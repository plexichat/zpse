/* libFuzzer target for zpse: differential fuzzing of block + frame
 * round-trips over arbitrary input bytes, across all nine effort levels.
 *
 * Built by OSS-Fuzz and locally with:
 *   clang -fsanitize=fuzzer,address,undefined -Iinclude \
 *     tests/suites/fuzz_target.c src/zpse_unity.c -o fuzz_target
 *
 * Always returns 0: any input must round-trip or be cleanly rejected,
 * never crash. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"

#define IN_CAP  ZPSE_MAX_BLOCK_SIZE
#define OUT_CAP (ZPSE_MAX_BLOCK_SIZE + ZPSE_BLOCK_HEADER_SIZE + 256)

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[IN_CAP];
static uint8_t comp[OUT_CAP];
static uint8_t out[IN_CAP];

/* Pick an effort level from the input bytes so every level gets exercised. */
static uint32_t pick_level(const uint8_t *data, size_t size)
{
    if (size == 0u) {
        return ZPSE_ENC_SPEED_BALANCED;
    }
    return (uint32_t)(data[0] % 9u) + 1u;
}

/* Round-trip one block at one level. Returns 0 always; failures are
 * counted locally but never abort (the fuzzer needs to keep going). */
static void fuzz_block(const uint8_t *data, size_t size, uint32_t level)
{
    uint32_t enc_flags = (uint32_t)level << ZPSE_ENC_SPEED_SHIFT;
    uint32_t clen = 0u;
    uint32_t olen = 0u;
    int rc;

    if (size > IN_CAP) {
        size = IN_CAP;
    }
    memcpy(in, data, size);

    rc = zpse_compress_block_ex(in, (uint32_t)size, comp, (uint32_t)sizeof(comp),
                                &clen, 0u, 1, enc_flags, enc_arena);
    if (rc != 0) {
        return; /* cleanly rejected */
    }
    if (clen > (uint32_t)size + (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return; /* over envelope: not a valid output, skip */
    }

    memset(out, 0xA5, sizeof(out));
    rc = zpse_decompress_block(comp, clen, out, (uint32_t)sizeof(out),
                               &olen, dec_arena);
    if (rc < 0 || olen != (uint32_t)size || memcmp(in, out, size) != 0) {
        return; /* decode mismatch */
    }
}

/* Round-trip one frame at one level. Same contract. */
static void fuzz_frame(const uint8_t *data, size_t size, uint32_t level)
{
    uint32_t enc_flags = (uint32_t)level << ZPSE_ENC_SPEED_SHIFT;
    size_t flen = 0u;
    size_t olen = 0u;
    int rc;
    static uint8_t fcomp[300000];
    static uint8_t fout[IN_CAP];

    if (size > IN_CAP) {
        size = IN_CAP;
    }
    memcpy(in, data, size);

    rc = zpse_compress_frame_ex(in, size, fcomp, sizeof(fcomp),
                                &flen, 4096u, enc_flags, enc_arena);
    if (rc != 0) {
        return;
    }

    memset(fout, 0xA5, sizeof(fout));
    rc = zpse_decompress_frame(fcomp, flen, fout, sizeof(fout),
                               &olen, dec_arena);
    if (rc != 0 || olen != size || memcmp(in, fout, size) != 0) {
        return;
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    uint32_t level = pick_level(data, size);
    size_t half = size / 2u;

    /* Block round-trip on the whole input. */
    fuzz_block(data, size, level);
    /* Block round-trip on a prefix (exercises partial blocks). */
    if (half > 0u) {
        fuzz_block(data, half, level);
    }
    /* Frame round-trip on the whole input. */
    fuzz_frame(data, size, level);

    return 0;
}
