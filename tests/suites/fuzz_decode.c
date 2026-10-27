/* libFuzzer decoder target: feeds arbitrary bytes to zpse_decompress_block
 * and zpse_decompress_frame. Unlike the round-trip fuzzer (which only ever
 * sees valid encoder output), this attacks the decoder's error-handling,
 * boundary, and corrupt-stream paths directly. ASan/UBSan catch any memory
 * safety bug; a clean non-zero return (never a crash) is the contract.
 *
 * Built by OSS-Fuzz and locally with:
 *   clang -fsanitize=fuzzer,address,undefined -Iinclude \
 *     tests/suites/fuzz_decode.c src/zpse_unity.c -o fuzz_decode
 *
 * Always returns 0: any input must be cleanly accepted or rejected,
 * never crash. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"

#define IN_CAP  ZPSE_MAX_BLOCK_SIZE
#define OUT_CAP (ZPSE_MAX_BLOCK_SIZE + ZPSE_BLOCK_HEADER_SIZE + 256)

static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t out[OUT_CAP];

/* Try to decode `data` as a block. The decoder must either succeed and
 * produce exactly the bytes it claims, or reject cleanly. Either way,
 * no crash, no memory corruption. */
static void fuzz_decode_block(const uint8_t *data, size_t size)
{
    uint32_t clen = 0u;
    uint32_t olen = 0u;
    int rc;

    /* Only the first ZPSE_BLOCK_HEADER_SIZE bytes matter as a header; the
     * rest is payload. Feed the whole buffer as compressed data. */
    if (size > IN_CAP) {
        size = IN_CAP;
    }
    rc = zpse_decompress_block(data, (uint32_t)size, out, (uint32_t)sizeof(out),
                               &olen, dec_arena);
    (void)rc;
    (void)olen;
    /* If rc == 0 the block decoded; if rc < 0 it was rejected. Both are
     * acceptable outcomes. The important thing is no crash/UB. */
}

/* Try to decode `data` as a frame. Same contract. */
static void fuzz_decode_frame(const uint8_t *data, size_t size)
{
    size_t flen = 0u;
    size_t olen = 0u;
    int rc;
    static uint8_t fout[OUT_CAP];

    if (size > sizeof(fout)) {
        size = sizeof(fout);
    }
    rc = zpse_decompress_frame(data, size, fout, sizeof(fout), &olen,
                               dec_arena);
    (void)rc;
    (void)olen;
    /* Same contract: accept or reject cleanly, never crash. */
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    /* Split the input: first half as a block, second half as a frame,
     * so both decoders get exercised from every fuzz input. */
    size_t half = size / 2u;

    fuzz_decode_block(data, size);
    if (half > 0u) {
        fuzz_decode_frame(data + half, size - half);
    }

    return 0;
}
