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

/* Deterministic FLAT seeds (sync 0x50B9). Arbitrary fuzz bytes rarely form a
 * valid FLAT header, so these hand-built blocks keep all four codecs and a
 * few corrupt envelopes in every invocation's corpus. */
static void flat_seed(uint8_t *b, uint32_t n, uint8_t codec, uint8_t param,
                      const uint8_t *payload, uint32_t plen)
{
    uint32_t i;

    memset(b, 0, 16u);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_FLAT);
    b[2] = (uint8_t)(ZPSE_B_RESET | ZPSE_B_LAST);
    b[3] = codec;
    b[4] = param;
    b[5] = 0u;
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)(16u + plen));
    zpse_store_u32le(b + 10, 0x50u);
    for (i = 0u; i < plen; i++) {
        b[16u + i] = payload[i];
    }
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

static void fuzz_flat_seeds(void)
{
    static uint8_t raw[16u + 32u];
    static uint8_t nib[16u + 32u];
    static uint8_t bit[16u + 16u];
    static uint8_t sub[16u + 22u];
    static const uint8_t pal[16] = {
        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
        8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u
    };
    uint8_t payload[32];
    uint32_t i;

    for (i = 0u; i < 32u; i++) {
        payload[i] = (uint8_t)(0x40u + i);
    }
    flat_seed(raw, 32u, ZPSE_FLAT_RAW, 0u, payload, 32u);
    /* NIBBLE4 needs only ceil(32/2) = 16 index bytes for n = 32; the palette
     * is 16 bytes. Passing 32 would read 16 bytes past `pal`. */
    flat_seed(nib, 32u, ZPSE_FLAT_NIBBLE4, 15u, pal, 16u);
    flat_seed(bit, 32u, ZPSE_FLAT_BIT2, 3u, pal, 16u);

    memset(payload, 0, sizeof payload);
    payload[0] = 4u;  /* period */
    payload[1] = 1u;  /* keepmask: position 0 of each group is kept */
    for (i = 2u; i < 10u; i++) {
        payload[i] = 0x5Au; /* template */
    }
    for (i = 10u; i < 18u; i++) {
        payload[i] = (uint8_t)(i - 10u); /* one sub byte per group */
    }
    /* [18..22) stays zero: the four pad bytes. */
    flat_seed(sub, 32u, ZPSE_FLAT_SUBLELANE, 0u, payload, 22u);

    fuzz_decode_block(raw, sizeof raw);
    fuzz_decode_block(nib, sizeof nib);
    fuzz_decode_block(bit, sizeof bit);
    fuzz_decode_block(sub, sizeof sub);

    /* Corrupt envelopes: each must be rejected cleanly, never crash. */
    raw[14] ^= 0x01u;                       /* bad CRC16 */
    fuzz_decode_block(raw, sizeof raw);
    nib[3] = 4u;                            /* codec > 3 */
    fuzz_decode_block(nib, sizeof nib);
    bit[2] = 0u;                            /* missing RESET */
    fuzz_decode_block(bit, sizeof bit);
    sub[5] = 1u;                            /* reserved byte */
    fuzz_decode_block(sub, sizeof sub);
    fuzz_decode_block(raw, 15u);            /* truncated header */
    fuzz_decode_block(sub, sizeof sub - 1u); /* truncated payload */
}

/* Deterministic TEXT seeds (sync 0x50BA). Arbitrary fuzz bytes almost never
 * form a valid TEXT header, so these hand-built blocks keep the decoder's
 * 7-bit-packed TEXT path in every invocation's corpus. Each is a legal packed
 * stream: the 16-byte header names n, and the payload is n - n/8 bytes, so
 * the block decodes exactly. */
static void text_seed(uint8_t *b, uint32_t n, uint8_t flags, uint32_t seq,
                      const uint8_t *payload, uint32_t plen)
{
    uint32_t i;

    memset(b, 0, 16u + plen);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
    b[2] = flags;
    b[3] = 0u; b[4] = 0u; b[5] = 0u; /* reserved */
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)(16u + plen));
    zpse_store_u32le(b + 10, seq);
    for (i = 0u; i < plen; i++) {
        b[16u + i] = payload[i];
    }
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

static void fuzz_text_seeds(void)
{
    static uint8_t t0[16u + 7u];   /* n=8:  8  - 8/8  = 7 packed bytes */
    static uint8_t t1[16u + 14u];  /* n=16: 16 - 16/8 = 14 */
    static uint8_t t2[16u + 28u];  /* n=32: 32 - 32/8 = 28 */
    uint8_t payload[32];
    uint32_t i;

    /* Three valid 7-bit packed streams of increasing size. Every bit pattern
     * is legal payload, so these vary while keeping the exact packed length
     * the decoder requires (n - n/8). */
    for (i = 0u; i < 8u; i++) {
        payload[i] = (uint8_t)(0x41u + i);
    }
    text_seed(t0, 8u, (uint8_t)(ZPSE_B_RESET | ZPSE_B_LAST), 11u, payload, 7u);

    for (i = 0u; i < 16u; i++) {
        payload[i] = (uint8_t)(0x61u + (i & 7u));
    }
    text_seed(t1, 16u, (uint8_t)(ZPSE_B_RESET | ZPSE_B_LAST), 12u, payload, 14u);

    for (i = 0u; i < 32u; i++) {
        payload[i] = (uint8_t)(0x20u + (i % 5u));
    }
    text_seed(t2, 32u, (uint8_t)(ZPSE_B_RESET | ZPSE_B_LAST), 13u, payload, 28u);

    fuzz_decode_block(t0, sizeof t0);
    fuzz_decode_block(t1, sizeof t1);
    fuzz_decode_block(t2, sizeof t2);

    /* Corrupt TEXT streams: every payload bit pattern is legal under the
     * 7-bit pack, so malformed means a bad header CRC, a truncated header,
     * or a payload shorter than the header's compressed-size claim. Each
     * must be rejected cleanly, never crash. */
    fuzz_decode_block(t0, sizeof t0 - 1u); /* payload short of c_text */
    t1[14] ^= 0x01u;                       /* CRC16 low byte */
    fuzz_decode_block(t1, sizeof t1);
    t2[15] ^= 0xFFu;                       /* CRC16 high byte */
    fuzz_decode_block(t2, sizeof t2);
    fuzz_decode_block(t2, 15u);            /* truncated TEXT header */
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    /* Split the input: first half as a block, second half as a frame,
     * so both decoders get exercised from every fuzz input. */
    size_t half = size / 2u;

    fuzz_flat_seeds();
    fuzz_text_seeds();

    fuzz_decode_block(data, size);
    if (half > 0u) {
        fuzz_decode_frame(data + half, size - half);
    }

    return 0;
}
