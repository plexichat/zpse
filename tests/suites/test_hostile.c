/* Hostile-bitstream tests: every corruption must yield a clean status
 * code, never a crash or silent acceptance. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
#include "zpse/text.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"
#include "zpse/dpcm1.h"
#include "zpse/mrd.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t input[2048];
static uint8_t comp[8192];
static uint8_t work[8192];
static uint8_t out[2048];
static uint8_t big[20000];
static uint8_t big_comp[24000];
static uint8_t big_out[20000];
static uint32_t big_len = 0;
static int fails = 0;
static int checks = 0;
#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { printf("FAIL " __VA_ARGS__); printf("\n"); fails++; } \
    else { printf("ok   " __VA_ARGS__); printf("\n"); } \
} while (0)

static uint32_t clen = 0;

static void refresh_flags(uint32_t flags) {
    uint32_t dummy = 0;
    int rc = zpse_compress_block_ex(input, sizeof input, comp, sizeof comp,
                                    &dummy, 0, 1, flags, enc_arena);
    (void)rc;
    clen = dummy;
}

/* The default stream (no payload checksum, as in LZ4). */
static void refresh(void) {
    refresh_flags(0u);
}

/* The same input with the optional payload CRC tag. */
static void refresh_crc(void) {
    refresh_flags((uint32_t)ZPSE_ENC_CHECKSUM);
}

/* The same, on a block too long for a page's 16-byte header, so the decoded
 * CRC has a field to live in. */
static void refresh_crc_big(void) {
    uint32_t dummy = 0;
    int rc = zpse_compress_block_ex(big, sizeof big, big_comp, sizeof big_comp,
                                    &dummy, 0, 1,
                                    (uint32_t)ZPSE_ENC_CHECKSUM, enc_arena);
    (void)rc;
    big_len = dummy;
}

/* These cases corrupt a header field and repair the checksum that covers it.
 * A block of a page's size carries the 16-byte header (flags at byte 2,
 * transform at 3, checksum over 14 bytes) and a longer one the 32-byte
 * header (flags at 3, transform at 12, checksum over 28), so both the field's
 * offset and the repair follow from the sync. */
static uint32_t hostile_short(const uint8_t *b, uint32_t len)
{
    return zpse_block_header_size(b, len) ==
           (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
}

static void hostile_fix_crc(uint8_t *b, uint32_t len)
{
    if (hostile_short(b, len)) {
        zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14) & 0xFFFFu));
    } else {
        zpse_store_u32le(b + 28, zpse_crc32c(b, 28));
    }
}

/* A canonical one-value token (v1.1): 16 bytes, no payload, fill byte in [3].
 * Built by hand so these cases do not depend on the encoder choosing this
 * form. token_fix_crc repairs the 16-bit header CRC after a field is changed,
 * so the field check -- not the checksum -- is what rejects a bad token. */
static void token_make(uint8_t *b, uint32_t n, uint8_t fill, uint8_t flags,
                       uint32_t seq)
{
    memset(b, 0, 16u);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_TOKEN);
    b[2] = flags;
    b[3] = fill;
    b[4] = 1u;
    b[5] = 0u;
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)ZPSE_BLOCK_HEADER_SIZE_SHORT);
    zpse_store_u32le(b + 10, seq);
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

static void token_fix_crc(uint8_t *b)
{
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

/* The bytes just past an n-byte decode must stay untouched: the token filler
 * is a memset, so this is where an out-of-bounds write would show up. */
static int token_canary(const uint8_t *b, uint32_t n)
{
    uint32_t i;
    for (i = n; i < n + 16u; i++) {
        if (b[i] != 0xCCu) {
            return 0;
        }
    }
    return 1;
}

/* Every corrupt token must get a clean (negative) status from both the public
 * validator and the whole decode path, leave the caller's length untouched and
 * write nothing into the output buffer. */
static void token_reject(const char *what, const uint8_t *tok, uint32_t len,
                         int expect)
{
    uint8_t obuf[256u + 16u];
    ZpseBlockHeader h;
    uint32_t olen = 0xDEADBEEFu;
    int brc = zpse_block_token_build(&h, tok, len);
    int rc;

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_block(tok, len, obuf, 256u, &olen, dec_arena);
    CHECK(brc == expect, "token build: %s -> %s", what, zpse_status_name(brc));
    CHECK(rc < 0 && olen == 0xDEADBEEFu && token_canary(obuf, 256u),
          "token: %s -> clean status, no write past cap", what);
}

/* A minimal FLAT RAW block (sync 0x50B9): 16-byte header plus 32 verbatim
 * bytes, RESET only. flat_fix_crc refreshes the 16-bit header CRC after a
 * field is changed, so the field check -- not the checksum -- is what rejects
 * a corrupt envelope. */
static void flat_fix_crc(uint8_t *b)
{
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

static void flat_make(uint8_t *b, uint32_t n, uint8_t codec, uint8_t param,
                      uint32_t seq)
{
    uint32_t i;

    memset(b, 0, 16u + 32u);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_FLAT);
    b[2] = (uint8_t)ZPSE_B_RESET;
    b[3] = codec;
    b[4] = param;
    b[5] = 0u;
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)(16u + n));
    zpse_store_u32le(b + 10, seq);
    for (i = 0u; i < 32u; i++) {
        b[16u + i] = (uint8_t)(0x40u + i);
    }
    flat_fix_crc(b);
}

/* A corrupt FLAT block must get the expected parser status and a clean
 * negative from the whole decode path, leave the caller's length untouched
 * and write nothing into or past the output cap. */
static void flat_reject(const char *what, const uint8_t *blk, uint32_t len,
                        int expect_parse)
{
    uint8_t obuf[64u + 16u];
    uint32_t olen = 0xDEADBEEFu;
    ZpseBlockHeader h;
    int prc = zpse_block_header_parse_flat(&h, blk, len);
    int rc, i, clean;

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_block(blk, len, obuf, 64u, &olen, dec_arena);
    CHECK(prc == expect_parse, "flat parse: %s -> %s", what,
          zpse_status_name(prc));
    clean = (rc < 0 && olen == 0xDEADBEEFu);
    for (i = 0; i < (int)sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) {
            clean = 0;
        }
    }
    CHECK(clean, "flat: %s -> clean status, no write past cap", what);
}

/* Build a FLAT block with an arbitrary payload of `plen` bytes; the header's
 * c field is 16 + plen so the decode path derives exactly that payload length
 * and the codec's own exact-length check is what accepts or rejects it. */
static void flat_seed_p(uint8_t *b, uint32_t n, uint8_t codec, uint8_t param,
                        const uint8_t *payload, uint32_t plen)
{
    uint32_t i;

    memset(b, 0, 16u + plen);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_FLAT);
    b[2] = (uint8_t)ZPSE_B_RESET;
    b[3] = codec;
    b[4] = param;
    b[5] = 0u;
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)(16u + plen));
    zpse_store_u32le(b + 10, 17u);
    for (i = 0u; i < plen; i++) {
        b[16u + i] = payload[i];
    }
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

/* FLAT payload lengths are exact: NIBBLE4 needs 16 + ceil(n/2) and BIT2 needs
 * 8 + ceil(n/4), RAW needs n and SUBLANE 14 + k*ceil(n/period). Supply one byte
 * too few and one too many at each boundary (n chosen to include non-multiples
 * of 4 and 2) and require clean rejection with an untouched output canary. */
static void test_flat_length_mismatch(void)
{
    static uint8_t pay[4300];
    static uint8_t blk[16u + 4300u];
    static const uint32_t ns[] = {
        1u, 3u, 5u, 7u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u
    };
    uint32_t i, d;

    for (i = 0u; i < sizeof ns / sizeof ns[0]; i++) {
        uint32_t n = ns[i];
        uint32_t nib = 16u + ((n + 1u) >> 1);
        uint32_t bit = 8u + ((n + 3u) >> 2);

        for (d = 0u; d < 2u; d++) {
            uint32_t plen = (d == 0u) ? nib - 1u : nib + 1u;

            memset(pay, 0x5A, sizeof pay);
            flat_seed_p(blk, n, ZPSE_FLAT_NIBBLE4, 15u, pay, plen);
            flat_reject("nibble length", blk, 16u + plen, ZPSE_OK);
        }
        for (d = 0u; d < 2u; d++) {
            uint32_t plen = (d == 0u) ? bit - 1u : bit + 1u;

            memset(pay, 0x5A, sizeof pay);
            flat_seed_p(blk, n, ZPSE_FLAT_BIT2, 3u, pay, plen);
            flat_reject("bit2 length", blk, 16u + plen, ZPSE_OK);
        }
    }

    /* RAW and SUBLANE get the same boundary treatment on small n (kept below
     * flat_reject's 64-byte cap so the length check, not the cap, rejects). */
    {
        static const uint32_t small[] = { 3u, 17u, 32u };

        for (i = 0u; i < sizeof small / sizeof small[0]; i++) {
            uint32_t n = small[i];
            uint32_t raw = n;
            /* period 4, keepmask 0x5 => k = 2 kept lanes per group. */
            uint32_t sub = 14u + 2u * ((n + 3u) >> 2);
            uint32_t d;

            for (d = 0u; d < 2u; d++) {
                uint32_t plen = (d == 0u) ? raw - 1u : raw + 1u;

                memset(pay, 0x5A, sizeof pay);
                flat_seed_p(blk, n, ZPSE_FLAT_RAW, 0u, pay, plen);
                flat_reject("raw length", blk, 16u + plen, ZPSE_OK);
            }
            for (d = 0u; d < 2u; d++) {
                uint32_t plen = (d == 0u) ? sub - 1u : sub + 1u;

                memset(pay, 0, sizeof pay);
                pay[0] = 4u;  /* period */
                pay[1] = 0x5u; /* keepmask: two kept lanes -> k = 2 */
                flat_seed_p(blk, n, ZPSE_FLAT_SUBLELANE, 0u, pay, plen);
                flat_reject("sublane length", blk, 16u + plen, ZPSE_OK);
            }
        }
    }
}

/* Both CRC bytes of a 16-byte FLAT header are covered: flip every bit of each
 * and the header parser must report BAD_HEADER_CRC. A full flip of either byte
 * must also be rejected by the whole decode path with an intact output canary. */
static void test_flat_crc_bytes(void)
{
    uint8_t hb[16u + 32u];
    uint8_t work2[16u + 32u];
    ZpseBlockHeader fh;
    uint32_t cidx, b;

    flat_make(hb, 32u, ZPSE_FLAT_RAW, 0u, 31u);
    for (cidx = 14u; cidx <= 15u; cidx++) {
        for (b = 0u; b < 8u; b++) {
            memcpy(work2, hb, 16u);
            work2[cidx] ^= (uint8_t)(1u << b);
            CHECK(zpse_block_header_parse_flat(&fh, work2, 16u) ==
                      ZPSE_BAD_HEADER_CRC,
                  "flat CRC byte %u bit %u rejected", cidx, b);
        }
    }
    for (cidx = 14u; cidx <= 15u; cidx++) {
        memcpy(work2, hb, sizeof hb);
        work2[cidx] ^= 0xFFu;
        flat_reject("flat CRC byte flip", work2, (uint32_t)sizeof hb,
                    ZPSE_BAD_HEADER_CRC);
    }
}

/* The TEXT page form (sync 0x50BA): a 16-byte header with no codec field --
 * bytes [3..6) are reserved zero, n at [6..8) is the decoded size (0 spells
 * 65536), c at [8..10) is the total block size. text_fix_crc repairs the
 * 16-bit header CRC after a field is changed, so the field check -- not the
 * checksum -- is what rejects a corrupt envelope. The payload is a fixed-rate
 * 7-bit pack: eight ASCII bytes become seven, so its length is n - (n>>3) and
 * length is the only structural constraint. Packed bytes may have bit 7 set. */

#define TEXT_N     64u
#define TEXT_PLEN  (TEXT_N - (TEXT_N >> 3))

/* Independent 8->7 pack: bit 7*i+k of each group is bit k of byte i. Built here
 * rather than calling zpse_text_encode so a bug in the encoder cannot make the
 * hand-built streams self-consistently wrong. */
static uint32_t text_pack_bits(const uint8_t *src, uint32_t n, uint8_t *dst)
{
    uint32_t ip = 0u, op = 0u;

    while (ip < n) {
        uint32_t r = n - ip, i, k, nb;
        uint64_t v = 0u;

        if (r > 8u) {
            r = 8u;
        }
        for (i = 0u; i < r; i++) {
            for (k = 0u; k < 7u; k++) {
                if (((src[ip + i] >> k) & 1u) != 0u) {
                    v |= UINT64_C(1) << (7u * i + k);
                }
            }
        }
        nb = (7u * r + 7u) >> 3;
        for (k = 0u; k < nb; k++) {
            dst[op + k] = (uint8_t)(v >> (8u * k));
        }
        ip += r;
        op += nb;
    }
    return op;
}

/* Independent 7->8 unpack, used to name the exact bytes that any packed bit
 * pattern (including bit 7 set) must decode to. */
static void text_unpack_bits(const uint8_t *packed, uint32_t n, uint8_t *dst)
{
    uint32_t ip = 0u, op = 0u;

    while (op < n) {
        uint32_t r = n - op, i, k, nb;
        uint64_t v = 0u;

        if (r > 8u) {
            r = 8u;
        }
        nb = (7u * r + 7u) >> 3;
        for (k = 0u; k < nb; k++) {
            v |= (uint64_t)packed[ip + k] << (8u * k);
        }
        for (i = 0u; i < r; i++) {
            uint8_t b = 0u;

            for (k = 0u; k < 7u; k++) {
                if (((v >> (7u * i + k)) & 1u) != 0u) {
                    b = (uint8_t)(b | (1u << k));
                }
            }
            dst[op + i] = b;
        }
        ip += nb;
        op += r;
    }
}

static void text_fix_crc(uint8_t *b)
{
    zpse_store_u16le(b + 14, (uint16_t)(zpse_crc32c(b, 14u) & 0xFFFFu));
}

/* A minimal TEXT block: RESET only, n ASCII 'A' bytes packed to n-n/8 bytes so
 * the stream decodes to exactly n bytes. Built by hand so these cases do not
 * depend on the encoder choosing this form. */
static void text_make(uint8_t *b, uint32_t n, uint8_t flags, uint32_t seq)
{
    static uint8_t src[65536u];
    uint32_t i, plen;

    for (i = 0u; i < n; i++) {
        src[i] = 0x41u;
    }
    plen = n - (n >> 3);
    memset(b, 0, 16u + plen);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
    b[2] = flags;
    b[3] = 0u; b[4] = 0u; b[5] = 0u;
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)(16u + plen));
    zpse_store_u32le(b + 10, seq);
    (void)text_pack_bits(src, n, b + 16u);
    text_fix_crc(b);
}

/* Build a TEXT block with an arbitrary packed payload of `plen` bytes. n is
 * the DECLARED decoded size (0 means 65536) and c is 16 + plen, so a mismatch
 * between n and the fixed rate n - (n>>3) is what decode has to catch. */
static void text_seed_p(uint8_t *b, uint32_t n, const uint8_t *payload,
                        uint32_t plen)
{
    uint32_t i;

    memset(b, 0, 16u + plen);
    zpse_store_u16le(b + 0, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
    b[2] = (uint8_t)ZPSE_B_RESET;
    b[3] = 0u; b[4] = 0u; b[5] = 0u;
    zpse_store_u16le(b + 6, (uint16_t)n);
    zpse_store_u16le(b + 8, (uint16_t)(16u + plen));
    zpse_store_u32le(b + 10, 17u);
    for (i = 0u; i < plen; i++) {
        b[16u + i] = payload[i];
    }
    text_fix_crc(b);
}

/* A corrupt TEXT block must get the expected parser status and a clean
 * negative from the whole decode path, leave the caller's length untouched and
 * write nothing into or past the output cap. */
static void text_reject(const char *what, const uint8_t *blk, uint32_t len,
                        int expect_parse)
{
    uint8_t obuf[64u + 16u];
    uint32_t olen = 0xDEADBEEFu;
    ZpseBlockHeader h;
    int prc = zpse_block_header_parse_text(&h, blk, len);
    int rc, i, clean;

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_block(blk, len, obuf, 64u, &olen, dec_arena);
    CHECK(prc == expect_parse, "text parse: %s -> %s", what,
          zpse_status_name(prc));
    clean = (rc < 0 && olen == 0xDEADBEEFu);
    for (i = 0; i < (int)sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) {
            clean = 0;
        }
    }
    CHECK(clean, "text: %s -> clean status, no write past cap", what);
}

/* Hostile-bitstream cases for the TEXT block form (sync 0x50BA). Every case
 * repairs the CRC where needed so a field check -- not the checksum -- is what
 * rejects a corrupt envelope, and checks the canary past the output cap as well
 * as the status. */
static void test_text_hostile(void)
{
    static uint8_t t64_in[65536u];
    static uint8_t t64_stream[65536u];
    static uint8_t t64_blk[16u + 65536u];
    static uint8_t t64_out[65536u + 16u];
    uint8_t tb[16u + TEXT_PLEN];
    uint8_t obuf[64u + 16u];
    uint8_t decscratch[32u];
    uint8_t tok[256u * 7u];
    uint8_t texp[2048u];
    uint8_t tok1[1u];
    uint8_t tok2[2u];
    uint8_t tok3[8u];
    uint8_t scratch[2048u + 16u];
    ZpseBlockHeader h, h2;
    uint32_t olen, i, j, v, L, total;
    int rc, ok;

    /* Baseline: a valid RESET-only TEXT block of 64 ASCII 'A' bytes, packed to
     * 56 payload bytes, parses and decodes byte-exactly, with an intact canary
     * past the cap. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    CHECK(zpse_block_header_parse_text(&h, tb, sizeof tb) == ZPSE_OK &&
              h.sync == (uint16_t)ZPSE_BLOCK_SYNC_TEXT &&
              h.block_flags == (uint8_t)ZPSE_B_RESET &&
              h.uncompressed_size == 64u &&
              h.compressed_block_size == sizeof tb &&
              h.block_sequence_number == 23u &&
              h.header_size == ZPSE_BLOCK_HEADER_SIZE_SHORT,
          "text: baseline block parses");
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_block(tb, sizeof tb, obuf, 64u, &olen, dec_arena);
    ok = (rc == ZPSE_OK && olen == 64u);
    for (j = 0u; j < 64u; j++) {
        if (obuf[j] != 0x41u) { ok = 0; }
    }
    for (j = 64u; j < sizeof obuf; j++) {
        if (obuf[j] != 0xCCu) { ok = 0; }
    }
    CHECK(ok, "text: valid stream decodes to n bytes, canary intact");

    /* LAST is the one optional flag bit: still legal, still fills exactly n. */
    text_make(tb, 64u, (uint8_t)(ZPSE_B_RESET | ZPSE_B_LAST), 23u);
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_block(tb, sizeof tb, obuf, 64u, &olen, dec_arena);
    ok = (rc == ZPSE_OK && olen == 64u);
    for (j = 0u; j < 64u; j++) {
        if (obuf[j] != 0x41u) { ok = 0; }
    }
    for (j = 64u; j < sizeof obuf; j++) {
        if (obuf[j] != 0xCCu) { ok = 0; }
    }
    CHECK(ok, "text: LAST bit accepted, fills exactly n");

    /* Header parse <-> serialize round-trip: serialising a parsed header and
     * parsing it back must reproduce every field of the parsed header. */
    memset(&h2, 0, sizeof h2);
    zpse_block_header_serialize_text(tb, &h);
    CHECK(zpse_block_header_parse_text(&h2, tb, 16u) == ZPSE_OK &&
              h2.sync == h.sync &&
              h2.block_version == h.block_version &&
              h2.block_flags == h.block_flags &&
              h2.uncompressed_size == h.uncompressed_size &&
              h2.compressed_block_size == h.compressed_block_size &&
              h2.transform_id == h.transform_id &&
              h2.segment_count == h.segment_count &&
              h2.transform_meta_size == h.transform_meta_size &&
              h2.tile_size == h.tile_size &&
              h2.predictor_profile == h.predictor_profile &&
              h2.reserved0 == h.reserved0 &&
              h2.block_sequence_number == h.block_sequence_number &&
              h2.decoded_crc32c == h.decoded_crc32c &&
              h2.header_crc32c == h.header_crc32c &&
              h2.header_size == h.header_size,
          "text: serialize -> parse round-trips every header field");

    /* Truncated header: one byte short. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    text_reject("truncated header", tb, 15u, ZPSE_TRUNCATED);

    /* c past the bytes supplied (parser accepts, decode path's length
     * arithmetic rejects with TRUNCATED). */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    zpse_store_u16le(tb + 8, (uint16_t)(sizeof tb + 1u));
    text_fix_crc(tb);
    text_reject("comp past input", tb, sizeof tb, ZPSE_OK);
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0xDEADBEEFu;
    CHECK(zpse_decompress_block(tb, sizeof tb, obuf, 64u, &olen, dec_arena) ==
              ZPSE_TRUNCATED,
          "text: comp past input -> TRUNCATED from decode path");

    /* c below the 16-byte header, c==0, and c==16 with no payload at all. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    zpse_store_u16le(tb + 8, 15u);
    text_fix_crc(tb);
    text_reject("comp below header", tb, sizeof tb, ZPSE_BAD_BLOCK_SIZE);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    zpse_store_u16le(tb + 8, 0u);
    text_fix_crc(tb);
    text_reject("comp zero", tb, sizeof tb, ZPSE_BAD_BLOCK_SIZE);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    zpse_store_u16le(tb + 8, 16u);
    text_fix_crc(tb);
    text_reject("comp == header, empty payload", tb, sizeof tb, ZPSE_OK);
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0xDEADBEEFu;
    CHECK(zpse_decompress_block(tb, sizeof tb, obuf, 64u, &olen, dec_arena) ==
              ZPSE_BAD_BLOCK_SIZE,
          "text: empty payload -> BAD_BLOCK_SIZE from decode path");

    /* Bad CRC16: flip a covered byte, do not repair. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[6] ^= 0xFFu;
    text_reject("bad CRC16", tb, sizeof tb, ZPSE_BAD_HEADER_CRC);

    /* Both CRC bytes are covered: flip every bit of each and the header parser
     * must report BAD_HEADER_CRC; a full flip of either byte must also be
     * rejected by the whole decode path with an intact output canary. */
    {
        uint8_t hb[16u + TEXT_PLEN];
        uint8_t work2[16u + TEXT_PLEN];

        text_make(hb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
        for (i = 14u; i <= 15u; i++) {
            for (j = 0u; j < 8u; j++) {
                memcpy(work2, hb, 16u);
                work2[i] ^= (uint8_t)(1u << j);
                CHECK(zpse_block_header_parse_text(&h2, work2, 16u) ==
                          ZPSE_BAD_HEADER_CRC,
                      "text CRC byte %u bit %u rejected", i, j);
            }
        }
        for (i = 14u; i <= 15u; i++) {
            memcpy(work2, hb, sizeof hb);
            work2[i] ^= 0xFFu;
            text_reject("text CRC byte flip", work2, (uint32_t)sizeof hb,
                        ZPSE_BAD_HEADER_CRC);
        }
    }

    /* n=0 on the wire is legal and spells 65536. Build a real 64 KiB packed
     * stream with the encoder (57344 bytes, still small enough for the 16-bit
     * c field), then check both the valid decode and a length mismatch. */
    {
        static const char phrase[] = "the quick brown fox jumps over the lazy dog ";
        uint32_t plen, pk = 0u, kk;

        for (kk = 0u; kk < 65536u; kk++) {
            if (phrase[pk] == '\0') { pk = 0u; }
            t64_in[kk] = (uint8_t)phrase[pk++];
        }
        plen = zpse_text_encode(t64_in, 65536u, t64_stream,
                                (uint32_t)sizeof t64_stream);
        CHECK(plen != 0u && plen + 16u <= 65535u,
              "text: 64 KiB input encodes to a u16-sized stream");
        if (plen != 0u) {
            uint32_t k, plen2;

            memset(t64_blk, 0, 16u);
            zpse_store_u16le(t64_blk + 0, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
            t64_blk[2] = (uint8_t)ZPSE_B_RESET;
            zpse_store_u16le(t64_blk + 6, 0u);
            zpse_store_u16le(t64_blk + 8, (uint16_t)(16u + plen));
            zpse_store_u32le(t64_blk + 10, 5u);
            memcpy(t64_blk + 16u, t64_stream, plen);
            text_fix_crc(t64_blk);
            CHECK(zpse_block_header_parse_text(&h, t64_blk,
                                               16u + plen) == ZPSE_OK &&
                      h.uncompressed_size == 65536u,
                  "text: n=0 on wire parses as 65536");
            memset(t64_out, 0xCC, sizeof t64_out);
            olen = 0u;
            rc = zpse_decompress_block(t64_blk, 16u + plen, t64_out, 65536u,
                                       &olen, dec_arena);
            ok = (rc == ZPSE_OK && olen == 65536u);
            for (k = 0u; k < 65536u; k++) {
                if (t64_out[k] != t64_in[k]) { ok = 0; }
            }
            for (k = 65536u; k < 65536u + 16u; k++) {
                if (t64_out[k] != 0xCCu) { ok = 0; }
            }
            CHECK(ok, "text: n=0 stream decodes to 64 KiB, canary intact");

            /* Drop the last packed byte: the header still declares 65536 but
             * the payload is now one byte short of n - (n>>3), so the fixed
             * rate check must reject it. */
            plen2 = plen - 1u;
            zpse_store_u16le(t64_blk + 8, (uint16_t)(16u + plen2));
            text_fix_crc(t64_blk);
            memset(t64_out, 0xCC, sizeof t64_out);
            olen = 0xDEADBEEFu;
            rc = zpse_decompress_block(t64_blk, 16u + plen2, t64_out, 65536u,
                                       &olen, dec_arena);
            ok = (rc < 0 && olen == 0xDEADBEEFu);
            for (k = 0u; k < sizeof t64_out; k++) {
                if (t64_out[k] != 0xCCu) { ok = 0; }
            }
            CHECK(ok, "text: n=65536 vs shorter packed stream -> clean reject");

            /* The serializer writes 65536 back as n=0 on the wire. */
            memset(&h2, 0, sizeof h2);
            h2.uncompressed_size = 65536u;
            zpse_block_header_serialize_text(tb, &h2);
            CHECK(zpse_load_u16le(tb + 6) == 0u,
                  "text: 65536 serialises as n=0 on the wire");
        }
    }

    /* Length mismatches at the codec boundary. A fixed-rate pack has exactly
     * one structural check: in_len must be n - (n>>3). Every bit pattern is
     * legal payload, so a byte >= 0x80 is not an error -- a wrong length is. */
    tok1[0] = 0x80u; /* bit 7 set is legal; 1 byte is short for n = 8 */
    CHECK(zpse_text_decode(tok1, 1u, decscratch, 8u) < 0,
          "text: packed length short for n -> negative");
    tok2[0] = 0x41u; tok2[1] = 0x41u; /* 2 bytes, n=1 expects 1 */
    CHECK(zpse_text_decode(tok2, 2u, decscratch, 1u) < 0,
          "text: packed length long for n -> negative");
    memset(tok3, 0x41u, sizeof tok3); /* 6 bytes, n=8 expects 7 */
    CHECK(zpse_text_decode(tok3, 6u, decscratch, 8u) < 0,
          "text: packed length one short of n -> negative");

    /* The same two mismatches at the block level, with an intact header so
     * the pack decoder -- not the envelope check -- is what rejects them. */
    text_seed_p(tb, 8u, tok1, 1u);
    text_reject("short stream for n", tb, 17u, ZPSE_OK);
    tok2[0] = 0x41u; tok2[1] = 0x41u;
    text_seed_p(tb, 1u, tok2, 2u);
    text_reject("long stream for n", tb, 18u, ZPSE_OK);

    /* Every packed byte value 0x00..0xFF is legal bits, including bit 7. Feed
     * all 256 values (each as a full eight-byte group's seven packed bytes)
     * as one valid-length stream: n = 256*8 and in_len = 256*7. An unmapped
     * or overflowing spread must show up as a wrong byte or a canary hit. */
    total = 0u;
    for (v = 0u; v < 256u; v++) {
        for (L = 0u; L < 7u; L++) {
            tok[total++] = (uint8_t)v;
        }
    }
    text_unpack_bits(tok, 2048u, texp);
    memset(scratch, 0xCC, sizeof scratch);
    rc = zpse_text_decode(tok, total, scratch, 2048u);
    ok = (rc == 0 && memcmp(scratch, texp, 2048u) == 0);
    for (j = 2048u; j < (uint32_t)sizeof scratch; j++) {
        if (scratch[j] != 0xCCu) { ok = 0; }
    }
    CHECK(ok, "text: all 256 packed byte values decode, no write past n");

    /* Wrong flags: RESET is required, LAST the only optional bit. Every other
     * bit names a form or feature this header has no room for. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] = 0u;
    text_fix_crc(tb);
    text_reject("missing RESET", tb, sizeof tb, ZPSE_BAD_HEADER);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] |= ZPSE_B_RAW;
    text_fix_crc(tb);
    text_reject("RAW flag set", tb, sizeof tb, ZPSE_BAD_HEADER);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] |= ZPSE_B_CRC;
    text_fix_crc(tb);
    text_reject("CRC flag set", tb, sizeof tb, ZPSE_BAD_HEADER);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] |= ZPSE_B_HAS_META;
    text_fix_crc(tb);
    text_reject("HAS_META flag set", tb, sizeof tb, ZPSE_BAD_HEADER);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] |= ZPSE_B_LZ;
    text_fix_crc(tb);
    text_reject("LZ flag set", tb, sizeof tb, ZPSE_BAD_HEADER);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] |= ZPSE_B_LZ_TILE;
    text_fix_crc(tb);
    text_reject("LZ_TILE flag set", tb, sizeof tb, ZPSE_BAD_HEADER);
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] |= ZPSE_B_HUF;
    text_fix_crc(tb);
    text_reject("HUF flag set", tb, sizeof tb, ZPSE_BAD_HEADER);
    /* ZPSE_B_RESERVED_MASK is empty: there is no reserved FLAG bit left to
     * set, the reserved BYTES are [3..6) and covered below. Still refuse a
     * byte with every forbidden bit set at once. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    tb[2] = (uint8_t)(ZPSE_B_RAW | ZPSE_B_CRC | ZPSE_B_HAS_META |
                      ZPSE_B_LZ | ZPSE_B_LZ_TILE | ZPSE_B_HUF);
    text_fix_crc(tb);
    text_reject("all forbidden flag bits", tb, sizeof tb, ZPSE_BAD_HEADER);

    /* Nonzero reserved bytes [3],[4],[5]. */
    for (i = 3u; i <= 5u; i++) {
        text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
        tb[i] = 1u;
        text_fix_crc(tb);
        text_reject("reserved byte nonzero", tb, sizeof tb, ZPSE_BAD_HEADER);
    }

    /* Wrong sync, CRC repaired: the block layer sees no TEXT header and the
     * generic parser refuses the unknown sync. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    zpse_store_u16le(tb, (uint16_t)(ZPSE_BLOCK_SYNC_TEXT + 1u));
    text_fix_crc(tb);
    text_reject("wrong sync", tb, sizeof tb, ZPSE_BAD_HEADER);

    /* Output cap one byte short: OVERFLOW with zero writes. */
    text_make(tb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0xDEADBEEFu;
    rc = zpse_decompress_block(tb, sizeof tb, obuf, 63u, &olen, dec_arena);
    ok = (rc == ZPSE_OUTPUT_OVERFLOW && olen == 0xDEADBEEFu);
    for (j = 0u; j < sizeof obuf; j++) {
        if (obuf[j] != 0xCCu) { ok = 0; }
    }
    CHECK(ok, "text: cap n-1 -> OUTPUT_OVERFLOW, no write");

    /* Random garbage and 1-4 random bit flips of a valid block: each must
     * either reject cleanly (length and every canary byte untouched) or decode
     * byte-exactly to what the same packed stream gives the codec directly. */
    {
        uint8_t rb[16u + TEXT_PLEN];
        uint8_t refbuf[64u + 16u];
        uint32_t r = 0x1BADB002u;
        uint32_t nref, k;
        int bad = 0, phase;

        for (phase = 0; phase < 2; phase++) {
            uint32_t cases = (phase == 0) ? 50u : 20u;
            for (i = 0u; i < cases; i++) {
                if (phase == 0) {
                    for (j = 0u; j < sizeof rb; j++) {
                        r = r * 1664525u + 1013904223u;
                        rb[j] = (uint8_t)(r >> 24);
                    }
                    /* Keep the form selected so the TEXT path is dispatched. */
                    zpse_store_u16le(rb, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
                } else {
                    uint32_t nf = 1u + (i & 3u); /* 1..4 bit flips */
                    text_make(rb, 64u, (uint8_t)ZPSE_B_RESET, 23u);
                    for (k = 0u; k < nf; k++) {
                        uint32_t pos;
                        r = r * 1664525u + 1013904223u;
                        pos = (r >> 8) % (uint32_t)sizeof rb;
                        r = r * 1664525u + 1013904223u;
                        rb[pos] ^= (uint8_t)(1u << ((r >> 24) & 7u));
                    }
                }
                /* Independent reference: parse + direct decode. */
                memset(refbuf, 0xCC, sizeof refbuf);
                if (zpse_block_header_parse_text(&h2, rb,
                                                 (uint32_t)sizeof rb) == ZPSE_OK &&
                    h2.uncompressed_size <= 64u) {
                    nref = h2.uncompressed_size;
                    if (zpse_text_decode(rb + 16u, (uint32_t)sizeof rb - 16u,
                                         refbuf, nref) < 0) {
                        nref = 0xFFFFFFFFu;
                    }
                } else {
                    nref = 0xFFFFFFFFu;
                }
                memset(obuf, 0xCC, sizeof obuf);
                olen = 0xDEADBEEFu;
                rc = zpse_decompress_block(rb, (uint32_t)sizeof rb, obuf, 64u,
                                           &olen, dec_arena);
                if (rc < 0) {
                    if (olen != 0xDEADBEEFu) { bad = 1; }
                    for (j = 0u; j < sizeof obuf; j++) {
                        if (obuf[j] != 0xCCu) { bad = 1; }
                    }
                } else if (nref == 0xFFFFFFFFu) {
                    bad = 1;
                } else {
                    if (rc != ZPSE_OK || olen != nref ||
                        memcmp(obuf, refbuf, nref) != 0) {
                        bad = 1;
                    }
                    for (j = nref; j < sizeof obuf; j++) {
                        if (obuf[j] != 0xCCu) { bad = 1; }
                    }
                }
            }
        }
        CHECK(!bad, "50x random TEXT streams + random bit flips cleanly handled");
    }
}

/* ---- DPCM1 whole-frame form ("DCM1") ----------------------------------
 * This form has no block-layer parser, so every case goes through
 * zpse_decompress_frame, which sniffs the magic and hands the bytes to
 * zpse_dpcm1_decode. The builder writes legal streams by hand, independent of
 * the encoder; each hostile case then mutates exactly one field. Wire layout:
 * 16-byte header, verbatim prefix, ns-1 signed delta bytes, then 6 bytes per
 * escape (u32 delta index, i16 true delta). */
static uint32_t dpcm1_build(uint8_t *b, uint32_t ns, uint16_t first,
                            const uint8_t *deltas,
                            const uint8_t *prefix, uint32_t plen,
                            const uint8_t *esctab, uint32_t ne)
{
    uint32_t nd = ns - 1u;
    uint32_t n = 2u * ns + plen;
    uint32_t i;

    memset(b, 0, 16u + plen + nd + 6u * ne);
    b[0] = 'D'; b[1] = 'C'; b[2] = 'M'; b[3] = '1';
    zpse_store_u32le(b + 4u, n);
    zpse_store_u16le(b + 8u, first);
    zpse_store_u16le(b + 10u, (uint16_t)plen);
    zpse_store_u32le(b + 12u, ne);
    for (i = 0u; i < plen; i++) b[16u + i] = prefix[i];
    for (i = 0u; i < nd; i++) b[16u + plen + i] = deltas[i];
    for (i = 0u; i < 6u * ne; i++) b[16u + plen + nd + i] = esctab[i];
    return 16u + plen + nd + 6u * ne;
}

/* The shared baseline: 16 samples, no prefix, no escapes, all deltas +20. */
static uint32_t dpcm1_base(uint8_t *b, uint8_t *deltas)
{
    uint32_t i;

    for (i = 0u; i < 16u; i++) deltas[i] = 20u;
    return dpcm1_build(b, 16u, 0x1000u, deltas, 0, 0u, 0, 0u);
}

/* A corrupt DCM1 frame must get a negative status from the public frame path,
 * leave the caller's length untouched and write nothing into the output. */
static void dpcm1_reject(const char *what, const uint8_t *blk, uint32_t len)
{
    uint8_t obuf[128u + 16u];
    size_t olen = (size_t)0xDEADBEEFu;
    uint32_t i;
    int rc, clean;

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_frame(blk, len, obuf, 128u, &olen, dec_arena);
    clean = (rc < 0 && olen == (size_t)0xDEADBEEFu);
    for (i = 0u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) clean = 0;
    }
    CHECK(clean, "dpcm1: %s -> clean status, no write past cap", what);
}

static void test_dpcm1_hostile(void)
{
    uint8_t b[128u];
    uint8_t deltas[16u];
    uint8_t prefix[4u] = { 0xDEu, 0xADu, 0xBEu, 0xEFu };
    uint8_t esc[12u];
    uint8_t obuf[128u + 16u];
    uint8_t ref[64u];
    uint32_t len, i;
    size_t olen;
    uint16_t v;
    int rc, ok;

    /* Valid baseline, no prefix: byte-exact decode and intact canary. */
    len = dpcm1_base(b, deltas);
    CHECK(len == 31u, "dpcm1: baseline stream is 31 bytes");
    v = 0x1000u;
    ref[0] = (uint8_t)v; ref[1] = (uint8_t)(v >> 8);
    for (i = 0u; i < 15u; i++) {
        v = (uint16_t)(v + 20u);
        ref[2u + 2u * i] = (uint8_t)v;
        ref[3u + 2u * i] = (uint8_t)(v >> 8);
    }
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_frame(b, len, obuf, 128u, &olen, dec_arena);
    ok = (rc == ZPSE_OK && olen == 32u && memcmp(obuf, ref, 32u) == 0);
    for (i = 32u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) ok = 0;
    }
    CHECK(ok, "dpcm1: valid stream decodes byte-exactly, canary intact");

    /* Valid prefix + one escape at delta index 5. */
    for (i = 0u; i < 16u; i++) deltas[i] = 20u;
    deltas[5] = 0u; /* placeholder for the escaped delta */
    zpse_store_u32le(esc, 5u);
    zpse_store_u16le(esc + 4u, (uint16_t)1000u);
    len = dpcm1_build(b, 16u, 0x2000u, deltas, prefix, 4u, esc, 1u);
    CHECK(len == 41u, "dpcm1: prefix+escape stream is 41 bytes");
    memcpy(ref, prefix, 4u);
    v = 0x2000u;
    ref[4] = (uint8_t)v; ref[5] = (uint8_t)(v >> 8);
    for (i = 0u; i < 15u; i++) {
        v = (uint16_t)(v + (i == 5u ? 1000u : 20u));
        ref[6u + 2u * i] = (uint8_t)v;
        ref[7u + 2u * i] = (uint8_t)(v >> 8);
    }
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_frame(b, len, obuf, 128u, &olen, dec_arena);
    ok = (rc == ZPSE_OK && olen == 36u && memcmp(obuf, ref, 36u) == 0);
    for (i = 36u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) ok = 0;
    }
    CHECK(ok, "dpcm1: prefix + escape decodes byte-exactly, canary intact");

    /* Wrong magic: the frame sniffer sees no DCM1 and falls through. */
    len = dpcm1_base(b, deltas);
    b[0] = 'X';
    dpcm1_reject("wrong magic", b, len);

    /* Truncation: magic only, then a header short of the 16-byte minimum. */
    len = dpcm1_base(b, deltas);
    dpcm1_reject("magic only", b, 4u);
    dpcm1_reject("short header", b, 15u);

    /* Original length field: a wrong value makes the exact need disagree with
     * the bytes supplied; the decoder itself also rejects a mismatched n. */
    len = dpcm1_base(b, deltas);
    zpse_store_u32le(b + 4u, 34u);
    dpcm1_reject("bad original length", b, len);
    CHECK(zpse_dpcm1_decode(b, len, obuf, 34u) == ZPSE_DPCM1_DECLINE,
          "dpcm1: decoder rejects n that disagrees with the header");

    zpse_store_u32le(b + 4u, 33u);
    dpcm1_reject("odd n", b, len);

    /* Prefix field: odd parity, longer than n-2, and leaving fewer than two
     * samples for the predictor. */
    len = dpcm1_base(b, deltas);
    zpse_store_u16le(b + 10u, 3u);
    dpcm1_reject("odd prefix", b, len);
    len = dpcm1_base(b, deltas);
    zpse_store_u16le(b + 10u, 32u);
    dpcm1_reject("prefix past n-2", b, len);
    len = dpcm1_base(b, deltas);
    zpse_store_u16le(b + 10u, 30u);
    dpcm1_reject("prefix leaves one sample", b, len);

    /* Escape count above ns-1 (= 15 for this stream). */
    len = dpcm1_base(b, deltas);
    zpse_store_u32le(b + 12u, 16u);
    dpcm1_reject("escape count > ns-1", b, len);

    /* Escape positions: must be strictly increasing and below ns-1. */
    for (i = 0u; i < 16u; i++) deltas[i] = 20u;
    zpse_store_u32le(esc, 5u);
    zpse_store_u16le(esc + 4u, (uint16_t)1000u);
    zpse_store_u32le(esc + 6u, 3u);
    zpse_store_u16le(esc + 10u, (uint16_t)2000u);
    len = dpcm1_build(b, 16u, 0x1000u, deltas, 0, 0u, esc, 2u);
    dpcm1_reject("non-monotonic escapes", b, len);

    zpse_store_u32le(esc, 15u); /* == ns-1 is out of range */
    zpse_store_u16le(esc + 4u, (uint16_t)1000u);
    len = dpcm1_build(b, 16u, 0x1000u, deltas, 0, 0u, esc, 1u);
    dpcm1_reject("escape position out of range", b, len);

    /* The exact byte count is part of the format: one byte short or long. */
    len = dpcm1_base(b, deltas);
    dpcm1_reject("exact need one byte short", b, len - 1u);
    dpcm1_reject("exact need one byte long", b, len + 1u);
    /* A header with no payload at all still has to decline cleanly. */
    dpcm1_reject("exact header only", b, 16u);
}

/* ---- MRD1 whole-frame form ("MRD1") -----------------------------------
 * Like DPCM1 there is no block-layer parser, so every case goes through
 * zpse_decompress_frame, which sniffs the magic and hands the bytes to
 * zpse_mrd_decode. The builder writes legal streams by hand, independent of
 * the encoder; each hostile case then mutates exactly one field. Wire layout:
 * a 24-byte header, verbatim prefix, a control stream (0x80 + u16 marks a
 * zero run, other bytes are signed deltas), then 6 bytes per escape (u32
 * delta index, i16 true delta). */
static uint32_t mrd_build(uint8_t *b, uint32_t n, uint16_t first,
                          const uint8_t *cs, uint32_t coff,
                          const uint8_t *esctab, uint32_t ne,
                          uint32_t nruns)
{
    uint32_t i;

    memset(b, 0, 24u + coff + 6u * ne);
    b[0] = 'M'; b[1] = 'R'; b[2] = 'D'; b[3] = '1';
    zpse_store_u32le(b + 4u, n);
    zpse_store_u16le(b + 8u, first);
    zpse_store_u16le(b + 10u, 0u);
    zpse_store_u32le(b + 12u, ne);
    zpse_store_u32le(b + 16u, coff);
    zpse_store_u32le(b + 20u, nruns);
    for (i = 0u; i < coff; i++) b[24u + i] = cs[i];
    for (i = 0u; i < 6u * ne; i++) b[24u + coff + i] = esctab[i];
    return 24u + coff + 6u * ne;
}

/* Header-only seed for cases whose declared counts are deliberately bogus;
 * the decoder must reject on the field check before it reads any payload. */
static void mrd_hdr(uint8_t *b, uint32_t n, uint32_t ne,
                    uint32_t coff, uint32_t nruns)
{
    memset(b, 0, 1024u);
    b[0] = 'M'; b[1] = 'R'; b[2] = 'D'; b[3] = '1';
    zpse_store_u32le(b + 4u, n);
    zpse_store_u16le(b + 8u, 1000u);
    zpse_store_u16le(b + 10u, 0u);
    zpse_store_u32le(b + 12u, ne);
    zpse_store_u32le(b + 16u, coff);
    zpse_store_u32le(b + 20u, nruns);
}

/* A corrupt MRD1 frame must get a negative status from the public frame path,
 * leave the caller's length untouched and write nothing into the output. */
static void mrd_reject(const char *what, const uint8_t *blk, uint32_t len)
{
    uint8_t obuf[256u + 16u];
    size_t olen = (size_t)0xDEADBEEFu;
    uint32_t i;
    int rc, clean;

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_frame(blk, len, obuf, 256u, &olen, dec_arena);
    clean = (rc < 0 && olen == (size_t)0xDEADBEEFu);
    for (i = 0u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) clean = 0;
    }
    CHECK(clean, "mrd1: %s -> clean status, no write past cap", what);
}

/* A few MRD1 rejections are found only while walking the control stream,
 * after the decoder has already emitted the leading sample(s). For those the
 * only bytes it may have touched are the first `lead`; everything after is
 * still the canary. This still bounds any write to the decoded prefix. */
static void mrd_reject_lead(const char *what, const uint8_t *blk,
                            uint32_t len, uint32_t lead)
{
    uint8_t obuf[256u + 16u];
    size_t olen = (size_t)0xDEADBEEFu;
    uint32_t i;
    int rc, clean;

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_frame(blk, len, obuf, 256u, &olen, dec_arena);
    clean = (rc < 0 && olen == (size_t)0xDEADBEEFu);
    for (i = lead; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) clean = 0;
    }
    CHECK(clean, "mrd1: %s -> clean status, no write past %u leading bytes",
          what, lead);
}

static void test_mrd1_hostile(void)
{
    uint8_t b[1024u];
    uint8_t obuf[256u + 16u];
    uint8_t ref[162u];
    uint8_t deltas[80u];
    uint32_t i, len;
    size_t olen;
    uint16_t v;
    int rc, ok;

    /* Valid baseline: 80 literal +10 deltas, n = 162, ns = 81. */
    for (i = 0u; i < 80u; i++) deltas[i] = 10u;
    len = mrd_build(b, 162u, 1000u, deltas, 80u, 0, 0u, 0u);
    CHECK(len == 104u, "mrd1: baseline literal stream is 104 bytes");
    v = 1000u;
    ref[0] = (uint8_t)v; ref[1] = (uint8_t)(v >> 8);
    for (i = 0u; i < 80u; i++) {
        v = (uint16_t)(v + 10u);
        ref[2u + 2u * i] = (uint8_t)v;
        ref[3u + 2u * i] = (uint8_t)(v >> 8);
    }
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_frame(b, len, obuf, 256u, &olen, dec_arena);
    ok = (rc == ZPSE_OK && olen == 162u && memcmp(obuf, ref, 162u) == 0);
    for (i = 162u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) ok = 0;
    }
    CHECK(ok, "mrd1: literal baseline decodes byte-exactly, canary intact");

    /* Valid baseline: one zero run of 80 samples (0x80, u16 80). */
    deltas[0] = 0x80u; deltas[1] = 80u; deltas[2] = 0u;
    len = mrd_build(b, 162u, 1000u, deltas, 3u, 0, 0u, 1u);
    CHECK(len == 27u, "mrd1: baseline RLE stream is 27 bytes");
    memset(ref, 0, sizeof ref);
    for (i = 0u; i < 81u; i++) {
        ref[2u * i] = (uint8_t)1000u;
        ref[2u * i + 1u] = (uint8_t)(1000u >> 8);
    }
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_frame(b, len, obuf, 256u, &olen, dec_arena);
    ok = (rc == ZPSE_OK && olen == 162u && memcmp(obuf, ref, 162u) == 0);
    for (i = 162u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) ok = 0;
    }
    CHECK(ok, "mrd1: RLE baseline decodes byte-exactly, canary intact");

    /* Wrong magic: the frame sniffer sees no MRD1 and falls through. */
    len = mrd_build(b, 162u, 1000u, deltas, 3u, 0, 0u, 1u);
    b[0] = 'X';
    mrd_reject("wrong magic", b, len);

    /* Truncation: magic only, a header short of 24 bytes, then the exact 24
     * bytes with no control stream (need is 27 for this stream). */
    len = mrd_build(b, 162u, 1000u, deltas, 3u, 0, 0u, 1u);
    mrd_reject("magic only", b, 4u);
    mrd_reject("short header", b, 15u);
    mrd_reject("exact header one short", b, 23u);
    mrd_reject("exact header only", b, 24u);

    /* RLE run longer than the samples left (65535 > ns-1 = 80), and one
     * shorter than the minimum run of 16. */
    mrd_hdr(b, 162u, 0u, 3u, 1u);
    b[24] = 0x80u; b[25] = 0xFFu; b[26] = 0xFFu;
    mrd_reject_lead("RLE run past n", b, 27u, 2u);
    mrd_hdr(b, 162u, 0u, 3u, 1u);
    b[24] = 0x80u; b[25] = 0x08u; b[26] = 0x00u;
    mrd_reject_lead("RLE run under minimum", b, 27u, 2u);

    /* A run reached after a literal still has to fit the deltas that remain:
     * ns-1 = 80, one literal uses didx 0, so 80 overshoots while 79 lands
     * exactly on the final sample. */
    mrd_hdr(b, 162u, 0u, 4u, 1u);
    b[24] = 0x01u; b[25] = 0x80u; zpse_store_u16le(b + 26u, 80u);
    mrd_reject_lead("RLE run crosses final sample", b, 28u, 4u);
    mrd_hdr(b, 162u, 0u, 4u, 1u);
    b[24] = 0x01u; b[25] = 0x80u; zpse_store_u16le(b + 26u, 79u);
    memset(obuf, 0xCC, sizeof obuf);
    olen = 0u;
    rc = zpse_decompress_frame(b, 28u, obuf, 256u, &olen, dec_arena);
    ref[0] = (uint8_t)1000u; ref[1] = (uint8_t)(1000u >> 8);
    v = 1001u; /* mrd_hdr's first sample, plus the +1 literal */
    for (i = 0u; i < 80u; i++) {
        ref[2u + 2u * i] = (uint8_t)v;
        ref[3u + 2u * i] = (uint8_t)(v >> 8);
    }
    ok = (rc == ZPSE_OK && olen == 162u && memcmp(obuf, ref, 162u) == 0);
    for (i = 162u; i < sizeof obuf; i++) {
        if (obuf[i] != 0xCCu) ok = 0;
    }
    CHECK(ok, "mrd1: RLE run ending at final sample decodes byte-exactly, canary intact");

    /* Escape position out of range, and two that are not strictly increasing. */
    mrd_hdr(b, 162u, 1u, 1u, 0u);
    b[24] = 0x01u;
    zpse_store_u32le(b + 25u, 99999u);
    zpse_store_u16le(b + 29u, 5u);
    mrd_reject("escape pos out of range", b, 31u);
    /* ns-1 = 80 is one past the last delta index (0..79). */
    mrd_hdr(b, 162u, 1u, 1u, 0u);
    b[24] = 0x01u;
    zpse_store_u32le(b + 25u, 80u);
    zpse_store_u16le(b + 29u, 5u);
    mrd_reject("escape pos == ns-1", b, 31u);
    mrd_hdr(b, 162u, 2u, 2u, 0u);
    b[24] = 0x01u; b[25] = 0x01u;
    zpse_store_u32le(b + 26u, 6u); zpse_store_u16le(b + 30u, 5u);
    zpse_store_u32le(b + 32u, 6u); zpse_store_u16le(b + 36u, 7u);
    mrd_reject("non-monotonic escapes", b, 38u);

    /* Escape count above ns-1 (= 80): rejected before any escape is read. */
    mrd_hdr(b, 162u, 100u, 1u, 0u);
    b[24] = 0x01u;
    mrd_reject("escape count > ns-1", b, 25u);

    /* Odd n, and a control stream too short to cover ns-1 deltas. */
    mrd_hdr(b, 161u, 0u, 1u, 0u);
    b[24] = 0x01u;
    mrd_reject("odd n", b, 25u);
    CHECK(zpse_mrd_decode(b, 25u, obuf, 161u) == ZPSE_MRD_DECLINE,
          "mrd1: decoder rejects odd n directly");
    mrd_hdr(b, 162u, 0u, 1u, 0u);
    b[24] = 0x01u;
    mrd_reject_lead("didx short of ns-1", b, 25u, 4u);

    /* Exact byte count: one byte short, one byte long, and a coff so large
     * the derived need cannot equal the supplied length. */
    len = mrd_build(b, 162u, 1000u, deltas, 3u, 0, 0u, 1u);
    mrd_reject("exact need one byte short", b, len - 1u);
    mrd_reject("exact need one byte long", b, len + 1u);
    mrd_hdr(b, 162u, 0u, 0xFFFFFFF0u, 0u);
    b[24] = 0x01u;
    mrd_reject("coff overflow", b, 25u);

    /* RLE token count above coff/3. */
    mrd_hdr(b, 162u, 0u, 3u, 100u);
    b[24] = 0x80u; b[25] = 0x10u; b[26] = 0x00u;
    mrd_reject("RLE count > coff/3", b, 27u);

    /* Prefix field: odd, past n-2 (=160), and exactly n-2 (which leaves only
     * one sample). Each is otherwise self-consistent with the supplied length. */
    mrd_hdr(b, 162u, 0u, 1u, 0u);
    zpse_store_u16le(b + 10u, 3u);
    b[27u] = 0x01u;
    mrd_reject("odd prefix", b, 28u);
    mrd_hdr(b, 162u, 0u, 1u, 0u);
    zpse_store_u16le(b + 10u, 162u);
    b[186u] = 0x01u;
    mrd_reject("prefix past n-2", b, 187u);
    mrd_hdr(b, 162u, 0u, 1u, 0u);
    zpse_store_u16le(b + 10u, 160u);
    b[184u] = 0x01u;
    mrd_reject("prefix leaves one sample", b, 185u);

    /* BUG 1: coff > ns-1. n=4 has ns-1=1 delta but the header names 100
     * literal bytes; the literal loops used to walk didx to ~100 and write
     * ~200 bytes into the four-byte output. */
    mrd_hdr(b, 4u, 0u, 100u, 0u);
    for (i = 24u; i < 124u; i++) b[i] = 0x01u;
    mrd_reject_lead("coff > ns-1", b, 124u, 2u);

    /* BUG 2: even with coff <= ns-1, a literal stretch can cross the final
     * sample. Here n=40 (ns-1=19), a 16-sample RLE token uses didx 0..15 and
     * the remaining eight literal bytes form one 8-lane SIMD store at didx 16
     * -- only three deltas remain, so the store used to write five samples
     * past the end. */
    mrd_hdr(b, 40u, 0u, 11u, 1u);
    b[24] = 0x80u; zpse_store_u16le(b + 25u, 16u);
    for (i = 27u; i < 35u; i++) b[i] = 0x01u;
    mrd_reject_lead("literal stretch past final sample", b, 35u, 34u);
    /* The tiny-frame shape: eight literals for a one-delta frame. */
    mrd_hdr(b, 4u, 0u, 8u, 0u);
    for (i = 24u; i < 32u; i++) b[i] = 0x01u;
    mrd_reject_lead("literal overflow, tiny frame", b, 32u, 2u);
}

/* A zero run longer than one u16 token is split by the encoder. A split at
 * exactly 65535 used to leave a 1..15 remainder, which the decoder rejects
 * (runs must be >=16), so the encoder now folds that tail into the previous
 * token. This round-trip is the regression for that long-run failure. */
static void test_mrd_long_run(void)
{
    enum { NS = 73729u, N = 2u * NS, RUN = 65543u }; /* RUN = 65535 + 8 */
    static uint8_t src[N];
    static uint8_t enc[16384u];
    static uint8_t dec[N + 16u];
    uint32_t i, elen = 0u;
    uint16_t v = 1000u;
    int rc, ok;

    /* RUN zero deltas followed by +100 deltas. The gate samples every 9th
     * delta (8192 points), sees ~909 of the +100 deltas and accepts. */
    zpse_store_u16le(src, v);
    for (i = 1u; i < NS; i++) {
        if (i > RUN) v = (uint16_t)(v + 100u);
        zpse_store_u16le(src + 2u * i, v);
    }

    rc = zpse_mrd_encode(src, N, enc, (uint32_t)sizeof enc, &elen);
    CHECK(rc == ZPSE_MRD_COMMIT, "mrd1: 65535+8 zero run encodes");
    if (rc != ZPSE_MRD_COMMIT) return;

    /* Two legal tokens: 65527 and 16, never a 1..15 tail. */
    CHECK(zpse_load_u32le(enc + 16u) == 8191u && /* 6 token bytes + 8185 literals */
          enc[24u] == 0x80u && zpse_load_u16le(enc + 25u) == 65527u &&
          enc[27u] == 0x80u && zpse_load_u16le(enc + 28u) == 16u,
          "mrd1: long run split into two legal tokens");

    memset(dec, 0xCC, sizeof dec);
    rc = zpse_mrd_decode(enc, elen, dec, N);
    ok = (rc == ZPSE_MRD_COMMIT && memcmp(dec, src, N) == 0);
    for (i = N; i < sizeof dec; i++) {
        if (dec[i] != 0xCCu) ok = 0;
    }
    CHECK(ok, "mrd1: 65535+8 zero run round-trips byte-exactly, canary intact");
}

int main(void) {
    uint32_t olen = 0;
    int rc, i;
    for (i = 0; i < (int)sizeof input; i++) input[i] = (uint8_t)(i & 0xFF);
    refresh();

    /* Truncations. */
    memcpy(work, comp, clen);
    rc = zpse_decompress_block(work, 31, out, sizeof out, &olen, dec_arena);
    CHECK(rc == ZPSE_TRUNCATED, "truncated header -> TRUNCATED");
    rc = zpse_decompress_block(work, clen - 1, out, sizeof out, &olen, dec_arena);
    CHECK(rc != 0, "truncated tail rejected");

    /* Header CRC corruption. */
    memcpy(work, comp, clen); work[10] ^= 0xFF;
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc == ZPSE_BAD_HEADER_CRC, "bad header CRC rejected");

    /* Payload corruption with intact header -> BAD_PAYLOAD_CRC, when the
     * stream asks for the checksum. Use incompressible input so the block
     * takes the RAW path, where any payload byte flip provably changes
     * only values (structure of a coded block could break earlier with a
     * different code). A page is asked for no checksum at all (see
     * ZPSE_BLOCK_HEADER_SIZE_SHORT: the field has nowhere to live), so the
     * stream here is longer than a page and the block keeps the long
     * header. */
    {
        uint32_t r = 0x12345u;
        int i;

        for (i = 0; i < (int)sizeof input; i++) {
            r = r * 1664525u + 1013904223u;
            input[i] = (uint8_t)(r >> 24);
        }
        memset(big, 0, sizeof big);
        for (i = 0; i < (int)sizeof big; i++) {
            r = r * 1664525u + 1013904223u;
            big[i] = (uint8_t)(r >> 24);
        }
        refresh_crc_big();
        big_comp[100] ^= 0xFF;
        rc = zpse_decompress_block(big_comp, big_len, big_out,
                                   (uint32_t)sizeof big_out, &olen,
                                   dec_arena);
        CHECK(rc == ZPSE_BAD_PAYLOAD_CRC, "bad payload CRC rejected");
        /* The default stream carries no checksum, so the same flip is a
         * value change the format never promised to detect -- that is the
         * trade the default makes, and it is asserted here so a future
         * "always emit the CRC" change cannot pass unnoticed. */
        refresh();
        memcpy(work, comp, clen);
        work[100] ^= 0xFF;
        rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
        CHECK(rc == ZPSE_OK && memcmp(out, input, sizeof input) != 0,
              "default stream has no payload CRC (flip undetected)");
        for (i = 0; i < (int)sizeof input; i++) input[i] = (uint8_t)(i & 0xFF);
        refresh();

        /* A page asked for a checksum gets a page's header anyway: the
         * decoded CRC has no field in sixteen bytes, so the flag and the
         * field are dropped rather than eight header bytes spent on them,
         * and the block says what it is ("no checksum"). */
        refresh_crc();
        CHECK(zpse_block_header_size(comp, clen) ==
                  (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT,
              "a page's checksummed block takes the short header");
        CHECK((comp[2] & (uint8_t)ZPSE_B_CRC) == 0u,
              "and carries no CRC flag");
        rc = zpse_decompress_block(comp, clen, out, sizeof out, &olen,
                                   dec_arena);
        CHECK(rc == ZPSE_OK && olen == sizeof input &&
                  memcmp(out, input, sizeof input) == 0,
              "and the page still round-trips");
    }

    /* Every block flag bit now names a form: bit 5 the raw-LZ form, bit 6
     * LZ plus tiles (LZ_TILE), bit 7 the Huffman-coded LZ form (HUF). So a
     * bit set on a payload that is not that form has to be refused by the
     * form's own envelope check rather than by a reserved-bit mask -- the
     * flag alone must never be enough to read a tile payload as commands.
     * Three cases follow: HUF on a tile payload, LZ on a tile payload, and
     * the invalid LZ_TILE|HUF combination. */
    /* The Huffman form on a tile payload: bit 7 alone must not be enough to
     * make a tile payload look like a Huffman-wrapped LZ stream. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 2u : 3u] |= 0x80;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc < 0, "Huffman flag on a tile payload rejected");

    /* The LZ form on a stream that is not one: the flag alone must not be
     * enough to read a tile payload as commands. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 2u : 3u] |= 0x20;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc < 0, "LZ flag on a tile payload rejected");

    /* The invalid LZ_TILE|HUF combination (bits 6 and 7): bit 7 is a fresh
     * single-bit encoding with no combined pattern, so a block claiming both
     * forms must be rejected rather than decoded as either one. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 2u : 3u] |= 0xC0;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc < 0, "invalid LZ_TILE|HUF flags on a tile payload rejected");

    /* Bad transform_id with repaired CRC -> BAD_TRANSFORM. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 3u : 12u] = 12;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc == ZPSE_BAD_TRANSFORM, "transform 12 rejected");

    /* Output too small -> OUTPUT_OVERFLOW. */
    rc = zpse_decompress_block(comp, clen, out, sizeof input - 1, &olen, dec_arena);
    CHECK(rc == ZPSE_OUTPUT_OVERFLOW, "small output -> OVERFLOW");

    /* Frame-level corruptions. */
    {
        static uint8_t fcomp[8192], fout[2048];
        size_t flen = 0, folen = 0;
        rc = zpse_compress_frame(input, sizeof input, fcomp, sizeof fcomp,
                                 &flen, 1024, enc_arena);
        CHECK(rc == 0, "frame compress baseline");
        memcpy(work, fcomp, flen); work[0] ^= 0xFF;
        rc = zpse_decompress_frame(work, flen, fout, sizeof fout, &folen, dec_arena);
        CHECK(rc == ZPSE_BAD_MAGIC, "frame bad magic");
        memcpy(work, fcomp, flen); work[4] ^= 0xFF;
        rc = zpse_decompress_frame(work, flen, fout, sizeof fout, &folen, dec_arena);
        CHECK(rc == ZPSE_BAD_VERSION, "frame bad version");
        memcpy(work, fcomp, flen); work[30] ^= 0xFF;
        rc = zpse_decompress_frame(work, flen, fout, sizeof fout, &folen, dec_arena);
        CHECK(rc == ZPSE_BAD_HEADER_CRC, "frame bad CRC");
    }

    /* Unit-level residual hostility. */
    {
        uint8_t exc[4]; uint8_t bad1[1] = {255};
        uint8_t dummy_pred[1] = {0};
        rc = zpse_residual_decode(0, bad1, 1, 1, dummy_pred, exc, exc);
        CHECK(rc == ZPSE_BAD_RESIDUAL, "residual 255 rejected (mode 0)");
        /* Base-M tail: M=3 -> G=20; K=1 valid word=1, invalid word=4. */
        uint8_t good[5] = {3, 1, 0, 0, 0};
        uint8_t badw[5] = {3, 4, 0, 0, 0};
        rc = zpse_residual_decode(3, good, 5, 1, dummy_pred, exc, exc);
        CHECK(rc == 0 && exc[0] == 1, "base-M valid tail accepted");
        rc = zpse_residual_decode(3, badw, 5, 1, dummy_pred, exc, exc);
        CHECK(rc == ZPSE_BAD_RESIDUAL, "base-M nonzero tail rejected");
        /* Signed delta zero unit. */
        uint8_t z4[2] = {0, 0x00};
        rc = zpse_residual_decode(4, z4, 2, 1, dummy_pred, exc, exc);
        CHECK(rc == ZPSE_BAD_RESIDUAL, "delta d=0 rejected");
        /* Palette index out of range: M=3 needs 2-bit indices, so the
         * value 3 is illegal. K=4 packs 4 indices into 1 byte; 0xFF makes
         * every index 3 -> rejected. (Power-of-two M cannot trigger OOR
         * since every representable value is < M.) */
        uint8_t pal[5] = {3, 10, 20, 30, 0xFF};
        uint8_t pexc[8];
        rc = zpse_residual_decode(2, pal, 5, 4, dummy_pred, pexc, pexc);
        CHECK(rc == ZPSE_BAD_RESIDUAL, "palette OOR index rejected");
    }

    /* Unit-level mask hostility: nonzero pad bits (Nt=9 -> 2 bytes). */
    {
        uint32_t w[8] = {0,0,0,0,0,0,0,0};
        uint8_t badmap[2] = {0x01, 0xFE}; /* bit0 set (K=1) + pad garbage */
        rc = zpse_mask_decode(1, badmap, 2, 9, 1, w, NULL);
        CHECK(rc == ZPSE_BAD_MASK, "bitmap pad bits rejected");
    }

    /* One-value page token (v1.1). The 16-byte header carries the fill byte
     * where a short header carries a transform id, so every other field has
     * one legal value. Each case below repairs the CRC where needed so the
     * field check is what rejects it, and checks the canary past the output
     * cap as well as the status. */
    {
        const uint32_t n = 256u;
        uint8_t tok[16];
        uint8_t obuf[256u + 16u];
        uint32_t olen;
        ZpseBlockHeader bh;
        int ok, j;

        /* Baseline: a valid token fills exactly n bytes, no more. */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        CHECK(zpse_block_token_build(&bh, tok, 16u) == ZPSE_OK &&
              bh.sync == ZPSE_BLOCK_SYNC_TOKEN &&
              bh.transform_id == 0x5Au && bh.uncompressed_size == n &&
              bh.segment_count == 1u && bh.transform_meta_size == 0u &&
              bh.compressed_block_size == 16u && bh.block_sequence_number == 7u,
              "token: build fills parsed header exactly");
        memset(obuf, 0xCC, sizeof obuf);
        olen = 0u;
        rc = zpse_decompress_block(tok, 16u, obuf, n, &olen, dec_arena);
        ok = rc == ZPSE_OK && olen == n && token_canary(obuf, n);
        for (j = 0; j < (int)n; j++) {
            if (obuf[j] != 0x5Au) { ok = 0; }
        }
        CHECK(ok, "token: valid page fills exactly n, canary intact");

        /* LAST is the one optional flag bit: the token is still legal and
         * still fills exactly n, then a cap one byte short must overflow
         * without writing anything. */
        token_make(tok, n, 0x00u,
                   (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET | ZPSE_B_LAST), 7u);
        memset(obuf, 0xCC, sizeof obuf);
        olen = 0u;
        rc = zpse_decompress_block(tok, 16u, obuf, n, &olen, dec_arena);
        ok = rc == ZPSE_OK && olen == n && token_canary(obuf, n);
        for (j = 0; j < (int)n; j++) {
            if (obuf[j] != 0x00u) { ok = 0; }
        }
        CHECK(ok, "token with LAST bit: accepted, fills exactly n");
        CHECK(zpse_block_token_build(&bh, tok, 16u) == ZPSE_OK &&
              (bh.block_flags & ZPSE_B_LAST) != 0u,
              "token with LAST bit: validator keeps LAST");
        memset(obuf, 0xCC, sizeof obuf);
        olen = 0xFFFFFFFFu;
        rc = zpse_decompress_block(tok, 16u, obuf, n - 1u, &olen, dec_arena);
        CHECK(rc == ZPSE_OUTPUT_OVERFLOW && obuf[0] == 0xCCu &&
              obuf[n - 1u] == 0xCCu && token_canary(obuf, n),
              "token with LAST bit: short cap -> OVERFLOW, no write");

        /* Bad CRC: flip a covered byte, do not repair. */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        tok[3] ^= 0xFFu;
        token_reject("bad CRC", tok, 16u, ZPSE_BAD_HEADER_CRC);

        /* Wrong flags: a forbidden form bit, CRC repaired. */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        tok[2] |= ZPSE_B_LZ;
        token_fix_crc(tok);
        token_reject("wrong flags", tok, 16u, ZPSE_BAD_HEADER);

        /* Wrong segment count. */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        tok[4] = 2u;
        token_fix_crc(tok);
        token_reject("wrong segcount", tok, 16u, ZPSE_BAD_HEADER);

        /* comp != 16. */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        zpse_store_u16le(tok + 8, 17u);
        token_fix_crc(tok);
        token_reject("comp != 16", tok, 16u, ZPSE_BAD_HEADER);

        /* Wrong sync: an unrecognised value (the block layer sees no header). */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        zpse_store_u16le(tok, (uint16_t)(ZPSE_BLOCK_SYNC_TOKEN + 1u));
        token_fix_crc(tok);
        token_reject("wrong sync", tok, 16u, ZPSE_BAD_HEADER);

        /* Truncated header: one byte short. */
        token_make(tok, n, 0x5Au, (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET), 7u);
        token_reject("truncated header", tok, 15u, ZPSE_TRUNCATED);

        /* Round-trip: a token emitted by the new serializer parses back to
         * the same fields, proving the serializer and validator agree. */
        memset(&bh, 0, sizeof bh);
        bh.block_flags = (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET | ZPSE_B_LAST);
        bh.transform_id = 0xA5u;
        bh.uncompressed_size = n;
        bh.block_sequence_number = 99u;
        memset(tok, 0xCC, sizeof tok);
        zpse_block_header_serialize_token(tok, &bh);
        CHECK(zpse_block_token_build(&bh, tok, 16u) == ZPSE_OK &&
              bh.transform_id == 0xA5u && bh.uncompressed_size == n &&
              bh.compressed_block_size == 16u &&
              bh.block_sequence_number == 99u &&
              (bh.block_flags & ZPSE_B_LAST) != 0u,
              "token: serialize -> build round-trips");
    }

    /* TEXT page form: the hostile-stream cases live in test_text_hostile. */
    test_text_hostile();

    /* DPCM1 whole-frame form: hostile fields and exact-length checks. */
    test_dpcm1_hostile();

    /* MRD1 whole-frame form: hostile fields and exact-length checks. */
    test_mrd1_hostile();
    test_mrd_long_run();

    /* FLAT page form (sync 0x50B9). A minimal RAW block is built by hand so
     * these cases do not depend on the encoder choosing this form. Each case
     * repairs the CRC where needed so the field check is what rejects it, and
     * checks the canary past the output cap as well as the status. */
    {
        uint8_t fb[16u + 32u];
        ZpseBlockHeader fh;
        uint32_t folen = 0u;
        int rc;

        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        CHECK(zpse_block_header_parse_flat(&fh, fb, sizeof fb) == ZPSE_OK &&
                  fh.sync == (uint16_t)ZPSE_BLOCK_SYNC_FLAT &&
                  fh.uncompressed_size == 32u &&
                  fh.compressed_block_size == sizeof fb &&
                  fh.block_sequence_number == 11u,
              "flat: baseline block parses");
        memset(out, 0, sizeof out);
        rc = zpse_decompress_block(fb, sizeof fb, out, 32u, &folen, dec_arena);
        CHECK(rc == ZPSE_OK && folen == 32u &&
                  memcmp(out, fb + 16u, 32u) == 0,
              "flat: baseline RAW block decodes verbatim");

        /* Bad CRC16: flip a covered byte, do not repair. */
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[3] ^= 0xFFu;
        flat_reject("bad CRC16", fb, sizeof fb, ZPSE_BAD_HEADER_CRC);

        /* Wrong flags: RESET is required, LAST the only optional bit. */
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[2] = 0u;
        flat_fix_crc(fb);
        flat_reject("missing RESET", fb, sizeof fb, ZPSE_BAD_HEADER);
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[2] |= ZPSE_B_RAW;
        flat_fix_crc(fb);
        flat_reject("RAW flag set", fb, sizeof fb, ZPSE_BAD_HEADER);
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[2] |= ZPSE_B_CRC;
        flat_fix_crc(fb);
        flat_reject("CRC flag set", fb, sizeof fb, ZPSE_BAD_HEADER);
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[2] |= ZPSE_B_HAS_META;
        flat_fix_crc(fb);
        flat_reject("HAS_META flag set", fb, sizeof fb, ZPSE_BAD_HEADER);

        /* Codec above the last one (3), and a nonzero reserved byte. */
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[3] = 4u;
        flat_fix_crc(fb);
        flat_reject("codec 4", fb, sizeof fb, ZPSE_BAD_TRANSFORM);
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        fb[5] = 1u;
        flat_fix_crc(fb);
        flat_reject("reserved byte nonzero", fb, sizeof fb, ZPSE_BAD_HEADER);

        /* Comp sizes: below the 16-byte header (parser) and past the bytes
         * actually supplied (length arithmetic in the decode path). */
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        zpse_store_u16le(fb + 8, 15u);
        flat_fix_crc(fb);
        flat_reject("comp below header", fb, sizeof fb, ZPSE_BAD_BLOCK_SIZE);
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        zpse_store_u16le(fb + 8, (uint16_t)(sizeof fb + 1u));
        flat_fix_crc(fb);
        flat_reject("comp past input", fb, sizeof fb, ZPSE_OK);

        /* Uncompressed size outside 1..4096. */
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        zpse_store_u16le(fb + 6, 0u);
        flat_fix_crc(fb);
        flat_reject("uncomp 0", fb, sizeof fb, ZPSE_BAD_BLOCK_SIZE);
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        zpse_store_u16le(fb + 6, 4097u);
        flat_fix_crc(fb);
        flat_reject("uncomp 4097", fb, sizeof fb, ZPSE_BAD_BLOCK_SIZE);

        /* Truncated header, then a truncated payload under an intact header. */
        flat_make(fb, 32u, ZPSE_FLAT_RAW, 0u, 11u);
        flat_reject("truncated header", fb, 15u, ZPSE_TRUNCATED);
        flat_reject("truncated payload", fb, sizeof fb - 1u, ZPSE_OK);
    }

    /* FLAT payload-length boundaries and exhaustive CRC-byte corruption. */
    test_flat_length_mismatch();
    test_flat_crc_bytes();

    /* Random garbage must error, never crash. */
    {
        uint32_t r = 0xDEADBEEFu;
        int bad = 0;
        for (i = 0; i < 50; i++) {
            int k;
            for (k = 0; k < 256; k++) {
                r = r * 1664525u + 1013904223u;
                work[k] = (uint8_t)(r >> 24);
            }
            rc = zpse_decompress_block(work, 256, out, sizeof out, &olen, dec_arena);
            if (rc >= 0) { bad = 1; break; }
            rc = zpse_decompress_frame(work, 256, out, sizeof out,
                                       &(size_t){0}, dec_arena);
            if (rc >= 0) { bad = 1; break; }
        }
        CHECK(!bad, "50x random garbage rejected");
    }

    printf(fails ? "HOSTILE: %d FAILURES\n" : "HOSTILE: ALL OK\n", fails);
    printf("HOSTILE checks=%d fails=%d\n", checks, fails);
    return fails != 0;
}
