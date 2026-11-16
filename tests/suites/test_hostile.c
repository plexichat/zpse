/* Hostile-bitstream tests: every corruption must yield a clean status
 * code, never a crash or silent acceptance. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"

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

    /* Every block flag bit now names a form: 5 the LZ form, 6 LZ plus
     * tiles, 7 the fast LZ form. So a bit set on a payload that is not that
     * form has to be refused by the form's own envelope check rather than by
     * a reserved-bit mask -- the flag alone must never be enough to read a
     * tile payload as commands. Bit 7 first, then the LZ bit. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 2u : 3u] |= 0x80;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc < 0, "fast LZ flag on a tile payload rejected");

    /* The LZ form on a stream that is not one: the flag alone must not be
     * enough to read a tile payload as commands. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 2u : 3u] |= 0x20;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc < 0, "LZ flag on a tile payload rejected");

    /* The entropy-coded form (flag pattern 110) on a stream that is not one.
     * Its envelope is the LZ forms' own, so a tile payload carrying the flag
     * is refused by the transform check before the entropy decoder sees a
     * byte of it -- and the decoder's own header is what has to agree with
     * the payload length, not the other way round. */
    memcpy(work, comp, clen);
    work[hostile_short(work, clen) ? 2u : 3u] |= 0xC0;
    hostile_fix_crc(work, clen);
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc < 0, "Huffman flag on a tile payload rejected");

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
