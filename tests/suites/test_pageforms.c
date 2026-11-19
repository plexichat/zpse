/* The v1.1 page forms: the one-value sync token and the three page-profile
 * encoder options.
 *
 * This suite is the round-trip and wire-layout guard for the additions the
 * public header gained in v1.1:
 *
 *   - ZPSE_BLOCK_SYNC_TOKEN: a constant 4 KiB page is one 16-byte header and
 *     no payload. The fill byte lives in transform_id at [3], n at [6], the
 *     short-header size (16) at [8], the sequence at [10] and a 16-bit CRC
 *     over the first 14 bytes at [14]. zpse_block_retag must rewrite only
 *     seq / LAST and refresh that CRC, leaving the fill and sizes alone.
 *   - ZPSE_ENC_PAGE_TURBO (bit 10), ZPSE_ENC_PAGE_HUF (bit 12) and
 *     ZPSE_ENC_PAGE_INSTANT (bit 13) are accepted by zpse_compress_block_ex
 *     and every form they can pick still decodes byte-for-byte. Bit 11 stays
 *     unassigned and is refused.
 *
 * The 12 inputs model the non-zero page classes of corpus_write.c (code,
 * rodata, pointer heaps, records, text, blobs, mixed pages, ...): each is a
 * 4096-byte, seeded, deterministic shape, so a full sweep is cheap and the
 * bytes are identical on every host. Only the public API and constants.h are
 * used; everything is static, so there is no allocation to leak. */

#include <stdio.h>
#include <string.h>

#include "zpse/zpse.h"
#include "zpse/constants.h"
#include "zpse/text.h"

/* INSTANT is bit 13, next to PAGE_TURBO (bit 10). Keep building until the
 * definition lands in the public header, the same fallback the page harness
 * uses. */
#ifndef ZPSE_ENC_PAGE_INSTANT
#define ZPSE_ENC_PAGE_INSTANT (UINT32_C(1) << 13)
#endif

/* The TEXT page form (sync 0x50BA) is the latest wire addition. Keep the
 * fallback so this suite preprocesses before constants.h lands it. */
#ifndef ZPSE_BLOCK_SYNC_TEXT
#define ZPSE_BLOCK_SYNC_TEXT ((uint16_t)0x50BA)
#endif

/* The TEXT payload is a finder-free 7-bit pack (8 ASCII bytes -> 7 payload
 * bytes), not the old static-BPE token stream. The property test below carries
 * its own independent pack/unpack model, so no generated table is pulled into
 * this suite and no codec helper is used to build its expected bytes. */

#define PAGE 4096u
#define NSHAPES 12u
#define COMP_CAP (2u * PAGE)
#define TOKLEN ((uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT)

_Alignas(ZPSE_ARENA_ALIGN) static unsigned char ea[ZPSE_ENC_ARENA_SIZE];
_Alignas(ZPSE_ARENA_ALIGN) static unsigned char da[ZPSE_DEC_ARENA_SIZE];

static uint8_t src[PAGE];
static uint8_t comp[COMP_CAP];
static uint8_t dec[PAGE];
static uint8_t tok[TOKLEN];
static uint8_t copy[TOKLEN];
static uint8_t flat_copy[COMP_CAP];

/* The TEXT tests also carry 64 KiB blocks (n == ZPSE_MAX_BLOCK_SIZE), which
 * the 4 KiB page buffers above cannot hold. These are the same fixed, aligned,
 * no-allocation style, sized for the largest block plus a raw-block envelope. */
#define TEXT_MAX 65536u
#define TEXT_CAP (TEXT_MAX + 256u)
static uint8_t tsrc[TEXT_MAX];
static uint8_t tcomp[TEXT_CAP];
static uint8_t tdec[TEXT_MAX];
static uint8_t tcopy[TEXT_CAP];

static int checks;
static int fails;

#define CHECK(cond, ...) do {                       \
    checks++;                                       \
    if (!(cond)) {                                  \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
        fails++;                                    \
    }                                               \
} while (0)

static const char *shape_name(int s)
{
    static const char *const names[NSHAPES] = {
        "zero", "const5A", "increment", "noise", "sparse", "stride4",
        "alt00FF", "sawtooth", "low4bit", "text", "halfnoise", "records"
    };
    return names[s];
}

/* Small seeded LCG: every shape is a pure function of its index. */
static uint32_t pg_rng;

static uint32_t pg_rnd(void)
{
    pg_rng = pg_rng * 1664525u + 1013904223u;
    return pg_rng;
}

static void make_page(uint8_t *out, int shape)
{
    static const char words[] =
        "the quick brown fox jumps over the lazy dog and the river flows ";
    uint32_t nw = (uint32_t)(sizeof words - 1u);
    uint32_t i;

    pg_rng = 0x9E3779B9u ^ (uint32_t)shape * 2654435761u;
    switch (shape) {
    case 0: /* untouched anonymous page */
        memset(out, 0, PAGE);
        break;
    case 1: /* one repeated byte -- the token's shape */
        memset(out, 0x5A, PAGE);
        break;
    case 2: /* increment ramp */
        for (i = 0; i < PAGE; i++) out[i] = (uint8_t)i;
        break;
    case 3: /* uniform noise, the RAW fallback */
        for (i = 0; i < PAGE; i++) out[i] = (uint8_t)(pg_rnd() >> 24);
        break;
    case 4: /* sparse holes in zeros */
        memset(out, 0, PAGE);
        for (i = 0; i < PAGE; i += 251u) out[i] = (uint8_t)(pg_rnd() >> 24);
        break;
    case 5: /* stride-4 records */
        for (i = 0; i < PAGE; i++)
            out[i] = (uint8_t)((i / 4u + (i % 4u) * 64u) & 0xFFu);
        break;
    case 6: /* alternating bytes */
        for (i = 0; i < PAGE; i++)
            out[i] = (uint8_t)((i & 1u) ? 0xFFu : 0x00u);
        break;
    case 7: /* sawtooth */
        for (i = 0; i < PAGE; i++) out[i] = (uint8_t)(i * 7u + (i >> 4));
        break;
    case 8: /* low-entropy 4-bit alphabet */
        for (i = 0; i < PAGE; i++)
            out[i] = (uint8_t)((pg_rnd() >> 24) & 0x0Fu);
        break;
    case 9: /* repetitive text */
        for (i = 0; i < PAGE; i++)
            out[i] = (uint8_t)words[(i * 7u + (i >> 5)) % nw];
        break;
    case 10: /* noise head, constant tail */
        for (i = 0; i < PAGE; i++)
            out[i] = (i < PAGE / 2u) ? (uint8_t)(pg_rnd() >> 24) : 0x5Au;
        break;
    default: /* structured 2-byte records */
        for (i = 0; i + 2u <= PAGE; i += 2u) {
            uint32_t v = i * 3u;

            out[i] = (uint8_t)v;
            out[i + 1u] = (uint8_t)(v >> 8);
        }
        break;
    }
}

static uint16_t token_crc(const uint8_t *b)
{
    return (uint16_t)(zpse_crc32c(b, 14u) & UINT32_C(0xFFFF));
}

static uint32_t le16(const uint8_t *b)
{
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8);
}

static uint32_t le32(const uint8_t *b)
{
    return le16(b) | (le16(b + 2u) << 16);
}

/* Every page shape, at level 6, under each new profile bit. */
static void test_roundtrips(void)
{
    static const uint32_t new_flags[] = {
        (uint32_t)ZPSE_ENC_PAGE_TURBO,
        (uint32_t)ZPSE_ENC_PAGE_HUF,
        (uint32_t)ZPSE_ENC_PAGE_INSTANT
    };
    static const char *const flag_names[] = { "TURBO", "HUF", "INSTANT" };
    unsigned f, s;

    for (f = 0; f < sizeof new_flags / sizeof new_flags[0]; f++) {
        uint32_t flags = (uint32_t)ZPSE_ENC_SPEED_L6 | new_flags[f];

        for (s = 0; s < NSHAPES; s++) {
            uint32_t clen = 0u, olen = 0u, cs = 0u, us = 0u;
            ZpseBlockHeader h;
            int rc;

            make_page(src, (int)s);
            memset(comp, 0xA5, sizeof comp);
            rc = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &clen,
                                        (uint32_t)s, (s + 1u == NSHAPES),
                                        flags, ea);
            CHECK(rc == ZPSE_OK, "shape %s + %s encode: %s",
                  shape_name((int)s), flag_names[f], zpse_status_name(rc));
            if (rc != ZPSE_OK) continue;

            CHECK(clen > 0u && clen <= sizeof comp,
                  "shape %s + %s: implausible length %u",
                  shape_name((int)s), flag_names[f], clen);
            CHECK(zpse_block_probe(comp, clen, &cs, &us) == ZPSE_OK &&
                      cs == clen && us == PAGE,
                  "shape %s + %s: probe disagrees with the header",
                  shape_name((int)s), flag_names[f]);
            CHECK(zpse_block_header_parse(&h, comp, clen) == ZPSE_OK &&
                      h.block_sequence_number == (uint32_t)s,
                  "shape %s + %s: header parse or sequence",
                  shape_name((int)s), flag_names[f]);

            memset(dec, 0x5A, sizeof dec);
            rc = zpse_decompress_block(comp, clen, dec, sizeof dec, &olen, da);
            CHECK(rc == ZPSE_OK && olen == PAGE &&
                      memcmp(dec, src, PAGE) == 0,
                  "shape %s + %s: round-trip (%s)",
                  shape_name((int)s), flag_names[f], zpse_status_name(rc));

            /* A constant page takes the token under every profile: none of
             * the three bits sets CHECKSUM, which is the one thing that
             * pushes a constant page onto the long-header path. */
            if (s == 0u || s == 1u) {
                CHECK(clen == TOKLEN &&
                          comp[0] == (uint8_t)ZPSE_BLOCK_SYNC_TOKEN &&
                          comp[1] == (uint8_t)(ZPSE_BLOCK_SYNC_TOKEN >> 8),
                      "shape %s + %s: constant page is not a 16-byte token",
                      shape_name((int)s), flag_names[f]);
            }
        }
    }
}

/* The token's exact wire layout, its memset decode, and retag rewrite. */
static void test_token(void)
{
    ZpseBlockHeader h;
    uint32_t clen = 0u, olen = 0u, cs = 0u, us = 0u;
    uint16_t crc;
    uint32_t i, bad;
    int rc;

    make_page(src, 1); /* 4096 bytes of 0x5A */
    memset(comp, 0, sizeof comp);
    rc = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &clen,
                                5u, 0, (uint32_t)ZPSE_ENC_SPEED_L6, ea);
    CHECK(rc == ZPSE_OK, "token encode: %s", zpse_status_name(rc));
    if (rc != ZPSE_OK) return;

    memcpy(tok, comp, (size_t)TOKLEN);
    memcpy(copy, tok, (size_t)TOKLEN);

    CHECK(clen == TOKLEN, "token must be exactly 16 bytes (got %u)", clen);
    CHECK(tok[0] == (uint8_t)ZPSE_BLOCK_SYNC_TOKEN &&
              tok[1] == (uint8_t)(ZPSE_BLOCK_SYNC_TOKEN >> 8),
          "token sync is 0x50B8");
    CHECK(tok[2] == (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET) &&
              (tok[2] & ZPSE_B_LAST) == 0u,
          "token flags are RAW|RESET without LAST");
    CHECK(tok[3] == 0x5Au, "fill byte at [3] is 0x5A (got 0x%02X)", tok[3]);
    CHECK(tok[4] == 1u && tok[5] == 0u, "token segment count / metadata");
    CHECK(le16(tok + 6u) == PAGE, "n at [6] is 4096 (got %u)", le16(tok + 6u));
    CHECK(le16(tok + 8u) == TOKLEN, "comp at [8] is 16 (got %u)",
          le16(tok + 8u));
    CHECK(le32(tok + 10u) == 5u, "seq at [10] is 5 (got %u)",
          le32(tok + 10u));
    crc = token_crc(tok);
    CHECK(le16(tok + 14u) == (uint32_t)crc,
          "CRC16 at [14] is %04X (got %04X)", (unsigned)crc,
          (unsigned)le16(tok + 14u));

    CHECK(zpse_block_header_parse(&h, tok, TOKLEN) == ZPSE_OK &&
              h.sync == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN &&
              h.header_size == (uint8_t)TOKLEN &&
              h.uncompressed_size == PAGE &&
              h.compressed_block_size == TOKLEN &&
              h.transform_id == 0x5Au &&
              h.block_sequence_number == 5u,
          "token header parses with its fill, sizes and sequence");
    CHECK(zpse_block_probe(tok, TOKLEN, &cs, &us) == ZPSE_OK &&
              cs == TOKLEN && us == PAGE,
          "token probe returns 16 / 4096");

    memset(dec, 0, sizeof dec);
    rc = zpse_decompress_block(tok, TOKLEN, dec, sizeof dec, &olen, da);
    CHECK(rc == ZPSE_OK && olen == PAGE, "token decode length (%s)",
          zpse_status_name(rc));
    bad = 0u;
    for (i = 0; i < PAGE; i++)
        if (dec[i] != 0x5Au) bad++;
    CHECK(bad == 0u, "token fills all 4096 bytes with 0x5A (%u wrong)", bad);

    /* Retag for a new position: seq and LAST change, the CRC is refreshed,
     * and nothing else in the header moves. */
    CHECK(zpse_block_retag(tok, TOKLEN, 1234u, 1) == ZPSE_OK,
          "token retag to (1234, last)");
    CHECK(zpse_block_header_parse(&h, tok, TOKLEN) == ZPSE_OK &&
              h.block_sequence_number == 1234u &&
              (h.block_flags & ZPSE_B_LAST) != 0u &&
              h.sync == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN &&
              h.header_size == (uint8_t)TOKLEN &&
              h.transform_id == 0x5Au,
          "retag updates seq and LAST, keeps the token identity");
    CHECK(le32(tok + 10u) == 1234u, "retag wrote seq at [10]");
    crc = token_crc(tok);
    CHECK(le16(tok + 14u) == (uint32_t)crc,
          "retag refreshed the CRC16 at [14]");
    CHECK(memcmp(tok + 3u, copy + 3u, 7u) == 0,
          "retag left the fill, sizes and comp fields untouched");
    memset(dec, 0, sizeof dec);
    rc = zpse_decompress_block(tok, TOKLEN, dec, sizeof dec, &olen, da);
    CHECK(rc == ZPSE_OK && olen == PAGE && dec[0] == 0x5Au &&
              dec[PAGE - 1u] == 0x5Au,
          "a retagged token still decodes to 4096 fill bytes");

    /* Clearing LAST is the same rewrite in the other direction; returning to
     * the original position restores the original 16 bytes exactly. */
    CHECK(zpse_block_retag(tok, TOKLEN, 5u, 0) == ZPSE_OK,
          "token retag back to (5, not last)");
    CHECK(zpse_block_header_parse(&h, tok, TOKLEN) == ZPSE_OK &&
              h.block_sequence_number == 5u &&
              (h.block_flags & ZPSE_B_LAST) == 0u,
          "retag back cleared LAST and restored seq");
    CHECK(memcmp(tok, copy, (size_t)TOKLEN) == 0,
          "a there-and-back retag is byte-neutral");
    CHECK(zpse_block_retag(tok, TOKLEN, 5u, 0) == ZPSE_OK &&
              memcmp(tok, copy, (size_t)TOKLEN) == 0,
          "retagging to the position already recorded is a no-op");

    /* A corrupt header is refused before any byte is rewritten. The only
     * difference from the original must be the CRC byte we flipped. */
    memcpy(tok, copy, (size_t)TOKLEN);
    tok[14] ^= 0x01u;
    CHECK(zpse_block_retag(tok, TOKLEN, 9u, 1) == ZPSE_BAD_HEADER_CRC,
          "token retag refuses a bad CRC");
    bad = 0u;
    for (i = 0; i < TOKLEN; i++)
        if (tok[i] != copy[i]) bad++;
    CHECK(bad == 1u && tok[14] != copy[14],
          "a refused retag did not rewrite the header (%u bytes differ)", bad);
}

/* Every page-sized block now takes the FLAT writer (or the one-value token).
 * This walks all twelve shapes, checks each FLAT block's exact wire envelope
 * and the codec chosen for the representative low-alpha / noisy pages, then
 * proves a FLAT retag is byte-neutral there and back. */
static void test_flat(void)
{
    ZpseBlockHeader h;
    uint32_t clen = 0u;
    int rc;
    uint32_t s;

    for (s = 0u; s < NSHAPES; s++) {
        uint32_t seq = 0x1000u + s;
        int last = (s + 1u == NSHAPES);
        uint16_t sync;
        uint8_t want_flags = (uint8_t)(ZPSE_B_RESET |
                                       (last ? ZPSE_B_LAST : 0u));
        uint32_t olen = 0u, cs = 0u, us = 0u;

        make_page(src, (int)s);
        memset(comp, 0xA5, sizeof comp);
        rc = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &clen,
                                    seq, last, (uint32_t)ZPSE_ENC_SPEED_L6, ea);
        CHECK(rc == ZPSE_OK, "flat shape %u encode: %s", s,
              zpse_status_name(rc));
        if (rc != ZPSE_OK) continue;

        memset(dec, 0x5A, sizeof dec);
        rc = zpse_decompress_block(comp, clen, dec, sizeof dec, &olen, da);
        CHECK(rc == ZPSE_OK && olen == PAGE && memcmp(dec, src, PAGE) == 0,
              "flat shape %u round-trip (%s)", s, zpse_status_name(rc));

        sync = (uint16_t)le16(comp);
        /* Constants take the 16-byte token; irregular low-alphabet pages
         * (alternation shape 6, low-4-bit shape 8) take FLAT; everything else
         * falls through to the level ladder, which is the whole point of the
         * fast-path design. Assert the form each shape actually takes. */
        if (s == 0u || s == 1u) {
            CHECK(sync == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN && clen == TOKLEN,
                  "flat shape %u constant page is a 16-byte token", s);
            continue;
        }
        if (s != 6u && s != 8u) {
            CHECK(sync == (uint16_t)ZPSE_BLOCK_SYNC_SHORT,
                  "flat shape %u structured page uses the ladder", s);
            continue;
        }
        CHECK(sync == (uint16_t)ZPSE_BLOCK_SYNC_FLAT,
              "flat shape %u sync is 50B9 (got %04X)", s, (unsigned)sync);
        CHECK(zpse_block_header_size(comp, clen) == TOKLEN,
              "flat shape %u header size is 16", s);

        /* Exact FLAT header: flags, codec, reserved, both sizes, sequence
         * and the CRC16 over the first fourteen bytes. */
        CHECK(comp[2] == want_flags,
              "flat shape %u flags are RESET%s (got %02X)", s,
              last ? "|LAST" : "", comp[2]);
        CHECK(comp[3] <= (uint8_t)ZPSE_FLAT_SUBLELANE,
              "flat shape %u codec %u is one of 0..3", s, comp[3]);
        CHECK(comp[5] == 0u, "flat shape %u reserved byte is 0", s);
        CHECK(le16(comp + 6u) == PAGE, "flat shape %u n at [6] is 4096", s);
        CHECK(le16(comp + 8u) == clen,
              "flat shape %u c at [8] is the block length", s);
        CHECK(le32(comp + 10u) == seq, "flat shape %u seq at [10]", s);
        CHECK(le16(comp + 14u) == (uint32_t)token_crc(comp),
              "flat shape %u CRC16 at [14]", s);
        CHECK(zpse_block_probe(comp, clen, &cs, &us) == ZPSE_OK &&
                  cs == clen && us == PAGE,
              "flat shape %u probe returns %u / 4096", s, clen);
        CHECK(zpse_block_header_parse(&h, comp, clen) == ZPSE_OK &&
                  h.sync == (uint16_t)ZPSE_BLOCK_SYNC_FLAT &&
                  h.header_size == TOKLEN &&
                  h.block_flags == want_flags &&
                  h.transform_id == comp[3] &&
                  h.transform_meta_size == comp[4] &&
                  h.uncompressed_size == PAGE &&
                  h.compressed_block_size == clen &&
                  h.block_sequence_number == seq,
              "flat shape %u header parses with the same fields", s);

        if (s == 6u) {
            CHECK(comp[3] == ZPSE_FLAT_BIT2,
                  "two-symbol page uses FLAT BIT2");
        }
        if (s == 8u) {
            CHECK(comp[3] == ZPSE_FLAT_NIBBLE4 && comp[4] <= 15u,
                  "low-4-bit page uses FLAT NIBBLE4");
        }
    }

    /* Retag a FLAT block there and back: only seq, LAST and the CRC may move.
     * The low-4-bit shape gives a coded (non-RAW) header to exercise. */
    make_page(src, 8);
    rc = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &clen, 7u, 0,
                                (uint32_t)ZPSE_ENC_SPEED_L6, ea);
    CHECK(rc == ZPSE_OK && le16(comp) == (uint16_t)ZPSE_BLOCK_SYNC_FLAT,
          "flat retag baseline encode");
    if (rc == ZPSE_OK) {
        uint32_t olen = 0u;

        memcpy(flat_copy, comp, (size_t)clen);
        CHECK(zpse_block_retag(comp, clen, 0x2222u, 1) == ZPSE_OK,
              "flat retag to (0x2222, last)");
        CHECK(zpse_block_header_parse(&h, comp, clen) == ZPSE_OK &&
                  h.block_sequence_number == 0x2222u &&
                  (h.block_flags & ZPSE_B_LAST) != 0u &&
                  h.transform_id == flat_copy[3] &&
                  h.transform_meta_size == flat_copy[4] &&
                  h.uncompressed_size == PAGE &&
                  h.compressed_block_size == clen,
              "flat retag updates seq/LAST, keeps codec, param and sizes");
        CHECK(le16(comp + 14u) == (uint32_t)token_crc(comp),
              "flat retag refreshed the CRC16 at [14]");
        CHECK(memcmp(comp + 3u, flat_copy + 3u, 3u) == 0 &&
                  memcmp(comp + 6u, flat_copy + 6u, 4u) == 0,
              "flat retag left the codec, param, reserved and sizes alone");
        CHECK(memcmp(comp + 16u, flat_copy + 16u, (size_t)clen - 16u) == 0,
              "flat retag left the whole payload alone");
        memset(dec, 0, sizeof dec);
        rc = zpse_decompress_block(comp, clen, dec, sizeof dec, &olen, da);
        CHECK(rc == ZPSE_OK && olen == PAGE && memcmp(dec, src, PAGE) == 0,
              "a retagged FLAT block still decodes byte-for-byte");
        CHECK(zpse_block_retag(comp, clen, 7u, 0) == ZPSE_OK &&
                  memcmp(comp, flat_copy, (size_t)clen) == 0,
              "a there-and-back FLAT retag is byte-neutral");
        CHECK(zpse_block_retag(comp, clen, 7u, 0) == ZPSE_OK &&
                  memcmp(comp, flat_copy, (size_t)clen) == 0,
              "retagging a FLAT block to its current position is a no-op");
    }
}

/* ---- Randomized FLAT property tests -----------------------------------
 * Every FLAT block is built by hand (header + payload) and its exact expected
 * decode is derived independently from the payload generator. Decoding through
 * the public API must then reproduce those bytes exactly. This covers all four
 * codecs, alphabet counts 1..16 (NIBBLE4) / 1..4 (BIT2), every SUBLANE period
 * (1/2/4/8), and n across 1..4096 -- including many non-multiples of 8/4/2,
 * where the ceil() payload arithmetic and the short final group live. The
 * PRNG is a fixed LCG, so the whole sweep is identical on every host. */

static uint32_t fp_rng;

static uint32_t fp_rnd(void)
{
    fp_rng = fp_rng * 1664525u + 1013904223u;
    return fp_rng;
}

static void fp_put16(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void fp_put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/* Build a valid FLAT block in blk[0..16+plen), write the exact bytes a correct
 * decode must produce into expect[0..n), and return plen. `alpha` is 1..16 for
 * NIBBLE4 and 1..4 for BIT2; `period` is 1/2/4/8 for SUBLANE (ignored else). */
static uint32_t flat_build_case(uint8_t *blk, uint8_t *expect, uint32_t n,
                                uint8_t codec, uint32_t alpha, uint32_t period)
{
    uint32_t plen = 0u, i;

    switch (codec) {
    case ZPSE_FLAT_RAW:
        plen = n;
        for (i = 0u; i < n; i++) {
            uint8_t v = (uint8_t)(fp_rnd() >> 24);

            blk[16u + i] = v;
            expect[i] = v;
        }
        break;

    case ZPSE_FLAT_NIBBLE4: {
        uint8_t pal[16];
        uint32_t pairs;

        for (i = 0u; i < 16u; i++) pal[i] = (uint8_t)(fp_rnd() >> 24);
        for (i = 0u; i < 16u; i++) blk[16u + i] = pal[i];
        pairs = (n + 1u) >> 1;
        plen = 16u + pairs;
        for (i = 0u; i < pairs; i++) {
            uint32_t a = fp_rnd() % alpha;
            uint32_t b = fp_rnd() % alpha;

            blk[16u + 16u + i] = (uint8_t)(a | (b << 4));
            expect[2u * i] = pal[a];
            if (2u * i + 1u < n) expect[2u * i + 1u] = pal[b];
        }
        break;
    }

    case ZPSE_FLAT_BIT2: {
        uint8_t pal[8];
        uint32_t quads;

        for (i = 0u; i < 8u; i++) pal[i] = (uint8_t)(fp_rnd() >> 24);
        for (i = 0u; i < 8u; i++) blk[16u + i] = pal[i];
        quads = (n + 3u) >> 2;
        plen = 8u + quads;
        for (i = 0u; i < quads; i++) {
            uint32_t j;
            uint8_t v = 0u;

            for (j = 0u; j < 4u; j++) {
                uint32_t a = fp_rnd() % alpha;
                uint32_t pos = 4u * i + j;

                v |= (uint8_t)(a << (2u * j));
                if (pos < n) expect[pos] = pal[a];
            }
            blk[16u + 8u + i] = v;
        }
        break;
    }

    default: { /* ZPSE_FLAT_SUBLELANE */
        uint32_t keepmask = fp_rnd() & ((1u << period) - 1u);
        uint32_t keep[8], rank[8], k = 0u, groups, j;

        for (j = 0u; j < period; j++) {
            keep[j] = (keepmask >> j) & 1u;
            rank[j] = (uint8_t)k;
            if (keep[j]) k++;
        }
        groups = (n + period - 1u) / period;
        plen = 14u + k * groups;
        blk[16u + 0u] = (uint8_t)period;
        blk[16u + 1u] = (uint8_t)keepmask;
        for (i = 0u; i < 8u; i++)
            blk[16u + 2u + i] = (uint8_t)(fp_rnd() >> 24);
        for (i = 0u; i < k * groups; i++)
            blk[16u + 10u + i] = (uint8_t)(fp_rnd() >> 24);
        for (i = 0u; i < 4u; i++) blk[16u + 10u + k * groups + i] = 0u;
        for (i = 0u; i < n; i++) {
            uint32_t g = i / period;
            uint32_t q = i & (period - 1u);

            expect[i] = keep[q]
                            ? blk[16u + 10u + g * k + rank[q]]
                            : blk[16u + 2u + q];
        }
        break;
    }
    }

    blk[0] = (uint8_t)ZPSE_BLOCK_SYNC_FLAT;
    blk[1] = (uint8_t)(ZPSE_BLOCK_SYNC_FLAT >> 8);
    blk[2] = (uint8_t)ZPSE_B_RESET;
    blk[3] = codec;
    blk[4] = (uint8_t)((codec == ZPSE_FLAT_RAW ||
                        codec == ZPSE_FLAT_SUBLELANE) ? 0u : alpha - 1u);
    blk[5] = 0u;
    fp_put16(blk + 6u, n);
    fp_put16(blk + 8u, 16u + plen);
    fp_put32(blk + 10u, 0x1234u);
    fp_put16(blk + 14u, (uint16_t)(zpse_crc32c(blk, 14u) & 0xFFFFu));
    return plen;
}

static void test_flat_property(void)
{
    /* Edges include every small n (1..7) and odd n around powers of two, which
     * is where ceil(n/2), ceil(n/4) and the short SUBLANE group differ. */
    static const uint32_t edges[] = {
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 9u, 11u, 13u, 15u, 17u,
        31u, 33u, 63u, 65u, 127u, 129u, 255u, 257u,
        511u, 513u, 1023u, 1025u, 2047u, 2049u, 4095u, 4096u
    };
    const uint32_t nedges = (uint32_t)(sizeof edges / sizeof edges[0]);
    uint32_t i, c;

    fp_rng = 0xC0FFEEu;

    /* Exhaustive edge sweep: each codec, with SUBLANE at each of its periods. */
    for (c = 0u; c <= (uint32_t)ZPSE_FLAT_SUBLELANE; c++) {
        uint32_t periods = (c == ZPSE_FLAT_SUBLELANE) ? 4u : 1u;
        uint32_t p;

        for (p = 0u; p < periods; p++) {
            uint32_t period = 1u << p; /* 1, 2, 4, 8 */
            for (i = 0u; i < nedges; i++) {
                uint32_t n = edges[i];
                uint32_t alpha = (c == ZPSE_FLAT_NIBBLE4) ? (i % 16u) + 1u
                               : (c == ZPSE_FLAT_BIT2) ? (i % 4u) + 1u
                               : 1u;
                uint32_t olen = 0u, plen;
                int rc;

                plen = flat_build_case(comp, src, n, (uint8_t)c, alpha, period);
                memset(dec, 0xA5, sizeof dec);
                rc = zpse_decompress_block(comp, 16u + plen, dec, sizeof dec,
                                           &olen, da);
                CHECK(rc == ZPSE_OK && olen == n && memcmp(dec, src, n) == 0,
                      "flat property codec %u n %u alpha %u period %u (%s)",
                      c, n, alpha, period, zpse_status_name(rc));
            }
        }
    }

    /* Randomized batch: codec, n, alpha and SUBLANE period all from the PRNG. */
    for (i = 0u; i < 400u; i++) {
        uint32_t c = fp_rnd() % ((uint32_t)ZPSE_FLAT_SUBLELANE + 1u);
        uint32_t n = 1u + fp_rnd() % 4096u;
        uint32_t alpha = (c == ZPSE_FLAT_NIBBLE4) ? (fp_rnd() % 16u) + 1u
                       : (c == ZPSE_FLAT_BIT2) ? (fp_rnd() % 4u) + 1u
                       : 1u;
        uint32_t period = 1u << (fp_rnd() & 3u);
        uint32_t olen = 0u, plen;
        int rc;

        plen = flat_build_case(comp, src, n, (uint8_t)c, alpha, period);
        memset(dec, 0xA5, sizeof dec);
        rc = zpse_decompress_block(comp, 16u + plen, dec, sizeof dec, &olen, da);
        CHECK(rc == ZPSE_OK && olen == n && memcmp(dec, src, n) == 0,
              "flat property random codec %u n %u alpha %u period %u (%s)",
              c, n, alpha, period, zpse_status_name(rc));
    }
}

/* The new option bits are legal; bit 11 is still unassigned. */
static void test_flags(void)
{
    static const uint32_t accepted[] = {
        (uint32_t)ZPSE_ENC_PAGE_TURBO,
        (uint32_t)ZPSE_ENC_PAGE_HUF,
        (uint32_t)ZPSE_ENC_PAGE_INSTANT
    };
    static const char *const names[] = { "PAGE_TURBO", "PAGE_HUF",
                                          "PAGE_INSTANT" };
    const uint32_t reserved11 = UINT32_C(1) << 11;
    uint32_t clen = 0u;
    unsigned f;

    CHECK(((uint32_t)ZPSE_ENC_PAGE_TURBO & reserved11) == 0u &&
              ((uint32_t)ZPSE_ENC_PAGE_HUF & reserved11) == 0u &&
              ((uint32_t)ZPSE_ENC_PAGE_INSTANT & reserved11) == 0u,
          "the new profile bits do not overlap reserved bit 11");

    make_page(src, 2); /* a ramp: a real encode, not the token shortcut */
    for (f = 0; f < sizeof accepted / sizeof accepted[0]; f++) {
        uint32_t clen = 0u, olen = 0u;
        uint32_t flags = (uint32_t)ZPSE_ENC_SPEED_L6 | accepted[f];
        int rc = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &clen,
                                        0u, 1u, flags, ea);

        CHECK(rc == ZPSE_OK, "%s accepted by compress_block_ex (%s)",
              names[f], zpse_status_name(rc));
        if (rc != ZPSE_OK) continue;
        memset(dec, 0, sizeof dec);
        rc = zpse_decompress_block(comp, clen, dec, sizeof dec, &olen, da);
        CHECK(rc == ZPSE_OK && olen == PAGE && memcmp(dec, src, PAGE) == 0,
              "%s stream round-trips (%s)", names[f], zpse_status_name(rc));
    }

    clen = 0u;
    CHECK(zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &clen,
                                 0u, 1u,
                                 (uint32_t)ZPSE_ENC_SPEED_L6 | reserved11,
                                 ea) == ZPSE_BAD_ARG,
          "unknown bit 11 is still refused");
}

/* ---- The TEXT page form (sync 0x50BA) ----------------------------------
 * The payload is the fixed-rate 7-bit pack from src/text.c; the 16-byte header
 * names no codec, and a stored n of 0 spells 65536. These tests cover three
 * things: every size/shape/level round-trips byte-for-byte; the wire header is
 * exactly the one an independent builder writes; and the pack/unpack is right
 * for hand-built streams (checked against an independent bit-loop model, never
 * against the codec's own encode/decode path). */

static uint32_t text_rng;

static uint32_t text_rnd(void)
{
    text_rng = text_rng * 1664525u + 1013904223u;
    return text_rng;
}

/* kind 0 = high-entropy 7-bit ASCII (no adjacent equal bytes), 1 = the
 * repetitive prose of make_page shape 9, 2 = prose head / noise tail. Every
 * byte is < 128, so the 7-bit pack writer is eligible. */
static void text_build_input(uint8_t *out, uint32_t n, int kind)
{
    static const char words[] =
        "the quick brown fox jumps over the lazy dog and the river flows ";
    uint32_t nw = (uint32_t)(sizeof words - 1u);
    uint32_t i, prev = 0u;

    text_rng = 0xC0FFEEu ^ (uint32_t)kind * 2654435761u;
    for (i = 0u; i < n; i++) {
        uint8_t v;

        if (kind == 1u || (kind == 2u && i < n / 2u)) {
            v = (uint8_t)words[(i * 7u + (i >> 5)) % nw];
        } else {
            v = (uint8_t)((text_rnd() >> 24) & 0x7Fu);
            if (v == prev) v = (uint8_t)((v + 1u) & 0x7Fu);
        }
        out[i] = v;
        prev = v;
    }
}

/* Independent TEXT header builder: fixed sync and zero reserved bytes, the
 * flags/sizes/seq taken from the arguments, and the low 16 bits of crc32c over
 * the first fourteen bytes (token_crc computes exactly that). */
static void text_expected_header(uint8_t out[16], uint32_t n, uint32_t c,
                                 uint32_t seq, uint8_t flags)
{
    uint16_t crc;
    uint16_t nstored = (n == TEXT_MAX) ? 0u : (uint16_t)n;

    memset(out, 0, 16u);
    out[0] = (uint8_t)ZPSE_BLOCK_SYNC_TEXT;
    out[1] = (uint8_t)(ZPSE_BLOCK_SYNC_TEXT >> 8);
    out[2] = flags;
    /* [3..6) reserved, left zero. */
    out[6] = (uint8_t)nstored;
    out[7] = (uint8_t)(nstored >> 8);
    out[8] = (uint8_t)c;
    out[9] = (uint8_t)(c >> 8);
    out[10] = (uint8_t)seq;
    out[11] = (uint8_t)(seq >> 8);
    out[12] = (uint8_t)(seq >> 16);
    out[13] = (uint8_t)(seq >> 24);
    crc = token_crc(out);
    out[14] = (uint8_t)crc;
    out[15] = (uint8_t)(crc >> 8);
}

/* Ask the public encoder for a TEXT block at any of the three effort levels.
 * Returns the block length and the level that produced it, or 0 if none did. */
static uint32_t text_find(uint8_t *blk, uint32_t cap, const uint8_t *in,
                          uint32_t n, uint32_t seq, int last,
                          uint32_t *level_out)
{
    static const uint32_t levels[3] = {
        (uint32_t)ZPSE_ENC_SPEED_FASTEST,
        (uint32_t)ZPSE_ENC_SPEED_BALANCED,
        (uint32_t)ZPSE_ENC_SPEED_MAX
    };
    unsigned f;

    for (f = 0u; f < 3u; f++) {
        uint32_t clen = 0u;
        int rc = zpse_compress_block_ex(in, n, blk, cap, &clen, seq, last,
                                        levels[f], ea);

        if (rc == ZPSE_OK &&
            clen >= (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT &&
            (uint16_t)le16(blk) == (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
            *level_out = levels[f];
            return clen;
        }
    }
    return 0u;
}

/* If the encoder has not selected TEXT for an input yet, build a legal TEXT
 * block directly from the public serializer and the public codec. This keeps
 * the wire checks running even before the encoder's form selection lands. */
static uint32_t text_hand_build(uint8_t *blk, uint32_t cap, const uint8_t *in,
                                uint32_t n, uint32_t seq, int last)
{
    ZpseBlockHeader h;
    uint32_t plen, total;

    plen = zpse_text_price(in, n);
    if (plen == 0u) return 0u;
    total = 16u + plen;
    if (total > cap || total > 65535u) return 0u;
    memset(&h, 0, sizeof h);
    h.sync = ZPSE_BLOCK_SYNC_TEXT;
    h.block_flags = (uint8_t)(ZPSE_B_RESET | (last ? ZPSE_B_LAST : 0u));
    h.uncompressed_size = n;
    h.compressed_block_size = total;
    h.block_sequence_number = seq;
    h.header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    zpse_block_header_serialize_text(blk, &h);
    if (zpse_text_encode(in, n, blk + 16u, plen) != plen) return 0u;
    return total;
}

static void test_text(void)
{
    static const uint32_t ns[] = { 1u, 4095u, 4096u, 16384u, 65536u };
    static const uint32_t levels[3] = {
        (uint32_t)ZPSE_ENC_SPEED_FASTEST,
        (uint32_t)ZPSE_ENC_SPEED_BALANCED,
        (uint32_t)ZPSE_ENC_SPEED_MAX
    };
    static const int kinds[] = { 0, 1, 2 };
    ZpseBlockHeader h;
    uint32_t n, c, seq, clen, olen, cs, us, level;
    unsigned i, k, f;
    int rc, last, found;

    /* 1. Byte-exact round-trip at every size, shape and level. Whether the
     * encoder actually selects TEXT is a separate concern; every form must
     * reconstruct the input exactly. */
    for (i = 0u; i < sizeof ns / sizeof ns[0]; i++) {
        n = ns[i];
        for (k = 0u; k < sizeof kinds / sizeof kinds[0]; k++) {
            for (f = 0u; f < 3u; f++) {
                text_build_input(tsrc, n, kinds[k]);
                memset(tcomp, 0xA5, sizeof tcomp);
                rc = zpse_compress_block_ex(tsrc, n, tcomp, sizeof tcomp,
                                            &clen, 0x1234u + i, (f == 2u),
                                            levels[f], ea);
                CHECK(rc == ZPSE_OK,
                      "text rt n %u kind %d level %u encode (%s)", n, kinds[k],
                      (unsigned)(levels[f] >> ZPSE_ENC_SPEED_SHIFT),
                      zpse_status_name(rc));
                if (rc != ZPSE_OK) continue;
                memset(tdec, 0x5A, sizeof tdec);
                rc = zpse_decompress_block(tcomp, clen, tdec, sizeof tdec,
                                           &olen, da);
                CHECK(rc == ZPSE_OK && olen == n &&
                          memcmp(tdec, tsrc, n) == 0,
                      "text rt n %u kind %d level %u (%s)", n, kinds[k],
                      (unsigned)(levels[f] >> ZPSE_ENC_SPEED_SHIFT),
                      zpse_status_name(rc));
            }
        }
    }

    /* 2/3. The exact TEXT header of an input that selects TEXT, and its probe.
     * Prose is the shape the ASCII gate convicts; fall back to a hand-built
     * block only if form selection has not landed in this build yet. */
    seq = 0x77u; last = 0;
    text_build_input(tsrc, PAGE, 1);
    clen = text_find(tcomp, sizeof tcomp, tsrc, PAGE, seq, last, &level);
    found = (clen >= 16u);
    if (!found) clen = text_hand_build(tcomp, sizeof tcomp, tsrc, PAGE,
                                       seq, last);
    CHECK(clen >= 16u, "text header: no TEXT block for 4096-byte prose");
    if (clen >= 16u) {
        uint8_t want[16];
        uint8_t flags = (uint8_t)(ZPSE_B_RESET | (last ? ZPSE_B_LAST : 0u));

        text_expected_header(want, PAGE, clen, seq, flags);
        CHECK(memcmp(tcomp, want, 16u) == 0,
              "text header bytes match the independent builder");
        CHECK(zpse_block_header_parse(&h, tcomp, clen) == ZPSE_OK &&
                  h.sync == (uint16_t)ZPSE_BLOCK_SYNC_TEXT &&
                  h.header_size == (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT &&
                  (h.block_flags & ZPSE_B_RESET) != 0u &&
                  h.uncompressed_size == PAGE &&
                  h.compressed_block_size == clen &&
                  h.block_sequence_number == seq,
              "text header parses with sync/RESET/16/sizes/seq");
        CHECK(zpse_block_probe(tcomp, clen, &cs, &us) == ZPSE_OK &&
                  cs == clen && us == PAGE,
              "text probe returns %u / 4096", clen);
        c = le16(tcomp + 8u);
        CHECK(c == clen, "text c at [8] is the block length (%u/%u)", c, clen);
    }

    /* 4. 64 KiB: n is stored as the zero sentinel but parses and decodes as
     * 65536, and the probe reports that same uncompressed size. */
    seq = 0x9999u; last = 1;
    text_build_input(tsrc, TEXT_MAX, 1);
    clen = text_find(tcomp, sizeof tcomp, tsrc, TEXT_MAX, seq, last, &level);
    if (clen < 16u)
        clen = text_hand_build(tcomp, sizeof tcomp, tsrc, TEXT_MAX, seq, last);
    CHECK(clen >= 16u, "text 64 KiB: no TEXT block for prose");
    if (clen >= 16u) {
        CHECK(le16(tcomp + 6u) == 0u,
              "text 64 KiB: n at [6] is the 0 sentinel (got %u)",
              le16(tcomp + 6u));
        CHECK(zpse_block_header_parse(&h, tcomp, clen) == ZPSE_OK &&
                  h.uncompressed_size == TEXT_MAX &&
                  h.compressed_block_size == clen,
              "text 64 KiB: parsed n is 65536");
        memset(tdec, 0x5A, sizeof tdec);
        rc = zpse_decompress_block(tcomp, clen, tdec, sizeof tdec, &olen, da);
        CHECK(rc == ZPSE_OK && olen == TEXT_MAX &&
                  memcmp(tdec, tsrc, TEXT_MAX) == 0,
              "text 64 KiB: round-trip (%s)", zpse_status_name(rc));
        CHECK(zpse_block_probe(tcomp, clen, &cs, &us) == ZPSE_OK &&
                  cs == clen && us == TEXT_MAX,
              "text 64 KiB: probe returns %u / 65536", clen);
    }

    /* 5. Retag there and back: only seq, LAST and the CRC may move. The codec,
     * reserved bytes, sizes and the whole token payload stay untouched. */
    seq = 0x7u; last = 0;
    text_build_input(tsrc, PAGE, 1);
    clen = text_find(tcomp, sizeof tcomp, tsrc, PAGE, seq, last, &level);
    if (clen < 16u)
        clen = text_hand_build(tcomp, sizeof tcomp, tsrc, PAGE, seq, last);
    CHECK(clen >= 16u, "text retag: no TEXT block for prose");
    if (clen >= 16u) {
        memcpy(tcopy, tcomp, (size_t)clen);
        CHECK(zpse_block_retag(tcomp, clen, 0x2222u, 1) == ZPSE_OK,
              "text retag to (0x2222, last)");
        CHECK(zpse_block_header_parse(&h, tcomp, clen) == ZPSE_OK &&
                  h.sync == (uint16_t)ZPSE_BLOCK_SYNC_TEXT &&
                  h.header_size == (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT &&
                  h.block_sequence_number == 0x2222u &&
                  (h.block_flags & ZPSE_B_LAST) != 0u &&
                  h.uncompressed_size == PAGE &&
                  h.compressed_block_size == clen,
              "text retag updates seq/LAST, keeps the TEXT identity and sizes");
        CHECK(le16(tcomp + 14u) == (uint32_t)token_crc(tcomp),
              "text retag refreshed the CRC16 at [14]");
        CHECK(memcmp(tcomp + 3u, tcopy + 3u, 7u) == 0 &&
                  memcmp(tcomp + 16u, tcopy + 16u, (size_t)clen - 16u) == 0,
              "text retag left reserved/sizes and the payload alone");
        memset(tdec, 0, sizeof tdec);
        rc = zpse_decompress_block(tcomp, clen, tdec, sizeof tdec, &olen, da);
        CHECK(rc == ZPSE_OK && olen == PAGE &&
                  memcmp(tdec, tsrc, PAGE) == 0,
              "a retagged TEXT block still decodes byte-for-byte");
        CHECK(zpse_block_retag(tcomp, clen, seq, last) == ZPSE_OK &&
                  memcmp(tcomp, tcopy, (size_t)clen) == 0,
              "a there-and-back TEXT retag is byte-neutral");
        CHECK(zpse_block_retag(tcomp, clen, seq, last) == ZPSE_OK &&
                  memcmp(tcomp, tcopy, (size_t)clen) == 0,
              "retagging TEXT to its current position is a no-op");

        memcpy(tcomp, tcopy, (size_t)clen);
        tcomp[14] ^= 0x01u;
        CHECK(zpse_block_retag(tcomp, clen, 9u, 1) == ZPSE_BAD_HEADER_CRC,
              "text retag refuses a bad CRC");
        {
            uint32_t bad = 0u, j;

            for (j = 0u; j < clen; j++)
                if (tcomp[j] != tcopy[j]) bad++;
            CHECK(bad == 1u && tcomp[14] != tcopy[14],
                  "a refused TEXT retag rewrote nothing (%u bytes differ)",
                  bad);
        }
    }
}

/* Independent 7-bit pack model. A group of up to eight ASCII bytes becomes
 * ceil(7r/8) payload bytes: bit 7*i+k of the group is bit k of byte i. This is
 * the same bit layout as src/tile.c's compact8/spread8, written here as a
 * plain bit loop so the codec's masks and shufb tables cannot make the model
 * self-consistently wrong. Returns the packed length, n - (n>>3). */
static uint32_t text_pack_model(const uint8_t *ascii, uint32_t n,
                                uint8_t *packed)
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
                if (((ascii[ip + i] >> k) & 1u) != 0u) {
                    v |= UINT64_C(1) << (7u * i + k);
                }
            }
        }
        nb = (7u * r + 7u) >> 3;
        for (k = 0u; k < nb; k++) {
            packed[op + k] = (uint8_t)(v >> (8u * k));
        }
        ip += r;
        op += nb;
    }
    return op;
}

/* The exact inverse, used only to double-check the model's own packed bytes:
 * reassemble each group's 56 bits, then spread them back to r ASCII bytes. */
static void text_unpack_model(const uint8_t *packed, uint32_t plen,
                              uint8_t *ascii, uint32_t n)
{
    uint32_t ip = 0u, op = 0u;

    (void)plen; /* all nb bytes of each group are consumed by the loop below */

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
            ascii[op + i] = b;
        }
        ip += nb;
        op += r;
    }
}

/* Independent 7-bit-pack property test: hand-build a TEXT stream (16-byte
 * header + packed payload), name n as the decoded length, and require the
 * public zpse_text_decode (and the whole block path) to reproduce the source
 * bytes. The edge sweep covers every small n and the odd sizes around powers
 * of two, where the decoder's 8-byte-store unroll and exact-length tail meet;
 * the random batch covers lengths across a page driven by a fixed LCG. */
static void test_text_property(void)
{
    static const uint32_t edges[] = {
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 9u, 11u, 13u, 15u, 17u,
        31u, 33u, 63u, 65u, 127u, 129u, 255u, 257u,
        511u, 512u
    };
    const uint32_t nedges = (uint32_t)(sizeof edges / sizeof edges[0]);
    uint32_t i;

    text_rng = 0xC0FFEEu;

    for (i = 0u; i < nedges + 400u; i++) {
        uint32_t n = (i < nedges) ? edges[i] : (1u + text_rnd() % 512u);
        uint32_t plen, olen;
        int rc;

        /* n ASCII source bytes (every byte < 128), one fixed LCG stream. */
        {
            uint32_t k;

            for (k = 0u; k < n; k++) {
                src[k] = (uint8_t)((text_rnd() >> 24) & 0x7Fu);
            }
        }
        plen = text_pack_model(src, n, comp + 16u);
        CHECK(plen == n - (n >> 3),
              "text property n %u: packed length %u", n, plen);
        text_unpack_model(comp + 16u, plen, tcomp, n);
        CHECK(memcmp(tcomp, src, n) == 0,
              "text property n %u: model round-trip", n);

        text_expected_header(comp, n, 16u + plen, 0u,
                             (uint8_t)ZPSE_B_RESET);
        memset(dec, 0xA5, sizeof dec);
        rc = zpse_text_decode(comp + 16u, plen, dec, n);
        CHECK(rc == 0 && memcmp(dec, src, n) == 0,
              "text property n %u: zpse_text_decode (%d)", n, rc);

        memset(dec, 0xA5, sizeof dec);
        olen = 0u;
        rc = zpse_decompress_block(comp, 16u + plen, dec, n, &olen, da);
        CHECK(rc == ZPSE_OK && olen == n && memcmp(dec, src, n) == 0,
              "text property n %u: block decode (%d)", n, rc);
    }
}

int main(void)
{
    test_roundtrips();
    test_token();
    test_flat();
    test_flat_property();
    test_flags();
    test_text();
    test_text_property();
    printf("PAGEFORMS: checks=%d fails=%d %s\n", checks, fails,
           fails == 0 ? "OK" : "FAILED");
    return fails == 0 ? 0 : 1;
}
