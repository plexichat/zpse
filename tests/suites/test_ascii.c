/* ASCII pack form (tile mask mode ZPSE_M_ASCII_PACK).
 *
 * The form stores a tile's bytes with the eighth bit dropped: seven bits a
 * byte, no predictor, no mask, no residual codec. It is the one tile form
 * whose decode is not a prediction, so the things that can go wrong are
 * specific: a lossy pack, a decoded length that disagrees with the payload,
 * a lane history that stops matching what the next tile expects, and a
 * record shape the validator lets through with the wrong sizes.
 *
 * This suite pins all of those: the bit pack and its inverse at every tile
 * length, the exact header the packer writes, the round trip through
 * zpse_tile_decode, one-byte mutations of each header field (every one must
 * be refused, not decoded into something), and the encoder's use of the
 * form at each effort level on ASCII, mixed and non-ASCII blocks. It also
 * pins the block-level TEXT form: on prose at and above the ASCII gate's
 * minimum block size the block's sync must be ZPSE_BLOCK_SYNC_TEXT, while one
 * byte at or above 0x80 anywhere in the block must keep that sync from being
 * selected, at every level and at the 16-bit n sentinel.
 *
 * Exit 0 on success, 1 on any failure.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "zpse/zpse.h"
#include "zpse/predictor.h"
#include "zpse/tile.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[ZPSE_MAX_BLOCK_SIZE];
static uint8_t comp[ZPSE_MAX_BLOCK_SIZE * 2];
static uint8_t out[ZPSE_MAX_BLOCK_SIZE];
/* The frame case is four blocks, so it needs its own buffers. */
static uint8_t frin[200000];
static uint8_t frcomp[220000];
static uint8_t frout[200000];
static uint8_t rec[4096];
static uint8_t back[512];
static uint8_t comb[8192];

static long checks;
static long failures;

#define CHECK(cond, name)                                                  \
    do {                                                                   \
        checks++;                                                          \
        if (!(cond)) {                                                     \
            failures++;                                                    \
            printf("FAIL %s (line %d)\n", (name), __LINE__);               \
        }                                                                  \
    } while (0)

/* Same check with a printf-style message, for the sweeps whose failure must
 * name the size and level that caused it. */
#define CHECKF(cond, fmt, ...)                                             \
    do {                                                                   \
        checks++;                                                          \
        if (!(cond)) {                                                     \
            failures++;                                                    \
            printf("FAIL " fmt " (line %d)\n", __VA_ARGS__, __LINE__);     \
        }                                                                  \
    } while (0)

/* A readable, deterministic ASCII tile: printable, with a newline every 32
 * bytes so the tile is not one repeated byte. */
static void ascii_fill(uint8_t *p, uint16_t n, uint32_t seed)
{
    uint16_t i;

    for (i = 0; i < n; i++) {
        uint32_t r = seed + (uint32_t)i * 2654435761u;

        p[i] = (uint8_t)((i % 32u) == 31u ? '\n' : (uint8_t)(' ' + (r >> 24) % 95u));
    }
}

/* Deterministic English-like prose for the block-level TEXT test. A
 * first-order Markov walk over a fixed prose corpus is rich in repeated
 * digrams yet never repeats a long substring for the LZ match finder. That
 * is the shape the ASCII gate convicts and the fixed-rate 7-bit TEXT pack
 * wins: every byte < 0x80, no low alphabet, no easy prediction. */
static void prose_fill(uint8_t *p, uint32_t n, uint32_t seed)
{
    static const char corpus[] =
        "The quick brown fox jumps over the lazy dog. This is a sentence of "
        "English prose with common words and ordinary punctuation. The "
        "encoder must decide which form to use for each block of bytes. When "
        "the input is plain text the static table can replace frequent pairs "
        "of characters with single tokens. Each token expands to two or more "
        "bytes on decode, and the result is a smaller stream than the "
        "original. There is no predictor and no mask and no residual codec "
        "in this form, only the byte stream that the table describes. The "
        "header is short because the block is a page. A reader finds the "
        "extent of every block by reading one short header at a time, and "
        "checks the low sixteen bits of a checksum before it trusts any of "
        "the fields inside. Nothing in the table depends on the level that "
        "the encoder was asked to use, so the same bytes must come out at "
        "every effort from fastest to max. Deterministic input and a fixed "
        "seed keep the test repeatable on every machine and every build, "
        "which matters for a format that promises the same compressed stream "
        "on every platform that implements it correctly.";
    uint8_t next[128][128];
    uint8_t cnt[128];
    uint32_t clen = (uint32_t)(sizeof corpus - 1u);
    uint32_t prev = (uint32_t)'T';
    uint32_t r = seed;
    uint32_t i;

    memset(next, 0, sizeof next);
    memset(cnt, 0, sizeof cnt);
    for (i = 0u; i + 1u < clen; i++) {
        uint32_t c = (uint8_t)corpus[i];
        uint32_t d = (uint8_t)corpus[i + 1u];

        if (c < 128u && d < 128u && cnt[c] < 128u) {
            next[c][cnt[c]++] = (uint8_t)d;
        }
    }
    for (i = 0u; i < n; i++) {
        if ((i % 80u) == 79u) {
            p[i] = '\n';
            prev = (uint32_t)'T';
        } else {
            if (cnt[prev] == 0u) {
                prev = (uint32_t)'T';
            }
            r = r * 1664525u + 1013904223u;
            p[i] = next[prev][(r >> 16) % cnt[prev]];
            prev = p[i];
        }
    }
}

static void lane_init(ZpseLaneState *lanes, uint8_t literal)
{
    memset(lanes, 0, sizeof *lanes);
    zpse_lane_reset_hist(lanes, literal);
}

/* Pack/unpack and decode, for every tile length. */
static void test_lengths(void)
{
    uint16_t nt;

    for (nt = 1; nt <= 256; nt++) {
        int n;
        int got;
        ZpseLaneState lanes;

        ascii_fill(in, nt, (uint32_t)nt * 7919u);
        CHECK(zpse_tile_is_ascii(in, nt) == 1, "ascii detected");
        CHECK(zpse_tile_ascii_len(nt) == 8u + (uint32_t)nt - (uint32_t)(nt >> 3),
              "ascii length formula");
        n = zpse_tile_pack_ascii(in, nt, rec, sizeof rec);
        CHECK(n == (int)zpse_tile_ascii_len(nt), "pack length");
        /* Header: mode 7, CONST_ZERO, no exceptions, empty mask, packed
         * payload. */
        CHECK(zpse_tile_mask_mode(rec[0]) == 7, "pack flags mode");
        CHECK(zpse_tile_res_mode(rec[0]) == 0, "pack flags res mode");
        CHECK((rec[0] & ZPSE_TILE_FLAG_RESET) != 0, "pack reset bit");
        CHECK(rec[1] == (uint8_t)ZPSE_P_CONST_ZERO, "pack predictor");
        CHECK(rec[2] == (uint8_t)(nt - 1u), "pack length byte");
        CHECK(rec[3] == 0u, "pack exception count");
        CHECK(rec[4] == 0u && rec[5] == 0u, "pack empty mask");
        CHECK((uint32_t)(rec[6] | (rec[7] << 8)) ==
                  (uint32_t)nt - (uint32_t)(nt >> 3), "pack payload size");

        lane_init(&lanes, 'x');
        got = zpse_tile_decode(&lanes, rec, (uint32_t)n, 'x', 1, in, 0, back,
                                sizeof back, comb);
        CHECK(got == (int)nt, "decode length");
        CHECK(memcmp(back, in, nt) == 0, "decode bytes");
        /* The lane history a stateless-family tile of the same bytes
         * leaves: the last byte of each lane, and the one 8 back. */
        if (nt == 256u) {
            CHECK(memcmp(lanes.last, in + 248, 8) == 0, "history last");
            CHECK(memcmp(lanes.second_last, in + 240, 8) == 0,
                  "history second");
        }
    }
}

/* Header mutations: every one must be refused. */
static void test_rejects(void)
{
    const uint16_t nt = 256u;
    int n;
    ZpseLaneState lanes;

    ascii_fill(in, nt, 12345u);
    n = zpse_tile_pack_ascii(in, nt, rec, sizeof rec);

    /* High bit set: not ASCII at all. */
    in[100] = 0xC3u;
    CHECK(zpse_tile_is_ascii(in, nt) == 0, "high bit detected");
    ascii_fill(in, nt, 12345u);

    {
        /* Truncated record. */
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, rec, (uint32_t)n - 1u, 0, 1, in, 0,
                               back, sizeof back, comb) < 0,
              "truncated record refused");
    }
    {
        /* A mask payload where the form has none. */
        uint8_t bad[512];

        memcpy(bad, rec, (size_t)n);
        bad[5] = 1u;
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, bad, (uint32_t)n, 0, 1, in, 0, back,
                               sizeof back, comb) < 0,
              "nonzero mask payload refused");
    }
    {
        /* Residual payload one byte short. */
        uint8_t bad[512];

        memcpy(bad, rec, (size_t)n);
        bad[6] = (uint8_t)(bad[6] - 1u);
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, bad, (uint32_t)n, 0, 1, in, 0, back,
                               sizeof back, comb) < 0,
              "short payload refused");
    }
    {
        /* Exception count without a mask. */
        uint8_t bad[512];

        memcpy(bad, rec, (size_t)n);
        bad[3] = 1u;
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, bad, (uint32_t)n, 0, 1, in, 0, back,
                               sizeof back, comb) < 0,
              "exception count refused");
    }
    {
        /* A predictor other than CONST_ZERO: the form predicts nothing, so
         * any other id is a malformed record. */
        uint8_t bad[512];

        memcpy(bad, rec, (size_t)n);
        bad[1] = (uint8_t)ZPSE_P_LANE_PREVIOUS;
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, bad, (uint32_t)n, 0, 1, in, 0, back,
                               sizeof back, comb) < 0,
              "wrong predictor refused");
    }
    {
        /* A residual mode the form does not use. */
        uint8_t bad[512];

        memcpy(bad, rec, (size_t)n);
        bad[0] = zpse_tile_flags_make(7u, 1u, 1);
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, bad, (uint32_t)n, 0, 1, in, 0, back,
                               sizeof back, comb) < 0,
              "wrong residual mode refused");
    }
    {
        /* The reserved tile flag bit. */
        uint8_t bad[512];

        memcpy(bad, rec, (size_t)n);
        bad[0] |= ZPSE_TILE_FLAG_RESERVED_BIT;
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, bad, (uint32_t)n, 0, 1, in, 0, back,
                               sizeof back, comb) < 0,
              "reserved bit refused");
    }
    {
        /* An output buffer that cannot hold the tile. */
        lane_init(&lanes, 0);
        CHECK(zpse_tile_decode(&lanes, rec, (uint32_t)n, 0, 1, in, 0, back,
                               255u, comb) < 0,
              "short output refused");
    }
}

/* The encoder's use of the form, per level, on text, mixed and binary
 * blocks. Every block must round-trip; text must come out below its own
 * size (the form is 1.143:1 before framing) and binary near-random data
 * must not. */
static void test_blocks(void)
{
    static const uint32_t levels[9] = {
        ZPSE_ENC_SPEED_FASTEST, ZPSE_ENC_SPEED_L2,
        ZPSE_ENC_SPEED_FASTER, ZPSE_ENC_SPEED_L4,
        ZPSE_ENC_SPEED_BALANCED, ZPSE_ENC_SPEED_L6,
        ZPSE_ENC_SPEED_L7, ZPSE_ENC_SPEED_L8,
        ZPSE_ENC_SPEED_MAX
    };
    static const char *names[9] = {"fastest", "2", "faster", "4",
                                   "balanced", "6", "7", "8", "max"};
    uint32_t li;

    for (li = 0; li < 9u; li++) {
        uint32_t cl, ol;
        uint32_t i;
        int rc;

        /* Text: prose-like, every byte below 0x80. */
        for (i = 0; i < 65536u; i++) {
            uint32_t r = i * 2246822519u;

            in[i] = (uint8_t)(((i & 63u) == 63u) ? '\n'
                                                 : (uint8_t)('a' + (r >> 26) % 26u));
        }
        rc = zpse_compress_block_ex(in, 65536u, comp, sizeof comp, &cl, 0, 1,
                                    levels[li], E);
        CHECK(rc == 0, "text compress");
        if (rc == 0) {
            rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
            CHECK(rc == 0 && ol == 65536u && memcmp(in, out, 65536u) == 0,
                  names[li]);
            CHECK(cl < 65536u, "text must beat raw");
            /* 7 bits a byte plus a 10-byte record per 256-byte tile. */
            CHECK(cl < 60000u, "text near seven bits a byte");
        }

        /* Mixed: ASCII text half, high-bit binary half. The ASCII tiles
         * can still take the form; the round trip is what must hold. */
        for (i = 0; i < 65536u; i++) {
            in[i] = (uint8_t)((i < 32768u) ? ('0' + (i % 10u))
                                           : (0x80u | (uint8_t)(i * 7u)));
        }
        rc = zpse_compress_block_ex(in, 65536u, comp, sizeof comp, &cl, 0, 1,
                                    levels[li], E);
        CHECK(rc == 0, "mixed compress");
        if (rc == 0) {
            rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
            CHECK(rc == 0 && ol == 65536u && memcmp(in, out, 65536u) == 0,
                  "mixed roundtrip");
        }

        /* Random bytes with the high bit set: never ASCII, so the form
         * must not be used and the block must not grow. */
        for (i = 0; i < 65536u; i++) {
            uint32_t r = 1u + i;

            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            in[i] = (uint8_t)(0x80u | (uint8_t)(r >> 24));
        }
        rc = zpse_compress_block_ex(in, 65536u, comp, sizeof comp, &cl, 0, 1,
                                    levels[li], E);
        CHECK(rc == 0, "binary compress");
        if (rc == 0) {
            rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
            CHECK(rc == 0 && ol == 65536u && memcmp(in, out, 65536u) == 0,
                  "binary roundtrip");
            CHECK(cl <= 65536u + 64u, "binary does not grow");
        }
    }

    /* Frame level: the form must survive the frame path, per level. */
    for (li = 0; li < 4u; li++) {
        uint32_t i;
        size_t fl = 0;
        size_t got = 0;
        int rc;

        for (i = 0; i < 200000u; i++)
            frin[i] = (uint8_t)((i % 97u) == 0u ? '\n'
                                               : (uint8_t)('a' + (i >> 3) % 26u));
        rc = zpse_compress_frame_ex(frin, 200000u, frcomp, sizeof frcomp, &fl,
                                    65536u, levels[li] | ZPSE_ENC_CHECKSUM, E);
        CHECK(rc == 0, "frame compress");
        if (rc == 0) {
            rc = zpse_decompress_frame(frcomp, fl, frout, sizeof frout, &got,
                                       D);
            CHECK(rc == 0 && got == 200000u &&
                      memcmp(frin, frout, 200000u) == 0,
                  "frame roundtrip");
        }
    }
}

/* The block-level TEXT form. The ASCII gate fires at 4096 bytes (the
 * threshold lives in block.c as ZPSE_ASCII_MIN_BLOCK) and writes a 16-byte
 * header whose n field spells 65536 as zero. Below the threshold the form is
 * ineligible. One high bit anywhere rejects the whole block: the gate must
 * not fire and the bytes must still round-trip. Every level, three sizes and
 * three flip positions (head, middle, tail) are swept deterministically. */
static void test_text_form(void)
{
    static const uint32_t levels[9] = {
        ZPSE_ENC_SPEED_FASTEST, ZPSE_ENC_SPEED_L2,
        ZPSE_ENC_SPEED_FASTER, ZPSE_ENC_SPEED_L4,
        ZPSE_ENC_SPEED_BALANCED, ZPSE_ENC_SPEED_L6,
        ZPSE_ENC_SPEED_L7, ZPSE_ENC_SPEED_L8,
        ZPSE_ENC_SPEED_MAX
    };
    static const char *names[9] = {"fastest", "2", "faster", "4",
                                   "balanced", "6", "7", "8", "max"};
    static const uint32_t lens[3] = { 4095u, 4096u, 65536u };
    uint32_t li, lj;

    for (li = 0; li < 9u; li++) {
        for (lj = 0; lj < sizeof lens / sizeof lens[0]; lj++) {
            ZpseBlockHeader bh;
            uint32_t n = lens[lj];
            uint32_t cl, ol;
            uint32_t pk;
            int rc;

            prose_fill(in, n, 12345u);
            rc = zpse_compress_block_ex(in, n, comp, sizeof comp, &cl, 7, 1,
                                        levels[li], E);
            CHECKF(rc == 0, "TEXT prose compress n=%u %s", n, names[li]);
            if (rc != 0) {
                continue;
            }
            CHECKF(zpse_block_header_parse(&bh, comp, cl) == 0,
                   "TEXT prose header parse n=%u %s", n, names[li]);
            if (n >= 4096u) {
                CHECKF(bh.sync == ZPSE_BLOCK_SYNC_TEXT,
                       "prose n=%u %s takes TEXT (sync 0x%04X)", n,
                       names[li], (unsigned)bh.sync);
            } else {
                CHECKF(bh.sync != ZPSE_BLOCK_SYNC_TEXT,
                       "prose n=%u %s below the gate is not TEXT", n,
                       names[li]);
            }
            memset(out, 0x5A, sizeof out);
            rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
            CHECKF(rc == 0 && ol == n && memcmp(in, out, n) == 0,
                   "TEXT prose roundtrip n=%u %s", n, names[li]);
            CHECKF(cl < n, "TEXT prose n=%u %s beats raw (%u)", n,
                   names[li], cl);

            /* Flip a single byte to 0x80 at the head, middle or tail. */
            for (pk = 0u; pk < 3u; pk++) {
                uint32_t at = (pk == 0u) ? 0u
                            : (pk == 1u) ? n / 2u : n - 1u;
                uint8_t keep = in[at];

                in[at] = 0x80u;
                rc = zpse_compress_block_ex(in, n, comp, sizeof comp, &cl, 7,
                                            1, levels[li], E);
                CHECKF(rc == 0, "non-ASCII compress n=%u at=%u %s", n, at,
                       names[li]);
                if (rc == 0) {
                    CHECKF(zpse_block_header_parse(&bh, comp, cl) == 0,
                           "non-ASCII header parse n=%u at=%u", n, at);
                    CHECKF(bh.sync != ZPSE_BLOCK_SYNC_TEXT,
                           "0x80 at %u of %u %s rejects the TEXT form", at,
                           n, names[li]);
                    rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol,
                                               D);
                    CHECKF(rc == 0 && ol == n && memcmp(in, out, n) == 0,
                           "non-ASCII roundtrip n=%u at=%u %s", n, at,
                           names[li]);
                }
                in[at] = keep;
            }
        }
    }
}

int main(void)
{
    test_lengths();
    test_rejects();
    test_blocks();
    test_text_form();
    printf("ASCII: %ld checks, %ld failures\n", checks, failures);
    if (failures != 0) {
        return 1;
    }
    printf("ASCII: ALL OK\n");
    return 0;
}
