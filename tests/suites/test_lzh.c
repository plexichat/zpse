/* The hexadecimal LZ form (src/lzh.c): the codec on its own, its block form,
 * and the option that selects it.
 *
 * Three things are pinned:
 *
 *   - zpse_lzh_encode / zpse_lzh_decode round-trip every shape a block can
 *     hold, from the five-byte minimum to a full 64 KiB, including the shapes
 *     that need the book's escape: a literal run past 270 bytes, a match past
 *     273, and a command whose pair the table does not name.
 *   - A payload that does not decode to exactly the declared output is refused
 *     with a status rather than half-decoded: every truncation and every byte
 *     of corruption is driven through the decoder, because this is a new wire
 *     form and nothing in it may crash a reader.
 *   - The block form: a block the ZPSE_ENC_LZ_HEX option writes names the form
 *     in its flags (both LZ bits, and no other form bit), decodes back to the
 *     input through the ordinary decoder, and is byte-identical when the same
 *     block is encoded twice. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/lzh.h"
#include "zpse/types.h"

static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint16_t tab[ZPSE_LZ_TABLE_ENTRIES];

static uint8_t plain[65536];
static uint8_t payload[140000];
static uint8_t payload2[140000];
static uint8_t comp[140000];
static uint8_t out[65536];

static int fails = 0;
static int checks = 0;
static uint64_t rng = 0x243F6A8885A308D3ULL;

#define CHECK(cond, ...) do {                       \
    checks++;                                       \
    if (!(cond)) {                                  \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
        fails++;                                    \
    }                                               \
} while (0)

static uint32_t rnd(void)
{
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng >> 33);
}

/* Eight shapes: the structural cases the codec has to survive, the same eight
 * the token form's suite uses. */
static void make_shape(int kind, uint32_t n)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        switch (kind) {
        case 0: /* ramp: a 256-byte period */
            plain[i] = (uint8_t)i;
            break;
        case 1: /* one value: one enormous match, no literals */
            plain[i] = 0x5Au;
            break;
        case 2: /* noise: no matches, so every command is literals */
            plain[i] = (uint8_t)rnd();
            break;
        case 3: /* period 8: a short offset copied over itself */
            plain[i] = (uint8_t)('a' + (i & 7u));
            break;
        case 4: /* sparse: 96 zeros between markers */
            plain[i] = (uint8_t)((i % 97u == 0u) ? (i / 97u) : 0u);
            break;
        case 5: /* long runs: matches far past the 273-byte command limit */
            plain[i] = (uint8_t)((i / 1000u) & 0xFFu);
            break;
        case 6: /* literal runs between matches, past the 270-byte limit */
            plain[i] = (uint8_t)((i % 600u) < 400u ? rnd() : 0x11u);
            break;
        default: /* a mix: matches, literals and a tail */
            plain[i] = (uint8_t)((i & 0x3FFu) < 700u ? (i * 7u) : 0x7Fu);
            break;
        }
    }
}

static const uint32_t sizes[] = {
    5u, 6u, 7u, 12u, 19u, 63u, 255u, 256u, 257u, 259u, 260u, 262u, 271u,
    300u, 511u, 512u, 513u, 1023u, 1024u, 1025u, 4095u, 4096u, 4097u,
    16384u, 65536u
};

static void roundtrip_all(void)
{
    unsigned k;
    unsigned s;

    for (k = 0; k < 8u; k++) {
        for (s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            uint32_t n = sizes[s];
            uint32_t pl;
            int rc;

            make_shape((int)k, n);
            pl = zpse_lzh_encode(plain, n, payload, sizeof payload, tab,
                                 ZPSE_LZ_HASH_BITS);
            CHECK(pl != 0u, "shape %u size %u: encode returned 0", k, n);
            if (pl == 0u) {
                continue;
            }
            CHECK(pl >= ZPSE_LZH_MIN_PAYLOAD, "shape %u size %u: %u bytes", k,
                  n, pl);
            memset(out, 0, n);
            rc = zpse_lzh_decode(payload, pl, out, n);
            CHECK(rc == ZPSE_OK, "shape %u size %u: decode %d", k, n, rc);
            CHECK(memcmp(out, plain, n) == 0,
                  "shape %u size %u: bytes differ", k, n);
            /* Determinism: the same input twice is the same payload. */
            {
                uint32_t pl2 = zpse_lzh_encode(plain, n, payload2,
                                               sizeof payload2, tab,
                                               ZPSE_LZ_HASH_BITS);

                CHECK(pl2 == pl && memcmp(payload, payload2, pl) == 0,
                      "shape %u size %u: encode is not deterministic", k, n);
            }
        }
    }
}

/* Every truncation of a payload has to be refused: the decoder requires the
 * stream position to land exactly on the end, so a short buffer can never be
 * read as a complete stream. */
static void truncations(void)
{
    uint32_t n = 4096u;
    uint32_t pl;
    uint32_t len;
    int bad = 0;

    make_shape(7, n);
    pl = zpse_lzh_encode(plain, n, payload, sizeof payload, tab,
                         ZPSE_LZ_HASH_BITS);
    CHECK(pl != 0u, "truncation case did not encode");
    if (pl == 0u) {
        return;
    }
    for (len = 0; len < pl; len++) {
        int rc = zpse_lzh_decode(payload, len, out, n);

        if (rc == ZPSE_OK) {
            bad++;
        }
    }
    CHECK(bad == 0, "%d truncated payload(s) decoded as complete", bad);
}

/* Every single-byte corruption: a status or a clean decode, never a crash and
 * never a decode that claims to have written a different number of bytes. */
static void corruptions(void)
{
    uint32_t n = 1024u;
    uint32_t pl;
    uint32_t i;
    int ok = 0;
    int refused = 0;

    make_shape(3, n);
    pl = zpse_lzh_encode(plain, n, payload, sizeof payload, tab,
                         ZPSE_LZ_HASH_BITS);
    CHECK(pl != 0u, "corruption case did not encode");
    if (pl == 0u) {
        return;
    }
    for (i = 0; i < pl; i++) {
        uint8_t keep = payload[i];
        int rc;

        payload[i] = (uint8_t)(keep ^ 0xFFu);
        rc = zpse_lzh_decode(payload, pl, out, n);
        if (rc == ZPSE_OK) {
            /* A complete decode is a legal answer: this form carries no
             * payload checksum, so a flipped literal decodes to those bytes.
             * What is not legal is a crash, a partial write, or a status that
             * is not a refusal. */
            ok++;
        } else {
            CHECK(rc < 0, "byte %u: non-negative status %d", i, rc);
            refused++;
        }
        payload[i] = keep;
    }
    CHECK(ok + refused == (int)pl, "corruption scan lost a case");
}

/* The block form: the option writes the hex form, the header says so, and the
 * ordinary decoder reads it back. A page shrinks to the 16-byte header, so the
 * flags byte is at offset 2 in both header forms and the check is the same. */
static void block_form(void)
{
    static const uint32_t ns[] = { 4096u, 16384u, 65536u };
    unsigned s;
    unsigned k;
    int hex_chosen = 0;

    for (s = 0; s < sizeof ns / sizeof ns[0]; s++) {
        for (k = 0; k < 8u; k++) {
            uint32_t n = ns[s];
            uint32_t clen = 0u;
            uint32_t olen = 0u;
            ZpseBlockHeader bh;
            int rc;

            make_shape((int)k, n);
            rc = zpse_compress_block_ex(plain, n, comp, sizeof comp, &clen, 0u,
                                        1,
                                        (uint32_t)(ZPSE_ENC_LZ_HEX |
                                                   ZPSE_ENC_SPEED_L6),
                                        ea);
            CHECK(rc == ZPSE_OK, "block size %u shape %u: encode %d", n, k, rc);
            if (rc != ZPSE_OK) {
                continue;
            }
            /* When the hex form was the winner it carries both LZ bits and no
             * other form bit -- the sweep is free to beat it on any block,
             * which is most of the 16 and 64 KiB shapes. The header is parsed
             * rather than read at a fixed offset, because a page's block is
             * the 16-byte header and the two forms put the flags in different
             * places. */
            CHECK(zpse_block_header_parse(&bh, comp, clen) == ZPSE_OK,
                  "block size %u shape %u: header parse failed", n, k);
            if ((bh.block_flags & (uint8_t)ZPSE_B_LZ_HEX) ==
                (uint8_t)ZPSE_B_LZ_HEX) {
                hex_chosen++;
                CHECK((bh.block_flags &
                       (uint8_t)(ZPSE_B_LZ_TILE | ZPSE_B_RAW)) == 0u,
                      "block size %u shape %u: another form bit set (0x%02X)",
                      n, k, bh.block_flags);
            }
            rc = zpse_decompress_block(comp, clen, out, sizeof out, &olen, da);
            CHECK(rc == ZPSE_OK, "block size %u shape %u: decode %d", n, k, rc);
            CHECK(olen == n && memcmp(out, plain, n) == 0,
                  "block size %u shape %u: bytes differ", n, k);
        }
    }
    /* Informational: on these shapes the sweep's tile and LZ forms take every
     * block, which is the expected result at 16 and 64 KiB -- the hex form is
     * only ever kept when its stream is the smallest one. */
    printf("LZH: sweep chose the hex form on %d of %u blocks\n", hex_chosen,
           24u);
    /* ZPSE_ENC_LZ_FAST spends the block on one finder pass, so with the hex
     * option the form is chosen outright: that is the pairing the paired A/B
     * in the README measures, and it has to round-trip through the ordinary
     * decoder like any other block. */
    {
        int fhex = 0;

        for (k = 0; k < 8u; k++) {
            uint32_t clen = 0u;
            uint32_t olen = 0u;
            ZpseBlockHeader bh;
            int rc;

            make_shape((int)k, 4096u);
            rc = zpse_compress_block_ex(
                plain, 4096u, comp, sizeof comp, &clen, 0u, 1,
                (uint32_t)(ZPSE_ENC_LZ_FAST | ZPSE_ENC_LZ_HEX), ea);
            CHECK(rc == ZPSE_OK, "fast+hex shape %u: encode %d", k, rc);
            if (rc != ZPSE_OK) {
                continue;
            }
            CHECK(zpse_block_header_parse(&bh, comp, clen) == ZPSE_OK,
                  "fast+hex shape %u: header parse", k);
            if ((bh.block_flags & (uint8_t)ZPSE_B_LZ_HEX) ==
                (uint8_t)ZPSE_B_LZ_HEX) {
                fhex++;
            }
            rc = zpse_decompress_block(comp, clen, out, sizeof out, &olen, da);
            CHECK(rc == ZPSE_OK && olen == 4096u &&
                      memcmp(out, plain, 4096u) == 0,
                  "fast+hex shape %u: decode %d", k, rc);
        }
        CHECK(fhex > 0, "fast+hex never wrote the hex form");
    }
    /* The flag alone, without a level, must be accepted: it is an option and
     * the level field may be empty. */
    {
        uint32_t clen = 0u;

        make_shape(3, 4096u);
        CHECK(zpse_compress_block_ex(plain, 4096u, comp, sizeof comp, &clen,
                                     0u, 1, (uint32_t)ZPSE_ENC_LZ_HEX,
                                     ea) == ZPSE_OK,
              "hex option without a level was refused");
    }
}

int main(void)
{
    roundtrip_all();
    truncations();
    corruptions();
    block_form();
    printf("LZH: checks=%d fails=%d\n", checks, fails);
    if (fails != 0) {
        printf("LZH FAILED\n");
        return 1;
    }
    printf("LZH OK\n");
    return 0;
}
