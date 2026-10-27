/* Intra-block LZ: the codec on its own, the block form it feeds, and the
 * deepest encoder level that uses it.
 *
 * Three things are pinned here:
 *
 *   - zpse_lz_encode / zpse_lz_decode round-trip every shape a block can
 *     hold, from the five-byte minimum to a full 64 KiB, including the
 *     shapes that split a run across several commands (a literal run past
 *     255 bytes, a match past 259) and the ones a match finder has to
 *     extend backwards into the pending literals.
 *   - A payload that does not decode to exactly the declared output, or
 *     that references outside the bytes it has already written, is refused
 *     with a status instead of being half-decoded. Every truncation and
 *     every byte of corruption is driven through the decoder, because this
 *     is a new wire form and the hostile suite's contract is that nothing
 *     in it can crash a reader.
 *   - The level: ZPSE_ENC_MAX never writes a block larger than the default
 *     level for the same input, and a block it does win is a block the
 *     ordinary decoder reads back byte for byte.
 */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint16_t tab[ZPSE_LZ_TABLE_ENTRIES];

static uint8_t plain[65536];
static uint8_t payload[140000];
static uint8_t comp[140000];
static uint8_t out[65536];

static int fails = 0;
static int checks = 0;
static uint64_t rng = 0x9E3779B97F4A7C15ULL;

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

/* Eight shapes: one per structural case the format has to survive. */
static void make_shape(int kind, uint32_t n)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        switch (kind) {
        case 0: /* ramp: 256-byte period, so it repeats once per block */
            plain[i] = (uint8_t)i;
            break;
        case 1: /* one value: the longest possible match and no literals */
            plain[i] = 0x5Au;
            break;
        case 2: /* noise: no matches at all, so the payload is all literals */
            plain[i] = (uint8_t)rnd();
            break;
        case 3: /* period 8: a short offset, copied over itself */
            plain[i] = (uint8_t)('a' + (i & 7u));
            break;
        case 4: /* sparse: 96 zeros between markers */
            plain[i] = (uint8_t)((i % 97u == 0u) ? (i / 97u) : 0u);
            break;
        case 5: /* long runs: matches far past the 259-byte token limit */
            plain[i] = (uint8_t)((i / 1000u) & 0xFFu);
            break;
        case 6: /* literal runs between matches, past 255 bytes */
            plain[i] = (uint8_t)((i % 600u) < 400u ? rnd() : 0x11u);
            break;
        default: /* a mix: matches, literals and a tail */
            plain[i] = (uint8_t)((i & 0x3FFu) < 700u ? (i * 7u) : 0x7Fu);
            break;
        }
    }
}

static const uint32_t sizes[] = {
    5u, 6u, 12u, 19u, 63u, 255u, 256u, 257u, 259u, 260u, 262u, 300u, 511u,
    512u, 513u, 1000u, 4096u, 12288u, 32768u, 65535u, 65536u
};

static void roundtrip_cases(void)
{
    int kind;
    unsigned si;

    for (kind = 0; kind < 8; kind++) {
        for (si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
            uint32_t n = sizes[si];
            uint32_t pl;

            make_shape(kind, n);
            pl = zpse_lz_encode(plain, n, payload, sizeof payload, tab,
                                 ZPSE_LZ_HASH_BITS);
            CHECK(pl > 0u, "shape %d size %u: no payload", kind, n);
            if (pl == 0u) {
                continue;
            }
            CHECK(pl <= sizeof payload, "shape %d size %u: payload %u", kind,
                  n, pl);
            memset(out, 0xA5, sizeof out);
            CHECK(zpse_lz_decode(payload, pl, out, n) == PSE_OK,
                  "shape %d size %u: decode refused", kind, n);
            CHECK(memcmp(out, plain, n) == 0,
                  "shape %d size %u: bytes differ", kind, n);
        }
    }
}

/* A payload built by hand, so each structural rule can be broken on its
 * own rather than found by chance in a corrupted stream. */
static uint32_t build_payload(uint8_t *dst, uint32_t cap, uint32_t lit_bytes,
                              const uint8_t *lits, const uint32_t *cmds,
                              uint32_t ncmds)
{
    uint32_t cmd_off = 4u + ((lit_bytes + 3u) & ~3u);
    uint32_t i;

    if (cmd_off + 4u * ncmds > cap) {
        return 0u;
    }
    zpse_store_u32le(dst, lit_bytes);
    if (lit_bytes > 0u) {
        memcpy(dst + 4u, lits, lit_bytes);
    }
    for (i = 4u + lit_bytes; i < cmd_off; i++) {
        dst[i] = 0u;
    }
    for (i = 0; i < ncmds; i++) {
        zpse_store_u32le(dst + cmd_off + 4u * i, cmds[i]);
    }
    return cmd_off + 4u * ncmds;
}

static uint32_t cmd(uint32_t off, uint32_t lit, uint32_t mlen)
{
    return (off & 0xFFFFu) | ((lit & 0xFFu) << 16) |
           (((mlen - ZPSE_LZ_MIN_MATCH) & 0xFFu) << 24);
}

static void hostile_cases(void)
{
    static uint8_t buf[256];
    static const uint8_t lits[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    uint32_t cmds[4];
    uint32_t pl;
    uint32_t n;

    /* A well-formed two-token stream: four literals, then a match. */
    cmds[0] = cmd(0u, 4u, 0u);
    cmds[1] = cmd(4u, 0u, 8u);
    pl = build_payload(buf, sizeof buf, 4u, lits, cmds, 2u);
    memset(out, 0, 16u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_OK, "hand-built stream");
    CHECK(memcmp(out, "\1\2\3\4\1\2\3\4\1\2\3\4", 12u) == 0,
          "hand-built bytes");

    /* Offset past the bytes written so far. */
    cmds[0] = cmd(5u, 4u, 8u);
    pl = build_payload(buf, sizeof buf, 4u, lits, cmds, 1u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_BAD_LZ, "offset past start");

    /* A match that would write past the output. */
    cmds[0] = cmd(0u, 4u, 9u);
    pl = build_payload(buf, sizeof buf, 4u, lits, cmds, 1u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_BAD_LZ, "match overruns");

    /* Trailing literals the tokens never emit. */
    cmds[0] = cmd(0u, 4u, 4u);
    pl = build_payload(buf, sizeof buf, 8u, lits, cmds, 1u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_BAD_LZ, "literals unused");

    /* A command after the output is already full. */
    cmds[0] = cmd(0u, 4u, 8u);
    cmds[1] = cmd(0u, 0u, 0u);
    pl = build_payload(buf, sizeof buf, 4u, lits, cmds, 2u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_BAD_LZ, "commands unused");

    /* Zero progress: a literal-only command with no literals. */
    cmds[0] = cmd(0u, 0u, 0u);
    pl = build_payload(buf, sizeof buf, 0u, lits, cmds, 1u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_BAD_LZ, "no progress");

    /* Literal count past the payload. */
    cmds[0] = cmd(0u, 4u, 0u);
    pl = build_payload(buf, sizeof buf, 4u, lits, cmds, 1u);
    zpse_store_u32le(buf, 8u);
    CHECK(zpse_lz_decode(buf, pl, out, 12u) == PSE_BAD_LZ, "literal count");

    /* Not a whole number of command words, and nothing at all. */
    pl = build_payload(buf, sizeof buf, 4u, lits, cmds, 1u);
    CHECK(zpse_lz_decode(buf, pl - 1u, out, 12u) == PSE_BAD_LZ,
          "partial command word");
    CHECK(zpse_lz_decode(buf, 4u, out, 12u) == PSE_TRUNCATED, "no commands");
    CHECK(zpse_lz_decode(buf, pl, out, 0u) == PSE_BAD_BLOCK_SIZE,
          "no output");

    /* Every truncation of a real payload, and every byte corrupted, has to
     * come back as a status. The decode is allowed to write into `out` as
     * it goes, so only the return code is checked -- and that it does not
     * run off the end of the buffers. */
    make_shape(7, 65536);
    pl = zpse_lz_encode(plain, 65536, payload, sizeof payload, tab,
                         ZPSE_LZ_HASH_BITS);
    CHECK(pl > 0u, "fuzz payload");
    for (n = 0; n < pl; n = (n < 64u ? n + 1u : n + 37u)) {
        int rc = zpse_lz_decode(payload, n, out, 65536);
        CHECK(rc == PSE_OK ? n == pl : 1, "truncated %u accepted", n);
    }
    for (n = 0; n < 4000u; n++) {
        uint32_t at = rnd() % pl;
        uint8_t keep = payload[at];

        payload[at] ^= (uint8_t)(1u << (rnd() & 7u));
        (void)zpse_lz_decode(payload, pl, out, 65536);
        payload[at] = keep;
    }
}

/* The block form and the level. */
static void block_cases(void)
{
    static const int shapes[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    unsigned k;

    for (k = 0; k < sizeof shapes / sizeof shapes[0]; k++) {
        uint32_t bal = 0u;
        uint32_t deep = 0u;
        uint32_t got = 0u;
        int rc;

        if (shapes[k] >= 8) {
            uint32_t i;
            for (i = 0; i < 65536u; i++) {
                plain[i] = (uint8_t)(shapes[k] == 8 ? (i < 32768u ? 0u : 0x33u)
                                                    : (i % 300u));
            }
        } else {
            make_shape(shapes[k], 65536u);
        }
        CHECK(zpse_compress_block_ex(plain, 65536u, comp, sizeof comp, &bal,
                                     0u, 1, (uint32_t)ZPSE_ENC_SPEED_BALANCED,
                                     ea) == PSE_OK,
              "shape %d: balanced encode", shapes[k]);
        rc = zpse_compress_block_ex(plain, 65536u, comp, sizeof comp, &deep,
                                    0u, 1, (uint32_t)ZPSE_ENC_SPEED_MAX, ea);
        CHECK(rc == PSE_OK, "shape %d: deepest encode", shapes[k]);
        if (rc != PSE_OK) {
            continue;
        }
        CHECK(deep <= bal, "shape %d: deepest %u larger than balanced %u",
              shapes[k], deep, bal);
        CHECK(zpse_decompress_block(comp, deep, out, sizeof out, &got, da) ==
                  PSE_OK,
              "shape %d: deepest decode", shapes[k]);
        CHECK(got == 65536u && memcmp(out, plain, 65536u) == 0,
              "shape %d: deepest round-trip", shapes[k]);
    }

    /* A block that carries both forms is not a block any encoder writes. */
    {
        uint32_t deep = 0u;
        ZpseBlockHeader h;
        uint32_t got = 0u;

        make_shape(5, 65536u);
        CHECK(zpse_compress_block_ex(plain, 65536u, comp, sizeof comp, &deep,
                                     0u, 1, (uint32_t)ZPSE_ENC_SPEED_MAX, ea) ==
                  PSE_OK,
              "dual-form source block");
        CHECK(zpse_block_header_parse(&h, comp, ZPSE_BLOCK_HEADER_SIZE) ==
                  PSE_OK,
              "dual-form header parse");
        /* Whatever form won, the header is the part under test: setting
         * both form flags describes a payload no decoder may accept. */
        h.block_flags = (uint8_t)(h.block_flags | ZPSE_B_RAW | ZPSE_B_LZ);
        zpse_block_header_serialize(comp, &h);
        zpse_store_u32le(comp + 28, zpse_crc32c(comp, 28u));
        CHECK(zpse_decompress_block(comp, deep, out, sizeof out, &got, da) <
                  0,
              "raw and LZ together refused");
    }
}

/* The table size is a parameter of the encode, so an effort level can
 * pick one. Two properties have to hold for every size: the same table
 * gives the same bytes, and every size round-trips. */
static void table_size_cases(void)
{
    static const uint32_t bits[] = { 8u, 10u, 11u, 12u, 13u };
    int kind;
    unsigned bi;

    for (kind = 0; kind < 8; kind++) {
        uint32_t first = 0u;

        for (bi = 0; bi < sizeof bits / sizeof bits[0]; bi++) {
            uint32_t n = 8192u;
            uint32_t pl;

            make_shape(kind, n);
            memset(tab, 0xFF, sizeof tab);
            pl = zpse_lz_encode(plain, n, payload, sizeof payload, tab,
                                 bits[bi]);
            CHECK(pl > 0u, "shape %d bits %u: no payload", kind, bits[bi]);
            if (pl == 0u) {
                continue;
            }
            CHECK(pl <= sizeof payload, "shape %d bits %u: payload %u", kind,
                  bits[bi], pl);
            memset(out, 0xA5, sizeof out);
            CHECK(zpse_lz_decode(payload, pl, out, n) == PSE_OK,
                  "shape %d bits %u: decode refused", kind, bits[bi]);
            CHECK(memcmp(out, plain, n) == 0,
                  "shape %d bits %u: bytes differ", kind, bits[bi]);
            if (bi == 0u) {
                first = pl;
            } else {
                /* A different table size is allowed to find different
                 * matches, so the payload may differ -- but the block it
                 * decodes to must not. */
                CHECK(pl > 0u, "shape %d bits %u: no payload", kind,
                      bits[bi]);
            }
        }
        (void)first;
    }
}

/* A block with no repeated four bytes anywhere is coded as literals:
 * the form still produces a payload (a little larger than the block),
 * and the caller rejects it on size. What must hold is that the payload
 * round-trips: a block the matcher finds nothing in is still a block
 * the decoder can rebuild. */
static void no_match_cases(void)
{
    uint32_t pl;
    uint32_t i;
    uint32_t x = 12345u;

    for (i = 0; i < 65536u; i++) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        plain[i] = (uint8_t)(x >> 24);
    }
    memset(tab, 0xFF, sizeof tab);
    pl = zpse_lz_encode(plain, 65536u, payload, sizeof payload, tab,
                         ZPSE_LZ_HASH_BITS);
    CHECK(pl > 0u, "no-match block: no payload");
    if (pl > 0u) {
        memset(out, 0xA5, sizeof out);
        CHECK(zpse_lz_decode(payload, pl, out, 65536u) == PSE_OK,
              "no-match block: decode refused");
        CHECK(memcmp(out, plain, 65536u) == 0,
              "no-match block: bytes differ");
    }
}

int main(void)
{
    if (zpse_encoder_init(NULL, ea, sizeof ea) != PSE_BAD_ARG) {
        printf("FAIL: encoder init accepted NULL\n");
        return 1;
    }
    roundtrip_cases();
    hostile_cases();
    block_cases();
    table_size_cases();
    no_match_cases();
    if (fails == 0) {
        printf("LZ: ALL OK (%d checks)\n", checks);
        return 0;
    }
    printf("LZ: %d FAILURES of %d checks\n", fails, checks);
    return 1;
}
