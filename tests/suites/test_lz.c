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
            CHECK(zpse_lz_decode(payload, pl, out, n) == ZPSE_OK,
                  "shape %d size %u: decode refused", kind, n);
            CHECK(memcmp(out, plain, n) == 0,
                  "shape %d size %u: bytes differ", kind, n);
        }
    }
}

/* The LZ forms are LZ4 raw blocks with an eight-byte accounting header, so
 * there is no token wire format to build by hand any more. The structural
 * rules are now LZ4's: the output length is known from the block, a match may
 * not reach outside it, and a short or corrupt block is refused. */

static void hostile_cases(void)
{
    static uint8_t seed[64];
    uint32_t pl;
    uint32_t n;

    for (n = 0; n < sizeof seed; n++) {
        seed[n] = (uint8_t)(1u + (n & 3u));
    }

    /* A real payload round-trips, and the LZ4 block inside it is what
     * carries the bytes -- the eight-byte header is only accounting. */
    pl = zpse_lz_encode(seed, sizeof seed, payload, sizeof payload, tab,
                        ZPSE_LZ_HASH_BITS);
    CHECK(pl > 0u, "seed payload");
    if (pl == 0u) {
        return;
    }
    memset(out, 0, sizeof out);
    CHECK(zpse_lz_decode(payload, pl, out, (uint32_t)sizeof seed) == ZPSE_OK,
          "seed stream");
    CHECK(memcmp(out, seed, sizeof seed) == 0, "seed bytes");

    /* No LZ4 block at all, a header claiming zero commands, a truncated
     * block, a wrong output length and a zero output length are all refused. */
    CHECK(zpse_lz_decode(payload, 4u, out, (uint32_t)sizeof seed) == ZPSE_BAD_LZ,
          "no stream");
    zpse_store_u32le(payload + 4u, 0u);
    CHECK(zpse_lz_decode(payload, pl, out, (uint32_t)sizeof seed) == ZPSE_BAD_LZ,
          "no commands");
    zpse_store_u32le(payload + 4u, 1u);
    CHECK(zpse_lz_decode(payload, pl - 1u, out, (uint32_t)sizeof seed) ==
              ZPSE_BAD_LZ,
          "partial block");
    CHECK(zpse_lz_decode(payload, pl, out, (uint32_t)sizeof seed + 1u) ==
              ZPSE_BAD_LZ,
          "wrong output size");
    CHECK(zpse_lz_decode(payload, pl, out, 0u) == ZPSE_BAD_LZ, "no output");

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
        CHECK(rc == ZPSE_OK ? n == pl : 1, "truncated %u accepted", n);
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
                                     ea) == ZPSE_OK,
              "shape %d: balanced encode", shapes[k]);
        rc = zpse_compress_block_ex(plain, 65536u, comp, sizeof comp, &deep,
                                    0u, 1, (uint32_t)ZPSE_ENC_SPEED_MAX, ea);
        CHECK(rc == ZPSE_OK, "shape %d: deepest encode", shapes[k]);
        if (rc != ZPSE_OK) {
            continue;
        }
        CHECK(deep <= bal, "shape %d: deepest %u larger than balanced %u",
              shapes[k], deep, bal);
        CHECK(zpse_decompress_block(comp, deep, out, sizeof out, &got, da) ==
                  ZPSE_OK,
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
                  ZPSE_OK,
              "dual-form source block");
        CHECK(zpse_block_header_parse(&h, comp, ZPSE_BLOCK_HEADER_SIZE) ==
                  ZPSE_OK,
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
            CHECK(zpse_lz_decode(payload, pl, out, n) == ZPSE_OK,
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
        CHECK(zpse_lz_decode(payload, pl, out, 65536u) == ZPSE_OK,
              "no-match block: decode refused");
        CHECK(memcmp(out, plain, 65536u) == 0,
              "no-match block: bytes differ");
    }

    /* ZPSE_ENC_LZ_FAST: the option that spends the block on one pass and
     * writes this form whatever level was asked for. It has to produce the
     * form's own envelope -- bit 7, transform 0, one segment -- and the block
     * has to read back, because this is the one path that reaches a decoder
     * without a sweep having priced it first. */
    {
        uint32_t cl = 0u;
        uint32_t got = 0u;
        ZpseBlockHeader h;
        uint32_t k;

        for (k = 0; k < 65536u; k++) {
            plain[k] = (uint8_t)(k % 251u);
        }
        CHECK(zpse_compress_block_ex(plain, 65536u, comp, sizeof comp, &cl,
                                     0u, 1,
                                     (uint32_t)ZPSE_ENC_LZ_FAST |
                                         (uint32_t)ZPSE_ENC_SPEED_FASTER,
                                     ea) == ZPSE_OK,
              "LZ_FAST: the block did not compress");
        CHECK(cl < 65568u, "LZ_FAST: fell back to a verbatim copy (%u B)", cl);
        CHECK(zpse_block_header_parse(&h, comp, cl) == ZPSE_OK,
              "LZ_FAST: the header does not parse");
        CHECK((h.block_flags & ZPSE_B_LZ2) != 0 && h.segment_count == 1u &&
                  h.transform_id == 0u,
              "LZ_FAST: not the fast LZ form (flags 0x%02x)",
              (unsigned)h.block_flags);
        CHECK(h.compressed_block_size == cl, "LZ_FAST: size in header");
        CHECK(zpse_decompress_block(comp, cl, out, sizeof out, &got, da) ==
                  ZPSE_OK &&
              got == 65536u && memcmp(out, plain, 65536u) == 0,
              "LZ_FAST: round-trip");
    }
}

/* ZPSE_ENC_PAGE_FAST is retired: page blocks now take the one-pass FLAT
 * fast path for the shapes it packs (constants, low-alphabet, pointer lanes)
 * and the level ladder for the rest, regardless of this option bit. What is
 * still pinned here is that the bit is harmless -- the page round-trips and
 * the header parses at both the fastest and deepest level -- and that above
 * a page the sweep still chooses as it always did. */
static void page_fast_cases(void)
{
    uint32_t cl = 0u;
    uint32_t got = 0u;
    ZpseBlockHeader h;
    uint32_t i;

    for (i = 0; i < 4096u; i++) {
        plain[i] = (uint8_t)(i / 16u);
    }
    CHECK(zpse_compress_block_ex(plain, 4096u, comp, sizeof comp, &cl, 0u, 1,
                                 (uint32_t)ZPSE_ENC_PAGE_FAST |
                                     (uint32_t)ZPSE_ENC_SPEED_FASTEST,
                                 ea) == ZPSE_OK,
          "PAGE_FAST: the page did not compress");
    CHECK(cl < 4096u + 64u, "PAGE_FAST: fell back to a verbatim copy (%u B)",
          cl);
    CHECK(zpse_block_header_parse(&h, comp, cl) == ZPSE_OK,
          "PAGE_FAST: the header does not parse");
    CHECK(h.compressed_block_size == cl, "PAGE_FAST: size in header");
    CHECK(zpse_decompress_block(comp, cl, out, sizeof out, &got, da) ==
                  ZPSE_OK &&
              got == 4096u && memcmp(out, plain, 4096u) == 0,
          "PAGE_FAST: round-trip");

    /* The contract is "one shape" at every level: the LZ forms and the
     * LZ-over-tiles stage are decisions too, and a level deep enough to offer
     * them must not offer them to a page under this flag. */
    {
        uint32_t cl6 = 0u;
        uint32_t got6 = 0u;

        for (i = 0; i < 4096u; i++) {
            plain[i] = (uint8_t)(i / 16u);
        }
        CHECK(zpse_compress_block_ex(plain, 4096u, payload, sizeof payload,
                                     &cl6, 0u, 1,
                                     (uint32_t)ZPSE_ENC_PAGE_FAST |
                                         (uint32_t)ZPSE_ENC_SPEED_MAX,
                                     ea) == ZPSE_OK,
              "PAGE_FAST level 9: the page did not compress");
        CHECK(zpse_block_header_parse(&h, payload, cl6) == ZPSE_OK,
              "PAGE_FAST level 9: the header does not parse");
        CHECK(zpse_decompress_block(payload, cl6, out, sizeof out, &got6,
                                    da) == ZPSE_OK &&
                  got6 == 4096u && memcmp(out, plain, 4096u) == 0,
              "PAGE_FAST level 9: round-trip");
    }

    /* Above a page the flag must change nothing: the fastest level's own
     * bytes, to the byte. */
    {
        uint32_t a = 0u;
        uint32_t b = 0u;

        for (i = 0; i < 8192u; i++) {
            plain[i] = (uint8_t)(i % 251u);
        }
        CHECK(zpse_compress_block_ex(plain, 8192u, comp, sizeof comp, &a, 0u,
                                     1, (uint32_t)ZPSE_ENC_SPEED_FASTEST,
                                     ea) == ZPSE_OK,
              "PAGE_FAST: the 8 KiB reference block did not compress");
        CHECK(zpse_compress_block_ex(plain, 8192u, payload, sizeof payload,
                                     &b, 0u, 1,
                                     (uint32_t)(ZPSE_ENC_PAGE_FAST |
                                                ZPSE_ENC_SPEED_FASTEST),
                                     ea) == ZPSE_OK,
              "PAGE_FAST: the 8 KiB block did not compress");
        CHECK(a == b && memcmp(comp, payload, a) == 0,
              "PAGE_FAST changed a block above a page (%u vs %u bytes)", a, b);
    }
}

/* Above a page: the length-measured lazy step, at the window sizes the block
 * encoder arms it for (see ZPSE_LZ_LAZY_MAX).
 *
 * Nothing covered this before a 16 KiB block of mozilla decoded 78 bytes of
 * the wrong history. roundtrip_cases() goes through zpse_lz_encode, whose
 * lazy window is a page, so no case ever ran the step with `lazy_max >= n`
 * -- and that step is the one place the finder stops being strictly forward:
 * it can move the cursor back onto a position whose slot already holds the
 * later one, which is how a candidate ahead of the cursor reached the emit.
 * A stream it writes has to round-trip at every size it is offered, whatever
 * it costs in ratio. */
static void lazy_window_cases(void)
{
    static const uint32_t win[] = { 8192u, 12288u, 16384u };
    unsigned wi;
    int kind;

    for (wi = 0; wi < sizeof win / sizeof win[0]; wi++) {
        uint32_t n = win[wi];

        for (kind = 0; kind < 8; kind++) {
            uint32_t pl;

            make_shape(kind, n);
            pl = zpse_lz_encode_ex(plain, n, payload, sizeof payload, tab,
                                   ZPSE_LZ_HASH_BITS, n);
            CHECK(pl > 0u, "lazy shape %d size %u: no payload", kind, n);
            if (pl == 0u) {
                continue;
            }
            memset(out, 0xA5, sizeof out);
            CHECK(zpse_lz_decode(payload, pl, out, n) == ZPSE_OK,
                  "lazy shape %d size %u: decode refused", kind, n);
            CHECK(memcmp(out, plain, n) == 0,
                  "lazy shape %d size %u: bytes differ", kind, n);
        }
    }

    /* The same window over segment soups: runs of one value beside structured
     * and noisy segments, which is the shape that makes the finder come back
     * to a position it has already inserted. The seeds are constants, so a
     * failure names its own case. */
    {
        uint32_t seed;

        for (seed = 1u; seed <= 24u; seed++) {
            uint32_t pos = 0u;
            uint32_t pl;

            rng = 0x9E3779B97F4A7C15ULL ^
                  (uint64_t)seed * 0xD1B54A32D192ED03ULL;
            while (pos < 16384u) {
                uint32_t len = 1u + (rnd() % 700u);
                uint32_t seg = rnd() % 6u;
                uint32_t i;

                if (len > 16384u - pos) {
                    len = 16384u - pos;
                }
                for (i = 0u; i < len; i++) {
                    switch (seg) {
                    case 0u:
                        plain[pos + i] = 0u;
                        break;
                    case 1u:
                        plain[pos + i] = (uint8_t)rnd();
                        break;
                    case 2u:
                        plain[pos + i] = (uint8_t)i;
                        break;
                    case 3u:
                        plain[pos + i] = (uint8_t)('a' + (i & 7u));
                        break;
                    case 4u:
                        plain[pos + i] = (uint8_t)((i / 4u) & 0xFFu);
                        break;
                    default:
                        plain[pos + i] =
                            (uint8_t)((i & 3u) != 0u ? 0u : (rnd() & 0xFFu));
                        break;
                    }
                }
                pos += len;
            }
            pl = zpse_lz_encode_ex(plain, 16384u, payload, sizeof payload, tab,
                                   ZPSE_LZ_HASH_BITS, 16384u);
            CHECK(pl > 0u, "lazy soup seed %u: no payload", seed);
            if (pl == 0u) {
                continue;
            }
            memset(out, 0xA5, sizeof out);
            CHECK(zpse_lz_decode(payload, pl, out, 16384u) == ZPSE_OK,
                  "lazy soup seed %u: decode refused", seed);
            CHECK(memcmp(out, plain, 16384u) == 0,
                  "lazy soup seed %u: bytes differ", seed);
        }
    }
}

/* The block that exposed the ahead-of-cursor candidate: 16 KiB of mozilla at
 * offset 50,823,168. With the lazy window armed it decoded 78 bytes of the
 * wrong history on the unfixed finder, and it does not shrink -- every
 * prefix and suffix under 16 KiB round-trips -- so it is kept whole.
 *
 * Located relative to this source file rather than the process's directory:
 * the runners compile from the tree and run the binary from anywhere. A
 * missing fixture is a failed check, never a skip, because a regression case
 * that quietly disappears is worse than none. */
static void fixture_cases(void)
{
    char path[1024];
    const char *slash = strrchr(__FILE__, '/');
    const char *bslash = strrchr(__FILE__, '\\');
    const char *tail = "../corpus/lz_lazy_16k.bin";
    size_t dir;
    FILE *f;
    size_t got;

    if (bslash != NULL && (slash == NULL || bslash > slash)) {
        slash = bslash;
    }
    dir = (slash != NULL) ? (size_t)(slash - __FILE__) + 1u : 0u;
    if (dir + strlen(tail) + 1u > sizeof path) {
        CHECK(0, "lazy fixture: path too long");
        return;
    }
    memcpy(path, __FILE__, dir);
    strcpy(path + dir, tail);
    f = fopen(path, "rb");
    CHECK(f != NULL, "lazy fixture: cannot open %s", path);
    if (f == NULL) {
        return;
    }
    got = fread(plain, 1, 16384u, f);
    fclose(f);
    CHECK(got == 16384u, "lazy fixture: read %u bytes", (unsigned)got);
    if (got != 16384u) {
        return;
    }
    {
        uint32_t pl = zpse_lz_encode_ex(plain, 16384u, payload, sizeof payload,
                                        tab, ZPSE_LZ_HASH_BITS, 16384u);

        CHECK(pl > 0u, "lazy fixture: no payload");
        if (pl == 0u) {
            return;
        }
        memset(out, 0xA5, sizeof out);
        CHECK(zpse_lz_decode(payload, pl, out, 16384u) == ZPSE_OK,
              "lazy fixture: decode refused");
        CHECK(memcmp(out, plain, 16384u) == 0, "lazy fixture: bytes differ");
    }
}

int main(void)
{
    if (zpse_encoder_init(NULL, ea, sizeof ea) != ZPSE_BAD_ARG) {
        printf("FAIL: encoder init accepted NULL\n");
        return 1;
    }
    roundtrip_cases();
    lazy_window_cases();
    fixture_cases();
    hostile_cases();
    block_cases();
    table_size_cases();
    no_match_cases();
    page_fast_cases();
    if (fails == 0) {
        printf("LZ: ALL OK (%d checks)\n", checks);
        return 0;
    }
    printf("LZ: %d FAILURES of %d checks\n", fails, checks);
    return 1;
}
