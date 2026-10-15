/* Scalar/SIMD parity + roundtrip corpus.
 *
 * Builds the same corpus twice -- once with the default (SIMD) dispatch
 * and once with -DZPSE_FORCE_SCALAR -- and the digest line must match
 * byte for byte. tests/run.ps1 does that comparison; running this
 * binary on its own still verifies every roundtrip and prints the
 * digest to compare against.
 *
 * Exit 0 = all roundtrips ok and the digest is stable. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t out[65536];

static uint64_t digest = 1469598103934665603ull; /* FNV-1a 64 */
static int fails = 0;

static void mix(const uint8_t *p, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        digest ^= (uint64_t)p[i];
        digest *= 1099511628211ull;
    }
}

static uint32_t rng_state = 0xC0FFEEu;
static uint8_t rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t)(rng_state >> 24);
}

/* ------------------------------------------------------------------ *
 * Tile-primitive sweep state (see the sweep in main).
 * ------------------------------------------------------------------ */
#define TILE_N 12
#define LIT_N 3

static const uint8_t lits[LIT_N] = { 0x00u, 0x5Au, 0xFFu };
/* STRIDE predictors read backwards from seg_pos, so the sweep runs the
 * tile both at the very start of a plane (boundary path: positions
 * below the stride read initial_literal) and well inside one (the
 * shifted-load path that a vectorised classifier would use). */
static const uint16_t variants[2] = { 0u, 256u };

static uint8_t tile_src[256];
static uint8_t tile_plane[512];
static uint8_t tile_rec[1024];
static uint8_t tile_dec[256];
static uint8_t tile_comb[0x2000]; /* ZPSE_*_SZ_COMB */
static uint8_t tile_o1[8 * 256];
static uint8_t tile_o2v[8 * 4096];
static uint8_t tile_o2c[8 * 4096];
static uint8_t tile_rec2[1024]; /* second record of the chained sweep */

static uint64_t tile_rng;

static uint8_t nxr(void) {
    tile_rng = tile_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (uint8_t)(tile_rng >> 56);
}

/* One of TILE_N deterministic 256-byte tile shapes. */
static void make_tile(int idx, uint8_t *dst) {
    int i;
    for (i = 0; i < 256; i++) {
        switch (idx) {
        case 0:  dst[i] = 0x00u; break;
        case 1:  dst[i] = 0x5Au; break;
        case 2:  dst[i] = (uint8_t)i; break;
        case 3:  dst[i] = (uint8_t)((i & 1) ? 0xFFu : 0x00u); break;
        case 4:  dst[i] = (uint8_t)(nxr() & 0x0Fu); break;
        case 5:  dst[i] = (uint8_t)((i / 4 + (i % 4) * 64) & 0xFF); break;
        case 6:  dst[i] = nxr(); break;
        case 7:  dst[i] = (i == 100) ? 0x37u : 0x00u; break;
        case 8:  dst[i] = (i == 200) ? 0x11u : 0x5Au; break;
        case 9:  dst[i] = (uint8_t)(i * 7 + (i >> 4)); break;
        case 10: dst[i] = ((i % 97) == 0) ? (uint8_t)(i * 3) : 0x00u; break;
        default: dst[i] = (uint8_t)((i < 128) ? 0x10u : 0xE0u); break;
        }
    }
}

/* FNV-1a 64 folds of arbitrary scalars, into an explicit accumulator so
 * the sweep's digest stays separate from the corpus digest above (whose
 * value run.ps1 already reports). */
static void emix(uint64_t *d, const uint8_t *p, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        *d ^= (uint64_t)p[i];
        *d *= 1099511628211ull;
    }
}

static void emix_u32(uint64_t *d, uint32_t v) {
    uint8_t b[4];
    b[0] = (uint8_t)v;
    b[1] = (uint8_t)(v >> 8);
    b[2] = (uint8_t)(v >> 16);
    b[3] = (uint8_t)(v >> 24);
    emix(d, b, 4u);
}

static void emix_u64(uint64_t *d, uint64_t v) {
    uint8_t b[8];
    int bi;
    for (bi = 0; bi < 8; bi++) {
        b[bi] = (uint8_t)(v >> (8 * bi));
    }
    emix(d, b, 8u);
}

/* Bit-by-bit count, deliberately not the implementation's popcount, so
 * it is an independent check of a dense mask. */
static uint32_t popcount_positions(const uint32_t words[8], uint16_t nt) {
    uint32_t n = 0;
    uint16_t p;
    for (p = 0; p < nt; p++) {
        if (((words[p >> 5] >> (p & 31u)) & 1u) != 0u) {
            n++;
        }
    }
    return n;
}

/* Any mask bit at or above nt. The classifier masks the partial tail of
 * its last 32-byte chunk, so these are always zero, and the packers'
 * "k <= nt" assumption depends on it. */
static uint32_t high_word_bits(const uint32_t words[8], uint16_t nt)
{
    uint32_t acc = 0u;
    uint32_t wi = ((uint32_t)nt + 31u) >> 5;
    uint32_t bit = (uint32_t)nt & 31u;

    if (bit != 0u && wi < 8u) {
        acc |= words[wi] & ~((1u << bit) - 1u);
        wi++;
    }
    while (wi < 8u) {
        acc |= words[wi];
        wi++;
    }
    return acc;
}

/* ------------------------------------------------------------------ *
 * Small-nt and mask-word boundary coverage.
 *
 * The 32-wide chunking of the SIMD classifier, its masked partial tail
 * and the dense path's 32-exception compaction each have a boundary that
 * no 256-position corpus tile ever visits: a tile whose exception count
 * lands exactly on 32, a partial tail chunk, a tile shorter than one
 * chunk. The shapes below are built so the CONST_ZERO predictor sees
 * exactly 31, 32, 33, nt-1 or nt exceptions, so a first/second/last
 * chunk is completely set or completely clear, plus low-nibble and full
 * pseudorandom tiles for the distinct-residual path.
 * ------------------------------------------------------------------ */
#define SMALL_NT_N 25
#define SMALL_SHAPE_N 12

static const uint16_t small_nts[SMALL_NT_N] = {
    1u, 2u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 47u, 63u, 64u, 65u,
    95u, 96u, 127u, 128u, 129u, 159u, 191u, 223u, 224u, 255u, 256u
};

/* The eight predictors the encoder's prescreen can rank, i.e. the ids
 * zpse_tile_classify is ever called with on the encode side. */
static const uint8_t sm_pids[8] = { 0u, 1u, 2u, 5u, 6u, 7u, 8u, 10u };

/* The three ids the prescreen never ranks: the stateful table loop of
 * ORDER1/ORDER2 and the TILE_CONSTANT id. */
static const uint8_t tbl_pids[3] = { 3u, 4u, 9u };

static void small_tile(int shape, uint16_t nt, uint8_t *dst)
{
    uint16_t i;
    uint16_t k;

    for (i = 0; i < nt; i++) {
        dst[i] = 0x00u;
    }
    switch (shape) {
    case 0: /* all hits */
        break;
    case 1: /* all exceptions */
        for (i = 0; i < nt; i++) {
            dst[i] = 0x37u;
        }
        break;
    case 2: /* exactly 31 exceptions (all of them below nt = 31) */
    case 3: /* exactly 32 exceptions */
    case 4: /* exactly 33 exceptions */
        k = (shape == 2) ? 31u : ((shape == 3) ? 32u : 33u);
        if (k > nt) {
            k = nt;
        }
        /* i*nt/k is strictly increasing while k <= nt, so exactly k
         * positions end up nonzero. */
        for (i = 0; i < k; i++) {
            uint16_t p = (uint16_t)(((uint32_t)i * (uint32_t)nt) /
                                    (uint32_t)k);
            if (p < nt) {
                dst[p] = 0x37u;
            }
        }
        break;
    case 5: /* one exception short of every position */
        for (i = 0; i + 1u < nt; i++) {
            dst[i] = 0x37u;
        }
        break;
    case 6: /* the first 32 positions are exceptions */
        for (i = 0; i < nt && i < 32u; i++) {
            dst[i] = 0x37u;
        }
        break;
    case 7: /* the first 32 positions are hits, the rest exceptions */
        for (i = 32u; i < nt; i++) {
            dst[i] = 0x37u;
        }
        break;
    case 8: /* the last 32 positions are exceptions */
        for (i = (nt > 32u) ? (uint16_t)(nt - 32u) : 0u; i < nt; i++) {
            dst[i] = 0x37u;
        }
        break;
    case 9: /* two-symbol alternation */
        for (i = 0; i < nt; i++) {
            dst[i] = (uint8_t)((i & 1u) ? 0xFFu : 0x00u);
        }
        break;
    case 10: /* low-nibble pseudorandom: few distinct residuals */
        tile_rng = 0xD1B54A32D192ED03ull + (uint64_t)nt;
        for (i = 0; i < nt; i++) {
            dst[i] = (uint8_t)(nxr() & 0x0Fu);
        }
        break;
    default: /* full pseudorandom bytes */
        tile_rng = 0x94D049BB133111EBull + (uint64_t)nt;
        for (i = 0; i < nt; i++) {
            dst[i] = nxr();
        }
        break;
    }
}

/* One classification/pack/decode case at an arbitrary nt. Returns 1 when
 * the case completed and 0 when an invariant failed (counted in `fails`).
 * Everything it reads is folded into *acc, and *full counts the cases
 * whose mask has a completely set word -- the shape the whole-mask
 * reconstruction shortcut and the dense pack path exist for. `off` moves
 * the tile inside the plane scratch, so both the segment-start STRIDE
 * fallback (off == 0) and the shifted-load path (off == 256) are used. */
static int small_case(const uint8_t *tile, uint16_t nt, uint8_t pid,
                      uint8_t lit, uint32_t off, uint64_t *acc,
                      uint32_t *full)
{
    ZpseLaneState L;
    ZpseLaneState L2;
    ZpseTileClass cl;
    uint8_t mm = 0u;
    uint8_t rm = 0u;
    uint32_t wi;
    uint32_t ji;
    uint16_t i;
    int n;
    int rc;

    for (i = 0; i < 512u; i++) {
        tile_plane[i] = (uint8_t)((i * 3u + 7u) & 0xFFu);
    }
    memcpy(tile_plane + off, tile, (size_t)nt);

    memset(&L, 0, sizeof L);
    L.order1 = tile_o1;
    L.o2val = tile_o2v;
    L.o2conf = tile_o2c;
    zpse_lane_reset(&L, lit);
    memset(&cl, 0, sizeof cl);
    rc = zpse_tile_classify(&L, tile_plane + off, nt, pid, lit, 1,
                            tile_plane, off, &cl);
    if (rc != 0) {
        printf("FAIL tilesmall rc pid=%u nt=%u rc=%d\n", (unsigned)pid,
               (unsigned)nt, rc);
        fails++;
        return 0;
    }
    if ((uint32_t)cl.k != popcount_positions(cl.words, nt) ||
        high_word_bits(cl.words, nt) != 0u) {
        printf("FAIL tilesmall mask pid=%u nt=%u k=%u bits=%u high=%u\n",
               (unsigned)pid, (unsigned)nt, (unsigned)cl.k,
               (unsigned)popcount_positions(cl.words, nt),
               (unsigned)high_word_bits(cl.words, nt));
        fails++;
        return 0;
    }
    if ((uint32_t)cl.nu != popcount_positions(cl.seen, 256u)) {
        printf("FAIL tilesmall seen pid=%u nt=%u nu=%u distinct=%u\n",
               (unsigned)pid, (unsigned)nt, (unsigned)cl.nu,
               (unsigned)popcount_positions(cl.seen, 256u));
        fails++;
        return 0;
    }
    for (wi = 0; wi < 8u; wi++) {
        if (cl.words[wi] == 0xFFFFFFFFu) {
            *full += 1u;
        }
    }

    zpse_tile_choose_modes(&cl, nt, &mm, &rm);
    n = zpse_tile_pack(pid, &cl, nt, tile_plane + off, mm, rm, 1, tile_rec,
                       (uint32_t)sizeof tile_rec, tile_comb);
    if (n < 8) {
        printf("FAIL tilesmall pack pid=%u nt=%u mask=%u res=%u n=%d\n",
               (unsigned)pid, (unsigned)nt, (unsigned)mm, (unsigned)rm, n);
        fails++;
        return 0;
    }

    /* decode(pack(classify(tile))) == tile, at this nt. */
    memset(&L2, 0, sizeof L2);
    L2.order1 = tile_o1;
    L2.o2val = tile_o2v;
    L2.o2conf = tile_o2c;
    zpse_lane_reset(&L2, lit);
    memset(tile_dec, 0xA5, sizeof tile_dec);
    rc = zpse_tile_decode(&L2, tile_rec, (uint32_t)n, lit, 1, tile_plane, off,
                          tile_dec, nt, tile_comb);
    if (rc != (int)nt || memcmp(tile_dec, tile_plane + off, (size_t)nt) != 0) {
        printf("FAIL tilesmall round pid=%u nt=%u mask=%u res=%u n=%d "
               "rc=%d\n", (unsigned)pid, (unsigned)nt, (unsigned)mm,
               (unsigned)rm, n, rc);
        fails++;
        return 0;
    }

    emix_u32(acc, (uint32_t)cl.k);
    emix_u32(acc, (uint32_t)cl.nu);
    emix_u32(acc, (uint32_t)cl.rmax);
    emix_u32(acc, (uint32_t)cl.dmin);
    emix_u32(acc, (uint32_t)cl.dmax);
    emix_u32(acc, (uint32_t)cl.constant_C);
    for (wi = 0; wi < 8u; wi++) {
        emix_u32(acc, cl.words[wi]);
    }
    for (ji = 0; ji < (uint32_t)cl.k; ji++) {
        emix(acc, &cl.at_pred[ji], 1u);
    }
    for (ji = 0; ji < (uint32_t)cl.k; ji++) {
        emix(acc, &cl.res_at_exc[ji], 1u);
    }
    emix_u32(acc, (uint32_t)mm);
    emix_u32(acc, (uint32_t)rm);
    emix_u32(acc, (uint32_t)n);
    emix(acc, tile_rec, (size_t)n);
    return 1;
}

/* One tile whose mask is expected to contain a completely set word: pack
 * it, then decode the same record twice from a fresh state (both must
 * reproduce the tile byte for byte, which is what the whole-mask
 * reconstruction shortcut has to preserve) and once with
 * tile_reset = 0, where only the status shape is a build-independent
 * promise: a positive return must still be exactly nt, a negative one is
 * a status -- never a crash and never a partial byte count. */
static int deep_case(const uint8_t *tile, uint16_t nt, uint8_t pid,
                     uint8_t lit, uint64_t *acc, uint32_t *full)
{
    ZpseLaneState L;
    ZpseLaneState L2;
    ZpseTileClass cl;
    uint8_t mm = 0u;
    uint8_t rm = 0u;
    const uint32_t off = 256u;
    uint32_t wi;
    uint16_t i;
    int d;
    int n;
    int rc;
    int rc0;

    for (i = 0; i < 512u; i++) {
        tile_plane[i] = (uint8_t)((i * 5u + 1u) & 0xFFu);
    }
    memcpy(tile_plane + off, tile, (size_t)nt);

    memset(&L, 0, sizeof L);
    L.order1 = tile_o1;
    L.o2val = tile_o2v;
    L.o2conf = tile_o2c;
    zpse_lane_reset(&L, lit);
    memset(&cl, 0, sizeof cl);
    rc = zpse_tile_classify(&L, tile_plane + off, nt, pid, lit, 1,
                            tile_plane, off, &cl);
    if (rc != 0) {
        printf("FAIL tiledeep rc pid=%u nt=%u rc=%d\n", (unsigned)pid,
               (unsigned)nt, rc);
        fails++;
        return 0;
    }
    if ((uint32_t)cl.k != popcount_positions(cl.words, nt) ||
        high_word_bits(cl.words, nt) != 0u) {
        printf("FAIL tiledeep mask pid=%u nt=%u k=%u high=%u\n",
               (unsigned)pid, (unsigned)nt, (unsigned)cl.k,
               (unsigned)high_word_bits(cl.words, nt));
        fails++;
        return 0;
    }
    for (wi = 0; wi < 8u; wi++) {
        if (cl.words[wi] == 0xFFFFFFFFu) {
            *full += 1u;
        }
    }

    zpse_tile_choose_modes(&cl, nt, &mm, &rm);
    n = zpse_tile_pack(pid, &cl, nt, tile_plane + off, mm, rm, 1, tile_rec,
                       (uint32_t)sizeof tile_rec, tile_comb);
    if (n < 8) {
        printf("FAIL tiledeep pack pid=%u nt=%u mask=%u res=%u n=%d\n",
               (unsigned)pid, (unsigned)nt, (unsigned)mm, (unsigned)rm, n);
        fails++;
        return 0;
    }

    for (d = 0; d < 2; d++) {
        memset(&L2, 0, sizeof L2);
        L2.order1 = tile_o1;
        L2.o2val = tile_o2v;
        L2.o2conf = tile_o2c;
        zpse_lane_reset(&L2, lit);
        memset(tile_dec, 0x5A, sizeof tile_dec);
        rc = zpse_tile_decode(&L2, tile_rec, (uint32_t)n, lit, 1, tile_plane,
                              off, tile_dec, nt, tile_comb);
        if (rc != (int)nt ||
            memcmp(tile_dec, tile_plane + off, (size_t)nt) != 0) {
            printf("FAIL tiledeep round pid=%u nt=%u mask=%u res=%u d=%d "
                   "n=%d rc=%d\n", (unsigned)pid, (unsigned)nt,
                   (unsigned)mm, (unsigned)rm, d, n, rc);
            fails++;
            return 0;
        }
        emix(acc, tile_dec, (size_t)nt);
    }

    memset(&L2, 0, sizeof L2);
    L2.order1 = tile_o1;
    L2.o2val = tile_o2v;
    L2.o2conf = tile_o2c;
    zpse_lane_reset(&L2, lit);
    memset(tile_dec, 0xA5, sizeof tile_dec);
    rc0 = zpse_tile_decode(&L2, tile_rec, (uint32_t)n, lit, 0, tile_plane, off,
                           tile_dec, nt, tile_comb);
    if (rc0 > 0 && rc0 != (int)nt) {
        printf("FAIL tiledeep noreset pid=%u nt=%u rc=%d\n", (unsigned)pid,
               (unsigned)nt, rc0);
        fails++;
        return 0;
    }

    emix_u32(acc, (uint32_t)cl.k);
    emix_u32(acc, (uint32_t)cl.nu);
    for (wi = 0; wi < 8u; wi++) {
        emix_u32(acc, cl.words[wi]);
    }
    emix_u32(acc, (uint32_t)mm);
    emix_u32(acc, (uint32_t)rm);
    emix_u32(acc, (uint32_t)n);
    emix_u32(acc, (uint32_t)rc0 + 4096u); /* fold a negative status too */
    return 1;
}

static void run(const char *name, uint32_t n) {
    uint32_t clen = 0, olen = 0;
    int rc = zpse_compress_block(in, n, comp, sizeof comp, &clen, 7, 1,
                                 enc_arena);
    if (rc != 0) {
        printf("FAIL %-22s compress rc=%d(%s)\n", name, rc,
               zpse_status_name(rc));
        fails++;
        return;
    }
    memset(out, 0xA5, n);
    rc = zpse_decompress_block(comp, clen, out, n, &olen, dec_arena);
    if (rc < 0 || olen != n || memcmp(in, out, n) != 0) {
        printf("FAIL %-22s decode rc=%d olen=%u\n", name, rc, olen);
        fails++;
        return;
    }
    printf("ok   %-22s n=%5u clen=%6u\n", name, n, clen);
    /* Digest the length too, so a parity break in sizing is caught. */
    mix(comp, clen);
    {
        uint8_t l[4] = { (uint8_t)clen, (uint8_t)(clen >> 8),
                         (uint8_t)(clen >> 16), (uint8_t)(clen >> 24) };
        mix(l, 4);
    }
}/* Two tiles classified back to back on ONE lane state: the first with
 * tile_reset = 1 (what the encoder does for every tile, and what the
 * per-primitive sweep above does for its single tile), the second with
 * tile_reset = 0, so the second classification starts from the lane
 * histories AND the ORDER1/ORDER2 tables the first one left behind.
 *
 * The single-tile sweep can never see that end state: it builds a fresh
 * lane state per case and never reads one afterwards. Here it is made
 * observable twice over -- once as the state itself (both histories plus
 * every byte of the table the predictor reads, folded into the digest at
 * the end) and once through the round trip, because the second record is
 * decoded from a lane state this file re-derives by decoding the first
 * record with tile_reset = 0. So a classification that leaves the wrong
 * histories or writes the wrong table entries breaks the digest against
 * the -DZPSE_FORCE_SCALAR build, and a state the DECODER cannot
 * reproduce breaks as a decode mismatch here.
 *
 * pids 3 and 4 are the stateful tables the encoder's prescreen never
 * ranks (ORDER1, ORDER2); 0 and 1 are the controls. Returns 0 on
 * success, non-zero after incrementing `fails`, like small_case. */
static int chain_case(int shape_a, int shape_b, uint8_t pid, uint8_t lit,
                      uint64_t *acc)
{
    ZpseLaneState L;
    ZpseLaneState L2;
    ZpseTileClass clA;
    ZpseTileClass clB;
    uint8_t mmA = 0u;
    uint8_t rmA = 0u;
    uint8_t mmB = 0u;
    uint8_t rmB = 0u;
    uint32_t i;
    int nA;
    int nB;
    int rc;

    /* tile_plane[0,256) is the first tile, [256,512) the second: tile two
     * is classified at seg_pos = 256 against the same plane. */
    small_tile(shape_a, 256u, tile_src);
    memcpy(tile_plane, tile_src, 256u);
    small_tile(shape_b, 256u, tile_src);
    memcpy(tile_plane + 256u, tile_src, 256u);

    memset(&L, 0, sizeof L);
    L.order1 = tile_o1;
    L.o2val = tile_o2v;
    L.o2conf = tile_o2c;
    zpse_lane_reset(&L, lit);
    memset(&clA, 0, sizeof clA);
    memset(&clB, 0, sizeof clB);

    rc = zpse_tile_classify(&L, tile_plane, 256u, pid, lit, 1, tile_plane,
                            0u, &clA);
    if (rc != PSE_OK) {
        printf("FAIL chainclass a pid=%u rc=%d\n", (unsigned)pid, rc);
        fails++;
        return 1;
    }
    zpse_tile_choose_modes(&clA, 256u, &mmA, &rmA);
    nA = zpse_tile_pack(pid, &clA, 256u, tile_plane, mmA, rmA, 1, tile_rec,
                        (uint32_t)sizeof tile_rec, tile_comb);
    if (nA < 8) {
        printf("FAIL chainpack a pid=%u n=%d\n", (unsigned)pid, nA);
        fails++;
        return 1;
    }

    /* The second tile, on the state the first left, WITHOUT a reset. */
    rc = zpse_tile_classify(&L, tile_plane + 256u, 256u, pid, lit, 0,
                            tile_plane, 256u, &clB);
    if (rc != PSE_OK) {
        printf("FAIL chainclass b pid=%u rc=%d\n", (unsigned)pid, rc);
        fails++;
        return 1;
    }
    if ((uint32_t)clB.k != popcount_positions(clB.words, 256u)) {
        printf("FAIL chainmask pid=%u k=%u mask=%u\n", (unsigned)pid,
               (unsigned)clB.k, popcount_positions(clB.words, 256u));
        fails++;
        return 1;
    }
    if ((uint32_t)clB.nu != popcount_positions(clB.seen, 256u)) {
        printf("FAIL chainnu pid=%u nu=%u distinct=%u\n", (unsigned)pid,
               (unsigned)clB.nu, popcount_positions(clB.seen, 256u));
        fails++;
        return 1;
    }
    zpse_tile_choose_modes(&clB, 256u, &mmB, &rmB);
    nB = zpse_tile_pack(pid, &clB, 256u, tile_plane + 256u, mmB, rmB, 0,
                        tile_rec2, (uint32_t)sizeof tile_rec2, tile_comb);
    if (nB < 8) {
        printf("FAIL chainpack b pid=%u n=%d\n", (unsigned)pid, nB);
        fails++;
        return 1;
    }

    /* The state the second classification left: histories + tables. */
    emix_u32(acc, (uint32_t)clB.k);
    emix_u32(acc, (uint32_t)clB.nu);
    emix_u32(acc, (uint32_t)clB.rmax);
    emix_u32(acc, (uint32_t)clB.dmin);
    emix_u32(acc, (uint32_t)clB.dmax);
    emix_u32(acc, (uint32_t)clB.constant_C);
    for (i = 0; i < 8u; i++) {
        emix_u32(acc, clB.words[i]);
    }
    for (i = 0; i < (uint32_t)clB.k; i++) {
        emix(acc, &clB.at_pred[i], 1u);
    }
    for (i = 0; i < (uint32_t)clB.k; i++) {
        emix(acc, &clB.res_at_exc[i], 1u);
    }
    emix_u32(acc, (uint32_t)mmB);
    emix_u32(acc, (uint32_t)rmB);
    emix_u32(acc, (uint32_t)nB);
    emix(acc, tile_rec2, (size_t)nB);
    emix(acc, L.last, 8u);
    emix(acc, L.second_last, 8u);
    if (pid == 3u) {
        emix(acc, tile_o1, sizeof tile_o1);
    } else if (pid == 4u) {
        emix(acc, tile_o2v, sizeof tile_o2v);
        emix(acc, tile_o2c, sizeof tile_o2c);
    }

    /* Decode both records in sequence: the first resets, the second
     * continues from the state the first decode leaves, which is the
     * decoder's own re-derivation of the state the encoder classified
     * the second tile against. */
    memset(&L2, 0, sizeof L2);
    L2.order1 = tile_o1;
    L2.o2val = tile_o2v;
    L2.o2conf = tile_o2c;
    zpse_lane_reset(&L2, lit);
    memset(tile_dec, 0xA5, sizeof tile_dec);
    rc = zpse_tile_decode(&L2, tile_rec, (uint32_t)nA, lit, 1, tile_plane, 0u,
                          tile_dec, 256u, tile_comb);
    if (rc != 256 || memcmp(tile_dec, tile_plane, 256u) != 0) {
        printf("FAIL chainround a pid=%u n=%d rc=%d\n", (unsigned)pid, nA,
               rc);
        fails++;
        return 1;
    }
    memset(tile_dec, 0xA5, sizeof tile_dec);
    rc = zpse_tile_decode(&L2, tile_rec2, (uint32_t)nB, lit, 0, tile_plane,
                          256u, tile_dec, 256u, tile_comb);
    if (rc != 256 || memcmp(tile_dec, tile_plane + 256u, 256u) != 0) {
        printf("FAIL chainround b pid=%u n=%d rc=%d\n", (unsigned)pid, nB,
               rc);
        fails++;
        return 1;
    }
    return 0;
}

int main(void)
{
    int i;
    static const uint32_t sizes[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32,
                                      33, 63, 127, 255, 256, 257, 511, 1000,
                                      4095, 4096, 4097, 12345, 65535, 65536 };
    const uint32_t nsizes = (uint32_t)(sizeof sizes / sizeof sizes[0]);
    char nm[64];

    /* Deterministic corpora, each exercised at every block size. */
    for (uint32_t s = 0; s < nsizes; s++) {
        uint32_t n = sizes[s];

        memset(in, 0, n);
        sprintf(nm, "zeros/%u", n);
        run(nm, n);

        memset(in, 0x5A, n);
        sprintf(nm, "const5A/%u", n);
        run(nm, n);

        for (i = 0; i < (int)n; i++) in[i] = (uint8_t)(i & 0xFF);
        sprintf(nm, "incr/%u", n);
        run(nm, n);

        for (i = 0; i < (int)n; i++) in[i] = (uint8_t)((i / 4 + (i % 4) * 64));
        sprintf(nm, "strided4/%u", n);
        run(nm, n);

        for (i = 0; i < (int)n; i++) in[i] = (i & 1) ? 0xFF : 0x00;
        sprintf(nm, "alt/%u", n);
        run(nm, n);

        memset(in, 0, n);
        for (i = 0; i < (int)n; i += 97) in[i] = (uint8_t)(i * 7);
        sprintf(nm, "sparse/%u", n);
        run(nm, n);

        rng_state = 0x1234u + s;
        for (i = 0; i < (int)n; i++) in[i] = rnd();
        sprintf(nm, "random/%u", n);
        run(nm, n);

        rng_state = 0x777u + s;
        for (i = 0; i < (int)n; i++) {
            in[i] = (uint8_t)(rnd() & 0x0F);
        }
        sprintf(nm, "lowentropy/%u", n);
        run(nm, n);
    }

    /* Frame path over a multi-block payload. */
    {
        size_t flen = 0, olen = 0;
        static uint8_t fcomp[300000];
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i * 13) ^ (i >> 3));
        for (int bsz = 256; bsz <= 65536; bsz *= 8) {
            int rc = zpse_compress_frame(in, 65536, fcomp, sizeof fcomp,
                                         &flen, (uint32_t)bsz, enc_arena);
            if (rc != 0) {
                printf("FAIL frame/%d rc=%d\n", bsz, rc);
                fails++;
                continue;
            }
            memset(out, 0, 65536);
            rc = zpse_decompress_frame(fcomp, flen, out, 65536, &olen,
                                       dec_arena);
            if (rc != 0 || olen != 65536 || memcmp(in, out, 65536) != 0) {
                printf("FAIL frame/%d decode rc=%d olen=%u\n", bsz, rc,
                       (unsigned)olen);
                fails++;
                continue;
            }
            printf("ok   frame/%d flen=%u\n", bsz, (unsigned)flen);
            mix(fcomp, flen);
        }
    }

    /* ---- Per-primitive scalar/SIMD parity -------------------------- *
     * The corpus above proves the two builds agree on whole compressed
     * streams, which is strong but coarse: a SIMD path that is used only
     * for some tile shapes could stay wrong for the shapes the corpus
     * happens not to pick. This sweep drives the tile primitives
     * directly -- classify, then choose_modes, then pack -- across every
     * predictor id and a spread of tile shapes, and folds the result
     * fields into its own digest. tests/run.ps1 compares the printed
     * lines between the default and -DZPSE_FORCE_SCALAR builds, so any
     * divergence in a primitive shows up here with the predictor id that
     * caused it.
     *
     * The sweep also asserts build-independent invariants and exits
     * non-zero on them, so running this binary alone still checks that a
     * classification and its packed record are self-consistent:
     *   k == popcount(exception mask)
     *   nu == popcount(distinct residuals)
     *   decode(pack(classify(tile))) == tile
     * The last one is what would catch a vectorised classifier that
     * disagrees with the scalar reference about *which* positions are
     * exceptions. */
    {
        uint64_t group = 1469598103934665603ull; /* FNV-1a 64 */
        uint32_t pid;

        for (pid = 0; pid <= 10u; pid++) {
            uint64_t acc = 1469598103934665603ull;
            uint32_t cases = 0;
            uint32_t tile_i, lit_i, rst_i, var_i;

            for (tile_i = 0; tile_i < TILE_N; tile_i++) {
                tile_rng = 0x9E3779B97F4A7C15ull +
                           (uint64_t)tile_i * 0x2545F4914F6CDD1Dull;
                make_tile((int)tile_i, tile_src);
                for (lit_i = 0; lit_i < LIT_N; lit_i++) {
                    for (rst_i = 0; rst_i < 2u; rst_i++) {
                        for (var_i = 0; var_i < 2u; var_i++) {
                            uint16_t off = variants[var_i];
                            ZpseLaneState L;
                            ZpseLaneState L2;
                            ZpseTileClass cl;
                            uint8_t mm = 0u;
                            uint8_t rm = 0u;
                            uint32_t wi;
                            uint32_t ji;
                            int n;
                            int rc;

                            /* Fresh, identical starting state for both
                             * the classifier and the decoder. */
                            for (i = 0; i < 256; i++) {
                                tile_plane[i] =
                                    (uint8_t)((i * 3 + 7) & 0xFF);
                            }
                            memcpy(tile_plane + off, tile_src, 256u);

                            memset(&L, 0, sizeof L);
                            L.order1 = tile_o1;
                            L.o2val = tile_o2v;
                            L.o2conf = tile_o2c;
                            zpse_lane_reset(&L, lits[lit_i]);
                            memset(&cl, 0, sizeof cl);
                            rc = zpse_tile_classify(&L, tile_plane + off, 256u,
                                                    (uint8_t)pid, lits[lit_i],
                                                    (int)rst_i, tile_plane,
                                                    (uint32_t)off, &cl);
                            if (rc != 0) {
                                printf("FAIL tileclass pid=%u tile=%u lit=%u "
                                       "reset=%u off=%u rc=%d\n",
                                       (unsigned)pid, (unsigned)tile_i,
                                       (unsigned)lit_i, (unsigned)rst_i,
                                       (unsigned)off, rc);
                                fails++;
                                continue;
                            }

                            /* Independent re-derivation of the counts. */
                            if ((uint32_t)cl.k !=
                                popcount_positions(cl.words, 256u)) {
                                printf("FAIL tileclass pid=%u tile=%u k=%u "
                                       "mask=%u\n", (unsigned)pid,
                                       (unsigned)tile_i,
                                       (unsigned)cl.k,
                                       popcount_positions(cl.words, 256u));
                                fails++;
                                continue;
                            }
                            if ((uint32_t)cl.nu !=
                                popcount_positions(cl.seen, 256u)) {
                                printf("FAIL tileclass pid=%u tile=%u nu=%u "
                                       "distinct=%u\n", (unsigned)pid,
                                       (unsigned)tile_i,
                                       (unsigned)cl.nu,
                                       popcount_positions(cl.seen, 256u));
                                fails++;
                                continue;
                            }

                            zpse_tile_choose_modes(&cl, 256u, &mm, &rm);
                            n = zpse_tile_pack((uint8_t)pid, &cl, 256u,
                                               tile_plane + off, mm, rm,
                                               (int)rst_i, tile_rec,
                                               (uint32_t)sizeof tile_rec,
                                               tile_comb);
                            if (n < 8) {
                                printf("FAIL tilepack  pid=%u tile=%u "
                                       "mask=%u res=%u n=%d\n",
                                       (unsigned)pid, (unsigned)tile_i,
                                       (unsigned)mm, (unsigned)rm, n);
                                fails++;
                                continue;
                            }

                            /* decode(pack(classify(tile))) == tile */
                            memset(&L2, 0, sizeof L2);
                            L2.order1 = tile_o1;
                            L2.o2val = tile_o2v;
                            L2.o2conf = tile_o2c;
                            zpse_lane_reset(&L2, lits[lit_i]);
                            memset(tile_dec, 0xA5, sizeof tile_dec);
                            rc = zpse_tile_decode(&L2, tile_rec,
                                                  (uint32_t)n, lits[lit_i],
                                                  (int)rst_i, tile_plane,
                                                  (uint32_t)off, tile_dec,
                                                  256u, tile_comb);
                            if (rc != 256 ||
                                memcmp(tile_dec, tile_plane + off,
                                       256u) != 0) {
                                printf("FAIL tileround pid=%u tile=%u "
                                       "mask=%u res=%u n=%d rc=%d\n",
                                       (unsigned)pid, (unsigned)tile_i,
                                       (unsigned)mm, (unsigned)rm, n, rc);
                                fails++;
                                continue;
                            }

                            emix_u32(&acc, (uint32_t)cl.k);
                            emix_u32(&acc, (uint32_t)cl.nu);
                            emix_u32(&acc, (uint32_t)cl.rmax);
                            emix_u32(&acc, (uint32_t)cl.dmin);
                            emix_u32(&acc, (uint32_t)cl.dmax);
                            emix_u32(&acc, (uint32_t)cl.constant_C);
                            for (wi = 0; wi < 8u; wi++) {
                                emix_u32(&acc, cl.words[wi]);
                            }
                            for (ji = 0; ji < (uint32_t)cl.k; ji++) {
                                emix(&acc, &cl.at_pred[ji], 1u);
                            }
                            for (ji = 0; ji < (uint32_t)cl.k; ji++) {
                                emix(&acc, &cl.res_at_exc[ji], 1u);
                            }
                            emix_u32(&acc, (uint32_t)mm);
                            emix_u32(&acc, (uint32_t)rm);
                            emix_u32(&acc, (uint32_t)n);
                            emix(&acc, tile_rec, (size_t)n);
                            cases++;
                        }
                    }
                }
            }

            printf("tileclass pid=%2u cases=%4u %016llX\n", (unsigned)pid,
                   (unsigned)cases, (unsigned long long)acc);
            emix_u64(&group, acc);
        }
        /* ---- Small-nt and mask-word boundary sweep ---------------- *
         * Same primitive chain as the block above, over nt from 1 to
         * 256 and over shapes built to hit the classifier's own
         * boundaries: exactly 31/32/33 exceptions (both sides of the
         * dense path's 32-exception decision), a completely set first,
         * last or only 32-position chunk, one exception short of the
         * whole tile, and pseudorandom tiles for the distinct-residual
         * path. Half the cases sit at seg_pos == 0 (the STRIDE
         * fallback) and half inside the plane, so the partial tail of
         * the 32-wide chunking is exercised at both segment ends. */
        {
            uint64_t acc = 1469598103934665603ull;
            uint32_t cases = 0;
            uint32_t full = 0u;
            uint32_t si;
            uint32_t ni;
            uint32_t pi;
            uint32_t li;

            for (si = 0; si < (uint32_t)SMALL_SHAPE_N; si++) {
                for (ni = 0; ni < (uint32_t)SMALL_NT_N; ni++) {
                    uint16_t nt = small_nts[ni];
                    uint32_t off = (ni & 1u) ? 0u : 256u;

                    if (si == 5u && nt < 2u) {
                        continue; /* shape 5 needs a hit left */
                    }
                    small_tile((int)si, nt, tile_src);
                    for (li = 0; li < (uint32_t)LIT_N; li++) {
                        for (pi = 0; pi < 8u; pi++) {
                            if (small_case(tile_src, nt, sm_pids[pi],
                                           lits[li], off, &acc, &full) != 0) {
                                cases++;
                            }
                        }
                    }
                }
            }

            printf("tilesmall  cases=%4u full=%4u %016llX\n",
                   (unsigned)cases, (unsigned)full,
                   (unsigned long long)acc);
            emix_u64(&group, acc);
        }

        /* ORDER1/ORDER2 and TILE_CONSTANT at the same boundaries. These
         * are the ids the encoder's prescreen never ranks, so their
         * classification is the stateful per-position table loop rather
         * than the vector path -- and that loop has to survive a tile
         * shorter than one chunk just as the vector path does. */
        {
            uint64_t acc = 1469598103934665603ull;
            uint32_t cases = 0;
            uint32_t full = 0u;
            static const uint16_t tbl_nts[3] = { 32u, 64u, 256u };
            uint32_t si;
            uint32_t ni;
            uint32_t pi;

            for (si = 0; si < (uint32_t)SMALL_SHAPE_N; si++) {
                for (ni = 0; ni < 3u; ni++) {
                    uint16_t nt = tbl_nts[ni];

                    if (si == 5u && nt < 2u) {
                        continue;
                    }
                    small_tile((int)si, nt, tile_src);
                    for (pi = 0; pi < 3u; pi++) {
                        if (small_case(tile_src, nt, tbl_pids[pi], 0x00u,
                                       256u, &acc, &full) != 0) {
                            cases++;
                        }
                    }
                }
            }

            printf("tiletable  cases=%4u full=%4u %016llX\n",
                   (unsigned)cases, (unsigned)full,
                   (unsigned long long)acc);
            emix_u64(&group, acc);
        }

        /* ---- Completely set mask word, decoded three ways ---------- *
         * A tile whose mask has a full 32-bit word is exactly what the
         * whole-mask reconstruction shortcut and the dense pack path
         * treat specially, so build one on purpose at a chunk, two
         * chunks and a full tile, then require two fresh decodes to be
         * byte-exact and the non-resetting one to still return nt. */
        {
            uint64_t acc = 1469598103934665603ull;
            uint32_t cases = 0;
            uint32_t full = 0u;
            static const uint8_t deep_shapes[3] = { 1u, 6u, 8u };
            static const uint16_t deep_nts[3] = { 32u, 64u, 256u };
            static const uint8_t deep_pids[3] = { 0u, 1u, 5u };
            uint32_t si;
            uint32_t ni;
            uint32_t pi;

            for (si = 0; si < 3u; si++) {
                for (ni = 0; ni < 3u; ni++) {
                    uint16_t nt = deep_nts[ni];

                    small_tile((int)deep_shapes[si], nt, tile_src);
                    for (pi = 0; pi < 3u; pi++) {
                        if (deep_case(tile_src, nt, deep_pids[pi], 0x00u,
                                      &acc, &full) != 0) {
                            cases++;
                        }
                    }
                }
            }
            if (full == 0u) {
                /* The sweep failed to reach the shape it exists for. */
                printf("FAIL tiledeep: no completely set mask word\n");
                fails++;
            }
            printf("tiledeep   cases=%4u full=%4u %016llX\n",
                   (unsigned)cases, (unsigned)full,
                   (unsigned long long)acc);
            emix_u64(&group, acc);
        }

        /* ---- The state a classification leaves behind ------------ *
         * Two tiles back to back on one lane state, the second without a
         * reset, over the two stateful predictors and two controls; the
         * shape pairs are chosen so tile two starts from a warm table
         * (all-hits then all-exceptions, low-nibble pseudorandom then
         * full pseudorandom) and the literal sweep makes those warm
         * entries non-trivial. See chain_case for what is compared. */
        {
            uint64_t acc = 1469598103934665603ull;
            uint32_t cases = 0;
            static const uint8_t chain_pids[4] = { 0u, 1u, 3u, 4u };
            static const uint8_t chain_pair[3][2] = {
                { 0u, 1u }, { 10u, 11u }, { 1u, 10u }
            };
            uint32_t pi;
            uint32_t ci;
            uint32_t li;

            for (ci = 0; ci < 3u; ci++) {
                for (li = 0; li < (uint32_t)LIT_N; li++) {
                    for (pi = 0; pi < 4u; pi++) {
                        if (chain_case((int)chain_pair[ci][0],
                                       (int)chain_pair[ci][1],
                                       chain_pids[pi], lits[li], &acc) == 0) {
                            cases++;
                        }
                    }
                }
            }

            printf("tilechain  cases=%4u %016llX\n", (unsigned)cases,
                   (unsigned long long)acc);
            emix_u64(&group, acc);
        }

        printf("TILECLASS %016llX\n", (unsigned long long)group);
    }

    printf("DIGEST %016llX\n", (unsigned long long)digest);
    printf(fails ? "PARITY: %d FAILURES\n" : "PARITY: ALL OK\n", fails);
    return fails != 0;
}
