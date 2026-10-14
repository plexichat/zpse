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
}

int main(void) {
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
        printf("TILECLASS %016llX\n", (unsigned long long)group);
    }

    printf("DIGEST %016llX\n", (unsigned long long)digest);
    printf(fails ? "PARITY: %d FAILURES\n" : "PARITY: ALL OK\n", fails);
    return fails != 0;
}
