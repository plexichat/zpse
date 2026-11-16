/* Stage profiler for the ENCODE and (page-size) DECODE paths.
 *
 * tests/bench.c reports the end-to-end rate and nothing about where it
 * goes; this reports the pieces the block encoder is built from, for the
 * same ten patterns bench.c uses (the generators are copied from there so
 * a row here can be read next to the matching row there):
 *
 *   block      zpse_compress_block_ex, the number bench.c prints
 *   forward    one zpse_transform_forward of one block for the identity,
 *              DELTA8 and width-16 transpose shapes
 *   segment    one zpse_segment_encode of the whole identity plane --
 *              a trial's real work, without the sweep
 *   tile       zpse_tile_encode of one 256-byte tile, the unit the segment
 *              line is made of
 *
 * With -DZPSE_BLOCK_STATS on BOTH this tool and the library (the unity
 * build), a second line per pattern attributes the encode to the library's
 * own rdtsc cycle counters: the ASCII gate, the attempt-order sample, the
 * decision probe, the full trials, the match-finder pass, the tile-body
 * (v2) finder and the Huffman wrap, all divided by the number of blocks
 * actually encoded. A third line times a batch of page blocks through one
 * reused decoder arena: at 4 KiB the per-call decoder init is a visible
 * fraction of a block, which a 64 KiB frame decode amortises away.
 *
 * Usage: enc_profile [-b BYTES] [-m MODE] [reps] [pattern ...]
 *   -b BYTES   block size, 256..65536 (default 65536; use 4096 for pages)
 *   -m MODE    encoder mode: a level L1..L9 (or 1..9) and/or flags joined
 *              by '+', ',', '|': TURBO, INSTANT, HUF, FAST, LZFAST, LZHEX,
 *              DFAST, CRC (default: level 6, i.e. the library default)
 *   reps       best-of-N timing repetitions (default 5)
 *   pattern    any of the ten names below; no names runs all ten.
 *
 * Examples (from the repo root):
 *   cc -std=c11 -O2 -Iinclude tests/tools/enc_profile.c src/zpse_unity.c \
 *      -o /tmp/enc_profile
 *   /tmp/enc_profile 5 zeros random
 *   cc -std=c11 -O2 -DZPSE_BLOCK_STATS -Iinclude \
 *      tests/tools/enc_profile.c src/zpse_unity.c -o /tmp/enc_profile_stats
 *   /tmp/enc_profile_stats -b 4096 -m TURBO
 *   /tmp/enc_profile_stats -b 4096 -m L6+TURBO -m INSTANT zeros
 *
 * Every line is best-of-reps, in microseconds (encode) or nanoseconds and
 * rdtsc cycles per block (decode). Deterministic formatting: no timing
 * value decides whether a column is printed.
 */
/* clock_gettime is POSIX, and a strict -std=c11 hides it unless the feature
 * test macro says so -- and it has to be said BEFORE the first system header,
 * because those include features.h, which pins the feature set for the rest
 * of the translation unit. Defining it after <stdio.h> left this file
 * building here (where a host compiler defaults to GNU mode) and failing on
 * the CI runner, whose C99/C11 builds do not: "implicit declaration of
 * clock_gettime" and an undeclared CLOCK_MONOTONIC. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> /* struct timespec + clock_gettime on the POSIX path */
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "zpse/zpse.h"
#include "zpse/arena.h"
#include "zpse/predictor.h"
#include "zpse/tile.h"
#include "zpse/transform.h"

/* C11 alignment for the two context arenas; the GNU and MSVC spellings are
 * kept as fallbacks for the compilers that do not accept _Alignas in C. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define ALIGN64 _Alignas(64)
#elif defined(_MSC_VER)
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

/* The library's sweep counters live in src/block.c behind this macro. We
 * only name them when this translation unit is built with it too, so a
 * plain build links exactly as before; build both halves with the macro to
 * get the per-stage line. */
#if defined(ZPSE_BLOCK_STATS)
extern uint32_t zpse_block_blocks;
extern uint64_t zpse_block_gate_cycles;
extern uint64_t zpse_block_sample_cycles;
extern uint64_t zpse_block_probe_cycles;
extern uint64_t zpse_block_trial_cycles;
extern uint64_t zpse_block_lz_pass_cycles;
extern uint64_t zpse_block_v2_cycles;
extern uint64_t zpse_block_huf_cycles;
#endif

static ALIGN64 unsigned char E[ZPSE_ENC_ARENA_SIZE];
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t plane[65536];
/* ORDER2 state: 8 x 4096 {value, confidence, generation} entries. */
static uint8_t o1[8 * 256], o2[8 * 4096 * 4];
static uint8_t rec[2048];
static uint8_t scratch[8192];

/* Run-wide settings, set by -b and -m. */
static uint32_t g_n = 65536;   /* block bytes */
static uint32_t g_flags = 0u;  /* ZPSE_ENC_* for this run */

#define DEC_NBLK 32 /* pages in one reused-arena decode batch */

#define MAXP 16
static const char *names[MAXP];
static int np = 0;

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER f = {{0}};
    LARGE_INTEGER c;
    if (f.QuadPart == 0) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

/* rdtsc for the decode line, comparable to the library's own cycle
 * counters. The lfence pairs are the same serialisation src/block.c uses;
 * on a target without rdtsc the column is deterministically 0. */
static uint64_t cycles_now(void)
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    return (uint64_t)__rdtsc();
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(_M_AMD64))
    uint32_t lo;
    uint32_t hi;
    __asm__ __volatile__("lfence" ::: "memory");
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    __asm__ __volatile__("lfence" ::: "memory");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
    return 0u;
#endif
}

/* Identical to tests/bench.c's make_pattern, by index, over the first g_n
 * bytes. Keep them in step: the whole point is reading a row here next to
 * the same row there. */
static void make_pattern(int idx)
{
    int i;
    uint32_t r;
    int n = (int)g_n;

    switch (idx) {
    case 0:
        memset(in, 0, (size_t)n);
        break;
    case 1:
        for (i = 0; i < n; i++) in[i] = (uint8_t)(i & 0xFF);
        break;
    case 2:
        r = 0x1234;
        for (i = 0; i < n; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)(r >> 24);
        }
        break;
    case 3:
        memset(in, 0, (size_t)n);
        for (i = 0; i < n; i += 97) in[i] = (uint8_t)i;
        break;
    case 4:
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((i / 4 + (i % 4) * 64) & 0xFF);
        break;
    case 5:
        memset(in, 0x5A, (size_t)n);
        break;
    case 6:
        for (i = 0; i < n; i++) in[i] = (uint8_t)((i & 1) ? 0xFF : 0x00);
        break;
    case 7:
        for (i = 0; i < n; i++) in[i] = (uint8_t)(i * 7 + (i >> 4));
        break;
    case 8:
        r = 0xBEEF;
        for (i = 0; i < n; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)((r >> 24) & 0x0F);
        }
        break;
    default:
        memset(in, 0, (size_t)n);
        for (i = 0; i < n; i += 256) {
            in[i] = (uint8_t)i;
            if (i + 1 < n) in[i + 1] = (uint8_t)(i >> 8);
        }
        break;
    }
}

/* Best of `reps` of one block encode, in ns. The compressed length of the
 * first successful encode is returned for the row. */
static double time_block(int reps, uint32_t *clen_out)
{
    double best = 1e18;
    int r;

    *clen_out = 0;
    for (r = 0; r < reps; r++) {
        double t0 = now_ns(), t1;
        uint32_t cl = 0;
        if (zpse_compress_block_ex(in, g_n, comp, sizeof comp, &cl, 0, 1,
                                   g_flags, E) != 0) {
            *clen_out = 0;
            return 0.0;
        }
        t1 = now_ns();
        *clen_out = cl;
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

/* Best of `reps` of one transform forward, in ns. */
static double time_forward(uint8_t tid, uint8_t sc, int reps)
{
    static ZpseTransformMeta meta;
    uint8_t *segs[16];
    uint32_t lens[16];
    uint8_t w = 0;
    double best = 1e18;
    int r;

    if (zpse_transform_width(tid, sc, &w) != ZPSE_OK) return 0.0;
    if (zpse_segment_lengths(g_n, w, lens) != ZPSE_OK) return 0.0;
    meta.transform_id = tid;
    meta.width_W = w;
    meta.stride_flags = 0;
    meta.has_seeds = 1;
    memset(meta.seeds, in[0], sizeof meta.seeds);
    for (r = 0; r < (int)w; r++) segs[r] = plane + (size_t)r * lens[0];
    for (r = 0; r < reps; r++) {
        double t0 = now_ns(), t1;
        if (zpse_transform_forward(&meta, w, in, g_n, segs, lens) != ZPSE_OK)
            return 0.0;
        t1 = now_ns();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

/* A snapshot of the library's cumulative stage counters. Deltas are taken
 * around the encode timing loop, so only the blocks whose cycles we report
 * contribute -- the warm-up encode and the other stages do not. */
typedef struct {
    uint64_t gate;
    uint64_t sample;
    uint64_t probe;
    uint64_t trial;
    uint64_t lz;
    uint64_t v2;
    uint64_t huf;
    uint32_t blocks;
} StageSnap;

static void stage_snapshot(StageSnap *s)
{
#if defined(ZPSE_BLOCK_STATS)
    s->gate = zpse_block_gate_cycles;
    s->sample = zpse_block_sample_cycles;
    s->probe = zpse_block_probe_cycles;
    s->trial = zpse_block_trial_cycles;
    s->lz = zpse_block_lz_pass_cycles;
    s->v2 = zpse_block_v2_cycles;
    s->huf = zpse_block_huf_cycles;
    s->blocks = zpse_block_blocks;
#else
    memset(s, 0, sizeof *s);
#endif
}

static void print_stages(const StageSnap *a, const StageSnap *b, int reps)
{
#if defined(ZPSE_BLOCK_STATS)
    uint32_t nb = b->blocks - a->blocks;
    double d[7], sum, div;
    int i;

    if (nb == 0u) nb = (uint32_t)reps; /* a level/path that counts no blocks */
    div = (double)nb;
    d[0] = (double)(b->gate - a->gate) / div;
    d[1] = (double)(b->sample - a->sample) / div;
    d[2] = (double)(b->probe - a->probe) / div;
    d[3] = (double)(b->trial - a->trial) / div;
    d[4] = (double)(b->lz - a->lz) / div;
    d[5] = (double)(b->v2 - a->v2) / div;
    d[6] = (double)(b->huf - a->huf) / div;
    sum = 0.0;
    for (i = 0; i < 7; i++) sum += d[i];

    printf("  cycles/blk gate=%8.0f sample=%8.0f probe=%8.0f trial=%8.0f"
           " lz=%8.0f v2=%8.0f huf=%8.0f sum=%8.0f\n",
           d[0], d[1], d[2], d[3], d[4], d[5], d[6], sum);
    if (sum > 0.0) {
        printf("  share      gate=%5.1f%% sample=%5.1f%% probe=%5.1f%%"
               " trial=%5.1f%% lz=%5.1f%% v2=%5.1f%% huf=%5.1f%%\n",
               d[0] * 100.0 / sum, d[1] * 100.0 / sum, d[2] * 100.0 / sum,
               d[3] * 100.0 / sum, d[4] * 100.0 / sum, d[5] * 100.0 / sum,
               d[6] * 100.0 / sum);
    } else {
        printf("  share      n/a (no named stage ran)\n");
    }
#else
    (void)a;
    (void)b;
    (void)reps;
    printf("  cycles/blk n/a (build with -DZPSE_BLOCK_STATS)\n");
#endif
}

/* Time a batch of page decodes through ONE decoder arena. The compressed
 * blocks are made outside the clock and the arena is declared once, outside
 * both loops: every timed call therefore pays its own decoder init, which
 * is exactly the fixed overhead a page exposes that a frame amortises. */
static double time_decode_pages(int nb, int reps, uint32_t *clen_out,
                                double *cyc_out)
{
    static ALIGN64 unsigned char D[ZPSE_DEC_ARENA_SIZE];
    static uint8_t dec_out[65536];
    size_t stride = (size_t)g_n + 1024u;
    uint8_t *packed = (uint8_t *)malloc((size_t)nb * stride);
    uint32_t *lens = (uint32_t *)malloc((size_t)nb * sizeof *lens);
    double best_ns = 1e18, best_cyc = 1e18;
    int r, k;

    *clen_out = 0;
    *cyc_out = 0.0;
    if (packed == NULL || lens == NULL) {
        free(packed);
        free(lens);
        return 0.0;
    }

    for (k = 0; k < nb; k++) {
        uint32_t cl = 0;
        if (zpse_compress_block_ex(in, g_n, packed + (size_t)k * stride,
                                   (uint32_t)stride, &cl, (uint32_t)k, 1,
                                   g_flags, E) != 0 || cl == 0u) {
            free(packed);
            free(lens);
            return 0.0;
        }
        lens[k] = cl;
    }
    *clen_out = lens[0];

    /* Verify once, outside the clock, that the arena and the stream agree. */
    {
        uint32_t ol = 0;
        if (zpse_decompress_block(packed, lens[0], dec_out, g_n, &ol, D) != 0 ||
            ol != g_n || memcmp(dec_out, in, (size_t)g_n) != 0) {
            free(packed);
            free(lens);
            return 0.0;
        }
    }

    for (r = 0; r < reps; r++) {
        double t0 = now_ns(), t1;
        uint64_t c0 = cycles_now(), c1;

        for (k = 0; k < nb; k++) {
            uint32_t ol = 0;
            if (zpse_decompress_block(packed + (size_t)k * stride, lens[k],
                                      dec_out, g_n, &ol, D) != 0 ||
                ol != g_n) {
                free(packed);
                free(lens);
                return 0.0;
            }
        }
        t1 = now_ns();
        c1 = cycles_now();
        if (c1 >= c0) {
            double cyc = (double)(c1 - c0) / (double)nb;
            if (cyc < best_cyc) best_cyc = cyc;
        }
        if ((t1 - t0) / (double)nb < best_ns) best_ns = (t1 - t0) / (double)nb;
    }

    free(packed);
    free(lens);
    if (best_cyc < 1e17) *cyc_out = best_cyc;
    return best_ns;
}

static void profile_one(int idx, int reps)
{
    ZpseEncoder e;
    ZpseLaneState lanes;
    StageSnap s0, s1;
    double block, f0, f1, f6, segment, tile = 1e18, dec_ns, dec_cyc;
    uint32_t clen = 0, dclen = 0;
    uint32_t ntiles = (g_n + 255u) / 256u;
    int r;

    make_pattern(idx);
    zpse_encoder_init(&e, E, sizeof E);
    lanes.order1 = o1;
    lanes.o2 = o2;
    lanes.gen = 0u;

    stage_snapshot(&s0);
    block = time_block(reps, &clen);
    stage_snapshot(&s1);
    if (clen == 0u) {
        printf("%-10s compress failed\n", names[idx]);
        return;
    }

    f0 = time_forward(0, 1, reps);
    f1 = time_forward(1, 1, reps);
    f6 = time_forward(6, 16, reps);

    segment = 1e18;
    for (r = 0; r < reps; r++) {
        double t0, t1;
        zpse_lane_reset(&lanes, in[0]);
        t0 = now_ns();
        (void)zpse_segment_encode(&e, &lanes, in, g_n, in[0], comp,
                                  sizeof comp, 0u, 0u);
        t1 = now_ns();
        if (t1 - t0 < segment) segment = t1 - t0;
    }

    /* One tile, exactly as a segment would encode it: predict, search,
     * pack. ntiles of these make up the segment line above. */
    for (r = 0; r < reps * (int)ntiles; r++) {
        double t0, t1;
        int k;
        zpse_lane_reset(&lanes, in[0]);
        t0 = now_ns();
        for (k = 0; k < (int)ntiles; k++) {
            uint32_t off = (uint32_t)k * 256u;
            uint16_t tlen = (uint16_t)((g_n - off < 256u) ? g_n - off : 256u);
            (void)zpse_tile_encode(&lanes, in + off, tlen, 1, 1, 0,
                                    in[off], 1, in, 0u, rec, sizeof rec,
                                    scratch);
        }
        t1 = now_ns();
        if ((t1 - t0) / (double)ntiles < tile)
            tile = (t1 - t0) / (double)ntiles;
    }

    printf("%-10s n=%5u clen=%6u  block %7.2f  fwd id %5.2f d8 %5.2f"
           " t16 %6.2f  segment %7.2f  tile %6.1f (x%3u %7.2f) us\n",
           names[idx], g_n, clen, block / 1000.0, f0 / 1000.0,
           f1 / 1000.0, f6 / 1000.0, segment / 1000.0, tile / 1000.0,
           ntiles, tile * (double)ntiles / 1000.0);

    print_stages(&s0, &s1, reps);

    dec_ns = time_decode_pages(DEC_NBLK, reps, &dclen, &dec_cyc);
    if (dec_ns > 0.0) {
        printf("  decode     %2d pages, 1 arena: %7.1f ns/block"
               "  %8.0f cyc/block  clen=%6u\n",
               DEC_NBLK, dec_ns, dec_cyc, dclen);
    } else {
        printf("  decode     FAILED\n");
    }
}

static int ci_equal(const char *a, const char *b)
{
    while (*a != 0 && *b != 0) {
        unsigned char ca = (unsigned char)*a;
        unsigned char cb = (unsigned char)*b;
        if (ca >= (unsigned char)'A' && ca <= (unsigned char)'Z')
            ca = (unsigned char)(ca - (unsigned char)'A' + (unsigned char)'a');
        if (cb >= (unsigned char)'A' && cb <= (unsigned char)'Z')
            cb = (unsigned char)(cb - (unsigned char)'A' + (unsigned char)'a');
        if (ca != cb) return 0;
        a++;
        b++;
    }
    return *a == 0 && *b == 0;
}

/* Parse a -b value: a plain decimal byte count in [256, 65536]. */
static int parse_bytes(const char *s, uint32_t *out)
{
    char *end = NULL;
    long v;

    if (s == NULL || *s == 0) return -1;
    v = strtol(s, &end, 10);
    if (end == s || *end != 0) return -1;
    if (v < 256 || v > 65536) return -1;
    *out = (uint32_t)v;
    return 0;
}

/* Parse a -m value. A level is L1..L9 (or 1..9); the rest are ZPSE_ENC_*
 * page/rate flags. An empty level field means the library default (6). */
static int parse_mode(const char *s, uint32_t *out)
{
    uint32_t flags = 0u;
    uint32_t lvl = 0u;
    const char *p = s;

    if (s == NULL || *s == 0) return -1;
    while (*p != 0) {
        char tok[32];
        size_t n = 0;
        const char *q = p;

        while (*q != 0 && *q != '+' && *q != ',' && *q != '|') {
            if (n + 1u < sizeof tok) tok[n++] = *q;
            q++;
        }
        tok[n] = 0;
        if (tok[0] == 0) return -1;

        if (tok[0] == 'L' || tok[0] == 'l') {
            int v = atoi(tok + 1);
            if (v < 1 || v > 9) return -1;
            lvl = (uint32_t)v;
        } else if (tok[0] >= '0' && tok[0] <= '9') {
            int v = atoi(tok);
            if (v < 1 || v > 9) return -1;
            lvl = (uint32_t)v;
        } else if (ci_equal(tok, "TURBO")) {
            flags |= ZPSE_ENC_PAGE_TURBO;
        } else if (ci_equal(tok, "INSTANT")) {
            flags |= ZPSE_ENC_PAGE_INSTANT;
        } else if (ci_equal(tok, "HUF")) {
            flags |= ZPSE_ENC_PAGE_HUF;
        } else if (ci_equal(tok, "FAST")) {
            flags |= ZPSE_ENC_PAGE_FAST;
        } else if (ci_equal(tok, "LZFAST")) {
            flags |= ZPSE_ENC_LZ_FAST;
        } else if (ci_equal(tok, "LZHEX")) {
            flags |= ZPSE_ENC_LZ_HEX;
        } else if (ci_equal(tok, "DFAST")) {
            flags |= ZPSE_ENC_DECODE_FAST;
        } else if (ci_equal(tok, "CRC")) {
            flags |= ZPSE_ENC_CHECKSUM;
        } else {
            return -1;
        }
        p = (*q != 0) ? q + 1 : q;
    }
    if (lvl != 0u) flags |= lvl << ZPSE_ENC_SPEED_SHIFT;
    *out = flags;
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
            "usage: enc_profile [-b BYTES] [-m MODE] [reps] [pattern ...]\n"
            "  -b BYTES  256..65536 (default 65536; 4096 for pages)\n"
            "  -m MODE   L1..L9 and/or TURBO, INSTANT, HUF, FAST, LZFAST,"
            " LZHEX, DFAST, CRC, joined by '+'\n"
            "  patterns  zeros incr random sparse strided const5A alt00FF"
            " sawtooth lowentropy sparse2\n");
}

int main(int argc, char **argv)
{
    int reps = 5;
    int sel[MAXP], nsel = 0;
    const char *mode_str = "L6 (default)";
    int i;

    /* Same order as tests/bench.c's pat_names, which is also the order of
     * make_pattern's cases: a name here resolves to the same bytes it does
     * there, so the rows can be read against each other. */
    names[np++] = "zeros";
    names[np++] = "incr";
    names[np++] = "random";
    names[np++] = "sparse";
    names[np++] = "strided";
    names[np++] = "const5A";
    names[np++] = "alt00FF";
    names[np++] = "sawtooth";
    names[np++] = "lowentropy";
    names[np++] = "sparse2";

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];

        if (strcmp(a, "-b") == 0) {
            if (++i >= argc || parse_bytes(argv[i], &g_n) != 0) {
                usage();
                return 1;
            }
        } else if (strcmp(a, "-m") == 0) {
            if (++i >= argc || parse_mode(argv[i], &g_flags) != 0) {
                usage();
                return 1;
            }
            mode_str = argv[i];
        } else if (a[0] == '-' && a[1] != 0) {
            usage();
            return 1;
        } else if (a[0] >= '0' && a[0] <= '9') {
            reps = atoi(a);
            if (reps < 1) reps = 1;
        } else {
            int j, found = 0;
            for (j = 0; j < np; j++) {
                if (strcmp(a, names[j]) == 0) {
                    sel[nsel++] = j;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                usage();
                return 1;
            }
        }
    }

    printf("# enc_profile n=%u reps=%d mode=%s flags=0x%08x stats=%s\n",
           g_n, reps, mode_str, g_flags,
#if defined(ZPSE_BLOCK_STATS)
           "on"
#else
           "off"
#endif
    );
    if (nsel > 0) {
        for (i = 0; i < nsel; i++) profile_one(sel[i], reps);
    } else {
        for (i = 0; i < np; i++) profile_one(i, reps);
    }
    return 0;
}
