/* Stage profiler for the ENCODE path: why a block takes what it takes.
 *
 * tests/bench.c reports the end-to-end rate and nothing about where it
 * goes; this reports the pieces the block encoder is built from, for the
 * same ten patterns bench.c uses (the generators are copied from there so
 * a row here can be read next to the matching row there):
 *
 *   block      zpse_compress_block, the number bench.c prints
 *   forward    one zpse_transform_forward of 64 KiB for the identity,
 *              DELTA8 and width-16 transpose shapes
 *   segment    one zpse_segment_encode of the whole 64 KiB identity
 *              plane -- a trial's real work, without the sweep
 *   tile       zpse_tile_encode of one 256-byte tile, the unit the
 *              segment line is made of (256 of them per plane)
 *
 * Every line is best-of-reps, in microseconds. The point of the tile line
 * is that `segment / 256` and the measured tile time should agree: when
 * they do not, the segment's own bookkeeping (records, headers, the dense
 * test) is the cost rather than the per-tile work.
 *
 * Usage: enc_profile [reps] [pattern ...]; no arguments runs all ten.
 * Builds against the amalgamation like every other tool in this
 * directory. Exit 0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> /* struct timespec + clock_gettime on the POSIX path */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "zpse/zpse.h"
#include "zpse/arena.h"
#include "zpse/predictor.h"
#include "zpse/tile.h"
#include "zpse/transform.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t plane[65536];
static uint8_t o1[8 * 256], o2v[8 * 4096], o2c[8 * 4096];
static uint8_t rec[2048];
static uint8_t scratch[8192];

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

/* Identical to tests/bench.c's make_pattern, by index. Keep them in step:
 * the whole point is reading a row here next to the same row there. */
static void make_pattern(int idx)
{
    int i;
    uint32_t r;

    switch (idx) {
    case 0:
        memset(in, 0, sizeof in);
        break;
    case 1:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)(i & 0xFF);
        break;
    case 2:
        r = 0x1234;
        for (i = 0; i < 65536; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)(r >> 24);
        }
        break;
    case 3:
        memset(in, 0, sizeof in);
        for (i = 0; i < 65536; i += 97) in[i] = (uint8_t)i;
        break;
    case 4:
        for (i = 0; i < 65536; i++)
            in[i] = (uint8_t)((i / 4 + (i % 4) * 64) & 0xFF);
        break;
    case 5:
        memset(in, 0x5A, sizeof in);
        break;
    case 6:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i & 1) ? 0xFF : 0x00);
        break;
    case 7:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)(i * 7 + (i >> 4));
        break;
    case 8:
        r = 0xBEEF;
        for (i = 0; i < 65536; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)((r >> 24) & 0x0F);
        }
        break;
    default:
        memset(in, 0, sizeof in);
        for (i = 0; i < 65536; i += 256) {
            in[i] = (uint8_t)i;
            in[i + 1] = (uint8_t)(i >> 8);
        }
        break;
    }
}

/* Best of `reps` of one block encode, in ns. */
static double time_block(int reps)
{
    double best = 1e18;
    int r;
    for (r = 0; r < reps; r++) {
        double t0 = now_ns(), t1;
        uint32_t cl = 0;
        (void)zpse_compress_block(in, sizeof in, comp, sizeof comp, &cl, 0, 1, E);
        t1 = now_ns();
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

    if (zpse_transform_width(tid, sc, &w) != PSE_OK) return 0.0;
    if (zpse_segment_lengths(65536, w, lens) != PSE_OK) return 0.0;
    meta.transform_id = tid;
    meta.width_W = w;
    meta.stride_flags = 0;
    meta.has_seeds = 1;
    memset(meta.seeds, in[0], sizeof meta.seeds);
    for (r = 0; r < (int)w; r++) segs[r] = plane + (size_t)r * lens[0];
    for (r = 0; r < reps; r++) {
        double t0 = now_ns(), t1;
        if (zpse_transform_forward(&meta, w, in, 65536, segs, lens) != PSE_OK)
            return 0.0;
        t1 = now_ns();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

static void profile_one(int idx, int reps)
{
    ZpseEncoder e;
    ZpseLaneState lanes;
    double block, f0, f1, f6, segment, tile = 1e18;
    uint32_t clen = 0;
    int r;

    make_pattern(idx);
    zpse_encoder_init(&e, E, sizeof E);
    lanes.order1 = o1;
    lanes.o2val = o2v;
    lanes.o2conf = o2c;
    {
        uint32_t l = 0;
        if (zpse_compress_block(in, sizeof in, comp, sizeof comp, &l, 0, 1, E)
            != 0) {
            printf("%-10s compress failed\n", names[idx]);
            return;
        }
        clen = l;
    }

    block = time_block(reps);
    f0 = time_forward(0, 1, reps);
    f1 = time_forward(1, 1, reps);
    f6 = time_forward(6, 16, reps);

    segment = 1e18;
    for (r = 0; r < reps; r++) {
        double t0, t1;
        zpse_lane_reset(&lanes, in[0]);
        t0 = now_ns();
        (void)zpse_segment_encode(&e, &lanes, in, 65536, in[0], comp,
                                  sizeof comp, 0u, 0u);
        t1 = now_ns();
        if (t1 - t0 < segment) segment = t1 - t0;
    }

    /* One tile, exactly as a segment would encode it: predict, search,
     * pack. 256 of these make up the segment line above. */
    for (r = 0; r < reps * 16; r++) {
        double t0, t1;
        int k;
        zpse_lane_reset(&lanes, in[0]);
        t0 = now_ns();
        for (k = 0; k < 64; k++)
            (void)zpse_tile_encode(&lanes, in + (size_t)k * 256u, 256, 1, 1,
                                   0, in[(size_t)k * 256u], 1, in, 0u, rec,
                                   sizeof rec, scratch);
        t1 = now_ns();
        if ((t1 - t0) / 64.0 < tile) tile = (t1 - t0) / 64.0;
    }

    printf("%-10s clen=%6u  block %7.2f  fwd id %5.2f d8 %5.2f t16 %6.2f"
           "  segment %7.2f  tile %6.1f (x256 %7.2f) us\n",
           names[idx], clen, block / 1000.0, f0 / 1000.0, f1 / 1000.0,
           f6 / 1000.0, segment / 1000.0, tile / 1000.0,
           tile * 256.0 / 1000.0);
}

int main(int argc, char **argv)
{
    int reps = 5;
    int i;

    if (argc > 1) {
        reps = atoi(argv[1]);
        if (reps < 1) reps = 1;
    }
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

    if (argc > 2) {
        for (i = 2; i < argc; i++) {
            int j;
            for (j = 0; j < np; j++) {
                if (strcmp(argv[i], names[j]) == 0) {
                    profile_one(j, reps);
                    break;
                }
            }
        }
    } else {
        for (i = 0; i < np; i++) profile_one(i, reps);
    }
    return 0;
}
