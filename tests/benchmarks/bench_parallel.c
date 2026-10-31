/* Scaling across workers: encode and decode of one frame as the worker
 * count rises, at five frame sizes.
 *
 * Every other benchmark in this tree measures one core, because the numbers
 * in the README are single-core numbers. This one exists to answer the one
 * question those cannot: how much of a frame's encode a second, third or
 * sixth worker actually takes off the wall clock, and from what frame size
 * it starts to pay. Encode is the side that needs it -- the codec encodes at
 * one to two gigabytes a second and decodes at ten to twenty -- but the read
 * side is measured too, since a frame with the index decodes block by block
 * and the workers cost nothing there.
 *
 * The corpus is the columnar generator set cycled through megabyte regions,
 * so a frame is a mix of timestamp, text, float and offset data rather than
 * one pattern that happens to thread well or badly. Every frame is checked:
 * the workers must write the bytes one worker writes, and every decode is
 * compared whole against the input.
 *
 * Usage: bench_parallel [max_threads] [reps]
 *   max_threads  worker counts 1..max_threads (default 6)
 *   reps         timed runs per measurement, best of (default 4)
 *
 * Pin the workers to real cores from outside (`taskset -c 0-5`): two workers
 * sharing a core measure the scheduler, not the codec.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "zpse/zpse.h"

#include "../corpus/columnar_gen.h"

#define MAXT 6u
#define BLK 65536u
#define REGION (4u * 1024u * 1024u)

static unsigned char enc_mem[MAXT][ZPSE_ENC_ARENA_SIZE]
    __attribute__((aligned(64)));
static unsigned char dec_mem[MAXT][ZPSE_DEC_ARENA_SIZE]
    __attribute__((aligned(64)));
static void *enc_arenas[MAXT] = { enc_mem[0], enc_mem[1], enc_mem[2],
                                  enc_mem[3], enc_mem[4], enc_mem[5] };
static void *dec_arenas[MAXT] = { dec_mem[0], dec_mem[1], dec_mem[2],
                                  dec_mem[3], dec_mem[4], dec_mem[5] };
static unsigned char stage[MAXT * (BLK + 64u)];

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq = {{0}};
    LARGE_INTEGER c;

    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

/* A frame's worth of the columnar corpus: the seven typed generators in
 * turn, four megabytes each. */
static void fill_mixed(uint8_t *buf, size_t n)
{
    static const uint32_t kinds[] = {
        ZPSE_COL_TS_SEQ, ZPSE_COL_VAR_STRING, ZPSE_COL_FLOAT_SPARSE,
        ZPSE_COL_STRUCT_OFFSETS, ZPSE_COL_TS_NS, ZPSE_COL_SPARSE_MIX,
        ZPSE_COL_GEO_POINT
    };
    size_t off = 0;
    unsigned k = 0;

    while (off < n) {
        size_t chunk = n - off;
        ZpseColGen g;

        if (chunk > REGION) {
            chunk = REGION;
        }
        zpse_col_gen_init(&g, kinds[k % (sizeof kinds / sizeof kinds[0])]);
        zpse_col_gen_fill(&g, buf + off, chunk);
        off += chunk;
        k++;
    }
}

static double enc_once(const uint8_t *in, size_t n, uint8_t *comp, size_t cap,
                       size_t *len, unsigned threads)
{
    double t0 = now_ns();

    if (zpse_compress_frame_parallel(in, n, comp, cap, len, BLK, 0, 1,
                                     threads, (void *const *)enc_arenas,
                                     MAXT) != PSE_OK) {
        return 0.0;
    }
    return now_ns() - t0;
}

static double dec_once(const uint8_t *comp, size_t len, uint8_t *out, size_t n,
                       unsigned threads)
{
    double t0 = now_ns();
    size_t got = 0;
    int rc;

    rc = zpse_decompress_frame_parallel(comp, len, out, n, &got, threads,
                                        (void *const *)dec_arenas, MAXT, stage);
    if (rc != PSE_OK || got != n) {
        return 0.0;
    }
    return now_ns() - t0;
}

int main(int argc, char **argv)
{
    static const size_t sizes[] = { 256u * 1024u, 1024u * 1024u,
                                    4u * 1024u * 1024u, 16u * 1024u * 1024u,
                                    64u * 1024u * 1024u };
    unsigned max_t = argc > 1 ? (unsigned)atoi(argv[1]) : MAXT;
    int reps = argc > 2 ? atoi(argv[2]) : 4;
    size_t cap = zpse_parallel_bound(sizes[4], BLK, 1);
    uint8_t *in = malloc(sizes[4]);
    uint8_t *comp = malloc(cap);
    uint8_t *ref = malloc(cap);
    uint8_t *out = malloc(sizes[4]);
    size_t si;

    if (in == 0 || comp == 0 || ref == 0 || out == 0) {
        printf("oom\n");
        return 1;
    }
#if !defined(ZPSE_MT)
    /* Without the switch every worker count is the serial encoder, which
     * would print a flat table that says nothing about scaling. Say so
     * rather than measure it. */
    printf("built without ZPSE_MT: every row would be one core. Rebuild with "
           "-DZPSE_MT -pthread.\n");
    return 2;
#endif
    if (max_t < 1u || max_t > MAXT) {
        max_t = MAXT;
    }
    if (reps < 1) {
        reps = 1;
    }
    printf("zpse parallel scaling: 64 KiB blocks, mixed columnar corpus, "
           "best of %d rep(s)\n", reps);
    printf("| frame | workers | encode | speedup | decode | speedup | "
           "identical frames |\n|---|---:|---:|---:|---:|---:|---|\n");

    for (si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        size_t n = sizes[si];
        size_t ref_len = 0;
        double enc_gbs[MAXT];
        double dec_gbs[MAXT];
        unsigned t;
        int r;

        fill_mixed(in, n);
        if (enc_once(in, n, ref, cap, &ref_len, 1u) == 0.0) {
            printf("encode failed at %lu\n", (unsigned long)n);
            return 1;
        }

        for (t = 1; t <= max_t; t++) {
            double best_e = 0.0;
            double best_d = 0.0;
            int same = 1;
            size_t len = 0;

            for (r = 0; r < reps; r++) {
                double dt = enc_once(in, n, comp, cap, &len, t);

                if (dt == 0.0) {
                    printf("encode failed at %lu, %u workers\n",
                           (unsigned long)n, t);
                    return 1;
                }
                if (best_e == 0.0 || dt < best_e) {
                    best_e = dt;
                }
            }
            /* Bytes, not just length: a worker that wrote a different frame
             * would still decode, so the frame itself is the check. */
            if (len != ref_len || memcmp(comp, ref, ref_len) != 0) {
                same = 0;
            }
            for (r = 0; r < reps; r++) {
                double dt = dec_once(comp, len, out, n, t);

                if (dt == 0.0) {
                    printf("decode failed at %lu, %u workers\n",
                           (unsigned long)n, t);
                    return 1;
                }
                if (memcmp(out, in, n) != 0) {
                    printf("decode MISMATCH at %lu, %u workers\n",
                           (unsigned long)n, t);
                    return 1;
                }
                if (best_d == 0.0 || dt < best_d) {
                    best_d = dt;
                }
            }
            enc_gbs[t - 1u] = (double)n / best_e;
            dec_gbs[t - 1u] = (double)n / best_d;
            printf("| %lu KiB | %u | %.2f GB/s | %.2fx | %.2f GB/s | %.2fx | "
                   "%s |\n",
                   (unsigned long)(n / 1024u), t, enc_gbs[t - 1u],
                   enc_gbs[t - 1u] / enc_gbs[0], dec_gbs[t - 1u],
                   dec_gbs[t - 1u] / dec_gbs[0], same ? "yes" : "NO");
        }
    }
    free(in);
    free(comp);
    free(ref);
    free(out);
    return 0;
}
