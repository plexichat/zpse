/* zpse across all 9 effort levels on the columnar corpora.
 *
 * For each columnar corpus (same 7 shapes as bench_colref.c and
 * bench_columnar.c) this runs every level 1=FASTEST ... 9=MAX and
 * prints ratio, encode MB/s and decode MB/s per level, so the size
 * and speed ladder is visible column by column.
 *
 * The levels are monotonic in effort: sizes should never increase
 * from level 1 to level 9, while encode time should never decrease.
 *
 * Portable: no dlopen, no POSIX-only calls. Links zpse_unity.c.
 *
 * Usage: bench_levels [iters] [reps]
 *   iters  timed iterations per measurement (default 20)
 *   reps   timed runs, best-of (default 3)
 * Exit 0 on success. */

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

#define CHUNK (65536u)
#define CORPUS_BYTES ((size_t)CHUNK * 64u) /* 4 MiB per column */
#define NCHUNK (CORPUS_BYTES / CHUNK)

static uint8_t *g_in;
static uint8_t *g_comp;
static uint8_t *g_dec;
static unsigned char g_ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char g_da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq = {{0}};
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

static void fmt_ratio(char *b, size_t n, size_t in, size_t out)
{
    snprintf(b, n, "%.2f:1", out ? (double)in / (double)out : 0.0);
}

static void fmt_rate(char *b, size_t n, double mb_s)
{
    if (mb_s >= 1000.0) snprintf(b, n, "%.2f GB/s", mb_s / 1000.0);
    else snprintf(b, n, "%.0f MB/s", mb_s);
}

/* Total encoded size of the column under zpse at one level, block by
 * block, with every decode verified. */
static int zpse_size_at(uint32_t level, size_t *out_len)
{
    size_t total = 0;
    size_t c;
    uint32_t clen = 0;
    uint32_t olen = 0;

    for (c = 0; c < NCHUNK; c++) {
        const uint8_t *src = g_in + c * CHUNK;
        if (zpse_compress_block_ex(src, CHUNK, g_comp,
                                   (uint32_t)(CHUNK + 1024u), &clen, 0, 1,
                                   level, g_ea) != 0)
            return -1;
        if (zpse_decompress_block(g_comp, clen, g_dec, CHUNK, &olen, g_da) != 0)
            return -1;
        if (memcmp(g_dec, src, CHUNK) != 0) return -1;
        total += clen;
    }
    *out_len = total;
    return 0;
}

/* Best-of-reps encode throughput of zpse at one level over the column. */
static double zpse_enc_mbps(uint32_t level, int iters, int reps)
{
    double best = 0.0;
    int rep;
    size_t c;

    (void)iters;
    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns();
        double t1;
        for (c = 0; c < NCHUNK; c++) {
            uint32_t l = 0;
            (void)zpse_compress_block_ex(g_in + c * CHUNK, CHUNK, g_comp,
                                         (uint32_t)(CHUNK + 1024u), &l, 0, 1,
                                         level, g_ea);
        }
        t1 = now_ns();
        {
            double mb = (double)CORPUS_BYTES * 1000.0 / (t1 - t0);
            if (best == 0.0 || mb > best) best = mb;
        }
    }
    return best;
}

/* Best-of-reps decode throughput of zpse at one level over the column. */
static double zpse_dec_mbps(uint32_t level, int iters, int reps)
{
    double best = 0.0;
    int rep;
    size_t c;
    static uint8_t *comps;
    static uint32_t lens[NCHUNK];
    static uint32_t prepared_level = 0u;
    size_t c2;

    (void)iters;
    if (prepared_level != level) {
        if (comps == NULL) {
            comps = malloc((CHUNK + 1024u) * NCHUNK);
            if (comps == NULL) return 0.0;
        }
        for (c2 = 0; c2 < NCHUNK; c2++) {
            uint32_t l = 0;
            zpse_compress_block_ex(g_in + c2 * CHUNK, CHUNK,
                                   comps + c2 * (CHUNK + 1024u),
                                   (uint32_t)(CHUNK + 1024u), &l, 0, 1,
                                   level, g_ea);
            lens[c2] = l;
        }
        prepared_level = level;
    }

    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns();
        double t1;
        for (c = 0; c < NCHUNK; c++) {
            uint32_t l = 0;
            (void)zpse_decompress_block(comps + c * (CHUNK + 1024u), lens[c],
                                        g_dec, CHUNK, &l, g_da);
        }
        t1 = now_ns();
        {
            double mb = (double)CORPUS_BYTES * 1000.0 / (t1 - t0);
            if (best == 0.0 || mb > best) best = mb;
        }
    }
    return best;
}

static void report_levels(int kind, int iters, int reps)
{
    ZpseColGen g;
    size_t c;
    size_t lens[9];
    double encs[9];
    double decs[9];
    char rb[24];
    char eb[24];
    char db[24];
    size_t li;

    zpse_col_gen_init(&g, kind);
    for (c = 0; c < NCHUNK; c++)
        zpse_col_gen_fill(&g, g_in + c * CHUNK, CHUNK);

    for (li = 0u; li < 9u; li++) {
        uint32_t level = (uint32_t)(li + 1u);
        if (zpse_size_at(level, &lens[li]) != 0) {
            lens[li] = 0;
        }
        encs[li] = zpse_enc_mbps(level, iters, reps);
        decs[li] = zpse_dec_mbps(level, iters, reps);
    }

    printf("\n%s: zpse across 9 levels, 4 MiB as %zu x 64 KiB blocks\n",
           zpse_col_names[kind], NCHUNK);
    printf("| level | name | ratio | encode | decode | bytes |");
    printf("\n|---|---|---:|---:|---:|---:|");
    for (li = 0u; li < 9u; li++) {
        uint32_t level = (uint32_t)(li + 1u);
        const char *name;

        switch (level) {
        case 1u: name = "FASTEST"; break;
        case 2u: name = "2"; break;
        case 3u: name = "FASTER"; break;
        case 4u: name = "4"; break;
        case 5u: name = "BALANCED"; break;
        case 6u: name = "6"; break;
        case 7u: name = "TIGHTER"; break;
        case 8u: name = "8"; break;
        default: name = "MAX"; break;
        }
        fmt_ratio(rb, sizeof rb, CORPUS_BYTES, lens[li]);
        fmt_rate(eb, sizeof eb, encs[li]);
        fmt_rate(db, sizeof db, decs[li]);
        printf("\n| %u | %s | %s | %s | %s | %zu |",
               level, name, rb, eb, db, lens[li]);
    }
    /* Monotonicity check: bytes should not grow from level to level. */
    {
        int ok = 1;
        for (li = 1u; li < 9u; li++) {
            if (lens[li] != 0 && lens[li - 1u] != 0 &&
                lens[li] > lens[li - 1u]) {
                ok = 0;
                break;
            }
        }
        printf("\n  sizes monotonic: %s\n", ok ? "yes" : "NO");
    }
}

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    int k;

    if (iters < 1) iters = 1;
    if (reps < 1) reps = 1;

    g_in = malloc(CORPUS_BYTES);
    g_comp = malloc(CHUNK + 1024u);
    g_dec = malloc(CHUNK);
    if (!g_in || !g_comp || !g_dec) {
        printf("oom\n");
        return 1;
    }

    printf("zpse levels on columnar corpora: 4 MiB per column as %u x 64 KiB "
           "blocks, single core, best-of-%d reps\n",
           (unsigned)NCHUNK, reps);
    printf("levels: 1=FASTEST 2 3=FASTER 4 5=BALANCED 6 7=TIGHTER 8 9=MAX\n");

    for (k = 0; k < (int)ZPSE_COL_KINDS; k++)
        report_levels(k, iters, reps);

    free(g_in);
    free(g_comp);
    free(g_dec);
    return 0;
}
