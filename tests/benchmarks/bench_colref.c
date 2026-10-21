/* What a purpose-built codec for each columnar shape achieves.
 *
 * tests/benchmarks/bench_columnar.c answers "how does zpse compare with
 * the other general-purpose block codecs on this data". That comparison
 * says zpse loses to zstd -- but zstd is a *block* codec, so it only
 * shows that the data has structure a match finder likes. It does not
 * say what the structure is worth when it is coded for what it is.
 *
 * This tool closes that gap. For each columnar corpus it runs a
 * reference codec built for that shape -- a delta-of-delta for
 * nanosecond timestamps, a null bitmap for a nullable column, a frame
 * of reference for increasing offsets, Gorilla XOR for repeated
 * floats -- and prints that size and speed next to zpse's on the same
 * bytes.
 *
 * The gap between the two rows is the price of zpse's design: no
 * schema, no column semantics, one byte-oriented model for everything.
 * That is the number that decides whether a typed stage is worth
 * adding, and it is worth having before anyone argues for one.
 *
 * The reference codecs are not competitors and are not a proposal to
 * put a columnar stage inside zpse. They are reference points, and
 * tests/suites/test_colref.c proves each one round-trips, so a size
 * here is a size a real codec reached.
 *
 * Framing: both sides see the same 64 KiB chunks. zpse compresses a
 * 64 KiB block; the reference codec encodes a 64 KiB chunk of the
 * column. Neither sees the whole column at once, so the comparison is
 * about the coding model and not about the window.
 *
 * Portable: no dlopen, no POSIX-only calls, so it runs on Windows and
 * in CI. Links zpse_unity.c.
 *
 * Usage: bench_colref [iters] [reps]
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
#include "col_ref.h"

#define CHUNK (65536u)
#define CORPUS_BYTES ((size_t)CHUNK * 64u) /* 4 MiB per column */
#define NCHUNK (CORPUS_BYTES / CHUNK)

static uint8_t *g_in;
static uint8_t *g_scratch;
static uint8_t *g_dec;
static uint8_t *g_comp;
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

/* Total encoded size of the column under the reference codec, chunk by
 * chunk. Each chunk is decoded and compared against its source, so a
 * size is only reported for a codec that demonstrably round-trips. */
static int ref_size(int kind, size_t *out_len)
{
    const ZpseColRef *r = zpse_colref_lookup(kind);
    size_t total = 0;
    size_t c;

    if (r == NULL) return -1;
    for (c = 0; c < NCHUNK; c++) {
        const uint8_t *src = g_in + c * CHUNK;
        size_t n = 0;
        if (r->encode(src, CHUNK, g_scratch, 2u * CHUNK + 64u, &n) != 0)
            return -1;
        if (r->decode(g_scratch, n, g_dec, CHUNK) != 0) return -1;
        if (memcmp(g_dec, src, CHUNK) != 0) return -1;
        total += n;
    }
    *out_len = total;
    return 0;
}

/* Total encoded size of the column under zpse, block by block, with
 * every decode verified. */
static int zpse_size(size_t *out_len)
{
    size_t total = 0;
    size_t c;
    uint32_t clen = 0;
    uint32_t olen = 0;

    for (c = 0; c < NCHUNK; c++) {
        const uint8_t *src = g_in + c * CHUNK;
        if (zpse_compress_block(src, CHUNK, g_comp, (uint32_t)(CHUNK + 1024u),
                                &clen, 0, 1, g_ea) != 0)
            return -1;
        if (zpse_decompress_block(g_comp, clen, g_dec, CHUNK, &olen, g_da) != 0)
            return -1;
        if (memcmp(g_dec, src, CHUNK) != 0) return -1;
        total += clen;
    }
    *out_len = total;
    return 0;
}

/* Best-of-reps throughput of the reference codec over the whole
 * column, in MB/s of uncompressed bytes. Returns 0 when the column
 * has no reference codec. */
static double ref_mbps(int kind, int iters, int reps, int decode)
{
    const ZpseColRef *r = zpse_colref_lookup(kind);
    double best = 0.0;
    int rep;
    size_t c;

    if (r == NULL) return 0.0;
    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns();
        double t1;
        if (decode) {
            for (c = 0; c < NCHUNK; c++) {
                size_t n = 0;
                (void)r->encode(g_in + c * CHUNK, CHUNK, g_scratch,
                                2u * CHUNK + 64u, &n);
                (void)r->decode(g_scratch, n, g_dec, CHUNK);
            }
        } else {
            for (c = 0; c < NCHUNK; c++) {
                size_t n = 0;
                (void)r->encode(g_in + c * CHUNK, CHUNK, g_scratch,
                                2u * CHUNK + 64u, &n);
            }
        }
        t1 = now_ns();
        {
            double mb = (double)CORPUS_BYTES * 1000.0 / (t1 - t0);
            if (best == 0.0 || mb > best) best = mb;
        }
    }
    (void)iters;
    return best;
}

/* Best-of-reps throughput of zpse over the whole column, in MB/s. */
static double zpse_mbps(int iters, int reps, int decode)
{
    double best = 0.0;
    int rep;
    size_t c;

    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns();
        double t1;
        if (decode) {
            for (c = 0; c < NCHUNK; c++) {
                uint32_t l = 0;
                (void)zpse_compress_block(g_in + c * CHUNK, CHUNK, g_comp,
                                          (uint32_t)(CHUNK + 1024u), &l, 0, 1,
                                          g_ea);
                (void)zpse_decompress_block(g_comp, l, g_dec, CHUNK, &l, g_da);
            }
        } else {
            for (c = 0; c < NCHUNK; c++) {
                uint32_t l = 0;
                (void)zpse_compress_block(g_in + c * CHUNK, CHUNK, g_comp,
                                          (uint32_t)(CHUNK + 1024u), &l, 0, 1,
                                          g_ea);
            }
        }
        t1 = now_ns();
        {
            double mb = (double)CORPUS_BYTES * 1000.0 / (t1 - t0);
            if (best == 0.0 || mb > best) best = mb;
        }
    }
    (void)iters;
    return best;
}

static void report(int kind, int iters, int reps)
{
    const ZpseColRef *r = zpse_colref_lookup(kind);
    size_t ref_len = 0;
    size_t zp_len = 0;
    char rb[24];
    char zb[24];
    char eb[24];
    char db[24];
    ZpseColGen g;
    size_t c;

    zpse_col_gen_init(&g, kind);
    for (c = 0; c < NCHUNK; c++)
        zpse_col_gen_fill(&g, g_in + c * CHUNK, CHUNK);

    if (zpse_size(&zp_len) != 0) {
        printf("| %s | %s | FAILED | - | - | - | - |\n", zpse_col_names[kind],
               r ? r->name : "-");
        return;
    }
    if (r == NULL) {
        /* No column codec helps here: this is high-cardinality text,
         * where the right tool is a block codec with a match finder. */
        fmt_ratio(zb, sizeof zb, CORPUS_BYTES, zp_len);
        fmt_rate(eb, sizeof eb, zpse_mbps(iters, reps, 0));
        fmt_rate(db, sizeof db, zpse_mbps(iters, reps, 1));
        printf("| %s | (block codec) | - | %s | %s | %s | - |\n",
               zpse_col_names[kind], zb, eb, db);
        return;
    }
    if (ref_size(kind, &ref_len) != 0) {
        printf("| %s | %s | FAILED | - | - | - | - |\n", zpse_col_names[kind],
               r->name);
        return;
    }
    fmt_ratio(rb, sizeof rb, CORPUS_BYTES, ref_len);
    fmt_ratio(zb, sizeof zb, CORPUS_BYTES, zp_len);
    fmt_rate(eb, sizeof eb, ref_mbps(kind, iters, reps, 0));
    fmt_rate(db, sizeof db, ref_mbps(kind, iters, reps, 1));
    printf("| %s | %s | %s | %s | %s | %s | %.1fx |\n", zpse_col_names[kind],
           r->name, rb, zb, eb, db, ref_len ? (double)zp_len / (double)ref_len : 0.0);
}

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    int k;

    if (iters < 1) iters = 1;
    if (reps < 1) reps = 1;

    g_in = malloc(CORPUS_BYTES);
    g_scratch = malloc(2u * CHUNK + 64u);
    g_dec = malloc(CHUNK);
    g_comp = malloc(CHUNK + 1024u);
    if (!g_in || !g_scratch || !g_dec || !g_comp) {
        printf("oom\n");
        return 1;
    }

    printf("columnar reference codecs: 4 MiB per column as %u x 64 KiB "
           "chunks, same bytes both sides\n",
           (unsigned)NCHUNK);
    printf("reference = a codec built for that column shape; zpse = the "
           "general-purpose block codec; 'zpse is' = how many times bigger "
           "zpse's output is (below 1 means zpse wins)\n");
    printf("| column | reference codec | reference ratio | zpse ratio | "
           "reference MB/s | zpse MB/s | zpse is |\n");
    printf("|---|---|---:|---:|---:|---:|---:|\n");
    for (k = 0; k < (int)ZPSE_COL_KINDS; k++)
        report(k, iters, reps);

    free(g_in);
    free(g_scratch);
    free(g_dec);
    free(g_comp);
    return 0;
}
