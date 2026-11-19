/* Focused A/B harness for FLAT7: encode rate, decode rate and ratio against
 * the normal frame path, on deterministic ASCII generated in C. Nothing is
 * read from disk and nothing is written into the repository, so two builds of
 * the same source see byte-identical input.
 *
 * Two corpora of the same size are measured: word-repetitive prose (which the
 * redundancy gate would decline) and high-entropy random printable ASCII
 * (which it accepts). Each is timed through both forms:
 *
 *   flat7  -- zpse_flat7_encode / zpse_flat7_decode, the whole-input 7-bit pack
 *   normal -- zpse_compress_frame_parallel with one worker and ZPSE_ENC_CHECKSUM
 *             (which FLAT7 declines) / zpse_decompress_frame, the ordinary
 *             frame path. One worker keeps the comparison single-threaded.
 *
 * Each case is round-tripped and compared whole before any timing is believed;
 * encode and decode are then timed best-of-N with no I/O inside the loops.
 *
 * Portable: clock_gettime on POSIX, QueryPerformanceCounter on Windows.
 * Links src/zpse_unity.c:
 *
 *   cc -std=c11 -O2 -Iinclude tests/benchmarks/bench_flat7.c \
 *      src/zpse_unity.c -o bench_flat7
 *
 * Usage: bench_flat7 [--reps=N] [--size=N]   (exit 0 on success)
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "zpse/zpse.h"

#if defined(__GNUC__) || defined(__clang__)
#define ALIGN64 __attribute__((aligned(64)))
#else
#define ALIGN64
#endif

#define DEFAULT_CORPUS (4u * 1024u * 1024u)
#define BLOCK_SIZE     (65536u)

/* Buffers, sized for the largest corpus we ever run. */
static uint8_t *g_in;        /* generated corpus */
static uint8_t *g_flat;      /* FLAT7 frame + the packer's 8-byte slack */
static uint8_t *g_norm;      /* normal frame, reserved slot layout */
static uint8_t *g_dec;       /* decode target */
static unsigned char g_ea[ZPSE_ENC_ARENA_SIZE] ALIGN64;
static unsigned char g_da[ZPSE_DEC_ARENA_SIZE] ALIGN64;
static void *g_enc_arenas[1];

static size_t g_corpus = DEFAULT_CORPUS;
static size_t g_norm_cap;
static int g_reps = 5;

/* ---- timing ------------------------------------------------------------- */

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq;
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

/* ---- deterministic corpora --------------------------------------------- */

/* Word-repetitive prose: a fixed pool of sentences chosen by an LCG. This is
 * all ASCII, but repetitive, so the redundancy gate would decline FLAT7 on it
 * -- the row shows what the fixed 1.143:1 pack costs against the ladder. */
static const char *const SENTENCES[] = {
    "the quick brown fox jumps over the lazy dog",
    "and the river flows quietly through the valley",
    "the lazy dog sleeps beneath the warm morning sun",
    "the quick brown fox runs across the open field",
    "the river flows past the old stone bridge",
    "and the sun rises over the quiet hills",
    "a soft wind blows through the tall green trees",
    "the dog and the fox rest by the flowing river",
    "the morning light falls across the quiet valley",
    "the brown fox jumps over the sleeping dog again",
    "the river and the wind move through the valley",
    "and the quick brown fox runs toward the river"
};

static void gen_prose(uint8_t *dst, size_t n)
{
    const size_t ns = sizeof SENTENCES / sizeof SENTENCES[0];
    size_t o = 0;
    unsigned col = 0;
    uint32_t s = 0x9E3779B9u;

    while (o < n) {
        const char *p;
        size_t l;
        s = s * UINT32_C(1664525) + UINT32_C(1013904223);
        p = SENTENCES[(s >> 8) % ns];
        l = strlen(p);

        if (col && (size_t)col + l + 1u > 76u) {
            dst[o++] = '\n'; col = 0;
            if (o >= n) break;
        }
        if (col) {
            dst[o++] = ' '; col++;
            if (o >= n) break;
        }
        while (l-- != 0 && o < n) {
            dst[o++] = (uint8_t)*p++;
            col++;
        }
    }
    while (o < n) dst[o++] = '\n';
}

/* splitmix64 -> one byte in [0x20, 0x7E]: deterministic, high-entropy
 * printable ASCII. Each 1 KiB window has almost no repeated 4-gram, which is
 * the class the redundancy gate accepts. */
static uint64_t rng_next(uint64_t *s)
{
    uint64_t z = (*s += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static void gen_printable(uint8_t *dst, size_t n)
{
    uint64_t s = UINT64_C(0x243F6A8885A308D3);
    size_t i;
    for (i = 0; i < n; i++) {
        dst[i] = (uint8_t)(0x20u + (unsigned)(rng_next(&s) % 95u));
    }
}

/* ---- the two paths ------------------------------------------------------ */

enum { MODE_FLAT7 = 0, MODE_NORMAL = 1 };

static uint8_t *frame_buf(int mode)
{
    return mode == MODE_FLAT7 ? g_flat : g_norm;
}

/* Encode the whole corpus into the mode's frame buffer; return its length. */
static size_t encode_all(int mode, const uint8_t *in)
{
    size_t len = 0;
    int rc;

    if (mode == MODE_FLAT7) {
        rc = zpse_flat7_encode(in, g_corpus, g_flat, g_corpus + 128u, &len);
    } else {
        rc = zpse_compress_frame_parallel(in, g_corpus, g_norm, g_norm_cap,
                                          &len, BLOCK_SIZE,
                                          ZPSE_ENC_CHECKSUM, 0, 1u,
                                          g_enc_arenas, 1u);
    }
    if (rc != ZPSE_OK) {
        fprintf(stderr, "bench_flat7: encode failed (%s)\n",
                zpse_status_name(rc));
        exit(1);
    }
    return len;
}

/* Decode the mode's frame into g_dec. */
static void decode_all(int mode, const uint8_t *frame, size_t len)
{
    size_t ol = 0;
    int rc;

    if (mode == MODE_FLAT7) {
        rc = zpse_flat7_decode(frame, len, g_dec, g_corpus, &ol);
    } else {
        rc = zpse_decompress_frame(frame, len, g_dec, g_corpus, &ol, g_da);
    }
    if (rc != ZPSE_OK || ol != g_corpus) {
        fprintf(stderr, "bench_flat7: decode failed (%s, ol=%lu)\n",
                zpse_status_name(rc), (unsigned long)ol);
        exit(1);
    }
}

typedef struct {
    double ratio;
    double enc;
    double dec;
} Result;

static Result run_case(int mode, const uint8_t *in)
{
    Result r;
    size_t len, rep;
    double best_e = 0.0, best_d = 0.0;
    uint8_t *frame;

    r.ratio = 0.0; r.enc = 0.0; r.dec = 0.0;

    /* Verify pass: exact bytes back, before any timing is believed. */
    len = encode_all(mode, in);
    frame = frame_buf(mode);
    decode_all(mode, frame, len);
    if (memcmp(g_dec, in, g_corpus) != 0) {
        fprintf(stderr, "bench_flat7: round-trip mismatch in mode %d\n", mode);
        exit(1);
    }
    r.ratio = len ? (double)g_corpus / (double)len : 0.0;

    for (rep = 0; rep < (size_t)g_reps; rep++) {
        double t0 = now_ns(), t1;

        (void)encode_all(mode, in);
        t1 = now_ns();
        if (best_e == 0.0 || t1 - t0 < best_e) best_e = t1 - t0;

        t0 = now_ns();
        decode_all(mode, frame_buf(mode), len);
        t1 = now_ns();
        if (best_d == 0.0 || t1 - t0 < best_d) best_d = t1 - t0;
    }
    if (best_e > 0.0) r.enc = (double)g_corpus * 1000.0 / best_e;
    if (best_d > 0.0) r.dec = (double)g_corpus * 1000.0 / best_d;
    return r;
}

static void run_workload(const char *name)
{
    static const char *const modes[] = { "flat7", "normal" };
    int m;

    /* The caller has already generated g_in for this workload. */
    for (m = 0; m < 2; m++) {
        Result r = run_case(m, g_in);
        printf("| %-11s | %-7s | %7.2f:1 | %9.0f | %9.0f |\n",
               name, modes[m], r.ratio, r.enc, r.dec);
    }
}

int main(int argc, char **argv)
{
    int a;

    for (a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--reps=", 7) == 0) {
            g_reps = atoi(argv[a] + 7);
            if (g_reps < 1) g_reps = 1;
        } else if (strncmp(argv[a], "--size=", 7) == 0) {
            unsigned long long v = strtoull(argv[a] + 7, NULL, 10);
            if (v >= BLOCK_SIZE && (v % BLOCK_SIZE) == 0) {
                g_corpus = (size_t)v;
            } else {
                fprintf(stderr, "bench_flat7: --size must be a multiple of %u\n",
                        BLOCK_SIZE);
                return 2;
            }
        } else {
            fprintf(stderr, "usage: bench_flat7 [--reps=N] [--size=N]\n");
            return 2;
        }
    }

    g_in = (uint8_t *)malloc(g_corpus);
    g_flat = (uint8_t *)malloc(g_corpus + 128u);
    g_norm_cap = zpse_parallel_bound(g_corpus, BLOCK_SIZE, 0);
    g_norm = (uint8_t *)malloc(g_norm_cap);
    g_dec = (uint8_t *)malloc(g_corpus);
    if (!g_in || !g_flat || !g_norm || !g_dec) {
        fprintf(stderr, "bench_flat7: out of memory\n");
        return 1;
    }
    g_enc_arenas[0] = g_ea;

    printf("zpse FLAT7 A/B harness: deterministic ASCII, %.2f MiB per corpus, "
           "best-of-%d, one worker\n", (double)g_corpus / 1048576.0, g_reps);
    printf("\n| workload | mode | ratio | encode MB/s | decode MB/s |\n");
    printf("| --- | --- | ---: | ---: | ---: |\n");

    gen_prose(g_in, g_corpus);
    run_workload("flat7-prose");
    gen_printable(g_in, g_corpus);
    run_workload("flat7-rand");

    free(g_in); free(g_flat); free(g_norm); free(g_dec);
    return 0;
}
