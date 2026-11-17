/* Focused encode + decode + ratio harness for TEXT, so a "faster on text"
 * claim can be measured honestly before and after a change.
 *
 * The corpus is generated in memory from a fixed pool of word-repetitive
 * sentences (an LCG picks the next sentence). Nothing is read from disk and
 * nothing is written into the repository, so two builds of the same source
 * see byte-identical input. Two workloads are measured: 4 KiB pages and 64
 * KiB blocks, over a fixed 4 MiB of prose each. For each workload and level
 * the harness first verifies every block round-trips, then times best-of-N
 * encode and best-of-N decode (no I/O inside the timed loops) and prints a
 * markdown table.
 *
 * Portable: clock_gettime on POSIX, QueryPerformanceCounter on Windows.
 * No dlopen. Links src/zpse_unity.c. lz4 rows are opt-in at build time:
 *
 *   cc -std=c11 -O2 -Iinclude tests/benchmarks/bench_text.c \
 *      src/zpse_unity.c -o bench_text
 *
 * With lz4 (define ZPSE_TEXT_HAVE_LZ4 and add its lib dir):
 *
 *   cc -std=c11 -O2 -DZPSE_TEXT_HAVE_LZ4=1 -Iinclude -I<lz4>/lib \
 *      tests/benchmarks/bench_text.c <lz4>/lib/lz4.c <lz4>/lib/lz4hc.c \
 *      src/zpse_unity.c -o bench_text
 *
 * Usage: bench_text [--reps=N] [--levels=1,6,9] [--size=N] [--no-lz4]
 * Exit 0 on success.
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

#ifdef ZPSE_TEXT_HAVE_LZ4
#include "lz4.h"
#include "lz4hc.h"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ALIGN64 __attribute__((aligned(64)))
#else
#define ALIGN64
#endif

#define DEFAULT_CORPUS (4u * 1024u * 1024u)
#define MAX_BLOCK      (65536u)

/* Buffers, sized for the largest block we ever run. */
static uint8_t *g_in;        /* the prose corpus */
static uint8_t *g_comp;      /* one scratch compressed block */
static uint8_t *g_dec;       /* whole decoded corpus (verify target) */
static uint8_t *g_packed;    /* all compressed blocks, laid out back to back */
static uint32_t *g_lens;     /* stored length per block */
static size_t *g_offs;       /* offset of each block in g_packed */
static unsigned char g_ea[ZPSE_ENC_ARENA_SIZE] ALIGN64;
static unsigned char g_da[ZPSE_DEC_ARENA_SIZE] ALIGN64;

static size_t g_corpus = DEFAULT_CORPUS;
static size_t g_block;       /* block size of the current workload */
static size_t g_nblocks;     /* number of blocks in the current workload */
static size_t g_cap;         /* capacity of g_comp */
static int g_reps = 3;
static int g_no_lz4;
static const char *g_levels = "6";

#ifdef ZPSE_TEXT_HAVE_LZ4
static void *g_lz4hc_state;
#endif

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

/* ---- deterministic prose generator ------------------------------------ */

/* Word-repetitive prose: a fixed pool of sentences, including the canonical
 * "the quick brown fox ..." phrase and the river/valley family, chosen in a
 * fixed order by an LCG. Repeating whole sentences gives the match finder
 * real redundancy to find, which is the workload "faster on text" is about. */
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

/* ---- codec abstraction ------------------------------------------------- */

enum { CODEC_ZPSE = 0, CODEC_LZ4 = 1, CODEC_LZ4HC = 2 };

typedef struct {
    const char *name;
    int kind;
    int param;
} Codec;

static uint32_t level_flags(int level)
{
    return (uint32_t)level << ZPSE_ENC_SPEED_SHIFT;
}

/* Compress one block into g_comp, return stored length in *cl. */
static void compress_one(const Codec *c, const uint8_t *src, uint32_t *cl)
{
    switch (c->kind) {
    default:
        (void)zpse_compress_block_ex(src, (uint32_t)g_block, g_comp,
                                     (uint32_t)g_cap, cl, 0, 1,
                                     level_flags(c->param), g_ea);
        break;
#ifdef ZPSE_TEXT_HAVE_LZ4
    case CODEC_LZ4:
        *cl = (uint32_t)LZ4_compress_default((const char *)src, (char *)g_comp,
                                             (int)g_block, (int)g_cap);
        break;
    case CODEC_LZ4HC:
        *cl = (uint32_t)LZ4_compress_HC_extStateHC(g_lz4hc_state,
                                                   (const char *)src,
                                                   (char *)g_comp,
                                                   (int)g_block, (int)g_cap,
                                                   c->param);
        break;
#endif
    }
}

/* Decode one block from g_packed at index b into g_dec + b*g_block. */
static void decode_one(const Codec *c, size_t b)
{
    const uint8_t *src = g_packed + g_offs[b];
    uint32_t cl = g_lens[b];
    uint8_t *dst = g_dec + b * g_block;

#ifdef ZPSE_TEXT_HAVE_LZ4
    if (c->kind != CODEC_ZPSE) {
        cl = (uint32_t)src[0] | ((uint32_t)src[1] << 8);
        src += 2;
        (void)LZ4_decompress_safe((const char *)src, (char *)dst,
                                  (int)cl, (int)g_block);
        return;
    }
#else
    (void)c;
#endif
    {
        uint32_t ol = 0;
        (void)zpse_decompress_block(src, cl, dst, (uint32_t)g_block, &ol, g_da);
    }
}

/* Encode every block; return total stored bytes (lz4 pays 2 length bytes a
 * block so its ratio is charged the framing a block store would need). */
static size_t encode_all(const Codec *c, const uint8_t *in)
{
    size_t b, off = 0;
    int prefix = 0;

#ifdef ZPSE_TEXT_HAVE_LZ4
    prefix = (c->kind != CODEC_ZPSE);
#endif
    for (b = 0; b < g_nblocks; b++) {
        uint32_t cl = 0;
        compress_one(c, in + b * g_block, &cl);
        g_offs[b] = off;
        if (prefix) {
            g_packed[off++] = (uint8_t)(cl & 0xFFu);
            g_packed[off++] = (uint8_t)((cl >> 8) & 0xFFu);
        }
        g_lens[b] = cl;
        memcpy(g_packed + off, g_comp, cl);
        off += cl;
    }
    return off;
}

static void decode_all(const Codec *c)
{
    size_t b;
    for (b = 0; b < g_nblocks; b++) decode_one(c, b);
}

typedef struct {
    double ratio;
    double enc;
    double dec;
    int ok;
} Result;

static Result run_case(const Codec *c, const uint8_t *in)
{
    Result r;
    size_t total, rep;
    double best_e = 0.0, best_d = 0.0;

    r.ratio = 0.0; r.enc = 0.0; r.dec = 0.0; r.ok = 1;

    /* Verify pass: exact bytes back, before any timing is believed. */
    total = encode_all(c, in);
    decode_all(c);
    if (memcmp(g_dec, in, g_nblocks * g_block) != 0) {
        r.ok = 0;
        return r;
    }
    r.ratio = total ? (double)(g_nblocks * g_block) / (double)total : 0.0;

    for (rep = 0; rep < (size_t)g_reps; rep++) {
        double t0 = now_ns(), t1;
        (void)encode_all(c, in);
        t1 = now_ns();
        if (best_e == 0.0 || t1 - t0 < best_e) best_e = t1 - t0;

        t0 = now_ns();
        decode_all(c);
        t1 = now_ns();
        if (best_d == 0.0 || t1 - t0 < best_d) best_d = t1 - t0;
    }
    if (best_e > 0.0) r.enc = (double)(g_nblocks * g_block) * 1000.0 / best_e;
    if (best_d > 0.0) r.dec = (double)(g_nblocks * g_block) * 1000.0 / best_d;
    return r;
}

/* ---- driver ------------------------------------------------------------ */

static int is_level(const char *s, int *lv)
{
    char *end;
    long v;
    if (s[0] < '0' || s[0] > '9') return 0;
    v = strtol(s, &end, 10);
    if (*end != '\0' || v < 1 || v > 9) return 0;
    *lv = (int)v;
    return 1;
}

static void run_levels(const char *workload)
{
    const char *p = g_levels;

    while (*p != '\0') {
        const char *end = strchr(p, ',');
        size_t n = end ? (size_t)(end - p) : strlen(p);
        char key[16];
        int level;
        Codec c;
        Result r;

        if (n == 0 || n >= sizeof key || !is_level(p, &level)) {
            fprintf(stderr, "bench_text: bad level in --levels=%s\n", g_levels);
            return;
        }
        memcpy(key, p, n); key[n] = '\0';
        c.name = key;
        c.kind = CODEC_ZPSE;
        c.param = level;

        r = run_case(&c, g_in);
        if (!r.ok)
            printf("| %s | %s | FAILED | | |\n", workload, c.name);
        else
            printf("| %s | %-7s | %7.2f:1 | %8.0f | %8.0f |\n",
                   workload, c.name, r.ratio, r.enc, r.dec);

        if (end == NULL) break;
        p = end + 1;
    }
}

#ifndef ZPSE_TEXT_HAVE_LZ4
static void run_lz4(const char *workload) { (void)workload; }
#else
static void run_lz4(const char *workload)
{
    static const struct { const char *name; int kind; int param; } specs[] = {
        { "lz4",  CODEC_LZ4,   0 },
        { "hc9",  CODEC_LZ4HC, 9 },
        { "hc12", CODEC_LZ4HC, 12 }
    };
    size_t i;
    if (g_no_lz4) return;
    for (i = 0; i < sizeof specs / sizeof specs[0]; i++) {
        Codec c;
        Result r;
        c.name = specs[i].name;
        c.kind = specs[i].kind;
        c.param = specs[i].param;
        r = run_case(&c, g_in);
        if (!r.ok)
            printf("| %s | %s | FAILED | | |\n", workload, c.name);
        else
            printf("| %s | %-7s | %7.2f:1 | %8.0f | %8.0f |\n",
                   workload, c.name, r.ratio, r.enc, r.dec);
    }
}
#endif

int main(int argc, char **argv)
{
    int a;

    for (a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--reps=", 7) == 0) {
            g_reps = atoi(argv[a] + 7);
            if (g_reps < 1) g_reps = 1;
        } else if (strncmp(argv[a], "--levels=", 9) == 0) {
            g_levels = argv[a] + 9;
        } else if (strncmp(argv[a], "--size=", 7) == 0) {
            unsigned long long v = strtoull(argv[a] + 7, NULL, 10);
            if (v >= MAX_BLOCK && (v % MAX_BLOCK) == 0) g_corpus = (size_t)v;
            else {
                fprintf(stderr, "bench_text: --size must be a multiple of %u\n", MAX_BLOCK);
                return 2;
            }
        } else if (strcmp(argv[a], "--no-lz4") == 0) {
            g_no_lz4 = 1;
        } else {
            fprintf(stderr,
                    "usage: bench_text [--reps=N] [--levels=1,6,9] "
                    "[--size=N] [--no-lz4]\n");
            return 2;
        }
    }

    g_in = (uint8_t *)malloc(g_corpus);
    g_comp = (uint8_t *)malloc(MAX_BLOCK + 1024u);
    g_dec = (uint8_t *)malloc(g_corpus);
    g_packed = (uint8_t *)malloc(g_corpus * 2u);
    g_lens = (uint32_t *)malloc((g_corpus / 4096u + 1u) * sizeof *g_lens);
    g_offs = (size_t *)malloc((g_corpus / 4096u + 1u) * sizeof *g_offs);
    if (!g_in || !g_comp || !g_dec || !g_packed || !g_lens || !g_offs) {
        fprintf(stderr, "bench_text: out of memory\n");
        return 1;
    }
    gen_prose(g_in, g_corpus);

#ifdef ZPSE_TEXT_HAVE_LZ4
    if (!g_no_lz4) {
        g_lz4hc_state = malloc((size_t)LZ4_sizeofStateHC());
        if (g_lz4hc_state == NULL) {
            fprintf(stderr, "bench_text: out of memory (lz4)\n");
            return 1;
        }
    }
    g_cap = (size_t)LZ4_compressBound((int)MAX_BLOCK);
#else
    g_cap = MAX_BLOCK + 1024u;
#endif

    printf("zpse text harness: deterministic word-repetitive prose, "
           "%.2f MiB per workload, best-of-%d reps, levels=%s\n",
           (double)g_corpus / 1048576.0, g_reps, g_levels);
#ifdef ZPSE_TEXT_HAVE_LZ4
    if (!g_no_lz4) printf("lz4 rows enabled (2 length bytes charged per block)\n");
#endif
    printf("\n| workload | mode | ratio | encode MB/s | decode MB/s |\n");
    printf("| --- | --- | ---: | ---: | ---: |\n");

    {
        static const struct { const char *name; size_t block; } shapes[] = {
            { "text-4K",  4096u },
            { "text-64K", MAX_BLOCK }
        };
        size_t s;
        for (s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
            g_block = shapes[s].block;
            g_nblocks = g_corpus / g_block;
            run_levels(shapes[s].name);
            run_lz4(shapes[s].name);
        }
    }

    free(g_in); free(g_comp); free(g_dec); free(g_packed);
    free(g_lens); free(g_offs);
#ifdef ZPSE_TEXT_HAVE_LZ4
    free(g_lz4hc_state);
#endif
    return 0;
}
