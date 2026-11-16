/* Per-page codec comparison, statically linked: every 4 KiB page is its own
 * block. This is the Windows/no-dlopen counterpart of bench_pages.c: lz4
 * 1.10.0 is linked in rather than loaded at run time, so the 4 KiB cross-codec
 * table can be regenerated on hosts without dlopen (see README's 4 KiB
 * section). Build with the vendored lz4 sources:
 *
 *   clang -std=c11 -O2 -Iinclude -I<lz4>/lib \
 *       tests/benchmarks/bench_pages_static.c \
 *       <lz4>/lib/lz4.c <lz4>/lib/lz4hc.c src/zpse_unity.c -o bench_pages_static
 *
 * Every fairness rule of bench_pages.c is kept: each page is an independent
 * block, every page round-trips under every codec before timing, packed bytes
 * are exact, and two length bytes are charged to each raw lz4 block (zpse's
 * 16-byte page header is already in its block). Timed loops are best-of-N
 * and contain no I/O.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#endif

#include "zpse/zpse.h"
#include "lz4.h"
#include "lz4hc.h"

/* ZPSE_ENC_PAGE_INSTANT is being added to the public header. This harness
 * must keep building before that lands, so supply the bit value here when
 * the installed header does not define it yet. It is bit 13, next to the
 * other page-profile bits (PAGE_TURBO is bit 10). */
#ifndef ZPSE_ENC_PAGE_INSTANT
#define ZPSE_ENC_PAGE_INSTANT (UINT32_C(1) << 13)
#endif

#define PAGE 4096u
#define COMP_CAP (PAGE * 2u)

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char comp[COMP_CAP];
static unsigned char out[PAGE];
static int exclude_zero;
static int no_lz4;                 /* set by --no-lz4 */
static const char *modes_filter;    /* comma-separated keys, set by --modes= */

/* Return 1 when a table row should be emitted. --no-lz4 drops every lz4 key
 * (lz4 and lz4hc) outright; --modes=KEY,KEY,... keeps only the named keys.
 * Keys are short, space-free names: level1, level6, level9, flat, fast, lzf,
 * lzh, turbo, instant, lz4, hc4, hc9, hc12. With no filter every row runs. */
static int mode_wanted(const char *key)
{
    const char *p;

    if (no_lz4 && (strcmp(key, "lz4") == 0 || strncmp(key, "hc", 2) == 0))
        return 0;
    if (modes_filter == NULL) return 1;

    p = modes_filter;
    while (*p != '\0') {
        const char *end = strchr(p, ',');
        size_t n = end ? (size_t)(end - p) : strlen(p);
        if (strlen(key) == n && strncmp(p, key, n) == 0) return 1;
        if (end == NULL) break;
        p = end + 1;
    }
    return 0;
}

static int run_codec(const char *name, const unsigned char *in, size_t npages,
                     int kind, int param, int reps);

static void emit(const char *key, const char *name, const unsigned char *in,
                 size_t npages, int kind, int param, int reps)
{
    if (mode_wanted(key)) run_codec(name, in, npages, kind, param, reps);
}

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq, have;
    LARGE_INTEGER t;
    if (!have.QuadPart) {
        QueryPerformanceFrequency(&freq);
        have.QuadPart = 1;
    }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

static int all_zero(const unsigned char *p)
{
    size_t i;
    for (i = 0; i < PAGE; i++)
        if (p[i] != 0) return 0;
    return 1;
}

/* kind: 0 zpse level, 9 zpse FLAT (default level flags, no flag bit),
 * 1 zpse PAGE_FAST, 5 zpse LZ_FAST, 6 zpse LZ_HEX, 7 zpse PAGE_TURBO,
 * 8 zpse PAGE_INSTANT, 3 lz4hc, 4 lz4 default. */
static int run_codec(const char *name, const unsigned char *in, size_t npages,
                     int kind, int param, int reps)
{
    unsigned char *packed = (unsigned char *)malloc(npages * COMP_CAP);
    uint32_t *lens = (uint32_t *)malloc(npages * sizeof *lens);
    void *lstate = NULL;
    size_t i, p, packed_total = 0;
    double best_e = 0.0, best_d = 0.0;
    int rep, ok = 1;

    if (packed == NULL || lens == NULL) {
        fprintf(stderr, "bench_pages_static: out of memory\n");
        free(packed); free(lens);
        return 1;
    }
    if ((kind == 3 || kind == 4)) lstate = malloc((size_t)LZ4_sizeofStateHC());
    if ((kind == 3 || kind == 4) && lstate == NULL) ok = 0;

    for (p = 0, i = 0; ok && p < npages; p++) {
        const unsigned char *src = in + p * PAGE;
        uint32_t cl = 0;
        int dl;

        if (exclude_zero && all_zero(src)) continue;
        switch (kind) {
        case 0:
        case 9:  /* FLAT: default level flags; the encoder emits FLAT itself. */
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        (uint32_t)param << 1, enc_arena);
            break;
        case 1:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        ((uint32_t)param << 1) | ZPSE_ENC_PAGE_FAST,
                                        enc_arena);
            break;
        case 5:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        ((uint32_t)param << 1) | ZPSE_ENC_LZ_FAST,
                                        enc_arena);
            break;
        case 6:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        ((uint32_t)param << 1) | ZPSE_ENC_LZ_HEX,
                                        enc_arena);
            break;
        case 7:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        ((uint32_t)param << 1) | ZPSE_ENC_PAGE_TURBO,
                                        enc_arena);
            break;
        case 8:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        ((uint32_t)param << 1) | ZPSE_ENC_PAGE_INSTANT,
                                        enc_arena);
            break;
        case 3:
            cl = (uint32_t)LZ4_compress_HC_extStateHC(lstate, (const char *)src,
                                                      (char *)comp, PAGE,
                                                      (int)sizeof comp, param);
            dl = (int)cl <= 0;
            break;
        default:
            cl = (uint32_t)LZ4_compress_default((const char *)src, (char *)comp,
                                                PAGE, (int)sizeof comp);
            dl = (int)cl <= 0;
            break;
        }
        if (dl != 0 || cl == 0 || cl + 2u > COMP_CAP) { ok = 0; break; }
        {
            uint32_t stored = cl + ((kind == 3 || kind == 4) ? 2u : 0u);
            lens[p] = stored;
            memcpy(packed + i, comp, cl);
            i += stored;
        }
        if ((kind == 3 || kind == 4)) {
            int stored = (int)lens[p] - 2;
            if (LZ4_decompress_safe((const char *)comp, (char *)out, stored,
                                    PAGE) != (int)PAGE ||
                memcmp(out, src, PAGE) != 0) ok = 0;
        } else {
            uint32_t ol = 0;
            if (zpse_decompress_block(comp, cl, out, PAGE, &ol, dec_arena) != 0 ||
                ol != PAGE || memcmp(out, src, PAGE) != 0) ok = 0;
        }
    }
    packed_total = i;
    if (!ok) {
        printf("| %s | FAILED | | |\n", name);
        free(packed); free(lens); free(lstate);
        return 1;
    }

    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns(), t1;
        for (p = 0; p < npages; p++) {
            const unsigned char *src = in + p * PAGE;
            uint32_t cl = 0;
            if (exclude_zero && all_zero(src)) continue;
            switch (kind) {
            case 0:
            case 9:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       (uint32_t)param << 1, enc_arena);
                break;
            case 1:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       ((uint32_t)param << 1) | ZPSE_ENC_PAGE_FAST,
                                       enc_arena);
                break;
            case 5:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       ((uint32_t)param << 1) | ZPSE_ENC_LZ_FAST,
                                       enc_arena);
                break;
            case 6:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       ((uint32_t)param << 1) | ZPSE_ENC_LZ_HEX,
                                       enc_arena);
                break;
            case 7:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       ((uint32_t)param << 1) | ZPSE_ENC_PAGE_TURBO,
                                       enc_arena);
                break;
            case 8:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       ((uint32_t)param << 1) | ZPSE_ENC_PAGE_INSTANT,
                                       enc_arena);
                break;
            case 3:
                LZ4_compress_HC_extStateHC(lstate, (const char *)src,
                                           (char *)comp, PAGE,
                                           (int)sizeof comp, param);
                break;
            default:
                LZ4_compress_default((const char *)src, (char *)comp, PAGE,
                                     (int)sizeof comp);
                break;
            }
        }
        t1 = now_ns();
        if (best_e == 0.0 || t1 - t0 < best_e) best_e = t1 - t0;

        t0 = now_ns();
        {
            size_t off = 0;
            for (p = 0; p < npages; p++) {
                if (exclude_zero && all_zero(in + p * PAGE)) continue;
                if ((kind == 3 || kind == 4)) {
                    int stored = (int)lens[p] - 2;
                    LZ4_decompress_safe((const char *)packed + off, (char *)out,
                                        stored, PAGE);
                } else {
                    uint32_t ol = 0;
                    zpse_decompress_block(packed + off, lens[p], out, PAGE, &ol,
                                          dec_arena);
                }
                off += lens[p];
            }
        }
        t1 = now_ns();
        if (best_d == 0.0 || t1 - t0 < best_d) best_d = t1 - t0;
    }

    {
        size_t measured = 0;
        for (p = 0; p < npages; p++)
            if (!(exclude_zero && all_zero(in + p * PAGE))) measured++;
        if (measured == 0u) {
            printf("| %-24s | (no pages) | | |\n", name);
        } else {
            printf("| %-24s | %8.2f:1 | %8.0f | %8.0f |\n", name,
                   (double)(measured * PAGE) / (double)packed_total,
                   (measured * PAGE) * 1000.0 / best_e,
                   (measured * PAGE) * 1000.0 / best_d);
        }
    }
    free(packed); free(lens); free(lstate);
    return 0;
}

int main(int argc, char **argv)
{
    FILE *f;
    long sz;
    unsigned char *in;
    size_t npages, p;
    int i, start = 1, reps = 3, rc = 0;
    const char *base;

    /* All options precede the first file and may appear in any order. */
    while (start < argc && argv[start][0] == '-') {
        if (strcmp(argv[start], "--exclude-zero") == 0) {
            exclude_zero = 1;
        } else if (strcmp(argv[start], "--no-lz4") == 0) {
            no_lz4 = 1;
        } else if (strncmp(argv[start], "--reps=", 7) == 0) {
            reps = atoi(argv[start] + 7);
            if (reps < 1) reps = 1;
        } else if (strncmp(argv[start], "--modes=", 8) == 0) {
            modes_filter = argv[start] + 8;
        } else {
            fprintf(stderr, "bench_pages_static: unknown option %s\n", argv[start]);
            return 2;
        }
        start++;
    }
    if (start >= argc) {
        fprintf(stderr,
                "usage: bench_pages_static [--exclude-zero] [--no-lz4] "
                "[--reps=N] [--modes=KEY,KEY,...] FILE...\n"
                "  modes: level1 level6 level9 flat fast lzf lzh turbo instant "
                "lz4 hc4 hc9 hc12\n");
        return 2;
    }
    for (i = start; i < argc; i++) {
        f = fopen(argv[i], "rb");
        if (f == NULL) {
            fprintf(stderr, "bench_pages_static: cannot open %s\n", argv[i]);
            return 1;
        }
        fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz <= 0 || (sz % (long)PAGE) != 0) {
            fprintf(stderr, "bench_pages_static: %s is not a page multiple\n", argv[i]);
            fclose(f); return 1;
        }
        in = (unsigned char *)malloc((size_t)sz);
        if (in == NULL || fread(in, 1, (size_t)sz, f) != (size_t)sz) {
            fprintf(stderr, "bench_pages_static: cannot read %s\n", argv[i]);
            free(in); fclose(f); return 1;
        }
        fclose(f);
        npages = (size_t)sz / PAGE;
        for (p = 0; p < npages; p++) (void)all_zero(in + p * PAGE);

        base = strrchr(argv[i], '/');
        if (!base) base = strrchr(argv[i], '\\');
        base = base ? base + 1 : argv[i];
        printf("### %s\n\n", base);
        printf("%zu pages\n", npages);
        if (exclude_zero) printf("zero pages excluded\n");
        printf("\n");
        printf("| codec | ratio | encode MB/s | decode MB/s |\n");
        printf("| --- | ---: | ---: | ---: |\n");
        emit("level1",  "zpse 1 FASTEST",     in, npages, 0, 1, reps);
        emit("level6",  "zpse 6 (default)",   in, npages, 0, 6, reps);
        emit("level9",  "zpse 9 MAX",         in, npages, 0, 9, reps);
        emit("flat",    "zpse FLAT (default)", in, npages, 9, 6, reps);
        emit("fast",    "zpse 6 PAGE_FAST",   in, npages, 1, 6, reps);
        emit("lzf",     "zpse 6 LZ_FAST",     in, npages, 5, 6, reps);
        emit("lzh",     "zpse 6 LZ_HEX",      in, npages, 6, 6, reps);
        emit("turbo",   "zpse 6 TURBO",       in, npages, 7, 6, reps);
        emit("instant", "zpse 6 PAGE_INSTANT", in, npages, 8, 6, reps);
        emit("lz4",     "lz4 1",              in, npages, 4, 1, reps);
        emit("hc4",     "lz4hc 4",            in, npages, 3, 4, reps);
        emit("hc9",     "lz4hc 9",            in, npages, 3, 9, reps);
        emit("hc12",    "lz4hc 12",           in, npages, 3, 12, reps);
        printf("\n");
        free(in);
    }
    return rc;
}
