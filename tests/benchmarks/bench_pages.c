/* Per-page codec comparison: every 4 KiB page is compressed on its own.
 *
 * Usage: bench_pages [--exclude-zero] FILE...
 *
 * zpse is linked in; zstd and lz4/lz4hc are loaded from the system shared
 * libraries at run time (dlopen) so the harness builds with no -dev packages.
 * Each page is one independent block for every codec, which is what a zram
 * device does and what a frame-block comparison at 64 KiB does not.
 *
 * `--exclude-zero` drops all-zero pages from the measurement and reports how
 * many were skipped. The page corpus keeps them in `pages-all.bin` on purpose:
 * untouched anonymous pages are common and compress almost to nothing, so a
 * ratio that silently includes them flatters every codec. `pages.txt` records
 * the count; the README quotes the non-zero number and shows the all-pages
 * number separately.
 *
 * Packed bytes include each codec's own framing. zpse's page header is 16
 * bytes; a zstd one-shot frame carries its own; an lz4 block has no header, so
 * two bytes of length are charged per page to make the comparison like for
 * like. Ratios are exact byte counts; the rates include the loop and are best
 * of three runs on one process.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zpse/zpse.h"

#define PAGE 4096u
#define COMP_CAP (PAGE * 2u)
#define LEVELS 4

typedef size_t (*zstd_bound_fn)(size_t);
typedef void *(*zstd_create_fn)(void);
typedef size_t (*zstd_free_fn)(void *);
typedef size_t (*zstd_cctx_fn)(void *, void *, size_t, const void *, size_t, int);
typedef size_t (*zstd_dec_fn)(void *, size_t, const void *, size_t);
typedef unsigned (*zstd_iserr_fn)(size_t);

typedef int (*lz4_hc_state_fn)(void *, const char *, char *, int, int, int);
typedef int (*lz4_def_fn)(const char *, char *, int, int);
typedef int (*lz4_dec_fn)(const char *, char *, int, int);
typedef int (*lz4_size_fn)(void);

static zstd_create_fn z_create;
static zstd_free_fn z_free;
static zstd_cctx_fn z_compress;
static zstd_dec_fn z_decompress;
static zstd_iserr_fn z_iserr;

static lz4_hc_state_fn l_hc;
static lz4_def_fn l_def;
static lz4_dec_fn l_dec;
static lz4_size_fn l_size;

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char comp[COMP_CAP];
static unsigned char out[PAGE];

static int exclude_zero;

static double now_ns(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int all_zero(const unsigned char *p)
{
    size_t i;

    for (i = 0; i < PAGE; i++)
        if (p[i] != 0) return 0;
    return 1;
}

static void *load_tools(void)
{
    void *zh = dlopen("libzstd.so.1", RTLD_LAZY);
    void *lh = dlopen("liblz4.so.1", RTLD_LAZY);

    if (zh == NULL) zh = dlopen("libzstd.so", RTLD_LAZY);
    if (lh == NULL) lh = dlopen("liblz4.so", RTLD_LAZY);
    if (zh != NULL) {
        z_create = (zstd_create_fn)dlsym(zh, "ZSTD_createCCtx");
        z_free = (zstd_free_fn)dlsym(zh, "ZSTD_freeCCtx");
        z_compress = (zstd_cctx_fn)dlsym(zh, "ZSTD_compressCCtx");
        z_decompress = (zstd_dec_fn)dlsym(zh, "ZSTD_decompress");
        z_iserr = (zstd_iserr_fn)dlsym(zh, "ZSTD_isError");
        if (!z_create || !z_free || !z_compress ||
            !z_decompress || !z_iserr) z_compress = NULL;
    }
    if (lh != NULL) {
        l_hc = (lz4_hc_state_fn)dlsym(lh, "LZ4_compress_HC_extStateHC");
        l_def = (lz4_def_fn)dlsym(lh, "LZ4_compress_default");
        l_dec = (lz4_dec_fn)dlsym(lh, "LZ4_decompress_safe");
        l_size = (lz4_size_fn)dlsym(lh, "LZ4_sizeofStateHC");
        if (!l_hc || !l_def || !l_dec || !l_size) l_hc = NULL;
    }
    return NULL;
}

/* One codec's page loop. `kind`: 0 zpse level, 1 zpse PAGE_FAST, 2 zstd,
 * 3 lz4hc, 4 lz4 default. `param`: level or flag. */
static int run_codec(const char *name, const unsigned char *in, size_t npages,
                     int kind, int param, int reps)
{
    unsigned char *packed = (unsigned char *)malloc(npages * COMP_CAP);
    uint32_t *lens = (uint32_t *)malloc(npages * sizeof *lens);
    size_t i, p, packed_total = 0;
    double best_e = 0.0, best_d = 0.0;
    void *zc = NULL;
    void *lstate = NULL;
    int rep, ok = 1;

    if (packed == NULL || lens == NULL) {
        fprintf(stderr, "bench_pages: out of memory\n");
        free(packed);
        free(lens);
        return 1;
    }
    if (kind == 2) zc = z_create();
    if (kind >= 3) lstate = malloc((size_t)l_size());
    if ((kind == 2 && zc == NULL) || (kind >= 3 && lstate == NULL)) ok = 0;

    /* One pass fills `packed` and `lens`; the same pass verifies decode. */
    for (p = 0, i = 0; ok && p < npages; p++) {
        const unsigned char *src = in + p * PAGE;
        uint32_t cl = 0;
        int dl;

        if (exclude_zero && all_zero(src)) continue;
        switch (kind) {
        case 0:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        (uint32_t)param << 1, enc_arena);
            break;
        case 1:
            dl = zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                        ((uint32_t)param << 1) | ZPSE_ENC_PAGE_FAST,
                                        enc_arena);
            break;
        case 2:
            cl = (uint32_t)z_compress(zc, comp, sizeof comp, src, PAGE, param);
            dl = z_iserr(cl) ? -1 : 0;
            break;
        case 3:
            cl = (uint32_t)l_hc(lstate, (const char *)src, (char *)comp, PAGE,
                                (int)sizeof comp, param);
            dl = (int)cl <= 0;
            break;
        default:
            cl = (uint32_t)l_def((const char *)src, (char *)comp, PAGE,
                                 (int)sizeof comp);
            dl = (int)cl <= 0;
            break;
        }
        if (dl != 0 || cl == 0 || cl + 2u > COMP_CAP) { ok = 0; break; }
        {
            uint32_t stored = cl + (kind >= 3 ? 2u : 0u);

            lens[p] = stored;
            memcpy(packed + i, comp, cl);
            i += stored;
        }

        if (kind == 0 || kind == 1) {
            uint32_t ol = 0;
            if (zpse_decompress_block(comp, cl, out, PAGE, &ol, dec_arena) != 0 ||
                ol != PAGE || memcmp(out, src, PAGE) != 0) ok = 0;
        } else if (kind == 2) {
            if (z_decompress(out, PAGE, comp, cl) != PAGE ||
                memcmp(out, src, PAGE) != 0) ok = 0;
        } else {
            if (l_dec((const char *)comp, (char *)out, (int)cl, PAGE) != (int)PAGE ||
                memcmp(out, src, PAGE) != 0) ok = 0;
        }
    }
    packed_total = i;
    if (!ok) {
        printf("| %s | FAILED | | |\n", name);
        free(packed);
        free(lens);
        if (zc) z_free(zc);
        free(lstate);
        return 1;
    }

    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns(), t1;
        size_t count = 0;

        for (p = 0; p < npages; p++) {
            const unsigned char *src = in + p * PAGE;
            uint32_t cl = 0;

            if (exclude_zero && all_zero(src)) continue;
            switch (kind) {
            case 0: case 1:
                zpse_compress_block_ex(src, PAGE, comp, sizeof comp, &cl, 0, 1,
                                       ((uint32_t)param << 1) |
                                           (kind == 1 ? ZPSE_ENC_PAGE_FAST : 0u),
                                       enc_arena);
                break;
            case 2:
                cl = (uint32_t)z_compress(zc, comp, sizeof comp, src, PAGE, param);
                break;
            case 3:
                cl = (uint32_t)l_hc(lstate, (const char *)src, (char *)comp, PAGE,
                                    (int)sizeof comp, param);
                break;
            default:
                cl = (uint32_t)l_def((const char *)src, (char *)comp, PAGE,
                                     (int)sizeof comp);
                break;
            }
            (void)cl;
            count++;
        }
        t1 = now_ns();
        if (best_e == 0.0 || t1 - t0 < best_e) best_e = t1 - t0;

        t0 = now_ns();
        {
            size_t off = 0;

            for (p = 0; p < npages; p++) {
                if (exclude_zero && all_zero(in + p * PAGE)) continue;
                if (kind == 0 || kind == 1) {
                    uint32_t ol = 0;
                    zpse_decompress_block(packed + off, lens[p], out, PAGE, &ol,
                                          dec_arena);
                } else if (kind == 2) {
                    z_decompress(out, PAGE, packed + off, lens[p]);
                } else {
                    int stored = lens[p] - 2;
                    l_dec((const char *)packed + off, (char *)out, stored, PAGE);
                }
                off += lens[p];
            }
        }
        t1 = now_ns();
        if (best_d == 0.0 || t1 - t0 < best_d) best_d = t1 - t0;
        (void)count;
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
    free(packed);
    free(lens);
    if (zc) z_free(zc);
    free(lstate);
    return 0;
}

static int bench_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    long sz;
    unsigned char *in;
    size_t npages, zeros = 0, p;
    const char *base = strrchr(path, '/');

    base = base ? base + 1 : path;
    if (f == NULL) {
        fprintf(stderr, "bench_pages: cannot open %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % (long)PAGE) != 0) {
        fprintf(stderr, "bench_pages: %s is not a page multiple\n", path);
        fclose(f);
        return 1;
    }
    in = (unsigned char *)malloc((size_t)sz);
    if (in == NULL || fread(in, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "bench_pages: cannot read %s\n", path);
        free(in);
        fclose(f);
        return 1;
    }
    fclose(f);
    npages = (size_t)sz / PAGE;
    for (p = 0; p < npages; p++)
        if (all_zero(in + p * PAGE)) zeros++;

    printf("### %s\n\n", base);
    printf("%zu pages, %zu zero (%.1f%%)%s\n\n", npages, zeros,
           npages ? 100.0 * (double)zeros / (double)npages : 0.0,
           exclude_zero ? ", zero pages excluded from the measurement" : "");
    printf("| codec | ratio | encode MB/s | decode MB/s |\n");
    printf("| --- | ---: | ---: | ---: |\n");
    run_codec("zpse 1 FASTEST", in, npages, 0, 1, 3);
    run_codec("zpse 6 (default)", in, npages, 0, 6, 3);
    run_codec("zpse 9 MAX", in, npages, 0, 9, 3);
    run_codec("zpse 6 PAGE_FAST", in, npages, 1, 6, 3);
    if (z_compress) {
        run_codec("zstd 1", in, npages, 2, 1, 3);
        run_codec("zstd 6", in, npages, 2, 6, 3);
        run_codec("zstd 19", in, npages, 2, 19, 1);
    }
    if (l_hc) {
        run_codec("lz4 1", in, npages, 4, 1, 3);
        run_codec("lz4hc 4", in, npages, 3, 4, 3);
        run_codec("lz4hc 9", in, npages, 3, 9, 3);
        run_codec("lz4hc 12", in, npages, 3, 12, 3);
    }
    printf("\n");
    free(in);
    return 0;
}

int main(int argc, char **argv)
{
    int i, start = 1, rc = 0;

    load_tools();
    if (argc > 1 && strcmp(argv[1], "--exclude-zero") == 0) {
        exclude_zero = 1;
        start = 2;
    }
    if (start >= argc) {
        fprintf(stderr, "usage: bench_pages [--exclude-zero] FILE...\n");
        return 2;
    }
    for (i = start; i < argc; i++)
        if (bench_file(argv[i]) != 0) rc = 1;
    return rc;
}
