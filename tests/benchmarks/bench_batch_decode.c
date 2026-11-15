/* Batch decode across workers, with the corpus already in memory.
 *
 * The streaming figure in docs/HISTORY.md is a process figure: a 202 MiB
 * frame read and written around the codec, where reading the frame is most of
 * the wall clock. That says how much a second worker takes off a decode of a
 * file, not what the codec does with one, so this holds the payloads in
 * memory and times zpse_decompress_batch_parallel alone at every worker
 * count. The difference between the two tables is the I/O.
 *
 * The blocks are packed once, by the same block encoder the frame encoder
 * calls, and every row is compared byte for byte against the input before it
 * is reported: a batch that decodes a block into the wrong slot is a wrong
 * answer, not a fast one. All rows decode the same blocks, so the rate column
 * is the only thing that changes.
 *
 * Usage: bench_batch_decode [FILE ...]
 *   Files: their bytes, 64 KiB at a time (up to 256 MiB).
 *   None:  64 MiB of the columnar generator set, so CI needs no corpus.
 *
 *   ZPSE_BD_MAXT  highest worker count (default 6, at most 8)
 *   ZPSE_BD_REPS  timed runs per row, best of (default 3)
 *   ZPSE_BD_MB    generated corpus size in MiB when there are no files
 *   ZPSE_BD_LEVEL block level to pack with (default 6)
 *   ZPSE_BD_PAGE  pack 4 KiB pages with ZPSE_ENC_PAGE_FAST instead of
 *                 64 KiB blocks, the shape a page frame has
 *
 * Prints a markdown table. Links src/zpse_unity.c; build with -DZPSE_MT for
 * more than one worker, or every row decodes on the caller.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "zpse/zpse.h"
#include "zpse/block.h"
#include "zpse/parallel.h"

#include "../corpus/columnar_gen.h"

#define BLK 65536u
#define PAGE 4096u
/* 256 MiB at either block size, so the two tables can cover the same bytes. */
#define MAXB 65536u
#define MAXT 8u
#define REGION (4u * 1024u * 1024u)

static unsigned char enc_mem[MAXT][ZPSE_ENC_ARENA_SIZE]
    __attribute__((aligned(64)));
static unsigned char dec_mem[MAXT][ZPSE_DEC_ARENA_SIZE]
    __attribute__((aligned(64)));
static void *enc_arenas[MAXT] = { enc_mem[0], enc_mem[1], enc_mem[2],
                                  enc_mem[3], enc_mem[4], enc_mem[5],
                                  enc_mem[6], enc_mem[7] };
static void *dec_arenas[MAXT] = { dec_mem[0], dec_mem[1], dec_mem[2],
                                  dec_mem[3], dec_mem[4], dec_mem[5],
                                  dec_mem[6], dec_mem[7] };

static uint32_t offs[MAXB];
static uint32_t lens[MAXB];
static uint32_t blen[MAXB];
static uint32_t got[MAXB];

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

static unsigned env_u(const char *name, unsigned fallback)
{
    const char *s = getenv(name);
    long v;

    if (s == NULL || *s == '\0') {
        return fallback;
    }
    v = strtol(s, NULL, 10);
    return v > 0 ? (unsigned)v : fallback;
}

/* The columnar corpus, the same mix bench_parallel uses: a frame of typed
 * columns rather than one pattern that happens to thread well. */
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

int main(int argc, char **argv)
{
    unsigned max_t = env_u("ZPSE_BD_MAXT", 6u);
    unsigned reps = env_u("ZPSE_BD_REPS", 3u);
    unsigned level = env_u("ZPSE_BD_LEVEL", 6u);
    unsigned page = env_u("ZPSE_BD_PAGE", 0u);
    uint32_t blk = page != 0u ? PAGE : BLK;
    uint32_t enc_flags = page != 0u ? (uint32_t)ZPSE_ENC_PAGE_FAST : 0u;
    size_t cap = (size_t)env_u("ZPSE_BD_MB", 64u) * 1024u * 1024u;
    size_t total = 0;
    uint32_t nblk, i, poff = 0;
    uint8_t *in, *pay, *out;
    double base = 0.0;
    int a;

    if (max_t > MAXT) {
        max_t = MAXT;
    }
    if (cap == 0 || cap > (size_t)MAXB * (size_t)blk) {
        cap = (size_t)MAXB * (size_t)blk;
    }
    in = malloc(cap);
    if (in == NULL) {
        fprintf(stderr, "bench_batch_decode: out of memory\n");
        return 1;
    }
    if (argc > 1) {
        for (a = 1; a < argc && total < cap; a++) {
            FILE *f = fopen(argv[a], "rb");
            size_t n;

            if (f == NULL) {
                perror(argv[a]);
                free(in);
                return 1;
            }
            n = fread(in + total, 1u, cap - total, f);
            total += n;
            fclose(f);
        }
    } else {
        fill_mixed(in, cap);
        total = cap;
    }
    if (total < blk) {
        fprintf(stderr, "bench_batch_decode: need at least one block\n");
        free(in);
        return 1;
    }
    nblk = (uint32_t)((total + blk - 1u) / blk);
    if (nblk > MAXB) {
        nblk = MAXB;
    }
    for (i = 0; i < nblk; i++) {
        size_t left = total - (size_t)i * (size_t)blk;

        blen[i] = (uint32_t)(left < blk ? left : blk);
    }
    pay = malloc(total + (size_t)nblk * 256u + 4096u);
    out = malloc((size_t)nblk * (size_t)blk);
    if (pay == NULL || out == NULL) {
        fprintf(stderr, "bench_batch_decode: out of memory\n");
        free(in);
        free(pay);
        free(out);
        return 1;
    }
    for (i = 0; i < nblk; i++) {
        uint32_t pl = 0;
        int rc = zpse_compress_block_ex(in + (size_t)i * (size_t)blk,
                                        blen[i], pay + poff, blk + 128u,
                                        &pl, i, 0,
                                        (uint32_t)(level << 1) | enc_flags,
                                        enc_arenas[0]);

        if (rc != ZPSE_OK || pl == 0u) {
            fprintf(stderr, "bench_batch_decode: block %u did not compress\n",
                    i);
            free(in);
            free(pay);
            free(out);
            return 1;
        }
        offs[i] = poff;
        lens[i] = pl;
        poff += pl;
    }

    printf("| workers | MB/s | speedup |\n|---:|---:|---:|\n");
    for (i = 1; i <= max_t; i++) {
        double best = 0.0;
        unsigned r;

        for (r = 0; r < reps; r++) {
            double t0 = now_ns();
            int rc = zpse_decompress_batch_parallel(
                pay, offs, lens, nblk, out, blk, got, i, dec_arenas, MAXT);

            if (rc != ZPSE_OK) {
                fprintf(stderr,
                        "bench_batch_decode: batch decode failed (%d) at %u "
                        "worker(s)\n",
                        rc, i);
                free(in);
                free(pay);
                free(out);
                return 1;
            }
            t0 = now_ns() - t0;
            if (best == 0.0 || t0 < best) {
                best = t0;
            }
        }
        for (uint32_t k = 0; k < nblk; k++) {
            if (got[k] != blen[k] ||
                memcmp(out + (size_t)k * (size_t)blk,
                       in + (size_t)k * (size_t)blk, blen[k]) != 0) {
                fprintf(stderr,
                        "bench_batch_decode: block %u differs at %u "
                        "worker(s)\n",
                        k, i);
                free(in);
                free(pay);
                free(out);
                return 1;
            }
        }
        if (base == 0.0) {
            base = best;
        }
        printf("| %u | %.0f | %.2f |\n", i,
               (double)total / 1e6 / (best / 1e9), base / best);
    }
    fprintf(stderr,
            "bench_batch_decode: %u blocks of %u B, %.1f MiB, level %u%s, "
            "%u rep(s)\n",
            nblk, blk, (double)total / 1048576.0, level,
            page != 0u ? " + PAGE_FAST" : "", reps);
#if !defined(ZPSE_MT)
    fprintf(stderr, "bench_batch_decode: built without ZPSE_MT; every row "
                    "decoded on the caller\n");
#endif
    free(in);
    free(pay);
    free(out);
    return 0;
}
