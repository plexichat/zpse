/* Rate benchmark for the two intra-block LZ codecs, by file.
 *
 * tests/suites/test_lz.c pins what the codecs write; nothing pinned how fast
 * they write it, which is how a finder can lose a third of its rate to a
 * change that leaves every stream identical. This prints that rate beside the
 * payload sizes, so a finder regression is a row that moved.
 *
 * Usage: bench_lz2 [file ...]
 *   No files: three built-in 64 KiB patterns -- english prose, a sixteen-value
 *             alphabet and random bytes -- so CI needs no corpus.
 *   Files:    every complete 64 KiB block of each file, which is the unit the
 *             block encoder actually hands the finder.
 *
 *   ZPSE_LZ_MIN_MBPS      exit non-zero when any row's encode rate is below
 *                         this, for a job that wants a floor (0 = report
 *                         only). A floor has to be loose -- CI hosts differ by
 *                         2-4x -- so it catches a codec that collapsed, not a
 *                         few percent.
 *   ZPSE_LZ_MIN_DEC_MBPS  the same floor for the fast form's decode rate.
 *
 * Both codecs get the same table, cleared before each call the way the
 * encoder clears it, and the same best-of-3 timing of whole-block calls. The
 * last column is the fast form's decode rate over payloads packed once, so a
 * regression in the command loop is a row that moved rather than a claim
 * nobody re-measured. Links src/zpse_unity.c.
 */
/* clock_gettime needs POSIX >= 199309 on glibc when -std=c11 is used
 * strictly, which is how CI builds this file. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

#include "zpse/zpse.h"
#include "zpse/lz.h"
#include "zpse/lz2.h"

#define BLOCK 65536u
/* One block's payload ceiling: the fast form codes a token per four bytes at
 * worst, which is 16 KiB over the block, and every payload is held so the
 * decode is timed on its own. */
#define PAYCAP 70000u

static uint8_t in[BLOCK];
static uint8_t payload[200000];
static uint8_t outbuf[BLOCK];
static uint16_t tab[ZPSE_LZ_TABLE_ENTRIES];
static uint8_t *filebuf;

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

/* One codec over `blocks` blocks of `data`, best of three passes. Returns the
 * rate in MB/s and, through *bytes_out, the last block's payload size. */
static double rate(int which, const uint8_t *data, uint32_t blocks,
                   uint32_t *bytes_out)
{
    double best = 0.0;
    uint32_t last = 0u;
    int rep;

    for (rep = 0; rep < 3; rep++) {
        double t0, t1;
        uint32_t b;

        t0 = now_ns();
        for (b = 0; b < blocks; b++) {
            memcpy(in, data + (size_t)b * BLOCK, BLOCK);
            memset(tab, 0xFF, sizeof tab);
            last = (which == 0)
                       ? zpse_lz_encode(in, BLOCK, payload, sizeof payload,
                                        tab, ZPSE_LZ_HASH_BITS)
                       : zpse_lz2_encode(in, BLOCK, payload, sizeof payload,
                                         tab, ZPSE_LZ_HASH_BITS);
            if (last == 0u) {
                return -1.0;
            }
        }
        t1 = now_ns();
        {
            double ns = t1 - t0;

            if (ns > 0.0) {
                double mb = (double)blocks * (double)BLOCK / (ns / 1e9) / 1e6;

                if (mb > best) {
                    best = mb;
                }
            }
        }
    }
    *bytes_out = last;
    return best;
}

/* The fast form's decode rate: every block packed once, then decoded best of
 * three out of one reused block buffer. The payloads are held so the clock
 * covers the decoder alone -- packing inside it would time the finder, which
 * is the column to the left. *packed_out reports the total payload, so a row
 * says which stream was decoded. Returns -1.0 when a codec refuses a block. */
static double dec_rate(const uint8_t *data, uint32_t blocks,
                       uint32_t *packed_out)
{
    uint8_t *packed = (uint8_t *)malloc((size_t)blocks * PAYCAP);
    uint32_t *lens = (uint32_t *)malloc((size_t)blocks * sizeof(uint32_t));
    double best = 0.0;
    uint32_t total = 0u;
    uint32_t b;
    int rep;

    *packed_out = 0u;
    if (packed == 0 || lens == 0) {
        free(packed);
        free(lens);
        return -1.0;
    }
    for (b = 0; b < blocks; b++) {
        memset(tab, 0xFF, sizeof tab);
        lens[b] = zpse_lz2_encode(data + (size_t)b * BLOCK, BLOCK,
                                  packed + (size_t)b * PAYCAP, PAYCAP, tab,
                                  ZPSE_LZ_HASH_BITS);
        if (lens[b] == 0u) {
            free(packed);
            free(lens);
            return -1.0;
        }
        total += lens[b];
    }
    for (rep = 0; rep < 3; rep++) {
        double t0 = now_ns();
        double ns;

        for (b = 0; b < blocks; b++) {
            if (zpse_lz2_decode(packed + (size_t)b * PAYCAP, lens[b], outbuf,
                                BLOCK) != 0) {
                free(packed);
                free(lens);
                return -1.0;
            }
        }
        ns = now_ns() - t0;
        if (ns > 0.0) {
            double mb = (double)blocks * (double)BLOCK / (ns / 1e9) / 1e6;

            if (mb > best) {
                best = mb;
            }
        }
    }
    *packed_out = total;
    free(packed);
    free(lens);
    return best;
}

int main(int argc, char **argv)
{
    const char *floor_env = getenv("ZPSE_LZ_MIN_MBPS");
    const char *dec_env = getenv("ZPSE_LZ_MIN_DEC_MBPS");
    double floor_mb = floor_env != 0 ? atof(floor_env) : 0.0;
    double dec_floor = dec_env != 0 ? atof(dec_env) : 0.0;
    int bad = 0;
    int arg;

    printf("block=%u B, best of 3 passes, one thread, table %u entries\n",
           (unsigned)BLOCK, (unsigned)ZPSE_LZ_TABLE_ENTRIES);
    printf("%-14s %7s %10s %10s %8s %10s\n", "input", "blocks", "lz1 MB/s",
           "lz2 MB/s", "lz2 B", "dec MB/s");

    if (argc > 1) {
        for (arg = 1; arg < argc; arg++) {
            FILE *f = fopen(argv[arg], "rb");
            long sz;
            size_t got;
            uint32_t blocks;
            uint32_t b1 = 0u, b2 = 0u, dp = 0u;
            double r1, r2, d2;

            if (f == 0) {
                fprintf(stderr, "bench_lz2: cannot open %s\n", argv[arg]);
                bad = 1;
                continue;
            }
            fseek(f, 0, SEEK_END);
            sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz < (long)BLOCK) {
                fclose(f);
                continue;
            }
            filebuf = (uint8_t *)malloc((size_t)sz);
            if (filebuf == 0) {
                fclose(f);
                bad = 1;
                continue;
            }
            got = fread(filebuf, 1, (size_t)sz, f);
            fclose(f);
            blocks = (uint32_t)(got / BLOCK);
            r1 = rate(0, filebuf, blocks, &b1);
            r2 = rate(1, filebuf, blocks, &b2);
            d2 = dec_rate(filebuf, blocks, &dp);
            free(filebuf);
            filebuf = 0;
            if (r1 < 0.0 || r2 < 0.0 || d2 < 0.0) {
                printf("%-14s %7u   codec refused a block\n", argv[arg],
                       (unsigned)blocks);
                bad = 1;
                continue;
            }
            printf("%-14s %7u %10.1f %10.1f %8u %10.1f\n", argv[arg],
                   (unsigned)blocks, r1, r2, (unsigned)b2, d2);
            if ((floor_mb > 0.0) && (r1 < floor_mb || r2 < floor_mb)) {
                bad = 1;
            }
            if ((dec_floor > 0.0) && (d2 < dec_floor)) {
                bad = 1;
            }
        }
        return bad;
    }

    /* The three built-in patterns: 64 KiB each, in the order a reader would
     * care about them -- prose is the finder's slow case and random its
     * fastest. */
    {
        static const char prose[] =
            "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor "
            "sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
            "incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
            "veniam, quis nostrud exercitation ullamco laboris nisi ut "
            "aliquip ex ea commodo consequat. ";
        static const char *const names[3] = { "english", "lowentropy",
                                             "random" };
        uint32_t r = 0x1234u;
        int k;

        for (k = 0; k < 3; k++) {
            uint32_t i, b1 = 0u, b2 = 0u, dp = 0u;
            double r1, r2, d2;

            if (k == 0) {
                size_t pl = strlen(prose);

                for (i = 0; i < BLOCK; i++) {
                    in[i] = (uint8_t)prose[i % pl];
                }
            } else {
                for (i = 0; i < BLOCK; i++) {
                    r = r * 1664525u + 1013904223u;
                    in[i] = (uint8_t)((k == 1) ? (r >> 24) & 0x0Fu
                                               : (r >> 24));
                }
            }
            memcpy(filebuf = (uint8_t *)malloc(BLOCK), in, BLOCK);
            r1 = rate(0, filebuf, 1u, &b1);
            r2 = rate(1, filebuf, 1u, &b2);
            d2 = dec_rate(filebuf, 1u, &dp);
            free(filebuf);
            filebuf = 0;
            if (r1 < 0.0 || r2 < 0.0 || d2 < 0.0) {
                printf("%-14s %7u   codec refused the block\n", names[k], 1u);
                bad = 1;
                continue;
            }
            printf("%-14s %7u %10.1f %10.1f %8u %10.1f\n", names[k], 1u, r1, r2,
                   (unsigned)b2, d2);
            if ((floor_mb > 0.0) && (r1 < floor_mb || r2 < floor_mb)) {
                bad = 1;
            }
            if ((dec_floor > 0.0) && (d2 < dec_floor)) {
                bad = 1;
            }
        }
    }
    return bad;
}
