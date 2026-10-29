/* Standalone benchmark for zpse_lz_encode -- the intra-block LZ match
 * finder by itself, not the full encoder. Links zpse_unity.c.
 *
 * Usage: bench_lz.exe [iters] [pattern ...]
 *
 * Patterns: zeros, const5A, random, incr, strided, alt00FF, sparse, sawtooth,
 * lowentropy, textlike.  With no arguments all run.
 *
 * Best-of-reps timing; reports MB/s (1e6 B/s) of uncompressed input and
 * the payload size the matcher produced.
 *
 * These are the match finder's own numbers, so the slowest row is not a
 * worst case for the encoder: `lowentropy` (a random stream over sixteen
 * byte values) is the slow one, at 0.3-0.4 GB/s against `random`'s 17 GB/s
 * depending on the host's state, and the difference is what the data leaves
 * the finder to do.
 * `random` never passes the 4-byte compare, so the miss accelerator walks
 * the block in about 2 000 iterations; `lowentropy` passes it 4 655 times,
 * each hit resetting that accelerator and restarting the loop, and the
 * 4-byte matches it finds are worth 400 bytes in 65 536 (0.994 of the
 * input) because a length-4 match pays for its own command word and no
 * more. On 64 KiB blocks of this pattern no level pays that pass: the LZ
 * screen prices the head's tiles at 766 bytes against the finder's 1 044
 * and refuses at every rung from FASTEST to MAX, so the row measures work
 * the encoder does not do. What it spends its 81 us on instead is the
 * decision probe, which prices ten candidates exactly on the block's head,
 * and two full trials: the identity, which is written at 33 288 bytes, and
 * a transpose behind it, whose stream has already outgrown the block when
 * it overflows the candidate buffer (PSE_OUTPUT_OVERFLOW, not a budget
 * stop). See tests/tools/enc_profile and the ZPSE_BLOCK_STATS counters in
 * src/block.c for the split.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif
#include "zpse/zpse.h"
#include "zpse/lz.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t payload[140000];
static uint16_t tab[ZPSE_LZ_TABLE_ENTRIES];

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq = {{0}};
    LARGE_INTEGER c;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

static void make_pattern(int idx)
{
    int i;
    uint32_t r;
    switch (idx) {
    case 0: /* zeros */
        memset(in, 0, sizeof in);
        break;
    case 1: /* const5A */
        memset(in, 0x5A, sizeof in);
        break;
    case 2: /* random */
        r = 0x1234;
        for (i = 0; i < 65536; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)(r >> 24);
        }
        break;
    case 3: /* incr */
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)i;
        break;
    case 4: /* strided */
        for (i = 0; i < 65536; i++)
            in[i] = (uint8_t)((i / 4 + (i % 4) * 64) & 0xFF);
        break;
    case 5: /* alt00FF */
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i & 1) ? 0xFF : 0x00);
        break;
    case 6: /* sparse */
        memset(in, 0, sizeof in);
        for (i = 0; i < 65536; i += 97) in[i] = (uint8_t)i;
        break;
    case 7: /* sawtooth */
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i * 7u) & 0xFF);
        break;
    case 8: /* lowentropy (4-bit alphabet) */
        r = 0x1234;
        for (i = 0; i < 65536; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)((r >> 24) & 0x0F);
        }
        break;
    case 9: /* textlike */
        for (i = 0; i < 65536; i++)
            in[i] = (uint8_t)("the quick brown fox "[(i * 3) % 20]);
        break;
    case 10: /* english prose: word boundaries, spaces, punctuation, mixed case */
        {
            static const char prose[] =
                "The quick brown fox jumps over the lazy dog. "
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna "
                "aliqua. Ut enim ad minim veniam, quis nostrud exercitation.";
            size_t plen = strlen(prose);
            for (i = 0; i < 65536; i++)
                in[i] = (uint8_t)prose[i % plen];
        }
        break;
    default:
        memset(in, 0, sizeof in);
        break;
    }
}

static const char *const pat_names[] = {
    "zeros",   "const5A", "random",  "incr",    "strided",
    "alt00FF", "sparse",  "sawtooth","lowentropy","textlike","english"
};
static const int npat = (int)(sizeof pat_names / sizeof pat_names[0]);

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 500;
    int reps = argc > 2 ? atoi(argv[2]) : 7;
    int order[16];
    int nord = 0;
    int rep, j, k, i;
    double best[16];

    for (i = 3; i < argc; i++) {
        int found = -1;
        for (j = 0; j < npat; j++)
            if (strcmp(argv[i], pat_names[j]) == 0) { found = j; break; }
        if (found < 0) {
            fprintf(stderr, "unknown pattern: %s\n", argv[i]);
            return 2;
        }
        if (nord >= 16) return 2;
        order[nord++] = found;
    }
    if (nord == 0)
        for (j = 0; j < npat; j++) order[nord++] = j;

    for (j = 0; j < nord; j++) best[j] = 0.0;

    printf("iters=%d reps=%d block=65536 B, single core, best-of-reps\n",
           iters, reps);
    printf("%-11s %10s %10s %10s %10s\n",
           "pattern", "payload B", "ratio", "MB/s (enc)", "ns/B");

    for (rep = 0; rep < reps; rep++) {
        for (j = 0; j < nord; j++) {
            make_pattern(order[j]);
            memset(tab, 0xFF, sizeof tab);

            /* warmup */
            (void)zpse_lz_encode(in, sizeof in, payload, sizeof payload,
                                 tab, ZPSE_LZ_HASH_BITS);

            {
                double t0 = now_ns();
                uint32_t pl = 0u;
                for (k = 0; k < iters; k++) {
                    memset(tab, 0xFF, sizeof tab);
                    pl = zpse_lz_encode(in, sizeof in, payload, sizeof payload,
                                        tab, ZPSE_LZ_HASH_BITS);
                }
                double t1 = now_ns();
                double ns = (t1 - t0) / iters;
                if (best[j] == 0.0 || ns < best[j]) best[j] = ns;

                if (rep == reps - 1) {
                    double mb = 65536.0 / (ns / 1e9) / 1e6;
                    double ratio = (double)pl / 65536.0;
                    printf("%-11s %10u %10.3f %10.2f %10.2f\n",
                           pat_names[order[j]], pl, ratio, mb, ns);
                }
            }
        }
    }
    return 0;
}
