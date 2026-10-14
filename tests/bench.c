/* Throughput bench: compress + decompress GB/s per pattern, 64 KiB
 * blocks, single core. Usage: bench.exe [iters] [reps]. Links
 * zpse_unity.c. Exit 0.
 *
 * Every pattern is measured `reps` times and the best result is kept.
 * A single long pass is not a fair comparison: the CPU clock drops
 * under a sustained load, so a pattern measured late in the run
 * reports a slower rate than the same code measured early. Best-of-N
 * is the estimator that is stable against that drift.
 *
 * Also prints the measured hardware ceilings for this machine
 * (memcpy / memset / CRC32-C), which bound what any codec built on
 * this wire format can reach. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "zpse/zpse.h"
#include "zpse/crc32c.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t out[65536];
static uint8_t scratch[65536] __attribute__((aligned(64)));
static volatile uint32_t vsink = 0;

#define MAXP 16
static const char *names[MAXP];
static double best_enc_ns[MAXP] = { 0 }; /* 0 == not yet measured */
static double best_dec_ns[MAXP] = { 0 };
static uint32_t clen_of[MAXP];
static int np = 0;

static double now_ns(void) {
    static LARGE_INTEGER freq = {{0}};
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
}

/* Fill `in` with pattern number idx (order matches `names`). */
static void make_pattern(int idx) {
    int i;
    uint32_t r;
    switch (idx) {
    case 0:
        memset(in, 0, sizeof in);
        break;
    case 1:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)(i & 0xFF);
        break;
    case 2:
        r = 0x1234;
        for (i = 0; i < 65536; i++) { r = r * 1664525u + 1013904223u; in[i] = (uint8_t)(r >> 24); }
        break;
    case 3:
        memset(in, 0, sizeof in);
        for (i = 0; i < 65536; i += 97) in[i] = (uint8_t)i;
        break;
    case 4:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i / 4 + (i % 4) * 64) & 0xFF);
        break;
    case 5:
        memset(in, 0x5A, sizeof in);
        break;
    case 6:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i & 1) ? 0xFF : 0x00);
        break;
    case 7:
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)(i * 7 + (i >> 4));
        break;
    case 8:
        r = 0xBEEF;
        for (i = 0; i < 65536; i++) { r = r * 1664525u + 1013904223u; in[i] = (uint8_t)((r >> 24) & 0x0F); }
        break;
    default:
        memset(in, 0, sizeof in);
        for (i = 0; i < 65536; i += 256) { in[i] = (uint8_t)i; in[i + 1] = (uint8_t)(i >> 8); }
        break;
    }
}

/* One measurement round for pattern idx: stores the round's rates into
 * best_* when they beat what is already there. Returns 0 on success,
 * non-zero if the block failed to round-trip. */
static int bench(int idx, int iters) {
    const char *name = names[idx];
    uint32_t cl = 0, ol = 0;
    double t0, t1, enc_ns, dec_ns;
    int i, rc;

    if (np <= idx) np = idx + 1;
    rc = zpse_compress_block(in, sizeof in, comp, sizeof comp, &cl, 0, 1, E);
    if (rc != 0) { printf("%-10s compress failed rc=%d\n", name, rc); return 1; }
    rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
    if (rc != 0 || ol != sizeof in || memcmp(in, out, sizeof in) != 0) {
        printf("%-10s roundtrip FAILED rc=%d\n", name, rc);
        return 1;
    }
    clen_of[idx] = cl;

    t0 = now_ns();
    for (i = 0; i < iters; i++) {
        uint32_t l = 0;
        rc = zpse_compress_block(in, sizeof in, comp, sizeof comp, &l, 0, 1, E);
        if (rc != 0) break;
    }
    t1 = now_ns();
    enc_ns = (t1 - t0) / iters;

    t0 = now_ns();
    for (i = 0; i < iters * 10; i++) {
        uint32_t l = 0;
        rc = zpse_decompress_block(comp, cl, out, sizeof out, &l, D);
        if (rc != 0) break;
    }
    t1 = now_ns();
    dec_ns = (t1 - t0) / (iters * 10);

    if (best_enc_ns[idx] == 0.0 || enc_ns < best_enc_ns[idx]) best_enc_ns[idx] = enc_ns;
    if (best_dec_ns[idx] == 0.0 || dec_ns < best_dec_ns[idx]) best_dec_ns[idx] = dec_ns;
    return 0;
}

static void report(void) {
    int i;
    for (i = 0; i < np; i++) {
        double e = best_enc_ns[i], d = best_dec_ns[i];
        printf("%-10s ratio=%5.3f clen=%6u  enc=%8.2f GB/s (comp %7.2f ns/B)  dec=%8.2f GB/s (decomp %6.2f ns/B)\n",
               names[i], (double)clen_of[i] / 65536.0, clen_of[i],
               65536.0 / e, e / 65536.0, 65536.0 / d, d / 65536.0);
    }
}

static void ceilings(void) {
    const int n = 3000;
    double t0, t1;
    int i;
    printf("--- hardware ceilings (same 64 KiB working set, single core) ---\n");
    t0 = now_ns();
    for (i = 0; i < n; i++) { memcpy(scratch, in, 65536); vsink = scratch[i & 63]; }
    t1 = now_ns();
    printf("memcpy 64KiB            %8.2f GB/s\n", 65536.0 / ((t1 - t0) / n));
    t0 = now_ns();
    for (i = 0; i < n; i++) { memset(scratch, 0x5A, 65536); vsink = scratch[i & 63]; }
    t1 = now_ns();
    printf("memset 64KiB            %8.2f GB/s\n", 65536.0 / ((t1 - t0) / n));
    t0 = now_ns();
    for (i = 0; i < n; i++) vsink += zpse_crc32c(in, 65536);
    t1 = now_ns();
    printf("crc32c 64KiB            %8.2f GB/s\n", 65536.0 / ((t1 - t0) / n));
    printf("(both directions must also verify a CRC32-C of every 64 KiB block)\n");
}

int main(int argc, char **argv) {
    static const char *const pat_names[] = {
        "zeros", "incr", "random", "sparse", "strided",
        "const5A", "alt00FF", "sawtooth", "lowentropy", "sparse2"
    };
    const int npat = (int)(sizeof pat_names / sizeof pat_names[0]);
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int rep, i;

    for (i = 0; i < npat; i++) names[i] = pat_names[i];
    printf("iters=%d reps=%d (decode x10), block=65536 B, single core, best-of-reps\n",
           iters, reps);

    for (rep = 0; rep < reps; rep++) {
        for (i = 0; i < npat; i++) {
            make_pattern(i);
            if (bench(i, iters) != 0) return 1;
        }
    }
    report();
    ceilings();
    return 0;
}
