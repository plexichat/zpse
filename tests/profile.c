/* Profiler: attribute encode/decode cost to components. */
#include <stdio.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif
#include "zpse/zpse.h"
#include "zpse/arena.h"
#include "zpse/predictor.h"
#include "zpse/tile.h"
#include "zpse/segment.h"
#include "zpse/combinadic.h"
#include "zpse/mask.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t o1[8*256], o2v[8*4096], o2c[8*4096];
static uint8_t scratch[8192];
static uint8_t big[66*1024];

static double now_ns(void) {
#if defined(_WIN32)
    static LARGE_INTEGER f = {{0}};
    LARGE_INTEGER c;
    if (f.QuadPart == 0) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}
#define TIMEIT(name, n, stmt) do { \
    double t0 = now_ns(); \
    for (int _i = 0; _i < (n); _i++) { stmt; } \
    double t1 = now_ns(); \
    printf("%-28s %9.3f ms total %9.1f ns/op\n", name, (t1-t0)/1e6, (t1-t0)/(n)); \
    fflush(stdout); \
} while (0)

int main(void) {
    ZpseEncoder e; ZpseDecoder d;
    ZpseLaneState lanes = {{0},{0},o1,o2v,o2c};
    static uint8_t tile[256], rec[1024], dec[256], plane[1024], pay[4096];
    uint16_t c128[128];
    int i;
    for (i = 0; i < 256; i++) tile[i] = (uint8_t)(i & 0xFF);
    for (i = 0; i < 128; i++) c128[i] = (uint16_t)(i * 2);
    for (i = 0; i < 1024; i++) plane[i] = (uint8_t)(i & 0xFF);
    zpse_encoder_init(&e, E, sizeof E);
    zpse_decoder_init(&d, D, sizeof D);

    TIMEIT("memset 66KB x200", 200, memset(big, 0x5A, sizeof big));
    TIMEIT("lane_reset x500", 500, zpse_lane_reset(&lanes, 0));
    TIMEIT("tile_encode 1combo x200", 200,
        (zpse_lane_reset(&lanes, tile[0]),
         zpse_tile_encode(&lanes, tile, 256, 1, 1, 0, tile[0], 1, tile, 0, rec, sizeof rec, scratch)));
    int wlen;
    zpse_lane_reset(&lanes, tile[0]);
    wlen = zpse_tile_encode(&lanes, tile, 256, 1, 1, 0, tile[0], 1, tile, 0, rec, sizeof rec, scratch);
    printf("tile record len=%d\n", wlen); fflush(stdout);
    TIMEIT("tile_decode 1tile x200", 200,
        (zpse_lane_reset(&lanes, tile[0]),
         zpse_tile_decode(&lanes, rec, (uint32_t)wlen, tile[0], 1, tile, 0, dec, sizeof dec, scratch)));
    TIMEIT("segment_encode 1KB x20", 20,
        (zpse_lane_reset(&lanes, plane[0]),
         zpse_segment_encode(&e, &lanes, plane, 1024, plane[0], pay, sizeof pay, 0u)));
    TIMEIT("comb rank K=128 x20", 20,
        zpse_comb_rank_to_bytes(c128, 128, 256, rec, sizeof rec));
    TIMEIT("comb rank K=8 x200", 200,
        zpse_comb_rank_to_bytes(c128, 8, 256, rec, sizeof rec));
    { /* mask mode-5 encode cost */
        uint32_t w[8] = {0}; uint16_t cc[8] = {10,20,30,40,50,60,70,80};
        uint16_t k2; zpse_mask_derive_k(5, 8, 256, &k2);
        for (i = 0; i < 8; i++) w[cc[i] >> 5] |= 1u << (cc[i] & 31);
        TIMEIT("mask_encode mode5 K=8 x200", 200,
            zpse_mask_encode(5, w, 256, 8, rec, sizeof rec, scratch));
    }
    return 0;
}
