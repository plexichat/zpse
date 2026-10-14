#include <stdio.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "zpse/zpse.h"
#include "zpse/predictor.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/combinadic.h"

static uint8_t o1[8*256], o2v[8*4096], o2c[8*4096];
static uint8_t scratch[8192];
static double now_ns(void) {
    static LARGE_INTEGER f = {{0}};
    LARGE_INTEGER c;
    if (f.QuadPart == 0) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
#define TIMEIT(name, n, stmt) do { \
    double t0 = now_ns(); \
    for (int _i = 0; _i < (n); _i++) { stmt; } \
    double t1 = now_ns(); \
    printf("%-30s %9.1f ns/op\n", name, (t1-t0)/(n)); \
    fflush(stdout); \
} while (0)

int main(void) {
    ZpseLaneState lanes = {{0},{0},o1,o2v,o2c};
    static uint8_t tile[256], pred[256], map[32], res[256], exc[256];
    static uint32_t words[8];
    int i;
    for (i = 0; i < 256; i++) tile[i] = (uint8_t)(i & 0xFF);
    memset(map, 0x01, sizeof map);
    for (i = 0; i < 256; i++) res[i] = (uint8_t)(i % 255);
    zpse_lane_reset(&lanes, 0);

    TIMEIT("predict_block stride", 2000,
        zpse_predict_block(&lanes, 5, 0, tile, 0, 0, pred, 256));
    TIMEIT("predict_update stride", 2000,
        zpse_predict_update(&lanes, 5, tile, pred, 256));
    TIMEIT("mask_decode bitmap", 2000,
        (words[0]=words[1]=words[2]=words[3]=words[4]=words[5]=words[6]=words[7]=0,
         zpse_mask_decode(1, map, 32, 256, 8, words, scratch)));
    TIMEIT("residual_decode excl", 2000,
        zpse_residual_decode(0, res, 8, 8, pred, exc, exc));
    TIMEIT("comb_payload_len(256,128)", 200,
        zpse_comb_payload_len(256, 128, NULL));
    TIMEIT("comb_payload_len(256,8)", 2000,
        zpse_comb_payload_len(256, 8, NULL));
    TIMEIT("mask_derive_k", 20000,
        { uint16_t k; zpse_mask_derive_k(1, 8, 256, &k); });
    return 0;
}
