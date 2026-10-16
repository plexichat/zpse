/* STRIDE_3 (predictor id 11): interleaved 3-byte records must round-trip
 * and must actually be recognised -- the compressed size has to collapse
 * on data whose only redundancy is "same as 3 bytes back". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "zpse/zpse.h"

static uint8_t g_enc[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t g_dec[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static int fails = 0;

static void check(const char *name, const uint8_t *in, uint32_t n,
                  double max_ratio)
{
    static uint8_t out[ZPSE_MAX_BLOCK_SIZE + 4096];
    static uint8_t back[ZPSE_MAX_BLOCK_SIZE];
    uint32_t clen = 0, olen = 0;
    int rc = zpse_compress_block(in, n, out, sizeof out, &clen, 0, 1, g_enc);

    if (rc < 0) {
        printf("FAIL %s: encode rc=%d\n", name, rc);
        fails++;
        return;
    }
    rc = zpse_decompress_block(out, clen, back, sizeof back, &olen, g_dec);
    if (rc < 0 || olen != n || memcmp(back, in, n) != 0) {
        printf("FAIL %s: roundtrip (rc=%d olen=%u)\n", name, rc, olen);
        fails++;
        return;
    }
    if ((double)clen > max_ratio * (double)n) {
        printf("FAIL %s: %u -> %u bytes, expected ratio <= %.2f\n", name, n,
               clen, max_ratio);
        fails++;
        return;
    }
    printf("ok   %-34s %6u -> %6u\n", name, n, clen);
}

int main(void)
{
    static uint8_t buf[65536];
    uint32_t i;
    uint32_t n;

    /* Constant pixel runs: byte i equals byte i-3, changing every 8 pixels. */
    for (i = 0; i < 65536u; i++) {
        uint32_t px = i / 3u;
        buf[i] = (uint8_t)(((px >> 3) * (i % 3u + 1u) * 37u) & 0xFFu);
    }
    check("rgb runs of 8 pixels", buf, 65536u, 0.35);
    check("rgb runs, odd length 65533", buf, 65533u, 0.35);
    check("rgb runs, short 300", buf, 300u, 1.20);
    check("rgb runs, tiny 17 (raw envelope)", buf, 17u, 3.00);

    /* Period-3 repeat with a rare change. */
    for (i = 0; i < 65536u; i++)
        buf[i] = (uint8_t)((i % 3u) * 50u + ((i / 1000u) & 3u));
    check("period-3 pattern", buf, 65536u, 0.10);

    /* Period-3 pattern with noise in 1 byte of 40: still roundtrips. */
    srand(7);
    for (i = 0; i < 65536u; i++) {
        buf[i] = (uint8_t)((i % 3u) * 50u);
        if (rand() % 40 == 0)
            buf[i] = (uint8_t)rand();
    }
    check("period-3 + sparse noise", buf, 65536u, 0.40);

    /* Pure noise must not be hurt by the extra candidate. */
    for (i = 0; i < 65536u; i++)
        buf[i] = (uint8_t)rand();
    n = 65536u;
    check("random (must stay raw-sized)", buf, n, 1.001);

    if (fails != 0) {
        printf("STRIDE3: %d FAILED\n", fails);
        return 1;
    }
    printf("STRIDE3: ALL OK\n");
    return 0;
}
