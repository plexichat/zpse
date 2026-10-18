/* Block-decision guard: the compressed size of the shapes the encoder's
 * candidate pricing exists to get right, pinned exactly.
 *
 * Every number here is the size the encoder writes today, and each is small
 * because the block decision saw the form that would be used rather than a
 * per-tile estimate of it:
 *
 *   alternating 0x00/0xFF  the identity costs 2618 bytes; transposed, all
 *                          sixteen planes are one byte value and the dense
 *                          record form writes twelve bytes per 4 KiB, so
 *                          the block is 608.
 *   two-byte spikes        the identity costs 3126; the transposed phases
 *                          are constant, so 1255.
 *   4-bit noise            one palette stream is cheaper than any predictor
 *                          search per tile, so the tile loop stops as soon
 *                          as the palette is priced and 33288 is written.
 *   zeros                  nothing may ever cost less than the floor, 248.
 *
 * A change that regresses one of these still round-trips, so nothing else in
 * the suite would notice; that is why the sizes are pinned. Links
 * src/zpse_unity.c; exits non-zero on any size or round-trip mismatch. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "zpse/zpse.h"

static uint8_t enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t out[65536];

static int fails = 0;

static void check(const char *name, uint32_t n, uint32_t want)
{
    uint32_t clen = 0;
    uint32_t olen = 0;
    int rc = zpse_compress_block(in, n, comp, sizeof comp, &clen, 0, 1,
                                 enc_arena);

    if (rc != 0) {
        printf("FAIL %-22s compress rc=%d(%s)\n", name, rc,
               zpse_status_name(rc));
        fails++;
        return;
    }
    if (clen != want) {
        printf("FAIL %-22s n=%u clen=%u, expected %u\n", name, n, clen, want);
        fails++;
        return;
    }
    memset(out, 0xA5, n);
    rc = zpse_decompress_block(comp, clen, out, n, &olen, dec_arena);
    if (rc < 0 || olen != n || memcmp(out, in, n) != 0) {
        printf("FAIL %-22s decode rc=%d olen=%u\n", name, rc, olen);
        fails++;
        return;
    }
    printf("ok   %-22s n=%5u clen=%5u (%u.%03u bits/byte)\n", name, n, clen,
           (unsigned)((uint64_t)clen * 8u / n),
           (unsigned)(((uint64_t)clen * 8000u / n) % 1000u));
}

int main(void)
{
    uint32_t i;

    for (i = 0; i < 65536u; i++) {
        in[i] = (uint8_t)((i & 1u) ? 0xFF : 0x00);
    }
    check("alt00FF", 65536u, 608u);

    memset(in, 0, sizeof in);
    for (i = 0; i < 65536u; i += 256u) {
        in[i] = (uint8_t)i;
        in[i + 1u] = (uint8_t)(i >> 8);
    }
    check("spike every 256", 65536u, 1255u);

    {
        uint32_t r = 0xBEEFu;

        for (i = 0; i < 65536u; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)((r >> 24) & 0x0Fu);
        }
    }
    check("4-bit noise", 65536u, 33288u);

    memset(in, 0, sizeof in);
    check("zeros", 65536u, 248u);

    if (fails != 0) {
        printf("DECIDE: %d FAILED\n", fails);
        return 1;
    }
    printf("DECIDE: ALL OK\n");
    return 0;
}
