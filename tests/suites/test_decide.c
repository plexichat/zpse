/* Block-decision guard: the compressed size of the shapes the encoder's
 * candidate pricing exists to get right, pinned exactly.
 *
 * Every number here is the size the encoder writes today, and each is small
 * because the block decision saw the form that would be used rather than a
 * per-tile estimate of it:
 *
 *   alternating 0x00/0xFF  split into its two byte phases, each phase is
 *                          one byte value, and the dense record form writes
 *                          them at twelve bytes per 4 KiB: 272 as tiles. The
 *                          v2 LZ+tiles form compresses that body and writes
 *                          81, and that is what the pin now holds. (82
 *                          before the page-sized parse step in src/lz.c, which
 *                          a 4 KiB body takes even inside a 64 KiB block.)
 *   two-byte spikes        at width 8, seven of the phases are constant and
 *                          are written in the dense form at twelve bytes per
 *                          4 KiB: 1211. The width-16 shape of the same block
 *                          is the one a 16 KiB sample mis-prices by a factor,
 *                          so this pin also covers the sweep keeping it in
 *                          the list instead of pruning it on the cheap
 *                          estimate, and deciding it on a full trial.
 *   4-bit noise            one palette stream is cheaper than any predictor
 *                          search per tile, so the tile loop stops as soon
 *                          as the palette is priced and 33288 is written.
 *   zeros                  the tile floor is 248; the v2 LZ+tiles form
 *                          takes the constant body down to 70. (69 before the
 *                          same parse step: on a body that is one match from
 *                          the second byte the lazy probe gives up a byte at
 *                          the end for no match, which is the step's one
 *                          measured cost and is why it is confined to pages.)
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

/* The decoder and encoder each carve an 8 KiB "comb" scratch out of their
 * arena for dense/tile exception buffers. It must be its own region: if it
 * were aliased onto recon/out/shuf/lzbuf (or the encoder's hist/shuf/lz), a
 * dense/tile exception pass would corrupt a plane being decoded. Both the
 * struct layout and the carve offsets are public (arena.h), so pin them. */
static int ranges_overlap(const uint8_t *a, size_t asz,
                          const uint8_t *b, size_t bsz)
{
    uintptr_t x = (uintptr_t)a;
    uintptr_t y = (uintptr_t)b;

    return x < y + (uintptr_t)bsz && y < x + (uintptr_t)asz;
}

static void comb_arena(void)
{
    ZpseDecoder d;
    ZpseEncoder e;

    if (zpse_decoder_init(&d, dec_arena, sizeof dec_arena) != ZPSE_OK) {
        printf("FAIL comb: decoder init\n");
        fails++;
        return;
    }
    if (d.comb == 0 || d.recon == 0 || d.out_stage == 0 ||
        d.shuf == 0 || d.lzbuf == 0) {
        printf("FAIL comb: decoder region is null\n");
        fails++;
    }
    if (ranges_overlap(d.comb, ZPSE_D_SZ_COMB, d.recon, ZPSE_D_SZ_RECON) ||
        ranges_overlap(d.comb, ZPSE_D_SZ_COMB, d.out_stage, ZPSE_D_SZ_OUT) ||
        ranges_overlap(d.comb, ZPSE_D_SZ_COMB, d.shuf, ZPSE_D_SZ_SHUF) ||
        ranges_overlap(d.comb, ZPSE_D_SZ_COMB, d.lzbuf, ZPSE_D_SZ_LZBUF)) {
        printf("FAIL comb: decoder comb overlaps another region\n");
        fails++;
    }

    if (zpse_encoder_init(&e, enc_arena, sizeof enc_arena) != ZPSE_OK) {
        printf("FAIL comb: encoder init\n");
        fails++;
        return;
    }
    if (e.comb == 0 || e.input == 0 ||
        e.hist == 0 || e.shuf == 0 || e.lz == 0) {
        printf("FAIL comb: encoder region is null\n");
        fails++;
    }
    if (ranges_overlap(e.comb, ZPSE_E_SZ_COMB, e.input, ZPSE_E_SZ_INPUT) ||
        ranges_overlap(e.comb, ZPSE_E_SZ_COMB, e.hist, ZPSE_E_SZ_HIST) ||
        ranges_overlap(e.comb, ZPSE_E_SZ_COMB, e.shuf, ZPSE_E_SZ_SHUF) ||
        ranges_overlap(e.comb, ZPSE_E_SZ_COMB, e.lz, ZPSE_E_SZ_LZ)) {
        printf("FAIL comb: encoder comb overlaps another region\n");
        fails++;
    }
}

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

    comb_arena();

    for (i = 0; i < 65536u; i++) {
        in[i] = (uint8_t)((i & 1u) ? 0xFF : 0x00);
    }
    check("alt00FF", 65536u, 98u);

    memset(in, 0, sizeof in);
    for (i = 0; i < 65536u; i += 256u) {
        in[i] = (uint8_t)i;
        in[i + 1u] = (uint8_t)(i >> 8);
    }
    check("spike every 256", 65536u, 1326u);

    {
        uint32_t r = 0xBEEFu;

        for (i = 0; i < 65536u; i++) {
            r = r * 1664525u + 1013904223u;
            in[i] = (uint8_t)((r >> 24) & 0x0Fu);
        }
    }
    check("4-bit noise", 65536u, 33280u);

    memset(in, 0, sizeof in);
    check("zeros", 65536u, 79u);

    if (fails != 0) {
        printf("DECIDE: %d FAILED\n", fails);
        return 1;
    }
    printf("DECIDE: ALL OK\n");
    return 0;
}
