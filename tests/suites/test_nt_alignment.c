/* Opt-in NT decode alignment harness.
 *
 * Every decode path that writes its output once, contiguously and without
 * reading it back is exercised at all 32 output alignments and 16 sizes
 * (512 cases per path), with the streaming hint off and on, and a 0xCC canary
 * on both sides of the destination. The RAW/no-CRC case is the shipped
 * f8447d1 path; the others are the paths that reuse its exact flag, cutoff
 * and per-thread hot-destination guard.
 *
 * usage: test_nt_alignment [raw|raw-crc|token|flat|flat7|all]
 * Exit 0 only when every case decodes byte-exactly and leaves both canaries
 * intact in both flag states. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zpse/zpse.h"
#include "zpse/api.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/flat7.h"
#include "zpse/types.h"

#define MAXN 65536u
#define CANARY 0xCCu

_Alignas(ZPSE_ARENA_ALIGN) static unsigned char da[ZPSE_DEC_ARENA_SIZE];
_Alignas(64) static uint8_t payload[MAXN];
_Alignas(64) static uint8_t blk[MAXN + 128u];
_Alignas(64) static uint8_t outbuf[MAXN + 128u];
_Alignas(64) static uint8_t ref[MAXN];

static const uint32_t sizes[] = {
    1u, 63u, 64u, 65u, 127u, 128u, 129u, 255u,
    4095u, 4096u, 4097u, 4160u, 8191u, 8192u, 12345u, 65535u
};
static const uint32_t page_sizes[] = {
    1u, 2u, 3u, 7u, 8u, 9u, 15u, 16u,
    63u, 64u, 65u, 127u, 128u, 255u, 1024u, 4096u
};

static unsigned long total_checks = 0;
static unsigned long total_fails = 0;

static void fill_pattern(uint32_t n)
{
    uint32_t x = 0x13579BDFu;
    uint32_t i;

    for (i = 0u; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        payload[i] = (uint8_t)(x >> 24);
    }
}

static ZpseBlockHeader base_header(uint32_t n, uint32_t c)
{
    ZpseBlockHeader h;

    memset(&h, 0, sizeof h);
    h.sync = ZPSE_BLOCK_SYNC;
    h.block_version = ZPSE_BLOCK_VERSION;
    h.block_flags = (uint8_t)(ZPSE_B_RESET);
    h.uncompressed_size = n;
    h.compressed_block_size = c;
    h.transform_id = 0u;
    h.segment_count = 1u;
    h.transform_meta_size = 0u;
    h.tile_size = (uint16_t)ZPSE_TILE_SIZE;
    h.header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE;
    return h;
}

/* RAW/no-CRC block, verbatim 32-byte header then n payload bytes. */
static void build_raw(uint32_t n)
{
    ZpseBlockHeader h = base_header(n, 32u + n);

    h.block_flags |= ZPSE_B_RAW;
    zpse_block_header_serialize(blk, &h);
    memcpy(blk + 32u, payload, n);
}

/* RAW with a payload CRC: the shipped path keeps its fused copy+checksum,
 * so this case checks that a streaming caller still gets byte-exact output
 * and a verified CRC when the hint is set. */
static void build_raw_crc(uint32_t n)
{
    ZpseBlockHeader h = base_header(n, 32u + n);

    h.block_flags |= (uint8_t)(ZPSE_B_RAW | ZPSE_B_CRC);
    h.decoded_crc32c = zpse_crc32c(payload, n);
    zpse_block_header_serialize(blk, &h);
    memcpy(blk + 32u, payload, n);
}

/* One-value page token: the fill byte travels in transform_id. */
static void build_token(uint32_t n)
{
    ZpseBlockHeader h = base_header(n, 16u);

    h.block_flags |= (uint8_t)(ZPSE_B_RAW);
    h.transform_id = payload[0];
    zpse_block_header_serialize_token(blk, &h);
}

/* TEXT page: a fixed-rate 7-bit pack, payload length n - (n>>3). */
static void build_flat(uint32_t n)
{
    ZpseBlockHeader h = base_header(n, 16u + n);

    h.block_flags = (uint8_t)ZPSE_B_RESET;
    h.transform_id = (uint8_t)ZPSE_FLAT_RAW;
    zpse_block_header_serialize_flat(blk, &h);
    memcpy(blk + 16u, payload, n);
}

/* FLAT7 whole-frame, RAW region: 40-byte header, one verbatim copy. */
static uint32_t build_flat7(uint32_t n)
{
    uint32_t crc;
    uint32_t i;

    memset(blk, 0, 40u);
    blk[0] = 'F'; blk[1] = '7'; blk[2] = 'P'; blk[3] = '1';
    blk[4] = 1u; blk[5] = 0u;
    blk[6] = 0u; blk[7] = 0u;
    for (i = 0u; i < 8u; i++) {
        blk[8u + i] = (uint8_t)(((uint64_t)n) >> (8u * i));
    }
    blk[16] = 1u; blk[17] = 0u; blk[18] = 0u; blk[19] = 0u; /* nregions */
    for (i = 0u; i < 4u; i++) { blk[20u + i] = 0u; }          /* offset */
    blk[24] = (uint8_t)n; blk[25] = (uint8_t)(n >> 8);
    blk[26] = (uint8_t)(n >> 16); blk[27] = (uint8_t)(n >> 24); /* packed_len */
    blk[28] = (uint8_t)n; blk[29] = (uint8_t)(n >> 8);
    blk[30] = (uint8_t)(n >> 16); blk[31] = (uint8_t)(n >> 24); /* out_len */
    blk[32] = (uint8_t)ZPSE_FLAT7_REGION_RAW;
    crc = zpse_crc32c(blk, 36u);
    for (i = 0u; i < 4u; i++) {
        blk[36u + i] = (uint8_t)(crc >> (8u * i));
    }
    memcpy(blk + 40u, payload, n);
    return 40u + n;
}

/* Run every alignment for one prepared stream, with both flag states. */
static void run_cases(const char *name, uint32_t n, uint32_t clen,
                      const uint8_t *expect, int whole)
{
    unsigned a;

    memcpy(ref, expect, n);
    for (a = 0u; a < 32u; a++) {
        uint8_t *out = outbuf + 64u + a;
        uint32_t fl;

        /* Hint ON first at each fresh destination, so the hot-destination
         * guard actually admits the NT store; the off pass then reuses the
         * same address and stays on the write-back path. */
        {
        uint32_t fl_order[2] = { 1u, 0u };
        uint32_t fl_i;

        for (fl_i = 0u; fl_i < 2u; fl_i++) {
            fl = fl_order[fl_i];
            size_t ol = 0;
            int rc;
            int ok = 1;
            uint32_t i;

            memset(outbuf, CANARY, sizeof outbuf);
            if (whole) {
                rc = zpse_decompress_frame_ex(blk, clen, out, n, &ol,
                                              fl ? ZPSE_DEC_STREAMING_OUTPUT : 0u,
                                              da);
            } else {
                uint32_t uol = 0;

                rc = zpse_decompress_block_ex(blk, clen, out, n, &uol,
                                              fl ? ZPSE_DEC_STREAMING_OUTPUT : 0u,
                                              da);
                ol = uol;
            }
            if (rc != 0 || ol != n || memcmp(out, ref, n) != 0) {
                ok = 0;
            }
            for (i = 0u; i < 64u + a; i++) {
                if (outbuf[i] != CANARY) { ok = 0; }
            }
            for (i = 64u + a + n; i < sizeof outbuf; i++) {
                if (outbuf[i] != CANARY) { ok = 0; }
            }
            total_checks++;
            if (!ok) {
                if (total_fails < 12) {
                    printf("FAIL %s n=%u align=%u flags=%u rc=%d ol=%lu\n",
                           name, n, a, fl, rc, (unsigned long)ol);
                }
                total_fails++;
            }
        }
        }
    }
}

static void path_raw(void)
{
    uint32_t si;
    for (si = 0u; si < sizeof sizes / sizeof sizes[0]; si++) {
        uint32_t n = sizes[si];
        fill_pattern(n);
        build_raw(n);
        run_cases("raw", n, 32u + n, payload, 0);
    }
}

static void path_raw_crc(void)
{
    uint32_t si;
    for (si = 0u; si < sizeof sizes / sizeof sizes[0]; si++) {
        uint32_t n = sizes[si];

        fill_pattern(n);
        build_raw_crc(n);
        run_cases("raw-crc", n, 32u + n, payload, 0);
    }
}

static void path_token(void)
{
    uint32_t si;
    for (si = 0u; si < sizeof sizes / sizeof sizes[0]; si++) {
        uint32_t n = sizes[si];
        uint8_t v = (uint8_t)(0x40u + (n & 0x3Fu));

        memset(payload, v, n);
        build_token(n);
        run_cases("token", n, 16u, payload, 0);
    }
}

static void path_flat(void)
{
    uint32_t si;
    for (si = 0u; si < sizeof page_sizes / sizeof page_sizes[0]; si++) {
        uint32_t n = page_sizes[si];

        fill_pattern(n);
        build_flat(n);
        run_cases("flat", n, 16u + n, payload, 0);
    }
}

static void path_flat7(void)
{
    uint32_t si;
    for (si = 0u; si < sizeof page_sizes / sizeof page_sizes[0]; si++) {
        uint32_t n = page_sizes[si];

        fill_pattern(n);
        run_cases("flat7", n, build_flat7(n), payload, 1);
    }
}

int main(int argc, char **argv)
{
    const char *which = (argc > 1) ? argv[1] : "all";
    int all = strcmp(which, "all") == 0;

    if (all || strcmp(which, "raw") == 0) path_raw();
    if (all || strcmp(which, "raw-crc") == 0) path_raw_crc();
    if (all || strcmp(which, "token") == 0) path_token();
    if (all || strcmp(which, "flat") == 0) path_flat();
    if (all || strcmp(which, "flat7") == 0) path_flat7();
    printf("nt-alignment(%s): %lu checks, %lu failures\n",
           which, total_checks, total_fails);
    return total_fails != 0;
}
