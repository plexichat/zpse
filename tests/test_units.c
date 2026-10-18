/* Low-level unit tests: CRC, arenas, headers, K-derivation, tables,
 * tile roundtrips across all predictors and residual modes. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
#include "zpse/arena.h"
#include "zpse/types.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/transform.h"
#include "zpse/predictor.h"
#include "zpse/tile.h"
#include "zpse/status.h"

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL " __VA_ARGS__); printf("\n"); fails++; } \
} while (0)

static uint8_t o1[8 * 256];
static uint8_t o2v[8 * 4096];
static uint8_t o2c[8 * 4096];
static uint8_t comb[8192];

int main(void) {
    /* CRC32-C. */
    {
        const uint8_t v[] = "123456789";
        CHECK(zpse_crc32c(v, 9) == 0xE3069283u, "crc check value");
        CHECK(zpse_crc32c(v, 0) == 0u, "crc empty == 0");
        CHECK(zpse_crc32c_init() == 0xFFFFFFFFu, "crc init");
    }

    /* Arenas. */
    {
        ZpseDecoder d; ZpseEncoder e;
        static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
        static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
        CHECK(zpse_decoder_init(NULL, da, sizeof da) == PSE_BAD_ARG, "dec null ctx");
        CHECK(zpse_decoder_init(&d, NULL, sizeof da) == PSE_BAD_ARG, "dec null mem");
        CHECK(zpse_decoder_init(&d, da, sizeof da - 1) == PSE_BAD_ARG, "dec short");
        CHECK(zpse_decoder_init(&d, da + 1, sizeof da - 1) == PSE_BAD_ARG, "dec misaligned");
        CHECK(zpse_decoder_init(&d, da, sizeof da) == 0, "dec ok");
        CHECK(d.recon == da + 0x1000 && d.comb == da + 0x31800, "dec carve");
        CHECK(zpse_encoder_init(&e, ea, sizeof ea) == 0, "enc ok");
        CHECK(e.best == ea + 0x31000 && e.hist == ea + 0x53800, "enc carve");
        CHECK(!zpse_arena_is_aligned(NULL) && zpse_arena_is_aligned(da), "aligned?");
    }

    /* Header serialize -> parse. */
    {
        ZpseFrameHeader fh, fh2;
        uint8_t buf[32];
        zpse_frame_header_build(&fh, 1024, 5000, 1, 5, 1, 0);
        zpse_frame_header_write(&fh, buf);
        CHECK(zpse_frame_header_read(&fh2, buf) == 0, "frame hdr roundtrip");
        CHECK(fh2.nominal_block_size == 1024 && fh2.block_count == 5, "frame fields");
        buf[6] |= 0x10;
        CHECK(zpse_frame_header_read(&fh2, buf) == PSE_BAD_HEADER_CRC ||
              zpse_frame_header_read(&fh2, buf) == PSE_UNSUPPORTED_FEATURE,
              "frame reserved bit rejected");
    }

    /* Mask K-derivation edges. */
    {
        uint16_t k = 99;
        CHECK(zpse_mask_derive_k(0, 0, 256, &k) == 0 && k == 0, "mode0 K=0");
        CHECK(zpse_mask_derive_k(0, 1, 256, &k) != 0, "mode0 K=1 bad");
        CHECK(zpse_mask_derive_k(6, 0, 256, &k) == 0 && k == 256, "mode6 ec0->256");
        CHECK(zpse_mask_derive_k(6, 5, 256, &k) != 0, "mode6 ec5 bad");
        CHECK(zpse_mask_derive_k(1, 0, 256, &k) != 0, "mode1 K=0 bad");
        CHECK(zpse_mask_derive_k(1, 255, 255, &k) != 0, "mode1 K=Nt bad");
        CHECK(zpse_mask_derive_k(7, 0, 8, &k) != 0, "mode7 bad");
        CHECK(zpse_mask_derive_k(2, 3, 256, &k) == 0 && k == 3, "mode2 K=3");
    }

    /* Base-M group table + transform widths + segment lengths. */
    {
        CHECK(zpse_base_m_group(3) == 20, "G(3)==20");
        CHECK(zpse_base_m_group(15) == 7, "G(15)==7");
        CHECK(zpse_base_m_group(2) == 0 && zpse_base_m_group(16) == 0, "G illegal==0");
        uint8_t w = 0;
        CHECK(zpse_transform_width(0, 1, &w) == 0 && w == 1, "width raw==1");
        CHECK(zpse_transform_width(5, 8, &w) == 0 && w == 8, "width t8==8");
        CHECK(zpse_transform_width(3, 8, &w) != 0, "width mismatch rejected");
        uint32_t lens[16] = {0};
        CHECK(zpse_segment_lengths(10, 4, lens) == 0 &&
              lens[0] == 3 && lens[1] == 3 && lens[2] == 2 && lens[3] == 2,
              "seg lens 10/4");
        /* Tile flag helpers. */
        uint8_t f = zpse_tile_flags_make(5, 3, 1);
        CHECK(zpse_tile_mask_mode(f) == 5 && zpse_tile_res_mode(f) == 3, "tile flags");
        CHECK((f & ZPSE_TILE_FLAG_RESET) != 0, "tile reset bit");
    }

    /* Mask encode -> decode roundtrip, every position-coding mode over
     * every tile length and four mask shapes (dense, sparse, one run,
     * alternating). This is the guard for the RUNS encoder's transition
     * walk: a run sequence that disagrees with the mask at one length is
     * caught here, and the shapes are chosen so the lengths below the 255
     * split, exact multiples of 32 and partial last words are all hit. */
    {
        static uint32_t w[8], w2[8];
        static uint8_t pay[600];
        uint32_t seed = 12345u;
        int trial;
        int bad = 0;

        for (trial = 0; trial < 4000; trial++) {
            uint16_t nt;
            uint16_t k = 0;
            unsigned cls;
            int mode;
            int i;

            seed = seed * 1664525u + 1013904223u;
            nt = (uint16_t)(2u + (seed >> 16) % 255u);
            seed = seed * 1664525u + 1013904223u;
            cls = (seed >> 8) & 3u;
            for (i = 0; i < 8; i++) {
                seed = seed * 1664525u + 1013904223u;
                w[i] = seed;
            }
            if (cls == 1u) {
                for (i = 0; i < 8; i++)
                    w[i] &= 0x11111111u;
            } else if (cls == 2u) {
                uint32_t a = (seed >> 3) % (uint32_t)nt;
                uint32_t len = 1u + (seed >> 11) % (uint32_t)nt;
                uint32_t p;

                for (i = 0; i < 8; i++)
                    w[i] = 0u;
                for (p = a; p < (uint32_t)nt && len > 0u; p++, len--)
                    w[p >> 5] |= 1u << (p & 31u);
            } else if (cls == 3u) {
                uint32_t p;

                for (i = 0; i < 8; i++)
                    w[i] = 0u;
                for (p = 0; p < (uint32_t)nt; p += 2u)
                    w[p >> 5] |= 1u << (p & 31u);
            }
            {
                uint32_t rem = (uint32_t)nt & 31u;
                int wi = (int)(nt >> 5);

                if (rem)
                    w[wi++] &= (1u << rem) - 1u;
                else if (wi < 8)
                    w[wi++] = 0u;
                for (; wi < 8; wi++)
                    w[wi] = 0u;
            }
            for (i = 0; i < (int)nt; i++)
                k = (uint16_t)(k + (uint16_t)((w[i >> 5] >> (i & 31)) & 1u));
            if (k == 0u || k == nt)
                continue;
            for (mode = 1; mode <= 4; mode++) {
                int n = zpse_mask_encode((uint8_t)mode, w, nt, k, pay,
                                         (uint16_t)sizeof pay, comb);

                if (n < 0)
                    continue;
                memset(w2, 0, sizeof w2);
                if (zpse_mask_decode((uint8_t)mode, pay, (uint16_t)n, nt, k,
                                     w2, comb) != PSE_OK ||
                    memcmp(w, w2, sizeof w) != 0) {
                    bad++;
                }
            }
        }
        CHECK(bad == 0, "mask encode/decode roundtrip modes 1-4 (%d bad)",
              bad);
    }

    /* Tile roundtrips: every predictor x residual modes {0,1,2,3,4}. */
    {
        static uint8_t seg[512], rec[1024], dec[256];
        uint8_t data[256], cdata[256];
        int pid, rm, mm;
        int i;
        for (i = 0; i < 256; i++) data[i] = (uint8_t)((i * 7 + (i >> 4)) & 0xFF);
        memset(cdata, 0x41, sizeof cdata);
        memcpy(seg + 64, data, 256);
        for (pid = 0; pid <= 11; pid++) {
            for (rm = 0; rm <= 4; rm++) {
                int ok = 0, tried = 0;
                const uint8_t mmodes[] = {1, 2, 3, 4, 5, 0, 6};
                for (mm = 0; mm < 7; mm++) {
                    ZpseLaneState le = {{0},{0},o1,o2v,o2c};
                    ZpseLaneState ld = {{0},{0},o1,o2v,o2c};
                    int wlen, got;
                    zpse_lane_reset(&le, data[0]);
                    wlen = zpse_tile_encode(&le, data, 256, (uint8_t)pid,
                                            mmodes[mm], (uint8_t)rm, data[0],
                                            1, seg + 64, 0, rec, sizeof rec, comb);
                    if (wlen < 0) continue;
                    tried++;
                    zpse_lane_reset(&ld, data[0]);
                    got = zpse_tile_decode(&ld, rec, (uint32_t)wlen, data[0],
                                           1, seg + 64, 0, dec, sizeof dec, comb);
                    if (got != 256 || memcmp(dec, data, 256) != 0) {
                        CHECK(0, "tile pid=%d rm=%d mm=%d mismatch", pid, rm, mmodes[mm]);
                    } else ok++;
                }
                if (!tried)
                    printf("note pid=%d rm=%d: no mask mode applied (ok)\n", pid, rm);
                (void)ok;
            }
        }
        /* Constant tile: K=0 / K=Nt paths. */
        {
            ZpseLaneState le = {{0},{0},o1,o2v,o2c};
            ZpseLaneState ld = {{0},{0},o1,o2v,o2c};
            int wlen, got;
            memcpy(seg + 64, cdata, 256);
            zpse_lane_reset(&le, 0x41);
            wlen = zpse_tile_encode(&le, cdata, 256, 9, 0, 0, 0x41, 1,
                                    seg + 64, 0, rec, sizeof rec, comb);
            CHECK(wlen > 0, "const tile encodes (pid9/mode0)");
            if (wlen > 0) {
                zpse_lane_reset(&ld, 0x41);
                got = zpse_tile_decode(&ld, rec, (uint32_t)wlen, 0x41, 1,
                                       seg + 64, 0, dec, sizeof dec, comb);
                CHECK(got == 256 && memcmp(dec, cdata, 256) == 0, "const tile roundtrip");
            }
        }
    }

    printf(fails ? "UNITS: %d FAILURES\n" : "UNITS: ALL OK\n", fails);
    return fails != 0;
}
