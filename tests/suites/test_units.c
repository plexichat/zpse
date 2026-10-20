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
#include "zpse/dense.h"

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

    /* Repeated-byte CRC: the arithmetic form the block decoder uses for a
     * one-value plane. Must agree with the buffer CRC at every length,
     * including the ones whose binary expansion is not a single power of
     * two (and the 64 KiB block size). */
    {
        static uint8_t rep[65536];
        static const size_t lens[] = { 0u, 1u, 2u, 3u, 7u, 8u, 255u, 256u,
                                       257u, 4095u, 4096u, 4097u, 65535u,
                                       65536u };
        int b, bad = 0;
        size_t li;

        for (b = 0; b < 256; b++) {
            memset(rep, b, sizeof rep);
            for (li = 0; li < sizeof lens / sizeof lens[0]; li++) {
                if (zpse_crc32c_repeat((uint8_t)b, lens[li]) !=
                    zpse_crc32c(rep, lens[li])) {
                    bad++;
                }
            }
        }
        CHECK(bad == 0, "crc repeat matches buffer crc (%d bad)", bad);
    }

    /* A one-value block takes the arithmetic payload CRC on decode, and
     * that path must still reject a corrupt one -- and must not claim the
     * shortcut when the dense records disagree about their value. */
    {
        static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
        static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
        static uint8_t in[65536], comp[512], out[65536];
        ZpseBlockHeader h;
        uint32_t cl = 0, ol = 0;
        int rc;

        memset(in, 0x5A, sizeof in);
        /* The payload CRC is off by default (LZ4-style), so the arithmetic
         * CRC path -- and its rejection -- is exercised through the opt-in.
         * The default stream must not carry the flag at all. */
        rc = zpse_compress_block(in, sizeof in, comp, sizeof comp, &cl, 0, 1,
                                 ea);
        CHECK(rc == PSE_OK, "const block encodes (rc=%d)", rc);
        CHECK(zpse_block_header_parse(&h, comp, cl) == 0 &&
              (h.block_flags & ZPSE_B_CRC) == 0,
              "default block carries no payload CRC");
        rc = zpse_compress_block_ex(in, sizeof in, comp, sizeof comp, &cl, 0,
                                    1, ZPSE_ENC_CHECKSUM, ea);
        CHECK(rc == PSE_OK, "const block encodes with a CRC (rc=%d)", rc);
        CHECK(cl < sizeof comp && zpse_block_header_parse(&h, comp, cl) == 0,
              "const block header parses");
        CHECK((h.block_flags & ZPSE_B_CRC) != 0 &&
              (h.block_flags & ZPSE_B_RAW) == 0,
              "const block is coded with a payload CRC");
        rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, da);
        CHECK(rc == PSE_OK && ol == sizeof in &&
              memcmp(out, in, sizeof in) == 0, "const block roundtrip");

        /* Corrupt the stored payload CRC (the serializer rewrites the
         * header CRC, so the header still parses). */
        h.decoded_crc32c ^= 1u;
        zpse_block_header_serialize(comp, &h);
        CHECK(zpse_decompress_block(comp, cl, out, sizeof out, &ol, da) ==
              PSE_BAD_PAYLOAD_CRC, "const block corrupt CRC rejected");

        /* Restore, then make one dense record reconstruct a different
         * value: the plane is no longer one value, so the arithmetic form
         * must not be used and the old CRC must now mismatch. */
        h.decoded_crc32c ^= 1u;
        zpse_block_header_serialize(comp, &h);
        {
            uint32_t rec0 = (uint32_t)ZPSE_BLOCK_HEADER_SIZE +
                            (uint32_t)h.transform_meta_size +
                            (uint32_t)h.segment_count * 8u + 16u;
            CHECK((comp[rec0 + 2] & 0x40) != 0,
                  "const block uses dense records");
            comp[rec0 + 3] = 9u;    /* TILE_CONSTANT */
            comp[rec0 + 4] = 0x11u; /* its constant byte */
        }
        CHECK(zpse_decompress_block(comp, cl, out, sizeof out, &ol, da) ==
              PSE_BAD_PAYLOAD_CRC,
              "disagreeing dense records fall back to the byte CRC");
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

    /* Mode-3 base-M packed residuals: every legal M, every K in 1..256,
     * encoded and decoded. This is the guard for the AVX2 nibble unpack
     * (M >= 11), whose magic-multiply division is only exact for words
     * below M^G -- the sweep covers both group widths it vectorises (7 and
     * 8), whole and partial final groups, and K on both sides of the
     * eight-group batch boundary, so a wrong constant or a mis-shaped
     * batch fails here rather than in the fuzz corpus. */
    {
        static const uint8_t ms[] = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
        uint8_t digits[256];
        uint8_t payload[1024];
        uint8_t got[256];
        unsigned mi;
        unsigned k;
        uint32_t rng = UINT32_C(0x5EED1234);
        int bad = 0;

        for (mi = 0; mi < (unsigned)(sizeof ms / sizeof ms[0]); mi++) {
            for (k = 1; k <= 256; k++) {
                unsigned i;
                int n;

                for (i = 0; i < k; i++) {
                    rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
                    digits[i] = (uint8_t)((rng >> 24) % ms[mi]);
                }
                n = zpse_residual_encode(ZPSE_R_PACKED_BASE_M, digits, 0, 0,
                                         (uint16_t)k, payload,
                                         (uint16_t)sizeof payload);
                if (n < 0) {
                    bad++;
                    continue;
                }
                if (zpse_residual_decode(ZPSE_R_PACKED_BASE_M, payload,
                                         (uint16_t)n, (uint16_t)k, 0, got,
                                         0) != 0 ||
                    memcmp(got, digits, k) != 0) {
                    bad++;
                }
            }
        }
        CHECK(bad == 0, "mode-3 M/K sweep: %d cases wrong", bad);
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

    /* Palette-bits residual roundtrip at every width. The packer has a
     * vector path per field width (1 bit: two distinct values, 2 bits:
     * three or four, 3 bits: five to eight, 4 bits: nine to sixteen), and
     * each is only taken for whole blocks of 16 or 32 values, so the
     * lengths below are chosen to land on and past every boundary. The
     * decoder cross-checks the pad bits, the payload length and every
     * index against M, so a packer that writes the wrong bit at the wrong
     * place fails here rather than in a roundtrip of some other shape.
     * The 64- and 128-value blocks the vector decoders consume at 2 and 1
     * bits per value make 64/65, 127/128/129 and 255/256/259/511/512/1023/
     * 1024 the lengths that matter most: each of those is a whole number
     * of the larger blocks plus a tail, so a ladder that mixes up the
     * block order or drops the tail fails here. */
    {
        static uint8_t exc[1200];
        static uint8_t back[1200];
        static uint8_t pay[1200];
        static const uint16_t lens[] = { 1u, 2u, 3u, 7u, 8u, 15u, 16u, 17u,
                                         31u, 32u, 33u, 63u, 64u, 65u,
                                         127u, 128u, 129u, 255u, 256u, 259u,
                                         300u, 511u, 512u, 1023u, 1024u };
        int m;
        int lt;
        int bad = 0;

        for (m = 2; m <= 16; m++) {
            for (lt = 0; lt < (int)(sizeof lens / sizeof lens[0]); lt++) {
                uint16_t k = lens[lt];
                uint16_t nu;
                unsigned b;
                uint32_t want;
                int n;
                uint16_t i;

                /* The first positions hold one of each value so the
                 * palette is the whole alphabet, then a scattered mix. */
                for (i = 0; i < k; i++) {
                    exc[i] = (i < (uint16_t)m)
                                 ? (uint8_t)i
                                 : (uint8_t)((i * 7u + (i >> 3)) %
                                            (unsigned)m);
                }
                /* The payload's M is the distinct count the packer finds
                 * (padded to two), not the loop's m: a short k cannot hold
                 * every value. */
                {
                    unsigned present[16];
                    unsigned nv = 0u;
                    uint16_t j;

                    for (j = 0; j < k; j++) {
                        unsigned v = exc[j];
                        unsigned s;

                        for (s = 0; s < nv; s++) {
                            if (present[s] == v) {
                                break;
                            }
                        }
                        if (s == nv) {
                            present[nv++] = v;
                        }
                    }
                    nu = (uint16_t)((nv < 2u) ? 2u : nv);
                }
                b = zpse_ceil_log2_u32((uint32_t)nu);
                want = 1u + (uint32_t)nu +
                       (((uint32_t)k * b + 7u) / 8u);
                n = zpse_residual_encode_hint((uint8_t)ZPSE_R_PALETTE_BITS,
                                              exc, 0, 0, k, pay,
                                              (uint16_t)sizeof pay, 0u, 0);
                if (n != (int)want) {
                    bad++;
                    continue;
                }
                memset(back, 0xEE, sizeof back);
                if (zpse_residual_decode((uint8_t)ZPSE_R_PALETTE_BITS, pay,
                                         (uint16_t)n, k, 0, back,
                                         back) != PSE_OK ||
                    memcmp(exc, back, (size_t)k) != 0) {
                    bad++;
                }
            }
        }
        CHECK(bad == 0, "palette-bits roundtrip m=2..16 (%d bad)", bad);
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
