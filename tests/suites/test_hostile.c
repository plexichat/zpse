/* Hostile-bitstream tests: every corruption must yield a clean status
 * code, never a crash or silent acceptance. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t input[2048];
static uint8_t comp[8192];
static uint8_t work[8192];
static uint8_t out[2048];
static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL " __VA_ARGS__); printf("\n"); fails++; } \
    else { printf("ok   " __VA_ARGS__); printf("\n"); } \
} while (0)

static uint32_t clen = 0;

static void refresh_flags(uint32_t flags) {
    uint32_t dummy = 0;
    int rc = zpse_compress_block_ex(input, sizeof input, comp, sizeof comp,
                                    &dummy, 0, 1, flags, enc_arena);
    (void)rc;
    clen = dummy;
}

/* The default stream (no payload checksum, as in LZ4). */
static void refresh(void) {
    refresh_flags(0u);
}

/* The same input with the optional payload CRC tag. */
static void refresh_crc(void) {
    refresh_flags((uint32_t)ZPSE_ENC_CHECKSUM);
}

int main(void) {
    uint32_t olen = 0;
    int rc, i;
    for (i = 0; i < (int)sizeof input; i++) input[i] = (uint8_t)(i & 0xFF);
    refresh();

    /* Truncations. */
    memcpy(work, comp, clen);
    rc = zpse_decompress_block(work, 31, out, sizeof out, &olen, dec_arena);
    CHECK(rc == PSE_TRUNCATED, "truncated header -> TRUNCATED");
    rc = zpse_decompress_block(work, clen - 1, out, sizeof out, &olen, dec_arena);
    CHECK(rc != 0, "truncated tail rejected");

    /* Header CRC corruption. */
    memcpy(work, comp, clen); work[10] ^= 0xFF;
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc == PSE_BAD_HEADER_CRC, "bad header CRC rejected");

    /* Payload corruption with intact header -> BAD_PAYLOAD_CRC, when the
     * stream asks for the checksum. Use incompressible input so the block
     * takes the RAW path, where any payload byte flip provably changes
     * only values (structure of a coded block could break earlier with a
     * different code). */
    {
        uint32_t r = 0x12345u;
        int i;
        for (i = 0; i < (int)sizeof input; i++) {
            r = r * 1664525u + 1013904223u;
            input[i] = (uint8_t)(r >> 24);
        }
        refresh_crc();
        memcpy(work, comp, clen);
        work[100] ^= 0xFF;
        rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
        CHECK(rc == PSE_BAD_PAYLOAD_CRC, "bad payload CRC rejected");
        /* The default stream carries no checksum, so the same flip is a
         * value change the format never promised to detect -- that is the
         * trade the default makes, and it is asserted here so a future
         * "always emit the CRC" change cannot pass unnoticed. */
        refresh();
        memcpy(work, comp, clen);
        work[100] ^= 0xFF;
        rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
        CHECK(rc == PSE_OK && memcmp(out, input, sizeof input) != 0,
              "default stream has no payload CRC (flip undetected)");
        for (i = 0; i < (int)sizeof input; i++) input[i] = (uint8_t)(i & 0xFF);
        refresh();
    }

    /* Reserved block flag bit with repaired CRC -> UNSUPPORTED_FEATURE. */
    memcpy(work, comp, clen); work[3] |= 0x20;
    zpse_store_u32le(work + 28, zpse_crc32c(work, 28));
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc == PSE_UNSUPPORTED_FEATURE, "reserved block bit rejected");

    /* Bad transform_id with repaired CRC -> BAD_TRANSFORM. */
    memcpy(work, comp, clen); work[12] = 12;
    zpse_store_u32le(work + 28, zpse_crc32c(work, 28));
    rc = zpse_decompress_block(work, clen, out, sizeof out, &olen, dec_arena);
    CHECK(rc == PSE_BAD_TRANSFORM, "transform 12 rejected");

    /* Output too small -> OUTPUT_OVERFLOW. */
    rc = zpse_decompress_block(comp, clen, out, sizeof input - 1, &olen, dec_arena);
    CHECK(rc == PSE_OUTPUT_OVERFLOW, "small output -> OVERFLOW");

    /* Frame-level corruptions. */
    {
        static uint8_t fcomp[8192], fout[2048];
        size_t flen = 0, folen = 0;
        rc = zpse_compress_frame(input, sizeof input, fcomp, sizeof fcomp,
                                 &flen, 1024, enc_arena);
        CHECK(rc == 0, "frame compress baseline");
        memcpy(work, fcomp, flen); work[0] ^= 0xFF;
        rc = zpse_decompress_frame(work, flen, fout, sizeof fout, &folen, dec_arena);
        CHECK(rc == PSE_BAD_MAGIC, "frame bad magic");
        memcpy(work, fcomp, flen); work[4] ^= 0xFF;
        rc = zpse_decompress_frame(work, flen, fout, sizeof fout, &folen, dec_arena);
        CHECK(rc == PSE_BAD_VERSION, "frame bad version");
        memcpy(work, fcomp, flen); work[30] ^= 0xFF;
        rc = zpse_decompress_frame(work, flen, fout, sizeof fout, &folen, dec_arena);
        CHECK(rc == PSE_BAD_HEADER_CRC, "frame bad CRC");
    }

    /* Unit-level residual hostility. */
    {
        uint8_t exc[4]; uint8_t bad1[1] = {255};
        uint8_t dummy_pred[1] = {0};
        rc = zpse_residual_decode(0, bad1, 1, 1, dummy_pred, exc, exc);
        CHECK(rc == PSE_BAD_RESIDUAL, "residual 255 rejected (mode 0)");
        /* Base-M tail: M=3 -> G=20; K=1 valid word=1, invalid word=4. */
        uint8_t good[5] = {3, 1, 0, 0, 0};
        uint8_t badw[5] = {3, 4, 0, 0, 0};
        rc = zpse_residual_decode(3, good, 5, 1, dummy_pred, exc, exc);
        CHECK(rc == 0 && exc[0] == 1, "base-M valid tail accepted");
        rc = zpse_residual_decode(3, badw, 5, 1, dummy_pred, exc, exc);
        CHECK(rc == PSE_BAD_RESIDUAL, "base-M nonzero tail rejected");
        /* Signed delta zero unit. */
        uint8_t z4[2] = {0, 0x00};
        rc = zpse_residual_decode(4, z4, 2, 1, dummy_pred, exc, exc);
        CHECK(rc == PSE_BAD_RESIDUAL, "delta d=0 rejected");
        /* Palette index out of range: M=3 needs 2-bit indices, so the
         * value 3 is illegal. K=4 packs 4 indices into 1 byte; 0xFF makes
         * every index 3 -> rejected. (Power-of-two M cannot trigger OOR
         * since every representable value is < M.) */
        uint8_t pal[5] = {3, 10, 20, 30, 0xFF};
        uint8_t pexc[8];
        rc = zpse_residual_decode(2, pal, 5, 4, dummy_pred, pexc, pexc);
        CHECK(rc == PSE_BAD_RESIDUAL, "palette OOR index rejected");
    }

    /* Unit-level mask hostility: nonzero pad bits (Nt=9 -> 2 bytes). */
    {
        uint32_t w[8] = {0,0,0,0,0,0,0,0};
        uint8_t badmap[2] = {0x01, 0xFE}; /* bit0 set (K=1) + pad garbage */
        rc = zpse_mask_decode(1, badmap, 2, 9, 1, w, NULL);
        CHECK(rc == PSE_BAD_MASK, "bitmap pad bits rejected");
    }

    /* Random garbage must error, never crash. */
    {
        uint32_t r = 0xDEADBEEFu;
        int bad = 0;
        for (i = 0; i < 50; i++) {
            int k;
            for (k = 0; k < 256; k++) {
                r = r * 1664525u + 1013904223u;
                work[k] = (uint8_t)(r >> 24);
            }
            rc = zpse_decompress_block(work, 256, out, sizeof out, &olen, dec_arena);
            if (rc >= 0) { bad = 1; break; }
            rc = zpse_decompress_frame(work, 256, out, sizeof out,
                                       &(size_t){0}, dec_arena);
            if (rc >= 0) { bad = 1; break; }
        }
        CHECK(!bad, "50x random garbage rejected");
    }

    printf(fails ? "HOSTILE: %d FAILURES\n" : "HOSTILE: ALL OK\n", fails);
    return fails != 0;
}
