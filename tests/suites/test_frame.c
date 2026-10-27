/* FrameIndex and random access.
 *
 * A frame's blocks are independent, so a reader that has a block's offset
 * can decode that block and nothing else. What it did not have was a way
 * to find the offset without decoding (and therefore walking) every block
 * before it. The index is that table, and this suite holds down the four
 * things it has to get right:
 *
 *   - every entry points at the block it names, and seeking to it returns
 *     exactly the bytes a sequential decode returns for that block;
 *   - the table is not trusted: a corrupted entry, a corrupted checksum,
 *     an index offset past the stream and a block number past the end are
 *     all refused rather than followed;
 *   - a frame without an index says so (count 0, seek refused), and a
 *     sequential decode of an indexed frame still validates the table;
 *   - the index composes with the other options (the deepest encoder
 *     level, a short final block, one block, many blocks).
 */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"

#define FRAME_N   (256u * 1024u) /* 4 blocks at 64 KiB, 16 at 16 KiB */
#define BLK       65536u

static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static uint8_t in[FRAME_N];
static uint8_t comp[FRAME_N + 2u * BLK];
static uint8_t out[FRAME_N];
static uint8_t blk[BLK];

static int fails = 0;
static int checks = 0;
static uint32_t rng = 0x2545F491u;

#define CHECK(cond, ...) do {                       \
    checks++;                                       \
    if (!(cond)) {                                  \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
        fails++;                                    \
    }                                               \
} while (0)

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void fill_input(uint32_t n, int kind)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        switch (kind) {
        case 0:
            in[i] = (uint8_t)(i / 7u);
            break;
        case 1:
            in[i] = (uint8_t)rnd();
            break;
        default:
            in[i] = (uint8_t)((i & 0x1FFu) < 300u ? (uint8_t)(i & 0x3Fu) : 0u);
            break;
        }
    }
}

/* Rewrite the frame header so a deliberate field edit survives its CRC. */
static void reseal_header(uint8_t *frame)
{
    zpse_store_u32le(frame + 28u, zpse_crc32c(frame, 28u));
}

static void seek_all(uint32_t n, uint32_t blk_size, uint32_t enc_flags,
                     int kind)
{
    size_t clen = 0;
    uint32_t nblocks;
    uint32_t count = 0u;
    uint32_t k;
    uint32_t i;

    fill_input(n, kind);
    CHECK(zpse_compress_frame_indexed(in, n, comp, sizeof comp, &clen,
                                      blk_size, enc_flags, ea) == PSE_OK,
          "indexed frame encode (n %u, block %u, flags %u)", n, blk_size,
          enc_flags);
    nblocks = n / blk_size + (n % blk_size != 0u ? 1u : 0u);
    CHECK(zpse_frame_index_count(comp, clen, &count) == PSE_OK,
          "index count reads");
    CHECK(count == nblocks, "index count %u, expected %u", count, nblocks);

    /* The whole frame still decodes sequentially, and its index agrees. */
    {
        size_t got = 0;
        CHECK(zpse_decompress_frame(comp, clen, out, sizeof out, &got, da) ==
                  PSE_OK,
              "indexed frame sequential decode");
        CHECK(got == n && memcmp(out, in, n) == 0,
              "indexed frame sequential bytes");
    }

    /* Every block, in an order that is not the stream's: a seek that
     * worked only because the previous seek left a cache warm would show
     * up here as a wrong block. */
    for (i = 0; i < nblocks; i++) {
        uint32_t got = 0u;
        uint32_t want;
        uint32_t off;

        k = (i * 7u + 3u) % nblocks;
        off = k * blk_size;
        want = n - off;
        if (want > blk_size) {
            want = blk_size;
        }
        CHECK(zpse_seek_frame(comp, clen, k, blk, sizeof blk, &got, da) ==
                  PSE_OK,
              "seek block %u of %u", k, nblocks);
        CHECK(got == want, "seek block %u produced %u of %u", k, got, want);
        CHECK(memcmp(blk, in + off, want) == 0, "seek block %u bytes", k);
    }

    /* Out of range: one past the end, and the far end of the space. */
    {
        uint32_t got = 0u;
        CHECK(zpse_seek_frame(comp, clen, nblocks, blk, sizeof blk, &got,
                              da) == PSE_BAD_ARG,
              "seek past the last block refused");
        CHECK(zpse_seek_frame(comp, clen, 0xFFFFFFFFu, blk, sizeof blk, &got,
                              da) == PSE_BAD_ARG,
              "seek far past the end refused");
    }
}

static void corrupt_cases(void)
{
    size_t clen = 0;
    uint32_t count = 0u;
    uint32_t frame_index_off;
    uint32_t got = 0u;

    fill_input(FRAME_N, 2);
    CHECK(zpse_compress_frame_indexed(in, FRAME_N, comp, sizeof comp, &clen,
                                      BLK, 0u, ea) == PSE_OK,
          "corruption source frame");
    if (zpse_frame_header_read(NULL, comp) != PSE_BAD_ARG) {
        CHECK(0, "header read NULL ctx");
    }
    frame_index_off = zpse_load_u32le(comp + 24u);
    CHECK(frame_index_off > 32u && (size_t)frame_index_off < clen,
          "index offset in range (%u)", frame_index_off);
    CHECK(zpse_frame_index_count(comp, clen, &count) == PSE_OK && count == 4u,
          "source index count %u", count);

    /* An entry whose size does not describe the block it points at. */
    {
        uint32_t size_at = frame_index_off + ZPSE_INDEX_HEADER_SIZE + 4u;
        uint32_t keep = zpse_load_u32le(comp + size_at);

        zpse_store_u32le(comp + size_at, keep + 1u);
        /* The entry table is checksummed, so re-seal it the way a writer
         * that agreed with itself would: the point is that the ENTRY is
         * wrong, not that the table is. */
        zpse_store_u32le(comp + frame_index_off + 8u,
                         zpse_crc32c(comp + frame_index_off +
                                         ZPSE_INDEX_HEADER_SIZE,
                                     (size_t)count * ZPSE_INDEX_ENTRY_SIZE));
        CHECK(zpse_seek_frame(comp, clen, 0u, blk, sizeof blk, &got, da) ==
                  PSE_BAD_HEADER,
              "entry/block size mismatch refused");
        zpse_store_u32le(comp + size_at, keep);
        zpse_store_u32le(comp + frame_index_off + 8u,
                         zpse_crc32c(comp + frame_index_off +
                                         ZPSE_INDEX_HEADER_SIZE,
                                     (size_t)count * ZPSE_INDEX_ENTRY_SIZE));
    }

    /* An entry that points at the middle of a block. */
    {
        uint32_t off_at = frame_index_off + ZPSE_INDEX_HEADER_SIZE;
        uint32_t keep = zpse_load_u32le(comp + off_at);

        zpse_store_u32le(comp + off_at, keep + 1u);
        zpse_store_u32le(comp + frame_index_off + 8u,
                         zpse_crc32c(comp + frame_index_off +
                                         ZPSE_INDEX_HEADER_SIZE,
                                     (size_t)count * ZPSE_INDEX_ENTRY_SIZE));
        CHECK(zpse_seek_frame(comp, clen, 0u, blk, sizeof blk, &got, da) < 0,
              "entry pointing inside a block refused");
        zpse_store_u32le(comp + off_at, keep);
        zpse_store_u32le(comp + frame_index_off + 8u,
                         zpse_crc32c(comp + frame_index_off +
                                         ZPSE_INDEX_HEADER_SIZE,
                                     (size_t)count * ZPSE_INDEX_ENTRY_SIZE));
    }

    /* A corrupted entry table: the checksum is what catches this. */
    {
        uint8_t keep = comp[frame_index_off + ZPSE_INDEX_HEADER_SIZE + 2u];

        comp[frame_index_off + ZPSE_INDEX_HEADER_SIZE + 2u] ^= 0xFFu;
        CHECK(zpse_seek_frame(comp, clen, 0u, blk, sizeof blk, &got, da) ==
                  PSE_BAD_HEADER_CRC,
              "index checksum refuses a flipped entry");
        CHECK(zpse_frame_index_count(comp, clen, &count) ==
                  PSE_BAD_HEADER_CRC,
              "index count refuses a flipped entry");
        comp[frame_index_off + ZPSE_INDEX_HEADER_SIZE + 2u] = keep;
    }

    /* A header that claims an index where there is none. */
    {
        uint32_t keep = zpse_load_u32le(comp + 24u);

        zpse_store_u32le(comp + 24u, 0u);
        reseal_header(comp);
        CHECK(zpse_frame_index_count(comp, clen, &count) == PSE_BAD_HEADER,
              "zero index offset refused");
        CHECK(zpse_seek_frame(comp, clen, 0u, blk, sizeof blk, &got, da) ==
                  PSE_BAD_HEADER,
              "seek with a zero index offset refused");
        zpse_store_u32le(comp + 24u, keep);
        reseal_header(comp);
    }

    /* Truncation: the table has to fit in what is there. */
    {
        CHECK(zpse_frame_index_count(comp, clen - 4u, &count) == PSE_TRUNCATED,
              "truncated index refused");
    }

    /* A frame encoded WITHOUT an index: count 0, seek unsupported. */
    {
        size_t nlen = 0;

        CHECK(zpse_compress_frame(in, FRAME_N, comp, sizeof comp, &nlen, BLK,
                                  ea) == PSE_OK,
              "plain frame encode");
        CHECK(zpse_frame_index_count(comp, nlen, &count) == PSE_OK &&
                  count == 0u,
              "plain frame reports no index");
        CHECK(zpse_seek_frame(comp, nlen, 0u, blk, sizeof blk, &got, da) ==
                  PSE_UNSUPPORTED_FEATURE,
              "seek on a plain frame refused");
    }
}

int main(void)
{
    static const uint32_t blocks[] = { 4096u, 16384u, 65536u, 32768u };

    seek_all(FRAME_N, 65536u, 0u, 0);
    seek_all(FRAME_N, 16384u, 0u, 1);
    seek_all(FRAME_N, 65536u, (uint32_t)ZPSE_ENC_SPEED_MAX, 2);
    seek_all(FRAME_N, 16384u, (uint32_t)ZPSE_ENC_SPEED_FASTEST, 0);
    seek_all(65536u, 65536u, 0u, 1); /* one block */
    corrupt_cases();

    /* A block size that does not divide the input leaves a short block
     * at the end: the index's last entry must describe that one too. */
    {
        uint32_t i;

        for (i = 0; i < sizeof blocks / sizeof blocks[0]; i++) {
            uint32_t count = 0u;
            uint32_t k;
            uint32_t nblocks;

            fill_input(FRAME_N, 0);
            {
                size_t clen = 0;

                CHECK(zpse_compress_frame_indexed(in, FRAME_N, comp,
                                                  sizeof comp, &clen,
                                                  blocks[i], 0u, ea) == PSE_OK,
                      "block size %u encode", blocks[i]);
                CHECK(zpse_frame_index_count(comp, clen, &count) == PSE_OK,
                      "block size %u count", blocks[i]);
                nblocks = (FRAME_N + blocks[i] - 1u) / blocks[i];
                CHECK(count == nblocks, "block size %u: %u of %u entries",
                      blocks[i], count, nblocks);
                for (k = 0; k < count; k++) {
                    uint32_t got = 0u;
                    uint32_t want = FRAME_N - k * blocks[i];
                    uint32_t off = k * blocks[i];

                    if (want > blocks[i]) {
                        want = blocks[i];
                    }
                    CHECK(zpse_seek_frame(comp, clen, k, blk, sizeof blk,
                                          &got, da) == PSE_OK,
                          "block size %u: seek %u", blocks[i], k);
                    CHECK(got == want && memcmp(blk, in + off, want) == 0,
                          "block size %u: seek %u bytes", blocks[i], k);
                }
            }
        }
    }

    if (fails == 0) {
        printf("FRAME: ALL OK (%d checks)\n", checks);
        return 0;
    }
    printf("FRAME: %d FAILURES of %d checks\n", fails, checks);
    return 1;
}
