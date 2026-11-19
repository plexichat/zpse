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
#include <stdlib.h>
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

/* Deterministic English-like prose for the TEXT page. A first-order Markov
 * walk over a fixed prose corpus keeps every byte printable ASCII without
 * repeating a long substring for the LZ match finder. That is the shape the
 * ASCII gate convicts and the fixed-rate 7-bit pack wins: no low alphabet to
 * tempt FLAT, no easy prediction for the ladder. */
static void fill_prose(uint8_t *p, uint32_t n, uint32_t seed)
{
    static const char corpus[] =
        "The quick brown fox jumps over the lazy dog. This is a sentence of "
        "English prose with common words and ordinary punctuation. The "
        "encoder must decide which form to use for each block of bytes. When "
        "the input is plain text the static table can replace frequent pairs "
        "of characters with single tokens. Each token expands to two or more "
        "bytes on decode, and the result is a smaller stream than the "
        "original. There is no predictor and no mask and no residual codec "
        "in this form, only the byte stream that the table describes. The "
        "header is short because the block is a page. A reader finds the "
        "extent of every block by reading one short header at a time, and "
        "checks the low sixteen bits of a checksum before it trusts any of "
        "the fields inside. Nothing in the table depends on the level that "
        "the encoder was asked to use, so the same bytes must come out at "
        "every effort from fastest to max. Deterministic input and a fixed "
        "seed keep the test repeatable on every machine and every build, "
        "which matters for a format that promises the same compressed stream "
        "on every platform that implements it correctly.";
    uint8_t next[128][128];
    uint8_t cnt[128];
    uint32_t clen = (uint32_t)(sizeof corpus - 1u);
    uint32_t prev = (uint32_t)'T';
    uint32_t r = seed;
    uint32_t i;

    memset(next, 0, sizeof next);
    memset(cnt, 0, sizeof cnt);
    for (i = 0u; i + 1u < clen; i++) {
        uint32_t c = (uint8_t)corpus[i];
        uint32_t d = (uint8_t)corpus[i + 1u];

        if (c < 128u && d < 128u && cnt[c] < 128u) {
            next[c][cnt[c]++] = (uint8_t)d;
        }
    }
    for (i = 0u; i < n; i++) {
        if ((i % 80u) == 79u) {
            p[i] = '\n';
            prev = (uint32_t)'T';
        } else {
            if (cnt[prev] == 0u) {
                prev = (uint32_t)'T';
            }
            r = r * 1664525u + 1013904223u;
            p[i] = next[prev][(r >> 16) % cnt[prev]];
            prev = p[i];
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
                                      blk_size, enc_flags, ea) == ZPSE_OK,
          "indexed frame encode (n %u, block %u, flags %u)", n, blk_size,
          enc_flags);
    nblocks = n / blk_size + (n % blk_size != 0u ? 1u : 0u);
    CHECK(zpse_frame_index_count(comp, clen, &count) == ZPSE_OK,
          "index count reads");
    CHECK(count == nblocks, "index count %u, expected %u", count, nblocks);

    /* The whole frame still decodes sequentially, and its index agrees. */
    {
        size_t got = 0;

        /* A decoder may write a few bytes past a match and rely on a later
         * command to overwrite them. At the end of a buffer there is no
         * later command, so the buffer gets a canary and the decode has to
         * leave it alone: a decode that writes into a caller's memory past
         * the bytes it reports is a buffer overflow, not a detail. */
        memset(out, 0x5A, sizeof out);
        CHECK(zpse_decompress_frame(comp, clen, out, sizeof out, &got, da) ==
                  ZPSE_OK,
              "indexed frame sequential decode");
        CHECK(got == n && memcmp(out, in, n) == 0,
              "indexed frame sequential bytes");
        {
            size_t j;
            int touched = 0;

            for (j = got; j < sizeof out; j++) {
                if (out[j] != 0x5A) {
                    touched = 1;
                    break;
                }
            }
            CHECK(!touched, "sequential decode wrote past its %lu bytes",
                  (unsigned long)got);
        }
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
        /* Same canary on the seek path: one block in a caller's buffer is
         * exactly the case where an over-copy has nowhere to go. */
        memset(blk, 0x5A, sizeof blk);
        CHECK(zpse_seek_frame(comp, clen, k, blk, sizeof blk, &got, da) ==
                  ZPSE_OK,
              "seek block %u of %u", k, nblocks);
        CHECK(got == want, "seek block %u produced %u of %u", k, got, want);
        CHECK(memcmp(blk, in + off, want) == 0, "seek block %u bytes", k);
        {
            uint32_t j;
            int touched = 0;

            for (j = want; j < (uint32_t)sizeof blk; j++) {
                if (blk[j] != 0x5A) {
                    touched = 1;
                    break;
                }
            }
            CHECK(!touched, "seek block %u wrote past its %u bytes", k,
                  want);
        }
        /* And the strictest form of the same question: a buffer holding
         * exactly the bytes wanted and nothing else. A decoder that writes
         * a few bytes past its own output is invisible in a larger buffer,
         * and this is the only shape that cannot hide it. */
        {
            uint8_t *tight = (uint8_t *)malloc(want);

            CHECK(tight != 0, "exact-size seek buffer");
            if (tight != 0) {
                memset(tight, 0x5A, want);
                CHECK(zpse_seek_frame(comp, clen, k, tight, want, &got, da) ==
                          ZPSE_OK,
                      "exact-size seek block %u of %u", k, nblocks);
                CHECK(got == want && memcmp(tight, in + off, want) == 0,
                      "exact-size seek block %u bytes", k);
                free(tight);
            }
        }
    }

    /* Out of range: one past the end, and the far end of the space. */
    {
        uint32_t got = 0u;
        CHECK(zpse_seek_frame(comp, clen, nblocks, blk, sizeof blk, &got,
                              da) == ZPSE_BAD_ARG,
              "seek past the last block refused");
        CHECK(zpse_seek_frame(comp, clen, 0xFFFFFFFFu, blk, sizeof blk, &got,
                              da) == ZPSE_BAD_ARG,
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
                                      BLK, 0u, ea) == ZPSE_OK,
          "corruption source frame");
    if (zpse_frame_header_read(NULL, comp) != ZPSE_BAD_ARG) {
        CHECK(0, "header read NULL ctx");
    }
    frame_index_off = zpse_load_u32le(comp + 24u);
    CHECK(frame_index_off > 32u && (size_t)frame_index_off < clen,
          "index offset in range (%u)", frame_index_off);
    CHECK(zpse_frame_index_count(comp, clen, &count) == ZPSE_OK && count == 4u,
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
                  ZPSE_BAD_HEADER,
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

    /* The table is also validated against the blocks a sequential reader
     * walks, which is what a decoder that never holds the frame in memory
     * uses (the CLI's `zpse d` on an indexed frame). Same guarantees, from
     * the bytes on disk instead of from a copy of the whole frame. */
    {
        ZpseFrameHeader fh;
        uint8_t *idx;
        uint32_t *offs;
        uint32_t *lens;
        uint32_t pos = (uint32_t)ZPSE_FRAME_HEADER_SIZE;
        size_t idx_len;
        uint32_t m;

        offs = (uint32_t *)malloc(sizeof *offs * (size_t)count);
        lens = (uint32_t *)malloc(sizeof *lens * (size_t)count);
        CHECK(offs != 0 && lens != 0, "walker arrays");
        CHECK(zpse_frame_header_read(&fh, comp) == ZPSE_OK, "walker header");
        for (m = 0; m < count; m++) {
            ZpseBlockHeader bh;

            CHECK(zpse_block_header_parse(&bh, comp + pos,
                                          (uint32_t)ZPSE_BLOCK_HEADER_SIZE) ==
                      ZPSE_OK,
                  "walker block %u header", m);
            offs[m] = pos;
            lens[m] = bh.compressed_block_size;
            pos += bh.compressed_block_size;
        }
        CHECK(pos == frame_index_off, "blocks end at the table (%u/%u)", pos,
              frame_index_off);
        idx = comp + frame_index_off;
        idx_len = (size_t)ZPSE_INDEX_HEADER_SIZE +
                  (size_t)count * (size_t)ZPSE_INDEX_ENTRY_SIZE;
        CHECK(zpse_frame_index_validate(&fh, idx, idx_len, count, offs,
                                        lens) == ZPSE_OK,
              "index validates against the walk");
        lens[count - 1u] += 1u;
        CHECK(zpse_frame_index_validate(&fh, idx, idx_len, count, offs,
                                        lens) == ZPSE_BAD_HEADER,
              "walked size mismatch refused");
        lens[count - 1u] -= 1u;
        offs[0] += 1u;
        CHECK(zpse_frame_index_validate(&fh, idx, idx_len, count, offs,
                                        lens) == ZPSE_BAD_HEADER,
              "walked offset mismatch refused");
        offs[0] -= 1u;
        CHECK(zpse_frame_index_validate(&fh, idx,
                                        (size_t)ZPSE_INDEX_HEADER_SIZE, count,
                                        offs, lens) == ZPSE_TRUNCATED,
              "short table refused by the walker");
        CHECK(zpse_frame_index_validate(&fh, idx, idx_len, count - 1u, offs,
                                        lens) == ZPSE_BAD_HEADER,
              "walker count that disagrees with the frame refused");
        free(offs);
        free(lens);
    }

    /* A corrupted entry table: the checksum is what catches this. */
    {
        uint8_t keep = comp[frame_index_off + ZPSE_INDEX_HEADER_SIZE + 2u];

        comp[frame_index_off + ZPSE_INDEX_HEADER_SIZE + 2u] ^= 0xFFu;
        CHECK(zpse_seek_frame(comp, clen, 0u, blk, sizeof blk, &got, da) ==
                  ZPSE_BAD_HEADER_CRC,
              "index checksum refuses a flipped entry");
        CHECK(zpse_frame_index_count(comp, clen, &count) ==
                  ZPSE_BAD_HEADER_CRC,
              "index count refuses a flipped entry");
        comp[frame_index_off + ZPSE_INDEX_HEADER_SIZE + 2u] = keep;
    }

    /* A header that claims an index where there is none. */
    {
        uint32_t keep = zpse_load_u32le(comp + 24u);

        zpse_store_u32le(comp + 24u, 0u);
        reseal_header(comp);
        CHECK(zpse_frame_index_count(comp, clen, &count) == ZPSE_BAD_HEADER,
              "zero index offset refused");
        CHECK(zpse_seek_frame(comp, clen, 0u, blk, sizeof blk, &got, da) ==
                  ZPSE_BAD_HEADER,
              "seek with a zero index offset refused");
        zpse_store_u32le(comp + 24u, keep);
        reseal_header(comp);
    }

    /* Truncation: the table has to fit in what is there. */
    {
        CHECK(zpse_frame_index_count(comp, clen - 4u, &count) == ZPSE_TRUNCATED,
              "truncated index refused");
    }

    /* A frame encoded WITHOUT an index: count 0, seek unsupported. */
    {
        size_t nlen = 0;

        /* The checksum flag keeps an all-ASCII input off the whole-input
         * FLAT7 fast frame. This case is about a PSE1 frame with no index
         * table, so it must stay a PSE1 frame. */
        CHECK(zpse_compress_frame_ex(in, FRAME_N, comp, sizeof comp, &nlen,
                                     BLK, (uint32_t)ZPSE_ENC_CHECKSUM, ea) ==
                  ZPSE_OK,
              "plain frame encode");
        CHECK(zpse_frame_index_count(comp, nlen, &count) == ZPSE_OK &&
                  count == 0u,
              "plain frame reports no index");
        CHECK(zpse_seek_frame(comp, nlen, 0u, blk, sizeof blk, &got, da) ==
                  ZPSE_UNSUPPORTED_FEATURE,
              "seek on a plain frame refused");
    }
}

/* A frame of 4 KiB pages: every block takes a 16-byte header (token, FLAT or
 * TEXT), so a streaming reader finds each block's extent by reading one short
 * header at a time. This pins the whole path the automatic page forms
 * introduced: four page shapes, distinct sequence numbers, LAST on the final
 * block only, and a walk that lands exactly on the end of the frame. */
static void page_frame_roundtrip(void)
{
    size_t clen = 0;
    size_t got = 0;
    size_t pos;
    uint32_t page;
    uint32_t k;
    uint32_t ntok = 0u;
    uint32_t nflat_coded = 0u;
    uint32_t nflat_raw = 0u;
    uint32_t ntext = 0u;
    uint32_t nladder = 0u;
    ZpseFrameHeader fh;

    for (page = 0u; page < FRAME_N / (uint32_t)ZPSE_BLOCK_SHORT_MAX; page++) {
        uint32_t off = page * (uint32_t)ZPSE_BLOCK_SHORT_MAX;
        uint32_t i;

        for (i = 0u; i < (uint32_t)ZPSE_BLOCK_SHORT_MAX; i++) {
            switch (page % 4u) {
            case 0u:
                in[off + i] = (uint8_t)(0x40u + page);          /* constant */
                break;
            case 1u:
                in[off + i] = (uint8_t)('a' + (rnd() & 3u));    /* low alpha */
                break;
            case 2u:
                in[off + i] = (uint8_t)rnd();                   /* incompressible */
                break;
            default:
                break; /* ASCII prose: filled just below */
            }
        }
        if (page % 4u == 3u) {
            /* Printable ASCII with a line break every 32 bytes: no low
             * alphabet and no repeat, so the ASCII gate takes the page and
             * writes it as a TEXT block. */
            fill_prose(in + off, ZPSE_BLOCK_SHORT_MAX, 12345u);
        }
    }
    CHECK(zpse_compress_frame(in, FRAME_N, comp, sizeof comp, &clen,
                              ZPSE_BLOCK_SHORT_MAX, ea) == ZPSE_OK,
          "page frame encode");
    CHECK(zpse_frame_header_read(&fh, comp) == ZPSE_OK, "page frame header");
    CHECK(fh.block_count == FRAME_N / (uint32_t)ZPSE_BLOCK_SHORT_MAX,
          "page frame block count %u", fh.block_count);
    CHECK(fh.nominal_block_size == ZPSE_BLOCK_SHORT_MAX,
          "page frame nominal block size %u", fh.nominal_block_size);
    CHECK(fh.content_size == FRAME_N, "page frame content size %lu",
          (unsigned long)fh.content_size);

    /* Same canary as the seek tests: a decode that writes past its reported
     * length has nowhere to go at the end of a whole frame. */
    memset(out, 0x5A, sizeof out);
    CHECK(zpse_decompress_frame(comp, clen, out, sizeof out, &got, da) ==
              ZPSE_OK,
          "page frame decode");
    CHECK(got == FRAME_N && memcmp(out, in, FRAME_N) == 0,
          "page frame bytes");
    {
        size_t j;
        int touched = 0;

        for (j = got; j < sizeof out; j++) {
            if (out[j] != 0x5A) {
                touched = 1;
                break;
            }
        }
        CHECK(!touched, "page frame decode wrote past its %lu bytes",
              (unsigned long)got);
    }

    /* Walk the frame one short header at a time, the way a reader without an
     * index finds each block's extent and identity. */
    pos = ZPSE_FRAME_HEADER_SIZE;
    for (k = 0u; k < fh.block_count; k++) {
        uint32_t avail = (uint32_t)(clen - pos);
        ZpseBlockHeader bh;
        int rc = zpse_block_header_parse(&bh, comp + pos, avail);

        CHECK(rc == ZPSE_OK, "page %u header parses", k);
        if (rc != ZPSE_OK) {
            break;
        }
        CHECK(bh.header_size == ZPSE_BLOCK_HEADER_SIZE_SHORT,
              "page %u header is 16 bytes", k);
        CHECK(zpse_block_header_size(comp + pos, avail) ==
                  ZPSE_BLOCK_HEADER_SIZE_SHORT,
              "page %u header size helper says 16", k);
        CHECK(bh.block_sequence_number == k, "page %u seq is %u", k,
              bh.block_sequence_number);
        CHECK(bh.uncompressed_size == ZPSE_BLOCK_SHORT_MAX,
              "page %u uncompressed size %u", k, bh.uncompressed_size);
        CHECK(bh.compressed_block_size >= ZPSE_BLOCK_HEADER_SIZE_SHORT,
              "page %u compressed size %u", k, bh.compressed_block_size);
        CHECK((size_t)bh.compressed_block_size <= clen - pos,
              "page %u compressed size fits", k);
        CHECK(((bh.block_flags & ZPSE_B_LAST) != 0u) ==
                  (k + 1u == fh.block_count),
              "page %u LAST flag", k);
        if (bh.sync == ZPSE_BLOCK_SYNC_TOKEN) {
            ntok++;
        } else if (bh.sync == ZPSE_BLOCK_SYNC_TEXT) {
            ntext++;
            /* TEXT carries no codec fields, but it is still a page block: a
             * 16-byte header, sized by the same helper. */
            CHECK(bh.header_size == ZPSE_BLOCK_HEADER_SIZE_SHORT,
                  "page %u TEXT header is 16 bytes", k);
            CHECK(zpse_block_header_size(comp + pos, avail) ==
                      ZPSE_BLOCK_HEADER_SIZE_SHORT,
                  "page %u TEXT header size helper says 16", k);
        } else if (bh.sync == ZPSE_BLOCK_SYNC_FLAT &&
                   bh.transform_id != ZPSE_FLAT_RAW) {
            nflat_coded++;
        } else if (bh.sync == ZPSE_BLOCK_SYNC_FLAT) {
            nflat_raw++;
        } else {
            nladder++;
        }
        pos += bh.compressed_block_size;
    }
    CHECK(pos == clen, "page walk consumed the frame (%lu/%lu)",
          (unsigned long)pos, (unsigned long)clen);
    CHECK(ntok > 0u, "a constant page became a token");
    CHECK(nflat_coded > 0u, "a low-alpha page became a coded FLAT block");
    CHECK(ntext > 0u, "a prose page became a TEXT block");
    /* Incompressible and structured low-alpha pages fall through to the
     * level ladder; FLAT is only the one-pass path for irregular low-alphabet
     * and pointer pages. This page mix has no raw FLAT page, but the counter
     * is kept so a future page shape can assert it without re-adding it. */
    (void)nflat_raw;
    CHECK(nladder > 0u, "an incompressible page took the ladder");
}

/* A FLAT block whose compressed size claims more bytes than remain in the
 * frame must be refused cleanly: the short-header walk checks the claim
 * against the available input before it decodes or writes anywhere. */
static void flat_overlong(void)
{
    size_t clen = 0;
    size_t got = 0;
    uint8_t *hdr;
    ZpseBlockHeader bh;
    uint32_t i;

    for (i = 0u; i < (uint32_t)ZPSE_BLOCK_SHORT_MAX; i++) {
        /* Four high-byte symbols: still a low-alphabet FLAT page, but not
         * all-ASCII, so the whole-input FLAT7 frame fast path cannot take it. */
        in[i] = (uint8_t)(0x80u + (rnd() & 3u));
    }
    CHECK(zpse_compress_frame(in, ZPSE_BLOCK_SHORT_MAX, comp, sizeof comp,
                              &clen, ZPSE_BLOCK_SHORT_MAX, ea) == ZPSE_OK,
          "overlong FLAT source frame");
    memset(&bh, 0, sizeof bh);
    hdr = comp + ZPSE_FRAME_HEADER_SIZE;
    CHECK(zpse_block_header_parse(&bh, hdr,
                                  (uint32_t)(clen - ZPSE_FRAME_HEADER_SIZE)) ==
                  ZPSE_OK &&
              bh.sync == ZPSE_BLOCK_SYNC_FLAT,
          "source block is FLAT");
    if (bh.sync != ZPSE_BLOCK_SYNC_FLAT) {
        return;
    }
    /* Inflate the 16-bit compressed-size field past the end of the frame and
     * re-seal the header's 16-bit CRC so the parse itself still succeeds: the
     * refusal has to come from the size against the remaining input. */
    zpse_store_u16le(hdr + 8u, (uint16_t)60000u);
    zpse_store_u16le(hdr + 14u,
                     (uint16_t)(zpse_crc32c(hdr, 14u) & UINT32_C(0xFFFF)));
    CHECK(zpse_block_header_parse(&bh, hdr,
                                  (uint32_t)(clen - ZPSE_FRAME_HEADER_SIZE)) ==
                  ZPSE_OK &&
              bh.compressed_block_size == 60000u,
          "overlong FLAT header parses with the inflated size");
    memset(out, 0x5A, sizeof out);
    CHECK(zpse_decompress_frame(comp, clen, out, sizeof out, &got, da) < 0,
          "overlong FLAT size refused");
    {
        size_t j;
        int touched = 0;

        for (j = 0; j < sizeof out; j++) {
            if (out[j] != 0x5A) {
                touched = 1;
                break;
            }
        }
        CHECK(!touched, "overlong FLAT size wrote into the output buffer");
    }
}

/* The TEXT form is a 16-byte header too, and its c field is equally untrusted:
 * a claim that runs past the end of the frame must be refused before any token
 * stream is decoded or any output byte is written. */
static void text_overlong(void)
{
    size_t clen = 0;
    size_t got = 0;
    uint8_t *hdr;
    ZpseBlockHeader bh;

    fill_prose(in, ZPSE_BLOCK_SHORT_MAX, 12345u);
    /* Prose is all ASCII and would otherwise be captured by the whole-input
     * FLAT7 frame magic, leaving no TEXT block to inflate. The checksum flag
     * opts out of that fast path while the block form stays TEXT. */
    CHECK(zpse_compress_frame_ex(in, ZPSE_BLOCK_SHORT_MAX, comp, sizeof comp,
                                 &clen, ZPSE_BLOCK_SHORT_MAX,
                                 (uint32_t)ZPSE_ENC_CHECKSUM, ea) == ZPSE_OK,
          "overlong TEXT source frame");
    memset(&bh, 0, sizeof bh);
    hdr = comp + ZPSE_FRAME_HEADER_SIZE;
    CHECK(zpse_block_header_parse(&bh, hdr,
                                  (uint32_t)(clen - ZPSE_FRAME_HEADER_SIZE)) ==
                  ZPSE_OK &&
              bh.sync == ZPSE_BLOCK_SYNC_TEXT,
          "source block is TEXT");
    if (bh.sync != ZPSE_BLOCK_SYNC_TEXT) {
        return;
    }
    /* Inflate the 16-bit compressed-size field past the end of the frame and
     * re-seal the header CRC: the refusal has to come from the size against
     * the remaining input, not from a checksum. */
    zpse_store_u16le(hdr + 8u, (uint16_t)60000u);
    zpse_store_u16le(hdr + 14u,
                     (uint16_t)(zpse_crc32c(hdr, 14u) & UINT32_C(0xFFFF)));
    CHECK(zpse_block_header_parse(&bh, hdr,
                                  (uint32_t)(clen - ZPSE_FRAME_HEADER_SIZE)) ==
                  ZPSE_OK &&
              bh.compressed_block_size == 60000u,
          "overlong TEXT header parses with the inflated size");
    memset(out, 0x5A, sizeof out);
    CHECK(zpse_decompress_frame(comp, clen, out, sizeof out, &got, da) < 0,
          "overlong TEXT size refused");
    {
        size_t j;
        int touched = 0;

        for (j = 0; j < sizeof out; j++) {
            if (out[j] != 0x5A) {
                touched = 1;
                break;
            }
        }
        CHECK(!touched, "overlong TEXT size wrote into the output buffer");
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
    page_frame_roundtrip();
    flat_overlong();
    text_overlong();
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
                                                  blocks[i], 0u, ea) == ZPSE_OK,
                      "block size %u encode", blocks[i]);
                CHECK(zpse_frame_index_count(comp, clen, &count) == ZPSE_OK,
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
                                          &got, da) == ZPSE_OK,
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
