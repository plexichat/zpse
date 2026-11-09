/* The caller-side page dictionary: the content tag, the index and the
 * header rewrite that lets a stored block stand at a new position.
 *
 * Four contracts, each of which a page store depends on:
 *
 *   - zpse_page_hash identifies a page's CONTENT: equal pages tag equal, any
 *     single-bit change anywhere in the page changes the tag, the length is
 *     part of the tag, and 0/0 (the empty entry) is never handed out.
 *   - zpse_pagedict_* stores and returns a caller's own slot id, compares the
 *     whole 128-bit tag (a 64-bit half agreeing is not a hit), maps directly
 *     (a new tag evicts the entry it indexes), and refuses a size it cannot
 *     mask with rather than indexing out of bounds.
 *   - zpse_block_retag rewrites exactly the two header fields a position owns
 *     -- block_sequence_number and the is-last flag -- in whichever header form
 *     the block carries, and fixes the header CRC so the block still parses.
 *   - Reuse is byte-neutral: a stored block retagged for a new position is
 *     byte-identical to a fresh encode of that page at that position, in both
 *     header forms. That is what makes a dictionary hit indistinguishable from
 *     a miss in the stream, and it is why nothing in the decoder changes.
 *
 * The store loop at the end is the caller's sequence from
 * docs/page-dictionary.md run end to end: hash, look up, serve or encode,
 * insert only what compressed, and decode every hit. */

#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

#define PAGE 4096u
/* 0..7 unique and compressible, 8..15 copies of them, 16..17 two copies of one
 * page that does not compress (so neither is worth an entry). */
#define NPAGES 18u
/* Wide enough that the corpus's eight compressible tags do not collide: with
 * 8 tags in 1,024 slots the index is the mapping being tested, not the hash.
 * The collision and eviction behaviour is pinned separately in test_index. */
#define NENT 1024u
#define CAP (PAGE + 64u)

static uint8_t src[NPAGES][PAGE];
static uint8_t blk[NPAGES][CAP];
static uint32_t blen[NPAGES];
static uint8_t ref[CAP];
static uint8_t dec[PAGE];

/* The "store": one grow-only pool, and a slot -> {offset, length} pair. Slots
 * are handed out by the caller, which is what the index stores. */
static uint8_t pool[NPAGES * CAP];
static uint32_t poff[NPAGES];
static uint32_t plen[NPAGES];
static uint32_t pool_used;
static uint32_t next_slot;

static uint64_t tag_lo[NENT];
static uint64_t tag_hi[NENT];
static uint32_t tag_slot[NENT];

static int fails = 0;
static int checks = 0;
static uint64_t rng = 0x9E3779B97F4A7C15ULL;

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
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng >> 33);
}

/* Compressible: a ramp with structure a predictor can follow. */
static void fill_pattern(uint8_t *p, uint32_t seed, uint32_t n)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        p[i] = (uint8_t)((seed * 31u + i * 7u) ^ (i >> 3));
    }
}

/* Incompressible: uniform noise, so the encoder falls back to RAW. */
static void fill_noise(uint8_t *p, uint32_t n)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        p[i] = (uint8_t)(rnd() >> 11);
    }
}

static void make_corpus(void)
{
    uint32_t q;

    for (q = 0; q < 8u; q++) {
        fill_pattern(src[q], q + 1u, PAGE);
    }
    for (q = 8u; q < 16u; q++) {
        memcpy(src[q], src[q - 8u], PAGE);
    }
    fill_noise(src[16], PAGE);
    memcpy(src[17], src[16], PAGE);
}

/* ---- the tag ------------------------------------------------------------ */

static void test_hash(void)
{
    uint8_t a[PAGE];
    uint8_t b[PAGE];
    uint64_t lo, hi, lo2, hi2;
    uint32_t i;

    fill_pattern(a, 7u, PAGE);
    memcpy(b, a, PAGE);
    zpse_page_hash(a, PAGE, &lo, &hi);
    zpse_page_hash(b, PAGE, &lo2, &hi2);
    CHECK(lo == lo2 && hi == hi2, "equal pages must tag equal");
    CHECK((lo | hi) != 0u, "a real page must not tag 0/0");

    /* Every byte position matters, and both halves move with it: the index
     * compares both, and a tag whose halves were correlated would waste one. */
    for (i = 0; i < PAGE; i += 61u) {
        b[i] ^= 0x01u;
        zpse_page_hash(b, PAGE, &lo2, &hi2);
        CHECK(lo2 != lo, "byte %u flipped, low tag unchanged", i);
        CHECK(hi2 != hi, "byte %u flipped, high tag unchanged", i);
        b[i] ^= 0x01u;
    }
    /* A one-byte change at either end, because the tail is the part a
     * length-blind hash often drops. */
    b[0] ^= 0x80u;
    zpse_page_hash(b, PAGE, &lo2, &hi2);
    CHECK(lo2 != lo, "first byte is not in the tag");
    b[0] ^= 0x80u;
    b[PAGE - 1u] ^= 0x01u;
    zpse_page_hash(b, PAGE, &lo2, &hi2);
    CHECK(lo2 != lo, "last byte is not in the tag");
    b[PAGE - 1u] ^= 0x01u;

    /* The length is part of the tag: a prefix must not collide with the whole
     * page, or a short final page could be served a full page's block. */
    zpse_page_hash(a, PAGE - 1u, &lo2, &hi2);
    CHECK(lo2 != lo || hi2 != hi, "length is not in the tag");

    zpse_page_hash(0, PAGE, &lo2, &hi2);
    CHECK(lo2 == 0u && hi2 == 0u, "a NULL page must tag as empty");
    zpse_page_hash(a, 0u, &lo2, &hi2);
    CHECK(lo2 == 0u && hi2 == 0u, "a zero-length page must tag as empty");
}

/* ---- the index ---------------------------------------------------------- */

static void test_index(void)
{
    ZpsePageDict d;
    uint32_t s = 0xFFFFFFFFu;

    zpse_pagedict_init(&d, tag_lo, tag_hi, tag_slot, NENT);
    CHECK(d.entries == NENT, "a valid size must be accepted");
    CHECK(zpse_pagedict_find(&d, 0x1234u, 0x5678u, &s) == 0,
          "an empty dictionary must miss");
    zpse_pagedict_insert(&d, 0x1234u, 0x5678u, 7u);
    CHECK(zpse_pagedict_find(&d, 0x1234u, 0x5678u, &s) == 1 && s == 7u,
          "a stored tag must find its slot");
    CHECK(zpse_pagedict_find(&d, 0x1234u, 0x5679u, &s) == 0,
          "the high half must be compared");
    CHECK(zpse_pagedict_find(&d, 0x1235u, 0x5678u, &s) == 0,
          "the low half must be compared");

    /* The empty tag is not a tag: inserting it must not make a lookup of 0/0
     * succeed, because a caller that forgot to hash would then be served an
     * arbitrary block. */
    CHECK(zpse_pagedict_find(&d, 0u, 0u, &s) == 0, "0/0 must never be found");
    zpse_pagedict_insert(&d, 0u, 0u, 3u);
    CHECK(zpse_pagedict_find(&d, 0u, 0u, &s) == 0,
          "inserting 0/0 must not make it findable");
    CHECK(zpse_pagedict_find(&d, 1u, 0u, &s) == 1 && s == 3u,
          "the empty tag must be stored as 1/0");

    /* Direct mapping: two tags on the same index keep only the second. */
    s = 0xFFFFFFFFu;
    zpse_pagedict_insert(&d, (uint64_t)NENT + 5u, 11u, 100u);
    zpse_pagedict_insert(&d, 5u, 12u, 101u);
    CHECK(zpse_pagedict_find(&d, (uint64_t)NENT + 5u, 11u, &s) == 0,
          "a colliding insert must evict the entry it overwrote");
    CHECK(zpse_pagedict_find(&d, 5u, 12u, &s) == 1 && s == 101u,
          "the evicting entry must be found");

    /* A size that cannot be masked is refused outright rather than indexed
     * with a wrong mask. */
    zpse_pagedict_init(&d, tag_lo, tag_hi, tag_slot, 100u);
    CHECK(d.entries == 0u, "a non-power-of-two size must be refused");
    CHECK(zpse_pagedict_find(&d, 1u, 1u, &s) == 0,
          "a refused dictionary must miss");
    zpse_pagedict_insert(&d, 1u, 1u, 1u); /* must not touch the arrays */
    zpse_pagedict_init(&d, tag_lo, tag_hi, tag_slot, 8u);
    CHECK(d.entries == 0u, "fewer than 16 entries must be refused");
    zpse_pagedict_init(&d, 0, tag_hi, tag_slot, NENT);
    CHECK(d.entries == 0u, "a NULL array must be refused");
}

/* ---- the store loop ----------------------------------------------------- */

static void test_store(void)
{
    ZpsePageDict d;
    uint32_t q;
    uint32_t hits = 0;
    uint32_t misses = 0;
    int last_ok = 1;

    make_corpus();
    zpse_pagedict_init(&d, tag_lo, tag_hi, tag_slot, NENT);
    pool_used = 0u;
    next_slot = 0u;

    for (q = 0; q < NPAGES; q++) {
        int last = (q + 1u == NPAGES) ? 1 : 0;
        uint64_t lo, hi;
        uint32_t len = 0u;
        uint32_t slot = 0u;
        uint32_t rlen = 0u;
        int rc;

        zpse_page_hash(src[q], PAGE, &lo, &hi);
        if (zpse_pagedict_find(&d, lo, hi, &slot) &&
            slot < next_slot && plen[slot] > 0u) {
            /* Serve the stored block, retagged for this position. */
            len = plen[slot];
            memcpy(blk[q], pool + poff[slot], len);
            rc = zpse_block_retag(blk[q], len, q, last);
            CHECK(rc == PSE_OK, "retag failed: %s", zpse_status_name(rc));
            hits++;
        } else {
            rc = zpse_compress_block(src[q], PAGE, blk[q], CAP, &len, q, last,
                                     ea);
            CHECK(rc == PSE_OK, "encode failed: %s", zpse_status_name(rc));
            misses++;
            /* Only what compressed is worth an entry: a RAW page's "block" is
             * longer than the page, so storing it would cost more than the
             * hash it saves. */
            if (len < PAGE) {
                poff[next_slot] = pool_used;
                plen[next_slot] = len;
                memcpy(pool + pool_used, blk[q], len);
                pool_used += len;
                zpse_pagedict_insert(&d, lo, hi, next_slot);
                next_slot++;
            }
        }
        blen[q] = len;

        /* The dictionary's whole contract: the bytes written for a hit are the
         * bytes a fresh encode of this page at this position writes. */
        rc = zpse_compress_block(src[q], PAGE, ref, CAP, &rlen, q, last, ea);
        CHECK(rc == PSE_OK, "reference encode failed: %s",
              zpse_status_name(rc));
        CHECK(rlen == len && memcmp(ref, blk[q], len) == 0,
              "page %u: served block differs from a fresh encode", q);

        /* And every block, served or encoded, decodes to the page. */
        {
            uint32_t got = 0u;

            rc = zpse_decompress_block(blk[q], len, dec, PAGE, &got, da);
            CHECK(rc == PSE_OK && got == PAGE &&
                      memcmp(dec, src[q], PAGE) == 0,
                  "page %u: block decoded wrong", q);
            if (rc == PSE_OK && got == PAGE) {
                ZpseBlockHeader h;

                if (zpse_block_header_parse(&h, blk[q], len) == PSE_OK) {
                    if ((h.block_flags & ZPSE_B_LAST) != 0u ? !last : last) {
                        last_ok = 0;
                    }
                }
            }
        }
    }
    /* Pages 8..15 repeat 0..7 and hit; 16,17 are one noise page twice, and a
     * RAW block is never inserted, so both encode. */
    CHECK(hits == 8u, "expected 8 hits, got %u", hits);
    CHECK(misses == NPAGES - 8u, "expected %u misses, got %u", NPAGES - 8u,
          misses);
    CHECK(last_ok, "a block's is-last flag disagrees with its position");
    CHECK(pool_used > 0u && pool_used < (uint32_t)sizeof pool,
          "store use %u is implausible", pool_used);
}

/* ---- the header rewrite ------------------------------------------------- */

static void test_retag_edges(void)
{
    static uint8_t big[16384];
    static uint8_t cbig[65536];
    static uint8_t copy[65536];
    static uint8_t out[16384];
    uint8_t page[PAGE];
    uint32_t clen = 0u, olen = 0u, cs = 0u, us = 0u;
    ZpseBlockHeader h;
    int rc;

    /* A long-header block: 16 KiB is above ZPSE_BLOCK_SHORT_MAX, so the header
     * stays 32 bytes and its CRC32-C covers seq and flags. */
    fill_pattern(big, 3u, (uint32_t)sizeof big);
    rc = zpse_compress_block(big, (uint32_t)sizeof big, cbig,
                             (uint32_t)sizeof cbig, &clen, 3u, 0, ea);
    CHECK(rc == PSE_OK && clen < sizeof big,
          "16 KiB pattern should compress (rc %s)", zpse_status_name(rc));

    CHECK(zpse_block_retag(cbig, clen, 1234u, 1) == PSE_OK,
          "retag of a long-header block failed");
    CHECK(zpse_block_header_parse(&h, cbig, clen) == PSE_OK &&
              h.header_size == (uint8_t)ZPSE_BLOCK_HEADER_SIZE &&
              h.block_sequence_number == 1234u &&
              (h.block_flags & ZPSE_B_LAST) != 0u,
          "the long header did not take the new position");
    CHECK(zpse_block_probe(cbig, clen, &cs, &us) == PSE_OK && cs == clen &&
              us == (uint32_t)sizeof big,
          "a retagged block must still probe as itself");
    rc = zpse_decompress_block(cbig, clen, out, (uint32_t)sizeof out, &olen,
                               da);
    CHECK(rc == PSE_OK && olen == (uint32_t)sizeof big &&
              memcmp(out, big, sizeof big) == 0,
          "a retagged long-header block decoded wrong");

    /* Clearing the flag is the same rewrite in the other direction, and the
     * payload must not move: the header CRC is the only other byte that can
     * change. */
    memcpy(copy, cbig, clen);
    CHECK(zpse_block_retag(cbig, clen, 3u, 0) == PSE_OK,
          "retag back to the original position failed");
    CHECK(memcmp(copy + ZPSE_BLOCK_HEADER_SIZE, cbig + ZPSE_BLOCK_HEADER_SIZE,
                 clen - ZPSE_BLOCK_HEADER_SIZE) == 0,
          "retag moved the payload");
    CHECK(zpse_block_header_parse(&h, cbig, clen) == PSE_OK &&
              h.block_sequence_number == 3u &&
              (h.block_flags & ZPSE_B_LAST) == 0u,
          "the long header did not take the cleared position");

    /* A page's block carries the 16-byte header, whose checksum is a 16-bit
     * truncation over 14 bytes: the same rewrite, different bytes. */
    fill_pattern(page, 9u, PAGE);
    clen = 0u;
    rc = zpse_compress_block(page, PAGE, cbig, (uint32_t)sizeof cbig, &clen,
                             5u, 0, ea);
    CHECK(rc == PSE_OK && clen < PAGE, "the page should compress");
    CHECK(zpse_block_header_parse(&h, cbig, clen) == PSE_OK &&
              h.header_size == (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT,
          "a page's block should carry the 16-byte header");
    CHECK(zpse_block_retag(cbig, clen, 99u, 1) == PSE_OK,
          "retag of a page's block failed");
    CHECK(zpse_block_header_parse(&h, cbig, clen) == PSE_OK &&
              h.header_size == (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT &&
              h.block_sequence_number == 99u &&
              (h.block_flags & ZPSE_B_LAST) != 0u &&
              h.compressed_block_size == clen,
          "the 16-byte header did not take the new position");
    rc = zpse_decompress_block(cbig, clen, out, (uint32_t)sizeof out, &olen,
                               da);
    CHECK(rc == PSE_OK && olen == PAGE && memcmp(out, page, PAGE) == 0,
          "a retagged page block decoded wrong");

    /* Refusals. Each one has to leave the block readable: a caller that
     * retags a corrupt or truncated block must not write a stream that only
     * fails later. */
    CHECK(zpse_block_retag(0, clen, 1u, 0) == PSE_BAD_ARG,
          "a NULL block must be refused");
    CHECK(zpse_block_retag(cbig, 8u, 1u, 0) == PSE_TRUNCATED,
          "a block shorter than a page header must be truncated");
    CHECK(zpse_block_retag(cbig, clen - 1u, 1u, 0) == PSE_BAD_HEADER,
          "a length the header disagrees with must be refused");
    CHECK(zpse_block_retag(cbig, clen + 1u, 1u, 0) == PSE_BAD_HEADER,
          "a length with trailing bytes must be refused");
    memcpy(copy, cbig, clen);
    copy[3] ^= 0x01u; /* transform id: inside the header CRC */
    CHECK(zpse_block_retag(copy, clen, 1u, 0) == PSE_BAD_HEADER_CRC,
          "a bad header CRC must be refused");
    memcpy(copy, cbig, clen);
    copy[0] ^= 0xFFu; /* sync */
    CHECK(zpse_block_retag(copy, clen, 1u, 0) == PSE_BAD_HEADER,
          "a block with no sync must be refused");

    /* A RAW block is a block too: its header takes a position, and its
     * payload is the page. */
    {
        static uint8_t noise[PAGE];
        uint32_t rlen = 0u;

        fill_noise(noise, PAGE);
        rc = zpse_compress_block(noise, PAGE, cbig, (uint32_t)sizeof cbig,
                                 &rlen, 1u, 0, ea);
        CHECK(rc == PSE_OK && rlen >= PAGE, "noise should be written raw");
        CHECK(zpse_block_retag(cbig, rlen, 2u, 1) == PSE_OK,
              "retag of a RAW block failed");
        rc = zpse_decompress_block(cbig, rlen, out, (uint32_t)sizeof out,
                                   &olen, da);
        CHECK(rc == PSE_OK && olen == PAGE && memcmp(out, noise, PAGE) == 0,
              "a retagged RAW block decoded wrong");
    }
}

int main(void)
{
    test_hash();
    test_index();
    test_store();
    test_retag_edges();
    printf("PAGEDICT: checks=%d fails=%d %s\n", checks, fails,
           fails == 0 ? "OK" : "FAILED");
    return fails == 0 ? 0 : 1;
}
