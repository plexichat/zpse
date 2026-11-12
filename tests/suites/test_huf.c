/* The entropy-coded LZ form (src/huf.c): the codec on its own, its block
 * form, and the option that selects it.
 *
 * Four things are pinned:
 *
 *   - zpse_huf_encode / zpse_huf_decode round-trip the payloads the LZ forms
 *     actually produce, from the smallest one the form accepts to a full
 *     block, over shapes with very different alphabets (a run of one byte, a
 *     pointer plane, prose, noise). This is the layer's own correctness: the
 *     bit accounting, the four-way split and the pair lookups either agree
 *     with the encoder's bit writer or they do not.
 *   - The encoding is a function of the bytes and nothing else: encoding the
 *     same payload twice writes the same bits, because a block's output may
 *     not depend on when it was encoded.
 *   - A payload that does not decode is refused with a status rather than
 *     half-decoded. Every truncation and every single-byte corruption goes
 *     through the decoder, because this is a new wire form and nothing in it
 *     may walk off a buffer.
 *   - The block form: the deepest level names it with flag pattern 110 and
 *     the LZ envelope (transform 0, one segment, no metadata), the ordinary
 *     decoder reads it back, and the form is never written at a level that
 *     did not ask for it.
 *
 * The encoder does NOT refuse to code a payload it cannot shrink -- the
 * decision to keep the form belongs to the block encoder (ZPSE_HUF_DIV), and
 * a layer that returned "no" here would be deciding it twice. So the noise
 * shape below legitimately grows, and only the round trip is checked on it.
 */
#include <stdio.h>
#include <string.h>

#include "zpse/zpse.h"
#include "zpse/huf.h"
#include "zpse/lz.h"
#include "zpse/types.h"
/* The shared, seeded corpus -- the same 768 cases tests/test_sizes.c pins, so
 * the block this suite exercises is the same block on every run and on every
 * machine. */
#include "corpus_gen.h"

static unsigned char ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint16_t tab[ZPSE_LZ_TABLE_ENTRIES];

static uint8_t plain[65536];
static uint8_t inner[70000];
static uint8_t huf1[70000];
static uint8_t huf2[70000];
static uint8_t back[70000];
static uint8_t comp[140000];
static uint8_t work[140000];

static int fails = 0;
static int checks = 0;
static uint64_t rng = 0x452821E638D01377ULL;

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

/* The alphabets LZ payloads really have. A Huffman code has a different job
 * in each, and shape 0 is the one the block form is exercised on: a repeated
 * record is where the LZ forms win a page outright. */
#define N_SHAPES 5

static void make_shape(uint32_t kind, uint32_t n)
{
    uint32_t i = 0u;

    while (i < n) {
        switch (kind) {
        case 0: {
            /* A fixed 24-byte record with a counter in it: long matches and
             * a narrow token alphabet, which is what a page of pointers and
             * a page of table rows both look like to the finder. */
            static const char rec[] = "recid=0000 slot=0000 flag=0 pad=";
            uint32_t k;

            for (k = 0u; rec[k] != '\0' && i < n; k++) {
                plain[i] = (uint8_t)rec[k];
                i++;
            }
            for (k = 0u; k < 8u && i < n; k++) {
                plain[i] = (uint8_t)('a' + (i % 3u));
                i++;
            }
            break;
        }
        case 1:
            /* 4-byte words a fixed distance apart: the shape of a heap. */
            plain[i] = (uint8_t)(((i & 3u) == 3u) ? 0u
                                                 : (i / 4u % 61u) + 32u);
            i++;
            break;
        case 2: {
            static const char *words[] = { "the ", "block ", "of ", "a ",
                                           "payload ", "and ", "it ",
                                           "compresses ", "well ", "now " };
            const char *w = words[rnd() % 10u];
            uint32_t k;

            for (k = 0u; w[k] != '\0' && i < n; k++) {
                plain[i] = (uint8_t)w[k];
                i++;
            }
            break;
        }
        case 3:
            plain[i] = (uint8_t)(rnd() & 0xFFu);
            i++;
            break;
        default:
            plain[i] = 0x5A;
            i++;
            break;
        }
    }
}

/* The LZ payload of a shape, which is what this form wraps. */
static uint32_t make_inner(uint32_t kind, uint32_t n)
{
    make_shape(kind, n);
    return zpse_lz_encode(plain, n, inner, (uint32_t)sizeof inner, tab,
                          (uint32_t)ZPSE_LZ_HASH_BITS);
}

static void codec_roundtrip(void)
{
    static const uint32_t sizes[] = { 1024u, 2048u, 4096u, 16384u, 65536u };
    uint32_t k;
    uint32_t s;

    for (k = 0; k < (uint32_t)N_SHAPES; k++) {
        for (s = 0; s < (uint32_t)(sizeof sizes / sizeof sizes[0]); s++) {
            uint32_t nl = make_inner(k, sizes[s]);
            uint32_t ilen = 0u;
            uint8_t iform = 0u;
            uint32_t hl;
            int rc;

            if (nl < (uint32_t)ZPSE_HUF_MIN_INNER ||
                nl > (uint32_t)ZPSE_HUF_MAX_INNER) {
                continue;
            }
            hl = zpse_huf_encode(inner, nl, huf1, (uint32_t)sizeof huf1,
                                 (uint8_t)ZPSE_B_LZ);
            CHECK(hl != 0u, "shape %u size %u: payload %u not coded", k,
                  sizes[s], nl);
            if (hl == 0u) {
                continue;
            }
            /* Every shape but noise has to get smaller once the payload is
             * big enough for the 12-byte header not to dominate it; noise is
             * the case the block encoder's gate exists for. */
            if (k != 3u && nl >= 512u) {
                CHECK(hl < nl, "shape %u size %u: %u -> %u bytes", k, sizes[s],
                      nl, hl);
            }
            rc = zpse_huf_decode(huf1, hl, back, (uint32_t)sizeof back, &ilen,
                                 &iform);
            CHECK(rc == ZPSE_OK, "shape %u size %u: decode rc %d", k, sizes[s],
                  rc);
            CHECK(ilen == nl && iform == (uint8_t)ZPSE_B_LZ,
                  "shape %u: inner %u/%u form %u", k, ilen, nl,
                  (unsigned)iform);
            CHECK(memcmp(back, inner, nl) == 0,
                  "shape %u size %u: payload differs", k, sizes[s]);

            /* Twice is the same bytes: a block's output is a function of the
             * block, so no state may leak into the choice of table. */
            (void)zpse_huf_encode(inner, nl, huf2, (uint32_t)sizeof huf2,
                                  (uint8_t)ZPSE_B_LZ);
            CHECK(memcmp(huf1, huf2, hl) == 0,
                  "shape %u size %u: not reproducible", k, sizes[s]);
        }
    }
}

static void refusals(void)
{
    uint32_t nl = make_inner(0, 4096u);
    uint32_t hl;
    uint32_t t;
    uint32_t ilen = 0u;
    uint8_t iform = 0u;

    hl = zpse_huf_encode(inner, nl, huf1, (uint32_t)sizeof huf1,
                         (uint8_t)ZPSE_B_LZ);
    CHECK(hl != 0u, "refusal baseline did not encode");
    if (hl == 0u) {
        return;
    }

    /* Below ZPSE_HUF_MIN_INNER there is nothing to code. */
    CHECK(zpse_huf_encode(inner, (uint32_t)ZPSE_HUF_MIN_INNER - 1u, huf2,
                          (uint32_t)sizeof huf2,
                          (uint8_t)ZPSE_B_LZ) == 0u,
          "a payload under the minimum was coded");
    /* A form the decoder does not know is not written. */
    CHECK(zpse_huf_encode(inner, nl, huf2, (uint32_t)sizeof huf2,
                          (uint8_t)ZPSE_B_LZ_TILE) == 0u,
          "the tile form was accepted as an inner form");
    /* A destination that cannot hold the payload is declined. */
    CHECK(zpse_huf_encode(inner, nl, huf2, 16u, (uint8_t)ZPSE_B_LZ) == 0u,
          "an undersized destination was used");

    /* Every truncation is refused. */
    for (t = 1; t < hl; t++) {
        int rc = zpse_huf_decode(huf1, t, back, (uint32_t)sizeof back, &ilen,
                                 &iform);

        if (rc == ZPSE_OK) {
            CHECK(0, "truncation at %u of %u was accepted", t, hl);
            break;
        }
    }

    /* Every single-byte corruption either is refused or decodes to bytes
     * that are not the payload. Nothing may crash. */
    for (t = 0; t < hl; t++) {
        int rc;

        memcpy(work, huf1, hl);
        work[t] ^= 0xFF;
        rc = zpse_huf_decode(work, hl, back, (uint32_t)sizeof back, &ilen,
                             &iform);
        if (rc == ZPSE_OK) {
            CHECK(ilen != nl || memcmp(back, inner, nl) != 0,
                  "corruption at %u decoded to the original payload", t);
        } else {
            CHECK(rc == ZPSE_BAD_HUF || rc == ZPSE_BAD_HEADER ||
                      rc == ZPSE_BAD_BLOCK_SIZE || rc == ZPSE_OUTPUT_OVERFLOW,
                  "corruption at %u returned %d", t, rc);
        }
    }
}

/* Encode one block at a level and report the form its flags name. */
static int encode_form(const uint8_t *p, uint32_t n, uint32_t flags,
                       uint32_t *clen, uint8_t *flags_out)
{
    int rc = zpse_compress_block_ex(p, n, comp, (uint32_t)sizeof comp, clen,
                                    0u, 1, flags, ea);

    if (rc != ZPSE_OK) {
        return rc;
    }
    /* A page's block takes the 16-byte header, whose flags are one byte
     * earlier -- reading a fixed offset would have this test checking the
     * transform id. */
    *flags_out = comp[zpse_block_header_size(comp, *clen) ==
                              (uint32_t)ZPSE_BLOCK_HEADER_SIZE
                          ? 3u
                          : 2u];
    return ZPSE_OK;
}

#define IS_HUF(fl) (((fl) & (uint8_t)ZPSE_B_HUF) == (uint8_t)ZPSE_B_HUF)

static void block_form(void)
{
    uint32_t clen = 0u;
    uint8_t fl = 0u;
    uint32_t olen = 0u;
    uint32_t n = 4096u;
    uint32_t si;
    uint32_t seed = 0u;
    int kind;
    int found_kind = 0;
    int found = 0;
    uint32_t saved_len = 0u;

    /* Which block the deepest level takes the new form on is a property of
     * the finder, not of this test, so the shared corpus is swept for one
     * rather than a shape being invented for it. corpus_gen.h is the
     * generator tests/test_sizes.c pins, so this search is deterministic and
     * cannot drift away from the corpus those sizes measure. */
    for (kind = 0; kind < ZPSE_CORPUS_KINDS && !found; kind++) {
        for (si = 0; si < (uint32_t)ZPSE_CORPUS_SEEDS && !found; si++) {
            uint32_t sd = corpus_seed((int)si, kind, n);

            corpus_gen(kind, n, sd, plain);
            CHECK(encode_form(plain, n, (uint32_t)ZPSE_ENC_SPEED_MAX, &clen,
                              &fl) == ZPSE_OK,
                  "corpus kind %d did not encode", kind);
            if (IS_HUF(fl)) {
                found = 1;
                found_kind = kind;
                seed = sd;
                memcpy(work, comp, clen);
                saved_len = clen;
            }
        }
    }
    CHECK(found, "no corpus block at the deepest level took the new form");
    if (!found) {
        return;
    }
    corpus_gen(found_kind, n, seed, plain);
    CHECK((fl & (uint8_t)ZPSE_B_RAW) == 0u, "a Huffman block is also RAW");

    {
        ZpseBlockHeader h;

        CHECK(zpse_block_header_parse(&h, work, saved_len) == ZPSE_OK,
              "Huffman block header does not parse");
        /* The envelope is the LZ forms' own: the payload names its inner
         * form, so the header has no room for a transform or a directory. */
        CHECK(h.transform_id == 0u && h.segment_count == 1u &&
                  h.transform_meta_size == 0u &&
                  (h.block_flags & (uint8_t)ZPSE_B_HAS_META) == 0u,
              "Huffman block carries a transform (tid %u, sc %u, meta %u)",
              (unsigned)h.transform_id, (unsigned)h.segment_count,
              (unsigned)h.transform_meta_size);
        CHECK(h.uncompressed_size == n, "Huffman block size word is %u",
              (unsigned)h.uncompressed_size);
        /* What it wraps is named in the payload, and it has to be one of the
         * three LZ forms -- never RAW and never a tile body. */
        {
            uint32_t hdr = zpse_block_header_size(work, saved_len);
            uint8_t iform = work[hdr + 1u];

            CHECK(iform == (uint8_t)ZPSE_B_LZ ||
                      iform == (uint8_t)ZPSE_B_LZ2 ||
                      iform == (uint8_t)ZPSE_B_LZ_HEX,
                  "inner form %02x is not an LZ form", (unsigned)iform);
        }
    }

    CHECK(zpse_decompress_block(work, saved_len, back,
                                (uint32_t)sizeof back, &olen, da) == ZPSE_OK,
          "Huffman block did not decode");
    CHECK(olen == n && memcmp(back, plain, n) == 0,
          "Huffman block did not round-trip");

    /* Byte-identical on a second encode. */
    CHECK(encode_form(plain, n, (uint32_t)ZPSE_ENC_SPEED_MAX, &clen, &fl) ==
              ZPSE_OK,
          "second encode failed");
    CHECK(clen == saved_len && memcmp(work, comp, clen) == 0,
          "the same block encoded to different bytes (%u then %u)", saved_len,
          clen);

    /* The option asks for it by itself, below the level that offers it. */
    CHECK(encode_form(plain, n,
                      (uint32_t)ZPSE_ENC_SPEED_L6 | (uint32_t)ZPSE_ENC_HUF,
                      &clen, &fl) == ZPSE_OK,
          "ZPSE_ENC_HUF at level 6 was refused");
    CHECK(IS_HUF(fl), "ZPSE_ENC_HUF did not write the form (flags %02x)",
          (unsigned)fl);

    /* And a level that did not ask for it never writes one. */
    CHECK(encode_form(plain, n, (uint32_t)ZPSE_ENC_SPEED_BALANCED, &clen,
                      &fl) == ZPSE_OK,
          "level 5 encode failed");
    CHECK(!IS_HUF(fl), "level 5 wrote the Huffman form (flags %02x)",
          (unsigned)fl);

    /* The payload names its own inner form, and a form the decoder does not
     * know is refused -- a Huffman block cannot be pointed at the tile path,
     * at RAW, or at itself. (The other half of this -- the flag alone over a
     * payload that is not one -- needs the header CRC re-sealing, so it lives
     * in test_hostile.c beside the helper that does it.) */
    {
        static const uint8_t bad[] = { 0x00u, 0x01u, 0x10u, 0x40u, 0xC0u,
                                       0xFFu };
        uint32_t hdr = zpse_block_header_size(work, saved_len);
        uint32_t b;

        for (b = 0; b < (uint32_t)(sizeof bad / sizeof bad[0]); b++) {
            memcpy(comp, work, saved_len);
            comp[hdr + 1u] = bad[b];
            CHECK(zpse_decompress_block(comp, saved_len, back,
                                        (uint32_t)sizeof back, &olen,
                                        da) < 0,
                  "inner form %02x was accepted", (unsigned)bad[b]);
        }
    }
}

int main(void)
{
    codec_roundtrip();
    refusals();
    block_form();
    printf("HUF: checks=%d fails=%d\n", checks, fails);
    if (fails != 0) {
        printf("HUF FAILED\n");
        return 1;
    }
    printf("HUF OK\n");
    return 0;
}
