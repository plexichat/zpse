/* Size regression gate: no case of the ratio corpus may grow.
 *
 * The LZ-first stage in zpse_block_encode is the reason this stayed at zero:
 * an early version of it skipped the tile sweep on any full-size block the
 * screen convicted, and nine ramps and sawtooths grew by 554 bytes each (802
 * against 248) while a block of zeros went from 69 to 802. The rule now
 * refuses to skip the sweep when the finder's own stream is under 1/64 of the
 * block, which is exactly the set of blocks the tiles own, so the contract
 * here is still "no case may grow" and this suite is what holds the trade
 * honest.
 *
 * tests/clen_corpus.c prints each case's compressed length so two builds
 * can be diffed; that catches a change only when someone runs the diff.
 * This suite pins the same 768 cases (3 seeds x 16 kinds x 16 sizes, from
 * the shared corpus in tests/corpus_gen.h) and fails when one gets
 * bigger, so a decision that silently stops choosing the form it used to
 * -- a pruned candidate, a priced form that no longer wins, a proxy that
 * mis-ranks -- is caught by the ordinary test run.
 *
 * Only GROWTH fails. A case that shrinks is counted in the summary
 * (shrank=N, delta=-N bytes) and does not fail the suite: an improvement
 * must not need a table update to pass, and a shrink that still
 * round-trips is not a regression. The pinned numbers are the sizes this
 * encoder wrote at the last intentional re-pin.
 *
 * The FLAT page form re-pinned this table: page-sized blocks now take a
 * 16-byte token for a constant page and fixed-width NIBBLE4/BIT2 payloads
 * for irregular low-alphabet pages, so 705 of the 768 cases shrank. The 27
 * that grew are structured low-alphabet pages (alternation, stride records)
 * where FLAT's fixed-width packing is larger than the old ladder's DELTA
 * form; they are re-pinned here as the cost of the one-pass SIMD decode.
 *
 * Every case is also decoded and byte-compared, so a size that stays the
 * same while the payload becomes wrong still fails. Links
 * src/zpse_unity.c; exits non-zero on any growth or mismatch. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "zpse/zpse.h"

#include "corpus_gen.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[ZPSE_MAX_BLOCK_SIZE];
static uint8_t comp[ZPSE_MAX_BLOCK_SIZE * 2];
static uint8_t out[ZPSE_MAX_BLOCK_SIZE];

/* case = (seed * KINDS + kind) * NSIZES + size_index. The numbers are the
 * compressed length of that block, eight per line in case order, so an
 * intentional change re-pins the values it moved and nothing else. */
static const uint32_t pinned[ZPSE_CORPUS_CASES] = {
   16,   16,   16,   16,   70,   72,   72,   84,   84,   96,   80,   78,   78,   80,   78,   78,
   16,   16,   16,   16,   70,   72,   72,   84,   84,   96,   81,   79,   79,   81,   79,   79,
   75,   73,   73,   73,   93,   89,   91,   94,   91,   92,   95,   93,   94,   98,   97,   99,
 1040, 2064, 3088, 4112, 5032, 6176, 8224,10032,12320,16416,20032,24608,32800,40032,49184,65568,
  544, 1056, 1568, 2080, 2606, 3178, 4202, 5135, 6279, 8356,10193,12510,16664,20338,24972,33280,
  544,  141,  190,  244,  306,  366,  471,  565,  681,  891, 1080, 1316, 1364, 1393, 1428, 1493,
   80,   78,   78,   78,   98,   93,   96,   99,   96,   97,  100,   98,   99,  103,  102,  104,
  280,  536,  792, 1048,   71,   75,   84,   96,   96,   95,   96,   96,   95,   97,   97,   97,
   83,   88,   88,   88,  107,  104,  104,  107,  104,  107,  110,  107,  109,  113,  111,  113,
   79,   83,   87,   91,  111,  115,  123,  130,  139,  155,  170,  187,  220,  248,  284,  289,
  553, 1071, 1588, 2106, 2791, 3415, 4532, 5527, 6529, 8695,10604,12960,17241,21020,25727,34331,
  545, 1061, 1577, 2093, 2565, 3141, 4174, 5085, 6239, 8303,10125,12431,16559,20203,24815,33071,
  280,  536,  792, 1048, 1234, 1524, 2000, 2301, 2835, 3814, 4611, 5513, 7128, 8788,10745,13956,
  544, 1056, 1568, 2080, 1963, 2379, 3166, 3818, 4696, 6111, 7502, 9153,12144,14802,18172,24247,
  111,  113,  113,  113,  146,  129,  130,  192,  188,  132,  146,  134,  136,  150,  140,  143,
  368,  646,  917, 1192, 1439, 1716, 2215, 2673, 3248, 4377, 5271, 6624, 8672,10480,12768,16864,
   16,   16,   16,   16,   70,   72,   72,   84,   84,   96,   80,   78,   78,   80,   78,   78,
   16,   16,   16,   16,   70,   72,   72,   84,   84,   96,   81,   79,   79,   81,   79,   79,
   75,   73,   73,   73,   93,   89,   91,   94,   91,   92,   95,   93,   94,   98,   97,   99,
 1040, 2064, 3088, 4112, 5032, 6176, 8224,10032,12320,16416,20032,24608,32800,40032,49184,65568,
  544, 1056, 1568, 2080, 2606, 3178, 4202, 5135, 6279, 8356,10193,12510,16664,20338,24972,33280,
  544,  141,  190,  244,  306,  366,  471,  565,  681,  891, 1080, 1316, 1364, 1393, 1428, 1493,
   80,   78,   78,   78,   98,   93,   96,   99,   96,   97,  100,   98,   99,  103,  102,  104,
  280,  536,  792, 1048,   71,   75,   84,   96,   96,   95,   96,   96,   95,   97,   97,   97,
   83,   88,   88,   88,  107,  104,  104,  107,  104,  107,  110,  107,  109,  113,  111,  113,
   79,   83,   87,   91,  111,  115,  123,  130,  139,  155,  170,  187,  220,  248,  284,  289,
  553, 1071, 1588, 2106, 2791, 3411, 4376, 5540, 6530, 8680,10612,12978,17211,21035,25760,34313,
  545, 1061, 1577, 2093, 2565, 3141, 4174, 5085, 6239, 8303,10125,12431,16559,20203,24815,33071,
  280,  536,  792, 1048, 1210, 1526, 1947, 2330, 2867, 3695, 4567, 5555, 7188, 8777,10710,14227,
  544, 1056, 1568, 2080, 1960, 2369, 3098, 3786, 4635, 6115, 7495, 9144,12107,14818,18179,24125,
  111,  113,  113,  113,  146,  129,  130,  192,  188,  132,  146,  134,  136,  150,  140,  143,
  368,  646,  917, 1192, 1428, 1716, 2215, 2751, 3248, 4364, 5260, 6624, 8672,10480,12768,16864,
   16,   16,   16,   16,   70,   72,   72,   84,   84,   96,   80,   78,   78,   80,   78,   78,
   16,   16,   16,   16,   70,   72,   72,   84,   84,   96,   81,   79,   79,   81,   79,   79,
   75,   73,   73,   73,   93,   89,   91,   94,   91,   92,   95,   93,   94,   98,   97,   99,
 1040, 2064, 3088, 4112, 5032, 6176, 8224,10032,12320,16416,20032,24608,32800,40032,49184,65568,
  544, 1056, 1568, 2080, 2606, 3178, 4202, 5135, 6279, 8356,10193,12510,16664,20338,24972,33280,
  544,  141,  190,  244,  306,  366,  471,  565,  681,  891, 1080, 1316, 1364, 1393, 1428, 1493,
   80,   78,   78,   78,   98,   93,   96,   99,   96,   97,  100,   98,   99,  103,  102,  104,
  280,  536,  792, 1048,   71,   75,   84,   96,   96,   95,   96,   96,   95,   97,   97,   97,
   83,   88,   88,   88,  107,  104,  104,  107,  104,  107,  110,  107,  109,  113,  111,  113,
   79,   83,   87,   91,  111,  115,  123,  130,  139,  155,  170,  187,  220,  248,  284,  289,
  553, 1071, 1588, 2106, 2789, 3306, 4372, 5364, 6785, 9021,10584,12932,17257,20971,25725,34337,
  545, 1061, 1577, 2093, 2565, 3141, 4174, 5085, 6239, 8303,10125,12431,16559,20203,24815,33071,
  280,  536,  792, 1048, 1251, 1523, 1910, 2315, 2864, 3731, 4590, 5554, 7288, 8740,10516,14088,
  544, 1056, 1568, 2080, 1959, 2392, 3115, 3813, 4608, 6174, 7497, 9166,12125,14878,18161,24162,
  111,  113,  113,  113,  146,  129,  130,  192,  188,  132,  146,  134,  136,  150,  140,  143,
  372,  646,  917, 1192, 1428, 1716, 2215, 2673, 3248, 4364, 5260, 6624, 8672,10480,12768,16864,
};

/* ---- The 64 KiB prose and run guards ---------------------------------
 *
 * The 768 cases above are synthetic shapes at sixteen sizes, and the one
 * class of block they do not model is a full-size page of prose. That is how
 * the 64 KiB Silesia total moved by +2.99 MB with every case in the table
 * unchanged: the loss was osdb +1.0 MB, ooffice +0.64, dickens +0.55, nci
 * +0.31 -- all text -- and the shapes below all stayed flat. A full-size text
 * block reaches the encoder through the ASCII gate and the tile-body stage
 * rather than through a generated shape, so nothing in the 768 could see it.
 *
 * Two blocks are guarded here instead. The prose case is the one the TEXT
 * page form is built for (word-level repetition, which is what a 64 KiB finder
 * exploits first); its ceiling is generous -- the block must at least beat
 * raw -- because its exact size moves with every text-path change, so pinning
 * a byte count here would only make each such change re-pin this file. The
 * run case is the one the tile-body stage wins, where tiles plus LZ write 116
 * bytes and the finder's own stream is 766 -- the exact pair that made the
 * deepest level write a larger block than the balanced one; that one stays an
 * absolute cap. Both are decoded and byte-compared, so a payload that keeps
 * its size and stops being right still fails. */
/* Generous: the prose block must at least beat raw. */
#define TEXT64_CEIL 65536u
#define RUNS64_PINNED 804u

static uint8_t guard_in[65536];

/* Deterministic prose from a 24-word vocabulary, a full stop every sentence.
 * No RNG: the pinned number is this exact byte stream. */
static void make_prose(uint32_t n)
{
    static const char *const w[24] = {
        "the", "of", "and", "a", "to", "in", "is", "that", "for",
        "it", "as", "was", "with", "be", "by", "on", "not", "he",
        "this", "are", "or", "his", "from", "at"
    };
    uint32_t at = 0u;
    uint32_t k = 0u;

    while (at < n) {
        const char *s = w[k % 24u];
        uint32_t j;

        for (j = 0; s[j] != '\0' && at < n; j++) {
            guard_in[at++] = (uint8_t)s[j];
        }
        if (at < n) {
            guard_in[at++] = (uint8_t)((k % 17u == 16u) ? '.' : ' ');
        }
        k++;
    }
}

/* Encode, decode and compare one guarded block; returns 0 when it is inside
 * its ceiling and correct. */
static int guard_case(uint32_t n, uint32_t pin, const char *name,
                      uint32_t *clen)
{
    uint32_t cl = 0u;
    uint32_t ol = 0u;
    int rc;

    rc = zpse_compress_block(guard_in, n, comp, sizeof comp, &cl, 0, 1, E);
    if (rc != 0) {
        printf("FAIL %s: compress rc=%d(%s)\n", name, rc,
               zpse_status_name(rc));
        return 1;
    }
    rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
    if (rc != 0 || ol != n || memcmp(guard_in, out, n) != 0) {
        printf("FAIL %s: roundtrip rc=%d olen=%u\n", name, rc, ol);
        return 1;
    }
    *clen = cl;
    if (cl > pin) {
        printf("FAIL %s: clen=%u grew from %u (+%u)\n", name, cl, pin,
               cl - pin);
        return 1;
    }
    return 0;
}

int main(void)
{
    long cases = 0;
    long grew = 0;
    long shrank = 0;
    long roundtrip_bad = 0;
    long long delta = 0;
    const int repin = (getenv("ZPSE_REPIN") != NULL);
    int seed;
    int kind;
    int s;

    for (seed = 0; seed < ZPSE_CORPUS_SEEDS; seed++) {
        for (kind = 0; kind < ZPSE_CORPUS_KINDS; kind++) {
            for (s = 0; s < ZPSE_CORPUS_NSIZES; s++) {
                uint32_t n = corpus_sizes[s];
                uint32_t cl = 0;
                uint32_t ol = 0;
                uint32_t want;
                long case_no = ((long)seed * ZPSE_CORPUS_KINDS + kind) *
                                   ZPSE_CORPUS_NSIZES + s;
                int rc;

                corpus_gen(kind, n, corpus_seed(seed, kind, n), in);
                rc = zpse_compress_block(in, n, comp, sizeof comp, &cl, 0, 1,
                                         E);
                if (rc != 0) {
                    printf("FAIL case=%ld kind=%d n=%u compress rc=%d(%s)\n",
                           case_no, kind, n, rc, zpse_status_name(rc));
                    grew++;
                    cases++;
                    continue;
                }
                rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
                if (rc != 0 || ol != n || memcmp(in, out, n) != 0) {
                    printf("FAIL case=%ld kind=%d n=%u roundtrip rc=%d olen=%u\n",
                           case_no, kind, n, rc, ol);
                    roundtrip_bad++;
                    cases++;
                    continue;
                }
                if (repin) {
                    printf("%5u,", cl);
                    if ((case_no + 1) % 16 == 0) {
                        putchar('\n');
                    }
                    cases++;
                    continue;
                }
                want = pinned[case_no];
                if (cl > want) {
                    printf("FAIL case=%ld kind=%2d n=%5u clen=%6u grew from %u"
                           " (+%u)\n",
                           case_no, kind, n, cl, want, cl - want);
                    grew++;
                    delta += (long long)cl - (long long)want;
                } else if (cl < want) {
                    shrank++;
                    delta += (long long)cl - (long long)want;
                }
                cases++;
            }
        }
    }

    /* The two full-size guards (see above). */
    {
        uint32_t ct = 0u;
        uint32_t cr = 0u;
        uint32_t i;
        int bad = 0;

        make_prose(TEXT64_CEIL);
        bad += guard_case(TEXT64_CEIL, TEXT64_CEIL, "text64", &ct);
        for (i = 0; i < 65536u; i++) {
            guard_in[i] = (uint8_t)(i < 32768u ? 0u : 0x33u);
        }
        bad += guard_case(65536u, RUNS64_PINNED, "runs64", &cr);
        printf("SIZES: guard text64=%u runs64=%u (text64 ceiling %u,"
               " runs64 pinned %u)\n", ct, cr, (unsigned)TEXT64_CEIL,
               (unsigned)RUNS64_PINNED);
        if (bad != 0) {
            grew += bad;
        }
    }

    printf("SIZES: cases=%ld grew=%ld shrank=%ld roundtrip_bad=%ld"
           " delta=%+lld bytes\n",
           cases, grew, shrank, roundtrip_bad, delta);
    if (grew != 0 || roundtrip_bad != 0) {
        printf("SIZES: FAILED (%ld growth(s), %ld roundtrip failure(s))\n",
               grew, roundtrip_bad);
        return 1;
    }
    printf("SIZES: ALL OK (no case grew)\n");
    return 0;
}
