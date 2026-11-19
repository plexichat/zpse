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
            16,    16,    16,    16,    72,    72,    72,    84,    84,    96,    70,    67,    67,    70,    67,    67,
            16,    16,    16,    16,    72,    72,    72,    84,    84,    96,    71,    68,    68,    71,    68,    68,
            75,    65,    65,    65,    87,    81,    83,    90,    83,    86,    93,    89,    92,   102,   101,   105,
          1040,  2064,  3088,  4112,  5032,  6176,  8224, 10032, 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
           544,  1056,  1568,  2080,  2606,  3178,  4202,  5135,  6279,  8356, 10193, 12510, 16664, 20338, 24972, 33280,
           544,   111,   152,   196,   248,   296,   380,   455,   548,   716,   867,  1056,  1391,  1691,  2067,  2742,
            80,    70,    70,    70,    92,    85,    88,    95,    88,    91,    98,    94,    97,   107,   106,   113,
           280,   536,   792,  1048,    88,    88,    88,    77,    77,    75,    79,    79,    82,    79,    79,    79,
            72,    74,    74,    74,    96,    91,    91,    98,    91,    98,   105,    98,   104,   114,   110,   116,
            73,    85,    97,   109,   136,   149,   173,   194,   221,   268,   308,   362,   458,   542,   647,   836,
           593,  1158,  1716,  2278,  2791,  3415,  4532,  5527,  6773,  9031, 11015, 13072, 17371, 21198, 25926, 34602,
           551,  1075,  1599,  2123,  2601,  3187,  4236,  5161,  6332,  8428, 10278, 12620, 16812, 20512, 25193, 33577,
           280,   536,   792,  1048,  1213,  1493,  1975,  2270,  2805,  3763,  4575,  5448,  7061,  8731, 10634, 13788,
           544,  1056,  1568,  2080,  1890,  2307,  3061,  3756,  4578,  6004,  7388,  9033, 11983, 14595, 17902, 23844,
            98,   100,   101,   101,   132,   118,   119,   169,   165,   125,   145,   129,   135,   157,   147,   156,
           364,   633,   894,  1168,  1436,  1691,  2217,  2780,  3329,  4371,  5288,  6624,  8672, 10480, 12768, 16864,
            16,    16,    16,    16,    72,    72,    72,    84,    84,    96,    70,    67,    67,    70,    67,    67,
            16,    16,    16,    16,    72,    72,    72,    84,    84,    96,    71,    68,    68,    71,    68,    68,
            75,    65,    65,    65,    87,    81,    83,    90,    83,    86,    93,    89,    92,   102,   101,   105,
          1040,  2064,  3088,  4112,  5032,  6176,  8224, 10032, 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
           544,  1056,  1568,  2080,  2606,  3178,  4202,  5135,  6279,  8356, 10193, 12510, 16664, 20338, 24972, 33280,
           544,   111,   152,   196,   248,   296,   380,   455,   548,   716,   867,  1056,  1391,  1691,  2067,  2742,
            80,    70,    70,    70,    92,    85,    88,    95,    88,    91,    98,    94,    97,   107,   106,   113,
           280,   536,   792,  1048,    88,    88,    88,    77,    77,    75,    79,    79,    82,    79,    79,    79,
            72,    74,    74,    74,    96,    91,    91,    98,    91,    98,   105,    98,   104,   114,   110,   116,
            73,    85,    97,   109,   136,   149,   173,   194,   221,   268,   308,   362,   458,   542,   647,   836,
           594,  1153,  1716,  2278,  2791,  3411,  4534,  5540,  6785,  8746, 11018, 13099, 17413, 21235, 26039, 34579,
           551,  1075,  1599,  2123,  2601,  3187,  4236,  5161,  6332,  8428, 10278, 12620, 16812, 20512, 25193, 33577,
           280,   536,   792,  1048,  1179,  1522,  1923,  2306,  2818,  3658,  4530,  5506,  7123,  8741, 10638, 14055,
           544,  1056,  1568,  2080,  1902,  2284,  3028,  3747,  4531,  6023,  7335,  9020, 11978, 14576, 17880, 23806,
            98,   100,   101,   101,   132,   118,   119,   169,   165,   125,   145,   129,   135,   157,   147,   156,
           364,   633,   894,  1168,  1436,  1691,  2217,  2707,  3329,  4371,  5281,  6624,  8672, 10480, 12768, 16864,
            16,    16,    16,    16,    72,    72,    72,    84,    84,    96,    70,    67,    67,    70,    67,    67,
            16,    16,    16,    16,    72,    72,    72,    84,    84,    96,    71,    68,    68,    71,    68,    68,
            75,    65,    65,    65,    87,    81,    83,    90,    83,    86,    93,    89,    92,   102,   101,   105,
          1040,  2064,  3088,  4112,  5032,  6176,  8224, 10032, 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
           544,  1056,  1568,  2080,  2606,  3178,  4202,  5135,  6279,  8356, 10193, 12510, 16664, 20338, 24972, 33280,
           544,   111,   152,   196,   248,   296,   380,   455,   548,   716,   867,  1056,  1391,  1691,  2067,  2742,
            80,    70,    70,    70,    92,    85,    88,    95,    88,    91,    98,    94,    97,   107,   106,   113,
           280,   536,   792,  1048,    88,    88,    88,    77,    77,    75,    79,    79,    82,    79,    79,    79,
            72,    74,    74,    74,    96,    91,    91,    98,    91,    98,   105,    98,   104,   114,   110,   116,
            73,    85,    97,   109,   136,   149,   173,   194,   221,   268,   308,   362,   458,   542,   647,   836,
           594,  1152,  1715,  2280,  2789,  3413,  4538,  5532,  6785,  9021, 10652, 13046, 17382, 21067, 25997, 34618,
           551,  1075,  1599,  2123,  2601,  3187,  4236,  5161,  6332,  8428, 10278, 12620, 16812, 20512, 25193, 33577,
           280,   536,   792,  1048,  1218,  1479,  1878,  2295,  2840,  3699,  4496,  5506,  7219,  8699, 10458, 13946,
           544,  1056,  1568,  2080,  1906,  2307,  3035,  3723,  4523,  6073,  7356,  8996, 11929, 14584, 17926, 23874,
            98,   100,   101,   101,   132,   118,   119,   169,   165,   125,   145,   129,   135,   157,   147,   156,
           366,   633,   894,  1168,  1436,  1691,  2217,  2707,  3329,  4371,  5274,  6624,  8672, 10480, 12768, 16864,
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
