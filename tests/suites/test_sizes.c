/* Size regression gate: no case of the ratio corpus may grow.
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
 * encoder wrote when the gate was added.
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
68,    68,    68,    68,    80,    80,    80,    92,
    92,   104,   116,   128,   152,   176,   200,   248,
    68,    68,    68,    68,    80,    80,    80,    92,
    92,   104,   116,   128,   152,   176,   200,   248,
    99,   139,   179,   219,   259,   299,   379,   459,
   539,   699,   849,  1019,  1339,  1629,  1979,  2619,
  1056,  2080,  3104,  4128,  5032,  6176,  8224, 10032,
 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
   597,  1109,  1621,  2133,  2614,  3186,  4210,  5143,
  6287,  8364, 10201, 12518, 16672, 20346, 24980, 33288,
   116,   178,   238,   300,   358,   422,   544,   662,
   788,  1032,  1258,  1522,  2008,  2448,  2986,  3962,
   104,   144,   184,   224,   264,   304,   384,   464,
   544,   704,   854,  1024,  1344,  1634,  1984,  2624,
    98,   104,   104,   104,   104,   104,   104,   128,
   128,   128,   152,   152,   176,   200,   224,   272,
   136,   176,   216,   256,   296,   336,   416,   496,
   576,   904,  1064,  1224,  1544,  1864,  2184,  2824,
  1056,  2080,  3104,  4128,  5032,  6176,   623,   703,
   783,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
   617,  1182,  1740,  2302,  2799,  3423,  4540,  5535,
  6781,  9039, 11023, 13519, 18002, 21977, 26974, 50891,
   616,  1175,  1738,  2304,  2803,  3421,  4572,  5570,
  6814,  9055, 11052, 13534, 17990, 21978, 26990, 35959,
   328,   584,   840,  1136,  1338,  1624,  2136,  2604,
  3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
   508,   960,  1424,  1864,  2277,  2776,  3660,  4486,
  5468,  7320,  8937, 10912, 14544, 17815, 21820, 29064,
   143,   203,   263,   323,   382,   443,   563,  1128,
  1308,  1644,  1952,  2316,  2988,  3660,  4392,  5736,
   444,   730,  1016,  1302,  1558,  1874,  2446,  2958,
  3680,  4704,  5608,  6752,  8800, 10608, 12896, 16992,
    68,    68,    68,    68,    80,    80,    80,    92,
    92,   104,   116,   128,   152,   176,   200,   248,
    68,    68,    68,    68,    80,    80,    80,    92,
    92,   104,   116,   128,   152,   176,   200,   248,
    99,   139,   179,   219,   259,   299,   379,   459,
   539,   699,   849,  1019,  1339,  1629,  1979,  2619,
  1056,  2080,  3104,  4128,  5032,  6176,  8224, 10032,
 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
   597,  1109,  1621,  2133,  2614,  3186,  4210,  5143,
  6287,  8364, 10201, 12518, 16672, 20346, 24980, 33288,
   116,   178,   238,   300,   358,   422,   544,   662,
   788,  1032,  1258,  1522,  2008,  2448,  2986,  3962,
   104,   144,   184,   224,   264,   304,   384,   464,
   544,   704,   854,  1024,  1344,  1634,  1984,  2624,
    98,   104,   104,   104,   104,   104,   104,   128,
   128,   128,   152,   152,   176,   200,   224,   272,
   136,   176,   216,   256,   296,   336,   416,   496,
   576,   904,  1064,  1224,  1544,  1864,  2184,  2824,
  1056,  2080,  3104,  4128,  5032,  6176,   623,   703,
   783,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
   618,  1177,  1740,  2302,  2799,  3419,  4542,  5548,
  6793,  9030, 11026, 13526, 18015, 21985, 26992, 35973,
   617,  1178,  1738,  2296,  2802,  3421,  4570,  5571,
  6812,  9050, 11044, 13545, 17997, 21986, 26976, 35937,
   323,   584,   840,  1096,  1338,  1624,  2136,  2604,
  3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
   512,   964,  1428,  1852,  2273,  2780,  3668,  4512,
  5480,  7296,  8925, 10932, 14548, 17775, 21804, 29056,
   143,   203,   263,   323,   382,   443,   563,  1128,
  1308,  1644,  1952,  2316,  2988,  3660,  4392,  5736,
   444,   730,  1016,  1302,  1558,  1874,  2446,  2958,
  3590,  4704,  5608,  6752,  8800, 10608, 12896, 16992,
    68,    68,    68,    68,    80,    80,    80,    92,
    92,   104,   116,   128,   152,   176,   200,   248,
    68,    68,    68,    68,    80,    80,    80,    92,
    92,   104,   116,   128,   152,   176,   200,   248,
    99,   139,   179,   219,   259,   299,   379,   459,
   539,   699,   849,  1019,  1339,  1629,  1979,  2619,
  1056,  2080,  3104,  4128,  5032,  6176,  8224, 10032,
 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
   597,  1109,  1621,  2133,  2614,  3186,  4210,  5143,
  6287,  8364, 10201, 12518, 16672, 20346, 24980, 33288,
   116,   178,   238,   300,   358,   422,   544,   662,
   788,  1032,  1258,  1522,  2008,  2448,  2986,  3962,
   104,   144,   184,   224,   264,   304,   384,   464,
   544,   704,   854,  1024,  1344,  1634,  1984,  2624,
    98,   104,   104,   104,   104,   104,   104,   128,
   128,   128,   152,   152,   176,   200,   224,   272,
   136,   176,   216,   256,   296,   336,   416,   496,
   576,   904,  1064,  1224,  1544,  1864,  2184,  2824,
  1056,  2080,  3104,  4128,  5032,  6176,   623,   703,
   783,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
   618,  1176,  1739,  2304,  2797,  3421,  4546,  5540,
  6793,  9029, 11018, 13512, 18031, 31113, 26981, 35952,
   617,  1175,  1739,  2297,  2801,  3418,  4567,  5570,
  6819,  9052, 11059, 13543, 17996, 21968, 26979, 35952,
   326,   584,   840,  1096,  1338,  1624,  2136,  2604,
  3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
   512,   964,  1420,  1864,  2285,  2784,  3672,  4488,
  5512,  7292,  8929, 10932, 14580, 17803, 21788, 29076,
   143,   203,   263,   323,   382,   443,   563,  1128,
  1308,  1644,  1952,  2316,  2988,  3660,  4392,  5736,
   444,   730,  1016,  1302,  1558,  1874,  2446,  2958,
  3590,  4704,  5608,  6752,  8800, 10608, 12896, 16992,};

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
