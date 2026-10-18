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
      105,   145,   185,   225,   265,   305,   385,   465,
      545,   705,   855,  1025,  1345,  1635,  1985,  2625,
       98,   138,   178,   218,   258,   298,   378,   458,
      608,   608,   608,   608,   608,   608,   608,   608,
      249,   441,   633,   825,  1002,  1209,  1696,  1943,
     2336,  2976,  3616,  4256,  5536,  6816,  8096, 10656,
     1056,  2080,  3104,  4128,  5032,  6176,   830,   990,
      990,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
      669,  1276,  1889,  2499,  3053,  3721,  4945,  6047,
     7391,  9829, 12007, 14709, 19601, 23955, 29382, 39152,
      666,  1277,  1889,  2496,  3067,  3721,  8224,  6597,
     8001, 16416, 12606, 24608, 19624, 23930, 29373, 39134,
      328,   584,   840,  1167,  1338,  1624,  2136,  2604,
     3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
      508,   960,  1424,  1864,  2277,  2776,  3660,  4486,
     5468,  7320,  8937, 10912, 14544, 17815, 21820, 29064,
      306,   570,   836,  1109,  1366,  1661,  2213,  2714,
     3317,  3840,  4656,  5552,  7264,  8880, 10688, 14112,
      735,  1140,  1428,  1715,  2030,  2349,  2918,  3467,
     4099,  5282,  6327,  7649, 10008, 12107, 14741, 19463,
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
      105,   145,   185,   225,   265,   305,   385,   465,
      545,   705,   855,  1025,  1345,  1635,  1985,  2625,
       98,   138,   178,   218,   258,   298,   378,   458,
      608,   608,   608,   608,   608,   608,   608,   608,
      249,   441,   633,   825,  1002,  1209,  1696,  1943,
     2336,  2976,  3616,  4256,  5536,  6816,  8096, 10656,
     1056,  2080,  3104,  4128,  5032,  6176,   830,   990,
      990,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
      668,  1281,  1890,  2500,  3050,  3722,  4948,  6044,
     7387,  9833, 12008, 14718, 19608, 23952, 29363, 39143,
      667,  1276,  1890,  2498,  3067,  3723,  8224,  6584,
     7996, 16416, 12605, 24608, 19615, 23921, 29383, 39162,
      323,   584,   840,  1096,  1338,  1624,  2136,  2604,
     3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
      512,   964,  1428,  1852,  2273,  2780,  3668,  4512,
     5480,  7296,  8925, 10932, 14548, 17775, 21804, 29056,
      306,   570,   836,  1109,  1366,  1661,  2213,  2714,
     3317,  3840,  4656,  5552,  7264,  8880, 10688, 14112,
      735,  1143,  1429,  1712,  2021,  2346,  2919,  3454,
     4094,  5287,  6343,  7648, 10011, 12115, 14733, 19455,
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
      105,   145,   185,   225,   265,   305,   385,   465,
      545,   705,   855,  1025,  1345,  1635,  1985,  2625,
       98,   138,   178,   218,   258,   298,   378,   458,
      608,   608,   608,   608,   608,   608,   608,   608,
      249,   441,   633,   825,  1002,  1209,  1696,  1943,
     2336,  2976,  3616,  4256,  5536,  6816,  8096, 10656,
     1056,  2080,  3104,  4128,  5032,  6176,   830,   990,
      990,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
      668,  1280,  1890,  2498,  3052,  3718,  4944,  6050,
     7383,  9832, 12019, 14711, 19594, 23942, 29389, 39134,
      667,  1278,  1889,  2501,  3068,  3720,  8224,  6595,
     7996, 16416, 12602, 24608, 19613, 23920, 29383, 39136,
      326,   584,   840,  1096,  1338,  1624,  2136,  2604,
     3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
      512,   964,  1420,  1864,  2285,  2784,  3672,  4488,
     5512,  7292,  8929, 10932, 14580, 17803, 21788, 29076,
      306,   570,   836,  1109,  1366,  1661,  2213,  2714,
     3317,  3840,  4656,  5552,  7264,  8880, 10688, 14112,
      735,  1143,  1425,  1713,  2030,  2348,  2920,  3461,
     4104,  5281,  6340,  7654,  9998, 12109, 14741, 19466,
};

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
