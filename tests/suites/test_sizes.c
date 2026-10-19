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
       68,    68,    68,    68,    80,    80,    80,    92,    92,   104,   116,   128,   152,   176,   200,   248,
       68,    68,    68,    68,    80,    80,    80,    92,    92,   104,   116,   128,   152,   176,   200,   248,
       99,   139,   179,   219,   259,   299,   379,   459,   539,   699,   849,  1019,  1339,  1629,  1979,  2619,
     1056,  2080,  3104,  4128,  5032,  6176,  8224, 10032, 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
      597,  1109,  1621,  2133,  2614,  3186,  4210,  5143,  6287,  8364, 10201, 12518, 16672, 20346, 24980, 33288,
      116,   178,   238,   300,   358,   422,   544,   662,   788,  1032,  1258,  1522,  2008,  2448,  2986,  3962,
      104,   144,   184,   224,   264,   304,   384,   464,   544,   704,   854,  1024,  1344,  1634,  1984,  2624,
       98,   138,   178,   218,   258,   298,   378,   458,   608,   608,   608,   608,   608,   608,   608,   608,
      249,   441,   633,   825,  1002,  1209,  1312,  1760,  1760,  2208,  2656,  3104,  4000,  4896,  5792,  7584,
     1056,  2080,  3104,  4128,  5032,  6176,   830,   990,   990,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
      617,  1182,  1740,  2302,  2799,  3423,  4540,  5535,  6781,  9046, 11023, 13531, 18002, 21977, 26974, 35965,
      616,  1175,  1738,  2304,  2803,  3421,  8224,  6007,  7167, 16416, 11410, 24608, 17990, 21978, 26990, 35979,
      328,   584,   840,  1167,  1338,  1624,  2136,  2604,  3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
      508,   960,  1424,  1864,  2277,  2776,  3660,  4486,  5468,  7320,  8937, 10912, 14544, 17815, 21820, 29064,
      287,   551,   815,  1079,  1325,  1607,  2213,  2714,  3317,  3456,  4176,  4976,  6496,  7920,  9536, 12576,
      735,  1100,  1356,  1613,  1918,  2203,  2721,  3225,  3803,  4884,  5851,  7042,  9213, 11135, 13540, 17871,
       68,    68,    68,    68,    80,    80,    80,    92,    92,   104,   116,   128,   152,   176,   200,   248,
       68,    68,    68,    68,    80,    80,    80,    92,    92,   104,   116,   128,   152,   176,   200,   248,
       99,   139,   179,   219,   259,   299,   379,   459,   539,   699,   849,  1019,  1339,  1629,  1979,  2619,
     1056,  2080,  3104,  4128,  5032,  6176,  8224, 10032, 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
      597,  1109,  1621,  2133,  2614,  3186,  4210,  5143,  6287,  8364, 10201, 12518, 16672, 20346, 24980, 33288,
      116,   178,   238,   300,   358,   422,   544,   662,   788,  1032,  1258,  1522,  2008,  2448,  2986,  3962,
      104,   144,   184,   224,   264,   304,   384,   464,   544,   704,   854,  1024,  1344,  1634,  1984,  2624,
       98,   138,   178,   218,   258,   298,   378,   458,   608,   608,   608,   608,   608,   608,   608,   608,
      249,   441,   633,   825,  1002,  1209,  1312,  1760,  1760,  2208,  2656,  3104,  4000,  4896,  5792,  7584,
     1056,  2080,  3104,  4128,  5032,  6176,   830,   990,   990,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
      618,  1177,  1740,  2302,  2799,  3419,  4543,  5549,  6793,  9041, 11026, 13526, 18015, 21985, 26992, 35973,
      617,  1178,  1738,  2296,  2802,  3421,  8224,  6015,  7171, 16416, 11408, 24608, 17999, 21986, 26976, 35937,
      323,   584,   840,  1096,  1338,  1624,  2136,  2604,  3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
      512,   964,  1428,  1852,  2273,  2780,  3668,  4512,  5480,  7296,  8925, 10932, 14548, 17775, 21804, 29056,
      287,   551,   815,  1079,  1325,  1607,  2213,  2714,  3317,  3456,  4176,  4976,  6496,  7920,  9536, 12576,
      735,  1097,  1357,  1616,  1915,  2206,  2721,  3219,  3810,  4882,  5839,  7046,  9213, 11121, 13547, 17879,
       68,    68,    68,    68,    80,    80,    80,    92,    92,   104,   116,   128,   152,   176,   200,   248,
       68,    68,    68,    68,    80,    80,    80,    92,    92,   104,   116,   128,   152,   176,   200,   248,
       99,   139,   179,   219,   259,   299,   379,   459,   539,   699,   849,  1019,  1339,  1629,  1979,  2619,
     1056,  2080,  3104,  4128,  5032,  6176,  8224, 10032, 12320, 16416, 20032, 24608, 32800, 40032, 49184, 65568,
      597,  1109,  1621,  2133,  2614,  3186,  4210,  5143,  6287,  8364, 10201, 12518, 16672, 20346, 24980, 33288,
      116,   178,   238,   300,   358,   422,   544,   662,   788,  1032,  1258,  1522,  2008,  2448,  2986,  3962,
      104,   144,   184,   224,   264,   304,   384,   464,   544,   704,   854,  1024,  1344,  1634,  1984,  2624,
       98,   138,   178,   218,   258,   298,   378,   458,   608,   608,   608,   608,   608,   608,   608,   608,
      249,   441,   633,   825,  1002,  1209,  1312,  1760,  1760,  2208,  2656,  3104,  4000,  4896,  5792,  7584,
     1056,  2080,  3104,  4128,  5032,  6176,   830,   990,   990,  1150,  1310,  1470,  1790,  2110,  2430,  3070,
      618,  1176,  1739,  2304,  2797,  3421,  4548,  5540,  6793,  9029, 11023, 13526, 18031, 21984, 26981, 35952,
      617,  1175,  1739,  2297,  2801,  3418,  8224,  6011,  7172, 16416, 11411, 24608, 17996, 21983, 26979, 35984,
      326,   584,   840,  1096,  1338,  1624,  2136,  2604,  3176,  4216,  5136,  6296,  8376, 10216, 12536, 16696,
      512,   964,  1420,  1864,  2285,  2784,  3672,  4488,  5512,  7292,  8929, 10932, 14580, 17803, 21788, 29076,
      287,   551,   815,  1079,  1325,  1607,  2213,  2714,  3317,  3456,  4176,  4976,  6496,  7920,  9536, 12576,
      735,  1099,  1361,  1615,  1916,  2202,  2720,  3230,  3796,  4887,  5840,  7044,  9223, 11134, 13541, 17864,
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
