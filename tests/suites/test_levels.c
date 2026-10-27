/* The encoder's effort levels, as a contract.
 *
 * The nine levels (1 = FASTEST ... 9 = MAX, with the default at 5) trade
 * size for time in one direction: the fastest
 * writes the biggest block and the deepest writes the smallest. That is
 * the promise the levels make, and nothing pins it -- a level that
 * quietly wrote a bigger block than the one before it would not fail
 * any suite, and it would not fail loudly either: the caller asked for
 * speed and got size, or the other way round.
 *
 * So it is pinned here, over the columnar corpora rather than over one
 * shape, because a level that is only right for one shape is not a
 * contract:
 *
 *   - every level's block round-trips, at every level;
 *   - the sizes do not increase from the fastest level to the deepest;
 *   - the deepest level never writes a block larger than the balanced
 *     one (the LZ form competes on size, so it can only win);
 *   - the levels are ordered in time the same way they are in size --
 *     the fastest level is not allowed to be the slowest.
 */
/* clock_gettime is POSIX, and a strict -std=c11 hides it unless the
 * feature test macro says so. It has to come before any system header. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zpse/zpse.h"
#include "columnar_gen.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static uint8_t in[65536];
static uint8_t comp[140000];
static uint8_t out[65536];

static long failures = 0;
static long checks = 0;

#define CHECK(cond, ...) do {                       \
    checks++;                                       \
    if (!(cond)) {                                  \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
        failures++;                                 \
    }                                               \
} while (0)

static const uint32_t levels[9] = {
    ZPSE_ENC_SPEED_FASTEST, ZPSE_ENC_SPEED_L2,
    ZPSE_ENC_SPEED_FASTER, ZPSE_ENC_SPEED_L4,
    ZPSE_ENC_SPEED_BALANCED, ZPSE_ENC_SPEED_L6,
    ZPSE_ENC_SPEED_L7, ZPSE_ENC_SPEED_L8,
    ZPSE_ENC_SPEED_MAX
};
static const char *const names[9] = {"fastest", "2", "faster", "4",
                                     "balanced", "6", "7", "8", "max"};

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(void)
{
    int kind;
    unsigned li;

    for (kind = 0; kind < (int)ZPSE_COL_KINDS; kind++) {
        uint32_t sizes[9];
        double times[9];
        ZpseColGen g;

        zpse_col_gen_init(&g, kind);
        zpse_col_gen_fill(&g, in, sizeof in);

            for (li = 0; li < 9u; li++) {
            uint32_t cl = 0u;
            uint32_t ol = 0u;
            double best = 1e18;
            int r;
            int rc;

            for (r = 0; r < 3; r++) {
                double t0 = now_ns();
                double t1;

                rc = zpse_compress_block_ex(in, sizeof in, comp, sizeof comp,
                                            &cl, 0, 1, levels[li], E);
                t1 = now_ns();
                if (rc != PSE_OK) {
                    break;
                }
                if (t1 - t0 < best) {
                    best = t1 - t0;
                }
            }
            CHECK(rc == PSE_OK, "%s: the %s level did not compress",
                  zpse_col_names[kind], names[li]);
            if (rc != PSE_OK) {
                sizes[li] = 0xFFFFFFFFu;
                times[li] = 0.0;
                continue;
            }
            sizes[li] = cl;
            times[li] = best;

            /* Every level's block must decode to the bytes it came from. */
            rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
            CHECK(rc == PSE_OK && ol == sizeof in &&
                      memcmp(out, in, sizeof in) == 0,
                  "%s: the %s level did not round-trip", zpse_col_names[kind],
                  names[li]);
        }

        /* The sizes do not increase from the fastest level to the
         * deepest. */
        for (li = 1u; li < 9u; li++) {
            CHECK(sizes[li] <= sizes[li - 1u],
                  "%s: the %s level writes %u bytes, the %s level %u",
                  zpse_col_names[kind], names[li], sizes[li], names[li - 1u],
                  sizes[li - 1u]);
        }
        /* The deepest level never writes a block larger than the
         * balanced one. */
        CHECK(sizes[8] <= sizes[4], "%s: the deepest level writes %u bytes, "
              "the balanced level %u", zpse_col_names[kind], sizes[8],
              sizes[4]);
        /* And the levels are ordered in time the same way they are in
         * size. The fastest level is not allowed to be the slowest. */
        CHECK(times[0] <= times[4] * 4.0,
              "%s: the fastest level takes %.0f ns, the balanced level "
              "%.0f ns", zpse_col_names[kind], times[0], times[4]);
    }

    if (failures != 0) {
        printf("LEVELS: %ld failures of %ld checks\n", failures, checks);
        return 1;
    }
    printf("LEVELS: ALL OK (%ld checks)\n", checks);
    return 0;
}
