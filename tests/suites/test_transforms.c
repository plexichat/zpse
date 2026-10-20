/* Every transform id against the scalar definition of the format.
 *
 * The encoder only ever emits ids 0 (RAW_BYTES), 1 (DELTA8) and 6
 * (TRANSPOSE_16), so the other nine transform shapes are reachable only
 * from streams another encoder produced -- and nothing in the tree tested
 * them: no suite called zpse_transform_forward or zpse_transform_inverse at
 * all. This is that test.
 *
 * The reference is computed here from the definition and never calls the
 * library: plane s holds the input bytes at positions s, s+w, s+2w, ...
 * (plane s one byte longer than plane s+1 when n is not a multiple of w),
 * and the delta (ids 1, 7-10) or xor (id 2) chain runs over each plane on
 * its own, seeded by seeds[s]. Both directions are checked, so a split that
 * agrees with its own merge but not with the format fails here.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "zpse/zpse.h"
#include "zpse/transform.h"

#define MAXN 65536u

static uint8_t in[MAXN] __attribute__((aligned(64)));
static uint8_t out[MAXN] __attribute__((aligned(64)));
/* A w=2 split of a 64 KiB block is two 32 KiB planes. */
static uint8_t planes[16][32768] __attribute__((aligned(64)));
static uint8_t ref[16][32768] __attribute__((aligned(64)));

static int is_delta(unsigned id)
{
    return id == 1u || (id >= 7u && id <= 10u);
}

static int is_xor(unsigned id)
{
    return id == 2u;
}

static uint32_t rng = 0x243F6A88u;

static uint8_t rnd(void)
{
    rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
    return (uint8_t)(rng >> 24);
}

int main(void)
{
    static const uint32_t sizes[] = { 65536u, 65535u, 4097u, 300u, 17u, 16u };
    unsigned checks = 0;
    int fails = 0;
    unsigned n_i, id;

    setvbuf(stdout, NULL, _IONBF, 0);

    for (n_i = 0; n_i < sizeof sizes / sizeof sizes[0]; n_i++) {
        uint32_t n = sizes[n_i];
        uint32_t i;

        for (i = 0; i < n; i++) {
            in[i] = rnd();
        }
        for (id = 0; id < 12u; id++) {
            ZpseTransformMeta m;
            uint32_t lens[16];
            uint8_t *segs[16];
            const uint8_t *csegs[16];
            uint8_t w = 0;
            uint8_t sc = 0;
            unsigned c;
            unsigned s;
            uint32_t j;
            int delta = is_delta(id);
            int xorr = is_xor(id);

            for (c = 1; c <= 16u; c <<= 1) {
                if (zpse_transform_width((uint8_t)id, (uint8_t)c, &w) == PSE_OK) {
                    sc = (uint8_t)c;
                    break;
                }
            }
            if (sc == 0u || w == 0u) {
                printf("FAIL id=%u: no width\n", id);
                fails++;
                continue;
            }
            memset(&m, 0, sizeof m);
            m.transform_id = (uint8_t)id;
            m.width_W = w;
            if (delta || xorr) {
                m.has_seeds = 1;
            }

            /* Segments: s, s+w, ... -- independently of zpse_segment_lengths,
             * which is then checked against this. */
            for (s = 0; s < (unsigned)w; s++) {
                uint32_t at = (uint32_t)s;
                uint32_t ok_len = (n + (uint32_t)w - 1u - (uint32_t)s) / (uint32_t)w;

                lens[s] = ok_len;
                segs[s] = planes[s];
                csegs[s] = planes[s];
                for (j = 0; j < ok_len; j++) {
                    if (in[at] != in[(uint32_t)s + j * (uint32_t)w]) {
                        printf("FAIL id=%u: segment walk\n", id);
                        fails++;
                    }
                    at += (uint32_t)w;
                }
                m.seeds[s] = (uint8_t)(0x40u + s * 7u);
            }
            {
                uint32_t other[16];

                if (zpse_segment_lengths(n, w, other) != PSE_OK) {
                    printf("FAIL id=%u n=%u: segment_lengths\n", id, n);
                    fails++;
                    continue;
                }
                for (s = 0; s < (unsigned)w; s++) {
                    if (other[s] != lens[s]) {
                        printf("FAIL id=%u n=%u: segment_lengths[%u] %u != %u\n",
                               id, n, s, other[s], lens[s]);
                        fails++;
                    }
                }
            }

            /* Reference, from the definition. */
            for (s = 0; s < (unsigned)w; s++) {
                uint8_t prev = m.seeds[s];

                for (j = 0; j < lens[s]; j++) {
                    uint8_t raw = in[(uint32_t)s + j * (uint32_t)w];

                    ref[s][j] = delta ? (uint8_t)(raw - prev)
                               : xorr ? (uint8_t)(raw ^ prev)
                                      : raw;
                    prev = raw;
                }
            }

            if (zpse_transform_forward(&m, w, in, n, segs, lens) != PSE_OK) {
                printf("FAIL id=%u n=%u: forward rc\n", id, n);
                fails++;
                continue;
            }
            for (s = 0; s < (unsigned)w; s++) {
                checks++;
                if (memcmp(planes[s], ref[s], lens[s]) != 0) {
                    printf("FAIL id=%u n=%u: plane %u differs from reference\n",
                           id, n, s);
                    fails++;
                    break;
                }
            }

            for (s = 0; s < (unsigned)w; s++) {
                csegs[s] = planes[s];
            }
            if (zpse_transform_inverse(&m, w, csegs, lens, out, n) != PSE_OK) {
                printf("FAIL id=%u n=%u: inverse rc\n", id, n);
                fails++;
                continue;
            }
            checks++;
            if (memcmp(out, in, n) != 0) {
                for (i = 0; i < n; i++) {
                    if (out[i] != in[i]) {
                        printf("FAIL id=%u n=%u: byte %u is %02X, input %02X\n",
                               id, n, i, (unsigned)out[i], (unsigned)in[i]);
                        break;
                    }
                }
                fails++;
            }
        }
    }

    if (fails == 0) {
        printf("TRANSFORMS: ALL OK (%u checks)\n", checks);
    } else {
        printf("TRANSFORMS: %d FAILURES\n", fails);
    }
    return fails != 0;
}
