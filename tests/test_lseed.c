/* Lane-state seeding test.
 *
 * The decoder establishes a segment's ORDER1/ORDER2 tables lazily: a tile
 * whose reset bit is set re-seeds the table it reads, and the segment's
 * initial state for a table is only materialised when a tile that does NOT
 * reset can read it. This suite is what holds that down, because the shape
 * it protects cannot come out of this encoder -- every tile the encoder
 * emits has its reset bit set, so nothing here is reachable from its own
 * output, only from a foreign stream.
 *
 * Each case is a hand-built segment: no encoder is used at all, the masks
 * and residuals are literal bytes, and the predictor/reset pattern is the
 * only variable. Every case is decoded twice:
 *
 *   - by zpse_segment_decode (the code under test), and
 *   - by the reference below, which walks the same records but hardcodes
 *     the documented segment-start state (histories = initial literal,
 *     order1 = initial literal, o2val = initial literal, o2conf = 0) and
 *     the documented per-tile reset rule instead of deriving either from
 *     the library.
 *
 * The two must agree byte for byte. If a segment-start table is never
 * seeded, the reference still predicts from the literal-filled table while
 * the decoder predicts from whatever the arena happened to hold, and the
 * case fails -- which is exactly the bug this covers ("the non-resetting
 * first ORDER tile reads an unseeded table"). The mask and residual
 * payloads are unpacked with the library's own mask/residual decoders, so
 * the suite is about lane state, not about re-testing the payload codecs
 * (test_units.c and test_hostile.c own those).
 *
 * Also checks the cheap invariants: decode succeeds, produces exactly the
 * segment's declared size, and the exit status follows the failure count.
 */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/arena.h"
#include "zpse/bits.h"
#include "zpse/mask.h"
#include "zpse/predictor.h"
#include "zpse/residual.h"
#include "zpse/segment.h"
#include "zpse/tile.h"
#include "zpse/types.h"

static unsigned char darena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static ZpseDecoder D;

static uint8_t payload[65536];
static uint8_t out[65536];
static uint8_t ref[65536];

/* Reference tables: the documented segment-start state. */
static uint8_t r_o1[8 * 256];
static uint8_t r_o2val[8 * 4096];
static uint8_t r_o2conf[8 * 4096];
static uint8_t r_last[8];
static uint8_t r_second[8];

static int fails = 0;
static unsigned nc = 0;

static void ref_start(uint8_t lit)
{
    unsigned l;

    for (l = 0; l < 8u; l++) {
        r_last[l] = lit;
        r_second[l] = lit;
    }
    memset(r_o1, lit, sizeof r_o1);
    memset(r_o2val, lit, sizeof r_o2val);
    memset(r_o2conf, 0, sizeof r_o2conf);
}

static void ref_tile(uint8_t pid, int reset, uint8_t lit, uint16_t nt,
                     const uint32_t words[8], const uint8_t *resid,
                     uint8_t *dst)
{
    unsigned l;

    if (reset != 0) {
        for (l = 0; l < 8u; l++) {
            r_last[l] = lit;
            r_second[l] = lit;
        }
        if (pid == (uint8_t)ZPSE_P_ORDER1) {
            memset(r_o1, lit, sizeof r_o1);
        } else if (pid == (uint8_t)ZPSE_P_ORDER2) {
            memset(r_o2val, lit, sizeof r_o2val);
            memset(r_o2conf, 0, sizeof r_o2conf);
        }
    }

    {
        uint16_t p;
        uint16_t j = 0;

        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint8_t last = r_last[lane];
            uint8_t second = r_second[lane];
            uint8_t xhat;
            uint8_t x;

            if (pid == (uint8_t)ZPSE_P_ORDER1) {
                xhat = r_o1[lane * 256u + (size_t)last];
            } else if (pid == (uint8_t)ZPSE_P_ORDER2) {
                xhat = r_o2val[lane * 4096u +
                               (size_t)zpse_order2_index(second, last)];
            } else { /* STRIDE_1 over the decoded stream */
                xhat = (p < 1u) ? lit : dst[p - 1u];
            }
            x = (((words[p >> 5] >> (p & 31u)) & 1u) != 0u)
                    ? zpse_exclusion_inverse(resid[j++], xhat)
                    : xhat;
            dst[p] = x;

            if (pid == (uint8_t)ZPSE_P_ORDER1) {
                size_t off = lane * 256u + (size_t)last;

                r_o1[off] = x;
            } else if (pid == (uint8_t)ZPSE_P_ORDER2) {
                size_t off = lane * 4096u +
                             (size_t)zpse_order2_index(second, last);

                if (r_o2conf[off] == 0u) {
                    r_o2val[off] = x;
                    r_o2conf[off] = 1u;
                } else if (x == xhat) {
                    if (r_o2conf[off] < 255u)
                        r_o2conf[off]++;
                } else {
                    r_o2conf[off]--;
                }
            }
            r_second[lane] = last;
            r_last[lane] = x;
        }
    }
}

/* One hand-built segment: `ntiles` tiles of 256 bytes, tile t using
 * pids[t] with reset flag resets[t]. The mask is a fixed 40-exception
 * RAW_BITMAP and the residual is EXCLUSION_U8, so both decoders see the
 * same payload and only the lane state can differ. */
static int run_case(const char *name, int ntiles, const uint8_t *pids,
                    const uint8_t *resets, uint8_t lit)
{
    static const uint16_t nex = 40u;
    uint32_t words[8];
    uint8_t resid[40];
    uint32_t pos;
    uint32_t got = 0;
    int t;
    int bad = 0;
    ZpseLaneState dl;
    int rc;

    memset(payload, 0, sizeof payload);
    {
        ZpseSegmentHeader sh;

        sh.segment_decoded_size = (uint32_t)ntiles * 256u;
        sh.tile_count = (uint16_t)ntiles;
        sh.initial_literal = lit;
        sh.segment_flags = 0x01u;
        sh.tile_directory_size = 0u;
        sh.reserved = 0u;
        zpse_segment_header_serialize(payload, &sh);
    }

    memset(words, 0, sizeof words);
    for (t = 0; t < (int)nex; t++) {
        uint16_t p = (uint16_t)(t * 6u + 2u); /* ascending, inside 256 */

        words[p >> 5] |= (uint32_t)1u << (p & 31u);
    }
    for (t = 0; t < (int)nex; t++)
        resid[t] = (uint8_t)(t * 5u + 3u); /* <= 127, so exclusion is legal */

    pos = 16u;
    for (t = 0; t < ntiles; t++) {
        uint8_t *rec = payload + pos + 2u;
        uint16_t rsz = (uint16_t)(ZPSE_TILE_HEADER_SIZE + 32u + nex);

        rec[0] = zpse_tile_flags_make((uint8_t)ZPSE_M_RAW_BITMAP,
                                      (uint8_t)ZPSE_R_EXCLUSION_U8,
                                      resets[t] != 0u ? 1 : 0);
        rec[1] = pids[t];
        rec[2] = 255u; /* nt - 1 */
        rec[3] = (uint8_t)nex;
        zpse_store_u16le(rec + 4, 32u);
        zpse_store_u16le(rec + 6, (uint16_t)nex);
        memcpy(rec + ZPSE_TILE_HEADER_SIZE, words, 32u);
        memcpy(rec + ZPSE_TILE_HEADER_SIZE + 32u, resid, nex);
        zpse_store_u16le(payload + pos, rsz);
        pos += 2u + (uint32_t)rsz;
    }

    /* Reference pass. */
    ref_start(lit);
    for (t = 0; t < ntiles; t++)
        ref_tile(pids[t], resets[t] != 0, lit, 256u, words, resid,
                 ref + (size_t)t * 256u);

    /* Library pass. */
    memset(out, 0xA5, sizeof out);
    dl.order1 = D.order1;
    dl.o2val = D.o2val;
    dl.o2conf = D.o2conf;
    rc = zpse_segment_decode(&D, &dl, payload, pos, out,
                             (uint32_t)ntiles * 256u, &got);
    if (rc != PSE_OK) {
        printf("FAIL %s: decode rc=%d\n", name, rc);
        fails++;
        return 1;
    }
    if (got != (uint32_t)ntiles * 256u) {
        printf("FAIL %s: produced %u of %u\n", name, (unsigned)got,
               (unsigned)(ntiles * 256));
        fails++;
        return 1;
    }
    if (memcmp(out, ref, (size_t)ntiles * 256u) != 0) {
        for (t = 0; t < ntiles * 256; t++) {
            if (out[t] != ref[t]) {
                printf("FAIL %s: byte %d is %02X, reference says %02X\n",
                       name, t, (unsigned)out[t], (unsigned)ref[t]);
                break;
            }
        }
        fails++;
        bad = 1;
    }
    nc++;
    if (bad == 0) {
        printf("ok   %s (%d tile%s)\n", name, ntiles, ntiles == 1 ? "" : "s");
    }
    return bad;
}

int main(void)
{
    /* Non-resetting first tile, chains, and mixed families: every shape
     * whose segment-start table state a missing lazy seed would change. */
    static const uint8_t p_o1[1] = { 3u };
    static const uint8_t p_o2[1] = { 4u };
    static const uint8_t p_st[1] = { 5u };
    static const uint8_t p_chain1[3] = { 3u, 3u, 4u };
    static const uint8_t p_chain2[3] = { 4u, 3u, 3u };
    static const uint8_t p_mix[2] = { 5u, 4u };
    static const uint8_t r0[1] = { 0u };
    static const uint8_t r1[1] = { 1u };
    static const uint8_t r010[3] = { 0u, 1u, 0u };
    static const uint8_t r101[3] = { 1u, 0u, 1u };
    static const uint8_t r00[2] = { 0u, 0u };

    if (zpse_decoder_init(&D, darena, sizeof darena) != 0) {
        printf("FAIL: decoder init\n");
        return 1;
    }

    run_case("order1-first-noreset", 1, p_o1, r0, 0x5Au);
    run_case("order2-first-noreset", 1, p_o2, r0, 0x5Au);
    run_case("order1-first-reset", 1, p_o1, r1, 0x5Au);
    run_case("order2-first-reset", 1, p_o2, r1, 0x5Au);
    run_case("stride-first-noreset", 1, p_st, r0, 0x5Au);
    run_case("order1-order1-order2", 3, p_chain1, r010, 0x5Au);
    run_case("order2-order1-order1", 3, p_chain2, r101, 0x5Au);
    run_case("stride-order2-noreset", 2, p_mix, r00, 0x5Au);
    run_case("order1-first-literal0", 1, p_o1, r0, 0x00u);
    run_case("order2-first-literal0", 1, p_o2, r0, 0x00u);

    if (fails == 0)
        printf("LSEED: ALL OK (%u cases)\n", nc);
    else
        printf("LSEED: %d FAILURES\n", fails);
    return fails != 0;
}
