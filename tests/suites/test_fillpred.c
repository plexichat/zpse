/* Lean classify + at_pred fill vs the full classification.
 *
 * zpse_tile_classify_lean skips cl->at_pred, the ~66-cycle compaction the
 * segment encoder does not need for the modes most tiles pack as
 * (RAW_ACTUAL). The predictions are produced on demand by
 * zpse_tile_fill_at_pred_reset. This suite is the guard on that pair:
 *
 *   for every predictor the encoder classifies, every tile length the
 *   format allows (full, ragged, single-byte), two segment offsets (so the
 *   staged-head and in-place paths are both covered), three initial
 *   literals and four data shapes (noise, small alphabet, ramp, constant):
 *
 *     full classify          -> reference
 *     lean classify + fill   -> must match the reference field by field,
 *                               including at_pred[0..k) and the lane state
 *
 * A fill that silently differed would change compressed bytes without
 * failing a round-trip, because the decoder mirrors whatever was written.
 * That is why this compares the classification itself rather than a stream
 * digest.
 *
 * Also checks the documented refusals: ORDER1/ORDER2 return
 * PSE_BAD_PREDICTOR, and a STRIDE fill without seg_base is PSE_BAD_ARG.
 *
 * Exit 0 with "FILLPRED: ALL OK (n checks)"; exits 1 on any mismatch.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "zpse/predictor.h"
#include "zpse/status.h"
#include "zpse/tile.h"

static uint8_t plane[1024 + 64];
static uint8_t o1[8 * 256];
static uint8_t o2[8 * 4096 * 4];


static ZpseTileClass ref_cl;
static ZpseTileClass lean_cl;

static uint32_t rng = 0x9E3779B9u;

static uint32_t rnd(void)
{
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}

/* The ten predictors the encoder's prescreen can name. ORDER1/ORDER2 are
 * deliberately absent: they are not trial candidates (see segment.c) and
 * the fill refuses them. */
static const uint8_t preds[10] = { 0u, 1u, 2u, 5u, 6u, 7u, 8u, 9u, 10u, 11u };
static const uint16_t lens[9] = { 256u, 200u, 137u, 64u, 33u, 32u, 17u, 8u,
                                  1u };
static const uint32_t offs[2] = { 0u, 100u };
static const uint8_t lits[3] = { 0u, 0x5Au, 0xFFu };

static void fill_shape(uint8_t *p, uint32_t n, int shape)
{
    uint32_t i;

    for (i = 0; i < n; i++) {
        switch (shape) {
        case 0: /* noise: dense masks, where classify costs the most */
            p[i] = (uint8_t)rnd();
            break;
        case 1: /* four values: sparse and clustered masks */
            p[i] = (uint8_t)("abcd"[rnd() & 3u]);
            break;
        case 2: /* ramp: STRIDE_1 is all hits, the others are not */
            p[i] = (uint8_t)(i * 7u);
            break;
        default: /* constant */
            p[i] = 0x33u;
            break;
        }
    }
}

static int cmp_cl(const char *what, const ZpseTileClass *a,
                  const ZpseTileClass *b)
{
    uint16_t i;

    if (a->k != b->k || a->nu != b->nu || a->rmax != b->rmax ||
        a->dmin != b->dmin || a->dmax != b->dmax ||
        a->act_valid != b->act_valid || a->constant_C != b->constant_C) {
        printf("FAIL %s: scalars differ (k %u/%u nu %u/%u)\n", what,
               (unsigned)a->k, (unsigned)b->k, (unsigned)a->nu,
               (unsigned)b->nu);
        return 1;
    }
    if (memcmp(a->words, b->words, sizeof a->words) != 0) {
        printf("FAIL %s: mask words differ\n", what);
        return 1;
    }
    if (memcmp(a->seen, b->seen, sizeof a->seen) != 0) {
        printf("FAIL %s: seen set differs\n", what);
        return 1;
    }
    /* Only [0, k) is defined: the arrays carry up to 8 bytes of store slack
     * past the last entry, and nothing beyond k is part of the contract. */
    for (i = 0; i < a->k; i++) {
        if (a->at_pred[i] != b->at_pred[i]) {
            printf("FAIL %s: at_pred[%u] %u != %u\n", what, (unsigned)i,
                   (unsigned)a->at_pred[i], (unsigned)b->at_pred[i]);
            return 1;
        }
        if (a->res_at_exc[i] != b->res_at_exc[i]) {
            printf("FAIL %s: res_at_exc[%u] %u != %u\n", what, (unsigned)i,
                   (unsigned)a->res_at_exc[i], (unsigned)b->res_at_exc[i]);
            return 1;
        }
        if (a->act_at_exc[i] != b->act_at_exc[i]) {
            printf("FAIL %s: act_at_exc[%u] %u != %u\n", what, (unsigned)i,
                   (unsigned)a->act_at_exc[i], (unsigned)b->act_at_exc[i]);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    uint32_t checks = 0;
    uint32_t fails = 0;
    unsigned pi;
    unsigned li;
    unsigned oi;
    unsigned si;
    unsigned ti;

    /* 16 bytes of lead-in so a tile can start mid-buffer, plus 32 bytes of
     * tail so a full-tile read past a short tile stays in bounds. */
    memset(plane, 0, sizeof plane);
    memset(o1, 0, sizeof o1);
    memset(o2, 0, sizeof o2);

    for (si = 0; si < 4u; si++) {
        for (pi = 0; pi < sizeof preds; pi++) {
            for (li = 0; li < sizeof lens / sizeof lens[0]; li++) {
                for (oi = 0; oi < 2u; oi++) {
                    for (ti = 0; ti < 3u; ti++) {
                        const uint8_t pid = preds[pi];
                        const uint16_t nt = lens[li];
                        const uint32_t off = offs[oi];
                        const uint8_t lit = lits[ti];
                        ZpseLaneState a;
                        ZpseLaneState b;
                        char what[96];
                        int rc;

                        fill_shape(plane, sizeof plane, (int)si);
                        snprintf(what, sizeof what,
                                 "shape=%u pid=%u nt=%u off=%u lit=%u",
                                 si, (unsigned)pid, (unsigned)nt,
                                 (unsigned)off, (unsigned)lit);

                        memset(&ref_cl, 0xA5, sizeof ref_cl);
                        memset(&lean_cl, 0x5A, sizeof lean_cl);
                        a.order1 = o1;
                        a.o2 = o2;
                        a.gen = 0u;
                        a.gen = 0u;
                        b = a;
                        rc = zpse_tile_classify(&a, plane + off, nt, pid, lit,
                                                1, plane, off, &ref_cl);
                        if (rc != PSE_OK) {
                            printf("FAIL %s: full classify rc=%d\n", what, rc);
                            fails++;
                            continue;
                        }
                        rc = zpse_tile_classify_lean(&b, plane + off, nt, pid,
                                                     lit, 1, plane, off,
                                                     &lean_cl);
                        if (rc != PSE_OK) {
                            printf("FAIL %s: lean classify rc=%d\n", what, rc);
                            fails++;
                            continue;
                        }
                        /* Everything except at_pred must already agree. */
                        {
                            const ZpseTileClass saved = lean_cl;

                            memcpy(lean_cl.at_pred, ref_cl.at_pred,
                                   sizeof lean_cl.at_pred);
                            fails += (uint32_t)cmp_cl(what, &ref_cl, &lean_cl);
                            lean_cl = saved;
                        }
                        /* And the fill must produce those predictions. */
                        rc = zpse_tile_fill_at_pred_reset(pid, plane + off, nt,
                                                          plane, off, lit,
                                                          &lean_cl);
                        if (rc != PSE_OK) {
                            printf("FAIL %s: fill rc=%d\n", what, rc);
                            fails++;
                            continue;
                        }
                        fails += (uint32_t)cmp_cl(what, &ref_cl, &lean_cl);
                        if (memcmp(&a.last, &b.last, sizeof a.last) != 0 ||
                            memcmp(&a.second_last, &b.second_last,
                                   sizeof a.second_last) != 0) {
                            printf("FAIL %s: lane state differs\n", what);
                            fails++;
                        }
                        checks++;
                    }
                }
            }
        }
    }

    /* Documented refusals. */
    memset(&lean_cl, 0, sizeof lean_cl);
    if (zpse_tile_fill_at_pred_reset((uint8_t)ZPSE_P_ORDER1, plane, 256u,
                                     plane, 16u, 0u, &lean_cl) !=
        PSE_BAD_PREDICTOR) {
        printf("FAIL ORDER1 fill was not refused\n");
        fails++;
    }
    if (zpse_tile_fill_at_pred_reset((uint8_t)ZPSE_P_ORDER2, plane, 256u,
                                     plane, 16u, 0u, &lean_cl) !=
        PSE_BAD_PREDICTOR) {
        printf("FAIL ORDER2 fill was not refused\n");
        fails++;
    }
    if (zpse_tile_fill_at_pred_reset((uint8_t)ZPSE_P_STRIDE_1, plane, 256u, 0,
                                     16u, 0u, &lean_cl) != PSE_BAD_ARG) {
        printf("FAIL STRIDE fill without seg_base was not refused\n");
        fails++;
    }
    checks += 3u;

    if (fails != 0) {
        printf("FILLPRED: %u FAILURES of %u checks\n", fails, checks);
        return 1;
    }
    printf("FILLPRED: ALL OK (%u checks)\n", checks);
    return 0;
}
