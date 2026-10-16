#include "zpse/predictor.h"

#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Histories only: 16 bytes, and the part of the entry state that every
 * tile's classify and decode depend on. Split out of zpse_lane_reset so
 * a caller can establish the entry state without paying the 66 KiB of
 * tables that only predictors 3 and 4 ever read. */
void zpse_lane_reset_hist(ZpseLaneState *s, uint8_t initial_literal)
{
    size_t l;

    if (s == NULL) {
        return;
    }
    for (l = 0; l < 8u; l++) {
        s->last[l] = initial_literal;
        s->second_last[l] = initial_literal;
    }
}

/* The tables one predictor reads, in the reset state: the same values
 * zpse_lane_reset_for writes for that predictor, and nothing for the
 * predictors that read no table at all. */
void zpse_lane_seed_table(ZpseLaneState *s, uint8_t initial_literal,
                          uint8_t predictor_id)
{
    if (s == NULL) {
        return;
    }
    /* pid 3 (ORDER1) is the only reader of order1; pid 4 (ORDER2) the
     * only reader of o2val (predict) / o2conf (update). All other
     * predictors touch histories only. */
    if (predictor_id == (uint8_t)ZPSE_P_ORDER1) {
        if (s->order1 != NULL) {
            memset(s->order1, initial_literal, (size_t)8u * 256u);
        }
    } else if (predictor_id == (uint8_t)ZPSE_P_ORDER2) {
        if (s->o2val != NULL) {
            memset(s->o2val, initial_literal, (size_t)8u * 4096u);
        }
        if (s->o2conf != NULL) {
            memset(s->o2conf, 0, (size_t)8u * 4096u);
        }
    }
}

void zpse_lane_reset(ZpseLaneState *s, uint8_t initial_literal)
{
    zpse_lane_reset_hist(s, initial_literal);
    zpse_lane_seed_table(s, initial_literal, (uint8_t)ZPSE_P_ORDER1);
    zpse_lane_seed_table(s, initial_literal, (uint8_t)ZPSE_P_ORDER2);
}

void zpse_lane_reset_for(ZpseLaneState *s, uint8_t initial_literal,
                           uint8_t predictor_id)
{
    zpse_lane_reset_hist(s, initial_literal);
    zpse_lane_seed_table(s, initial_literal, predictor_id);
}

static uint32_t zpse_stride_of(uint8_t predictor_id)
{
    switch (predictor_id) {
    case ZPSE_P_STRIDE_1:
        return 1u;
    case ZPSE_P_STRIDE_2:
        return 2u;
    case ZPSE_P_STRIDE_3:
        return 3u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u;
    }
}

int zpse_predict_block(const ZpseLaneState *s, uint8_t predictor_id,
                       uint8_t constant_C, const uint8_t *seg_base,
                       uint32_t seg_pos, uint8_t initial_literal,
                       uint8_t *pred_out, uint16_t nt)
{
    uint16_t p;

    if (s == NULL || pred_out == NULL) {
        return PSE_BAD_ARG;
    }
    if (nt == 0u || nt > 256u) {
        return PSE_BAD_ARG;
    }
    if (predictor_id > (uint8_t)ZPSE_P_STRIDE_3) {
        return PSE_BAD_PREDICTOR;
    }

    switch (predictor_id) {
    case ZPSE_P_CONST_ZERO: /* 0 */
        for (p = 0; p < nt; p++) {
            pred_out[p] = 0u;
        }
        break;

    case ZPSE_P_LANE_PREVIOUS: /* 1 */
        for (p = 0; p < nt; p++) {
            pred_out[p] = s->last[(size_t)p & 7u];
        }
        break;

    case ZPSE_P_LANE_DELTA: /* 2 */
        for (p = 0; p < nt; p++) {
            unsigned last = s->last[(size_t)p & 7u];
            unsigned prev = s->second_last[(size_t)p & 7u];
            pred_out[p] = (uint8_t)((2u * last - prev) & 0xFFu);
        }
        break;

    case ZPSE_P_ORDER1: /* 3 */
        if (s->order1 == NULL) {
            return PSE_BAD_ARG;
        }
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            pred_out[p] = s->order1[lane * 256u + (size_t)s->last[lane]];
        }
        break;

    case ZPSE_P_ORDER2: /* 4 */
        if (s->o2val == NULL) {
            return PSE_BAD_ARG;
        }
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint16_t idx = zpse_order2_index(s->second_last[lane],
                                            s->last[lane]);
            pred_out[p] = s->o2val[lane * 4096u + (size_t)idx];
        }
        break;

    case ZPSE_P_STRIDE_1: /* 5 */
    case ZPSE_P_STRIDE_2: /* 6 */
    case ZPSE_P_STRIDE_4: /* 7 */
    case ZPSE_P_STRIDE_8: /* 8 */
    case ZPSE_P_STRIDE_3: /* 11 */
        {
            uint32_t stride = zpse_stride_of(predictor_id);
            if (seg_pos + (uint32_t)nt > stride && seg_base == NULL) {
                return PSE_BAD_ARG;
            }
            for (p = 0; p < nt; p++) {
                /* Segment-absolute addressing (SPEC Sec. 7): the stream
                 * is segment-scoped, so the guard is on seg_pos + p.
                 * NOTE: batch form is still only exact for stateless
                 * predictors; tile reconstruction uses the online loop
                 * in tile.c. */
                uint32_t abs_pos = seg_pos + (uint32_t)p;
                if (abs_pos < stride) {
                    pred_out[p] = initial_literal;
                } else {
                    uint32_t idx = abs_pos - stride;
                    pred_out[p] = seg_base[idx];
                }
            }
        }
        break;

    case ZPSE_P_TILE_CONSTANT: /* 9 */
        for (p = 0; p < nt; p++) {
            pred_out[p] = constant_C;
        }
        break;

    default: /* 10 LANE_AVERAGE */
        for (p = 0; p < nt; p++) {
            unsigned last = s->last[(size_t)p & 7u];
            unsigned prev = s->second_last[(size_t)p & 7u];
            pred_out[p] = (uint8_t)((last + prev + 1u) >> 1);
        }
        break;
    }

    return PSE_OK;
}

void zpse_predict_update(ZpseLaneState *s, uint8_t predictor_id,
                         const uint8_t *actual, const uint8_t *pred,
                         uint16_t nt)
{
    uint16_t p;

    if (s == NULL || actual == NULL || nt == 0u || nt > 256u) {
        return;
    }

    for (p = 0; p < nt; p++) {
        size_t lane = (size_t)p & 7u;
        uint8_t old_last = s->last[lane];
        uint8_t old_second = s->second_last[lane];
        uint8_t x = actual[p];

        /* History update applies to every predictor, in position order. */
        s->second_last[lane] = old_last;
        s->last[lane] = x;

        if (predictor_id == ZPSE_P_ORDER1) { /* 3 */
            if (s->order1 != NULL) {
                s->order1[lane * 256u + (size_t)old_last] = x;
            }
        } else if (predictor_id == ZPSE_P_ORDER2) { /* 4 */
            uint16_t idx = zpse_order2_index(old_second, old_last);
            size_t off = lane * 4096u + (size_t)idx;
            int hit;

            if (s->o2val != NULL) {
                hit = (x == s->o2val[off]);
            } else if (pred != NULL) {
                hit = (x == pred[p]);
            } else {
                continue;
            }

            if (hit) {
                if (s->o2conf != NULL && s->o2conf[off] < 255u) {
                    s->o2conf[off]++;
                }
            } else {
                if (s->o2conf != NULL) {
                    if (s->o2conf[off] == 0u) {
                        if (s->o2val != NULL) {
                            s->o2val[off] = x;
                        }
                        s->o2conf[off] = 1u;
                    } else {
                        s->o2conf[off]--;
                    }
                } else if (s->o2val != NULL) {
                    s->o2val[off] = x;
                }
            }
        }
    }
}
