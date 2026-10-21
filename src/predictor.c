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
/* An ORDER2 entry is {value, confidence, generation}: the three bytes
 * a prediction needs, in one cache line (two tables meant two misses a
 * byte, one of them to a line that held nothing the other wanted). */
static inline int zpse_o2_learned(const uint8_t *tab, size_t off,
                                   uint16_t gen)
{
    return (uint16_t)tab[off + 2u] == (uint16_t)(gen & 0xFFu) &&
           (uint16_t)tab[off + 3u] == (uint16_t)(gen >> 8);
}

static inline void zpse_o2_stamp(uint8_t *tab, size_t off, uint16_t gen)
{
    tab[off + 2u] = (uint8_t)(gen & 0xFFu);
    tab[off + 3u] = (uint8_t)(gen >> 8);
}

void zpse_lane_seed_table(ZpseLaneState *s, uint8_t initial_literal,
                          uint8_t predictor_id)
{
    if (s == NULL) {
        return;
    }
    /* pid 3 (ORDER1) is the only reader of order1; pid 4 (ORDER2) the
     * only reader of o2 (predict and update). All other
     * predictors touch histories only. */
    if (predictor_id == (uint8_t)ZPSE_P_ORDER1) {
        if (s->order1 != NULL) {
            memset(s->order1, initial_literal, (size_t)8u * 256u);
        }
    } else if (predictor_id == (uint8_t)ZPSE_P_ORDER2) {
        /* Nothing to seed. An entry is learned only if its generation is
         * the tile's, and the tile's generation advances every tile, so
         * no entry can be mistaken for one this tile wrote. The wipe
         * that used to be here -- 64 KiB of memset, a third of the
         * classify -- is gone; the table is wiped once per trial
         * instead, where the generation restarts. */
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
        if (s->o2 == NULL) {
            return PSE_BAD_ARG;
        }
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint16_t idx = zpse_order2_index(s->second_last[lane],
                                            s->last[lane]);
            size_t off = (lane * 4096u + (size_t)idx) * 4u;

            /* An entry nobody has learned predicts the initial literal.
             * "Learned" means written by this tile: the generation says
             * so. */
            pred_out[p] = zpse_o2_learned(s->o2, off, s->gen)
                              ? s->o2[off]
                              : initial_literal;
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
                /* Segment-absolute addressing: the stream
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
            size_t off = (lane * 4096u + (size_t)idx) * 4u;
            uint8_t *val;
            uint8_t *conf;

            if (s->o2 == NULL) {
                /* No table to learn into: the caller has the
                 * predictions and nothing else to update. */
                (void)pred;
                continue;
            }
            val = &s->o2[off];
            conf = &s->o2[off + 1u];
            if (!zpse_o2_learned(s->o2, off, s->gen)) {
                /* Nobody has learned this entry, so its prediction was
                 * the initial literal; whether the byte matched it or
                 * not, the value now learned is the byte. */
                *val = x;
                *conf = 1u;
                zpse_o2_stamp(s->o2, off, s->gen);
            } else if (x == *val) {
                if (*conf < 255u) {
                    (*conf)++;
                }
            } else {
                (*conf)--;
            }
        }
    }
}
