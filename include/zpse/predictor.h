#ifndef ZPSE_PREDICTOR_H
#define ZPSE_PREDICTOR_H

/* 8-lane predictors, profile 0 (SPEC Sec. 7). Scalar formulas are
 * normative; vector.h provides bit-exact SIMD equivalents. */

#include <stdint.h>
#include "constants.h"

typedef enum ZpsePredictorId {
    ZPSE_P_CONST_ZERO     = 0,
    ZPSE_P_LANE_PREVIOUS  = 1,
    ZPSE_P_LANE_DELTA     = 2,
    ZPSE_P_ORDER1         = 3,
    ZPSE_P_ORDER2         = 4,
    ZPSE_P_STRIDE_1       = 5,
    ZPSE_P_STRIDE_2       = 6,
    ZPSE_P_STRIDE_4       = 7,
    ZPSE_P_STRIDE_8       = 8,
    ZPSE_P_TILE_CONSTANT  = 9,
    ZPSE_P_LANE_AVERAGE   = 10
} ZpsePredictorId;

/* Live lane histories (kept in registers/stack, <= 16 bytes). Tables
 * themselves live in the arena; this struct only borrows pointers. */
typedef struct ZpseLaneState {
    uint8_t last[8];
    uint8_t second_last[8];
    uint8_t *order1; /* 8 x 256, arena */
    uint8_t *o2val;  /* 8 x 4096, arena */
    uint8_t *o2conf; /* 8 x 4096, arena */
} ZpseLaneState;

void zpse_lane_reset(ZpseLaneState *s, uint8_t initial_literal);

/* Selective reset: histories always; tables only if predictor reads them
 * (ORDER1 iff pid==3; O2VAL/O2CONF iff pid==4; neither otherwise).
 * Same values as zpse_lane_reset. Saves ~66 KB memset on most tiles. */
void zpse_lane_reset_for(ZpseLaneState *s, uint8_t initial_literal,
                         uint8_t predictor_id);

/* ORDER2 index: ((second*251) ^ (last*17)) & 0xFFF. */
static inline uint16_t zpse_order2_index(uint8_t second_last, uint8_t last) {
    return (uint16_t)(((uint16_t)((uint16_t)second_last * 251u) ^
                       (uint16_t)((uint16_t)last * 17u)) & 0xFFFu);
}

/* Generate Nt predictions into pred_out (scalar reference). seg_base /
 * seg_pos_backing support STRIDE predictors: seg_base points at the
 * decoded transformed stream for this segment, seg_pos = count of bytes
 * already decoded in this segment before this tile. For non-stride
 * predictors these may be NULL/0. constant_C used iff predictor == 9.
 *
 * Snapshot semantics: predictions for all p in [0,Nt) derive from the
 * ENTRY state (no intra-tile chaining). Exact only for stateless
 * predictors (CONST_ZERO, TILE_CONSTANT, and STRIDE when its backing
 * bytes are already final). Stateful predictors (1-4, 10) require the
 * online predict/update loop in tile.c for tile reconstruction; do NOT
 * use this as a tile reference generator for those IDs. */
int zpse_predict_block(const ZpseLaneState *s, uint8_t predictor_id,
                       uint8_t constant_C, const uint8_t *seg_base,
                       uint32_t seg_pos, uint8_t initial_literal,
                       uint8_t *pred_out, uint16_t nt);

/* Advance histories + ORDER1/ORDER2 tables in position order given the
 * actual transformed bytes and their predictions. */
void zpse_predict_update(ZpseLaneState *s, uint8_t predictor_id,
                         const uint8_t *actual, const uint8_t *pred,
                         uint16_t nt);

#endif /* ZPSE_PREDICTOR_H */
