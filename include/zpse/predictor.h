#ifndef ZPSE_PREDICTOR_H
#define ZPSE_PREDICTOR_H

/* 8-lane predictors, profile 0. Scalar formulas are
 * normative; vector.h provides bit-exact SIMD equivalents. */

#include <stdint.h>
#include "constants.h"

typedef enum ZpsePredictorId {
    ZPSE_P_CONST_ZERO     = 0,
    ZPSE_P_LANE_PREVIOUS  = 1,
    ZPSE_P_LANE_DELTA     = 2,
    /* 3 and 4 are reserved gaps. Old streams naming them are refused, never
     * decoded: no live predictor may be assigned either number. */
    ZPSE_P_RESERVED_3     = 3,
    ZPSE_P_RESERVED_4     = 4,
    ZPSE_P_STRIDE_1       = 5,
    ZPSE_P_STRIDE_2       = 6,
    ZPSE_P_STRIDE_4       = 7,
    ZPSE_P_STRIDE_8       = 8,
    /* 9 is a reserved gap too: the dense form's byte 4 is now the fill for
     * CONST_ZERO ALL_EXCEPTION records, so the old constant predictor has
     * no wire representation. */
    ZPSE_P_RESERVED_9     = 9,
    ZPSE_P_LANE_AVERAGE   = 10,
    ZPSE_P_STRIDE_3       = 11  /* x_hat = out[p-3]; interleaved RGB etc. */
} ZpsePredictorId;

/* Live lane histories (kept in registers/stack, <= 16 bytes). Every tile
 * reads these as its entry state. The two arrays MUST stay adjacent: the
 * hot reset writes both with one 16-byte memset from last[]. */
typedef struct ZpseLaneState {
    uint8_t last[8];
    uint8_t second_last[8];
} ZpseLaneState;

void zpse_lane_reset(ZpseLaneState *s, uint8_t initial_literal);

/* Entry-state reset used by every tile. The predictor id is part of the
 * signature because callers already have it at hand; no live predictor
 * reads a table, so the reset no longer depends on it. */
void zpse_lane_reset_for(ZpseLaneState *s, uint8_t initial_literal,
                         uint8_t predictor_id);

/* The history half of the reset on its own: last[]/second_last[] set to
 * the literal, 16 bytes, no tables. This is the entry state every tile
 * reads, and it is the whole of zpse_lane_reset now. */
void zpse_lane_reset_hist(ZpseLaneState *s, uint8_t initial_literal);

/* zpse_predict_block and zpse_predict_update were removed: no callers in
 * src/ or tests/. The snapshot-semantics note worth keeping is that the
 * stateful predictors (1, 2, 10) require the online predict/update loop in
 * tile.c for tile reconstruction; bit-exact SIMD formulas live in vector.h. */

#endif /* ZPSE_PREDICTOR_H */
