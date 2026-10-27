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
    ZPSE_P_ORDER1         = 3,
    ZPSE_P_ORDER2         = 4,
    ZPSE_P_STRIDE_1       = 5,
    ZPSE_P_STRIDE_2       = 6,
    ZPSE_P_STRIDE_4       = 7,
    ZPSE_P_STRIDE_8       = 8,
    ZPSE_P_TILE_CONSTANT  = 9,
    ZPSE_P_LANE_AVERAGE   = 10,
    ZPSE_P_STRIDE_3       = 11  /* x_hat = out[p-3]; interleaved RGB etc. */
} ZpsePredictorId;

/* Live lane histories (kept in registers/stack, <= 16 bytes). Tables
 * themselves live in the arena; this struct only borrows pointers. */
typedef struct ZpseLaneState {
    uint8_t last[8];
    uint8_t second_last[8];
    uint8_t *order1; /* 8 x 256, arena */
    /* ORDER2's state: 8 x 4096 entries of {value, confidence,
     * generation}, interleaved so the three bytes a prediction needs are
     * in one cache line. Two separate tables meant two misses a byte,
     * one of them to a line that held nothing the other wanted.
     *
     * The generation is what makes the per-tile reset O(1): an entry
     * whose generation is not the current one was not learned by this
     * tile, so it reads as unlearned. The old table was wiped instead,
     * which cost 64 KiB of memset a tile. */
    uint8_t *o2;     /* 8 x 4096 x 4, arena */
    /* The generation this tile's ORDER2 entries carry. An entry whose
     * generation is not this one was not learned by this tile and reads
     * as unlearned -- which is what the per-tile wipe used to arrange,
     * at the price of 64 KiB of memset. The counter is advanced once a
     * tile by both sides, over the same tiles in the same order, so the
     * two agree on which entries are current. */
    uint16_t gen;
    /* Set once the ORDER2 table has been wiped for the block or trial this
     * lane state belongs to; cleared where the wipe used to be done. See
     * zpse_lane_o2_use. */
    int o2_ready;
} ZpseLaneState;

/* Wipe the ORDER2 table if this block or trial has not needed it yet.
 *
 * Entries carry a generation stamp and the stamp restarts with every block,
 * so a table left over from the previous block could make its first tiles
 * read entries they never wrote. Wiping at every block start is correct but
 * unconditional: 128 KiB of memset for a block whose tiles may never name
 * ORDER2, which on a one-value or stride block was the largest fixed cost
 * in the whole path. Every caller that is about to touch the table calls
 * this first, so the wipe happens exactly when it is needed and once. */
void zpse_lane_o2_use(ZpseLaneState *s);

void zpse_lane_reset(ZpseLaneState *s, uint8_t initial_literal);

/* Selective reset: histories always; tables only if predictor reads them
 * (ORDER1 iff pid==3; O2VAL/O2CONF iff pid==4; neither otherwise).
 * Same values as zpse_lane_reset. Saves ~66 KB memset on most tiles. */
void zpse_lane_reset_for(ZpseLaneState *s, uint8_t initial_literal,
                         uint8_t predictor_id);

/* The history half of the reset on its own: last[]/second_last[] set to
 * the literal, 16 bytes, no tables. This is the entry state every tile
 * reads, and it is the only part of zpse_lane_reset that every tile
 * depends on. */
void zpse_lane_reset_hist(ZpseLaneState *s, uint8_t initial_literal);

/* The table half of the reset on its own: exactly the memsets
 * zpse_lane_reset_for performs for the same predictor_id (order1 for
 * ORDER1; nothing for ORDER2, whose table the generation makes stale
 * instead). 2 KiB, so a caller should only reach for it when the
 * predictor it names is really about to read one of those tables. */
void zpse_lane_seed_table(ZpseLaneState *s, uint8_t initial_literal,
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
