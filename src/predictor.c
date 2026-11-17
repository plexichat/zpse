#include "zpse/predictor.h"
#include "zpse/hot.h"

#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

void zpse_lane_o2_use(ZpseLaneState *s)
{
    if (s->o2_ready != 0) {
        return;
    }
    if (s->o2 != 0) {
        memset(s->o2, 0, (size_t)8u * 4096u * 4u);
    }
    s->o2_ready = 1;
}

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
/* The ORDER2 {value, confidence, generation} helpers that used to sit here
 * died with zpse_predict_block/zpse_predict_update (no callers). The entry
 * layout they encoded is documented in predictor.h on ZpseLaneState::o2. */

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
         * no entry can be mistaken for one this tile wrote. The table is
         * wiped once per block or trial, where the generation restarts,
         * and only if that block or trial reaches ORDER2 at all -- see
         * zpse_lane_o2_use. */
    }
}

void zpse_lane_reset(ZpseLaneState *s, uint8_t initial_literal)
{
    zpse_lane_reset_hist(s, initial_literal);
    zpse_lane_seed_table(s, initial_literal, (uint8_t)ZPSE_P_ORDER1);
    zpse_lane_seed_table(s, initial_literal, (uint8_t)ZPSE_P_ORDER2);
}

ZPSE_HOT void zpse_lane_reset_for(ZpseLaneState *s, uint8_t initial_literal,
                           uint8_t predictor_id)
{
    if (s == NULL) {
        return;
    }
    /* last[] and second_last[] are adjacent in ZpseLaneState, so one
     * 16-byte memset writes the whole entry state. Inlined from
     * zpse_lane_reset_hist: that is a plain O2 function and this one is
     * ZPSE_HOT/O3, so gcc will not inline across the mismatch and every
     * classify was paying the call. */
    memset(s->last, initial_literal, 16u);
    /* Only ORDER1 (id 3) seeds a table; for every other id
     * zpse_lane_seed_table is a no-op. The nine ids the encoder classifies
     * (0,1,2,5,6,7,8,10,11 -- seg_screen_pred[]) read no table at all, so
     * this guard keeps their path call-free. The decoder can still pass 3
     * (ORDER1 from the stream), which is why the call is guarded, not gone. */
    if (predictor_id == (uint8_t)ZPSE_P_ORDER1) {
        zpse_lane_seed_table(s, initial_literal, predictor_id);
    }
}

/* zpse_stride_of died with zpse_predict_block below (no callers). */

/* zpse_predict_block removed: grep across src/ and tests/ found no callers.
 * It was the scalar batch (snapshot-semantics) reference generator; tile
 * reconstruction uses the online loop in tile.c and bit-exact SIMD formulas
 * live in vector.h. */

/* zpse_predict_update removed: grep across src/ and tests/ found no callers.
 * It was the ORDER1/ORDER2 table learner for the batch generator; the live
 * online update loop is in tile.c. */
