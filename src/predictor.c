#include "zpse/predictor.h"
#include "zpse/hot.h"

#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Histories only: 16 bytes, the part of the entry state that every tile's
 * classify and decode depend on. Keeping it separate lets a caller
 * establish the entry state without any table work (there is none left). */
void zpse_lane_reset_hist(ZpseLaneState *s, uint8_t initial_literal)
{
    if (s == NULL) {
        return;
    }
    memset(s->last, initial_literal, 16u);
}

void zpse_lane_reset(ZpseLaneState *s, uint8_t initial_literal)
{
    zpse_lane_reset_hist(s, initial_literal);
}

/* zpse_lane_reset_for writes both histories with one 16-byte memset starting
 * at s->last, which is exact only while the two 8-byte arrays are adjacent. */
_Static_assert(offsetof(ZpseLaneState, second_last) ==
               offsetof(ZpseLaneState, last) + 8u,
               "last/second_last must be adjacent for the 16-byte reset");

ZPSE_HOT void zpse_lane_reset_for(ZpseLaneState *s, uint8_t initial_literal,
                           uint8_t predictor_id)
{
    (void)predictor_id;
    if (s == NULL) {
        return;
    }
    /* last[] and second_last[] are adjacent in ZpseLaneState, so one
     * 16-byte memset writes the whole entry state. Inlined from
     * zpse_lane_reset_hist: that is a plain O2 function and this one is
     * ZPSE_HOT/O3, so gcc will not inline across the mismatch and every
     * classify was paying the call. */
    memset(s->last, initial_literal, 16u);
}

/* zpse_predict_block and zpse_predict_update were removed: no callers in
 * src/ or tests/. They were the scalar batch (snapshot-semantics) reference
 * generator and its table learner; tile reconstruction uses the online loop
 * in tile.c and bit-exact SIMD formulas live in vector.h. */
