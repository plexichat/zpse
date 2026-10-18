#ifndef ZPSE_TILE_H
#define ZPSE_TILE_H

/* Tile encode/decode (SPEC Sec. 6.2, 7-10). A tile is:
 * [u16 size][8-B hdr][C?][mask][residual]. This header covers framing,
 * validation, mode extraction, and the scalar reconstruct path. */

#include <stddef.h>
#include <stdint.h>
#include "predictor.h"
#include "types.h"

#define ZPSE_TILE_FLAG_MASK_MODE_MASK  UINT8_C(0x07)
#define ZPSE_TILE_FLAG_RES_MODE_SHIFT  3u
#define ZPSE_TILE_FLAG_RES_MODE_MASK   UINT8_C(0x38)
#define ZPSE_TILE_FLAG_RESERVED_BIT    (UINT8_C(1) << 6)
#define ZPSE_TILE_FLAG_RESET           (UINT8_C(1) << 7)

static inline uint8_t zpse_tile_mask_mode(uint8_t flags) {
    return (uint8_t)(flags & ZPSE_TILE_FLAG_MASK_MODE_MASK);
}
static inline uint8_t zpse_tile_res_mode(uint8_t flags) {
    return (uint8_t)((flags & ZPSE_TILE_FLAG_RES_MODE_MASK) >>
                     ZPSE_TILE_FLAG_RES_MODE_SHIFT);
}
static inline uint8_t zpse_tile_flags_make(uint8_t mask_mode,
                                           uint8_t res_mode, int reset) {
    return (uint8_t)((mask_mode & 0x07u) | ((res_mode & 0x07u) << 3) |
                     (reset ? ZPSE_TILE_FLAG_RESET : 0u));
}

/* Read the eight record header bytes into `th`. No validation: this is
 * the field load a caller does before it can ask zpse_tile_validate any
 * question, exposed so a caller that already holds the record (the
 * segment decoder's inline fast path) does not have to repeat it. */
void zpse_tile_read_header(const uint8_t *record, struct ZpseTileHeader *th);

/* Validate a tile header against (nt_expected, bytes_remaining).
 * Derives K, checks the size equation and reserved bits/modes. */
int zpse_tile_validate(const struct ZpseTileHeader *th, uint16_t nt_expected,
                       uint32_t bytes_remaining, uint16_t *nt_out,
                       uint16_t *k_out);

/* Decode one tile's record (record points at the 8-B header, record_len
 * == tile_record_size). Appends Nt transformed bytes at out (capacity
 * checked). Updates lane state. seg_base/seg_pos back STRIDE predictors
 * (Sec. 7). Returns bytes of `out` produced (== Nt) or negative status.
 * comb_scratch: 8 KiB arena region for mode-5 masks. */
int zpse_tile_decode(ZpseLaneState *lanes, const uint8_t *record,
                     uint32_t record_len, uint8_t initial_literal,
                     int tile_reset, const uint8_t *seg_base,
                     uint32_t seg_pos, uint8_t *out, uint32_t out_cap,
                     uint8_t *comb_scratch);

/* Encode one tile (encoder-side). Returns record bytes written
 * (excl. the u16 prefix) or negative status. */
int zpse_tile_encode(ZpseLaneState *lanes, const uint8_t *actual, uint16_t nt,
                     uint8_t predictor_id, uint8_t mask_mode,
                     uint8_t res_mode, uint8_t initial_literal,
                     int tile_reset, const uint8_t *seg_base,
                     uint32_t seg_pos, uint8_t *record_out,
                     uint32_t record_cap, uint8_t *comb_scratch);

/* Classification result for one tile: which positions the predictor
 * misses, and what it predicted there. Produced once per (tile,
 * predictor) and reused by every (mask_mode, res_mode) packing trial,
 * which is what makes the encoder's mode search cheap. */
typedef struct ZpseTileClass {
    uint32_t words[8];     /* 256-bit exception mask, bit p = exception */
    uint16_t k;            /* number of exceptions (== popcount) */
    uint16_t nu;           /* distinct exclusion-residual values seen,
                            * capped at 17: the palette mode is rejected
                            * above 16 entries, so a larger exact count
                            * could not change any packed byte and the
                            * dense classify stops counting there. */
    uint8_t  constant_C;   /* predictor 9 mode byte (encoder choice) */
    uint8_t  rmax;         /* largest exclusion residual seen */
    uint8_t  dmin;         /* signed-delta range, stored biased by +128 */
    uint8_t  dmax;
    uint32_t seen[8];      /* 256-bit set of exclusion residuals seen, cut
                            * to its lowest popcount(seen) == nu bits */
    uint8_t  at_pred[256]; /* predictions at exception positions, ascending */
    uint8_t  res_at_exc[256]; /* exclusion residuals at those positions:
                               * exclusion_forward(actual[p], at_pred[j]).
                               * Filled by classify (and by tile_encode),
                               * so the packers read them instead of
                               * walking the tile a second time. */
    uint8_t  act_valid;    /* 1 iff act_at_exc[0..k) was filled by classify */
    uint8_t  act_at_exc[256]; /* actual[p] at the exception positions,
                               * ascending: the RAW_ACTUAL residual payload,
                               * produced while classifying so the packer
                               * does not have to gather it again. */
} ZpseTileClass;

/* Pick the cheapest legal (mask_mode, res_mode) pair for a finished
 * classification, without encoding anything. */
void zpse_tile_choose_modes(const ZpseTileClass *cl, uint16_t nt,
                            uint8_t *mask_mode_out, uint8_t *res_mode_out);

/* Predict + compare one tile against `actual`, filling cl and advancing
 * the lane state exactly as zpse_tile_decode will while decoding it.
 * Does not touch the output buffer. */
int zpse_tile_classify(ZpseLaneState *lanes, const uint8_t *actual,
                       uint16_t nt, uint8_t predictor_id,
                       uint8_t initial_literal, int tile_reset,
                       const uint8_t *seg_base, uint32_t seg_pos,
                       ZpseTileClass *cl);

/* Pack a classification into a wire record for one (mask_mode,
 * res_mode) pair. Returns record bytes written (excl. the u16 prefix)
 * or negative status. Does not touch lane state. */
int zpse_tile_pack(uint8_t predictor_id, const ZpseTileClass *cl,
                   uint16_t nt, const uint8_t *actual,
                   uint8_t mask_mode, uint8_t res_mode,
                   int tile_reset, uint8_t *record_out,
                   uint32_t record_cap, uint8_t *comb_scratch);

#endif /* ZPSE_TILE_H */
