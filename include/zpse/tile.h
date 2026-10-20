#ifndef ZPSE_TILE_H
#define ZPSE_TILE_H

/* Tile encode/decode. A tile is:
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
    /* The three per-exception arrays hold up to 256 entries and are written
     * by 8-byte stores that may run up to 7 bytes past the last real entry
     * (the dense compaction and tile_gather_act_avx2 both store a whole
     * 64-bit group at a time), so each has 8 bytes of slack. */
    uint8_t  at_pred[264]; /* predictions at exception positions, ascending */
    uint8_t  res_at_exc[264]; /* exclusion residuals at those positions:
                               * exclusion_forward(actual[p], at_pred[j]).
                               * Filled by classify (and by tile_encode),
                               * so the packers read them instead of
                               * walking the tile a second time. */
    uint8_t  act_valid;    /* 1 iff act_at_exc[0..k) was filled by classify */
    uint8_t  act_at_exc[264]; /* actual[p] at the exception positions,
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

/* Same classification without cl->at_pred, the predictions at the
 * exception positions. It is the one product the encoder usually does not
 * read: a packed record takes its bytes from cl->res_at_exc or
 * cl->act_at_exc, and the predictions only for PALETTE_BITS,
 * PACKED_BASE_M and SIGNED_DELTA. Measured over the eighteen corpora,
 * 82% of the tiles the encoder searches pack as RAW_ACTUAL, and the
 * at_pred compaction is ~66 cycles of a ~320-cycle exception pass, so
 * skipping it is worth ~13% of a classify on a dense tile.
 *
 * Everything else -- words, k, constant_C, res_at_exc, act_at_exc, seen,
 * nu, rmax/dmin/dmax and the lane state -- is exactly what
 * zpse_tile_classify produces. at_pred is left untouched (not zeroed);
 * a caller that packs a mode which reads it must first call
 * zpse_tile_fill_at_pred_reset. Adding an entry point rather than
 * changing the existing one keeps zpse_tile_classify's contract (and
 * zpse_tile_encode, which passes through arbitrary modes) intact. */
int zpse_tile_classify_lean(ZpseLaneState *lanes, const uint8_t *actual,
                            uint16_t nt, uint8_t predictor_id,
                            uint8_t initial_literal, int tile_reset,
                            const uint8_t *seg_base, uint32_t seg_pos,
                            ZpseTileClass *cl);

/* Write cl->at_pred for a classification from zpse_tile_classify_lean: the
 * prediction at every exception position, ascending, byte for byte what
 * zpse_tile_classify would have written.
 *
 * Valid only for a tile classified with tile_reset != 0, where every lane
 * history starts at `initial_literal`; that is the discipline
 * zpse_segment_encode uses for every tile it writes (the per-tile reset
 * bit is always 1). It is a separate call so the cost lands only on the
 * tiles whose packed mode reads the predictions.
 *
 * Predictors 0, 1, 2, 5, 6, 7, 8, 9, 10 and 11 are supported -- the ten
 * the encoder classifies. ORDER1/ORDER2 return PSE_BAD_PREDICTOR: they
 * would have to rebuild their tables byte by byte, and the encoder never
 * asks for them. */
int zpse_tile_fill_at_pred_reset(uint8_t predictor_id, const uint8_t *actual,
                                 uint16_t nt, const uint8_t *seg_base,
                                 uint32_t seg_pos, uint8_t initial_literal,
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
