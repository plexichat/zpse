#ifndef ZPSE_TRANSFORM_H
#define ZPSE_TRANSFORM_H

/* Coordinate transforms (SPEC Sec. 5). Forward = encode-side split,
 * inverse = decode-side interleave. All routines bounds-checked,
 * no allocation, scalar-correct with SIMD fast paths in vector.h. */

#include <stddef.h>
#include <stdint.h>
#include "types.h"

int zpse_transform_width(uint8_t transform_id, uint8_t segment_count,
                         uint8_t *w_out);

int zpse_segment_lengths(uint32_t n, uint8_t w, uint32_t out_lens[16]);

/* Validate decoded segment sizes against the stride formula + clamping
 * rule (no empty segments, N < W forbidden). */
int zpse_transform_validate_sizes(uint8_t transform_id, uint8_t segment_count,
                                  uint32_t n, const uint32_t *seg_lens);

/* Forward: split native input into W segment buffers. */
int zpse_transform_forward(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *in, uint32_t n,
                           uint8_t *segs[16], const uint32_t seg_lens[16]);

/* Inverse: interleave W segment buffers into native output. Handles
 * non-divisible tails (scalar epilogue). */
int zpse_transform_inverse(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *segs[16], const uint32_t seg_lens[16],
                           uint8_t *out, uint32_t n);

/* Delta8 / Xor8 per-segment helpers (with explicit seed). */
void zpse_delta8_encode(const uint8_t *in, uint8_t *enc, uint32_t n, uint8_t seed);
void zpse_delta8_decode(const uint8_t *enc, uint8_t *out, uint32_t n, uint8_t seed);
void zpse_xor8_encode(const uint8_t *in, uint8_t *enc, uint32_t n, uint8_t seed);
void zpse_xor8_decode(const uint8_t *enc, uint8_t *out, uint32_t n, uint8_t seed);

#endif /* ZPSE_TRANSFORM_H */
