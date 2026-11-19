#ifndef ZPSE_TRANSFORM_H
#define ZPSE_TRANSFORM_H

/* Coordinate transforms. Forward = encode-side split,
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

/* The same transform for planes the caller has just laid out itself: the
 * segment lengths are the stride formula's and the buffers are disjoint by
 * construction, so the two validation passes here (a per-segment comparison
 * plus 16 + 120 overlap tests at w = 16) are skipped. Everything cheap stays
 * -- width, the null checks, the seed requirement -- so a caller that got the
 * shape wrong still cannot write outside its planes. Internal to the encoder:
 * block.c's three staging sites are the only callers, and the result is
 * byte-for-byte what zpse_transform_forward produces for the same
 * arguments. */
int zpse_transform_forward_planes(const ZpseTransformMeta *meta,
                                  uint8_t seg_count, const uint8_t *in,
                                  uint32_t n, uint8_t *segs[16],
                                  const uint32_t seg_lens[16]);

/* In-place delta/xor prefix chain over one contiguous plane already staged
 * by the forward transpose: p[j] = p[j] - p[j-1] (delta != 0) or
 * p[j] ^ p[j-1] (delta == 0), with p[-1] == seed. This is the same kernel the
 * full forward transform uses, so chaining a plain-staged plane produces the
 * same bytes as staging it with the chain in the first place. */
void zpse_plane_chain_inplace(uint8_t *p, uint32_t n, uint8_t seed, int delta);

/* Inverse: interleave W segment buffers into native output. Handles
 * non-divisible tails (scalar epilogue). */
int zpse_transform_inverse(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *segs[16], const uint32_t seg_lens[16],
                           uint8_t *out, uint32_t n);

/* Internal inverse entry for the decoder, which has already validated the
 * segment sizes and plane disjointness on the same arguments. It skips the
 * duplicate `zpse_transform_validate_sizes` and the O(w^2) plane-overlap
 * passes that zpse_transform_inverse repeats; every cheap check (nulls, n
 * range, width agreement, seed requirement) stays, and on a 4 KiB page with
 * 64-byte-aligned planes it takes a staging-free direct path (see
 * ZPSE_XFORM_PAGE_FAST in transform.c). Byte-for-byte identical output.
 * Only block.c's decode-side call should use this: a caller that has not
 * validated must use zpse_transform_inverse. */
int zpse_transform_inverse_trusted(const ZpseTransformMeta *meta,
                                   uint8_t seg_count,
                                   const uint8_t *segs[16],
                                   const uint32_t seg_lens[16], uint8_t *out,
                                   uint32_t n);

#endif /* ZPSE_TRANSFORM_H */
