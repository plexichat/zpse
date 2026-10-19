#ifndef ZPSE_DENSE_H
#define ZPSE_DENSE_H

/* Dense segment record (SPEC_V1.md Sec. 6.3).
 *
 * The tile framing pays one tile header, one mask and one residual header
 * per 256 bytes. A dense record codes a whole segment plane with ONE
 * predictor, ONE mask over the plane and ONE residual stream, so the same
 * bytes cost a dozen headers per plane instead of a dozen per tile. The
 * record sits where the tile records would: directly after the 16-byte
 * segment header, and a segment that uses the form uses nothing else.
 *
 * A dense record is marked by bit 6 of its flag byte -- the bit tile
 * records must leave clear (ZPSE_TILE_FLAG_RESERVED_BIT), so no stream
 * that decoded before can be read as a dense record and no dense record
 * is reachable as a tile.
 *
 * The record AREA of such a segment is a sequence of these records,
 * back to back, each coding at most ZPSE_DENSE_MAX_LEN bytes and exactly
 * one when the plane fits; the segment's decoded size is their sum. A
 * 64 KiB segment is therefore sixteen 4 KiB records, which is what lets
 * a block encoded as one segment use the form at all. Each record is
 * self-contained: its predictions never read a byte another record
 * reconstructed.
 *
 * `zpse_segment_encode` emits the form for two shapes in v1 (segment.c):
 * a plane that is a single byte value end to end, where 12 bytes per 4 KiB
 * beats 10 per 256-byte tile; and a plane of at most sixteen distinct byte
 * values, coded as ALL_EXCEPTION + PALETTE_BITS, which the decoder below
 * expands without running a predictor at all (see dense_record). This
 * decoder is not limited to those shapes: it accepts any conforming
 * record, which is what the reference-built cases in tests/test_dense.c
 * exercise.
 *
 * Layout (little-endian), offset from the start of the record:
 *   0  u16  rec_len        bytes after this field (header tail + payloads)
 *   2  u8   flags          bit 6 = marker; bits 0..2 = mask mode (0, 1,
 *                          2 or 6); bits 3..5 = residual mode (0..3);
 *                          bit 7 MUST be 0
 *   3  u8   predictor_id   0..11, not 3 (ORDER1) / 4 (ORDER2) in v1
 *   4  u8   constant_C     used iff predictor_id == 9
 *   5  u8   reserved       MUST be 0
 *   6  u16  exception_count K
 *   8  u16  mask_len
 *  10  u16  residual_len
 *  12  ...  mask payload (mask_len bytes), then residual payload
 *
 * v1 limits: planes of at most ZPSE_DENSE_MAX_LEN bytes and residual
 * modes 0..3 (mode 4 needs the predictions up front, which a dense
 * record only knows while reconstructing). Both limits are reported as
 * PSE_UNSUPPORTED_FEATURE; they are reserved space, not decoder bugs. */

#include <stdint.h>

#include "types.h"

#define ZPSE_DENSE_MARKER      (UINT8_C(1) << 6)
#define ZPSE_DENSE_HEADER_SIZE 12u
#define ZPSE_DENSE_MAX_LEN     4096u

/* Decode one dense record for one segment. `h` is the already-parsed
 * segment header; `payload`/`payload_len` are the segment payload
 * (header included). scratch must hold at least h->segment_decoded_size
 * bytes (the caller passes the 8 KiB combinadic region).
 * Returns PSE_OK, or the same class of status a tile record would:
 * PSE_BAD_TILE_SIZE, PSE_BAD_MASK, PSE_BAD_RESIDUAL,
 * PSE_UNSUPPORTED_FEATURE, PSE_OUTPUT_OVERFLOW.
 *
 * On success *all_const is 1 iff every record reconstructed one byte value
 * end to end and all of them agreed on it, which is what the dense constant
 * form written for a one-value plane decodes to; *const_byte is that value
 * then. The block layer uses it to checksum such a plane arithmetically. */
int zpse_dense_decode(const ZpseSegmentHeader *h, const uint8_t *payload,
                      uint32_t payload_len, uint8_t *out, uint32_t out_cap,
                      uint32_t *out_len, uint8_t *scratch, int *all_const,
                      uint8_t *const_byte);

#endif /* ZPSE_DENSE_H */
