#ifndef ZPSE_SEGMENT_H
#define ZPSE_SEGMENT_H

/* Segment encode/decode (SPEC Sec. 6). A segment payload is a 16-B
 * header followed by tile_count prefixed tile records. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"
#include "predictor.h"

int zpse_segment_decode(ZpseDecoder *d, ZpseLaneState *lanes,
                        const uint8_t *payload, uint32_t payload_len,
                        uint8_t *decoded_out, uint32_t decoded_cap,
                        uint32_t *decoded_len_out);

/* Encoder-side: transform-plane bytes -> segment payload. Returns
 * payload bytes written or negative status. Choice of per-tile
 * (predictor, mask, residual) modes is made internally (PSE SELECT).
 *
 * byte_budget bounds the payload this call may commit: 0 means no bound,
 * otherwise the encoder stops as soon as the bytes it has already
 * committed would exceed it and returns PSE_OUTPUT_OVERFLOW. The bound is
 * tested before a tile is started and `pos` only ever grows, so no tile
 * is ever truncated and the bytes written are a prefix of the unbounded
 * output. It exists so a caller that already knows a larger total cannot
 * win -- block.c's transform sweep against the raw envelope does -- can
 * abandon a hopeless trial without paying for the rest of the tiles. */
int zpse_segment_encode(ZpseEncoder *e, ZpseLaneState *lanes,
                        const uint8_t *plane, uint32_t plane_len,
                        uint8_t initial_literal,
                        uint8_t *payload_out, uint32_t payload_cap,
                        uint32_t byte_budget);

#endif /* ZPSE_SEGMENT_H */
