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
/* Prescreen-only size estimate of one transformed plane (see segment.c). */
uint32_t zpse_segment_proxy_cost(const uint8_t *plane, uint32_t plane_len);

int zpse_segment_encode(ZpseEncoder *e, ZpseLaneState *lanes,
                        const uint8_t *plane, uint32_t plane_len,
                        uint8_t initial_literal,
                        uint8_t *payload_out, uint32_t payload_cap,
                        uint32_t byte_budget);

/* The cheap nine-predictor screen zpse_segment_encode runs per tile,
 * exposed so a caller can ask how predictable a candidate plane is
 * without encoding it: ks[] receives the exact exception count of each
 * screened predictor (ids 3, 4 and 9 are unreachable sentinels). `a` is
 * the tile, `seg_base`/`seg_pos` the plane it sits in (the STRIDE
 * predictors read seg_base[seg_pos + p - s], and seg_pos == 0 means the
 * table's first tile, whose preceding plane bytes do not exist). */
void zpse_segment_prescreen(const uint8_t *a, uint16_t nt,
                            const uint8_t *seg_base, uint32_t seg_pos,
                            uint8_t initial_literal, uint16_t ks[12]);

#endif /* ZPSE_SEGMENT_H */
