#ifndef ZPSE_SEGMENT_H
#define ZPSE_SEGMENT_H

/* Segment encode/decode (SPEC Sec. 6). A segment payload is a 16-B
 * header followed by tile_count prefixed tile records -- or, for a plane
 * the encoder coded with the dense form (Sec. 6.3), a sequence of dense
 * records and no tiles. */

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
 * abandon a hopeless trial without paying for the rest of the tiles.
 *
 * flat_abort_min is a second, weaker bound for the same caller: 0 disables
 * it, otherwise the encoder abandons the trial (same status) as soon as it
 * has covered that many input bytes with not one tile paying for itself --
 * every tile it wrote cost at least as many bytes as the bytes it codes.
 * The two counters live in the encoder (e->flat_covered / e->flat_paid) so
 * the caller can make the threshold a property of the whole trial while the
 * test itself happens per tile; the caller resets them before each trial.
 * Unlike byte_budget this is a heuristic, not a bound: it only fires for a
 * trial that cannot beat the raw envelope anyway, and a trial it abandons
 * is reported exactly like one the budget stopped.
 *
 * Two shapes are written in the dense form (Sec. 6.3), and only when that
 * is strictly smaller than the tile records they replace: a plane that is
 * one byte value end to end (12 bytes per 4 KiB against 10 per 256-byte
 * tile, so the last tile of a short plane keeps its record), and a plane
 * of at most sixteen distinct byte values (one palette stream per 4 KiB,
 * ALL_EXCEPTION mask, no prediction to reconstruct). The dense records
 * reconstruct exactly the plane they replace, tile_count is 1, and every
 * other plane's bytes are unchanged. */
/* Prescreen-only size estimate of one transformed plane (see segment.c). */
uint32_t zpse_segment_proxy_cost(const uint8_t *plane, uint32_t plane_len);

int zpse_segment_encode(ZpseEncoder *e, ZpseLaneState *lanes,
                        const uint8_t *plane, uint32_t plane_len,
                        uint8_t initial_literal,
                        uint8_t *payload_out, uint32_t payload_cap,
                        uint32_t byte_budget, uint32_t flat_abort_min);

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
