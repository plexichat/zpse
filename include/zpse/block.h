#ifndef ZPSE_BLOCK_H
#define ZPSE_BLOCK_H

/* Block encode/decode. Envelope:
 * [32-B header][meta][segdir][payloads]. Includes RAW fallback. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"

/* Decode one block. in_len must be >= compressed_block_size taken from
 * the header; output must hold uncompressed_size bytes. Returns bytes
 * written (== uncompressed_size) or negative PseStatus. */
int zpse_block_decode(ZpseDecoder *d, const uint8_t *in, uint32_t in_len,
                      uint8_t *output, uint32_t out_cap,
                      uint32_t *out_len_out);

/* Encode one block (<= 65536 B). Selects transform + per-segment tiles,
 * falls back to RAW_BLOCK when compression does not pay. Returns block
 * bytes written or negative status. seq is block_sequence_number. */
int zpse_block_encode(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                      uint32_t seq, int is_last,
                      uint8_t *out, uint32_t out_cap);

/* Header-only helpers (validation without full decode). */
int zpse_block_probe(const uint8_t *in, uint32_t in_len,
                     uint32_t *compressed_size_out,
                     uint32_t *uncompressed_size_out);

#endif /* ZPSE_BLOCK_H */
