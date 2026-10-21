#ifndef ZPSE_MASK_H
#define ZPSE_MASK_H

/* Exception masks. Positions are ascending zero-based
 * byte indices within the tile. Dense output is a 256-bit bitmask as
 * 8x u32 words (bit p = word[p>>5] >> (p&31) & 1), matching the
 * combinadic reference decoder. */

#include <stddef.h>
#include <stdint.h>

typedef enum ZpseMaskMode {
    ZPSE_M_ALL_HIT   = 0,
    ZPSE_M_RAW_BITMAP = 1,
    ZPSE_M_SPARSE_U8  = 2,
    ZPSE_M_SPARSE_GAPS = 3,
    ZPSE_M_RUNS       = 4,
    ZPSE_M_ENUMERATIVE = 5,
    ZPSE_M_ALL_EXCEPTION = 6,
    /* No exception mask at all: the tile's payload is the tile itself with
     * the eighth bit of every byte dropped (Sec. 6.2, mode 7). Only a tile
     * whose bytes are all below 0x80 may use it -- the encoder checks, the
     * decoder cannot -- and it is the only mode whose mask payload is
     * empty while K is 0 AND the residual payload is non-empty: the
     * residual holds nt*7/8 packed bytes, and the predictor is CONST_ZERO
     * (the form predicts nothing). 224 bytes for a full 256-byte tile, so
     * a tile of ASCII text -- which no predictor here packs below its own
     * size -- codes at 1.143:1 for the price of a scan and a bit pack.
     * The bit layout is tile.h's (eight-byte groups, little endian). */
    ZPSE_M_ASCII_PACK = 7
} ZpseMaskMode;

/* Derive K from (mask_mode, exception_count, Nt) per Sec. 6.2.
 * Returns 0 on success with *k_out set; error code otherwise. */
int zpse_mask_derive_k(uint8_t mask_mode, uint8_t exception_count,
                       uint16_t nt, uint16_t *k_out);

/* Decode a mask payload into dense words[8] (zeroed by caller).
 * payload_len MUST equal the tile's mask_payload_size; nt/K already
 * validated. comb_scratch is the 8 KiB arena region (only mode 5 uses
 * it; may be NULL otherwise). */
int zpse_mask_decode(uint8_t mask_mode, const uint8_t *payload,
                     uint16_t payload_len, uint16_t nt, uint16_t k,
                     uint32_t words[8], uint8_t *comb_scratch);

/* Encode helper used by the encoder: dense words -> payload. Returns
 * payload bytes written or negative status. */
int zpse_mask_encode(uint8_t mask_mode, const uint32_t words[8],
                     uint16_t nt, uint16_t k, uint8_t *payload_out,
                     uint16_t payload_cap, uint8_t *comb_scratch);

/* Bit helpers. */
int zpse_mask_count_bits(const uint32_t words[8], uint16_t nt);
void zpse_mask_clear(uint32_t words[8]);
int zpse_mask_test(const uint32_t words[8], uint16_t pos);

#endif /* ZPSE_MASK_H */
