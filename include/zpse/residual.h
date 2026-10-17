#ifndef ZPSE_RESIDUAL_H
#define ZPSE_RESIDUAL_H

/* Residual codecs (SPEC Sec. 9). Exclusion mapping is the shared
 * primitive; per-mode packers handle the payload layout. All decoders
 * unpack into exclusion-residual bytes exc_out[K] in [0,254] (mode 1
 * yields raw actual bytes instead — see below), then tile.h inverts. */

#include <stddef.h>
#include <stdint.h>

typedef enum ZpseResidualMode {
    ZPSE_R_EXCLUSION_U8   = 0,
    ZPSE_R_RAW_ACTUAL     = 1,
    ZPSE_R_PALETTE_BITS   = 2,
    ZPSE_R_PACKED_BASE_M  = 3,
    ZPSE_R_SIGNED_DELTA   = 4
} ZpseResidualMode;

/* Exclusion primitives (Sec. 9). */
static inline uint8_t zpse_exclusion_forward(uint8_t x, uint8_t p_hat) {
    return (x < p_hat) ? x : (uint8_t)(x - 1u);
}
static inline uint8_t zpse_exclusion_inverse(uint8_t r, uint8_t p_hat) {
    return (r < p_hat) ? r : (uint8_t)(r + 1u);
}
static inline int zpse_signed_delta_encode(uint8_t x, uint8_t p_hat, int *d_out) {
    int t = ((int)x - (int)p_hat + 128) & 0xFF;
    int d = t - 128;
    *d_out = d;
    return (d == 0) ? -1 : 0;
}

/* Chunk size G per M (Sec. 9.2 table). Returns 0 for illegal M. */
uint8_t zpse_base_m_group(uint8_t m);

/* Smallest legal mode-3 M covering exclusion residuals up to maxv
 * (M >= maxv+1, M in {3,5,6,7,9,10,11,12,13,14,15}). Returns 0 when no
 * allowed M covers maxv, i.e. mode 3 is infeasible. */
uint8_t zpse_residual_mode3_pick_m(uint8_t maxv);

/* Generic decode: payload -> exc_out[K] (exclusion domain, [0,254]).
 * Mode 1 instead writes raw actual bytes to actual_out[K] ([0,255]).
 * pred_at_exc[K]: predictions at exception positions (needed by modes
 * that store deltas). Returns 0 or PSE_BAD_RESIDUAL. */
int zpse_residual_decode(uint8_t mode, const uint8_t *payload,
                         uint16_t payload_len, uint16_t k,
                         const uint8_t *pred_at_exc,
                         uint8_t *exc_out, uint8_t *actual_out);

/* Generic encode: returns payload bytes written or negative status.
 * For mode 1, actual_in[K] are raw bytes; otherwise exc_in[K] are
 * exclusion residuals. pred_at_exc as above. */
int zpse_residual_encode(uint8_t mode, const uint8_t *exc_in,
                         const uint8_t *actual_in, const uint8_t *pred_at_exc,
                         uint16_t k, uint8_t *payload_out,
                         uint16_t payload_cap);

/* Same as zpse_residual_encode, with the classification's certificate for
 * the mode-2 palette: `nu_hint` distinct exclusion residuals and their
 * `seen_hint` 256-bit set (bit v = value v is present). When the encoder
 * has already counted the distinct values of exc_in[0..k) -- tile packing
 * does, because the mode search needs the count anyway -- mode 2 can find
 * its first-appearance palette by scanning only up to the last NEW value
 * instead of all k. The certificate is verified before it is used (a
 * wrong one falls back to the exhaustive scan), so the payload and every
 * status are identical to zpse_residual_encode. */
int zpse_residual_encode_hint(uint8_t mode, const uint8_t *exc_in,
                              const uint8_t *actual_in,
                              const uint8_t *pred_at_exc, uint16_t k,
                              uint8_t *payload_out, uint16_t payload_cap,
                              uint16_t nu_hint, const uint32_t *seen_hint);

/* Payload-length predictors (encoder sizing). Exact for modes 0-3;
 * mode 4 reports the 8-bit worst case 1+K (exact submode needs
 * actual+pred, unavailable to this signature). */
int zpse_residual_payload_len(uint8_t mode, uint16_t k,
                              const uint8_t *exc_in, uint16_t *len_out);

#endif /* ZPSE_RESIDUAL_H */
