#ifndef ZPSE_DELTA264_H
#define ZPSE_DELTA264_H

/* ZPSE transform extension -- ID 21, DELTA2_64.
 *
 * 64-bit modular second-delta, followed by an 8-way byte transpose.
 * Length-preserving and lossless: forward and inverse both consume n bytes
 * and produce n bytes. The seed (x[0]) travels inline in the stream, so there
 * is no side metadata and no header.
 *
 * All three entry points are deterministic and block-local. Status codes:
 *   ZPSE_D264_OK      0   success
 *   ZPSE_D264_BAD_ARG (-1) null src/dst
 *   ZPSE_D264_SHORT  (-2) dst_cap < n
 *
 * zpse_d264_gate returns 1 (accept / commit) or 0 (reject / decline). */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZPSE_D264_OK       0
#define ZPSE_D264_BAD_ARG (-1)
#define ZPSE_D264_SHORT   (-2)

/* Transpose the count = n/8 full 8-byte words, then copy the n%8 tail bytes
 * verbatim to the END. Output layout (see README.txt):
 *   [plane 0: count bytes][plane 1: count bytes] ... [plane 7: count bytes]
 *   [tail: n%8 bytes] */
int zpse_d264_forward(const uint8_t *src, uint32_t n,
                      uint8_t *dst, uint32_t dst_cap);
int zpse_d264_inverse(const uint8_t *src, uint32_t n,
                      uint8_t *dst, uint32_t dst_cap);

/* Content gate, evaluated on the ORIGINAL input bytes (before transform):
 * accept iff count >= 4 and any of
 *   zero_dd_count >= dd_count / 4
 *   second_delta_bit_width + 4 <= first_delta_bit_width
 *   second_delta_bit_width <= 16                 (B/4, B = 64)
 * See README.txt for the exact definition of the two bit widths. */
int zpse_d264_gate(const uint8_t *src, uint32_t n);

/* ---- Runtime capability / test hooks ---------------------------------- *
 * These are exported so the parity suite can force each implementation and
 * compare it byte-for-byte against the scalar oracle. Normal callers use the
 * three functions above, which dispatch at runtime. */
int zpse_d264_has_sse2(void);
int zpse_d264_has_avx2(void);
int zpse_d264_forward_scalar(const uint8_t *src, uint32_t n,
                             uint8_t *dst, uint32_t dst_cap);
int zpse_d264_forward_sse2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap);
int zpse_d264_forward_avx2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap);
int zpse_d264_inverse_scalar(const uint8_t *src, uint32_t n,
                             uint8_t *dst, uint32_t dst_cap);
int zpse_d264_inverse_sse2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap);
int zpse_d264_inverse_avx2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_DELTA264_H */
