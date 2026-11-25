#ifndef ZPSE_BS32_H
#define ZPSE_BS32_H

/* ZPSE transform extension, ID 13: BITSHUFFLE_32 (W = 4 B, B = 32 bit).
 *
 * Standalone, self-contained module for the integration agent. The scalar
 * C routines are the normative oracle; the 128-bit SSSE3 and 256-bit AVX2
 * kernels are byte-identical fast paths selected at runtime.
 *
 * Layout (in -> out), with count = n/4, tail = n%4,
 * pbytes = ceil(count/8):
 *
 *   byte offset 0          : plane 0  (bit 0 of words 0..count-1, LSB first)
 *   byte offset pbytes     : plane 1
 *   ...
 *   byte offset 31*pbytes  : plane 31 (bit 31 of each word)
 *   byte offset 32*pbytes  : the `tail` unshuffled input bytes
 *
 * Plane p, byte q, bit i (i LSB-first) = bit p of word q*8+i.
 * Padding bits in the final byte of each plane (i >= count%8) are ZERO
 * on forward and ignored on inverse. No metadata is stored: the geometry
 * is derived from n alone. count divisible by 8 => output length == n.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes shared by all three entry points (negative = reject/error). */
#define ZPSE_BS32_OK          (0)
#define ZPSE_BS32_BAD_ARG     (-1)
#define ZPSE_BS32_CAPACITY    (-2)
#define ZPSE_BS32_OVERFLOW    (-3)
#define ZPSE_BS32_GATE_REJECT (-4)

/* Compute the output length. Returns ZPSE_BS32_OK and stores it in *out_len,
 * or a negative status (overflow) without storing. */
int zpse_bs32_output_len(uint32_t n, uint32_t *out_len);

/* Forward: native bytes -> bit-shuffled planes + tail. Writes exactly
 * zpse_bs32_output_len(n) bytes. Returns 0 on success, negative on error. */
int zpse_bs32_forward(const uint8_t *in, uint32_t n,
                      uint8_t *out, uint32_t out_cap);

/* Inverse: bit-shuffled planes + tail -> native bytes. Writes exactly n
 * bytes. Returns 0 on success, negative on error. */
int zpse_bs32_inverse(const uint8_t *in, uint32_t n,
                      uint8_t *out, uint32_t out_cap);

/* Gate: 0 = admit the transform for this block, negative = reject.
 * See README.txt / bitshuffle32.c for the exact rule. */
int zpse_bs32_gate(const uint8_t *in, uint32_t n);

/* Explicit, fully-validated per-path entry points. These exist so tests can
 * assert scalar == SSSE3 == AVX2 byte-for-byte. The *_sse / *_avx2 routines
 * transparently fall back to the scalar kernel on non-x86 builds. */
int zpse_bs32_forward_scalar(const uint8_t *in, uint32_t n,
                             uint8_t *out, uint32_t out_cap);
int zpse_bs32_inverse_scalar(const uint8_t *in, uint32_t n,
                             uint8_t *out, uint32_t out_cap);
int zpse_bs32_forward_sse(const uint8_t *in, uint32_t n,
                          uint8_t *out, uint32_t out_cap);
int zpse_bs32_inverse_sse(const uint8_t *in, uint32_t n,
                          uint8_t *out, uint32_t out_cap);
int zpse_bs32_forward_avx2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap);
int zpse_bs32_inverse_avx2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap);

/* Runtime capability bits (local, cached CPUID; honours ZPSE_FORCE_SCALAR).
 * Bit layout matches include/zpse/vector.h so the integrator may alias this
 * to zpse_cpu_caps(): bit0 SSE2, bit1 SSSE3, bit2 AVX2. */
uint32_t zpse_bs32_cpu_caps(void);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_BS32_H */
