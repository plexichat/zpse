#ifndef ZPSE_WORDXOR64_H
#define ZPSE_WORDXOR64_H

/* ZPSE transform extension -- ID 19, WORD_XOR_64.
 *
 * Length-preserving, lossless, block-local, zero metadata. The XOR seed is
 * inline: y[0] = x[0], so nothing beyond the transformed bytes is stored.
 *
 * Public API
 * ----------
 *   zpse_wx64_forward(src, n, dst, cap)
 *   zpse_wx64_inverse(src, n, dst, cap)
 *       src : input block, n bytes (n == 0 is legal).
 *       dst : output block, at least n writable bytes.
 *       cap : destination capacity in bytes. 0 means "caller guarantees n".
 *             If cap != 0 and cap < n, WX64_ERR_ARG is returned.
 *       src and dst must NOT overlap (no in-place transform is supported).
 *       Return WX64_OK (0) on success, WX64_ERR_ARG (-1) on bad arguments.
 *
 *   zpse_wx64_gate(src, n)
 *       Returns 1 when the XOR stream shows redundancy worth transforming,
 *       0 otherwise (including n < 64, the 8-word minimum). The decision is
 *       the OR of three signals measured on y[i] = x[i] ^ x[i-1]:
 *         zero_xor_count >= count/8
 *         zero bytes per XOR word >= 4
 *         popcount per XOR word <= 16
 *       High-entropy input fails all three and is rejected.
 *
 *   zpse_wx64_gate_strong(src, n)
 *       The spec's stricter high-confidence tier: nonzero XOR bytes <= 60%.
 *       Returns 1/0. This is a diagnostic/extra tier; the main gate above is
 *       the OR rule and does not require it.
 *
 * Test hooks
 * ----------
 *   The *_scalar / *_sse2 / *_avx2 entry points below are byte-identical
 *   implementations selected by the public entry points via runtime dispatch.
 *   They are exported only so tests can compare them directly; integrators
 *   should call the three public functions above.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WX64_OK        0
#define WX64_ERR_ARG  (-1)

int zpse_wx64_forward(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_inverse(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_gate(const uint8_t *src, uint32_t n);
int zpse_wx64_gate_strong(const uint8_t *src, uint32_t n);

int zpse_wx64_forward_scalar(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_forward_sse2(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_forward_avx2(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_inverse_scalar(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_inverse_sse2(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
int zpse_wx64_inverse_avx2(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_WORDXOR64_H */
