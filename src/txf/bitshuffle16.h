#ifndef ZPSE_BITSHUFFLE16_H
#define ZPSE_BITSHUFFLE16_H

/* ZPSE transform extension -- ID 12, BITSHUFFLE_16 (W = 2 bytes, B = 16).
 *
 * Self-contained module (no dependency on the zpse tree). The integration
 * agent owns wiring this into the codec; this file only defines the wire
 * format, the three public entry points, and per-path test hooks.
 *
 * All multibyte words are read/written with explicit little-endian byte
 * loads/stores. No unaligned casts, no floating point.
 *
 * ---------------------------------------------------------------------------
 * WIRE FORMAT (length-preserving for every gate-admitted codec block)
 * ---------------------------------------------------------------------------
 * Let count = n / 2 (number of 16-bit words) and tail = n & 1 (0 or 1).
 * Let pb = ceil(count / 8) = (count + 7) / 8.
 *
 *   out_len = 16 * pb + tail
 *
 * The output is 16 bit-planes followed by the (optional) odd tail byte:
 *
 *   plane p occupies out[p * pb .. p * pb + pb - 1], p = 0 (LSB) .. 15.
 *   plane p byte j, bit b  =  bit p of word (8*j + b),
 *                            or 0 when 8*j + b >= count (padding = ZERO).
 *   out[16 * pb] = in[2*count] when tail == 1.
 *
 * Inverse reads exactly out_len bytes and writes exactly n bytes. `n` is the
 * ORIGINAL native length and is passed unchanged to both directions -- that
 * is what makes the format metadata-free and unambiguous even when count is
 * not a multiple of 8. For codec-realistic n (a multiple of 256, hence
 * count a multiple of 64) out_len == n exactly: the transform is then
 * length-preserving in the buffer-size sense as well.
 *
 * ---------------------------------------------------------------------------
 * GATE
 * ---------------------------------------------------------------------------
 * zpse_bs16_gate() returns 0 (admit) or a negative status (reject).
 * It samples up to 256 evenly spaced 16-bit words (unsigned little-endian),
 * computes:
 *   required_bits = bit_width( OR_over_samples(value - min_sampled) )
 *   varying_bits  = popcount( aggregate_or ^ aggregate_and )
 * and admits iff (n >= 256, count >= 16) AND
 *   (required_bits <= B/2 = 8)  OR  (varying_bits <= 3*B/4 = 12).
 */

#include <stdint.h>

#define ZPSE_BS16_W 2u
#define ZPSE_BS16_B 16u
#define ZPSE_BS16_PLANES 16u

/* Return codes (negative on error, 0 on success). */
#define ZPSE_BS16_OK         0
#define ZPSE_BS16_BAD_ARG   (-1)  /* NULL pointer */
#define ZPSE_BS16_TOO_SMALL (-2)  /* cap smaller than required size */
#define ZPSE_BS16_REJECT    (-3)  /* gate: data does not qualify */

/* Number of bytes the forward transform writes for a native length n. */
uint32_t zpse_bs16_out_len(uint32_t n);

/* Public API. `n` is the ORIGINAL native byte length in both directions.
 * forward: reads n bytes from in, writes out_len bytes, needs cap >= out_len.
 * inverse: reads out_len bytes from in, writes n bytes, needs cap >= n. */
int zpse_bs16_forward(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_inverse(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_gate(const uint8_t *in, uint32_t n);

/* --- Per-path test/integration hooks -------------------------------------
 * Same signature/semantics as the public entry points but each forces one
 * implementation. All paths are byte-for-byte identical. The default
 * zpse_bs16_forward/inverse dispatch among them via runtime CPU detection.
 * The scalar path is the portable format oracle. */
int zpse_bs16_forward_scalar(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_inverse_scalar(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_forward_sse2(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_inverse_sse2(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_forward_ssse3(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_inverse_ssse3(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_forward_avx2(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);
int zpse_bs16_inverse_avx2(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap);

/* Bitmask of detected features (mirrors ZpseCpuCaps bits from zpse/vector.h:
 * bit0 = SSE2, bit1 = SSSE3, bit2 = AVX2). */
uint32_t zpse_bs16_cpu_caps(void);

#endif /* ZPSE_BITSHUFFLE16_H */
