#ifndef ZPSE_DPCM1_H
#define ZPSE_DPCM1_H

/* DPCM1 -- a fixed-predictor, one-byte-per-delta whole-frame codec for
 * 16-bit little-endian samples (measured on the Silesia x-ray medical image).
 *
 * There is no mask, residual or predictor search: the predictor is always the
 * previous 16-bit sample and every delta that fits in a signed byte is stored
 * as exactly one byte. Rare out-of-range deltas go to a small sorted escape
 * table (sample position u32, true delta i16). A leading raw byte prefix (the
 * x-ray file's 240-byte header) is stored verbatim and prediction begins at
 * the first sample after it. Decode is a byte read, an SSE2 inclusive scan and
 * a scatter of the escapes -- no Huffman or transform stage.
 *
 * REGISTERED IN THE UNITY BUILD: src/dpcm1.c is included by
 * src/zpse_unity.c and built as part of the library.
 *
 * OPT-IN: nothing is emitted unless the caller sets ZPSE_ENC_DPCM1 (bit 17)
 * on zpse_compress_frame_ex. There is no runtime toggle. The content gate
 * accepts only data that looks like the measured x-ray: at least 90% of
 * sampled deltas fit int8 AND the mean sampled |delta| is at least 18 (smooth
 * data such as Silesia "mr" is already cheaper under PSE1 and must fall
 * through to the normal encoder).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZPSE_DPCM1_DECLINE 0
#define ZPSE_DPCM1_COMMIT  1

/* "DCM1" as a little-endian u32: bytes 'D','C','M','1'. */
#define ZPSE_DPCM1_MAGIC       UINT32_C(0x314D4344)
#define ZPSE_DPCM1_HEADER_SIZE UINT32_C(16)

/* WIRE FORMAT -- self-delimiting, all integers little-endian.
 *
 *   offset  size  field
 *   0       4     magic = "DCM1" (u32 LE 0x314D4344)
 *   4       4     n = original byte length (u32 LE); total samples = n/2
 *   8       2     first sample after the prefix, raw (u16 LE)
 *   10      2     prefix_len, even byte count of the verbatim prefix (u16 LE)
 *   12      4     n_escapes (u32 LE)
 *   16      p     the raw prefix, copied byte for byte from input[0..p)
 *   16+p    ns-1  signed int8 delta for samples 1..ns-1 after the prefix;
 *                 an escaped delta has a zero placeholder here
 *   16+p+n-1  6*ne escape table, sorted by strictly increasing position:
 *                 (delta index u32 LE, true delta i16 LE)
 *
 * ns = n/2 - prefix_len/2 is the sample count from the prefix onward, so the
 * total length is 16 + prefix_len + (ns-1) + 6*n_escapes. Decoding restores
 * the prefix verbatim, then rebuilds sample i as sample[i-1] + delta[i-1]
 * (mod 2^16) and applies the escape patches.
 */

/* Conservative upper bound on the encoded length of an n-byte input (it
 * assumes no prefix). Returns 0 on overflow. The encoder itself checks the
 * exact need, which includes the discovered prefix, before any store. */
uint32_t zpse_dpcm1_bound(uint32_t n);

int zpse_dpcm1_encode(const uint8_t *in, uint32_t n,
                       uint8_t *out, uint32_t out_cap, uint32_t *out_len);

int zpse_dpcm1_decode(const uint8_t *in, uint32_t in_len,
                       uint8_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_DPCM1_H */
