#ifndef ZPSE_MRD_H
#define ZPSE_MRD_H

/* MRD -- previous-sample DPCM whole-frame codec with zero-run RLE, for
 * 16-bit little-endian sample streams (measured on the Silesia "mr" image).
 *
 * Like DPCM1: a single fixed predictor (the previous 16-bit sample), one
 * signed byte per fitting delta, and a small sorted escape table for the rare
 * deltas outside [-127,127]. Unlike DPCM1, a run of >=16 consecutive zero
 * deltas is stored as one 3-byte token (marker 0x80 + u16 run length) that
 * decodes by replicating the held sample -- no bitmap, no per-sample byte.
 * The marker byte 0x80 == int8 -128 is outside the fitting range [-127,127],
 * so it can never occur as a literal delta or as an escape placeholder (0x00)
 * and is unambiguous. Decode keeps DPCM1's SSE2 inclusive scan for stretches
 * of literal deltas and uses a 16-bit splat for RLE runs.
 *
 * REGISTERED IN THE UNITY BUILD: src/mrd.c is included by src/zpse_unity.c.
 *
 * OPT-IN: nothing is emitted unless the caller sets ZPSE_ENC_MRD (bit 18) on
 * zpse_compress_frame_ex. The content gate accepts only data like "mr":
 * >=95% int8 deltas and mean |delta| in [4,14) (so x-ray at 25.4 declines),
 * plus a zero-delta floor.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZPSE_MRD_DECLINE 0
#define ZPSE_MRD_COMMIT  1

/* "MRD1" as a little-endian u32: bytes 'M','R','D','1'. Distinct from PSE1,
 * F7P1, PXI1, F64F, RLE1, DCM1 and ZRPK (differs in >=1 byte). */
#define ZPSE_MRD_MAGIC       UINT32_C(0x3144524D)
#define ZPSE_MRD_HEADER_SIZE UINT32_C(24)

uint32_t zpse_mrd_bound(uint32_t n);

int zpse_mrd_encode(const uint8_t *in, uint32_t n,
                     uint8_t *out, uint32_t out_cap, uint32_t *out_len);

int zpse_mrd_decode(const uint8_t *in, uint32_t in_len,
                     uint8_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_MRD_H */
