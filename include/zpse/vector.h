#ifndef ZPSE_VECTOR_H
#define ZPSE_VECTOR_H

/* SIMD dispatch + always-inline hot-loop declarations (SPEC Sec. 2, 10).
 * Scalar C in the .c files is normative; these kernels MUST be
 * bit-exact replacements selected at runtime by zpse_cpu_caps(). */

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define ZPSE_ALWAYS_INLINE __forceinline
#else
#define ZPSE_ALWAYS_INLINE static inline __attribute__((always_inline))
#endif

typedef enum ZpseCpuCaps {
    ZPSE_CPU_SCALAR = 0,
    ZPSE_CPU_SSE2   = 1 << 0,
    ZPSE_CPU_SSSE3  = 1 << 1,
    ZPSE_CPU_AVX2   = 1 << 2,
    ZPSE_CPU_AVX512 = 1 << 3, /* BW + VBMI2 (vpexpandb) required */
    ZPSE_CPU_NEON   = 1 << 4
} ZpseCpuCaps;

uint32_t zpse_cpu_caps(void);

/* Inverse transpose-8: 8 lanes x 8 B in -> 64 B interleaved out.
 * segs[s] must each have >= 8 readable bytes at seg_offset. */
void zpse_inv_transpose8_64B(uint8_t *dst, const uint8_t *segs[8],
                             size_t seg_offset);

/* LANE_DELTA over 64 B: pred_out[64] from lane histories. */
void zpse_predict_lane_delta_64B(uint8_t *pred_out, const uint8_t last[8],
                                 const uint8_t prev[8]);

/* Monolithic tile reconstruct: RAW_BITMAP + EXCLUSION_U8 fast path.
 * Returns residual bytes consumed, or negative status. */
int zpse_reconstruct_bitmap_excl_u8(uint8_t *dst, const uint8_t *pred,
                                    const uint8_t *bitmap, uint16_t nt,
                                    const uint8_t *residuals,
                                    uint16_t res_len);

/* AVX2 8-B compaction helper (pshufb LUT lives in arena shuf region). */
int zpse_compact8(uint8_t *dst, const uint8_t data[8], uint8_t mask8,
                  const uint8_t *lut_2k);

#endif /* ZPSE_VECTOR_H */
