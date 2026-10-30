#ifndef ZPSE_VECTOR_H
#define ZPSE_VECTOR_H

/* SIMD dispatch + always-inline hot-loop declarations.
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

/* The same answer, cached at the call site.
 *
 * `zpse_cpu_caps` carries the whole CPUID/XGETBV body, so gcc will not
 * inline it: every ask is a call plus a relaxed atomic load plus a
 * compare, 6-12 cycles. That is fine once per block and not fine where it
 * is asked -- the per-tile paths ask it two to four times per tile, on
 * both encode (classify, the at_pred fill, pack) and decode (tile decode,
 * the residual dispatchers), which is 15-45 cycles of a 500-cycle tile.
 *
 * The answer cannot change while the process runs and the computation is
 * idempotent, so the hot sites read a cache of their own and only fall
 * back to the real function on the first call. `ZPSE_FORCE_SCALAR` is
 * honoured because the fallback is the function that implements it. */
#if defined(_MSC_VER)
#define ZPSE_CPU_CAPS_FAST() zpse_cpu_caps()
#else
#include <stdatomic.h>

static inline uint32_t zpse_cpu_caps_fast(void)
{
    static _Atomic uint32_t zpse_caps_cache = 0xFFFFFFFFu;
    uint32_t caps = atomic_load_explicit(&zpse_caps_cache,
                                         memory_order_relaxed);

    if (caps != 0xFFFFFFFFu) {
        return caps;
    }
    caps = zpse_cpu_caps();
    atomic_store_explicit(&zpse_caps_cache, caps, memory_order_relaxed);
    return caps;
}

#define ZPSE_CPU_CAPS_FAST() zpse_cpu_caps_fast()
#endif

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
