#include "zpse/flat.h"

#include <string.h>

#include "zpse/constants.h" /* ZPSE_FLAT_* codec ids and flags */
#include "zpse/hot.h"     /* ZPSE_AVX2_ABI_OK */
#include "zpse/status.h"
#include "zpse/vector.h"  /* ZPSE_CPU_CAPS_FAST, ZPSE_CPU_AVX2 */

/* ---- AVX2 target/ABI gates (same shape as src/dense.c) ---------------- *
 * The per-function target attribute lets these bodies compile in a TU
 * built at the baseline ISA; ZPSE_AVX2_ABI_OK keeps them out of builds
 * whose ABI cannot carry a 256-bit value (the gcc/Windows by-value vector
 * hazard described in zpse/hot.h). aligned(64) keeps each body inside one
 * pair of I-cache lines. Every __m256i below is a function LOCAL: no vector
 * is ever passed by value or returned, so the ABI hazard cannot recur. */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_FLAT_AVX2_TARGET __attribute__((target("avx2"), aligned(64)))
#define ZPSE_FLAT_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_FLAT_AVX2_TARGET __attribute__((aligned(64)))
#define ZPSE_FLAT_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_FLAT_HAVE_AVX2 0
#endif
#else
#define ZPSE_FLAT_HAVE_AVX2 0
#endif

#if ZPSE_FLAT_HAVE_AVX2
#include <immintrin.h>
#endif

#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2

/* ---- codec 0: FLAT_RAW ------------------------------------------------ *
 * 32 bytes per load/store pair; the scalar tail is a plain copy. The
 * fallback is memcpy, which is bit-exact for any overlap-free buffers. */
static ZPSE_FLAT_AVX2_TARGET void flat_raw_avx2(const uint8_t *src,
                                                uint8_t *dst, uint32_t n)
{
    uint32_t i = 0u;

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(src + i));

        _mm256_storeu_si256((__m256i *)(void *)(dst + i), v);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

/* ---- codec 1: FLAT_NIBBLE4 ------------------------------------------- *
 * 32 packed bytes -> 64 palette lookups. The 16-byte palette is broadcast
 * into both 128-bit halves of a __m256i, so pshufb's per-lane 16-byte
 * indexing maps every nibble (0..15) to pal[nibble] with no masking. */
static ZPSE_FLAT_AVX2_TARGET void flat_nibble4_avx2(const uint8_t *pal,
    const uint8_t *src, uint8_t *dst, uint32_t n)
{
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i plt = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(const void *)pal));
    uint32_t i = 0u;

    for (; i + 64u <= n; i += 64u) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(src + (i >> 1u)));
        __m256i lo = _mm256_and_si256(v, mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
        __m256i a = _mm256_unpacklo_epi8(lo, hi);
        __m256i b = _mm256_unpackhi_epi8(lo, hi);
        __m256i o0 = _mm256_permute2x128_si256(a, b, 0x20);
        __m256i o1 = _mm256_permute2x128_si256(a, b, 0x31);

        _mm256_storeu_si256((__m256i *)(void *)(dst + i),
                            _mm256_shuffle_epi8(plt, o0));
        _mm256_storeu_si256((__m256i *)(void *)(dst + i + 32u),
                            _mm256_shuffle_epi8(plt, o1));
    }
    for (; i < n; i++) {
        dst[i] = pal[src[i >> 1u] >> (4u * (i & 1u)) & 0x0Fu];
    }
}

/* ---- codec 2: FLAT_BIT2 ---------------------------------------------- *
 * 8 packed bytes -> 32 two-bit values (the unpack from vector.c), then one
 * pshufb through an 8-byte palette broadcast to both lanes. */
static ZPSE_FLAT_AVX2_TARGET void flat_bit2_avx2(const uint8_t *pal,
    const uint8_t *src, uint8_t *dst, uint32_t n)
{
    const __m128i ctl0 = _mm_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    const __m128i ctl4 = _mm_setr_epi8(
        4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m256i modd = _mm256_setr_epi8(
        0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
        0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
    const __m256i mtwo = _mm256_setr_epi8(
        0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1,
        0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1);
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i plt = _mm256_broadcastsi128_si256(
        _mm_loadl_epi64((const __m128i *)(const void *)pal));
    uint32_t i = 0u;

    for (; i + 32u <= n; i += 32u) {
        __m128i p8 = _mm_loadl_epi64(
            (const __m128i *)(const void *)(src + (i >> 2u)));
        __m256i b = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(p8, ctl0)),
            _mm_shuffle_epi8(p8, ctl4), 1);
        __m256i x0 = _mm256_and_si256(b, mask);
        __m256i x1 = _mm256_and_si256(_mm256_srli_epi16(b, 2), mask);
        __m256i x2 = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask);
        __m256i x3 = _mm256_and_si256(_mm256_srli_epi16(b, 6), mask);
        __m256i p01 = _mm256_blendv_epi8(x0, x1, modd);
        __m256i p23 = _mm256_blendv_epi8(x2, x3, modd);
        __m256i idx = _mm256_blendv_epi8(p01, p23, mtwo);

        _mm256_storeu_si256((__m256i *)(void *)(dst + i),
                            _mm256_shuffle_epi8(plt, idx));
    }
    for (; i < n; i++) {
        dst[i] = pal[src[i >> 2u] >> (2u * (i & 3u)) & 0x03u];
    }
}

/* ---- codec 3: FLAT_SUBLELANE ----------------------------------------- *
 * The pattern repeats every `period` bytes (period is a power of two <= 8)
 * and at most `k` of them are stored. For a 16-byte output slice the needed
 * sub bytes are subn = (16/period)*k <= 16. Build a 16-byte pshufb table
 * whose [0,subn) slots are the sub bytes for this slice and whose remaining
 * slots hold the template values for clear keep bits; one precomputed index
 * vector (identical for both halves and every slice) selects sub or template
 * per position. Two pshufb + two stores produce 32 output bytes per step. */
static ZPSE_FLAT_AVX2_TARGET void flat_sublane_avx2(const uint8_t *sub,
    uint8_t *dst, uint32_t n, uint32_t period, uint32_t k,
    const uint8_t keep[8], const uint8_t rank[8],
    const uint8_t tslot[8], const uint8_t templ[8])
{
    uint8_t idx_arr[16];
    uint8_t tbl[16];
    const uint32_t subn = (16u / period) * k;
    const uint32_t groups16 = 16u / period;
    uint32_t t;
    uint32_t at;

    memset(tbl, 0, sizeof tbl);
    for (t = 0u; t < 16u; t++) {
        uint32_t j = t & (period - 1u);
        uint32_t g = t / period;

        idx_arr[t] = (uint8_t)(keep[j] ? (g * k + rank[j])
                                       : (subn + tslot[j]));
        if (!keep[j]) {
            tbl[subn + tslot[j]] = templ[j];
        }
    }
    {
        const __m128i idx = _mm_loadu_si128(
            (const __m128i *)(const void *)idx_arr);

        for (at = 0u; at + 32u <= n; at += 32u) {
            uint32_t h;

            for (h = 0u; h < 2u; h++) {
                uint32_t base_g = (at + 16u * h) / period;
                uint32_t g;
                uint32_t b;

                for (g = 0u; g < groups16; g++) {
                    for (b = 0u; b < k; b++) {
                        tbl[g * k + b] = sub[(base_g + g) * k + b];
                    }
                }
                {
                    __m128i tv = _mm_loadu_si128(
                        (const __m128i *)(const void *)tbl);
                    __m128i r = _mm_shuffle_epi8(tv, idx);

                    _mm_storeu_si128(
                        (__m128i *)(void *)(dst + at + 16u * h), r);
                }
            }
        }
    }
    for (; at < n; at++) {
        uint32_t g = at / period;
        uint32_t j = at & (period - 1u);

        dst[at] = keep[j] ? sub[g * k + rank[j]] : templ[j];
    }
}

#endif /* ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2 */

/* ---- scalar, bit-exact fallbacks ------------------------------------- */

static void flat_nibble4_scalar(const uint8_t *restrict pal,
                                const uint8_t *restrict src,
                                uint8_t *restrict dst, uint32_t n)
{
    uint32_t i;

    for (i = 0u; i < n; i++) {
        dst[i] = pal[src[i >> 1u] >> (4u * (i & 1u)) & 0x0Fu];
    }
}

static void flat_bit2_scalar(const uint8_t *restrict pal,
                             const uint8_t *restrict src,
                             uint8_t *restrict dst, uint32_t n)
{
    uint32_t i;

    for (i = 0u; i < n; i++) {
        dst[i] = pal[src[i >> 2u] >> (2u * (i & 3u)) & 0x03u];
    }
}

static void flat_sublane_scalar(const uint8_t *restrict sub,
                                uint8_t *restrict dst, uint32_t n,
                                uint32_t period, uint32_t k,
                                const uint8_t keep[8], const uint8_t rank[8],
                                const uint8_t templ[8])
{
    uint32_t p;

    for (p = 0u; p < n; p++) {
        uint32_t g = p / period;
        uint32_t j = p & (period - 1u);

        dst[p] = keep[j] ? sub[g * k + rank[j]] : templ[j];
    }
}

/* ---- decoder + validation -------------------------------------------- */

int zpse_flat_decode(uint8_t codec, uint8_t param,
                     const uint8_t *payload, uint32_t payload_len,
                     uint8_t *out, uint32_t n)
{
    uint32_t expected;

    /* Common argument checks. n == 0 and n > one page are rejected before
     * any payload field is read; a caller with a non-page-size output has
     * the wrong wire form. */
    if (out == 0 || n == 0u || n > (uint32_t)ZPSE_FLAT_MAX_N) {
        return ZPSE_BAD_ARG;
    }
    if (codec > (uint8_t)ZPSE_FLAT_SUBLELANE) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (payload == 0) {
        return ZPSE_BAD_ARG;
    }

    switch (codec) {
    case ZPSE_FLAT_RAW: {
        if (param != 0u) {
            return ZPSE_BAD_ARG;
        }
        expected = n;
        if (payload_len < expected) {
            return ZPSE_TRUNCATED;
        }
        if (payload_len != expected) {
            return ZPSE_BAD_ARG;
        }
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        if (n >= 32u &&
            (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
            flat_raw_avx2(payload, out, n);
            return ZPSE_OK;
        }
#endif
        memcpy(out, payload, (size_t)n);
        return ZPSE_OK;
    }
    case ZPSE_FLAT_NIBBLE4: {
        const uint8_t *pal;
        const uint8_t *idx;

        if (param > 15u) {
            return ZPSE_BAD_ARG;
        }
        expected = 16u + ((n + 1u) >> 1);
        if (payload_len < expected) {
            return ZPSE_TRUNCATED;
        }
        if (payload_len != expected) {
            return ZPSE_BAD_ARG;
        }
        pal = payload;
        idx = payload + 16u;
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        if (n >= 64u &&
            (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
            flat_nibble4_avx2(pal, idx, out, n);
            return ZPSE_OK;
        }
#endif
        flat_nibble4_scalar(pal, idx, out, n);
        return ZPSE_OK;
    }
    case ZPSE_FLAT_BIT2: {
        const uint8_t *pal;
        const uint8_t *idx;

        if (param > 3u) {
            return ZPSE_BAD_ARG;
        }
        expected = 8u + ((n + 3u) >> 2);
        if (payload_len < expected) {
            return ZPSE_TRUNCATED;
        }
        if (payload_len != expected) {
            return ZPSE_BAD_ARG;
        }
        pal = payload;
        idx = payload + 8u;
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        if (n >= 32u &&
            (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
            flat_bit2_avx2(pal, idx, out, n);
            return ZPSE_OK;
        }
#endif
        flat_bit2_scalar(pal, idx, out, n);
        return ZPSE_OK;
    }
    default: { /* ZPSE_FLAT_SUBLELANE */
        uint32_t period;
        uint8_t keepmask;
        uint8_t keep[8];
        uint8_t rank[8];
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        uint8_t tslot[8];
#endif
        const uint8_t *templ;
        const uint8_t *sub;
        uint32_t k = 0u;
        uint32_t groups;
        uint32_t j;
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        uint32_t ts = 0u;
#endif

        /* The fixed 10-byte prefix must be present before its fields are
         * read; the exact-length check below covers the rest. */
        if (payload_len < 14u) {
            return ZPSE_TRUNCATED;
        }
        period = (uint32_t)payload[0];
        keepmask = payload[1];
        if (period != 1u && period != 2u && period != 4u && period != 8u) {
            return ZPSE_BAD_ARG;
        }

        for (j = 0u; j < period; j++) {
            keep[j] = (uint8_t)((keepmask >> j) & 1u);
            rank[j] = (uint8_t)k;
            if (keep[j]) {
                k++;
            }
        }
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        for (j = 0u; j < period; j++) {
            tslot[j] = 0u;
            if (!keep[j]) {
                tslot[j] = (uint8_t)ts;
                ts++;
            }
        }
#endif

        /* ceil(n/period) groups cover every output position, including a
         * short final group when n is not a multiple of the period. */
        groups = (n + period - 1u) / period;
        expected = 14u + k * groups;
        if (payload_len < expected) {
            return ZPSE_TRUNCATED;
        }
        if (payload_len != expected) {
            return ZPSE_BAD_ARG;
        }

        templ = payload + 2u;
        sub = payload + 10u;
#if ZPSE_FLAT_HAVE_AVX2 && ZPSE_FLAT_AVX2
        if (n >= 32u &&
            (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
            flat_sublane_avx2(sub, out, n, period, k, keep, rank, tslot,
                              templ);
            return ZPSE_OK;
        }
#endif
        flat_sublane_scalar(sub, out, n, period, k, keep, rank, templ);
        return ZPSE_OK;
    }
    }
}
