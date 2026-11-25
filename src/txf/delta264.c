/* delta264.c -- ZPSE transform ID 21 (DELTA2_64), W = 8 bytes, B = 64.
 *
 * Forward, on the count = n/8 little-endian 64-bit words of the input:
 *
 *   y[0] = x[0]
 *   y[1] = x[1] - x[0]                       (mod 2^64)
 *   y[i] = (x[i] - x[i-1]) - (x[i-1] - x[i-2])   for i >= 2
 *        = x[i] - 2*x[i-1] + x[i-2]          (mod 2^64)
 *
 * the last form is the one the SIMD kernels use: it is a 3-tap modular FIR,
 * so y[i] has no dependency on y[i-1] and 2/4 words can be done per vector.
 * uint64 arithmetic wraps naturally, which is exactly modular 2^64.
 *
 * y[] is then byte-transposed into 8 planes (plane p = byte p of every word)
 * and the n%8 tail bytes are copied verbatim after the planes.
 *
 * Inverse: reconstruct y[] from the planes, then
 *
 *   x[0] = y[0];  d = y[1];  x[1] = x[0] + d;
 *   for i >= 2:  d += y[i];  x[i] = x[i-1] + d;
 *
 * which is a two-stage prefix sum of y[] (d accumulates y, x accumulates d).
 * The SIMD kernels do that with shift/add prefix ladders plus a 64-bit carry
 * between vectors. Every path is byte-identical to the scalar oracle; the
 * scalar C remains normative.
 *
 * No FP, no allocation, no globals beyond a cached CPUID answer, no unaligned
 * casts: the scalar paths use explicit byte-wise LE helpers and the SIMD
 * paths use unaligned _mm*_loadu/_storeu intrinsics. */

#include "delta264.h"

#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || \
    defined(__i386__) || defined(_M_IX86)
#define D264_X86 1
#else
#define D264_X86 0
#endif

#if D264_X86
#if defined(__GNUC__) && !defined(_MSC_VER)
#define D264_TARGET_SSE2 __attribute__((target("sse2")))
#define D264_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define D264_TARGET_SSE2
#define D264_TARGET_AVX2
#endif
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#if !defined(_MSC_VER) && !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#endif
#endif

/* ---- Explicit little-endian helpers (no unaligned casts) -------------- */

static uint64_t d264_load64(const uint8_t *p)
{
    uint64_t v = 0;
    unsigned i;
    for (i = 0; i < 8u; i++) {
        v |= (uint64_t)p[i] << (8u * i);
    }
    return v;
}

static uint64_t d264_store64(uint8_t *p, uint64_t v)
{
    unsigned i;
    for (i = 0; i < 8u; i++) {
        p[i] = (uint8_t)(v >> (8u * i));
    }
    return v;
}

/* Number of significant bits of a modular uint64 value (0 -> 0). Used only
 * by the gate; it is the width of the stored value, not a signed width. */
static unsigned d264_bitlen(uint64_t v)
{
    if (v == 0u) {
        return 0u;
    }
#if defined(__GNUC__) || defined(__clang__)
    return 64u - (unsigned)__builtin_clzll(v);
#else
    {
        unsigned b = 0u;
        while (v != 0u) {
            v >>= 1;
            b++;
        }
        return b;
    }
#endif
}

/* ---- Runtime dispatch -------------------------------------------------- */

static uint32_t d264_detect_caps(void)
{
    uint32_t caps = 0u;

#if defined(D264_FORCE_SCALAR)
    return 0u;
#endif
#if D264_X86
#if defined(_MSC_VER)
    {
        int info[4];
        unsigned long long xcr0;
        __cpuid(info, 1);
        if (info[3] & (1 << 26)) {
            caps |= 1u; /* SSE2 */
        }
        if ((info[2] & (1 << 27)) && (info[2] & (1 << 28))) {
            xcr0 = _xgetbv(0);
            if ((xcr0 & 0x6ULL) == 0x6ULL) {
                __cpuid(info, 0);
                if (info[0] >= 7) {
                    __cpuidex(info, 7, 0);
                    if (info[1] & (1 << 5)) {
                        caps |= 2u; /* AVX2 */
                    }
                }
            }
        }
    }
#else
    {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (__get_cpuid(1, &a, &b, &c, &d)) {
            if (d & (1u << 26)) {
                caps |= 1u; /* SSE2 */
            }
            if ((c & (1u << 27)) && (c & (1u << 28))) {
                unsigned long long lo = 0, hi = 0;
                __asm__ volatile(".byte 0x0f,0x01,0xd0"
                                 : "=a"(lo), "=d"(hi) : "c"(0));
                if ((((hi << 32) | lo) & 0x6ULL) == 0x6ULL &&
                    __get_cpuid_max(0, NULL) >= 7u) {
                    unsigned int a7 = 0, b7 = 0, c7 = 0, d7 = 0;
                    (void)a7; (void)c7; (void)d7;
                    __get_cpuid_count(7, 0, &a7, &b7, &c7, &d7);
                    if (b7 & (1u << 5)) {
                        caps |= 2u; /* AVX2 */
                    }
                }
            }
        }
    }
#endif
#endif
    return caps;
}

static uint32_t d264_caps(void)
{
#if D264_X86 && !defined(_MSC_VER) && !defined(__STDC_NO_ATOMICS__)
    static _Atomic int cache = -1;
    int c = atomic_load_explicit(&cache, memory_order_relaxed);
    if (c >= 0) {
        return (uint32_t)c;
    }
    c = (int)d264_detect_caps();
    atomic_store_explicit(&cache, c, memory_order_relaxed);
    return (uint32_t)c;
#else
    static int cache = -1;
    if (cache < 0) {
        cache = (int)d264_detect_caps();
    }
    return (uint32_t)cache;
#endif
}

int zpse_d264_has_sse2(void) { return (d264_caps() & 1u) != 0u; }
int zpse_d264_has_avx2(void) { return (d264_caps() & 2u) != 0u; }

/* ---- 8x8 byte transpose ------------------------------------------------ *
 * in[r] holds 8 bytes of row r, out[c] receives 8 bytes of column c:
 * out[c][r] = in[r][c]. The permutation is its own inverse, so forward and
 * inverse share the same kernel. SSE2 unpacks only; it is used by both SIMD
 * paths because an 8x8 byte transpose is 64 bytes of pure shuffling where a
 * 256-bit register buys nothing over two 128-bit halves. */

static void d264_xpose8_scalar(const uint8_t *const in[8], uint8_t *const out[8])
{
    unsigned c, r;
    for (c = 0u; c < 8u; c++) {
        for (r = 0u; r < 8u; r++) {
            out[c][r] = in[r][c];
        }
    }
}

#if D264_X86
static D264_TARGET_SSE2 void d264_xpose8_sse2(const uint8_t *const in[8],
                                             uint8_t *const out[8])
{
    __m128i a0, a1, a2, a3, a4, a5, a6, a7;
    __m128i t0, t1, t2, t3, u0, u1, u2, u3, v0, v1, v2, v3;

    a0 = _mm_loadl_epi64((const __m128i *)(const void *)in[0]);
    a1 = _mm_loadl_epi64((const __m128i *)(const void *)in[1]);
    a2 = _mm_loadl_epi64((const __m128i *)(const void *)in[2]);
    a3 = _mm_loadl_epi64((const __m128i *)(const void *)in[3]);
    a4 = _mm_loadl_epi64((const __m128i *)(const void *)in[4]);
    a5 = _mm_loadl_epi64((const __m128i *)(const void *)in[5]);
    a6 = _mm_loadl_epi64((const __m128i *)(const void *)in[6]);
    a7 = _mm_loadl_epi64((const __m128i *)(const void *)in[7]);

    t0 = _mm_unpacklo_epi8(a0, a1);
    t1 = _mm_unpacklo_epi8(a2, a3);
    t2 = _mm_unpacklo_epi8(a4, a5);
    t3 = _mm_unpacklo_epi8(a6, a7);
    u0 = _mm_unpacklo_epi16(t0, t1);
    u1 = _mm_unpackhi_epi16(t0, t1);
    u2 = _mm_unpacklo_epi16(t2, t3);
    u3 = _mm_unpackhi_epi16(t2, t3);
    v0 = _mm_unpacklo_epi32(u0, u2); /* columns 0,1 */
    v2 = _mm_unpackhi_epi32(u0, u2); /* columns 2,3 */
    v1 = _mm_unpacklo_epi32(u1, u3); /* columns 4,5 */
    v3 = _mm_unpackhi_epi32(u1, u3); /* columns 6,7 */

    _mm_storel_epi64((__m128i *)(void *)out[0], v0);
    _mm_storel_epi64((__m128i *)(void *)out[1], _mm_srli_si128(v0, 8));
    _mm_storel_epi64((__m128i *)(void *)out[2], v2);
    _mm_storel_epi64((__m128i *)(void *)out[3], _mm_srli_si128(v2, 8));
    _mm_storel_epi64((__m128i *)(void *)out[4], v1);
    _mm_storel_epi64((__m128i *)(void *)out[5], _mm_srli_si128(v1, 8));
    _mm_storel_epi64((__m128i *)(void *)out[6], v3);
    _mm_storel_epi64((__m128i *)(void *)out[7], _mm_srli_si128(v3, 8));
}
#endif /* D264_X86 */

/* ---- Forward ----------------------------------------------------------- */

/* Write one reconstructed word's bytes into the planes at word index i. */
static void d264_put_word(uint8_t *const planes[8], uint32_t i, uint64_t y)
{
    unsigned p;
    for (p = 0u; p < 8u; p++) {
        planes[p][i] = (uint8_t)(y >> (8u * p));
    }
}

static void d264_fwd_group(uint8_t *const planes[8], uint32_t base,
                           const uint8_t ybytes[64], int mode)
{
    const uint8_t *in[8];
    uint8_t *out[8];
    unsigned p;
    for (p = 0u; p < 8u; p++) {
        in[p] = ybytes + (size_t)p * 8u;
        out[p] = planes[p] + base;
    }
#if D264_X86
    if (mode != 0) {
        d264_xpose8_sse2(in, out);
        return;
    }
#else
    (void)mode;
#endif
    d264_xpose8_scalar(in, out);
}

static D264_TARGET_AVX2 int d264_fwd_impl(const uint8_t *src, uint32_t count,
                         uint8_t *const planes[8], int mode)
{
    uint32_t base, i;
    uint64_t x0, x1;

    /* Words 0 and 1 are the special cases of the recurrence. */
    if (count >= 1u) {
        d264_put_word(planes, 0u, d264_load64(src));
    }
    if (count < 2u) {
        return ZPSE_D264_OK;
    }
    x0 = d264_load64(src);
    x1 = d264_load64(src + 8u);
    d264_put_word(planes, 1u, x1 - x0);

    /* Complete 8-word groups from index 2. The FIR form needs x[base-2],
     * which is in range for base >= 2. */
    base = 2u;
#if D264_X86
    for (; (size_t)base + 8u <= (size_t)count; base += 8u) {
        uint8_t ybytes[64];
        unsigned k;
        if (mode == 2) {
            __m256i c0 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(src + (size_t)base * 8u));
            __m256i c1 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(src + (size_t)(base + 4u) * 8u));
            __m256i p1a = _mm256_loadu_si256(
                (const __m256i *)(const void *)(src + (size_t)(base - 1u) * 8u));
            __m256i p1b = _mm256_loadu_si256(
                (const __m256i *)(const void *)(src + (size_t)(base + 3u) * 8u));
            __m256i p2a = _mm256_loadu_si256(
                (const __m256i *)(const void *)(src + (size_t)(base - 2u) * 8u));
            __m256i p2b = _mm256_loadu_si256(
                (const __m256i *)(const void *)(src + (size_t)(base + 2u) * 8u));
            /* y = cur - 2*prev + prev2, all mod 2^64. */
            __m256i y0 = _mm256_add_epi64(
                _mm256_sub_epi64(_mm256_sub_epi64(c0, p1a), p1a), p2a);
            __m256i y1 = _mm256_add_epi64(
                _mm256_sub_epi64(_mm256_sub_epi64(c1, p1b), p1b), p2b);
            _mm256_storeu_si256((__m256i *)(void *)(ybytes + 0), y0);
            _mm256_storeu_si256((__m256i *)(void *)(ybytes + 32), y1);
        } else {
            for (k = 0u; k < 8u; k += 2u) {
                __m128i c = _mm_loadu_si128((const __m128i *)(const void *)
                                  (src + (size_t)(base + k) * 8u));
                __m128i p1 = _mm_loadu_si128((const __m128i *)(const void *)
                                  (src + (size_t)(base + k - 1u) * 8u));
                __m128i p2 = _mm_loadu_si128((const __m128i *)(const void *)
                                  (src + (size_t)(base + k - 2u) * 8u));
                __m128i y = _mm_add_epi64(
                    _mm_sub_epi64(_mm_sub_epi64(c, p1), p1), p2);
                _mm_storeu_si128((__m128i *)(void *)(ybytes + (size_t)k * 8u), y);
            }
        }
        (void)k;
        d264_fwd_group(planes, base, ybytes, mode);
    }
#endif
    /* Scalar remainder (also the whole loop on a non-x86 build). */
    for (i = base; i < count; i++) {
        uint64_t c = d264_load64(src + (size_t)i * 8u);
        uint64_t p1 = d264_load64(src + (size_t)(i - 1u) * 8u);
        uint64_t p2 = d264_load64(src + (size_t)(i - 2u) * 8u);
        d264_put_word(planes, i, c - p1 - p1 + p2);
    }
    return ZPSE_D264_OK;
}

/* ---- Inverse prefix helpers -------------------------------------------- *
 * pfx2/pfx4 return the inclusive prefix of v with `carry` added to every
 * lane, and write the last lane (the carry for the next vector) to *total. */

static D264_TARGET_SSE2 __m128i d264_pfx2(__m128i v, uint64_t carry,
                                          uint64_t *total)
{
    v = _mm_add_epi64(v, _mm_slli_si128(v, 8));
    v = _mm_add_epi64(v, _mm_set1_epi64x((long long)carry));
    *total = (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(v, 8));
    return v;
}

static D264_TARGET_AVX2 __m256i d264_pfx4(__m256i v, uint64_t carry,
                                          uint64_t *total)
{
    __m256i addhi;
    v = _mm256_add_epi64(v, _mm256_slli_si256(v, 8));
    /* After the shift, the low 128-bit lane holds [y0, y0+y1]; its total is
     * byte 8..15, which must be added ONLY to the high lane. */
    addhi = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_setzero_si128()),
        _mm_set1_epi64x((long long)(uint64_t)_mm_cvtsi128_si64(
            _mm_srli_si128(_mm256_castsi256_si128(v), 8))), 1);
    v = _mm256_add_epi64(v, addhi);
    v = _mm256_add_epi64(v, _mm256_set1_epi64x((long long)carry));
    /* The block total is element 3 (the high half's high 64-bit lane), so
     * shift the high 128-bit lane right by 8 before extracting to a GPR.
     * A plain cvtsi128_si64 would return element 2 and break the carry. */
    *total = (uint64_t)_mm_cvtsi128_si64(
        _mm_srli_si128(_mm256_extracti128_si256(v, 1), 8));
    return v;
}

/* Reconstruct one y word from the planes at index i. */
static uint64_t d264_get_word(const uint8_t *const planes[8], uint32_t i)
{
    uint64_t y = 0u;
    unsigned p;
    for (p = 0u; p < 8u; p++) {
        y |= (uint64_t)planes[p][i] << (8u * p);
    }
    return y;
}

static D264_TARGET_AVX2 int d264_inv_impl(const uint8_t *const planes[8], uint32_t count,
                         uint8_t *dst, int mode)
{
    uint32_t i, base;
    uint64_t y0, dcarry, xcarry;

    if (count >= 1u) {
        d264_store64(dst, d264_get_word(planes, 0u));
    }
    if (count < 2u) {
        return ZPSE_D264_OK;
    }
    y0 = d264_get_word(planes, 0u);
    dcarry = d264_get_word(planes, 1u);          /* d = y[1] */
    xcarry = y0 + dcarry;                         /* x[1] = x[0] + y[1] */
    d264_store64(dst + 8u, xcarry);

    base = 2u;
#if D264_X86
    for (; (size_t)base + 8u <= (size_t)count; base += 8u) {
        uint8_t ybytes[64];
        const uint8_t *in[8];
        uint8_t *out[8];
        unsigned p;

        for (p = 0u; p < 8u; p++) {
            in[p] = planes[p] + base;
            out[p] = ybytes + (size_t)p * 8u;
        }
        /* planes -> y[] (the transpose is its own inverse). */
        if (mode != 0) {
            d264_xpose8_sse2(in, out);
        } else {
            d264_xpose8_scalar(in, out);
        }

        if (mode == 2) {
            unsigned j;
            for (j = 0u; j < 8u; j += 4u) {
                __m256i yv = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(ybytes + (size_t)j * 8u));
                __m256i dv = d264_pfx4(yv, dcarry, &dcarry);
                __m256i xv = d264_pfx4(dv, xcarry, &xcarry);
                _mm256_storeu_si256(
                    (__m256i *)(void *)(dst + (size_t)(base + j) * 8u), xv);
            }
        } else {
            unsigned j;
            for (j = 0u; j < 8u; j += 2u) {
                __m128i yv = _mm_loadu_si128(
                    (const __m128i *)(const void *)(ybytes + (size_t)j * 8u));
                __m128i dv = d264_pfx2(yv, dcarry, &dcarry);
                __m128i xv = d264_pfx2(dv, xcarry, &xcarry);
                _mm_storeu_si128(
                    (__m128i *)(void *)(dst + (size_t)(base + j) * 8u), xv);
            }
        }
    }
#endif
    /* Scalar remainder. */
    for (i = base; i < count; i++) {
        uint64_t y = d264_get_word(planes, i);
        dcarry += y;
        xcarry += dcarry;
        d264_store64(dst + (size_t)i * 8u, xcarry);
    }
    return ZPSE_D264_OK;
}

/* ---- Public wrappers --------------------------------------------------- */

static int d264_validate(const uint8_t *src, uint32_t n,
                         uint8_t *dst, uint32_t dst_cap)
{
    if (src == 0 || dst == 0) {
        return ZPSE_D264_BAD_ARG;
    }
    if (dst_cap < n) {
        return ZPSE_D264_SHORT;
    }
    return ZPSE_D264_OK;
}

static int d264_forward_call(const uint8_t *src, uint32_t n,
                             uint8_t *dst, int mode)
{
    uint8_t *planes[8];
    uint32_t count = n / 8u;
    uint32_t tail = n % 8u;
    unsigned p;

    if (n == 0u) {
        return ZPSE_D264_OK;
    }
    for (p = 0u; p < 8u; p++) {
        planes[p] = dst + (size_t)p * count;
    }
    d264_fwd_impl(src, count, planes, mode);
    if (tail != 0u) {
        memcpy(dst + (size_t)8u * count, src + (size_t)8u * count, tail);
    }
    return ZPSE_D264_OK;
}

static int d264_inverse_call(const uint8_t *src, uint32_t n,
                             uint8_t *dst, int mode)
{
    const uint8_t *planes[8];
    uint32_t count = n / 8u;
    uint32_t tail = n % 8u;
    unsigned p;

    if (n == 0u) {
        return ZPSE_D264_OK;
    }
    for (p = 0u; p < 8u; p++) {
        planes[p] = src + (size_t)p * count;
    }
    d264_inv_impl(planes, count, dst, mode);
    if (tail != 0u) {
        memcpy(dst + (size_t)8u * count, src + (size_t)8u * count, tail);
    }
    return ZPSE_D264_OK;
}

int zpse_d264_forward_scalar(const uint8_t *src, uint32_t n,
                             uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
    return d264_forward_call(src, n, dst, 0);
}

int zpse_d264_forward_sse2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
    return d264_forward_call(src, n, dst, 1);
}

int zpse_d264_forward_avx2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
    return d264_forward_call(src, n, dst, 2);
}

int zpse_d264_inverse_scalar(const uint8_t *src, uint32_t n,
                             uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
    return d264_inverse_call(src, n, dst, 0);
}

int zpse_d264_inverse_sse2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
    return d264_inverse_call(src, n, dst, 1);
}

int zpse_d264_inverse_avx2(const uint8_t *src, uint32_t n,
                           uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
    return d264_inverse_call(src, n, dst, 2);
}

int zpse_d264_forward(const uint8_t *src, uint32_t n,
                      uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
#if D264_X86
    if (d264_caps() & 2u) {
        return d264_forward_call(src, n, dst, 2);
    }
    if (d264_caps() & 1u) {
        return d264_forward_call(src, n, dst, 1);
    }
#endif
    return d264_forward_call(src, n, dst, 0);
}

int zpse_d264_inverse(const uint8_t *src, uint32_t n,
                      uint8_t *dst, uint32_t dst_cap)
{
    int rc = d264_validate(src, n, dst, dst_cap);
    if (rc != ZPSE_D264_OK) return rc;
#if D264_X86
    if (d264_caps() & 2u) {
        return d264_inverse_call(src, n, dst, 2);
    }
    if (d264_caps() & 1u) {
        return d264_inverse_call(src, n, dst, 1);
    }
#endif
    return d264_inverse_call(src, n, dst, 0);
}

/* ---- Gate -------------------------------------------------------------- */

int zpse_d264_gate(const uint8_t *src, uint32_t n)
{
    uint32_t count = n / 8u;
    uint32_t dd_count, zero_dd = 0u;
    unsigned first_bw = 0u, second_bw = 0u;
    uint64_t prev, dprev = 0u;
    uint32_t i;

    if (src == 0 || count < 4u) {
        return 0;
    }

    prev = d264_load64(src);
    /* First deltas d[i] = x[i+1] - x[i], i = 0 .. count-2. */
    for (i = 0u; i + 1u < count; i++) {
        uint64_t cur = d264_load64(src + (size_t)(i + 1u) * 8u);
        uint64_t d = cur - prev;
        unsigned bw = d264_bitlen(d);
        if (bw > first_bw) {
            first_bw = bw;
        }
        prev = cur;
        if (i == 0u) {
            dprev = d;
        } else {
            /* Second delta y[i+1] = d - dprev for i >= 1. */
            uint64_t dd = d - dprev;
            unsigned dbw = d264_bitlen(dd);
            if (dd == 0u) {
                zero_dd++;
            }
            if (dbw > second_bw) {
                second_bw = dbw;
            }
            dprev = d;
        }
    }
    dd_count = count - 2u;

    return (zero_dd >= dd_count / 4u) ||
           (second_bw + 4u <= first_bw) ||
           (second_bw <= 16u);
}
