/* ZPSE transform extension -- ID 12, BITSHUFFLE_16.
 *
 * Portable scalar reference + SSE2 / SSSE3 / AVX2 paths with runtime
 * dispatch. All paths are byte-for-byte identical to the scalar oracle.
 * Unity-build friendly: no globals that are written after init, no heap
 * allocation, all helpers static.
 *
 * Build note: function-target attributes let the SIMD bodies compile even
 * when the TU is built without -mssse3/-mavx2 (the same pattern as
 * S:\coding\zpse\src\vector.c). The integration agent may swap
 * zpse_bs16_cpu_caps() for zpse_cpu_caps() from zpse/vector.h; the bit
 * layout (SSE2=1, SSSE3=2, AVX2=4) is identical.
 */

#include "bitshuffle16.h"

#include <string.h>
#if !defined(_MSC_VER)
#include <stdatomic.h>
#endif

/* out_len = n + (0..15), so widths above UINT32_MAX-15 would wrap the size
 * computation. Real codec blocks are <= 64 KiB; reject the impossible case. */
#define BS16_VALID_N(n) ((n) <= 0xFFFFFFF0u)

/* ---- Architecture / target-attribute plumbing -------------------------- */

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#define ZPSE_BS16_HAVE_X86 1
#else
#define ZPSE_BS16_HAVE_X86 0
#endif

#if ZPSE_BS16_HAVE_X86
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

#if defined(__GNUC__) && ZPSE_BS16_HAVE_X86 && !defined(_MSC_VER)
#define ZPSE_T_SSE2  __attribute__((target("sse2")))
#define ZPSE_T_SSSE3 __attribute__((target("ssse3")))
#define ZPSE_T_AVX2  __attribute__((target("avx2")))
#else
#define ZPSE_T_SSE2
#define ZPSE_T_SSSE3
#define ZPSE_T_AVX2
#endif

/* ---- Feature detection -------------------------------------------------- */

#if ZPSE_BS16_HAVE_X86 && !defined(_MSC_VER)
static unsigned long long bs16_xgetbv0(void)
{
    uint32_t lo = 0;
    uint32_t hi = 0;
    __asm__ volatile(".byte 0x0f,0x01,0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}
#endif

uint32_t zpse_bs16_cpu_caps(void)
{
    uint32_t caps = 0;
#if !defined(_MSC_VER)
    static _Atomic uint32_t cached = 0xFFFFFFFFu;
    caps = atomic_load_explicit(&cached, memory_order_relaxed);
    if (caps != 0xFFFFFFFFu) return caps;
    caps = 0; /* discard the sentinel before accumulating detected bits */
#endif
#if ZPSE_BS16_HAVE_X86
#if defined(_MSC_VER)
    {
        int info[4];
        __cpuid(info, 1);
        if (info[3] & (1 << 26)) caps |= 1u;   /* SSE2 */
        if (info[2] & (1 << 9))  caps |= 2u;   /* SSSE3 */
        if ((info[2] & (1 << 28)) && (info[2] & (1 << 27))) {
            if ((_xgetbv(0) & 0x6ULL) == 0x6ULL) {
                __cpuid(info, 0);
                if (info[0] >= 7) {
                    __cpuidex(info, 7, 0);
                    if (info[1] & (1 << 5)) caps |= 4u; /* AVX2 */
                }
            }
        }
    }
#else
    {
        unsigned int a = 0, b = 0, c = 0, d = 0;
        if (__get_cpuid(1, &a, &b, &c, &d)) {
            if (d & (1u << 26)) caps |= 1u;
            if (c & (1u << 9))  caps |= 2u;
            if ((c & (1u << 28)) && (c & (1u << 27))) {
                if ((bs16_xgetbv0() & 0x6ULL) == 0x6ULL) {
                    if (__get_cpuid_max(0, NULL) >= 7u) {
                        unsigned int a7 = 0, b7 = 0, c7 = 0, d7 = 0;
                        (void)a7; (void)c7; (void)d7;
                        __get_cpuid_count(7, 0, &a7, &b7, &c7, &d7);
                        if (b7 & (1u << 5)) caps |= 4u;
                    }
                }
            }
        }
    }
#endif
#else
    caps = 0;
#endif
#if !defined(_MSC_VER)
    atomic_store_explicit(&cached, caps, memory_order_relaxed);
#endif
    return caps;
}

/* ---- Small helpers ------------------------------------------------------ */

static unsigned bs16_popcount16(uint16_t v)
{
    unsigned n = 0;
    while (v) { n += (unsigned)(v & 1u); v = (uint16_t)(v >> 1); }
    return n;
}

static unsigned bs16_bit_width(uint32_t v)
{
    unsigned n = 0;
    while (v) { n++; v >>= 1; }
    return n;
}

uint32_t zpse_bs16_out_len(uint32_t n)
{
    uint32_t count = n >> 1;
    uint32_t pb = (count + 7u) >> 3;
    return 16u * pb + (n & 1u);
}

/* Zero-padded 16/32-byte loads: never read beyond the logical word region,
 * never touch the odd tail byte. */
static void bs16_load16(uint8_t dst[16], const uint8_t *src, uint32_t avail)
{
    memset(dst, 0, 16);
    if (avail) {
        if (avail > 16u) avail = 16u;
        memcpy(dst, src, (size_t)avail);
    }
}

static void bs16_load32(uint8_t dst[32], const uint8_t *src, uint32_t avail)
{
    memset(dst, 0, 32);
    if (avail) {
        if (avail > 32u) avail = 32u;
        memcpy(dst, src, (size_t)avail);
    }
}

/* =========================================================================
 * SCALAR REFERENCE (format oracle) -- naive bit-by-bit, obviously correct.
 * ========================================================================= */

static void bs16_fwd_scalar_do(const uint8_t *in, uint32_t n,
                               uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t k, p;
    memset(out, 0, (size_t)(16u * pb) + (size_t)(n & 1u));
    for (k = 0; k < count; k++) {
        uint16_t w = (uint16_t)in[2u * k] |
                     ((uint16_t)in[2u * k + 1u] << 8);
        uint32_t j = k >> 3;
        uint32_t b = k & 7u;
        for (p = 0; p < 16u; p++) {
            if ((w >> p) & 1u) {
                out[p * pb + j] = (uint8_t)(out[p * pb + j] | (1u << b));
            }
        }
    }
    if (n & 1u) out[16u * pb] = in[2u * count];
}

static void bs16_inv_scalar_do(const uint8_t *in, uint32_t n,
                               uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t k, p;
    for (k = 0; k < count; k++) {
        uint32_t j = k >> 3;
        uint32_t b = k & 7u;
        uint16_t w = 0;
        for (p = 0; p < 16u; p++) {
            if ((in[p * pb + j] >> b) & 1u) {
                w = (uint16_t)(w | ((uint16_t)1u << p));
            }
        }
        out[2u * k]     = (uint8_t)w;
        out[2u * k + 1u] = (uint8_t)(w >> 8);
    }
    if (n & 1u) out[2u * count] = in[16u * pb];
}

int zpse_bs16_forward_scalar(const uint8_t *in, uint32_t n,
                             uint8_t *out, uint32_t cap)
{
    uint32_t pb, need;
    if (!in || !out || !BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
    need = zpse_bs16_out_len(n);
    if (cap < need) return ZPSE_BS16_TOO_SMALL;
    pb = ((n >> 1) + 7u) >> 3;
    if (need) bs16_fwd_scalar_do(in, n, out, pb);
    return ZPSE_BS16_OK;
}

int zpse_bs16_inverse_scalar(const uint8_t *in, uint32_t n,
                             uint8_t *out, uint32_t cap)
{
    uint32_t pb;
    if (!in || !out || !BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
    if (cap < n) return ZPSE_BS16_TOO_SMALL;
    pb = ((n >> 1) + 7u) >> 3;
    if (n) bs16_inv_scalar_do(in, n, out, pb);
    return ZPSE_BS16_OK;
}

/* =========================================================================
 * SIMD SHARED PRIMITIVES
 *
 * The 8x8 bit transpose of Hacker's Delight is applied independently to each
 * 64-bit lane. In the forward direction a lane holds 8 word-bytes (rows) and
 * the transpose yields 8 plane bytes (columns). The operation is its own
 * inverse, so the same primitive reconstructs words in the inverse path.
 * ========================================================================= */

#if ZPSE_BS16_HAVE_X86

static ZPSE_T_SSE2 __m128i bs16_t8_sse2(__m128i x)
{
    __m128i t;
    t = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64(x, 7)),
                      _mm_set1_epi64x(0x00AA00AA00AA00AAULL));
    x = _mm_xor_si128(x, t); x = _mm_xor_si128(x, _mm_slli_epi64(t, 7));
    t = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64(x, 14)),
                      _mm_set1_epi64x(0x0000CCCC0000CCCCULL));
    x = _mm_xor_si128(x, t); x = _mm_xor_si128(x, _mm_slli_epi64(t, 14));
    t = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi64(x, 28)),
                      _mm_set1_epi64x(0x00000000F0F0F0F0ULL));
    x = _mm_xor_si128(x, t); x = _mm_xor_si128(x, _mm_slli_epi64(t, 28));
    return x;
}

static ZPSE_T_AVX2 __m256i bs16_t8_avx2(__m256i x)
{
    __m256i t;
    t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(x, 7)),
                         _mm256_set1_epi64x(0x00AA00AA00AA00AAULL));
    x = _mm256_xor_si256(x, t); x = _mm256_xor_si256(x, _mm256_slli_epi64(t, 7));
    t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(x, 14)),
                         _mm256_set1_epi64x(0x0000CCCC0000CCCCULL));
    x = _mm256_xor_si256(x, t); x = _mm256_xor_si256(x, _mm256_slli_epi64(t, 14));
    t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(x, 28)),
                         _mm256_set1_epi64x(0x00000000F0F0F0F0ULL));
    x = _mm256_xor_si256(x, t); x = _mm256_xor_si256(x, _mm256_slli_epi64(t, 28));
    return x;
}

/* ---- SSE2 forward: deinterleave with mask+packus, then SWAR transpose ---- */

static ZPSE_T_SSE2 void bs16_fwd_sse2_do(const uint8_t *in, uint32_t n,
                                         uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t groups = (count + 7u) >> 3;
    uint32_t g;
    const __m128i m00ff = _mm_set1_epi16(0x00FF);
    const __m128i mff00 = _mm_set1_epi16(0xFF00);

    memset(out, 0, (size_t)(16u * pb) + (size_t)(n & 1u));

    for (g = 0; g < groups; g++) {
        uint8_t buf[16];
        __m128i v, ev, od, lo8, hi8, tlo, thi;
        uint32_t avail = (2u * count > 16u * g) ? (2u * count - 16u * g) : 0u;
        unsigned p;
        bs16_load16(buf, in + 16u * g, avail);
        v = _mm_loadu_si128((const __m128i *)(const void *)buf);
        ev = _mm_and_si128(v, m00ff);
        lo8 = _mm_packus_epi16(ev, ev);                 /* lo0..lo7 (dup) */
        od = _mm_srli_epi16(_mm_and_si128(v, mff00), 8);
        hi8 = _mm_packus_epi16(od, od);                 /* hi0..hi7 (dup) */
        tlo = bs16_t8_sse2(lo8);
        thi = bs16_t8_sse2(hi8);
        {
            uint8_t lo[16], hi[16];
            _mm_storeu_si128((__m128i *)(void *)lo, tlo);
            _mm_storeu_si128((__m128i *)(void *)hi, thi);
            for (p = 0; p < 8u; p++) {
                out[p * pb + g]       = lo[p];
                out[(p + 8u) * pb + g] = hi[p];
            }
        }
    }
    if (n & 1u) out[16u * pb] = in[2u * count];
}

/* ---- SSSE3 forward: pshufb deinterleave, then the same SWAR transpose ---- */

static ZPSE_T_SSSE3 void bs16_fwd_ssse3_do(const uint8_t *in, uint32_t n,
                                           uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t groups = (count + 7u) >> 3;
    uint32_t g;
    const __m128i sh_lo = _mm_setr_epi8(0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14);
    const __m128i sh_hi = _mm_setr_epi8(1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15);

    memset(out, 0, (size_t)(16u * pb) + (size_t)(n & 1u));

    for (g = 0; g < groups; g++) {
        uint8_t buf[16];
        __m128i v, lo8, hi8, tlo, thi;
        uint32_t avail = (2u * count > 16u * g) ? (2u * count - 16u * g) : 0u;
        unsigned p;
        bs16_load16(buf, in + 16u * g, avail);
        v = _mm_loadu_si128((const __m128i *)(const void *)buf);
        lo8 = _mm_shuffle_epi8(v, sh_lo);
        hi8 = _mm_shuffle_epi8(v, sh_hi);
        tlo = bs16_t8_sse2(lo8);
        thi = bs16_t8_sse2(hi8);
        {
            uint8_t lo[16], hi[16];
            _mm_storeu_si128((__m128i *)(void *)lo, tlo);
            _mm_storeu_si128((__m128i *)(void *)hi, thi);
            for (p = 0; p < 8u; p++) {
                out[p * pb + g]       = lo[p];
                out[(p + 8u) * pb + g] = hi[p];
            }
        }
    }
    if (n & 1u) out[16u * pb] = in[2u * count];
}

/* ---- SSE2 inverse: gather plane bytes, SWAR transpose, interleave ------- */

static ZPSE_T_SSE2 void bs16_inv_sse2_do(const uint8_t *in, uint32_t n,
                                         uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t groups = (count + 7u) >> 3;
    uint32_t g;
    for (g = 0; g < groups; g++) {
        uint8_t plo[16], phi[16], lb[8], hb[8];
        __m128i lo, hi, x, t;
        unsigned p, b;
        /* qword0 = planes 0..7, qword1 = planes 8..15. The transpose runs on
         * both 64-bit lanes in parallel: qword0 of t becomes the 8 low word
         * bytes, qword1 the 8 high word bytes. */
        memset(plo, 0, 16); memset(phi, 0, 16);
        for (p = 0; p < 8u; p++) {
            plo[p] = in[p * pb + g];
            phi[p] = in[(p + 8u) * pb + g];
        }
        lo = _mm_loadl_epi64((const __m128i *)(const void *)plo);
        hi = _mm_loadl_epi64((const __m128i *)(const void *)phi);
        x = _mm_unpacklo_epi64(lo, hi);
        t = bs16_t8_sse2(x);
        _mm_storel_epi64((__m128i *)(void *)lb, t);
        _mm_storel_epi64((__m128i *)(void *)hb, _mm_srli_si128(t, 8));
        {
            uint32_t base = 16u * g;
            uint32_t avail = (2u * count > base) ? (2u * count - base) : 0u;
            if (avail > 16u) avail = 16u;
            for (b = 0; b < 8u && (2u * b) < avail; b++) {
                out[base + 2u * b]     = lb[b];
                out[base + 2u * b + 1u] = hb[b];
            }
        }
    }
    if (n & 1u) out[2u * count] = in[16u * pb];
}

/* =========================================================================
 * AVX2 forward / inverse -- two 8-word groups per iteration.
 * ========================================================================= */

static ZPSE_T_AVX2 void bs16_fwd_avx2_do(const uint8_t *in, uint32_t n,
                                         uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t groups = (count + 7u) >> 3;
    uint32_t g = 0;
    const __m256i sh_lo = _mm256_setr_epi8(
        0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14,
        0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14);
    const __m256i sh_hi = _mm256_setr_epi8(
        1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15,
        1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15);

    memset(out, 0, (size_t)(16u * pb) + (size_t)(n & 1u));

    for (; g + 1 < groups; g += 2) {
        uint8_t buf[32];
        __m256i v, lo16, hi16, tlo, thi;
        uint32_t base = 16u * g;
        uint32_t avail = (2u * count > base) ? (2u * count - base) : 0u;
        unsigned p;
        bs16_load32(buf, in + base, avail);
        v = _mm256_loadu_si256((const __m256i *)(const void *)buf);
        lo16 = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(v, sh_lo), 0x08);
        hi16 = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(v, sh_hi), 0x08);
        tlo = bs16_t8_avx2(lo16);
        thi = bs16_t8_avx2(hi16);
        {
            uint8_t lo[32], hi[32];
            _mm256_storeu_si256((__m256i *)(void *)lo, tlo);
            _mm256_storeu_si256((__m256i *)(void *)hi, thi);
            for (p = 0; p < 8u; p++) {
                out[p * pb + g]         = lo[p];
                out[p * pb + g + 1u]    = lo[8u + p];
                out[(p + 8u) * pb + g]      = hi[p];
                out[(p + 8u) * pb + g + 1u] = hi[8u + p];
            }
        }
    }
    /* scalar cleanup for an odd number of groups (or all of them if < 2) */
    for (; g < groups; g++) {
        uint8_t buf[16];
        __m128i v, ev, od, lo8, hi8, tlo, thi;
        const __m128i m00ff = _mm_set1_epi16(0x00FF);
        const __m128i mff00 = _mm_set1_epi16(0xFF00);
        uint32_t base = 16u * g;
        uint32_t avail = (2u * count > base) ? (2u * count - base) : 0u;
        unsigned p;
        bs16_load16(buf, in + base, avail);
        v = _mm_loadu_si128((const __m128i *)(const void *)buf);
        ev = _mm_and_si128(v, m00ff);
        lo8 = _mm_packus_epi16(ev, ev);
        od = _mm_srli_epi16(_mm_and_si128(v, mff00), 8);
        hi8 = _mm_packus_epi16(od, od);
        tlo = bs16_t8_sse2(lo8);
        thi = bs16_t8_sse2(hi8);
        {
            uint8_t lo[16], hi[16];
            _mm_storeu_si128((__m128i *)(void *)lo, tlo);
            _mm_storeu_si128((__m128i *)(void *)hi, thi);
            for (p = 0; p < 8u; p++) {
                out[p * pb + g]        = lo[p];
                out[(p + 8u) * pb + g] = hi[p];
            }
        }
    }
    if (n & 1u) out[16u * pb] = in[2u * count];
}

static ZPSE_T_AVX2 void bs16_inv_avx2_do(const uint8_t *in, uint32_t n,
                                         uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t groups = (count + 7u) >> 3;
    uint32_t g = 0;
    for (; g + 1 < groups; g += 2) {
        uint8_t blo[32], bhi[32];
        uint8_t lo[32], hi[32];
        __m256i x, t;
        unsigned p, b;
        uint32_t base = 16u * g;
        uint32_t avail = (2u * count > base) ? (2u * count - base) : 0u;
        if (avail > 32u) avail = 32u;
        memset(blo, 0, 32); memset(bhi, 0, 32);
        for (p = 0; p < 8u; p++) {
            blo[p]      = in[p * pb + g];
            blo[8u + p] = in[p * pb + g + 1u];
            bhi[p]      = in[(p + 8u) * pb + g];
            bhi[8u + p] = in[(p + 8u) * pb + g + 1u];
        }
        x = _mm256_loadu_si256((const __m256i *)(const void *)blo);
        t = bs16_t8_avx2(x);
        _mm256_storeu_si256((__m256i *)(void *)lo, t);
        x = _mm256_loadu_si256((const __m256i *)(const void *)bhi);
        t = bs16_t8_avx2(x);
        _mm256_storeu_si256((__m256i *)(void *)hi, t);
        for (b = 0; b < 8u; b++) {
            uint32_t o0 = base + 2u * b;
            uint32_t o1 = base + 16u + 2u * b;
            if (o0 + 1u < base + avail) {
                out[o0] = lo[b];     out[o0 + 1u] = hi[b];
            }
            if (o1 + 1u < base + avail) {
                out[o1] = lo[8u + b]; out[o1 + 1u] = hi[8u + b];
            }
        }
    }
    /* scalar cleanup via the SSE2 inverse helper (one group at a time) */
    for (; g < groups; g++) {
        uint8_t plo[16], phi[16], lb[8], hb[8];
        __m128i x, t, hi;
        unsigned p, b;
        uint32_t base = 16u * g;
        uint32_t avail = (2u * count > base) ? (2u * count - base) : 0u;
        if (avail > 16u) avail = 16u;
        memset(plo, 0, 16); memset(phi, 0, 16);
        for (p = 0; p < 8u; p++) {
            plo[p] = in[p * pb + g];
            phi[p] = in[(p + 8u) * pb + g];
        }
        {
            __m128i qlo = _mm_loadl_epi64((const __m128i *)(const void *)plo);
            __m128i qhi = _mm_loadl_epi64((const __m128i *)(const void *)phi);
            x = _mm_unpacklo_epi64(qlo, qhi);
        }
        t = bs16_t8_sse2(x);
        _mm_storel_epi64((__m128i *)(void *)lb, t);
        hi = _mm_srli_si128(t, 8);
        _mm_storel_epi64((__m128i *)(void *)hb, hi);
        for (b = 0; b < 8u && (2u * b) < avail; b++) {
            out[base + 2u * b]     = lb[b];
            out[base + 2u * b + 1u] = hb[b];
        }
    }
    if (n & 1u) out[2u * count] = in[16u * pb];
}

/* ---- Fast inverse: eight groups per iteration ---------------------------
 * The AVX2 inverse above gathers sixteen plane bytes with sixteen vpinsrb
 * instructions per two groups; that dependent insert chain is the cost of the
 * path (measured 52-79 us for a 64 KiB block). Here eight groups are loaded
 * with eight-byte loads, transposed as two 8x8 byte matrices (planes 0..7 and
 * 8..15) with unpacklo/unpackhi, and the resulting groups are fed to the same
 * SWAR bit transpose as four AVX2 lanes. The scalar tail keeps the format
 * identical for a partial final batch.
 */
static ZPSE_T_AVX2 void bs16_inv_fast8(const uint8_t *in, uint8_t *out,
                                       uint32_t pb, uint32_t g);

static ZPSE_T_AVX2 void bs16_inv_fast_avx2_do(const uint8_t *in, uint32_t n,
                                              uint8_t *out, uint32_t pb)
{
    uint32_t count = n >> 1;
    uint32_t groups = (count + 7u) >> 3;
    uint32_t g = 0u;

    for (; g + 32u <= groups; g += 32u) {
        bs16_inv_fast8(in, out, pb, g);
        bs16_inv_fast8(in, out, pb, g + 8u);
        bs16_inv_fast8(in, out, pb, g + 16u);
        bs16_inv_fast8(in, out, pb, g + 24u);
    }
    for (; g + 8u <= groups; g += 8u) {
        bs16_inv_fast8(in, out, pb, g);
    }
    for (; g < groups; g++) {
        uint8_t plo[8], phi[8], lb[8], hb[8];
        __m128i x, t, hi;
        unsigned p, b;
        for (p = 0u; p < 8u; p++) {
            plo[p] = in[p * pb + g];
            phi[p] = in[(p + 8u) * pb + g];
        }
        x = _mm_unpacklo_epi64(
                _mm_loadl_epi64((const __m128i *)(const void *)plo),
                _mm_loadl_epi64((const __m128i *)(const void *)phi));
        t = bs16_t8_sse2(x);
        _mm_storel_epi64((__m128i *)(void *)lb, t);
        hi = _mm_srli_si128(t, 8);
        _mm_storel_epi64((__m128i *)(void *)hb, hi);
        for (b = 0u; b < 8u; b++) {
            out[16u * g + 2u * b]     = lb[b];
            out[16u * g + 2u * b + 1u] = hb[b];
        }
    }
    if (n & 1u) out[2u * count] = in[16u * pb];
}

static ZPSE_T_AVX2 void bs16_inv_fast8(const uint8_t *in, uint8_t *out,
                                       uint32_t pb, uint32_t g)
{
    { /* one eight-group batch */
        __m128i a[8], b[8], t0, t1, t2, t3, t4, t5, t6, t7;
        __m128i u0, u1, u2, u3, v0, v1, v2, v3;
        __m256i yl0, yl1, yh0, yh1;
        __m128i l0, l1, h0, h1;
        unsigned p;

        for (p = 0u; p < 8u; p++) {
            a[p] = _mm_loadl_epi64((const __m128i *)(const void *)(in + p * pb + g));
            b[p] = _mm_loadl_epi64((const __m128i *)(const void *)(in + (p + 8u) * pb + g));
        }
        t0 = _mm_unpacklo_epi8(a[0], a[1]); t1 = _mm_unpacklo_epi8(a[2], a[3]);
        t2 = _mm_unpacklo_epi8(a[4], a[5]); t3 = _mm_unpacklo_epi8(a[6], a[7]);
        u0 = _mm_unpacklo_epi16(t0, t1); u1 = _mm_unpackhi_epi16(t0, t1);
        u2 = _mm_unpacklo_epi16(t2, t3); u3 = _mm_unpackhi_epi16(t2, t3);
        v0 = _mm_unpacklo_epi32(u0, u2); v1 = _mm_unpackhi_epi32(u0, u2);
        v2 = _mm_unpacklo_epi32(u1, u3); v3 = _mm_unpackhi_epi32(u1, u3);
        yl0 = bs16_t8_avx2(_mm256_set_m128i(v1, v0));
        yl1 = bs16_t8_avx2(_mm256_set_m128i(v3, v2));

        t4 = _mm_unpacklo_epi8(b[0], b[1]); t5 = _mm_unpacklo_epi8(b[2], b[3]);
        t6 = _mm_unpacklo_epi8(b[4], b[5]); t7 = _mm_unpacklo_epi8(b[6], b[7]);
        u0 = _mm_unpacklo_epi16(t4, t5); u1 = _mm_unpackhi_epi16(t4, t5);
        u2 = _mm_unpacklo_epi16(t6, t7); u3 = _mm_unpackhi_epi16(t6, t7);
        v0 = _mm_unpacklo_epi32(u0, u2); v1 = _mm_unpackhi_epi32(u0, u2);
        v2 = _mm_unpacklo_epi32(u1, u3); v3 = _mm_unpackhi_epi32(u1, u3);
        yh0 = bs16_t8_avx2(_mm256_set_m128i(v1, v0));
        yh1 = bs16_t8_avx2(_mm256_set_m128i(v3, v2));

        /* Interleave low/high bytes in registers: no staging round trip. */
        l0 = _mm256_castsi256_si128(yl0); l1 = _mm256_extracti128_si256(yl0, 1);
        h0 = _mm256_castsi256_si128(yh0); h1 = _mm256_extracti128_si256(yh0, 1);
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * g),
                         _mm_unpacklo_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 1u)),
                         _mm_unpackhi_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 2u)),
                         _mm_unpacklo_epi8(l1, h1));
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 3u)),
                         _mm_unpackhi_epi8(l1, h1));
        l0 = _mm256_castsi256_si128(yl1); l1 = _mm256_extracti128_si256(yl1, 1);
        h0 = _mm256_castsi256_si128(yh1); h1 = _mm256_extracti128_si256(yh1, 1);
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 4u)),
                         _mm_unpacklo_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 5u)),
                         _mm_unpackhi_epi8(l0, h0));
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 6u)),
                         _mm_unpacklo_epi8(l1, h1));
        _mm_storeu_si128((__m128i *)(void *)(out + 16u * (g + 7u)),
                         _mm_unpackhi_epi8(l1, h1));
    }
}

#endif /* ZPSE_BS16_HAVE_X86 */

/* ---- Public per-path entry points --------------------------------------- */

static int bs16_check(const uint8_t *in, const uint8_t *out, uint32_t cap,
                      uint32_t need)
{
    if (!in || !out) return ZPSE_BS16_BAD_ARG;
    if (cap < need) return ZPSE_BS16_TOO_SMALL;
    return ZPSE_BS16_OK;
}

int zpse_bs16_forward_sse2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t cap)
{
    uint32_t pb, need;
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
#if ZPSE_BS16_HAVE_X86
    need = zpse_bs16_out_len(n);
    rc = bs16_check(in, out, cap, need);
    if (rc) return rc;
    pb = ((n >> 1) + 7u) >> 3;
    if (need) bs16_fwd_sse2_do(in, n, out, pb);
    return ZPSE_BS16_OK;
#else
    (void)pb; (void)need;
    return zpse_bs16_forward_scalar(in, n, out, cap);
#endif
}

int zpse_bs16_inverse_sse2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t cap)
{
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
#if ZPSE_BS16_HAVE_X86
    rc = bs16_check(in, out, cap, n);
    if (rc) return rc;
    if (n) bs16_inv_sse2_do(in, n, out, ((n >> 1) + 7u) >> 3);
    return ZPSE_BS16_OK;
#else
    (void)rc;
    return zpse_bs16_inverse_scalar(in, n, out, cap);
#endif
}

int zpse_bs16_forward_ssse3(const uint8_t *in, uint32_t n,
                            uint8_t *out, uint32_t cap)
{
    uint32_t pb, need;
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
#if ZPSE_BS16_HAVE_X86
    need = zpse_bs16_out_len(n);
    rc = bs16_check(in, out, cap, need);
    if (rc) return rc;
    pb = ((n >> 1) + 7u) >> 3;
    if (need) bs16_fwd_ssse3_do(in, n, out, pb);
    return ZPSE_BS16_OK;
#else
    (void)pb; (void)need; (void)rc;
    return zpse_bs16_forward_scalar(in, n, out, cap);
#endif
}

int zpse_bs16_inverse_ssse3(const uint8_t *in, uint32_t n,
                            uint8_t *out, uint32_t cap)
{
    /* The SSSE3 inverse uses the same SWAR primitive as SSE2 (pshufb does not
     * help the gather/transpose direction), so it is the identical kernel. */
    return zpse_bs16_inverse_sse2(in, n, out, cap);
}

int zpse_bs16_forward_avx2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t cap)
{
    uint32_t pb, need;
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
#if ZPSE_BS16_HAVE_X86
    need = zpse_bs16_out_len(n);
    rc = bs16_check(in, out, cap, need);
    if (rc) return rc;
    pb = ((n >> 1) + 7u) >> 3;
    if (need) bs16_fwd_avx2_do(in, n, out, pb);
    return ZPSE_BS16_OK;
#else
    (void)pb; (void)need; (void)rc;
    return zpse_bs16_forward_scalar(in, n, out, cap);
#endif
}

int zpse_bs16_inverse_avx2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t cap)
{
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
#if ZPSE_BS16_HAVE_X86
    rc = bs16_check(in, out, cap, n);
    if (rc) return rc;
    if (n) bs16_inv_avx2_do(in, n, out, ((n >> 1) + 7u) >> 3);
    return ZPSE_BS16_OK;
#else
    (void)rc;
    return zpse_bs16_inverse_scalar(in, n, out, cap);
#endif
}

/* ---- Runtime dispatch --------------------------------------------------- */

int zpse_bs16_forward(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap)
{
    uint32_t need;
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
    need = zpse_bs16_out_len(n);
    rc = bs16_check(in, out, cap, need);
    if (rc) return rc;
#if ZPSE_BS16_HAVE_X86
    {
        uint32_t caps = zpse_bs16_cpu_caps();
        if (caps & 4u) {
            if (need) bs16_fwd_avx2_do(in, n, out, ((n >> 1) + 7u) >> 3);
            return ZPSE_BS16_OK;
        }
        if (caps & 2u) {
            if (need) bs16_fwd_ssse3_do(in, n, out, ((n >> 1) + 7u) >> 3);
            return ZPSE_BS16_OK;
        }
        if (caps & 1u) {
            if (need) bs16_fwd_sse2_do(in, n, out, ((n >> 1) + 7u) >> 3);
            return ZPSE_BS16_OK;
        }
    }
#endif
    if (need) bs16_fwd_scalar_do(in, n, out, ((n >> 1) + 7u) >> 3);
    return ZPSE_BS16_OK;
}

int zpse_bs16_inverse(const uint8_t *in, uint32_t n, uint8_t *out, uint32_t cap)
{
    int rc;
    if (!BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
    rc = bs16_check(in, out, cap, n);
    if (rc) return rc;
    if (!n) return ZPSE_BS16_OK;
#if ZPSE_BS16_HAVE_X86
    {
        uint32_t caps = zpse_bs16_cpu_caps();
        if (caps & 4u) {
#if !defined(ZPSE_BS16_FAST_INV) || ZPSE_BS16_FAST_INV
            bs16_inv_fast_avx2_do(in, n, out, ((n >> 1) + 7u) >> 3);
#else
            bs16_inv_avx2_do(in, n, out, ((n >> 1) + 7u) >> 3);
#endif
            return ZPSE_BS16_OK;
        }
        if (caps & 2u) {
            bs16_inv_sse2_do(in, n, out, ((n >> 1) + 7u) >> 3);
            return ZPSE_BS16_OK;
        }
        if (caps & 1u) {
            bs16_inv_sse2_do(in, n, out, ((n >> 1) + 7u) >> 3);
            return ZPSE_BS16_OK;
        }
    }
#endif
    bs16_inv_scalar_do(in, n, out, ((n >> 1) + 7u) >> 3);
    return ZPSE_BS16_OK;
}

/* ---- Gate ---------------------------------------------------------------- */

int zpse_bs16_gate(const uint8_t *in, uint32_t n)
{
    uint32_t count;
    uint32_t ns, step, i;
    uint16_t mn = 0xFFFFu;
    uint16_t ag_or = 0u;
    uint16_t ag_and = 0xFFFFu;
    uint32_t or_diff = 0u;
    unsigned required_bits, varying_bits;

    if (!in || !BS16_VALID_N(n)) return ZPSE_BS16_BAD_ARG;
    if (n < 256u) return ZPSE_BS16_REJECT;
    count = n >> 1;
    if (count < 16u) return ZPSE_BS16_REJECT;   /* minimum 16 complete words */

    ns = count < 256u ? count : 256u;
    step = count / ns;
    if (!step) step = 1u;

    for (i = 0; i < ns; i++) {
        uint32_t k = i * step;
        uint16_t v = (uint16_t)in[2u * k] |
                     ((uint16_t)in[2u * k + 1u] << 8);
        if (v < mn) mn = v;
        ag_or  = (uint16_t)(ag_or | v);
        ag_and = (uint16_t)(ag_and & v);
    }
    for (i = 0; i < ns; i++) {
        uint32_t k = i * step;
        uint16_t v = (uint16_t)in[2u * k] |
                     ((uint16_t)in[2u * k + 1u] << 8);
        or_diff |= (uint32_t)(v - mn);
    }

    required_bits = bs16_bit_width(or_diff);
    varying_bits  = bs16_popcount16((uint16_t)(ag_or ^ ag_and));

    if (required_bits <= (ZPSE_BS16_B / 2u) ||
        varying_bits <= (3u * ZPSE_BS16_B / 4u)) {
        return ZPSE_BS16_OK;
    }
    return ZPSE_BS16_REJECT;
}
