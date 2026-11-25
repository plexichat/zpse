#include "bitshuffle32.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Architecture / dispatch plumbing.                                  */
/* Mirrors S:\coding\zpse\src\vector.c:45-53 (target attributes) and    */
/* include/zpse/vector.h (capability bit layout). When integrating,    */
/* define ZPSE_BS32_USE_ZPSE_CAPS and include zpse/vector.h so the      */
/* codec's own cached CPUID result is used instead of this local one.  */
/* ------------------------------------------------------------------ */

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#define BS32_X86 1
#else
#define BS32_X86 0
#endif

#if BS32_X86
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>
#endif

#if defined(__GNUC__) && BS32_X86 && !defined(_MSC_VER)
#define BS32_TARGET_SSE2  __attribute__((target("sse2")))
#define BS32_TARGET_SSSE3 __attribute__((target("ssse3")))
#define BS32_TARGET_AVX2  __attribute__((target("avx2")))
#else
#define BS32_TARGET_SSE2
#define BS32_TARGET_SSSE3
#define BS32_TARGET_AVX2
#endif

#ifdef ZPSE_BS32_USE_ZPSE_CAPS
#include "zpse/vector.h"
#define BS32_CAPS() zpse_cpu_caps()
#else
#if !defined(_MSC_VER)
#include <stdatomic.h>
#endif
#ifndef ZPSE_FORCE_SCALAR
/* Only the runtime-dispatch builds need CPUID detection; the forced-scalar
 * parity build returns 0 from zpse_bs32_cpu_caps() without calling any of it. */
static void bs32_cpuid(int leaf, int sub,
                       uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d)
{
#if BS32_X86 && defined(_MSC_VER)
    int info[4];
    (void)leaf;
    (void)sub;
    __cpuid(info, 1);
    *a = (uint32_t)info[0]; *b = (uint32_t)info[1];
    *c = (uint32_t)info[2]; *d = (uint32_t)info[3];
#else
    (void)b;
    __asm__ __volatile__("cpuid"
                         : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                         : "a"(leaf), "c"(sub));
#endif
}

static uint64_t bs32_xgetbv0(void)
{
#if BS32_X86 && defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t lo = 0;
    uint32_t hi = 0;
    __asm__ __volatile__(".byte 0x0f,0x01,0xd0"
                         : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#endif
}

static uint32_t bs32_detect_caps(void)
{
    uint32_t caps = 0;
    uint32_t a = 0, b = 0, c = 0, d = 0;
    uint32_t maxleaf;

    if (!BS32_X86) {
        return 0;
    }
    bs32_cpuid(0, 0, &maxleaf, &b, &c, &d);
    if (maxleaf < 1u) {
        return 0;
    }
    bs32_cpuid(1, 0, &a, &b, &c, &d);
    if (d & (1u << 26)) {
        caps |= 1u << 0; /* SSE2 */
    }
    if (c & (1u << 9)) {
        caps |= 1u << 1; /* SSSE3 */
    }
    if ((c & (1u << 28)) && (c & (1u << 27))) {
        uint64_t xcr0 = bs32_xgetbv0();
        if ((xcr0 & 0x6ULL) == 0x6ULL && maxleaf >= 7u) {
            uint32_t a7 = 0, b7 = 0, c7 = 0, d7 = 0;
            bs32_cpuid(7, 0, &a7, &b7, &c7, &d7);
            (void)a7; (void)c7; (void)d7;
            if (b7 & (1u << 5)) {
                caps |= 1u << 2; /* AVX2 */
            }
        }
    }
    return caps;
}
#endif /* !ZPSE_FORCE_SCALAR */

uint32_t zpse_bs32_cpu_caps(void)
{
#ifdef ZPSE_FORCE_SCALAR
    return 0;
#elif !defined(_MSC_VER)
    static _Atomic uint32_t cached = 0xFFFFFFFFu;
    uint32_t caps = atomic_load_explicit(&cached, memory_order_relaxed);
    if (caps != 0xFFFFFFFFu) {
        return caps;
    }
    caps = bs32_detect_caps();
    atomic_store_explicit(&cached, caps, memory_order_relaxed);
    return caps;
#else
    static uint32_t cached = 0xFFFFFFFFu;
    if (cached == 0xFFFFFFFFu) {
        cached = bs32_detect_caps();
    }
    return cached;
#endif
}
#define BS32_CAPS() zpse_bs32_cpu_caps()
#endif /* ZPSE_BS32_USE_ZPSE_CAPS */

/* ------------------------------------------------------------------ */
/* Geometry and validation.                                           */
/* ------------------------------------------------------------------ */

typedef struct bs32_geo {
    uint32_t count;     /* complete 4-byte words */
    uint32_t tail;      /* 0..3 unshuffled bytes */
    uint32_t pbytes;    /* ceil(count/8): bytes per bit plane */
    uint64_t out_len;   /* 32*pbytes + tail */
} bs32_geo;

static int bs32_geometry(uint32_t n, bs32_geo *g)
{
    uint64_t len;
    if (g == NULL) {
        return ZPSE_BS32_BAD_ARG;
    }
    g->count = n >> 2;
    g->tail = n & 3u;
    g->pbytes = (g->count + 7u) >> 3;
    len = (uint64_t)g->pbytes * 32u + (uint64_t)g->tail;
    if (len > (uint64_t)UINT32_MAX) {
        return ZPSE_BS32_OVERFLOW;
    }
    g->out_len = len;
    return ZPSE_BS32_OK;
}

static int bs32_validate(const uint8_t *in, uint32_t n,
                         uint8_t *out, uint32_t cap, bs32_geo *g,
                         int inverse)
{
    int rc = bs32_geometry(n, g);
    uint64_t need;
    if (rc != ZPSE_BS32_OK) {
        return rc;
    }
    if (in == NULL || out == NULL) {
        return ZPSE_BS32_BAD_ARG;
    }
    /* Forward writes the shuffled length; inverse writes n bytes. */
    need = inverse ? (uint64_t)n : g->out_len;
    if ((uint64_t)cap < need) {
        return ZPSE_BS32_CAPACITY;
    }
    return ZPSE_BS32_OK;
}

int zpse_bs32_output_len(uint32_t n, uint32_t *out_len)
{
    bs32_geo g;
    int rc = bs32_geometry(n, &g);
    if (rc == ZPSE_BS32_OK) {
        if (out_len == NULL) {
            return ZPSE_BS32_BAD_ARG;
        }
        *out_len = (uint32_t)g.out_len;
    }
    return rc;
}

static inline uint32_t bs32_load32le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void bs32_store32le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/* ------------------------------------------------------------------ */
/* Scalar oracle (normative).                                         */
/* ------------------------------------------------------------------ */

static void bs32_fwd_scalar_range(const uint8_t *in, uint32_t k0, uint32_t k1,
                                  uint32_t pbytes, uint8_t *out)
{
    uint32_t k;
    for (k = k0; k < k1; k++) {
        uint32_t w = bs32_load32le(in + 4u * k);
        uint32_t p;
        uint8_t *dst = out + (size_t)(k >> 3);
        for (p = 0; p < 32; p++) {
            if ((w >> p) & 1u) {
                dst[(size_t)p * pbytes] |= (uint8_t)(1u << (k & 7u));
            }
        }
    }
}

static void bs32_inv_scalar_range(const uint8_t *in, uint32_t k0, uint32_t k1,
                                  uint32_t pbytes, uint8_t *out)
{
    uint32_t k;
    for (k = k0; k < k1; k++) {
        uint32_t w = 0;
        uint32_t p;
        const uint8_t *src = in + (size_t)(k >> 3);
        for (p = 0; p < 32; p++) {
            w |= (((uint32_t)src[(size_t)p * pbytes] >> (k & 7u)) & 1u) << p;
        }
        bs32_store32le(out + 4u * k, w);
    }
}

/* ------------------------------------------------------------------ */
/* SIMD constants.                                                    */
/* ------------------------------------------------------------------ */

/* Byte-in-lane transpose: out[4*q+i] = in[4*i+q] (q,i in 0..3). */
static const uint8_t bs32_transpose16[16] = {
     0,  4,  8, 12,
     1,  5,  9, 13,
     2,  6, 10, 14,
     3,  7, 11, 15
};
static const uint8_t bs32_transpose32[32] = {
     0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15,
     0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15
};
static const uint8_t bs32_perm_lanes[32] = { /* AVX2 lane reorder 0,4,1,5,2,6,3,7 */
     0,0,0,0, 4,0,0,0, 1,0,0,0, 5,0,0,0,
     2,0,0,0, 6,0,0,0, 3,0,0,0, 7,0,0,0
};

/* For bit p, b = p>>3 is the byte of the word; shuf_b places the 1-bit
 * value (one bit per word) at byte b of every 4-byte lane. 0x80 zeroes
 * the other destinations. Row index is p>>3 (0..3). */
static const uint8_t bs32_shuf_b16[4][16] = {
    { 0, 0x80, 0x80, 0x80,  1, 0x80, 0x80, 0x80,
      2, 0x80, 0x80, 0x80,  3, 0x80, 0x80, 0x80 },
    { 0x80, 0, 0x80, 0x80,  0x80, 1, 0x80, 0x80,
      0x80, 2, 0x80, 0x80,  0x80, 3, 0x80, 0x80 },
    { 0x80, 0x80, 0, 0x80,  0x80, 0x80, 1, 0x80,
      0x80, 0x80, 2, 0x80,  0x80, 0x80, 3, 0x80 },
    { 0x80, 0x80, 0x80, 0,  0x80, 0x80, 0x80, 1,
      0x80, 0x80, 0x80, 2,  0x80, 0x80, 0x80, 3 }
};
/* ------------------------------------------------------------------ */
/* 128-bit SSSE3 kernels (process 8 words = two 128-bit vectors).     */
/* ------------------------------------------------------------------ */

#if BS32_X86
static BS32_TARGET_SSSE3 void bs32_fwd_sse_kernel(
    const uint8_t *in, uint32_t count, uint32_t pbytes, uint8_t *out)
{
    const __m128i mask80 = _mm_set1_epi8((char)0x80);
    const __m128i tr = _mm_loadu_si128((const __m128i *)(const void *)bs32_transpose16);
    uint32_t k;

    for (k = 0; k + 8u <= count; k += 8u) {
        __m128i a = _mm_loadu_si128((const __m128i *)(const void *)(in + 4u * k));
        __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(in + 4u * k + 16u));
        uint32_t g = k >> 3;
#define BS32_FWD_SSE_STEP(PP)                                                  \
        {                                                                      \
            __m128i ma = _mm_and_si128(_mm_slli_epi32(a, 7 - (PP)), mask80);   \
            __m128i mb = _mm_and_si128(_mm_slli_epi32(b, 7 - (PP)), mask80);   \
            uint32_t va = (uint32_t)_mm_movemask_epi8(_mm_shuffle_epi8(ma, tr));\
            uint32_t vb = (uint32_t)_mm_movemask_epi8(_mm_shuffle_epi8(mb, tr));\
            out[((PP) +  0u) * pbytes + g] =                                   \
                (uint8_t)((va & 0xFu) | ((vb & 0xFu) << 4));                   \
            out[((PP) +  8u) * pbytes + g] =                                   \
                (uint8_t)(((va >> 4) & 0xFu) | (((vb >> 4) & 0xFu) << 4));     \
            out[((PP) + 16u) * pbytes + g] =                                   \
                (uint8_t)(((va >> 8) & 0xFu) | (((vb >> 8) & 0xFu) << 4));     \
            out[((PP) + 24u) * pbytes + g] =                                   \
                (uint8_t)(((va >> 12) & 0xFu) | (((vb >> 12) & 0xFu) << 4));   \
        }
        BS32_FWD_SSE_STEP(0)
        BS32_FWD_SSE_STEP(1)
        BS32_FWD_SSE_STEP(2)
        BS32_FWD_SSE_STEP(3)
        BS32_FWD_SSE_STEP(4)
        BS32_FWD_SSE_STEP(5)
        BS32_FWD_SSE_STEP(6)
        BS32_FWD_SSE_STEP(7)
#undef BS32_FWD_SSE_STEP
    }
}

static BS32_TARGET_SSSE3 void bs32_inv_sse_kernel(
    const uint8_t *in, uint32_t count, uint32_t pbytes, uint8_t *out)
{
    /* 8 words per iteration = two 128-bit vectors of 4 words each.
     * Words 0..3 read plane bits 0..3; words 4..7 read plane bits 4..7. */
    const __m128i bm_lo =
        _mm_setr_epi8(1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i bm_hi =
        _mm_setr_epi8(16, 32, 64, (char)128, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0);
    uint32_t k;

    for (k = 0; k + 8u <= count; k += 8u) {
        __m128i lo = _mm_setzero_si128();
        __m128i hi = _mm_setzero_si128();
        uint32_t p;
        for (p = 0; p < 32; p++) {
            uint8_t plane = in[(size_t)p * pbytes + (k >> 3)];
            __m128i val = _mm_set1_epi8((char)(1u << (p & 7u)));
            __m128i shuf = _mm_loadu_si128(
                (const __m128i *)(const void *)bs32_shuf_b16[p >> 3]);
            __m128i tl = _mm_and_si128(_mm_set1_epi8((char)plane), bm_lo);
            __m128i th = _mm_and_si128(_mm_set1_epi8((char)plane), bm_hi);
            lo = _mm_or_si128(lo, _mm_shuffle_epi8(
                _mm_and_si128(_mm_cmpeq_epi8(tl, bm_lo), val), shuf));
            hi = _mm_or_si128(hi, _mm_shuffle_epi8(
                _mm_and_si128(_mm_cmpeq_epi8(th, bm_hi), val), shuf));
        }
        _mm_storeu_si128((__m128i *)(void *)(out + 4u * k), lo);
        _mm_storeu_si128((__m128i *)(void *)(out + 4u * k + 16u), hi);
    }
}
#endif /* BS32_X86 */

/* ------------------------------------------------------------------ */
/* 256-bit AVX2 kernels (process 8 words = one 256-bit vector).       */
/* ------------------------------------------------------------------ */

#if BS32_X86
static BS32_TARGET_AVX2 void bs32_fwd_avx2_kernel(
    const uint8_t *in, uint32_t count, uint32_t pbytes, uint8_t *out)
{
    const __m256i mask80 = _mm256_set1_epi8((char)0x80);
    const __m256i tr =
        _mm256_loadu_si256((const __m256i *)(const void *)bs32_transpose32);
    const __m256i perm = _mm256_loadu_si256((const __m256i *)(const void *)bs32_perm_lanes);
    uint32_t k;

    for (k = 0; k + 8u <= count; k += 8u) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(const void *)(in + 4u * k));
        uint32_t g = k >> 3;
#define BS32_FWD_AVX2_STEP(PP)                                                 \
        {                                                                      \
            __m256i m = _mm256_and_si256(_mm256_slli_epi32(x, 7 - (PP)),       \
                                         mask80);                              \
            __m256i s = _mm256_shuffle_epi8(m, tr);                            \
            __m256i q = _mm256_permutevar8x32_epi32(s, perm);                  \
            uint32_t v = (uint32_t)_mm256_movemask_epi8(q);                    \
            out[((PP) +  0u) * pbytes + g] = (uint8_t)(v);                     \
            out[((PP) +  8u) * pbytes + g] = (uint8_t)(v >> 8);                \
            out[((PP) + 16u) * pbytes + g] = (uint8_t)(v >> 16);               \
            out[((PP) + 24u) * pbytes + g] = (uint8_t)(v >> 24);               \
        }
        BS32_FWD_AVX2_STEP(0)
        BS32_FWD_AVX2_STEP(1)
        BS32_FWD_AVX2_STEP(2)
        BS32_FWD_AVX2_STEP(3)
        BS32_FWD_AVX2_STEP(4)
        BS32_FWD_AVX2_STEP(5)
        BS32_FWD_AVX2_STEP(6)
        BS32_FWD_AVX2_STEP(7)
#undef BS32_FWD_AVX2_STEP
    }
}

static BS32_TARGET_AVX2 void bs32_inv_avx2_kernel(
    const uint8_t *in, uint32_t count, uint32_t pbytes, uint8_t *out)
{
    /* pshufb is per-128-bit-lane, so each lane handles its own 4 words.
     * Low lane bits 0..3 of the plane feed words 0..3, high lane bits
     * 4..7 feed words 4..7; both use the same 16-byte shuffle row. */
    const __m256i bm = _mm256_setr_epi8(
        1, 2, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    uint32_t k;

    for (k = 0; k + 8u <= count; k += 8u) {
        __m256i acc = _mm256_setzero_si256();
        uint32_t p;
        for (p = 0; p < 32; p++) {
            uint8_t plane = in[(size_t)p * pbytes + (k >> 3)];
            __m128i shuf128 = _mm_loadu_si128(
                (const __m128i *)(const void *)bs32_shuf_b16[p >> 3]);
            __m256i shuf = _mm256_set_m128i(shuf128, shuf128);
            __m256i t = _mm256_and_si256(_mm256_set1_epi8((char)plane), bm);
            __m256i eq = _mm256_cmpeq_epi8(t, bm);
            __m256i val = _mm256_and_si256(eq,
                _mm256_set1_epi8((char)(1u << (p & 7u))));
            acc = _mm256_or_si256(acc, _mm256_shuffle_epi8(val, shuf));
        }
        _mm256_storeu_si256((__m256i *)(void *)(out + 4u * k), acc);
    }
}
#endif /* BS32_X86 */

/* ------------------------------------------------------------------ */
/* Public forward / inverse (dispatch + scalar epilogue).             */
/* ------------------------------------------------------------------ */

typedef enum bs32_path {
    BS32_PATH_AUTO = 0,
    BS32_PATH_SCALAR = 1,
    BS32_PATH_SSE = 2,
    BS32_PATH_AVX2 = 3
} bs32_path;

static int bs32_forward_ex(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t cap, bs32_path path)
{
    bs32_geo g;
    uint32_t full;
    int rc = bs32_validate(in, n, out, cap, &g, 0);
    if (rc != ZPSE_BS32_OK) {
        return rc;
    }
    memset(out, 0, (size_t)g.out_len);
    full = g.count & ~7u;

    if (full != 0u) {
        int used = BS32_PATH_SCALAR;
#if BS32_X86
        uint32_t caps = BS32_CAPS();
        if (path == BS32_PATH_AUTO) {
            if (caps & (1u << 2)) {
                used = BS32_PATH_AVX2;
            } else if (caps & (1u << 1)) {
                used = BS32_PATH_SSE;
            }
        } else {
            used = (int)path;
        }
        if (used == BS32_PATH_AVX2 && (caps & (1u << 2))) {
            bs32_fwd_avx2_kernel(in, full, g.pbytes, out);
        } else if (used == BS32_PATH_SSE && (caps & (1u << 1))) {
            bs32_fwd_sse_kernel(in, full, g.pbytes, out);
        } else
#endif
        {
            bs32_fwd_scalar_range(in, 0, full, g.pbytes, out);
        }
    }
    if (g.count > full) {
        bs32_fwd_scalar_range(in, full, g.count, g.pbytes, out);
    }
    if (g.tail != 0u) {
        memcpy(out + (size_t)(32u * g.pbytes), in + (size_t)(4u * g.count),
               g.tail);
    }
    return ZPSE_BS32_OK;
}

static int bs32_inverse_ex(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t cap, bs32_path path)
{
    bs32_geo g;
    uint32_t full;
    int rc = bs32_validate(in, n, out, cap, &g, 1);
    if (rc != ZPSE_BS32_OK) {
        return rc;
    }
    full = g.count & ~7u;

    if (full != 0u) {
        int used = BS32_PATH_SCALAR;
#if BS32_X86
        uint32_t caps = BS32_CAPS();
        if (path == BS32_PATH_AUTO) {
            if (caps & (1u << 2)) {
                used = BS32_PATH_AVX2;
            } else if (caps & (1u << 1)) {
                used = BS32_PATH_SSE;
            }
        } else {
            used = (int)path;
        }
        if (used == BS32_PATH_AVX2 && (caps & (1u << 2))) {
            bs32_inv_avx2_kernel(in, full, g.pbytes, out);
        } else if (used == BS32_PATH_SSE && (caps & (1u << 1))) {
            bs32_inv_sse_kernel(in, full, g.pbytes, out);
        } else
#endif
        {
            bs32_inv_scalar_range(in, 0, full, g.pbytes, out);
        }
    }
    if (g.count > full) {
        bs32_inv_scalar_range(in, full, g.count, g.pbytes, out);
    }
    if (g.tail != 0u) {
        memcpy(out + (size_t)(4u * g.count),
               in + (size_t)(32u * g.pbytes), g.tail);
    }
    return ZPSE_BS32_OK;
}

int zpse_bs32_forward(const uint8_t *in, uint32_t n,
                      uint8_t *out, uint32_t out_cap)
{
    return bs32_forward_ex(in, n, out, out_cap, BS32_PATH_AUTO);
}

int zpse_bs32_inverse(const uint8_t *in, uint32_t n,
                      uint8_t *out, uint32_t out_cap)
{
    return bs32_inverse_ex(in, n, out, out_cap, BS32_PATH_AUTO);
}

int zpse_bs32_forward_scalar(const uint8_t *in, uint32_t n,
                             uint8_t *out, uint32_t out_cap)
{
    return bs32_forward_ex(in, n, out, out_cap, BS32_PATH_SCALAR);
}

int zpse_bs32_inverse_scalar(const uint8_t *in, uint32_t n,
                             uint8_t *out, uint32_t out_cap)
{
    return bs32_inverse_ex(in, n, out, out_cap, BS32_PATH_SCALAR);
}

int zpse_bs32_forward_sse(const uint8_t *in, uint32_t n,
                          uint8_t *out, uint32_t out_cap)
{
    return bs32_forward_ex(in, n, out, out_cap, BS32_PATH_SSE);
}

int zpse_bs32_inverse_sse(const uint8_t *in, uint32_t n,
                          uint8_t *out, uint32_t out_cap)
{
    return bs32_inverse_ex(in, n, out, out_cap, BS32_PATH_SSE);
}

int zpse_bs32_forward_avx2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap)
{
    return bs32_forward_ex(in, n, out, out_cap, BS32_PATH_AVX2);
}

int zpse_bs32_inverse_avx2(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap)
{
    return bs32_inverse_ex(in, n, out, out_cap, BS32_PATH_AVX2);
}

/* ------------------------------------------------------------------ */
/* Gate.                                                              */
/* ------------------------------------------------------------------ */

static unsigned bs32_popcount32(uint32_t v)
{
    static const uint8_t bits[16] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    };
    return (unsigned)bits[ v        & 0xFu] + (unsigned)bits[(v >>  4) & 0xFu] +
           (unsigned)bits[(v >>  8) & 0xFu] + (unsigned)bits[(v >> 12) & 0xFu] +
           (unsigned)bits[(v >> 16) & 0xFu] + (unsigned)bits[(v >> 20) & 0xFu] +
           (unsigned)bits[(v >> 24) & 0xFu] + (unsigned)bits[(v >> 28) & 0xFu];
}

static unsigned bs32_bit_width(uint32_t v)
{
    unsigned n = 0;
    while (v != 0u) {
        v >>= 1;
        n++;
    }
    return n;
}

int zpse_bs32_gate(const uint8_t *in, uint32_t n)
{
    bs32_geo g;
    uint32_t count;
    uint32_t ns, j;
    uint32_t min = 0, or_all = 0, and_all = 0xFFFFFFFFu, diff_or = 0;

    if (in == NULL) {
        return ZPSE_BS32_BAD_ARG;
    }
    if (bs32_geometry(n, &g) != ZPSE_BS32_OK) {
        return ZPSE_BS32_OVERFLOW;
    }
    /* Block floor and minimum-word floor from the extension spec. */
    if (n < 256u || g.count < 16u) {
        return ZPSE_BS32_GATE_REJECT;
    }
    count = g.count;

    /* Deterministic evenly-spaced sample: all words if fewer than 256,
     * otherwise 256 words spread across the whole block. */
    ns = (count < 256u) ? count : 256u;
    for (j = 0; j < ns; j++) {
        uint32_t idx = (ns == count) ? j
            : (uint32_t)(((uint64_t)j * (uint64_t)count) / (uint64_t)ns);
        uint32_t w = bs32_load32le(in + (size_t)4u * idx);
        if (j == 0u || w < min) {
            min = w;
        }
        or_all |= w;
        and_all &= w;
    }
    for (j = 0; j < ns; j++) {
        uint32_t idx = (ns == count) ? j
            : (uint32_t)(((uint64_t)j * (uint64_t)count) / (uint64_t)ns);
        uint32_t w = bs32_load32le(in + (size_t)4u * idx);
        diff_or |= (w - min); /* unsigned wrap: value - min_sampled */
    }

    if (bs32_bit_width(diff_or) <= 16u ||
        bs32_popcount32(or_all ^ and_all) <= 24u) {
        return ZPSE_BS32_OK; /* admit */
    }
    return ZPSE_BS32_GATE_REJECT;
}
