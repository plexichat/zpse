/* ZPSE transform extension -- ID 19, WORD_XOR_64, W=8, B=64.
 *
 * Scalar C is normative. SSE2 and AVX2 paths are runtime-dispatched and must
 * produce byte-identical output. All loads/stores are explicit little-endian
 * byte accesses or unaligned-safe SIMD load/store intrinsics: no unaligned
 * pointer casts, no FP arithmetic, no data-dependent branches in the kernels.
 */

#include "wordxor64.h"

#include <stddef.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define WX64_TARGET_SSE2 __attribute__((target("sse2")))
#define WX64_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define WX64_TARGET_SSE2
#define WX64_TARGET_AVX2
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define WX64_X86 1
#else
#define WX64_X86 0
#endif

#if WX64_X86
#include <immintrin.h>
#endif

#if defined(__clang__) || defined(__GNUC__)
#include <stdatomic.h>
#endif

/* ------------------------------------------------------------------ */
/* Argument / endian helpers                                            */
/* ------------------------------------------------------------------ */

static int wx64_args_ok(const uint8_t *src, uint32_t n,
                        const uint8_t *dst, uint32_t cap)
{
    if (src == NULL || dst == NULL) return 0;
    if (cap != 0u && cap < n) return 0;
    return 1;
}

/* Explicit little-endian: never cast a uint8_t* to a wider type. */
static uint64_t wx64_load64le(const uint8_t *p)
{
    return (uint64_t)p[0]        | ((uint64_t)p[1] << 8)  |
           ((uint64_t)p[2] << 16)| ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32)| ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48)| ((uint64_t)p[7] << 56);
}

static void wx64_store64le(uint8_t *p, uint64_t v)
{
    p[0] = (uint8_t)(v);       p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32); p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48); p[7] = (uint8_t)(v >> 56);
}

/* Scatter one XOR word into the eight planes at word index i. */
static void wx64_scatter_word(uint8_t *dst, uint32_t count,
                              uint32_t i, uint64_t y)
{
    uint32_t j;
    for (j = 0; j < 8u; ++j) {
        dst[j * count + i] = (uint8_t)(y >> (8u * j));
    }
}

/* ------------------------------------------------------------------ */
/* Scalar oracle                                                        */
/* ------------------------------------------------------------------ */

int zpse_wx64_forward_scalar(const uint8_t *src, uint32_t n,
                             uint8_t *dst, uint32_t cap)
{
    uint32_t count, tail, i;
    uint64_t prev;

    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    count = n >> 3;
    tail  = n & 7u;
    if (count == 0u) {
        if (tail != 0u) memmove(dst, src, tail);
        return WX64_OK;
    }

    prev = 0u; /* inline seed: y[0] = x[0] ^ 0 */
    for (i = 0; i < count; ++i) {
        uint64_t xi = wx64_load64le(src + ((size_t)i << 3));
        wx64_scatter_word(dst, count, i, xi ^ prev);
        prev = xi;
    }
    if (tail != 0u) {
        memmove(dst + (size_t)count * 8u, src + (size_t)count * 8u, tail);
    }
    return WX64_OK;
}

int zpse_wx64_inverse_scalar(const uint8_t *src, uint32_t n,
                             uint8_t *dst, uint32_t cap)
{
    uint32_t count, tail, i;
    uint64_t prev;

    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    count = n >> 3;
    tail  = n & 7u;
    if (count == 0u) {
        if (tail != 0u) memmove(dst, src, tail);
        return WX64_OK;
    }

    prev = 0u;
    for (i = 0; i < count; ++i) {
        uint64_t yi = 0u;
        uint64_t xi;
        uint32_t j;
        for (j = 0; j < 8u; ++j) {
            yi |= (uint64_t)src[(size_t)j * count + i] << (8u * j);
        }
        xi = yi ^ prev;
        wx64_store64le(dst + ((size_t)i << 3), xi);
        prev = xi;
    }
    if (tail != 0u) {
        memmove(dst + (size_t)count * 8u, src + (size_t)count * 8u, tail);
    }
    return WX64_OK;
}

/* ------------------------------------------------------------------ */
/* 8x8 byte transpose helpers (SSE2 unpack network, used by both SIMD
 * paths). Forward: 64 B interleaved -> 8 planes of 8 B each.
 * Inverse: 8 planes -> 64 B interleaved.                                */
/* ------------------------------------------------------------------ */

#if WX64_X86

static WX64_TARGET_SSE2 void wx64_fwd_transpose8(uint8_t *dst, uint32_t count,
                                                  uint32_t base,
                                                  const uint8_t tmp[64])
{
    __m128i r0 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp +  0));
    __m128i r1 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp +  8));
    __m128i r2 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp + 16));
    __m128i r3 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp + 24));
    __m128i r4 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp + 32));
    __m128i r5 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp + 40));
    __m128i r6 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp + 48));
    __m128i r7 = _mm_loadl_epi64((const __m128i *)(const void *)(tmp + 56));
    __m128i t0 = _mm_unpacklo_epi8(r0, r1);
    __m128i t1 = _mm_unpacklo_epi8(r2, r3);
    __m128i t2 = _mm_unpacklo_epi8(r4, r5);
    __m128i t3 = _mm_unpacklo_epi8(r6, r7);
    __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    __m128i u2 = _mm_unpacklo_epi16(t2, t3);
    __m128i u3 = _mm_unpackhi_epi16(t2, t3);
    __m128i p01 = _mm_unpacklo_epi32(u0, u2); /* plane 0 | plane 1 */
    __m128i p23 = _mm_unpackhi_epi32(u0, u2); /* plane 2 | plane 3 */
    __m128i p45 = _mm_unpacklo_epi32(u1, u3); /* plane 4 | plane 5 */
    __m128i p67 = _mm_unpackhi_epi32(u1, u3); /* plane 6 | plane 7 */

    _mm_storel_epi64((__m128i *)(void *)(dst + 0u * count + base), p01);
    _mm_storel_epi64((__m128i *)(void *)(dst + 1u * count + base),
                     _mm_srli_si128(p01, 8));
    _mm_storel_epi64((__m128i *)(void *)(dst + 2u * count + base), p23);
    _mm_storel_epi64((__m128i *)(void *)(dst + 3u * count + base),
                     _mm_srli_si128(p23, 8));
    _mm_storel_epi64((__m128i *)(void *)(dst + 4u * count + base), p45);
    _mm_storel_epi64((__m128i *)(void *)(dst + 5u * count + base),
                     _mm_srli_si128(p45, 8));
    _mm_storel_epi64((__m128i *)(void *)(dst + 6u * count + base), p67);
    _mm_storel_epi64((__m128i *)(void *)(dst + 7u * count + base),
                     _mm_srli_si128(p67, 8));
}

static WX64_TARGET_SSE2 void wx64_inv_transpose8(uint8_t tmp[64],
                                                  const uint8_t *const plane[8],
                                                  uint32_t pos)
{
    __m128i r0 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[0] + pos));
    __m128i r1 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[1] + pos));
    __m128i r2 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[2] + pos));
    __m128i r3 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[3] + pos));
    __m128i r4 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[4] + pos));
    __m128i r5 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[5] + pos));
    __m128i r6 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[6] + pos));
    __m128i r7 = _mm_loadl_epi64((const __m128i *)(const void *)(plane[7] + pos));
    __m128i t0 = _mm_unpacklo_epi8(r0, r1);
    __m128i t1 = _mm_unpacklo_epi8(r2, r3);
    __m128i t2 = _mm_unpacklo_epi8(r4, r5);
    __m128i t3 = _mm_unpacklo_epi8(r6, r7);
    __m128i u0 = _mm_unpacklo_epi16(t0, t1);
    __m128i u1 = _mm_unpackhi_epi16(t0, t1);
    __m128i u2 = _mm_unpacklo_epi16(t2, t3);
    __m128i u3 = _mm_unpackhi_epi16(t2, t3);
    __m128i v0 = _mm_unpacklo_epi32(u0, u2);
    __m128i v1 = _mm_unpacklo_epi32(u1, u3);
    __m128i v2 = _mm_unpackhi_epi32(u0, u2);
    __m128i v3 = _mm_unpackhi_epi32(u1, u3);

    _mm_storeu_si128((__m128i *)(void *)(tmp +  0), v0);
    _mm_storeu_si128((__m128i *)(void *)(tmp + 16), v2);
    _mm_storeu_si128((__m128i *)(void *)(tmp + 32), v1);
    _mm_storeu_si128((__m128i *)(void *)(tmp + 48), v3);
}

/* Scalar prefix-XOR used for the <8-word remainder of SIMD paths.
 * carry is x[i-1] as a 64-bit value (0 at the very start). */
static void wx64_scalar_xor_remainder(const uint8_t *src, uint8_t *dst,
                                      uint32_t count, uint32_t from,
                                      uint32_t to, uint64_t carry)
{
    uint32_t i;
    for (i = from; i < to; ++i) {
        uint64_t xi = wx64_load64le(src + ((size_t)i << 3));
        wx64_scatter_word(dst, count, i, xi ^ carry);
        carry = xi;
    }
}

static void wx64_scalar_inv_remainder(const uint8_t *src, uint8_t *dst,
                                      uint32_t count, uint32_t from,
                                      uint32_t to, uint64_t carry)
{
    uint32_t i;
    for (i = from; i < to; ++i) {
        uint64_t yi = 0u, xi;
        uint32_t j;
        for (j = 0; j < 8u; ++j) {
            yi |= (uint64_t)src[(size_t)j * count + i] << (8u * j);
        }
        xi = yi ^ carry;
        wx64_store64le(dst + ((size_t)i << 3), xi);
        carry = xi;
    }
}

/* ------------------------------------------------------------------ */
/* SSE2 path                                                            */
/* ------------------------------------------------------------------ */

WX64_TARGET_SSE2 int zpse_wx64_forward_sse2(const uint8_t *src, uint32_t n,
                                            uint8_t *dst, uint32_t cap)
{
    uint32_t count, tail, i;
    __m128i carry;

    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    count = n >> 3;
    tail  = n & 7u;
    if (count == 0u) {
        if (tail != 0u) memmove(dst, src, tail);
        return WX64_OK;
    }

    carry = _mm_setzero_si128(); /* low 64 = x[i-1], high 64 = 0 */
    i = 0u;
    for (; i + 8u <= count; i += 8u) {
        uint8_t tmp[64];
        __m128i pair_carry = carry;
        uint32_t q;
        for (q = 0; q < 8u; q += 2u) {
            const uint8_t *p = src + ((size_t)(i + q) << 3);
            __m128i lo = _mm_loadl_epi64((const __m128i *)(const void *)p);
            __m128i hi = _mm_loadl_epi64((const __m128i *)(const void *)(p + 8));
            __m128i cur = _mm_or_si128(lo, _mm_slli_si128(hi, 8));
            /* y[2q]   = x[2q]   ^ x[2q-1]
             * y[2q+1] = x[2q+1] ^ x[2q] */
            __m128i y = _mm_xor_si128(_mm_xor_si128(cur,
                                                   _mm_slli_si128(cur, 8)),
                                      pair_carry);
            _mm_storel_epi64((__m128i *)(void *)(tmp + (size_t)q * 8u), y);
            _mm_storel_epi64((__m128i *)(void *)(tmp + (size_t)q * 8u + 8u),
                             _mm_srli_si128(y, 8));
            pair_carry = _mm_srli_si128(cur, 8); /* [x[2q+1], 0] */
        }
        carry = pair_carry;
        wx64_fwd_transpose8(dst, count, i, tmp);
    }

    if (i < count) {
        uint64_t c = (uint64_t)_mm_cvtsi128_si64(carry);
        wx64_scalar_xor_remainder(src, dst, count, i, count, c);
    }
    if (tail != 0u) {
        memmove(dst + (size_t)count * 8u, src + (size_t)count * 8u, tail);
    }
    return WX64_OK;
}

WX64_TARGET_SSE2 int zpse_wx64_inverse_sse2(const uint8_t *src, uint32_t n,
                                            uint8_t *dst, uint32_t cap)
{
    uint32_t count, tail, i;
    __m128i carry;

    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    count = n >> 3;
    tail  = n & 7u;
    if (count == 0u) {
        if (tail != 0u) memmove(dst, src, tail);
        return WX64_OK;
    }

    carry = _mm_setzero_si128(); /* low 64 = x[i-1] */
    i = 0u;
    for (; i + 8u <= count; i += 8u) {
        uint8_t tmp[64];
        const uint8_t *plane[8];
        __m128i y[4];
        __m128i c = carry;
        __m128i x[4];
        uint32_t k;

        for (k = 0; k < 8u; ++k) plane[k] = src + (size_t)k * count;
        wx64_inv_transpose8(tmp, plane, i);
        for (k = 0; k < 4u; ++k) {
            y[k] = _mm_loadu_si128((const __m128i *)(const void *)(tmp + 16u * k));
        }

        /* SIMD prefix XOR of eight 64-bit words, with the incoming carry
         * injected into lane 0. */
        for (k = 0; k < 4u; ++k) {
            __m128i shifted = _mm_slli_si128(y[k], 8); /* [0, y[2k]] */
            x[k] = _mm_xor_si128(_mm_xor_si128(y[k], shifted), c);
            /* Broadcast the second output word to BOTH qwords: the high
             * lane needs x[2k+1] to turn y[2k+3]^y[2k+2] into y[2k+3]^x[2k+2]. */
            c = _mm_srli_si128(x[k], 8);
            c = _mm_unpacklo_epi64(c, c); /* [x[2k+1], x[2k+1]] */
        }
        for (k = 0; k < 4u; ++k) {
            _mm_storeu_si128((__m128i *)(void *)(dst + ((size_t)(i + 2u * k) << 3)),
                             x[k]);
        }
        carry = c;
    }

    if (i < count) {
        uint64_t c = (uint64_t)_mm_cvtsi128_si64(carry);
        wx64_scalar_inv_remainder(src, dst, count, i, count, c);
    }
    if (tail != 0u) {
        memmove(dst + (size_t)count * 8u, src + (size_t)count * 8u, tail);
    }
    return WX64_OK;
}

/* ------------------------------------------------------------------ */
/* AVX2 path                                                            */
/* ------------------------------------------------------------------ */

WX64_TARGET_AVX2 int zpse_wx64_forward_avx2(const uint8_t *src, uint32_t n,
                                            uint8_t *dst, uint32_t cap)
{
    uint32_t count, tail, i;
    __m256i carry;

    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    count = n >> 3;
    tail  = n & 7u;
    if (count == 0u) {
        if (tail != 0u) memmove(dst, src, tail);
        return WX64_OK;
    }

    carry = _mm256_setzero_si256();
    i = 0u;
    for (; i + 8u <= count; i += 8u) {
        uint8_t tmp[64];
        const size_t off = (size_t)i << 3;
        __m256i xlo = _mm256_loadu_si256((const __m256i *)(const void *)(src + off));
        __m256i xhi = _mm256_loadu_si256((const __m256i *)(const void *)(src + off + 32));
        /* Predecessor vector for the eight x words:
         *  lanes 0..3 = [x_prev, x0, x1, x2]
         *  lanes 4..7 = [x3, x4, x5, x6]
         * where x_prev is the last x word of the previous 8-word block
         * (zero for the first block). Each lane group is built by broadcasting
         * the seed word and blending in lanes 0..2 of the corresponding
         * 4-word group, then shifted right one qword. */
        __m256i xprev_b = _mm256_permute4x64_epi64(carry, _MM_SHUFFLE(3, 3, 3, 3));
        __m256i xlo_s   = _mm256_permute4x64_epi64(xlo,  _MM_SHUFFLE(2, 1, 0, 0));
        __m256i xhi_s   = _mm256_permute4x64_epi64(xhi,  _MM_SHUFFLE(2, 1, 0, 0));
        __m256i x3_b    = _mm256_permute4x64_epi64(xlo,  _MM_SHUFFLE(3, 3, 3, 3));
        __m256i prevlo  = _mm256_blend_epi32(xprev_b, xlo_s, 0xFC);
        __m256i prevhi  = _mm256_blend_epi32(x3_b, xhi_s, 0xFC);
        __m256i ylo = _mm256_xor_si256(xlo, prevlo);
        __m256i yhi = _mm256_xor_si256(xhi, prevhi);

        _mm256_storeu_si256((__m256i *)(void *)(tmp +  0), ylo);
        _mm256_storeu_si256((__m256i *)(void *)(tmp + 32), yhi);
        wx64_fwd_transpose8(dst, count, i, tmp);
        carry = xhi; /* last four x words of this block */
    }

    if (i < count) {
        uint64_t c = (uint64_t)_mm256_extract_epi64(carry, 3);
        wx64_scalar_xor_remainder(src, dst, count, i, count, c);
    }
    if (tail != 0u) {
        memmove(dst + (size_t)count * 8u, src + (size_t)count * 8u, tail);
    }
    return WX64_OK;
}

WX64_TARGET_AVX2 int zpse_wx64_inverse_avx2(const uint8_t *src, uint32_t n,
                                            uint8_t *dst, uint32_t cap)
{
    uint32_t count, tail, i;
    __m256i carry;

    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    count = n >> 3;
    tail  = n & 7u;
    if (count == 0u) {
        if (tail != 0u) memmove(dst, src, tail);
        return WX64_OK;
    }

    carry = _mm256_setzero_si256();
    i = 0u;
    for (; i + 8u <= count; i += 8u) {
        uint8_t tmp[64];
        const uint8_t *plane[8];
        __m256i y[2], x[2];
        __m256i cvec;
        uint32_t k;

        for (k = 0; k < 8u; ++k) plane[k] = src + (size_t)k * count;
        wx64_inv_transpose8(tmp, plane, i);
        y[0] = _mm256_loadu_si256((const __m256i *)(const void *)(tmp +  0));
        y[1] = _mm256_loadu_si256((const __m256i *)(const void *)(tmp + 32));

        for (k = 0; k < 2u; ++k) {
            /* Hillis-Steele prefix over four qwords, then fold in the
             * incoming carry (x of the previous word) in ALL lanes -- every
             * output x[i] = xprev ^ y[0] ^ .. ^ y[i] depends on it. */
            __m256i sh = _mm256_permute4x64_epi64(y[k], _MM_SHUFFLE(2, 1, 0, 0));
            __m256i rot, z, sh2;
            z = _mm256_setzero_si256();
            sh = _mm256_blend_epi32(sh, z, 0x03); /* [0, y0, y1, y2] */
            x[k] = _mm256_xor_si256(y[k], sh); /* [y0, y1^y0, y2^y1, y3^y2] */
            rot = _mm256_permute4x64_epi64(x[k], _MM_SHUFFLE(1, 0, 3, 2));
            sh2 = _mm256_blend_epi32(z, rot, 0xF0); /* [0, 0, p0, p1] */
            x[k] = _mm256_xor_si256(x[k], sh2);     /* prefix of y */
            cvec = _mm256_set1_epi64x((long long)_mm256_extract_epi64(carry, 3));
            x[k] = _mm256_xor_si256(x[k], cvec);
            /* carry always holds the most recently produced four x words;
             * lane 3 is the running x[i-1] for the next group. */
            carry = x[k];
        }
        _mm256_storeu_si256((__m256i *)(void *)(dst + ((size_t)i << 3)), x[0]);
        _mm256_storeu_si256((__m256i *)(void *)(dst + ((size_t)i << 3) + 32), x[1]);
    }

    if (i < count) {
        uint64_t c = (uint64_t)_mm256_extract_epi64(carry, 3);
        wx64_scalar_inv_remainder(src, dst, count, i, count, c);
    }
    if (tail != 0u) {
        memmove(dst + (size_t)count * 8u, src + (size_t)count * 8u, tail);
    }
    return WX64_OK;
}

#else /* !WX64_X86 : non-x86 builds fall back to the scalar oracle */

int zpse_wx64_forward_sse2(const uint8_t *s, uint32_t n, uint8_t *d, uint32_t c)
{ return zpse_wx64_forward_scalar(s, n, d, c); }
int zpse_wx64_forward_avx2(const uint8_t *s, uint32_t n, uint8_t *d, uint32_t c)
{ return zpse_wx64_forward_scalar(s, n, d, c); }
int zpse_wx64_inverse_sse2(const uint8_t *s, uint32_t n, uint8_t *d, uint32_t c)
{ return zpse_wx64_inverse_scalar(s, n, d, c); }
int zpse_wx64_inverse_avx2(const uint8_t *s, uint32_t n, uint8_t *d, uint32_t c)
{ return zpse_wx64_inverse_scalar(s, n, d, c); }

#endif /* WX64_X86 */

/* ------------------------------------------------------------------ */
/* Runtime dispatch                                                     */
/* ------------------------------------------------------------------ */

enum { WX64_DISPATCH_AUTO = -1, WX64_DISPATCH_SCALAR = 0,
       WX64_DISPATCH_SSE2 = 1, WX64_DISPATCH_AVX2 = 2 };

static int wx64_dispatch(void)
{
#if defined(__clang__) || defined(__GNUC__)
    static _Atomic int cached = WX64_DISPATCH_AUTO;
    int d = atomic_load_explicit(&cached, memory_order_relaxed);
    if (d != WX64_DISPATCH_AUTO) return d;
#if WX64_X86
    if (__builtin_cpu_supports("avx2"))      d = WX64_DISPATCH_AVX2;
    else if (__builtin_cpu_supports("sse2")) d = WX64_DISPATCH_SSE2;
    else                                     d = WX64_DISPATCH_SCALAR;
#else
    d = WX64_DISPATCH_SCALAR;
#endif
    atomic_store_explicit(&cached, d, memory_order_relaxed);
    return d;
#else
    return WX64_DISPATCH_SCALAR;
#endif
}

int zpse_wx64_forward(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap)
{
    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    switch (wx64_dispatch()) {
    case WX64_DISPATCH_AVX2: return zpse_wx64_forward_avx2(src, n, dst, cap);
    case WX64_DISPATCH_SSE2: return zpse_wx64_forward_sse2(src, n, dst, cap);
    default:                 return zpse_wx64_forward_scalar(src, n, dst, cap);
    }
}

int zpse_wx64_inverse(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap)
{
    if (!wx64_args_ok(src, n, dst, cap)) return WX64_ERR_ARG;
    switch (wx64_dispatch()) {
    case WX64_DISPATCH_AVX2: return zpse_wx64_inverse_avx2(src, n, dst, cap);
    case WX64_DISPATCH_SSE2: return zpse_wx64_inverse_sse2(src, n, dst, cap);
    default:                 return zpse_wx64_inverse_scalar(src, n, dst, cap);
    }
}

/* ------------------------------------------------------------------ */
/* Gate                                                                 */
/* ------------------------------------------------------------------ */

static unsigned wx64_popcount64(uint64_t v)
{
    static const uint8_t bits[16] = {
        0u, 1u, 1u, 2u, 1u, 2u, 2u, 3u,
        1u, 2u, 2u, 3u, 2u, 3u, 3u, 4u
    };
    unsigned c = 0u;
    while (v != 0u) {
        c += bits[v & 0x0Fu];
        v >>= 4;
    }
    return c;
}

/* Compute all three signals in one pass over the XOR stream. */
static int wx64_gate_stats(const uint8_t *src, uint32_t n,
                           uint64_t *zero_xor_out, uint64_t *zero_bytes_out,
                           uint64_t *popcount_out)
{
    uint32_t count, i;
    uint64_t prev, zero_xor, zero_bytes, popcount;

    if (src == NULL) return 0;
    count = n >> 3;
    if (count < 8u) return 0; /* minimum 8 words */

    prev = 0u;
    zero_xor = 0u;
    zero_bytes = 0u;
    popcount = 0u;
    for (i = 0; i < count; ++i) {
        uint64_t xi = wx64_load64le(src + ((size_t)i << 3));
        uint64_t yi = xi ^ prev;
        uint32_t j;
        unsigned z = 0u;
        for (j = 0; j < 8u; ++j) {
            uint8_t b = (uint8_t)(yi >> (8u * j));
            if (b == 0u) ++z;
        }
        if (yi == 0u) ++zero_xor;
        zero_bytes += z;
        popcount += wx64_popcount64(yi);
        prev = xi;
    }
    if (zero_xor_out)   *zero_xor_out   = zero_xor;
    if (zero_bytes_out) *zero_bytes_out = zero_bytes;
    if (popcount_out)   *popcount_out   = popcount;
    return 1;
}

int zpse_wx64_gate(const uint8_t *src, uint32_t n)
{
    uint64_t zx, zb, pc;
    uint64_t count = (uint64_t)(n >> 3);

    if (!wx64_gate_stats(src, n, &zx, &zb, &pc)) return 0;

    /* zero_xor_count >= count/8 */
    if (zx >= count / 8u) return 1;
    /* avg zero bytes per XOR word >= W/2 == 4 */
    if (zb >= 4u * count) return 1;
    /* avg popcount <= B/4 == 16 */
    if (pc <= 16u * count) return 1;
    return 0;
}

int zpse_wx64_gate_strong(const uint8_t *src, uint32_t n)
{
    uint64_t zx, zb, pc;
    uint64_t count, nonzero_bytes;

    if (!wx64_gate_stats(src, n, &zx, &zb, &pc)) return 0;
    (void)zx;
    (void)pc;
    count = (uint64_t)(n >> 3);
    nonzero_bytes = 8u * count - zb;
    /* nonzero XOR bytes <= 60% (integer form: nz*5 <= 24*count) */
    return (nonzero_bytes * 5u <= 24u * count) ? 1 : 0;
}
