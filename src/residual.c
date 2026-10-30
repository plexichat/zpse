#include "zpse/residual.h"
#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/status.h"
#include "zpse/vector.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The mode-2 palette unpack is the densest residual path the decoder has:
 * a 256-position tile of low-entropy noise carries ~235 exceptions, all of
 * them palette indices, and unpacking them one shift-and-mask at a time
 * measured ~0.75 ns per value - the largest single pool left in the decode
 * benchmark. Sixteen values at a time come out of the bit-packed payload
 * with byte shuffles instead, which needs pshufb.
 *
 * Same per-function-target structure as the SIMD paths in tile.c: gcc and
 * clang enable SSSE3 for these routines alone, MSVC only when the whole
 * translation unit is built with an /arch level that defines __SSSE3__,
 * otherwise the SIMD form is compiled out and the scalar loop is used.
 * Execution is additionally gated on zpse_cpu_caps(), so the scalar path
 * stays reachable and tests/test_parity.c can require the two to agree. */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
/* Also pins 64-byte alignment: see the note on ZPSE_TILE_AVX2 in tile.c.
 * These are the loops that carry dense-tile decode and residual packing, and
 * their cost otherwise shifts with the placement of unrelated code. */
#define ZPSE_RES_SSSE3 __attribute__((target("ssse3"), aligned(64)))
#define ZPSE_RES_HAVE_SSSE3 1
#elif defined(__SSSE3__)
#define ZPSE_RES_SSSE3
#define ZPSE_RES_HAVE_SSSE3 1
#else
#define ZPSE_RES_SSSE3
#define ZPSE_RES_HAVE_SSSE3 0
#endif
#else
#define ZPSE_RES_HAVE_SSSE3 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_RES_AVX2 __attribute__((target("avx2")))
#define ZPSE_RES_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_RES_AVX2
#define ZPSE_RES_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_RES_AVX2
#define ZPSE_RES_HAVE_AVX2 0
#endif
#else
#define ZPSE_RES_HAVE_AVX2 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_RES_BMI2 __attribute__((target("bmi2"), aligned(64)))
#define ZPSE_RES_HAVE_BMI2 1
#elif defined(__BMI2__)
#define ZPSE_RES_BMI2
#define ZPSE_RES_HAVE_BMI2 1
#else
#define ZPSE_RES_BMI2
#define ZPSE_RES_HAVE_BMI2 0
#endif
#else
#define ZPSE_RES_HAVE_BMI2 0
#endif

#if ZPSE_RES_HAVE_SSSE3 || ZPSE_RES_HAVE_AVX2 || ZPSE_RES_HAVE_BMI2
#include <immintrin.h>
#endif

uint8_t zpse_base_m_group(uint8_t m)
{
    switch (m) {
    case 3: return 20;
    case 5: return 13;
    case 6: return 11;
    case 7: return 10;
    case 9: return 9;
    case 10: return 9;
    case 11: return 8;
    case 12: return 8;
    case 13: return 8;
    case 14: return 7;
    case 15: return 7;
    default: return 0;
    }
}

/* Smallest allowed mode-3 M covering residuals with max value maxv
 * (M >= maxv+1, M in {3,5,6,7,9,10,11,12,13,14,15}). Returns 0 when no
 * allowed M covers maxv (maxv+1 > 15). */
uint8_t zpse_residual_mode3_pick_m(uint8_t maxv)
{
    static const uint8_t allowed[] = {
        3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15
    };
    unsigned need = (unsigned)maxv + 1u;
    size_t i;

    for (i = 0; i < sizeof(allowed) / sizeof(allowed[0]); i++) {
        if ((unsigned)allowed[i] >= need) {
            return allowed[i];
        }
    }
    return 0;
}

static uint8_t residual_mode3_select_m(uint8_t maxv)
{
    return zpse_residual_mode3_pick_m(maxv);
}

/* Read w (<= 8) bits at bitpos, LSB-first.
 *
 * Every caller reads a packed run of fixed-width fields from a payload
 * whose length was already validated against exactly those fields, so a
 * field never crosses more than one byte boundary and the upper byte it
 * reaches is always inside the payload. That turns the general bit
 * walk into two shifts and an AND, which is the difference between a
 * per-unit branch loop and straight-line code in the dense-residual
 * decode paths (modes 2 and 4). */
static unsigned residual_get_bits(const uint8_t *s, uint32_t bitpos,
                                  unsigned w)
{
    uint32_t byte = bitpos >> 3u;
    unsigned off = (unsigned)(bitpos & 7u);
    unsigned v = (unsigned)s[byte] >> off;

    if (off + w > 8u) {
        v |= (unsigned)s[byte + 1u] << (8u - off);
    }
    return v & ((1u << w) - 1u);
}

/* OR w (<= 8) bits of val at bitpos, LSB-first. Target must be zeroed. */
static void residual_put_bits(uint8_t *d, uint32_t bitpos, unsigned w,
                              unsigned val)
{
    unsigned sh = 0;

    while (w > 0) {
        uint32_t byte = bitpos >> 3u;
        unsigned off = (unsigned)(bitpos & 7u);
        unsigned take = 8u - off;
        uint8_t chunk;

        if (take > w) {
            take = w;
        }
        chunk = (uint8_t)(((val >> sh) & ((1u << take) - 1u)) << off);
        d[byte] = (uint8_t)(d[byte] | chunk);
        bitpos += take;
        sh += take;
        w -= take;
    }
}

/* Signed wrap delta x - pred in [-128,127]. */
static int residual_delta(uint8_t x, uint8_t pred)
{
    return (int)(((unsigned)x - (unsigned)pred + 128u) & 0xFFu) - 128;
}

/* True when any of the first n bytes is exactly 0xFF.
 *
 * The per-byte rejection loop below (mode 0) is a branch inside the
 * loop body, which is what stops the compiler from vectorising or
 * unrolling it; the same question can be asked of eight bytes at once.
 * `~w` turns a 0xFF byte into a zero byte, and the classic "has zero
 * byte" idiom then reports it: (x - 0x01..01) & ~x & 0x80..80 is
 * non-zero exactly when x has a zero byte. Byte order is irrelevant here
 * because only byte equality is examined. */
static int residual_has_ff(const uint8_t *p, uint32_t n)
{
    uint32_t i = 0;

    while (i + 8u <= n) {
        uint64_t w;

        memcpy(&w, p + i, sizeof w);
        w = ~w;
        if (((w - UINT64_C(0x0101010101010101)) & ~w &
             UINT64_C(0x8080808080808080)) != 0) {
            return 1;
        }
        i += 8u;
    }
    for (; i < n; i++) {
        if (p[i] == 255u) {
            return 1;
        }
    }
    return 0;
}

/* Forward byte copy, a word at a time. Produces exactly the bytes the
 * element-wise loop produced, for every caller that does not overlap the
 * payload (the payload is the tile record inside the compressed stream
 * and the destination is the tile's scratch array). */
/* A tile's payload copy, k bytes with k <= 256, on every mode-0 and mode-1
 * tile the decoder rebuilds -- 5% of decode time on the corpus, and on a
 * dense tile the k is large enough that the step width is what the copy
 * costs. Thirty-two bytes a step (two 16-byte moves, which is what the
 * compiler makes of a constant-size memcpy) halves the loop and its two
 * counter updates per byte copied, and the 8-byte loop still covers the
 * last partial step. */
static void residual_copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n)
{
    uint32_t i = 0;

    while (i + 32u <= n) {
        memcpy(dst + i, src + i, 32);
        i += 32u;
    }
    while (i + 8u <= n) {
        uint64_t w;

        memcpy(&w, src + i, sizeof w);
        memcpy(dst + i, &w, sizeof w);
        i += 8u;
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

#if ZPSE_RES_HAVE_SSSE3

/* Sixteen palette indices, one per byte lane, from the low 64 bits of the
 * bit-packed stream `w` (already shifted down to the first value's bit 0).
 * Every field is b <= 4 bits wide and never crosses a 64-bit lane, so the
 * three wide cases are byte shuffles:
 *
 *   b == 4   each payload byte holds values 2k and 2k+1; the low and high
 *            nibbles interleave straight into value order.
 *   b == 2   each payload byte holds values 4k..4k+3 in bit pairs; the four
 *            masked copies interleave back into value order.
 *   b == 1   each payload byte holds eight values, one per bit, so the byte
 *            is broadcast against the per-lane bit values and compared.
 *
 * Only the low eight lanes of the two b == 1 compares are consumed by the
 * unpacklo_epi64 that joins them, so the all-ones the zeroed upper lanes of
 * bitv would produce past lane 7 never reaches an output byte. */
static ZPSE_RES_SSSE3 __m128i residual_pal_index(uint64_t w, unsigned b)
{
    const __m128i one = _mm_set1_epi8((char)1);
    __m128i v = _mm_cvtsi64_si128((long long)w);

    if (b == 4u) {
        __m128i nib = _mm_set1_epi8((char)0x0F);
        __m128i lo = _mm_and_si128(v, nib);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), nib);

        return _mm_unpacklo_epi8(lo, hi);
    }
    if (b == 2u) {
        __m128i two = _mm_set1_epi8((char)0x03);
        __m128i a = _mm_and_si128(v, two);
        __m128i c = _mm_and_si128(_mm_srli_epi16(v, 2), two);
        __m128i d = _mm_and_si128(_mm_srli_epi16(v, 4), two);
        __m128i e = _mm_and_si128(_mm_srli_epi16(v, 6), two);

        return _mm_unpacklo_epi16(_mm_unpacklo_epi8(a, c),
                                  _mm_unpacklo_epi8(d, e));
    }
    {
        const __m128i bitv = _mm_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char)0x80,
            0, 0, 0, 0, 0, 0, 0, 0);
        __m128i lo = _mm_cmpeq_epi8(
            _mm_and_si128(_mm_set1_epi8((char)(uint8_t)w), bitv), bitv);
        __m128i hi = _mm_cmpeq_epi8(
            _mm_and_si128(_mm_set1_epi8((char)(uint8_t)(w >> 8)), bitv),
            bitv);

        return _mm_unpacklo_epi64(_mm_and_si128(lo, one),
                                  _mm_and_si128(hi, one));
    }
}

/* Unpack whole blocks of sixteen mode-2 palette indices with SSSE3,
 * returning how many values were written. It stops at the first block it
 * cannot handle - a width other than 1, 2 or 4, or an index that is not
 * below m - and the scalar loop in the caller then re-decides from there,
 * so a rejected payload still produces the scalar PSE_BAD_RESIDUAL at the
 * scalar position and every accepted value is byte-identical to what that
 * loop would have written. Reads stop at the block's last needed byte, so
 * a partial final block never touches the byte after the packed region.
 * The palette is copied into a zero-padded 16-byte register rather than
 * loaded straight from the payload, because a payload with m < 16 has no
 * readable bytes past the palette. */
#if ZPSE_RES_HAVE_AVX2
static ZPSE_RES_AVX2 uint16_t residual_decode_mode2_avx2_m16(
    const uint8_t *pal, const uint8_t *src, uint16_t k, uint8_t *out);
#endif

#if ZPSE_RES_HAVE_BMI2
#define ZPSE_RES_BMI2_SSSE3 \
    __attribute__((target("ssse3,bmi2"), aligned(64)))

/* Asked once per process, like the tile path's own probe: the answer cannot
 * change while the program runs. MSVC has no builtin for it and keeps the
 * shift-and-mask form, which is byte-identical. */
static int residual_bmi2(void)
{
    static int cached = -1;

    if (cached < 0) {
#if defined(__GNUC__) || defined(__clang__)
        cached = __builtin_cpu_supports("bmi2") ? 1 : 0;
#else
        cached = 0;
#endif
    }
    return cached;
}

/* Sixteen three-bit indices per block. The block is 48 bits -- exactly six
 * payload bytes -- so, starting at value 0, every block starts on a byte
 * boundary and the phase is always zero: no shift of the stream is needed.
 * Each pdep spreads the low 24 bits of its word into the low three bits of
 * eight bytes, which is the index vector pshufb wants, so a block costs one
 * 6-byte load (a u32 and a u16, both inside the payload because
 * i + 16 <= k), two pdeps, one compare and one shuffling store. The caller's
 * shift-and-mask block stays for machines without BMI2 and produces the
 * same bytes. */
static ZPSE_RES_BMI2_SSSE3 uint16_t residual_mode2_b3_bmi2(
    const uint8_t *src, uint16_t k, unsigned m, __m128i palv, __m128i limv,
    uint8_t *exc_out)
{
    uint16_t i = 0u;
    uint32_t off = 0u;

    while ((uint32_t)i + 16u <= (uint32_t)k) {
        uint64_t w = (uint64_t)zpse_load_u32le(src + off) |
                     ((uint64_t)zpse_load_u16le(src + off + 4u) << 32);
        uint64_t lo = _pdep_u64(w, UINT64_C(0x0707070707070707));
        uint64_t hi = _pdep_u64(w >> 24, UINT64_C(0x0707070707070707));
        __m128i idx = _mm_set_epi64x((long long)hi, (long long)lo);

        /* m == 8 fills the width: no 3-bit index can name a palette entry
         * that is not there. */
        if (m != 8u && _mm_movemask_epi8(_mm_cmpgt_epi8(idx, limv)) != 0) {
            break;
        }
        _mm_storeu_si128((__m128i *)(void *)(exc_out + i),
                         _mm_shuffle_epi8(palv, idx));
        i = (uint16_t)(i + 16u);
        off += 6u;
    }
    return i;
}
#endif

static ZPSE_RES_SSSE3 uint16_t residual_decode_mode2_simd(
    const uint8_t *palette, const uint8_t *src, uint16_t k, unsigned b,
    unsigned m, uint8_t *exc_out)
{
    uint16_t i = 0u;
    uint8_t pal[16];
    __m128i palv;
    __m128i limv;

    /* m is in [2,16], so b is in [1,4] and every width has a path below. */
    if (b < 1u || b > 4u) {
        return 0u;
    }
    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_SSSE3) == 0u) {
        return 0u;
    }
#if ZPSE_RES_HAVE_AVX2
    /* The dense palette width first, on a wider unit when one is there:
     * 64 values per step instead of 32. It cannot change a byte -- only
     * the order in which independent values are produced -- and the
     * scalar tail below finishes whatever it leaves, exactly as it does
     * for the 128-bit loop. */
    if (b == 4u && m == 16u && k >= 64u &&
        (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        i = residual_decode_mode2_avx2_m16(palette, src, k, exc_out);
    }
#endif
    memcpy(pal, palette, (size_t)m);
    memset(pal + m, 0, sizeof pal - (size_t)m);
    palv = _mm_loadu_si128((const __m128i *)(const void *)pal);
    limv = _mm_set1_epi8((char)(uint8_t)(m - 1u));

    /* A 3-bit field does not tile a byte, so this width takes eight values
     * (24 bits) per block, extracted with plain shifts into eight index
     * bytes and then looked up with the same shuffle. The 24 bits plus the
     * phase always fit in the first 8-byte read, so there is no second
     * load here. */
    if (b == 3u) {
#if ZPSE_RES_HAVE_BMI2
        if (residual_bmi2()) {
            return residual_mode2_b3_bmi2(src, k, m, palv, limv, exc_out);
        }
#endif
        while ((uint32_t)i + 8u <= (uint32_t)k) {
            uint32_t bit = (uint32_t)i * 3u;
            uint32_t off = bit >> 3u;
            unsigned ph = (unsigned)(bit & 7u);
            uint32_t take = ((bit + 24u + 7u) >> 3u) - off;
            uint64_t w = 0u;
            uint8_t t8[8];
            __m128i idx;

            if (take > 8u) {
                take = 8u;
            }
            memcpy(&w, src + off, (size_t)take);
            w = (w >> ph) & UINT64_C(0xFFFFFF);
            t8[0] = (uint8_t)(w & 7u);
            t8[1] = (uint8_t)((w >> 3) & 7u);
            t8[2] = (uint8_t)((w >> 6) & 7u);
            t8[3] = (uint8_t)((w >> 9) & 7u);
            t8[4] = (uint8_t)((w >> 12) & 7u);
            t8[5] = (uint8_t)((w >> 15) & 7u);
            t8[6] = (uint8_t)((w >> 18) & 7u);
            t8[7] = (uint8_t)((w >> 21) & 7u);
            idx = _mm_loadl_epi64((const __m128i *)(const void *)t8);
            if (_mm_movemask_epi8(_mm_cmpgt_epi8(idx, limv)) != 0) {
                break;
            }
            _mm_storel_epi64((__m128i *)(void *)(exc_out + i),
                             _mm_shuffle_epi8(palv, idx));
            i = (uint16_t)(i + 8u);
        }
        return i;
    }

    /* One bit per value. The width means m == 2 (ceil_log2(m) == 1 has no
     * other solution), so every index is 0 or 1 and there is nothing to
     * validate: the value is pal[0] or pal[1] selected by one bit. Sixteen
     * payload bytes hold 128 values, bit j of byte i being value 8i + j,
     * so the eight bit planes have to be interleaved back into byte order
     * -- the 2-, 4- and 8-way unpack ladder below. That is 64 operations
     * for 128 values where the generic loop spends ~2.3 cycles per value
     * (measured: 1-bit palettes were 13x slower to expand than 4-bit ones).
     * Values are consumed in blocks of 128, always from value 0, so what
     * this consumes is a prefix and the loop below finishes the rest. */
    if (b == 1u) {
        const __m128i p0v = _mm_set1_epi8((char)pal[0]);
        const __m128i xv = _mm_set1_epi8((char)(pal[0] ^ pal[1]));
        const __m128i b0 = _mm_set1_epi8(1);
        const __m128i b1 = _mm_set1_epi8(2);
        const __m128i b2 = _mm_set1_epi8(4);
        const __m128i b3 = _mm_set1_epi8(8);
        const __m128i b4 = _mm_set1_epi8(16);
        const __m128i b5 = _mm_set1_epi8(32);
        const __m128i b6 = _mm_set1_epi8(64);
        const __m128i b7 = _mm_set1_epi8((char)0x80);

        while ((uint32_t)i + 128u <= (uint32_t)k) {
            __m128i x = _mm_loadu_si128(
                (const __m128i *)(const void *)(src + ((uint32_t)i >> 3u)));
            __m128i v0, v1, v2, v3, v4, v5, v6, v7;
            __m128i u0, u1, u2, u3, u4, u5, u6, u7;
            __m128i w0, w1, w2, w3, w4, w5, w6, w7;
            uint8_t *o = exc_out + i;

            /* One bit plane per vector: 0xFF where the bit is set, then
             * pal[0] ^ (mask & (pal[0] ^ pal[1])). */
            v0 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b0), b0), xv));
            v1 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b1), b1), xv));
            v2 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b2), b2), xv));
            v3 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b3), b3), xv));
            v4 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b4), b4), xv));
            v5 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b5), b5), xv));
            v6 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b6), b6), xv));
            v7 = _mm_xor_si128(p0v, _mm_and_si128(_mm_cmpeq_epi8(
                     _mm_and_si128(x, b7), b7), xv));

            /* Interleave bytes (2-way), then 16-bit units (4-way), then
             * 32-bit units (8-way): each level's lo/hi halves split the
             * input bytes in half, so the final eight vectors are input
             * bytes 0-1, 2-3, 4-5, ... in ascending value order. */
            u0 = _mm_unpacklo_epi8(v0, v1);
            u1 = _mm_unpackhi_epi8(v0, v1);
            u2 = _mm_unpacklo_epi8(v2, v3);
            u3 = _mm_unpackhi_epi8(v2, v3);
            u4 = _mm_unpacklo_epi8(v4, v5);
            u5 = _mm_unpackhi_epi8(v4, v5);
            u6 = _mm_unpacklo_epi8(v6, v7);
            u7 = _mm_unpackhi_epi8(v6, v7);
            w0 = _mm_unpacklo_epi16(u0, u2);
            w1 = _mm_unpackhi_epi16(u0, u2);
            w2 = _mm_unpacklo_epi16(u1, u3);
            w3 = _mm_unpackhi_epi16(u1, u3);
            w4 = _mm_unpacklo_epi16(u4, u6);
            w5 = _mm_unpackhi_epi16(u4, u6);
            w6 = _mm_unpacklo_epi16(u5, u7);
            w7 = _mm_unpackhi_epi16(u5, u7);
            _mm_storeu_si128((__m128i *)(void *)(o),
                             _mm_unpacklo_epi32(w0, w4));
            _mm_storeu_si128((__m128i *)(void *)(o + 16u),
                             _mm_unpackhi_epi32(w0, w4));
            _mm_storeu_si128((__m128i *)(void *)(o + 32u),
                             _mm_unpacklo_epi32(w1, w5));
            _mm_storeu_si128((__m128i *)(void *)(o + 48u),
                             _mm_unpackhi_epi32(w1, w5));
            _mm_storeu_si128((__m128i *)(void *)(o + 64u),
                             _mm_unpacklo_epi32(w2, w6));
            _mm_storeu_si128((__m128i *)(void *)(o + 80u),
                             _mm_unpackhi_epi32(w2, w6));
            _mm_storeu_si128((__m128i *)(void *)(o + 96u),
                             _mm_unpacklo_epi32(w3, w7));
            _mm_storeu_si128((__m128i *)(void *)(o + 112u),
                             _mm_unpackhi_epi32(w3, w7));
            i = (uint16_t)(i + 128u);
        }
        return i;
    }

    /* Two-bit indices tile a byte four ways: three or four distinct values,
     * which is what a near-binary plane or a two-tone mask residual comes
     * out as. One 16-byte load carries 64 values, extracted as four index
     * vectors (values 4i + 0..3) and interleaved back into value order by
     * an 8- and then a 16-bit unpack. Every index is validated before the
     * first store, so a rejected payload leaves nothing written here and
     * the loop below re-decides from value i, exactly as it would have. */
    if (b == 2u) {
        const __m128i low2 = _mm_set1_epi8(0x03);

        /* m == 4 fills the width, so the four range checks below are
         * provably dead for that plane (see the b == 4 note). */
        if (m == 4u) {
            while ((uint32_t)i + 64u <= (uint32_t)k) {
                __m128i x = _mm_loadu_si128(
                    (const __m128i *)(const void *)(src +
                                                    ((uint32_t)i >> 2u)));
                __m128i i0 = _mm_and_si128(x, low2);
                __m128i i1 = _mm_and_si128(_mm_srli_epi16(x, 2), low2);
                __m128i i2 = _mm_and_si128(_mm_srli_epi16(x, 4), low2);
                __m128i i3 = _mm_and_si128(_mm_srli_epi16(x, 6), low2);
                __m128i a0 = _mm_shuffle_epi8(palv, i0);
                __m128i a1 = _mm_shuffle_epi8(palv, i1);
                __m128i a2 = _mm_shuffle_epi8(palv, i2);
                __m128i a3 = _mm_shuffle_epi8(palv, i3);
                __m128i u0 = _mm_unpacklo_epi8(a0, a1);
                __m128i u1 = _mm_unpackhi_epi8(a0, a1);
                __m128i u2 = _mm_unpacklo_epi8(a2, a3);
                __m128i u3 = _mm_unpackhi_epi8(a2, a3);

                _mm_storeu_si128((__m128i *)(void *)(exc_out + i),
                                 _mm_unpacklo_epi16(u0, u2));
                _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 16u),
                                 _mm_unpackhi_epi16(u0, u2));
                _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 32u),
                                 _mm_unpacklo_epi16(u1, u3));
                _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 48u),
                                 _mm_unpackhi_epi16(u1, u3));
                i = (uint16_t)(i + 64u);
            }
            return i;
        }

        while ((uint32_t)i + 64u <= (uint32_t)k) {
            __m128i x = _mm_loadu_si128(
                (const __m128i *)(const void *)(src + ((uint32_t)i >> 2u)));
            __m128i i0 = _mm_and_si128(x, low2);
            __m128i i1 = _mm_and_si128(_mm_srli_epi16(x, 2), low2);
            __m128i i2 = _mm_and_si128(_mm_srli_epi16(x, 4), low2);
            __m128i i3 = _mm_and_si128(_mm_srli_epi16(x, 6), low2);
            __m128i a0, a1, a2, a3, u0, u1, u2, u3;

            if (_mm_movemask_epi8(_mm_or_si128(
                    _mm_or_si128(_mm_cmpgt_epi8(i0, limv),
                                 _mm_cmpgt_epi8(i1, limv)),
                    _mm_or_si128(_mm_cmpgt_epi8(i2, limv),
                                 _mm_cmpgt_epi8(i3, limv)))) != 0) {
                break;
            }
            a0 = _mm_shuffle_epi8(palv, i0);
            a1 = _mm_shuffle_epi8(palv, i1);
            a2 = _mm_shuffle_epi8(palv, i2);
            a3 = _mm_shuffle_epi8(palv, i3);
            u0 = _mm_unpacklo_epi8(a0, a1);
            u1 = _mm_unpackhi_epi8(a0, a1);
            u2 = _mm_unpacklo_epi8(a2, a3);
            u3 = _mm_unpackhi_epi8(a2, a3);
            _mm_storeu_si128((__m128i *)(void *)(exc_out + i),
                             _mm_unpacklo_epi16(u0, u2));
            _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 16u),
                             _mm_unpackhi_epi16(u0, u2));
            _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 32u),
                             _mm_unpacklo_epi16(u1, u3));
            _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 48u),
                             _mm_unpackhi_epi16(u1, u3));
            i = (uint16_t)(i + 64u);
        }
        return i;
    }

    /* Four-bit indices tile a byte exactly, so one 16-byte load carries 32
     * values: the load, the two index extractions, the two shuffles and the
     * two interleaved stores cover 32 values where the loop below covers
     * 16. Every plane of nine to sixteen distinct values lands on this
     * width -- a 4-bit image, a nibble-packed sensor, entropy-coded data
     * after a transform -- so it is the width that decides how fast such a
     * plane decodes (a dense palette record runs this over the whole
     * plane). Low nibble first and then high, which is the little-endian
     * bit order residual_get_bits uses, so the interleave below is the
     * ascending values 0, 1, 2, ... of the payload. Reads stay inside the
     * packed region: 32 values need bytes [i/2, i/2 + 16) and i + 32 <= k
     * implies i/2 + 16 <= ceil(k/2). */
    if (b == 4u) {
        const __m128i low4 = _mm_set1_epi8(0x0F);

        /* With sixteen values the width is exactly full, so no 4-bit index
         * can name a palette entry that is not there: the range check below
         * is provably dead. Hoisting it out of the loop drops two compares,
         * two movemasks and a branch from every 32 values, and this is the
         * loop a nine-to-sixteen-value plane spends its decode in -- every
         * palette plane dense records carry. */
        if (m == 16u) {
            while ((uint32_t)i + 32u <= (uint32_t)k) {
                __m128i x = _mm_loadu_si128(
                    (const __m128i *)(const void *)(src +
                                                    ((uint32_t)i >> 1u)));
                __m128i lo = _mm_and_si128(x, low4);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), low4);
                __m128i a = _mm_shuffle_epi8(palv, lo);
                __m128i c = _mm_shuffle_epi8(palv, hi);

                _mm_storeu_si128((__m128i *)(void *)(exc_out + i),
                                 _mm_unpacklo_epi8(a, c));
                _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 16u),
                                 _mm_unpackhi_epi8(a, c));
                i = (uint16_t)(i + 32u);
            }
            return i;
        }

        while ((uint32_t)i + 32u <= (uint32_t)k) {
            __m128i x = _mm_loadu_si128(
                (const __m128i *)(const void *)(src + ((uint32_t)i >> 1u)));
            __m128i lo = _mm_and_si128(x, low4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), low4);
            __m128i a;
            __m128i c;

            if (_mm_movemask_epi8(_mm_cmpgt_epi8(lo, limv)) != 0 ||
                _mm_movemask_epi8(_mm_cmpgt_epi8(hi, limv)) != 0) {
                break;
            }
            a = _mm_shuffle_epi8(palv, lo);
            c = _mm_shuffle_epi8(palv, hi);
            _mm_storeu_si128((__m128i *)(void *)(exc_out + i),
                             _mm_unpacklo_epi8(a, c));
            _mm_storeu_si128((__m128i *)(void *)(exc_out + i + 16u),
                             _mm_unpackhi_epi8(a, c));
            i = (uint16_t)(i + 32u);
        }
    }

    while ((uint32_t)i + 16u <= (uint32_t)k) {
        uint32_t bit = (uint32_t)i * b;
        uint32_t off = bit >> 3u;
        unsigned ph = (unsigned)(bit & 7u);
        /* One past the last payload byte this block reads: bit (i+16)*b is
         * still inside the packed region because i + 16 <= k, so every read
         * below is bounded by the payload and the bytes past the block are
         * never touched. A block spans 16*b + ph <= 71 bits, i.e. two reads
         * of at most 8 and 1 bytes. */
        uint32_t end = (bit + 16u * b + 7u) >> 3u;
        uint32_t take = end - off;
        uint64_t w = 0u;
        __m128i idx;

        if (take > 8u) {
            take = 8u;
        }
        memcpy(&w, src + off, (size_t)take);
        if (ph != 0u) {
            uint32_t rem = (end > off + 8u) ? end - (off + 8u) : 0u;
            uint64_t hi = 0u;

            if (rem > 0u) {
                memcpy(&hi, src + off + 8u,
                       (size_t)((rem > 8u) ? 8u : rem));
            }
            w = (w >> ph) | (hi << (64u - ph));
        }
        idx = residual_pal_index(w, b);
        if (_mm_movemask_epi8(_mm_cmpgt_epi8(idx, limv)) != 0) {
            break;
        }
        _mm_storeu_si128((__m128i *)(void *)(exc_out + i),
                         _mm_shuffle_epi8(palv, idx));
        i = (uint16_t)(i + 16u);
    }
    return i;
}

#endif /* ZPSE_RES_HAVE_SSSE3 */

#if ZPSE_RES_HAVE_AVX2

/* m == 16 (b == 4) -- the dense palette form's common width: a plane of
 * nine to sixteen distinct values, packed two per payload byte.
 *
 * The 128-bit loop below unpacks 32 values per step; a 256-bit load
 * carries 64, and the only thing that keeps this from being a plain
 * doubling is the ORDER. vpshufb and the two unpacks are lane-local, so
 * lane 0 would produce values 0..31 and lane 1 values 32..63 in the same
 * register, i.e. out[0..31] then out[32..63] would have to come from four
 * different halves. Both unpacks are therefore permuted once: the two
 * 128-bit halves of the low-byte unpack and of the high-byte unpack are
 * combined pairwise, which is exactly ascending order in two 32-byte
 * stores (0x20 takes the low half of each, 0x31 the high half).
 *
 * With sixteen values the width is full, so no 4-bit index can name a
 * palette entry that is not there and the range check the general form
 * needs is provably dead -- hence the `m == 16` guard at every caller. */
static ZPSE_RES_AVX2 uint16_t residual_decode_mode2_avx2_m16(
    const uint8_t *pal, const uint8_t *src, uint16_t k, uint8_t *out)
{
    const __m256i low4 = _mm256_set1_epi8(0x0F);
    const __m256i palv = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(const void *)pal));
    uint16_t i = 0u;

    while ((uint32_t)i + 64u <= (uint32_t)k) {
        __m256i x = _mm256_loadu_si256(
            (const __m256i *)(const void *)(src + ((uint32_t)i >> 1u)));
        __m256i lo = _mm256_and_si256(x, low4);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low4);
        __m256i a = _mm256_shuffle_epi8(palv, lo);
        __m256i c = _mm256_shuffle_epi8(palv, hi);
        __m256i u = _mm256_unpacklo_epi8(a, c);
        __m256i v = _mm256_unpackhi_epi8(a, c);

        _mm256_storeu_si256((__m256i *)(void *)(out + i),
                            _mm256_permute2x128_si256(u, v, 0x20));
        _mm256_storeu_si256((__m256i *)(void *)(out + i + 32u),
                            _mm256_permute2x128_si256(u, v, 0x31));
        i = (uint16_t)(i + 64u);
    }
    return i;
}

#endif /* ZPSE_RES_HAVE_AVX2 */

static int residual_decode_mode2(const uint8_t *payload,
                                 uint16_t payload_len, uint16_t k,
                                 uint8_t *exc_out)
{
    uint8_t m;
    unsigned b;
    uint32_t packed;
    uint32_t expect;
    uint32_t total_bits;
    uint16_t i;

    if (payload_len < 1u) {
        return PSE_BAD_RESIDUAL;
    }
    m = payload[0];
    if (m < 2u || m > 16u) {
        return PSE_BAD_RESIDUAL;
    }
    b = zpse_ceil_log2_u32(m);
    packed = ((uint32_t)k * b + 7u) / 8u;
    expect = 1u + (uint32_t)m + packed;
    if ((uint32_t)payload_len != expect) {
        return PSE_BAD_RESIDUAL;
    }
    for (i = 0; i < m; i++) {
        if (payload[1 + i] > 254u) {
            return PSE_BAD_RESIDUAL;
        }
    }
    if (packed > 0u) {
        total_bits = (uint32_t)k * b;
        if ((total_bits & 7u) != 0u) {
            /* Only the final byte can carry pad bits. */
            if ((payload[1u + m + packed - 1u] >> (total_bits & 7u)) != 0u) {
                return PSE_BAD_RESIDUAL;
            }
        }
    }
#if ZPSE_RES_HAVE_SSSE3
    /* Below a block there is nothing to unpack, and the setup (a caps query
     * plus the zero-padded palette) would cost more than the scalar walk. */
    i = (k >= 8u) ? residual_decode_mode2_simd(payload + 1u, payload + 1u + m,
                                               k, b, m, exc_out)
                  : 0u;
    /* A half-written tile after a rejection is unobservable: every caller
     * returns immediately on a non-OK status and exc_out is the tile's
     * resid[] scratch, which is never read afterwards. */
#else
    i = 0;
#endif
    for (; i < k; i++) {
        unsigned idx = residual_get_bits(payload + 1u + m,
                                         (uint32_t)i * b, b);
        if (idx >= m) {
            return PSE_BAD_RESIDUAL;
        }
        exc_out[i] = payload[1u + idx];
    }
    return PSE_OK;
}

/* Base-M digit pairs, one per byte pair: m3_pair_lut[off + r] holds
 * (r % M) | ((r / M) << 8) for every r below M*M, and m3_pair_off[M] is
 * where M's row starts. Rows are in the order residual_decode_mode3's
 * switch names them: 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15. The table is
 * what makes a two-digit step cost one division: see that function. */
static const uint16_t m3_pair_lut[1155] = {
    0x0000, 0x0001, 0x0002, 0x0100, 0x0101, 0x0102, 0x0200, 0x0201,
    0x0202, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0100, 0x0101,
    0x0102, 0x0103, 0x0104, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204,
    0x0300, 0x0301, 0x0302, 0x0303, 0x0304, 0x0400, 0x0401, 0x0402,
    0x0403, 0x0404, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005,
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0200, 0x0201,
    0x0202, 0x0203, 0x0204, 0x0205, 0x0300, 0x0301, 0x0302, 0x0303,
    0x0304, 0x0305, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405,
    0x0500, 0x0501, 0x0502, 0x0503, 0x0504, 0x0505, 0x0000, 0x0001,
    0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0100, 0x0101, 0x0102,
    0x0103, 0x0104, 0x0105, 0x0106, 0x0200, 0x0201, 0x0202, 0x0203,
    0x0204, 0x0205, 0x0206, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304,
    0x0305, 0x0306, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405,
    0x0406, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504, 0x0505, 0x0506,
    0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x0606, 0x0000,
    0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008,
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
    0x0108, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206,
    0x0207, 0x0208, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304, 0x0305,
    0x0306, 0x0307, 0x0308, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404,
    0x0405, 0x0406, 0x0407, 0x0408, 0x0500, 0x0501, 0x0502, 0x0503,
    0x0504, 0x0505, 0x0506, 0x0507, 0x0508, 0x0600, 0x0601, 0x0602,
    0x0603, 0x0604, 0x0605, 0x0606, 0x0607, 0x0608, 0x0700, 0x0701,
    0x0702, 0x0703, 0x0704, 0x0705, 0x0706, 0x0707, 0x0708, 0x0800,
    0x0801, 0x0802, 0x0803, 0x0804, 0x0805, 0x0806, 0x0807, 0x0808,
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007,
    0x0008, 0x0009, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105,
    0x0106, 0x0107, 0x0108, 0x0109, 0x0200, 0x0201, 0x0202, 0x0203,
    0x0204, 0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x0300, 0x0301,
    0x0302, 0x0303, 0x0304, 0x0305, 0x0306, 0x0307, 0x0308, 0x0309,
    0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407,
    0x0408, 0x0409, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504, 0x0505,
    0x0506, 0x0507, 0x0508, 0x0509, 0x0600, 0x0601, 0x0602, 0x0603,
    0x0604, 0x0605, 0x0606, 0x0607, 0x0608, 0x0609, 0x0700, 0x0701,
    0x0702, 0x0703, 0x0704, 0x0705, 0x0706, 0x0707, 0x0708, 0x0709,
    0x0800, 0x0801, 0x0802, 0x0803, 0x0804, 0x0805, 0x0806, 0x0807,
    0x0808, 0x0809, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905,
    0x0906, 0x0907, 0x0908, 0x0909, 0x0000, 0x0001, 0x0002, 0x0003,
    0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A, 0x0100,
    0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107, 0x0108,
    0x0109, 0x010A, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205,
    0x0206, 0x0207, 0x0208, 0x0209, 0x020A, 0x0300, 0x0301, 0x0302,
    0x0303, 0x0304, 0x0305, 0x0306, 0x0307, 0x0308, 0x0309, 0x030A,
    0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407,
    0x0408, 0x0409, 0x040A, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504,
    0x0505, 0x0506, 0x0507, 0x0508, 0x0509, 0x050A, 0x0600, 0x0601,
    0x0602, 0x0603, 0x0604, 0x0605, 0x0606, 0x0607, 0x0608, 0x0609,
    0x060A, 0x0700, 0x0701, 0x0702, 0x0703, 0x0704, 0x0705, 0x0706,
    0x0707, 0x0708, 0x0709, 0x070A, 0x0800, 0x0801, 0x0802, 0x0803,
    0x0804, 0x0805, 0x0806, 0x0807, 0x0808, 0x0809, 0x080A, 0x0900,
    0x0901, 0x0902, 0x0903, 0x0904, 0x0905, 0x0906, 0x0907, 0x0908,
    0x0909, 0x090A, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0A04, 0x0A05,
    0x0A06, 0x0A07, 0x0A08, 0x0A09, 0x0A0A, 0x0000, 0x0001, 0x0002,
    0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A,
    0x000B, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106,
    0x0107, 0x0108, 0x0109, 0x010A, 0x010B, 0x0200, 0x0201, 0x0202,
    0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208, 0x0209, 0x020A,
    0x020B, 0x0300, 0x0301, 0x0302, 0x0303, 0x0304, 0x0305, 0x0306,
    0x0307, 0x0308, 0x0309, 0x030A, 0x030B, 0x0400, 0x0401, 0x0402,
    0x0403, 0x0404, 0x0405, 0x0406, 0x0407, 0x0408, 0x0409, 0x040A,
    0x040B, 0x0500, 0x0501, 0x0502, 0x0503, 0x0504, 0x0505, 0x0506,
    0x0507, 0x0508, 0x0509, 0x050A, 0x050B, 0x0600, 0x0601, 0x0602,
    0x0603, 0x0604, 0x0605, 0x0606, 0x0607, 0x0608, 0x0609, 0x060A,
    0x060B, 0x0700, 0x0701, 0x0702, 0x0703, 0x0704, 0x0705, 0x0706,
    0x0707, 0x0708, 0x0709, 0x070A, 0x070B, 0x0800, 0x0801, 0x0802,
    0x0803, 0x0804, 0x0805, 0x0806, 0x0807, 0x0808, 0x0809, 0x080A,
    0x080B, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905, 0x0906,
    0x0907, 0x0908, 0x0909, 0x090A, 0x090B, 0x0A00, 0x0A01, 0x0A02,
    0x0A03, 0x0A04, 0x0A05, 0x0A06, 0x0A07, 0x0A08, 0x0A09, 0x0A0A,
    0x0A0B, 0x0B00, 0x0B01, 0x0B02, 0x0B03, 0x0B04, 0x0B05, 0x0B06,
    0x0B07, 0x0B08, 0x0B09, 0x0B0A, 0x0B0B, 0x0000, 0x0001, 0x0002,
    0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A,
    0x000B, 0x000C, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105,
    0x0106, 0x0107, 0x0108, 0x0109, 0x010A, 0x010B, 0x010C, 0x0200,
    0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207, 0x0208,
    0x0209, 0x020A, 0x020B, 0x020C, 0x0300, 0x0301, 0x0302, 0x0303,
    0x0304, 0x0305, 0x0306, 0x0307, 0x0308, 0x0309, 0x030A, 0x030B,
    0x030C, 0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406,
    0x0407, 0x0408, 0x0409, 0x040A, 0x040B, 0x040C, 0x0500, 0x0501,
    0x0502, 0x0503, 0x0504, 0x0505, 0x0506, 0x0507, 0x0508, 0x0509,
    0x050A, 0x050B, 0x050C, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604,
    0x0605, 0x0606, 0x0607, 0x0608, 0x0609, 0x060A, 0x060B, 0x060C,
    0x0700, 0x0701, 0x0702, 0x0703, 0x0704, 0x0705, 0x0706, 0x0707,
    0x0708, 0x0709, 0x070A, 0x070B, 0x070C, 0x0800, 0x0801, 0x0802,
    0x0803, 0x0804, 0x0805, 0x0806, 0x0807, 0x0808, 0x0809, 0x080A,
    0x080B, 0x080C, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905,
    0x0906, 0x0907, 0x0908, 0x0909, 0x090A, 0x090B, 0x090C, 0x0A00,
    0x0A01, 0x0A02, 0x0A03, 0x0A04, 0x0A05, 0x0A06, 0x0A07, 0x0A08,
    0x0A09, 0x0A0A, 0x0A0B, 0x0A0C, 0x0B00, 0x0B01, 0x0B02, 0x0B03,
    0x0B04, 0x0B05, 0x0B06, 0x0B07, 0x0B08, 0x0B09, 0x0B0A, 0x0B0B,
    0x0B0C, 0x0C00, 0x0C01, 0x0C02, 0x0C03, 0x0C04, 0x0C05, 0x0C06,
    0x0C07, 0x0C08, 0x0C09, 0x0C0A, 0x0C0B, 0x0C0C, 0x0000, 0x0001,
    0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009,
    0x000A, 0x000B, 0x000C, 0x000D, 0x0100, 0x0101, 0x0102, 0x0103,
    0x0104, 0x0105, 0x0106, 0x0107, 0x0108, 0x0109, 0x010A, 0x010B,
    0x010C, 0x010D, 0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205,
    0x0206, 0x0207, 0x0208, 0x0209, 0x020A, 0x020B, 0x020C, 0x020D,
    0x0300, 0x0301, 0x0302, 0x0303, 0x0304, 0x0305, 0x0306, 0x0307,
    0x0308, 0x0309, 0x030A, 0x030B, 0x030C, 0x030D, 0x0400, 0x0401,
    0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407, 0x0408, 0x0409,
    0x040A, 0x040B, 0x040C, 0x040D, 0x0500, 0x0501, 0x0502, 0x0503,
    0x0504, 0x0505, 0x0506, 0x0507, 0x0508, 0x0509, 0x050A, 0x050B,
    0x050C, 0x050D, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605,
    0x0606, 0x0607, 0x0608, 0x0609, 0x060A, 0x060B, 0x060C, 0x060D,
    0x0700, 0x0701, 0x0702, 0x0703, 0x0704, 0x0705, 0x0706, 0x0707,
    0x0708, 0x0709, 0x070A, 0x070B, 0x070C, 0x070D, 0x0800, 0x0801,
    0x0802, 0x0803, 0x0804, 0x0805, 0x0806, 0x0807, 0x0808, 0x0809,
    0x080A, 0x080B, 0x080C, 0x080D, 0x0900, 0x0901, 0x0902, 0x0903,
    0x0904, 0x0905, 0x0906, 0x0907, 0x0908, 0x0909, 0x090A, 0x090B,
    0x090C, 0x090D, 0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0A04, 0x0A05,
    0x0A06, 0x0A07, 0x0A08, 0x0A09, 0x0A0A, 0x0A0B, 0x0A0C, 0x0A0D,
    0x0B00, 0x0B01, 0x0B02, 0x0B03, 0x0B04, 0x0B05, 0x0B06, 0x0B07,
    0x0B08, 0x0B09, 0x0B0A, 0x0B0B, 0x0B0C, 0x0B0D, 0x0C00, 0x0C01,
    0x0C02, 0x0C03, 0x0C04, 0x0C05, 0x0C06, 0x0C07, 0x0C08, 0x0C09,
    0x0C0A, 0x0C0B, 0x0C0C, 0x0C0D, 0x0D00, 0x0D01, 0x0D02, 0x0D03,
    0x0D04, 0x0D05, 0x0D06, 0x0D07, 0x0D08, 0x0D09, 0x0D0A, 0x0D0B,
    0x0D0C, 0x0D0D, 0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005,
    0x0006, 0x0007, 0x0008, 0x0009, 0x000A, 0x000B, 0x000C, 0x000D,
    0x000E, 0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106,
    0x0107, 0x0108, 0x0109, 0x010A, 0x010B, 0x010C, 0x010D, 0x010E,
    0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207,
    0x0208, 0x0209, 0x020A, 0x020B, 0x020C, 0x020D, 0x020E, 0x0300,
    0x0301, 0x0302, 0x0303, 0x0304, 0x0305, 0x0306, 0x0307, 0x0308,
    0x0309, 0x030A, 0x030B, 0x030C, 0x030D, 0x030E, 0x0400, 0x0401,
    0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407, 0x0408, 0x0409,
    0x040A, 0x040B, 0x040C, 0x040D, 0x040E, 0x0500, 0x0501, 0x0502,
    0x0503, 0x0504, 0x0505, 0x0506, 0x0507, 0x0508, 0x0509, 0x050A,
    0x050B, 0x050C, 0x050D, 0x050E, 0x0600, 0x0601, 0x0602, 0x0603,
    0x0604, 0x0605, 0x0606, 0x0607, 0x0608, 0x0609, 0x060A, 0x060B,
    0x060C, 0x060D, 0x060E, 0x0700, 0x0701, 0x0702, 0x0703, 0x0704,
    0x0705, 0x0706, 0x0707, 0x0708, 0x0709, 0x070A, 0x070B, 0x070C,
    0x070D, 0x070E, 0x0800, 0x0801, 0x0802, 0x0803, 0x0804, 0x0805,
    0x0806, 0x0807, 0x0808, 0x0809, 0x080A, 0x080B, 0x080C, 0x080D,
    0x080E, 0x0900, 0x0901, 0x0902, 0x0903, 0x0904, 0x0905, 0x0906,
    0x0907, 0x0908, 0x0909, 0x090A, 0x090B, 0x090C, 0x090D, 0x090E,
    0x0A00, 0x0A01, 0x0A02, 0x0A03, 0x0A04, 0x0A05, 0x0A06, 0x0A07,
    0x0A08, 0x0A09, 0x0A0A, 0x0A0B, 0x0A0C, 0x0A0D, 0x0A0E, 0x0B00,
    0x0B01, 0x0B02, 0x0B03, 0x0B04, 0x0B05, 0x0B06, 0x0B07, 0x0B08,
    0x0B09, 0x0B0A, 0x0B0B, 0x0B0C, 0x0B0D, 0x0B0E, 0x0C00, 0x0C01,
    0x0C02, 0x0C03, 0x0C04, 0x0C05, 0x0C06, 0x0C07, 0x0C08, 0x0C09,
    0x0C0A, 0x0C0B, 0x0C0C, 0x0C0D, 0x0C0E, 0x0D00, 0x0D01, 0x0D02,
    0x0D03, 0x0D04, 0x0D05, 0x0D06, 0x0D07, 0x0D08, 0x0D09, 0x0D0A,
    0x0D0B, 0x0D0C, 0x0D0D, 0x0D0E, 0x0E00, 0x0E01, 0x0E02, 0x0E03,
    0x0E04, 0x0E05, 0x0E06, 0x0E07, 0x0E08, 0x0E09, 0x0E0A, 0x0E0B,
    0x0E0C, 0x0E0D, 0x0E0E,
};

static const uint16_t m3_pair_off[11] = {
    0, 9, 34, 70, 119, 200, 300, 421, 565, 734, 930
};

/* Two base-M digits of `W` at once: the quotient by M^2, then the pair
 * itself from the table (the remainder is below M^2 by construction, so
 * it indexes the row). Writes the two bytes through PTR. */
#define ZPSE_M3_PAIR(W, PTR, MM, LUT)                                       \
    do {                                                                    \
        uint32_t q_ = (W) / ((uint32_t)(MM) * (uint32_t)(MM));              \
        uint32_t r_ = (W) - q_ * ((uint32_t)(MM) * (uint32_t)(MM));         \
        memcpy((PTR), (LUT) + r_, sizeof(uint16_t));                        \
        (W) = q_;                                                           \
    } while (0)

#if ZPSE_RES_HAVE_AVX2

/* Magic constants for q = w / M, exact for every w < M^G (the only range a
 * group word can hold), as (magic, extra shift) with
 *
 *     q = ((uint64)w * magic) >> (32 + sh)
 *
 * magic = ceil(2^(32+sh) / M), chosen so the rounding error times M^G
 * stays inside 2^(32+sh): (magic*M - 2^(32+sh)) * M^G <= 2^(32+sh) is
 * what makes floor(w*magic / 2^(32+sh)) == floor(w/M) for every w below
 * the bound, and it holds for all eleven M. magic fits in 32 bits for each
 * of them, so one unsigned 32x32 -> 64 multiply is the whole quotient.
 * tests/suites/test_units.c sweeps every M and every K through the packer
 * and this unpacker, which is what pins these constants: the scalar loops
 * below share the table, so a change that only the vector path could see
 * would fail there. */
static const uint32_t m3_div_magic[16] = {
    0u, 0u, 0u, UINT32_C(2863311531), 0u, UINT32_C(1717986919),
    UINT32_C(715827883), UINT32_C(613566757), 0u, UINT32_C(477218589),
    UINT32_C(429496730), UINT32_C(390451573), UINT32_C(357913942),
    UINT32_C(330382100), UINT32_C(306783379), UINT32_C(286331154)
};

static const uint8_t m3_div_sh[16] = {
    0u, 0u, 0u, 1u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
};

/* High 32 bits of the eight unsigned 32x32 products, per 32-bit lane.
 * AVX2's multiply-high is 64-bit-lane only, so the even and the odd lanes
 * are multiplied separately (each 4 products) and blended back together;
 * `magic_hi` is magic with its low 32 bits in every 64-bit lane, which is
 * what the odd-lane multiply of the shifted word needs. */
static ZPSE_RES_AVX2 __attribute__((always_inline)) inline __m256i
m3_mulhi32(__m256i a, __m256i magic, __m256i magic_hi, int sh)
{
    __m256i lo = _mm256_mul_epu32(a, magic);
    __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), magic_hi);

    /* Same shift on both halves: the shift count is 32 + sh, which is a
     * compile-time constant in the two callers below, so this is one
     * immediate shift each. */
    lo = _mm256_srli_epi64(lo, 32);
    hi = _mm256_slli_epi64(_mm256_srli_epi64(hi, 32), 32);
    if (sh != 0) {
        lo = _mm256_srli_epi64(lo, sh);
        hi = _mm256_srli_epi64(hi, sh);
    }
    return _mm256_blend_epi32(lo, hi, 0xAA);
}

/* Eight groups' worth of nibble-packed digits, expanded to bytes.
 *
 * For M in [11,15] a digit is below 15 and a group holds 7 or 8 of them,
 * so the whole group fits in the nibbles of one 32-bit lane. The lanes
 * hold consecutive groups, so the expansion is two byte-shuffles: the
 * even digits (bit 4i of lane j) and the odd ones interleave per group,
 * and with G == 8 that interleave is already the output layout -- 8 bytes
 * per group, groups 0,1 in the low half of one register and 4,5 in the
 * high half. With G == 7 one fixed shuffle per 128-bit half closes each
 * 8-byte record up to 7 bytes; the two bytes of padding it leaves behind
 * are the start of the next half's record, which is stored afterwards. */
static ZPSE_RES_AVX2 __attribute__((always_inline)) inline void
m3_nib_expand8(uint8_t *dst, __m256i acc, uint32_t g)
{
    const __m256i nib = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(acc, nib);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(acc, 4), nib);
    __m256i a = _mm256_unpacklo_epi8(lo, hi); /* groups 0,1 | 4,5 */
    __m256i b = _mm256_unpackhi_epi8(lo, hi); /* groups 2,3 | 6,7 */

    if (g == 8u) {
        _mm_storeu_si128((__m128i *)(void *)(dst + 0u),
                         _mm256_castsi256_si128(a));
        _mm_storeu_si128((__m128i *)(void *)(dst + 16u),
                         _mm256_castsi256_si128(b));
        _mm_storeu_si128((__m128i *)(void *)(dst + 32u),
                         _mm256_extracti128_si256(a, 1));
        _mm_storeu_si128((__m128i *)(void *)(dst + 48u),
                         _mm256_extracti128_si256(b, 1));
    } else {
        const __m256i close = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, (char)0x80,
            (char)0x80, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
            (char)0x80, (char)0x80);
        __m256i ca = _mm256_shuffle_epi8(a, close);
        __m256i cb = _mm256_shuffle_epi8(b, close);

        /* Stored in this order: each store's two pad bytes are the start
         * of the record the next one writes (dst+14 and dst+28 are group
         * 2's and group 4's first byte), and the last one's two land in
         * the slack the caller's buffer has. */
        _mm_storeu_si128((__m128i *)(void *)(dst + 0u),
                         _mm256_castsi256_si128(ca));
        _mm_storeu_si128((__m128i *)(void *)(dst + 14u),
                         _mm256_castsi256_si128(cb));
        _mm_storeu_si128((__m128i *)(void *)(dst + 28u),
                         _mm256_extracti128_si256(ca, 1));
        _mm_storeu_si128((__m128i *)(void *)(dst + 42u),
                         _mm256_extracti128_si256(cb, 1));
    }
}

/* The same two helpers four groups wide, for the tail an eight-group batch
 * cannot take: k = 256 with M = 15 is 36 whole groups, so four of them are
 * left after four batches of eight. */
static ZPSE_RES_AVX2 __attribute__((always_inline)) inline __m128i
m3_mulhi32x4(__m128i a, __m128i magic, __m128i magic_hi, int sh)
{
    __m128i lo = _mm_mul_epu32(a, magic);
    __m128i hi = _mm_mul_epu32(_mm_srli_epi64(a, 32), magic_hi);

    lo = _mm_srli_epi64(lo, 32);
    hi = _mm_slli_epi64(_mm_srli_epi64(hi, 32), 32);
    if (sh != 0) {
        lo = _mm_srli_epi64(lo, sh);
        hi = _mm_srli_epi64(hi, sh);
    }
    /* 32-bit lanes 1 and 3 come from `hi`: two 16-bit lanes each, 0xCC. */
    return _mm_blend_epi16(lo, hi, 0xCC);
}

static ZPSE_RES_AVX2 __attribute__((always_inline)) inline void
m3_nib_expand4(uint8_t *dst, __m128i acc, uint32_t g)
{
    const __m128i nib = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(acc, nib);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(acc, 4), nib);
    __m128i a = _mm_unpacklo_epi8(lo, hi); /* groups 0,1 */
    __m128i b = _mm_unpackhi_epi8(lo, hi); /* groups 2,3 */

    if (g == 8u) {
        _mm_storeu_si128((__m128i *)(void *)(dst + 0u), a);
        _mm_storeu_si128((__m128i *)(void *)(dst + 16u), b);
    } else {
        const __m128i close = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 8, 9,
                                            10, 11, 12, 13, 14,
                                            (char)0x80, (char)0x80);

        _mm_storeu_si128((__m128i *)(void *)(dst + 0u),
                         _mm_shuffle_epi8(a, close));
        _mm_storeu_si128((__m128i *)(void *)(dst + 14u),
                         _mm_shuffle_epi8(b, close));
    }
}

/* Nibble-packed base-M unpack for M in [11,15]: eight groups (eight 32-bit
 * words, one 32-byte load) per step, each group's digits accumulated as
 * nibbles of its lane by a shift-or, then expanded by shuffles.
 *
 * The pair-LUT form the scalar loops use cannot be vectorised -- its table
 * lookup is a gather -- so this path divides instead. Every step is a
 * multiply-high, a multiply, a subtract and a shift-or on eight lanes at
 * once, i.e. one digit for eight groups for about the cost the scalar
 * pair loop pays for two digits of one group. Writes 8*g bytes per batch
 * to `dst` (plus two bytes of padding in the g == 7 case), which may
 * therefore reach 8*g + 2 bytes past the buffer start.
 *
 * Returns the number of groups consumed, or 0 if a group word was out of
 * range (any word >= M^G leaves a non-zero quotient after G digits, which
 * is exactly the scalar loops' own check). */
#define ZPSE_M3_NIB(MM, GG)                                                 \
    case (MM): {                                                           \
        const __m256i magic = _mm256_set1_epi32((int)m3_div_magic[(MM)]);   \
        const __m256i magic_hi = _mm256_srli_epi64(magic, 32);             \
        const __m256i mvec = _mm256_set1_epi32((MM));                      \
        const __m128i magic4 = _mm_set1_epi32((int)m3_div_magic[(MM)]);     \
        const __m128i magic4_hi = _mm_srli_epi64(magic4, 32);              \
        const __m128i mvec4 = _mm_set1_epi32((MM));                        \
        const int sh = (int)m3_div_sh[(MM)];                               \
        uint32_t n = 0u;                                                   \
        /* Two bytes may be written past the last group of a batch, so a     \
         * batch is only taken while at least two digits follow it: the      \
         * scalar loops that finish the tile then overwrite them. */        \
        while (n + 8u <= nfull &&                                          \
               (uint32_t)k - (n + 8u) * (uint32_t)(GG) >= 2u) {            \
            const uint8_t *p = payload + 1u + (j + n) * 4u;                \
            __m256i w = _mm256_loadu_si256(                                \
                (const __m256i *)(const void *)p);                         \
            __m256i acc = _mm256_setzero_si256();                          \
            __m256i q;                                                     \
            acc = _mm256_slli_epi32(                                       \
                _mm256_sub_epi32(w, _mm256_mullo_epi32(                    \
                    (q = m3_mulhi32(w, magic, magic_hi, sh)), mvec)), 0);   \
            w = q;                                                         \
            ZPSE_M3_NIB_STEPS(2);                                          \
            ZPSE_M3_NIB_STEPS(3);                                          \
            ZPSE_M3_NIB_STEPS(4);                                          \
            ZPSE_M3_NIB_STEPS(5);                                          \
            ZPSE_M3_NIB_STEPS(6);                                          \
            ZPSE_M3_NIB_STEPS(7);                                          \
            if ((GG) == 8) {                                               \
                ZPSE_M3_NIB_STEPS(8);                                      \
            }                                                              \
            if (_mm256_testz_si256(w, w) == 0) {                           \
                *consumed = 0u;                                            \
                return PSE_BAD_RESIDUAL;                                   \
            }                                                              \
            m3_nib_expand8(dst + n * (uint32_t)(GG), acc, (GG));           \
            n += 8u;                                                       \
        }                                                                  \
        while (n + 4u <= nfull &&                                          \
               (uint32_t)k - (n + 4u) * (uint32_t)(GG) >= 2u) {            \
            const uint8_t *p4 = payload + 1u + (j + n) * 4u;               \
            __m128i w4 = _mm_loadu_si128(                                  \
                (const __m128i *)(const void *)p4);                        \
            __m128i acc4 = _mm_setzero_si128();                            \
            __m128i q4;                                                    \
            acc4 = _mm_slli_epi32(                                         \
                _mm_sub_epi32(w4, _mm_mullo_epi32(                         \
                    (q4 = m3_mulhi32x4(w4, magic4, magic4_hi, sh)),         \
                    mvec4)), 0);                                           \
            w4 = q4;                                                       \
            ZPSE_M3_NIB_STEPS4(2);                                         \
            ZPSE_M3_NIB_STEPS4(3);                                         \
            ZPSE_M3_NIB_STEPS4(4);                                         \
            ZPSE_M3_NIB_STEPS4(5);                                         \
            ZPSE_M3_NIB_STEPS4(6);                                         \
            ZPSE_M3_NIB_STEPS4(7);                                         \
            if ((GG) == 8) {                                               \
                ZPSE_M3_NIB_STEPS4(8);                                     \
            }                                                              \
            if (_mm_testz_si128(w4, w4) == 0) {                            \
                *consumed = 0u;                                            \
                return PSE_BAD_RESIDUAL;                                   \
            }                                                              \
            m3_nib_expand4(dst + n * (uint32_t)(GG), acc4, (GG));          \
            n += 4u;                                                       \
        }                                                                  \
        *consumed = n;                                                     \
        return PSE_OK;                                                     \
    }                                                                      \
    break;

/* The four-wide forms of the same two macros. */
#define ZPSE_M3_NIB_STEPS4(I)                                              \
    do {                                                                   \
        acc4 = _mm_or_si128(                                               \
            acc4,                                                          \
            _mm_slli_epi32(                                                \
                _mm_sub_epi32(                                             \
                    w4, _mm_mullo_epi32(                                   \
                            (q4 = m3_mulhi32x4(w4, magic4, magic4_hi, sh)),  \
                            mvec4)),                                        \
                4 * ((I) - 1)));                                           \
        w4 = q4;                                                           \
    } while (0)

/* One more digit for the batch: acc |= (w - (w/M)*M) << 4*(i-1). The
 * shift count is a literal at each use, which is what keeps it an
 * immediate. */
#define ZPSE_M3_NIB_STEPS(I)                                               \
    do {                                                                   \
        acc = _mm256_or_si256(                                             \
            acc,                                                           \
            _mm256_slli_epi32(                                             \
                _mm256_sub_epi32(                                          \
                    w, _mm256_mullo_epi32(                                 \
                           (q = m3_mulhi32(w, magic, magic_hi, sh)),        \
                           mvec)),                                         \
                4 * ((I) - 1)));                                           \
        w = q;                                                             \
    } while (0)

static ZPSE_RES_AVX2 int residual_decode_mode3_nib(
    const uint8_t *payload, uint32_t nfull, uint32_t k, uint8_t m,
    uint8_t *dst, uint32_t *consumed)
{
    uint32_t j = 0u;

    *consumed = 0u;
    switch (m) {
    ZPSE_M3_NIB(11, 8)
    ZPSE_M3_NIB(12, 8)
    ZPSE_M3_NIB(13, 8)
    ZPSE_M3_NIB(14, 7)
    ZPSE_M3_NIB(15, 7)
    default:
        return PSE_OK;
    }
}

#undef ZPSE_M3_NIB
#undef ZPSE_M3_NIB_STEPS
#undef ZPSE_M3_NIB_STEPS4

#endif /* ZPSE_RES_HAVE_AVX2 */

static int residual_decode_mode3(const uint8_t *payload,
                                 uint16_t payload_len, uint16_t k,
                                 uint8_t *exc_out)
{
    uint8_t m;
    uint8_t g;
    uint32_t ng;
    uint32_t nfull;
    uint32_t j;
    uint32_t bad = 0u;

    if (payload_len < 1u) {
        return PSE_BAD_RESIDUAL;
    }
    m = payload[0];
    g = zpse_base_m_group(m);
    if (g == 0u) {
        return PSE_BAD_RESIDUAL;
    }
    ng = ((uint32_t)k + (uint32_t)g - 1u) / (uint32_t)g;
    if ((uint32_t)payload_len != 1u + ng * 4u) {
        return PSE_BAD_RESIDUAL;
    }

    /* Only whole groups are decoded by the two loops below, so a group
     * whose digit count is short of G -- the last one, when K is not a
     * multiple of G -- is always left to the single-group loop at the end.
     * Writing it in a batched loop would put bytes past k. */
    nfull = (((uint32_t)k % (uint32_t)g) == 0u) ? ng : ng - 1u;

    j = 0u;
#if ZPSE_RES_HAVE_AVX2
    /* Declared here, not with the others: the vector batch is the only
     * caller, and the kernel-less build (see ZPSE_AVX2_ABI_OK) must not
     * carry an unused local. */
    int rc;

    /* Nibble-packed batches of eight groups (see residual_decode_mode3_nib).
     * One group is always left to the scalar loops below, so the two bytes
     * of padding the g == 7 stores can leave past a batch always land in a
     * group those loops then write in full. */
    if (g >= 7u && g <= 8u && nfull >= 9u &&
        (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        rc = residual_decode_mode3_nib(payload, nfull, k, m, exc_out, &j);
        if (rc != PSE_OK) {
            return rc;
        }
    }
#endif

    /* Two digits per step.
     *
     * The single-digit form is bound by instruction count, not latency:
     * two divides cost one multiply-shift for the quotient, and the digit
     * itself -- w - M*q -- another multiply or a shift/subtract pair, so
     * every digit carried a multiply. Dividing by M^2 instead yields two
     * digits from the same one multiply, and the pair (d0, d1) of a
     * remainder below M^2 is a byte -> byte function, so the second
     * division becomes one table load. That is two multiplies per two
     * digits instead of two per digit, which measures at 0.76 cycles per
     * digit against 1.9 (6-wide, gcc 15, M = 15, K = 126).
     *
     * A group of odd G leaves one digit after the pairs, and it needs no
     * division at all: every pair step divides by M^2, so the leftover is
     * the most significant digit itself. It still has to be range-checked
     * -- a hostile word can leave M or more there -- which the scalar
     * single-digit loop expressed as "the final quotient is zero".
     *
     * Groups are independent, so the batches interleave them: the six
     * words, each of which is a chain of its own, run as six chains. Six
     * remains the measured optimum (see the earlier note): four runs 2.3
     * cycles a digit after GCC packs them into SSE lanes, eight spills.
     * What this loop leaves over (fewer than six groups) the four-group
     * loop below runs. */
    for (; j + 6u <= nfull; j += 6u) {
        const uint8_t *p = payload + 1u + j * 4u;
        uint32_t w0 = zpse_load_u32le(p);
        uint32_t w1 = zpse_load_u32le(p + 4u);
        uint32_t w2 = zpse_load_u32le(p + 8u);
        uint32_t w3 = zpse_load_u32le(p + 12u);
        uint32_t w4 = zpse_load_u32le(p + 16u);
        uint32_t w5 = zpse_load_u32le(p + 20u);
        uint8_t *o = exc_out + j * (uint32_t)g;
        uint32_t i;

        /* M is one of eleven fixed constants. Naming it at the point of
         * use lets the compiler replace the hardware integer divide (one
         * per unit, ~20+ cycles) with a multiply-shift pair. */
        switch (m) {
#define ZPSE_M3_PAIRS6(MM, GI, GG)                                         \
        case (MM): {                                                       \
            const uint16_t *lut = m3_pair_lut + m3_pair_off[(GI)];         \
            for (i = 0u; i + 2u <= (GG); i += 2u) {                        \
                ZPSE_M3_PAIR(w0, o + i, MM, lut);                          \
                ZPSE_M3_PAIR(w1, o + (GG) + i, MM, lut);                   \
                ZPSE_M3_PAIR(w2, o + 2u * (GG) + i, MM, lut);              \
                ZPSE_M3_PAIR(w3, o + 3u * (GG) + i, MM, lut);              \
                ZPSE_M3_PAIR(w4, o + 4u * (GG) + i, MM, lut);              \
                ZPSE_M3_PAIR(w5, o + 5u * (GG) + i, MM, lut);              \
            }                                                              \
            if (i < (GG)) {                                                \
                bad |= (w0 >= (uint32_t)(MM));                             \
                bad |= (w1 >= (uint32_t)(MM));                             \
                bad |= (w2 >= (uint32_t)(MM));                             \
                bad |= (w3 >= (uint32_t)(MM));                             \
                bad |= (w4 >= (uint32_t)(MM));                             \
                bad |= (w5 >= (uint32_t)(MM));                             \
                o[i] = (uint8_t)w0;                                        \
                o[(GG) + i] = (uint8_t)w1;                                 \
                o[2u * (GG) + i] = (uint8_t)w2;                            \
                o[3u * (GG) + i] = (uint8_t)w3;                            \
                o[4u * (GG) + i] = (uint8_t)w4;                            \
                o[5u * (GG) + i] = (uint8_t)w5;                            \
                w0 = w1 = w2 = w3 = w4 = w5 = 0u;                          \
            }                                                              \
        } break
        ZPSE_M3_PAIRS6(3, 0, 20);
        ZPSE_M3_PAIRS6(5, 1, 13);
        ZPSE_M3_PAIRS6(6, 2, 11);
        ZPSE_M3_PAIRS6(7, 3, 10);
        ZPSE_M3_PAIRS6(9, 4, 9);
        ZPSE_M3_PAIRS6(10, 5, 9);
        ZPSE_M3_PAIRS6(11, 6, 8);
        ZPSE_M3_PAIRS6(12, 7, 8);
        ZPSE_M3_PAIRS6(13, 8, 8);
        ZPSE_M3_PAIRS6(14, 9, 7);
        ZPSE_M3_PAIRS6(15, 10, 7);
#undef ZPSE_M3_PAIRS6
        default:
            return PSE_BAD_RESIDUAL;
        }
        /* Full groups: word < M^G. Same check per group. */
        if ((w0 | w1 | w2 | w3 | w4 | w5) != 0u) {
            return PSE_BAD_RESIDUAL;
        }
    }

    /* The same pairs four wide, for the 0..5 groups a batch of six could
     * not take. */
    for (; j + 4u <= nfull; j += 4u) {
        const uint8_t *p = payload + 1u + j * 4u;
        uint32_t w0 = zpse_load_u32le(p);
        uint32_t w1 = zpse_load_u32le(p + 4u);
        uint32_t w2 = zpse_load_u32le(p + 8u);
        uint32_t w3 = zpse_load_u32le(p + 12u);
        uint8_t *o = exc_out + j * (uint32_t)g;
        uint32_t i;

        switch (m) {
#define ZPSE_M3_PAIRS4(MM, GI, GG)                                         \
        case (MM): {                                                       \
            const uint16_t *lut = m3_pair_lut + m3_pair_off[(GI)];         \
            for (i = 0u; i + 2u <= (GG); i += 2u) {                        \
                ZPSE_M3_PAIR(w0, o + i, MM, lut);                          \
                ZPSE_M3_PAIR(w1, o + (GG) + i, MM, lut);                   \
                ZPSE_M3_PAIR(w2, o + 2u * (GG) + i, MM, lut);              \
                ZPSE_M3_PAIR(w3, o + 3u * (GG) + i, MM, lut);              \
            }                                                              \
            if (i < (GG)) {                                                \
                bad |= (w0 >= (uint32_t)(MM));                             \
                bad |= (w1 >= (uint32_t)(MM));                             \
                bad |= (w2 >= (uint32_t)(MM));                             \
                bad |= (w3 >= (uint32_t)(MM));                             \
                o[i] = (uint8_t)w0;                                        \
                o[(GG) + i] = (uint8_t)w1;                                 \
                o[2u * (GG) + i] = (uint8_t)w2;                            \
                o[3u * (GG) + i] = (uint8_t)w3;                            \
                w0 = w1 = w2 = w3 = 0u;                                    \
            }                                                              \
        } break
        ZPSE_M3_PAIRS4(3, 0, 20);
        ZPSE_M3_PAIRS4(5, 1, 13);
        ZPSE_M3_PAIRS4(6, 2, 11);
        ZPSE_M3_PAIRS4(7, 3, 10);
        ZPSE_M3_PAIRS4(9, 4, 9);
        ZPSE_M3_PAIRS4(10, 5, 9);
        ZPSE_M3_PAIRS4(11, 6, 8);
        ZPSE_M3_PAIRS4(12, 7, 8);
        ZPSE_M3_PAIRS4(13, 8, 8);
        ZPSE_M3_PAIRS4(14, 9, 7);
        ZPSE_M3_PAIRS4(15, 10, 7);
#undef ZPSE_M3_PAIRS4
        default:
            return PSE_BAD_RESIDUAL;
        }
        /* Full groups: word < M^G. Same check per group. */
        if ((w0 | w1 | w2 | w3) != 0u) {
            return PSE_BAD_RESIDUAL;
        }
    }

    for (; j < ng; j++) {
        uint32_t word = zpse_load_u32le(payload + 1u + j * 4u);
        uint32_t cnt;
        uint32_t i;
        uint32_t rem = (uint32_t)k - j * (uint32_t)g;
        uint8_t *o = exc_out + j * (uint32_t)g;

        cnt = (rem >= g) ? g : rem;
        /* Same constant-naming argument as the batched loops above. */
        switch (m) {
#define ZPSE_M3_PAIRS(MM, GI)                                              \
        case (MM): {                                                       \
            const uint16_t *lut = m3_pair_lut + m3_pair_off[(GI)];         \
            for (i = 0u; i + 2u <= cnt; i += 2u) {                         \
                ZPSE_M3_PAIR(word, o + i, MM, lut);                        \
            }                                                              \
            if (i < cnt) {                                                 \
                bad |= (word >= (uint32_t)(MM));                           \
                o[i] = (uint8_t)word;                                      \
                word = 0u;                                                 \
            }                                                              \
        } break
        ZPSE_M3_PAIRS(3, 0);
        ZPSE_M3_PAIRS(5, 1);
        ZPSE_M3_PAIRS(6, 2);
        ZPSE_M3_PAIRS(7, 3);
        ZPSE_M3_PAIRS(9, 4);
        ZPSE_M3_PAIRS(10, 5);
        ZPSE_M3_PAIRS(11, 6);
        ZPSE_M3_PAIRS(12, 7);
        ZPSE_M3_PAIRS(13, 8);
        ZPSE_M3_PAIRS(14, 9);
        ZPSE_M3_PAIRS(15, 10);
#undef ZPSE_M3_PAIRS
        default:
            return PSE_BAD_RESIDUAL;
        }
        /* Full groups: word < M^G; tail: high part zero. Same check. */
        if (word != 0u) {
            return PSE_BAD_RESIDUAL;
        }
    }
    if (bad != 0u) {
        return PSE_BAD_RESIDUAL;
    }
    return PSE_OK;
}

static int residual_decode_mode4(const uint8_t *payload,
                                 uint16_t payload_len, uint16_t k,
                                 const uint8_t *pred_at_exc,
                                 uint8_t *exc_out)
{
    uint8_t sub;
    uint32_t expect;
    uint16_t i;

    if (payload_len < 1u) {
        return PSE_BAD_RESIDUAL;
    }
    sub = payload[0];
    if (sub > 2u) {
        return PSE_BAD_RESIDUAL;
    }
    if (sub == 0u) {
        expect = 1u + (((uint32_t)k + 1u) / 2u);
    } else if (sub == 1u) {
        expect = 1u + ((6u * (uint32_t)k + 7u) / 8u);
    } else {
        expect = 1u + (uint32_t)k;
    }
    if ((uint32_t)payload_len != expect) {
        return PSE_BAD_RESIDUAL;
    }
    if (k > 0u && (pred_at_exc == 0 || exc_out == 0)) {
        return PSE_BAD_ARG;
    }
    if (sub == 0u) {
        if ((k & 1u) != 0u) {
            if ((payload[1u + (k >> 1u)] >> 4u) != 0u) {
                return PSE_BAD_RESIDUAL;
            }
        }
        for (i = 0; i < k; i++) {
            uint8_t byte = payload[1u + (i >> 1u)];
            unsigned enc = ((i & 1u) == 0u) ? (byte & 0xFu) : (byte >> 4u);
            int d;
            uint8_t x;

            if (enc == 0u) {
                return PSE_BAD_RESIDUAL;
            }
            d = (enc < 8u) ? (int)enc : (int)enc - 16;
            x = (uint8_t)(((int)pred_at_exc[i] + d) & 0xFF);
            exc_out[i] = zpse_exclusion_forward(x, pred_at_exc[i]);
        }
    } else if (sub == 1u) {
        const uint8_t *s = payload + 1u;
        uint32_t total_bits = 6u * (uint32_t)k;

        if ((total_bits & 7u) != 0u && expect > 1u) {
            if ((s[expect - 2u] >> (total_bits & 7u)) != 0u) {
                return PSE_BAD_RESIDUAL;
            }
        }
        for (i = 0; i < k; i++) {
            unsigned enc = residual_get_bits(s, (uint32_t)i * 6u, 6u);
            int d;
            uint8_t x;

            if (enc == 0u) {
                return PSE_BAD_RESIDUAL;
            }
            d = (enc < 32u) ? (int)enc : (int)enc - 64;
            x = (uint8_t)(((int)pred_at_exc[i] + d) & 0xFF);
            exc_out[i] = zpse_exclusion_forward(x, pred_at_exc[i]);
        }
    } else {
        for (i = 0; i < k; i++) {
            uint8_t enc = payload[1u + i];
            int d;
            uint8_t x;

            if (enc == 0u) {
                return PSE_BAD_RESIDUAL;
            }
            d = (enc < 128u) ? (int)enc : (int)enc - 256;
            x = (uint8_t)(((int)pred_at_exc[i] + d) & 0xFF);
            exc_out[i] = zpse_exclusion_forward(x, pred_at_exc[i]);
        }
    }
    return PSE_OK;
}

int zpse_residual_decode(uint8_t mode, const uint8_t *payload,
                         uint16_t payload_len, uint16_t k,
                         const uint8_t *pred_at_exc,
                         uint8_t *exc_out, uint8_t *actual_out)
{
    (void)pred_at_exc;
    if (mode > 4u) {
        return PSE_BAD_RESIDUAL;
    }
    if (payload_len > 0u && payload == 0) {
        return PSE_BAD_ARG;
    }
    if (mode == ZPSE_R_RAW_ACTUAL) {
        if ((uint32_t)payload_len != (uint32_t)k) {
            return PSE_BAD_RESIDUAL;
        }
        if (k > 0u && (actual_out == 0 || payload == 0)) {
            return PSE_BAD_ARG;
        }
        if (k > 0u) {
            residual_copy_bytes(actual_out, payload, (uint32_t)k);
        }
        return PSE_OK;
    }
    if (k > 0u && exc_out == 0) {
        return PSE_BAD_ARG;
    }
    switch (mode) {
    case ZPSE_R_EXCLUSION_U8:
        if ((uint32_t)payload_len != (uint32_t)k) {
            return PSE_BAD_RESIDUAL;
        }
        if (k > 0u && payload == 0) {
            return PSE_BAD_ARG;
        }
        /* Every unit must be <= 254 -- 255 is not a legal exclusion
         * residual -- and `payload[i] > 254` is exactly `payload[i] ==
         * 255` for a byte, so the same rejection can be answered for eight
         * units at a time and the payload then copied. The status is
         * identical; the only difference is that on the rejecting path
         * exc_out may hold fewer units than the per-byte loop had already
         * written, which no caller can observe: every one of them returns
         * immediately on a non-OK status and exc_out (the tile's `resid[]`
         * scratch) is never read afterwards. */
        if (k > 0u) {
            if (residual_has_ff(payload, (uint32_t)k)) {
                return PSE_BAD_RESIDUAL;
            }
            residual_copy_bytes(exc_out, payload, (uint32_t)k);
        }
        return PSE_OK;
    case ZPSE_R_PALETTE_BITS:
        return residual_decode_mode2(payload, payload_len, k, exc_out);
    case ZPSE_R_PACKED_BASE_M:
        return residual_decode_mode3(payload, payload_len, k, exc_out);
    default: /* ZPSE_R_SIGNED_DELTA */
        return residual_decode_mode4(payload, payload_len, k, pred_at_exc,
                                     exc_out);
    }
}

#if ZPSE_RES_HAVE_SSSE3

/* Sixteen mode-2 palette indices per step, from the residual bytes
 * themselves, for the dense half of the palette: b == 4 means one nibble
 * per value, and `idx_of` is the value -> index table the scalar packer
 * reads one byte at a time.
 *
 * Two shuffles do sixteen lookups. The table is split into its two 16-byte
 * halves and pshufb addresses a half by the low nibble of each byte, so for
 * a residual below 32 the low half answers values 0..15 directly and the
 * high half answers 16..31 as idx_of[16 + (v & 15)] == idx_of[v] -- the
 * caller guarantees no residual reaches 32, which is what keeps both halves
 * in range (and pshufb's zeroing for a byte with bit 7 set can never be
 * triggered by a byte of 31 or less). The halves are selected by the high
 * nibble with a signed compare, exact for values below 128.
 *
 * Each pair of adjacent index bytes is then merged into the payload byte
 * that holds it: maddubs multiplies the even lane by 1 and the odd lane by
 * 16 and adds them, which is exactly (idx_2j | idx_2j+1 << 4) -- 255 at the
 * most, so the packing widening saturates nothing -- and sixteen values are
 * eight whole payload bytes, stored as one 64-bit lane. That is the same
 * byte the scalar loop writes for those two values, so the payload is
 * identical whichever path produced it.
 *
 * Values are consumed in ascending order in blocks of sixteen, always
 * starting at value 0, so what it consumes is a prefix of the tile and the
 * scalar loop can finish the rest unchanged. */
static ZPSE_RES_SSSE3 uint16_t residual_encode_mode2_simd(
    const uint8_t *idx_of, const uint8_t *exc_in, uint16_t k, uint8_t *dst)
{
    uint8_t lo[16];
    uint8_t hi[16];
    __m128i tlo;
    __m128i thi;
    __m128i mult;
    __m128i nibv;
    uint16_t i = 0u;

    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_SSSE3) == 0u) {
        return 0u;
    }
    memcpy(lo, idx_of, sizeof lo);
    memcpy(hi, idx_of + 16, sizeof hi);
    tlo = _mm_loadu_si128((const __m128i *)(const void *)lo);
    thi = _mm_loadu_si128((const __m128i *)(const void *)hi);
    mult = _mm_setr_epi8(1, 16, 1, 16, 1, 16, 1, 16,
                         1, 16, 1, 16, 1, 16, 1, 16);
    nibv = _mm_set1_epi8((char)0x0F);

    while ((uint32_t)i + 16u <= (uint32_t)k) {
        __m128i v = _mm_loadu_si128(
            (const __m128i *)(const void *)(exc_in + (uint32_t)i));
        __m128i sel = _mm_cmpgt_epi8(v, nibv);
        __m128i l = _mm_shuffle_epi8(tlo, v);
        __m128i h = _mm_shuffle_epi8(thi, v);
        __m128i idx = _mm_or_si128(_mm_and_si128(sel, h),
                                   _mm_andnot_si128(sel, l));
        __m128i merged = _mm_maddubs_epi16(idx, mult);
        __m128i packedv = _mm_packus_epi16(merged, merged);

        /* Sixteen 4-bit fields are eight whole bytes and the block starts
         * at value 0, so its byte offset is value >> 1. */
        _mm_storel_epi64((__m128i *)(void *)(dst + ((uint32_t)i >> 1u)),
                         packedv);
        i = (uint16_t)((uint32_t)i + 16u);
    }
    return i;
}

/* One bit per value (b == 1, i.e. a palette of exactly two entries).
 *
 * The bit for value i is "index i is 1", and with m == 2 and two distinct
 * entries that is exactly "the value is pal[1]": index 0 is the other
 * entry. So sixteen compares become sixteen bits through movemask -- the
 * byte-lane sign bits in lane order are the ascending values -- and
 * thirty-two values become one 32-bit store. The scalar loop this replaces
 * builds the same byte eight values at a time with a table lookup and
 * seven shifts each (~1.2 cycles a value); this is ~0.25.
 *
 * Only taken when nu == 2. A single-entry palette is padded to m == 2 with
 * pal[1] == 0, and a value of 0 would then compare equal to the padding
 * and be packed as index 1 instead of 0. */
static ZPSE_RES_SSSE3 uint16_t residual_encode_mode2_bits(
    const uint8_t *exc_in, uint16_t k, uint8_t p1, uint8_t *dst)
{
    const __m128i onev = _mm_set1_epi8((char)p1);
    uint16_t i = 0u;

    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_SSSE3) == 0u) {
        return 0u;
    }
    while ((uint32_t)i + 32u <= (uint32_t)k) {
        __m128i a = _mm_loadu_si128(
            (const __m128i *)(const void *)(exc_in + (uint32_t)i));
        __m128i b = _mm_loadu_si128(
            (const __m128i *)(const void *)(exc_in + (uint32_t)i + 16u));
        uint32_t m =
            (uint32_t)(unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(a, onev)) |
            ((uint32_t)(unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(b, onev))
             << 16u);

        /* Thirty-two 1-bit fields are four whole bytes, at value >> 3. */
        zpse_store_u32le(dst + ((uint32_t)i >> 3u), m);
        i = (uint16_t)((uint32_t)i + 32u);
    }
    return i;
}

/* Two bits per value (b == 2, a palette of three or four entries).
 *
 * Sixteen values are four packed bytes, and the merge is the 4-bit one from
 * residual_encode_mode2_simd followed by one horizontal add: maddubs with
 * (1, 4, 16, 64) puts values 4j and 4j+1 in the low nibble of a 16-bit lane
 * and values 4j+2 and 4j+3 in its high nibble, so lane pairs add up to
 * whole bytes, which is what hadd gives. Indices never exceed 3, so neither
 * maddubs (signed saturation) nor packus can clamp anything.
 *
 * The index lookup is the same two-shuffle select as the 4-bit path, so it
 * needs the same precondition: every residual below 32, which the caller
 * checks in `wide`. */
static ZPSE_RES_SSSE3 uint16_t residual_encode_mode2_b2(
    const uint8_t *idx_of, const uint8_t *exc_in, uint16_t k, uint8_t *dst)
{
    uint8_t lo[16];
    uint8_t hi[16];
    __m128i tlo;
    __m128i thi;
    __m128i nibv;
    __m128i mult;
    uint16_t i = 0u;

    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_SSSE3) == 0u) {
        return 0u;
    }
    memcpy(lo, idx_of, sizeof lo);
    memcpy(hi, idx_of + 16, sizeof hi);
    tlo = _mm_loadu_si128((const __m128i *)(const void *)lo);
    thi = _mm_loadu_si128((const __m128i *)(const void *)hi);
    nibv = _mm_set1_epi8((char)0x0F);
    mult = _mm_setr_epi8(1, 4, 16, 64, 1, 4, 16, 64,
                         1, 4, 16, 64, 1, 4, 16, 64);

    while ((uint32_t)i + 16u <= (uint32_t)k) {
        __m128i v = _mm_loadu_si128(
            (const __m128i *)(const void *)(exc_in + (uint32_t)i));
        __m128i sel = _mm_cmpgt_epi8(v, nibv);
        __m128i l = _mm_shuffle_epi8(tlo, v);
        __m128i h = _mm_shuffle_epi8(thi, v);
        __m128i idx = _mm_or_si128(_mm_and_si128(sel, h),
                                   _mm_andnot_si128(sel, l));
        __m128i pair = _mm_maddubs_epi16(idx, mult);
        __m128i bytev = _mm_hadd_epi16(pair, pair);
        __m128i packedv = _mm_packus_epi16(bytev, bytev);

        /* Four bytes per sixteen values, at value >> 2. */
        zpse_store_u32le(dst + ((uint32_t)i >> 2u),
                         (uint32_t)_mm_cvtsi128_si32(packedv));
        i = (uint16_t)((uint32_t)i + 16u);
    }
    return i;
}

#endif /* ZPSE_RES_HAVE_SSSE3 */

/* `nu_hint`/`seen_hint` are the classification's distinct-residual count
 * and set, when the caller has one: with them the packer can skip the
 * O(k) discovery scan below and find the first-appearance palette by
 * scanning only as far as the last NEW value (typically a few dozen
 * values instead of all k). They are a certificate, not a hint to trust:
 * the scan only takes the short path when the prefix really does hold
 * nu_hint distinct values AND every one of them is set in seen_hint, at
 * which point prefix == set (a subset of a set of the same size) and
 * every remaining value is known to be a palette entry. Anything else
 * falls back to the exhaustive scan, so a wrong certificate costs time,
 * never bytes. */
ZPSE_HOT static int residual_encode_mode2(const uint8_t *exc_in, uint16_t k,
                                 uint8_t *out, uint16_t cap,
                                 uint16_t nu_hint, const uint32_t *seen_hint)
{
    /* pal has one byte of slack: the membership test below cannot be a
     * branch (see the loop), so a repeat stores its dead "next slot"
     * unconditionally, and with sixteen distinct values already recorded
     * that slot is pal[16]. Every live entry is above that index, so the
     * spare byte is the only thing a repeat can ever write. */
    uint8_t pal[17];
    uint8_t idx_of[256];
    unsigned nu = 0;
    uint8_t m;
    unsigned b;
    uint32_t packed;
    uint32_t total;
    uint16_t i;
    unsigned s;
#if ZPSE_RES_HAVE_SSSE3
    int wide = 0; /* set when a residual reaches the high index-table half */
#endif

    /* Palette membership and value->index are both direct table lookups.
     * The obvious linear scans are O(k*M) with a data-dependent branch
     * per probe, and this packer is the encoder's single hottest
     * function for dense-residual tiles (a 256-byte tile with k~240 was
     * measured at ~4.5 us in pack alone, almost all of it here).
     *
     * idx_of[255] == 0xFF is the "absent" sentinel, which is safe
     * because residuals above 254 are rejected below. First-appearance
     * index is preserved, so the payload is byte-identical to what the
     * scan produced.
     *
     * The sentinel test itself is not a branch: a dense tile answers "not
     * new" for all but about sixteen of its ~240 values, so the branch is
     * unpredictable at exactly the positions that matter, while the select
     * has nothing to hide behind. The three stores are the same ones the
     * branchy form performs, only unconditional: idx_of[v] is written back
     * with the index it already had on a repeat, nu advances by is_new, and
     * pal[nu] holds the dead next slot on a repeat -- which is pal[16] once
     * sixteen values are recorded, above every live entry. */
    {
        int certified = 0;

        if (seen_hint != 0 && nu_hint <= 16u) {
            /* Cleared inside the branch, not before it: the exhaustive
             * scan below clears the table itself, so the call without a
             * hint -- which is most of them -- used to write 256 bytes
             * that the next memset immediately overwrote. Same table, same
             * first-appearance order, one clear per call instead of two. */
            memset(idx_of, 0xFF, sizeof idx_of);
            /* Prefix scan with an early exit: stop as soon as every value
             * the certificate promises has been seen. Positions are
             * visited in the same ascending order as the exhaustive scan,
             * so `pal` comes out in the same first-appearance order and
             * the payload is identical. The scan alone proves the prefix
             * holds nu_hint distinct values; the loop after it proves each
             * of them is one the classification saw, so prefix == set. A
             * bitmap marker was measured here and lost: the predictable
             * load-and-compare against 0xFF is cheaper than a read-modify-
             * write of the bit itself. */
            for (i = 0; i < k && nu < nu_hint; i++) {
                unsigned v = exc_in[i];

                if (idx_of[v] == 0xFFu) {
                    if (v > 254u) {
                        return PSE_BAD_RESIDUAL;
                    }
                    idx_of[v] = (uint8_t)nu;
                    pal[nu] = (uint8_t)v;
                    nu++;
                }
            }
            if (nu == nu_hint) {
                certified = 1;
                for (s = 0; s < nu; s++) {
                    uint8_t v = pal[s];

                    if (((seen_hint[v >> 5] >> (v & 31u)) & 1u) == 0u) {
                        certified = 0;
                        break;
                    }
                }
            }
            if (!certified) {
                /* Not reachable from a valid certificate; rebuild from
                 * scratch exactly as the exhaustive scan would. The scan's
                 * own clear is the one that counts -- clearing here as
                 * well only moved the cost, never the contents. */
                nu = 0;
            }
        }
        if (!certified) {
            memset(idx_of, 0xFF, sizeof idx_of);
            for (i = 0; i < k; i++) {
                unsigned v = exc_in[i];
                unsigned is_new;

                if (v > 254u) {
                    return PSE_BAD_RESIDUAL;
                }
                is_new = (idx_of[v] == 0xFFu) ? 1u : 0u;
                if (is_new != 0u && nu == 16u) {
                    return PSE_BAD_RESIDUAL;
                }
                idx_of[v] = (uint8_t)(is_new != 0u ? nu : (unsigned)idx_of[v]);
                pal[nu] = (uint8_t)(is_new != 0u ? v : 0u);
                nu += is_new;
            }
        }
    }
    m = (nu < 2u) ? 2u : (uint8_t)nu;
    for (s = nu; s < m; s++) {
        pal[s] = 0;
    }
    /* The wide pack below shuffles a 16-entry half of the value -> index
     * table with the residual byte itself, so it addresses the right half
     * only while every residual is below 32. The palette holds at most 16
     * values, so one short pass settles it. */
#if ZPSE_RES_HAVE_SSSE3
    for (s = 0; s < nu; s++) {
        if (pal[s] >= 32u) {
            wide = 1;
            break;
        }
    }
#endif
    b = zpse_ceil_log2_u32(m);
    packed = ((uint32_t)k * b + 7u) / 8u;
    total = 1u + (uint32_t)m + packed;
    if ((uint32_t)cap < total) {
        return PSE_OUTPUT_OVERFLOW;
    }
    if (total > 0u && out == 0) {
        return PSE_BAD_ARG;
    }
    out[0] = m;
    for (s = 0; s < m; s++) {
        out[1u + s] = pal[s];
    }
    /* Zero the packed region in one go: the wide path below stores whole
     * bytes over it and the tail only ORs bits into it. */
    memset(out + 1u + m, 0, (size_t)packed);

    /* Eight values are b bytes together (8*b bits) and every 8-value group
     * starts on a byte boundary (group g begins at bit 8*b*g), so a group
     * is exactly the little-endian store of the 8b-bit word
     * sum(idx_i << (i*b)). That is bit-for-bit what residual_put_bits
     * writes for those eight values: value i's b bits land at bit i*b of
     * a zeroed target, which is this word's bit range for the group. m is
     * at most 16, so b <= 4 and every index is below 2^b; the accumulator
     * and the largest shift (7*4 = 28 bits) stay inside 32 bits. */
    {
        uint8_t *dst = out + 1u + m;
        uint32_t g = 0u;
        uint16_t gi = 0u;

#if ZPSE_RES_HAVE_SSSE3
        /* The dense palette (a nibble per value, every residual below 32)
         * retires sixteen values per step; everything it does not consume,
         * including the tail of the block, is left to the byte loop. */
        if (b == 4u && wide == 0) {
            gi = residual_encode_mode2_simd(idx_of, exc_in, k, dst);
            g = (uint32_t)gi >> 1u;
        } else if (b == 1u && nu == 2u) {
            gi = residual_encode_mode2_bits(exc_in, k, pal[1], dst);
            g = (uint32_t)gi >> 3u;
        } else if (b == 2u && wide == 0) {
            gi = residual_encode_mode2_b2(idx_of, exc_in, k, dst);
            g = (uint32_t)gi >> 2u;
        }
#endif

        for (; (uint32_t)gi + 8u <= (uint32_t)k;
             gi = (uint16_t)(gi + 8u)) {
            uint32_t w = (uint32_t)idx_of[exc_in[gi]];
            unsigned j;

            for (j = 1u; j < 8u; j++) {
                w |= (uint32_t)idx_of[exc_in[gi + j]] << (j * b);
            }
            /* The group is b whole little-endian bytes of w, so the widest
             * store that fits writes it: 4, 2 or 1 byte instead of b of
             * them. b == 3 (a palette of 5..8 values) is the only width
             * with no such store and keeps its three bytes. */
            switch (b) {
            case 1:
                dst[g] = (uint8_t)w;
                break;
            case 2:
                zpse_store_u16le(dst + g, (uint16_t)w);
                break;
            case 4:
                zpse_store_u32le(dst + g, w);
                break;
            default:
                dst[g] = (uint8_t)w;
                dst[g + 1u] = (uint8_t)(w >> 8);
                dst[g + 2u] = (uint8_t)(w >> 16);
                break;
            }
            g += b;
        }
        /* Fewer than eight values left. The groups above consumed whole
         * bytes -- 8*b bits is b bytes and every group starts on a byte
         * boundary -- so this tail starts on a byte boundary as well. */
        for (; gi < k; gi++) {
            residual_put_bits(dst, (uint32_t)gi * b, b,
                              (unsigned)idx_of[exc_in[gi]]);
        }
    }
    return (int)total;
}

ZPSE_HOT static int residual_encode_mode3(const uint8_t *exc_in, uint16_t k,
                                 uint8_t *out, uint16_t cap)
{
    uint8_t m;
    uint8_t g;
    uint32_t ng;
    uint32_t total;
    uint32_t j;

    if (k == 0u) {
        m = 3;
    } else {
        uint8_t maxv = 0;
        uint16_t i;

        for (i = 0; i < k; i++) {
            if (exc_in[i] > maxv) {
                maxv = exc_in[i];
            }
        }
        m = residual_mode3_select_m(maxv);
        if (m == 0u) {
            return PSE_BAD_RESIDUAL;
        }
    }
    g = zpse_base_m_group(m);
    ng = ((uint32_t)k + (uint32_t)g - 1u) / (uint32_t)g;
    total = 1u + ng * 4u;
    if ((uint32_t)cap < total) {
        return PSE_OUTPUT_OVERFLOW;
    }
    if (total > 0u && out == 0) {
        return PSE_BAD_ARG;
    }
    out[0] = m;
    j = 0u;
    /* One multiply a digit, six groups at a time.
     *
     * The power-sum form below carries two multiplies per digit: one for
     * the digit's own contribution and one to advance the M^i weight, and
     * that second one is a serial chain of its own. Horner's rule --
     * word = (((d0*M + d1)*M + d2)*M + ...) -- produces the same integer
     * (every prefix is below M^G < 2^32, so nothing wraps) for a single
     * multiply per digit, and with M named as a literal at each arm the
     * compiler turns that multiply into a shift-add pair. Six groups' chains
     * run side by side, so the loop is bound by mul/shift throughput rather
     * than by a seven-deep dependency. Measured in isolation on the
     * palette16 residual stream (119 digits, M = 15): ~440 cycles for the
     * power-sum form, ~270 for this one, and the tile's own encode moved
     * by ~7% overall. The groups after the last whole batch -- five at
     * most, and the short final one -- keep the power sum. */
    {
        uint32_t nfull =
            (((uint32_t)k % (uint32_t)g) == 0u) ? ng : ng - 1u;

        switch (m) {
#define ZPSE_M3_ENC6(MM, GG)                                                \
        case (MM): {                                                        \
            while (j + 6u <= nfull) {                                       \
                const uint8_t *b = exc_in + j * (uint32_t)(GG);             \
                uint32_t w0 = 0u;                                           \
                uint32_t w1 = 0u;                                           \
                uint32_t w2 = 0u;                                           \
                uint32_t w3 = 0u;                                           \
                uint32_t w4 = 0u;                                           \
                uint32_t w5 = 0u;                                           \
                uint32_t i;                                                 \
                /* Most significant digit first: the field carries d_i at     \
                 * M^i, so the loop starts at the last byte of each group. */ \
                for (i = (GG); i-- > 0u;) {                                 \
                    w0 = w0 * (MM) + b[i];                                  \
                    w1 = w1 * (MM) + b[(GG) + i];                           \
                    w2 = w2 * (MM) + b[2u * (GG) + i];                      \
                    w3 = w3 * (MM) + b[3u * (GG) + i];                      \
                    w4 = w4 * (MM) + b[4u * (GG) + i];                      \
                    w5 = w5 * (MM) + b[5u * (GG) + i];                      \
                }                                                           \
                zpse_store_u32le(out + 1u + j * 4u, w0);                    \
                zpse_store_u32le(out + 5u + j * 4u, w1);                    \
                zpse_store_u32le(out + 9u + j * 4u, w2);                    \
                zpse_store_u32le(out + 13u + j * 4u, w3);                   \
                zpse_store_u32le(out + 17u + j * 4u, w4);                   \
                zpse_store_u32le(out + 21u + j * 4u, w5);                   \
                j += 6u;                                                    \
            }                                                               \
        } break
        ZPSE_M3_ENC6(3, 20);
        ZPSE_M3_ENC6(5, 13);
        ZPSE_M3_ENC6(6, 11);
        ZPSE_M3_ENC6(7, 10);
        ZPSE_M3_ENC6(9, 9);
        ZPSE_M3_ENC6(10, 9);
        ZPSE_M3_ENC6(11, 8);
        ZPSE_M3_ENC6(12, 8);
        ZPSE_M3_ENC6(13, 8);
        ZPSE_M3_ENC6(14, 7);
        ZPSE_M3_ENC6(15, 7);
#undef ZPSE_M3_ENC6
        default:
            break;
        }
    }
    for (; j < ng; j++) {
        uint32_t base = j * (uint32_t)g;
        uint32_t rem = (uint32_t)k - base;
        uint32_t cnt = (rem >= g) ? g : rem;
        uint32_t word = 0;
        uint32_t p = 1;
        uint32_t i;

        for (i = 0; i < cnt; i++) {
            word += (uint32_t)exc_in[base + i] * p;
            p *= (uint32_t)m;
        }
        zpse_store_u32le(out + 1u + j * 4u, word);
    }
    return (int)total;
}

ZPSE_HOT static int residual_encode_mode4(const uint8_t *actual_in,
                                 const uint8_t *pred_at_exc, uint16_t k,
                                 uint8_t *out, uint16_t cap)
{
    int dmin = 0;
    int dmax = 0;
    uint8_t sub;
    uint32_t total;
    uint16_t i;

    if (k == 0u) {
        if (cap < 1u) {
            return PSE_OUTPUT_OVERFLOW;
        }
        if (out == 0) {
            return PSE_BAD_ARG;
        }
        out[0] = 0;
        return 1;
    }
    if (actual_in == 0 || pred_at_exc == 0) {
        return PSE_BAD_ARG;
    }
    for (i = 0; i < k; i++) {
        int d = residual_delta(actual_in[i], pred_at_exc[i]);

        if (d == 0) {
            return PSE_BAD_RESIDUAL;
        }
        if (i == 0u) {
            dmin = d;
            dmax = d;
        } else {
            if (d < dmin) {
                dmin = d;
            }
            if (d > dmax) {
                dmax = d;
            }
        }
    }
    if (dmin >= -8 && dmax <= 7) {
        sub = 0;
        total = 1u + (((uint32_t)k + 1u) / 2u);
    } else if (dmin >= -32 && dmax <= 31) {
        sub = 1;
        total = 1u + ((6u * (uint32_t)k + 7u) / 8u);
    } else {
        sub = 2;
        total = 1u + (uint32_t)k;
    }
    if ((uint32_t)cap < total) {
        return PSE_OUTPUT_OVERFLOW;
    }
    if (out == 0) {
        return PSE_BAD_ARG;
    }
    out[0] = sub;
    if (sub == 0u) {
        uint32_t packed = ((uint32_t)k + 1u) / 2u;
        uint32_t g = 0u;
        uint16_t gi;

        memset(out + 1u, 0, (size_t)packed);
        /* Four-bit fields. Eight of them are exactly four bytes and a
         * group always starts on a byte boundary, so one little-endian
         * store of sum(enc_i << (4*i)) is what the nibble form below
         * wrote: value 2j in the low nibble of byte j, value 2j+1 in the
         * high one, which is the same bit positions. */
        for (gi = 0u; (uint32_t)gi + 8u <= (uint32_t)k;
             gi = (uint16_t)(gi + 8u)) {
            uint32_t w = 0u;
            unsigned j;

            for (j = 0; j < 8u; j++) {
                int d =
                    residual_delta(actual_in[gi + j], pred_at_exc[gi + j]);

                w |= (uint32_t)(uint8_t)(d & 0xF) << (4u * j);
            }
            for (j = 0; j < 4u; j++) {
                out[1u + g + j] = (uint8_t)(w >> (8u * j));
            }
            g += 4u;
        }
        for (; gi < k; gi++) {
            int d = residual_delta(actual_in[gi], pred_at_exc[gi]);
            uint8_t enc = (uint8_t)(d & 0xF);

            if ((gi & 1u) == 0u) {
                out[1u + (gi >> 1u)] = enc;
            } else {
                out[1u + (gi >> 1u)] |= (uint8_t)(enc << 4u);
            }
        }
    } else if (sub == 1u) {
        uint32_t packed = (6u * (uint32_t)k + 7u) / 8u;
        uint32_t g = 0u;
        uint16_t gi;

        memset(out + 1u, 0, (size_t)packed);
        /* Six-bit fields. Eight are 48 bits, again a whole number of bytes
         * per group, so a group is one 64-bit little-endian store and the
         * tail keeps the bit form (which also starts on a byte boundary,
         * because the groups consumed whole bytes). */
        for (gi = 0u; (uint32_t)gi + 8u <= (uint32_t)k;
             gi = (uint16_t)(gi + 8u)) {
            uint64_t w = 0u;
            unsigned j;

            for (j = 0; j < 8u; j++) {
                int d =
                    residual_delta(actual_in[gi + j], pred_at_exc[gi + j]);

                w |= (uint64_t)(uint8_t)(d & 0x3F) << (6u * j);
            }
            for (j = 0; j < 6u; j++) {
                out[1u + g + j] = (uint8_t)(w >> (8u * j));
            }
            g += 6u;
        }
        for (; gi < k; gi++) {
            int d = residual_delta(actual_in[gi], pred_at_exc[gi]);

            residual_put_bits(out + 1u, (uint32_t)gi * 6u, 6u,
                              (unsigned)(d & 0x3F));
        }
    } else {
        for (i = 0; i < k; i++) {
            int d = residual_delta(actual_in[i], pred_at_exc[i]);

            out[1u + i] = (uint8_t)(d & 0xFF);
        }
    }
    return (int)total;
}

ZPSE_HOT int zpse_residual_encode_hint(uint8_t mode, const uint8_t *exc_in,
                              const uint8_t *actual_in,
                              const uint8_t *pred_at_exc, uint16_t k,
                              uint8_t *payload_out, uint16_t payload_cap,
                              uint16_t nu_hint, const uint32_t *seen_hint)
{
    if (mode > 4u) {
        return PSE_BAD_RESIDUAL;
    }
    if (mode == ZPSE_R_RAW_ACTUAL) {
        if (k > 0u && (actual_in == 0 || payload_out == 0)) {
            return PSE_BAD_ARG;
        }
        if ((uint32_t)payload_cap < (uint32_t)k) {
            return PSE_OUTPUT_OVERFLOW;
        }
        if (k > 0u)
            memcpy(payload_out, actual_in, (size_t)k);
        return (int)k;
    }
    if (mode == ZPSE_R_EXCLUSION_U8) {
        if (k > 0u && (exc_in == 0 || payload_out == 0)) {
            return PSE_BAD_ARG;
        }
        if ((uint32_t)payload_cap < (uint32_t)k) {
            return PSE_OUTPUT_OVERFLOW;
        }
        if (k > 0u) {
            /* `> 254` on a byte is `== 255`; test eight at a time, then copy. */
            if (residual_has_ff(exc_in, (uint32_t)k)) {
                return PSE_BAD_RESIDUAL;
            }
            memcpy(payload_out, exc_in, (size_t)k);
        }
        return (int)k;
    }
    if (mode == ZPSE_R_SIGNED_DELTA) {
        return residual_encode_mode4(actual_in, pred_at_exc, k, payload_out,
                                     payload_cap);
    }
    if (k > 0u && exc_in == 0) {
        return PSE_BAD_ARG;
    }
    if (mode == ZPSE_R_PALETTE_BITS) {
        return residual_encode_mode2(exc_in, k, payload_out, payload_cap,
                                     nu_hint, seen_hint);
    }
    return residual_encode_mode3(exc_in, k, payload_out, payload_cap);
}

int zpse_residual_encode(uint8_t mode, const uint8_t *exc_in,
                         const uint8_t *actual_in,
                         const uint8_t *pred_at_exc, uint16_t k,
                         uint8_t *payload_out, uint16_t payload_cap)
{
    return zpse_residual_encode_hint(mode, exc_in, actual_in, pred_at_exc, k,
                                     payload_out, payload_cap, 0u, 0);
}

int zpse_residual_payload_len(uint8_t mode, uint16_t k,
                              const uint8_t *exc_in, uint16_t *len_out)
{
    if (len_out == 0) {
        return PSE_BAD_ARG;
    }
    switch (mode) {
    case ZPSE_R_EXCLUSION_U8:
    case ZPSE_R_RAW_ACTUAL:
        *len_out = k;
        return PSE_OK;
    case ZPSE_R_PALETTE_BITS: {
        uint8_t seen[8];
        unsigned nu = 0;
        uint8_t m;
        unsigned b;
        uint16_t i;

        if (k > 0u && exc_in == 0) {
            return PSE_BAD_ARG;
        }
        /* Membership as a 256-bit bitmap rather than a linear probe of the
         * values seen so far: the probe is O(k*nu) with a data-dependent
         * branch per value (~1900 probes for a 235-exception tile) and
         * this is the encoder's mode-search length estimator, so it runs
         * once per candidate mode per tile. v <= 254, so v >> 5 is in
         * 0..7 and the bitmap is eight bytes. The statuses are unchanged
         * and are still decided in ascending value order: a value above
         * 254 is rejected before it is tested for membership, and the
         * seventeenth distinct value is the one that is rejected. */
        memset(seen, 0, sizeof seen);
        for (i = 0; i < k; i++) {
            uint8_t v = exc_in[i];
            uint8_t bit;

            if (v > 254u) {
                return PSE_BAD_RESIDUAL;
            }
            bit = (uint8_t)(1u << (v & 31u));
            if ((seen[v >> 5u] & bit) != 0u) {
                continue;
            }
            if (nu == 16u) {
                return PSE_BAD_RESIDUAL;
            }
            seen[v >> 5u] |= bit;
            nu++;
        }
        m = (nu < 2u) ? 2u : (uint8_t)nu;
        b = zpse_ceil_log2_u32(m);
        *len_out = (uint16_t)(1u + (uint32_t)m +
                              (((uint32_t)k * b + 7u) / 8u));
        return PSE_OK;
    }
    case ZPSE_R_PACKED_BASE_M: {
        uint8_t m;
        uint8_t g;

        if (k > 0u && exc_in == 0) {
            return PSE_BAD_ARG;
        }
        if (k == 0u) {
            m = 3;
        } else {
            uint8_t maxv = 0;
            uint16_t i;

            for (i = 0; i < k; i++) {
                if (exc_in[i] > maxv) {
                    maxv = exc_in[i];
                }
            }
            m = residual_mode3_select_m(maxv);
            if (m == 0u) {
                return PSE_BAD_RESIDUAL;
            }
        }
        g = zpse_base_m_group(m);
        *len_out = (uint16_t)(1u + (((uint32_t)k + (uint32_t)g - 1u) /
                                    (uint32_t)g) * 4u);
        return PSE_OK;
    }
    case ZPSE_R_SIGNED_DELTA:
        /* Exact length needs actual+pred (submode selection); report the
         * 8-bit worst case for buffer sizing. */
        *len_out = (uint16_t)(1u + (uint32_t)k);
        return PSE_OK;
    default:
        return PSE_BAD_RESIDUAL;
    }
}
