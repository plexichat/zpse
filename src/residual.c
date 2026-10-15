#include "zpse/residual.h"
#include "zpse/bits.h"
#include "zpse/status.h"
#include "zpse/vector.h"

#include <stddef.h>
#include <stdint.h>

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
#define ZPSE_RES_SSSE3 __attribute__((target("ssse3")))
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

#if ZPSE_RES_HAVE_SSSE3
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
static void residual_copy_bytes(uint8_t *dst, const uint8_t *src, uint32_t n)
{
    uint32_t i = 0;

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
    if ((zpse_cpu_caps() & (uint32_t)ZPSE_CPU_SSSE3) == 0u) {
        return 0u;
    }
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

static int residual_decode_mode3(const uint8_t *payload,
                                 uint16_t payload_len, uint16_t k,
                                 uint8_t *exc_out)
{
    uint8_t m;
    uint8_t g;
    uint32_t ng;
    uint32_t j;

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
    for (j = 0; j < ng; j++) {
        uint32_t word = zpse_load_u32le(payload + 1u + j * 4u);
        uint32_t cnt;
        uint32_t i;
        uint32_t rem = (uint32_t)k - j * (uint32_t)g;

        cnt = (rem >= g) ? g : rem;
        /* M is one of eleven fixed constants. Naming it at the point of
         * use lets the compiler replace the hardware integer divide
         * (one per unit, ~20+ cycles) with a multiply-shift pair, which
         * is the single largest cost in decoding a group of dense
         * base-M residuals. */
        switch (m) {
#define ZPSE_M3_DIGITS(MM)                                                   \
        case (MM):                                                           \
            for (i = 0; i < cnt; i++) {                                      \
                exc_out[j * (uint32_t)g + i] = (uint8_t)(word % (MM));        \
                word /= (MM);                                                \
            }                                                                \
            break
        ZPSE_M3_DIGITS(3);
        ZPSE_M3_DIGITS(5);
        ZPSE_M3_DIGITS(6);
        ZPSE_M3_DIGITS(7);
        ZPSE_M3_DIGITS(9);
        ZPSE_M3_DIGITS(10);
        ZPSE_M3_DIGITS(11);
        ZPSE_M3_DIGITS(12);
        ZPSE_M3_DIGITS(13);
        ZPSE_M3_DIGITS(14);
        ZPSE_M3_DIGITS(15);
#undef ZPSE_M3_DIGITS
        default:
            return PSE_BAD_RESIDUAL;
        }
        /* Full groups: word < M^G; tail: high part zero. Same check. */
        if (word != 0u) {
            return PSE_BAD_RESIDUAL;
        }
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

static int residual_encode_mode2(const uint8_t *exc_in, uint16_t k,
                                 uint8_t *out, uint16_t cap)
{
    uint8_t pal[16];
    uint8_t idx_of[256];
    unsigned nu = 0;
    uint8_t m;
    unsigned b;
    uint32_t packed;
    uint32_t total;
    uint16_t i;
    unsigned s;

    /* Palette membership and value->index are both direct table lookups.
     * The obvious linear scans are O(k*M) with a data-dependent branch
     * per probe, and this packer is the encoder's single hottest
     * function for dense-residual tiles (a 256-byte tile with k~240 was
     * measured at ~4.5 us in pack alone, almost all of it here).
     *
     * idx_of[255] == 0xFF is the "absent" sentinel, which is safe
     * because residuals above 254 are rejected below. First-appearance
     * index is preserved, so the payload is byte-identical to what the
     * scan produced. */
    memset(idx_of, 0xFF, sizeof idx_of);
    for (i = 0; i < k; i++) {
        uint8_t v = exc_in[i];

        if (v > 254u) {
            return PSE_BAD_RESIDUAL;
        }
        if (idx_of[v] == 0xFFu) {
            if (nu == 16u) {
                return PSE_BAD_RESIDUAL;
            }
            idx_of[v] = (uint8_t)nu;
            pal[nu++] = v;
        }
    }
    m = (nu < 2u) ? 2u : (uint8_t)nu;
    for (s = nu; s < m; s++) {
        pal[s] = 0;
    }
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
    for (s = 0; s < packed; s++) {
        out[1u + m + s] = 0;
    }
    for (i = 0; i < k; i++) {
        residual_put_bits(out + 1u + m, (uint32_t)i * b, b,
                          (unsigned)idx_of[exc_in[i]]);
    }
    return (int)total;
}

static int residual_encode_mode3(const uint8_t *exc_in, uint16_t k,
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
    for (j = 0; j < ng; j++) {
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

static int residual_encode_mode4(const uint8_t *actual_in,
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
        uint32_t s;

        for (s = 0; s < packed; s++) {
            out[1u + s] = 0;
        }
        for (i = 0; i < k; i++) {
            int d = residual_delta(actual_in[i], pred_at_exc[i]);
            uint8_t enc = (uint8_t)(d & 0xF);

            if ((i & 1u) == 0u) {
                out[1u + (i >> 1u)] = (uint8_t)enc;
            } else {
                out[1u + (i >> 1u)] |= (uint8_t)(enc << 4u);
            }
        }
    } else if (sub == 1u) {
        uint32_t packed = (6u * (uint32_t)k + 7u) / 8u;
        uint32_t s;

        for (s = 0; s < packed; s++) {
            out[1u + s] = 0;
        }
        for (i = 0; i < k; i++) {
            int d = residual_delta(actual_in[i], pred_at_exc[i]);

            residual_put_bits(out + 1u, (uint32_t)i * 6u, 6u,
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

int zpse_residual_encode(uint8_t mode, const uint8_t *exc_in,
                         const uint8_t *actual_in,
                         const uint8_t *pred_at_exc, uint16_t k,
                         uint8_t *payload_out, uint16_t payload_cap)
{
    uint16_t i;

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
        for (i = 0; i < k; i++) {
            payload_out[i] = actual_in[i];
        }
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
        return residual_encode_mode2(exc_in, k, payload_out, payload_cap);
    }
    return residual_encode_mode3(exc_in, k, payload_out, payload_cap);
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
        uint8_t seen[16];
        unsigned nu = 0;
        uint8_t m;
        unsigned b;
        uint16_t i;
        unsigned s;

        if (k > 0u && exc_in == 0) {
            return PSE_BAD_ARG;
        }
        for (i = 0; i < k; i++) {
            unsigned found = 0;
            uint8_t v = exc_in[i];

            if (v > 254u) {
                return PSE_BAD_RESIDUAL;
            }
            for (s = 0; s < nu; s++) {
                if (seen[s] == v) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (nu == 16u) {
                    return PSE_BAD_RESIDUAL;
                }
                seen[nu++] = v;
            }
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
