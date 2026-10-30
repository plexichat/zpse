#include "zpse/dense.h"

#include <string.h>

#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define ZPSE_DENSE_HAVE_SSE2 1
#include <emmintrin.h>
#endif

#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/mask.h"
#include "zpse/predictor.h"
#include "zpse/residual.h"
#include "zpse/status.h"
#include "zpse/vector.h"

/* 256-bit kernels for the two whole-record passes below (the exclusion
 * inverse of a dense record's residual stream). Same dispatch shape as
 * src/transform.c and src/residual.c: the per-function target attribute
 * lets a translation unit built at the baseline ISA still compile them,
 * and ZPSE_AVX2_ABI_OK keeps them out of builds whose ABI cannot pass a
 * 32-byte vector (see zpse/hot.h), where the SSE2 loop is what runs. */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_DENSE_AVX2 __attribute__((target("avx2")))
#define ZPSE_DENSE_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_DENSE_AVX2
#define ZPSE_DENSE_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_DENSE_HAVE_AVX2 0
#endif
#else
#define ZPSE_DENSE_HAVE_AVX2 0
#endif

#if ZPSE_DENSE_HAVE_AVX2
#include <immintrin.h>
#endif

/* out[0..n) += 1 at 32 bytes per step. The whole loop is one instruction
 * per 32 bytes either way -- PADDB -- so the win is simply half the
 * iterations, and it is taken for the same reason as every other 256-bit
 * kernel here: a dense record is up to 4 KiB and this pass is the entire
 * decode of a constant plane that does not sit on 255. */
#if ZPSE_DENSE_HAVE_AVX2
static ZPSE_DENSE_AVX2 void dense_add_one_avx2(uint8_t *out, uint32_t n)
{
    uint32_t i = 0u;
    const __m256i one = _mm256_set1_epi8(1);

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(out + i));

        _mm256_storeu_si256((__m256i *)(void *)(out + i),
                            _mm256_add_epi8(v, one));
    }
    for (; i < n; i++) {
        out[i] = (uint8_t)(out[i] + 1u);
    }
}

/* out[0..n) = zpse_exclusion_inverse(out[i], fill), 32 bytes per step. The
 * compare is unsigned and the result is r + (r >= fill), so the byte mask
 * of `max(r, fill) == r` IS the increment: one PADD of a compare mask, no
 * branch, and VPMINUB/VPCMPEQB are both single-uop on AVX2 just as their
 * 128-bit forms are. */
static ZPSE_DENSE_AVX2 void dense_exclusion_map_avx2(uint8_t *out, uint32_t n,
                                                     uint8_t fill)
{
    uint32_t i = 0u;
    const __m256i f = _mm256_set1_epi8((char)fill);
    const __m256i one = _mm256_set1_epi8(1);

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(out + i));
        __m256i ge = _mm256_cmpeq_epi8(_mm256_max_epu8(v, f), v);

        _mm256_storeu_si256((__m256i *)(void *)(out + i),
                            _mm256_add_epi8(v, _mm256_and_si256(ge, one)));
    }
    for (; i < n; i++) {
        out[i] = zpse_exclusion_inverse(out[i], fill);
    }
}
#endif /* ZPSE_DENSE_HAVE_AVX2 */

/* out[0..n) += 1, byte-wise with wrap, which is the exclusion inverse for
 * a zero prediction (zpse_exclusion_inverse(r, 0) == r + 1 for every r).
 * Byte addition cannot be carried out with a word add -- a carry out of
 * one byte would corrupt the next -- so the vector form adds with
 * PADDB, one byte lane at a time. */
static void dense_add_one(uint8_t *out, uint32_t n)
{
    uint32_t i = 0u;

#if ZPSE_DENSE_HAVE_AVX2
    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        dense_add_one_avx2(out, n);
        return;
    }
#endif
#if ZPSE_DENSE_HAVE_SSE2
    {
        const __m128i one = _mm_set1_epi8(1);

        for (; i + 16u <= n; i += 16u) {
            __m128i v = _mm_loadu_si128(
                (const __m128i *)(const void *)(out + i));

            _mm_storeu_si128((__m128i *)(void *)(out + i),
                             _mm_add_epi8(v, one));
        }
    }
#endif
    for (; i < n; i++) {
        out[i] = (uint8_t)(out[i] + 1u);
    }
}

/* out[0..n) = zpse_exclusion_inverse(out[i], fill), for a fill the two
 * cases above do not cover (0 is dense_add_one, 255 is the identity).
 * The map is r + (r >= fill) with an unsigned compare, so 16 bytes at a
 * time: PADDB of the compare's byte mask. This is the pass that used to
 * be a per-byte scalar loop over the whole 4 KiB record, and on a plane
 * of sixteen values that happens to include 255 (so the encoder's
 * preferred no-op fill is not available) it was the whole cost of
 * decoding a dense record. */
static void dense_exclusion_map(uint8_t *out, uint32_t n, uint8_t fill)
{
    uint32_t i = 0u;

#if ZPSE_DENSE_HAVE_AVX2
    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        dense_exclusion_map_avx2(out, n, fill);
        return;
    }
#endif
#if ZPSE_DENSE_HAVE_SSE2
    {
        const __m128i f = _mm_set1_epi8((char)fill);
        const __m128i one = _mm_set1_epi8(1);

        for (; i + 16u <= n; i += 16u) {
            __m128i v = _mm_loadu_si128(
                (const __m128i *)(const void *)(out + i));
            __m128i ge = _mm_cmpeq_epi8(_mm_max_epu8(v, f), v);

            _mm_storeu_si128((__m128i *)(void *)(out + i),
                             _mm_add_epi8(v, _mm_and_si128(ge, one)));
        }
    }
#endif
    for (; i < n; i++) {
        out[i] = zpse_exclusion_inverse(out[i], fill);
    }
}

/* Period of a STRIDE predictor (mirrors tile_stride_of in tile.c). */
static uint32_t dense_stride(uint8_t predictor_id)
{
    switch (predictor_id) {
    case ZPSE_P_STRIDE_1:
        return 1u;
    case ZPSE_P_STRIDE_2:
        return 2u;
    case ZPSE_P_STRIDE_3:
        return 3u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u; /* ZPSE_P_STRIDE_8 */
    }
}

static unsigned dense_ctz(unsigned w)
{
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_ctz(w);
#else
    unsigned n = 0u;

    while ((w & 1u) == 0u) {
        w >>= 1;
        n++;
    }
    return n;
#endif
}

/* Number of set bits in a dense bitmap of `need` bytes.
 *
 * SWAR over four bytes at a time rather than a bit-scan per byte: a
 * 4 KiB record's bitmap is 512 bytes and half of them are typically set,
 * so the bit loop was ~1500 dependent-friendly iterations of a branch
 * and a clear-lowest-bit per unit of validation. Same count, no
 * data-dependent branch. */
static uint32_t dense_popcount(const uint8_t *bits, uint32_t need)
{
    uint32_t n = 0u;
    uint32_t i = 0u;

    for (; i + 4u <= need; i += 4u) {
        uint32_t w;

        memcpy(&w, bits + i, sizeof w);
        w = w - ((w >> 1) & UINT32_C(0x55555555));
        w = (w & UINT32_C(0x33333333)) +
            ((w >> 2) & UINT32_C(0x33333333));
        n += (((w + (w >> 4)) & UINT32_C(0x0F0F0F0F)) *
              UINT32_C(0x01010101)) >> 24;
    }
    for (; i < need; i++) {
        unsigned b = bits[i];

        while (b != 0u) {
            b &= b - 1u;
            n++;
        }
    }
    return n;
}

/* First exception position >= `from`, or nt when the mask has none left.
 * Positions only ever move forward, so this scan is entered once per
 * exception and costs O(nt/8 + K) over the whole record. */
static uint32_t dense_bitmap_next(const uint8_t *bits, uint32_t nt,
                                  uint32_t from)
{
    uint32_t p = from;

    while (p < nt) {
        uint32_t byte_i = p >> 3;
        unsigned w = (unsigned)bits[byte_i] >> (p & 7u);

        if (w != 0u) {
            return p + (uint32_t)dense_ctz(w);
        }
        p = (byte_i + 1u) << 3;
    }
    return nt;
}

/* Position of exception j, given the mask. */
static uint32_t dense_next(uint8_t mask_mode, const uint8_t *mask,
                           uint32_t nt, uint32_t j, uint32_t from)
{
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        return nt;
    case ZPSE_M_ALL_EXCEPTION:
        return j;
    case ZPSE_M_SPARSE_U8:
        return mask[j];
    default: /* ZPSE_M_RAW_BITMAP */
        return dense_bitmap_next(mask, nt, from);
    }
}

/* Actual byte at exception j: raw for mode 1, un-excluded for the rest. */
static uint8_t dense_exc_value(const uint8_t *resid, uint8_t res_mode,
                               uint32_t j, uint8_t xhat)
{
    if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
        return resid[j];
    }
    return zpse_exclusion_inverse(resid[j], xhat);
}

static int dense_check_mask(uint8_t mask_mode, const uint8_t *mask,
                            uint32_t nt, uint32_t k, uint32_t mask_len)
{
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (k != 0u || mask_len != 0u) {
            return PSE_BAD_MASK;
        }
        return PSE_OK;
    case ZPSE_M_ALL_EXCEPTION:
        if (k != nt || mask_len != 0u) {
            return PSE_BAD_MASK;
        }
        return PSE_OK;
    case ZPSE_M_RAW_BITMAP: {
        uint32_t need = (nt + 7u) >> 3;

        if (k < 1u || k >= nt) {
            return PSE_BAD_MASK;
        }
        if (mask_len != need) {
            return PSE_BAD_MASK;
        }
        if ((nt & 7u) != 0u && (mask[need - 1u] >> (nt & 7u)) != 0u) {
            return PSE_BAD_MASK;
        }
        if (dense_popcount(mask, need) != k) {
            return PSE_BAD_MASK;
        }
        return PSE_OK;
    }
    case ZPSE_M_SPARSE_U8: {
        uint32_t j;

        /* Positions are one byte each, so this form only exists for
         * records whose positions all fit in a byte. */
        if (nt > 256u) {
            return PSE_BAD_MASK;
        }
        if (k < 1u || k >= nt) {
            return PSE_BAD_MASK;
        }
        if (mask_len != k) {
            return PSE_BAD_MASK;
        }
        for (j = 0u; j < k; j++) {
            if ((uint32_t)mask[j] >= nt) {
                return PSE_BAD_MASK;
            }
            if (j > 0u && mask[j] <= mask[j - 1u]) {
                return PSE_BAD_MASK;
            }
        }
        return PSE_OK;
    }
    default:
        /* Mask modes 3, 4, 5 and 7 are reserved inside a dense record. */
        return PSE_UNSUPPORTED_FEATURE;
    }
}

/* One record at payload[base..], coding exactly nt (<= ZPSE_DENSE_MAX_LEN)
 * bytes into out. `literal` is the segment's initial literal, which the
 * record's entry state and STRIDE fallbacks use. */
ZPSE_HOT static int dense_record(uint32_t nt, const uint8_t *payload,
                        uint32_t payload_len, uint32_t base, uint8_t *out,
                        uint8_t *scratch, uint8_t literal, int *is_const,
                        uint8_t *fill_out)
{
    uint32_t k;
    uint32_t mask_len;
    uint32_t res_len;
    uint32_t rec_len;
    uint8_t flags;
    uint8_t mask_mode;
    uint8_t res_mode;
    uint8_t pred;
    uint8_t constant_c;
    const uint8_t *mask;
    const uint8_t *res;
    uint8_t last[8];
    uint8_t second[8];
    uint32_t p;
    uint32_t j;
    uint32_t next;
    int rc;
    unsigned l;

    *is_const = 0;
    if (nt == 0u || nt > ZPSE_DENSE_MAX_LEN) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (payload_len < base || payload_len - base < ZPSE_DENSE_HEADER_SIZE) {
        return PSE_BAD_TILE_SIZE;
    }

    rec_len = (uint32_t)zpse_load_u16le(payload + base);
    flags = payload[base + 2u];
    pred = payload[base + 3u];
    constant_c = payload[base + 4u];
    k = (uint32_t)zpse_load_u16le(payload + base + 6u);
    mask_len = (uint32_t)zpse_load_u16le(payload + base + 8u);
    res_len = (uint32_t)zpse_load_u16le(payload + base + 10u);
    mask_mode = (uint8_t)(flags & UINT8_C(0x07));
    res_mode = (uint8_t)((flags >> 3) & UINT8_C(0x07));

    if ((flags & ZPSE_DENSE_MARKER) == 0u) {
        return PSE_BAD_TILE_SIZE;
    }
    if ((flags & UINT8_C(0x80)) != 0u) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (payload[base + 5u] != 0u) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (rec_len < 10u ||
        (uint64_t)base + 2u + (uint64_t)rec_len > (uint64_t)payload_len) {
        return PSE_BAD_TILE_SIZE;
    }
    if (mask_len + res_len != rec_len - 10u) {
        return PSE_BAD_TILE_SIZE;
    }
    /* Mode 4 (signed delta) is prediction-dependent, and a record only
     * knows a position's prediction while reconstructing it, so it is
     * reserved here and reported as an unsupported feature rather than
     * decoded wrongly. */
    if (res_mode > (uint8_t)ZPSE_R_PACKED_BASE_M) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (pred > (uint8_t)ZPSE_P_STRIDE_3) {
        return PSE_BAD_PREDICTOR;
    }
    if (pred == (uint8_t)ZPSE_P_ORDER1 || pred == (uint8_t)ZPSE_P_ORDER2) {
        return PSE_UNSUPPORTED_FEATURE;
    }

    mask = payload + base + ZPSE_DENSE_HEADER_SIZE;
    res = mask + mask_len;
    rc = dense_check_mask(mask_mode, mask, nt, k, mask_len);
    if (rc != PSE_OK) {
        return rc;
    }

    /* ---- No prediction at all (Sec. 6.3) ------------------------- *
     * An ALL_EXCEPTION mask makes every position an exception, and the
     * constant families predict the same byte everywhere, so no
     * prediction chain exists to follow: the record IS its residual
     * stream, expanded and mapped through the (constant) exclusion
     * inverse. That is the shape the encoder emits for a plane with few
     * distinct byte values (segment.c), and decoding it straight into the
     * output -- rather than into the exception scratch and back out
     * through a per-position loop -- is what makes such a plane cost one
     * palette expansion instead of one predictor step per byte. The
     * residual decoder validates the payload exactly as it does on the
     * general path, so every rejection is unchanged; validation of K and
     * of the mask already happened above. */
    if (mask_mode == (uint8_t)ZPSE_M_ALL_EXCEPTION &&
        (pred == (uint8_t)ZPSE_P_CONST_ZERO ||
         pred == (uint8_t)ZPSE_P_TILE_CONSTANT)) {
        uint8_t fill = (pred == (uint8_t)ZPSE_P_CONST_ZERO) ? 0u : constant_c;

        rc = zpse_residual_decode(res_mode, res, (uint16_t)res_len,
                                  (uint16_t)k, 0, out, out);
        if (rc != PSE_OK) {
            return rc;
        }
        if (res_mode != (uint8_t)ZPSE_R_RAW_ACTUAL) {
            if (fill == 0u) {
                dense_add_one(out, nt);
            } else if (fill == 255u) {
                /* zpse_exclusion_inverse(r, 255) is r for every residual a
                 * payload can hold (255 itself is not a legal residual), and
                 * the encoder picks 255 exactly when the plane does not
                 * contain it, so the expansion IS the reconstruction. */
            } else {
                /* Any other prediction is not a constant offset, so this
                 * takes the byte test -- vectorised, over the whole
                 * record, since it is the only pass the form has left. */
                dense_exclusion_map(out, nt, fill);
            }
        }
        return PSE_OK;
    }

    rc = zpse_residual_decode(res_mode, res, (uint16_t)res_len, (uint16_t)k,
                              0, scratch, scratch);
    if (rc != PSE_OK) {
        return rc;
    }

    /* A record with no exceptions is pure prediction, and every
     * predictor's fixed point from the entry state is the literal: CONST
     * gives 0 or C, the lane histories stay at the literal (2L - L = L,
     * (L + L + 1) >> 1 = L) and STRIDE reads the literal it just wrote.
     * So the whole record is one fill, which is what makes the form cheap
     * to decode as well as to encode. */
    if (k == 0u) {
        uint8_t fill = (pred == (uint8_t)ZPSE_P_CONST_ZERO)
                           ? 0u
                           : (pred == (uint8_t)ZPSE_P_TILE_CONSTANT)
                                 ? constant_c
                                 : literal;

        memset(out, fill, (size_t)nt);
        /* This record reconstructed one byte value end to end; the
         * block layer turns a whole plane of these into arithmetic
         * instead of a CRC pass (crc32c.h: zpse_crc32c_repeat). */
        *is_const = 1;
        *fill_out = fill;
        return PSE_OK;
    }

    for (l = 0; l < 8u; l++) {
        last[l] = literal;
        second[l] = literal;
    }
    j = 0u;
    next = dense_next(mask_mode, mask, nt, 0u, 0u);

    if (pred == (uint8_t)ZPSE_P_CONST_ZERO ||
        pred == (uint8_t)ZPSE_P_TILE_CONSTANT) {
        uint8_t fill = (pred == (uint8_t)ZPSE_P_CONST_ZERO) ? 0u : constant_c;

        for (p = 0u; p < nt; p++) {
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, fill);
                j++;
                next = (j < k)
                           ? dense_next(mask_mode, mask, nt, j, p + 1u)
                           : nt;
            } else {
                x = fill;
            }
            out[p] = x;
        }
    } else if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
        for (p = 0u; p < nt; p++) {
            unsigned lane = p & 7u;
            uint8_t xhat = last[lane];
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k)
                           ? dense_next(mask_mode, mask, nt, j, p + 1u)
                           : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
            second[lane] = xhat;
            last[lane] = x;
        }
    } else if (pred == (uint8_t)ZPSE_P_LANE_DELTA) {
        for (p = 0u; p < nt; p++) {
            unsigned lane = p & 7u;
            uint8_t prev = last[lane];
            uint8_t xhat =
                (uint8_t)((2u * (unsigned)prev - (unsigned)second[lane]) &
                          UINT8_C(0xFF));
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k)
                           ? dense_next(mask_mode, mask, nt, j, p + 1u)
                           : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
            second[lane] = prev;
            last[lane] = x;
        }
    } else if (pred == (uint8_t)ZPSE_P_LANE_AVERAGE) {
        for (p = 0u; p < nt; p++) {
            unsigned lane = p & 7u;
            uint8_t prev = last[lane];
            uint8_t xhat =
                (uint8_t)(((unsigned)prev + (unsigned)second[lane] + 1u) >> 1);
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k)
                           ? dense_next(mask_mode, mask, nt, j, p + 1u)
                           : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
            second[lane] = prev;
            last[lane] = x;
        }
    } else {
        uint32_t s = dense_stride(pred);

        for (p = 0u; p < nt; p++) {
            uint8_t xhat = (p < s) ? literal : out[p - s];
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k)
                           ? dense_next(mask_mode, mask, nt, j, p + 1u)
                           : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
        }
    }

    if (j != k) {
        return PSE_BAD_MASK;
    }
    return PSE_OK;
}

int zpse_dense_decode(const ZpseSegmentHeader *h, const uint8_t *payload,
                      uint32_t payload_len, uint8_t *out, uint32_t out_cap,
                      uint32_t *out_len, uint8_t *scratch, int *all_const,
                      uint8_t *const_byte)
{
    uint32_t total;
    uint32_t produced;
    uint32_t pos;
    int one_value = 1;
    uint8_t fill = 0u;

    if (h == 0 || payload == 0 || out == 0 || out_len == 0 || scratch == 0 ||
        all_const == 0 || const_byte == 0 || h->tile_count != 1u) {
        return PSE_BAD_ARG;
    }
    *all_const = 0;
    *const_byte = 0u;
    total = h->segment_decoded_size;
    if (total == 0u || total > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (total > out_cap) {
        return PSE_OUTPUT_OVERFLOW;
    }

    produced = 0u;
    pos = ZPSE_SEGMENT_HEADER_SIZE;
    while (produced < total) {
        uint32_t ns = total - produced;
        uint32_t rec_len;
        int rc;

        if (ns > ZPSE_DENSE_MAX_LEN) {
            ns = ZPSE_DENSE_MAX_LEN;
        }
        if (pos + 2u > payload_len) {
            return PSE_BAD_TILE_SIZE;
        }
        rec_len = (uint32_t)zpse_load_u16le(payload + pos);
        {
            int rec_const = 0;
            uint8_t rec_fill = 0u;

            rc = dense_record(ns, payload, payload_len, pos, out + produced,
                              scratch, h->initial_literal, &rec_const,
                              &rec_fill);
            if (rc != PSE_OK) {
                return rc;
            }
            if (rec_const == 0 || (one_value != 0 && produced != 0u &&
                                   rec_fill != fill)) {
                one_value = 0;
            } else if (produced == 0u) {
                fill = rec_fill;
            }
        }
        pos += 2u + rec_len;
        if (pos > payload_len) {
            return PSE_BAD_TILE_SIZE;
        }
        produced += ns;
    }
    if (pos != payload_len) {
        return PSE_BAD_TILE_SIZE;
    }
    *out_len = total;
    *all_const = one_value;
    *const_byte = fill;
    return PSE_OK;
}
