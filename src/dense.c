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

/* 256-bit kernels for the whole-record passes below. Same dispatch shape as
 * src/transform.c and src/residual.c: the per-function target attribute lets
 * a translation unit built at the baseline ISA still compile them, and
 * ZPSE_AVX2_ABI_OK keeps them out of builds whose ABI cannot pass a 32-byte
 * vector (see zpse/hot.h), where the SSE2/scalar loop is what runs.
 * aligned(64) keeps each kernel's body inside one pair of I-cache lines: a
 * kernel is entered once per record but then runs for its whole 4 KiB. */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_DENSE_AVX2_TARGET __attribute__((target("avx2"), aligned(64)))
#define ZPSE_DENSE_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_DENSE_AVX2_TARGET __attribute__((aligned(64)))
#define ZPSE_DENSE_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_DENSE_HAVE_AVX2 0
#endif
#else
#define ZPSE_DENSE_HAVE_AVX2 0
#endif

/* Master switch for the new vectorized whole-record reconstruction kernels.
 * Default on; define to 0 for a scalar-only build. The scalar loops in
 * dense_record remain in place and are bit-identical (that is what
 * tests/test_parity.c pins), so either setting decodes the same stream. */
#ifndef ZPSE_DENSE_AVX2
#define ZPSE_DENSE_AVX2 1
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
static ZPSE_DENSE_AVX2_TARGET void dense_add_one_avx2(uint8_t *out, uint32_t n)
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
static ZPSE_DENSE_AVX2_TARGET void dense_exclusion_map_avx2(uint8_t *out,
                                                             uint32_t n,
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

/* Store `v` at p[0..n) without a per-byte exception test: a record's
 * prediction runs are what the position loop used to walk one byte at a
 * time, and a run of a few hundred bytes is a memset. The short form stays
 * a byte loop -- a call to memset for five bytes costs more than the five
 * stores it replaces, and a whole-record store would have to be bounded by
 * the record, whose end is exactly where writing past the last run would
 * land in the next record's bytes. */
static void dense_fill(uint8_t *p, uint32_t n, uint8_t v)
{
    if (n >= 8u) {
        memset(p, v, (size_t)n);
        return;
    }
    while (n-- != 0u) {
        *p++ = v;
    }
}

static int dense_check_mask(uint8_t mask_mode, const uint8_t *mask,
                            uint32_t nt, uint32_t k, uint32_t mask_len)
{
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (k != 0u || mask_len != 0u) {
            return ZPSE_BAD_MASK;
        }
        return ZPSE_OK;
    case ZPSE_M_ALL_EXCEPTION:
        if (k != nt || mask_len != 0u) {
            return ZPSE_BAD_MASK;
        }
        return ZPSE_OK;
    case ZPSE_M_RAW_BITMAP: {
        uint32_t need = (nt + 7u) >> 3;

        if (k < 1u || k >= nt) {
            return ZPSE_BAD_MASK;
        }
        if (mask_len != need) {
            return ZPSE_BAD_MASK;
        }
        if ((nt & 7u) != 0u && (mask[need - 1u] >> (nt & 7u)) != 0u) {
            return ZPSE_BAD_MASK;
        }
        if (dense_popcount(mask, need) != k) {
            return ZPSE_BAD_MASK;
        }
        return ZPSE_OK;
    }
    case ZPSE_M_SPARSE_U8: {
        uint32_t j;

        /* Positions are one byte each, so this form only exists for
         * records whose positions all fit in a byte. */
        if (nt > 256u) {
            return ZPSE_BAD_MASK;
        }
        if (k < 1u || k >= nt) {
            return ZPSE_BAD_MASK;
        }
        if (mask_len != k) {
            return ZPSE_BAD_MASK;
        }
        for (j = 0u; j < k; j++) {
            if ((uint32_t)mask[j] >= nt) {
                return ZPSE_BAD_MASK;
            }
            if (j > 0u && mask[j] <= mask[j - 1u]) {
                return ZPSE_BAD_MASK;
            }
        }
        return ZPSE_OK;
    }
    default:
        /* Mask modes 3, 4, 5 and 7 are reserved inside a dense record. */
        return ZPSE_UNSUPPORTED_FEATURE;
    }
}

/* ---- Vectorized whole-record reconstruction ------------------------ *
 * Everything below is compiled only when the ABI can carry 256-bit values
 * and the ZPSE_DENSE_AVX2 gate is on. Each kernel walks the whole record
 * with 32-byte stores, consuming the residual units in ascending position
 * order exactly as the scalar loops in dense_record do, and returns the
 * number of units it consumed. The scalar loops stay in place, untouched,
 * as the bit-exact fallback: no new format, status or lane-state rule is
 * introduced here, so a build with the gate off (or a machine without
 * AVX2) produces the same digest.
 *
 * Only three predictor *shapes* exist in a dense record in v1:
 *   CONST (already a memset between exceptions -- left alone),
 *   LANE_* (8-byte lane history, period 8), and
 *   STRIDE_* (out[p] = out[p-s]).
 * ORDER1 / ORDER2 are rejected by dense_record before reconstruction (their
 * tables are per tile, not per segment), so there is no table-chained family
 * to vectorize here; the STRIDE family is the one whose prediction is not
 * lane-local. See src/tile.c for the table-chained 8-lane kernels. */

#if ZPSE_DENSE_HAVE_AVX2 && ZPSE_DENSE_AVX2

/* Stride fill shuffle indices: byte i of row [s_index][phase] selects
 * chain[(phase + i) % s] from an eight-byte chain vector duplicated into
 * both 128-bit halves. s_index is 0..4 for s = 1,2,3,4,8; phase counts the
 * bytes emitted so far in the current run modulo s. A 32-byte pshufb is
 * per-128-bit-lane, so the row is built for all 32 byte positions (the high
 * half's index at i is (phase + 16 + i) % s, not (phase + i) % s). */
static const uint8_t dense_stride_shuf[5][8][32] = {
    { /* s = 1 */
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } /* phase 0 */
    },
    { /* s = 2 */
        { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 }, /* phase 0 */
        { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
          1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 }  /* phase 1 */
    },
    { /* s = 3 */
        { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
          1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1 }, /* phase 0 */
        { 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
          2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 }, /* phase 1 */
        { 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
          0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 }  /* phase 2 */
    },
    { /* s = 4 */
        { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
          0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 }, /* phase 0 */
        { 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
          1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0 }, /* phase 1 */
        { 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
          2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1 }, /* phase 2 */
        { 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
          3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2 }  /* phase 3 */
    },
    { /* s = 8 */
        { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
          0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 }, /* phase 0 */
        { 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0,
          1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0 }, /* phase 1 */
        { 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
          2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1 }, /* phase 2 */
        { 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2,
          3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2 }, /* phase 3 */
        { 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
          4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 }, /* phase 4 */
        { 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
          5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4 }, /* phase 5 */
        { 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
          6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5 }, /* phase 6 */
        { 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6,
          7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6 }  /* phase 7 */
    }
};

static uint32_t dense_stride_index(uint32_t s)
{
    switch (s) {
    case 1u: return 0u;
    case 2u: return 1u;
    case 3u: return 2u;
    case 4u: return 3u;
    default: return 4u; /* s == 8 */
    }
}

/* The dense mask forms expressed as a 32-bit word for the 32 positions
 * starting at `at` (a multiple of 32). Bit i is the exception flag of
 * position at+i. For SPARSE_U8 the sorted positions are walked once across
 * the whole record through *sparse_j; for RAW_BITMAP the four mask bytes
 * are read little-endian (the format's bit order) and the partial tail is
 * cleared so a short final block never consumes a unit past K. No bit at or
 * past nt can be set: dense_check_mask validated that before we got here. */
static uint32_t dense_mask_word(uint8_t mask_mode, const uint8_t *mask,
                                uint32_t nt, uint32_t at,
                                uint32_t *sparse_j, uint32_t k)
{
    uint32_t rem = nt - at;
    uint32_t w = 0u;

    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        return 0u;
    case ZPSE_M_ALL_EXCEPTION:
        return (rem >= 32u) ? UINT32_C(0xFFFFFFFF)
                            : (uint32_t)((UINT64_C(1) << rem) - 1u);
    case ZPSE_M_SPARSE_U8:
        while (*sparse_j < k && (uint32_t)mask[*sparse_j] < at + 32u) {
            uint32_t pp = (uint32_t)mask[*sparse_j];

            if (pp >= at) {
                w |= UINT32_C(1) << (pp - at);
            }
            (*sparse_j)++;
        }
        return w;
    default: { /* ZPSE_M_RAW_BITMAP */
        uint32_t mb = at >> 3u;
        uint32_t have = ((nt + 7u) >> 3u) - mb;

        if (have >= 4u) {
            memcpy(&w, mask + mb, 4);
        } else if (have != 0u) {
            memcpy(&w, mask + mb, (size_t)have);
        }
        if (rem < 32u) {
            w &= (UINT32_C(1) << rem) - 1u;
        }
        return w;
    }
    }
}

/* Exclusive popcount of one group's eight mask bits: the index, inside the
 * group's eight-byte unit window, of the unit at each position. Byte 7 is
 * the count of lanes 0..6, so adding bit 7 of the mask byte gives the group's
 * unit count. Same ladder as tile.c's tile_unit_index(). */
static ZPSE_DENSE_AVX2_TARGET __m128i dense_unit_index(__m128i bits,
                                                        __m128i bitv,
                                                        __m128i one)
{
    __m128i x = _mm_and_si128(_mm_cmpeq_epi8(bits, bitv), one);

    x = _mm_slli_si128(x, 1);
    x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
    return x;
}

/* LANE_PREVIOUS / LANE_DELTA / LANE_AVERAGE, 32 positions per outer step.
 * The prediction is a pure function of the eight-byte entry histories
 * (prev1 = last[], prev2 = second[]); each eight-byte group consumes its own
 * units, blends them with the prediction at the mask's set lanes, and feeds
 * the resulting bytes back as the next group's history. That is exactly the
 * scalar loop's update, eight lanes at a time. LANE_PREVIOUS and LANE_DELTA
 * additionally have a closed form over a whole exception-free 32-byte block,
 * which keeps a sparse-mask record at two stores per 32 bytes. */
static ZPSE_DENSE_AVX2_TARGET uint32_t dense_recon_lane_avx2(
    uint8_t *out, uint32_t nt, uint8_t pred,
    uint8_t mask_mode, const uint8_t *mask, uint32_t k,
    const uint8_t *units, uint8_t res_mode,
    const uint8_t last[8], const uint8_t second[8])
{
    const __m128i bitv = _mm_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i one = _mm_set1_epi8((char)1);
    __m128i prev1 = _mm_loadl_epi64((const __m128i *)(const void *)last);
    __m128i prev2 = _mm_loadl_epi64((const __m128i *)(const void *)second);
    uint32_t consumed = 0u;
    uint32_t sparse_j = 0u;
    uint32_t at;

    for (at = 0u; at < nt; at += 32u) {
        uint32_t rem = nt - at;
        uint32_t w = dense_mask_word(mask_mode, mask, nt, at, &sparse_j, k);
        uint32_t g;

        if (w == 0u && rem >= 32u &&
            (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS ||
             pred == (uint8_t)ZPSE_P_LANE_DELTA)) {
            if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
                __m128i v = _mm_unpacklo_epi64(prev1, prev1);

                _mm_storeu_si128((__m128i *)(void *)(out + at), v);
                _mm_storeu_si128((__m128i *)(void *)(out + at + 16u), v);
            } else {
                __m128i d = _mm_sub_epi8(prev1, prev2);
                __m128i x0 = _mm_add_epi8(prev1, d);
                __m128i x1 = _mm_add_epi8(x0, d);
                __m128i x2 = _mm_add_epi8(x1, d);
                __m128i x3 = _mm_add_epi8(x2, d);

                _mm_storeu_si128((__m128i *)(void *)(out + at),
                                 _mm_unpacklo_epi64(x0, x1));
                _mm_storeu_si128((__m128i *)(void *)(out + at + 16u),
                                 _mm_unpacklo_epi64(x2, x3));
                prev2 = x2;
                prev1 = x3;
            }
            continue;
        }

        for (g = 0u; g < 4u && at + g * 8u < nt; g++) {
            uint32_t m8 = (w >> (g * 8u)) & UINT32_C(0xFF);
            uint32_t group_rem = nt - (at + g * 8u);
            __m128i xhat;
            __m128i res;
            uint32_t cnt = 0u;

            if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
                xhat = prev1;
            } else if (pred == (uint8_t)ZPSE_P_LANE_DELTA) {
                xhat = _mm_sub_epi8(_mm_add_epi8(prev1, prev1), prev2);
            } else {
                /* LANE_AVERAGE rounds up, as the scalar formula does. */
                xhat = _mm_avg_epu8(prev1, prev2);
            }

            if (m8 == 0u) {
                res = xhat;
            } else if (m8 == UINT32_C(0xFF) && group_rem >= 8u) {
                __m128i r = _mm_loadl_epi64(
                    (const __m128i *)(const void *)(units + consumed));

                cnt = 8u;
                if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                    res = r;
                } else {
                    __m128i ge = _mm_cmpeq_epi8(_mm_max_epu8(r, xhat), r);

                    res = _mm_add_epi8(r, _mm_and_si128(ge, one));
                }
            } else {
                const __m128i bits =
                    _mm_and_si128(_mm_set1_epi8((char)m8), bitv);
                __m128i idx = dense_unit_index(bits, bitv, one);
                __m128i r = _mm_shuffle_epi8(
                    _mm_loadl_epi64(
                        (const __m128i *)(const void *)(units + consumed)),
                    idx);
                __m128i mb = _mm_cmpeq_epi8(bits, bitv);
                __m128i x;

                cnt = (uint32_t)(uint8_t)_mm_extract_epi8(idx, 7) +
                      (m8 >> 7u);
                if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                    x = r;
                } else {
                    __m128i ge = _mm_cmpeq_epi8(_mm_max_epu8(r, xhat), r);

                    x = _mm_add_epi8(r, _mm_and_si128(ge, one));
                }
                res = _mm_or_si128(_mm_and_si128(mb, x),
                                   _mm_andnot_si128(mb, xhat));
            }
            consumed += cnt;

            if (group_rem >= 8u) {
                _mm_storel_epi64((__m128i *)(void *)(out + at + g * 8u), res);
            } else {
                uint8_t ob[16];

                _mm_storeu_si128((__m128i *)(void *)ob, res);
                memcpy(out + at + g * 8u, ob, (size_t)group_rem);
            }
            prev2 = prev1;
            prev1 = res;
        }
    }
    return consumed;
}

/* STRIDE_1/2/3/4/8. The prediction at p is out[p-s] (the literal for p<s),
 * so between two exceptions every position inherits the value of the most
 * recent output in its residue class mod s. `chain[r]` holds that value; it
 * starts at the literal for all r and changes only at an exception, where
 * xhat is chain[next % s]. An exception-free run is therefore a repeating
 * period-s pattern, filled 32 bytes at a time with one pshufb + store per
 * step. This is the same "chain over the exceptions, fill the runs" shape
 * tile.c uses for STRIDE, specialised to a segment whose backing bytes are
 * all the entry literal. */
static ZPSE_DENSE_AVX2_TARGET uint32_t dense_recon_stride_avx2(
    uint8_t *out, uint32_t nt, uint32_t s, uint8_t literal,
    uint8_t mask_mode, const uint8_t *mask, uint32_t k,
    const uint8_t *units, uint8_t res_mode)
{
    uint8_t chain[8];
    uint32_t sidx = dense_stride_index(s);
    uint32_t consumed = 0u;
    uint32_t pos = 0u;
    unsigned l;

    for (l = 0u; l < 8u; l++) {
        chain[l] = literal;
    }

    while (pos < nt) {
        uint32_t next = (consumed >= k)
                            ? nt
                            : dense_next(mask_mode, mask, nt, consumed, pos);
        uint32_t run = (next < nt) ? next : nt;

        if (run > pos) {
            uint32_t phase = pos % s;
            __m256i cv = _mm256_broadcastsi128_si256(
                _mm_loadl_epi64((const __m128i *)(const void *)chain));
            uint8_t *d = out + pos;
            uint32_t n = run - pos;

            while (n >= 32u) {
                __m256i pat = _mm256_shuffle_epi8(
                    cv, _mm256_loadu_si256(
                            (const __m256i *)(const void *)
                                dense_stride_shuf[sidx][phase]));

                _mm256_storeu_si256((__m256i *)(void *)d, pat);
                d += 32u;
                n -= 32u;
                phase = (phase + 32u) % s;
            }
            while (n-- != 0u) {
                *d++ = chain[phase];
                phase++;
                if (phase == s) {
                    phase = 0u;
                }
            }
        }
        if (next >= nt) {
            break;
        }
        {
            uint32_t r = next % s;
            uint8_t xhat = chain[r];
            uint8_t x = (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL)
                            ? units[consumed]
                            : zpse_exclusion_inverse(units[consumed], xhat);

            out[next] = x;
            chain[r] = x;
            consumed++;
            pos = next + 1u;
        }
    }
    return consumed;
}

#endif /* ZPSE_DENSE_HAVE_AVX2 && ZPSE_DENSE_AVX2 */

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
#if ZPSE_DENSE_HAVE_AVX2 && ZPSE_DENSE_AVX2
    int use_avx2 = nt >= 32u &&
                   (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u;
#endif

    *is_const = 0;
    if (nt == 0u || nt > ZPSE_DENSE_MAX_LEN) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (payload_len < base || payload_len - base < ZPSE_DENSE_HEADER_SIZE) {
        return ZPSE_BAD_TILE_SIZE;
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
        return ZPSE_BAD_TILE_SIZE;
    }
    if ((flags & UINT8_C(0x80)) != 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (payload[base + 5u] != 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (rec_len < 10u ||
        (uint64_t)base + 2u + (uint64_t)rec_len > (uint64_t)payload_len) {
        return ZPSE_BAD_TILE_SIZE;
    }
    if (mask_len + res_len != rec_len - 10u) {
        return ZPSE_BAD_TILE_SIZE;
    }
    /* Mode 4 (signed delta) is prediction-dependent, and a record only
     * knows a position's prediction while reconstructing it, so it is
     * reserved here and reported as an unsupported feature rather than
     * decoded wrongly. */
    if (res_mode > (uint8_t)ZPSE_R_PACKED_BASE_M) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (pred > (uint8_t)ZPSE_P_STRIDE_3) {
        return ZPSE_BAD_PREDICTOR;
    }
    if (pred == (uint8_t)ZPSE_P_ORDER1 || pred == (uint8_t)ZPSE_P_ORDER2) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }

    mask = payload + base + ZPSE_DENSE_HEADER_SIZE;
    res = mask + mask_len;
    rc = dense_check_mask(mask_mode, mask, nt, k, mask_len);
    if (rc != ZPSE_OK) {
        return rc;
    }

    /* ---- No prediction at all (Sec. 6.3) ------------------------- *
     * An ALL_EXCEPTION mask makes every position an exception, so no
     * prediction chain is ever followed. Two shapes then let the residual
     * stream go straight into the output:
     *
     *   RAW_ACTUAL: the unit IS the actual byte at its position, for EVERY
     *   predictor. The record decodes as one copy (the "no structure left"
     *   plane of dense.h), so the residual decoder's own byte copy is the
     *   whole reconstruction. This generalises the old constant-only case:
     *   xhat never reaches an exception, whatever predicted it.
     *
     *   CONST family (prediction repeated): the record is that constant
     *   expanded through the exclusion inverse; only a byte-wide map of
     *   the constant is left, which is the palette plane of segment.c.
     *
     * Decoding into `out` directly -- rather than into the exception scratch
     * and back out through a per-position loop -- is what makes these planes
     * cost one copy/expansion instead of one predictor step per byte. The
     * residual decoder validates the payload exactly as it does on the
     * general path, so every rejection is unchanged; validation of K and of
     * the mask already happened above, and ALL_EXCEPTION forces K == nt. */
    if (mask_mode == (uint8_t)ZPSE_M_ALL_EXCEPTION &&
        (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL ||
         pred == (uint8_t)ZPSE_P_CONST_ZERO ||
         pred == (uint8_t)ZPSE_P_TILE_CONSTANT)) {
        uint8_t fill = (pred == (uint8_t)ZPSE_P_CONST_ZERO) ? 0u : constant_c;

        rc = zpse_residual_decode(res_mode, res, (uint16_t)res_len,
                                  (uint16_t)k, 0, out, out);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
            return ZPSE_OK; /* units ARE the plane; no inverse, no prediction */
        }
        if (fill == 0u) {
            dense_add_one(out, nt);
        } else if (fill == 255u) {
            /* zpse_exclusion_inverse(r, 255) is r for every residual a
             * payload can hold (255 itself is not a legal residual), and
             * the encoder picks 255 exactly when the plane does not
             * contain it, so the expansion IS the reconstruction. */
        } else {
            /* Any other prediction is not a constant offset, so this
             * takes the byte test -- vectorised, over the whole record,
             * since it is the only pass the form has left. */
            dense_exclusion_map(out, nt, fill);
        }
        return ZPSE_OK;
    }

    rc = zpse_residual_decode(res_mode, res, (uint16_t)res_len, (uint16_t)k,
                              0, scratch, scratch);
    if (rc != ZPSE_OK) {
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
        return ZPSE_OK;
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
        uint32_t at = 0u;

        /* The prediction is the same byte everywhere, so the record is that
         * byte with k exceptions on top of it. Walking the positions one at
         * a time made the exception test a data-dependent branch taken at
         * k of nt positions -- mispredicting at a third of them on the mask
         * this form is chosen for -- and stored every byte twice over. The
         * runs between exceptions are filled instead, and the units are
         * consumed in the same ascending order as before, so j still ends
         * at k and the residual stream is read identically. The mask has
         * k < nt here (ALL_EXCEPTION took the no-prediction path above and
         * k == 0 the fill above that), but a position at or past nt is
         * refused rather than written: that is the same ZPSE_BAD_MASK the
         * position loop's j != k check answered with. */
        for (j = 0; j < (uint32_t)k; j++) {
            uint32_t pos = dense_next(mask_mode, mask, nt, j, at);

            if (pos >= (uint32_t)nt) {
                return ZPSE_BAD_MASK;
            }
            if (pos > at) {
                dense_fill(out + at, pos - at, fill);
            }
            out[pos] = dense_exc_value(scratch, res_mode, j, fill);
            at = pos + 1u;
        }
        if (at < (uint32_t)nt) {
            dense_fill(out + at, (uint32_t)nt - at, fill);
        }
        j = k;
#if ZPSE_DENSE_HAVE_AVX2 && ZPSE_DENSE_AVX2
    } else if (use_avx2) {
        /* The three lane-history predictors and the stride family are the
         * only remaining shapes (ORDER1/ORDER2 were rejected above). Both
         * kernels consume exactly K units in ascending position order; a
         * mismatch is the same ZPSE_BAD_MASK the scalar loops' own j != k
         * test would have returned. The caller's scratch already holds the
         * decoded residual stream, so no new buffer -- stack or heap -- is
         * needed and dense_record's frame stays at its original 16 bytes. */
        uint32_t done;

        if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS ||
            pred == (uint8_t)ZPSE_P_LANE_DELTA ||
            pred == (uint8_t)ZPSE_P_LANE_AVERAGE) {
            done = dense_recon_lane_avx2(out, nt, pred, mask_mode, mask, k,
                                         scratch, res_mode, last, second);
        } else {
            done = dense_recon_stride_avx2(out, nt, dense_stride(pred), literal,
                                           mask_mode, mask, k, scratch,
                                           res_mode);
        }
        if (done != k) {
            return ZPSE_BAD_MASK;
        }
        j = k;
#endif
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
        uint32_t head = (s < (uint32_t)nt) ? s : (uint32_t)nt;

        /* Positions below the stride read the entry literal rather than the
         * record, so the first `s` of them are handled here and the steady
         * state below reads out[p - s] unconditionally: the `p < s` select
         * was one compare per position of the whole record for a decision
         * that is fixed after the first s bytes. The order of the stores,
         * the exception positions and the units consumed are unchanged. */
        for (p = 0u; p < head; p++) {
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, literal);
                j++;
                next = (j < k)
                           ? dense_next(mask_mode, mask, nt, j, p + 1u)
                           : nt;
            } else {
                x = literal;
            }
            out[p] = x;
        }
        for (; p < nt; p++) {
            uint8_t xhat = out[p - s];
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
        return ZPSE_BAD_MASK;
    }
    return ZPSE_OK;
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
        return ZPSE_BAD_ARG;
    }
    *all_const = 0;
    *const_byte = 0u;
    total = h->segment_decoded_size;
    if (total == 0u || total > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (total > out_cap) {
        return ZPSE_OUTPUT_OVERFLOW;
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
            return ZPSE_BAD_TILE_SIZE;
        }
        rec_len = (uint32_t)zpse_load_u16le(payload + pos);
        {
            int rec_const = 0;
            uint8_t rec_fill = 0u;

            rc = dense_record(ns, payload, payload_len, pos, out + produced,
                              scratch, h->initial_literal, &rec_const,
                              &rec_fill);
            if (rc != ZPSE_OK) {
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
            return ZPSE_BAD_TILE_SIZE;
        }
        produced += ns;
    }
    if (pos != payload_len) {
        return ZPSE_BAD_TILE_SIZE;
    }
    *out_len = total;
    *all_const = one_value;
    *const_byte = fill;
    return ZPSE_OK;
}
