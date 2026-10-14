#include "zpse/transform.h"
#include "zpse/constants.h"
#include "zpse/status.h"

#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 * SSE2 fast paths for the two transform shapes the encoder actually
 * trials. For a block of 16 bytes or more `zpse_block_encode` only ever
 * forwards transform ids 0 (RAW_BYTES), 6 (TRANSPOSE_16) and 1 (DELTA8),
 * so the width `w` is 1 or 16 there; every other width (2, 4, 8, the
 * delta/xor transposes and the stride transpose) reaches the scalar
 * loops below, which stay the normative reference.
 *
 * SSE2 is architectural on x86-64, so this needs no runtime dispatch and
 * no per-function target attribute. That is deliberate: the normal build
 * and the -DZPSE_FORCE_SCALAR conformance build compile this very same
 * code, so the scalar/SIMD parity test cannot observe a difference here
 * by construction. Every path below is byte-for-byte the scalar loop's
 * output; only the number of bytes each iteration covers changes.
 * ------------------------------------------------------------------ */
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#define ZPSE_T_HAVE_SSE2 1
#else
#define ZPSE_T_HAVE_SSE2 0
#endif

#if ZPSE_T_HAVE_SSE2
/* 16x16 byte transpose, in place: entering with row q in r[q] (byte c of
 * row q at position c) leaves column c in r[c] (byte q of column c at
 * position q). The permutation is its own inverse, which is exactly why
 * the forward gather and the inverse scatter can share it: both only
 * need "16 rows of 16 become 16 columns of 16". */
static void zpse_t16x16(__m128i r[16])
{
    __m128i t[16];
    __m128i u[16];
    __m128i v[16];
    unsigned q;

    /* Stage 1: interleave adjacent rows bytewise. */
    for (q = 0; q < 8u; q++) {
        t[2u * q] = _mm_unpacklo_epi8(r[2u * q], r[2u * q + 1u]);
        t[2u * q + 1u] = _mm_unpackhi_epi8(r[2u * q], r[2u * q + 1u]);
    }
    /* Stage 2: 16-bit units. */
    u[0] = _mm_unpacklo_epi16(t[0], t[2]);
    u[1] = _mm_unpackhi_epi16(t[0], t[2]);
    u[2] = _mm_unpacklo_epi16(t[1], t[3]);
    u[3] = _mm_unpackhi_epi16(t[1], t[3]);
    u[4] = _mm_unpacklo_epi16(t[4], t[6]);
    u[5] = _mm_unpackhi_epi16(t[4], t[6]);
    u[6] = _mm_unpacklo_epi16(t[5], t[7]);
    u[7] = _mm_unpackhi_epi16(t[5], t[7]);
    u[8] = _mm_unpacklo_epi16(t[8], t[10]);
    u[9] = _mm_unpackhi_epi16(t[8], t[10]);
    u[10] = _mm_unpacklo_epi16(t[9], t[11]);
    u[11] = _mm_unpackhi_epi16(t[9], t[11]);
    u[12] = _mm_unpacklo_epi16(t[12], t[14]);
    u[13] = _mm_unpackhi_epi16(t[12], t[14]);
    u[14] = _mm_unpacklo_epi16(t[13], t[15]);
    u[15] = _mm_unpackhi_epi16(t[13], t[15]);
    /* Stage 3: 32-bit units. */
    v[0] = _mm_unpacklo_epi32(u[0], u[4]);
    v[1] = _mm_unpackhi_epi32(u[0], u[4]);
    v[2] = _mm_unpacklo_epi32(u[1], u[5]);
    v[3] = _mm_unpackhi_epi32(u[1], u[5]);
    v[4] = _mm_unpacklo_epi32(u[2], u[6]);
    v[5] = _mm_unpackhi_epi32(u[2], u[6]);
    v[6] = _mm_unpacklo_epi32(u[3], u[7]);
    v[7] = _mm_unpackhi_epi32(u[3], u[7]);
    v[8] = _mm_unpacklo_epi32(u[8], u[12]);
    v[9] = _mm_unpackhi_epi32(u[8], u[12]);
    v[10] = _mm_unpacklo_epi32(u[9], u[13]);
    v[11] = _mm_unpackhi_epi32(u[9], u[13]);
    v[12] = _mm_unpacklo_epi32(u[10], u[14]);
    v[13] = _mm_unpackhi_epi32(u[10], u[14]);
    v[14] = _mm_unpacklo_epi32(u[11], u[15]);
    v[15] = _mm_unpackhi_epi32(u[11], u[15]);
    /* Stage 4: 64-bit units -> one column per register. */
    r[0] = _mm_unpacklo_epi64(v[0], v[8]);
    r[1] = _mm_unpackhi_epi64(v[0], v[8]);
    r[2] = _mm_unpacklo_epi64(v[1], v[9]);
    r[3] = _mm_unpackhi_epi64(v[1], v[9]);
    r[4] = _mm_unpacklo_epi64(v[2], v[10]);
    r[5] = _mm_unpackhi_epi64(v[2], v[10]);
    r[6] = _mm_unpacklo_epi64(v[3], v[11]);
    r[7] = _mm_unpackhi_epi64(v[3], v[11]);
    r[8] = _mm_unpacklo_epi64(v[4], v[12]);
    r[9] = _mm_unpackhi_epi64(v[4], v[12]);
    r[10] = _mm_unpacklo_epi64(v[5], v[13]);
    r[11] = _mm_unpackhi_epi64(v[5], v[13]);
    r[12] = _mm_unpacklo_epi64(v[6], v[14]);
    r[13] = _mm_unpackhi_epi64(v[6], v[14]);
    r[14] = _mm_unpacklo_epi64(v[7], v[15]);
    r[15] = _mm_unpackhi_epi64(v[7], v[15]);
}
#endif /* ZPSE_T_HAVE_SSE2 */

static int zpse_valid_count(uint8_t v)
{
    return (v == 1 || v == 2 || v == 4 || v == 8 || v == 16) ? 1 : 0;
}

static int zpse_needs_delta(uint8_t tid, uint8_t flags)
{
    if (tid == 1) {
        return 1;
    }
    if (tid >= 7 && tid <= 10) {
        return 1;
    }
    if (tid == 11 && (flags & (uint8_t)1)) {
        return 1;
    }
    return 0;
}

static int zpse_needs_xor(uint8_t tid)
{
    return (tid == 2) ? 1 : 0;
}

static int zpse_ranges_overlap(const void *a, uint32_t alen,
                               const void *b, uint32_t blen)
{
    uintptr_t x;
    uintptr_t y;
    if (alen == 0 || blen == 0) {
        return 0;
    }
    x = (uintptr_t)a;
    y = (uintptr_t)b;
    return (x < y + (uintptr_t)blen && y < x + (uintptr_t)alen) ? 1 : 0;
}

int zpse_transform_width(uint8_t transform_id, uint8_t segment_count,
                         uint8_t *w_out)
{
    uint8_t w = 0;

    if (w_out == 0) {
        return PSE_BAD_ARG;
    }
    switch (transform_id) {
    case 0:
    case 1:
    case 2:
        w = 1;
        break;
    case 3:
    case 7:
        w = 2;
        break;
    case 4:
    case 8:
        w = 4;
        break;
    case 5:
    case 9:
        w = 8;
        break;
    case 6:
    case 10:
        w = 16;
        break;
    case 11:
        if (!zpse_valid_count(segment_count)) {
            return PSE_BAD_TRANSFORM;
        }
        w = segment_count;
        break;
    default:
        return PSE_BAD_TRANSFORM;
    }
    if (transform_id != 11 && segment_count != w) {
        return PSE_BAD_TRANSFORM;
    }
    *w_out = w;
    return PSE_OK;
}

int zpse_segment_lengths(uint32_t n, uint8_t w, uint32_t out_lens[16])
{
    uint32_t s;

    if (out_lens == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (!zpse_valid_count(w)) {
        return PSE_BAD_ARG;
    }
    for (s = 0; s < (uint32_t)w; s++) {
        out_lens[s] = (n + (uint32_t)w - 1u - s) / (uint32_t)w;
    }
    for (s = (uint32_t)w; s < 16u; s++) {
        out_lens[s] = 0;
    }
    return PSE_OK;
}

int zpse_transform_validate_sizes(uint8_t transform_id, uint8_t segment_count,
                                  uint32_t n, const uint32_t *seg_lens)
{
    uint8_t w = 0;
    uint8_t s;
    int rc;

    if (seg_lens == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    rc = zpse_transform_width(transform_id, segment_count, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    for (s = 0; s < w; s++) {
        uint32_t expect = (n + (uint32_t)w - 1u - (uint32_t)s) / (uint32_t)w;
        if (seg_lens[s] != expect) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
        if (seg_lens[s] < 1) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
    }
    return PSE_OK;
}

int zpse_transform_forward(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *in, uint32_t n,
                           uint8_t *segs[16], const uint32_t seg_lens[16])
{
    uint8_t w = 0;
    uint8_t s;
    uint32_t j;
    int rc;
    int delta;
    int xorr;

    if (meta == 0 || in == 0 || segs == 0 || seg_lens == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (meta->transform_id > 11) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_width(meta->transform_id, seg_count, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    if (meta->width_W != w || seg_count != w) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_validate_sizes(meta->transform_id, seg_count, n,
                                       seg_lens);
    if (rc != PSE_OK) {
        return rc;
    }
    for (s = 0; s < w; s++) {
        if (segs[s] == 0) {
            return PSE_BAD_ARG;
        }
    }
    delta = zpse_needs_delta(meta->transform_id, meta->stride_flags);
    xorr = zpse_needs_xor(meta->transform_id);
    if ((delta || xorr) && !meta->has_seeds) {
        return PSE_BAD_ARG;
    }
    for (s = 0; s < w; s++) {
        if (zpse_ranges_overlap(in, n, segs[s], seg_lens[s])) {
            return PSE_BAD_ARG;
        }
    }
    for (s = 0; s < w; s++) {
        uint8_t t;
        for (t = (uint8_t)(s + 1u); t < w; t++) {
            if (zpse_ranges_overlap(segs[s], seg_lens[s],
                                    segs[t], seg_lens[t])) {
                return PSE_BAD_ARG;
            }
        }
    }
#if ZPSE_T_HAVE_SSE2
    if (w == 1u) {
        uint8_t *dst = segs[0];

        if (!delta && !xorr) {
            /* A plain width-1 split is the identity: 64 KiB of bytewise
             * `dst[j] = in[j]` becomes one memcpy. */
            memcpy(dst, in, (size_t)n);
            return PSE_OK;
        }
        /* Only dst[0] touches the seed. For j >= 1 the chain's
         * `prev = raw` makes the stored value in[j] - in[j-1] (or the
         * xor), i.e. a bytewise op between the stream and itself shifted
         * by one byte -- no dependency chain to serialise, unlike the
         * inverse below. */
        dst[0] = delta ? (uint8_t)(in[0] - meta->seeds[0])
                       : (uint8_t)(in[0] ^ meta->seeds[0]);
        j = 1;
        for (; (size_t)j + 16u <= (size_t)n; j += 16u) {
            __m128i cur = _mm_loadu_si128(
                (const __m128i *)(const void *)(in + j));
            __m128i prv = _mm_loadu_si128(
                (const __m128i *)(const void *)(in + j - 1u));
            _mm_storeu_si128((__m128i *)(void *)(dst + j),
                             delta ? _mm_sub_epi8(cur, prv)
                                   : _mm_xor_si128(cur, prv));
        }
        for (; j < n; j++) {
            dst[j] = delta ? (uint8_t)(in[j] - in[j - 1u])
                           : (uint8_t)(in[j] ^ in[j - 1u]);
        }
        return PSE_OK;
    }
    if (w == 16u && !delta && !xorr) {
        uint32_t rows = seg_lens[0];
        uint32_t g;

        for (s = 1; s < w; s++) {
            if (seg_lens[s] < rows) {
                rows = seg_lens[s];
            }
        }
        /* A group is 16 rows of 16 bytes: one contiguous 256-byte input
         * block becomes 16 x 16-byte per-segment stores. The group bound
         * uses the SHORTEST segment, so every segment still has all 16
         * rows and every load stays inside the input. */
        for (g = 0; (g + 1u) * 16u <= rows; g++) {
            __m128i r[16];
            unsigned t;

            for (t = 0; t < 16u; t++) {
                r[t] = _mm_loadu_si128((const __m128i *)(const void *)
                                       (in + (size_t)g * 256u +
                                        (size_t)t * 16u));
            }
            zpse_t16x16(r);
            for (t = 0; t < 16u; t++) {
                _mm_storeu_si128(
                    (__m128i *)(void *)(segs[t] + (size_t)g * 16u), r[t]);
            }
        }
        /* Ragged tail (segment lengths differ by at most one row). */
        for (s = 0; s < w; s++) {
            uint32_t len = seg_lens[s];

            for (j = g * 16u; j < len; j++) {
                segs[s][j] = in[(uint32_t)s + j * 16u];
            }
        }
        return PSE_OK;
    }
#endif /* ZPSE_T_HAVE_SSE2 */

    for (s = 0; s < w; s++) {
        uint32_t len = seg_lens[s];
        uint8_t *dst = segs[s];
        if (delta) {
            uint8_t seed = meta->seeds[s];
            uint8_t prev = seed;
            for (j = 0; j < len; j++) {
                uint8_t raw = in[(uint32_t)s + j * (uint32_t)w];
                dst[j] = (uint8_t)(raw - prev);
                prev = raw;
            }
        } else if (xorr) {
            uint8_t seed = meta->seeds[s];
            uint8_t prev = seed;
            for (j = 0; j < len; j++) {
                uint8_t raw = in[(uint32_t)s + j * (uint32_t)w];
                dst[j] = (uint8_t)(raw ^ prev);
                prev = raw;
            }
        } else {
            for (j = 0; j < len; j++) {
                dst[j] = in[(uint32_t)s + j * (uint32_t)w];
            }
        }
    }
    return PSE_OK;
}

int zpse_transform_inverse(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *segs[16], const uint32_t seg_lens[16],
                           uint8_t *out, uint32_t n)
{
    uint8_t w = 0;
    uint8_t s;
    uint32_t i;
    int rc;
    int delta;
    int xorr;
    uint8_t prev[16];

    if (meta == 0 || segs == 0 || seg_lens == 0 || out == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (meta->transform_id > 11) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_width(meta->transform_id, seg_count, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    if (meta->width_W != w || seg_count != w) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_validate_sizes(meta->transform_id, seg_count, n,
                                       seg_lens);
    if (rc != PSE_OK) {
        return rc;
    }
    for (s = 0; s < w; s++) {
        if (segs[s] == 0) {
            return PSE_BAD_ARG;
        }
    }
    delta = zpse_needs_delta(meta->transform_id, meta->stride_flags);
    xorr = zpse_needs_xor(meta->transform_id);
    if ((delta || xorr) && !meta->has_seeds) {
        return PSE_BAD_ARG;
    }
    for (s = 0; s < w; s++) {
        if (zpse_ranges_overlap(out, n, segs[s], seg_lens[s])) {
            return PSE_BAD_ARG;
        }
    }
    for (s = 0; s < w; s++) {
        uint8_t t;
        for (t = (uint8_t)(s + 1u); t < w; t++) {
            if (zpse_ranges_overlap(segs[s], seg_lens[s],
                                    segs[t], seg_lens[t])) {
                return PSE_BAD_ARG;
            }
        }
    }
    /* Per-segment loops instead of a single i loop with `i % w` and
     * `i / w`: those are two integer divisions per output byte when `w`
     * is not a compile-time constant, which dominated decode for the
     * single-segment transforms. Segment s owns exactly the output
     * positions s, s+w, s+2w, ... in ascending order, and the delta/xor
     * chains are segment-local, so this is bit-identical to the i loop. */
    (void)i;
    if (!delta && !xorr) {
        if (w == 1u) {
            memcpy(out, segs[0], (size_t)n);
            return PSE_OK;
        }
#if ZPSE_T_HAVE_SSE2
        if (w == 16u) {
            uint32_t rows = seg_lens[0];
            uint32_t g;

            for (s = 1; s < w; s++) {
                if (seg_lens[s] < rows) {
                    rows = seg_lens[s];
                }
            }
            /* The mirror image of the forward gather: load 16 segment
             * columns of 16 bytes and transposing them yields the 16
             * contiguous output rows out[16*j + s]. */
            for (g = 0; (g + 1u) * 16u <= rows; g++) {
                __m128i r[16];
                unsigned t;

                for (t = 0; t < 16u; t++) {
                    r[t] = _mm_loadu_si128((const __m128i *)(const void *)
                                           (segs[t] + (size_t)g * 16u));
                }
                zpse_t16x16(r);
                for (t = 0; t < 16u; t++) {
                    _mm_storeu_si128(
                        (__m128i *)(void *)(out + (size_t)g * 256u +
                                            (size_t)t * 16u),
                        r[t]);
                }
            }
            for (s = 0; s < w; s++) {
                const uint8_t *src = segs[s];
                uint32_t len = seg_lens[s];
                uint32_t jj;

                for (jj = g * 16u; jj < len; jj++) {
                    out[(uint32_t)s + jj * 16u] = src[jj];
                }
            }
            return PSE_OK;
        }
#endif /* ZPSE_T_HAVE_SSE2 */
        for (s = 0; s < w; s++) {
            const uint8_t *src = segs[s];
            uint32_t len = seg_lens[s];
            uint32_t j;
            for (j = 0; j < len; j++) {
                out[(uint32_t)s + j * (uint32_t)w] = src[j];
            }
        }
        return PSE_OK;
    }
    for (s = 0; s < w; s++) {
        prev[s] = meta->seeds[s];
    }
    if (w == 1u) {
        const uint8_t *src = segs[0];
        uint8_t pv = prev[0];
        uint32_t j = 0;

#if ZPSE_T_HAVE_SSE2
        /* out[j] = seed + sum(enc[0..j]) for delta, the xor prefix for
         * xor: a prefix scan. SSE2 does 16 bytes at a time with the
         * standard log-shift ladder (1, 2, 4, 8) to build the within-
         * block prefix sums, then one broadcast of the running carry
         * added to all 16 lanes -- valid because the carry is exactly
         * the constant every lane's prefix is missing. The carry for the
         * next block is lane 15 of the result. */
        {
            __m128i carry = _mm_set1_epi8((char)pv);

            for (; (size_t)j + 16u <= (size_t)n; j += 16u) {
                __m128i v = _mm_loadu_si128(
                    (const __m128i *)(const void *)(src + j));

                v = delta ? _mm_add_epi8(v, _mm_slli_si128(v, 1))
                          : _mm_xor_si128(v, _mm_slli_si128(v, 1));
                v = delta ? _mm_add_epi8(v, _mm_slli_si128(v, 2))
                          : _mm_xor_si128(v, _mm_slli_si128(v, 2));
                v = delta ? _mm_add_epi8(v, _mm_slli_si128(v, 4))
                          : _mm_xor_si128(v, _mm_slli_si128(v, 4));
                v = delta ? _mm_add_epi8(v, _mm_slli_si128(v, 8))
                          : _mm_xor_si128(v, _mm_slli_si128(v, 8));
                v = delta ? _mm_add_epi8(v, carry) : _mm_xor_si128(v, carry);
                _mm_storeu_si128((__m128i *)(void *)(out + j), v);
                /* Word 7 holds bytes 14 and 15, so the carry is its HIGH
                 * byte: a plain uint8_t cast of the word would keep byte
                 * 14 and carry the wrong value into every later block. */
                carry = _mm_set1_epi8(
                    (char)(uint8_t)(_mm_extract_epi16(v, 7) >> 8));
            }
            /* Every lane of `carry` holds the same byte, so lane 0 is
             * the running value the scalar tail must continue from --
             * and if n < 16 the block above never ran, leaving the seed
             * itself, which is what the scalar loop starts from too. */
            pv = (uint8_t)_mm_extract_epi16(carry, 0);
        }
#endif /* ZPSE_T_HAVE_SSE2 */
        for (; j < n; j++) {
            uint8_t e = src[j];
            uint8_t d = delta ? (uint8_t)(e + pv) : (uint8_t)(e ^ pv);
            pv = d;
            out[j] = d;
        }
        return PSE_OK;
    }
    for (s = 0; s < w; s++) {
        const uint8_t *src = segs[s];
        uint32_t len = seg_lens[s];
        uint8_t pv = prev[s];
        uint32_t j;
        for (j = 0; j < len; j++) {
            uint8_t e = src[j];
            uint8_t d = delta ? (uint8_t)(e + pv) : (uint8_t)(e ^ pv);
            pv = d;
            out[(uint32_t)s + j * (uint32_t)w] = d;
        }
    }
    return PSE_OK;
}

void zpse_delta8_encode(const uint8_t *in, uint8_t *enc, uint32_t n,
                        uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (in == 0 || enc == 0 || n == 0) {
        return;
    }
    prev = in[0];
    enc[0] = (uint8_t)(in[0] - seed);
    for (i = 1; i < n; i++) {
        uint8_t cur = in[i];
        enc[i] = (uint8_t)(cur - prev);
        prev = cur;
    }
}

void zpse_delta8_decode(const uint8_t *enc, uint8_t *out, uint32_t n,
                        uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (enc == 0 || out == 0 || n == 0) {
        return;
    }
    out[0] = (uint8_t)(enc[0] + seed);
    prev = out[0];
    for (i = 1; i < n; i++) {
        uint8_t e = enc[i];
        uint8_t d = (uint8_t)(e + prev);
        out[i] = d;
        prev = d;
    }
}

void zpse_xor8_encode(const uint8_t *in, uint8_t *enc, uint32_t n,
                      uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (in == 0 || enc == 0 || n == 0) {
        return;
    }
    prev = in[0];
    enc[0] = (uint8_t)(in[0] ^ seed);
    for (i = 1; i < n; i++) {
        uint8_t cur = in[i];
        enc[i] = (uint8_t)(cur ^ prev);
        prev = cur;
    }
}

void zpse_xor8_decode(const uint8_t *enc, uint8_t *out, uint32_t n,
                      uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (enc == 0 || out == 0 || n == 0) {
        return;
    }
    out[0] = (uint8_t)(enc[0] ^ seed);
    prev = out[0];
    for (i = 1; i < n; i++) {
        uint8_t e = enc[i];
        uint8_t d = (uint8_t)(e ^ prev);
        out[i] = d;
        prev = d;
    }
}
