/* dpcm1 -- fixed-predictor DPCM whole-frame codec for 16-bit samples.
 *
 * Included by src/zpse_unity.c (built as part of the library, not as a
 * separate translation unit). It is a single-shape driver: previous-sample
 * predictor, one signed byte per delta, sorted escapes for the rare deltas
 * that do not fit a byte. A leading raw byte prefix is stored verbatim and
 * prediction begins at the first sample after it. Decode is a byte read, an
 * SSE2 inclusive scan and a scatter of the escapes; nothing here calls the
 * block/segment machinery.
 *
 * PRODUCTION FORM: the frame encoder tries this codec by default at every
 * level, before MRD and before the PSE1 ladder. ZPSE_ENC_DPCM1 is accepted but
 * ignored (deprecated no-op); only CHECKSUM skips the attempt. The prefix probe
 * and the content gate below are the exact predicates the measured prototype
 * used, with the raw prefix added so the x-ray file's 240-byte header is
 * preserved byte for byte instead of being escaped as 120 samples.
 */

#include "zpse/dpcm1.h"

#include "zpse/bits.h"

#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DPCM1_SIMD 1
#include <emmintrin.h> /* SSE2: baseline on every x86 target we build for */
#endif

/* Wrap a 32-bit sample difference into the signed 16-bit predictor domain. */
static int32_t dpcm1_delta(uint16_t cur, uint16_t prev)
{
    int32_t d = (int32_t)cur - (int32_t)prev;

    if (d > 32767) d -= 65536;
    else if (d < -32768) d += 65536;
    return d;
}

/* Is a delta representable by one signed delta byte? The measured prototype
 * used [-127,127]; keeping that exact range reproduces its fit counts and
 * escape table, which the reported sizes and gate verdicts depend on. */
static int dpcm1_fits(int32_t d)
{
    return d >= -127 && d <= 127;
}

/* Smallest even p <= min(4096, n-2) where the 16 bytes [p-16,p) are all zero,
 * the sample at p is nonzero (so p is the end of the zero run, the start of
 * the signal), and the next 64 deltas have at least 60 int8 fits with mean
 * |delta| in [18,200]. Returns 0 when no such boundary exists. On the Silesia
 * x-ray file this returns 240, the length of its header. */
static uint32_t dpcm1_prefix(const uint8_t *in, uint32_t n)
{
    uint32_t lim = n - 2u;
    uint32_t p;

    if (lim > 4096u) lim = 4096u;
    for (p = 16u; p <= lim; p += 2u) {
        uint32_t k, fit = 0u, sumabs = 0u;
        uint16_t prev;

        /* Need sample[p .. p+128] for 64 deltas (65 samples). */
        if (p + 130u > n) break;
        if (in[p] == 0u && in[p + 1u] == 0u) continue;
        for (k = 0u; k < 16u; ++k) {
            if (in[p - 16u + k] != 0u) break;
        }
        if (k != 16u) continue;

        prev = zpse_load_u16le(in + p);
        for (k = 0u; k < 64u; ++k) {
            uint16_t s = zpse_load_u16le(in + p + 2u * (k + 1u));
            int32_t d = dpcm1_delta(s, prev);

            if (dpcm1_fits(d)) ++fit;
            sumabs += (uint32_t)(d < 0 ? -d : d);
            prev = s;
        }
        if (fit >= 60u && sumabs >= 18u * 64u && sumabs <= 200u * 64u) {
            return p;
        }
    }
    return 0u;
}

/* Necessary condition for commit, sampled evenly from the first sample after
 * the prefix. Both thresholds are integer comparisons: 90% fit and mean
 * |delta| >= 18 (sum >= 18*sample, no division). */
static int dpcm1_gate(const uint8_t *in, uint32_t prefix, uint32_t ns)
{
    uint32_t sample = ns - 1u;
    uint32_t step, s, fit = 0u, sumabs = 0u;

    if (sample > 8192u) sample = 8192u;
    step = (ns - 1u) / sample;
    if (step == 0u) step = 1u;
    for (s = 0u; s < sample; ++s) {
        uint32_t at = prefix + 2u * (s * step);
        int32_t d = dpcm1_delta(zpse_load_u16le(in + at + 2u),
                                zpse_load_u16le(in + at));

        if (dpcm1_fits(d)) ++fit;
        sumabs += (uint32_t)(d < 0 ? -d : d);
    }
    return (fit * 10u >= sample * 9u && sumabs >= 18u * sample);
}

/* Worst case (no prefix, every delta escaped) as a public bound. */
uint32_t zpse_dpcm1_bound(uint32_t n)
{
    uint64_t ns, b;

    if (n < 2u || (n & 1u) != 0u) return 0u;
    ns = (uint64_t)n / 2u;
    b = (uint64_t)ZPSE_DPCM1_HEADER_SIZE + (ns - 1u) + 6u * (ns - 1u);
    return b > UINT32_MAX ? 0u : (uint32_t)b;
}

int zpse_dpcm1_encode(const uint8_t *in, uint32_t n,
                       uint8_t *out, uint32_t out_cap, uint32_t *out_len)
{
    uint32_t prefix, ns, ne = 0u, i;
    uint64_t need;
    uint16_t first, prev;
    uint8_t *delta, *esc;

    if (out_len != 0) *out_len = 0u;
    if (in == 0 || out == 0 || out_len == 0) return ZPSE_DPCM1_DECLINE;
    if (n < 4u || (n & 1u) != 0u) return ZPSE_DPCM1_DECLINE;

    prefix = dpcm1_prefix(in, n);
    if ((prefix & 1u) != 0u || prefix > n - 2u) return ZPSE_DPCM1_DECLINE;
    ns = n / 2u - prefix / 2u;
    if (ns < 2u) return ZPSE_DPCM1_DECLINE;
    if (!dpcm1_gate(in, prefix, ns)) return ZPSE_DPCM1_DECLINE;

    /* Pass 1: count escapes so the exact capacity/size requirement is known
     * before a single byte of out is written. A decline here leaves out
     * untouched. */
    first = zpse_load_u16le(in + prefix);
    prev = first;
    for (i = 1u; i < ns; ++i) {
        uint16_t s = zpse_load_u16le(in + prefix + 2u * i);
        int32_t d = dpcm1_delta(s, prev);

        if (!dpcm1_fits(d)) ++ne;
        prev = s;
    }
    need = (uint64_t)ZPSE_DPCM1_HEADER_SIZE + (uint64_t)prefix +
           (uint64_t)(ns - 1u) + 6u * (uint64_t)ne;
    /* Strict size bound before any store: the whole-frame form must never be
     * larger than the raw input it replaces, and must fit the caller's
     * buffer. Checked in 64 bits; no truncation can hide an overflow. */
    if (need > (uint64_t)n || (uint64_t)out_cap < need) {
        return ZPSE_DPCM1_DECLINE;
    }

    /* Pass 2: header, verbatim prefix, delta bytes, then the sorted escape
     * table (already increasing because the scan is). */
    zpse_store_u32le(out, ZPSE_DPCM1_MAGIC);
    zpse_store_u32le(out + 4u, n);
    zpse_store_u16le(out + 8u, first);
    zpse_store_u16le(out + 10u, (uint16_t)prefix);
    zpse_store_u32le(out + 12u, ne);
    memcpy(out + ZPSE_DPCM1_HEADER_SIZE, in, prefix);
    delta = out + ZPSE_DPCM1_HEADER_SIZE + prefix;
    esc = delta + (ns - 1u);

    prev = first;
    for (i = 1u; i < ns; ++i) {
        uint16_t s = zpse_load_u16le(in + prefix + 2u * i);
        int32_t d = dpcm1_delta(s, prev);

        if (dpcm1_fits(d)) {
            delta[i - 1u] = (uint8_t)(int8_t)d;
        } else {
            delta[i - 1u] = 0u;
            zpse_store_u32le(esc, i - 1u);
            zpse_store_u16le(esc + 4u, (uint16_t)(int16_t)d);
            esc += 6u;
        }
        prev = s;
    }
    *out_len = (uint32_t)need;
    return ZPSE_DPCM1_COMMIT;
}

int zpse_dpcm1_decode(const uint8_t *in, uint32_t in_len,
                       uint8_t *out, uint32_t n)
{
    uint32_t prefix, ns, ne, i, ei, dpos;
    uint64_t need;
    uint16_t first;
    int32_t base;
    const uint8_t *delta, *esc;

    /* Bounded header parse: magic, length, parity, prefix and escape count
     * are all validated before any byte of out is touched. */
    if (in == 0 || out == 0 || in_len < (uint32_t)ZPSE_DPCM1_HEADER_SIZE) {
        return ZPSE_DPCM1_DECLINE;
    }
    if (zpse_load_u32le(in) != ZPSE_DPCM1_MAGIC) return ZPSE_DPCM1_DECLINE;
    if (n < 4u || (n & 1u) != 0u) return ZPSE_DPCM1_DECLINE;
    if (zpse_load_u32le(in + 4u) != n) return ZPSE_DPCM1_DECLINE;
    prefix = zpse_load_u16le(in + 10u);
    if ((prefix & 1u) != 0u || prefix > n - 2u) return ZPSE_DPCM1_DECLINE;
    ns = n / 2u - prefix / 2u;
    if (ns < 2u) return ZPSE_DPCM1_DECLINE;
    ne = zpse_load_u32le(in + 12u);
    if (ne > ns - 1u) return ZPSE_DPCM1_DECLINE;
    need = (uint64_t)ZPSE_DPCM1_HEADER_SIZE + (uint64_t)prefix +
           (uint64_t)(ns - 1u) + 6u * (uint64_t)ne;
    if (need != (uint64_t)in_len) return ZPSE_DPCM1_DECLINE;

    esc = in + ZPSE_DPCM1_HEADER_SIZE + prefix + (ns - 1u);
    /* Escape positions must be in range and strictly increasing; the encoder
     * writes them sorted, and a hostile stream must not be able to steer the
     * scatter below out of the delta window. */
    for (i = 0u; i < ne; ++i) {
        uint32_t p = zpse_load_u32le(esc + (size_t)6u * i);

        if (p >= ns - 1u) return ZPSE_DPCM1_DECLINE;
        if (i != 0u && p <= zpse_load_u32le(esc + (size_t)6u * (i - 1u))) {
            return ZPSE_DPCM1_DECLINE;
        }
    }

    first = zpse_load_u16le(in + 8u);
    memcpy(out, in + ZPSE_DPCM1_HEADER_SIZE, prefix); /* verbatim prefix */
    zpse_store_u16le(out + prefix, first);
    base = (int32_t)(int16_t)first;
    delta = in + ZPSE_DPCM1_HEADER_SIZE + prefix;
    dpos = 0u;
    ei = 0u;

#ifdef DPCM1_SIMD
    /* Eight 16-bit lanes per SSE2 vector. The inclusive scan is the
     * Hillis-Steele scans (shifts of 1, 2, 4 lanes = 2, 4, 8 bytes); the carry
     * between vectors is a single wrapped 16-bit running base, so no raw
     * bases are stored in the stream. */
    for (; dpos + 8u <= ns - 1u; dpos += 8u) {
        __m128i raw = _mm_loadl_epi64((const __m128i *)(delta + dpos));
        __m128i sign = _mm_cmplt_epi8(raw, _mm_setzero_si128());
        __m128i v = _mm_unpacklo_epi8(raw, sign); /* 8 sign-extended deltas */

        while (ei < ne && zpse_load_u32le(esc + (size_t)6u * ei) < dpos + 8u) {
            uint32_t p = zpse_load_u32le(esc + (size_t)6u * ei);

            if (p >= dpos) {
                int16_t ed = (int16_t)zpse_load_u16le(esc + (size_t)6u * ei + 4u);
                int16_t tmp[8];

                _mm_storeu_si128((__m128i *)tmp, v);
                tmp[p - dpos] = ed;
                v = _mm_loadu_si128((const __m128i *)tmp);
            }
            ++ei;
        }

        v = _mm_add_epi16(v, _mm_slli_si128(v, 2));
        v = _mm_add_epi16(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi16(v, _mm_slli_si128(v, 8));
        {
            int32_t sum = (int16_t)_mm_extract_epi16(v, 7);

            v = _mm_add_epi16(v, _mm_set1_epi16((int16_t)base));
            _mm_storeu_si128((__m128i *)(out + prefix + (size_t)2u * (dpos + 1u)), v);
            base = (int32_t)(int16_t)(base + sum);
        }
    }
#endif

    /* Scalar tail (and the whole loop on a non-x86 build). */
    for (; dpos < ns - 1u; ++dpos) {
        int32_t d = (int32_t)(int8_t)delta[dpos];

        while (ei < ne && zpse_load_u32le(esc + (size_t)6u * ei) == dpos) {
            d = (int16_t)zpse_load_u16le(esc + (size_t)6u * ei + 4u);
            ++ei;
        }
        base = (int32_t)(int16_t)(base + d);
        zpse_store_u16le(out + prefix + (size_t)2u * (dpos + 1u), (uint16_t)base);
    }
    return ZPSE_DPCM1_COMMIT;
}
