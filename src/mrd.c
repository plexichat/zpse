/* mrd -- previous-sample DPCM + zero-run RLE whole-frame codec for 16-bit
 * samples. Included by src/zpse_unity.c (built as part of the library).
 *
 * Wire format (all integers little-endian):
 *   0   4  magic "MRD1"
 *   4   4  n = original byte length
 *   8   2  first sample after the prefix (u16)
 *   10  2  prefix length, even (u16)
 *   12  4  number of escapes (u32)
 *   16  4  control-stream length in bytes (u32)
 *   20  4  number of RLE tokens (u32, informational)
 *   24  p  verbatim prefix
 *   24+p .. +cstream_len  control stream:
 *        0x81..0xFF / 0x01..0x7F : one literal int8 delta (-127..-1 / 1..127)
 *        0x00                    : one zero delta, or an escape placeholder
 *        0x80 + u16 run (>=16)   : `run` consecutive zero deltas (hold value)
 *   tail 6*ne escape table, sorted by strictly increasing delta index:
 *        (delta index u32, true delta i16)
 *
 * Byte 0x80 is int8 -128, outside the [-127,127] fit range, so no literal
 * delta and no escape placeholder (0x00) ever equals the RLE marker.
 */

#include "zpse/mrd.h"

#include "zpse/bits.h"

#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MRD_SIMD 1
#include <emmintrin.h>
#endif

#define MRD_MARKER 0x80u
#define MRD_MIN_RUN 16u
#define MRD_MAX_RUN 65535u

static int32_t mrd_delta(uint16_t cur, uint16_t prev)
{
    int32_t d = (int32_t)cur - (int32_t)prev;
    if (d > 32767) d -= 65536;
    else if (d < -32768) d += 65536;
    return d;
}

static int mrd_fits(int32_t d)
{
    return d >= -127 && d <= 127;
}

/* Verbatim prefix probe, copied from DPCM1. On mr the header is 9092 bytes
 * (>4096) and the signal's mean |delta| is ~11 (<18), so this returns 0 -- the
 * entire mr file is decoded as int16 samples with no prefix. */
static uint32_t mrd_prefix(const uint8_t *in, uint32_t n)
{
    uint32_t lim = n - 2u;
    uint32_t p;

    if (lim > 4096u) lim = 4096u;
    for (p = 16u; p <= lim; p += 2u) {
        uint32_t k, fit = 0u, sumabs = 0u;
        uint16_t prev;

        if (p + 130u > n) break;
        if (in[p] == 0u && in[p + 1u] == 0u) continue;
        for (k = 0u; k < 16u; ++k) {
            if (in[p - 16u + k] != 0u) break;
        }
        if (k != 16u) continue;

        prev = zpse_load_u16le(in + p);
        for (k = 0u; k < 64u; ++k) {
            uint16_t s = zpse_load_u16le(in + p + 2u * (k + 1u));
            int32_t d = mrd_delta(s, prev);
            if (mrd_fits(d)) ++fit;
            sumabs += (uint32_t)(d < 0 ? -d : d);
            prev = s;
        }
        if (fit >= 60u && sumabs >= 18u * 64u && sumabs <= 200u * 64u) {
            return p;
        }
    }
    return 0u;
}

/* Content gate. Necessary conditions, all integer comparisons:
 *   >=95% of sampled deltas fit int8,
 *   mean |delta| in [4,14) (accepts mr ~11, declines x-ray ~25),
 *   >=5% of sampled deltas are zero (a zero-run RLE opportunity). */
static int mrd_gate(const uint8_t *in, uint32_t prefix, uint32_t ns)
{
    uint32_t sample = ns - 1u;
    uint32_t step, s, fit = 0u, sumabs = 0u, zeros = 0u;

    if (sample > 8192u) sample = 8192u;
    step = (ns - 1u) / sample;
    if (step == 0u) step = 1u;
    for (s = 0u; s < sample; ++s) {
        uint32_t at = prefix + 2u * (s * step);
        int32_t d = mrd_delta(zpse_load_u16le(in + at + 2u),
                              zpse_load_u16le(in + at));
        if (mrd_fits(d)) ++fit;
        sumabs += (uint32_t)(d < 0 ? -d : d);
        if (d == 0) ++zeros;
    }
    return (fit * 20u >= sample * 19u &&
            sumabs >= 4u * sample && sumabs < 14u * sample &&
            zeros * 20u >= sample);
}

uint32_t zpse_mrd_bound(uint32_t n)
{
    uint64_t ns, b;

    if (n < 2u || (n & 1u) != 0u) return 0u;
    ns = (uint64_t)n / 2u;
    /* Worst case (no RLE, every delta escaped): 24 + (ns-1) + 6*(ns-1). */
    b = (uint64_t)ZPSE_MRD_HEADER_SIZE + (ns - 1u) + 6u * (ns - 1u);
    return b > UINT32_MAX ? 0u : (uint32_t)b;
}

/* Emit a zero run as one or more RLE tokens (u16 run length caps at 65535).
 * `dst` may be NULL (count-only pass); returns bytes written. */
static uint32_t mrd_emit_run(uint8_t *dst, uint32_t run)
{
    uint32_t written = 0u;

    while (run != 0u) {
        uint32_t take = run > MRD_MAX_RUN ? MRD_MAX_RUN : run;

        /* A token's run must be >=MRD_MIN_RUN or the decoder declines it.
         * Splitting at exactly 65535 can leave a 1..15 tail, so shorten this
         * token just enough to make the tail a legal >=16 run instead. */
        if (run > MRD_MAX_RUN && run - take < MRD_MIN_RUN) {
            take -= MRD_MIN_RUN - (run - take);
        }
        if (dst != 0) {
            dst[0] = (uint8_t)MRD_MARKER;
            zpse_store_u16le(dst + 1u, (uint16_t)take);
            dst += 3u;
        }
        written += 3u;
        run -= take;
    }
    return written;
}

int zpse_mrd_encode(const uint8_t *in, uint32_t n,
                     uint8_t *out, uint32_t out_cap, uint32_t *out_len)
{
    uint32_t prefix, ns, coff = 0u, ne = 0u, nruns = 0u, i;
    uint64_t need;
    uint16_t first;
    uint8_t *cs = 0, *esc = 0;
    uint32_t zero_run = 0u, eoff = 0u;
    int rc = ZPSE_MRD_DECLINE;

    if (out_len != 0) *out_len = 0u;
    if (in == 0 || out == 0 || out_len == 0) return ZPSE_MRD_DECLINE;
    if (n < 4u || (n & 1u) != 0u) return ZPSE_MRD_DECLINE;

    prefix = mrd_prefix(in, n);
    if ((prefix & 1u) != 0u || prefix > n - 2u) return ZPSE_MRD_DECLINE;
    ns = n / 2u - prefix / 2u;
    if (ns < 2u) return ZPSE_MRD_DECLINE;
    if (!mrd_gate(in, prefix, ns)) return ZPSE_MRD_DECLINE;

    /* Pass 1: measure the exact control-stream length and escape count. */
    {
        uint16_t prev = zpse_load_u16le(in + prefix);
        zero_run = 0u;
        for (i = 1u; i < ns; ++i) {
            uint16_t s = zpse_load_u16le(in + prefix + 2u * i);
            int32_t d = mrd_delta(s, prev);

            if (d == 0) {
                ++zero_run;
            } else {
                if (zero_run >= MRD_MIN_RUN) {
                    coff += mrd_emit_run(0, zero_run);
                    ++nruns;
                } else {
                    coff += zero_run; /* literal zero bytes */
                }
                zero_run = 0u;
                if (mrd_fits(d)) {
                    ++coff;
                } else {
                    ++coff; /* escape placeholder 0x00 */
                    ++ne;
                }
            }
            prev = s;
        }
        if (zero_run >= MRD_MIN_RUN) {
            coff += mrd_emit_run(0, zero_run);
            ++nruns;
        } else {
            coff += zero_run;
        }
    }

    need = (uint64_t)ZPSE_MRD_HEADER_SIZE + (uint64_t)prefix +
           (uint64_t)coff + 6u * (uint64_t)ne;
    if (need > (uint64_t)n || (uint64_t)out_cap < need) {
        return ZPSE_MRD_DECLINE;
    }

    cs = (uint8_t *)malloc((size_t)coff + 4u);
    if (ne != 0u) esc = (uint8_t *)malloc((size_t)6u * ne);
    if (cs == 0 || (ne != 0u && esc == 0)) {
        free(cs); free(esc);
        return ZPSE_MRD_DECLINE;
    }

    /* Pass 2: fill the control stream and the sorted escape table. */
    coff = 0u;
    eoff = 0u;
    zero_run = 0u;
    {
        uint16_t prev = zpse_load_u16le(in + prefix);
        for (i = 1u; i < ns; ++i) {
            uint16_t s = zpse_load_u16le(in + prefix + 2u * i);
            int32_t d = mrd_delta(s, prev);

            if (d == 0) {
                ++zero_run;
            } else {
                if (zero_run >= MRD_MIN_RUN) {
                    coff += mrd_emit_run(cs + coff, zero_run);
                } else {
                    uint32_t k;
                    for (k = 0u; k < zero_run; ++k) cs[coff + k] = 0u;
                    coff += zero_run;
                }
                zero_run = 0u;
                if (mrd_fits(d)) {
                    cs[coff++] = (uint8_t)(int8_t)d;
                } else {
                    cs[coff++] = 0u; /* escape placeholder */
                    zpse_store_u32le(esc + eoff, i - 1u);
                    zpse_store_u16le(esc + eoff + 4u, (uint16_t)(int16_t)d);
                    eoff += 6u;
                }
            }
            prev = s;
        }
        if (zero_run >= MRD_MIN_RUN) {
            coff += mrd_emit_run(cs + coff, zero_run);
        } else {
            uint32_t k;
            for (k = 0u; k < zero_run; ++k) cs[coff + k] = 0u;
            coff += zero_run;
        }
    }

    /* Header, verbatim prefix, control stream, escape table. */
    first = zpse_load_u16le(in + prefix);
    zpse_store_u32le(out, ZPSE_MRD_MAGIC);
    zpse_store_u32le(out + 4u, n);
    zpse_store_u16le(out + 8u, first);
    zpse_store_u16le(out + 10u, (uint16_t)prefix);
    zpse_store_u32le(out + 12u, ne);
    zpse_store_u32le(out + 16u, coff);
    zpse_store_u32le(out + 20u, nruns);
    memcpy(out + ZPSE_MRD_HEADER_SIZE, in, prefix);
    memcpy(out + ZPSE_MRD_HEADER_SIZE + prefix, cs, coff);
    if (ne != 0u) {
        memcpy(out + ZPSE_MRD_HEADER_SIZE + prefix + coff, esc, (size_t)6u * ne);
    }

    *out_len = (uint32_t)need;
    rc = ZPSE_MRD_COMMIT;
    free(cs); free(esc);
    return rc;
}

int zpse_mrd_decode(const uint8_t *in, uint32_t in_len,
                     uint8_t *out, uint32_t n)
{
    uint32_t prefix, ns, ne, coff, nruns, didx = 0u, ei = 0u, i;
    uint64_t need;
    uint16_t first;
    int32_t base;
    const uint8_t *cs, *esc, *cp, *cend;

    if (in == 0 || out == 0 || in_len < (uint32_t)ZPSE_MRD_HEADER_SIZE) {
        return ZPSE_MRD_DECLINE;
    }
    if (zpse_load_u32le(in) != ZPSE_MRD_MAGIC) return ZPSE_MRD_DECLINE;
    if (n < 4u || (n & 1u) != 0u) return ZPSE_MRD_DECLINE;
    if (zpse_load_u32le(in + 4u) != n) return ZPSE_MRD_DECLINE;
    prefix = zpse_load_u16le(in + 10u);
    if ((prefix & 1u) != 0u || prefix > n - 2u) return ZPSE_MRD_DECLINE;
    ns = n / 2u - prefix / 2u;
    if (ns < 2u) return ZPSE_MRD_DECLINE;
    ne = zpse_load_u32le(in + 12u);
    if (ne > ns - 1u) return ZPSE_MRD_DECLINE;
    coff = zpse_load_u32le(in + 16u);
    nruns = zpse_load_u32le(in + 20u);
    /* Invariant: every control byte accounts for at least one of the ns-1
     * deltas -- a literal (or escape placeholder) is one byte for one delta,
     * and a 3-byte RLE token stands for >=16. So coff may never exceed ns-1.
     * Without this, a self-consistent header can give the literal loops more
     * bytes than there are deltas, walking didx past the final sample and
     * writing outside `out` before the end-of-stream check gets a say. */
    if (coff > ns - 1u) return ZPSE_MRD_DECLINE;
    need = (uint64_t)ZPSE_MRD_HEADER_SIZE + (uint64_t)prefix +
           (uint64_t)coff + 6u * (uint64_t)ne;
    if (need != (uint64_t)in_len) return ZPSE_MRD_DECLINE;
    if (coff > in_len - ZPSE_MRD_HEADER_SIZE - prefix) return ZPSE_MRD_DECLINE;
    if (nruns > coff / 3u) return ZPSE_MRD_DECLINE;

    esc = in + ZPSE_MRD_HEADER_SIZE + prefix + coff;
    /* Escape positions must be in range and strictly increasing. */
    for (i = 0u; i < ne; ++i) {
        uint32_t p = zpse_load_u32le(esc + (size_t)6u * i);
        if (p >= ns - 1u) return ZPSE_MRD_DECLINE;
        if (i != 0u && p <= zpse_load_u32le(esc + (size_t)6u * (i - 1u))) {
            return ZPSE_MRD_DECLINE;
        }
    }

    first = zpse_load_u16le(in + 8u);
    memcpy(out, in + ZPSE_MRD_HEADER_SIZE, prefix); /* verbatim prefix */
    zpse_store_u16le(out + prefix, first);
    base = (int32_t)(int16_t)first;
    cs = in + ZPSE_MRD_HEADER_SIZE + prefix;
    cend = cs + coff;
    cp = cs;

    while (cp < cend) {
        if (*cp == (uint8_t)MRD_MARKER) {
            uint32_t run, k;

            if ((size_t)(cend - cp) < 3u) return ZPSE_MRD_DECLINE;
            run = zpse_load_u16le(cp + 1u);
            cp += 3u;
            if (run < MRD_MIN_RUN || run > ns - 1u - didx) {
                return ZPSE_MRD_DECLINE;
            }
#ifdef MRD_SIMD
            {
                __m128i pat = _mm_set1_epi16((int16_t)base);
                uint8_t *o = out + prefix + (size_t)2u * (didx + 1u);

                for (k = 0u; k + 8u <= run; k += 8u) {
                    _mm_storeu_si128((__m128i *)(o + (size_t)2u * k), pat);
                }
                for (; k < run; ++k) {
                    zpse_store_u16le(o + (size_t)2u * k, (uint16_t)base);
                }
            }
#else
            {
                uint8_t *o = out + prefix + (size_t)2u * (didx + 1u);
                for (k = 0u; k < run; ++k) {
                    zpse_store_u16le(o + (size_t)2u * k, (uint16_t)base);
                }
            }
#endif
            didx += run;
            continue;
        }

        /* A stretch of literal delta bytes; vectorise in 8-lane blocks. */
        {
            uint32_t avail = 0u;
            while (cp + avail < cend && avail < 8u &&
                   cp[avail] != (uint8_t)MRD_MARKER) {
                ++avail;
            }
            /* Each of these bytes emits one sample, starting at didx+1. Even
             * with coff <= ns-1 an RLE token earlier in the stream can leave
             * didx close to ns-1, so bound this stretch by the deltas left
             * before any vector store (which always emits eight). */
            if (avail > ns - 1u - didx) return ZPSE_MRD_DECLINE;
#ifdef MRD_SIMD
            if (avail == 8u) {
                __m128i raw, sign, v;

                /* One store emits eight samples, so all eight output slots
                 * must still be inside the ns samples. */
                if (ns - 1u - didx < 8u) return ZPSE_MRD_DECLINE;
                raw = _mm_loadl_epi64((const __m128i *)cp);
                sign = _mm_cmplt_epi8(raw, _mm_setzero_si128());
                v = _mm_unpacklo_epi8(raw, sign);

                while (ei < ne && zpse_load_u32le(esc + (size_t)6u * ei) < didx + 8u) {
                    uint32_t p = zpse_load_u32le(esc + (size_t)6u * ei);

                    if (p >= didx) {
                        int16_t tmp[8];
                        int16_t ed = (int16_t)zpse_load_u16le(esc + (size_t)6u * ei + 4u);

                        _mm_storeu_si128((__m128i *)tmp, v);
                        tmp[p - didx] = ed;
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
                    _mm_storeu_si128((__m128i *)(out + prefix + (size_t)2u * (didx + 1u)), v);
                    base = (int32_t)(int16_t)(base + sum);
                }
                cp += 8u;
                didx += 8u;
            } else
#endif
            {
                uint32_t k;
                for (k = 0u; k < avail; ++k) {
                    int32_t d;

                    /* Per-store backstop: didx+1 is the sample slot written
                     * below, and only slots 0..ns-1 exist. */
                    if (didx >= ns - 1u) return ZPSE_MRD_DECLINE;
                    d = (int32_t)(int8_t)cp[k];
                    while (ei < ne && zpse_load_u32le(esc + (size_t)6u * ei) == didx) {
                        d = (int16_t)zpse_load_u16le(esc + (size_t)6u * ei + 4u);
                        ++ei;
                    }
                    base = (int32_t)(int16_t)(base + d);
                    zpse_store_u16le(out + prefix + (size_t)2u * (didx + 1u),
                                     (uint16_t)base);
                    ++didx;
                }
                cp += avail;
            }
        }
    }

    if (didx != ns - 1u || ei != ne) return ZPSE_MRD_DECLINE;
    return ZPSE_MRD_COMMIT;
}
