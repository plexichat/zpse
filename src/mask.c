#include "zpse/mask.h"
#include "zpse/combinadic.h"
#include "zpse/status.h"

#include <string.h>

/* Unity-build safe: only static helpers plus the public API. No
 * allocation, no globals, C11. */

static void zpse_mask_set_at(uint32_t words[8], uint16_t pos)
{
    if (pos >= 256u) {
        return; /* words[] has 8 entries = 256 bits; pos must be < nt <= 256 */
    }
    words[pos >> 5] |= (uint32_t)1u << (pos & 31u);
}

/* Set every bit of the range [from, from+len).
 *
 * The RUNS decoder knows each miss run as one length, and a near-dense
 * mask -- 253 exceptions in 256 positions, which is three hit bytes and
 * seven bytes of payload -- was previously written one position at a time
 * through zpse_mask_set_at: a load, OR and store to the same word for
 * every position, so each bit paid the store-to-load latency of the one
 * before it (measured ~5 cycles per exception, 56% of that shape's whole
 * decode). Setting a whole run with a word-at-a-time OR does the same
 * eight or sixteen bits in one store, and the caller's `sum <= nt` check
 * means every bit it sets is still below nt exactly as before. */
static void zpse_mask_set_range(uint32_t words[8], uint32_t from,
                                uint32_t len)
{
    uint32_t wi = from >> 5u;
    uint32_t fi = from & 31u;
    uint32_t bits;

    if (len == 0u) {
        return;
    }
    if (fi != 0u) {
        bits = 32u - fi;
        if (bits > len) {
            bits = len;
        }
        words[wi] |= (uint32_t)(((UINT64_C(1) << bits) - 1u) << fi);
        wi++;
        len -= bits;
        if (len == 0u) {
            return;
        }
    }
    while (len >= 32u) {
        words[wi] |= UINT32_C(0xFFFFFFFF);
        wi++;
        len -= 32u;
    }
    if (len != 0u) {
        words[wi] |= (uint32_t)((UINT64_C(1) << len) - 1u);
    }
}

/* Population count on one dense word. The builtin is available on both
 * compilers this codebase targets through gcc/clang; the fallback keeps
 * the file buildable elsewhere without changing the result. */
static int zpse_mask_popcount32(uint32_t w)
{
    /* Branch-free SWAR popcount. __builtin_popcount lowers to a libgcc
     * call (__popcountdi2) unless the whole build is compiled with
     * -mpopcnt, which this portable codebase is not. */
    w = w - ((w >> 1) & 0x55555555u);
    w = (w & 0x33333333u) + ((w >> 2) & 0x33333333u);
    w = (w + (w >> 4)) & 0x0F0F0F0Fu;
    return (int)((w * 0x01010101u) >> 24);
}

/* Index of the lowest set bit. Caller guarantees w != 0. */
static unsigned zpse_mask_ctz32(uint32_t w)
{
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_ctz(w);
#else
    unsigned n = 0;

    while ((w & 1u) == 0u) {
        w >>= 1;
        n++;
    }
    return n;
#endif
}

static int zpse_mask_nt_ok(uint16_t nt)
{
    return nt >= 1 && nt <= 256;
}

/* Collect ascending exception positions. Returns count (<= nt). */
static uint16_t zpse_mask_collect(const uint32_t words[8], uint16_t nt,
                                  uint16_t *out /* [256] */)
{
    uint16_t n = 0;
    uint16_t p;

    for (p = 0; p < nt; p++) {
        if (zpse_mask_test(words, p))
            out[n++] = p;
    }
    return n;
}

int zpse_mask_derive_k(uint8_t mask_mode, uint8_t exception_count,
                       uint16_t nt, uint16_t *k_out)
{
    uint16_t k;

    if (k_out == 0)
        return PSE_BAD_ARG;
    if (!zpse_mask_nt_ok(nt))
        return PSE_BAD_MASK;
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (exception_count != 0)
            return PSE_BAD_MASK;
        *k_out = 0;
        return PSE_OK;
    case ZPSE_M_ALL_EXCEPTION:
        k = (exception_count == 0) ? nt : (uint16_t)exception_count;
        if (k != nt)
            return PSE_BAD_MASK;
        *k_out = k;
        return PSE_OK;
    case ZPSE_M_ASCII_PACK:
        /* No mask: the form has no exceptions. K is 0, and the tile's
         * residual payload is the packed bytes (tile.c owns the length). */
        if (exception_count != 0)
            return PSE_BAD_MASK;
        *k_out = 0;
        return PSE_OK;
    case ZPSE_M_RAW_BITMAP:
    case ZPSE_M_SPARSE_U8:
    case ZPSE_M_SPARSE_GAPS:
    case ZPSE_M_RUNS:
    case ZPSE_M_ENUMERATIVE:
        k = (uint16_t)exception_count;
        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        *k_out = k;
        return PSE_OK;
    default:
        return PSE_BAD_MASK;
    }
}

void zpse_mask_clear(uint32_t words[8])
{
    int i;

    if (words == 0)
        return;
    for (i = 0; i < 8; i++)
        words[i] = 0;
}

int zpse_mask_test(const uint32_t words[8], uint16_t pos)
{
    if (words == 0 || pos >= 256)
        return 0;
    return (int)((words[pos >> 5] >> (pos & 31u)) & 1u);
}

/* 256-bit population count in one pass, for the modes whose count has to
 * agree with the header on every tile (RAW_BITMAP decodes 32 payload bytes
 * and then has to prove they hold exactly k bits; eight SWAR popcounts
 * measured ~95 cycles of that tile's ~1125, which is why the dense path
 * gets a vectorised count). The bit-parallel fold needs only AND/ADD/SHIFT,
 * so SSE2 -- part of the x86-64 baseline -- is enough; SAD then sums the
 * four 64-bit lanes. Same per-function target structure as the other SIMD
 * helpers in this tree, and the scalar loop stays the fallback both when
 * the target is compiled out and when the CPU lacks the level. */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define ZPSE_MASK_SIMD __attribute__((target("sse2")))
#define ZPSE_MASK_HAVE_SIMD 1
#elif defined(_M_X64)
#define ZPSE_MASK_SIMD
#define ZPSE_MASK_HAVE_SIMD 1
#else
#define ZPSE_MASK_HAVE_SIMD 0
#endif
#else
#define ZPSE_MASK_HAVE_SIMD 0
#endif

#if ZPSE_MASK_HAVE_SIMD
#include <emmintrin.h>

/* Count of the one bits in one 16-byte half. Straight-line on purpose:
 * the same fold written as a four-iteration loop was miscompiled by the
 * MinGW GCC used for the Windows builds -- it summed stale lanes and
 * returned 20..26 bits for a mask holding one -- while this unrolled form
 * is exactly what the scalar count says. Two halves per call, no loop. */
static ZPSE_MASK_SIMD int zpse_mask_popcount16_sse2(__m128i v)
{
    const __m128i m1 = _mm_set1_epi8((char)0x55);
    const __m128i m2 = _mm_set1_epi8((char)0x33);
    const __m128i m4 = _mm_set1_epi8((char)0x0F);
    __m128i s;

    v = _mm_sub_epi8(v, _mm_and_si128(_mm_srli_epi16(v, 1), m1));
    v = _mm_add_epi8(_mm_and_si128(v, m2),
                     _mm_and_si128(_mm_srli_epi16(v, 2), m2));
    v = _mm_and_si128(_mm_add_epi8(v, _mm_srli_epi16(v, 4)), m4);
    s = _mm_sad_epu8(v, _mm_setzero_si128());
    return (int)_mm_cvtsi128_si32(s) +
           (int)_mm_cvtsi128_si32(_mm_srli_si128(s, 8));
}

static ZPSE_MASK_SIMD int zpse_mask_popcount256_sse2(const uint32_t words[8])
{
    const unsigned char *b = (const unsigned char *)(const void *)words;

    return zpse_mask_popcount16_sse2(
               _mm_loadu_si128((const __m128i *)(const void *)(b + 0u))) +
           zpse_mask_popcount16_sse2(
               _mm_loadu_si128((const __m128i *)(const void *)(b + 16u)));
}
#endif /* ZPSE_MASK_HAVE_SIMD */

int zpse_mask_count_bits(const uint32_t words[8], uint16_t nt)
{
    int n = 0;
    int i;

    if (words == 0 || nt == 0)
        return 0;
    if (nt > 256)
        nt = 256;
#if ZPSE_MASK_HAVE_SIMD
    /* A full 256-position mask needs no trimming, so the copy, the
     * zero-fill and the final AND below are all no-ops for it -- and a full
     * tile is what every dense shape has. Counting the caller's words
     * directly is one 32-byte read instead of a read, a copy and two
     * writes; this runs once per packed tile (the encoder re-checks the
     * count it is about to size the mask from). */
    if (nt == 256u)
        return zpse_mask_popcount256_sse2(words);
    {
        /* Trim the bits at or above nt in a local copy (the caller's
         * array is read-only here), then one vector pass counts the lot:
         * a short final tile costs a 32-byte copy, not eight SWAR
         * popcounts. SSE2 is part of the x86-64 baseline, so no runtime
         * capability check is needed -- the compile-time test above is the
         * only gate. */
        uint32_t w[8];
        uint16_t last = (uint16_t)((nt - 1u) >> 5);
        uint32_t keep = (uint32_t)(((uint64_t)1u << (((nt - 1u) & 31u) + 1u))
                                   - 1u);
        int k;

        memcpy(w, words, sizeof w);
        for (k = (int)last + 1; k < 8; k++)
            w[k] = 0u;
        w[last] &= keep;
        return zpse_mask_popcount256_sse2(w);
    }
#endif
    for (i = 0; i < 8; i++) {
        uint32_t w = words[i];
        uint16_t base = (uint16_t)(i * 32);
        uint32_t m;

        if (base >= nt) {
            break;
        } else if (nt - base < 32) {
            m = (uint32_t)(((uint64_t)1u << (nt - base)) - 1u);
            w &= m;
        }
        n += zpse_mask_popcount32(w);
    }
    return n;
}

int zpse_mask_decode(uint8_t mask_mode, const uint8_t *payload,
                     uint16_t payload_len, uint16_t nt, uint16_t k,
                     uint32_t words[8], uint8_t *comb_scratch)
{
    int i;

    if (words == 0)
        return PSE_BAD_ARG;
    for (i = 0; i < 8; i++)
        words[i] = 0;
    if (!zpse_mask_nt_ok(nt))
        return PSE_BAD_MASK;
    if (payload_len > 0 && payload == 0)
        return PSE_BAD_ARG;

    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (k != 0 || payload_len != 0)
            return PSE_BAD_MASK;
        return PSE_OK;
    case ZPSE_M_ALL_EXCEPTION: {
        uint16_t p;

        if (k != nt || payload_len != 0)
            return PSE_BAD_MASK;
        for (p = 0; p < nt; p++)
            zpse_mask_set_at(words, p);
        return PSE_OK;
    }
    case ZPSE_M_RAW_BITMAP: {
        uint16_t need;
        unsigned byte_i;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        need = (uint16_t)((nt + 7u) >> 3);
        if (payload_len != need)
            return PSE_BAD_MASK;
        /* The dense mask IS a 256-bit little-endian bitmap: payload
         * byte b carries bits 8b..8b+7, which is exactly byte lane
         * (b & 3) of words[b >> 2]. `words` was zeroed above, so byte
         * lane b of word b>>2 is zero before the OR and the four bytes
         * of a whole word can simply be assembled into one store instead
         * of four read-modify-writes with a variable shift each. Only
         * whole words are assembled that way; the tail (at most three
         * bytes) keeps the byte loop, so no byte past the payload -- whose
         * length is exactly `need` here -- is ever read. Identical bytes
         * either way, and lanes past `need` stay zero. */
        for (byte_i = 0; byte_i + 4u <= (unsigned)need; byte_i += 4u) {
            words[byte_i >> 2] =
                (uint32_t)payload[byte_i] |
                ((uint32_t)payload[byte_i + 1u] << 8) |
                ((uint32_t)payload[byte_i + 2u] << 16) |
                ((uint32_t)payload[byte_i + 3u] << 24);
        }
        for (; byte_i < (unsigned)need; byte_i++) {
            words[byte_i >> 2] |=
                (uint32_t)payload[byte_i] << ((byte_i & 3u) * 8u);
        }
        /* Bits >= nt sit in the high bits of the final byte and MUST be
         * zero (Sec. 8). The old scan ran p from nt to need*8 and is
         * therefore only ever non-empty when nt is not a multiple of 8. */
        if ((nt & 7u) != 0u && (payload[need - 1u] >> (nt & 7u)) != 0u)
            return PSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return PSE_BAD_MASK;
        return PSE_OK;
    }
    case ZPSE_M_SPARSE_U8: {
        uint16_t j;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (payload_len != k)
            return PSE_BAD_MASK;
        {
            /* The ascending-order test compares against the previous
             * decoded position. Keeping it in a local removes the reload
             * of payload[j - 1] per iteration and cannot change any
             * decision: the loop has already returned unless every
             * earlier byte was < nt <= 256, so the local holds exactly
             * the byte the reload would have produced. */
            uint16_t prev = 0;

            for (j = 0; j < k; j++) {
                uint16_t v = payload[j];

                if (v >= nt)
                    return PSE_BAD_MASK;
                if (j > 0 && v <= prev)
                    return PSE_BAD_MASK;
                prev = v;
                zpse_mask_set_at(words, v);
            }
        }
        return PSE_OK;
    }
    case ZPSE_M_SPARSE_GAPS: {
        uint8_t flag;
        uint16_t need;
        uint32_t pos = 0;
        uint16_t j;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (payload_len < 1)
            return PSE_BAD_MASK;
        flag = payload[0];
        if (flag > 1)
            return PSE_BAD_MASK;
        need = flag ? (uint16_t)(1u + 2u * (uint16_t)k)
                    : (uint16_t)(1u + (uint16_t)k);
        if (payload_len != need)
            return PSE_BAD_MASK;
        for (j = 0; j < k; j++) {
            uint32_t gap;

            if (flag)
                gap = (uint32_t)payload[1 + 2 * j] |
                      ((uint32_t)payload[1 + 2 * j + 1] << 8);
            else
                gap = payload[1 + j];
            if (j == 0)
                pos = gap;
            else
                pos = pos + 1u + gap;
            if (pos >= nt)
                return PSE_BAD_MASK;
            zpse_mask_set_at(words, (uint16_t)pos);
        }
        return PSE_OK;
    }
    case ZPSE_M_RUNS: {
        uint32_t sum = 0;
        uint32_t miss = 0;
        uint32_t pos = 0;
        uint16_t r;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (payload_len < 2)
            return PSE_BAD_MASK;
        for (r = 0; r < payload_len; r++) {
            uint32_t len = payload[r];

            sum += len;
            if (sum > nt)
                return PSE_BAD_MASK;
            if (r & 1u) {
                miss += len;
                if (miss > k)
                    return PSE_BAD_MASK;
                zpse_mask_set_range(words, pos, len);
            }
            pos += len;
        }
        if (sum != nt || miss != k)
            return PSE_BAD_MASK;
        return PSE_OK;
    }
    case ZPSE_M_ENUMERATIVE:
        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (payload_len == 0)
            return PSE_BAD_MASK;
        return zpse_comb_unrank(nt, k, payload, payload_len, words,
                               comb_scratch);
    default:
        return PSE_BAD_MASK;
    }
}

int zpse_mask_encode(uint8_t mask_mode, const uint32_t words[8],
                     uint16_t nt, uint16_t k, uint8_t *payload_out,
                     uint16_t payload_cap, uint8_t *comb_scratch)
{
    uint16_t c[256];
    uint16_t nc;
    uint16_t j;

    if (words == 0)
        return PSE_BAD_ARG;
    if (!zpse_mask_nt_ok(nt))
        return PSE_BAD_MASK;

    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (k != 0)
            return PSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != 0)
            return PSE_BAD_MASK;
        return 0;
    case ZPSE_M_ALL_EXCEPTION:
        if (k != nt)
            return PSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)nt)
            return PSE_BAD_MASK;
        return 0;
    case ZPSE_M_RAW_BITMAP: {
        uint16_t need;
        unsigned byte_i;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return PSE_BAD_MASK;
        need = (uint16_t)((nt + 7u) >> 3);
        if (payload_cap < need)
            return PSE_OUTPUT_OVERFLOW;
        if (need > 0 && payload_out == 0)
            return PSE_BAD_ARG;
        /* Little-endian serialisation of the dense words: payload byte b
         * is byte lane (b & 3) of words[b >> 2]. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        memcpy(payload_out, words, (size_t)need);
        (void)byte_i;
#else
        for (byte_i = 0; byte_i < (unsigned)need; byte_i++) {
            payload_out[byte_i] =
                (uint8_t)(words[byte_i >> 2] >> ((byte_i & 3u) * 8u));
        }
#endif
        /* Pad bits (>= nt) MUST be zero on the wire. The caller's array
         * may hold stray bits above nt, which zpse_mask_count_bits()
         * deliberately ignores, so clear them explicitly rather than
         * relying on them being absent. */
        if ((nt & 7u) != 0u) {
            payload_out[need - 1u] &= (uint8_t)((1u << (nt & 7u)) - 1u);
        }
        return (int)need;
    }
    case ZPSE_M_SPARSE_U8: {
        /* One __builtin_ctz step per exception instead of a full
         * 0..nt-1 scan, with the same masking zpse_mask_count_bits()
         * applies so positions >= nt can never be reported. */
        unsigned wi;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        nc = 0;
        for (wi = 0; wi < 8u; wi++) {
            uint32_t w = words[wi];
            uint16_t base = (uint16_t)(wi * 32u);

            if (base >= nt)
                break;
            if (nt - base < 32) {
                w &= (uint32_t)(((uint64_t)1u << (nt - base)) - 1u);
            }
            while (w != 0u) {
                c[nc++] = (uint16_t)(base + (uint16_t)zpse_mask_ctz32(w));
                w &= w - 1u;
            }
        }
        if (nc != k)
            return PSE_BAD_MASK;
        if (payload_cap < k)
            return PSE_OUTPUT_OVERFLOW;
        if (k > 0 && payload_out == 0)
            return PSE_BAD_ARG;
        for (j = 0; j < k; j++)
            payload_out[j] = (uint8_t)c[j];
        return (int)k;
    }
    case ZPSE_M_SPARSE_GAPS: {
        int wide = 0;
        uint16_t need;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        nc = zpse_mask_collect(words, nt, c);
        if (nc != k)
            return PSE_BAD_MASK;
        for (j = 0; j < k; j++) {
            uint32_t gap = (j == 0) ? c[0]
                                    : (uint32_t)(c[j] - c[j - 1] - 1u);

            if (gap > 255u) {
                wide = 1;
                break;
            }
        }
        need = wide ? (uint16_t)(1u + 2u * (uint16_t)k)
                    : (uint16_t)(1u + (uint16_t)k);
        if (payload_cap < need)
            return PSE_OUTPUT_OVERFLOW;
        if (payload_out == 0)
            return PSE_BAD_ARG;
        payload_out[0] = wide ? 1u : 0u;
        for (j = 0; j < k; j++) {
            uint32_t gap = (j == 0) ? c[0]
                                    : (uint32_t)(c[j] - c[j - 1] - 1u);

            if (wide) {
                payload_out[1 + 2 * j] = (uint8_t)(gap & 0xFFu);
                payload_out[1 + 2 * j + 1] = (uint8_t)((gap >> 8) & 0xFFu);
            } else {
                payload_out[1 + j] = (uint8_t)gap;
            }
        }
        return (int)need;
    }
    case ZPSE_M_RUNS: {
        /* Logical runs alternate starting with a hit run; a leading
         * zero-length hit run covers masks starting with an exception.
         * Runs longer than 255 are split with zero-length alternates,
         * e.g. 300 hits -> [255, 0, 45].
         *
         * The runs are the gaps between transition positions, not a
         * bit-at-a-time scan: with `prev` as the bit below the word, bit p
         * of (w ^ (w << 1 | prev)) is set exactly when p differs from
         * p - 1, and bit 0 of the first word is the comparison against the
         * implicit hit below the tile -- so a mask that opens with an
         * exception yields a transition at 0 and therefore a leading
         * zero-length run, which is the shape the sequence needs. Eight
         * word operations plus one ctz per transition replace 256 bit
         * tests and the alternating branch that follows them (measured:
         * the bit scan cost ~640 cycles per record, 18% of a tile). */
        /* The edge list is the only array this needs: a run length is the
         * difference of two consecutive edges, so the second array of run
         * lengths -- and the pass that filled it -- is gone. The price still
         * has to be known before the first payload byte is written, but it
         * is a property of the list rather than a second walk over it: every
         * run is exactly one byte (see below), so the payload is `nb - 1`
         * bytes and the sizing walk is gone with the split loop it existed
         * to pay for. Neither the list nor the mask words are re-read. */
        uint16_t edge[258];
        int nb = 0;
        uint32_t need;
        int r;
        uint16_t o = 0;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return PSE_BAD_MASK;
        {
            uint32_t prev = 0u;
            int wi;

            edge[nb++] = 0u;
            for (wi = 0; wi < 8; wi++) {
                uint32_t w = words[wi];
                uint32_t keep = ~0u;
                uint32_t trans;

                if (wi * 32 >= (int)nt)
                    break;
                if ((wi + 1) * 32 > (int)nt) {
                    uint32_t rem = (uint32_t)nt - (uint32_t)wi * 32u;

                    keep = (1u << rem) - 1u;
                    w &= keep;
                }
                /* Keep only transitions inside the tile: shifting up sets a
                 * bit one past the last valid one, and that bit is at nt.
                 */
                trans = (w ^ ((w << 1) | prev)) & keep;
                while (trans != 0u) {
                    unsigned b = zpse_mask_ctz32(trans);

                    if (nb >= 257)
                        return PSE_BAD_MASK;
                    edge[nb++] = (uint16_t)((uint32_t)wi * 32u + b);
                    trans &= trans - 1u;
                }
                prev = w >> 31;
            }
            edge[nb++] = nt;
        }
        /* Every run is one byte, so the payload is exactly `nb - 1` bytes.
         *
         * A run longer than 255 would have to be split as the 255, 0, rest
         * form, and that cannot happen here: `nt <= 256` (the mask contract),
         * and the k guard above keeps at least one hit and at least one
         * exception, so no run of one value covers the whole tile and the
         * longest possible run is `nt - 1 <= 255`. The split loop therefore
         * never ran, and the sizing walk that existed to price it was
         * arithmetic on a constant. */
        need = (uint32_t)(nb - 1);
        if (need > payload_cap)
            return PSE_OUTPUT_OVERFLOW;
        if (need > 0 && payload_out == 0)
            return PSE_BAD_ARG;
        for (r = 0; r + 1 < nb; r++) {
            payload_out[o++] = (uint8_t)(edge[r + 1] - edge[r]);
        }
        return (int)o;
    }
    case ZPSE_M_ENUMERATIVE:
        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        nc = zpse_mask_collect(words, nt, c);
        if (nc != k)
            return PSE_BAD_MASK;
        return zpse_comb_rank_to_bytes(c, k, nt, payload_out, payload_cap);
    default:
        break;
    }
    (void)comb_scratch;
    return PSE_BAD_MASK;
}
