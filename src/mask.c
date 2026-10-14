#include "zpse/mask.h"
#include "zpse/combinadic.h"
#include "zpse/status.h"

/* Unity-build safe: only static helpers plus the public API. No
 * allocation, no globals, C11. */

static void zpse_mask_set_at(uint32_t words[8], uint16_t pos)
{
    words[pos >> 5] |= (uint32_t)1u << (pos & 31u);
}

/* Population count on one dense word. The builtin is available on both
 * compilers this codebase targets through gcc/clang; the fallback keeps
 * the file buildable elsewhere without changing the result. */
static int zpse_mask_popcount32(uint32_t w)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(w);
#else
    int n = 0;

    while (w != 0u) {
        w &= w - 1u;
        n++;
    }
    return n;
#endif
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

int zpse_mask_count_bits(const uint32_t words[8], uint16_t nt)
{
    int n = 0;
    int i;

    if (words == 0 || nt == 0)
        return 0;
    if (nt > 256)
        nt = 256;
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
            uint32_t t;

            sum += len;
            if (sum > nt)
                return PSE_BAD_MASK;
            if (r & 1u) {
                miss += len;
                if (miss > k)
                    return PSE_BAD_MASK;
                for (t = 0; t < len; t++)
                    zpse_mask_set_at(words, (uint16_t)(pos + t));
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
        for (byte_i = 0; byte_i < (unsigned)need; byte_i++) {
            payload_out[byte_i] =
                (uint8_t)(words[byte_i >> 2] >> ((byte_i & 3u) * 8u));
        }
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
         * e.g. 300 hits -> [255, 0, 45]. */
        uint16_t runs[257];
        int nruns = 0;
        uint16_t p = 0;
        int want_miss = 0;
        uint32_t need = 0;
        int r;
        uint16_t o = 0;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return PSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return PSE_BAD_MASK;
        while (p < nt) {
            int cls = zpse_mask_test(words, p);

            if (cls != want_miss) {
                if (nruns >= 257)
                    return PSE_BAD_MASK;
                runs[nruns++] = 0;
                want_miss = !want_miss;
                continue;
            } else {
                uint16_t q = (uint16_t)(p + 1u);

                while (q < nt && zpse_mask_test(words, q) == cls)
                    q++;
                if (nruns >= 257)
                    return PSE_BAD_MASK;
                runs[nruns++] = (uint16_t)(q - p);
                p = q;
                want_miss = !want_miss;
            }
        }
        for (r = 0; r < nruns; r++) {
            uint32_t L = runs[r];

            if (L == 0) {
                need += 1;
            } else {
                uint32_t chunks = (L + 254u) / 255u;

                need += 2u * chunks - 1u;
            }
        }
        if (need > payload_cap)
            return PSE_OUTPUT_OVERFLOW;
        if (need > 0 && payload_out == 0)
            return PSE_BAD_ARG;
        for (r = 0; r < nruns; r++) {
            uint32_t L = runs[r];

            if (L == 0) {
                payload_out[o++] = 0;
            } else {
                uint32_t rem = L;

                while (rem > 255u) {
                    payload_out[o++] = 255;
                    payload_out[o++] = 0;
                    rem -= 255u;
                }
                payload_out[o++] = (uint8_t)rem;
            }
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
