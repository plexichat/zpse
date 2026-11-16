#include "zpse/pagefast.h"

#include <stddef.h>
#include <string.h>

#include "zpse/hot.h"
#include "zpse/vector.h"

/* Runtime-dispatched 256-bit kernel. Same shape as src/dense.c: a per-function
 * target attribute lets this translation unit stay at the baseline ISA, and
 * ZPSE_AVX2_ABI_OK keeps the kernel out of builds that cannot pass a 32-byte
 * vector safely (see zpse/hot.h). Every __m256i here is local to one function,
 * so there is no by-value vector argument at a call boundary. */
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && \
    ZPSE_PAGEFAST_AVX2
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_PF_AVX2 __attribute__((target("avx2")))
#define ZPSE_PF_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_PF_AVX2
#define ZPSE_PF_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_PF_HAVE_AVX2 0
#endif
#else
#define ZPSE_PF_HAVE_AVX2 0
#endif

#if ZPSE_PF_HAVE_AVX2
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ZPSE_PF_CTZ32(x) ((uint32_t)__builtin_ctz(x))
#else
static uint32_t pf_ctz32(uint32_t x)
{
    uint32_t c = 0u;
    while ((x & 1u) == 0u) { x >>= 1; c++; }
    return c;
}
#define ZPSE_PF_CTZ32(x) pf_ctz32(x)
#endif

/* The one accumulator block both kernels fill. The scalar kernel writes the
 * byte arrays directly; the AVX2 kernel keeps the vectors in registers for
 * the whole page and stores them once at the end. pf_finish() is shared, so
 * the two paths are bit-identical by construction.
 *
 *   ref[j]        first byte seen at vector-lane position j (j = offset % 32)
 *   lane_diff[j]  OR of (byte ^ ref[j]) over the page; 0 means that position
 *                 held the same byte in every lane (the POINTER signal)
 *   zero_cnt[j]   number of zero bytes at position j, max 128 per lane
 *   abits[8]      the 256-bit alphabet bitmap: bit v of word v>>5 == value seen
 */
typedef struct ZpsePfAcc {
    uint8_t  alpha[16];     /* first-appearance order, zero-filled past nalpha */
    uint32_t abits[8];      /* 256-bit alphabet bitmap */
    uint8_t  nalpha;
    uint8_t  alpha_over;    /* a 17th distinct value was seen */
    uint8_t  ref[32];
    uint8_t  lane_diff[32];
    uint8_t  zero_cnt[32];
    uint32_t nzero;
    uint8_t  high_or;       /* OR of every byte & 0x80 */
    uint8_t  first;
} ZpsePfAcc;

static void pf_acc_init(ZpsePfAcc *a, const uint8_t *in, uint32_t n)
{
    uint32_t i, k;

    memset(a, 0, sizeof(*a));
    k = n < 32u ? n : 32u;
    for (i = 0u; i < k; i++) {
        a->ref[i] = in[i];
    }
    a->first = k != 0u ? in[0] : 0u;
}

/* Append a newly discovered value to the first-appearance alphabet. */
static void pf_record(ZpsePfAcc *a, uint8_t v)
{
    uint32_t i;

    for (i = 0u; i < a->nalpha; i++) {
        if (a->alpha[i] == v) {
            return;
        }
    }
    if (a->nalpha >= 16u) {
        a->alpha_over = 1u;
        return;
    }
    a->alpha[a->nalpha++] = v;
}

static void pf_add_byte(ZpsePfAcc *a, uint8_t b, uint32_t lane)
{
    if (b == 0u) {
        a->zero_cnt[lane] = (uint8_t)(a->zero_cnt[lane] + 1u);
    }
    a->lane_diff[lane] |= (uint8_t)(b ^ a->ref[lane]);
    a->high_or |= (uint8_t)(b & 0x80u);
    if (!a->alpha_over) {
        uint32_t word = (uint32_t)(b >> 5);
        uint32_t bit = (uint32_t)1u << (b & 31u);
        if ((a->abits[word] & bit) == 0u) {
            a->abits[word] |= bit;
            pf_record(a, b);
        }
    }
}

static uint32_t pf_popcount32(uint32_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcount(x);
#else
    uint32_t c = 0u;
    while (x != 0u) { x &= x - 1u; c++; }
    return c;
#endif
}

/* A page looks like an array of pointers when, for one of the 8/4/2-byte
 * strides, the upper half of every lane is held constant across the page
 * while the lower half varies. A zero high byte is allowed to be merely
 * "mostly" zero (7 of 8 bytes), which covers pages where one lane carries
 * an occasional non-canonical value. */
static int pf_looks_pointer(const ZpsePfAcc *a, uint32_t n)
{
    static const uint32_t strides[3] = { 8u, 4u, 2u };
    uint32_t si;

    for (si = 0u; si < 3u; si++) {
        uint32_t s = strides[si];
        uint32_t p, bad = 0u, good = 0u, varies = 0u;
        /* A 64-bit address commonly has three constant high bytes and one
         * varying one (0x00007fXX........); allow that one exception at 8. */
        uint32_t max_bad = s >= 8u ? 1u : 0u;

        if (n < 4u * s) {
            continue;
        }
        for (p = 0u; p < s; p++) {
            uint32_t j, mismatch = 0u;

            for (j = p; j < 32u; j += s) {
                mismatch |= (uint32_t)a->lane_diff[j];
            }
            if (p >= s / 2u) {
                if (mismatch == 0u) {
                    good++;
                } else if (a->ref[p] == 0u) {
                    uint32_t z = 0u, lanes = 0u;
                    for (j = p; j < 32u; j += s) {
                        z += (uint32_t)a->zero_cnt[j];
                        if (j < n) {
                            lanes++;
                        }
                    }
                    if (z * 8u >= lanes * 7u) {
                        good++;
                    } else {
                        bad++;
                    }
                } else {
                    bad++;
                }
            } else if (mismatch != 0u) {
                varies = 1u;
            }
        }
        if (good + max_bad >= s / 2u && bad <= max_bad && varies != 0u) {
            return 1;
        }
    }
    return 0;
}

static int pf_finish(const ZpsePfAcc *a, uint32_t n, ZpsePageInfo *info)
{
    uint32_t i, bits = 0u, nalpha;

    info->value = a->first;
    info->flags = 0u;
    info->nzero = (uint16_t)a->nzero;
    /* Publish the per-lane stats from the shared accumulator. Both kernels
     * have finished filling a->lane_diff/zero_cnt by now (AVX2 stores them
     * after its main loop, scalar writes them via pf_add_byte), so copying
     * here keeps the two paths bit-identical without a second scan. */
    memcpy(info->lane_diff, a->lane_diff, 32u);
    memcpy(info->zero_cnt, a->zero_cnt, 32u);
    for (i = 0u; i < 8u; i++) {
        bits += pf_popcount32(a->abits[i]);
    }
    nalpha = a->alpha_over != 0u ? 0u : bits;
    info->nalpha = (uint8_t)nalpha;
    if (a->alpha_over == 0u) {
        memcpy(info->alpha, a->alpha, 16u);
    } else {
        memset(info->alpha, 0, 16u);
    }

    if (n == 0u || a->nzero == n) {
        return ZPSE_PAGE_ZERO;
    }
    {
        uint32_t j, same = 1u, k = n < 32u ? n : 32u;

        for (j = 1u; j < k; j++) {
            if (a->ref[j] != a->ref[0]) {
                same = 0u;
                break;
            }
        }
        if (same != 0u) {
            for (j = 0u; j < 32u; j++) {
                if (a->lane_diff[j] != 0u) {
                    same = 0u;
                    break;
                }
            }
            if (same != 0u) {
                return ZPSE_PAGE_ONE_VALUE;
            }
        }
    }
    if (a->alpha_over == 0u && nalpha <= 16u) {
        return ZPSE_PAGE_LOW_ALPHA;
    }
    if (a->high_or == 0u) {
        return ZPSE_PAGE_ASCII;
    }
    if (pf_looks_pointer(a, n) != 0) {
        return ZPSE_PAGE_POINTER;
    }
    return ZPSE_PAGE_GENERIC;
}

#if ZPSE_PF_HAVE_AVX2

/* The alphabet accumulator's in-register layout is a byte-indexed bitmap:
 * 128-bit half 0 holds byte lanes for values 0..127, half 1 for 128..255,
 * byte (v>>3)&15 holds bit 1<<(v&7). Each source half is broadcast across
 * both 256-bit sides so one pshufb per half can gather every lane's bitmap
 * byte by its OWN value; a blend by the value's high bit then selects the
 * right half. That is a fixed 32-byte membership test instead of up to 16
 * scalar compares against the known alphabet. */
static ZPSE_PF_AVX2 __m256i pf_bits_set(__m256i bits, uint8_t v)
{
    const __m256i byte_lanes = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i lo_half = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    __m256i sel = _mm256_cmpeq_epi8(byte_lanes,
        _mm256_set1_epi8((char)((v >> 3) & 15u)));
    __m256i val = _mm256_set1_epi8((char)(1u << (v & 7u)));
    __m256i half = (v & 0x80u) != 0u
        ? _mm256_xor_si256(lo_half,
              _mm256_set1_epi8((char)0xFF))
        : lo_half;

    return _mm256_or_si256(bits,
        _mm256_and_si256(half, _mm256_and_si256(sel, val)));
}

/* The whole page in one register-resident pass: 32-byte loads, four vector
 * accumulators (zero histogram, lane xor, high-bit OR, alphabet bitmap) and
 * scalar work only for the at-most-17 newly discovered alphabet values. */
static ZPSE_PF_AVX2 void pf_classify_avx2(const uint8_t *in, uint32_t n,
                                          ZpsePfAcc *a)
{
    const __m256i ref = _mm256_loadu_si256(
        (const __m256i *)(const void *)in);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i mask1f = _mm256_set1_epi8((char)0x1F);
    const __m256i mask0f = _mm256_set1_epi8((char)0x0F);
    const __m256i mask07 = _mm256_set1_epi8((char)0x07);
    const __m256i high_bit = _mm256_set1_epi8((char)0x80);
    const __m256i bit_lut = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80);
    __m256i zcnt = zero;
    __m256i ldiff = zero;
    __m256i hacc = zero;
    __m256i bits = zero;
    uint32_t i = 0u;
    uint8_t buf[32];

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(in + i));
        __m256i isz = _mm256_cmpeq_epi8(v, zero);

        zcnt = _mm256_add_epi8(zcnt, _mm256_and_si256(isz, one));
        ldiff = _mm256_or_si256(ldiff, _mm256_xor_si256(v, ref));
        hacc = _mm256_or_si256(hacc, v);

        if (a->alpha_over == 0u) {
            __m256i hi3 = _mm256_and_si256(
                _mm256_srli_epi16(v, 3), mask1f);
            __m256i idx = _mm256_and_si256(hi3, mask0f);
            /* Select the SOURCE half before pshufb, not the result: each
             * lane must gather with its own index. Broadcasting the shuffled
             * result would mix lanes and report false membership. */
            __m256i bits_lo = _mm256_permute2x128_si256(bits, bits, 0x00);
            __m256i bits_hi = _mm256_permute2x128_si256(bits, bits, 0x11);
            __m256i glo = _mm256_shuffle_epi8(bits_lo, idx);
            __m256i ghi = _mm256_shuffle_epi8(bits_hi, idx);
            __m256i sel = _mm256_cmpeq_epi8(
                _mm256_and_si256(v, high_bit), high_bit);
            __m256i gathered = _mm256_blendv_epi8(glo, ghi, sel);
            __m256i bit = _mm256_shuffle_epi8(bit_lut,
                _mm256_and_si256(v, mask07));
            __m256i known = _mm256_and_si256(gathered, bit);
            uint32_t mask = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(known, zero));

            if (mask != 0u) {
                _mm256_storeu_si256((__m256i *)(void *)buf, v);
                while (mask != 0u && a->alpha_over == 0u) {
                    uint8_t val = buf[ZPSE_PF_CTZ32(mask)];

                    bits = pf_bits_set(bits, val);
                    pf_record(a, val);
                    mask &= mask - 1u;
                }
            }
        }
    }

    _mm256_storeu_si256((__m256i *)(void *)a->zero_cnt, zcnt);
    _mm256_storeu_si256((__m256i *)(void *)a->lane_diff, ldiff);
    {
        uint32_t hm = (uint32_t)_mm256_movemask_epi8(hacc);

        a->high_or = hm != 0u ? (uint8_t)0x80 : (uint8_t)0u;
    }
    /* Convert the byte-indexed runtime bitmap to the word layout shared with
     * the scalar path; the tail loop uses it as its membership oracle. */
    memset(a->abits, 0, sizeof(a->abits));
    {
        uint32_t k;
        for (k = 0u; k < a->nalpha; k++) {
            uint8_t val = a->alpha[k];
            a->abits[val >> 5] |= (uint32_t)1u << (val & 31u);
        }
    }
    for (; i < n; i++) {
        pf_add_byte(a, in[i], i & 31u);
    }
}
#endif /* ZPSE_PF_HAVE_AVX2 */

int zpse_page_classify(const uint8_t *in, uint32_t n, ZpsePageInfo *info)
{
    ZpsePfAcc a;
    uint32_t i;

    if (info == NULL) {
        return ZPSE_PAGE_GENERIC;
    }
    if (in == NULL || n == 0u) {
        memset(info, 0, sizeof(*info));
        return in == NULL ? ZPSE_PAGE_GENERIC : ZPSE_PAGE_ZERO;
    }

    pf_acc_init(&a, in, n);
#if ZPSE_PF_HAVE_AVX2
    if (n >= 32u && (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        pf_classify_avx2(in, n, &a);
    } else
#endif
    {
        for (i = 0u; i < n; i++) {
            pf_add_byte(&a, in[i], i & 31u);
        }
    }
    for (i = 0u; i < 32u; i++) {
        a.nzero += (uint32_t)a.zero_cnt[i];
    }
    return pf_finish(&a, n, info);
}
