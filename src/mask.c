#include "zpse/mask.h"
#include "zpse/hot.h"    /* ZPSE_AVX2_ABI_OK */
#include "zpse/status.h"
#include "zpse/vector.h" /* ZPSE_CPU_CAPS_FAST(): process-wide CPUID cache */

#include <string.h>

/* Unity-build safe: only static helpers plus the public API. No
 * allocation, no globals, C11. (The cached CPUID probes below use
 * function-local statics only, exactly as vector.c and crc32c.c do.) */

/* Master switch for the page-oriented fast paths. 0 restores the scalar
 * decoders and the baseline SSE2 count on every build. */
#ifndef ZPSE_MASK_FAST4K
#define ZPSE_MASK_FAST4K 1u
#endif

static void zpse_mask_set_at(uint32_t words[8], uint16_t pos)
{
    if (pos >= 256u) {
        return; /* words[] has 8 entries = 256 bits; pos must be < nt <= 256 */
    }
    words[pos >> 5] |= (uint32_t)1u << (pos & 31u);
}

/* Set every bit of the range [from, from+len).
 *
 * Two callers hand it a range rather than a position: the ALL_EXCEPTION
 * decoder, whose mask is the whole of [0, nt) (one call, at most eight
 * stores, in place of nt read-modify-writes through zpse_mask_set_at and
 * its bounds test), and the RUNS decoder, which knows each miss run as one
 * length. A near-dense mask -- 253 exceptions in 256 positions, which is
 * three hit bytes and seven bytes of payload -- was previously written one
 * position at a time
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
        return ZPSE_BAD_ARG;
    if (!zpse_mask_nt_ok(nt))
        return ZPSE_BAD_MASK;
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (exception_count != 0)
            return ZPSE_BAD_MASK;
        *k_out = 0;
        return ZPSE_OK;
    case ZPSE_M_ALL_EXCEPTION:
        k = (exception_count == 0) ? nt : (uint16_t)exception_count;
        if (k != nt)
            return ZPSE_BAD_MASK;
        *k_out = k;
        return ZPSE_OK;
    case ZPSE_M_ASCII_PACK:
        /* No mask: the form has no exceptions. K is 0, and the tile's
         * residual payload is the packed bytes (tile.c owns the length). */
        if (exception_count != 0)
            return ZPSE_BAD_MASK;
        *k_out = 0;
        return ZPSE_OK;
    case ZPSE_M_RAW_BITMAP:
    case ZPSE_M_SPARSE_U8:
    case ZPSE_M_SPARSE_GAPS:
    case ZPSE_M_RUNS:
        k = (uint16_t)exception_count;
        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        *k_out = k;
        return ZPSE_OK;
    default:
        return ZPSE_BAD_MASK;
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

/* ---- ISA gates -------------------------------------------------------- *
 * SSE2 is the x86-64 baseline and needs no runtime check. AVX2 and POPCNT
 * do. The per-function target attributes let this TU be built at the
 * baseline ISA -- the same shape dense.c, tile.c and vector.c use.
 * ZPSE_AVX2_ABI_OK keeps the 256-bit helpers out of builds whose ABI cannot
 * pass a 32-byte vector (MinGW GCC on Windows); those fall back to the
 * SSE2 count and the scalar decoders, which are bit-identical. */
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#define ZPSE_MASK_HAVE_X86 1
#define ZPSE_MASK_HAVE_SIMD 1
#define ZPSE_MASK_HAVE_POPCNT 1
#if ZPSE_MASK_FAST4K
#define ZPSE_MASK_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_MASK_HAVE_AVX2 0
#endif
#if defined(_MSC_VER)
#define ZPSE_MASK_SIMD
#define ZPSE_MASK_AVX2
#define ZPSE_MASK_POPCNT
#else
#define ZPSE_MASK_SIMD   __attribute__((target("sse2")))
#define ZPSE_MASK_AVX2   __attribute__((target("avx2")))
#define ZPSE_MASK_POPCNT __attribute__((target("popcnt")))
#endif
#else
#define ZPSE_MASK_HAVE_X86 0
#define ZPSE_MASK_HAVE_SIMD 0
#define ZPSE_MASK_HAVE_POPCNT 0
#define ZPSE_MASK_HAVE_AVX2 0
#endif

#if ZPSE_MASK_HAVE_SIMD
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <nmmintrin.h> /* _mm_popcnt_u32 */
#include <stdatomic.h>
#endif
#include <emmintrin.h>
#endif
#if ZPSE_MASK_HAVE_AVX2
#include <immintrin.h>
#endif

/* Branch-free SWAR popcount, kept as the fallback on every build. The
 * builtin lowers to a libgcc call (__popcountdi2) unless the whole build is
 * compiled with -mpopcnt, which this portable codebase is not. */
static int zpse_mask_popcount32_swar(uint32_t w)
{
    w = w - ((w >> 1) & 0x55555555u);
    w = (w & 0x33333333u) + ((w >> 2) & 0x33333333u);
    w = (w + (w >> 4)) & 0x0F0F0F0Fu;
    return (int)((w * 0x01010101u) >> 24);
}

#if ZPSE_MASK_HAVE_POPCNT
/* One-time CPUID probe, cached exactly like crc32c.c's SSE4.2 probe and
 * vector.c's caps word: CPUID is serialising (microseconds under
 * virtualisation) and a mask count happens once or twice per tile. The
 * caps word is the first gate, so ZPSE_FORCE_SCALAR and non-x86 builds
 * never issue the probe. Idempotent computation, relaxed atomics, no
 * barrier on the hot path. */
#if defined(_MSC_VER)
static int zpse_mask_popcnt_cached = -1; /* -1 = not yet computed */
static int zpse_mask_popcnt_get(void)
{
    return zpse_mask_popcnt_cached;
}
static void zpse_mask_popcnt_set(int v)
{
    zpse_mask_popcnt_cached = v;
}
#else
static _Atomic int zpse_mask_popcnt_cached = -1;
static int zpse_mask_popcnt_get(void)
{
    return atomic_load_explicit(&zpse_mask_popcnt_cached,
                                memory_order_relaxed);
}
static void zpse_mask_popcnt_set(int v)
{
    atomic_store_explicit(&zpse_mask_popcnt_cached, v, memory_order_relaxed);
}
#endif

static int zpse_mask_have_popcnt(void)
{
    int present = zpse_mask_popcnt_get();

    if (present >= 0) {
        return present;
    }
    if (ZPSE_CPU_CAPS_FAST() == 0u) {
        present = 0;
    } else {
#if defined(_MSC_VER)
        int info[4];

        __cpuid(info, 1);
        present = (info[2] >> 23) & 1; /* POPCNT: leaf 1, ECX bit 23 */
#else
        unsigned int a = 0;
        unsigned int b = 0;
        unsigned int c = 0;
        unsigned int d = 0;

        present = 0;
        if (__get_cpuid(1, &a, &b, &c, &d)) {
            present = (int)((c >> 23) & 1u);
        }
#endif
    }
    zpse_mask_popcnt_set(present);
    return present;
}

static ZPSE_MASK_POPCNT int zpse_mask_popcount32_hw(uint32_t w)
{
    return (int)_mm_popcnt_u32(w);
}

/* Eight words with the dispatch hoisted out of the loop: the cached answer
 * is read once, not once per word. */
static ZPSE_MASK_POPCNT int zpse_mask_popcount256_hw(const uint32_t words[8])
{
    return zpse_mask_popcount32_hw(words[0]) +
           zpse_mask_popcount32_hw(words[1]) +
           zpse_mask_popcount32_hw(words[2]) +
           zpse_mask_popcount32_hw(words[3]) +
           zpse_mask_popcount32_hw(words[4]) +
           zpse_mask_popcount32_hw(words[5]) +
           zpse_mask_popcount32_hw(words[6]) +
           zpse_mask_popcount32_hw(words[7]);
}
#endif /* ZPSE_MASK_HAVE_POPCNT */

/* Runtime-dispatched count of one dense word: hardware POPCNT when CPUID
 * says it is there, the SWAR fold otherwise (all non-x86 builds). */
static int zpse_mask_popcount32(uint32_t w)
{
#if ZPSE_MASK_HAVE_POPCNT
    if (zpse_mask_have_popcnt()) {
        return zpse_mask_popcount32_hw(w);
    }
#endif
    return zpse_mask_popcount32_swar(w);
}

#if ZPSE_MASK_HAVE_SIMD
/* Count of the one bits in one 16-byte half. Straight-line on purpose:
 * the same fold written as a four-iteration loop was miscompiled by the
 * MinGW GCC used for the Windows builds -- it summed stale lanes and
 * returned 20..26 bits for a mask holding one -- while this unrolled form
 * is exactly what the scalar count says. Two halves per call, no loop.
 * Kept as the fallback for CPUs without POPCNT; the bit-parallel fold needs
 * only AND/ADD/SHIFT, so SSE2 is enough. */
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

/* 256-bit count, one cached dispatch for the whole mask: hardware POPCNT
 * where CPUID says it exists, the baseline SSE2 fold next, and the scalar
 * word loop (through the runtime-dispatched one-word count) elsewhere. */
static int zpse_mask_popcount256(const uint32_t words[8])
{
#if ZPSE_MASK_HAVE_POPCNT
    if (zpse_mask_have_popcnt()) {
        return zpse_mask_popcount256_hw(words);
    }
#endif
#if ZPSE_MASK_HAVE_SIMD
    /* The single-word dispatcher is the fallback on builds without this
     * fold; keep it live here so -Wunused-function stays quiet. */
    (void)zpse_mask_popcount32;
    return zpse_mask_popcount256_sse2(words);
#else
    {
        int n = 0;
        int i;

        for (i = 0; i < 8; i++)
            n += zpse_mask_popcount32(words[i]);
        return n;
    }
#endif
}

int zpse_mask_count_bits(const uint32_t words[8], uint16_t nt)
{
    uint32_t w[8];
    const uint32_t *src = words;

    if (words == 0 || nt == 0)
        return 0;
    if (nt > 256)
        nt = 256;
    /* A full 256-position mask needs no trimming, so the copy, the zero-fill
     * and the final AND are all no-ops for it -- and a full tile is what
     * every dense shape has. Counting the caller's words directly is one
     * 32-byte read instead of a read, a copy and two writes. */
    if (nt != 256u) {
        /* Trim the bits at or above nt in a local copy (the caller's array
         * is read-only here). The same bits were ignored by the old scalar
         * loop's per-word mask. */
        uint16_t last = (uint16_t)((nt - 1u) >> 5);
        uint32_t keep = (uint32_t)(((uint64_t)1u << (((nt - 1u) & 31u) + 1u))
                                   - 1u);
        int ki;

        memcpy(w, words, sizeof w);
        for (ki = (int)last + 1; ki < 8; ki++)
            w[ki] = 0u;
        w[last] &= keep;
        src = w;
    }
    return zpse_mask_popcount256(src);
}

/* The two scalar decoders are also the error-path reference for their AVX2
 * twins. A rejected payload leaves partially built words behind; callers
 * discard them, but replaying the scalar walk from a clean mask on rejection
 * keeps the two paths bit-identical even there. */
static int zpse_mask_sparse_u8_scalar(uint32_t words[8], const uint8_t *payload,
                                      uint16_t k, uint16_t nt)
{
    uint16_t prev = 0;
    uint16_t j;

    for (j = 0; j < k; j++) {
        uint16_t v = payload[j];

        if (v >= nt || (j > 0u && v <= prev))
            return ZPSE_BAD_MASK;
        prev = v;
        zpse_mask_set_at(words, v);
    }
    return ZPSE_OK;
}

static int zpse_mask_runs_scalar(uint32_t words[8], const uint8_t *payload,
                                 uint16_t payload_len, uint16_t nt, uint16_t k)
{
    uint32_t sum = 0u;
    uint32_t miss = 0u;
    uint32_t pos = 0u;
    uint16_t r;

    for (r = 0; r < payload_len; r++) {
        uint32_t len = payload[r];

        sum += len;
        if (sum > nt)
            return ZPSE_BAD_MASK;
        if (r & 1u) {
            miss += len;
            if (miss > k)
                return ZPSE_BAD_MASK;
            zpse_mask_set_range(words, pos, len);
        }
        pos += len;
    }
    if (sum != nt || miss != k)
        return ZPSE_BAD_MASK;
    return ZPSE_OK;
}

/* ---- AVX2 mask decoders ------------------------------------------------ *
 * Every helper below is bit-exact to the scalar branch it replaces: same
 * words, same ZPSE_BAD_MASK decisions. They are called only when the cached
 * CPU caps say AVX2 is present, and every one has a scalar fall-through in
 * zpse_mask_decode for the other builds and CPUs. */
#if ZPSE_MASK_HAVE_AVX2

/* ALL_EXCEPTION: words = bits [0, nt). words[] is already zero. One masked
 * store writes whole words for the 32-position blocks below nt; only the
 * partial final word needs a scalar AND. */
static ZPSE_MASK_AVX2 void zpse_mask_all_exception_avx2(uint32_t words[8],
                                                        uint16_t nt)
{
    const __m256i idx = _mm256_setr_epi32(0, 32, 64, 96, 128, 160, 192, 224);
    const __m256i ntv = _mm256_set1_epi32((int)nt);
    const __m256i ones = _mm256_set1_epi32(-1);
    const __m256i have = _mm256_cmpgt_epi32(ntv, idx); /* lane base < nt */

    _mm256_maskstore_epi32((int *)(void *)words, have, ones);
    if ((nt & 31u) != 0u) {
        uint16_t last = (uint16_t)((nt - 1u) >> 5u);

        words[last] &= (uint32_t)((UINT64_C(1) << (nt & 31u)) - 1u);
    }
}

/* RAW_BITMAP, whole payload: one 32-byte load and store. Only called when
 * need == 32 (nt in 249..256), so it never reads past the payload. On
 * little-endian hosts this is the same byte-for-byte copy memcpy makes. */
static ZPSE_MASK_AVX2 void zpse_mask_bitmap_copy_avx2(
    uint32_t words[8], const uint8_t *payload)
{
    __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)payload);

    _mm256_storeu_si256((__m256i *)(void *)words, v);
}

/* SPARSE_U8 for k >= 8. The eight ascending positions in a chunk are
 * validated with two compares (range, order) instead of one branch per
 * byte; one variable-shift vector builds the eight bit masks in parallel.
 * The stores into words stay scalar -- AVX2 has no scatter -- but positions
 * are ascending, so consecutive lanes hit the same or the next word. */
static ZPSE_MASK_AVX2 int zpse_mask_sparse_u8_avx2(
    uint32_t words[8], const uint8_t *payload, uint16_t k, uint16_t nt)
{
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i ntv = _mm256_set1_epi32((int)nt);
    const __m256i mask31 = _mm256_set1_epi32(31);
    /* Lane 7 compares against itself (false); it is checked against the
     * first byte of the next chunk by the cross-chunk test below. */
    const __m256i next_lane = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);
    uint32_t bits[8];
    uint16_t prev = 0;
    uint16_t j = 0;
    int i;

    for (; (uint16_t)(j + 8u) <= k; j = (uint16_t)(j + 8u)) {
        __m256i v = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64((const __m128i *)(const void *)(payload + j)));
        __m256i nxt = _mm256_permutevar8x32_epi32(v, next_lane);
        __m256i range_ok = _mm256_cmpgt_epi32(ntv, v);  /* v < nt */
        __m256i order_ok = _mm256_cmpgt_epi32(nxt, v);  /* v[i] < v[i+1] */
        unsigned good;

        if ((uint32_t)_mm256_movemask_epi8(range_ok) != 0xFFFFFFFFu)
            goto reject;
        good = (unsigned)_mm256_movemask_epi8(order_ok) & 0x0FFFFFFFu;
        if (good != 0x0FFFFFFFu || (j > 0u && (uint16_t)payload[j] <= prev))
            goto reject;

        _mm256_storeu_si256(
            (__m256i *)(void *)bits,
            _mm256_sllv_epi32(one, _mm256_and_si256(v, mask31)));
        for (i = 0; i < 8; i++)
            words[payload[j + (uint16_t)i] >> 5u] |= bits[i];
        prev = payload[j + 7u];
    }
    /* Tail (0..7 positions) and the cross-chunk order check, scalar. */
    for (; j < k; j++) {
        uint16_t v = payload[j];

        if (v >= nt || (j > 0u && v <= prev))
            goto reject;
        prev = v;
        words[v >> 5u] |= (uint32_t)1u << (v & 31u);
    }
    return ZPSE_OK;

reject:
    /* Malformed stream: replay the scalar walk so even the partial words it
     * leaves behind are the ones the scalar decoder would have produced. */
    zpse_mask_clear(words);
    return zpse_mask_sparse_u8_scalar(words, payload, k, nt);
}

/* RUNS for payload_len >= 8. The eight run lengths are prefix-summed in
 * one vector (three shuffles plus one cross-lane add), which removes the
 * serial pos += len chain; the miss-run lengths are prefix-summed the same
 * way for the k check. Both totals are non-decreasing, so their last lanes
 * are the running maxima and one test each replaces a branch per run. */
static ZPSE_MASK_AVX2 int zpse_mask_runs_avx2(
    uint32_t words[8], const uint8_t *payload, uint16_t payload_len,
    uint16_t nt, uint16_t k)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i odd = _mm256_setr_epi32(0, -1, 0, -1, 0, -1, 0, -1);
    uint32_t base = 0u; /* lengths of all runs so far */
    uint32_t miss = 0u; /* lengths of the miss (odd) runs so far */
    unsigned r = 0u;

    for (; r + 8u <= (unsigned)payload_len; r += 8u) {
        __m256i lens = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64((const __m128i *)(const void *)(payload + r)));
        __m256i s;
        __m256i ml;
        __m256i carry;
        uint32_t pos[8];
        uint32_t mp[8];
        unsigned i;

        s = _mm256_add_epi32(lens, _mm256_slli_si256(lens, 4));
        s = _mm256_add_epi32(s, _mm256_slli_si256(s, 8));
        carry = _mm256_shuffle_epi32(s, _MM_SHUFFLE(3, 3, 3, 3));
        s = _mm256_add_epi32(s, _mm256_permute2x128_si256(zero, carry, 0x20));

        ml = _mm256_and_si256(lens, odd);
        ml = _mm256_add_epi32(ml, _mm256_slli_si256(ml, 4));
        ml = _mm256_add_epi32(ml, _mm256_slli_si256(ml, 8));
        carry = _mm256_shuffle_epi32(ml, _MM_SHUFFLE(3, 3, 3, 3));
        ml = _mm256_add_epi32(ml, _mm256_permute2x128_si256(zero, carry, 0x20));

        s = _mm256_add_epi32(s, _mm256_set1_epi32((int)base));
        ml = _mm256_add_epi32(ml, _mm256_set1_epi32((int)miss));
        _mm256_storeu_si256((__m256i *)(void *)pos, s);
        _mm256_storeu_si256((__m256i *)(void *)mp, ml);

        if (pos[7] > (uint32_t)nt || mp[7] > (uint32_t)k)
            goto reject;

        /* Only the miss runs write bits: (r + i) is odd exactly when i has
         * the parity opposite to r, so start at that i and step by two. This
         * visits the same indices, in the same order, as testing every i
         * against the old ((r + i) & 1u) branch -- r is a multiple of 8 here
         * so the first miss run is i = 1 and `from` is the chunk-local
         * pos[i - 1] (the `(i == 0) ? base` guard is kept for the general
         * case). Byte-identical set_range calls, four instead of eight. */
        for (i = (r & 1u) ^ 1u; i < 8u; i += 2u) {
            uint32_t from = (i == 0u) ? base : pos[i - 1u];

            if (pos[i] > from)
                zpse_mask_set_range(words, from, pos[i] - from);
        }
        base = pos[7];
        miss = mp[7];
    }

    /* Tail runs, same checks and same order as the scalar decoder. */
    for (; r < (unsigned)payload_len; r++) {
        uint32_t len = payload[r];

        base += len;
        if (base > (uint32_t)nt)
            goto reject;
        if ((r & 1u) != 0u) {
            miss += len;
            if (miss > (uint32_t)k)
                goto reject;
            zpse_mask_set_range(words, base - len, len);
        }
    }
    if (base != (uint32_t)nt || miss != (uint32_t)k)
        goto reject;
    return ZPSE_OK;

reject:
    zpse_mask_clear(words);
    return zpse_mask_runs_scalar(words, payload, payload_len, nt, k);
}

#endif /* ZPSE_MASK_HAVE_AVX2 */

int zpse_mask_decode(uint8_t mask_mode, const uint8_t *payload,
                     uint16_t payload_len, uint16_t nt, uint16_t k,
                     uint32_t words[8], uint8_t *comb_scratch)
{
    (void)comb_scratch; /* kept in the signature; only mode 5 ever used it */
#if ZPSE_MASK_HAVE_AVX2
    int have_avx2 = (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u;
#endif

    if (words == 0)
        return ZPSE_BAD_ARG;
    /* words[] is zeroed by the caller (see the mask.h contract); every
     * success path below ORs into those zero words, so no clear is needed
     * here. The AVX2 reject paths clear before their scalar replay and keep
     * doing so -- that reset is not redundant. */
    if (!zpse_mask_nt_ok(nt))
        return ZPSE_BAD_MASK;
    if (payload_len > 0 && payload == 0)
        return ZPSE_BAD_ARG;

    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (k != 0 || payload_len != 0)
            return ZPSE_BAD_MASK;
        return ZPSE_OK;
    case ZPSE_M_ALL_EXCEPTION:
        if (k != nt || payload_len != 0)
            return ZPSE_BAD_MASK;
        /* Every position is an exception, so the mask is [0, nt) as one
         * run: a word at a time instead of nt load-OR-store rounds on the
         * same eight words, each waiting on the store before it. */
#if ZPSE_MASK_HAVE_AVX2
        if (have_avx2) {
            zpse_mask_all_exception_avx2(words, nt);
            return ZPSE_OK;
        }
#endif
        zpse_mask_set_range(words, 0u, nt);
        return ZPSE_OK;
    case ZPSE_M_RAW_BITMAP: {
        uint16_t need;
        unsigned byte_i;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        need = (uint16_t)((nt + 7u) >> 3);
        if (payload_len != need)
            return ZPSE_BAD_MASK;
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
        /* Every byte of that assembly is a byte of the wire on a
         * little-endian host, so there the whole assembly is the copy the
         * encoder already does the other way round: one 32-byte copy for a
         * full tile, against eight reassembled stores. The words past
         * `need` stay zero from the clear above either way. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#if ZPSE_MASK_HAVE_AVX2
        /* A full 32-byte bitmap is exactly eight words. One vector load
         * and store; need < 32 keeps the memcpy so no byte past the payload
         * is ever read. */
        if (have_avx2 && need == 32u) {
            zpse_mask_bitmap_copy_avx2(words, payload);
        } else {
            memcpy(words, payload, (size_t)need);
        }
#else
        memcpy(words, payload, (size_t)need);
#endif
        (void)byte_i;
#else
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
#endif
        /* Bits >= nt sit in the high bits of the final byte and MUST be
         * zero (Sec. 8). The old scan ran p from nt to need*8 and is
         * therefore only ever non-empty when nt is not a multiple of 8. */
        if ((nt & 7u) != 0u && (payload[need - 1u] >> (nt & 7u)) != 0u)
            return ZPSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return ZPSE_BAD_MASK;
        return ZPSE_OK;
    }
    case ZPSE_M_SPARSE_U8:
        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        if (payload_len != k)
            return ZPSE_BAD_MASK;
#if ZPSE_MASK_HAVE_AVX2
        /* For a short exception list the scalar loop is cheaper than the
         * vector setup; from eight positions the batched validation and
         * variable shifts pay for themselves. */
        if (have_avx2 && k >= 8u)
            return zpse_mask_sparse_u8_avx2(words, payload, k, nt);
#endif
        /* The ascending-order test compares against the previous decoded
         * position; zpse_mask_sparse_u8_scalar keeps that byte in a local
         * instead of reloading payload[j - 1] each iteration. */
        return zpse_mask_sparse_u8_scalar(words, payload, k, nt);
    case ZPSE_M_SPARSE_GAPS: {
        uint8_t flag;
        uint16_t need;
        uint32_t pos = 0;
        uint16_t j;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        if (payload_len < 1)
            return ZPSE_BAD_MASK;
        flag = payload[0];
        if (flag > 1)
            return ZPSE_BAD_MASK;
        need = flag ? (uint16_t)(1u + 2u * (uint16_t)k)
                    : (uint16_t)(1u + (uint16_t)k);
        if (payload_len != need)
            return ZPSE_BAD_MASK;
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
                return ZPSE_BAD_MASK;
            zpse_mask_set_at(words, (uint16_t)pos);
        }
        return ZPSE_OK;
    }
    case ZPSE_M_RUNS:
        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        if (payload_len < 2)
            return ZPSE_BAD_MASK;
#if ZPSE_MASK_HAVE_AVX2
        if (have_avx2 && payload_len >= 8u)
            return zpse_mask_runs_avx2(words, payload, payload_len, nt, k);
#endif
        return zpse_mask_runs_scalar(words, payload, payload_len, nt, k);
    default:
        return ZPSE_BAD_MASK;
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
        return ZPSE_BAD_ARG;
    if (!zpse_mask_nt_ok(nt))
        return ZPSE_BAD_MASK;

    switch (mask_mode) {
    /* The two constant forms have to prove their count against the header,
     * and both counts have a cheaper question behind them: "is the count
     * zero" is "is every bit below nt clear" and "is the count nt" is "is
     * every bit below nt set". A masked compare answers each in at most
     * eight tests, where the count costs a 32-byte copy and a vector pass
     * (or a popcount of the whole mask) to produce a number that is then
     * compared with a constant. Bits at or above nt are ignored, exactly as
     * the count ignores them. */
    case ZPSE_M_ALL_HIT: {
        uint16_t last;
        uint16_t wi;
        uint32_t keep;

        if (k != 0)
            return ZPSE_BAD_MASK;
        last = (uint16_t)((nt - 1u) >> 5u);
        keep = (uint32_t)(((UINT64_C(1) << (((nt - 1u) & 31u) + 1u)) - 1u));
        for (wi = 0; wi < last; wi++) {
            if (words[wi] != 0u)
                return ZPSE_BAD_MASK;
        }
        if ((words[last] & keep) != 0u)
            return ZPSE_BAD_MASK;
        return 0;
    }
    case ZPSE_M_ALL_EXCEPTION: {
        uint16_t last;
        uint16_t wi;
        uint32_t keep;

        if (k != nt)
            return ZPSE_BAD_MASK;
        last = (uint16_t)((nt - 1u) >> 5u);
        keep = (uint32_t)(((UINT64_C(1) << (((nt - 1u) & 31u) + 1u)) - 1u));
        for (wi = 0; wi < last; wi++) {
            if (words[wi] != UINT32_C(0xFFFFFFFF))
                return ZPSE_BAD_MASK;
        }
        if ((words[last] & keep) != keep)
            return ZPSE_BAD_MASK;
        return 0;
    }
    case ZPSE_M_RAW_BITMAP: {
        uint16_t need;
        unsigned byte_i;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return ZPSE_BAD_MASK;
        need = (uint16_t)((nt + 7u) >> 3);
        if (payload_cap < need)
            return ZPSE_OUTPUT_OVERFLOW;
        if (need > 0 && payload_out == 0)
            return ZPSE_BAD_ARG;
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
            return ZPSE_BAD_MASK;
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
            return ZPSE_BAD_MASK;
        if (payload_cap < k)
            return ZPSE_OUTPUT_OVERFLOW;
        if (k > 0 && payload_out == 0)
            return ZPSE_BAD_ARG;
        for (j = 0; j < k; j++)
            payload_out[j] = (uint8_t)c[j];
        return (int)k;
    }
    case ZPSE_M_SPARSE_GAPS: {
        int wide = 0;
        uint16_t need;

        if (k < 1 || k > (uint16_t)(nt - 1u))
            return ZPSE_BAD_MASK;
        nc = zpse_mask_collect(words, nt, c);
        if (nc != k)
            return ZPSE_BAD_MASK;
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
            return ZPSE_OUTPUT_OVERFLOW;
        if (payload_out == 0)
            return ZPSE_BAD_ARG;
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
            return ZPSE_BAD_MASK;
        if (zpse_mask_count_bits(words, nt) != (int)k)
            return ZPSE_BAD_MASK;
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
                        return ZPSE_BAD_MASK;
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
            return ZPSE_OUTPUT_OVERFLOW;
        if (need > 0 && payload_out == 0)
            return ZPSE_BAD_ARG;
        for (r = 0; r + 1 < nb; r++) {
            payload_out[o++] = (uint8_t)(edge[r + 1] - edge[r]);
        }
        return (int)o;
    }
    default:
        break;
    }
    (void)comb_scratch;
    return ZPSE_BAD_MASK;
}
