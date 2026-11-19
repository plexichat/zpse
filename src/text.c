#include "zpse/text.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "zpse/hot.h"     /* ZPSE_AVX2_ABI_OK */
#include "zpse/vector.h"  /* ZPSE_CPU_CAPS_FAST, ZPSE_CPU_AVX2 */

/* Non-temporal decode stores need a sfence to make them visible, so below
 * this output length the fence costs more than the RFO it avoids. Typical
 * cache-resident blocks take the ordinary write-back path. */
#define ZPSE_TEXT_NT_MIN 4096u

/* Internal entry point used by block.c after its ASCII gate has already
 * scanned the input. Byte-identical to zpse_text_encode on eligible input but
 * without the redundant eligibility scan. Not part of the public text.h API. */
uint32_t zpse_text_encode_ascii(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap);

/* ---- Payload form: finder-free 7-bit pack ------------------------------- *
 * Every input byte must be ASCII (<128). Eight bytes become seven: bit 7i+k
 * of the packed group is bit k of input byte i. This is the identical layout
 * to src/tile.c's tile_ascii_compact8 / tile_ascii_spread8, so a tile decode
 * kernel could read this stream. The tail group of r<8 bytes occupies
 * group_out(r) = (7r+7)>>3 == r bytes, masked to 7r bits. Hence the payload
 * length is always n - (n>>3), with no table, no search and no per-block
 * setup: price is one ASCII scan plus one subtraction and encode is one pack
 * pass. The wire form (sync 0x50BA, 16-byte header) and the three entry
 * points are unchanged; block.c still gates this form to blocks where the
 * LZ finder cannot win. */

#include "text_tables.inc"

/* ---- Target/ABI gates (same shape as src/flat.c:10-33, src/tile.c:370) -- *
 * Per-function target attributes keep this TU at the baseline ISA; each
 * vector type is a function LOCAL, never passed or returned by value, so the
 * gcc/MinGW 256-bit-by-value ABI hazard in zpse/hot.h cannot recur. aligned(64)
 * keeps a hot kernel inside one pair of I-cache lines. */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_TEXT_AVX2_TARGET __attribute__((target("avx2"), aligned(64)))
/* The kernels below keep every __m256i in a function LOCAL used only by
 * intrinsics: no 256-bit value is passed or returned by value, so the
 * MinGW gcc ABI hazard documented in zpse/hot.h (an aligned stack temporary
 * for a by-value 256-bit argument) cannot occur here. The per-function
 * target("avx2") attribute keeps the rest of the TU at baseline ISA. */
#define ZPSE_TEXT_HAVE_AVX2 1
#elif defined(__AVX2__)
#define ZPSE_TEXT_AVX2_TARGET __attribute__((aligned(64)))
#define ZPSE_TEXT_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_TEXT_HAVE_AVX2 0
#endif
#else
#define ZPSE_TEXT_HAVE_AVX2 0
#endif

/* vector.h has no BMI2 cap bit (ZPSE_CPU_BMI2 does not exist), so the BMI2
 * body is guarded the way tile.c guards its own: a target("bmi2") function
 * entered only after a cached __builtin_cpu_supports("bmi2") answer. */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(ZPSE_FORCE_SCALAR)
#define ZPSE_TEXT_HAVE_BMI2 1
#if defined(__GNUC__) || defined(__clang__)
#define ZPSE_TEXT_BMI2_TARGET __attribute__((target("bmi2"), aligned(64)))
#else
#define ZPSE_TEXT_BMI2_TARGET
#endif
#else
#define ZPSE_TEXT_HAVE_BMI2 0
#endif

/* SSE2 is baseline on x86-64, so the 2-lane kernels below need no runtime
 * check: they are the middle tier between the scalar reference and the
 * BMI2/AVX2 kernels. FORCE_SCALAR parity builds leave only the scalar path. */
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(ZPSE_FORCE_SCALAR)
#define ZPSE_TEXT_HAVE_SSE2 1
#else
#define ZPSE_TEXT_HAVE_SSE2 0
#endif

#if ZPSE_TEXT_HAVE_AVX2 || ZPSE_TEXT_HAVE_BMI2
#include <immintrin.h>
#endif
#if ZPSE_TEXT_HAVE_SSE2
#include <emmintrin.h>
#endif

#if ZPSE_TEXT_HAVE_BMI2
/* Asked once per process; the answer cannot change while it runs. */
static int text_bmi2(void)
{
    static int cached = -1;

    if (cached < 0) {
#if defined(__GNUC__) || defined(__clang__)
        cached = __builtin_cpu_supports("bmi2") ? 1 : 0;
#else
        cached = 0;
#endif
    }
    return cached;
}
#endif

/* ---- Shared scalar primitives ------------------------------------------ */

/* Bytes of payload for a final group of r input bytes (r<=8). For r=1..7
 * this is exactly r; r=8 never reaches here (it is a full group). */
static uint32_t text_group_out(uint32_t r)
{
    return (7u * r + 7u) >> 3;
}

/* r bytes as a little-endian 64-bit word, zero above byte r-1. */
static uint64_t text_gather(const uint8_t *p, uint32_t r)
{
    uint64_t w = 0u;
    uint32_t i;

    if (r == 8u) {
        memcpy(&w, p, sizeof w);
        return w;
    }
    for (i = 0u; i < r; i++) {
        w |= (uint64_t)p[i] << (8u * i);
    }
    return w;
}

/* Eight ASCII bytes -> their 56 packed bits; bit 7i+k = bit k of byte i. */
static uint64_t text_compact8(uint64_t w)
{
    return (w & UINT64_C(0x000000000000007F)) |
           ((w & UINT64_C(0x0000000000007F00)) >> 1) |
           ((w & UINT64_C(0x00000000007F0000)) >> 2) |
           ((w & UINT64_C(0x000000007F000000)) >> 3) |
           ((w & UINT64_C(0x0000007F00000000)) >> 4) |
           ((w & UINT64_C(0x00007F0000000000)) >> 5) |
           ((w & UINT64_C(0x007F000000000000)) >> 6) |
           ((w & UINT64_C(0x7F00000000000000)) >> 7);
}

static uint64_t text_spread8(uint64_t v)
{
    return (v & UINT64_C(0x000000000000007F)) |
           ((v & UINT64_C(0x0000000000003F80)) << 1) |
           ((v & UINT64_C(0x00000000001FC000)) << 2) |
           ((v & UINT64_C(0x000000000FE00000)) << 3) |
           ((v & UINT64_C(0x00000007F0000000)) << 4) |
           ((v & UINT64_C(0x000003F800000000)) << 5) |
           ((v & UINT64_C(0x0001FC0000000000)) << 6) |
           ((v & UINT64_C(0x00FE000000000000)) << 7);
}

/* 64-bit OR tree: ~4 bytes/cycle, used for price/encode eligibility. No
 * vector needed at the call sites. */
static int text_all_ascii(const uint8_t *p, uint32_t n)
{
    uint32_t i = 0u;
    const uint64_t hi = UINT64_C(0x8080808080808080);

    for (; i + 64u <= n; i += 64u) {
        uint64_t a, b, c, d, e, f, g, h;

        memcpy(&a, p + i,      8u);
        memcpy(&b, p + i +  8u, 8u);
        memcpy(&c, p + i + 16u, 8u);
        memcpy(&d, p + i + 24u, 8u);
        memcpy(&e, p + i + 32u, 8u);
        memcpy(&f, p + i + 40u, 8u);
        memcpy(&g, p + i + 48u, 8u);
        memcpy(&h, p + i + 56u, 8u);
        {
            uint64_t acc = (a | b | c | d) | (e | f | g | h);

            if ((acc & hi) != 0u) {
                return 0;
            }
        }
    }
    for (; i + 8u <= n; i += 8u) {
        uint64_t w;

        memcpy(&w, p + i, 8u);
        if ((w & hi) != 0u) {
            return 0;
        }
    }
    for (; i < n; i++) {
        if ((p[i] & 0x80u) != 0u) {
            return 0;
        }
    }
    return 1;
}

/* ---- PACK --------------------------------------------------------------- */

/* Portable correctness reference. The 8-byte store for a full group writes
 * one spare byte that belongs to the NEXT group; taking it only while another
 * group follows (groups*7 is the all-full-groups payload end) keeps the last
 * group's 7-byte store exact and avoids writing past the payload. */
static uint32_t text_pack_scalar(const uint8_t *in, uint32_t n, uint8_t *out)
{
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g, p = 0u;

    for (g = 0u; g < groups; g++) {
        uint64_t w, v;

        memcpy(&w, in + (size_t)g * 8u, sizeof w);
        v = text_compact8(w);
        if (p + 8u <= groups * 7u) {
            memcpy(out + p, &v, sizeof v);
        } else {
            memcpy(out + p, &v, 7u);
        }
        p += 7u;
    }
    if (rest != 0u) {
        uint64_t v = text_compact8(text_gather(in + groups * 8u, rest));
        uint32_t nb = text_group_out(rest);
        uint32_t k;

        v &= (UINT64_C(1) << (7u * rest)) - UINT64_C(1);
        for (k = 0u; k < nb; k++) {
            out[p++] = (uint8_t)(v >> (8u * k));
        }
    }
    return p;
}

#if ZPSE_TEXT_HAVE_BMI2
/* pext gathers the low seven bits of every byte, in order, into the low 56
 * bits: one instruction per group against the scalar form's 22 ALU ops. Four
 * independent pexts per iteration; the four 8-byte stores each spill one byte
 * into the next group's slot, which that group's store overwrites. The strict
 * `<` bound leaves at least one group for the exact-length tail so the spill
 * is never past the payload. */
static ZPSE_TEXT_BMI2_TARGET uint32_t text_pack_bmi2(const uint8_t *in,
                                                      uint32_t n, uint8_t *out)
{
    const uint64_t mask = UINT64_C(0x7F7F7F7F7F7F7F7F);
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 4u < groups) {
        uint64_t a, b, c, d;

        memcpy(&a, in + (size_t)g * 8u,        8u);
        memcpy(&b, in + (size_t)g * 8u +  8u,  8u);
        memcpy(&c, in + (size_t)g * 8u + 16u,  8u);
        memcpy(&d, in + (size_t)g * 8u + 24u,  8u);
        a = _pext_u64(a, mask);
        b = _pext_u64(b, mask);
        c = _pext_u64(c, mask);
        d = _pext_u64(d, mask);
        memcpy(out + p,       &a, sizeof a);
        memcpy(out + p +  7u, &b, sizeof b);
        memcpy(out + p + 14u, &c, sizeof c);
        memcpy(out + p + 21u, &d, sizeof d);
        p += 28u;
        g += 4u;
    }
    for (; g < groups; g++) {
        uint64_t w, v;

        memcpy(&w, in + (size_t)g * 8u, sizeof w);
        v = _pext_u64(w, mask);
        if (p + 8u <= groups * 7u) {
            memcpy(out + p, &v, sizeof v);
        } else {
            memcpy(out + p, &v, 7u);
        }
        p += 7u;
    }
    if (rest != 0u) {
        uint64_t w = text_gather(in + groups * 8u, rest);
        uint64_t v = _pext_u64(w, mask);
        uint32_t nb = text_group_out(rest);
        uint32_t k;

        v &= (UINT64_C(1) << (7u * rest)) - UINT64_C(1);
        for (k = 0u; k < nb; k++) {
            out[p++] = (uint8_t)(v >> (8u * k));
        }
    }
    return p;
}
#endif /* ZPSE_TEXT_HAVE_BMI2 */

#if ZPSE_TEXT_HAVE_AVX2
/* The scalar compact8 idiom applied to all four 64-bit lanes of a __m256i at
 * once, then one pshufb per 128-bit lane to drop the four zero holes. Each
 * 32-byte load consumes four groups; the 32-byte store emits their 28 payload
 * bytes plus four spare bytes at p+28..31 belonging to the next group, which
 * the strict `<` bound guarantees will be overwritten by the tail. */
#define ZPSE_TEXT_COMPACT8_256(v) do {                                        \
    __m256i _a = _mm256_and_si256((v), zc0);                                  \
    __m256i _b = _mm256_srli_epi64(_mm256_and_si256((v), zc1), 1);            \
    __m256i _c = _mm256_srli_epi64(_mm256_and_si256((v), zc2), 2);            \
    __m256i _d = _mm256_srli_epi64(_mm256_and_si256((v), zc3), 3);            \
    __m256i _e = _mm256_srli_epi64(_mm256_and_si256((v), zc4), 4);            \
    __m256i _f = _mm256_srli_epi64(_mm256_and_si256((v), zc5), 5);            \
    __m256i _g = _mm256_srli_epi64(_mm256_and_si256((v), zc6), 6);            \
    __m256i _h = _mm256_srli_epi64(_mm256_and_si256((v), zc7), 7);            \
    (v) = _mm256_or_si256(                                                    \
        _mm256_or_si256(_mm256_or_si256(_a, _b), _mm256_or_si256(_c, _d)),    \
        _mm256_or_si256(_mm256_or_si256(_e, _f), _mm256_or_si256(_g, _h)));   \
} while (0)

static ZPSE_TEXT_AVX2_TARGET uint32_t text_pack_avx2(const uint8_t *in,
                                                      uint32_t n, uint8_t *out)
{
    const __m256i zc0 = _mm256_set1_epi64x(INT64_C(0x000000000000007F));
    const __m256i zc1 = _mm256_set1_epi64x(INT64_C(0x0000000000007F00));
    const __m256i zc2 = _mm256_set1_epi64x(INT64_C(0x00000000007F0000));
    const __m256i zc3 = _mm256_set1_epi64x(INT64_C(0x000000007F000000));
    const __m256i zc4 = _mm256_set1_epi64x(INT64_C(0x0000007F00000000));
    const __m256i zc5 = _mm256_set1_epi64x(INT64_C(0x00007F0000000000));
    const __m256i zc6 = _mm256_set1_epi64x(INT64_C(0x007F000000000000));
    const __m256i zc7 = _mm256_set1_epi64x(INT64_C(0x7F00000000000000));
    const __m128i zmask = _mm_loadu_si128(
        (const __m128i *)(const void *)ztext_shuf_compact);
    const __m256i zshuf = _mm256_broadcastsi128_si256(zmask);
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 4u < groups) {
        __m256i w = _mm256_loadu_si256(
            (const __m256i *)(const void *)(in + (size_t)g * 8u));

        __m128i hi;

        ZPSE_TEXT_COMPACT8_256(w);
        w = _mm256_shuffle_epi8(w, zshuf);
        /* Each 128-bit lane now holds 14 payload bytes and two zero holes.
         * The two stores splice them into 28 contiguous bytes: the low lane
         * lands at p, the high lane at p+14, so its first 14 bytes overwrite
         * the low lane's holes. Each store spills 2 spare bytes at p+14 and
         * p+28 respectively; the strict `<` loop guarantees a following group
         * overwrites them. */
        _mm256_storeu_si256((__m256i *)(void *)(out + p), w);
        hi = _mm256_extracti128_si256(w, 1);
        _mm_storeu_si128((__m128i *)(void *)(out + p + 14u), hi);
        p += 28u;
        g += 4u;
    }
    for (; g < groups; g++) {
        uint64_t w, v;

        memcpy(&w, in + (size_t)g * 8u, sizeof w);
        v = text_compact8(w);
        if (p + 8u <= groups * 7u) {
            memcpy(out + p, &v, sizeof v);
        } else {
            memcpy(out + p, &v, 7u);
        }
        p += 7u;
    }
    if (rest != 0u) {
        uint64_t v = text_compact8(text_gather(in + groups * 8u, rest));
        uint32_t nb = text_group_out(rest);
        uint32_t k;

        v &= (UINT64_C(1) << (7u * rest)) - UINT64_C(1);
        for (k = 0u; k < nb; k++) {
            out[p++] = (uint8_t)(v >> (8u * k));
        }
    }
    return p;
}
#undef ZPSE_TEXT_COMPACT8_256
#endif /* ZPSE_TEXT_HAVE_AVX2 */

#if ZPSE_TEXT_HAVE_SSE2
/* The scalar compact8 idiom applied to both 64-bit lanes of a __m128i at
 * once: eight pand + seven psrlq + seven por, all two-lanes-per-instruction.
 * Each 16-byte load consumes two groups; after compaction each lane holds one
 * group's 56 payload bits in bytes 0..6 with a zero hole in byte 7. The low
 * lane is stored as 8 bytes at p; the high lane is shifted down 8 bytes and
 * stored as 8 bytes at p+7, so its seven payload bytes splice after the low
 * lane's seven and its trailing zero byte spills into the next group's slot.
 * The strict g+2 < groups bound guarantees a following group overwrites that
 * spill, exactly as the BMI2/AVX2 kernels do. No pshufb is needed (SSE2). */
#define ZPSE_TEXT_COMPACT8_128(v) do {                                        \
    __m128i _a = _mm_and_si128((v), zc0);                                      \
    __m128i _b = _mm_srli_epi64(_mm_and_si128((v), zc1), 1);                  \
    __m128i _c = _mm_srli_epi64(_mm_and_si128((v), zc2), 2);                  \
    __m128i _d = _mm_srli_epi64(_mm_and_si128((v), zc3), 3);                  \
    __m128i _e = _mm_srli_epi64(_mm_and_si128((v), zc4), 4);                  \
    __m128i _f = _mm_srli_epi64(_mm_and_si128((v), zc5), 5);                  \
    __m128i _g = _mm_srli_epi64(_mm_and_si128((v), zc6), 6);                  \
    __m128i _h = _mm_srli_epi64(_mm_and_si128((v), zc7), 7);                  \
    (v) = _mm_or_si128(                                                        \
        _mm_or_si128(_mm_or_si128(_a, _b), _mm_or_si128(_c, _d)),             \
        _mm_or_si128(_mm_or_si128(_e, _f), _mm_or_si128(_g, _h)));            \
} while (0)

static uint32_t text_pack_sse2(const uint8_t *in, uint32_t n, uint8_t *out)
{
    const __m128i zc0 = _mm_set1_epi64x(INT64_C(0x000000000000007F));
    const __m128i zc1 = _mm_set1_epi64x(INT64_C(0x0000000000007F00));
    const __m128i zc2 = _mm_set1_epi64x(INT64_C(0x00000000007F0000));
    const __m128i zc3 = _mm_set1_epi64x(INT64_C(0x000000007F000000));
    const __m128i zc4 = _mm_set1_epi64x(INT64_C(0x0000007F00000000));
    const __m128i zc5 = _mm_set1_epi64x(INT64_C(0x00007F0000000000));
    const __m128i zc6 = _mm_set1_epi64x(INT64_C(0x007F000000000000));
    const __m128i zc7 = _mm_set1_epi64x(INT64_C(0x7F00000000000000));
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 2u < groups) {
        __m128i w = _mm_loadu_si128(
            (const __m128i *)(const void *)(in + (size_t)g * 8u));
        __m128i hi;

        ZPSE_TEXT_COMPACT8_128(w);
        _mm_storel_epi64((__m128i *)(void *)(out + p), w);
        hi = _mm_srli_si128(w, 8);
        _mm_storel_epi64((__m128i *)(void *)(out + p + 7u), hi);
        p += 14u;
        g += 2u;
    }
    for (; g < groups; g++) {
        uint64_t w, v;

        memcpy(&w, in + (size_t)g * 8u, sizeof w);
        v = text_compact8(w);
        if (p + 8u <= groups * 7u) {
            memcpy(out + p, &v, sizeof v);
        } else {
            memcpy(out + p, &v, 7u);
        }
        p += 7u;
    }
    if (rest != 0u) {
        uint64_t v = text_compact8(text_gather(in + groups * 8u, rest));
        uint32_t nb = text_group_out(rest);
        uint32_t k;

        v &= (UINT64_C(1) << (7u * rest)) - UINT64_C(1);
        for (k = 0u; k < nb; k++) {
            out[p++] = (uint8_t)(v >> (8u * k));
        }
    }
    return p;
}
#undef ZPSE_TEXT_COMPACT8_128
#endif /* ZPSE_TEXT_HAVE_SSE2 */

/* Pack dispatch shared by both encode entry points. The caller has already
 * checked n, the pointers, ASCII eligibility and capacity. */
static uint32_t text_pack_dispatch(const uint8_t *in, uint32_t n, uint8_t *dst)
{
#if ZPSE_TEXT_HAVE_BMI2
    if (text_bmi2()) {
        return text_pack_bmi2(in, n, dst);
    }
#endif
#if ZPSE_TEXT_HAVE_AVX2
    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        return text_pack_avx2(in, n, dst);
    }
#endif
#if ZPSE_TEXT_HAVE_SSE2
    return text_pack_sse2(in, n, dst);
#endif
    return text_pack_scalar(in, n, dst);
}

uint32_t zpse_text_encode(const uint8_t *restrict in, uint32_t n,
                          uint8_t *restrict dst, uint32_t cap)
{
    uint32_t need;

    if (n == 0u || in == NULL || dst == NULL) {
        return 0u;
    }
    if (!text_all_ascii(in, n)) {
        return 0u; /* non-ASCII input is ineligible for this form */
    }
    need = n - (n >> 3);
    if (cap < need) {
        return 0u;
    }
    return text_pack_dispatch(in, n, dst);
}

/* Precondition: every byte of in[0..n) is <128. The block writer establishes
 * this with its ASCII gate and zpse_text_price before handing over the exact
 * payload length; skipping the scan here is byte-for-byte identical to
 * zpse_text_encode for every eligible input and saves a full pass over the
 * block. The public entry point keeps its scan for direct callers. */
uint32_t zpse_text_encode_ascii(const uint8_t *restrict in, uint32_t n,
                                uint8_t *restrict dst, uint32_t cap)
{
    uint32_t need;

    if (n == 0u || in == NULL || dst == NULL) {
        return 0u;
    }
    need = n - (n >> 3);
    if (cap < need) {
        return 0u;
    }
    return text_pack_dispatch(in, n, dst);
}

/* ---- UNPACK ------------------------------------------------------------- */

/* Portable reference. Every store is guarded by a subtraction from the
 * remaining capacity (n-op for output, in_len-p for input) so neither buffer
 * can be overrun, and the routine reports exact fill. */
static int text_unpack_scalar(const uint8_t *in, uint32_t in_len,
                              uint8_t *out, uint32_t n)
{
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g, p = 0u;

    for (g = 0u; g < groups; g++) {
        uint64_t v, w;

        if (p + 8u <= in_len) {
            memcpy(&v, in + p, sizeof v);
        } else {
            v = text_gather(in + p, 7u);
        }
        w = text_spread8(v);
        memcpy(out + (size_t)g * 8u, &w, sizeof w);
        p += 7u;
    }
    if (rest != 0u) {
        uint32_t nb = text_group_out(rest);
        uint64_t v = text_gather(in + p, nb);
        uint64_t w = text_spread8(v);
        uint32_t k;

        if (p + nb > in_len) {
            return -1;
        }
        for (k = 0u; k < rest; k++) {
            out[groups * 8u + k] = (uint8_t)(w >> (8u * k));
        }
        p += nb;
    }
    return (p == in_len && groups * 8u + rest == n) ? 0 : -1;
}

#if ZPSE_TEXT_HAVE_BMI2
/* pdep is the exact inverse of pext with the same mask: the low 56 source
 * bits become the low seven bits of eight separate bytes. Four independent
 * pdeps per iteration overlap their latency. The unmasked 8-byte load is safe
 * because pdep ignores the high byte the mask does not select. */
static ZPSE_TEXT_BMI2_TARGET uint64_t text_load7(const uint8_t *p)
{
    uint64_t v;

    memcpy(&v, p, sizeof v);
    return _pdep_u64(v, UINT64_C(0x7F7F7F7F7F7F7F7F));
}

static ZPSE_TEXT_BMI2_TARGET int text_unpack_bmi2(const uint8_t *in,
                                                   uint32_t in_len,
                                                   uint8_t *out, uint32_t n)
{
    const uint64_t mask = UINT64_C(0x7F7F7F7F7F7F7F7F);
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 4u < groups) {
        uint64_t a = text_load7(in + p);
        uint64_t b = text_load7(in + p +  7u);
        uint64_t c = text_load7(in + p + 14u);
        uint64_t d = text_load7(in + p + 21u);

        memcpy(out + (size_t)g * 8u,        &a, sizeof a);
        memcpy(out + (size_t)g * 8u +  8u,  &b, sizeof b);
        memcpy(out + (size_t)g * 8u + 16u,  &c, sizeof c);
        memcpy(out + (size_t)g * 8u + 24u,  &d, sizeof d);
        g += 4u;
        p += 28u;
    }
    for (; g < groups; g++) {
        uint64_t v, w;

        if (p + 8u <= in_len) {
            memcpy(&v, in + p, sizeof v);
        } else {
            v = text_gather(in + p, 7u);
        }
        w = _pdep_u64(v, mask);
        memcpy(out + (size_t)g * 8u, &w, sizeof w);
        p += 7u;
    }
    if (rest != 0u) {
        uint32_t nb = text_group_out(rest);
        uint64_t v;
        uint64_t w;
        uint32_t k;

        if (p + nb > in_len) {
            return -1;
        }
        v = text_gather(in + p, nb);
        w = _pdep_u64(v, mask);
        for (k = 0u; k < rest; k++) {
            out[groups * 8u + k] = (uint8_t)(w >> (8u * k));
        }
        p += nb;
    }
    return (p == in_len && groups * 8u + rest == n) ? 0 : -1;
}
#endif /* ZPSE_TEXT_HAVE_BMI2 */

#if ZPSE_TEXT_HAVE_AVX2
/* E7's verified 7-bit unpack for one group: build a 16-bit window for each of
 * the eight output bytes with vpshufb, extract its 7-bit field with vpsrlvd,
 * gather byte 0 of each 32-bit lane with a second vpshufb, then splice the two
 * 128-bit halves into one 8-byte store. Byte 7 of the source is a don't-care
 * (the field's top bit is masked away), so one 8-byte load per group needs no
 * extra masking. Constants are 32-byte aligned for the vmovdqa loads. */
static const int8_t ztext_unpack_pos[32] __attribute__((aligned(32))) = {
     0,  1, -128, -128,   0,  1, -128, -128,
     1,  2, -128, -128,   2,  3, -128, -128,
     3,  4, -128, -128,   4,  5, -128, -128,
     5,  6, -128, -128,   6,  7, -128, -128
};
static const int32_t ztext_unpack_shift[8] __attribute__((aligned(32))) = {
     0, 7, 6, 5, 4, 3, 2, 1
};
static const int8_t ztext_unpack_gather[32] __attribute__((aligned(32))) = {
     0,  4,  8, 12, -128, -128, -128, -128,
    -128, -128, -128, -128, -128, -128, -128, -128,
     0,  4,  8, 12, -128, -128, -128, -128,
    -128, -128, -128, -128, -128, -128, -128, -128
};

#define ZPSE_TEXT_UNPACK8_AVX2(ip, rd) do {                                    \
    __m256i _pb = _mm256_broadcastsi128_si256(                                  \
        _mm_loadl_epi64((const __m128i *)(const void *)(ip)));                  \
    __m256i _w = _mm256_shuffle_epi8(_pb, _mm256_load_si256(                    \
        (const __m256i *)(const void *)ztext_unpack_pos));                      \
    __m256i _t = _mm256_srlv_epi32(_w, _mm256_load_si256(                       \
        (const __m256i *)(const void *)ztext_unpack_shift));                    \
    _t = _mm256_and_si256(_t, _mm256_set1_epi32(0x7f));                         \
    __m256i _pk = _mm256_shuffle_epi8(_t, _mm256_load_si256(                    \
        (const __m256i *)(const void *)ztext_unpack_gather));                   \
    (rd) = _mm_or_si128(_mm256_castsi256_si128(_pk),                            \
        _mm_slli_si128(_mm256_extracti128_si256(_pk, 1), 4));                   \
} while (0)

/* Write-back form: eight groups (56 input, 64 output) per iteration, no
 * alignment requirement on out. The p+56 < in_len guard keeps the 8-byte load
 * of the last group inside the caller's buffer; any remainder takes the exact
 * scalar tail below. */
static ZPSE_TEXT_AVX2_TARGET int text_unpack_avx2(const uint8_t *in,
                                                   uint32_t in_len,
                                                   uint8_t *out, uint32_t n)
{
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 8u <= groups && p + 56u < in_len) {
        __m128i r;

        ZPSE_TEXT_UNPACK8_AVX2(in + p, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)g * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p +  7u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 1u) * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 14u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 2u) * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 21u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 3u) * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 28u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 4u) * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 35u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 5u) * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 42u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 6u) * 8u), r);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 49u, r);
        _mm_storel_epi64((__m128i *)(void *)(out + (size_t)(g + 7u) * 8u), r);
        g += 8u;
        p += 56u;
    }
    for (; g < groups; g++) {
        uint64_t v, w;

        if (p + 8u <= in_len) {
            memcpy(&v, in + p, sizeof v);
        } else {
            v = text_gather(in + p, 7u);
        }
        w = text_spread8(v);
        memcpy(out + (size_t)g * 8u, &w, sizeof w);
        p += 7u;
    }
    if (rest != 0u) {
        uint32_t nb = text_group_out(rest);
        uint64_t v, w;
        uint32_t k;

        if (p + nb > in_len) {
            return -1;
        }
        v = text_gather(in + p, nb);
        w = text_spread8(v);
        for (k = 0u; k < rest; k++) {
            out[groups * 8u + k] = (uint8_t)(w >> (8u * k));
        }
        p += nb;
    }
    return (p == in_len && groups * 8u + rest == n) ? 0 : -1;
}

/* Non-temporal form: eight groups (56 input, 64 output) are assembled directly
 * in registers and emitted as four 16-byte streaming stores -- no scratch line,
 * so gcc never pays a store-to-load forward through a stack staging buffer.
 * Four 16-byte stores also beat two 32-byte ones here: gcc emits each 32-byte
 * store as a pair of 16-byte uops anyway, and the narrower form schedules them
 * sooner. The caller guarantees out is 32-byte aligned (so every 16-byte store
 * is aligned too) and n >= ZPSE_TEXT_NT_MIN; p advances in multiples of 56 and
 * g in multiples of 8, so g*8 is 64-byte aligned and each iteration ends at
 * (g+8)*8 <= groups*8 <= n -- no store reaches the tail or the caller's canary.
 * The sfence before the scalar tail drains the write-combining buffers both for
 * visibility and so a write-back tail byte sharing a line cannot resurrect
 * stale data. */
static ZPSE_TEXT_AVX2_TARGET int text_unpack_avx2_nt(const uint8_t *in,
                                                      uint32_t in_len,
                                                      uint8_t *out, uint32_t n)
{
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 8u <= groups && p + 56u < in_len) {
        __m128i a, b, c, d, lo, hi;

        ZPSE_TEXT_UNPACK8_AVX2(in + p,       a);
        ZPSE_TEXT_UNPACK8_AVX2(in + p +  7u, b);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 14u, c);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 21u, d);
        lo = _mm_or_si128(a, _mm_slli_si128(b, 8));
        hi = _mm_or_si128(c, _mm_slli_si128(d, 8));
        _mm_stream_si128((__m128i *)(void *)(out + (size_t)g * 8u), lo);
        _mm_stream_si128((__m128i *)(void *)(out + (size_t)g * 8u + 16u), hi);

        ZPSE_TEXT_UNPACK8_AVX2(in + p + 28u, a);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 35u, b);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 42u, c);
        ZPSE_TEXT_UNPACK8_AVX2(in + p + 49u, d);
        lo = _mm_or_si128(a, _mm_slli_si128(b, 8));
        hi = _mm_or_si128(c, _mm_slli_si128(d, 8));
        _mm_stream_si128((__m128i *)(void *)(out + (size_t)g * 8u + 32u), lo);
        _mm_stream_si128((__m128i *)(void *)(out + (size_t)g * 8u + 48u), hi);
        g += 8u;
        p += 56u;
    }
    _mm_sfence();
    for (; g < groups; g++) {
        uint64_t v, w;

        if (p + 8u <= in_len) {
            memcpy(&v, in + p, sizeof v);
        } else {
            v = text_gather(in + p, 7u);
        }
        w = text_spread8(v);
        memcpy(out + (size_t)g * 8u, &w, sizeof w);
        p += 7u;
    }
    if (rest != 0u) {
        uint32_t nb = text_group_out(rest);
        uint64_t v, w;
        uint32_t k;

        if (p + nb > in_len) {
            return -1;
        }
        v = text_gather(in + p, nb);
        w = text_spread8(v);
        for (k = 0u; k < rest; k++) {
            out[groups * 8u + k] = (uint8_t)(w >> (8u * k));
        }
        p += nb;
    }
    return (p == in_len && groups * 8u + rest == n) ? 0 : -1;
}
#undef ZPSE_TEXT_UNPACK8_AVX2
#endif /* ZPSE_TEXT_HAVE_AVX2 */

#if ZPSE_TEXT_HAVE_SSE2
/* spread8 on both 64-bit lanes of a __m128i. Two 8-byte loads rebuild one
 * group per lane: the low lane is group g (bytes p..p+6), the high lane is
 * group g+1 shifted up with _mm_slli_si128. The byte 7 of each lane is a
 * don't care that spread8 masks away, so the overlapping loads need no extra
 * masking. One 16-byte store emits the 16 decompressed bytes. */
#define ZPSE_TEXT_SPREAD8_128(v) do {                                         \
    __m128i _a = _mm_and_si128((v), zc0);                                      \
    __m128i _b = _mm_slli_epi64(_mm_and_si128((v), zc1), 1);                  \
    __m128i _c = _mm_slli_epi64(_mm_and_si128((v), zc2), 2);                  \
    __m128i _d = _mm_slli_epi64(_mm_and_si128((v), zc3), 3);                  \
    __m128i _e = _mm_slli_epi64(_mm_and_si128((v), zc4), 4);                  \
    __m128i _f = _mm_slli_epi64(_mm_and_si128((v), zc5), 5);                  \
    __m128i _g = _mm_slli_epi64(_mm_and_si128((v), zc6), 6);                  \
    __m128i _h = _mm_slli_epi64(_mm_and_si128((v), zc7), 7);                  \
    (v) = _mm_or_si128(                                                        \
        _mm_or_si128(_mm_or_si128(_a, _b), _mm_or_si128(_c, _d)),             \
        _mm_or_si128(_mm_or_si128(_e, _f), _mm_or_si128(_g, _h)));            \
} while (0)

static int text_unpack_sse2(const uint8_t *in, uint32_t in_len,
                             uint8_t *out, uint32_t n)
{
    const __m128i zc0 = _mm_set1_epi64x(INT64_C(0x000000000000007F));
    const __m128i zc1 = _mm_set1_epi64x(INT64_C(0x0000000000003F80));
    const __m128i zc2 = _mm_set1_epi64x(INT64_C(0x00000000001FC000));
    const __m128i zc3 = _mm_set1_epi64x(INT64_C(0x000000000FE00000));
    const __m128i zc4 = _mm_set1_epi64x(INT64_C(0x00000007F0000000));
    const __m128i zc5 = _mm_set1_epi64x(INT64_C(0x000003F800000000));
    const __m128i zc6 = _mm_set1_epi64x(INT64_C(0x0001FC0000000000));
    const __m128i zc7 = _mm_set1_epi64x(INT64_C(0x00FE000000000000));
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g = 0u, p = 0u;

    while (g + 2u < groups) {
        __m128i a = _mm_loadl_epi64(
            (const __m128i *)(const void *)(in + p));
        __m128i b = _mm_slli_si128(
            _mm_loadl_epi64(
                (const __m128i *)(const void *)(in + p + 7u)), 8);
        __m128i v = _mm_or_si128(a, b);

        ZPSE_TEXT_SPREAD8_128(v);
        _mm_storeu_si128((__m128i *)(void *)(out + (size_t)g * 8u), v);
        g += 2u;
        p += 14u;
    }
    for (; g < groups; g++) {
        uint64_t v, w;

        if (p + 8u <= in_len) {
            memcpy(&v, in + p, sizeof v);
        } else {
            v = text_gather(in + p, 7u);
        }
        w = text_spread8(v);
        memcpy(out + (size_t)g * 8u, &w, sizeof w);
        p += 7u;
    }
    if (rest != 0u) {
        uint32_t nb = text_group_out(rest);
        uint64_t v, w;
        uint32_t k;

        if (p + nb > in_len) {
            return -1;
        }
        v = text_gather(in + p, nb);
        w = text_spread8(v);
        for (k = 0u; k < rest; k++) {
            out[groups * 8u + k] = (uint8_t)(w >> (8u * k));
        }
        p += nb;
    }
    return (p == in_len && groups * 8u + rest == n) ? 0 : -1;
}
#undef ZPSE_TEXT_SPREAD8_128
#endif /* ZPSE_TEXT_HAVE_SSE2 */

int zpse_text_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                     uint32_t n)
{
    uint32_t expected;

    if (n == 0u || in == NULL || out == NULL) {
        return -1;
    }
    expected = n - (n >> 3);
    if (in_len != expected) {
        return -1; /* the only structural constraint of a fixed-rate pack:
                    * every bit pattern is payload, so length is all there is
                    * to validate. Bit 7 is routinely set on packed bytes. */
    }
/* E7 measured the vpshufb+vpsrlvd unpack above pdep on Zen 3 by a wide margin
 * (10.6 vs 5.4 GB/s in L2), so decode prefers AVX2 even though encode stays on
 * pext. A suitably aligned, large output streams through non-temporal stores;
 * small or misaligned output keeps the write-back form. Both are bit-exact
 * replacements for the scalar reference; the runtime AVX2 gate is unchanged. */
#if ZPSE_TEXT_HAVE_AVX2
    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        if (n >= ZPSE_TEXT_NT_MIN && ((uintptr_t)out & 31u) == 0u) {
            return text_unpack_avx2_nt(in, in_len, out, n);
        }
        return text_unpack_avx2(in, in_len, out, n);
    }
#endif
#if ZPSE_TEXT_HAVE_BMI2
    if (text_bmi2()) {
        return text_unpack_bmi2(in, in_len, out, n);
    }
#endif
#if ZPSE_TEXT_HAVE_SSE2
    return text_unpack_sse2(in, in_len, out, n);
#endif
    return text_unpack_scalar(in, in_len, out, n);
}

/* ---- PRICE -------------------------------------------------------------- */

uint32_t zpse_text_price(const uint8_t *in, uint32_t n)
{
    if (n == 0u || in == NULL) {
        return 0u;
    }
    if (!text_all_ascii(in, n)) {
        return 0u;
    }
    return n - (n >> 3);
}
