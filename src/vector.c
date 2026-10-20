#include "zpse/vector.h"
#include "zpse/status.h"

#include <string.h>

/* Unity-build safe: no globals, no allocation, helpers are `static`.
 * Scalar C paths are normative and bit-exact;
 * SIMD paths are dispatched at runtime and MUST match scalar output. */

/* ---- Architecture detection (compile time) ---- */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#define ZPSE_HAVE_X86 1
#else
#define ZPSE_HAVE_X86 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || \
    defined(_M_ARM64) || defined(__arm64__) || defined(__ARM64__)
#define ZPSE_HAVE_NEON_BASELINE 1
#else
#define ZPSE_HAVE_NEON_BASELINE 0
#endif

/* Guard all arch intrinsics/includes: x86-only headers below. */
#if ZPSE_HAVE_X86
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

/* GCC/Clang function-target attributes so SIMD intrinsics compile even
 * when the TU is built without -mssse3/-msse2. MSVC needs none. */
#if defined(__GNUC__) && ZPSE_HAVE_X86 && !defined(_MSC_VER)
#define ZPSE_TARGET_SSE2 __attribute__((target("sse2")))
#define ZPSE_TARGET_SSSE3 __attribute__((target("ssse3")))
#else
#define ZPSE_TARGET_SSE2
#define ZPSE_TARGET_SSSE3
#endif

static unsigned zpse_popcount8(uint8_t v)
{
    unsigned n = 0;
    while (v != 0) {
        n += (unsigned)(v & 1u);
        v = (uint8_t)(v >> 1);
    }
    return n;
}

#if ZPSE_HAVE_X86 && !defined(_MSC_VER) && !defined(ZPSE_FORCE_SCALAR)
static unsigned long long zpse_xgetbv_gcc(void)
{
    uint32_t lo = 0;
    uint32_t hi = 0;
    /* XGETBV with ECX=0; raw opcode avoids <immintrin.h> feature gates. */
    __asm__ volatile(".byte 0x0f,0x01,0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}
#endif

uint32_t zpse_cpu_caps(void)
{
#ifdef ZPSE_FORCE_SCALAR
    /* Conformance builds: pretend no SIMD exists so the scalar reference
     * paths run. tests/test_parity.c builds the same corpus both ways
     * and requires byte-identical output. */
    return 0;
#else
    /* CPUID is a serialising instruction and an expensive one under
     * virtualisation: measured at ~2.5 us per call on the development
     * machine, which is more than the work of an entire 256-byte tile.
     * Dispatch that asks once per tile therefore spends more on the
     * question than on the code it selects -- the AVX2 classify path was
     * 17x slower than the scalar one for exactly this reason. The answer
     * cannot change while the process runs, so compute it once and cache
     * it; the computation is idempotent and a racing thread can at worst
     * repeat it. */
    static uint32_t cached = 0xFFFFFFFFu; /* sentinel: not yet computed */
    uint32_t caps;

    if (cached != 0xFFFFFFFFu) {
        return cached;
    }
    caps = 0;

#if ZPSE_HAVE_NEON_BASELINE
    /* AArch64/NEON builds: ASIMD is an architectural baseline. */
    caps |= (uint32_t)ZPSE_CPU_NEON;
#endif

#if ZPSE_HAVE_X86
#if defined(_MSC_VER)
    {
        int info[4];
        __cpuid(info, 1);
        if (info[3] & (1 << 26)) {
            caps |= (uint32_t)ZPSE_CPU_SSE2;
        }
        if (info[2] & (1 << 9)) {
            caps |= (uint32_t)ZPSE_CPU_SSSE3;
        }
        if ((info[2] & (1 << 28)) && (info[2] & (1 << 27))) {
            unsigned long long xcr0 = _xgetbv(0);
            if ((xcr0 & 0x6ULL) == 0x6ULL) {
                __cpuid(info, 0);
                if (info[0] >= 7) {
                    __cpuidex(info, 7, 0);
                    if (info[1] & (1 << 5)) {
                        caps |= (uint32_t)ZPSE_CPU_AVX2;
                    }
                    /* AVX512 = BW (EBX bit 30) + VBMI2 for vpexpandb
                     * (ECX bit 6), plus opmask/ZMM OS state. */
                    if ((info[1] & (1 << 30)) && (info[2] & (1 << 6))) {
                        if ((xcr0 & 0xE0ULL) == 0xE0ULL) {
                            caps |= (uint32_t)ZPSE_CPU_AVX512;
                        }
                    }
                }
            }
        }
    }
#else
    {
        unsigned int a = 0;
        unsigned int b = 0;
        unsigned int c = 0;
        unsigned int d = 0;
        if (__get_cpuid(1, &a, &b, &c, &d)) {
            if (d & (1u << 26)) {
                caps |= (uint32_t)ZPSE_CPU_SSE2;
            }
            if (c & (1u << 9)) {
                caps |= (uint32_t)ZPSE_CPU_SSSE3;
            }
            if ((c & (1u << 28)) && (c & (1u << 27))) {
                unsigned long long xcr0 = zpse_xgetbv_gcc();
                if ((xcr0 & 0x6ULL) == 0x6ULL) {
                    unsigned int maxleaf = __get_cpuid_max(0, NULL);
                    if (maxleaf >= 7u) {
                        unsigned int a7 = 0;
                        unsigned int b7 = 0;
                        unsigned int c7 = 0;
                        unsigned int d7 = 0;
                        __get_cpuid_count(7, 0, &a7, &b7, &c7, &d7);
                        (void)a7;
                        (void)d7;
                        if (b7 & (1u << 5)) {
                            caps |= (uint32_t)ZPSE_CPU_AVX2;
                        }
                        /* AVX512 = BW (EBX bit 30) + VBMI2 for vpexpandb
                         * (ECX bit 6), plus opmask/ZMM OS state. */
                        if ((b7 & (1u << 30)) && (c7 & (1u << 6))) {
                            if ((xcr0 & 0xE0ULL) == 0xE0ULL) {
                                caps |= (uint32_t)ZPSE_CPU_AVX512;
                            }
                        }
                    }
                }
            }
        }
    }
#endif
#endif

    cached = caps;
    return caps;
#endif /* ZPSE_FORCE_SCALAR */
}

/* ---- Inverse transpose-8 ---- */

static void zpse_inv_transpose8_scalar(uint8_t *dst, const uint8_t *segs[8],
                                       size_t seg_offset)
{
    uint8_t lane[8][8];
    size_t s;
    size_t i;
    for (s = 0; s < 8; s++) {
        /* memcpy: no unaligned UB, no alignment assumption. */
        memcpy(lane[s], segs[s] + seg_offset, 8);
    }
    for (i = 0; i < 8; i++) {
        for (s = 0; s < 8; s++) {
            dst[8 * i + s] = lane[s][i];
        }
    }
}

#if ZPSE_HAVE_X86
/* 8x8 byte transpose via unpack matrix (128-bit unpacks; each __m128i
 * holds one lane's 8 B). Dispatched when AVX2 is present (AVX2 implies
 * the SSE2 unpack path used here). Bit-exact: pure byte shuffles. */
static ZPSE_TARGET_SSE2 void zpse_inv_transpose8_unpack(
    uint8_t *dst, const uint8_t *segs[8], size_t seg_offset)
{
    __m128i a0;
    __m128i a1;
    __m128i a2;
    __m128i a3;
    __m128i a4;
    __m128i a5;
    __m128i a6;
    __m128i a7;
    __m128i t0;
    __m128i t1;
    __m128i t2;
    __m128i t3;
    __m128i u0;
    __m128i u1;
    __m128i u2;
    __m128i u3;
    __m128i v0;
    __m128i v1;
    __m128i v2;
    __m128i v3;

    a0 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[0] + seg_offset));
    a1 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[1] + seg_offset));
    a2 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[2] + seg_offset));
    a3 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[3] + seg_offset));
    a4 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[4] + seg_offset));
    a5 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[5] + seg_offset));
    a6 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[6] + seg_offset));
    a7 = _mm_loadl_epi64((const __m128i *)(const void *)(segs[7] + seg_offset));

    t0 = _mm_unpacklo_epi8(a0, a1);
    t1 = _mm_unpacklo_epi8(a2, a3);
    t2 = _mm_unpacklo_epi8(a4, a5);
    t3 = _mm_unpacklo_epi8(a6, a7);

    u0 = _mm_unpacklo_epi16(t0, t1);
    u1 = _mm_unpackhi_epi16(t0, t1);
    u2 = _mm_unpacklo_epi16(t2, t3);
    u3 = _mm_unpackhi_epi16(t2, t3);

    /* v0 = rows 0,1; v2 = rows 2,3; v1 = rows 4,5; v3 = rows 6,7. */
    v0 = _mm_unpacklo_epi32(u0, u2);
    v1 = _mm_unpacklo_epi32(u1, u3);
    v2 = _mm_unpackhi_epi32(u0, u2);
    v3 = _mm_unpackhi_epi32(u1, u3);

    _mm_storeu_si128((__m128i *)(void *)(dst + 0), v0);
    _mm_storeu_si128((__m128i *)(void *)(dst + 16), v2);
    _mm_storeu_si128((__m128i *)(void *)(dst + 32), v1);
    _mm_storeu_si128((__m128i *)(void *)(dst + 48), v3);
}
#endif

void zpse_inv_transpose8_64B(uint8_t *dst, const uint8_t *segs[8],
                             size_t seg_offset)
{
    size_t s;
    if (dst == 0 || segs == 0) {
        return;
    }
    for (s = 0; s < 8; s++) {
        if (segs[s] == 0) {
            return;
        }
    }
#if ZPSE_HAVE_X86
    if (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) {
        zpse_inv_transpose8_unpack(dst, segs, seg_offset);
        return;
    }
#endif
    zpse_inv_transpose8_scalar(dst, segs, seg_offset);
}

/* ---- LANE_DELTA prediction over 64 B ----
 * 64 B = 8 steps x 8 lanes. History evolves across steps: for each lane
 * L, step q predicts (2*last - prev) mod 256, then last/prev advance to
 * (pred, last). This matches the scalar LANE_DELTA recurrence of
 * applied in position order. */

static void zpse_lane_delta_scalar(uint8_t *out, const uint8_t last[8],
                                   const uint8_t prev[8])
{
    uint8_t cur_last[8];
    uint8_t cur_prev[8];
    unsigned q;
    unsigned lane;
    memcpy(cur_last, last, 8);
    memcpy(cur_prev, prev, 8);
    for (q = 0; q < 8; q++) {
        for (lane = 0; lane < 8; lane++) {
            unsigned p = (2u * (unsigned)cur_last[lane] -
                          (unsigned)cur_prev[lane]) &
                         0xFFu;
            out[8 * q + lane] = (uint8_t)p;
        }
        for (lane = 0; lane < 8; lane++) {
            cur_prev[lane] = cur_last[lane];
            cur_last[lane] = out[8 * q + lane];
        }
    }
}

#if ZPSE_HAVE_X86
/* Byte-vector stepping: sub/add_epi8 are mod-256 lane arithmetic, hence
 * bit-exact to the scalar recurrence. */
static ZPSE_TARGET_SSE2 void zpse_lane_delta_vec(uint8_t *out,
                                                 const uint8_t last[8],
                                                 const uint8_t prev[8])
{
    __m128i L;
    __m128i P;
    unsigned q;
    L = _mm_loadl_epi64((const __m128i *)(const void *)last);
    P = _mm_loadl_epi64((const __m128i *)(const void *)prev);
    for (q = 0; q < 8; q++) {
        __m128i D = _mm_sub_epi8(L, P);
        __m128i X = _mm_add_epi8(L, D);
        _mm_storel_epi64((__m128i *)(void *)(out + 8 * q), X);
        P = L;
        L = X;
    }
}
#endif

void zpse_predict_lane_delta_64B(uint8_t *pred_out, const uint8_t last[8],
                                 const uint8_t prev[8])
{
    if (pred_out == 0 || last == 0 || prev == 0) {
        return;
    }
#if ZPSE_HAVE_X86
    if (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) {
        zpse_lane_delta_vec(pred_out, last, prev);
        return;
    }
#endif
    zpse_lane_delta_scalar(pred_out, last, prev);
}

/* ---- Monolithic reconstruct: RAW_BITMAP + EXCLUSION_U8 ----
 * Scalar bit-exact reference. Bit p (LSB-first) set =>
 * exception: consume next residual r, x = (r < pred) ? r : r + 1;
 * else x = pred. AVX-512 (vpexpandb) path is intentionally omitted;
 * scalar is used on all arches (allowed: SIMD paths are optional). */

int zpse_reconstruct_bitmap_excl_u8(uint8_t *dst, const uint8_t *pred,
                                    const uint8_t *bitmap, uint16_t nt,
                                    const uint8_t *residuals, uint16_t res_len)
{
    unsigned consumed = 0;
    unsigned p;
    if (dst == 0 || pred == 0) {
        return PSE_BAD_ARG;
    }
    if (nt < 1u || nt > 256u) {
        return PSE_BAD_TILE_SIZE; /* Nt normative range 1..256 (Sec. 6.2) */
    }
    if (nt > 0u && bitmap == 0) {
        return PSE_BAD_ARG;
    }
    for (p = 0; p < (unsigned)nt; p++) {
        unsigned bit = (unsigned)((bitmap[p >> 3] >> (p & 7u)) & 1u);
        if (bit) {
            unsigned r;
            unsigned ph;
            if (residuals == 0) {
                return PSE_BAD_ARG;
            }
            if (consumed >= (unsigned)res_len) {
                return PSE_TRUNCATED;
            }
            r = (unsigned)residuals[consumed];
            consumed++;
            /* EXCLUSION_U8 residuals are in [0,254]; 255 is forbidden
             * outside RAW_ACTUAL_BYTES. */
            if (r > 254u) {
                return PSE_BAD_RESIDUAL;
            }
            ph = (unsigned)pred[p];
            dst[p] = (uint8_t)((r < ph) ? r : (r + 1u));
        } else {
            dst[p] = pred[p];
        }
    }
    return (int)consumed;
}

/* ---- AVX2 8-B compaction helper ---- */

static int zpse_compact8_scalar(uint8_t *dst, const uint8_t data[8],
                                uint8_t mask8)
{
    int n = 0;
    unsigned i;
    for (i = 0; i < 8; i++) {
        if ((mask8 >> i) & 1u) {
            dst[n] = data[i];
            n++;
        }
    }
    return n;
}

#if ZPSE_HAVE_X86
/* pshufb LUT path: 8-B entry at lut + mask*8; 0x80 entries zero the
 * corresponding output byte, of which only the first popcount(mask)
 * bytes are stored, so padding is ignored per scalar semantics. */
static ZPSE_TARGET_SSSE3 int zpse_compact8_pshufb(
    uint8_t *dst, const uint8_t data[8], uint8_t mask8, const uint8_t *lut_2k)
{
    __m128i d;
    __m128i sh;
    __m128i r;
    uint8_t tmp[16];
    int n;
    d = _mm_loadl_epi64((const __m128i *)(const void *)data);
    sh = _mm_loadl_epi64(
        (const __m128i *)(const void *)(lut_2k + (size_t)mask8 * 8u));
    r = _mm_shuffle_epi8(d, sh);
    _mm_storeu_si128((__m128i *)(void *)tmp, r);
    n = (int)zpse_popcount8(mask8);
    memcpy(dst, tmp, (size_t)n);
    return n;
}
#endif

int zpse_compact8(uint8_t *dst, const uint8_t data[8], uint8_t mask8,
                  const uint8_t *lut_2k)
{
    if (dst == 0 || data == 0) {
        return PSE_BAD_ARG;
    }
#if ZPSE_HAVE_X86
    if (lut_2k != 0 && (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_SSSE3)) {
        return zpse_compact8_pshufb(dst, data, mask8, lut_2k);
    }
#else
    (void)lut_2k;
#endif
    return zpse_compact8_scalar(dst, data, mask8);
}
