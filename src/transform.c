#include "zpse/transform.h"
#include "zpse/constants.h"
#include "zpse/hot.h"
#include "zpse/status.h"
#include "zpse/vector.h"

#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 * SSE2 fast paths for the transform widths. For a block of 16 bytes or
 * more `zpse_block_encode` only ever forwards transform ids 0 (RAW_BYTES),
 * 6 (TRANSPOSE_16) and 1 (DELTA8), so the width `w` is 1 or 16 there; the
 * narrow widths (2, 4 and 8 -- the plain and delta transposes) are decoded
 * from streams other encoders produce, and they are the slowest shapes in
 * the format, so they get the same treatment: a parity split, or its
 * inverse merge, in place of one strided byte per load and per store.
 * Every path below is byte-for-byte the scalar loops' output, which stay
 * the normative reference.
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

/* ------------------------------------------------------------------ *
 * SSE2 prefix scans for the delta8 / xor8 chains.
 *
 * A delta8 or xor8 stream is a prefix scan: out[j] = enc[j] (op) out[j-1]
 * with out[-1] = seed, so byte j needs byte j-1. But only the *carry*
 * truly chains: inside one 16-byte block the four log-shift steps build
 * every lane's partial prefix with no cross-lane dependency at all, and
 * the block's total is just its last lane. So the scan runs over a chunk
 * of eight such blocks -- eight completely independent ladders, each
 * producing both its prefix and its total -- then a short scalar carry
 * chain over the eight block totals, then one broadcast-and-op per block.
 * That puts eight independent ladders and eight independent stores
 * between two links of the carry chain instead of one; the arithmetic per
 * byte is exactly the arithmetic the scalar loop does.
 *
 * SSE2, like every other fast path in this file: architectural on x86-64,
 * so there is no runtime dispatch, the -DZPSE_FORCE_SCALAR conformance
 * build runs these very intrinsics, and the scalar/SIMD parity test
 * cannot observe a difference here by construction. Every length is
 * handled: whole 128-byte chunks, then whole 16-byte blocks, then the
 * scalar remainder, all three continuing from the same carry.
 * ------------------------------------------------------------------ */
#if ZPSE_T_HAVE_SSE2
#define ZPSE_ADD8(a_, b_) ((uint8_t)((uint8_t)(a_) + (uint8_t)(b_)))
#define ZPSE_XOR8(a_, b_) ((uint8_t)((uint8_t)(a_) ^ (uint8_t)(b_)))

/* Defines `kernel_name(src, dst, n, seed)` for one of the two operators:
 * OPV is the 128-bit byte-wise op and OP8 its scalar twin. */
#define ZPSE_DEFINE_PREFIX_SCAN(kernel_name, OPV, OP8)                       \
static void kernel_name(const uint8_t *src, uint8_t *dst, uint32_t n,        \
                        uint8_t seed)                                       \
{                                                                           \
    uint32_t j = 0;                                                         \
    uint8_t pv = seed;                                                      \
                                                                            \
    if (src == 0 || dst == 0 || n == 0u) {                                  \
        return;                                                             \
    }                                                                       \
    /* Whole 128-byte chunks: eight independent ladders, then the block    \
     * totals folded into one carry per block, then the blocks rebased.    \
     * The carry for block b is the running value BEFORE block b, which    \
     * is the exclusive prefix of the totals over the chunks before it     \
     * plus the seed, and every lane of the block is missing exactly that  \
     * constant, so one broadcast add/xor per block applies it. */         \
    for (; (size_t)j + 128u <= (size_t)n; j += 128u) {                      \
        __m128i v[8];                                                       \
        uint8_t tot[8];                                                     \
        uint8_t acc = pv;                                                   \
        unsigned b;                                                         \
                                                                            \
        for (b = 0; b < 8u; b++) {                                          \
            __m128i x = _mm_loadu_si128((const __m128i *)(const void *)     \
                                        (src + (size_t)j +                  \
                                         (size_t)b * 16u));                 \
            x = OPV(x, _mm_slli_si128(x, 1));                               \
            x = OPV(x, _mm_slli_si128(x, 2));                               \
            x = OPV(x, _mm_slli_si128(x, 4));                               \
            x = OPV(x, _mm_slli_si128(x, 8));                               \
            v[b] = x;                                                       \
            /* Word 7 holds bytes 14 and 15, so the total is its HIGH      \
             * byte: a plain uint8_t cast of the word would keep byte 14    \
             * and carry the wrong value into every later block. */         \
            tot[b] = (uint8_t)(_mm_extract_epi16(x, 7) >> 8);               \
        }                                                                   \
        for (b = 0; b < 8u; b++) {                                          \
            uint8_t c = acc;                                                \
            acc = OP8(acc, tot[b]);                                         \
            tot[b] = c;                                                     \
        }                                                                   \
        for (b = 0; b < 8u; b++) {                                          \
            _mm_storeu_si128((__m128i *)(void *)(dst + (size_t)j +          \
                                               (size_t)b * 16u),            \
                             OPV(v[b], _mm_set1_epi8((char)tot[b])));        \
        }                                                                   \
        pv = acc;                                                           \
    }                                                                       \
    /* Whole 16-byte blocks: one ladder plus its carry. */                  \
    for (; (size_t)j + 16u <= (size_t)n; j += 16u) {                        \
        __m128i x = _mm_loadu_si128(                                        \
            (const __m128i *)(const void *)(src + (size_t)j));              \
        x = OPV(x, _mm_slli_si128(x, 1));                                   \
        x = OPV(x, _mm_slli_si128(x, 2));                                   \
        x = OPV(x, _mm_slli_si128(x, 4));                                   \
        x = OPV(x, _mm_slli_si128(x, 8));                                   \
        x = OPV(x, _mm_set1_epi8((char)pv));                                \
        _mm_storeu_si128((__m128i *)(void *)(dst + (size_t)j), x);          \
        pv = (uint8_t)(_mm_extract_epi16(x, 7) >> 8);                       \
    }                                                                       \
    /* Scalar remainder (fewer than 16 bytes left, or n < 16 overall). */   \
    for (; j < n; j++) {                                                    \
        pv = OP8(pv, src[j]);                                               \
        dst[j] = pv;                                                        \
    }                                                                       \
}

ZPSE_DEFINE_PREFIX_SCAN(zpse_prefix_sum16, _mm_add_epi8, ZPSE_ADD8)
ZPSE_DEFINE_PREFIX_SCAN(zpse_prefix_xor16, _mm_xor_si128, ZPSE_XOR8)
#endif /* ZPSE_T_HAVE_SSE2 */

/* ------------------------------------------------------------------ *
 * AVX2 prefix scans: the same eight-ladders-per-carry structure as the
 * SSE2 kernels above, one width wider, so a carry link now covers 256
 * bytes instead of 128.
 *
 * A 32-byte block needs five log steps where a 16-byte block needs four.
 * The first four are the in-lane shifts (1, 2, 4, 8) that build each
 * 16-byte lane's own inclusive prefix, exactly as in the SSE2 ladder.
 * The fifth is the step ACROSS the 128-bit lane boundary, and it is NOT a
 * plain 16-byte shift of the register: after the four in-lane steps the
 * high lane still needs the low lane's TOTAL -- its last byte, byte 15 --
 * added to every one of its bytes, while shifting the low lane up by 16
 * bytes would add the low lane's per-position prefixes and leave every
 * position but the last one wrong. So the low lane's total is broadcast
 * (`zpse_avx2_lane_total`: one vpshufb plus one lane permute) and added to
 * the high lane, and because that helper zeroes the low lane the low lane
 * is left untouched. The block's total is byte 31, the HIGH byte of
 * 16-bit word 15 -- a plain cast of the word would keep byte 30, the same
 * trap the SSE2 kernels document.
 *
 * Unlike the SSE2 kernels this needs runtime dispatch, because AVX2 is
 * not architectural: each kernel carries the per-function target
 * attribute so a translation unit built without -mavx2 still compiles
 * them, and execution is gated on zpse_cpu_caps(), which answers from a
 * cache and returns 0 under -DZPSE_FORCE_SCALAR. A machine without AVX2,
 * a non-x86 build and the conformance build therefore keep the SSE2 or
 * scalar loops, and since all three are bit-exact the scalar/SIMD parity
 * test still compares like for like.
 * ------------------------------------------------------------------ */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_T_AVX2 __attribute__((target("avx2")))
#define ZPSE_T_HAVE_AVX2 1
#elif defined(__AVX2__)
#define ZPSE_T_AVX2
#define ZPSE_T_HAVE_AVX2 1
#else
#define ZPSE_T_AVX2
#define ZPSE_T_HAVE_AVX2 0
#endif
#else
#define ZPSE_T_AVX2
#define ZPSE_T_HAVE_AVX2 0
#endif

#if ZPSE_T_HAVE_AVX2
#include <immintrin.h>

/* Runtime gate for the AVX2 kernels, cached by zpse_cpu_caps(). */
static int zpse_t_avx2(void)
{
    return (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u;
}

/* Defines `kernel_name(src, dst, n, seed)` for one of the two operators:
 * OPV is the 256-bit byte-wise op and OP8 its scalar twin. The tail is
 * scalar, which is what keeps every length exact, so no masked store or
 * over-read is needed anywhere. */
/* After the four in-lane shift/op steps each 128-bit half of x holds its
 * own inclusive prefix, so the high half still needs the low half's TOTAL
 * added to every byte of it -- not the low half's prefixes, which is what
 * adding the shifted vector would do. Broadcast that single byte (the low
 * half's last byte) into the high half and leave the low half zero.
 *
 * The byte to broadcast is byte 15, the high byte of word 7 of the LOW
 * half. Selecting it with `vpshufb` and then moving that half up costs two
 * shuffle-port ops and leaves the total in the vector domain, where the
 * obvious `_mm_extract_epi16` / `_mm_set1_epi8` / `_mm256_inserti128_si256`
 * form pays a vector->GPR->vector round trip (two extra shuffle-port ops
 * plus three to five cycles of latency) on the critical path of every
 * 32-byte block. Shuffles are the binding resource in this kernel -- four
 * in-lane `vpslldq` per 32 bytes cannot be avoided, everything else can --
 * so the round trip goes. */
static const uint8_t zpse_avx2_ctl15[32] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
};

/* Same idea as the control above, but with byte 15 selected in BOTH lanes,
 * for `zpse_avx2_bcast_hi_byte` (the high half's byte 15 is mirrored into
 * the low half first, so one table serves both halves). */
static const uint8_t zpse_avx2_ctl15_all[32] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};

/* Both helpers take their shuffle control as a parameter: it is one of the
 * two constant tables above, loaded once by the kernel, because a byte-wise
 * scan runs these helpers twice per 32-byte block and a register the kernel
 * already holds beats a memory operand in that position. */
static ZPSE_T_AVX2 __m256i zpse_avx2_lane_total(__m256i ctl, __m256i x)
{
    /* vpshufb is per 128-bit lane, so index 15 in the low lane copies
     * byte 15 into every byte of that lane, and the 0x80 indices (bit 7
     * set) zero the high lane. The permute then puts that broadcast into
     * the high half and zeroes the low one. */
    __m256i t = _mm256_shuffle_epi8(x, ctl);

    return _mm256_permute2x128_si256(t, t, 0x08);
}

/* Broadcast a 32-byte block's LAST byte (its total, for this scan) to all
 * 32 bytes of the result. Byte 31 is the high byte of word 15 in the high
 * 128-bit lane, so the permute first mirrors that lane into the low one
 * (imm 0x11 selects a.high for both halves) and the shuffle then copies
 * byte 15 of each lane into every byte of that lane. Two shuffle-port ops
 * and no vector->GPR round trip, which is what lets the chunk loop below
 * keep the whole carry chain in vector registers. */
static ZPSE_T_AVX2 __m256i zpse_avx2_bcast_hi_byte(__m256i ctl, __m256i x)
{
    __m256i m = _mm256_permute2x128_si256(x, x, 0x11);

    return _mm256_shuffle_epi8(m, ctl);
}

#define ZPSE_DEFINE_PREFIX_SCAN32(kernel_name, OPV, OP8)                     \
static ZPSE_T_AVX2 void kernel_name(const uint8_t *src, uint8_t *dst,        \
                                    uint32_t n, uint8_t seed)               \
{                                                                           \
    uint32_t j = 0;                                                         \
    uint8_t pv = seed;                                                      \
                                                                            \
    if (src == 0 || dst == 0 || n == 0u) {                                  \
        return;                                                             \
    }                                                                       \
    const __m256i ctl_lo = _mm256_loadu_si256(                              \
        (const __m256i *)(const void *)zpse_avx2_ctl15);                    \
    const __m256i ctl_all = _mm256_loadu_si256(                             \
        (const __m256i *)(const void *)zpse_avx2_ctl15_all);                \
    /* Whole 256-byte chunks: eight independent 32-byte ladders, the carry  \
     * chain kept in a vector register, and one broadcast op per block. */ \
    for (; (size_t)j + 256u <= (size_t)n; j += 256u) {                      \
        /* The carry lives in a vector register as a 32-byte broadcast of   \
         * the running value, one block at a time: block b is rebased by    \
         * the broadcast of everything before it, and the broadcast for     \
         * the next block is this block's own total, which is final as      \
         * soon as its five ladder steps are done. Nothing leaves the       \
         * vector domain inside the chunk -- no per-block 256-bit extract,  \
         * no eight-byte `tot[]` array and no second pass over the blocks   \
         * -- which is what a byte-wise scan wants, because its binding     \
         * resource is shuffle-port throughput: four `vpslldq` per 32       \
         * bytes are irreducible, and the per-block bookkeeping around      \
         * them is the part that can be made cheaper. The op below is the   \
         * same op the scalar chain used, applied to a broadcast, and       \
         * broadcasting commutes with it, so this is the same arithmetic.   \
         */                                                                 \
        __m256i c = _mm256_set1_epi8((char)pv);                             \
        unsigned b;                                                         \
                                                                            \
        for (b = 0; b < 8u; b++) {                                          \
            __m256i x = _mm256_loadu_si256((const __m256i *)(const void *)  \
                                           (src + (size_t)j +               \
                                            (size_t)b * 32u));              \
            __m256i nb;                                                     \
                                                                            \
            x = OPV(x, _mm256_slli_si256(x, 1));                            \
            x = OPV(x, _mm256_slli_si256(x, 2));                            \
            x = OPV(x, _mm256_slli_si256(x, 4));                            \
            x = OPV(x, _mm256_slli_si256(x, 8));                            \
            x = OPV(x, zpse_avx2_lane_total(ctl_lo, x));                    \
            /* The block's total, broadcast, taken BEFORE the carry is      \
             * folded in: byte 31 of the rebased block is the total op the  \
             * carry, and `c` is the carry broadcast, so folding first      \
             * would fold it in twice for the additive scan. */             \
            nb = zpse_avx2_bcast_hi_byte(ctl_all, x);                       \
            _mm256_storeu_si256((__m256i *)(void *)(dst + (size_t)j +       \
                                                   (size_t)b * 32u),        \
                                OPV(x, c));                                 \
            c = OPV(c, nb);                                                 \
        }                                                                   \
        /* Every byte of c is the running value after the chunk, so any     \
         * one of them is the scalar `pv` the tail loops continue from. */  \
        pv = (uint8_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(c));         \
    }                                                                       \
    /* Whole 32-byte blocks: one ladder plus its carry. */                  \
    for (; (size_t)j + 32u <= (size_t)n; j += 32u) {                        \
        __m256i x = _mm256_loadu_si256(                                     \
            (const __m256i *)(const void *)(src + (size_t)j));              \
        x = OPV(x, _mm256_slli_si256(x, 1));                                \
        x = OPV(x, _mm256_slli_si256(x, 2));                                \
        x = OPV(x, _mm256_slli_si256(x, 4));                                \
        x = OPV(x, _mm256_slli_si256(x, 8));                                \
        x = OPV(x, zpse_avx2_lane_total(ctl_lo, x));                        \
        x = OPV(x, _mm256_set1_epi8((char)pv));                             \
        _mm256_storeu_si256((__m256i *)(void *)(dst + (size_t)j), x);       \
        /* The carry into the next block is this block's last byte, which  \
         * is byte 31 == the high byte of word 15 as in the chunk loop. */ \
        pv = (uint8_t)(_mm256_extract_epi16(x, 15) >> 8);                   \
    }                                                                       \
    /* Scalar remainder (fewer than 32 bytes left, or n < 32 overall). */   \
    for (; j < n; j++) {                                                    \
        pv = OP8(pv, src[j]);                                               \
        dst[j] = pv;                                                        \
    }                                                                       \
}

ZPSE_DEFINE_PREFIX_SCAN32(zpse_prefix_sum32, _mm256_add_epi8, ZPSE_ADD8)
ZPSE_DEFINE_PREFIX_SCAN32(zpse_prefix_xor32, _mm256_xor_si256, ZPSE_XOR8)

/* ------------------------------------------------------------------ *
 * AVX2 16x16 byte transpose.
 *
 * The 128-bit network in zpse_t16x16 above is op-optimal -- four stages of
 * unpack operations, four per input row -- but it needs sixteen live
 * 16-byte values per stage, and a 16-entry stage array plus the caller's
 * row array will not fit the SSE2 register file, so the intermediates
 * live in memory: measured, the inverse of a 16-segment block (transform
 * id 6) costs 8.2 us per 64 KiB, about 2.8x the unpack-op floor for the
 * shape, and store/reload traffic is most of the difference. This kernel
 * runs the same network on 256-bit registers, where one unpack processes
 * both 128-bit lanes and therefore does two rows' worth of work, and it
 * keeps at most nine live registers, so nothing needs to spill.
 *
 * Layout: a[k] = [row k | row k+8]. Every unpack in the network acts per
 * 128-bit lane with identical immediate semantics, so the low lane
 * transposes rows 0..7 by itself while the high lane transposes rows
 * 8..15, both following the same index pattern:
 *
 *   stage 1  t = unpack{lo,hi}_epi8   on the (a0,a1) (a2,a3) (a4,a5) (a6,a7) pairs
 *   stage 2  u = unpack{lo,hi}_epi16  on the (t0,t2) (t1,t3) (t4,t6) (t5,t7) pairs
 *   stage 3  v = unpack{lo,hi}_epi32  on the (u0,u4) (u1,u5) (u2,u6) (u3,u7) pairs
 *   stage 4  w = unpack{lo,hi}_epi64  on the (v0,v4) (v1,v5) (v2,v6) (v3,v7) pairs
 *
 * which leaves w[k] = [column k of its eight rows | column k+8 of them],
 * eight bytes each: the low lane carries rows 0..7's two columns and the
 * high lane rows 8..15's. Column k is therefore [w[k].low[0..7] |
 * w[k].high[0..7]] and column k+8 is [w[k].low[8..15] |
 * w[k].high[8..15]] -- one extract and two 64-bit unpacks per register,
 * so 16 columns come out of 8 registers.
 *
 * Same contract as zpse_t16x16: in[r] is row r on entry, out[c] receives
 * column c, so the two are bit-identical. The AVX2 form runs when
 * zpse_cpu_caps() reports AVX2 and the 128-bit one otherwise, including
 * under -DZPSE_FORCE_SCALAR -- which is what lets the parity suite keep
 * comparing two builds.
 * ------------------------------------------------------------------ */
#define ZPSE_T16_ROW128(p_) ((const __m128i *)(const void *)(p_))

/* Groups of sixteen rows the forward transpose collects before writing
 * them out; see zpse_t16x16_forward_blocked. */
#ifndef ZPSE_T16_GROUPS
#define ZPSE_T16_GROUPS 32u
#endif

/* Groups of sixteen rows the inverse stages per plane before transposing
 * them; see zpse_t16x16_inverse_blocked. It sat at four (64 bytes, one
 * cache line per plane) while the forward half of this file collects
 * thirty-two, so each 4 KiB-strided line load bought the inverse only
 * four groups. Staging 256 bytes per plane -- sixteen groups, 4 KiB of
 * buffer -- measured 3100 -> 2760 ns per 64 KiB for this transform alone
 * (G = 4/8/16/32 giving 3100/2950/2760/2700, best of six runs each),
 * about +12% and in line with the forward's blocking. Thirty-two groups
 * (8 KiB) is no faster in isolation and slower in place: the transpose's
 * output has to share L1 with that much staging. */
#ifndef ZPSE_T16_ISTAGE
#define ZPSE_T16_ISTAGE 16u
#endif

/* One 16-row group: 16 rows of 16 bytes become 16 columns of 16 bytes.
 * The sixteen bases are passed once for the whole plane and the group's
 * byte offset separately, so a caller that walks many groups builds no
 * pointer array per group; assembling those two sixteen-entry arrays was
 * a third of this kernel's cost on a 16-way transposed plane. */
static ZPSE_T_AVX2 void zpse_t16x16_avx2(const uint8_t *const in[16],
                                         uint32_t in_off,
                                         uint8_t *const out[16],
                                         uint32_t out_off)
{
    const __m256i a0 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[8] + in_off),
                                           ZPSE_T16_ROW128(in[0] + in_off));
    const __m256i a1 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[9] + in_off),
                                           ZPSE_T16_ROW128(in[1] + in_off));
    const __m256i a2 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[10] + in_off),
                                           ZPSE_T16_ROW128(in[2] + in_off));
    const __m256i a3 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[11] + in_off),
                                           ZPSE_T16_ROW128(in[3] + in_off));
    const __m256i a4 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[12] + in_off),
                                           ZPSE_T16_ROW128(in[4] + in_off));
    const __m256i a5 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[13] + in_off),
                                           ZPSE_T16_ROW128(in[5] + in_off));
    const __m256i a6 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[14] + in_off),
                                           ZPSE_T16_ROW128(in[6] + in_off));
    const __m256i a7 = _mm256_loadu2_m128i(ZPSE_T16_ROW128(in[15] + in_off),
                                           ZPSE_T16_ROW128(in[7] + in_off));
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    __m256i u0, u1, u2, u3, u4, u5, u6, u7;
    __m256i v0, v1, v2, v3, v4, v5, v6, v7;
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;

    t0 = _mm256_unpacklo_epi8(a0, a1);
    t1 = _mm256_unpackhi_epi8(a0, a1);
    t2 = _mm256_unpacklo_epi8(a2, a3);
    t3 = _mm256_unpackhi_epi8(a2, a3);
    t4 = _mm256_unpacklo_epi8(a4, a5);
    t5 = _mm256_unpackhi_epi8(a4, a5);
    t6 = _mm256_unpacklo_epi8(a6, a7);
    t7 = _mm256_unpackhi_epi8(a6, a7);
    u0 = _mm256_unpacklo_epi16(t0, t2);
    u1 = _mm256_unpackhi_epi16(t0, t2);
    u2 = _mm256_unpacklo_epi16(t1, t3);
    u3 = _mm256_unpackhi_epi16(t1, t3);
    u4 = _mm256_unpacklo_epi16(t4, t6);
    u5 = _mm256_unpackhi_epi16(t4, t6);
    u6 = _mm256_unpacklo_epi16(t5, t7);
    u7 = _mm256_unpackhi_epi16(t5, t7);
    v0 = _mm256_unpacklo_epi32(u0, u4);
    v1 = _mm256_unpackhi_epi32(u0, u4);
    v2 = _mm256_unpacklo_epi32(u1, u5);
    v3 = _mm256_unpackhi_epi32(u1, u5);
    v4 = _mm256_unpacklo_epi32(u2, u6);
    v5 = _mm256_unpackhi_epi32(u2, u6);
    v6 = _mm256_unpacklo_epi32(u3, u7);
    v7 = _mm256_unpackhi_epi32(u3, u7);
    w0 = _mm256_unpacklo_epi64(v0, v4);
    w1 = _mm256_unpackhi_epi64(v0, v4);
    w2 = _mm256_unpacklo_epi64(v1, v5);
    w3 = _mm256_unpackhi_epi64(v1, v5);
    w4 = _mm256_unpacklo_epi64(v2, v6);
    w5 = _mm256_unpackhi_epi64(v2, v6);
    w6 = _mm256_unpacklo_epi64(v3, v7);
    w7 = _mm256_unpackhi_epi64(v3, v7);

    /* w[k] carries column k in its two first 64-bit halves, one per row
     * group, so column k is their low-low join and column k+8 their
     * high-high join. */
#define ZPSE_T16_EMIT(WK, IDX)                                                \
    do {                                                                      \
        const __m128i lo_ = _mm256_castsi256_si128(WK);                       \
        const __m128i hi_ = _mm256_extracti128_si256(WK, 1);                  \
        _mm_storeu_si128((__m128i *)(void *)(out[IDX] + out_off),             \
                         _mm_unpacklo_epi64(lo_, hi_));                       \
        _mm_storeu_si128((__m128i *)(void *)(out[(IDX) + 8u] + out_off),      \
                         _mm_unpackhi_epi64(lo_, hi_));                       \
    } while (0)

    ZPSE_T16_EMIT(w0, 0u);
    ZPSE_T16_EMIT(w1, 1u);
    ZPSE_T16_EMIT(w2, 2u);
    ZPSE_T16_EMIT(w3, 3u);
    ZPSE_T16_EMIT(w4, 4u);
    ZPSE_T16_EMIT(w5, 5u);
    ZPSE_T16_EMIT(w6, 6u);
    ZPSE_T16_EMIT(w7, 7u);
#undef ZPSE_T16_EMIT
}

#undef ZPSE_T16_ROW128

/* Forward-transpose a whole plane: 16 rows of 16 bytes per group, `rows`
 * rows per segment, written out in blocks rather than one group at a time.
 *
 * The sixteen planes of a transposed block sit `rows` bytes apart, and for
 * a 64 KiB block that is exactly 4096 -- 64 cache lines, i.e. the same L1
 * set for every plane. One group at a time therefore leaves sixteen lines
 * of ONE set live while each of them receives 16 bytes, so every store
 * re-allocates a line the previous group just evicted: measured 0.49
 * cycles/byte, against 0.15 for the same kernel writing planes 1 KiB apart
 * (which is what a 16 KiB segment does, so the block sweep's proxy never
 * saw the problem and only a winning trial paid for it -- 8 us of a 64 KiB
 * transpose, a third of `alt00FF`'s whole encode).
 *
 * Transposing ZPSE_T16_GROUPS groups into a buffer first -- one plane per
 * contiguous run, stride 16 * GROUPS, so the buffer itself has no conflicts
 * -- and then copying whole runs into the planes turns four partial-line
 * stores per line into one full-line store: 0.20 cycles/byte, the same
 * number a padded layout gives. 32 groups (an 8 KiB buffer, 512 bytes per
 * plane) measured best; 16 leaves 0.22 and 64 overruns L1 at 0.31. Eight
 * planes per copy pass, so no more than the L1 associativity of them share
 * a set. The buffer lives here rather than in zpse_transform_forward so
 * that the identity, delta and xor paths never pay for its frame.
 *
 * Same bytes, written in a different order. */
static ZPSE_T_AVX2 void zpse_t16x16_forward_blocked(const uint8_t *const in[16],
                                                   uint8_t *const out[16],
                                                   uint32_t rows)
{
    uint8_t col[16u * 16u * ZPSE_T16_GROUPS];
    uint8_t *cseg[16];
    uint32_t q;
    uint32_t g;
    unsigned t;

    for (t = 0; t < 16u; t++) {
        cseg[t] = col + (size_t)t * 16u * (size_t)ZPSE_T16_GROUPS;
    }
    for (g = 0; (g + ZPSE_T16_GROUPS) * 16u <= rows; g += ZPSE_T16_GROUPS) {
        for (q = 0; q < ZPSE_T16_GROUPS; q++) {
            zpse_t16x16_avx2(in, (g + q) * 256u, cseg, q * 16u);
        }
        for (t = 0; t < 16u; t += 8u) {
            unsigned u;

            for (u = 0; u < 8u; u++) {
                memcpy(out[t + u] + (size_t)g * 16u, cseg[t + u],
                       (size_t)16u * (size_t)ZPSE_T16_GROUPS);
            }
        }
    }
    for (; (g + 1u) * 16u <= rows; g++) {
        zpse_t16x16_avx2(in, g * 256u, out, g * 16u);
    }
}

/* Direct gather: the same network reading the sixteen planes in place.
 *
 * Staging exists only to break the L1 set conflict of planes exactly
 * `rows` apart -- for a 64 KiB block that is 4096 bytes, and the L1 set
 * index is (address / 64) mod 64, so all sixteen lines of a group land
 * on one set. A caller that lays the planes out with a skew has already
 * broken that conflict, and for it the copy is pure cost. Measured with
 * the same kernel and the same output buffer, in one process,
 * alternating, 64 KiB per run:
 *
 *   staged, 256 B per plane        2730-2900 ns
 *   direct, planes 4096 B apart    4670-4830 ns  (0.58x -- the conflict)
 *   direct, planes 4160 B apart    2000-2600 ns  (1.28-1.46x)
 *
 * 4160 rather than a fractional skew because the planes then stay on
 * cache-line boundaries, which the segment decoder's stores care about:
 * a 16-byte skew measured faster in this harness and ~7% *slower* end to
 * end, every plane's writes splitting a line. Block decode lays the
 * planes of a 16-wide interleave out that way (ZPSE_D_PLANE_SKEW), so it
 * reaches this path; everything else keeps the staging, and both produce
 * identical bytes -- same kernel, only the source pointers differ. */
static ZPSE_T_AVX2 void zpse_t16x16_inverse_direct(const uint8_t *const in[16],
                                                   uint8_t *const out[16],
                                                   uint32_t groups)
{
    uint32_t g;

    for (g = 0; g < groups; g++) {
        zpse_t16x16_avx2(in, g * 16u, out, g * 256u);
    }
}

/* Inverse of the above, blocked the other way round: one 64-byte line from
 * each of the sixteen planes into a 1 KiB buffer, then four groups out of
 * the buffer.
 *
 * A group of the inverse reads sixteen bytes from each source plane, and
 * the planes of a 64 KiB block sit exactly 4096 bytes apart -- so all
 * sixteen lines of a group have the SAME L1 set index (the same problem
 * the forward's 4 KiB-strided stores had, now on the load side). Measured
 * 11 GB/s for the direct form against 30 GB/s for the same kernel with
 * L1-resident sources: conflict misses, not arithmetic. Staging writes
 * plane t at t*64, so its sixteen-byte chunks map to set t and the
 * transpose itself reads from sixteen different sets: 16 GB/s, and the
 * copy is one full line per plane instead of four partial ones. */
static ZPSE_T_AVX2 void zpse_t16x16_inverse_blocked(const uint8_t *const in[16],
                                                     uint8_t *const out[16],
                                                     uint32_t rows)
{
    /* Aligned: the staging stores are aligned _mm256 stores, and every
     * plane's run inside it starts on a 64-byte boundary -- which is why
     * ZPSE_T16_ISTAGE must be even (a run of 16 * G bytes, aligned when
     * that is a multiple of 32). */
    uint8_t buf[16u * 16u * ZPSE_T16_ISTAGE] __attribute__((aligned(64)));
    const uint8_t *bin[16];
    uint32_t g = 0u;
    uint32_t q;
    unsigned t;

    for (t = 0; t < 16u; t++) {
        bin[t] = buf + (size_t)t * 16u * (size_t)ZPSE_T16_ISTAGE;
    }
    for (; (g + ZPSE_T16_ISTAGE) * 16u <= rows; g += ZPSE_T16_ISTAGE) {
        for (t = 0; t < 16u; t++) {
            const uint8_t *src = in[t] + (size_t)g * 16u;
            uint8_t *dst = buf + (size_t)t * 16u * (size_t)ZPSE_T16_ISTAGE;
            uint32_t j;

            for (j = 0; j < (uint32_t)(16u * ZPSE_T16_ISTAGE); j += 32u) {
                _mm256_store_si256((__m256i *)(void *)(dst + j),
                                   _mm256_loadu_si256(
                                       (const __m256i *)(const void *)(src + j)));
            }
        }
        for (q = 0; q < (uint32_t)ZPSE_T16_ISTAGE; q++) {
            zpse_t16x16_avx2(bin, q * 16u, out, (g + q) * 256u);
        }
    }
    for (; (g + 1u) * 16u <= rows; g++) {
        zpse_t16x16_avx2(in, g * 16u, out, g * 256u);
    }
}

/* Width-1 delta/xor staging, 32 bytes per step. No dependency chain to
 * satisfy: for j >= 1 the stored byte is an op between the stream and
 * itself shifted by one, so the two loads are independent of the store.
 * Only dst[0] touches the seed. 16 bytes per step measured 2 us of a
 * 12 us `incr` encode, where this pass runs once per DELTA8 trial. */
static ZPSE_T_AVX2 void zpse_prefix_stream_avx2(uint8_t *dst, const uint8_t *in,
                                                uint32_t n, uint8_t seed,
                                                int delta)
{
    uint32_t q = 1u;

    dst[0] = delta ? (uint8_t)(in[0] - seed) : (uint8_t)(in[0] ^ seed);
    for (; (size_t)q + 32u <= (size_t)n; q += 32u) {
        __m256i cur = _mm256_loadu_si256(
            (const __m256i *)(const void *)(in + q));
        __m256i prv = _mm256_loadu_si256(
            (const __m256i *)(const void *)(in + q - 1u));

        _mm256_storeu_si256((__m256i *)(void *)(dst + q),
                            delta ? _mm256_sub_epi8(cur, prv)
                                  : _mm256_xor_si256(cur, prv));
    }
    for (; q < n; q++) {
        dst[q] = delta ? (uint8_t)(in[q] - in[q - 1u])
                       : (uint8_t)(in[q] ^ in[q - 1u]);
    }
}

#endif /* ZPSE_T_HAVE_AVX2 */

/* In-place delta/xor over one contiguous plane: p[j] = p[j] - p[j-1]
 * (delta) or p[j] ^ p[j-1] (xor), with p[-1] == seed. The stride-16
 * gather the transposed delta used to do per byte alongside the chain is
 * what made it slow; the plane is contiguous by the time this runs, so a
 * 16-byte step only needs the previous chunk's last byte as the incoming
 * "prv[0]". That byte is read before the store overwrites it. */
static void zpse_plane_chain_inplace(uint8_t *p, uint32_t n, uint8_t seed,
                                     int delta)
{
    uint8_t prev = seed;
    uint32_t q;

#if ZPSE_T_HAVE_SSE2
    /* Four chunks per iteration. Each chunk's incoming byte is the last
     * byte of the PREVIOUS chunk's original content, which is a byte of a
     * load that has already happened, so all four shifts come out of
     * registers and only the seed for the next group of four crosses the
     * loop -- one psrldq per 64 bytes. The 16-byte form that still takes
     * the remainder reads that seed back out of the plane, at a position
     * the store of the same iteration has just written, so the store queue
     * can only forward it after the store retires; the wide form above
     * reads nothing it has just written. The residual 16-byte loop and the
     * byte tail are unchanged, and every position gets the same byte as
     * before. */
    for (q = 0; (size_t)q + 64u <= (size_t)n; q += 64u) {
        __m128i a = _mm_loadu_si128((const __m128i *)(const void *)(p + q));
        __m128i b =
            _mm_loadu_si128((const __m128i *)(const void *)(p + q + 16u));
        __m128i c =
            _mm_loadu_si128((const __m128i *)(const void *)(p + q + 32u));
        __m128i d =
            _mm_loadu_si128((const __m128i *)(const void *)(p + q + 48u));
        __m128i pa = _mm_or_si128(_mm_slli_si128(a, 1),
                                  _mm_cvtsi32_si128((int)prev));
        __m128i pb =
            _mm_or_si128(_mm_slli_si128(b, 1), _mm_srli_si128(a, 15));
        __m128i pc =
            _mm_or_si128(_mm_slli_si128(c, 1), _mm_srli_si128(b, 15));
        __m128i pd =
            _mm_or_si128(_mm_slli_si128(d, 1), _mm_srli_si128(c, 15));

        prev = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(d, 15));
        if (delta) {
            _mm_storeu_si128((__m128i *)(void *)(p + q), _mm_sub_epi8(a, pa));
            _mm_storeu_si128((__m128i *)(void *)(p + q + 16u),
                             _mm_sub_epi8(b, pb));
            _mm_storeu_si128((__m128i *)(void *)(p + q + 32u),
                             _mm_sub_epi8(c, pc));
            _mm_storeu_si128((__m128i *)(void *)(p + q + 48u),
                             _mm_sub_epi8(d, pd));
        } else {
            _mm_storeu_si128((__m128i *)(void *)(p + q),
                             _mm_xor_si128(a, pa));
            _mm_storeu_si128((__m128i *)(void *)(p + q + 16u),
                             _mm_xor_si128(b, pb));
            _mm_storeu_si128((__m128i *)(void *)(p + q + 32u),
                             _mm_xor_si128(c, pc));
            _mm_storeu_si128((__m128i *)(void *)(p + q + 48u),
                             _mm_xor_si128(d, pd));
        }
    }
    for (; (size_t)q + 16u <= (size_t)n; q += 16u) {
        __m128i cur = _mm_loadu_si128((const __m128i *)(const void *)(p + q));
        __m128i prv = _mm_or_si128(_mm_slli_si128(cur, 1),
                                   _mm_cvtsi32_si128((int)prev));
        uint8_t carry = p[q + 15u];

        _mm_storeu_si128((__m128i *)(void *)(p + q),
                         delta ? _mm_sub_epi8(cur, prv)
                               : _mm_xor_si128(cur, prv));
        prev = carry;
    }
#else
    q = 0;
#endif /* ZPSE_T_HAVE_SSE2 */
    for (; q < n; q++) {
        uint8_t raw = p[q];

        p[q] = delta ? (uint8_t)(raw - prev) : (uint8_t)(raw ^ prev);
        prev = raw;
    }
}

#if ZPSE_T_HAVE_SSE2
/* The evens of two 16-byte vectors, back to back, and the odds likewise.
 * The words of a vector ARE its byte pairs, so masking the low bytes and
 * shifting out the high ones is exactly a byte-parity split; packus then
 * compacts both halves into one 16-byte result. Every split below is a
 * nest of these, and every merge its inverse (an unpack). */
static __m128i zpse_parity_evens(__m128i a, __m128i b)
{
    const __m128i mask = _mm_set1_epi16(0x00FF);

    return _mm_packus_epi16(_mm_and_si128(a, mask), _mm_and_si128(b, mask));
}

static __m128i zpse_parity_odds(__m128i a, __m128i b)
{
    return _mm_packus_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
}

/* Undo one plane's sixteen delta (or xor) bytes in registers: an inclusive
 * byte prefix sum is log2(16) shift-adds, the xor chain the same with xors,
 * and the byte that crosses the chunk is the chunk's own last byte --
 * exactly the value the byte-at-a-time chain carries, so the result is
 * byte-identical to it. `carry` is threaded through the chunk. */
ZPSE_HOT static __m128i zpse_chain_undo(__m128i x, uint8_t *carry, int delta)
{
    uint8_t add;

    if (delta) {
        x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
        add = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(x, 15));
        x = _mm_add_epi8(x, _mm_set1_epi8((char)*carry));
        *carry = (uint8_t)(*carry + add);
    } else {
        x = _mm_xor_si128(x, _mm_slli_si128(x, 1));
        x = _mm_xor_si128(x, _mm_slli_si128(x, 2));
        x = _mm_xor_si128(x, _mm_slli_si128(x, 4));
        x = _mm_xor_si128(x, _mm_slli_si128(x, 8));
        add = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(x, 15));
        x = _mm_xor_si128(x, _mm_set1_epi8((char)*carry));
        *carry ^= add;
    }
    return x;
}

/* The narrow widths are one byte permutation: byte i of the records belongs
 * to plane (i mod w), so the split is log2(w) levels of parity splits and
 * the merge log2(w) levels of byte interleaves. A group is sixteen rows --
 * 16*w input bytes in, 16 bytes per plane out -- which keeps the whole
 * working set in registers: nothing here allocates, so these paths stay
 * usable on the decode side, which has no scratch of its own.
 *
 * The scalar loops they replace moved one strided byte per load and one per
 * store: 15.2-15.6 us forward and 18.1-18.6 us inverse per 64 KiB, against
 * 2.9/3.0 for the 16-wide transpose. */

static void zpse_split2(uint8_t *const segs[16], const uint8_t *in,
                        uint32_t rows)
{
    uint32_t g;

    for (g = 0; (size_t)g + 16u <= (size_t)rows; g += 16u) {
        const uint8_t *q = in + (size_t)g * 2u;
        __m128i a = _mm_loadu_si128((const __m128i *)(const void *)q);
        __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(q + 16));

        _mm_storeu_si128((__m128i *)(void *)(segs[0] + g),
                         zpse_parity_evens(a, b));
        _mm_storeu_si128((__m128i *)(void *)(segs[1] + g),
                         zpse_parity_odds(a, b));
    }
}

static void zpse_split4(uint8_t *const segs[16], const uint8_t *in,
                        uint32_t rows)
{
    uint32_t g;

    for (g = 0; (size_t)g + 16u <= (size_t)rows; g += 16u) {
        const uint8_t *q = in + (size_t)g * 4u;
        __m128i a = _mm_loadu_si128((const __m128i *)(const void *)q);
        __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(q + 16));
        __m128i c = _mm_loadu_si128((const __m128i *)(const void *)(q + 32));
        __m128i d = _mm_loadu_si128((const __m128i *)(const void *)(q + 48));
        /* Level 1: bytes of class 0|2 (evens) and 1|3 (odds), per half. */
        __m128i e1 = zpse_parity_evens(a, b);
        __m128i o1 = zpse_parity_odds(a, b);
        __m128i e2 = zpse_parity_evens(c, d);
        __m128i o2 = zpse_parity_odds(c, d);

        /* Level 2: even and odd positions of each half of those streams. */
        _mm_storeu_si128((__m128i *)(void *)(segs[0] + g),
                         zpse_parity_evens(e1, e2));
        _mm_storeu_si128((__m128i *)(void *)(segs[1] + g),
                         zpse_parity_evens(o1, o2));
        _mm_storeu_si128((__m128i *)(void *)(segs[2] + g),
                         zpse_parity_odds(e1, e2));
        _mm_storeu_si128((__m128i *)(void *)(segs[3] + g),
                         zpse_parity_odds(o1, o2));
    }
}

static void zpse_split8(uint8_t *const segs[16], const uint8_t *in,
                        uint32_t rows)
{
    uint32_t g;

    for (g = 0; (size_t)g + 16u <= (size_t)rows; g += 16u) {
        const uint8_t *q = in + (size_t)g * 8u;
        __m128i a0 = _mm_loadu_si128((const __m128i *)(const void *)q);
        __m128i a1 = _mm_loadu_si128((const __m128i *)(const void *)(q + 16));
        __m128i a2 = _mm_loadu_si128((const __m128i *)(const void *)(q + 32));
        __m128i a3 = _mm_loadu_si128((const __m128i *)(const void *)(q + 48));
        __m128i a4 = _mm_loadu_si128((const __m128i *)(const void *)(q + 64));
        __m128i a5 = _mm_loadu_si128((const __m128i *)(const void *)(q + 80));
        __m128i a6 = _mm_loadu_si128((const __m128i *)(const void *)(q + 96));
        __m128i a7 = _mm_loadu_si128((const __m128i *)(const void *)(q + 112));
        /* Level 1: the parity split of each 32-byte span. */
        __m128i ev01 = zpse_parity_evens(a0, a1);
        __m128i od01 = zpse_parity_odds(a0, a1);
        __m128i ev23 = zpse_parity_evens(a2, a3);
        __m128i od23 = zpse_parity_odds(a2, a3);
        __m128i ev45 = zpse_parity_evens(a4, a5);
        __m128i od45 = zpse_parity_odds(a4, a5);
        __m128i ev67 = zpse_parity_evens(a6, a7);
        __m128i od67 = zpse_parity_odds(a6, a7);
        /* Level 2: the evens and odds of the two spans of each parity.
         * Slots 0-3 hold the even residues of the group's first and second
         * half in turn, 4-7 the odd ones. */
        __m128i p0 = zpse_parity_evens(ev01, ev23);
        __m128i p4 = zpse_parity_odds(ev01, ev23);
        __m128i p1 = zpse_parity_evens(ev45, ev67);
        __m128i p5 = zpse_parity_odds(ev45, ev67);
        __m128i p2 = zpse_parity_evens(od01, od23);
        __m128i p6 = zpse_parity_odds(od01, od23);
        __m128i p3 = zpse_parity_evens(od45, od67);
        __m128i p7 = zpse_parity_odds(od45, od67);

        /* Level 3: one residue per plane. */
        _mm_storeu_si128((__m128i *)(void *)(segs[0] + g),
                         zpse_parity_evens(p0, p1));
        _mm_storeu_si128((__m128i *)(void *)(segs[1] + g),
                         zpse_parity_evens(p2, p3));
        _mm_storeu_si128((__m128i *)(void *)(segs[2] + g),
                         zpse_parity_evens(p4, p5));
        _mm_storeu_si128((__m128i *)(void *)(segs[3] + g),
                         zpse_parity_evens(p6, p7));
        _mm_storeu_si128((__m128i *)(void *)(segs[4] + g),
                         zpse_parity_odds(p0, p1));
        _mm_storeu_si128((__m128i *)(void *)(segs[5] + g),
                         zpse_parity_odds(p2, p3));
        _mm_storeu_si128((__m128i *)(void *)(segs[6] + g),
                         zpse_parity_odds(p4, p5));
        _mm_storeu_si128((__m128i *)(void *)(segs[7] + g),
                         zpse_parity_odds(p6, p7));
    }
}

static void zpse_merge2(uint8_t *out, const uint8_t *const segs[16],
                        uint32_t rows, uint8_t *carry, int delta)
{
    uint32_t g;

    for (g = 0; (size_t)g + 16u <= (size_t)rows; g += 16u) {
        __m128i a = _mm_loadu_si128(
            (const __m128i *)(const void *)(segs[0] + g));
        __m128i b = _mm_loadu_si128(
            (const __m128i *)(const void *)(segs[1] + g));
        uint8_t *d = out + (size_t)g * 2u;

        if (carry != 0) {
            a = zpse_chain_undo(a, carry, delta);
            b = zpse_chain_undo(b, carry + 1, delta);
        }
        _mm_storeu_si128((__m128i *)(void *)d, _mm_unpacklo_epi8(a, b));
        _mm_storeu_si128((__m128i *)(void *)(d + 16),
                         _mm_unpackhi_epi8(a, b));
    }
}

static void zpse_merge4(uint8_t *out, const uint8_t *const segs[16],
                        uint32_t rows, uint8_t *carry, int delta)
{
    uint32_t g;

    for (g = 0; (size_t)g + 16u <= (size_t)rows; g += 16u) {
        __m128i p[4];
        __m128i lo[2];
        __m128i hi[2];
        uint8_t *d = out + (size_t)g * 4u;
        unsigned t;

        for (t = 0; t < 4u; t++) {
            __m128i x = _mm_loadu_si128(
                (const __m128i *)(const void *)(segs[t] + g));

            p[t] = carry != 0 ? zpse_chain_undo(x, carry + t, delta) : x;
        }
        /* Classes 0|1 and 2|3 interleave pairwise, then the two halves. */
        lo[0] = _mm_unpacklo_epi8(p[0], p[1]);
        hi[0] = _mm_unpackhi_epi8(p[0], p[1]);
        lo[1] = _mm_unpacklo_epi8(p[2], p[3]);
        hi[1] = _mm_unpackhi_epi8(p[2], p[3]);
        _mm_storeu_si128((__m128i *)(void *)d,
                         _mm_unpacklo_epi16(lo[0], lo[1]));
        _mm_storeu_si128((__m128i *)(void *)(d + 16),
                         _mm_unpackhi_epi16(lo[0], lo[1]));
        _mm_storeu_si128((__m128i *)(void *)(d + 32),
                         _mm_unpacklo_epi16(hi[0], hi[1]));
        _mm_storeu_si128((__m128i *)(void *)(d + 48),
                         _mm_unpackhi_epi16(hi[0], hi[1]));
    }
}

static void zpse_merge8(uint8_t *out, const uint8_t *const segs[16],
                        uint32_t rows, uint8_t *carry, int delta)
{
    uint32_t g;

    for (g = 0; (size_t)g + 16u <= (size_t)rows; g += 16u) {
        __m128i c[8];
        uint8_t *d = out + (size_t)g * 8u;
        unsigned t;

        for (t = 0; t < 8u; t++) {
            __m128i x = _mm_loadu_si128(
                (const __m128i *)(const void *)(segs[t] + g));

            c[t] = carry != 0 ? zpse_chain_undo(x, carry + t, delta) : x;
        }
        /* Level 3, undone: class t and class t+4 were the evens and odds of
         * the same pair of streams. */
        __m128i p0 = _mm_unpacklo_epi8(c[0], c[4]);
        __m128i p1 = _mm_unpackhi_epi8(c[0], c[4]);
        __m128i p2 = _mm_unpacklo_epi8(c[1], c[5]);
        __m128i p3 = _mm_unpackhi_epi8(c[1], c[5]);
        __m128i p4 = _mm_unpacklo_epi8(c[2], c[6]);
        __m128i p5 = _mm_unpackhi_epi8(c[2], c[6]);
        __m128i p6 = _mm_unpacklo_epi8(c[3], c[7]);
        __m128i p7 = _mm_unpackhi_epi8(c[3], c[7]);
        /* Level 2, undone: the same pairing one level down. */
        __m128i ev01 = _mm_unpacklo_epi8(p0, p4);
        __m128i ev23 = _mm_unpackhi_epi8(p0, p4);
        __m128i ev45 = _mm_unpacklo_epi8(p1, p5);
        __m128i ev67 = _mm_unpackhi_epi8(p1, p5);
        __m128i od01 = _mm_unpacklo_epi8(p2, p6);
        __m128i od23 = _mm_unpackhi_epi8(p2, p6);
        __m128i od45 = _mm_unpacklo_epi8(p3, p7);
        __m128i od67 = _mm_unpackhi_epi8(p3, p7);
        /* Level 1, undone: the parity-2 spans rebuild the eight 16-byte
         * pieces of the group, in input order. */
        _mm_storeu_si128((__m128i *)(void *)d,
                         _mm_unpacklo_epi8(ev01, od01));
        _mm_storeu_si128((__m128i *)(void *)(d + 16),
                         _mm_unpackhi_epi8(ev01, od01));
        _mm_storeu_si128((__m128i *)(void *)(d + 32),
                         _mm_unpacklo_epi8(ev23, od23));
        _mm_storeu_si128((__m128i *)(void *)(d + 48),
                         _mm_unpackhi_epi8(ev23, od23));
        _mm_storeu_si128((__m128i *)(void *)(d + 64),
                         _mm_unpacklo_epi8(ev45, od45));
        _mm_storeu_si128((__m128i *)(void *)(d + 80),
                         _mm_unpackhi_epi8(ev45, od45));
        _mm_storeu_si128((__m128i *)(void *)(d + 96),
                         _mm_unpacklo_epi8(ev67, od67));
        _mm_storeu_si128((__m128i *)(void *)(d + 112),
                         _mm_unpackhi_epi8(ev67, od67));
    }
}

/* `rows` interleaved rows in, w contiguous planes out. The caller handles
 * the ragged tail (plane lengths differ by at most one row) and the
 * delta/xor chains, which run on the contiguous planes. */
static void zpse_split_narrow(uint8_t *const segs[16], const uint8_t *in,
                              uint32_t rows, uint8_t w)
{
    if (w == 2u) {
        zpse_split2(segs, in, rows);
    } else if (w == 4u) {
        zpse_split4(segs, in, rows);
    } else {
        zpse_split8(segs, in, rows);
    }
}

/* The inverse: w contiguous planes in, the interleaved rows out. When
 * `carry` is not NULL each plane's sixteen bytes are first restored from the
 * chain in registers, and `carry[t]` holds the byte that crosses the group. */
static void zpse_merge_narrow(uint8_t *out, const uint8_t *const segs[16],
                              uint32_t rows, uint8_t w, uint8_t *carry,
                              int delta)
{
    if (w == 2u) {
        zpse_merge2(out, segs, rows, carry, delta);
    } else if (w == 4u) {
        zpse_merge4(out, segs, rows, carry, delta);
    } else {
        zpse_merge8(out, segs, rows, carry, delta);
    }
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

ZPSE_HOT int zpse_transform_forward(const ZpseTransformMeta *meta, uint8_t seg_count,
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
#if ZPSE_T_HAVE_AVX2
        if (zpse_t_avx2()) {
            zpse_prefix_stream_avx2(dst, in, n, meta->seeds[0], delta);
            return PSE_OK;
        }
#endif /* ZPSE_T_HAVE_AVX2 */
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
    if (w == 16u) {
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
#if ZPSE_T_HAVE_AVX2
        if (zpse_t_avx2()) {
            /* Bases once: row t of a group is 16 bytes into the next
             * contiguous 256-byte input block, and its destination is 16
             * rows into segment t. */
            const uint8_t *tin[16];
            uint8_t *tout[16];
            unsigned t;

            for (t = 0; t < 16u; t++) {
                tin[t] = in + (size_t)t * 16u;
                tout[t] = segs[t];
            }
            zpse_t16x16_forward_blocked(tin, tout, rows);
            g = rows / 16u;
        } else
#endif /* ZPSE_T_HAVE_AVX2 */
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
        /* The delta/xor chain runs on the contiguous plane the transpose
         * just produced, not on the stride-16 gather: same bytes (each
         * plane is one byte position across consecutive records, chain and
         * seed included), but 16 bytes per step instead of one strided load
         * per byte. */
        if (delta || xorr) {
            for (s = 0; s < w; s++) {
                zpse_plane_chain_inplace(segs[s], seg_lens[s],
                                         meta->seeds[s], delta);
            }
        }
        return PSE_OK;
    }
    /* The narrow widths (2, 4 and 8) are one byte permutation, and a
     * permutation does not care whether a chain follows it. The parity
     * splits above produce the contiguous planes; the delta/xor chains then
     * run on them exactly as the 16-wide path runs them, sixteen bytes per
     * step instead of one strided load per byte. */
    if (w == 2u || w == 4u || w == 8u) {
        uint32_t rows = seg_lens[0];
        uint32_t full, r2;

        for (s = 1; s < w; s++) {
            if (seg_lens[s] < rows) {
                rows = seg_lens[s];
            }
        }
        full = (rows / 16u) * 16u;
        zpse_split_narrow(segs, in, full, w);
        /* Ragged rows: the planes can differ in length by one. */
        for (s = 0; s < w; s++) {
            uint32_t len = seg_lens[s];

            for (r2 = full; r2 < len; r2++) {
                segs[s][r2] = in[(uint32_t)s + r2 * (uint32_t)w];
            }
        }
        if (delta || xorr) {
            for (s = 0; s < w; s++) {
                zpse_plane_chain_inplace(segs[s], seg_lens[s], meta->seeds[s],
                                         delta);
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

ZPSE_HOT int zpse_transform_inverse(const ZpseTransformMeta *meta, uint8_t seg_count,
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
#if ZPSE_T_HAVE_AVX2
            if (zpse_t_avx2()) {
                /* The inverse gather, with the same hoisted bases: a group
                 * is 16 bytes into each segment and 256 into the output. */
                const uint8_t *tin[16];
                uint8_t *tout[16];
                unsigned t;

                uintptr_t stride;
                int skewed;

                for (t = 0; t < 16u; t++) {
                    tin[t] = segs[t];
                    tout[t] = out + (size_t)t * 16u;
                }
                /* The L1 set index is (address / 64) mod 64, so segment
                 * bases a multiple of 4096 apart put every group's
                 * sixteen lines on one set -- that is the case staging
                 * exists for. Any other stride has the conflict broken
                 * already, and then reading the planes in place beats
                 * copying them (see zpse_t16x16_inverse_direct); block
                 * decode lays out a 16-wide interleave that way. Both
                 * paths produce identical bytes. */
                stride = (uintptr_t)segs[1] - (uintptr_t)segs[0];
                skewed = ((stride & (uintptr_t)4095u) != 0u) ? 1 : 0;
                if (skewed != 0 && rows >= 16u) {
                    zpse_t16x16_inverse_direct(tin, tout, rows / 16u);
                } else {
                    zpse_t16x16_inverse_blocked(tin, tout, rows);
                }
                g = rows / 16u; /* the tail loop below is already covered */
            } else
#endif /* ZPSE_T_HAVE_AVX2 */
            for (g = 0; (g + 1u) * 16u <= rows; g++) {
                __m128i r[16];
                unsigned t;

                for (t = 0; t < 16u; t++) {
                    r[t] = _mm_loadu_si128(
                        (const __m128i *)(const void *)
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
        /* The narrow widths have no chain to restore here: the merge IS the
         * transform, so the same helper covers them. */
        if (w == 2u || w == 4u || w == 8u) {
            uint32_t rows = seg_lens[0];
            uint32_t full, r2;

            for (s = 1; s < w; s++) {
                if (seg_lens[s] < rows) {
                    rows = seg_lens[s];
                }
            }
            full = (rows / 16u) * 16u;
            zpse_merge_narrow(out, segs, full, w, 0, 0);
            /* Ragged rows: the planes can differ in length by one. */
            for (s = 0; s < w; s++) {
                uint32_t len = seg_lens[s];

                for (r2 = full; r2 < len; r2++) {
                    out[(uint32_t)s + r2 * (uint32_t)w] = segs[s][r2];
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

#if ZPSE_T_HAVE_SSE2
        /* out[j] = seed + sum(enc[0..j]) for delta, the xor prefix for
         * xor, from the seed exactly like the scalar loop below: the
         * prefix scans defined above, 256 bytes per carry link when AVX2
         * is available and 128 bytes otherwise. */
        if (delta) {
#if ZPSE_T_HAVE_AVX2
            if (zpse_t_avx2()) {
                zpse_prefix_sum32(src, out, n, pv);
                return PSE_OK;
            }
#endif
            zpse_prefix_sum16(src, out, n, pv);
        } else {
#if ZPSE_T_HAVE_AVX2
            if (zpse_t_avx2()) {
                zpse_prefix_xor32(src, out, n, pv);
                return PSE_OK;
            }
#endif
            zpse_prefix_xor16(src, out, n, pv);
        }
        return PSE_OK;
#else
        {
            uint32_t j;

            for (j = 0; j < n; j++) {
                uint8_t e = src[j];
                uint8_t d = delta ? (uint8_t)(e + pv) : (uint8_t)(e ^ pv);
                pv = d;
                out[j] = d;
            }
        }
        return PSE_OK;
#endif /* ZPSE_T_HAVE_SSE2 */
    }
#if ZPSE_T_HAVE_SSE2
    /* The narrow widths with a chain: the same permutation, with each
     * plane's sixteen bytes restored from the chain in registers on the way
     * in; see zpse_merge_narrow for why the carry is the chain's own last
     * byte. */
    if (w == 2u || w == 4u || w == 8u) {
        uint32_t rows = seg_lens[0];
        uint32_t full, r2;
        uint8_t carry[16];

        for (s = 1; s < w; s++) {
            if (seg_lens[s] < rows) {
                rows = seg_lens[s];
            }
        }
        for (s = 0; s < w; s++) {
            carry[s] = prev[s];
        }
        full = (rows / 16u) * 16u;
        zpse_merge_narrow(out, segs, full, w, carry, delta);
        /* Ragged rows, continued from each plane's carry. */
        for (s = 0; s < w; s++) {
            uint32_t len = seg_lens[s];
            uint8_t p = carry[s];

            for (r2 = full; r2 < len; r2++) {
                uint8_t e = segs[s][r2];
                uint8_t d = delta ? (uint8_t)(e + p) : (uint8_t)(e ^ p);

                p = d;
                out[(uint32_t)s + r2 * (uint32_t)w] = d;
            }
        }
        return PSE_OK;
    }

    /* The 16-wide delta/xor inverse: the chain is what the byte-at-a-time
     * loop below serialises, so it is done 16 rows at a time inside
     * registers -- an inclusive byte prefix sum (log2(16) shift-adds) or
     * prefix xor per plane -- and the 16 planes are then interleaved by the
     * same 16x16 transpose the non-delta inverse uses. The carry that crosses
     * a group is the chain's own last byte, exactly the value the scalar loop
     * carries, so the two produce identical output. */
    if (w == 16u) {
        uint32_t rows = seg_lens[0];
        uint32_t g;
        uint8_t pv[16];

        for (s = 1; s < w; s++) {
            if (seg_lens[s] < rows) {
                rows = seg_lens[s];
            }
        }
        for (s = 0; s < w; s++) {
            pv[s] = prev[s];
        }
        for (g = 0; (g + 1u) * 16u <= rows; g++) {
            __m128i r[16];
            unsigned t;

            for (t = 0; t < 16u; t++) {
                __m128i v = _mm_loadu_si128(
                    (const __m128i *)(const void *)(segs[t] + (size_t)g * 16u));
                uint8_t add;

                if (delta) {
                    v = _mm_add_epi8(v, _mm_slli_si128(v, 1));
                    v = _mm_add_epi8(v, _mm_slli_si128(v, 2));
                    v = _mm_add_epi8(v, _mm_slli_si128(v, 4));
                    v = _mm_add_epi8(v, _mm_slli_si128(v, 8));
                } else {
                    v = _mm_xor_si128(v, _mm_slli_si128(v, 1));
                    v = _mm_xor_si128(v, _mm_slli_si128(v, 2));
                    v = _mm_xor_si128(v, _mm_slli_si128(v, 4));
                    v = _mm_xor_si128(v, _mm_slli_si128(v, 8));
                }
                /* Byte 15 of the prefix is the whole chunk's total (or xor),
                 * which is what the next group's carry becomes. */
                add = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(v, 15));
                v = delta ? _mm_add_epi8(v, _mm_set1_epi8((char)pv[t]))
                          : _mm_xor_si128(v, _mm_set1_epi8((char)pv[t]));
                pv[t] = (uint8_t)(delta ? (uint8_t)(pv[t] + add)
                                        : (uint8_t)(pv[t] ^ add));
                r[t] = v;
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
            uint8_t p = pv[s];
            uint32_t jj;

            for (jj = g * 16u; jj < len; jj++) {
                uint8_t e = src[jj];
                uint8_t d = delta ? (uint8_t)(e + p) : (uint8_t)(e ^ p);
                p = d;
                out[(uint32_t)s + jj * 16u] = d;
            }
        }
        return PSE_OK;
    }
#endif /* ZPSE_T_HAVE_SSE2 */
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

    if (in == 0 || enc == 0 || n == 0) {
        return;
    }
    enc[0] = (uint8_t)(in[0] - seed);
#if ZPSE_T_HAVE_SSE2
    /* Only enc[0] touches the seed. For j >= 1 the stored value is
     * in[j] - in[j-1], a byte-wise op between the stream and itself
     * shifted by one byte, so there is no chain to serialise -- the same
     * idiom zpse_transform_forward already uses for this transform. */
    i = 1;
    for (; (size_t)i + 16u <= (size_t)n; i += 16u) {
        __m128i cur = _mm_loadu_si128(
            (const __m128i *)(const void *)(in + i));
        __m128i prv = _mm_loadu_si128(
            (const __m128i *)(const void *)(in + i - 1u));
        _mm_storeu_si128((__m128i *)(void *)(enc + i),
                         _mm_sub_epi8(cur, prv));
    }
#else
    i = 1;
#endif /* ZPSE_T_HAVE_SSE2 */
    for (; i < n; i++) {
        enc[i] = (uint8_t)(in[i] - in[i - 1u]);
    }
}

void zpse_delta8_decode(const uint8_t *enc, uint8_t *out, uint32_t n,
                        uint8_t seed)
{
    if (enc == 0 || out == 0 || n == 0) {
        return;
    }
#if ZPSE_T_HAVE_SSE2
    /* Same scan as the w == 1 inverse: out[0] = enc[0] + seed, then the
     * byte-wise prefix sum. */
#if ZPSE_T_HAVE_AVX2
    if (zpse_t_avx2()) {
        zpse_prefix_sum32(enc, out, n, seed);
        return;
    }
#endif
    zpse_prefix_sum16(enc, out, n, seed);
#else
    {
        uint32_t i;
        uint8_t prev;

        out[0] = (uint8_t)(enc[0] + seed);
        prev = out[0];
        for (i = 1; i < n; i++) {
            uint8_t e = enc[i];
            uint8_t d = (uint8_t)(e + prev);
            out[i] = d;
            prev = d;
        }
    }
#endif /* ZPSE_T_HAVE_SSE2 */
}

void zpse_xor8_encode(const uint8_t *in, uint8_t *enc, uint32_t n,
                      uint8_t seed)
{
    uint32_t i;

    if (in == 0 || enc == 0 || n == 0) {
        return;
    }
    enc[0] = (uint8_t)(in[0] ^ seed);
#if ZPSE_T_HAVE_SSE2
    /* As above, with xor: enc[j] = in[j] ^ in[j-1] for j >= 1. */
    i = 1;
    for (; (size_t)i + 16u <= (size_t)n; i += 16u) {
        __m128i cur = _mm_loadu_si128(
            (const __m128i *)(const void *)(in + i));
        __m128i prv = _mm_loadu_si128(
            (const __m128i *)(const void *)(in + i - 1u));
        _mm_storeu_si128((__m128i *)(void *)(enc + i),
                         _mm_xor_si128(cur, prv));
    }
#else
    i = 1;
#endif /* ZPSE_T_HAVE_SSE2 */
    for (; i < n; i++) {
        enc[i] = (uint8_t)(in[i] ^ in[i - 1u]);
    }
}

void zpse_xor8_decode(const uint8_t *enc, uint8_t *out, uint32_t n,
                      uint8_t seed)
{
    if (enc == 0 || out == 0 || n == 0) {
        return;
    }
#if ZPSE_T_HAVE_SSE2
    /* Same scan as the w == 1 inverse: out[0] = enc[0] ^ seed, then the
     * byte-wise prefix xor. */
#if ZPSE_T_HAVE_AVX2
    if (zpse_t_avx2()) {
        zpse_prefix_xor32(enc, out, n, seed);
        return;
    }
#endif
    zpse_prefix_xor16(enc, out, n, seed);
#else
    {
        uint32_t i;
        uint8_t prev;

        out[0] = (uint8_t)(enc[0] ^ seed);
        prev = out[0];
        for (i = 1; i < n; i++) {
            uint8_t e = enc[i];
            uint8_t d = (uint8_t)(e ^ prev);
            out[i] = d;
            prev = d;
        }
    }
#endif /* ZPSE_T_HAVE_SSE2 */
}
