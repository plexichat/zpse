#include "zpse/segment.h"

#include <string.h>

#include "zpse/bits.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"
#include "zpse/tile.h"
#include "zpse/types.h"
#include "zpse/vector.h"

int zpse_segment_decode(ZpseDecoder *d, ZpseLaneState *lanes,
                        const uint8_t *payload, uint32_t payload_len,
                        uint8_t *decoded_out, uint32_t decoded_cap,
                        uint32_t *decoded_len_out)
{
    ZpseSegmentHeader h;
    int rc;
    uint32_t expect_tc;
    uint32_t pos;
    uint32_t produced;
    uint32_t tile_count;
    uint32_t seg_size;
    uint8_t literal;
    uint16_t t;

    if (d == 0 || lanes == 0 || payload == 0 || decoded_out == 0 ||
        decoded_len_out == 0) {
        return PSE_BAD_ARG;
    }
    rc = zpse_segment_header_parse(&h, payload, payload_len);
    if (rc != PSE_OK) {
        return rc;
    }
    if (h.segment_decoded_size < 1u ||
        h.segment_decoded_size > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_SEGMENT_DIRECTORY;
    }
    expect_tc = (h.segment_decoded_size + 255u) / 256u;
    if (h.tile_count == 0u || (uint32_t)h.tile_count != expect_tc) {
        return PSE_BAD_TILE_SIZE;
    }
    if (h.segment_decoded_size > decoded_cap) {
        return PSE_OUTPUT_OVERFLOW;
    }

    /* First tile of the segment always re-seeds lane state,
     * regardless of that tile's reset bit (Sec. 6.2 / 7). */
    zpse_lane_reset(lanes, h.initial_literal);

    pos = 16u;
    produced = 0u;
    tile_count = (uint32_t)h.tile_count;
    seg_size = h.segment_decoded_size;
    literal = h.initial_literal;
    for (t = 0; t < h.tile_count; t++) {
        uint16_t nt_exp;
        uint16_t rsz;
        const uint8_t *rec;
        int tile_reset;
        int got;

        if ((uint32_t)t + 1u < tile_count) {
            nt_exp = 256u;
        } else {
            nt_exp = (uint16_t)(seg_size - produced);
        }
        if (pos + 2u > payload_len) {
            return PSE_BAD_TILE_SIZE;
        }
        rsz = zpse_load_u16le(payload + pos);
        if (rsz == 0u) {
            return PSE_BAD_TILE_SIZE;
        }
        if ((uint32_t)rsz > payload_len - pos - 2u) {
            return PSE_BAD_TILE_SIZE;
        }
        if ((uint32_t)nt_exp > seg_size - produced) {
            return PSE_BAD_TILE_SIZE;
        }
        if (produced + (uint32_t)nt_exp > decoded_cap) {
            return PSE_OUTPUT_OVERFLOW;
        }
        rec = payload + pos + 2u;

        /* ---- Batched all-hit run -------------------------------- *
         * The record below is the entire tile header with every mode bit
         * clear except the reset flag: mask mode ALL_HIT, residual mode
         * EXCLUSION_U8, a zero exception count, both payload sizes zero,
         * and one of the CONST / lane-history predictors. Such a tile
         * decodes to a single repeated byte -- 0 for CONST_ZERO, the
         * initial literal otherwise -- and leaves the lane state at that
         * byte's fixed point, which is the state one tile leaves. A
         * maximal run of them is therefore one memset over the run's
         * contiguous output plus one history update, instead of one
         * zpse_tile_decode call and one tile_header_validate per tile.
         *
         * ONLY this exact byte pattern is taken here. Anything else -- in
         * particular every malformed record -- falls through to the
         * general path below, which re-derives and re-checks everything,
         * so not one rejection changes. The scan makes the same four
         * per-tile checks the loop above makes and simply STOPS at the
         * first one that fails, leaving the general path to report it at
         * the same position with the same status.
         *
         * The predictor byte is tested FIRST because it is the selective
         * one: a STRIDE tile (the whole point of the STRIDE family -- a
         * constant-period plane) is rejected by one compare against the
         * range [CONST_ZERO, LANE_DELTA] plus one against LANE_AVERAGE,
         * instead of after eight byte comparisons for the fields that
         * only the rare all-hit record has. */
        if ((rec[1] <= (uint8_t)ZPSE_P_LANE_DELTA ||
             rec[1] == (uint8_t)ZPSE_P_LANE_AVERAGE) &&
            rsz == 8u && nt_exp == 256u &&
            rec[0] == (uint8_t)ZPSE_TILE_FLAG_RESET &&
            rec[2] == 255u && rec[3] == 0u && rec[4] == 0u &&
            rec[5] == 0u && rec[6] == 0u && rec[7] == 0u) {
            uint8_t fill = (rec[1] == (uint8_t)ZPSE_P_CONST_ZERO)
                               ? 0u : literal;
            uint32_t run = 1u;
            uint32_t rpos = pos + 10u;
            uint32_t rprod = produced + 256u;
            uint32_t rt = (uint32_t)t + 1u;

            while (rt < tile_count) {
                uint16_t nt2;
                const uint8_t *r2;

                if (rpos + 2u > payload_len) {
                    break;
                }
                if (zpse_load_u16le(payload + rpos) != 8u) {
                    break;
                }
                if (payload_len - rpos - 2u < 8u) {
                    break;
                }
                nt2 = ((rt + 1u < tile_count)
                           ? 256u
                           : (uint16_t)(seg_size - rprod));
                if (nt2 != 256u) {
                    break;
                }
                if ((uint32_t)nt2 > seg_size - rprod) {
                    break;
                }
                if (rprod + (uint32_t)nt2 > decoded_cap) {
                    break;
                }
                r2 = payload + rpos + 2u;
                if (r2[1] != rec[1] ||
                    r2[0] != (uint8_t)ZPSE_TILE_FLAG_RESET ||
                    r2[2] != 255u || r2[3] != 0u || r2[4] != 0u ||
                    r2[5] != 0u || r2[6] != 0u || r2[7] != 0u) {
                    break;
                }
                run++;
                rpos += 10u;
                rprod += 256u;
                rt++;
            }

            memset(decoded_out + produced, fill, (size_t)run * 256u);
            {
                unsigned l;

                for (l = 0; l < 8u; l++) {
                    lanes->last[l] = fill;
                    lanes->second_last[l] = fill;
                }
            }
            produced = rprod;
            pos = rpos;
            t = (uint16_t)(rt - 1u);
            continue;
        }

        tile_reset = (rec[0] & ZPSE_TILE_FLAG_RESET) != 0 ? 1 : 0;
        got = zpse_tile_decode(lanes, rec, (uint32_t)rsz,
                               literal, tile_reset,
                               decoded_out, produced,
                               decoded_out + produced,
                               decoded_cap - produced, d->comb);
        if (got < 0) {
            return got;
        }
        if ((uint32_t)got != (uint32_t)nt_exp) {
            return PSE_BAD_TILE_SIZE;
        }
        produced += (uint32_t)got;
        pos += 2u + (uint32_t)rsz;
    }
    if (produced != h.segment_decoded_size) {
        return PSE_BAD_TILE_SIZE;
    }
    if (pos != payload_len) {
        return PSE_BAD_TILE_SIZE;
    }
    *decoded_len_out = produced;
    return PSE_OK;
}

/* ---- Encoder tile search (PSE SELECT) -------------------------------
 *
 * A tile always starts from the reset lane state, so the predictions of
 * the cheap predictor families have closed forms in the already-known
 * plane bytes:
 *
 *   0  CONST_ZERO     0
 *   1  LANE_PREVIOUS  p < 8 ? init : a[p-8]
 *   2  LANE_DELTA     p <  8 ? init
 *                     p < 16 ? 2*a[p-8] - init
 *                            : 2*a[p-8] - a[p-16]
 *   5..8 STRIDE_S     seg_pos+p < S ? init : a[p-S]
 *   10 LANE_AVERAGE   p <  8 ? init
 *                     p < 16 ? (a[p-8] + init + 1) >> 1
 *                            : (a[p-8] + a[p-16] + 1) >> 1
 *
 * (Families 1/2/10 differ from the STRIDE forms only at the tile start,
 * which is exactly the difference between "lane histories reset to
 * init" and "read the already-decoded stream".)
 *
 * One pass therefore yields the exact exception count K for eight
 * predictors at once. K alone orders them: a record is
 * 8 + mask_len(K) + residual_len(K) bytes and both terms are monotone
 * in K. The best one or two candidates are then classified exactly once
 * each (which also yields the residual statistics) and packed once each
 * with modes picked by pure length arithmetic. Nothing is
 * encode-and-measured, so the per-tile cost is a fixed handful of
 * passes instead of ~50.
 *
 * Predictors 3/4 (ORDER1/ORDER2) are not trial candidates: theirs are
 * the only resets that cost 2 KiB / 64 KiB of table memset, and they
 * lose to STRIDE and the lane-history family on essentially every input
 * this format targets. They stay fully decodable (tile.c) so foreign
 * and hostile streams remain conformant.
 *
 * 9 (TILE_LOCAL_CONSTANT) is only trial-ed when the eight cheap
 * predictors all miss badly, which is the only situation where a
 * different constant can beat "predict the byte 8 positions back".
 */

/* Exact K for the closed-form families. ks[] is indexed by predictor
 * id; ids outside the set are set to an unreachable sentinel. */
static void seg_prescreen_counts(const uint8_t *a, uint16_t nt,
                                 const uint8_t *seg_base, uint32_t seg_pos,
                                 uint8_t init, uint16_t ks[11])
{
    uint32_t k0 = 0u, k1 = 0u, k2 = 0u, k5 = 0u;
    uint32_t k6 = 0u, k7 = 0u, k8 = 0u, k10 = 0u;
    uint8_t last[8];
    uint8_t second[8];
    unsigned l;
    uint16_t p;

    for (l = 0; l < 8u; l++) {
        last[l] = init;
        second[l] = init;
    }

    for (p = 0; p < nt; p++) {
        unsigned lane = (unsigned)p & 7u;
        uint8_t x = a[p];
        uint8_t h8 = last[lane];
        uint8_t h16 = second[lane];
        uint32_t d = seg_pos + (uint32_t)p;
        uint8_t s1 = (d < 1u) ? init : seg_base[d - 1u];
        uint8_t s2 = (d < 2u) ? init : seg_base[d - 2u];
        uint8_t s4 = (d < 4u) ? init : seg_base[d - 4u];
        uint8_t s8 = (d < 8u) ? init : seg_base[d - 8u];

        k0 += (uint32_t)(x != 0u);
        k1 += (uint32_t)(x != h8);
        k2 += (uint32_t)(x != (uint8_t)((2u * (unsigned)h8 -
                              (unsigned)h16) & 0xFFu));
        k5 += (uint32_t)(x != s1);
        k6 += (uint32_t)(x != s2);
        k7 += (uint32_t)(x != s4);
        k8 += (uint32_t)(x != s8);
        k10 += (uint32_t)(x != (uint8_t)(((unsigned)h8 + (unsigned)h16 + 1u)
                                         >> 1));

        second[lane] = h8;
        last[lane] = x;
    }

    ks[0] = (uint16_t)k0;
    ks[1] = (uint16_t)k1;
    ks[2] = (uint16_t)k2;
    ks[5] = (uint16_t)k5;
    ks[6] = (uint16_t)k6;
    ks[7] = (uint16_t)k7;
    ks[8] = (uint16_t)k8;
    ks[10] = (uint16_t)k10;
    ks[3] = 0xFFFFu;
    ks[4] = 0xFFFFu;
    ks[9] = 0xFFFFu;
}

static const uint8_t seg_screen_pred[8] = { 0u, 1u, 2u, 5u, 6u, 7u, 8u, 10u };

/* ------------------------------------------------------------------ *
 * Vectorised prescreen. Every one of the eight counts above is a
 * byte-wise equality test between the current 32 bytes and a shifted
 * (or arithmetically derived) vector, so AVX2 evaluates all of them for
 * 32 positions at once: compare, movemask, popcount. The three
 * predictions that are not a plain shift are still plain vector ops:
 * `2*h8 - h16` is add+sub, and `(h8 + h16 + 1) >> 1` is exactly
 * _mm256_avg_epu8 (it rounds up).
 *
 * Positions below 16 are the only ones with boundary behaviour (lane
 * histories still at `init`, stride reads possibly before seg_pos), so
 * they stay scalar; the vector loop starts at p = 16, where
 * last[l] = a[p-8+l] and second[l] = a[p-16+l] hold identically.
 *
 * Note that this boundary is anchored by the per-tile lane reset, not
 * by seg_pos: the encoder resets every tile to `initial_literal`, so at
 * p < 8 the lane histories are `init` even in the middle of a segment
 * and the prologue cannot be skipped there. Only the STRIDE reads look
 * back into the plane.
 * ------------------------------------------------------------------ */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
/* GCC/clang accept a per-function target attribute, so AVX2 is enabled
 * for this one routine and the rest of the translation unit stays at the
 * baseline ISA. The runtime capability check still decides whether it
 * runs. */
#define ZPSE_PS_AVX2 __attribute__((target("avx2")))
#define ZPSE_PS_HAVE_AVX2 1
#elif defined(__AVX2__)
/* MSVC has no per-function target attribute: the AVX2 intrinsics exist
 * only when the whole translation unit is built with /arch:AVX2, which
 * is what defines __AVX2__. Build with it and this prescreen is used;
 * build without it and the file compiles to the verified-equivalent
 * scalar path rather than failing outright. Either way the runtime
 * check in seg_prescreen() gates execution, so a /arch:AVX2 binary is
 * still correct on a CPU without AVX2. */
#define ZPSE_PS_AVX2
#define ZPSE_PS_HAVE_AVX2 1
#else
#define ZPSE_PS_AVX2
#define ZPSE_PS_HAVE_AVX2 0
#endif
#else
#define ZPSE_PS_HAVE_AVX2 0
#endif

#if ZPSE_PS_HAVE_AVX2
#include <immintrin.h>

/* Partition of the staging buffer below. */
#define ZPSE_PS_STG_LEN 304u /* 16 head + 256 body + 32 pad */

/* Accumulate one 32-byte chunk's hits into the eight counters.
 * Each comparison contributes 0xFF (-1) per matching byte, so a byte-lane
 * counter sums the hits directly: no movemask plus popcount round trip per
 * comparison, and no 3-cycle latency chain to serialise on. `MASK` is all
 * ones for a full chunk and the validity vector for the last, partial one
 * (whose bytes past the tile end are zero padding that must not match). */
#define SEG_PS_CHUNK(v_, h8L_, h8S_, h16_, MASK_)                          \
    do {                                                                   \
        __m256i _d2 = _mm256_sub_epi8(_mm256_add_epi8(h8L_, h8L_), h16_);  \
        __m256i _av = _mm256_avg_epu8(h8L_, h16_);                         \
        a0 = _mm256_sub_epi8(a0, _mm256_and_si256(                         \
                 _mm256_cmpeq_epi8(v_, zero), MASK_));                     \
        a1 = _mm256_sub_epi8(a1, _mm256_and_si256(                         \
                 _mm256_cmpeq_epi8(v_, h8L_), MASK_));                     \
        a2 = _mm256_sub_epi8(a2, _mm256_and_si256(                         \
                 _mm256_cmpeq_epi8(v_, _d2), MASK_));                      \
        a3 = _mm256_sub_epi8(a3, _mm256_and_si256(_mm256_cmpeq_epi8(v_,    \
                 _mm256_loadu_si256((const __m256i *)(const void *)        \
                     (stg + 15u + (size_t)p))), MASK_));                   \
        a4 = _mm256_sub_epi8(a4, _mm256_and_si256(_mm256_cmpeq_epi8(v_,    \
                 _mm256_loadu_si256((const __m256i *)(const void *)        \
                     (stg + 14u + (size_t)p))), MASK_));                   \
        a5 = _mm256_sub_epi8(a5, _mm256_and_si256(_mm256_cmpeq_epi8(v_,    \
                 _mm256_loadu_si256((const __m256i *)(const void *)        \
                     (stg + 12u + (size_t)p))), MASK_));                   \
        a6 = _mm256_sub_epi8(a6, _mm256_and_si256(                         \
                 _mm256_cmpeq_epi8(v_, h8S_), MASK_));                     \
        a7 = _mm256_sub_epi8(a7, _mm256_and_si256(                         \
                 _mm256_cmpeq_epi8(v_, _av), MASK_));                      \
    } while (0)

/* One pass for the whole tile, so there is no scalar prologue or tail to
 * pay for and no per-position lane state at all: the staging buffer makes
 * every prediction a constant-offset load.
 *
 *   stg[0, 16)        the 16 plane bytes BEFORE the tile
 *   stg[16, 16+nt)    the tile itself
 *   stg[16+nt, ...)   32 zero bytes of pad
 *
 * The head exists because `seg_base[d - s]` for the STRIDE family is a
 * plain `a[p - s]` (a[p-s] == stg[16+p-s] for p >= s, and for p < s it is
 * one of the head bytes) and because the LANEHIST family needs the entry
 * lane state. Both are real plane data once the segment has 16 bytes
 * behind the tile, and both collapse to `init` at seg_pos == 0, where
 * the encoder's reset makes the lane state `init` and the scalar
 * `d < s` fallback (d == p) covers exactly the bytes before the plane
 * start: the caller only uses this path for those two positions, and
 * this function fills the head with `init` for the second. Note it
 * requires `a == seg_base + seg_pos`, which is how
 * zpse_segment_encode calls it (and the only caller).
 *
 * The pad exists so the last, partial chunk can be evaluated as a full
 * 32-byte chunk; its accidental matches are masked out.
 *
 * The two lane-history predictions need the tile's ENTRY state at the
 * start (every tile resets its lanes, SPEC Sec. 7), so for p < 8 the
 * LANE_PREVIOUS operand is `init` and for p < 16 so is the LANE_DELTA /
 * LANE_AVERAGE second operand. Those are the first 8 and first 16 lanes
 * of the first chunk, so one blend each fixes chunk 0 and every later
 * chunk uses the plain shifted load. STRIDE_8 uses the same shifted load
 * but WITHOUT the blend: `seg_base[d-8]` is a real preceding byte even
 * for p < 8, so sharing LANE_PREVIOUS's blended operand there would be
 * wrong (they are distinct candidates and are counted separately). */
static ZPSE_PS_AVX2 void seg_prescreen_avx2(const uint8_t *a, uint16_t nt,
                                            const uint8_t *seg_base,
                                            uint32_t seg_pos, uint8_t init,
                                            uint16_t ks[11])
{
    uint8_t stg[ZPSE_PS_STG_LEN];
    __m256i a0, a1, a2, a3, a4, a5, a6, a7;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i all = _mm256_cmpeq_epi8(zero, zero);
    const __m256i initv = _mm256_set1_epi8((char)init);
    const __m256i idx = _mm256_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                         16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31);
    const __m256i m8 = _mm256_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,
                                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0);
    const __m256i m16 = _mm256_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,
                                         -1,-1,-1,-1,-1,-1,-1,-1,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    uint32_t p = 0;

    (void)seg_base;
    a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = zero;

    if (seg_pos == 0u) {
        /* A segment's first tile. The encoder resets the lane state for
         * every tile, so both things this head has to stand for are
         * `init` here: the STRIDE `d < s` fallback, and the LANEHIST
         * entry state (whose first 8 and first 16 lanes the p == 0
         * blends below force to `init` regardless of what the head
         * holds). Reading a - 16 would step outside the plane, so fill
         * the head instead -- it is the same value the scalar loop
         * would have used. */
        memset(stg, init, 16);
    } else {
        memcpy(stg, a - 16, 16);
    }
    memcpy(stg + 16, a, nt);
    memset(stg + 16 + nt, 0, 32);

    while (p + 32u <= (uint32_t)nt) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + 16u + p));
        __m256i h8S = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + 8u + p));
        __m256i h8L = h8S;
        __m256i h16 = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + p));

        if (p == 0u) {
            h8L = _mm256_blendv_epi8(h8L, initv, m8);
            h16 = _mm256_blendv_epi8(h16, initv, m16);
        }
        SEG_PS_CHUNK(v, h8L, h8S, h16, all);
        p += 32u;
    }
    if (p < (uint32_t)nt) {
        uint32_t rem = (uint32_t)nt - p;
        __m256i valid = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)rem), idx);
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + 16u + p));
        __m256i h8S = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + 8u + p));
        __m256i h8L = h8S;
        __m256i h16 = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + p));

        if (p == 0u) {
            h8L = _mm256_blendv_epi8(h8L, initv, m8);
            h16 = _mm256_blendv_epi8(h16, initv, m16);
        }
        SEG_PS_CHUNK(v, h8L, h8S, h16, valid);
    }

    {
        __m256i acc[8];
        uint64_t hits[8];
        uint64_t lane[4];
        int i;
        int j;

        acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
        acc[4] = a4; acc[5] = a5; acc[6] = a6; acc[7] = a7;
        for (i = 0; i < 8; i++) {
            hits[i] = 0u;
            /* Sum of absolute differences from zero is the sum of the
             * byte-lane counters, i.e. the total number of hits. */
            _mm256_storeu_si256((__m256i *)(void *)lane,
                                _mm256_sad_epu8(acc[i], zero));
            for (j = 0; j < 4; j++) {
                hits[i] += lane[j];
            }
        }
        ks[0] = (uint16_t)((uint32_t)nt - (uint32_t)hits[0]);
        ks[1] = (uint16_t)((uint32_t)nt - (uint32_t)hits[1]);
        ks[2] = (uint16_t)((uint32_t)nt - (uint32_t)hits[2]);
        ks[5] = (uint16_t)((uint32_t)nt - (uint32_t)hits[3]);
        ks[6] = (uint16_t)((uint32_t)nt - (uint32_t)hits[4]);
        ks[7] = (uint16_t)((uint32_t)nt - (uint32_t)hits[5]);
        ks[8] = (uint16_t)((uint32_t)nt - (uint32_t)hits[6]);
        ks[10] = (uint16_t)((uint32_t)nt - (uint32_t)hits[7]);
        ks[3] = 0xFFFFu;
        ks[4] = 0xFFFFu;
        ks[9] = 0xFFFFu;
    }
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* Dispatch helper: `have_avx2` is computed once per segment encode so
 * no capability state has to be cached in a global. */
static void seg_prescreen(const uint8_t *a, uint16_t nt,
                          const uint8_t *seg_base, uint32_t seg_pos,
                          uint8_t init, int have_avx2, uint16_t ks[11])
{
#if ZPSE_PS_HAVE_AVX2
    /* The vector path folds the tile boundaries into the whole-tile
     * pass and reads 16 plane bytes before the tile: real ones once the
     * segment has that much history, and `init` for a segment's first
     * tile, where the reset lane state and the STRIDE `d < s` fallback
     * both make every byte before the tile `init` (seg_prescreen_avx2
     * fills the head for that case). Those are the only two positions
     * that occur -- seg_pos is a tile index times 256 -- and any other
     * keeps the scalar path rather than silently widening the
     * contract. The scalar first-tile prescreen this replaces cost
     * 853 ns against 30 ns for every later tile, i.e. ~13 us per block
     * on the 16-segment transform trial. */
    if (have_avx2 && (seg_pos == 0u || seg_pos >= 16u)) {
        seg_prescreen_avx2(a, nt, seg_base, seg_pos, init, ks);
        return;
    }
#else
    (void)have_avx2;
#endif
    seg_prescreen_counts(a, nt, seg_base, seg_pos, init, ks);
}

/* Emit the minimal all-hit tile record: mask ALL_HIT and residual
 * EXCLUSION_U8 with k == 0, so the whole tile is prediction and both
 * payloads are empty. This is byte for byte what zpse_tile_pack's own
 * k == 0 arm writes for that pair, which is the pair
 * zpse_tile_choose_modes picks whenever k == 0. 10 bytes including the
 * u16 record-length prefix; the caller has checked the capacity. `nt` is
 * the tile length, so the last tile of a segment still writes nt-1 into
 * tile_length_minus_1. */
static void seg_write_all_hit(uint8_t *dst, uint8_t predictor_id,
                              uint16_t nt)
{
    zpse_store_u16le(dst, (uint16_t)ZPSE_TILE_HEADER_SIZE);
    dst[2] = zpse_tile_flags_make((uint8_t)ZPSE_M_ALL_HIT,
                                  (uint8_t)ZPSE_R_EXCLUSION_U8, 1);
    dst[3] = predictor_id;
    dst[4] = (uint8_t)(nt - (uint16_t)1u);
    dst[5] = 0u; /* exception count: unused by both extreme mask modes */
    zpse_store_u16le(dst + 6, 0u); /* mask payload size */
    zpse_store_u16le(dst + 8, 0u); /* residual payload size */
}

int zpse_segment_encode(ZpseEncoder *e, ZpseLaneState *lanes,
                        const uint8_t *plane, uint32_t plane_len,
                        uint8_t initial_literal,
                        uint8_t *payload_out, uint32_t payload_cap,
                        uint32_t byte_budget)
{
    ZpseSegmentHeader h;
    uint32_t tc;
    uint32_t pos;
    uint32_t t;
    uint8_t *tmp;
    uint32_t tmp_cap;
    int have_avx2 = (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;

    if (e == 0 || lanes == 0 || plane == 0 || payload_out == 0) {
        return PSE_BAD_ARG;
    }
    if (plane_len < 1u || plane_len > 65536u) {
        return PSE_BAD_ARG;
    }
    if (e->hist == 0 || e->comb == 0 || e->input == 0) {
        return PSE_BAD_ARG;
    }
    if (payload_cap < 16u) {
        return PSE_OUTPUT_OVERFLOW;
    }
    tc = (plane_len + 255u) / 256u;

    h.segment_decoded_size = plane_len;
    h.tile_count = (uint16_t)tc;
    h.initial_literal = initial_literal;
    h.segment_flags = 0x01u;
    h.tile_directory_size = 0u;
    h.reserved = 0u;
    zpse_segment_header_serialize(payload_out, &h);

    /* Scratch record buffer for the speculative trials (the worst tile
     * record is under 512 B; the hist region is 4 KiB and unused by the
     * tile path). */
    tmp = e->hist;
    tmp_cap = 4096u;

    /* Reset-first lane discipline; the per-tile reset bit is always 1
     * (simplest legal choice, keeps every tile independent). */
    zpse_lane_reset(lanes, initial_literal);

    pos = 16u;
    for (t = 0; t < tc; t++) {
        uint32_t seg_pos = t * 256u;
        uint16_t nt =
            (uint16_t)((t + 1u < tc) ? 256u : (plane_len - seg_pos));
        const uint8_t *actual = plane + seg_pos;
        uint16_t ks[11];
        uint8_t cand[3];
        uint32_t ncand = 0;
        uint32_t ci;
        int best_len = -1;
        uint16_t best_k = 0u;
        uint8_t best_p = 0;
        uint8_t *bst = e->input; /* staging for the winning record */
        int n;

        /* Bound check for the byte budget, before any work for this
         * tile: `pos` grows monotonically, so once it is past the budget
         * nothing later could come back under it. No exception path can
         * be skipped by this -- the caller only passes a budget when a
         * total that large cannot win -- and no tile is half-written. */
        if (byte_budget != 0u && pos > byte_budget) {
            return PSE_OUTPUT_OVERFLOW;
        }

        seg_prescreen(actual, nt, plane, seg_pos, initial_literal,
                      have_avx2, ks);
        best_p = 0;
        {
            uint16_t bk = ks[seg_screen_pred[0]];
            uint32_t i;
            for (i = 1; i < 8u; i++) {
                uint16_t c = ks[seg_screen_pred[i]];
                if (c < bk) {
                    bk = c;
                    best_p = seg_screen_pred[i];
                }
            }
            best_k = bk;
            cand[ncand++] = best_p;

            /* Second trial only when it is genuinely competitive. */
            {
                uint8_t second_p = 0xFFu;
                uint16_t sk = 0xFFFFu;
                for (i = 0; i < 8u; i++) {
                    uint16_t c = ks[seg_screen_pred[i]];
                    if (seg_screen_pred[i] == best_p) {
                        continue;
                    }
                    if (c < sk) {
                        sk = c;
                        second_p = seg_screen_pred[i];
                    }
                }
                /* Second trial only when it is genuinely competitive,
                 * and never when the best candidate is already perfect:
                 * bk == 0 selects mask ALL_HIT with residual
                 * EXCLUSION_U8, whose payloads are both empty, so
                 * zpse_tile_pack returns exactly ZPSE_TILE_HEADER_SIZE
                 * -- the smallest record that can exist. The candidate
                 * loop replaces the winner only on a strictly smaller
                 * packed size, which no second candidate can be against
                 * an 8-byte record, so appending one here could not have
                 * changed the output. */
                if (bk != 0u && second_p != 0xFFu &&
                    (uint32_t)sk <= (uint32_t)bk + 4u) {
                    cand[ncand++] = second_p;
                }
            }
            /* Near-random tile (every cheap predictor misses almost
             * everywhere): one trial is enough. A second classification
             * plus its mode search would cost the same again for a
             * difference of a byte or two. */
            if ((uint32_t)bk * 8u >= (uint32_t)nt * 7u) {
                ncand = 1u;
            }
        }

        /* ---- All-hit run ------------------------------------------ *
         * best_k == 0 means the winning screened predictor predicts the
         * whole tile: mask ALL_HIT with an empty residual, i.e. exactly
         * the 8-byte record the packer writes for (k == 0, ALL_HIT,
         * EXCLUSION_U8). Neither the mode search nor the packer can turn
         * that classification into any other record, no second candidate
         * is ever appended when bk == 0, and 8 bytes is the smallest
         * record a tile can have, so the winner byte and the record are
         * both forced -- which is what lets this write them directly
         * instead of classifying, choosing and packing per tile.
         *
         * A run is just more of the same: past the first tile the
         * prescreen is the only thing that decides which predictor byte
         * the general path would have written, so stay here while it
         * keeps reporting k == 0 for the same winner. Any other tile
         * falls out to the general path, which re-derives everything
         * from the same prescreen result -- each tile is written by
         * exactly one of the two paths, and the bytes agree, so the
         * payload is unchanged. The lane state needs no update: the
         * classify call for the next tile is always made with
         * tile_reset == 1, which resets the histories and tables itself,
         * and nothing after this loop reads `lanes`.
         *
         * The run stops (rather than overflowing) on the same capacity
         * condition the general path returns PSE_OUTPUT_OVERFLOW for. */
        if (best_k == 0u) {
            uint8_t pred = best_p;

            if (payload_cap - pos < 10u) {
                return PSE_OUTPUT_OVERFLOW;
            }
            seg_write_all_hit(payload_out + pos, pred, nt);
            pos += 10u;

            while (t + 1u < tc) {
                uint32_t nsseg_pos = (t + 1u) * 256u;
                uint16_t nnt = (uint16_t)((t + 2u < tc)
                                              ? 256u
                                              : (plane_len - nsseg_pos));
                uint16_t nks[11];
                uint16_t nbk;
                uint8_t npred = seg_screen_pred[0];
                uint32_t i;

                if (byte_budget != 0u && pos > byte_budget) {
                    return PSE_OUTPUT_OVERFLOW;
                }
                if (payload_cap - pos < 10u) {
                    return PSE_OUTPUT_OVERFLOW;
                }
                seg_prescreen(plane + nsseg_pos, nnt, plane, nsseg_pos,
                              initial_literal, have_avx2, nks);
                nbk = nks[npred];
                for (i = 1; i < 8u; i++) {
                    uint16_t c = nks[seg_screen_pred[i]];
                    if (c < nbk) {
                        nbk = c;
                        npred = seg_screen_pred[i];
                    }
                }
                if (nbk != 0u || npred != pred) {
                    break;
                }
                seg_write_all_hit(payload_out + pos, npred, nnt);
                pos += 10u;
                t++;
            }
            continue;
        }

        for (ci = 0; ci < ncand; ci++) {
            uint8_t pred = cand[ci];
            ZpseTileClass cl;
            uint8_t mm;
            uint8_t rr;
            int rc = zpse_tile_classify(lanes, actual, nt, pred,
                                        initial_literal, 1, plane, seg_pos,
                                        &cl);
            if (rc != PSE_OK) {
                continue;
            }
            zpse_tile_choose_modes(&cl, nt, &mm, &rr);
            n = zpse_tile_pack(pred, &cl, nt, actual, mm, rr, 1, tmp,
                               tmp_cap, e->comb);
            if (n < 8) {
                /* Defensive fallback: the always-legal minimal pair. */
                mm = (cl.k == 0u) ? 0u
                     : (cl.k == nt) ? 6u
                     : ((uint32_t)cl.k <= (((uint32_t)nt + 7u) >> 3) ? 2u
                                                                     : 1u);
                n = zpse_tile_pack(pred, &cl, nt, actual, mm, 0u, 1, tmp,
                                   tmp_cap, e->comb);
                rr = 0u;
            }
            if (n < 8) {
                continue;
            }
            if (best_len < 0 || n < best_len) {
                memcpy(bst, tmp, (size_t)n);
                best_len = n;
                best_p = pred;
            }
            /* A record of just the 8-byte tile header (9 with the
             * predictor-9 mode byte) carries no mask and no residual
             * payload at all, so it is the global floor for this tile:
             * nothing left to try can be smaller. This is the case for
             * every fully predicted tile. */
            if (best_len <= 8 + (best_p == 9u ? 1 : 0)) {
                break;
            }
        }
        (void)best_p;

        if (best_len < 0) {
            return PSE_BAD_RESIDUAL;
        }
        if (pos + 2u > payload_cap ||
            (uint32_t)best_len > payload_cap - pos - 2u) {
            return PSE_OUTPUT_OVERFLOW;
        }
        memcpy(payload_out + pos + 2u, bst, (size_t)best_len);
        zpse_store_u16le(payload_out + pos, (uint16_t)best_len);
        pos += 2u + (uint32_t)best_len;
    }
    return (int)pos;
}
