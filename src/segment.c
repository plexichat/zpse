#include "zpse/segment.h"

#include <string.h>

#include "zpse/bits.h"
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
    for (t = 0; t < h.tile_count; t++) {
        uint16_t nt_exp;
        uint16_t rsz;
        const uint8_t *rec;
        int tile_reset;
        int got;

        if (t + 1u < (uint32_t)h.tile_count) {
            nt_exp = 256u;
        } else {
            nt_exp = (uint16_t)(h.segment_decoded_size - produced);
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
        if ((uint32_t)nt_exp > h.segment_decoded_size - produced) {
            return PSE_BAD_TILE_SIZE;
        }
        if (produced + (uint32_t)nt_exp > decoded_cap) {
            return PSE_OUTPUT_OVERFLOW;
        }
        rec = payload + pos + 2u;
        tile_reset = (rec[0] & ZPSE_TILE_FLAG_RESET) != 0 ? 1 : 0;
        got = zpse_tile_decode(lanes, rec, (uint32_t)rsz,
                               h.initial_literal, tile_reset,
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
static ZPSE_PS_AVX2 void seg_prescreen_avx2(const uint8_t *a, uint16_t nt,
                                            const uint8_t *seg_base,
                                            uint32_t seg_pos, uint8_t init,
                                            uint16_t ks[11])
{
    uint32_t c[8]; /* k0 k1 k2 k5 k6 k7 k8 k10 */
    uint8_t last[8];
    uint8_t second[8];
    unsigned l;
    uint16_t p = 0;
    uint16_t stop = (nt < 16u) ? nt : 16u;

    for (l = 0; l < 8u; l++) {
        c[l] = 0u;
        last[l] = init;
        second[l] = init;
    }

    /* Scalar prologue [0, 16). */
    for (; p < stop; p++) {
        unsigned lane = (unsigned)p & 7u;
        uint8_t x = a[p];
        uint8_t h8 = last[lane];
        uint8_t h16 = second[lane];
        uint32_t d = seg_pos + (uint32_t)p;
        uint8_t s1 = (d < 1u) ? init : seg_base[d - 1u];
        uint8_t s2 = (d < 2u) ? init : seg_base[d - 2u];
        uint8_t s4 = (d < 4u) ? init : seg_base[d - 4u];
        uint8_t s8 = (d < 8u) ? init : seg_base[d - 8u];

        c[0] += (uint32_t)(x != 0u);
        c[1] += (uint32_t)(x != h8);
        c[2] += (uint32_t)(x != (uint8_t)((2u * (unsigned)h8 -
                              (unsigned)h16) & 0xFFu));
        c[3] += (uint32_t)(x != s1);
        c[4] += (uint32_t)(x != s2);
        c[5] += (uint32_t)(x != s4);
        c[6] += (uint32_t)(x != s8);
        c[7] += (uint32_t)(x != (uint8_t)(((unsigned)h8 + (unsigned)h16 + 1u)
                                          >> 1));
        second[lane] = h8;
        last[lane] = x;
    }

    /* Vector body [16, nt) in 32-byte chunks. */
    if (p == 16u) {
        const __m256i zero = _mm256_setzero_si256();
        while ((uint32_t)p + 32u <= (uint32_t)nt) {
            const uint8_t *q = a + p;
            __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)q);
            __m256i h8 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(q - 8));
            __m256i h16 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(q - 16));
            __m256i q1 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(q - 1));
            __m256i q2 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(q - 2));
            __m256i q4 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(q - 4));
            __m256i d2 = _mm256_sub_epi8(_mm256_add_epi8(h8, h8), h16);
            __m256i av = _mm256_avg_epu8(h8, h16);
            /* These are exception counts, so each chunk contributes
             * (32 - hits). Every chunk is full, so the width is fixed. */
            uint32_t m1 = (uint32_t)_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(v, h8));

            c[0] += 32u - (uint32_t)__builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, zero)));
            c[1] += 32u - (uint32_t)__builtin_popcount((unsigned)m1);
            c[2] += 32u - (uint32_t)__builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, d2)));
            c[3] += 32u - (uint32_t)__builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, q1)));
            c[4] += 32u - (uint32_t)__builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, q2)));
            c[5] += 32u - (uint32_t)__builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, q4)));
            /* STRIDE_8 reads the same byte LANE_PREVIOUS predicts for
             * every p >= 8, so the count is shared. */
            c[6] += 32u - (uint32_t)__builtin_popcount((unsigned)m1);
            c[7] += 32u - (uint32_t)__builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, av)));
            p = (uint16_t)(p + 32u);
        }
        if (p > 16u) {
            /* Lane histories at p are plain memory: a[p-8+l], a[p-16+l]. */
            for (l = 0; l < 8u; l++) {
                last[l] = a[p - 8u + l];
                second[l] = a[p - 16u + l];
            }
        }
    }

    /* Scalar tail (< 32 bytes). */
    for (; p < nt; p++) {
        unsigned lane = (unsigned)p & 7u;
        uint8_t x = a[p];
        uint8_t h8 = last[lane];
        uint8_t h16 = second[lane];
        uint8_t s1 = a[p - 1u];
        uint8_t s2 = a[p - 2u];
        uint8_t s4 = a[p - 4u];
        uint8_t s8 = a[p - 8u];

        c[0] += (uint32_t)(x != 0u);
        c[1] += (uint32_t)(x != h8);
        c[2] += (uint32_t)(x != (uint8_t)((2u * (unsigned)h8 -
                              (unsigned)h16) & 0xFFu));
        c[3] += (uint32_t)(x != s1);
        c[4] += (uint32_t)(x != s2);
        c[5] += (uint32_t)(x != s4);
        c[6] += (uint32_t)(x != s8);
        c[7] += (uint32_t)(x != (uint8_t)(((unsigned)h8 + (unsigned)h16 + 1u)
                                          >> 1));
        second[lane] = h8;
        last[lane] = x;
    }

    ks[0] = (uint16_t)c[0];
    ks[1] = (uint16_t)c[1];
    ks[2] = (uint16_t)c[2];
    ks[5] = (uint16_t)c[3];
    ks[6] = (uint16_t)c[4];
    ks[7] = (uint16_t)c[5];
    ks[8] = (uint16_t)c[6];
    ks[10] = (uint16_t)c[7];
    ks[3] = 0xFFFFu;
    ks[4] = 0xFFFFu;
    ks[9] = 0xFFFFu;
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* Dispatch helper: `have_avx2` is computed once per segment encode so
 * no capability state has to be cached in a global. */
static void seg_prescreen(const uint8_t *a, uint16_t nt,
                          const uint8_t *seg_base, uint32_t seg_pos,
                          uint8_t init, int have_avx2, uint16_t ks[11])
{
#if ZPSE_PS_HAVE_AVX2
    if (have_avx2) {
        seg_prescreen_avx2(a, nt, seg_base, seg_pos, init, ks);
        return;
    }
#else
    (void)have_avx2;
#endif
    seg_prescreen_counts(a, nt, seg_base, seg_pos, init, ks);
}

int zpse_segment_encode(ZpseEncoder *e, ZpseLaneState *lanes,
                        const uint8_t *plane, uint32_t plane_len,
                        uint8_t initial_literal,
                        uint8_t *payload_out, uint32_t payload_cap)
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
        uint8_t best_p = 0;
        uint8_t *bst = e->input; /* staging for the winning record */
        int n;

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
                if (second_p != 0xFFu &&
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
