#include "zpse/segment.h"

#include <string.h>

#include "zpse/bits.h"
#include "zpse/dense.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"
#include "zpse/tile.h"
#include "zpse/types.h"
#include "zpse/vector.h"

/* STRIDE period of an all-hit STRIDE record (predictor ids 5..8). Mirrors
 * tile_stride_of in src/tile.c, spelled out here so a change to the enum
 * ordering cannot silently change what a batched run fills. */
static uint32_t seg_all_hit_stride(uint8_t predictor_id)
{
    switch (predictor_id) {
    case ZPSE_P_STRIDE_1:
        return 1u;
    case ZPSE_P_STRIDE_2:
        return 2u;
    case ZPSE_P_STRIDE_3:
        return 3u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u; /* ZPSE_P_STRIDE_8 */
    }
}

/* One unaligned 64-bit load. A tile record's 8-byte header is compared as
 * a single word below; loading it through memcpy is how the rest of the
 * codebase reads unaligned words (crc32c.c, residual.c), and it compiles
 * to one load on the targets that matter. */
static uint64_t seg_load_u64(const uint8_t *p)
{
    uint64_t w;

    memcpy(&w, p, sizeof w);
    return w;
}

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
    int o1_seed;
    int o2_seed;

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
    if (h.tile_count == 0u) {
        return PSE_BAD_TILE_SIZE;
    }
    if (h.segment_decoded_size > decoded_cap) {
        return PSE_OUTPUT_OVERFLOW;
    }

    /* ---- Dense record (Sec. 6.3) -------------------------------- *
     * One predictor, one mask and one residual stream for the whole
     * plane, either marked by bit 6 of the record's flag byte. That bit
     * is reserved in a tile record and tile_header_validate rejects it,
     * so no payload that decodes as tiles can be read as a dense record:
     * the two interpretations cannot overlap. A payload carrying it and
     * failing validation below is rejected with the dense record's own
     * statuses, exactly as a malformed tile is rejected with the tile
     * ones. The record is the segment's only record, so the whole plane
     * (and the segment's decoded size) comes from one call. */
    if (payload_len > 18u && (payload[18] & ZPSE_DENSE_MARKER) != 0u) {
        if ((uint32_t)h.tile_count != 1u) {
            return PSE_BAD_TILE_SIZE;
        }
        return zpse_dense_decode(&h, payload, payload_len, decoded_out,
                                 decoded_cap, decoded_len_out, d->comb);
    }
    if ((uint32_t)h.tile_count != expect_tc) {
        return PSE_BAD_TILE_SIZE;
    }

    /* First tile of the segment always re-seeds lane state,
     * regardless of that tile's reset bit (Sec. 6.2 / 7). What is
     * established here is the HISTORIES: they are what every tile reads
     * as its entry state, and they are 16 bytes. The ORDER1/ORDER2
     * tables are 66 KiB of memset that only predictors 3 and 4 read, and
     * a resetting tile re-seeds the tables it reads itself (inside
     * zpse_tile_decode -> zpse_lane_reset_for), so a segment whose tiles
     * all reset -- which is every segment the encoder emits, and the only
     * shape the format's own encoder produces -- never touches them. A
     * tile that does NOT reset reads the state the segment start
     * establishes, so that state is seeded below, lazily, before the
     * first such tile that can read a table: same values, same order of
     * observations as the eager reset, without the 66 KiB for tables no
     * tile in the segment reads. */
    zpse_lane_reset_hist(lanes, h.initial_literal);

    pos = 16u;
    produced = 0u;
    o1_seed = 0;
    o2_seed = 0;
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
         * and one of the CONST, lane-history or STRIDE predictors. Such a
         * tile is pure prediction: a single repeated byte for the CONST
         * and lane-history families, and a period-s pattern for STRIDE.
         * A maximal run of them is therefore one fill of the run's
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
         * one: an ORDER1/ORDER2/TILE_CONSTANT tile fails it on the first
         * (or third) compare, instead of after eight byte comparisons for
         * the fields that only an all-hit record has.
         *
         * The STRIDE family (ids 5..8) is batched here too, and it is the
         * one case whose fill is not a single byte. An all-hit STRIDE tile
         * is exactly the constant-period recurrence
         *
         *     out[d] = (d < s) ? literal : out[d - s],   s in {1,2,4,8},
         *
         * over segment-absolute positions d, and that rule does not depend
         * on where a tile starts. A run of R such tiles therefore fills
         * [run_start, run_start + 256*R) with one ascending evaluation of
         * it: an s-byte seed, read from decoded_out (which already holds
         * every byte before the run -- in-place is what the per-tile path
         * relies on as well), then the same repeated doubling the per-tile
         * path uses to expand a seed to a whole tile. */
        if ((rec[1] <= (uint8_t)ZPSE_P_LANE_DELTA ||
             rec[1] == (uint8_t)ZPSE_P_LANE_AVERAGE ||
             rec[1] == (uint8_t)ZPSE_P_STRIDE_3 ||
             (rec[1] >= (uint8_t)ZPSE_P_STRIDE_1 &&
              rec[1] <= (uint8_t)ZPSE_P_STRIDE_8)) &&
            rsz == 8u && nt_exp == 256u &&
            rec[0] == (uint8_t)ZPSE_TILE_FLAG_RESET &&
            rec[2] == 255u && rec[3] == 0u && rec[4] == 0u &&
            rec[5] == 0u && rec[6] == 0u && rec[7] == 0u) {
            uint32_t run = 1u;
            uint32_t rpos = pos + 10u;
            uint32_t rprod = produced + 256u;
            uint32_t rt = (uint32_t)t + 1u;
            /* Every record of the run is the same ten bytes -- [u16 size]
             * followed by the 8-byte header that opened it -- so membership
             * is one size compare plus ONE word compare against that header,
             * instead of the byte-at-a-time comparisons below. The word
             * covers the whole header (reset flag, predictor, both payload
             * sizes), which is exactly what those comparisons test, and the
             * run's first record has already passed them, so a record whose
             * word matches is a member for the same reason it was. */
            uint64_t pat = seg_load_u64(rec);

            /* Two records per step, and only while the scalar loop below
             * would accept both. The three checks it makes beyond the
             * record's own bytes are all implied here, which is why they are
             * not repeated: a step requires rt + 2 < tile_count, so neither
             * record is the segment's LAST tile, and for any tile but the
             * last `nt2 == 256`, `seg_size - rprod >= 256` (the tile count
             * is ceil(seg_size/256), so rprod + 256 <= 256*(tile_count-1) <
             * seg_size) and `rprod + 256 <= seg_size <= decoded_cap` (the
             * entry check) all hold by construction. The payload bound is
             * tested before any load, so both records are inside it. A step
             * that cannot be taken -- the last tile, a short tail, a record
             * that differs -- leaves rpos/rprod/rt on that record and the
             * loop below re-decides it exactly as before. */
            while (rt + 2u < tile_count && payload_len - rpos >= 20u &&
                   zpse_load_u16le(payload + rpos) == 8u &&
                   zpse_load_u16le(payload + rpos + 10u) == 8u &&
                   seg_load_u64(payload + rpos + 2u) == pat &&
                   seg_load_u64(payload + rpos + 12u) == pat) {
                run += 2u;
                rpos += 20u;
                rprod += 512u;
                rt += 2u;
            }

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

            if (rec[1] >= (uint8_t)ZPSE_P_STRIDE_1) {
                /* STRIDE run: one ascending evaluation of the period-s
                 * recurrence over the whole run (see the comment above).
                 * Every tile of the run is a full 256-byte tile -- the
                 * scan below stops at the first one that is not -- and
                 * every one of them was already bounded against
                 * decoded_cap and seg_size by that same scan. */
                uint32_t s = seg_all_hit_stride(rec[1]);
                uint32_t run_len = run * 256u;
                uint8_t *base = decoded_out + produced;
                uint32_t seed = (run_len < s) ? run_len : s;
                uint32_t filled;
                uint32_t i;

                for (i = 0; i < seed; i++) {
                    uint32_t d = produced + i;

                    base[i] = (d < s) ? literal : decoded_out[d - s];
                }
                filled = seed;
                while (filled < run_len) {
                    uint32_t chunk = filled;

                    if (chunk > run_len - filled) {
                        chunk = run_len - filled;
                    }
                    memcpy(base + filled, base, chunk);
                    filled += chunk;
                }
                /* End state = the state the run's LAST tile leaves, which
                 * is what a following non-resetting tile reads. For a full
                 * tile the epilogue of tile_history_fixup (src/tile.c)
                 * reduces to: lane l ends on position 248 + l of that tile
                 * (every lastp is >= 8 for nt == 256, so no entry state is
                 * involved) and its second history is 8 positions back. */
                {
                    unsigned l;
                    uint32_t base_last = produced + run_len - 256u;

                    for (l = 0; l < 8u; l++) {
                        lanes->last[l] =
                            decoded_out[base_last + 248u + l];
                        lanes->second_last[l] =
                            decoded_out[base_last + 240u + l];
                    }
                }
            } else {
                /* CONST / lane-history tile: 0 for CONST_ZERO, the initial
                 * literal otherwise, and a freshly reset lane state is a
                 * fixed point at that byte (which is what one tile
                 * leaves). */
                uint8_t fill = (rec[1] == (uint8_t)ZPSE_P_CONST_ZERO)
                                   ? 0u : literal;

                memset(decoded_out + produced, fill, (size_t)run * 256u);
                {
                    unsigned l;

                    for (l = 0; l < 8u; l++) {
                        lanes->last[l] = fill;
                        lanes->second_last[l] = fill;
                    }
                }
            }
            produced = rprod;
            pos = rpos;
            t = (uint16_t)(rt - 1u);
            continue;
        }

        tile_reset = (rec[0] & ZPSE_TILE_FLAG_RESET) != 0 ? 1 : 0;
        /* Lazy table seeding (see the reset above). A tile that reads a
         * table it does not reset reads this segment's initial state for
         * it, so seed exactly that table here, once. rec[1] is the
         * predictor byte and is read only when the record has one; a
         * predictor that is not 3 or 4 reads no table, and an invalid
         * predictor byte fails validation below whatever this does. */
        if (rsz >= 2u) {
            uint8_t pid = rec[1];

            if (pid == (uint8_t)ZPSE_P_ORDER1) {
                if (o1_seed == 0) {
                    if (tile_reset == 0) {
                        zpse_lane_seed_table(lanes, literal, pid);
                    }
                    o1_seed = 1;
                }
            } else if (pid == (uint8_t)ZPSE_P_ORDER2) {
                if (o2_seed == 0) {
                    if (tile_reset == 0) {
                        zpse_lane_seed_table(lanes, literal, pid);
                    }
                    o2_seed = 1;
                }
            }
        }
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
                                 uint8_t init, uint16_t ks[12])
{
    uint32_t k0 = 0u, k1 = 0u, k2 = 0u, k5 = 0u;
    uint32_t k6 = 0u, k7 = 0u, k8 = 0u, k10 = 0u, k11 = 0u;
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
        uint8_t s3 = (d < 3u) ? init : seg_base[d - 3u];
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
        k11 += (uint32_t)(x != s3);
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
    ks[11] = (uint16_t)k11;
    ks[3] = 0xFFFFu;
    ks[4] = 0xFFFFu;
    ks[9] = 0xFFFFu;
}

/* STRIDE_3 is last on purpose: the winner is the first predictor with the
 * strictly smallest miss count, so appending it can only change a tile
 * where it is strictly better than every earlier candidate. */
static const uint8_t seg_screen_pred[9] = {0u, 1u, 2u, 5u, 6u,
                                           7u, 8u, 10u, 11u};

/* ------------------------------------------------------------------ *
 * Vectorised prescreen. Every one of the nine counts above is a
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

/* Accumulate one 32-byte chunk's hits into the nine counters.
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
        a8 = _mm256_sub_epi8(a8, _mm256_and_si256(_mm256_cmpeq_epi8(v_,    \
                 _mm256_loadu_si256((const __m256i *)(const void *)        \
                     (stg + 13u + (size_t)p))), MASK_));                   \
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
                                            uint16_t ks[12])
{
    uint8_t stg[ZPSE_PS_STG_LEN];
    __m256i a0, a1, a2, a3, a4, a5, a6, a7, a8;
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
    a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = zero;

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
        __m256i acc[9];
        uint64_t hits[9];
        uint64_t lane[4];
        int i;
        int j;

        acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
        acc[4] = a4; acc[5] = a5; acc[6] = a6; acc[7] = a7; acc[8] = a8;
        for (i = 0; i < 9; i++) {
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
        ks[11] = (uint16_t)((uint32_t)nt - (uint32_t)hits[8]);
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
                          uint8_t init, int have_avx2, uint16_t ks[12])
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

/* ------------------------------------------------------------------ *
 * Cheap run continuation for the all-hit batch in zpse_segment_encode.
 *
 * Once an all-hit run is under way, the only thing the general path would
 * still consult for the next tile is seg_prescreen, and a tile that keeps
 * satisfying the run's own predictor cannot change the winner. For the
 * cases below that follows from the tile's own bytes plus the segment
 * literal, so a single broadcast compare can replace the whole
 * nine-predictor pass (~30 ns per tile, the dominant cost of encoding a
 * constant or periodic stream). The winner of the prescreen's scan is the
 * FIRST predictor of seg_screen_pred[] whose miss count is 0, so each
 * case also has to exclude every predictor that precedes it:
 *
 *   CONST_ZERO     (first of all) all bytes are 0.
 *   LANE_PREVIOUS  all bytes equal the tile's entry lane value, which for
 *                  the encoder's always-reset tiles is the segment
 *                  literal; requires a nonzero literal, otherwise
 *                  CONST_ZERO would also be all-hit and would precede it.
 *   STRIDE_1       the tile is constant and equal to the plane byte just
 *                  before it, which is exactly what `d - 1` predicts for
 *                  every position since seg_pos >= 1 throughout the scan.
 *                  That byte must be neither 0 (CONST_ZERO would be
 *                  all-hit and precedes) nor the literal (LANE_PREVIOUS
 *                  would be all-hit and precedes - and LANE_DELTA and
 *                  LANE_AVERAGE too: their prediction starts from the
 *                  literal while the histories still hold it, so a run
 *                  that never departs from the literal keeps agreeing).
 *
 * LANE_DELTA / LANE_AVERAGE / the other STRIDE periods are deliberately
 * not here: a tile all-hit for LANE_DELTA or LANE_AVERAGE is constant at
 * the literal, which makes LANE_PREVIOUS all-hit as well, and STRIDE_2/4/8
 * would additionally have to exclude the shorter strides. Both fall
 * through to seg_prescreen, which stays authoritative.
 *
 * Returns 1 only when the tile provably keeps the run's predictor and its
 * k == 0; 0 means "ask the prescreen". Being wrong towards 0 only costs
 * time, while returning 1 for a tile the prescreen would have rejected
 * would change the encoded bytes - so every condition above is a proof,
 * not a heuristic.
 * ------------------------------------------------------------------ */
static int seg_all_hit_cheap_scalar(const uint8_t *a, uint16_t nt,
                                    uint8_t pred, uint8_t literal,
                                    uint32_t seg_pos, const uint8_t *plane)
{
    uint16_t i;

    if (pred == (uint8_t)ZPSE_P_CONST_ZERO) {
        for (i = 0; i < nt; i++) {
            if (a[i] != 0u)
                return 0;
        }
        return 1;
    }
    if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
        if (literal == 0u)
            return 0;
        for (i = 0; i < nt; i++) {
            if (a[i] != literal)
                return 0;
        }
        return 1;
    }
    if (pred == (uint8_t)ZPSE_P_STRIDE_1 && seg_pos >= 1u) {
        uint8_t v = plane[seg_pos - 1u];

        if (v == 0u || v == literal)
            return 0;
        for (i = 0; i < nt; i++) {
            if (a[i] != v)
                return 0;
        }
        return 1;
    }
    return 0;
}

#if ZPSE_PS_HAVE_AVX2
/* Same three proofs, one broadcast compare each: 32 bytes per step. */
static ZPSE_PS_AVX2 int seg_all_hit_cheap_avx2(const uint8_t *a, uint16_t nt,
                                               uint8_t pred, uint8_t literal,
                                               uint32_t seg_pos,
                                               const uint8_t *plane)
{
    uint8_t v;
    uint32_t p = 0;
    __m256i want;

    if (pred == (uint8_t)ZPSE_P_CONST_ZERO) {
        v = 0u;
    } else if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
        if (literal == 0u)
            return 0;
        v = literal;
    } else if (pred == (uint8_t)ZPSE_P_STRIDE_1 && seg_pos >= 1u) {
        v = plane[seg_pos - 1u];
        if (v == 0u || v == literal)
            return 0;
    } else {
        return 0;
    }

    want = _mm256_set1_epi8((char)v);
    while (p + 32u <= (uint32_t)nt) {
        __m256i cur = _mm256_loadu_si256(
            (const __m256i *)(const void *)(a + p));

        /* All 32 bytes equal: movemask of the compare is all ones. */
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(cur, want)) != -1)
            return 0;
        p += 32u;
    }
    while (p < (uint32_t)nt) {
        if (a[p] != v)
            return 0;
        p++;
    }
    return 1;
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* Dispatch helper, mirroring seg_prescreen: `have_avx2` is computed once
 * per segment encode. */
static int seg_all_hit_cheap_ok(const uint8_t *a, uint16_t nt, uint8_t pred,
                                uint8_t literal, uint32_t seg_pos,
                                const uint8_t *plane, int have_avx2)
{
#if ZPSE_PS_HAVE_AVX2
    if (have_avx2) {
        return seg_all_hit_cheap_avx2(a, nt, pred, literal, seg_pos, plane);
    }
#else
    (void)have_avx2;
#endif
    return seg_all_hit_cheap_scalar(a, nt, pred, literal, seg_pos, plane);
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

void zpse_segment_prescreen(const uint8_t *a, uint16_t nt,
                            const uint8_t *seg_base, uint32_t seg_pos,
                            uint8_t initial_literal, uint16_t ks[12])
{
    int have_avx2 = (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;

    seg_prescreen(a, nt, seg_base, seg_pos, initial_literal, have_avx2, ks);
}


/* 1 iff p[0..n) is one byte value, which is the only plane shape the dense
 * form pays on (see seg_dense_emit). Word-wise with an early exit, so a
 * plane that is not constant costs one compare. */
static int seg_plane_is_const(const uint8_t *p, uint32_t n)
{
    uint64_t splat = UINT64_C(0x0101010101010101) * (uint64_t)p[0];
    uint32_t i = 0u;

    while (i + 8u <= n) {
        uint64_t w;

        memcpy(&w, p + i, 8);
        if (w != splat) {
            return 0;
        }
        i += 8u;
    }
    while (i < n) {
        if (p[i] != p[0]) {
            return 0;
        }
        i++;
    }
    return 1;
}

/* Cheap size estimate for a transformed plane, from the prescreen alone.
 *
 * zpse_block_encode ranks transform candidates with it before spending a
 * classify + mode search + pack on each: per tile it takes the best
 * closed-form predictor's exact exception count k and prices the record at
 * 10 (length prefix + header) + k residual bytes + min(k, 32) mask bytes
 * (no mask at the two extremes). It ignores the bit-packed residual modes,
 * which shave a similar fraction off every candidate, so the ranking is
 * what matters, not the absolute number. */
uint32_t zpse_segment_proxy_cost(const uint8_t *plane, uint32_t plane_len)
{
    int have_avx2 = (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;
    uint32_t tc = (plane_len + 255u) / 256u;
    uint32_t cost = 16u;
    uint32_t sum = 0u;
    uint32_t t;
    uint8_t init;

    if (plane == 0 || plane_len == 0u)
        return UINT32_MAX;
    init = plane[0];
    /* A constant plane is emitted in the dense form (seg_dense_emit): twelve
     * bytes per 4 KiB instead of ten per tile. Pricing it as tiles would
     * overstate it by an order of magnitude, and this estimate is what
     * decides which candidates are worth a trial at all -- a candidate ten
     * times smaller than the estimate says would be pruned before it could
     * win. Non-constant planes stop at the first differing word. */
    if (seg_plane_is_const(plane, plane_len)) {
        uint32_t ch = (plane_len + ZPSE_DENSE_MAX_LEN - 1u) /
                      ZPSE_DENSE_MAX_LEN;

        return (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
               (uint32_t)ZPSE_DENSE_HEADER_SIZE * ch;
    }
    /* Every other tile is enough to rank candidates (the estimate is a
     * sum over many tiles and only its order is used), and it makes the
     * proxy a small fraction of one trial instead of a tenth of it. The
     * skipped tiles are priced at the sampled tiles' average. */
    for (t = 0; t < tc; t += 2u) {
        uint32_t seg_pos = t * 256u;
        uint16_t nt =
            (uint16_t)((t + 1u < tc) ? 256u : (plane_len - seg_pos));
        uint16_t ks[12];
        uint16_t bk;
        uint32_t i;

        seg_prescreen(plane + seg_pos, nt, plane, seg_pos, init, have_avx2,
                      ks);
        bk = ks[seg_screen_pred[0]];
        for (i = 1; i < 9u; i++) {
            uint16_t c = ks[seg_screen_pred[i]];
            if (c < bk)
                bk = c;
        }
        sum += 10u;
        if (bk != 0u) {
            sum += bk;
            if (bk != nt)
                sum += (bk < 32u) ? bk : 32u;
        }
    }
    return cost + sum * 2u;
}

/* Trial-level noise probe, called once per tile the general path writes.
 * `flat_covered` accumulates the length of the trial's LEADING run of tiles
 * that did not pay for themselves -- a tile whose record (the u16 length
 * prefix included) is not smaller than the bytes it codes. The run ends at
 * the first tile that does pay (`flat_paid`), and the probe is over for the
 * rest of the trial then: no later coverage can make a paying tile stop
 * having paid, so one compare per tile is all the probe costs from that
 * point on.
 *
 * A run of that shape reaching flat_abort_min means the trial has coded that
 * many input bytes into records at least as large as the bytes they code,
 * without one tile anywhere in the stretch paying for itself. On a block
 * whose every transform shape has to beat the RAW envelope, that is the
 * block being noise. Returns 1 when the caller must abandon the trial.
 *
 * The counters live in the encoder so one threshold spans the trial's
 * segments; the caller resets them per trial. flat_abort_min == 0 disables
 * the test entirely (that is what the direct tile/segment callers pass). */
static int seg_flat_abort(ZpseEncoder *e, uint16_t nt, uint32_t coded,
                          uint32_t flat_abort_min)
{
    if (flat_abort_min == 0u || e->flat_paid != 0u)
        return 0;
    if (coded < (uint32_t)nt) {
        e->flat_paid = 1u;
        return 0;
    }
    e->flat_covered += nt;
    return (e->flat_covered >= flat_abort_min) ? 1 : 0;
}

/* Dense record form for a constant plane (SPEC Sec. 6.3).
 *
 * An all-hit tile costs ten bytes (two of length prefix, eight of tile
 * header) and a dense record costs twelve for 4 KiB, so a plane that is a
 * single byte value everywhere is the one shape this form beats the tile
 * path on: 248 bytes instead of 2616 for a 64 KiB block of zeroes. A
 * dense record's K == 0 reconstruction is a fill -- every predictor's
 * fixed point under a record's entry state is one value -- which is
 * precisely the constant plane, so no other plane is worth the form in
 * v1 and none is offered it.
 *
 * Returns the payload length written, or 0 when the plane is not constant,
 * when the form would not be strictly smaller than `pos` bytes of tile
 * records, or when it does not fit `byte_budget` / `payload_cap`. The
 * constant test is word-wise with an early exit, so a plane that is not
 * constant costs one compare per segment and the tile path is otherwise
 * untouched; the budget-exhausted paths call it too, because a trial can
 * reach a size the tile path cannot encode but these records can. */
static int seg_dense_emit(ZpseSegmentHeader *h, uint8_t *payload_out,
                          const uint8_t *plane, uint32_t plane_len,
                          uint8_t initial_literal, uint32_t pos,
                          uint32_t byte_budget, uint32_t payload_cap,
                          int known_const)
{
    uint32_t chunks = (plane_len + ZPSE_DENSE_MAX_LEN - 1u) /
                      ZPSE_DENSE_MAX_LEN;
    uint32_t dtotal = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
                      chunks * (uint32_t)ZPSE_DENSE_HEADER_SIZE;
    uint8_t v = plane[0];
    uint8_t pred;
    uint8_t *q;
    uint32_t c;

    if (dtotal >= pos || dtotal > payload_cap) {
        return 0;
    }
    if (byte_budget != 0u && dtotal > byte_budget) {
        return 0;
    }
    if (!known_const && !seg_plane_is_const(plane, plane_len)) {
        return 0;
    }

    /* STRIDE_1's fixed point is the segment's initial literal, so it needs
     * no constant byte; CONST_ZERO needs none either; anything else needs
     * TILE_CONSTANT and its byte. */
    pred = (v == 0u) ? (uint8_t)ZPSE_P_CONST_ZERO
           : (v == initial_literal) ? (uint8_t)ZPSE_P_STRIDE_1
                                    : (uint8_t)ZPSE_P_TILE_CONSTANT;
    q = payload_out + ZPSE_SEGMENT_HEADER_SIZE;
    for (c = 0u; c < chunks; c++) {
        zpse_store_u16le(q, (uint16_t)(ZPSE_DENSE_HEADER_SIZE - 2u));
        q[2] = (uint8_t)(ZPSE_DENSE_MARKER |
                         ((uint8_t)ZPSE_R_EXCLUSION_U8 << 3));
        q[3] = pred;
        q[4] = (pred == (uint8_t)ZPSE_P_TILE_CONSTANT) ? v : 0u;
        q[5] = 0u;
        zpse_store_u16le(q + 6u, 0u);
        zpse_store_u16le(q + 8u, 0u);
        zpse_store_u16le(q + 10u, 0u);
        q += ZPSE_DENSE_HEADER_SIZE;
    }
    /* A dense segment always reports tile_count == 1 (Sec. 6.3). */
    h->tile_count = 1u;
    zpse_segment_header_serialize(payload_out, h);
    return (int)dtotal;
}

/* How much of the plane the scalar gather below reads before the vector
 * test takes over. It only has to be long enough to hold the whole
 * alphabet of a plane the dense palette form can use (at most sixteen
 * values); every later occurrence of a value costs a vector compare
 * instead of a read-modify-write of the set. */
#define ZPSE_ALPHA_HEAD 1024u

#if ZPSE_PS_HAVE_AVX2
/* First position in p[0, n) whose byte is NOT in the set `bits`, or n when
 * every byte is.
 *
 * The set is turned into two sixteen-entry tables indexed by the LOW
 * nibble of a byte: table A holds, for each low nibble, the eight high
 * nibbles 0..7 that occur with it, and table B the same for 8..15. A byte
 * is therefore a member exactly when bit (v >> 4) & 7 of A[v & 15] (or of
 * B, for a high nibble above 7) is set, which is three pshufb lookups and
 * a masked select for 32 bytes at a time -- no per-value broadcast, so the
 * cost does not grow with the alphabet. This is what replaces the scalar
 * bitmap probe: that probe is a load-modify-store of the same eight-word
 * set per byte, i.e. a dependent chain through L1, and it accounted for
 * most of the 46 us the palette form's alphabet scan cost on 64 KiB of
 * four-bit data.
 *
 * The answer is the same the scalar probe gives -- membership of a byte in
 * a set of at most sixteen values -- so the caller can hand the offset
 * back to the scalar loop and get the identical alphabet and count. */
static ZPSE_PS_AVX2 uint32_t seg_alphabet_first_outside_avx2(
    const uint8_t *p, uint32_t n, const uint32_t bits[8])
{
    uint8_t ta[16];
    uint8_t tb[16];
    const __m256i nib = _mm256_set1_epi8(0x0F);
    const __m256i hi7 = _mm256_set1_epi8(7);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i onehot = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80);
    __m256i tav;
    __m256i tbv;
    uint32_t i = 0u;
    unsigned v;

    memset(ta, 0, sizeof ta);
    memset(tb, 0, sizeof tb);
    for (v = 0u; v < 256u; v++) {
        if (((bits[v >> 5] >> (v & 31u)) & 1u) == 0u) {
            continue;
        }
        if ((v >> 4) < 8u) {
            ta[v & 15u] |= (uint8_t)(1u << (v >> 4));
        } else {
            tb[v & 15u] |= (uint8_t)(1u << ((v >> 4) - 8u));
        }
    }
    tav = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(const void *)ta));
    tbv = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((const __m128i *)(const void *)tb));

    while (i + 32u <= n) {
        const __m256i cur = _mm256_loadu_si256(
            (const __m256i *)(const void *)(p + i));
        const __m256i lo = _mm256_and_si256(cur, nib);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(cur, 4), nib);
        const __m256i sel = _mm256_cmpgt_epi8(hi, hi7);
        const __m256i mask = _mm256_or_si256(
            _mm256_and_si256(sel, _mm256_shuffle_epi8(tbv, lo)),
            _mm256_andnot_si256(sel, _mm256_shuffle_epi8(tav, lo)));
        const __m256i bit = _mm256_shuffle_epi8(
            onehot, _mm256_and_si256(hi, hi7));
        const unsigned miss = (unsigned)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_and_si256(mask, bit), zero));

        if (miss != 0u) {
            /* movemask sets bit k for the k-th byte, so the lowest set bit
             * is the first byte outside the set. */
            unsigned k = 0u;

            while (((miss >> k) & 1u) == 0u) {
                k++;
            }
            return i + k;
        }
        i += 32u;
    }
    for (; i < n; i++) {
        unsigned c = p[i];

        if (((bits[c >> 5] >> (c & 31u)) & 1u) == 0u) {
            return i;
        }
    }
    return n;
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* Which byte values the plane actually contains, as a 256-bit set.
 * Stops as soon as the alphabet exceeds `limit`, because every caller only
 * asks whether it is small: a text, code or noise plane bails out after 17
 * distinct values instead of scanning 64 KiB.
 *
 * The gather is a scalar bitmap probe, which is a dependent read-modify-
 * write per byte. On a plane that really does have a small alphabet that
 * probe runs over every byte of the plane, and on 64 KiB of four-bit data
 * it was measured at 46 us per block -- more than the palette form's own
 * pack. So the first ZPSE_ALPHA_HEAD bytes are gathered scalar (they hold
 * a palette plane's whole alphabet many times over) and the rest of the
 * plane is only TESTED for membership in what the head found, with one
 * vector pass whose cost does not depend on the alphabet size. A plane
 * with more than sixteen values still leaves the gather the first time the
 * test finds one, so the early exit that keeps text and noise cheap is
 * unchanged. */
static uint32_t seg_plane_alphabet(const uint8_t *plane, uint32_t plane_len,
                                   uint32_t limit, uint32_t bits[8])
{
    uint32_t n = 0u;
    uint32_t i;
    uint32_t head = (plane_len < ZPSE_ALPHA_HEAD) ? plane_len
                                                  : ZPSE_ALPHA_HEAD;

    memset(bits, 0, 8u * sizeof bits[0]);
    for (i = 0u; i < head; i++) {
        unsigned v = plane[i];
        uint32_t bit = 1u << (v & 31u);

        if ((bits[v >> 5] & bit) != 0u) {
            continue;
        }
        bits[v >> 5] |= bit;
        n++;
        if (n > limit) {
            return n;
        }
    }
#if ZPSE_PS_HAVE_AVX2
    /* The membership tables hold at most sixteen values, so the vector test
     * is only taken when the head found no more than that -- which is the
     * case every caller cares about (`limit` is 16 at both call sites) and
     * the case where the scalar probe would otherwise walk the whole
     * plane. */
    if (n <= 16u && head < plane_len &&
        (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        uint32_t q = seg_alphabet_first_outside_avx2(plane + head,
                                                     plane_len - head, bits);

        if (q == plane_len - head) {
            return n;
        }
        i = head + q;
    }
#endif
    for (; i < plane_len; i++) {
        unsigned v = plane[i];
        uint32_t bit = 1u << (v & 31u);

        if ((bits[v >> 5] & bit) != 0u) {
            continue;
        }
        bits[v >> 5] |= bit;
        n++;
        if (n > limit) {
            return n;
        }
    }
    return n;
}

/* Size of the dense palette form for a plane of `m` distinct values
 * (SPEC Sec. 6.3): one record per ZPSE_DENSE_MAX_LEN bytes, each a 12-byte
 * header, a palette of m and ceil(ns * ceil(log2 m) / 8) packed indices.
 *
 * The palette is sized by the plane's whole alphabet, not by what each
 * chunk happens to contain, so this is an upper bound on what the encoder
 * writes -- the packer re-derives m per record and can only be smaller.
 * An upper bound is the safe direction: it decides whether the form beats
 * the tiles it would replace. */
static uint32_t seg_dense_palette_cost(uint32_t plane_len, uint32_t m)
{
    uint32_t b = zpse_ceil_log2_u32(m);
    uint32_t total = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;
    uint32_t off;

    for (off = 0u; off < plane_len; off += ZPSE_DENSE_MAX_LEN) {
        uint32_t ns = plane_len - off;

        if (ns > ZPSE_DENSE_MAX_LEN) {
            ns = ZPSE_DENSE_MAX_LEN;
        }
        total += (uint32_t)ZPSE_DENSE_HEADER_SIZE + 1u + m +
                 ((ns * b + 7u) / 8u);
    }
    return total;
}

/* Dense palette record form (SPEC Sec. 6.3) for a plane of at most
 * sixteen distinct byte values.
 *
 * This is the shape a 4-bit image has before any transform and the shape
 * an entropy-coded plane has after one: every byte is one of a handful of
 * symbols. The tile path spends ten bytes of framing per 256 bytes on it
 * and then a prediction step per byte, because each tile must pick a
 * predictor, which for such a plane is close to noise; the dense form
 * spends one 12-byte header per 4 KiB and codes the whole plane as a
 * palette plus indices, so it is both smaller and, at decode, one palette
 * expansion for 4 KiB instead of a predictor chain per tile (dense.c).
 *
 * The record uses predictor TILE_CONSTANT, mask ALL_EXCEPTION and residual
 * PALETTE_BITS. ALL_EXCEPTION makes every position an exception, so the
 * record codes every byte and the decoder runs no prediction at all; the
 * constant predictor only has to supply the value exceptions are measured
 * against, and that value is chosen to be one the plane does NOT contain
 * -- otherwise the exclusion mapping would have no residual for the bytes
 * equal to it. 255 is preferred when absent because the exclusion residual
 * of every value below it is the value itself, which makes the decoder's
 * inverse a no-op.
 *
 * Returns the payload length written, or 0 when the plane's alphabet is not
 * small enough, when the form is not strictly smaller than the `pos` bytes
 * of tile records already written (and `replace` is 0), or when it does not
 * fit payload_cap / byte_budget.
 *
 * `replace` is the caller saying it has already decided the form wins and
 * the tile records written so far are to be abandoned: the `pos` test is
 * then not a comparison but a formality, because the only reason `pos` can
 * still be below the form's size is that the caller stopped the tile loop
 * early to save the work (see the estimate in zpse_segment_encode). */
static int seg_dense_palette_emit(ZpseSegmentHeader *h, uint8_t *payload_out,
                                  const uint8_t *plane, uint32_t plane_len,
                                  uint32_t m, const uint32_t bits[8],
                                  uint32_t pos, int replace,
                                  uint32_t byte_budget,
                                  uint32_t payload_cap)
{
    uint8_t buf[ZPSE_DENSE_MAX_LEN];
    uint32_t seen2[8];
    uint32_t nu2;
    uint32_t cost;
    uint32_t fill;
    uint32_t off;
    uint32_t at;

    if (m < 2u || m > 16u) {
        return 0;
    }
    cost = seg_dense_palette_cost(plane_len, m);
    if ((cost >= pos && replace == 0) || cost > payload_cap) {
        return 0;
    }
    if (byte_budget != 0u && cost > byte_budget) {
        return 0;
    }

    fill = 255u;
    while (fill > 0u && ((bits[fill >> 5] >> (fill & 31u)) & 1u) != 0u) {
        fill--;
    }

    /* The residual alphabet every record can draw from, i.e. the plane's
     * alphabet under the exclusion map for `fill` (a function of one byte,
     * so distinct values can only merge, never appear). It is a
     * certificate for the packer, which would otherwise scan all 4096
     * bytes of every record to discover a palette it can see in the first
     * few dozen; on 64 KiB of four-bit data that scan was measured at 30 us
     * of the block's encode. It cannot change a byte: the packer takes the
     * short path only when the prefix it scanned already holds every value
     * of the certificate, at which point the palette order and size are the
     * ones the exhaustive scan produces. */
    memset(seen2, 0, sizeof seen2);
    nu2 = 0u;
    {
        uint32_t v;

        for (v = 0u; v < 256u; v++) {
            uint8_t r;
            uint32_t bit;

            if (((bits[v >> 5] >> (v & 31u)) & 1u) == 0u) {
                continue;
            }
            r = zpse_exclusion_forward((uint8_t)v, (uint8_t)fill);
            bit = 1u << ((uint32_t)r & 31u);
            if ((seen2[(uint32_t)r >> 5] & bit) != 0u) {
                continue;
            }
            seen2[(uint32_t)r >> 5] |= bit;
            nu2++;
        }
    }

    at = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;
    for (off = 0u; off < plane_len; off += ZPSE_DENSE_MAX_LEN) {
        uint32_t ns = plane_len - off;
        uint32_t i;
        uint32_t room;
        const uint8_t *exc;
        int rl;

        if (ns > ZPSE_DENSE_MAX_LEN) {
            ns = ZPSE_DENSE_MAX_LEN;
        }
        /* With fill == 255 the exclusion residual of every byte the plane
         * can hold is the byte itself, so the packer reads the plane in
         * place; any other fill needs the mapped copy. */
        if (fill == 255u) {
            exc = plane + off;
        } else {
            for (i = 0u; i < ns; i++) {
                buf[i] = zpse_exclusion_forward(plane[off + i],
                                                (uint8_t)fill);
            }
            exc = buf;
        }
        room = payload_cap - (at + (uint32_t)ZPSE_DENSE_HEADER_SIZE);
        if (room > 65535u) {
            room = 65535u;
        }
        rl = zpse_residual_encode_hint((uint8_t)ZPSE_R_PALETTE_BITS, exc, 0, 0,
                                       (uint16_t)ns,
                                       payload_out + at +
                                           (uint32_t)ZPSE_DENSE_HEADER_SIZE,
                                       (uint16_t)room, (uint16_t)nu2,
                                       seen2);
        if (rl < 0) {
            /* The size bound above said this fits; the packer rejecting a
             * residual value would mean the exclusion map produced 255,
             * which a fill the plane does not contain cannot do. Leave the
             * tile records in place rather than guessing. */
            return 0;
        }
        zpse_store_u16le(payload_out + at,
                         (uint16_t)(10u + (uint32_t)rl));
        payload_out[at + 2u] = (uint8_t)(ZPSE_DENSE_MARKER |
                                         (uint8_t)ZPSE_M_ALL_EXCEPTION |
                                         ((uint8_t)ZPSE_R_PALETTE_BITS << 3));
        payload_out[at + 3u] = (uint8_t)ZPSE_P_TILE_CONSTANT;
        payload_out[at + 4u] = (uint8_t)fill;
        payload_out[at + 5u] = 0u;
        zpse_store_u16le(payload_out + at + 6u, (uint16_t)ns);
        zpse_store_u16le(payload_out + at + 8u, 0u);
        zpse_store_u16le(payload_out + at + 10u, (uint16_t)rl);
        /* 2 bytes of rec_len, then the rest of the header (10) and the
         * residual payload -- what rec_len counts starts after itself. */
        at += 2u + 10u + (uint32_t)rl;
    }

    /* A dense segment always reports tile_count == 1 (Sec. 6.3). */
    h->tile_count = 1u;
    zpse_segment_header_serialize(payload_out, h);
    return (int)at;
}

/* Tiles a palette decision prices before deciding, and the plane size in
 * tiles below which it does not price at all. See seg_tile_estimate. */
#define ZPSE_EST_TILES 8u
#define ZPSE_EST_MIN_TILES 16u

/* Estimated size of the tile records a plane would take, from a sample of
 * ZPSE_EST_TILES tiles spread evenly over it.
 *
 * This exists because the two forms are priced in very different ways. The
 * palette form's price is exact and costs one alphabet scan. The tiles'
 * price is exact only after every tile has been classified, mode-searched
 * and packed -- which is the whole trial. On 64 KiB of four-bit data the
 * tiles lose by a quarter, and the exact comparison finds that out by
 * writing 190 of them and then replacing all 190 with the palette: measured
 * 110 us of that block's 114 us of encode. The sample is eight tiles
 * instead. On a plane whose tiles are alike (what a four-bit plane is) the
 * per-tile average is within a percent, and on a plane where the palette
 * loses the sample costs under 4 us -- noise against a trial.
 *
 * Each sampled tile goes through exactly the pipeline the trial loop runs,
 * INCLUDING the second candidate when it is competitive, so this is the
 * trial's own per-tile size rather than a proxy for it. Returns 0 (the
 * caller then keeps the exact comparison) when a tile cannot be priced. */
static int seg_tile_estimate(ZpseEncoder *e, ZpseLaneState *lanes,
                             const uint8_t *plane, uint32_t plane_len,
                             uint32_t tc, uint8_t initial_literal,
                             int have_avx2, uint32_t *out)
{
    uint32_t step = tc / ZPSE_EST_TILES;
    uint32_t sum = 0u;
    uint32_t nsum = 0u;
    uint32_t t;

    if (step == 0u) {
        step = 1u;
    }
    for (t = 0u; t < tc; t += step) {
        uint32_t seg_pos = t * 256u;
        uint16_t nt =
            (uint16_t)((t + 1u < tc) ? 256u : (plane_len - seg_pos));
        const uint8_t *actual = plane + seg_pos;
        uint16_t ks[12];
        uint16_t best_k;
        uint8_t best_p;
        uint8_t cand[2];
        uint32_t ncand = 0u;
        uint32_t i;
        uint32_t ci;
        int tile_len = -1;

        seg_prescreen(actual, nt, plane, seg_pos, initial_literal, have_avx2,
                      ks);
        best_p = seg_screen_pred[0];
        best_k = ks[best_p];
        for (i = 1u; i < 9u; i++) {
            uint16_t c = ks[seg_screen_pred[i]];

            if (c < best_k) {
                best_k = c;
                best_p = seg_screen_pred[i];
            }
        }
        if (best_k == 0u) {
            /* The all-hit run above writes these directly: a 2-byte length
             * prefix and the 8-byte record. */
            sum += 10u;
            nsum++;
            continue;
        }
        cand[ncand++] = best_p;
        {
            uint8_t second_p = 0xFFu;
            uint16_t sk = 0xFFFFu;

            for (i = 0u; i < 9u; i++) {
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
                (uint32_t)sk <= (uint32_t)best_k + 4u &&
                (uint32_t)best_k * 8u < (uint32_t)nt * 7u) {
                cand[ncand++] = second_p;
            }
        }
        for (ci = 0u; ci < ncand; ci++) {
            uint8_t pred = cand[ci];
            ZpseTileClass cl;
            uint8_t mm;
            uint8_t rr;
            int n;

            if (zpse_tile_classify(lanes, actual, nt, pred, initial_literal, 1,
                                   plane, seg_pos, &cl) != PSE_OK) {
                continue;
            }
            zpse_tile_choose_modes(&cl, nt, &mm, &rr);
            n = zpse_tile_pack(pred, &cl, nt, actual, mm, rr, 1, e->hist,
                               4096u, e->comb);
            if (n < 8) {
                mm = (cl.k == 0u) ? 0u
                     : (cl.k == nt) ? 6u
                     : ((uint32_t)cl.k <= (((uint32_t)nt + 7u) >> 3) ? 2u
                                                                     : 1u);
                n = zpse_tile_pack(pred, &cl, nt, actual, mm, 0u, 1, e->hist,
                                   4096u, e->comb);
            }
            if (n < 8) {
                continue;
            }
            if (tile_len < 0 || n < tile_len) {
                tile_len = n;
            }
        }
        if (tile_len < 0) {
            return 0;
        }
        sum += 2u + (uint32_t)tile_len;
        nsum++;
    }
    if (nsum == 0u) {
        return 0;
    }
    *out = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
           (uint32_t)(((uint64_t)sum * tc + nsum / 2u) / nsum);
    return 1;
}

/* Exit helper for the paths that run out of room: the dense form may still
 * fit where the tile records no longer do, so it is offered before the
 * overflow is reported. Nothing is known about the plane here, so the
 * constant test runs (it is a word-wise scan with an early exit). Returns a
 * payload length or PSE_OUTPUT_OVERFLOW. */
static int seg_no_room(ZpseSegmentHeader *h, uint8_t *payload_out,
                       const uint8_t *plane, uint32_t plane_len,
                       uint8_t initial_literal, uint32_t pos,
                       uint32_t byte_budget, uint32_t payload_cap)
{
    int dn = seg_dense_emit(h, payload_out, plane, plane_len, initial_literal,
                            pos, byte_budget, payload_cap, 0);

    return dn != 0 ? dn : PSE_OUTPUT_OVERFLOW;
}

int zpse_segment_encode(ZpseEncoder *e, ZpseLaneState *lanes,
                        const uint8_t *plane, uint32_t plane_len,
                        uint8_t initial_literal,
                        uint8_t *payload_out, uint32_t payload_cap,
                        uint32_t byte_budget, uint32_t flat_abort_min)
{
    ZpseSegmentHeader h;
    uint32_t tc;
    uint32_t pos;
    uint32_t t;
    uint8_t *tmp;
    uint32_t tmp_cap;
    int have_avx2 = (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;
    uint8_t dense_v;
    int dense_ok;
    uint32_t pal_len;
    uint32_t pal_m;
    int pal_try;
    uint32_t pal_bits[8];

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
     * (simplest legal choice, keeps every tile independent). Only the
     * histories need establishing here: every tile below is classified
     * with tile_reset == 1, so zpse_tile_classify calls
     * zpse_lane_reset_for for it (histories + the table its predictor
     * reads), and no predictor this encoder ever classifies -- the nine
     * in seg_screen_pred[] -- reads a table at all. The 66 KiB table
     * memset zpse_lane_reset would do for ORDER1/ORDER2 here was for a
     * classification this encoder never asks for. */
    zpse_lane_reset_hist(lanes, initial_literal);

    /* Constant-plane tracking, which lets the dense form (SPEC Sec. 6.3)
     * skip its own scan: a tile the all-hit run writes is constant -- k ==
     * 0 forces prediction == actual at every position, and under a reset
     * tile every predictor's fixed point is one value -- so one byte per
     * tile decides whether the whole plane is that same value, and the
     * general path, which can code anything, clears the flag. The tile
     * path is otherwise untouched: the records it writes are replaced at
     * the end when the dense form is smaller. */
    dense_v = plane[0];
    dense_ok = 1;

    /* Dense palette form (SPEC Sec. 6.3): a plane whose alphabet is small
     * is cheaper as one palette stream than as one predictor search per
     * tile -- both in bytes and, at decode, in work (segment.c's
     * seg_dense_palette_emit). Pricing it means scanning the plane, so it
     * is only priced when the tile path is *losing*: the first tile is
     * probed for its alphabet before the loop (a plane's alphabet contains
     * that tile's, so 256 bytes with more than sixteen values settles it),
     * and the whole plane is scanned only when the first tile turns out to
     * need a predictor and a residual at all. A plane of all-hit tiles --
     * every periodic and every near-constant plane -- therefore never pays
     * for the scan, and keeps the tile coding it has always had, which is
     * far smaller than a palette. pal_len is an upper bound on the bytes
     * the form will write (the packer re-derives the palette per record),
     * so stopping the loop when pos reaches it can only leave the smaller
     * form. */
    pal_try = (seg_plane_alphabet(plane,
                                  (plane_len < 256u) ? plane_len : 256u, 16u,
                                  pal_bits) <= 16u)
                  ? 1
                  : 0;
    pal_len = 0u;
    pal_m = 0u;

    pos = 16u;
    for (t = 0; t < tc; t++) {
        uint32_t seg_pos = t * 256u;
        uint16_t nt =
            (uint16_t)((t + 1u < tc) ? 256u : (plane_len - seg_pos));
        const uint8_t *actual = plane + seg_pos;

        if (pal_len != 0u && pos >= pal_len) {                int dn = seg_dense_palette_emit(&h, payload_out, plane, plane_len,
                                                pal_m, pal_bits, pos, 0,
                                                byte_budget, payload_cap);

            if (dn != 0) {
                return dn;
            }
            pal_len = 0u; /* not emittable after all: keep tiling */
        }
        uint16_t ks[12];
        uint8_t cand[3];
        uint32_t ncand = 0;
        uint32_t ci;
        int best_len = -1;
        uint16_t best_k = 0u;
        uint8_t best_p = 0;
        uint8_t *bst = e->input; /* staging for the winning record */
        int n;
        int direct;    /* packing the single candidate in place */
        uint8_t *wout; /* where the candidate's record is packed */
        uint32_t wcap; /* and the capacity it is packed with */

        /* Bound check for the byte budget, before any work for this
         * tile: `pos` grows monotonically, so once it is past the budget
         * nothing later could come back under it. No exception path can
         * be skipped by this -- the caller only passes a budget when a
         * total that large cannot win -- and no tile is half-written. */
        if (byte_budget != 0u && pos > byte_budget) {
            return seg_no_room(&h, payload_out, plane, plane_len,
                               initial_literal, pos, byte_budget,
                               payload_cap);
        }

        seg_prescreen(actual, nt, plane, seg_pos, initial_literal,
                      have_avx2, ks);
        best_p = 0;
        {
            uint16_t bk = ks[seg_screen_pred[0]];
            uint32_t i;
            for (i = 1; i < 9u; i++) {
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
                for (i = 0; i < 9u; i++) {
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
         * keeps reporting k == 0 for the same winner. For the three cases
         * seg_all_hit_cheap_ok can settle from the tile's own bytes that
         * answer is already known and the prescreen is skipped
         * altogether; for everything else it is asked, exactly as before.
         * Any other tile falls out to the general path, which re-derives
         * everything from the same prescreen result -- each tile is
         * written by exactly one of the two paths, and the bytes agree,
         * so the payload is unchanged. The lane state needs no update: the
         * classify call for the next tile is always made with
         * tile_reset == 1, which resets the histories and tables itself,
         * and nothing after this loop reads `lanes`.
         *
         * The run stops (rather than overflowing) on the same capacity
         * condition the general path returns PSE_OUTPUT_OVERFLOW for. */
        if (best_k == 0u) {
            uint8_t pred = best_p;

            if (payload_cap - pos < 10u) {
                return seg_no_room(&h, payload_out, plane, plane_len,
                                   initial_literal, pos, byte_budget,
                                   payload_cap);
            }
            /* The probe is not consulted on this path. A run writes 10 bytes
             * per tile, which pays for itself unless the tile is 10 bytes or
             * shorter -- and only a segment's LAST tile can be that short,
             * against a threshold of a sixteenth of a block of at least
             * 8 KiB. So every run tile in the stretch the probe measures
             * pays, the leading non-paying run stays empty here, and the
             * abort could not fire even if it were consulted. All-hit runs
             * are the hot path for constant and periodic blocks, so they pay
             * nothing for a test that cannot fire -- and a block whose front
             * is a run of predicting tiles is exactly a block this probe is
             * not looking for. */
            seg_write_all_hit(payload_out + pos, pred, nt);
            pos += 10u;

            while (t + 1u < tc) {
                uint32_t nsseg_pos = (t + 1u) * 256u;
                uint16_t nnt = (uint16_t)((t + 2u < tc)
                                              ? 256u
                                              : (plane_len - nsseg_pos));
                uint16_t nks[12];
                uint8_t npred = pred;

                if (byte_budget != 0u && pos > byte_budget) {
                    return seg_no_room(&h, payload_out, plane, plane_len,
                                       initial_literal, pos, byte_budget,
                                       payload_cap);
                }
                if (payload_cap - pos < 10u) {
                    return seg_no_room(&h, payload_out, plane, plane_len,
                                       initial_literal, pos, byte_budget,
                                       payload_cap);
                }
                if (plane[nsseg_pos] != dense_v) {
                    dense_ok = 0;
                }
                /* The tile keeps the run's predictor and its k == 0 only
                 * when the cheap proof holds; otherwise the prescreen is
                 * authoritative, exactly as it was before. When the cheap
                 * proof holds the winner is unchanged, so npred stays
                 * `pred` -- which is what the prescreen branch would also
                 * have concluded, since no other predictor can have a
                 * lower (negative) count and the scan order cannot
                 * promote one that precedes it. */
                if (!seg_all_hit_cheap_ok(plane + nsseg_pos, nnt, pred,
                                          initial_literal, nsseg_pos, plane,
                                          have_avx2)) {
                    uint16_t nbk;
                    uint32_t i;

                    npred = seg_screen_pred[0];
                    seg_prescreen(plane + nsseg_pos, nnt, plane, nsseg_pos,
                                  initial_literal, have_avx2, nks);
                    nbk = nks[npred];
                    for (i = 1; i < 9u; i++) {
                        uint16_t c = nks[seg_screen_pred[i]];
                        if (c < nbk) {
                            nbk = c;
                            npred = seg_screen_pred[i];
                        }
                    }
                    if (nbk != 0u || npred != pred) {
                        break;
                    }
                }
                seg_write_all_hit(payload_out + pos, npred, nnt);
                pos += 10u;
                t++;
            }
            continue;
        }

        /* A tile that needs a mask or a residual is not an all-hit tile, so
         * the dense *constant* form cannot describe this segment. The dense
         * palette form still can: this is the first moment the encoder knows
         * the plane is dense rather than periodic, so this is where the form
         * is priced (once) -- and, if the tiles already written are bigger
         * than it, where they are replaced by it. */
        dense_ok = 0;

        /* One candidate and room for the worst record: pack it straight
         * into the output instead of into the 4 KiB scratch and copying
         * the winner back out of it. That is the dense-tile case (every
         * tile of a text, code or noise block) and the single-candidate
         * case generally, and it removes a per-tile read-modify-write of
         * up to 4096 bytes plus the copy.
         *
         * `direct` is only taken when the room that is left is at least
         * the scratch capacity, so the packer is handed a capacity it
         * cannot exhaust and returns exactly what it would have returned
         * for `tmp`; the overflow decision stays where it was, on the
         * final record size below. With two candidates the second one
         * still needs somewhere to be packed before the sizes are
         * compared, so that case keeps the staging buffer. */
        direct = (ncand == 1u && payload_cap - pos - 2u >= tmp_cap) ? 1 : 0;
        wout = direct ? (payload_out + pos + 2u) : tmp;
        wcap = direct ? (payload_cap - pos - 2u) : tmp_cap;

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
            n = zpse_tile_pack(pred, &cl, nt, actual, mm, rr, 1, wout,
                               wcap, e->comb);
            if (n < 8) {
                /* Defensive fallback: the always-legal minimal pair. */
                mm = (cl.k == 0u) ? 0u
                     : (cl.k == nt) ? 6u
                     : ((uint32_t)cl.k <= (((uint32_t)nt + 7u) >> 3) ? 2u
                                                                     : 1u);
                n = zpse_tile_pack(pred, &cl, nt, actual, mm, 0u, 1, wout,
                                   wcap, e->comb);
                rr = 0u;
            }
            if (n < 8) {
                continue;
            }
            if (best_len < 0 || n < best_len) {
                if (!direct) {
                    memcpy(bst, tmp, (size_t)n);
                }
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
        }        (void)best_p;

        if (best_len < 0) {
            return PSE_BAD_RESIDUAL;
        }
        if (pos + 2u > payload_cap ||
            (uint32_t)best_len > payload_cap - pos - 2u) {
            return seg_no_room(&h, payload_out, plane, plane_len,
                               initial_literal, pos, byte_budget,
                               payload_cap);
        }
        if (!direct) {
            memcpy(payload_out + pos + 2u, bst, (size_t)best_len);
        }
        zpse_store_u16le(payload_out + pos, (uint16_t)best_len);
        pos += 2u + (uint32_t)best_len;

        /* The palette form (SPEC Sec. 6.3) is priced from the first tile
         * that needs a mask or a residual -- the moment the plane is known
         * to be dense rather than periodic -- and only when that tile is
         * losing: a record that fits in an eighth of its own bytes pays for
         * itself many times over, and a plane of those (a sparse flag
         * bitmap, a slow ramp) is not worth the plane-wide scan the pricing
         * needs. An eighth still admits a random one-bit plane, whose tiles
         * cost about a quarter of their bytes. Once priced, the loop above
         * the tiles stops the moment the tiles can no longer win. */
        if (pal_try != 0 &&
            2u + (uint32_t)best_len >= (uint32_t)(nt >> 3)) {
            int dn;

            pal_try = 0;
            pal_m = seg_plane_alphabet(plane, plane_len, 16u, pal_bits);
            if (pal_m >= 2u && pal_m <= 16u) {
                uint32_t pc = seg_dense_palette_cost(plane_len, pal_m);

                if (pc <= payload_cap &&
                    (byte_budget == 0u || pc <= byte_budget)) {
                    pal_len = pc;
                }
            }
            if (pal_len != 0u && pos >= pal_len) {
                dn = seg_dense_palette_emit(&h, payload_out, plane, plane_len,
                                            pal_m, pal_bits, pos, 0,
                                            byte_budget, payload_cap);
                if (dn != 0) {
                    return dn;
                }
                pal_len = 0u;
            }
            /* The form is priced and the tiles have not caught up with it.
             * Ask the sample whether they are going to: if the palette is
             * smaller than the tiles by a margin, the rest of the loop is
             * work whose bytes the form will throw away, so emit it now
             * and return. The margin (an eighth) is what keeps a mispriced
             * plane -- tiles that differ region by region -- from losing
             * more bytes than the sample saved; below it the exact rule
             * still runs the tiles and compares, as it always did. */
            if (pal_len != 0u && pos < pal_len && tc >= ZPSE_EST_MIN_TILES) {
                uint32_t est;

                if (seg_tile_estimate(e, lanes, plane, plane_len, tc,
                                      initial_literal, have_avx2, &est) &&
                    est - (est >> 3) >= pal_len) {
                    dn = seg_dense_palette_emit(&h, payload_out, plane,
                                                plane_len, pal_m, pal_bits,
                                                pos, 1, byte_budget,
                                                payload_cap);
                    if (dn != 0) {
                        return dn;
                    }
                }
            }
        }
        if (seg_flat_abort(e, nt, 2u + (uint32_t)best_len, flat_abort_min)) {
            return PSE_OUTPUT_OVERFLOW;
        }
    }

    /* The plane may be one byte value end to end, which the dense record
     * form (SPEC Sec. 6.3) codes in 12 bytes per 4 KiB instead of the ten
     * bytes per 256-byte tile the run path above just wrote. The rewrite
     * only ever shrinks the record area, so every bound checked above still
     * holds. Offered first so a constant plane keeps the form it has
     * always had. */
    {
        int dn = seg_dense_emit(&h, payload_out, plane, plane_len,
                                initial_literal, pos, byte_budget,
                                payload_cap, dense_ok);        if (dn != 0) {
            return dn;
        }
    }

    /* The loop above stops early once the palette form is the smaller of
     * the two; this is the case where it finished and the form still has
     * to be compared with every tile record written. Offered after the
     * constant form so a constant plane keeps the form it has always had
     * -- it is also the smaller of the two for a constant. */
    if (pal_len != 0u) {
        int dn = seg_dense_palette_emit(&h, payload_out, plane, plane_len,
                                        pal_m, pal_bits, pos, 0, byte_budget,
                                        payload_cap);

        if (dn != 0) {
            return dn;
        }
    }
    return (int)pos;
}
