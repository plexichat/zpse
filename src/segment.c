#include "zpse/segment.h"

#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#endif

#include "zpse/bits.h"
#include "zpse/dense.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"
#include "zpse/tile.h"
#include "zpse/types.h"
#include "zpse/vector.h"

/* The ASCII-pack tile record in its one shape (tile.h): an 8-byte tile
 * header, no mask payload, and 224 packed bytes for a 256-byte tile, with
 * mask mode 7, residual mode 0 and the per-tile reset bit -- the bytes the
 * packer writes and nothing else. The segment decoder's run shortcut
 * tests these four header bytes and the record length, and hands every
 * other record to the general path. */
#define ZPSE_ASCII_REC   232u
#define ZPSE_ASCII_FLAGS (uint8_t)((uint8_t)ZPSE_M_ASCII_PACK | \
                                   (uint8_t)ZPSE_TILE_FLAG_RESET)

/* Every header byte of that exact record, so the shortcut below cannot
 * accept a record the general path would reject: flags, CONST_ZERO, a
 * 256-byte length, no exceptions, an empty mask and a 224-byte payload.
 * rs is the length prefix the caller already read. */
#define ZPSE_ASCII_HEAD_OK(p, rs)                                          \
    ((rs) == (uint32_t)ZPSE_ASCII_REC && (p)[0] == ZPSE_ASCII_FLAGS &&      \
     (p)[1] == 0u && (p)[2] == 255u && (p)[3] == 0u && (p)[4] == 0u &&      \
     (p)[5] == 0u && (p)[6] == 224u && (p)[7] == 0u)

#if defined(ZPSE_TRIAL_STATS)
/* Which (mask_mode, res_mode) the mode search actually picks, indexed
 * mask_mode * 8 + res_mode. This is what says whether a classify product
 * is worth producing at all: the per-exception arrays it compacts are
 * read one mode at a time. */
uint32_t zpse_stats_modes[64];
/* Opt-in accounting for the encoder's own decisions (see tile.c for the
 * per-predictor half). Defined here rather than next to the code that
 * bumps them because the dense-form counters are read by functions that
 * sit above the segment encoder body in this file. */
/* Tiles whose predictor search ran (see the increment in
 * zpse_segment_encode). The partner of zpse_trial_calls[] in tile.c. */
uint32_t zpse_trial_search_tiles;
/* Dense-form accounting: planes the full alphabet scan found palettable
 * (2..16 values), planes the palette form actually replaced the tiles of,
 * and planes the literal form replaced them of. A plane that is
 * palettable but never emits costs a full alphabet scan and nothing else,
 * so the pair separates "the form is used" from "the form was considered". */
uint32_t zpse_stats_palettable;
uint32_t zpse_stats_palette_emits;
uint32_t zpse_stats_literal_emits;
#endif

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

/* ---- 256-byte fill for a full constant tile ------------------------- *
 * A const tile's prediction is one byte, so the tile is that byte
 * everywhere: the fill is a memset of exactly 256 bytes, and on a sparse
 * plane (K of ~3 in 256) it is the largest single item of the tile.
 * `memset` with a runtime length does not inline, and both the call and
 * the `rep stosb` glibc picks at this size pay a start-up that is several
 * times what the 256 bytes themselves cost; 256 is known here, so this is
 * sixteen 16-byte stores with no branch and no call. SSE2 is baseline on
 * x86-64, so no dispatch is needed; every other target keeps the call. */
ZPSE_ALWAYS_INLINE void seg_fill_256(uint8_t *dst, uint8_t c)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    const __m128i v = _mm_set1_epi8((char)c);
    unsigned i;

    for (i = 0; i < 256u; i += 16u) {
        _mm_storeu_si128((__m128i *)(void *)(dst + i), v);
    }
#else
    memset(dst, c, 256u);
#endif
}

/* ---- Inline constant-tile decode ------------------------------------ *
 *
 * The dominant record on a sparse plane -- and the shape of every tile of
 * a constant one -- is a full 256-byte tile whose prediction is a single
 * byte: CONST_ZERO or TILE_CONSTANT, any mask mode, any residual mode but
 * SIGNED_DELTA (whose payload decodes through a private helper in
 * tile.c), at most 24 exceptions. Such a tile is a memset, one store per
 * exception in mask order, and the two-load history epilogue. The general
 * entry point reaches exactly that code, but only after it has parsed the
 * header a second time, re-derived the sizes, taken the family dispatch,
 * tested the all-hit case and checked for AVX2, and its frame carries the
 * delta buffer and every family's loop.
 *
 * So the tile loop takes this shape itself, using the same public pieces
 * in the same order (validate, mask decode, residual decode, fill,
 * scatter, history epilogue). The constants are the ones tile.c uses:
 * the exception stream is resolved as RAW_ACTUAL or exclusion exactly as
 * its CONST loop does, and a full tile's history fixup is the two 8-byte
 * loads below.
 *
 * Everything the general path would do about anything else is left to it:
 * a record of any other shape, and every record this path cannot prove
 * well formed, returns 0 and the loop calls zpse_tile_decode, which
 * re-validates and reports the identical status. A record this path does
 * accept gets the identical bytes and the identical status. */
static int seg_tile_decode_const(ZpseLaneState *lanes, const uint8_t *rec,
                                 uint16_t rsz, uint8_t literal,
                                 int tile_reset, uint8_t *out,
                                 uint8_t *comb)
{
    struct ZpseTileHeader th;
    uint16_t nt = 0;
    uint16_t k = 0;
    uint8_t mask_mode;
    uint8_t res_mode;
    uint8_t c = 0;
    const uint8_t *mp;
    uint32_t words[8];
    uint8_t resid[256];
    uint16_t j = 0;
    unsigned wi;
    int rc;

    if (rsz < (uint16_t)ZPSE_TILE_HEADER_SIZE) {
        return 0;
    }
    if (rec[1] != (uint8_t)ZPSE_P_CONST_ZERO &&
        rec[1] != (uint8_t)ZPSE_P_TILE_CONSTANT) {
        return 0;
    }
    mask_mode = zpse_tile_mask_mode(rec[0]);
    res_mode = zpse_tile_res_mode(rec[0]);
    if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
        return 0;
    }
    /* The ASCII form carries CONST_ZERO's predictor byte -- it predicts
     * nothing -- but its residual is the packed tile, not an exception
     * list, and its mask mode is not one any constant fill understands.
     * Decline: the general path decodes it. */
    if (mask_mode == (uint8_t)ZPSE_M_ASCII_PACK) {
        return 0;
    }
    zpse_tile_read_header(rec, &th);
    rc = zpse_tile_validate(&th, (uint16_t)ZPSE_MAX_TILE_LEN, rsz, &nt, &k);
    if (rc != PSE_OK) {
        return 0;
    }
    /* record_need == record_len, the identity zpse_tile_decode insists on
     * after its own validate call -- without it a record whose payload
     * sizes do not add up would be decoded here instead of rejected. */
    if ((uint32_t)rsz != (uint32_t)ZPSE_TILE_HEADER_SIZE +
            (th.predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT ? 1u : 0u) +
            (uint32_t)th.mask_payload_size +
            (uint32_t)th.residual_payload_size) {
        return 0;
    }
    /* Above this the general path's 32-wide kernel is the faster one, and
     * below it the memset plus k stores is; see the CONST crossover in
     * src/tile.c. */
    if (k > 24u) {
        return 0;
    }

    mp = rec + ZPSE_TILE_HEADER_SIZE;
    if (th.predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT) {
        c = mp[0];
        mp++;
    }

    /* ---- Sparse mask, direct residuals --------------------------- *
     * The two sparse mask modes ARE an ascending position list, and the
     * two direct residual modes ARE one byte per exception (the exclusion
     * residual, or the actual byte). For those the 8-word mask and the
     * residual buffer are both detours: a 32-byte clear, a bit-walk per
     * exception, and a call to expand K bytes -- on the planes this path
     * exists for (K of ~3 in 256) most of what a tile costs. So decode
     * the positions into `pos[]` and resolve each exception byte straight
     * from the payload, with the same checks zpse_mask_decode and
     * zpse_residual_decode make, in the same order and with the same
     * statuses: a payload this rejects is rejected identically, and one
     * it accepts produces the identical bytes.
     *
     * The lane reset is dead in both paths below as well as in this one.
     * A full-length CONST tile reads no lane state, and the epilogue at
     * the bottom sets both histories from the tile's own last sixteen
     * bytes, so nothing of the entry state survives the tile. */
    if ((mask_mode == (uint8_t)ZPSE_M_SPARSE_U8 ||
         mask_mode == (uint8_t)ZPSE_M_SPARSE_GAPS) &&
        (res_mode == (uint8_t)ZPSE_R_EXCLUSION_U8 ||
         res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) && k >= 1u) {
        uint8_t pos[24];
        const uint8_t *mpl = mp;
        uint16_t mlen = th.mask_payload_size;
        const uint8_t *rpl;
        uint16_t jj;

        if (mask_mode == (uint8_t)ZPSE_M_SPARSE_U8) {
            uint16_t prev = 0;

            if (mlen != k) {
                return PSE_BAD_MASK;
            }
            for (jj = 0; jj < k; jj++) {
                uint16_t v = mpl[jj];

                /* v >= Nt cannot happen for a byte when Nt is 256, so
                 * the ascending test is the whole of what the generic
                 * decoder checks here. */
                if (jj > 0u && v <= prev) {
                    return PSE_BAD_MASK;
                }
                prev = v;
                pos[jj] = (uint8_t)v;
            }
        } else {
            uint8_t flag;
            uint32_t gp = 0;

            if (mlen < 1u) {
                return PSE_BAD_MASK;
            }
            flag = mpl[0];
            if (flag > 1u) {
                return PSE_BAD_MASK;
            }
            if (mlen != (uint16_t)(flag != 0u ? 1u + 2u * k : 1u + k)) {
                return PSE_BAD_MASK;
            }
            for (jj = 0; jj < k; jj++) {
                uint32_t gap =
                    (flag != 0u)
                        ? ((uint32_t)mpl[1u + 2u * jj] |
                           ((uint32_t)mpl[2u + 2u * jj] << 8))
                        : (uint32_t)mpl[1u + jj];

                gp = (jj == 0u) ? gap : gp + 1u + gap;
                if (gp >= 256u) {
                    return PSE_BAD_MASK;
                }
                pos[jj] = (uint8_t)gp;
            }
        }
        /* Both direct residual modes are exactly K bytes; the exclusion
         * one additionally forbids 255, which is the only value that is
         * not a legal exclusion residual. */
        if ((uint32_t)th.residual_payload_size != (uint32_t)k) {
            return PSE_BAD_RESIDUAL;
        }
        rpl = mpl + mlen;
        if (res_mode == (uint8_t)ZPSE_R_EXCLUSION_U8) {
            for (jj = 0; jj < k; jj++) {
                if (rpl[jj] == 255u) {
                    return PSE_BAD_RESIDUAL;
                }
            }
        }
        seg_fill_256(out, c);
        for (jj = 0; jj < k; jj++) {
            out[pos[jj]] = (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL)
                               ? rpl[jj]
                               : zpse_exclusion_inverse(rpl[jj], c);
        }
        memcpy(lanes->last, out + 248u, sizeof lanes->last);
        memcpy(lanes->second_last, out + 240u, sizeof lanes->second_last);
        return 1;
    }

    /* CONST reads no table, so a reset is the histories only (this is what
     * zpse_lane_reset_for does for predictor 0 and 9). */
    if (tile_reset != 0) {
        zpse_lane_reset_hist(lanes, literal);
    }

    zpse_mask_clear(words);
    rc = zpse_mask_decode(mask_mode, mp, th.mask_payload_size, nt, k, words,
                          comb);
    if (rc != PSE_OK) {
        return rc;
    }
    if (k > 0u || th.residual_payload_size > 0u) {
        rc = zpse_residual_decode(res_mode, mp + th.mask_payload_size,
                                  th.residual_payload_size, k, 0, resid,
                                  resid);
        if (rc != PSE_OK) {
            return rc;
        }
    }

    /* A constant prediction is that constant everywhere but its
     * exceptions, so fill and then write them, in mask order -- the order
     * the residual units were packed in. */
    memset(out, c, nt);
    for (wi = 0; wi < 8u && j < k; wi++) {
        uint32_t w = words[wi];

        while (w != 0u) {
            uint16_t p = (uint16_t)((uint32_t)wi * 32u + zpse_ctz32(w));
            uint8_t e;

            if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                e = resid[j];
            } else {
                e = zpse_exclusion_inverse(resid[j], c);
            }
            out[p] = e;
            j++;
            w &= w - 1u;
        }
    }
    /* Full tile: lane l ends on position 248 + l, second history 8 back.
     * This is tile_history_fixup's nt == 256 branch. */
    memcpy(lanes->last, out + 248u, sizeof lanes->last);
    memcpy(lanes->second_last, out + 240u, sizeof lanes->second_last);
    if (j != k) {
        return PSE_BAD_MASK;
    }
    return 1;
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
    /* Repeated-record shortcut (see the block in the tile loop). The
     * previous tile's record, its length and whether it was a reset tile
     * this path may copy the output of. */
    const uint8_t *prev_rec = 0;
    uint16_t prev_rsz = 0u;
    uint64_t prev_key = 0u;
    int prev_ok = 0;

    if (d == 0 || lanes == 0 || payload == 0 || decoded_out == 0 ||
        decoded_len_out == 0) {
        return PSE_BAD_ARG;
    }
    /* Cleared here, set only by the dense constant form below: block
     * decode reads it after the loop to decide whether the payload CRC
     * can be arithmetic, so it must never describe an earlier segment. */
    d->seg_const = 0u;
    d->seg_const_byte = 0u;
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
        {
            int all_const = 0;
            uint8_t const_byte = 0u;

            rc = zpse_dense_decode(&h, payload, payload_len, decoded_out,
                                   decoded_cap, decoded_len_out, d->comb,
                                   &all_const, &const_byte);
            if (rc == PSE_OK && all_const != 0) {
                d->seg_const = 1u;
                d->seg_const_byte = const_byte;
            }
            return rc;
        }
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

        /* Once a tile, on both sides: see the encoder's loop. */
        lanes->gen++;

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
            prev_ok = 0;
            continue;
        }

        /* ---- Repeated record -------------------------------------- *
         * A reset tile's reconstruction is a function of its record and,
         * for the STRIDE family, of the eight output bytes before it: a
         * reset tile starts from the segment literal, and nothing else is
         * read (CONST's byte, the lane histories and the STRIDE reads are
         * all the predictors here consult). So a record that repeats the
         * previous tile's, at a position whose eight preceding output
         * bytes repeat too, decodes to the same bytes -- and those bytes
         * are already written, one tile back.
         *
         * Copying them is exactly what the tile loop below would have
         * produced. The previous tile must have been a reset tile as well
         * (otherwise the bytes one tile back are not what a reset tile
         * yields), and the lane state is left alone: a full 256-byte
         * tile's epilogue is the O(8) history update, which for nt == 256
         * reads only the tile's own output bytes, so the state after the
         * previous tile IS the state after this one.
         *
         * Nothing is skipped that a hostile stream could notice: the bytes
         * compared are at the same length and already passed validation at
         * this position, so any rejection they earn has already happened.
         * ORDER1/ORDER2 are excluded (stateful tables, and predictors this
         * encoder never writes), as are tiles that are not full length. */
        if (prev_ok && rsz == prev_rsz &&
            rsz >= (uint16_t)ZPSE_TILE_HEADER_SIZE) {
            uint64_t key;

            /* The record's leading word is the filter, exactly as in the
             * encoder's cache: on a plane that repeats nothing it rejects
             * the tile in one load and one compare, and the full compare
             * below only runs for a tile that already matches eight bytes
             * (on `sawtooth` -- 224 of 256 tiles -- the filter passes and
             * the compare confirms it). The length test in front of it is
             * what makes the eight-byte load safe: rsz is a field of the
             * stream and a record shorter than a header is a legal length
             * prefix, rejected later by the header parse. */
            memcpy(&key, rec, sizeof key);
            if (key == prev_key && nt_exp == (uint16_t)256u &&
                produced >= 264u &&
                (rec[0] & ZPSE_TILE_FLAG_RESET) != 0u &&
                rec[1] != (uint8_t)ZPSE_P_ORDER1 &&
                rec[1] != (uint8_t)ZPSE_P_ORDER2 &&
                memcmp(rec, prev_rec, (size_t)rsz) == 0 &&
                (rec[1] < (uint8_t)ZPSE_P_STRIDE_1 ||
                 rec[1] > (uint8_t)ZPSE_P_STRIDE_8 ||
                 memcmp(decoded_out + produced - 8u,
                        decoded_out + produced - 264u, 8u) == 0)) {
                memcpy(decoded_out + produced, decoded_out + produced - 256u,
                       256u);
                produced += 256u;
                pos += 2u + (uint32_t)rsz;
                continue;
            }
        }

        /* ---- ASCII run -------------------------------------------- *
         * Consecutive ASCII-pack records (tile.c, mask mode 7) are
         * independent of each other and of the lane state: the form has no
         * predictor, no mask and no residual codec, so a tile is a bit
         * unpack of its own 224-byte payload, and the state a run leaves
         * is just its last sixteen bytes. Taking a run here removes the
         * per-tile header parse, validate, dispatch and history epilogue
         * -- the difference between 3.9 GB/s through the general path and
         * the unpack's own rate.
         *
         * Only the exact shape the encoder writes is accepted: flags,
         * CONST_ZERO, no exceptions, and 232 record bytes for a 256-byte
         * tile. Anything else -- including a malformed record that shares
         * those four header bytes but not the length -- falls through to
         * the general path, which re-validates everything, so no rejection
         * changes. */
        if (nt_exp == (uint16_t)ZPSE_MAX_TILE_LEN &&
            ZPSE_ASCII_HEAD_OK(rec, (uint32_t)rsz)) {
            uint32_t run = 1u;
            uint32_t at = pos + 2u + (uint32_t)rsz;
            uint32_t o = produced;
            uint32_t k;
            const uint8_t *rp = rec;

            while ((uint32_t)t + run < tile_count &&
                   at + 2u + (uint32_t)ZPSE_ASCII_REC <= payload_len) {
                const uint8_t *nr = payload + at + 2u;

                if (!ZPSE_ASCII_HEAD_OK(nr, zpse_load_u16le(payload + at))) {
                    break;
                }
                at += 2u + (uint32_t)ZPSE_ASCII_REC;
                run++;
            }
            /* The run only holds for a stream whose remaining tiles really
             * are 256 bytes each; anything else is the general path's. */
            if ((uint64_t)run * 256u <= (uint64_t)(seg_size - produced) &&
                (uint64_t)run * 256u <= (uint64_t)(decoded_cap - produced)) {
                for (k = 0; k < run; k++) {
                    /* rp is the record (header) pointer; the length prefix
                     * sits two bytes before it. */
                    zpse_tile_ascii_unpack(
                        rp + (uint32_t)ZPSE_TILE_HEADER_SIZE,
                        (uint32_t)ZPSE_ASCII_REC -
                            (uint32_t)ZPSE_TILE_HEADER_SIZE,
                        decoded_out + o, (uint16_t)ZPSE_MAX_TILE_LEN);
                    o += 256u;
                    rp += 2u + (uint32_t)ZPSE_ASCII_REC;
                }
                memcpy(lanes->last, decoded_out + o - 8u, 8u);
                memcpy(lanes->second_last, decoded_out + o - 16u, 8u);
                /* One advance per tile, as the per-tile loop does; this
                 * iteration already counted the first one. */
                lanes->gen = (uint16_t)(lanes->gen + (uint16_t)(run - 1u));
                produced = o;
                pos = at;
                t = (uint16_t)(t + (uint16_t)(run - 1u));
                prev_rec = 0;
                prev_ok = 0;
                continue;
            }
        }

        tile_reset = (rec[0] & ZPSE_TILE_FLAG_RESET) != 0 ? 1 : 0;
        /* ---- Inline constant tile -------------------------------- *
         * See seg_tile_decode_const. 1 means the tile is written and only
         * the cursors are left; 0 means the record is some other shape (or
         * one this path will not vouch for) and the general entry point
         * below handles it; negative is a status it would have returned
         * at the same point. The lazy ORDER table seeding further down
         * only applies to predictors 3 and 4, which this path never
         * takes, so running it first would be dead work -- it is reached
         * exactly when the record is not a constant one. */
        if (nt_exp == (uint16_t)ZPSE_MAX_TILE_LEN) {
            int fc = seg_tile_decode_const(lanes, rec, rsz, literal,
                                           tile_reset, decoded_out + produced,
                                           d->comb);

            if (fc < 0) {
                return fc;
            }
            if (fc > 0) {
                produced += (uint32_t)ZPSE_MAX_TILE_LEN;
                pos += 2u + (uint32_t)rsz;
                prev_rec = rec;
                prev_rsz = rsz;
                memcpy(&prev_key, rec, sizeof prev_key);
                prev_ok = (tile_reset != 0) ? 1 : 0;
                continue;
            }
        }
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
        /* The repeated-record test above may look at this tile next, and
         * only a reset tile's output is the fixed point it assumes. */
        prev_rec = rec;
        prev_rsz = rsz;
        memcpy(&prev_key, rec, sizeof prev_key);
        prev_ok = (tile_reset != 0) ? 1 : 0;
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
 * id; ids outside the set are set to an unreachable sentinel. When
 * `fast_tier` is set, only the three uniform predictors (CONST_ZERO,
 * LANE_PREVIOUS, STRIDE_1) are computed; the rest are left as sentinel so
 * they can never win. */
static void seg_prescreen_counts(const uint8_t *a, uint16_t nt,
                                 const uint8_t *seg_base, uint32_t seg_pos,
                                 uint8_t init, int fast_tier, uint16_t ks[12])
{
    uint32_t k0 = 0u, k1 = 0u, k5 = 0u;
    uint32_t k2 = 0u, k6 = 0u, k7 = 0u, k8 = 0u, k10 = 0u, k11 = 0u;
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
        if (!fast_tier) {
            k2 += (uint32_t)(x != (uint8_t)((2u * (unsigned)h8 -
                                  (unsigned)h16) & 0xFFu));
            k5 += (uint32_t)(x != s1);
            k6 += (uint32_t)(x != s2);
            k7 += (uint32_t)(x != s4);
            k8 += (uint32_t)(x != s8);
            k11 += (uint32_t)(x != s3);
            k10 += (uint32_t)(x != (uint8_t)(((unsigned)h8 + (unsigned)h16 + 1u)
                                             >> 1));
        } else {
            k5 += (uint32_t)(x != s1);
        }

        second[lane] = h8;
        last[lane] = x;
    }

    ks[0] = (uint16_t)k0;
    ks[1] = (uint16_t)k1;
    ks[5] = (uint16_t)k5;
    if (!fast_tier) {
        ks[2] = (uint16_t)k2;
        ks[6] = (uint16_t)k6;
        ks[7] = (uint16_t)k7;
        ks[8] = (uint16_t)k8;
        ks[10] = (uint16_t)k10;
        ks[11] = (uint16_t)k11;
    }
    ks[3] = 0xFFFFu;
    ks[4] = 0xFFFFu;
    ks[9] = 0xFFFFu;
    if (fast_tier) {
        ks[2] = 0xFFFFu;
        ks[6] = 0xFFFFu;
        ks[7] = 0xFFFFu;
        ks[8] = 0xFFFFu;
        ks[10] = 0xFFFFu;
        ks[11] = 0xFFFFu;
    }
}

/* STRIDE_3 is last on purpose: the winner is the first predictor with the
 * strictly smallest miss count, so appending it can only change a tile
 * where it is strictly better than every earlier candidate. The order is
 * static at startup but can be reordered by the dynamic ranker (see
 * `seg_screen_rank`) at block boundaries, so the most-winning predictor is
 * tried first. */
static uint8_t seg_screen_pred[9] = {0u, 1u, 2u, 5u, 6u,
                                     7u, 8u, 10u, 11u};

/* Win count per predictor id, used by the dynamic ranker. Indexed by
 * predictor id (0-11); entries for ORDER1/ORDER2/TILE_CONSTANT stay at 0
 * because they are never in seg_screen_pred. */
static uint32_t seg_pred_wins[12];

/* Reorder seg_screen_pred so the predictor with the most wins runs first,
 * ties broken by the existing order. Called once per block (or whenever the
 * caller wants a fresh ranking); the rank is a pure function of the wins
 * seen so far, so it is deterministic across builds. */
void seg_screen_rank(void)
{
    uint8_t order[9];
    uint8_t used[9];
    uint32_t i;
    uint32_t j;
    uint32_t n;

    for (i = 0; i < 9u; i++) {
        order[i] = seg_screen_pred[i];
        used[i] = 0u;
    }
    for (n = 0u; n < 9u; n++) {
        uint32_t best = 0u;
        uint8_t best_p = 0u;
        uint32_t bi = 0u;
        int found = 0;

        for (i = 0; i < 9u; i++) {
            if (used[i] != 0u) {
                continue;
            }
            if (!found || (uint32_t)seg_pred_wins[order[i]] > best) {
                best = seg_pred_wins[order[i]];
                best_p = order[i];
                bi = i;
                found = 1;
            }
        }
        used[bi] = 1u;
        seg_screen_pred[n] = best_p;
    }
    for (j = 0; j < 12u; j++) {
        seg_pred_wins[j] = 0u;
    }
}

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
#define ZPSE_PS_AVX2 __attribute__((target("avx2"), aligned(64)))
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
 * start (every tile resets its lanes), so for p < 8 the
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
                                            int fast_tier, uint16_t ks[12])
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
        if (fast_tier) {
            ks[2] = 0xFFFFu;
            ks[6] = 0xFFFFu;
            ks[7] = 0xFFFFu;
            ks[8] = 0xFFFFu;
            ks[10] = 0xFFFFu;
            ks[11] = 0xFFFFu;
        }
    }
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* Dispatch helper: `have_avx2` is computed once per segment encode so
 * no capability state has to be cached in a global. */
static void seg_prescreen(const uint8_t *a, uint16_t nt,
                          const uint8_t *seg_base, uint32_t seg_pos,
                          uint8_t init, int fast_tier, int have_avx2,
                          uint16_t ks[12])
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
        seg_prescreen_avx2(a, nt, seg_base, seg_pos, init, fast_tier, ks);
        return;
    }
#else
    (void)have_avx2;
#endif
    seg_prescreen_counts(a, nt, seg_base, seg_pos, init, fast_tier, ks);
}

/* Miss count of CONST_ZERO over the whole tile: the number of nonzero
 * bytes. Same result from both forms, so the encoder's decision cannot
 * depend on which one a build has. */
static uint16_t seg_zero_misses_scalar(const uint8_t *a, uint16_t nt)
{
    uint32_t p;
    uint32_t hits = 0u;

    for (p = 0; p < (uint32_t)nt; p++) {
        hits += (a[p] == 0u) ? 1u : 0u;
    }
    return (uint16_t)((uint32_t)nt - hits);
}

#if ZPSE_PS_HAVE_AVX2
static ZPSE_PS_AVX2 uint16_t seg_zero_misses_avx2(const uint8_t *a, uint16_t nt)
{
    const __m256i zero = _mm256_setzero_si256();
    __m256i acc = zero;
    uint64_t lane[4];
    uint32_t hits = 0u;
    uint32_t p = 0u;
    int j;

    while (p + 32u <= (uint32_t)nt) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(a + p));

        /* A byte-lane counter: -1 for a zero byte, added in. */
        acc = _mm256_sub_epi8(acc, _mm256_cmpeq_epi8(v, zero));
        p += 32u;
    }
    _mm256_storeu_si256((__m256i *)(void *)lane, _mm256_sad_epu8(acc, zero));
    for (j = 0; j < 4; j++) {
        hits += (uint32_t)lane[j];
    }
    for (; p < (uint32_t)nt; p++) {
        hits += (a[p] == 0u) ? 1u : 0u;
    }
    return (uint16_t)((uint32_t)nt - hits);
}
#endif

/* Prescreen with one shortcut: a tile that CONST_ZERO misses at most
 * ZPSE_PS_ZERO_ONLY positions is classified against CONST_ZERO alone.
 *
 * CONST_ZERO is the first predictor in seg_screen_pred[], so it wins every
 * tie -- which means the other eight can only beat it by a strictly lower
 * miss count. At <= 3 misses a lower count is at most 2, and the record it
 * would buy differs by a couple of bytes, while the nine-predictor pass is
 * ~30 ns of a tile that costs ~170 ns to encode (`sparse`: nearly every
 * tile is zeroes plus a spike or two, and that row is the slowest encoder).
 * This is a heuristic, not a proof, so it is gated by the pinned ratio
 * corpus: no case may grow (`tests/test_sizes.c` fails if one does).
 *
 * The shortcut is armed by the PREVIOUS tile rather than measured on every
 * tile: `*zero_cheap` says the last tile's prescreen found CONST_ZERO
 * within the threshold, so a plane whose tiles are mostly not near-zero
 * (random, alternation, a ramp) never pays for the zero count at all --
 * measuring unconditionally cost 1.5-5.8% on every row the shortcut does
 * not help. A plane of near-zero tiles pays one 256-byte compare instead
 * of nine, which is the 16% `sparse` gains.
 *
 * Both forms of the miss count return the same value, and the arming
 * decision comes from ks[0], so a -DZPSE_FORCE_SCALAR build makes exactly
 * the same decisions as the vector one (the parity digests depend on it). */
#define ZPSE_PS_ZERO_ONLY 3u

static void seg_prescreen_tile(const uint8_t *a, uint16_t nt,
                               const uint8_t *seg_base, uint32_t seg_pos,
                               uint8_t init, int fast_tier, int have_avx2,
                               int *zero_cheap, uint16_t ks[12])
{
    if (*zero_cheap) {
        uint16_t zk;

#if ZPSE_PS_HAVE_AVX2
        zk = have_avx2 ? seg_zero_misses_avx2(a, nt)
                       : seg_zero_misses_scalar(a, nt);
#else
        (void)have_avx2;
        zk = seg_zero_misses_scalar(a, nt);
#endif
        if (zk <= (uint16_t)ZPSE_PS_ZERO_ONLY) {
            int i;

            for (i = 0; i < 12; i++) {
                ks[i] = 0xFFFFu;
            }
            ks[0] = zk;
            return;
        }
        *zero_cheap = 0;
    }
    seg_prescreen(a, nt, seg_base, seg_pos, init, fast_tier, have_avx2, ks);
    *zero_cheap = (ks[seg_screen_pred[0]] <= (uint16_t)ZPSE_PS_ZERO_ONLY)
                      ? 1 : 0;
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

/* ------------------------------------------------------------------ *
 * Generalised run continuation: "does this tile keep the run's predictor,
 * with k == 0?" proved for any of the nine screened predictors, not just
 * the three uniform cases seg_all_hit_cheap_ok covers.
 *
 * A run may continue only when the prescreen would name the same predictor
 * with k == 0, because that is what the general path would otherwise have
 * chosen. The prescreen's winner is the FIRST predictor of
 * seg_screen_pred[] with the strictly smallest miss count, so keeping the
 * run needs two facts:
 *
 *   (a) the run's predictor misses nothing in this tile, and
 *   (b) every predictor BEFORE it in scan order misses somewhere.
 *
 * (a) is one prediction pass over the tile (a shifted compare for the
 * families below), (b) is a witness search: one miss each is enough, and
 * the search stops as soon as it finds one. `(b)` is bounded to the first
 * SEG_WITNESS positions per predictor, so this can never cost more than a
 * fixed few dozen cycles; when a witness is not found that fast the caller
 * asks the prescreen, which stays authoritative. Being wrong towards "ask"
 * costs time, never bytes.
 *
 * This exists because the periodic rows (`incr`'s delta plane, `strided`,
 * `sparse2`) are runs of tiles that are NOT one repeated byte, so the
 * uniform proof above could not settle them and every tile paid a full
 * nine-predictor prescreen: measured 255 prescreens per 64 KiB block on
 * `strided`, 36k of that block's 74k cycles.
 * ------------------------------------------------------------------ */

/* Positions the witness search may look at before giving up on ruling a
 * predictor out. Eight is one chunk: a predictor that hits a whole chunk of
 * a tile the run's predictor hits exactly is one whose count is worth the
 * exact prescreen. */
#define SEG_WITNESS 8u

static uint32_t seg_stride_of(uint8_t pred)
{
    switch (pred) {
    case ZPSE_P_STRIDE_1:
        return 1u;
    case ZPSE_P_STRIDE_2:
        return 2u;
    case ZPSE_P_STRIDE_3:
        return 3u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u;
    }
}

/* 1 iff `pred` misses at least one of the first `limit` positions. The lane
 * histories start at `literal` because every tile the encoder writes is
 * classified with the reset bit set. Returns 0 both for "no
 * miss in the window" and for a predictor this cannot reason about, which
 * the caller treats the same way: ask the prescreen. */
static int seg_pred_miss_within(const uint8_t *a, uint16_t nt, uint8_t pred,
                                uint8_t literal, uint32_t seg_pos,
                                const uint8_t *plane, uint16_t limit)
{
    uint8_t last[8];
    uint8_t second[8];
    uint16_t n = (nt < limit) ? nt : limit;
    uint16_t p;
    unsigned l;

    for (l = 0; l < 8u; l++) {
        last[l] = literal;
        second[l] = literal;
    }
    for (p = 0; p < n; p++) {
        unsigned lane = (unsigned)p & 7u;
        uint8_t x = a[p];
        uint32_t d = seg_pos + (uint32_t)p;
        uint8_t xhat;

        switch (pred) {
        case (uint8_t)ZPSE_P_CONST_ZERO:
            xhat = 0u;
            break;
        case (uint8_t)ZPSE_P_LANE_PREVIOUS:
            xhat = last[lane];
            break;
        case (uint8_t)ZPSE_P_LANE_DELTA:
            xhat = (uint8_t)((2u * (unsigned)last[lane] -
                              (unsigned)second[lane]) & 0xFFu);
            break;
        case (uint8_t)ZPSE_P_LANE_AVERAGE:
            xhat = (uint8_t)(((unsigned)last[lane] +
                              (unsigned)second[lane] + 1u) >> 1);
            break;
        case (uint8_t)ZPSE_P_STRIDE_1:
        case (uint8_t)ZPSE_P_STRIDE_2:
        case (uint8_t)ZPSE_P_STRIDE_3:
        case (uint8_t)ZPSE_P_STRIDE_4:
        case (uint8_t)ZPSE_P_STRIDE_8: {
            uint32_t s = seg_stride_of(pred);

            xhat = (d < s) ? literal : plane[d - s];
            break;
        }
        default:
            return 0;
        }
        if (x != xhat) {
            return 1;
        }
        second[lane] = last[lane];
        last[lane] = x;
    }
    return 0;
}

#if ZPSE_PS_HAVE_AVX2
/* One prediction pass over the tile for the shifted families, checking that
 * every position matches. The staging buffer is the tile.c one (16 head
 * bytes + tile + 32 pad), so each prediction is a constant-offset load:
 *
 *   LANE_DELTA    2*stg[8+p] - stg[p]      (head = the entry lane state)
 *   LANE_AVERAGE  avg(stg[8+p], stg[p])
 *   STRIDE_s      stg[16+p-s]              (head = real plane bytes)
 *
 * The first chunk blends its first 8 / 16 lanes with `literal`, which is
 * what the lane state holds at p < 8 / p < 16 after the per-tile reset.
 * STRIDE needs real bytes before the tile, so the caller only takes this
 * path when seg_pos >= 16. */
static ZPSE_PS_AVX2 int seg_shift_all_hit_avx2(const uint8_t *a, uint16_t nt,
                                               uint8_t pred, uint8_t literal,
                                               uint32_t seg_pos,
                                               const uint8_t *plane)
{
    uint8_t stg[ZPSE_PS_STG_LEN];
    const __m256i initv = _mm256_set1_epi8((char)literal);
    const __m256i idx = _mm256_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                         16,17,18,19,20,21,22,23,24,25,26,27,
                                         28,29,30,31);
    const __m256i m8 = _mm256_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,
                                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0);
    const __m256i m16 = _mm256_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,
                                         -1,-1,-1,-1,-1,-1,-1,-1,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    uint32_t s = (pred == (uint8_t)ZPSE_P_LANE_DELTA ||
                  pred == (uint8_t)ZPSE_P_LANE_AVERAGE)
                     ? 0u
                     : seg_stride_of(pred);
    uint32_t p = 0u;

    if (s != 0u) {
        /* Real preceding plane bytes (the caller guarantees seg_pos >= 16). */
        memcpy(stg, plane + seg_pos - 16u, 16);
    } else {
        memset(stg, literal, 16);
    }
    memcpy(stg + 16, a, nt);
    memset(stg + 16 + nt, 0, 32);

    while (p + 32u <= (uint32_t)nt) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + 16u + p));
        __m256i pv;

        if (s != 0u) {
            pv = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + 16u + p - s));
        } else {
            __m256i h8 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + 8u + p));
            __m256i h16 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + p));

            if (p == 0u) {
                h8 = _mm256_blendv_epi8(h8, initv, m8);
                h16 = _mm256_blendv_epi8(h16, initv, m16);
            }
            if (pred == (uint8_t)ZPSE_P_LANE_DELTA) {
                pv = _mm256_sub_epi8(_mm256_add_epi8(h8, h8), h16);
            } else {
                pv = _mm256_avg_epu8(h8, h16);
            }
        }
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, pv)) != -1) {
            return 0;
        }
        p += 32u;
    }
    if (p < (uint32_t)nt) {
        uint32_t rem = (uint32_t)nt - p;
        __m256i valid = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)rem), idx);
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(stg + 16u + p));
        __m256i pv;

        if (s != 0u) {
            pv = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + 16u + p - s));
        } else {
            __m256i h8 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + 8u + p));
            __m256i h16 = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + p));

            pv = (pred == (uint8_t)ZPSE_P_LANE_DELTA)
                     ? _mm256_sub_epi8(_mm256_add_epi8(h8, h8), h16)
                     : _mm256_avg_epu8(h8, h16);
        }
        /* Valid lanes must all match; the padding lanes are masked out. */
        if ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, pv)) |
             ~_mm256_movemask_epi8(valid)) != -1) {
            return 0;
        }
    }
    return 1;
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* 1 iff predictor `pred` predicts every position of this tile (k == 0).
 * The three uniform cases reuse the specialised routine above; the rest are
 * a shifted compare, or the scalar walk when AVX2 is unavailable or the
 * tile starts a segment (where STRIDE's preceding bytes are not all
 * available). */
static int seg_pred_all_hit(const uint8_t *a, uint16_t nt, uint8_t pred,
                            uint8_t literal, uint32_t seg_pos,
                            const uint8_t *plane, int have_avx2)
{
    if (pred == (uint8_t)ZPSE_P_CONST_ZERO ||
        pred == (uint8_t)ZPSE_P_LANE_PREVIOUS ||
        pred == (uint8_t)ZPSE_P_STRIDE_1) {
        return seg_all_hit_cheap_ok(a, nt, pred, literal, seg_pos, plane,
                                    have_avx2);
    }
#if ZPSE_PS_HAVE_AVX2
    if (have_avx2 && (seg_pos >= 16u ||
                      (pred != (uint8_t)ZPSE_P_STRIDE_2 &&
                       pred != (uint8_t)ZPSE_P_STRIDE_3 &&
                       pred != (uint8_t)ZPSE_P_STRIDE_4 &&
                       pred != (uint8_t)ZPSE_P_STRIDE_8))) {
        return seg_shift_all_hit_avx2(a, nt, pred, literal, seg_pos, plane);
    }
#else
    (void)have_avx2;
#endif
    return !seg_pred_miss_within(a, nt, pred, literal, seg_pos, plane, nt);
}

/* 1 iff the run may continue with `pred` on this tile: the prescreen would
 * still name it, with k == 0. See the block comment above. */
static int seg_run_stays(const uint8_t *a, uint16_t nt, uint8_t pred,
                         uint8_t literal, uint32_t seg_pos,
                         const uint8_t *plane, int have_avx2)
{
    int i;

    /* The uniform three carry their own ordering proof. */
    if (pred == (uint8_t)ZPSE_P_CONST_ZERO ||
        pred == (uint8_t)ZPSE_P_LANE_PREVIOUS ||
        pred == (uint8_t)ZPSE_P_STRIDE_1) {
        return seg_all_hit_cheap_ok(a, nt, pred, literal, seg_pos, plane,
                                    have_avx2);
    }
    for (i = 0; i < 9; i++) {
        if (seg_screen_pred[i] == pred) {
            break;
        }
    }
    if (i == 9) {
        return 0; /* not a screened predictor: let the prescreen decide */
    }
    /* Every predictor earlier in the scan must miss somewhere, or it would
     * win the tile instead (first minimum). One miss each is a proof. */
    {
        int j;

        for (j = 0; j < i; j++) {
            if (!seg_pred_miss_within(a, nt, seg_screen_pred[j], literal,
                                      seg_pos, plane, SEG_WITNESS)) {
                return 0;
            }
        }
    }
    return seg_pred_all_hit(a, nt, pred, literal, seg_pos, plane, have_avx2);
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
                            uint8_t initial_literal, uint16_t ks[12],
                            uint32_t enc_flags)
{
    int have_avx2 = (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;
    int fast_tier = zpse_enc_speed(enc_flags) <= 2u;

    seg_prescreen(a, nt, seg_base, seg_pos, initial_literal, fast_tier,
                  have_avx2, ks);
}


#if ZPSE_PS_HAVE_AVX2
/* 1 iff p[0..n) is one byte value. 32 bytes per step: the word loop below
 * is a load, a compare and a branch per eight bytes, which is 8192
 * iterations (2 us) on a 64 KiB plane -- more than the tile loop the
 * constant shortcut exists to skip. This is byte-for-byte the same test. */
static ZPSE_PS_AVX2 int seg_plane_is_const_avx2(const uint8_t *p, uint32_t n)
{
    const __m256i v = _mm256_set1_epi8((char)p[0]);
    const __m256i eq = _mm256_cmpeq_epi8(v, v);
    uint32_t i = 0u;

    while (i + 32u <= n) {
        __m256i q = _mm256_loadu_si256(
            (const __m256i *)(const void *)(p + i));

        if ((uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(q, v)) !=
            (uint32_t)_mm256_movemask_epi8(eq)) {
            return 0;
        }
        i += 32u;
    }
    for (; i < n; i++) {
        if (p[i] != p[0]) {
            return 0;
        }
    }
    return 1;
}
#endif /* ZPSE_PS_HAVE_AVX2 */

/* 1 iff p[0..n) is one byte value, which is the only plane shape the dense
 * form pays on (see seg_dense_emit). Word-wise with an early exit, so a
 * plane that is not constant costs one compare. */
static int seg_plane_is_const(const uint8_t *p, uint32_t n)
{
    uint64_t splat = UINT64_C(0x0101010101010101) * (uint64_t)p[0];
    uint32_t i = 0u;

#if ZPSE_PS_HAVE_AVX2
    if ((zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        return seg_plane_is_const_avx2(p, n);
    }
#endif
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
    uint32_t step;
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
    /* Every second tile, or every fourth once the window holds enough of
     * them: the estimate is a sum over the window and only its ORDER
     * against another candidate's sum is read, so the sample has to be
     * representative, not complete. The skipped tiles are priced at the
     * sampled tiles' average, which keeps the estimate unbiased; what the
     * wider stride costs is variance.
     *
     * The wide stride is taken only for a FULL proxy window (16 KiB, 64
     * tiles), where a quarter of it is still 16 tiles of evidence.
     * Measured: it takes the proxy from ~18.0k to ~11.5k cycles of a 64 KiB
     * block -- 6% of a lowentropy block's encode, and the proxy is a fifth
     * of that. Nothing below a full window gets it: a short block is
     * proxied by a quarter of its own tiles (8 KiB is 32 tiles, so 8), and
     * that is where the size gate (tests/test_sizes.c) caught it -- three
     * 8 KiB cases of the corpus grew by 281 bytes each. With the window
     * rule every one of the 768 pinned lengths is unchanged, and so is the
     * total of the 2,718-case fuzz corpus, whose mixed-region shapes are
     * the ones a coarse sample would mis-rank. */
    step = (tc >= 64u) ? 4u : 2u;
    for (t = 0; t < tc; t += step) {
        uint32_t seg_pos = t * 256u;
        uint16_t nt =
            (uint16_t)((t + 1u < tc) ? 256u : (plane_len - seg_pos));
        uint16_t ks[12];
        uint16_t bk;
        uint32_t i;

        seg_prescreen(plane + seg_pos, nt, plane, seg_pos, init, 0,
                      have_avx2, ks);
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
    return cost + sum * step;
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

/* Dense record form for a constant plane.
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

int zpse_segment_plane_is_const(const uint8_t *p, uint32_t n)
{
    if (p == 0 || n == 0u) {
        return 0;
    }
    return seg_plane_is_const(p, n);
}

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
 *: one record per ZPSE_DENSE_MAX_LEN bytes, each a 12-byte
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

/* Dense palette record form for a plane of at most
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
#if defined(ZPSE_TRIAL_STATS)
    zpse_stats_palette_emits++;
#endif
    return (int)at;
}

/* Dense record form for a plane with no structure to predict: a plane whose
 * bytes are close to uniform is coded by the tile path as one record per 256
 * bytes that carries the bytes themselves -- an ALL_EXCEPTION-ish mask of a
 * handful of bytes and a RAW_ACTUAL residual of nearly all 256 -- so the
 * plane pays ten bytes of framing and a full nine-predictor search per tile
 * to store its own bytes, growing them by ~6%.
 *
 * The dense form already has the shape: an ALL_EXCEPTION mask makes every
 * position an exception and RAW_ACTUAL then means the residual IS the plane,
 * so the decoder runs no prediction at all and reconstructs the plane with
 * one copy (dense.c, the ALL_EXCEPTION + constant-predictor path). One
 * 12-byte header per ZPSE_DENSE_MAX_LEN bytes replaces the per-tile framing,
 * so a 4 KiB plane costs plane_len + 12 instead of ~plane_len + 250: both
 * smaller AND a copy instead of a predictor chain per tile.
 *
 * Only offered to a plane whose alphabet is too wide for the palette form
 * (more than sixteen distinct values) and only when the tiles already coded
 * or their projection say they will lose, exactly like the palette form --
 * `lit_len` is exact (nothing about the form is re-derived per record), so
 * the comparison in the loop can be trusted where the palette's is an upper
 * bound. */
static uint32_t seg_dense_literal_cost(uint32_t plane_len)
{
    uint32_t chunks = (plane_len + ZPSE_DENSE_MAX_LEN - 1u) /
                      ZPSE_DENSE_MAX_LEN;

    return (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
           chunks * (uint32_t)ZPSE_DENSE_HEADER_SIZE + plane_len;
}

static int seg_dense_literal_emit(ZpseSegmentHeader *h, uint8_t *payload_out,
                                  const uint8_t *plane, uint32_t plane_len,
                                  uint32_t pos, int replace,
                                  uint32_t byte_budget, uint32_t payload_cap)
{
    uint32_t cost = seg_dense_literal_cost(plane_len);
    uint32_t at;
    uint32_t off;

    if ((cost >= pos && replace == 0) || cost > payload_cap) {
        return 0;
    }
    if (byte_budget != 0u && cost > byte_budget) {
        return 0;
    }
    at = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;
    for (off = 0u; off < plane_len; off += ZPSE_DENSE_MAX_LEN) {
        uint32_t ns = plane_len - off;

        if (ns > ZPSE_DENSE_MAX_LEN) {
            ns = ZPSE_DENSE_MAX_LEN;
        }
        zpse_store_u16le(payload_out + at, (uint16_t)(10u + ns));
        payload_out[at + 2u] = (uint8_t)(ZPSE_DENSE_MARKER |
                                         (uint8_t)ZPSE_M_ALL_EXCEPTION |
                                         ((uint8_t)ZPSE_R_RAW_ACTUAL << 3));
        /* TILE_CONSTANT is the predictor the ALL_EXCEPTION path accepts, and
         * its constant is unused: RAW_ACTUAL supplies every byte, so the
         * decoder never resolves a prediction for this record. */
        payload_out[at + 3u] = (uint8_t)ZPSE_P_TILE_CONSTANT;
        payload_out[at + 4u] = 0u;
        payload_out[at + 5u] = 0u;
        zpse_store_u16le(payload_out + at + 6u, (uint16_t)ns);
        zpse_store_u16le(payload_out + at + 8u, 0u);
        zpse_store_u16le(payload_out + at + 10u, (uint16_t)ns);
        if (ns != 0u) {
            memcpy(payload_out + at + (uint32_t)ZPSE_DENSE_HEADER_SIZE,
                   plane + off, (size_t)ns);
        }
        at += (uint32_t)ZPSE_DENSE_HEADER_SIZE + ns;
    }
    /* A dense segment always reports tile_count == 1 (Sec. 6.3). */
    h->tile_count = 1u;
    zpse_segment_header_serialize(payload_out, h);
#if defined(ZPSE_TRIAL_STATS)
    zpse_stats_literal_emits++;
#endif
    return (int)at;
}

/* Tiles a palette decision prices before deciding, and the plane size in
 * tiles below which it does not price at all. See the projection in
 * zpse_segment_encode. */
#define ZPSE_EST_TILES 8u
#define ZPSE_EST_MIN_TILES 16u
/* Tiles whose records the projection extrapolates from. Two, not four: the
 * tiles of a palette plane are alike by construction (that is why the form
 * wins on it), so half the plane already says what the rest costs, and every
 * tile coded before the form is taken is a pipeline whose bytes are thrown
 * away. The 1/8 margin the projection uses is unchanged, so a plane the early
 * extrapolation misprices keeps its tiles and only loses the comparison. */
#define ZPSE_PAL_PROJ_TILES 2u

/* Price of a plane's tile records, from a sample of `tiles` tiles spread
 * evenly over it.
 *
 * This exists because the two forms are priced in very different ways. A
 * dense form's price is exact and costs one pass over the plane. The tiles'
 * price is exact only after every tile has been classified, mode-searched
 * and packed -- which is the whole trial. On 64 KiB of four-bit data the
 * tiles lose by a quarter, and the exact comparison finds that out by
 * writing 190 of them and then replacing all 190 with the palette: measured
 * 110 us of that block's 114 us of encode. The sample is a few tiles
 * instead. On a plane whose tiles are alike (what a four-bit plane is) the
 * per-tile average is within a percent, and on a plane where the dense form
 * loses the sample costs a few microseconds -- noise against a trial.
 *
 * Each sampled tile goes through exactly the pipeline the trial loop runs,
 * including the second candidate when it is competitive, so this is the
 * trial's own per-tile size rather than a proxy for it. Returns the price
 * INCLUDING the 16-byte segment header, or UINT32_MAX when a tile cannot be
 * priced (the caller then has no estimate). */
static uint32_t seg_tile_price(ZpseEncoder *e, ZpseLaneState *lanes,
                               const uint8_t *plane, uint32_t plane_len,
                               uint32_t tc, unsigned tiles,
                               uint8_t initial_literal, int have_avx2,
                               uint32_t limit)
{
    uint32_t step = tc / tiles;
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

        seg_prescreen(actual, nt, plane, seg_pos, initial_literal,
                      zpse_enc_speed(e->enc_flags) <= 2u, have_avx2, ks);
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
            /* The all-hit run writes these directly: a 2-byte length prefix
             * and the 8-byte record. */
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
            if (second_p != 0xFFu && sk == best_k &&
                (uint32_t)best_k * 8u < (uint32_t)nt * 7u) {
                cand[ncand++] = second_p;
            }
        }
        /* A generation no real tile uses, so the entries this pass
         * leaves behind read as unlearned to the tile that follows. The
         * lane state's own generation is put back afterwards: the
         * pricing pass is not a tile. */
        uint16_t gen_save = lanes->gen;

        lanes->gen = 0xFFFFu;
        for (ci = 0u; ci < ncand; ci++) {
            uint8_t pred = cand[ci];
            ZpseTileClass cl;
            uint8_t mm;
            uint8_t rr;
            int n;

            if (zpse_tile_classify_lean(lanes, actual, nt, pred,
                                        initial_literal, 1, plane, seg_pos,
                                        &cl) != PSE_OK) {
                continue;
            }
            zpse_tile_choose_modes(&cl, nt, &mm, &rr);
            if (rr == (uint8_t)ZPSE_R_PALETTE_BITS ||
                rr == (uint8_t)ZPSE_R_PACKED_BASE_M ||
                rr == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                (void)zpse_tile_fill_at_pred_reset(pred, actual, nt, plane,
                                                   seg_pos, initial_literal,
                                                   &cl);
            }
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
        lanes->gen = gen_save;
        if (tile_len < 0) {
            return UINT32_MAX;
        }
        /* The encoder offers the seven-bit form for an ASCII tile as one
         * more candidate and keeps whichever record is smaller, so the
         * tile's cost -- and the price the sweep must rank the plane by --
         * is the smaller of the two. Pricing only the searched record is
         * what kept a text plane priced above RAW, which is how a text
         * block ended up written raw at every level but MAX. The clamp can
         * only lower a price, and only to what the tile loop really
         * writes, so no shape can be preferred on a promise it does not
         * keep. */
        if (zpse_tile_is_ascii(actual, nt) &&
            (uint32_t)tile_len > zpse_tile_ascii_len(nt)) {
            tile_len = (int)zpse_tile_ascii_len(nt);
        }
        sum += 2u + (uint32_t)tile_len;
        nsum++;
        /* `limit` is the sweep's running verdict: a price above it can never
         * win, so there is no point finishing the sample. The folded price
         * is header + round((S_final*tc)/nsum_final), with S_final only
         * growing and tc/nsum_final >= 1 (we never price more tiles than
         * exist), so it can never fall below header + sum. Once that floor
         * reaches `limit` the sample is convicted; UINT32_MAX means "could
         * not be priced", so the caller keeps the candidate -- a safe,
         * perf-only degradation, never a wrongly dropped winner. */
        if (limit != UINT32_MAX &&
            (uint64_t)ZPSE_SEGMENT_HEADER_SIZE + (uint64_t)sum >=
                (uint64_t)limit) {
            /* Convicted: return the proven floor (header + sum), a finite
             * value >= limit that is a valid lower bound on the true price.
             * UINT32_MAX stays reserved for "could not be priced" (keep),
             * so a convicted value is distinguishable from an unpriceable
             * one and the caller can drop the candidate instead. */
            return (uint32_t)ZPSE_SEGMENT_HEADER_SIZE + sum;
        }
    }
    if (nsum == 0u) {
        return UINT32_MAX;
    }
    return (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
           (uint32_t)(((uint64_t)sum * tc + nsum / 2u) / nsum);
}

/* Price of the dense constant form for a plane: the
 * segment header plus twelve bytes per 4 KiB chunk, no payload. Returns
 * UINT32_MAX when the plane is not one byte value end to end, which the
 * word-wise test settles in one compare for every plane that is not. */
uint32_t zpse_segment_const_cost(const uint8_t *plane, uint32_t plane_len)
{
    uint32_t chunks;

    if (plane == 0 || plane_len == 0u) {
        return UINT32_MAX;
    }
    if (!seg_plane_is_const(plane, plane_len)) {
        return UINT32_MAX;
    }
    chunks = (plane_len + ZPSE_DENSE_MAX_LEN - 1u) / ZPSE_DENSE_MAX_LEN;
    return (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
           chunks * (uint32_t)ZPSE_DENSE_HEADER_SIZE;
}

/* Smallest size the segment encoder can write this plane in, whole: the
 * smaller of the tile records (sampled) and the dense constant form, which
 * is priced exactly because it costs one word-wise pass.
 *
 * The constant form is what a per-tile estimate cannot see and what changes
 * the ranking: a plane of one byte value costs twelve bytes per 4 KiB as a
 * dense record against ten bytes per 256-byte tile, and the phases of a
 * transposed plane are constant whenever the source repeats with the
 * transpose period (alternating bytes, a spike every 256). A 64 KiB block
 * of two-byte spikes is 1255 bytes transposed and 3126 as tiles, and the
 * per-tile price had it backwards; alternating 0x00/0xFF is 608 against
 * 2618.
 *
 * The palette form is deliberately NOT priced here. Recognising it costs a
 * scalar gather of the plane's alphabet (the vector pass can only test
 * membership), and a plane whose tiles are worse than a palette is one
 * whose candidate is palettable either way, so the ranking it would change
 * is a tie the segment encoder resolves exactly on its own.
 *
 * The result is what zpse_segment_encode would return for the plane, up to
 * the tile sample's error, and includes the 16-byte segment header.
 * UINT32_MAX means the plane could not be priced.
 *
 * `limit` threads the sweep's running verdict into the sample: planes that
 * cannot beat it stop after the tiles that convicted them (see
 * seg_tile_price), while UINT32_MAX prices the whole sample, as before. */
uint32_t zpse_segment_probe_cost(ZpseEncoder *e, ZpseLaneState *lanes,
                                 const uint8_t *plane, uint32_t plane_len,
                                 unsigned tiles, uint32_t limit)
{
    uint32_t tc;
    uint32_t best;
    int have_avx2 = (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;

    if (e == 0 || lanes == 0 || plane == 0 || plane_len == 0u) {
        return UINT32_MAX;
    }
    if (tiles == 0u) {
        tiles = ZPSE_EST_TILES;
    }
    tc = (plane_len + 255u) / 256u;
    best = seg_tile_price(e, lanes, plane, plane_len, tc, tiles,
                          plane[0], have_avx2, limit);

    /* A constant plane's sampled tiles are all-hit, so its tile price is
     * exactly the plane floor (a 2-byte prefix and an 8-byte header per
     * tile); a plane priced above that cannot be constant and does not need
     * the scan at all. The test is therefore free on every plane that is
     * not all-hit, which is most of them. */
    if (best == (uint32_t)ZPSE_SEGMENT_HEADER_SIZE + 10u * tc) {
        uint32_t d = zpse_segment_const_cost(plane, plane_len);

        if (d != UINT32_MAX && d < best) {
            best = d;
        }
    }
    return best;
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

/* Repeat cache geometry: 128 direct-mapped slots, indexed by the top bits
 * of the tile key. A plane is 16 tiles, a 64 KiB block 256, so this holds
 * every distinct tile of any plane that repeats with a period of up to ~128
 * tiles (32 KiB) -- which is what a block of fixed-size records is. The
 * arrays are 12 bytes a slot and are zeroed once per segment; on the 16
 * tiles of a 4 KiB plane that clear is under 3 cycles a tile. */
#define ZPSE_REP_SLOTS 128u
#define ZPSE_REP_SHIFT (64u - 7u)

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
    /* 1 iff the block layer already proved this very plane is one byte
     * value (see the constant-plane shortcut below). */
    int plane_known_const;
    uint32_t pal_len;
    uint32_t pal_m;
    int pal_try;
    /* Price of the dense literal form, or 0 when the plane is not a
     * candidate for it (its alphabet fits the palette form, or the form
     * does not fit the budget). */
    uint32_t lit_len;
    int lit_try;
    /* Tile count at which the literal form's projection is evaluated next,
     * and the shift that divides by it (see the tile loop). */
    uint32_t lit_proj_at;
    uint32_t lit_proj_shift;
    /* Tile count at which the palette projection is evaluated next, and the
     * shift that divides by it (see the projection in the tile loop). */
    uint32_t proj_at;
    uint32_t proj_shift;
    /* Whether the previous tile armed the CONST_ZERO shortcut in
     * seg_prescreen_tile (see there). */
    int zero_cheap = 0;
    uint32_t pal_bits[8];
    /* Repeat cache: source tile offset in the plane, where its record
     * starts in the payload, its length (with the u16 prefix) and its tile
     * length. A zero offset means the slot is empty and is the only other
     * thing the entry holds, so the arrays are zeroed together. */
    uint32_t rep_off[ZPSE_REP_SLOTS];
    uint32_t rep_pos[ZPSE_REP_SLOTS];
    uint16_t rep_len[ZPSE_REP_SLOTS];
    uint16_t rep_nt[ZPSE_REP_SLOTS];
    /* The tile key the slot was filled with, so a tile whose key differs
     * can be rejected by one 64-bit compare instead of the 264-byte
     * memcmp below. Equal keys say nothing (the key is a 4-word XOR, and
     * different tiles can collide); different keys prove the tiles differ,
     * which is what makes this a filter and not a second hash check. On a
     * plane with no repeats -- every dense shape -- that saves a libc call
     * and a 264-byte compare per tile. */
    uint64_t rep_key[ZPSE_REP_SLOTS];

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

    /* Constant-plane tracking, which lets the dense form
     * skip its own scan: a tile the all-hit run writes is constant -- k ==
     * 0 forces prediction == actual at every position, and under a reset
     * tile every predictor's fixed point is one value -- so one byte per
     * tile decides whether the whole plane is that same value, and the
     * general path, which can code anything, clears the flag. The tile
     * path is otherwise untouched: the records it writes are replaced at
     * the end when the dense form is smaller. */
    dense_v = plane[0];
    dense_ok = 1;

    /* The block layer may already have established that this plane is one
     * byte value -- it has to know, to pick its CRC form -- and when it did
     * it records the pointer it tested. That pointer is only ever the
     * block's own input, i.e. the identity trial's plane, so another
     * candidate's staged plane can never inherit the answer. */
    plane_known_const = (e->const_plane == plane) ? 1 : 0;

    /* ---- A constant plane: the dense form, without writing a tile ----- *
     * The tile loop below writes ten bytes for every 256-byte tile of a
     * one-value plane (each is an all-hit record) and then `seg_dense_emit`
     * rewrites the payload as the dense form, which is twelve bytes per
     * 4 KiB. Both are cheap per tile, but 256 of them is ~5.5k cycles of
     * work whose only purpose is to be thrown away -- 20% of what a
     * constant block's encode costs after its CRC.
     *
     * So ask first: if the plane is one byte value and the dense form is
     * strictly smaller than the all-hit tiles it would replace (the same
     * comparison `seg_dense_emit` would make at the end of the loop), write
     * it now and return. The bytes and the statuses are exactly the ones
     * the loop would have produced: the tile size of a constant plane is
     * 10 bytes per tile by construction, `seg_dense_emit` is handed that
     * projected size as its `pos` so it applies its own `dtotal < pos`
     * rule, and the shortcut is skipped whenever the byte budget or the
     * payload capacity could have stopped the loop short of 16 + 10*tc
     * (where the old path could still have reached a different ending).
     *
     * `flat_paid` is set the way the loop's first all-hit tile sets it: a
     * 10-byte record for a 256-byte tile pays for itself, and the probe's
     * state spans the segments of one trial, so leaving it clear here could
     * arm the abort for a later segment that is not constant. */
    {
        uint32_t chunks = (plane_len + ZPSE_DENSE_MAX_LEN - 1u) /
                          ZPSE_DENSE_MAX_LEN;
        uint32_t dtotal = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE +
                          chunks * (uint32_t)ZPSE_DENSE_HEADER_SIZE;
        uint32_t tiles = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE + 10u * tc;

        if (dtotal < tiles && dtotal <= payload_cap &&
            (byte_budget == 0u || dtotal <= byte_budget) &&
            (plane_known_const != 0 || seg_plane_is_const(plane, plane_len))) {
            int dn = seg_dense_emit(&h, payload_out, plane, plane_len,
                                    initial_literal, tiles, byte_budget,
                                    payload_cap, 1);

            if (dn != 0) {
                e->flat_paid = 1u;
                return dn;
            }
        }
    }

    /* Dense palette form: a plane whose alphabet is small
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
    lit_len = 0u;
    lit_try = 1;
    lit_proj_at = 2u;
    lit_proj_shift = 1u;
    memset(rep_off, 0, sizeof rep_off);
    memset(rep_key, 0, sizeof rep_key);

    /* The projection is evaluated after 4, 8, 16, ... tiles coded, so the
     * extrapolation is a shift of a product rather than a 64-bit division
     * per tile (that division cost a 4 KiB text plane 4% of its encode).
     * Later checkpoints are also better evidence, and the exact rule above
     * the loop still fires the moment the tiles have overtaken the form. */
    proj_at = ZPSE_PAL_PROJ_TILES;
    proj_shift = 1u;
    pos = 16u;
    for (t = 0; t < tc; t++) {
        uint32_t seg_pos = t * 256u;
        uint16_t nt =
            (uint16_t)((t + 1u < tc) ? 256u : (plane_len - seg_pos));
        const uint8_t *actual = plane + seg_pos;

        /* The generation advances once a tile, on both sides, over the
         * same tiles in the same order: the encoder's winning trial and
         * the decoder's block. An ORDER2 entry is learned only if its
         * generation is this one, so this is the reset that replaced the
         * table wipe. */
        lanes->gen++;

        if (pal_len != 0u && pos >= pal_len) {                int dn = seg_dense_palette_emit(&h, payload_out, plane, plane_len,
                                                pal_m, pal_bits, pos, 0,
                                                byte_budget, payload_cap);

            if (dn != 0) {
                return dn;
            }
            pal_len = 0u; /* not emittable after all: keep tiling */
        }
        /* The literal form's price is exact, so this is a real comparison: the
         * moment the tiles written exceed it, the form is the smaller of the
         * two and the rest of the plane is work whose bytes would be thrown
         * away. */
        if (lit_len != 0u && pos >= lit_len) {
            int dn = seg_dense_literal_emit(&h, payload_out, plane, plane_len,
                                            pos, 0, byte_budget, payload_cap);

            if (dn != 0) {
                return dn;
            }
            lit_len = 0u;
        }
        uint16_t ks[12];
        uint8_t cand[5];
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
        int rep_slot = -1; /* repeat-cache slot this tile fills, or -1 */
        uint64_t rep_slot_key = 0u; /* and the key it is filled with */
        int ascii_ok; /* 1 iff every byte of the tile is below 0x80 */

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

        /* ---- Repeated tile --------------------------------------- *
         * A tile record is a pure function of the tile's bytes, the eight
         * plane bytes before it, its length and the segment literal.
         * Every tile here is classified with tile_reset == 1, so no state
         * carries across a boundary, and seg_pos reaches nothing else:
         * the STRIDE predictors read it, through exactly those eight
         * bytes. So when a tile repeats an earlier one with the same
         * head, its record IS that record, and copying it writes byte for
         * byte what the pipeline would have.
         *
         * The cache is direct-mapped on four words of the tile (offsets 0,
         * 64, 128 and 192, XORed into one product), one slot per 4 KiB of a
         * 64 KiB plane. That key is only a hint: the full 264-byte compare
         * (the tile and its head) decides, so a collision can only cost the
         * compare, never a wrong byte -- and the four words are chosen so a
         * tile of zeroes and a tile of one spike do not collide, which
         * keying on the leading bytes alone made them do (measured: the
         * sparse row lost 25% to compares that ran to the spike). A
         * periodic plane -- a ramp, a repeated 4 KiB block, a transposed
         * plane whose period is one tile -- repeats most of its tiles, and
         * each hit replaces a prescreen, a classify, a mode search and a
         * pack with one compare and one copy.
         *
         * Tile 0 is never a target or a source: its STRIDE reads fall back
         * to the literal instead of reading the plane, and its head is not
         * before it. The palette form is the one piece of tile-loop state a
         * copy would skip; it is priced once from a tile's own record
         * length, so the copy is only taken when that pricing could not
         * have run at this tile either -- either it has already happened
         * (pal_try == 0) or the record is too small to trigger it, exactly
         * as at the tile this record came from -- and never while the form
         * is still in play (pal_len != 0), which is the one state where the
         * payload written so far may be rewritten in place. */
        if (t != 0u && nt == (uint16_t)256u) {
            uint64_t w0;
            uint64_t w1;
            uint64_t w2;
            uint64_t w3;
            uint64_t key;
            unsigned slot;

            memcpy(&w0, actual, sizeof w0);
            memcpy(&w1, actual + 64, sizeof w1);
            memcpy(&w2, actual + 128, sizeof w2);
            memcpy(&w3, actual + 192, sizeof w3);
            key = (w0 ^ w1 ^ w2 ^ w3) * UINT64_C(0x9E3779B97F4A7C15);
            slot = (unsigned)(key >> ZPSE_REP_SHIFT);
            if (rep_off[slot] != 0u && rep_nt[slot] == (uint16_t)256u &&
                rep_key[slot] == key &&
                memcmp(plane + rep_off[slot] - 8u, actual - 8u, 264u) == 0 &&
                pal_len == 0u &&
                (pal_try == 0 ||
                 (uint32_t)rep_len[slot] < (uint32_t)(nt >> 3))) {
                uint32_t rpos = rep_pos[slot];
                uint32_t rlen = (uint32_t)rep_len[slot];

                if (pos + rlen > payload_cap) {
                    return seg_no_room(&h, payload_out, plane, plane_len,
                                       initial_literal, pos, byte_budget,
                                       payload_cap);
                }
                memcpy(payload_out + pos, payload_out + rpos, (size_t)rlen);
                /* Re-point the slot at this copy, so a run of repeats keeps
                 * finding the most recent one (and keeps the compare's
                 * second operand in the payload's hot lines). */
                rep_off[slot] = seg_pos;
                rep_pos[slot] = pos;
                rep_key[slot] = key;
                pos += rlen;
                /* A record with a mask or a residual is not a constant
                 * tile, which is what the general path records here. */
                dense_ok = 0;
                if (seg_flat_abort(e, nt, rlen, flat_abort_min)) {
                    return PSE_OUTPUT_OVERFLOW;
                }
                continue;
            }
            rep_slot = (int)slot;
            rep_slot_key = key;
        }

        /* ---- ASCII pack ------------------------------------------ *
         * One 32-word scan decides it (tile.c's zpse_tile_is_ascii). When
         * it holds, the tile has a form that costs seven bits a byte --
         * 224 bytes for a full tile, less 2 bytes of framing than the 256
         * the tile itself would take -- and no predictor here can be
         * trusted to beat that on the shapes that reach this point, which
         * is why the sweep below is skipped at the gate levels once the
         * prescreen confirms no predictor is close. `ascii_len` is also
         * what the comparison after the sweep uses at the ratio levels, so
         * the scan is not wasted there either. */
        /* A gated block has already had every one of its bytes scanned for
         * the high bit (zpse_ascii_scan over the whole block, which is what
         * let the gate fire), so the per-tile scan here would re-prove it 256
         * times per block. Skip it, and keep the per-tile test for the blocks
         * that did not gate. */
        ascii_ok = (e->ascii_gate != 0u) ? 1
                   : ((nt >= 8u && zpse_tile_is_ascii(actual, nt)) ? 1 : 0);
        /* The block encoder's gate: its samples have already shown that
         * no cheap shape beats seven bits a byte on this block, so the
         * record is written here and the prescreen, the classify, the mode
         * search and the pack are skipped for the whole block. That is the
         * difference between a text block costing a full sweep per tile
         * and costing a scan and a bit pack -- and the form is the same
         * bytes either way, so the stream is unchanged. */
        if (ascii_ok && e->ascii_gate != 0u) {
            uint32_t alen = zpse_tile_ascii_len(nt);

            if (pos + 2u + alen > payload_cap) {
                return seg_no_room(&h, payload_out, plane, plane_len,
                                   initial_literal, pos, byte_budget,
                                   payload_cap);
            }
            zpse_store_u16le(payload_out + pos, (uint16_t)alen);
            (void)zpse_tile_pack_ascii(actual, nt, payload_out + pos + 2u,
                                       payload_cap - pos - 2u);
            pos += 2u + alen;
            dense_ok = 0;
            if (seg_flat_abort(e, nt, 2u + alen, flat_abort_min)) {
                return PSE_OUTPUT_OVERFLOW;
            }
            continue;
        }

        /* Opt-in trial accounting (-DZPSE_TRIAL_STATS, see tile.c).
         * Counted here, not at the top of the loop, so the denominator is
         * the tiles whose predictor search actually ran: a tile the
         * all-hit run or the repeat cache wrote without one contributes
         * no trials and no tile. "Trials per tile" below is therefore
         * trials per SEARCHED tile. */
#if defined(ZPSE_TRIAL_STATS)
        zpse_trial_search_tiles++;
#endif
        seg_prescreen_tile(actual, nt, plane, seg_pos, initial_literal,
                           zpse_enc_speed(e->enc_flags) <= 2u, have_avx2,
                           &zero_cheap, ks);
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
                /* Second trial only on an exact tie in the prescreen's
                 * miss count, and never when the best candidate is
                 * already perfect: bk == 0 selects mask ALL_HIT with
                 * residual EXCLUSION_U8, whose payloads are both empty,
                 * so zpse_tile_pack returns exactly ZPSE_TILE_HEADER_SIZE
                 * -- the smallest record that can exist. The candidate
                 * loop replaces the winner only on a strictly smaller
                 * packed size, which no second candidate can be against
                 * an 8-byte record, so appending one here could not have
                 * changed the output.
                 *
                 * The tie is exact because a miss count is a poor price
                 * of the record: two predictors that miss one position
                 * each usually pack to the same size, so a candidate one
                 * miss behind almost never wins, while classifying and
                 * packing it costs a full tile. With the old +4 slack
                 * every sparse-plane tile paid for two of everything
                 * (measured: 1.03 -> 1.63 GB/s on the sparse row), and
                 * the 768-case clen corpus came out byte-identical
                 * without it. */
                if (bk != 0u && second_p != 0xFFu && sk == bk) {
                    cand[ncand++] = second_p;
                }
            }
            /* ORDER1 and ORDER2 are the two predictors that read a
             * table (order1, and o2), and they are not in the
             * prescreen's candidate list, so they can never win a tile.
             * They can be tried when the screened predictors do badly.
             *
             * At most ONE of them may be in the list. A classify leaves
             * the table it reads in the state its own chain produced, and
             * the table is only re-seeded for the predictor the NEXT
             * classify is for. So with both in the list, the loser's
             * classify leaves the winner's table dirty -- and the
             * decoder, which applies only the winner, never dirties it.
             * The two sides then disagree about the table, and the block
             * decodes to different bytes (measured: the decimal8 corpus
             * failed to round-trip, the differing bytes at the tail of
             * the tiles of the later segments).
             *
             * One table user with anything else is fine: the other
             * candidates read no table, and the history fixup at the end
             * of a tile depends on the bytes, not on the predictor. */
#if defined(ZPSE_TRIAL_ORDER1)
            if (bk * 2u > (uint32_t)nt) {
                if (ncand < 4u) {
                    cand[ncand++] = (uint8_t)ZPSE_P_ORDER1;
                }
            }
#endif
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
                if (!seg_run_stays(plane + nsseg_pos, nnt, pred,
                                   initial_literal, nsseg_pos, plane,
                                   have_avx2)) {
                    uint16_t nbk;
                    uint32_t i;

                    npred = seg_screen_pred[0];
                    seg_prescreen(plane + nsseg_pos, nnt, plane, nsseg_pos,
                                  initial_literal,
                                  zpse_enc_speed(e->enc_flags) <= 2u,
                                  have_avx2, nks);
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
            int rc = zpse_tile_classify_lean(lanes, actual, nt, pred,
                                             initial_literal, 1, plane,
                                             seg_pos, &cl);
            if (rc != PSE_OK) {
                continue;
            }
            zpse_tile_choose_modes(&cl, nt, &mm, &rr);
            /* Only the modes that read the predictions pay for them. */
            if (rr == (uint8_t)ZPSE_R_PALETTE_BITS ||
                rr == (uint8_t)ZPSE_R_PACKED_BASE_M ||
                rr == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                (void)zpse_tile_fill_at_pred_reset(pred, actual, nt, plane,
                                                   seg_pos, initial_literal,
                                                   &cl);
            }
#if defined(ZPSE_TRIAL_STATS)
            zpse_stats_modes[((uint32_t)mm << 3) | (uint32_t)rr]++;
#endif
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
        }
        /* seg_pred_wins[best_p]++; -- disabled for debugging */

        if (best_len < 0) {
            return PSE_BAD_RESIDUAL;
        }
        if (pos + 2u > payload_cap ||
            (uint32_t)best_len > payload_cap - pos - 2u) {
            return seg_no_room(&h, payload_out, plane, plane_len,
                               initial_literal, pos, byte_budget,
                               payload_cap);
        }
        /* The ratio levels take the seven-bit form when it is smaller than
         * what the sweep found; on text it always is, on structured ASCII
         * the sweep wins and this costs one compare. */
        if (ascii_ok && zpse_tile_ascii_len(nt) < (uint32_t)best_len) {
            best_len = (int)zpse_tile_ascii_len(nt);
            (void)zpse_tile_pack_ascii(actual, nt, payload_out + pos + 2u,
                                       payload_cap - pos - 2u);
        } else if (!direct) {
            memcpy(payload_out + pos + 2u, bst, (size_t)best_len);
        }
        zpse_store_u16le(payload_out + pos, (uint16_t)best_len);
        if (rep_slot >= 0) {
            rep_off[rep_slot] = seg_pos;
            rep_pos[rep_slot] = pos;
            rep_len[rep_slot] = (uint16_t)(2u + (uint32_t)best_len);
            rep_nt[rep_slot] = nt;
            rep_key[rep_slot] = rep_slot_key;
        }
        pos += 2u + (uint32_t)best_len;

        /* The palette form is priced from the first tile
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
#if defined(ZPSE_TRIAL_STATS)
                zpse_stats_palettable++;
#endif

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
        }

        /* ---- Dense literal form -------------------------------- *
         * A plane with a wide alphabet and no structure to predict has no
         * form smaller than its own bytes, and the tile path spends ten
         * bytes of framing and a nine-predictor search per 256 bytes to
         * discover that -- record for record, ~6% larger than the bytes it
         * codes. The dense literal form stores the plane with one 12-byte
         * header per 4 KiB instead, so it is both smaller and a copy at
         * decode (see seg_dense_literal_emit).
         *
         * Priced on the same evidence as the palette form -- a tile that is
         * losing by more than 8x cannot be paying for itself -- and only
         * when the palette form is not on the table: a plane of at most
         * sixteen values is always smaller as a palette than as a literal,
         * so that plane keeps the form it has always had. `lit_len` is
         * exact, so the comparisons below are real size comparisons rather
         * than the upper bound the palette's price is. */
        if (lit_try != 0 && pal_len == 0u &&
            2u + (uint32_t)best_len >= (uint32_t)(nt >> 3)) {
            uint32_t lc = seg_dense_literal_cost(plane_len);

            lit_try = 0;
            if (lc <= payload_cap &&
                (byte_budget == 0u || lc <= byte_budget)) {
                lit_len = lc;
            }
        }

        /* The form is priced and the tiles have not caught up with it. Ask
         * the tiles ALREADY CODED whether they are going to: if the plane's
         * tile total is going to be larger than the palette by a margin, the
         * rest of the loop is work whose bytes the form would throw away, so
         * emit it now and return. The margin (an eighth) is what keeps a
         * mispriced plane -- tiles that differ region by region -- from
         * losing more bytes than the guess saved; below it the exact rule
         * still runs the tiles and compares, as it always did.
         *
         * This used to be a separate sample of ZPSE_EST_TILES tiles, each
         * run through the whole classify/mode-search/pack pipeline. That is
         * a third of a plane's tile work paid on top of the tiles
         * themselves, and it is pure loss on the planes where the tiles do
         * win: palette16 (4-bit data) measured 24 tile pipelines per 16
         * tiles, 8 of them the sample, for a decision its own first tiles
         * had already made. The prefix below is exactly the evidence the
         * sample was trying to obtain -- per-tile bytes, from real tiles --
         * and it is free, because those tiles are coded either way. Coding
         * a few more of them before deciding costs a losing palette plane a
         * handful of pipelines; it saves every winning plane the sample. */
        if (pal_len != 0u && t + 1u == proj_at) {
            const uint32_t hdr = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;
            uint32_t proj = hdr +
                            (uint32_t)(((uint64_t)(pos - hdr) * tc) >>
                                       proj_shift);
            int dn;

            proj_shift++;
            proj_at <<= 1;
            if (pos < pal_len && proj - (proj >> 3) >= pal_len) {
                dn = seg_dense_palette_emit(&h, payload_out, plane,
                                            plane_len, pal_m, pal_bits,
                                            pos, 1, byte_budget,
                                            payload_cap);
                if (dn != 0) {
                    return dn;
                }
            }
        }

        /* The literal form's own projection, from HALF the plane rather than
         * a quarter and on its own schedule: its tiles are the same size to
         * within a byte or two (a random plane has no regions), so two tiles
         * already say what the rest will cost, and every tile coded before
         * the form is taken is a pipeline whose bytes are thrown away. The
         * margin is a thirty-second rather than the palette's eighth because
         * this form wins by a few percent, not by a quarter -- an eighth
         * would never let it fire. */
        if (lit_len != 0u && pos < lit_len && t + 1u == lit_proj_at) {
            const uint32_t hdr = (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;
            uint32_t proj = hdr +
                            (uint32_t)(((uint64_t)(pos - hdr) * tc) >>
                                       lit_proj_shift);
            int dn;

            lit_proj_shift++;
            lit_proj_at <<= 1;
            if (proj - (proj >> 5) >= lit_len) {
                dn = seg_dense_literal_emit(&h, payload_out, plane, plane_len,
                                            pos, 1, byte_budget, payload_cap);
                if (dn != 0) {
                    return dn;
                }
            }
        }
        if (seg_flat_abort(e, nt, 2u + (uint32_t)best_len, flat_abort_min)) {
            return PSE_OUTPUT_OVERFLOW;
        }
    }

    /* The plane may be one byte value end to end, which the dense record
     * form codes in 12 bytes per 4 KiB instead of the ten
     * bytes per 256-byte tile the run path above just wrote. The rewrite
     * only ever shrinks the record area, so every bound checked above still
     * holds. Offered first so a constant plane keeps the form it has
     * always had. */
    {
        int dn = seg_dense_emit(&h, payload_out, plane, plane_len,
                                initial_literal, pos, byte_budget,
                                payload_cap,
                                (dense_ok || plane_known_const) ? 1 : 0);        if (dn != 0) {
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
    /* Same for the literal form: offered last so a plane that is constant or
     * has a small alphabet keeps the form it has always had, and only taken
     * when every tile record written turned out to cost more than a copy of
     * the plane. */
    if (lit_len != 0u) {
        int dn = seg_dense_literal_emit(&h, payload_out, plane, plane_len,
                                        pos, 0, byte_budget, payload_cap);

        if (dn != 0) {
            return dn;
        }
    }
    return (int)pos;
}
