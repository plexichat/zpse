#include "zpse/block.h"

#include "zpse/arena.h"
#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/predictor.h"
#include "zpse/segment.h"
#include "zpse/status.h"
#include "zpse/transform.h"
#include "zpse/types.h"

#include <string.h>

/* Block envelope codec (SPEC Sec. 4, 5, 11).
 * No allocation; staging uses caller arenas. Helpers static. */

static int zpse_block_scount_valid(uint8_t sc)
{
    return sc == 1 || sc == 2 || sc == 4 || sc == 8 || sc == 16;
}

/* Fix up header CRC bytes 28..31 after serializing. */
static void zpse_block_fix_header_crc(uint8_t hdr[32])
{
    uint32_t c = zpse_crc32c(hdr, 28);
    zpse_store_u32le(hdr + 28, c);
}

static void zpse_block_fill_header(ZpseBlockHeader *h, uint8_t flags,
                                   uint32_t n, uint32_t c, uint8_t tid,
                                   uint8_t sc, uint16_t meta_len,
                                   uint32_t seq, uint32_t decoded_crc)
{
    h->sync = ZPSE_BLOCK_SYNC;
    h->block_version = ZPSE_BLOCK_VERSION;
    h->block_flags = flags;
    h->uncompressed_size = n;
    h->compressed_block_size = c;
    h->transform_id = tid;
    h->segment_count = sc;
    h->transform_meta_size = meta_len;
    h->tile_size = (uint16_t)ZPSE_TILE_SIZE;
    h->predictor_profile = 0;
    h->reserved0 = 0;
    h->block_sequence_number = seq;
    h->decoded_crc32c = decoded_crc;
    h->header_crc32c = 0;
}

int zpse_block_probe(const uint8_t *in, uint32_t in_len,
                     uint32_t *compressed_size_out,
                     uint32_t *uncompressed_size_out)
{
    ZpseBlockHeader h;
    int rc;

    if (in == 0 || compressed_size_out == 0 || uncompressed_size_out == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    rc = zpse_block_header_parse(&h, in, in_len);
    if (rc != PSE_OK) {
        return rc;
    }
    if (h.compressed_block_size < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (in_len < h.compressed_block_size) {
        return PSE_TRUNCATED;
    }
    *compressed_size_out = h.compressed_block_size;
    *uncompressed_size_out = h.uncompressed_size;
    return PSE_OK;
}

int zpse_block_decode(ZpseDecoder *d, const uint8_t *in, uint32_t in_len,
                      uint8_t *output, uint32_t out_cap,
                      uint32_t *out_len_out)
{
    ZpseBlockHeader h;
    ZpseTransformMeta meta;
    uint32_t n, c, meta_size, scount;
    uint64_t dir_end, area;
    const uint8_t *dir, *payloads;
    uint32_t seg_lens[16];
    uint8_t init_lit[16];
    const uint8_t *csegs[16];
    uint32_t cum, s;
    int identity_shape;
    int rc;

    if (d == 0 || in == 0 || output == 0) {
        return PSE_BAD_ARG;
    }
    if (d->recon == 0 || d->out_stage == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    rc = zpse_block_header_parse(&h, in, in_len);
    if (rc != PSE_OK) {
        return rc;
    }
    /* Header CRC defence in depth (parse is expected to check it). */
    if (zpse_crc32c(in, 28) != h.header_crc32c) {
        return PSE_BAD_HEADER_CRC;
    }

    n = h.uncompressed_size;
    c = h.compressed_block_size;
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (c < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (in_len < c) {
        return PSE_TRUNCATED;
    }

    if ((h.block_flags & ZPSE_B_RESERVED_MASK) != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if ((h.block_flags & ZPSE_B_RESET) == 0) {
        return PSE_BAD_HEADER;
    }
    if (h.reserved0 != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (h.predictor_profile != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (h.tile_size != (uint16_t)ZPSE_TILE_SIZE) {
        return PSE_BAD_TILE_SIZE;
    }
    if (h.transform_id >= 12) {
        return PSE_BAD_TRANSFORM;
    }
    if (!zpse_block_scount_valid(h.segment_count)) {
        return PSE_BAD_TRANSFORM;
    }
    meta_size = h.transform_meta_size;
    {
        int want_meta = (h.block_flags & ZPSE_B_HAS_META) != 0;
        int have_meta = meta_size > 0;
        if (want_meta != have_meta) {
            return PSE_BAD_HEADER;
        }
    }

    if (out_cap < n) {
        return PSE_OUTPUT_OVERFLOW;
    }

    /* RAW fallback path: exactly N raw bytes at offset 32. */
    if ((h.block_flags & ZPSE_B_RAW) != 0) {
        uint32_t calc;
        if (h.transform_id != 0 || h.segment_count != 1 || meta_size != 0) {
            return PSE_BAD_HEADER;
        }
        if (c != (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
            return PSE_BAD_BLOCK_SIZE;
        }
        if ((h.block_flags & ZPSE_B_CRC) != 0) {
            /* Fused copy + checksum. The fallback payload is a verbatim
             * copy of the input, so a single pass both fills the caller's
             * buffer and returns the CRC of exactly the bytes copied --
             * no second sweep over 64 KiB that has already left L1. */
            calc = zpse_crc32c_copy(output, in + ZPSE_BLOCK_HEADER_SIZE, n);
            if (calc != h.decoded_crc32c) {
                return PSE_BAD_PAYLOAD_CRC;
            }
        } else {
            memcpy(output, in + ZPSE_BLOCK_HEADER_SIZE, n);
        }
        if (out_len_out != 0) {
            *out_len_out = n;
        }
        return (int)n;
    }

    /* Coded path: structural directory checks (PHASE1). */
    scount = h.segment_count;
    dir_end = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)meta_size +
              (uint64_t)scount * (uint64_t)ZPSE_SEG_DIR_ENTRY_SIZE;
    if (dir_end > (uint64_t)c) {
        return PSE_BAD_SEGMENT_DIRECTORY;
    }
    rc = zpse_transform_meta_parse(&meta, h.transform_id,
                                   in + ZPSE_BLOCK_HEADER_SIZE,
                                   (uint16_t)meta_size, h.segment_count);
    if (rc != PSE_OK) {
        return rc;
    }
    dir = in + ZPSE_BLOCK_HEADER_SIZE + meta_size;
    payloads = in + dir_end;
    area = (uint64_t)c - dir_end;

    {
        uint32_t expect = 0;
        for (s = 0; s < scount; s++) {
            uint32_t off = zpse_load_u32le(dir + (size_t)s * 8);
            uint32_t sz = zpse_load_u32le(dir + (size_t)s * 8 + 4);
            if (off != expect) {
                return PSE_BAD_SEGMENT_DIRECTORY;
            }
            if ((uint64_t)off + (uint64_t)sz > area) {
                return PSE_BAD_SEGMENT_DIRECTORY;
            }
            if (sz < (uint32_t)ZPSE_SEGMENT_HEADER_SIZE) {
                return PSE_BAD_SEGMENT_DIRECTORY;
            }
            expect = off + sz;
        }
        if ((uint64_t)expect != area) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
    }

    /* Per-segment decoded sizes from segment headers. */
    {
        uint64_t sum = 0;
        for (s = 0; s < scount; s++) {
            ZpseSegmentHeader sh;
            uint32_t off = zpse_load_u32le(dir + (size_t)s * 8);
            uint32_t sz = zpse_load_u32le(dir + (size_t)s * 8 + 4);
            rc = zpse_segment_header_parse(&sh, payloads + off, sz);
            if (rc != PSE_OK) {
                return rc;
            }
            if (sh.segment_decoded_size == 0) {
                return PSE_BAD_SEGMENT_DIRECTORY;
            }
            seg_lens[s] = sh.segment_decoded_size;
            init_lit[s] = sh.initial_literal;
            sum += (uint64_t)sh.segment_decoded_size;
        }
        if (sum != (uint64_t)n) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
    }

    rc = zpse_transform_validate_sizes(h.transform_id, h.segment_count, n,
                                       seg_lens);
    if (rc != PSE_OK) {
        return rc;
    }

    /* A single-segment ZPSE_T_RAW_BYTES block has an inverse transform that
     * is exactly `memcpy(out, segs[0], n)`: zpse_transform_meta_parse only
     * accepts that id with segment_count == 1, an empty metadata region and
     * width_W == 1, and the delta/xor wrap is keyed on ids 1, 2, 7..10 and
     * 11-with-flag-bit-0, so id 0 is neither. The plane for such a block is
     * therefore the caller's buffer itself: decoding the segment straight
     * into it produces the same bytes the copy would have produced, and
     * PHASE4 then becomes a copy of a buffer onto itself, which the
     * transform's own overlap guard refuses. So for this shape the copy is
     * not just redundant, it is not expressible at all - and it is the
     * shape the encoder's identity trial produces, i.e. the common case for
     * a smooth or constant block. Nothing the inverse would have validated
     * is skipped: this function already established that meta, segs and out
     * are non-null, that n is in [ZPSE_MIN_BLOCK_SIZE,
     * ZPSE_MAX_BLOCK_SIZE], that the width agrees with the parse, and that
     * zpse_transform_validate_sizes() passes for exactly these arguments;
     * the overlap guard exists to protect the copy that no longer happens. */
    identity_shape = (h.transform_id == (uint8_t)ZPSE_T_RAW_BYTES &&
                      scount == 1u) ? 1 : 0;

    /* PHASE3: decode segments into recon planes (or straight into the
     * caller's buffer for the identity shape above). */
    {
        ZpseLaneState lanes;
        lanes.order1 = d->order1;
        lanes.o2val = d->o2val;
        lanes.o2conf = d->o2conf;
        if (lanes.order1 == 0 || lanes.o2val == 0 || lanes.o2conf == 0) {
            return PSE_BAD_ARG;
        }
        cum = 0;
        for (s = 0; s < scount; s++) {
            uint32_t off = zpse_load_u32le(dir + (size_t)s * 8);
            uint32_t sz = zpse_load_u32le(dir + (size_t)s * 8 + 4);
            uint8_t *plane = d->recon + cum;
            uint32_t got = 0;

            if (identity_shape != 0) {
                plane = output;
            }
            /* One full init per segment: zpse_lane_reset covers
             * histories + order1/o2val/o2conf (same values as the
             * memsets it replaces). Segment predictors vary per
             * tile, so no selective reset here. */
            zpse_lane_reset(&lanes, init_lit[s]);
            if ((uint64_t)cum + (uint64_t)seg_lens[s] > ZPSE_MAX_BLOCK_SIZE) {
                return PSE_BAD_SEGMENT_DIRECTORY;
            }
            rc = zpse_segment_decode(d, &lanes, payloads + off, sz,
                                     plane, seg_lens[s], &got);
            if (rc != PSE_OK) {
                return rc;
            }
            if (got != seg_lens[s]) {
                return PSE_BAD_SEGMENT_DIRECTORY;
            }
            csegs[s] = plane;
            cum += seg_lens[s];
        }
        if (cum != n) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
    }

    /* PHASE4: inverse transform straight into the caller's buffer (the
     * segment planes live in d->recon, so there is no overlap and the
     * 64 KiB staging copy is unnecessary). The identity shape above has no
     * transform left to apply - its plane IS the output - so this phase is
     * skipped for it, which is also the only way to express it (see the
     * identity_shape comment). */
    if (identity_shape == 0) {
        rc = zpse_transform_inverse(&meta, h.segment_count, csegs, seg_lens,
                                    output, n);
        if (rc != PSE_OK) {
            return rc;
        }
    }

    /* PHASE5: payload CRC iff the block flag tags it. */
    if ((h.block_flags & ZPSE_B_CRC) != 0) {
        uint32_t calc = zpse_crc32c(output, n);
        if (calc != h.decoded_crc32c) {
            return PSE_BAD_PAYLOAD_CRC;
        }
    }

    if (out_len_out != 0) {
        *out_len_out = n;
    }
    return (int)n;
}

/* Encode one transform trial into dst[0..cap). Returns PSE_OK with
 * *total_out set, or a negative status / skip code. *budget_hit is set
 * when the trial was stopped by the RAW-envelope budget below rather
 * than by a real error, because the caller has to count that case
 * exactly like a completed trial that failed to beat RAW. */
static int zpse_block_try_one(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                              uint32_t in_crc,
                              uint8_t tid, uint8_t sc,
                              const uint8_t *meta_bytes, uint16_t meta_len,
                              int is_last, uint32_t seq,
                              uint8_t *dst, uint32_t cap, uint32_t *total_out,
                              int *budget_hit)
{
    /* A trial is only ever chosen when its total is below the RAW
     * envelope (32 + N), so it cannot matter once it has reached it:
     * every payload byte written is committed and later segments only add
     * to the total. Handing the segment encoder that remaining budget
     * lets it stop instead of finishing a payload that is thrown away.
     * The envelope is also a strict upper bound on the budget, and the
     * cap the caller passes is the candidate buffer, which is larger than
     * any envelope a valid block can have, so an overflow with an active
     * budget is always the budget -- never the cap. */
    ZpseTransformMeta tm;
    uint8_t w = 0;
    uint32_t lens[16];
    uint32_t seg_sizes[16];
    uint8_t *planes[16];
    uint64_t base, cum;
    uint32_t s, total;
    ZpseBlockHeader h;
    ZpseLaneState lanes;
    uint8_t flags;
    int rc;

    if (budget_hit != 0) {
        *budget_hit = 0;
    }
    rc = zpse_transform_width(tid, sc, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_segment_lengths(n, w, lens);
    if (rc != PSE_OK) {
        return rc;
    }
    /* Clamping: N < W forbidden; empty segments forbidden. */
    for (s = 0; s < sc; s++) {
        if (lens[s] == 0) {
            return PSE_BAD_TRANSFORM;
        }
    }
    rc = zpse_transform_meta_parse(&tm, tid, meta_bytes, meta_len, sc);
    if (rc != PSE_OK) {
        return rc;
    }
    {
        uint64_t sum = 0;
        uint32_t off = 0;
        for (s = 0; s < sc; s++) {
            planes[s] = e->tstage + off;
            if ((uint64_t)off + (uint64_t)lens[s] > ZPSE_E_SZ_TSTAGE) {
                return PSE_OUTPUT_OVERFLOW;
            }
            off += lens[s];
            sum += (uint64_t)lens[s];
        }
        if (sum != (uint64_t)n) {
            return PSE_BAD_TRANSFORM;
        }
    }
    rc = zpse_transform_forward(&tm, sc, in, n, planes, lens);
    if (rc != PSE_OK) {
        return rc;
    }

    base = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)meta_len +
           (uint64_t)sc * (uint64_t)ZPSE_SEG_DIR_ENTRY_SIZE;
    if (base > (uint64_t)cap) {
        return PSE_OUTPUT_OVERFLOW;
    }
    lanes.order1 = e->order1;
    lanes.o2val = e->o2val;
    lanes.o2conf = e->o2conf;
    if (lanes.order1 == 0 || lanes.o2val == 0 || lanes.o2conf == 0) {
        return PSE_BAD_ARG;
    }
    cum = 0;
    for (s = 0; s < sc; s++) {
        uint8_t lit = lens[s] > 0 ? planes[s][0] : 0;
        uint64_t at = base + cum;
        uint64_t raw = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n;
        uint32_t room;
        uint32_t budget;
        int wlen;
        if (at > (uint64_t)cap) {
            return PSE_OUTPUT_OVERFLOW;
        }
        room = (uint32_t)((uint64_t)cap - at);
        budget = (raw > at) ? (uint32_t)(raw - at) : 0u;
        wlen = zpse_segment_encode(e, &lanes, planes[s], lens[s], lit,
                                   dst + at, room, budget);
        if (wlen < 0) {
            if (wlen == PSE_OUTPUT_OVERFLOW && budget != 0u &&
                budget_hit != 0) {
                *budget_hit = 1;
            }
            return wlen;
        }
        if (wlen < (int)ZPSE_SEGMENT_HEADER_SIZE) {
            return PSE_BAD_TILE_SIZE;
        }
        seg_sizes[s] = (uint32_t)wlen;
        cum += (uint64_t)wlen;
    }
    if (base + cum > (uint64_t)cap) {
        return PSE_OUTPUT_OVERFLOW;
    }
    total = (uint32_t)(base + cum);

    if (meta_len > 0) {
        memcpy(dst + ZPSE_BLOCK_HEADER_SIZE, meta_bytes, meta_len);
    }
    for (s = 0; s < sc; s++) {
        uint32_t off = 0;
        uint32_t k;
        for (k = 0; k < s; k++) {
            off += seg_sizes[k];
        }
        zpse_store_u32le(dst + ZPSE_BLOCK_HEADER_SIZE + meta_len + (size_t)s * 8,
                          off);
        zpse_store_u32le(dst + ZPSE_BLOCK_HEADER_SIZE + meta_len + (size_t)s * 8 + 4,
                          seg_sizes[s]);
    }

    flags = (uint8_t)(ZPSE_B_RESET | ZPSE_B_CRC);
    if (meta_len > 0) {
        flags |= ZPSE_B_HAS_META;
    }
    if (is_last) {
        flags |= ZPSE_B_LAST;
    }
    /* The decoded CRC covers the uncompressed input, so it is a property
     * of (in, n) and not of the trial -- the caller computes it once and
     * every candidate header reuses it. */
    zpse_block_fill_header(&h, flags, n, total, tid, sc, meta_len, seq,
                           in_crc);
    zpse_block_header_serialize(dst, &h);
    zpse_block_fix_header_crc(dst);
    *total_out = total;
    return PSE_OK;
}

/* Smallest size a trial with this (transform, segment count, metadata)
 * shape can possibly produce: the envelope (block header, metadata, the
 * segment directory, each segment's 16-byte header) plus 2 length-prefix
 * bytes and the 8-byte tile header for every tile. Used to stop the
 * transform sweep as soon as nothing left can beat the running best. */
static uint32_t zpse_block_floor(uint32_t n, uint8_t tid, uint8_t sc,
                                 uint16_t meta_len)
{
    uint32_t lens[16];
    uint32_t tiles = 0u;
    uint8_t w = 0;
    uint8_t s;

    if (zpse_transform_width(tid, sc, &w) != PSE_OK) {
        return UINT32_MAX;
    }
    if (zpse_segment_lengths(n, w, lens) != PSE_OK) {
        return UINT32_MAX;
    }
    for (s = 0; s < sc; s++) {
        tiles += (lens[s] + 255u) / 256u;
    }
    return (uint32_t)ZPSE_BLOCK_HEADER_SIZE + (uint32_t)meta_len +
           (uint32_t)sc * ((uint32_t)ZPSE_SEG_DIR_ENTRY_SIZE +
                           (uint32_t)ZPSE_SEGMENT_HEADER_SIZE) +
           10u * tiles;
}

int zpse_block_encode(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                      uint32_t seq, int is_last,
                      uint8_t *out, uint32_t out_cap)
{
    uint8_t tids[4];
    uint8_t scounts[4];
    uint8_t meta_bufs[4][2];
    uint16_t meta_lens[4];
    uint32_t in_crc;
    int ncand = 0;
    uint32_t best_total = 0;
    int have_best = 0;
    int fails = 0;
    int i;
    int rc;

    if (e == 0 || in == 0 || out == 0) {
        return PSE_BAD_ARG;
    }
    if (e->tstage == 0 || e->cand == 0 || e->best == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }

    /* Trial 1: RAW_BYTES (the identity, as a single segment).
     *
     * Ordered FIRST because it carries the smallest envelope of any
     * candidate -- one segment and no transform metadata -- so it is the
     * one that can reach the absolute floor of 32 + 8 + 16 + 10 per tile.
     * Reaching a candidate's floor is what lets the sweep stop, so a
     * constant or all-one block costs exactly one trial. */
    tids[0] = 0;
    scounts[0] = 1;
    meta_lens[0] = 0;
    ncand = 1;

    /* Trial 2: DELTA8 with seed = first input byte. */
    tids[ncand] = 1;
    scounts[ncand] = 1;
    meta_bufs[ncand][0] = in[0];
    meta_lens[ncand] = 1;
    ncand++;

    /* Trial 3: width-appropriate plain transpose. */
    if (n >= 16) {
        tids[ncand] = 6;
        scounts[ncand] = 16;
        meta_lens[ncand] = 0;
        ncand++;
    } else if (n >= 8) {
        tids[ncand] = 5;
        scounts[ncand] = 8;
        meta_lens[ncand] = 0;
        ncand++;
    } else if (n >= 4) {
        tids[ncand] = 4;
        scounts[ncand] = 4;
        meta_lens[ncand] = 0;
        ncand++;
    } else if (n >= 2) {
        tids[ncand] = 3;
        scounts[ncand] = 2;
        meta_lens[ncand] = 0;
        ncand++;
    }

    /* Every candidate writes the same decoded CRC into its header (it is
     * the CRC of the uncompressed block), so it is computed once here
     * instead of once per trial over the same 64 KiB. */
    in_crc = zpse_crc32c(in, n);

    for (i = 0; i < ncand; i++) {
        uint32_t t = 0;
        int budget_hit = 0;
        const uint8_t *mb = meta_lens[i] > 0 ? meta_bufs[i] : in;
        rc = zpse_block_try_one(e, in, n, in_crc, tids[i], scounts[i], mb,
                                meta_lens[i], is_last, seq,
                                e->cand, (uint32_t)ZPSE_E_SZ_CAND, &t,
                                &budget_hit);
        if (rc != PSE_OK) {
            /* A trial the budget stopped had already reached the RAW
             * envelope, which is exactly the condition the completed
             * trial's `t + 1 > 32 + N` test below would have counted, so
             * count it here and the sweep decides as it always did. */
            if (budget_hit) {
                fails++;
                if (fails >= 2) {
                    break;
                }
            }
            continue;
        }
        if (!have_best || t < best_total) {
            if (t > (uint32_t)ZPSE_E_SZ_BEST) {
                continue;
            }
            memcpy(e->best, e->cand, t);
            best_total = t;
            have_best = 1;
        }
        /* Early exit over the candidates that are LEFT, not over all of
         * them: a trial whose own envelope is bigger (16 segments, or a
         * metadata byte) cannot reach a size the identity trial can, so
         * comparing the running best against each remaining candidate's
         * own floor stops the sweep as soon as nothing left could be
         * strictly smaller. That is why the identity runs first -- it is
         * the only candidate that can reach the global minimum, so for
         * constant/ramp blocks the sweep ends here after one trial while a
         * ramp whose winner is DELTA8 still gets its second trial. */
        {
            uint32_t min_floor = UINT32_MAX;
            int j;

            for (j = i + 1; j < ncand; j++) {
                uint32_t f = zpse_block_floor(n, tids[j], scounts[j],
                                              meta_lens[j]);
                if (f < min_floor) {
                    min_floor = f;
                }
            }
            if (have_best && best_total <= min_floor) {
                break;
            }
            /* Two candidates that both fail to beat the raw envelope
             * mean the block is incompressible under every transform
             * shape tried; the raw fallback wins and the remaining
             * trials are pure cost. */
            if (t + 1u > (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
                fails++;
                if (fails >= 2) {
                    break;
                }
            }
        }
    }

    /* RAW fallback when coded does not pay: coded >= 32 + N. */
    if (!have_best || (uint64_t)best_total >=
                      (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n) {
        uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n;
        ZpseBlockHeader h;
        uint8_t flags = (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET | ZPSE_B_CRC);
        uint32_t decoded_crc;
        if (is_last) {
            flags |= ZPSE_B_LAST;
        }
        if (out_cap < total) {
            return PSE_OUTPUT_OVERFLOW;
        }
        decoded_crc = in_crc;
        zpse_block_fill_header(&h, flags, n, total, 0, 1, 0, seq,
                               decoded_crc);
        zpse_block_header_serialize(out, &h);
        zpse_block_fix_header_crc(out);
        memcpy(out + ZPSE_BLOCK_HEADER_SIZE, in, n);
        return (int)total;
    }

    if (out_cap < best_total) {
        return PSE_OUTPUT_OVERFLOW;
    }
    memcpy(out, e->best, best_total);
    return (int)best_total;
}
