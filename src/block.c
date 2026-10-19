#include "zpse/block.h"

#include "zpse/arena.h"
#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/dense.h"
#include "zpse/predictor.h"
#include "zpse/segment.h"
#include "zpse/status.h"
#include "zpse/tile.h"
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
            /* No lane reset here. zpse_segment_decode established the
             * segment's entry state itself, from the initial_literal of
             * the very header this loop just parsed the decoded size out
             * of, so resetting from init_lit[s] here was the same state
             * memoized a second time in front of every segment -- and the
             * expensive half of it (the ORDER tables) 66 KiB a segment.
             * The histories and the lazily seeded tables that
             * zpse_segment_decode sets up are exactly what this call would
             * have produced, and it can see everything it needs: lanes
             * below already points at the decoder's own tables. */
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
        uint32_t calc;

        /* A one-value plane is the dense constant form (Sec. 6.3), and
         * only the identity shape reaches here with the decoded plane
         * already in the caller's buffer, so d->seg_const describes
         * exactly the bytes output[0..n). Its CRC is O(log n) matrix
         * applies instead of a 64 KiB pass over the buffer the decoder
         * just wrote -- and that pass is most of the time a constant
         * block takes to decode. Nothing is skipped on a claim the
         * decoder made up: seg_const is only ever set by a record whose
         * every byte the decoder itself wrote as one value, and the
         * value compared below is still the full 64 KiB's CRC. */
        if (identity_shape != 0 && d->seg_const != 0u && n != 0u) {
            calc = zpse_crc32c_repeat(d->seg_const_byte, (size_t)n);
        } else {
            calc = zpse_crc32c(output, n);
        }
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
 * exactly like a completed trial that failed to beat RAW.
 *
 * `have_best` / `best_total` carry the sweep's running winner: this trial
 * is only ever chosen on a STRICTLY smaller total, so once the bytes it
 * has produced already reach `best_total` it has lost and the rest of the
 * payload is work for nothing. That bound is the tighter of the two
 * whenever a previous trial completed, and stopping on it is exact -- a
 * trial's total only grows as its remaining segments are written, so
 * `at >= best_total` (or a segment that crosses it) is a proof, not an
 * estimate. The raw-envelope case keeps its own meaning: it is the only
 * one that sets `*budget_hit`, because it is the only one that means
 * "reached the size where RAW wins" for the caller's incompressible
 * heuristic. */
/* The nine predictors the segment encoder's prescreen ranks, spelled out
 * here for the sample below (the same list, and the same reason for
 * spelling it out, as seg_screen_pred in src/segment.c: appending to the
 * enum cannot silently reorder it). ids 3, 4 and 9 are the sentinels that
 * prescreen leaves at 0xFFFF. */
static const uint8_t zpse_block_screen[9] = { 0u, 1u, 2u, 5u, 6u, 7u, 8u,
                                              10u, 11u };

/* Whole-block noise sample: is EVERY one of sixteen tiles spread evenly
 * across the block's transformed planes a tile the segment encoder itself
 * would call near-random -- one where all nine screened predictors miss at
 * least 7/8 of the positions (the same threshold the segment encoder's own
 * single-candidate rule uses)?
 *
 * This is the second half of the trial-level flat abort's evidence, and it
 * is what keeps that heuristic off a block whose FRONT is noise and whose
 * rest compresses: the leading-run test (seg_flat_abort) only ever sees
 * the front, so it cannot tell such a block from a block that is noise
 * everywhere, and the whole-block sample sees the rest. Sixteen 256-byte
 * tiles out of a block is a sample, not a proof -- but the points are the
 * odd thirty-seconds of the block (1/32, 3/32, ... 31/32), a sixteenth of
 * the block apart, so a compressible stretch of the block wider than that
 * contains one of them whatever the segment shape, and each point is judged
 * by the encoder's own near-random rule on the very plane the trial is
 * about to write. The abort further requires a leading run of unpredicted
 * tiles. Returns 1 only when the whole block reads as noise at every sample
 * point. */
static int zpse_block_is_noise(const uint8_t *const planes[16], uint8_t sc,
                               const uint32_t lens[16], uint32_t n)
{
    unsigned i;

    for (i = 0; i < 16u; i++) {
        uint64_t target = ((uint64_t)n * (uint64_t)(2u * i + 1u)) / 32u;
        uint64_t seen = 0;
        uint32_t len, off;
        const uint8_t *pl;
        uint16_t nt;
        uint16_t ks[12];
        uint16_t best;
        unsigned s, j;

        for (s = 0; s + 1u < sc; s++) {
            if (target < seen + lens[s])
                break;
            seen += lens[s];
        }
        pl = planes[s];
        len = lens[s];
        if (pl == 0 || len < 64u)
            continue;
        off = (uint32_t)(target - seen);
        off &= ~255u;
        if (off + 64u > len)
            off = (len >= 320u) ? (len - 320u) : 0u;
        nt = (uint16_t)((len - off > 256u) ? 256u : (len - off));
        if (nt < 64u)
            continue;
        zpse_segment_prescreen(pl + off, nt, pl, off, pl[0], ks);
        best = ks[zpse_block_screen[0]];
        for (j = 1u; j < 9u; j++) {
            if (ks[zpse_block_screen[j]] < best)
                best = ks[zpse_block_screen[j]];
        }
        if ((uint32_t)best * 8u < (uint32_t)nt * 7u)
            return 0; /* a tile the encoder can predict: not a noise block */
    }
    return 1;
}

static int zpse_block_try_one(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                              uint32_t in_crc,
                              uint8_t tid, uint8_t sc,
                              const uint8_t *meta_bytes, uint16_t meta_len,
                              int is_last, uint32_t seq,
                              uint8_t *dst, uint32_t cap, uint32_t *total_out,
                              int *budget_hit, int have_best,
                              uint32_t best_total, uint32_t flat_abort_min)
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
    int is_noise;
    int rc;

    if (budget_hit != 0) {
        *budget_hit = 0;
    }
    /* The flat abort's coverage counter and the flag that ends its run span
     * this trial's segments, so they are reset here rather than inside
     * zpse_segment_encode. */
    e->flat_covered = 0u;
    e->flat_paid = 0u;
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
    /* A width-1 split with neither DELTA8 nor XOR8 is the identity: the
     * segment encoder only reads its plane, so the input itself can serve
     * as segment 0 and a 64 KiB staging copy disappears from every block
     * that tries the identity (which is all of them). The overlap
     * rejection inside zpse_transform_forward exists because it writes the
     * plane; not copying at all is what makes it unnecessary here. */
    if (tid == 0u && sc == 1u) {
        planes[0] = (uint8_t *)(uintptr_t)in;
    } else {
        rc = zpse_transform_forward(&tm, sc, in, n, planes, lens);
        if (rc != PSE_OK) {
            return rc;
        }
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
    /* Sixteen prescreens against the tiles this trial is about to write,
     * asked for only while the flat abort could still fire: the abort is
     * armed per segment only while beating the RAW envelope is the trial's
     * only way to win, and a winner already smaller than RAW means no
     * segment can be armed. */
    is_noise = (flat_abort_min != 0u &&
                (!have_best ||
                 (uint64_t)best_total >=
                     (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n))
                   ? zpse_block_is_noise((const uint8_t *const *)planes, sc,
                                         lens, n)
                   : 0;
    cum = 0;
    for (s = 0; s < sc; s++) {
        uint8_t lit = lens[s] > 0 ? planes[s][0] : 0;
        uint64_t at = base + cum;
        uint64_t raw = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n;
        uint32_t room;
        uint32_t budget;
        int budget_is_raw = 1;
        int wlen;
        if (at > (uint64_t)cap) {
            return PSE_OUTPUT_OVERFLOW;
        }
        /* Already at or past the running best before this segment writes
         * a byte: every remaining payload byte only adds to the total, so
         * this trial cannot be strictly smaller than the winner. A loss,
         * not a budget hit -- nothing here says anything about RAW. */
        if (have_best && (uint64_t)best_total <= at) {
            return PSE_OUTPUT_OVERFLOW;
        }
        room = (uint32_t)((uint64_t)cap - at);
        budget = (raw > at) ? (uint32_t)(raw - at) : 0u;
        if (have_best && (uint64_t)best_total < raw) {
            /* RAW is out of the running (the winner is already smaller
             * than it), so the tighter cap is the winner itself, and it
             * is the one that counts now -- a segment stopped by it is a
             * known loss, never an incompressible-block signal. */
            uint32_t bb = (uint32_t)((uint64_t)best_total - at);
            if (budget == 0u || bb < budget) {
                budget = bb;
                budget_is_raw = 0;
            }
        }
        /* The flat abort is a *noise* signal: it only means something
         * while the trial's only way to win is to beat the RAW envelope.
         * Once a winner exists the budget above is a proof of the same
         * thing, so the heuristic is switched off and only exact bounds
         * stop the trial. The whole-block sample is asked first because it
         * is a property of the trial's transformed planes, not of the
         * segment being written, so a block that is predictable anywhere
         * keeps the trial it has always had. */
        wlen = zpse_segment_encode(e, &lanes, planes[s], lens[s], lit,
                                   dst + at, room, budget,
                                   (budget_is_raw != 0 && is_noise != 0)
                                       ? flat_abort_min
                                       : 0u);
        if (wlen < 0) {
            if (wlen == PSE_OUTPUT_OVERFLOW && budget != 0u &&
                budget_is_raw != 0 && budget_hit != 0) {
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
        uint8_t *ent =
            dst + ZPSE_BLOCK_HEADER_SIZE + meta_len + (size_t)s * 8;
        uint32_t k;

        for (k = 0; k < s; k++) {
            off += seg_sizes[k];
        }
        zpse_store_u32le(ent, off);
        zpse_store_u32le(ent + 4, seg_sizes[s]);
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

/* Divisor of the trial-level noise probe's threshold: see flat_min in
 * zpse_block_encode and seg_flat_abort in segment.c.
 *
 * 64, not 16. The probe exists only to stop a trial that can no longer
 * beat the RAW envelope, and a noise tile costs about 370 ns to encode
 * (versus 75 ns for one that predicts), so the old 4 KiB threshold spent
 * 16 of them -- 6 us of a 26 us encode -- before giving up. A quarter of
 * that decides on the same evidence; the floor in zpse_block_encode means
 * 64 and anything larger are the same threshold, so this is the smallest
 * the divisor can usefully be. The noise row goes 3.01 -> 3.44 GB/s and
 * nothing else moves: tests/test_fuzz.c prints byte-identical sizes for
 * all 2,718 cases and tests/test_sizes.c still reports grew=0 over 768. */
#ifndef ZPSE_FLAT_MIN_DIV
#define ZPSE_FLAT_MIN_DIV 64u
#endif

/* Smallest size a trial with this (transform, segment count, metadata)
 * shape can possibly produce: the envelope (block header, metadata, the
 * segment directory, each segment's 16-byte header) plus 2 length-prefix
 * bytes and the 8-byte tile header for every tile. Used to stop the
 * transform sweep as soon as nothing left can beat the running best. */
static uint32_t zpse_block_floor(uint32_t n, uint8_t tid, uint8_t sc,
                                 uint16_t meta_len)
{
    uint32_t lens[16];
    uint32_t records_bytes = 0u;
    uint8_t w = 0;
    uint8_t s;

    if (zpse_transform_width(tid, sc, &w) != PSE_OK) {
        return UINT32_MAX;
    }
    if (zpse_segment_lengths(n, w, lens) != PSE_OK) {
        return UINT32_MAX;
    }
    for (s = 0; s < sc; s++) {
        uint32_t ch = (lens[s] + ZPSE_DENSE_MAX_LEN - 1u) /
                      ZPSE_DENSE_MAX_LEN;
        uint32_t tile_cost = 10u * ((lens[s] + 255u) / 256u);
        uint32_t dense_cost = ZPSE_DENSE_HEADER_SIZE * ch;

        /* The two forms have very different floors -- ten bytes per
         * 256-byte tile, twelve per 4 KiB dense record -- and a constant
         * plane is coded at the second rate (zpse_segment_encode emits it
         * that way), so a candidate's bound is the smaller of the two.
         * Anything larger would prune a candidate whose plane happens to
         * be constant, which is a real shape (the DELTA8 plane of a
         * strided block). */
        records_bytes += (tile_cost < dense_cost ? tile_cost : dense_cost);
    }
    return (uint32_t)ZPSE_BLOCK_HEADER_SIZE + (uint32_t)meta_len +
           (uint32_t)sc * ((uint32_t)ZPSE_SEG_DIR_ENTRY_SIZE +
                           (uint32_t)ZPSE_SEGMENT_HEADER_SIZE) +
           records_bytes;
}

/* ------------------------------------------------------------------ *
 * Candidate attempt order.
 *
 * The sweep's result does not depend on the order the candidates are
 * attempted in, which makes that order a free performance knob:
 *
 *   - the winner is the smallest total with ties going to the earlier
 *     candidate in the LIST (index) order, which is applied explicitly at
 *     the comparison below;
 *   - the running-winner bound hands a later candidate the tightest size
 *     it must beat, so attempting the likely winner first is what makes
 *     the losers stop early;
 *   - the floor-based exit is evaluated over the candidates NOT yet
 *     attempted, and the two-failure exit needs two attempts, so the two
 *     exits fire at the same points their candidate sets allow.
 *
 * Only the two width-1 candidates can be swapped, and that is what makes
 * the argument tight: the attempted SET is {0, 1, ...} in both orders, so
 * the two-failure exit still costs exactly two attempts over the same two
 * candidates, while the floor check after the first attempt sees a
 * superset of what it saw before (the identity's floor is one byte below
 * DELTA8's, so its threshold is never the higher one and the sweep can
 * only stop later, never earlier). No candidate is ever skipped that the
 * other order would have attempted, and no tie can resolve differently.
 * Defaulting to the identity first is therefore safe, and swapping is safe
 * whenever the sample below says DELTA8's plane is much more predictable
 * than the raw bytes -- which is the case for ramps and strides, where the
 * identity trial is a full block of near-all-exception tiles -- the most
 * expensive work the tile encoder does -- that loses to DELTA8 by a
 * factor of four or more.
 *
 * The estimate is a sample, not a bound: it is used only to pick the
 * order, so a bad estimate costs the work the other order would have
 * saved, never a different block. A bound strong enough to PRUNE a
 * candidate cannot be built this cheaply -- a tile's floor given its
 * exception count k is roughly 8 + (mask bits) + k/8 bytes, about 40
 * bytes even for k near Nt, against the ~200 bytes such a tile actually
 * packs to, so no sum of per-tile floors reaches the sizes this format
 * produces. Ordering is the only use of the prescreen that pays.
 *
 * The sample: six tiles spread across the block, prescreened on both
 * planes. The identity's plane is the input; DELTA8's is
 * in[q] - in[q - 1] (its first byte is 0 because its seed is in[0]), so a
 * sample tile at plane offset p needs only in[p - 17, p + 256) and both
 * planes are prescreened with the same call the segment encoder makes,
 * with seg_pos == 16 so a locally-built 16-byte head stands in for the
 * plane bytes before the tile.
 * ------------------------------------------------------------------ */
#define ZPSE_ORDER_SAMPLES 6
#define ZPSE_ORDER_MIN_TILES 16u

static int zpse_block_delta_trial_first(const uint8_t *in, uint32_t n)
{
    uint32_t tc = n / (uint32_t)ZPSE_TILE_SIZE;
    uint32_t est_id = 0u;
    uint32_t est_d1 = 0u;
    unsigned ns;
    uint8_t loc[16u + ZPSE_TILE_SIZE];
    unsigned i;
    unsigned j;
    unsigned c;

    /* The estimate costs a fixed handful of prescreens while the work it
     * can save is proportional to the block, so it only runs once a block
     * has enough tiles to lose: below ZPSE_ORDER_MIN_TILES the identity
     * trial cannot waste more than its own few tiles and the sweep's floor
     * test usually ends it after one trial anyway. */
    if (tc < ZPSE_ORDER_MIN_TILES) {
        return 0;
    }
    ns = (tc >= 8u * ZPSE_ORDER_MIN_TILES) ? (unsigned)ZPSE_ORDER_SAMPLES : 3u;

    for (i = 0; i < ns; i++) {
        uint32_t t = ((uint32_t)(i + 1u) * tc) / (uint32_t)(ns + 1u);
        uint32_t p = t * (uint32_t)ZPSE_TILE_SIZE;
        uint16_t ks[12];
        uint32_t kmin;

        /* Interior tiles only: the head has to be real plane bytes, and
         * the DELTA8 window reaches one byte further back still. */
        if (p < 17u || (uint64_t)p + (uint64_t)ZPSE_TILE_SIZE > (uint64_t)n) {
            continue;
        }

        /* Identity plane. `in[0]` is the segment's initial literal. */
        memcpy(loc, in + p - 16u, 16u);
        memcpy(loc + 16u, in + p, (size_t)ZPSE_TILE_SIZE);
        zpse_segment_prescreen(loc + 16u, (uint16_t)ZPSE_TILE_SIZE, loc, 16u,
                               in[0], ks);
        kmin = ks[0];
        for (c = 1u; c < 11u; c++) {
            if (ks[c] < kmin) {
                kmin = ks[c];
            }
        }
        est_id += kmin;
        /* A raw plane that predicts every sampled tile so far is about to
         * reach the global floor and end the sweep -- and it is the
         * cheap kind of block (constant, all-hit), where the estimate's
         * own cost would be the largest part of the encode. Stop before
         * paying for the other plane, and report the third answer: this
         * is not just "the identity goes first", it is "the identity is
         * about to END the sweep", so the caller can also skip the proxy
         * that would otherwise price every candidate first. */
        if (est_id == 0u) {
            return 2;
        }

        /* DELTA8 plane over the same window, one byte behind itself. */
        for (j = 0; j < 16u + (unsigned)ZPSE_TILE_SIZE; j++) {
            uint32_t q = p - 16u + j;

            loc[j] = (uint8_t)(in[q] - in[q - 1u]);
        }
        zpse_segment_prescreen(loc + 16u, (uint16_t)ZPSE_TILE_SIZE, loc, 16u,
                               0u, ks);
        kmin = ks[0];
        for (c = 1u; c < 11u; c++) {
            if (ks[c] < kmin) {
                kmin = ks[c];
            }
        }
        est_d1 += kmin;
    }

    /* Needs a clear margin, not a hair: the two candidates can be close
     * (low-entropy data favours the identity while both planes look
     * equally bad to a prescreen), and a wrong swap costs the trial the
     * other order would have stopped early. */
    return (est_d1 * 4u <= est_id) ? 1 : 0;
}


/* Tiles an exact probe prices per plane: enough to tell a shape that packs
 * (a nibble alphabet, a mask-friendly record) from one that cannot, at a
 * fixed cost. */
#ifndef ZPSE_PROBE_TILES
#define ZPSE_PROBE_TILES 8u
#endif

/* Tiles the decision probe prices per candidate (see the sweep decision in
 * zpse_block_encode): paid once per candidate against a full trial's 256
 * tiles. Four is measured against sixteen: the 768-case clen corpus came
 * out byte-identical either way, the 2,718-case fuzz corpus came out
 * 34,174 bytes SMALLER at four (a coarse estimate ties more often, and a
 * tie goes to the earlier candidate -- the identity, which is the better
 * prior on the mixed blocks where the sample is noisiest), and it took
 * 22 us off a 64 KiB block's encode that the probe was spending to price
 * candidates it then rejected. */
#ifndef ZPSE_DECIDE_TILES
#define ZPSE_DECIDE_TILES 4u
#endif

static int zpse_block_planes(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint8_t tid, uint8_t sc,
                             const uint8_t *meta_bytes, uint16_t meta_len,
                             uint8_t *planes[16], uint32_t lens[16]);

/* Estimated size of one transform trial: stage the transform, then price
 * each plane with the prescreen alone (no classify, no packing).
 * UINT32_MAX means "could not be estimated" and the caller keeps the
 * candidate. */
static uint32_t zpse_block_proxy(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                                 uint8_t tid, uint8_t sc,
                                 const uint8_t *meta_bytes, uint16_t meta_len)
{
    uint32_t lens[16];
    uint8_t *planes[16];
    uint32_t total;
    uint32_t s;

    if (zpse_block_planes(e, in, n, tid, sc, meta_bytes, meta_len, planes,
                          lens) != PSE_OK) {
        return UINT32_MAX;
    }
    total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + meta_len +
            (uint32_t)sc * (uint32_t)ZPSE_SEG_DIR_ENTRY_SIZE;
    for (s = 0; s < sc; s++) {
        uint32_t c = zpse_segment_proxy_cost(planes[s], lens[s]);
        if (c == UINT32_MAX)
            return UINT32_MAX;
        total += c;
    }
    return total;
}

/* Exact price of a plane: the smaller of its tile records (a spread of its
 * tiles, classified, mode-searched and packed exactly as a trial does) and
 * the two dense forms, which are priced exactly because each takes one pass
 * over the plane. zpse_segment_probe_cost owns all three; this only gives
 * it the lane state a segment encode would start from.
 *
 * `tiles` is how many tiles to price (0 = ZPSE_PROBE_TILES), spread evenly
 * over the plane and folded back up to its full tile count. `limit` is the
 * running verdict the caller holds: a plane whose own proven floor already
 * exceeds it stops there (see seg_tile_price in segment.c). Returns the
 * estimate for the whole plane, or UINT32_MAX when the plane cannot be
 * priced (the caller then keeps the candidate). */
static uint32_t zpse_block_exact_plane(ZpseEncoder *e, const uint8_t *plane,
                                 uint32_t len, unsigned tiles,
                                 uint32_t limit)
{
    ZpseLaneState lanes;

    if (plane == 0 || len == 0u || e->comb == 0) {
        return UINT32_MAX;
    }
    if (tiles == 0u || tiles > ZPSE_DECIDE_TILES) {
        tiles = (tiles == 0u) ? ZPSE_PROBE_TILES : ZPSE_DECIDE_TILES;
    }
    lanes.order1 = e->order1;
    lanes.o2val = e->o2val;
    lanes.o2conf = e->o2conf;
    return zpse_segment_probe_cost(e, &lanes, plane, len, tiles, limit);
}

/* Smallest price zpse_block_exact_plane can return for a plane of `len`
 * bytes: the segment envelope, plus either a 2-byte length prefix and the
 * 8-byte header of every tile (the record of a tile whose prediction is
 * exact: mask ALL_HIT, empty residual) or the dense constant form's 12
 * bytes per 4 KiB, whichever is smaller. No record can be below either
 * bound, so a plane priced at this floor is one whose every tile was an
 * exact prediction -- a constant plane, which the dense form then writes. */
static uint32_t zpse_block_plane_floor(uint32_t len)
{
    uint32_t tiles = 16u + 10u * ((len + 255u) / 256u);
    uint32_t dense = 16u + 12u * ((len + ZPSE_DENSE_MAX_LEN - 1u) /
                                  ZPSE_DENSE_MAX_LEN);

    return tiles < dense ? tiles : dense;
}

/* Smallest price zpse_block_exact_sum can return for a candidate whose
 * planes total `len` bytes: the block envelope zpse_block_exact_sum starts
 * from, plus one plane at its floor. Any candidate with more than one plane
 * pays the 16-byte segment envelope once per plane and so cannot be below
 * this; a single-segment one reaches it exactly when its plane is at the
 * floor. */
static uint32_t zpse_block_cand_floor(uint32_t len)
{
    return (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
           zpse_block_plane_floor(len);
}

/* Exact price of one candidate: every plane probed, summed the same way the
 * prescreen proxy sums its planes. */
static uint32_t zpse_block_exact_sum(ZpseEncoder *e, uint8_t *planes[16],
                                      const uint32_t lens[16], uint8_t sc,
                                      unsigned tiles, uint32_t limit)
{
    uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u;
    uint32_t records = 0u;
    uint32_t avg_rec;
    uint32_t sum_rec;
    unsigned priced = 0u;
    unsigned step = 1u;
    unsigned s;

    /* 0 means "the probe's own default", exactly as zpse_block_exact_plane
     * reads it; it is resolved here because it divides the budget. */
    if (tiles == 0u || tiles > ZPSE_DECIDE_TILES) {
        tiles = (tiles == 0u) ? ZPSE_PROBE_TILES : ZPSE_DECIDE_TILES;
    }
    /* The budget is per CANDIDATE, not per plane: a 16-segment transpose has
     * sixteen planes, and pricing a tile in every one of them measured
     * sixteen tiles to buy four tiles' worth of information (12.5 us of a
     * 27 us block on the lowentropy row). Price `tiles` of the planes
     * instead, one tile each.
     *
     * The planes that are not priced still get the constant test, because
     * that is the one form a tile price cannot see and it is cheap: the
     * phases of a transpose are constant whenever the source repeats with
     * the transpose period, and that is exactly the case a spread of four
     * planes can miss (a spike every 256 bytes lands in one phase). A plane
     * that is not constant is priced at the priced planes' average. */
    if (sc > 1u) {
        step = ((unsigned)sc + tiles - 1u) / tiles;
        if (step == 0u) {
            step = 1u;
        }
    }
    for (s = 0u; s < sc; s += step) {
        /* The running share: everything still unspent below `limit`
         * belongs to the planes not yet priced, and this plane may spend
         * its envelope plus a fair piece of it. UINT32_MAX (no verdict in
         * hand) prices the whole sample, as before. */
        uint32_t share = UINT32_MAX;
        if (limit != UINT32_MAX) {
            uint64_t envelope = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
                                (uint64_t)sc * 16u;
            if ((uint64_t)records + envelope <= (uint64_t)limit) {
                share = (uint32_t)(16u +
                          ((uint64_t)limit - (uint64_t)records - envelope));
            }
        }
        {
            uint32_t c = zpse_block_exact_plane(e, planes[s], lens[s],
                                                sc > 1u ? 1u : tiles, share);

            if (c == UINT32_MAX) {
                return UINT32_MAX;
            }
            /* c is 16 + the plane's record area folded up to its tile count. */
            records += c - 16u;
            priced++;

            /* Early exit: if the running lower bound on the candidate total
             * already exceeds `limit`, this candidate cannot win. Return the
             * bound (finite, > limit) so the caller drops it instead of
             * pricing the rest of the planes. UINT32_MAX remains "could not
             * be priced" (keep), so a convicted value is distinguishable. */
            if (limit != UINT32_MAX) {
                uint64_t lb = (uint64_t)(ZPSE_BLOCK_HEADER_SIZE + 16u) +
                              (uint64_t)sc * 16u + (uint64_t)records;
                if (lb > (uint64_t)limit) {
                    return (uint32_t)(lb > (uint64_t)UINT32_MAX
                                          ? (uint64_t)UINT32_MAX
                                          : lb);
                }
            }
        }
    }
    if (priced == 0u) {
        return UINT32_MAX;
    }
    sum_rec = records;
    avg_rec = records / priced;
    if (step > 1u) {
        for (s = 0u; s < sc; s++) {
            uint32_t d;

            if ((s % step) == 0u) {
                continue;
            }
            d = zpse_segment_const_cost(planes[s], lens[s]);
            sum_rec += (d != UINT32_MAX) ? (d - 16u) : avg_rec;
        }
    }
    total += (uint32_t)sc * 16u + sum_rec;
    return total;
}

/* Stage one candidate's transform into the encoder's staging area and hand
 * back its planes and lengths. A width-1 split with neither DELTA8 nor XOR8
 * is the identity, so the input itself serves as the plane for it. */
static int zpse_block_planes(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint8_t tid, uint8_t sc,
                             const uint8_t *meta_bytes, uint16_t meta_len,
                             uint8_t *planes[16], uint32_t lens[16])
{
    ZpseTransformMeta tm;
    uint8_t w = 0;
    uint32_t off = 0;
    uint32_t s;

    if (zpse_transform_width(tid, sc, &w) != PSE_OK)
        return PSE_BAD_TRANSFORM;
    if (zpse_segment_lengths(n, w, lens) != PSE_OK)
        return PSE_BAD_TRANSFORM;
    if (zpse_transform_meta_parse(&tm, tid, meta_bytes, meta_len, sc) != PSE_OK)
        return PSE_BAD_TRANSFORM;
    for (s = 0; s < sc; s++) {
        if (lens[s] == 0u || (uint64_t)off + lens[s] > ZPSE_E_SZ_TSTAGE)
            return PSE_OUTPUT_OVERFLOW;
        planes[s] = e->tstage + off;
        off += lens[s];
    }
    if (off != n)
        return PSE_BAD_TRANSFORM;
    if (tid == 0u && sc == 1u) {
        planes[0] = (uint8_t *)(uintptr_t)in;
        return PSE_OK;
    }
    return zpse_transform_forward(&tm, sc, in, n, planes, lens);
}

/* Exact price of one candidate, used to break a prescreen tie: see
 * zpse_block_exact_plane. UINT32_MAX means "could not be priced". */
static uint32_t zpse_block_proxy_exact(ZpseEncoder *e, const uint8_t *in,
                                       uint32_t n, uint8_t tid, uint8_t sc,
                                       const uint8_t *meta_bytes,
                                       uint16_t meta_len, unsigned tiles,
                                       uint32_t limit)
{
    uint32_t lens[16];
    uint8_t *planes[16];

    if (zpse_block_planes(e, in, n, tid, sc, meta_bytes, meta_len, planes,
                          lens) != PSE_OK) {
        return UINT32_MAX;
    }
    return zpse_block_exact_sum(e, planes, lens, sc, tiles, limit);
}/* Price candidates [from, *ncand_io) exactly -- a few packed tiles each --
 * and drop the ones more than ~6% worse than the best of that range, the
 * same margin the prescreen prune uses. Anything that cannot be priced is
 * kept, and so is everything when the exact prices tie, so this can only
 * remove a candidate the prescreen would have kept on no evidence at all. */
static void zpse_block_prune_exact(ZpseEncoder *e, const uint8_t *in,
                                   uint32_t pn, uint8_t tids[4],
                                   uint8_t scounts[4],
                                   uint8_t meta_bufs[4][2],
                                   uint16_t meta_lens[4], uint8_t idx[4],
                                   int from, int *ncand_io)
{
    uint32_t ex[4];
    uint32_t pmin = UINT32_MAX;
    uint32_t limit;
    int to = *ncand_io;
    int kept = 0;
    int ci;

    for (ci = from; ci < to; ci++) {
        const uint8_t *mb = meta_lens[ci] > 0 ? meta_bufs[ci] : in;
        /* Feed the running verdict down as `limit`: the first candidate is
         * priced without one (UINT32_MAX), each later one with the best
         * price seen so far plus the sweep's ~6% margin. A plane whose
         * proven floor already exceeds its share stops early, so a
         * hopeless candidate prices only a tile or two instead of all of
         * them. Convicted candidates return a finite value > limit, never
         * UINT32_MAX, so they are dropped, not kept. */
        uint32_t lim = (pmin == UINT32_MAX) ? UINT32_MAX : pmin + (pmin >> 4);

        ex[ci] = zpse_block_proxy_exact(e, in, pn, tids[ci], scounts[ci], mb,
                                       meta_lens[ci], 0u, lim);
        if (ex[ci] != UINT32_MAX && ex[ci] < pmin) {
            pmin = ex[ci];
        }
    }
    if (pmin == UINT32_MAX) {
        return;
    }
    limit = pmin + (pmin >> 4);
    for (ci = from; ci < to; ci++) {
        if (ex[ci] == UINT32_MAX || ex[ci] <= limit) {
            tids[from + kept] = tids[ci];
            scounts[from + kept] = scounts[ci];
            meta_lens[from + kept] = meta_lens[ci];
            meta_bufs[from + kept][0] = meta_bufs[ci][0];
            meta_bufs[from + kept][1] = meta_bufs[ci][1];
            idx[from + kept] = idx[ci];
            kept++;
        }
    }
    *ncand_io = from + kept;
}

/* Exchange two candidate slots, moving the candidate itself in every array
 * that describes it -- including idx[], which carries its original list
 * index and is what the winner's tie-break compares. The attempt order is
 * realized this way and never as an index permutation over the list, so
 * the proxy pruning below (which compacts the list) cannot leave the
 * attempt loop pointing at a slot it has since dropped: after the swap the
 * loop simply walks the surviving slots in order. */
static void zpse_block_swap_cand(uint8_t tids[4], uint8_t scounts[4],
                                 uint8_t meta_bufs[4][2],
                                 uint16_t meta_lens[4], uint8_t idx[4],
                                 int a, int b)
{
    uint8_t t8 = tids[a];
    uint8_t s8 = scounts[a];
    uint8_t i8 = idx[a];
    uint8_t m0 = meta_bufs[a][0];
    uint8_t m1 = meta_bufs[a][1];
    uint16_t l16 = meta_lens[a];

    tids[a] = tids[b];
    scounts[a] = scounts[b];
    idx[a] = idx[b];
    meta_bufs[a][0] = meta_bufs[b][0];
    meta_bufs[a][1] = meta_bufs[b][1];
    meta_lens[a] = meta_lens[b];

    tids[b] = t8;
    scounts[b] = s8;
    idx[b] = i8;
    meta_bufs[b][0] = m0;
    meta_bufs[b][1] = m1;
    meta_lens[b] = l16;
}

int zpse_block_encode(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                      uint32_t seq, int is_last,
                      uint8_t *out, uint32_t out_cap)
{
    uint8_t tids[4];
    uint8_t scounts[4];
    uint8_t meta_bufs[4][2];
    uint16_t meta_lens[4];
    uint8_t idx[4];
    uint32_t in_crc;
    int ncand = 0;
    uint32_t best_total = 0;
    uint32_t best_idx = 0;
    uint32_t flat_min = 0u;
    int have_best = 0;
    int fails = 0;
    int sample;
    int exact_priced = 0;
    int oi;
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

    /* Trial-level noise probe threshold: a trial whose only way to win is
     * the RAW envelope is abandoned once it has covered this many input
     * bytes without one tile paying for itself (zpse_segment_encode,
     * seg_flat_abort). A sixteenth of the block, floored at four tiles, and
     * only for blocks big enough for either to be worth measuring -- the
     * same 8 KiB gate the prescreen proxy uses -- so small blocks keep
     * exactly the trial structure they have always had. */
    flat_min = (n >= 8192u) ? (n / ZPSE_FLAT_MIN_DIV) : 0u;
    if (flat_min != 0u && flat_min < 1024u)
        flat_min = 1024u;

    /* The candidate list is built in slot order below and then reordered
     * and compacted in place, so every slot a candidate does not fill must
     * start at a defined value rather than whatever the stack held. */
    memset(tids, 0, sizeof tids);
    memset(scounts, 0, sizeof scounts);
    memset(meta_bufs, 0, sizeof meta_bufs);
    memset(meta_lens, 0, sizeof meta_lens);
    memset(idx, 0, sizeof idx);

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
    /* A one-value block can take its CRC arithmetically. The scan is the
     * segment encoder's own AVX2 one, and its answer is handed back to that
     * encoder by pointer (e->const_plane) so the identity trial skips its
     * own scan: a constant block is scanned once, not twice. The scan exits
     * at the first differing byte, so a block that is not constant pays one
     * compare and writes no pointer. */
    e->const_plane = 0;
    if (zpse_segment_plane_is_const(in, n)) {
        in_crc = zpse_crc32c_repeat(in[0], (size_t)n);
        e->const_plane = in;
    } else {
        in_crc = zpse_crc32c(in, n);
    }

    /* Attempt order: identity first unless the sample says DELTA8's plane
     * is much more predictable, and never anything but a swap of those
     * two (see the argument above the sample). The sample settles one
     * more thing: whether the proxy below is worth running at all. */
    {
        int i;

        for (i = 0; i < ncand; i++) {
            idx[i] = (uint8_t)i;
        }
    }
    sample = (ncand >= 2) ? zpse_block_delta_trial_first(in, n) : 0;
    if (ncand >= 2) {
        int swap = (sample == 1);

#if defined(ZPSE_FORCE_TRIAL_SWAP)
        /* Test hook: take the other decision wherever a decision exists,
         * so a build with this macro attempts these two candidates in the
         * opposite order from the default build for every block. The
         * parity suite builds test_parity.c and test_fuzz.c this way and
         * requires their streams and digests to be identical to the
         * default build's, which is what proves the order neutral. */
        swap = swap ? 0 : 1;
#endif
        if (swap) {
            zpse_block_swap_cand(tids, scounts, meta_bufs, meta_lens, idx, 0,
                                 1);
        }
    }

    /* Prune with the prescreen proxy: price every candidate cheaply and
     * fully encode only those within ~6% of the best estimate (always at
     * least one). A full trial is a classify + mode search + pack of every
     * tile, so where one transform is clearly better this skips the rest.
     * The decision is a pure function of the input, so it is identical in
     * the scalar, SIMD and swapped-order builds.
     *
     * sample == 2 means the identity's plane predicts every sampled tile,
     * so its trial is about to reach the global floor and end the sweep on
     * the first attempt: pricing the other candidates would then be pure
     * cost, and on a constant block it is most of the encode. The sample
     * says nothing about "is one candidate clearly better than another"
     * unless it looked at more than one plane, so the gate is exact about
     * what it knows. */
#ifndef ZPSE_KEEP_EXACT_SWEEP
    /* ---- Decide from a sample, then encode once ---------------- *
     * DEFAULT; build with -DZPSE_KEEP_EXACT_SWEEP for the old sweep, which
     * encodes every survivor in full and keeps the smallest stream. A
     * trial is a classify, a mode search and a pack of every tile, so the
     * candidates the prescreen cannot separate -- a 4-bit alphabet, a
     * nibble-packed record, a text block, where all of them miss about as
     * often as each other -- cost a whole block encode each for a
     * difference of a few bytes.
     *
     * This prices every survivor exactly instead: the same classify, mode
     * search and pack on ZPSE_DECIDE_TILES tiles spread over the block,
     * folded up to the block's tile count. The winner is then encoded
     * once. Measured 1.65-2.07x on the lowentropy row of the bench (949k ->
     * 575k-510k cycles per 64 KiB block) and byte-identical on all 768 cases
     * of tests/clen_corpus.c, INCLUDING their CRC32-Cs.
     *
     * The price is a sample, and that is the trade. On a block whose tiles
     * are not statistically alike -- tests/test_fuzz.c shape 16, regions of
     * random texture at random boundaries -- a 16-tile sample can pick the
     * wrong transform outright: 74 of that suite's 2,718 cases moved by one
     * byte and one (8 KiB of random mixed regions) went from 1,735 to 3,645
     * bytes. Every case of tests/clen_corpus.c is byte-identical. See the
     * README for what that costs and why it is worth it.
     *
     * Ordering keeps the sweep's own tie-break: the smallest price wins and
     * an exact tie goes to the earlier candidate, so a candidate can only
     * be picked over one it would have lost to on a strict inequality. A
     * price that cannot be computed never wins over one that can, and if
     * nothing can be priced the list is left alone and the sweep runs
     * exactly as it always did. */
    /* Rank the candidates on the prescreen alone before pricing any of
     * them, and drop everything more than ~6% worse: a trial is a classify,
     * a mode search and a pack of every tile, and a candidate the cheap
     * estimate has already lost by more than that is not worth the exact
     * probe either. That estimate covers every other tile of the first
     * 16 KiB -- broad enough that a block whose structure changes inside
     * the probe window still ranks correctly, where a four-tile exact
     * sample alone mis-picks by a factor (measured on the fuzz suite's
     * mixed-region shapes: 400 KB worse without this step). */
    if (sample != 2 && ncand > 1 && n >= 8192u) {
        uint32_t px[4];
        uint32_t pmin = UINT32_MAX;
        int kept = 0;
        int ci;
        uint32_t pn = n < 16384u ? n : 16384u;

        /* The identity is priced first (see the exact probe below: it is
         * the first candidate in the list, and a swap of the attempt order
         * does not move it). Its price is therefore in hand when the others
         * are reached, and a candidate can then be dropped BEFORE its proxy
         * runs when its own floor -- the smallest total its transform and
         * segment shape can possibly produce -- already exceeds the 6%
         * margin around that price. Such a candidate cannot be kept by the
         * prune below whatever its proxy would say, because a proxy is never
         * below its floor, so skipping the computation cannot change the
         * survivor list, and the pruned price is written as limit + 1 to
         * record exactly that. The margin only shrinks if a later proxy
         * undercuts the running minimum, and a candidate dropped under the
         * wider margin is dropped under the narrower one too, so the order
         * of the loop stays irrelevant. The floor is arithmetic on the
         * shape -- no plane is touched -- and this is what removes the
         * transpose candidate from every block whose identity plane is
         * already near-perfect (a ramp, a constant, an all-hit plane). */
        for (ci = 0; ci < ncand; ci++) {
            const uint8_t *mb = meta_lens[ci] > 0 ? meta_bufs[ci] : in;
            uint32_t cfloor;

            if (ci > 0 && pmin != UINT32_MAX) {
                /* The floor is taken over the PROXY's window, not the block:
                 * it is compared with a proxy, and a proxy prices `pn`
                 * bytes. A block-sized floor would be several times the
                 * window's price and would drop candidates the prune is
                 * meant to keep. */
                cfloor = zpse_block_floor(pn, tids[ci], scounts[ci],
                                          meta_lens[ci]);
                if (cfloor != UINT32_MAX &&
                    cfloor > pmin + (pmin >> 4)) {
                    px[ci] = pmin + (pmin >> 4) + 1u;
                    continue;
                }
            }
            px[ci] = zpse_block_proxy(e, in, pn, tids[ci], scounts[ci], mb,
                                      meta_lens[ci]);
            if (px[ci] < pmin) {
                pmin = px[ci];
            }
        }
        if (pmin != UINT32_MAX) {
            uint32_t limit = pmin + (pmin >> 4);

            for (ci = 0; ci < ncand; ci++) {
                if (px[ci] == UINT32_MAX || px[ci] <= limit) {
                    tids[kept] = tids[ci];
                    scounts[kept] = scounts[ci];
                    meta_lens[kept] = meta_lens[ci];
                    meta_bufs[kept][0] = meta_bufs[ci][0];
                    meta_bufs[kept][1] = meta_bufs[ci][1];
                    idx[kept] = idx[ci];
                    kept++;
                }
            }
            ncand = kept;
        }
    }

    if (ncand > 1 && n >= 8192u) {
        uint32_t ex[4];
        uint8_t order[4];
        int nord = 0;
        /* Price on the first 16 KiB only: staging a transform costs about
         * as much as the estimate itself, and a block's structure (record
         * width, delta-friendliness) is the same throughout, so a quarter
         * of it orders the candidates just as well. (8 KiB was measured
         * too: it is faster, but it mis-ranks a block whose structure only
         * appears past the first quarter -- a long noise prefix in front
         * of a compressible body went to RAW -- so the window stays at
         * 16 KiB.) */
        uint32_t pn = n < 16384u ? n : 16384u;
        int pick = -1;
        int ci;
        int k;

        /* The identity is priced first, whatever the attempt order is. Its
         * shape carries the smallest envelope (one segment, no transform
         * metadata) so its floor is the block's, and the identity wins every
         * tie on list index 0: the moment it prices at that floor the winner
         * is settled and the remaining candidates' staging and probes are
         * pure cost. That is the constant and all-hit case, where it was
         * 14.6 us of a 19.2 us encode before. The comparison below keeps the
         * sweep's own tie-break, so the pricing order cannot change who
         * wins -- only how much of the list is priced. */
        for (ci = 0; ci < ncand; ci++) {
            if (idx[ci] == 0u) {
                order[nord++] = (uint8_t)ci;
            }
        }
        for (ci = 0; ci < ncand; ci++) {
            if (idx[ci] != 0u) {
                order[nord++] = (uint8_t)ci;
            }
        }

        for (k = 0; k < nord; k++) {
            const uint8_t *mb;

            ci = order[k];
            mb = meta_lens[ci] > 0 ? meta_bufs[ci] : in;
            ex[ci] = zpse_block_proxy_exact(e, in, pn, tids[ci],
                                            scounts[ci], mb, meta_lens[ci],
                                            ZPSE_DECIDE_TILES, UINT32_MAX);
            if (ex[ci] == UINT32_MAX) {
                continue;
            }
            if (pick < 0 || ex[ci] < ex[pick] ||
                (ex[ci] == ex[pick] && idx[ci] < idx[pick])) {
                pick = ci;
            }
            if (idx[ci] == 0u && scounts[ci] == 1u &&
                ex[ci] == zpse_block_cand_floor(pn)) {
                break;
            }
        }
        if (pick >= 0) {
            tids[0] = tids[pick];
            scounts[0] = scounts[pick];
            meta_lens[0] = meta_lens[pick];
            meta_bufs[0][0] = meta_bufs[pick][0];
            meta_bufs[0][1] = meta_bufs[pick][1];
            idx[0] = idx[pick];
            ncand = 1;
        }
    }
#endif /* !ZPSE_KEEP_EXACT_SWEEP */

    for (oi = 0; oi < ncand; oi++) {
        int i = oi;
        uint32_t t = 0;
        int budget_hit = 0;
        const uint8_t *mb;
        /* Exact bound before the work: the smallest total this candidate's
         * (transform, segment count, metadata) shape can produce is a
         * strictly greater size than the winner, so the trial cannot be
         * chosen or tie and the transform staging is pure cost. Unlike the
         * post-trial floor check, this one is per candidate, so a single
         * hopeless shape left in the list no longer keeps the others
         * alive. `>` (not `>=`) on purpose: a candidate whose floor equals
         * the winner could still tie it, and a tie is decided by list
         * order, so it has to be attempted.
         *
         * "Could still tie" is only true for a candidate EARLIER in the
         * list than the current winner: a tie goes to the lower index, so
         * a later one cannot win it. The list-order test is what keeps the
         * bound exact wherever the attempt order is not the list order
         * (ZPSE_FORCE_TRIAL_SWAP, and any future reordering): skipping an
         * earlier candidate because it merely ties would change the
         * stream, and the parity suite proves it does not. */
        if (have_best && (uint32_t)idx[i] > best_idx &&
            zpse_block_floor(n, tids[i], scounts[i], meta_lens[i]) >
                best_total) {
            continue;
        }
        mb = meta_lens[i] > 0 ? meta_bufs[i] : in;
        rc = zpse_block_try_one(e, in, n, in_crc, tids[i], scounts[i], mb,
                                meta_lens[i], is_last, seq,
                                e->cand, (uint32_t)ZPSE_E_SZ_CAND, &t,
                                &budget_hit, have_best, best_total,
                                flat_min);
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
        /* Smallest total wins; a tie goes to the candidate EARLIER in
         * the list, so the attempt order cannot change the winner. */
        if (!have_best || t < best_total ||
            (t == best_total && (uint32_t)idx[i] < best_idx)) {
            if (t > (uint32_t)ZPSE_E_SZ_BEST) {
                continue;
            }
            memcpy(e->best, e->cand, t);
            best_total = t;
            best_idx = (uint32_t)idx[i];
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
            int earlier_left = 0;
            int oj;

            for (oj = oi + 1; oj < ncand; oj++) {
                int j = oj;
                uint32_t f;

                /* A candidate earlier in the list than the winner cannot
                 * produce a smaller total, but it can produce the same one
                 * and win the tie on order, so the bound does not cover it
                 * and the sweep does not end while one is left. */
                if ((uint32_t)idx[j] < best_idx) {
                    earlier_left = 1;
                    continue;
                }
                f = zpse_block_floor(n, tids[j], scounts[j], meta_lens[j]);
                if (f < min_floor) {
                    min_floor = f;
                }
            }
            if (have_best && !earlier_left && best_total <= min_floor) {
                break;
            }
            /* The sweep IS going to spend another trial, so price the rest
             * exactly and keep only what can still win. This is the case
             * the prescreen is worst at: when it prices every candidate
             * alike -- data on a 4-bit alphabet, a nibble-packed record, a
             * text block -- it keeps them all and every one is encoded in
             * full, while a few exact tiles per candidate tell them apart.
             * Asked here, after a winner exists and after the floor test, a
             * block the sweep is about to finish, or one whose first trial
             * aborted (noise), pays nothing for it. */
            if (!exact_priced && oi + 1 < ncand && n >= 8192u) {
                exact_priced = 1;
                zpse_block_prune_exact(e, in,
                                       n < 16384u ? n : 16384u,
                                       tids, scounts,
                                       meta_bufs, meta_lens, idx, oi + 1,
                                       &ncand);
                min_floor = UINT32_MAX;
                earlier_left = 0;
                for (oj = oi + 1; oj < ncand; oj++) {
                    int j = oj;
                    uint32_t f;

                    if ((uint32_t)idx[j] < best_idx) {
                        earlier_left = 1;
                        continue;
                    }
                    f = zpse_block_floor(n, tids[j], scounts[j],
                                         meta_lens[j]);
                    if (f < min_floor) {
                        min_floor = f;
                    }
                }
                if (have_best && !earlier_left && best_total <= min_floor) {
                    break;
                }
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
