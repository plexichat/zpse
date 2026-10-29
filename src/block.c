#include "zpse/block.h"

#include "zpse/arena.h"
#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/dense.h"
#include "zpse/lz.h"
#include "zpse/predictor.h"
#include "zpse/segment.h"
#include "zpse/status.h"
#include "zpse/tile.h"
#include "zpse/transform.h"
#include "zpse/types.h"

#include <string.h>

/* Capacity of the transform sweep's candidate list, and of one candidate's
 * transform metadata. The list used to hold three shapes -- RAW_BYTES,
 * DELTA8 and a single plain transpose -- so both were sized to a small
 * constant and the metadata to two bytes (DELTA8's one seed byte plus slack).
 *
 * The sweep now also offers the narrower plain transposes and the
 * delta-transposes, which the format has always been able to encode and
 * decode but the encoder never picked. DELTA_TRANSPOSE_16 carries sixteen
 * seed bytes, so metadata grows to 16. See zpse_block_encode for why these
 * particular shapes and how their seeds are chosen. */
#ifndef ZPSE_MAX_CAND
#define ZPSE_MAX_CAND 10u
#endif
#ifndef ZPSE_MAX_CAND_META
#define ZPSE_MAX_CAND_META 16u
#endif

/* Block envelope codec.
 * No allocation; staging uses caller arenas. Helpers static. */

/* Opt-in block-sweep accounting (-DZPSE_BLOCK_STATS): what the transform
 * sweep actually spent on a block. The encoder's candidate list is not
 * observable from outside it, so "how many candidates did this block
 * really try, and how many of those tries were wasted?" cannot be measured
 * without counters -- and that answer decides where the sweep's cost is.
 *
 * Each counter is a global, so a multi-threaded caller would race; the
 * counters exist to answer questions about a benchmark run, not to be
 * part of the format. */
#if defined(ZPSE_BLOCK_STATS)
/* Blocks encoded (one call to zpse_block_encode that got past the size
 * checks). The denominator for every per-block average below. */
uint32_t zpse_block_blocks;
/* Candidates the list BUILT, before any pruning. */
uint32_t zpse_block_cands_built;
/* Candidates the decision probe PRICED (one exact probe per candidate). */
uint32_t zpse_block_cands_priced;
/* Candidates that survived the probe's margin and went into the trial
 * loop. */
uint32_t zpse_block_cands_kept;
/* Full block encodes actually SPENT (a trial that ran to the end of the
 * block). */
uint32_t zpse_block_trials;
/* Trials skipped by the per-candidate floor bound without any work. */
uint32_t zpse_block_trials_skipped;
/* How many times a trial was won, by transform id (0..10: RAW_BYTES,
 * DELTA8, and the plain/delta transposes at widths 2/4/8/16). */
uint32_t zpse_block_wins[16];
/* Total compressed bytes each transform id produced across all trials. */
uint32_t zpse_block_bytes[16];
/* How many trials each transform id ran. */
uint32_t zpse_block_trials_by_tid[16];
/* The winning transform id of each block, counted (one entry per block).
 * Slot 15 is the RAW fallback: the sweep found nothing that paid and the
 * block went out raw. */
uint32_t zpse_block_win_tid[16];
/* Where the winner sat in the trial order: the histogram of "which trial
 * actually won", 0 = the first trial. It is the direct answer to "is the
 * sweep's own ordering already right most of the time?", and therefore to
 * "would trying fewer candidates cost compressed size?". */
uint32_t zpse_block_win_pos[32];
/* How much smaller the final winner was than the FIRST trial's total,
 * summed over the blocks, and how many blocks had a first trial that was
 * not the winner. A non-zero sum means the later trials earned their
 * keep; a zero sum means they did not. */
uint32_t zpse_block_first_trial_delta;
uint32_t zpse_block_first_trial_lost;
/* The margin, in bytes, between the probe's best price and the worst
 * survivor's price -- the width of the band the probe kept. Summed. */
uint32_t zpse_block_margin_sum;
/* The best price the probe produced. Summed, for a per-block average. */
uint32_t zpse_block_price_sum;
/* Blocks where the probe's winner was also the trial loop's winner: the
 * probe got it right. */
uint32_t zpse_block_probe_correct;
/* Cycles spent in the decision probe (exact prices) and in the trial
 * loop (full block encodes), so the split between "deciding which shape to
 * try" and "trying the shapes" is measured rather than assumed. Read
 * with rdtsc, so the two counters are comparable to each other but not
 * to wall-clock time. */
uint64_t zpse_block_probe_cycles;
uint64_t zpse_block_trial_cycles;
/* A serialising cycle read: the counter must not be reordered across the
 * work it measures. LFENCE is enough on every x86-64 that runs this
 * build's AVX2 paths. */
static uint64_t blk_rdtsc(void)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    uint32_t lo;
    uint32_t hi;

    __asm__ __volatile__("lfence" ::: "memory");
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    __asm__ __volatile__("lfence" ::: "memory");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
    return 0u;
#endif
}
#endif

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

    /* Intra-block LZ form: one literal stream, one command stream, no
     * transform and no tiles. The block keeps the shape the raw fallback
     * uses (transform 0, one segment, no metadata) so a reader can tell
     * the two apart by flag alone, and every reference it carries is a
     * distance inside this block. */
    if ((h.block_flags & ZPSE_B_LZ) != 0) {
        if (h.transform_id != 0 || h.segment_count != 1u || meta_size != 0u) {
            return PSE_BAD_HEADER;
        }
        if (c <= (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
            return PSE_BAD_BLOCK_SIZE;
        }
        rc = zpse_lz_decode(in + ZPSE_BLOCK_HEADER_SIZE,
                            c - (uint32_t)ZPSE_BLOCK_HEADER_SIZE, output, n);
        if (rc != PSE_OK) {
            return rc;
        }
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
        /* A 16-wide interleave transposes sixteen planes whose bases are
         * then `seg_len` apart -- for a 64 KiB block exactly 4096, and the
         * L1 set index is (address / 64) mod 64, so a group's sixteen
         * lines all land on one set; that is why the inverse stages a
         * copy of every plane before transposing it. Laying the planes
         * out 4160 apart (one extra cache line per plane) walks the set
         * index along by one per plane and keeps every plane aligned, and
         * then the inverse can read them in place: measured 1.28-1.46x
         * faster than the staged form (see zpse_t16x16_inverse_direct).
         * Only id 6 reaches the w == 16 inverse path, so only it is laid
         * out this way; the planes go eight and eight into the two 64 KiB
         * decoder regions, which needs 7 * 4160 + 4096 = 33216 bytes of
         * each. */
        int skew_layout = (identity_shape == 0 &&
                           h.transform_id == (uint8_t)ZPSE_T_TRANSPOSE_16 &&
                           scount == 16u) ? 1 : 0;
        uint32_t skew_off = 0;

        lanes.order1 = d->order1;
        lanes.o2 = d->o2;
        lanes.gen = 0u;
        /* The generation restarts with the lane state, so the previous
         * block's entries would carry the same stamps; the first tile that
         * names ORDER2 wipes the table (zpse_lane_o2_use). Wiping here
         * instead cost every block 128 KiB of memset, whichever predictors
         * its tiles turned out to use. */
        lanes.o2_ready = 0;
        if (lanes.order1 == 0 || lanes.o2 == 0) {
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
            } else if (skew_layout != 0) {
                uint8_t *region = (s < 8) ? d->recon : d->out_stage;

                if (s == 8u) {
                    skew_off = 0;
                }
                plane = region + skew_off;
                skew_off += seg_lens[s] + ZPSE_D_PLANE_SKEW;
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
        zpse_segment_prescreen(pl + off, nt, pl, off, pl[0], ks, 0u);
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
    /* Reorder the prescreen scan order by win count at every block start,
     * so the predictor that wins most tiles runs first. */
    seg_screen_rank();
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
    lanes.o2 = e->o2;
    lanes.gen = 0u;
    /* One wipe per trial, and only when the trial reaches ORDER2: the
     * generation restarts with the lane state, so the previous trial's
     * entries would carry the same stamps, but a trial whose tiles never
     * name ORDER2 has nothing to wipe. */
    lanes.o2_ready = 0;
    if (lanes.order1 == 0 || lanes.o2 == 0) {
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

    flags = (uint8_t)(ZPSE_B_RESET |
                      (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                           ? ZPSE_B_CRC : 0u));
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
                               in[0], ks, 0u);
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
                               0u, ks, 0u);
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

/* Tiles the decision probe prices per plane of a SINGLE-segment candidate
 * (see the sweep decision in zpse_block_encode): paid once per candidate
 * against a full trial's 256 tiles. Four is measured against sixteen: the
 * 768-case clen corpus came out byte-identical either way, the 2,718-case
 * fuzz corpus came out 34,174 bytes SMALLER at four (a coarse estimate ties
 * more often, and a tie goes to the earlier candidate -- the identity, which
 * is the better prior on the mixed blocks where the sample is noisiest), and
 * it took 22 us off a 64 KiB block's encode that the probe was spending to
 * price candidates it then rejected.
 *
 * A candidate with more than one plane is priced one tile per plane instead,
 * whatever this says: the planes of a transpose are not interchangeable, so
 * there is no fewer-planes estimate of it that is not biased (see
 * zpse_block_exact_sum). */
#ifndef ZPSE_DECIDE_TILES
#define ZPSE_DECIDE_TILES 4u
#endif

/* Full block encodes the sweep may spend on a block whose own probe says it
 * does not compress at all -- the winner's price over the 16 KiB window is at
 * or above the same window written raw. Such a block falls back to RAW unless
 * a shape wins on a part of it the sample did not cover, which is the one
 * thing a sample cannot rule out, so the sweep still spends this many; it just
 * does not spend one encode per survivor on a block that is very likely to be
 * incompressible. Blocks the probe says DO compress are swept in full, which
 * is what keeps the ratio: the 3-value-alphabet case of the ratio corpus is
 * decided by its fifth survivor (16696 bytes against 16864 for the first
 * three), and "zero with one byte every 256" by a shape the probe ranks
 * fourth. */
#ifndef ZPSE_PROBE_WINDOW
#define ZPSE_PROBE_WINDOW 12288u
#endif


#ifndef ZPSE_FLAT_TRIALS
#define ZPSE_FLAT_TRIALS 3
#endif

/* The band, as a fraction of the best window price (in 1/2^N), inside which
 * the probe cannot separate two candidates, and how many of a field inside it
 * are still tried in full (see the near-tie prune in zpse_block_encode).
 *
 * The band is 1/64, i.e. 1.6%: a full trial costs about six times a window
 * price, so a field that prices within 1.6% of itself is a field where the
 * trials are being paid to break a tie the block total barely reflects.
 * dict_u8 is the measured case: its five survivors price within a fraction of
 * a percent of each other, the four extra trials buy 216 bytes in 4 MiB
 * (0.01%), and they are most of the block's encode time (342 -> 656 MB/s with
 * the field cut to its best two).
 *
 * The floor is the other half of the trade, and it is 64 KiB: only a block at
 * the format's full size prunes. A full block's trial is the most expensive
 * thing the sweep can do (256 tiles, up to sixteen planes), so that is where
 * the trade pays; every smaller block is settled exactly, because a short
 * block's trial is cheap enough that pruning cannot buy anything worth its
 * risk -- and the probe's sample covers a bigger fraction of it anyway, which
 * is the opposite of the case the prune exists for. */
#ifndef ZPSE_TIE_BAND_SHIFT
#define ZPSE_TIE_BAND_SHIFT 6u
#endif
#ifndef ZPSE_TIE_MIN_N
#define ZPSE_TIE_MIN_N 65536u
#endif
#ifndef ZPSE_TIE_KEEP
#define ZPSE_TIE_KEEP 2u
#endif

static int zpse_block_planes(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint8_t tid, uint8_t sc,
                             const uint8_t *meta_bytes, uint16_t meta_len,
                             uint8_t *planes[16], uint32_t lens[16]);

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
    lanes.o2 = e->o2;
    lanes.gen = 0u;
    lanes.o2_ready = 0;
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

/* Exact price of one candidate, summed the same way the prescreen proxy sums
 * its planes.
 *
 * `every_plane` is BALANCED and deeper (level 5+) and prices one tile
 * in every plane, which is the only way to rank a transpose correctly (see
 * the comment below). The cheap levels (1-4) price `tiles` planes instead
 * and charge the rest their average, or nothing at all when they are
 * constant: that is the estimate this code used to make everywhere, and it
 * is wrong by design in exactly one direction -- it under-prices the wide
 * transposes,
 * whose planes differ most from each other. */
static uint32_t zpse_block_exact_sum(ZpseEncoder *e, uint8_t *planes[16],
                                      const uint32_t lens[16], uint8_t sc,
                                      unsigned tiles, uint32_t limit,
                                      int every_plane)
{
    uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u;
    uint32_t records = 0u;
    uint32_t tile_records = 0u;
    unsigned tile_planes = 0u;
    unsigned step = 1u;
    unsigned s;

    /* 0 means "the probe's own default", exactly as zpse_block_exact_plane
     * reads it; it is resolved here because it is the depth this function
     * asks a single-segment plane for. */
    if (tiles == 0u || tiles > ZPSE_DECIDE_TILES) {
        tiles = (tiles == 0u) ? ZPSE_PROBE_TILES : ZPSE_DECIDE_TILES;
    }
    /* Every plane is priced, one tile each. A transpose's planes are NOT
     * interchangeable, and an estimate that prices a stride of them and
     * charges the rest the average of those is not merely coarse -- it is
     * biased towards whatever the priced planes happen to be worth. On the
     * decimal8 corpus the planes of the width-16 delta-transpose differ by
     * several times over, and pricing four of the sixteen (0, 4, 8, 12) put
     * the whole candidate at 1648 bytes on the 16 KiB window against the
     * 3728 it actually costs over the block, so it beat the width-8 plain
     * transpose (3276 -- and the cheapest shape the block has) by a margin
     * the block does not have. One tile in each of the sixteen planes prices
     * it at its cost, and the transpose that really wins is the one the
     * probe picks.
     *
     * The price of the deeper pass is 60 plane-tiles for the ten candidates
     * of a 64 KiB block instead of 36, which is why a constant plane is
     * tested BEFORE its tile is priced (see below). Measured end to end,
     * the pass is what buys decimal8 4.62:1 -> 5.82:1 (3.96:1 -> 4.01:1 over
     * the corpus).
     *
     * The cheap levels (1-4) used to price only a stride of the planes and
     * charge the rest their average -- which under-prices wide transposes
     * whose planes differ (DT8 on 2-byte records came out 49% larger than
     * BALANCED's pick). They now price every plane, one tile each; a single
     * tile per plane is enough to rank a transpose, and single-segment
     * candidates still pay fewer tiles than BALANCED's four,
     * which is where the cheap levels' speed comes from. */
    if (!every_plane) {
        step = 1u;
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
            uint32_t c;

            /* A constant plane does not need a tile priced: the dense record
             * form is the cheapest way to write it (12 bytes per 4 KiB
             * against 10 bytes per 256-byte all-hit tile), and the scan that
             * finds this out exits at the first differing byte, so it costs
             * nothing on a plane that is not constant. It is what keeps the
             * deeper pass affordable on the blocks whose planes mostly are
             * constant -- zeros, a run of offsets, a repeated string -- where
             * a tile in each of sixteen constant planes was most of the
             * probe. */
            if (sc > 1u) {
                uint32_t d = zpse_segment_const_cost(planes[s], lens[s]);

                if (d != UINT32_MAX) {
                    records += d - 16u;
                    continue;
                }
            }
            c = zpse_block_exact_plane(e, planes[s], lens[s],
                                       sc > 1u ? 1u : tiles, share);

            if (c == UINT32_MAX) {
                return UINT32_MAX;
            }
            /* c is 16 + the plane's record area folded up to its tile count. */
            records += c - 16u;
            tile_records += c - 16u;
            tile_planes++;

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
    /* The planes the cheap levels did not price: a constant one is priced by
     * the scan, and anything else is charged the average of the planes that
     * were. Zero tile-priced planes means the whole candidate was constant,
     * so there is nothing left to fill in. */
    if (step > 1u && tile_planes > 0u) {
        uint32_t avg = tile_records / tile_planes;

        for (s = 0u; s < sc; s++) {
            uint32_t d;

            if ((s % step) == 0u) {
                continue;
            }
            d = zpse_segment_const_cost(planes[s], lens[s]);
            records += (d != UINT32_MAX) ? (d - 16u) : avg;
        }
    }
    total += (uint32_t)sc * 16u + records;
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
                                       uint32_t limit, int every_plane)
{
    uint32_t lens[16];
    uint8_t *planes[16];

    if (zpse_block_planes(e, in, n, tid, sc, meta_bytes, meta_len, planes,
                          lens) != PSE_OK) {
        return UINT32_MAX;
    }
    return zpse_block_exact_sum(e, planes, lens, sc, tiles, limit, every_plane);
}

/* Exchange two candidate slots, moving the candidate itself in every array
 * that describes it -- including idx[], which carries its original list
 * index and is what the winner's tie-break compares. The attempt order is
 * realized this way and never as an index permutation over the list, so the
 * exact prune (which compacts the list) cannot leave the attempt loop
 * pointing at a slot it has since dropped: after the swap the loop simply
 * walks the surviving slots in order. */
static void zpse_block_swap_cand(uint8_t tids[ZPSE_MAX_CAND],
                                 uint8_t scounts[ZPSE_MAX_CAND],
                                 uint8_t meta_bufs[ZPSE_MAX_CAND][ZPSE_MAX_CAND_META],
                                 uint16_t meta_lens[ZPSE_MAX_CAND],
                                 uint8_t idx[ZPSE_MAX_CAND],
                                 int a, int b)
{
    uint8_t t8 = tids[a];
    uint8_t s8 = scounts[a];
    uint8_t i8 = idx[a];
    uint8_t mbuf[ZPSE_MAX_CAND_META];
    uint16_t l16 = meta_lens[a];
    unsigned m;

    /* Every metadata byte moves, not just the first two: a delta-transpose
     * candidate carries up to 16 seed bytes and swapping only two of them
     * would hand the other candidate's trial a seed it never priced. */
    for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
        mbuf[m] = meta_bufs[a][m];
    }

    tids[a] = tids[b];
    scounts[a] = scounts[b];
    idx[a] = idx[b];
    meta_lens[a] = meta_lens[b];
    for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
        meta_bufs[a][m] = meta_bufs[b][m];
    }

    tids[b] = t8;
    scounts[b] = s8;
    idx[b] = i8;
    meta_lens[b] = l16;
    for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
        meta_bufs[b][m] = mbuf[m];
    }
}

/* Encode the block as the intra-block LZ form into dst[0..cap). Returns
 * PSE_OK with *total_out set, or a negative status when the payload does
 * not fit (which the caller reads as "this candidate lost", not as an
 * error). The block header is written last-to-first: the payload at
 * dst + 32, then the envelope around it. */
/* ---- The LZ form's decode bill, as a per-level acceleration factor ---- *
 *
 * An LZ stream decodes at the speed of its own command loop, one 4-byte
 * command per output run, so its rate follows the AVERAGE RUN LENGTH.
 * Measured on this corpus, in bytes of output per command: 5.6 -> 0.6 GB/s
 * (prose), 9 -> 1.2 (json), 17 -> 2.9 (var strings), 64 -> 8.8 (sparse),
 * 128 -> 8.4 (geo), 246 -> 24 (zeros). A tile stream decodes at 5-13 GB/s
 * whatever it holds (~12 GB/s on the shapes this encoder picks), so the two
 * cross at about ZPSE_LZ_BPC_CROSS output bytes per command and the only
 * question a block has to answer is how much decode speed it will trade for
 * size.
 *
 * That trade is the level's ACCELERATION FACTOR A -- the same idea as
 * zstd's -1/-2/-4, moved to the decode side: the form is taken when its
 * commands average at least ZPSE_LZ_BPC_CROSS / A output bytes, i.e. when
 * the LZ stream is expected to decode at least A times SLOWER than the tile
 * stream it replaces. A = 2 is the 48 bytes per command this rule used to
 * be hard-coded to (and it is nearly inert: it keeps LZ only for blocks
 * whose runs are long enough to out-decode the tiles); a large A buys ratio
 * on text and pays for it in decode rate, which is what the fastest and
 * default levels are being asked for now.
 *
 * Measured with every level set to each value in turn, on the four text
 * cases and a repetitive one, against the tile stream's own 1.09:1 at
 * ~14 GB/s (encode falls to ~100 MB/s on the cases that take the form,
 * because the match finder is a full pass):
 *
 *   A=2  (48 B/cmd)  no text case moves; the shipped rule before this knob.
 *   A=4  (24 B/cmd)  no text case moves either: var_string, the longest-run
 *                    text shape here, sits just under 24 bytes per command.
 *   A=8  (12 B/cmd)  var_string 1.09 -> 2.52 at 2.6 GB/s, and log_lines
 *                    1.09 -> 1.68 at 2.0 (part of its blocks clear the bar).
 *   A=12 (8 B/cmd)   + json_rows 1.09 -> 2.05 at 1.1 GB/s and log_lines the
 *                    rest of the way to 2.63 at 1.5. This is the value the
 *                    two deepest levels ship.
 *   A=16 (6 B/cmd)   unchanged from A=12: prose sits at 5.6 bytes per
 *                    command, just under the bar.
 *   A=24 (4 B/cmd)   + prose, at MAX only: text_en 1.09 -> 1.33 at 0.6 GB/s.
 *                    The default level does not take it because its sample
 *                    margin (1.25) rejects prose before this rule is asked;
 *                    MAX's (1.0625) does not. That is the whole difference
 *                    between the two levels' LZ behaviour.
 *
 * So A=12 is the setting that buys the structured-text cases while keeping a
 * GB/s decoder, and A=24 buys prose at a command-loop 0.6 GB/s. Each level
 * names its own; the levels must stay ordered by permissiveness -- a deeper
 * level that accepted LESS than the level before it would write a larger
 * block than that level, which tests/suites/test_levels.c fails on. */
#ifndef ZPSE_LZ_BPC_CROSS
#define ZPSE_LZ_BPC_CROSS 96u
#endif
#ifndef ZPSE_LZ_ACCEL_MAX
#define ZPSE_LZ_ACCEL_MAX 24u
#endif
#ifndef ZPSE_LZ_ACCEL_BALANCED
#define ZPSE_LZ_ACCEL_BALANCED 12u /* levels 5-6: 8 bytes per command */
#endif
#ifndef ZPSE_LZ_ACCEL_L7
#define ZPSE_LZ_ACCEL_L7 16u       /* level 7: 6 bytes per command */
#endif
#ifndef ZPSE_LZ_ACCEL_L8
#define ZPSE_LZ_ACCEL_L8 20u       /* level 8: 4.8 bytes per command */
#endif

/* The sample margins, in sixteenths: the win on the sample a level will pay
 * a match-finder pass for is 16 / num. Smaller is looser, and the ladder
 * must never loosen upward (see the level table: a deeper level that
 * accepted less would write a larger block than the one above it). */
#ifndef ZPSE_LZ_SCREEN_NUM_MAX
#define ZPSE_LZ_SCREEN_NUM_MAX 17u /* 1.0625x, levels 8-9 */
#endif
#ifndef ZPSE_LZ_SCREEN_NUM_L7
#define ZPSE_LZ_SCREEN_NUM_L7 18u      /* 1.125x, level 7 */
#endif
#ifndef ZPSE_LZ_SCREEN_NUM_L6
#define ZPSE_LZ_SCREEN_NUM_L6 19u      /* 1.1875x, level 6 */
#endif
#ifndef ZPSE_LZ_SCREEN_NUM_BALANCED
#define ZPSE_LZ_SCREEN_NUM_BALANCED 20u /* 1.25x, level 5 */
#endif

/* The smallest win that pays for the LZ form's slower decode. A stream one
 * byte smaller is not worth trading a 12 GB/s decoder for a command loop,
 * and the rule is deliberately coarse: the LZ form's own size is estimated
 * well (the sweep compares real totals), the win test only has to stop the
 * one-byte "wins", which are noise on the sampling and cost 20x the decode.
 * 1/32 is the same margin the candidate sweep keeps. */
#ifndef ZPSE_LZ_MIN_WIN_DIV
#define ZPSE_LZ_MIN_WIN_DIV 32u
#endif

/* ---- The effort ladder, one row per level --------------------------- *
 *
 * Every knob that decides WHICH shape a level writes is in this table. The
 * table exists because the thresholds it replaces made the ladder shorter
 * than it claimed to be: the sweep tested `lvl >= 5`, `lvl >= 7`, `lvl >= 9`
 * and `lvl <= 2`, so levels 2, 4, 6 and 8 evaluated every test exactly as
 * their neighbour below did and were aliases of it, not rungs. A level that
 * appears on the ladder has to be a level, so each row differs from the row
 * above it in at least one column and the columns are ordered so that the
 * deeper row can only choose from a superset of the same shapes:
 *
 *   trials      full block encodes the sweep may spend; 0 = every survivor
 *               (a level that tries more candidates can only take a smaller
 *               total, never a larger one).
 *   screen_num  the LZ sample margin in sixteenths -- the win on the sample
 *               that pays for a match-finder pass is 16/num. Smaller num is
 *               a looser margin, and a pass that runs can only be kept when
 *               it is smaller, so the deeper row is never the larger one.
 *   samples     how many samples the screen may spend (2 = head and middle,
 *               1 = head only). A screen that refuses fewer passes keeps
 *               more candidates, which again can only lower the total.
 *   accel       the LZ decode bill (ZPSE_LZ_BPC_CROSS / accel bytes per
 *               command); a larger factor accepts streams with shorter runs,
 *               so it is a superset of the row above for the same reason.
 *
 * Levels 1-4 write no LZ form, so their LZ columns are zero and unused.
 * The prescreen width (level 1 cheap, 2-9 full) and the every-plane pricing
 * (5-9) are not here: they are single thresholds shared by whole tiers of
 * the ladder, and they live in zpse_enc_speed's own comment in
 * include/zpse/constants.h. */
typedef struct {
    uint8_t trials;
    uint8_t screen_num;
    uint8_t samples;
    uint8_t accel;
} ZpseLevel;

static const ZpseLevel zpse_level[10] = {
    /* 0: never used -- an empty level field normalises to BALANCED. */
    { 0u, 0u, 0u, 0u },
    /* 1 FASTEST: the probe's winner only, no match finder. */
    { 1u, 0u, 0u, 0u },
    /* 2: full nine-predictor prescreen, two trials. */
    { 2u, 0u, 0u, 0u },
    /* 3 FASTER: the same prescreen, three trials. */
    { 3u, 0u, 0u, 0u },
    /* 4: four trials, still no match finder. */
    { 4u, 0u, 0u, 0u },
    /* 5 BALANCED (default): every survivor, 1.25x screen, 8 B/command. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_BALANCED, 2u,
      (uint8_t)ZPSE_LZ_ACCEL_BALANCED },
    /* 6: the screen loosened to 1.1875x, same decode bill. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_L6, 2u,
      (uint8_t)ZPSE_LZ_ACCEL_BALANCED },
    /* 7: 1.125x, 6 B/command. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_L7, 2u, (uint8_t)ZPSE_LZ_ACCEL_L7 },
    /* 8: MAX's 1.0625x margin, two samples, 5 B/command. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_MAX, 2u, (uint8_t)ZPSE_LZ_ACCEL_L8 },
    /* 9 MAX: the head sample only, 4 B/command. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_MAX, 1u,
      (uint8_t)ZPSE_LZ_ACCEL_MAX }
};

/* The row for a level number (1-9). An out-of-range level cannot reach here
 * -- enc_flags_ok rejects it -- so the default is a bound, not a policy. */
static const ZpseLevel *zpse_level_row(uint32_t lvl)
{
    return &zpse_level[lvl <= 9u ? lvl : 9u];
}

/* Minimum output bytes per LZ command at this level's acceleration factor. */
static uint32_t zpse_lz_min_bpc(uint32_t speed)
{
    uint32_t accel = (uint32_t)zpse_level_row(zpse_enc_speed(speed))->accel;
    uint32_t bpc;

    if (accel == 0u) {
        accel = 1u;
    }
    bpc = (uint32_t)ZPSE_LZ_BPC_CROSS / accel;
    return bpc == 0u ? 1u : bpc;
}

/* Do the two cheap levels run the match finder too? Off by default, and the
 * measurement is why. On the three structured-text corpora the form wins
 * 1.09 -> 2.03/2.63/2.52, but the pass that finds it runs at ~600 MB/s while
 * the tile encoder runs at 2.4 GB/s on the same blocks, so FASTEST's encode
 * on those cases falls from 2320 MB/s to 391 -- slower than the DEFAULT
 * level, which is the one thing an effort ladder must not do: the levels
 * would no longer be ordered in time, and a caller asking for the fastest
 * setting would get a five-times-slower encoder. The default and the deepest
 * level pay that pass for the ratio (BALANCED 1900 -> 426 MB/s on JSON for
 * 1.09 -> 2.05, and prose at MAX 1.09 -> 1.33); FASTEST and FASTER keep the
 * tile stream and their own 2.4 GB/s. */
#ifndef ZPSE_LZ_CHEAP_LEVELS
#define ZPSE_LZ_CHEAP_LEVELS 0
#endif

/* Which levels run the match finder at all: BALANCED and deeper price
 * every block -- the pass is what those levels are for -- and, when
 * ZPSE_LZ_CHEAP_LEVELS is on, the cheap levels price only the blocks the
 * ASCII gate has already convicted of being text, which is the one piece of
 * evidence about the tile family the gate hands back for free. */
static int zpse_block_lz_level(uint32_t speed, uint32_t ascii_gate)
{
    uint32_t lvl = zpse_enc_speed(speed);
    (void)ascii_gate;
#if ZPSE_LZ_CHEAP_LEVELS
    return lvl >= ZPSE_ENC_SPEED_LZ_MIN || ascii_gate != 0u;
#else
    return lvl >= ZPSE_ENC_SPEED_LZ_MIN;
#endif
}

/* ---- The sample screen the LZ candidate passes first ---------------- *
 *
 * The match finder is the most expensive candidate in the sweep: measured
 * with tests/tools on 64 KiB blocks it runs at 5.4-10.9 GB/s on the
 * repetitive corpora but 575-610 MB/s on the text-shaped ones, against
 * 2.3-2.5 GB/s for the tile encoder on the same blocks. So a level that
 * offers the candidate on every block pays several trials' worth of time
 * for a form that usually loses -- which is why only MAX used to offer it.
 *
 * Before the full pass, the same question is asked on the head of the
 * block: the match finder encodes ZPSE_LZ_SCREEN_BYTES of it, the shape the
 * sweep wants prices the same span with the probe's own machinery, and the
 * pass only goes ahead when the LZ form wins that sample by the level's
 * margin (ZPSE_LZ_SCREEN_*, in sixteenths). The screen is a lower bound on
 * the win, so it can only cost a block the ratio a full pass would have
 * bought on a sample that lies -- and it is what makes the candidate
 * affordable outside MAX: a block with nothing to find never reaches the
 * match finder at all, and the shapes that do reach it keep their 2.0-2.6:1
 * (see the acceleration table above).
 *
 * The sample is 1 KiB, and that is a measurement rather than a default: at
 * MAX, zeros_burst's LZ form is taken at 12.23:1 with a 1 KiB sample, 11.55
 * with 2 KiB, 10.22 with 4 and 10.18 with 8, because a sample bigger than
 * the burst spacing describes the block no better while giving the tile side
 * a bigger span to win on. Prose's win on the sample lies between MAX's
 * margin (accepted) and the default level's (rejected), which is exactly
 * where a threshold should sit: the level that exists to buy ratio pays the
 * pass for it, the balanced level does not.
 *
 * The margins differ per level because they are the encode side of the same
 * trade the acceleration factor makes on the decode side: the deeper the
 * level, the smaller the sample win it will pay a pass for. MAX's margin is
 * the smallest of the four (it is the level that exists to buy ratio) and it
 * must be, since a deeper level that accepted LESS than the one before it
 * would write a larger block than that level -- which is what
 * tests/suites/test_levels.c pins. */
#ifndef ZPSE_LZ_SCREEN_BYTES
#define ZPSE_LZ_SCREEN_BYTES 1024u
#endif
#ifndef ZPSE_LZ_SCREEN_DEN
#define ZPSE_LZ_SCREEN_DEN 16u
#endif
#ifndef ZPSE_LZ_SCREEN_SLACK_NUM
#define ZPSE_LZ_SCREEN_SLACK_NUM 5u
#endif
#ifndef ZPSE_LZ_SCREEN_SLACK_DEN
#define ZPSE_LZ_SCREEN_SLACK_DEN 4u
#endif

/* Could a sample of `sn` bytes code to `lz_s` and still beat the block?
 *
 * The sample's bytes per input byte, scaled to the block, is the estimate;
 * `best_total` is what the sweep actually reached. The estimate reads high
 * against the block's own pass because the match finder only sees the bytes
 * it has already scanned: a 1 KiB sample cannot code the repeat that
 * arrives 40 KiB later (zeros_burst's sample-scaled LZ is 6.8 KiB against
 * the 5.3 KiB its pass produces, 28%). The slack is that correction, and a
 * verdict is only taken from a sample that loses by more than it. */
static int zpse_block_lz_sample_can_win(uint32_t lz_s, uint32_t sn,
                                        uint32_t n, uint32_t best_total)
{
    /* The sample's size before scaling: what a payload carries whatever its
     * input, i.e. the literal-count word and one command
     * (ZPSE_LZ_MIN_PAYLOAD). At 1 KiB it is 1% of the sample and forgettable;
     * scaled from 1 KiB to 64 KiB it is 8 bytes against a block that may code
     * to a few hundred, and repeating it 64 times turns a 30-byte period into
     * a block whose estimate is four times its real size (the text-ish corpus
     * case: 60 bytes on the sample, 584 on the block, and an estimate of 3840
     * that convicts it). */
    uint32_t body = lz_s > (uint32_t)ZPSE_LZ_MIN_PAYLOAD
                        ? lz_s - (uint32_t)ZPSE_LZ_MIN_PAYLOAD
                        : (uint32_t)ZPSE_LZ_MIN_PAYLOAD;
    uint64_t est = ((uint64_t)body * (uint64_t)n + ((uint64_t)sn - 1u)) /
                   (uint64_t)sn;
    uint32_t need = best_total -
                    (best_total / (uint32_t)ZPSE_LZ_MIN_WIN_DIV + 1u);

    return est * (uint64_t)ZPSE_LZ_SCREEN_SLACK_DEN <=
           (uint64_t)need * (uint64_t)ZPSE_LZ_SCREEN_SLACK_NUM;
}

/* Should the full match-finder pass be paid for? `tid`, `sc` and the
 * metadata describe the shape the sweep wants for this block, which is what
 * the sample's tile price is measured with; the sample itself is the head of
 * the block. Returns 1 (go ahead) when the sample cannot be coded or priced
 * -- the screen is an optimisation on the sweep, never a new way to drop a
 * candidate the sweep could not judge.
 *
 * `have_best` / `best_total` are the sweep's own verdict on the whole block,
 * and they are asked of the sample once its tile price has agreed with it:
 * the pass is what the sample is estimating, so the comparison it makes when
 * the stream is finished (zpse_block_lz_wins, against the same total) can be
 * made here from the sample. It is what stops the pass where LZ cannot win for a
 * reason the head cannot show -- a block whose tile win comes from a shape
 * the sample's 64-byte planes cannot express. uuid_mix is that block:
 * sixteen 6-byte-prefix records code as one command per record under LZ
 * (0.875 of a byte per byte) and 0.63 under the width-16 transpose, and the
 * pass aborted at its own size cap after spending 68% of the block's encode
 * time. The tile price of the same shape on the same 1 KiB sample is far
 * worse than its 64 KiB price, so the old test accepted it; the sweep's own
 * total cannot make that mistake.
 *
 * The estimate is taken from two samples, the head and the middle, and only
 * a block both of them convict is refused. One sample is not enough: a
 * block's head can be a burst of noise in front of 64 KiB of zeros
 * (zeros_burst), and that head prices LZ at several times what the block's
 * pass reaches. The second sample is paid only by a block the first has
 * already convicted, which is the block that was about to pay a full pass. */
static int zpse_block_lz_screen(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                                uint8_t tid, uint8_t sc, const uint8_t *mb,
                                uint16_t meta_len, uint32_t speed,
                                int have_best, uint32_t best_total)
{
    uint32_t lvl = zpse_enc_speed(speed);
    uint32_t sn = n < (uint32_t)ZPSE_LZ_SCREEN_BYTES
                      ? n
                      : (uint32_t)ZPSE_LZ_SCREEN_BYTES;
    const ZpseLevel *row = zpse_level_row(lvl);
    uint32_t num = (uint32_t)row->screen_num;
    uint32_t lz_s;
    uint32_t tile_s;

    if (sn < ZPSE_LZ_MIN_MATCH + 1u) {
        return 1;
    }
    lz_s = zpse_lz_encode(in, sn, e->cand, (uint32_t)ZPSE_E_SZ_CAND, e->lz,
                          lvl == 1u
                              ? (uint32_t)ZPSE_LZ_HASH_BITS_FAST
                              : (uint32_t)ZPSE_LZ_HASH_BITS);
    if (lz_s == 0u) {
        return 1;
    }
    tile_s = zpse_block_proxy_exact(e, in, sn, tid, sc, mb, meta_len,
                                    ZPSE_DECIDE_TILES, UINT32_MAX, 1);
    if (tile_s == UINT32_MAX) {
        return 1;
    }
    if ((uint64_t)tile_s * (uint64_t)ZPSE_LZ_SCREEN_DEN <
        (uint64_t)lz_s * (uint64_t)num) {
        /* The shapes the sweep offered cannot beat the sample. That is the
         * ordinary answer on prose, and it costs one priced sample. */
        return 0;
    }
    /* The tile price agrees with LZ on the sample; the block's own total may
     * still disagree, and that is the case this second question is for (see
     * the comment above). Two samples, the head and the middle, and only a
     * block both of them convict loses its pass. One is not enough: a
     * block's head can be a burst of noise in front of 64 KiB of zeros
     * (zeros_burst), and that head prices LZ at several times what the
     * block's pass reaches. Asked only after the tile price has already said
     * yes, so a block the old screen refused pays nothing new. */
    if (row->samples > 1u && have_best &&
        !zpse_block_lz_sample_can_win(lz_s, sn, n, best_total)) {
        uint32_t off = (n - sn) / 2u;
        uint32_t lz_m = zpse_lz_encode(in + off, sn, e->cand,
                                       (uint32_t)ZPSE_E_SZ_CAND, e->lz,
                                       lvl == 1u
                                           ? (uint32_t)ZPSE_LZ_HASH_BITS_FAST
                                           : (uint32_t)ZPSE_LZ_HASH_BITS);

        if (lz_m != 0u &&
            !zpse_block_lz_sample_can_win(lz_m, sn, n, best_total)) {
            return 0;
        }
    }
    return 1;
}

/* Is this LZ stream worth taking? Smaller than the tile winner (or the
 * sweep found nothing at all, in which case there is no tile stream to
 * compare against), inside the level's acceleration factor, and smaller by
 * enough to pay for the decode it gives up. `lz_cmds == 0` cannot happen
 * from zpse_lz_encode (its stream always ends in a command) but is kept as
 * "no evidence either way" rather than a division by zero. */
static int zpse_block_lz_wins(uint32_t n, uint32_t lz_cmds, uint32_t lz_total,
                              int have_best, uint32_t best_total,
                              uint32_t speed)
{
    if (have_best &&
        (lz_total >= best_total ||
         best_total - lz_total < best_total / (uint32_t)ZPSE_LZ_MIN_WIN_DIV +
                                     1u)) {
        return 0;
    }
    if (lz_cmds != 0u && n / lz_cmds < zpse_lz_min_bpc(speed)) {
        return 0;
    }
    return 1;
}

static int zpse_block_try_lz(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint32_t in_crc, int is_last, uint32_t seq,
                             uint8_t *dst, uint32_t cap, uint32_t *total_out,
                             uint32_t *cmds_out)
{
    ZpseBlockHeader h;
    uint8_t flags;
    uint32_t payload;
    uint32_t total;

    if (e->lz == 0 || dst == 0 || total_out == 0) {
        return PSE_BAD_ARG;
    }
    if (cap <= (uint32_t)ZPSE_BLOCK_HEADER_SIZE + ZPSE_LZ_MIN_PAYLOAD) {
        return PSE_OUTPUT_OVERFLOW;
    }
    payload = zpse_lz_encode(in, n, dst + ZPSE_BLOCK_HEADER_SIZE,
                             cap - (uint32_t)ZPSE_BLOCK_HEADER_SIZE, e->lz,
                             zpse_enc_speed(e->enc_flags) == 1u
                                 ? (uint32_t)ZPSE_LZ_HASH_BITS_FAST
                                 : (uint32_t)ZPSE_LZ_HASH_BITS);
    if (payload == 0u) {
        return PSE_OUTPUT_OVERFLOW;
    }
    /* The stream's own layout gives the command count for free: a four-byte
     * literal total, the literals padded to a word, then commands to the
     * end of the payload. */
    if (cmds_out != 0) {
        uint32_t lit_total = zpse_load_u32le(dst + ZPSE_BLOCK_HEADER_SIZE);
        uint32_t cmd_off = 4u + ((lit_total + 3u) & ~3u);

        *cmds_out = (payload > cmd_off) ? (payload - cmd_off) / 4u : 0u;
    }
    total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + payload;
    flags = (uint8_t)(ZPSE_B_LZ | ZPSE_B_RESET |
                      (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                           ? ZPSE_B_CRC : 0u));
    if (is_last) {
        flags |= ZPSE_B_LAST;
    }
    zpse_block_fill_header(&h, flags, n, total, 0u, 1u, 0u, seq, in_crc);
    zpse_block_header_serialize(dst, &h);
    zpse_block_fix_header_crc(dst);
    *total_out = total;
    return PSE_OK;
}

/* The ASCII gate's sample: how many bytes of the block are priced, and with
 * how many tiles. Eight tiles is enough for the prescreen to tell prose
 * from structure, and small enough that the whole decision costs a few
 * thousand cycles against the ~200,000 a 64 KiB sweep spends. */
#ifndef ZPSE_ASCII_SAMPLE
#define ZPSE_ASCII_SAMPLE 2048u
#endif
#ifndef ZPSE_ASCII_SAMPLE_TILES
#define ZPSE_ASCII_SAMPLE_TILES 4u
#endif
#ifndef ZPSE_ASCII_MIN_BLOCK
#define ZPSE_ASCII_MIN_BLOCK 4096u
#endif

/* Should this block take the seven-bit form without a sweep?
 *
 * Three shapes are priced on the head of the block and again on its tail,
 * each with the probe's limit set to what the form would cost over the same
 * span. The probe stops a candidate the moment its running price crosses
 * the limit, so on prose each of these prices is a few tile classifications
 * rather than a transform of the whole window. A shape that comes back at
 * or under the limit is a shape that could beat the form, and then the
 * answer is no: the sweep keeps the block, and the tile loop still offers
 * the form per tile, so nothing is lost but the shortcut.
 *
 * The tail sample is what keeps a block of text with a compressible tail --
 * or the reverse -- out of the shortcut; both ends have to look like prose.
 * The three shapes are the ones a text block could plausibly want:
 * identity, DELTA8 (digits and numbers), and a width-4 transpose (tabular
 * text). Widths 2, 8 and 16 price within a few percent of width 4 on the
 * data this fires on, and they cost the same samples again. */
/* 1 iff the first `n` bytes are all one value.
 *
 * The gate compares three shapes' exact price against what the seven-bit
 * form would cost over the same span, and a plane that is one value prices
 * at its own floor -- far below seven bits a byte -- so a constant sample
 * cannot convict. Testing for it on the two kilobytes the gate would have
 * sampled anyway is what keeps a one-value block from paying a whole-block
 * scan and six priced planes to learn nothing: that scan is ~1.3 us of a
 * ~3 us encode (measured: `zeros` encodes at 20 GB/s through the gate and
 * 37 at MAX, which does not run it).
 *
 * Skipping the gate can only send the block down the ordinary sweep, which
 * is the path every non-ASCII and every non-gated block takes, so the bytes
 * written are decided by the sweep either way. */
static int zpse_block_head_one_value(const uint8_t *in, uint32_t n)
{
    uint64_t rep;
    uint32_t i;
    uint8_t v;

    if (in == 0 || n < 64u) {
        return 0;
    }
    v = in[0];
    rep = UINT64_C(0x0101010101010101) * (uint64_t)v;
    for (i = 0u; i + 8u <= n; i += 8u) {
        uint64_t w;

        memcpy(&w, in + i, sizeof w);
        if (w != rep) {
            return 0;
        }
    }
    for (; i < n; i++) {
        if (in[i] != v) {
            return 0;
        }
    }
    return 1;
}

static int zpse_block_ascii_gate(ZpseEncoder *e, const uint8_t *in,
                                 uint32_t n)
{
    static const uint8_t tids[3] = {0u, 1u, 4u};
    static const uint8_t scs[3] = {1u, 1u, 4u};
    static const uint16_t metas[3] = {0u, 1u, 0u};
    uint32_t pn = n < (uint32_t)ZPSE_ASCII_SAMPLE ? n
                                                  : (uint32_t)ZPSE_ASCII_SAMPLE;
    uint32_t alim = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + zpse_ascii_len(pn);
    unsigned j;

    if (e == 0 || in == 0 || pn < 256u) {
        return 0;
    }
    for (j = 0; j < 3u; j++) {
        uint32_t h;

        if ((uint32_t)scs[j] > pn) {
            continue;
        }
        h = zpse_block_proxy_exact(e, in, pn, tids[j], scs[j], in, metas[j],
                                   ZPSE_ASCII_SAMPLE_TILES, alim, 1);
        if (h == UINT32_MAX || h <= alim) {
            return 0;
        }
        if (n > pn) {
            uint32_t t = zpse_block_proxy_exact(e, in + (n - pn), pn, tids[j],
                                                scs[j], in, metas[j],
                                                ZPSE_ASCII_SAMPLE_TILES,
                                                alim, 1);

            if (t == UINT32_MAX || t <= alim) {
                return 0;
            }
        }
    }
    return 1;
}

int zpse_block_encode(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                      uint32_t seq, int is_last,
                      uint8_t *out, uint32_t out_cap)
{
    uint8_t tids[ZPSE_MAX_CAND];
    uint8_t scounts[ZPSE_MAX_CAND];
    uint8_t meta_bufs[ZPSE_MAX_CAND][ZPSE_MAX_CAND_META];
    uint16_t meta_lens[ZPSE_MAX_CAND];
    uint8_t idx[ZPSE_MAX_CAND];
    uint32_t in_crc;
    int ncand = 0;
    uint32_t best_total = 0;
    uint32_t best_idx = 0;
    uint32_t flat_min = 0u;
    int have_best = 0;
    int fails = 0;
    int sample;
    int sample_flat = 0;
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

    /* Trials 3..: the transposed shapes, plain and delta, at every width the
     * block is long enough to support.
     *
     * The list used to offer ONE plain transpose and to pick its width from
     * n alone, so a 64 KiB block was only ever offered the width-16 shape.
     * That left the sweep's result on the table for every columnar corpus
     * whose record width is 4 or 8 bytes -- and, more importantly, it never
     * offered a DELTA-TRANSPOSE at all, although the format has always been
     * able to encode and decode ids 7..10 and tests/suites/test_transforms.c
     * has always covered them. A delta-transpose is the shape a smooth column
     * wants: the transpose puts one byte position per plane, and the delta
     * chain then runs down that plane, so a slow ramp becomes a plane of
     * near-zeros that the tile encoder can predict.
     *
     * Priced over the shootout's 18 corpora with tests/tools/cand_probe.c
     * (64 KiB blocks), best shape the OLD list could pick versus best shape
     * in the full set:
     *
     *   i32_seq      18904 ->  8996  DT4   -52.4%
     *   ts_us        20640 -> 13264  DT8   -35.7%
     *   sorted_near  21044 -> 16636  DT4   -20.9%
     *   decimal8     14386 -> 12672  T8    -11.9%
     *   ts_gapped    34960 -> 31680  T4     -9.4%
     *   i64_mixed    11644 -> 10804  DT8    -7.2%
     *
     * 13 of the 18 improved, mean 10.99% over those, and no corpus chose a
     * shape outside this list. The cost of the extra candidates is bounded by
     * the prescreen below, which prices them on a 16 KiB window and drops
     * everything more than ~6% behind the best estimate before any of them is
     * encoded in full.
     *
     * Seeds: a delta-transpose carries one seed byte per plane, and the chain
     * it seeds is p[j] = plane[j] - plane[j-1] with plane[-1] == seed, so the
     * first byte the plane writes is plane[0] - seed. Setting the seed to the
     * plane's own first byte makes that difference zero. Any seed round-trips
     * (it travels in the metadata), so only size depends on the choice, and
     * zero is the best of them. Plane s of a width-w transpose holds input
     * positions s, s+w, s+2w, ..., so its first byte is in[s] -- which is
     * exactly the rule DELTA8 already uses at width 1 (seed = in[0]).
     * Verified round-trip by the full suite and by test_transforms.c. */
    {
        unsigned j;

        for (j = 0; j < 4u; j++) {
            uint8_t w = (uint8_t)(2u << j);
            unsigned s;

            /* N < W is forbidden by the format, and in[] must hold w bytes
             * for the seeds below, so a block shorter than the width simply
             * does not get that pair of candidates. */
            if (n < (uint32_t)w) {
                continue;
            }
            if (ncand + 2 > (int)ZPSE_MAX_CAND) {
                break;
            }
            /* Plain transpose: no transform metadata. */
            tids[ncand] = (uint8_t)(3u + j);
            scounts[ncand] = w;
            meta_lens[ncand] = 0;
            ncand++;
            /* Delta-transpose: one seed per plane, each the plane's own
             * first byte, so the chain's first difference is zero. */
            tids[ncand] = (uint8_t)(7u + j);
            scounts[ncand] = w;
            for (s = 0; s < w; s++) {
                meta_bufs[ncand][s] = in[s];
            }
            meta_lens[ncand] = w;
            ncand++;
        }
    }

    /* Every candidate writes the same decoded CRC into its header (it is
     * the CRC of the uncompressed block), so it is computed once here
     * instead of once per trial over the same 64 KiB. */
#if defined(ZPSE_BLOCK_STATS)
    zpse_block_blocks++;
    zpse_block_cands_built += (uint32_t)ncand;
#endif
    /* A one-value block can take its CRC arithmetically. The scan is the
     * segment encoder's own AVX2 one, and its answer is handed back to that
     * encoder by pointer (e->const_plane) so the identity trial skips its
     * own scan: a constant block is scanned once, not twice. The scan exits
     * at the first differing byte, so a block that is not constant pays one
     * compare and writes no pointer. */
    e->const_plane = 0;
    if ((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u) {
        if (zpse_segment_plane_is_const(in, n)) {
            in_crc = zpse_crc32c_repeat(in[0], (size_t)n);
            e->const_plane = in;
        } else {
            in_crc = zpse_crc32c(in, n);
        }
    } else {
        /* No checksum asked for: nothing reads the field, so it is left
         * 0 and the pass over the block is not paid at all. */
        in_crc = 0u;
    }

    /* ---- ASCII gate -------------------------------------------- *
     * A block whose bytes are all below 0x80 has a form that costs seven
     * bits a byte -- 224 bytes per 256-byte tile -- with no predictor, no
     * mask and no residual codec. On prose that is the best this encoder
     * can do, and today it finds it the expensive way: the sweep prices
     * every shape, then encodes one in full, then the tile loop offers the
     * seven-bit form per tile and compares. Nine tenths of that work is
     * the sweep discovering that nothing predicts English.
     *
     * So the decision is taken here, before the sweep, and it is taken on
     * the evidence the sweep itself would have used: four tiles of the
     * identity, DELTA8 and width-4 transposed planes, priced exactly with
     * the probe's own limit set to what the seven-bit form would cost over
     * the same span. A probe that comes back ABOVE that limit has been
     * convicted -- it stopped early on a plane that could not win -- and if
     * all three shapes are convicted there is nothing left for the sweep to
     * try: a transform is one of those three or a width the sample did not
     * see, and the widths all price alike on the data this fires on.
     *
     * The gate never fires on data that compresses: a constant or periodic
     * ASCII plane prices at its own floor, far below seven bits a byte, so
     * the sample does not convict and the block keeps the sweep it has
     * always had.
     *
     * Every level gates, MAX included. MAX used to opt out on the argument
     * that it has the LZ form, which beats seven bits on prose, and that
     * the trial is the point of the level -- but the gate does not compete
     * with that trial: it decides only what the TILE side of a block costs
     * (one pack of the seven-bit form instead of a priced sweep), and
     * zpse_block_lz_level reads the level, not e->ascii_gate, so MAX still
     * runs the match finder and its stream still replaces the gated one
     * when it is smaller (see the candidate below). Opting out therefore
     * bought no ratio and cost the sweep on every convicted block: measured
     * at MAX, text_en 88 -> 294 MB/s, json_rows 101 -> 407, var_string 91 ->
     * 410 on 4 MiB blocks, with the streams identical -- and identical too
     * over the 779-case size corpus at levels 5, 7, 8 and 9. */
    e->ascii_gate = 0u;
    if (n >= (uint32_t)ZPSE_ASCII_MIN_BLOCK &&
        !zpse_block_head_one_value(
            in, (n < (uint32_t)ZPSE_ASCII_SAMPLE) ? n
                                                  : (uint32_t)ZPSE_ASCII_SAMPLE) &&
        zpse_ascii_scan(in, n) && zpse_block_ascii_gate(e, in, n)) {
        e->ascii_gate = 1u;
        /* Identity only: one trial, which now packs instead of searching,
         * and no pricing pass or second opinion to pay for first. */
        ncand = 1;
        sample_flat = 0;
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
    /* Nothing is pruned on the cheap proxy any more. It is a prescreen-shaped
     * estimate -- the ideal packing rate implied by each plane's miss count --
     * and it cannot order shapes that pack BELOW that rate. On a block that is
     * zero except for one byte every 256 it priced the width-4 transpose at
     * 448 bytes per 16 KiB and never priced the width-16 one at all: that
     * candidate's own floor (624 bytes, a valid bound) already exceeded the
     * margin around 448, so it was dropped unmeasured, while over the whole
     * block it is the cheapest shape there is. Widening the margin does not
     * fix that; it only moves which blocks the estimate gets wrong, and the
     * same margin is what kept decimal8's width-8 transpose, which is the
     * shape that block actually wants.
     *
     * What replaces it is one exact pass: every candidate is priced with the
     * segment encoder's own classify/mode/pack on a spread of tiles, with the
     * running verdict fed down as the probe's limit so a candidate the best
     * price has already beaten by the margin stops on the plane that convicted
     * it. The pass keeps the candidates within the margin (a like-for-like
     * comparison) and orders them by price, which is the order the trial loop
     * below walks. */
    if (ncand > 1 && n >= 8192u) {
        uint32_t ex[ZPSE_MAX_CAND];
        uint32_t pc[ZPSE_MAX_CAND];
        uint8_t order[ZPSE_MAX_CAND];
        int nord = 0;
        /* Price on the first 16 KiB only: staging a transform costs about
         * as much as the estimate itself, and a block's structure (record
         * width, delta-friendliness) is the same throughout, so a quarter
         * of it orders the candidates just as well. (8 KiB was measured
         * too: it is faster, but it mis-ranks a block whose structure only
         * appears past the first quarter -- a long noise prefix in front
         * of a compressible body went to RAW -- so the window stays at
         * 16 KiB.) */
        /* The window the candidates are priced on: the head of the block.
     *
     * It was 16 KiB -- a quarter of a 64 KiB block -- and it is now 12 KiB,
     * because the price of the window is paid once per candidate and the
     * sweep prices ten of them. Cutting it by a quarter is 5.4% of the whole encode
     * over the columnar corpus, measured with the two builds alternating
     * (698 -> 736 MB/s mean, ratio unchanged at 3.96:1, decode
     * unchanged). The window has to be big enough for the price to
     * mean something: the sizes corpus is what catches a window that
     * mis-ranks, and it did not move. */
        uint32_t speed = e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK;
        /* One window for every level. FASTEST was given its own, 8 KiB
         * against the others' 12 (the window is paid once per candidate, and
         * FASTEST acts on a single candidate), and it is not worth it: on the
         * geo_point corpus the shorter window picks a shape that is 13%
         * larger AND decodes at 1.9 GB/s against the 5.1 the 12 KiB window
         * picks (13.06:1), because 8 KiB is not enough of a burst-heavy
         * columnar block to see the repeats that rank it. The speed it bought
         * was ~4%, inside the run-to-run noise of this machine. */
        uint32_t pn = n < (uint32_t)ZPSE_PROBE_WINDOW
                          ? n
                          : (uint32_t)ZPSE_PROBE_WINDOW;
        /* BALANCED and deeper price every plane of a candidate, which is
         * what ranks a transpose correctly; the cheap levels price a stride
         * of them and fill the rest in (see zpse_block_exact_sum). */
        int every_plane = zpse_enc_speed(speed) >= ZPSE_ENC_SPEED_EVERY_PLANE_MIN;
        int pick = -1;
        /* The head window turned out NOT to describe the block: it priced
         * at the raw envelope and the tail window was the one that
         * compressed (or could not be priced at all). The near-tie prune
         * below must not trust a price on a window like that. */
        int head_misleads = 0;
        /* A survivor was kept because its CONTENT beat the winner's, not
         * because its total was close (see the compaction). Such a field
         * is not a tie, however close the totals are. */
        int content_kept = 0;
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

        for (ci = 0; ci < ncand; ci++) {
            ex[ci] = UINT32_MAX;
            pc[ci] = UINT32_MAX;
        }
        for (k = 0; k < nord; k++) {
            const uint8_t *mb;
            uint32_t lim = UINT32_MAX;
#if defined(ZPSE_BLOCK_STATS)
            uint64_t t_cycle = blk_rdtsc();
#endif

            ci = order[k];
            mb = meta_lens[ci] > 0 ? meta_bufs[ci] : in;
            /* The running verdict: a candidate the best price has already
             * beaten by the margin is offered only that margin, and the probe
             * stops it on the plane whose running bound crosses it. A
             * convicted candidate comes back as a finite bound above `lim`,
             * never UINT32_MAX, so it reads as priced-and-lost rather than
             * unpriced. */
            if (pick >= 0 && ex[pick] != UINT32_MAX) {
                lim = ex[pick] + (ex[pick] >> 4);
            }
            /* Arithmetic floor before any staging: a shape's smallest
             * possible total over the window is a function of its plane
             * lengths alone (ten bytes per 256-byte tile, twelve per 4 KiB
             * dense record), so a candidate already past the limit cannot
             * be a survivor and the transform staging plus the plane prices
             * that would prove it are pure cost. The probe's own early exit
             * cannot see this: it stops a candidate only after its first
             * plane has been staged and priced, which is most of the price
             * of a width-16 shape. Reported as a finite price above `lim`
             * -- like a candidate the probe convicted -- so the compaction
             * below drops it rather than treating it as unpriced (UINT32_MAX
             * is kept on purpose: the sweep must never drop something it
             * could not price). */
            if (lim != UINT32_MAX) {
                uint32_t fl = zpse_block_floor(pn, tids[ci], scounts[ci],
                                               meta_lens[ci]);

                if (fl != UINT32_MAX && fl > lim) {
                    ex[ci] = fl;
                    continue;
                }
            }
            ex[ci] = zpse_block_proxy_exact(e, in, pn, tids[ci],
                                            scounts[ci], mb, meta_lens[ci],
                                            ZPSE_DECIDE_TILES, lim,
                                            every_plane);
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_cands_priced++;
            zpse_block_probe_cycles += blk_rdtsc() - t_cycle;
#endif
            if (ex[ci] == UINT32_MAX) {
                pc[ci] = UINT32_MAX;
                continue;
            }
            /* The content price: what the sample actually measured, with the
             * candidate's fixed structural envelope -- block header, block
             * segment directory and one 16-byte segment header per plane --
             * taken back out. The sample can only see content; the envelope
             * is a constant it cannot judge, and it is what made the two
             * orders disagree on "spike every 256": the width-4 shape prices
             * at 404 + 112 bytes of envelope against 436 + 176 for the
             * width-8, so totals said 404 < 436 while the content said
             * 292 > 260 -- and the block is 1419 against 1211. A candidate
             * whose content beats the winner's can therefore not be dropped
             * on its total alone (see the prune below). */
            {
                uint32_t env = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
                               16u * (uint32_t)scounts[ci];

                pc[ci] = ex[ci] > env ? ex[ci] - env : 0u;
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
            /* Keep what the winner has not beaten by the margin, and order
             * the survivors by price: the trial loop walks them in that order,
             * so the cheapest shape is attempted first and every later trial
             * is handed the tightest budget it could still win against.
             * Anything that could not be priced is kept and sorts last (it has
             * no price to order by, and the sweep must not drop it).
             *
             * Two things get a trial: a total within the margin, and a
             * candidate whose CONTENT beats the winner's (see where pc[] is
             * filled in). The second is the case the margin cannot cover --
             * a shape whose envelope is what puts it behind -- and it costs
             * one bounded trial only on blocks where the two orders actually
             * disagree. The winner is always kept, so the loss there is a
             * trial, never a different block. */
            uint32_t limit = ex[pick] + (ex[pick] >> 4);
            int kept = 0;
            int win = -1;
            uint8_t win_idx = idx[pick];

            for (ci = 0; ci < ncand; ci++) {
                if (ci == pick || ex[ci] == UINT32_MAX ||
                    ex[ci] <= limit || pc[ci] < pc[pick]) {
                    if (ci == pick) {
                        win = kept;
                    } else if (ex[ci] != UINT32_MAX && ex[ci] > limit &&
                               pc[ci] < pc[pick]) {
                        content_kept = 1;
                    }
                    if (kept != ci) {
                        unsigned m;

                        tids[kept] = tids[ci];
                        scounts[kept] = scounts[ci];
                        meta_lens[kept] = meta_lens[ci];
                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            meta_bufs[kept][m] = meta_bufs[ci][m];
                        }
                        idx[kept] = idx[ci];
                        ex[kept] = ex[ci];
                    }
                    kept++;
                }
            }
            ncand = kept;
#if defined(ZPSE_BLOCK_STATS)
            /* The compaction above moved the winner to slot `win` (the
             * slot `kept` held when the winner was copied), so its price
             * is ex[win], not ex[pick] -- `pick` still names the slot the
             * winner had BEFORE the compaction. */
            {
                uint32_t best_price = ex[win];
                uint32_t worst = 0u;

                zpse_block_cands_kept += (uint32_t)kept;
                zpse_block_price_sum += best_price;
                for (ci = 0; ci < kept; ci++) {
                    if (ex[ci] != UINT32_MAX && ex[ci] > worst) {
                        worst = ex[ci];
                    }
                }
                zpse_block_margin_sum += worst - best_price;
            }
#endif
            /* Insertion sort on the survivors: at most ten of them, and a tie
             * keeps the original list order (idx ascending) so the attempt
             * order still cannot decide a tie. */
            {
                int a;

                for (a = 1; a < ncand; a++) {
                    int b = a;

                    while (b > 0 &&
                           (ex[b - 1] > ex[b] ||
                            (ex[b - 1] == ex[b] && idx[b - 1] > idx[b]))) {
                        uint8_t  st = tids[b - 1], ss = scounts[b - 1];
                        uint8_t  si = idx[b - 1];
                        uint16_t sl = meta_lens[b - 1];
                        uint32_t sx = ex[b - 1];
                        uint8_t  sb[ZPSE_MAX_CAND_META];
                        unsigned m;

                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            sb[m] = meta_bufs[b - 1][m];
                        }
                        tids[b - 1] = tids[b];
                        scounts[b - 1] = scounts[b];
                        idx[b - 1] = idx[b];
                        meta_lens[b - 1] = meta_lens[b];
                        ex[b - 1] = ex[b];
                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            meta_bufs[b - 1][m] = meta_bufs[b][m];
                        }
                        tids[b] = st;
                        scounts[b] = ss;
                        idx[b] = si;
                        meta_lens[b] = sl;
                        ex[b] = sx;
                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            meta_bufs[b][m] = sb[m];
                        }
                        b--;
                    }
                }
                /* The sort moved the winner: find its slot again by the
                 * original list index it carries. */
                pick = win;
                for (a = 0; a < ncand; a++) {
                    if (idx[a] == win_idx) {
                        pick = a;
                        break;
                    }
                }
            }
            /* Does the probe's own winner fail to beat the raw form over the
             * window it priced? Then the block is a raw-fallback candidate and
             * the trial loop below is capped (ZPSE_FLAT_TRIALS). */
            sample_flat = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn <= ex[pick];
            /* ...and the same window at the OTHER end of the block, because a
             * block whose front is noise and whose back is not is exactly the
             * case a front-only probe calls incompressible: the "noise then
             * constant" shapes of the ratio corpus, where capping the trials
             * costs a quarter of the block's compressed size. One extra price,
             * of the candidate the front window already likes, is enough to
             * tell that apart from a block that is noise everywhere -- and
             * only the winner is priced, not the whole list. */
            if (sample_flat && n > pn) {
                const uint8_t *wmb = meta_lens[pick] > 0 ? meta_bufs[pick] : in;
                uint32_t tp = zpse_block_proxy_exact(
                    e, in + (n - pn), pn, tids[pick], scounts[pick], wmb,
                    meta_lens[pick], ZPSE_DECIDE_TILES, UINT32_MAX,
                    every_plane);

                if (tp == UINT32_MAX ||
                    tp < (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn) {
                    sample_flat = 0;
                    head_misleads = 1;
                }
            }
        }
        if (pick >= 0 && pick != 0) {
            /* Move the exact-probe winner to slot 0 so the trial loop tries
             * the most likely winner first: its total becomes best_total and
             * the per-candidate floor exits discard shapes that provably
             * cannot win without a full block encode. ncand is deliberately
             * left at the prescreen survivor count -- every survivor is still
             * encoded in full (bounded by the running-best budget), so the
             * sweep keeps the true minimum even when the 16 KiB sample
             * mis-ranks the candidates (a noise prefix in front of a
             * compressible body, or two transforms within proxy error on a
             * uniform block). The tie-break still uses the original list
             * index carried in idx[], so the result stays order neutral. */
            unsigned m;
            uint8_t  tt = tids[0],  ts = scounts[0];
            uint16_t tl = meta_lens[0];
            uint8_t  ti = idx[0];
            uint8_t  tb[ZPSE_MAX_CAND_META];

            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                tb[m] = meta_bufs[0][m];
            tids[0] = tids[pick];
            scounts[0] = scounts[pick];
            meta_lens[0] = meta_lens[pick];
            idx[0] = idx[pick];
            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                meta_bufs[0][m] = meta_bufs[pick][m];
            tids[pick] = tt;
            scounts[pick] = ts;
            meta_lens[pick] = tl;
            idx[pick] = ti;
            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                meta_bufs[pick][m] = tb[m];
        }
        /* ---- A near-tied field is cut to its best two ------------------
         *
         * The survivor margin above keeps everything the winner has not
         * beaten by 1/16, which on a homogeneous column keeps almost the
         * whole list: every transposed shape prices the same record width
         * on the same data, so the candidates differ by a fraction of a
         * percent and the margin has no one to drop. The trial loop then
         * encodes each of them in full to find out what the price already
         * said -- and a full trial is ~6x a window price, so the tie is the
         * most expensive thing the sweep does.
         *
         * A field whose THIRD-best price is inside ZPSE_TIE_BAND of its best
         * is a field the price cannot order, and the block total it produces
         * differs by about the band. So the two cheapest survivors (the
         * winner, already in slot 0, and the next by price) are kept and the
         * rest stop here. This is the same trade the levels make one rung at
         * a time -- it just applies to the one shape of block where the rung
         * would otherwise spend every trial it has on a coin flip.
         *
         * Four things hold it back, each of them a case where a tied price is
         * known not to describe the block: a block short of the format's full
         * size (see ZPSE_TIE_MIN_N), a head window that the tail disproved, a
         * survivor kept because its CONTENT beat the winner's rather than its
         * total, and a field of fewer than four. The size gate in
         * tests/suites/test_sizes.c is what keeps those honest: without any
         * of them a tied field on "three-value alphabet" blocks grows by 24
         * bytes in 768 cases. */
        {
            uint32_t cap = (uint32_t)zpse_level_row(
                zpse_enc_speed(speed))->trials;

            /* Only a level that prices EVERY survivor can cut a field: the
             * cheap rungs already stop at their own trial count, and
             * cutting under them would only reorder the handful they try. */
            if (cap == 0u && ncand > (int)ZPSE_TIE_KEEP && n >= ZPSE_TIE_MIN_N) {
                /* The survivors are sorted by price after the compaction
                 * (ties in list order), so the third-cheapest is slot 2 and
                 * the ones to drop are the tail of that order. */
                uint32_t best = ex[0];
                uint32_t third = ex[ZPSE_TIE_KEEP];

                if (pick >= 0 && ncand >= 4 && !head_misleads &&
                    !content_kept && third != UINT32_MAX &&
                    third - best <= best >> ZPSE_TIE_BAND_SHIFT) {
                    ncand = (int)ZPSE_TIE_KEEP;
                }
            }
            /* The trials this level is willing to spend, from the level
             * table: FASTEST keeps the probe's winner only, each cheap rung
             * above it buys one more trial, and BALANCED and deeper price
             * every survivor in full. Buying a trial is the only thing the
             * cheap rungs do, so the table is also the whole difference
             * between 1, 2, 3 and 4. */
            if (pick >= 0 && cap != 0u && (uint32_t)ncand > cap) {
                ncand = (int)cap;
            }
        }
    }
#endif /* !ZPSE_KEEP_EXACT_SWEEP */

    /* Full block encodes actually spent. The flat-block cap below counts
     * SLOTS, and a slot can be skipped (its floor cannot beat the winner)
     * without a trial being run -- but the two paths that leave the loop
     * early after a trial do spend one. Counting the trials themselves is
     * what the cap means: "the sweep has spent N full encodes". */
    int ntried = 0;
    /* The gate's single trial writes straight into `out` (see the loop). */
    int direct_out = (e->ascii_gate != 0u && ncand == 1) ? 1 : 0;
    uint8_t *tdst = e->cand;
    uint32_t tcap = (uint32_t)ZPSE_E_SZ_CAND;
#if defined(ZPSE_BLOCK_STATS)
    /* The total the first trial produced, and whether it succeeded at all.
     * The difference between it and the block's final winner is exactly
     * what the later trials bought. */
    uint32_t first_trial_total = 0u;
    int have_first_trial = 0;
    /* The slot (in trial order) that produced the current winner. */
    int win_pos = -1;
#endif

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
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_trials_skipped++;
#endif
            continue;
        }
        mb = meta_lens[i] > 0 ? meta_bufs[i] : in;
        ntried++;
        /* The gated block has exactly one trial and it is going to win:
         * the seven-bit form is smaller than the raw fallback by
         * construction. Write it straight into the caller's buffer instead
         * of staging it through cand/best and copying it out again -- two
         * full-block copies that were most of what the gate's own encode
         * cost (measured: dickens 841 -> 1.3 GB/s). */
        if (direct_out) {
            tdst = out;
            tcap = out_cap;
        } else {
            tdst = e->cand;
            tcap = (uint32_t)ZPSE_E_SZ_CAND;
        }
#if defined(ZPSE_BLOCK_STATS)
        zpse_block_trials++;
        zpse_block_trials_by_tid[tids[i] & 15u]++;
#endif
#if defined(ZPSE_BLOCK_STATS)
        {
            uint64_t t_cycle = blk_rdtsc();

            rc = zpse_block_try_one(e, in, n, in_crc, tids[i], scounts[i], mb,
                                    meta_lens[i], is_last, seq, tdst, tcap,
                                    &t, &budget_hit, have_best, best_total,
                                    flat_min);
            zpse_block_trial_cycles += blk_rdtsc() - t_cycle;
        }
#else
        rc = zpse_block_try_one(e, in, n, in_crc, tids[i], scounts[i], mb,
                                meta_lens[i], is_last, seq, tdst, tcap, &t,
                                &budget_hit, have_best, best_total, flat_min);
#endif
        if (rc != PSE_OK) {
            /* A trial the budget stopped cannot win: its total only grows as
             * its remaining segments are written. That is a verdict on THIS
             * candidate and says nothing about the ones after it -- a shape
             * that packs worse on the front of a block can be the best one
             * over the whole of it -- so on its own it does not end the
             * sweep. The per-candidate floor test at the top of the loop
             * still skips a shape whose own bound cannot beat the winner
             * without spending the trial. */
            (void)budget_hit;
#if defined(ZPSE_BLOCK_STATS)
            if (!have_first_trial) {
                /* The first trial failed, so there is no first-trial total to
                 * compare against: the block's winner was found by a later
                 * trial with nothing to beat. */
                have_first_trial = 1;
                first_trial_total = UINT32_MAX;
            }
#endif
            /* A trial the budget stopped is a trial whose stream was already
             * bigger than the raw envelope before it reached the end of the
             * block: the budget IS the raw envelope and the trial ran into
             * it. On a block the sweep has priced as noise -- the winner's
             * price over the head window and over the tail window is at
             * least the raw envelope -- that is the verdict for the whole
             * sweep, because the shapes behind this one are priced no better
             * and run into the same budget. It saves two whole block encodes
             * on every block of noise, which is most of what such a block
             * costs to encode.
             *
             * The verdict is the trial's own, not the price's, which is what
             * keeps it honest: a block whose head and tail are noise but
             * whose middle is not -- the "noise then constant" shape of the
             * size corpus -- has a price that says flat and a trial that
             * says otherwise, and the trial wins here because it encodes the
             * whole block. A block that compresses still gets its
             * compression; only a block that really is noise stops after
             * one trial. */
            if (sample_flat) {
                break;
            }
            continue;
        }
        /* Smallest total wins; a tie goes to the candidate EARLIER in
         * the list, so the attempt order cannot change the winner. */
        if (!have_best || t < best_total ||
            (t == best_total && (uint32_t)idx[i] < best_idx)) {
            if (!direct_out && t > (uint32_t)ZPSE_E_SZ_BEST) {
                /* The trial ran and its result cannot be kept: the block is
                 * bigger than the best buffer, so the RAW fallback will
                 * carry it. That is a whole block encode spent for nothing,
                 * and on an incompressible block EVERY candidate lands
                 * here -- so this path has to reach the flat-block cap
                 * too, or the cap never fires for exactly the blocks that
                 * need it. */
                if (sample_flat && ntried >= ZPSE_FLAT_TRIALS) {
                    break;
                }
                continue;
            }
            if (!direct_out) {
                memcpy(e->best, tdst, t);
            }
            best_total = t;
            best_idx = (uint32_t)idx[i];
            have_best = 1;
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_bytes[tids[i] & 15u] += t;
            /* The winner changed: the previous holder's win is undone, so
             * the per-tid win count is tracked at the end of the block
             * rather than here. */
            win_pos = oi;
#endif
        }
#if defined(ZPSE_BLOCK_STATS)
        if (!have_first_trial) {
            have_first_trial = 1;
            first_trial_total = t;
        }
#endif
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
            /* The probe says this block does not compress (see
             * ZPSE_FLAT_TRIALS): stop after that many full encodes. */
            if (sample_flat && ntried >= ZPSE_FLAT_TRIALS) {
                break;
            }
            /* A trial that finished above the raw envelope is the same
             * verdict the budget gives a trial that ran into it (see the
             * budget-stopped path above): the shape does not beat the raw
             * bytes, and on a block the sweep has priced as noise the shapes
             * behind it are priced no better, so the sweep stops here too. */
            if (sample_flat &&
                t + 1u > (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
                break;
            }
            /* Two candidates that both finish above the raw envelope are
             * evidence that the block is incompressible, and the sweep spends
             * two whole trials to find out -- the rest abort on the noise
             * probe long before they write the block out. RAW itself (id 0)
             * is the baseline -- it never beats itself -- so it does not count
             * as a failure; otherwise a block whose first candidates are RAW
             * and DELTA8 would abort before reaching a transpose that actually
             * pays. */
            if (tids[i] != 0u &&
                t + 1u > (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
                fails++;
                if (fails >= 2) {
                    break;
                }
            }
        }
    }

    /* One more candidate, competing on size alone: the intra-block LZ form.
     * A level that takes it can never write a block larger than the same
     * level without it, and a block it wins is a copy loop to decode instead
     * of a tile reconstruction. Which levels run the match finder at all is
     * zpse_block_lz_level (MAX and the default level price every block, the
     * two cheap levels only the blocks the ASCII gate convicted of being
     * text), and what they will trade for it is the level's acceleration
     * factor (see ZPSE_LZ_ACCEL_*): the match finder is a full pass over the
     * block, so it is the most expensive candidate in the sweep. */
    if (zpse_block_lz_level(e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK,
                            e->ascii_gate) &&
        e->lz != 0) {
        /* Exact floor first: one command codes at most 259 bytes, so an
         * LZ payload can never be smaller than its 8-byte header plus one
         * 4-byte command per 259 bytes of output. A winner already under
         * that cannot be beaten, and on the blocks the tiles already
         * handle well -- a constant or all-hit plane is 248 bytes, far
         * below the ~1 KiB floor of a 64 KiB block -- the whole match
         * finder is skipped. */
        uint32_t lz_floor = (uint32_t)ZPSE_BLOCK_HEADER_SIZE +
                            ZPSE_LZ_MIN_PAYLOAD +
                            4u * ((n + ZPSE_LZ_MAX_MATCH - 1u) /
                                  ZPSE_LZ_MAX_MATCH);

        if ((!have_best || best_total > lz_floor) &&
            zpse_block_lz_screen(e, in, n, tids[0], scounts[0],
                                 meta_lens[0] > 0 ? meta_bufs[0] : in,
                                 meta_lens[0],
                                 e->enc_flags &
                                     (uint32_t)ZPSE_ENC_SPEED_MASK,
                                 have_best, best_total)) {
            uint32_t lz_total = 0u;
            uint32_t lz_cmds = 0u;
            uint32_t lz_speed = e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK;

            /* Cap the pass at the size it could still win at. `lz_wins`
             * keeps the form only when its total is smaller than the
             * winner's by ZPSE_LZ_MIN_WIN_DIV, so a payload that reaches
             * that cap is one the sweep would reject anyway -- and a stream
             * only grows as it is written, so stopping there cannot change
             * the verdict, only the time spent reaching it. This is the
             * whole of the win on the blocks where the form LOSES after
             * its screen accepted it (the screen samples 1 KiB and can be
             * wrong about the block): measured on the level-5 corpus,
             * uuid_mix 254 -> 289 MB/s (+13.7%) and json_rows +1.1%, every
             * stream unchanged. */
            uint32_t lz_cap = (uint32_t)ZPSE_E_SZ_CAND;

            if (have_best && best_total < (uint32_t)ZPSE_E_SZ_CAND) {
                uint32_t win = best_total / (uint32_t)ZPSE_LZ_MIN_WIN_DIV +
                               1u;

                if (best_total > win) {
                    lz_cap = best_total - win;
                }
            }
            rc = zpse_block_try_lz(e, in, n, in_crc, is_last, seq, e->cand,
                                   lz_cap, &lz_total, &lz_cmds);
            /* Smaller is not enough: the two forms decode at speeds that
             * have nothing to do with their ratio, so the win also has to be
             * big enough to pay for the decode it gives up and the command
             * loop has to be inside this level's acceleration factor. Both
             * tests are in zpse_block_lz_wins, with the measurements they
             * came from. */
            if (rc == PSE_OK &&
                zpse_block_lz_wins(n, lz_cmds, lz_total, have_best,
                                   best_total, lz_speed)) {
                if (direct_out) {
                    /* In direct_out mode, `out` is the caller's buffer with
                     * capacity `tcap` (= out_cap). `lz_total` can be up to
                     * `lz_cap` (ZPSE_E_SZ_CAND = 64 KiB when !have_best),
                     * which may exceed `tcap`. Guard the copy so we never
                     * write past the caller's buffer; if LZ doesn't fit,
                     * skip it and fall through to RAW (which checks out_cap
                     * itself at line 2744). */
                    if (lz_total <= tcap) {
                        memcpy(out, e->cand, lz_total);
                        best_total = lz_total;
                        best_idx = 15u;
                        have_best = 1;
                    }
                } else {
                    memcpy(e->best, e->cand, lz_total);
                    best_total = lz_total;
                    best_idx = 15u;
                    have_best = 1;
                }
            }
        }
    }

    /* RAW fallback when coded does not pay: coded >= 32 + N. */
    if (!have_best || (uint64_t)best_total >=
                      (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n) {
        uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n;
        ZpseBlockHeader h;
        uint8_t flags = (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET |
                                  (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                                       ? ZPSE_B_CRC : 0u));
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
#if defined(ZPSE_BLOCK_STATS)
        /* The RAW fallback won, but it is not a transform win: the sweep
         * found nothing that paid, and the block went out raw. */
        zpse_block_win_tid[15u]++;
#endif
        return (int)total;
    }

    if (out_cap < best_total) {
        return PSE_OUTPUT_OVERFLOW;
    }
    if (!direct_out) {
        memcpy(out, e->best, best_total);
    }
#if defined(ZPSE_BLOCK_STATS)
    {
        /* The winner's transform id: the candidate that carried best_total.
         * best_idx is the LIST index of the winner, and the list is sorted
         * by price, so the winner is found by scanning for it. */
        int wi;

        for (wi = 0; wi < ncand; wi++) {
            if ((uint32_t)idx[wi] == best_idx) {
                break;
            }
        }
        if (wi < ncand) {
            zpse_block_wins[tids[wi] & 15u]++;
            zpse_block_win_tid[tids[wi] & 15u]++;
        }
        /* Which trial won, and what the later trials bought over the first
         * one. A winner at position 0 means the sweep's own ordering was
         * right and every trial after it was spent for nothing. */
        if (win_pos >= 0 && win_pos < 32) {
            zpse_block_win_pos[win_pos]++;
        }
        if (have_first_trial && first_trial_total != UINT32_MAX &&
            first_trial_total > best_total) {
            zpse_block_first_trial_delta += first_trial_total - best_total;
            zpse_block_first_trial_lost++;
        }
    }
#endif
    return (int)best_total;
}
