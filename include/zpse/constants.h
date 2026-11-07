#ifndef ZPSE_CONSTANTS_H
#define ZPSE_CONSTANTS_H

#include <stdint.h>

/* PSE v1.0.0 wire constants. */

#define ZPSE_MAGIC              UINT32_C(0x31455350) /* "PSE1" LE */
#define ZPSE_FRAME_MAJOR        ((uint8_t)1)
#define ZPSE_FRAME_MINOR        ((uint8_t)0)
#define ZPSE_BLOCK_VERSION      ((uint8_t)1)
#define ZPSE_BLOCK_SYNC         ((uint16_t)0x50B6)
/* A page's header, half the size. The 32-byte header spends 12 of its bytes
 * on fields a page does not need to spell out -- the version, the tile size,
 * the predictor profile and reserved byte (all fixed by the format), and the
 * decoded CRC (a page stream that asks for a payload checksum keeps the long
 * header, because the field has nowhere to go here). What is left fits in
 * 16: sync, flags, transform, segment count, metadata length, both sizes as
 * 16-bit, the sequence number and a 16-bit header checksum.
 *
 * It is worth 16 bytes on every page that compresses, and a page is 4096: at
 * the shared-library dump's 63.6% that is 0.30 points of its ratio for no
 * work at all, and 0.34 points over the three dumps together (52 065 pages,
 * 45 000 of which shrink, 719 591 bytes). The sync value chooses the header: a reader that
 * sees 0x50B7 knows the block is a page's and reads 16 bytes before the
 * payload, and any other sync is the 32-byte header it has always been. */
#define ZPSE_BLOCK_SYNC_SHORT   ((uint16_t)0x50B7)

#define ZPSE_FRAME_HEADER_SIZE  32u
#define ZPSE_BLOCK_HEADER_SIZE  32u
#define ZPSE_BLOCK_HEADER_SIZE_SHORT 16u
/* Largest block a short header can describe: its sizes are 16-bit, and the
 * encoder only writes one for the page this format is asked for. */
#define ZPSE_BLOCK_SHORT_MAX    4096u

/* FrameIndex (Sec. 3.4): the random-access table a frame carries when its
 * ZPSE_F_FRAME_INDEX flag is set. It sits after the last block -- the
 * header's frame_index_offset points at it -- and holds one {offset, size}
 * pair per block, both relative to the start of the frame, so a reader can
 * decode block k without walking blocks 0..k-1. */
#define ZPSE_INDEX_MAGIC        UINT32_C(0x31584950) /* "PXI1" LE */
#define ZPSE_INDEX_HEADER_SIZE  16u
#define ZPSE_INDEX_ENTRY_SIZE   8u
#define ZPSE_SEGMENT_HEADER_SIZE 16u
#define ZPSE_TILE_HEADER_SIZE    8u
#define ZPSE_SEG_DIR_ENTRY_SIZE  8u

#define ZPSE_MAX_BLOCK_SIZE     UINT32_C(65536)
#define ZPSE_MIN_BLOCK_SIZE     UINT32_C(1)
#define ZPSE_TILE_SIZE          256u
#define ZPSE_MAX_TILE_LEN       256u
#define ZPSE_NUM_LANES          8u

#define ZPSE_MAX_SEGMENTS       16u /* max segment_count value (1,2,4,8,16) */
#define ZPSE_ORDER2_SIZE        4096u
#define ZPSE_COMB_MAX_BYTES     32u /* C(256,128) fits in 4x u64 limbs */

/* Frame flags (Sec. 3.2). */
#define ZPSE_F_BLOCK_CRC        (UINT16_C(1) << 0)
#define ZPSE_F_CONTENT_SIZE     (UINT16_C(1) << 1)
#define ZPSE_F_BLOCK_COUNT      (UINT16_C(1) << 2)
#define ZPSE_F_FRAME_INDEX      (UINT16_C(1) << 3)
#define ZPSE_F_RESERVED_MASK    (UINT16_C(0xFFF0))

/* Block flags (Sec. 4.2). */
#define ZPSE_B_RAW              (UINT8_C(1) << 0)
#define ZPSE_B_LAST             (UINT8_C(1) << 1)
#define ZPSE_B_CRC              (UINT8_C(1) << 2)
#define ZPSE_B_RESET            (UINT8_C(1) << 3)
#define ZPSE_B_HAS_META         (UINT8_C(1) << 4)
/* The block payload is the intra-block LZ form (include/zpse/lz.h)
 * rather than transform + segments + tiles. Mutually exclusive with
 * ZPSE_B_RAW; transform_id 0, segment_count 1, no transform metadata. */
#define ZPSE_B_LZ               (UINT8_C(1) << 5)
/* v2: LZ *and* tiles, in one block. The payload is
 *   [u32 body_len][intra-block LZ stream]
 * where the LZ stream expands to exactly the body the tile form would have
 * written after its 32-byte header (transform metadata, segment directory,
 * segment payloads). The transform, segment count and metadata length in
 * the header are the tile form's own, so the decoder unpacks the body once
 * and then runs the ordinary tile path over it.
 *
 * Mutually exclusive with ZPSE_B_RAW and ZPSE_B_LZ: a block is still one
 * form, it is just a form that carries the other one inside it. There is no
 * compatibility shim -- a v1 reader sees bit 6 in the reserved mask and
 * refuses the block, which is what the reserved bits were for. */
#define ZPSE_B_LZ_TILE          (UINT8_C(1) << 6)
/* The fast LZ form (src/lz2.c): a token, its literals and a fixed 16-bit
 * distance, with neither the repeat-distance cache nor the one-byte
 * lookahead that ZPSE_B_LZ spends its per-position budget on. It exists as
 * its own form because the two codecs want different rungs: the cheap levels
 * take this one on every block, the deep ones keep the repeat cache where
 * its size shows. Bit 7 was the reserved mask and is now the last form, so
 * the mask is empty rather than wrong. */
#define ZPSE_B_LZ2              (UINT8_C(1) << 7)
#define ZPSE_B_RESERVED_MASK    (UINT8_C(0x00))

#define ZPSE_ENC_CHECKSUM       (UINT32_C(1) << 0)

/* Write the fast LZ form and nothing else. With this set the block is one
 * greedy match-finder pass: no probe, no trial, no candidate list, no
 * decision -- which is the trade lz4 makes, and the reason to ask for it.
 * The streams it writes are the same ZPSE_B_LZ2 blocks the deepest level
 * offers as its last candidate, so a decoder needs to know nothing about the
 * option. What the caller gives up is the tile forms, and with them the ratio
 * on typed columns, where a transpose plus a predictor beats any LZ parse;
 * what it gets is one pass at the finder's own rate, with the literal and
 * command writing as the only other work. */
#define ZPSE_ENC_LZ_FAST        (UINT32_C(1) << 5)

/* Encoder effort (ZPSE_ENC_* bits 1-4). Nine levels, 1 = FASTEST ... 9 =
 * MAX. Stored as level << ZPSE_ENC_SPEED_SHIFT, so the raw field value is
 * 2,4,6,...,18. This trades compressed size for encode time and never
 * changes the format: every level writes a stream any decoder reads, and a
 * tile stream decodes at the same speed whichever level chose it, because
 * the shapes themselves are the same. What a level changes is how the sweep
 * decides which shape to write. The one exception is the intra-block LZ
 * form, which levels 5-9 may also pick: it decodes at the speed of its
 * command loop rather than the tile reconstruction, which is why the level
 * caps how many commands per output byte it will accept.
 *
 *  1  ZPSE_ENC_SPEED_FASTEST  one trial per block: the probe's winner, on the
 *                              cheap prescreen that tries only the three
 *                              uniform predictors (CONST_ZERO,
 *                              LANE_PREVIOUS, STRIDE_1). ASCII gate on every
 *                              block, no match finder.
 *  2                          full nine-predictor prescreen, two trials, no
 *                              match finder
 *  3  ZPSE_ENC_SPEED_FASTER   the same prescreen, three trials, no match
 *                              finder
 *  4                          four trials, still no match finder
 *  5  ZPSE_ENC_SPEED_BALANCED price the planes of every candidate at every
 *                              plane, keep those within a margin of the best,
 *                              and encode each survivor in full -- except on
 *                              a full-size block whose survivors the price
 *                              can order, where the best two are tried and
 *                              the rest dropped (see ZPSE_TIE_*, and the
 *                              tie_field column of the ladder). Also offer
 *                              the intra-block LZ form as one more candidate.
 *                              The match finder is a full pass at roughly a
 *                              quarter of the tile encoder's rate, so a sample
 *                              of the block is coded first and the pass is
 *                              paid for only when that sample shows the form
 *                              winning by 1.25x.
 *  6                          the LZ sample margin loosened to 1.1875x, and
 *                              the decode bill raised to an acceleration
 *                              factor of 16 (6 output bytes per command).
 *                              The bill is free on the encode side: it only
 *                              filters streams the pass has already produced,
 *                              so it can never make a block pay for a pass.
 *  7                          the margin loosened to 1.125x; the decode bill
 *                              unchanged from 6.
 *  8                          margin 1.0625x with an acceleration factor of
 *                              20 (4.8 bytes per command), and the field cut
 *                              back to a tied tail only: the deepest rungs
 *                              encode every survivor their price separated.
 *  9  ZPSE_ENC_SPEED_MAX      the same 1.0625x margin, one sample instead of
 *                              two (so more blocks pay for a pass), and an
 *                              acceleration factor of 24 (4 bytes per
 *                              command) -- the loosest decode bill any level
 *                              accepts
 *
 * The ladder is a table in src/block.c (zpse_level): each level differs from
 * the one above it in at least one knob, and every knob moves in the
 * direction that can only add candidates to the choice, so a deeper level
 * cannot write a larger block than a shallower one (tests/suites/
 * test_levels.c pins that). The cheap rungs buy trials one at a time; the
 * deep rungs loosen the LZ screen one notch at a time. No level's LZ
 * candidate can cost it size -- the form is kept only when it is smaller than
 * the best tile stream -- and the trade it makes is decode speed: an LZ
 * stream decodes at the speed of its command loop, so the level's
 * acceleration factor (see ZPSE_LZ_ACCEL_*) decides how much slower than the
 * tile stream it is allowed to be.
 *
 * An unsupported level is rejected with PSE_BAD_ARG rather than rounded to
 * one of these. */
#define ZPSE_ENC_SPEED_SHIFT    1
#define ZPSE_ENC_SPEED_MASK     (UINT32_C(0xF) << ZPSE_ENC_SPEED_SHIFT)

#define ZPSE_ENC_SPEED_FASTEST  (UINT32_C(1) << ZPSE_ENC_SPEED_SHIFT)  /* level 1 */
#define ZPSE_ENC_SPEED_L2       (UINT32_C(2) << ZPSE_ENC_SPEED_SHIFT)  /* level 2 */
#define ZPSE_ENC_SPEED_FASTER   (UINT32_C(3) << ZPSE_ENC_SPEED_SHIFT)  /* level 3 */
#define ZPSE_ENC_SPEED_L4       (UINT32_C(4) << ZPSE_ENC_SPEED_SHIFT)  /* level 4 */
#define ZPSE_ENC_SPEED_BALANCED (UINT32_C(5) << ZPSE_ENC_SPEED_SHIFT)  /* level 5 */
#define ZPSE_ENC_SPEED_L6       (UINT32_C(6) << ZPSE_ENC_SPEED_SHIFT)  /* level 6 */
#define ZPSE_ENC_SPEED_L7       (UINT32_C(7) << ZPSE_ENC_SPEED_SHIFT)  /* level 7 */
#define ZPSE_ENC_SPEED_L8       (UINT32_C(8) << ZPSE_ENC_SPEED_SHIFT)  /* level 8 */
#define ZPSE_ENC_SPEED_MAX      (UINT32_C(9) << ZPSE_ENC_SPEED_SHIFT)  /* level 9 */

/* The level an options field with no level bits encodes at: the default is
 * level 6, one rung below BALANCED.
 *
 * BALANCED (5) was the default, and level 6 is a superset of its trade rather
 * than a step away from it: the screen margin goes from 1.25x to 1.1875x and
 * the decode bill from 8 to 6 bytes per command, and neither can refuse a
 * block BALANCED would have taken -- a stream the finder offers can only be
 * kept when it is smaller, and a bill that accepts shorter runs only admits
 * more of them. Measured over Silesia (202 MiB of real data, one pinned core,
 * `tests/benchmarks/silesia.sh`): 113,246,135 packed bytes at level 5 against
 * 106,140,388 at level 6, i.e. 6.3% smaller, for 6% of the encode rate (315
 * against 296 MB/s on the prose pair).
 *
 * The two knobs are not equal in cost, which is what the level table's row 6
 * is built from: raising the bill alone is free -- it only filters streams the
 * match-finder pass has already produced, so it cannot add a pass -- while
 * the margin costs one, because it offers the finder more blocks. So the
 * default takes the free knob and leaves the paid one to level 7.
 *
 * What it costs is the ladder's cheapest rung below it: level 5 still exists
 * and still writes what it wrote, so a caller who wants that exact stream can
 * ask for it by number. */
#define ZPSE_ENC_SPEED_DEFAULT  (ZPSE_ENC_SPEED_L6 >> ZPSE_ENC_SPEED_SHIFT)

/* Extract the level number (1-9) from the options field, normalising the
 * empty field to the default level. The encoder still has a few tier tests --
 * "at least 5 prices every plane", "at least 5 runs the match finder",
 * "at most 1 screens three predictors" -- so a raw zero would put the default
 * encoder on the cheap side of all of them, which is how the default level
 * silently became the most expensive-quality regression the size gate has
 * ever caught. The normalisation is here, in the one accessor, rather than at
 * each test. */
static inline uint32_t zpse_enc_speed(uint32_t enc_flags) {
    uint32_t lvl = (uint32_t)((enc_flags & ZPSE_ENC_SPEED_MASK) >>
                              ZPSE_ENC_SPEED_SHIFT);

    return (lvl != 0u) ? lvl : (uint32_t)ZPSE_ENC_SPEED_DEFAULT;
}

/* Levels whose sweep prices every plane of every candidate (5-9). */
#define ZPSE_ENC_SPEED_EVERY_PLANE_MIN 5u
/* Levels that run the match finder at all (5-9). */
#define ZPSE_ENC_SPEED_LZ_MIN 5u
/* Levels whose segment prescreen is cut to the three uniform predictors (1
 * only). The prescreen is what half of FASTEST's encode is; every rung above
 * it has a second trial to order, and a runner-up chosen by a prescreen that
 * cannot see STRIDE or the lane-history family is the wrong runner-up. */
#define ZPSE_ENC_SPEED_CHEAP_PRESCREEN_MAX 1u
/* Every level uses the ASCII gate, MAX included: it decides only what the
 * tile side of a convicted block costs, and the LZ candidate is offered
 * afterwards whatever it decided, so no level can lose its LZ trial to it.
 * (MAX used to opt out of it, for nothing; see src/block.c.) */

/* Segment flags (Sec. 6.1). Only 0x01 is legal in v1. */
#define ZPSE_S_LEN_PREFIX       (UINT8_C(1) << 0)
#define ZPSE_S_DIR_BIT          (UINT8_C(1) << 1)

/* CRC32-C check value for "123456789". */
#define ZPSE_CRC32C_CHECK       UINT32_C(0xE3069283)

#endif /* ZPSE_CONSTANTS_H */
