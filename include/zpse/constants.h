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
 * decoded CRC (a page stream that asks for a payload checksum drops it: the
 * field has nowhere to go here, and eight header bytes on every page are
 * worth more than a check the caller can have at any larger block size).
 * What is left fits in
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
/* A one-value page: the 16-byte page header with the value in transform_id
 * and no payload at all. The decoder fills the block with that byte. A zero
 * page used to spend 16 header + 8 segment directory + 8 segment header +
 * 12 dense record = 44 bytes to say "4096 zeros"; the token says it in 16.
 * This is a v1.1 wire addition: an older reader sees an unknown sync and
 * refuses the block, which is the point of the sync byte. */
#define ZPSE_BLOCK_SYNC_TOKEN   ((uint16_t)0x50B8)
/* The FLAT page form: a 16-byte header that names a flat payload codec
 * instead of a transform + segment directory. This is a BREAKING wire change
 * (allowed here); the new sync is what selects it. All header bytes are
 * little-endian:
 *   [0..2)   u16 sync = ZPSE_BLOCK_SYNC_FLAT (0x50B9)
 *   [2]      u8  flags -- ZPSE_B_RESET required, ZPSE_B_LAST optional; every
 *                         other bit (RAW, CRC, HAS_META, LZ*, reserved) is
 *                         rejected
 *   [3]      u8  codec -- one of the ZPSE_FLAT_* values below
 *   [4]      u8  param -- codec parameter (becomes transform_meta_size)
 *   [5]      u8  reserved, MUST be 0
 *   [6..8)   u16 n     -- uncompressed size, 1..ZPSE_BLOCK_SHORT_MAX
 *   [8..10)  u16 c     -- compressed total size, >= 16
 *   [10..14) u32 seq   -- block sequence number
 *   [14..16) u16 crc16 -- low 16 bits of crc32c over bytes [0..14)
 */
#define ZPSE_BLOCK_SYNC_FLAT    ((uint16_t)0x50B9)
/* Values of FLAT header byte [3]. */
#define ZPSE_FLAT_RAW           ((uint8_t)0)
#define ZPSE_FLAT_NIBBLE4       ((uint8_t)1)
#define ZPSE_FLAT_BIT2          ((uint8_t)2)
#define ZPSE_FLAT_SUBLELANE     ((uint8_t)3)
/* The TEXT page form (sync ZPSE_BLOCK_SYNC_TEXT): a 16-byte header naming no
 * codec -- the payload is the caller's own byte stream. All header bytes
 * little-endian:
 *   [0..2)   u16 sync = ZPSE_BLOCK_SYNC_TEXT (0x50BA)
 *   [2]      u8  flags -- ZPSE_B_RESET required, ZPSE_B_LAST optional; every
 *                         other bit (RAW, CRC, HAS_META, LZ*, reserved) is
 *                         rejected
 *   [3..6)   u8  reserved, MUST be 0
 *   [6..8)   u16 n     -- uncompressed size; 0 means 65536, else literal n
 *   [8..10)  u16 c     -- total block size incl. this 16-byte header, >= 16
 *   [10..14) u32 seq   -- block sequence number
 *   [14..16) u16 crc16 -- low 16 bits of crc32c over bytes [0..14)
 * The writer declines a block when 16 + plen > 65535, since c is 16-bit. */
#define ZPSE_BLOCK_SYNC_TEXT    ((uint16_t)0x50BA)

/* FLAT7 stream format constants (src/flat7.c). A FLAT7 stream carries a
 * 20-byte base header followed by 16-byte region records (plus a 4-byte CRC
 * on some regions). The magic is stored little-endian, so the four bytes on
 * the wire spell "F7P1". */
#define ZPSE_FLAT7_MAGIC            UINT32_C(0x31503746) /* "F7P1" little-endian */
#define ZPSE_FLAT7_VER_MAJOR        ((uint8_t)1)
#define ZPSE_FLAT7_VER_MINOR        ((uint8_t)0)
#define ZPSE_FLAT7_BASE_SIZE        20u
#define ZPSE_FLAT7_REGION_SIZE      16u
#define ZPSE_FLAT7_CRC_SIZE         4u
#define ZPSE_FLAT7_REGION_ASCII7    ((uint8_t)0)
#define ZPSE_FLAT7_REGION_RAW       ((uint8_t)1)
#define ZPSE_FLAT7_MIN_SIZE         4096u

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
/* Eight bytes, not sixteen. The last eight carried a tile-directory size and
 * a reserved word, and the parser refused any value but zero for either --
 * so they were eight bytes of nothing on every segment. At the page this
 * format is asked for there is one segment a block, which is eight bytes off
 * every page's payload: 8 * 52 065 pages is 416 520 bytes on the dump corpus,
 * for no change in what a segment can express. */
#define ZPSE_SEGMENT_HEADER_SIZE 8u
/* A PRICE charges sixteen bytes a segment -- the envelope every pruning
 * threshold was swept with -- but only at the levels where the price is a
 * screen rather than the decision (see zpse_segment_price). Charging it
 * everywhere was measured and refused: it recovers the 64 KiB level-6 cost of
 * the shorter header (81 043 247 -> 80 571 956 bytes) and buys 16 KiB
 * 178 687, but levels 1 to 4 decide by that price and nothing else, so the
 * eight bytes a segment that are not there made 4 KiB level 1 grow 124 889 146
 * -> 125 592 685 (0.33 points of ratio), on the page size this format is for.
 * From level 5 up the price only orders candidates that are then encoded in
 * full, so the same envelope is free there. */
#ifndef ZPSE_SEGMENT_PRICE_MAX
#define ZPSE_SEGMENT_PRICE_MAX 16u /* -DZPSE_SEGMENT_PRICE_MAX=8u charges the
                                    * exact price at every level, which is the
                                    * arm this gate was measured against. */
#endif
#define ZPSE_TILE_HEADER_SIZE    8u
#define ZPSE_SEG_DIR_ENTRY_SIZE  8u

#define ZPSE_MAX_BLOCK_SIZE     UINT32_C(65536)
#define ZPSE_MIN_BLOCK_SIZE     UINT32_C(1)
#define ZPSE_TILE_SIZE          256u
#define ZPSE_MAX_TILE_LEN       256u

/* segment_count is one of 1, 2, 4, 8 or 16. */

/* Frame flags (Sec. 3.2). */
#define ZPSE_F_CONTENT_SIZE     (UINT16_C(1) << 1)
#define ZPSE_F_BLOCK_COUNT      (UINT16_C(1) << 2)
#define ZPSE_F_FRAME_INDEX      (UINT16_C(1) << 3)
/* Bits 4-15 were always reserved. Bit 0 used to be the frame-level payload
 * CRC flag; that CRC and the flag are gone, so the bit has no meaning left.
 * Refuse it like any other reserved bit rather than accept a frame whose
 * payload goes unchecked, which is what an accepted bit 0 would mean now. */
#define ZPSE_F_RESERVED_MASK    (UINT16_C(0xFFF1))

/* Block flags (Sec. 4.2). */
#define ZPSE_B_RAW              (UINT8_C(1) << 0)
#define ZPSE_B_LAST             (UINT8_C(1) << 1)
#define ZPSE_B_CRC              (UINT8_C(1) << 2)
#define ZPSE_B_RESET            (UINT8_C(1) << 3)
#define ZPSE_B_HAS_META         (UINT8_C(1) << 4)
/* The block payload is a single raw-LZ4 block (include/zpse/lz.h), rather
 * than transform + segments + tiles: [u32 uncompressed][u32 cmds] followed
 * by the raw LZ4 block. Mutually exclusive with ZPSE_B_RAW; transform_id 0,
 * segment_count 1, no transform metadata. This is the body-is-LZ4 form;
 * ZPSE_B_LZ_TILE (bit 6) is the other one, which LZ4-compresses a tile body. */
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
/* The entropy-coded LZ form: the raw LZ4 payload, Huffman coded. Its payload
 * is
 *   [u8 table_id][u8 inner_form][u32 inner_len][u16 seg0][u16 seg1][u16 seg2]
 *   [four Huffman streams]
 * and `inner_form` is a fixed tag: it must be ZPSE_B_LZ (0x20), the raw-LZ4
 * form selected by bit 5. The tables are fixed in the format
 * (src/huf_tables.inc), so a block costs 12 bytes of overhead and no table;
 * see include/zpse/huf.h.
 *
 * This is a fresh single-bit encoding: ZPSE_B_HUF alone. Bit 7 was reserved
 * by v1, so a block with it set is refused by an old reader -- which is what
 * makes the bit available to select this form. There is no combined pattern:
 * 0xA0 (ZPSE_B_LZ | ZPSE_B_HUF) and 0xC0 (ZPSE_B_LZ_TILE | ZPSE_B_HUF) are
 * invalid combinations and must be rejected. The envelope is the LZ form's
 * own -- transform 0, one segment, no metadata -- because the tile form's
 * transform and segment count live in the header, and a Huffman payload only
 * ever wraps a pure LZ stream. */
#define ZPSE_B_HUF              (UINT8_C(1) << 7)
#define ZPSE_B_RESERVED_MASK    (UINT8_C(0x00))

/* Decoder hints (ZPSE_DEC_*): in-process caller options, never stream bytes.
 * Setting one changes no compressed byte and no decoded byte; it only tells
 * the decoder something the caller knows about its own output buffer. Bit 0
 * promises that the caller will not reread the output soon, which lets the
 * RAW/no-CRC copy use streaming (non-temporal) stores on x86 for blocks big
 * enough to pay for the fence. The hint lives in its own namespace: encoder
 * option bit 0 (ZPSE_ENC_CHECKSUM) has a completely different meaning. */
#define ZPSE_DEC_STREAMING_OUTPUT (UINT32_C(1) << 0)
#define ZPSE_DEC_VALID_MASK       ZPSE_DEC_STREAMING_OUTPUT

#define ZPSE_ENC_CHECKSUM       (UINT32_C(1) << 0)

/* Write an LZ block and nothing else: no probe, no trial, no candidate list,
 * no decision. The payload is the same raw LZ4 block the deepest level offers,
 * so a decoder needs to know nothing about the option. What the caller gives
 * up is the tile forms, and with them the ratio on typed columns, where a
 * transpose plus a predictor beats any LZ parse; what it gets is one LZ4 pass
 * per block with no decision machinery. */
#define ZPSE_ENC_LZ_FAST        (UINT32_C(1) << 5)

/* Also Huffman code the winning LZ payload (include/zpse/huf.h). Off by
 * default; the deepest level turns it on for itself, because it is the rung
 * whose trade is size and nothing else. Set explicitly, it applies at any
 * level that writes an LZ form at all (5-9). The streams it writes need no
 * decoder option, so a caller who does not ask for it never sees one.
 * Levels 1-4 write no LZ form, so the bit has nothing to act on there. */
#define ZPSE_ENC_HUF            (UINT32_C(1) << 8)

/* Encoder effort (ZPSE_ENC_* bits 1-4). Nine levels, 1 = FASTEST ... 9 =
 * MAX. Stored as level << ZPSE_ENC_SPEED_SHIFT, so the raw field value is
 * 2,4,6,...,18. This trades compressed size for encode time and never
 * changes the format: every level writes a stream any decoder reads, and a
 * tile stream decodes at the same speed whichever level chose it, because
 * the shapes themselves are the same. What a level changes is how the sweep
 * decides which shape to write. The one exception is the intra-block LZ
 * form, which levels 5-9 may also pick: it decodes at the speed of its
 * command loop rather than the tile reconstruction, which is why one decode
 * bill caps how many commands per output byte it will accept, the same at
 * every level. Levels 5-9 also share one LZ4 screen (see
 * ZPSE_LZ4_SCREEN_NUM in src/block.c); the per-level LZ margins and bills
 * once listed in the ladder are historical, from before the raw-LZ4 codec.
 * The bill is free on the encode side: it only filters streams the pass has
 * already produced, so it can never make a block pay for a pass. Down the
 * ladder only trials and tie_field vary.
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
 *                              the intra-block LZ form as one more candidate:
 *                              the match finder is a full pass at roughly a
 *                              quarter of the tile encoder's rate, so a sample
 *                              of the block is coded first and the pass is
 *                              paid for only when the shared screen shows the
 *                              form winning.
 *  6                          the same LZ screen and decode bill as 5
 *  7                          the same LZ screen and decode bill as 5
 *  8                          the same LZ screen and bill; the field cut is
 *                              off: the deepest rungs encode every survivor
 *                              their price separated
 *  9  ZPSE_ENC_SPEED_MAX      the same LZ screen and decode bill as 5
 *
 * The ladder is a table in src/block.c (zpse_level): each level differs from
 * the one above it in at least one knob, and every knob moves in the
 * direction that can only add candidates to the choice, so a deeper level
 * cannot write a larger block than a shallower one (tests/suites/
 * test_levels.c pins that). The cheap rungs buy trials one at a time; levels
 * 5-9 all apply the same LZ4 screen and decode bill, not a notch looser per
 * rung. No level's LZ candidate can cost it size -- the form is kept only when
 * it is smaller than the best tile stream -- and the trade it makes is decode
 * speed: an LZ stream decodes at the speed of its command loop, so the one
 * global bill (see src/block.c) decides how much slower than the tile stream
 * it is allowed to be.
 *
 * An unsupported level is rejected with ZPSE_BAD_ARG rather than rounded to
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
 * level 6.
 *
 * Why 6 rather than BALANCED (5) or 7 is historical -- it was chosen for the
 * old, pre-LZ4 ladder. Back then level 6 was a superset of level 5's trade
 * rather than a step away from it: the screen margin went from 1.25x to
 * 1.1875x and the decode bill from 8 to 6 bytes per command, and neither could
 * refuse a block BALANCED would have taken -- a stream the finder offers can
 * only be kept when it is smaller, and a bill that accepts shorter runs only
 * admits more of them. Measured over Silesia (202 MiB of real data, one pinned
 * core, `tests/benchmarks/silesia.sh`): 113,246,135 packed bytes at level 5
 * against 106,140,388 at level 6, i.e. 6.3% smaller, for 6% of the encode rate
 * (315 against 296 MB/s on the prose pair).
 *
 * The two knobs were not equal in cost, which is what row 6 was built from:
 * raising the bill alone was free -- it only filtered streams the match-finder
 * pass had already produced, so it could not add a pass -- while the margin
 * cost one, because it offered the finder more blocks. The default took the
 * free knob and left the paid one to level 7; level 5 stayed as the cheapest
 * rung for a caller who wanted its exact trade.
 *
 * Those per-level margins and bills are historical now: levels 5-9 share one
 * LZ4 screen and one decode bill, and only trials and tie_field vary by level
 * (see ZPSE_LZ4_SCREEN_NUM in src/block.c). The default stays at 6. */
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
/* How many candidates a page block's sweep may price.
 *
 * The sweep prices a candidate by staging its transform over its probe window
 * and pricing every plane of it, and a page has nine of them, so the candidate
 * list -- not the probe's order or its floor test -- is where a 4 KiB encode's
 * time goes. Capping the list at seven drops the two shapes the sweep reaches
 * last. On the dump corpus that is 1.1% of packed bytes for a fifth of the
 * encode rate; the mixed and Silesia corpora pay 0.008 and 0.064% for 13 and
 * 12%. Only a page: a 64 KiB block's body is where the candidates behind the
 * seventh earn their price. With the whole-page price (see
 * ZPSE_PROBE_SHORT_TILES) even that cost is gone -- the cap writes the same
 * bytes with and without it on all three corpora, rate inside the host's band
 * -- so it is kept for the pages the price cannot settle, not for a measured
 * saving here. And only a level that prices every plane (5-9): a
 * cheap rung acts on the price alone, and the same two shapes are worth 2.5
 * points of ratio there (1.0-1.1% for BALANCED).
 *
 * It is a knob now, and it was not one before: the value was a plain #define
 * in this header, so a build passing the macro on the command line silently
 * kept seven and every arm measured against it ran the same binary. Measured
 * as one at 4 KiB level 6, four interleaved rounds on one pinned core, seven
 * against six: six writes 145 420 891 bytes over the dumps against 142 699 989,
 * i.e. 1.276 ratio points, for 118.8/126.7/122.3/128.3 MB/s against
 * 118.5/114.9/121.4/118.1, and Silesia pays 0.154 points for 3-5%. Five is the
 * same bytes as six (145 427 697) and reads 125.6 against 116.5 MB/s by mean
 * in the same run, four costs 1.459 points for +17%, and three is the
 * same bytes as four and the fastest of the four (+20%).
 *
 * Six is the default now, and it is the one arm below seven the ratio line was
 * moved for: 1.276 points over the dumps and 0.154 over Silesia is what the
 * last priced shape costs, and it buys 5-10% of the page encoder's rate, which
 * no other knob on the price has offered without costing both corpora at once.
 * Seven is still buildable (-DZPSE_PAGE_CAND_CAP=7u) for a caller whose bytes
 * matter more than that, and the arms below six remain refused: they buy their
 * extra rate by dropping a shape the sweep has to be able to reach.
 *
 * Re-measured at 4 KiB level 6 on the current build: six writes 88,706,614
 * bytes over the dumps against 86,168,376 raised to nine (1.19 points) for
 * 159.0 against 128.8 MB/s, and Silesia 116,303,066 against 115,915,276 (0.18)
 * for 158.7 against 135.0, so the cap is a real trade there and not a ceiling
 * the price has already made free. */
#ifndef ZPSE_PAGE_CAND_CAP
#define ZPSE_PAGE_CAND_CAP 6u
#endif
/* Levels that run the match finder at all (5-9). */
#define ZPSE_ENC_SPEED_LZ_MIN 5u
/* Levels whose segment prescreen is cut to the three uniform predictors (1
 * only). The prescreen is what half of FASTEST's encode is; every rung above
 * it has a second trial to order, and a runner-up chosen by a prescreen that
 * cannot see STRIDE or the lane-history family is the wrong runner-up. */
#define ZPSE_ENC_SPEED_CHEAP_PRESCREEN_MAX 1u
/* Every level uses the ASCII gate, MAX included: it decides only what the
 * tile side of a convicted block costs, and the LZ candidate is offered
 * afterwards whatever it decided, so no level can lose its LZ trial to it. */

/* What a PRICE charges for a segment. The format writes eight bytes, and
 * that is what a price charges wherever the price decides: levels 1 to 4 rank
 * the candidates by it and encode only the winner, so an envelope eight bytes
 * too small is a size decision, not a ranking one. Levels 5 to 9 price every
 * candidate and then encode the survivors in full, so there the price only
 * screens, and the sixteen bytes those thresholds were swept with are the
 * ranking they were tuned against -- an eight-byte error costs a trial there,
 * not a byte.
 *
 * Measured on the dump corpus, 213 MB, against the same build with the
 * envelope set to the exact price everywhere: gated this way, 4 KiB level 1
 * stays at its exact-price 124 889 146 bytes, while the envelope buys 11 190
 * at 4 KiB level 6, 178 687 at 16 KiB level 6, 471 291 at 64 KiB level 6,
 * 126 728 at 64 KiB level 9, 0.24 points of the mixed workload and 0.12 of
 * Silesia, and costs 31 910 at 4 KiB level 9. Its price is on synthetic data:
 * the one-value 64 KiB block encodes at 6.1 GB/s against 10.9 without it, for
 * the same 67 output bytes, because more of its candidates survive the
 * price. Real corpora show no encode cost -- 4 KiB level 6 is 118 MB/s either
 * way -- so the trade is kept and the synthetic rate is the one it costs. */
static inline uint32_t zpse_segment_price(uint32_t enc_flags) {
    return (zpse_enc_speed(enc_flags) >= ZPSE_ENC_SPEED_EVERY_PLANE_MIN)
               ? (uint32_t)ZPSE_SEGMENT_PRICE_MAX
               : (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;
}

/* Segment flags (Sec. 6.1). Only 0x01 is legal in v1. */
#define ZPSE_S_LEN_PREFIX       (UINT8_C(1) << 0)

/* The standard CRC32-C check value for "123456789" is 0xE3069283. */

#endif /* ZPSE_CONSTANTS_H */
