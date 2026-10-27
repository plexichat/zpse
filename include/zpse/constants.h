#ifndef ZPSE_CONSTANTS_H
#define ZPSE_CONSTANTS_H

#include <stdint.h>

/* PSE v1.0.0 wire constants. */

#define ZPSE_MAGIC              UINT32_C(0x31455350) /* "PSE1" LE */
#define ZPSE_FRAME_MAJOR        ((uint8_t)1)
#define ZPSE_FRAME_MINOR        ((uint8_t)0)
#define ZPSE_BLOCK_VERSION      ((uint8_t)1)
#define ZPSE_BLOCK_SYNC         ((uint16_t)0x50B6)

#define ZPSE_FRAME_HEADER_SIZE  32u
#define ZPSE_BLOCK_HEADER_SIZE  32u

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
#define ZPSE_B_RESERVED_MASK    (UINT8_C(0xC0))

/* Encoder options (API): which optional fields a stream carries. The
 * payload CRC32-C is optional in the format (block flag ZPSE_B_CRC,
 * Sec. 4), and it is OFF by default -- the same choice LZ4 makes for its
 * frame checksum. Emitting it costs the encoder a pass over the whole
 * block it just produced, and costing the DECODER a pass over an output
 * it just computed, so a stream that does not ask for it does not pay.
 * Every decoder here verifies it whenever a stream carries it. */
#define ZPSE_ENC_CHECKSUM       (UINT32_C(1) << 0)

/* Encoder effort (ZPSE_ENC_* bits 1-2). This trades compressed size for
 * encode time and never changes the format: every level writes a stream any
 * decoder reads, and every level's stream decodes at the same speed, because
 * the shapes themselves are the same. What a level changes is how the sweep
 * decides which shape to write.
 *
 *   ZPSE_ENC_SPEED_BALANCED  the default: price the planes of every
 *                            candidate, keep those within a margin of the
 *                            best, encode each in full -- and offer the
 *                            intra-block LZ form as one more candidate on
 *                            every block. The match finder is a full pass
 *                            at roughly a quarter of the tile encoder's
 *                            rate, so a sample of the block is coded first
 *                            and the pass is paid for only when that sample
 *                            shows the form winning by a margin (see
 *                            ZPSE_LZ_SCREEN_* in block.c).
 *   ZPSE_ENC_FASTER          the same sweep with at most two full trials,
 *                            and no match finder at all.
 *   ZPSE_ENC_FASTEST         the same cheap price, then encode only the
 *                            single shape it likes best: one trial per
 *                            block, no second opinion, no match finder.
 *                            This is the level to reach for when the
 *                            encoder is the bottleneck.
 *
 *   ZPSE_ENC_MAX             everything the default level does, with the
 *                            match finder's sample screen at its loosest
 *                            (see ZPSE_LZ_SCREEN_NUM_*). The two deepest
 *                            levels therefore differ mostly in how small a
 *                            win on that sample is worth a full pass, which
 *                            is what puts prose's LZ form at MAX and not at
 *                            the default level. No level's LZ candidate can
 *                            cost it size -- the form is kept only when it
 *                            is smaller than the best tile stream -- and
 *                            the trade it makes is decode speed: an LZ
 *                            stream decodes at the speed of its command
 *                            loop, so the level's acceleration factor (see
 *                            ZPSE_LZ_ACCEL_* in block.c) decides how slow
 *                            than the tile stream it is allowed to be.
 *
 * An unsupported level is rejected with PSE_BAD_ARG rather than rounded to
 * one of these. */
#define ZPSE_ENC_SPEED_SHIFT    1
#define ZPSE_ENC_SPEED_MASK     (UINT32_C(3) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_SPEED_BALANCED (UINT32_C(0) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_FASTER         (UINT32_C(1) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_FASTEST        (UINT32_C(2) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_MAX            (UINT32_C(3) << ZPSE_ENC_SPEED_SHIFT)

/* Segment flags (Sec. 6.1). Only 0x01 is legal in v1. */
#define ZPSE_S_LEN_PREFIX       (UINT8_C(1) << 0)
#define ZPSE_S_DIR_BIT          (UINT8_C(1) << 1)

/* CRC32-C check value for "123456789". */
#define ZPSE_CRC32C_CHECK       UINT32_C(0xE3069283)

#endif /* ZPSE_CONSTANTS_H */
