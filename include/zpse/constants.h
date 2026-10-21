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
#define ZPSE_B_RESERVED_MASK    (UINT8_C(0xE0))

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
 *   ZPSE_ENC_SPEED_BALANCED  the default: price one tile in every plane of
 *                            every candidate, keep those within a margin of
 *                            the best, encode each in full.
 *   ZPSE_ENC_FASTER          price a quarter of the planes and estimate the
 *                            rest, keeping the same survivor rule. The
 *                            estimate under-prices the wide transposes, so
 *                            a block whose best shape is one of those can
 *                            come out larger than at the default.
 *   ZPSE_ENC_FASTEST         the same cheap price, then encode only the
 *                            single shape it likes best: one trial per
 *                            block, no second opinion.
 *
 * An unsupported level is rejected with PSE_BAD_ARG rather than rounded to
 * one of these. */
#define ZPSE_ENC_SPEED_SHIFT    1
#define ZPSE_ENC_SPEED_MASK     (UINT32_C(3) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_SPEED_BALANCED (UINT32_C(0) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_FASTER         (UINT32_C(1) << ZPSE_ENC_SPEED_SHIFT)
#define ZPSE_ENC_FASTEST        (UINT32_C(2) << ZPSE_ENC_SPEED_SHIFT)

/* Segment flags (Sec. 6.1). Only 0x01 is legal in v1. */
#define ZPSE_S_LEN_PREFIX       (UINT8_C(1) << 0)
#define ZPSE_S_DIR_BIT          (UINT8_C(1) << 1)

/* CRC32-C check value for "123456789". */
#define ZPSE_CRC32C_CHECK       UINT32_C(0xE3069283)

#endif /* ZPSE_CONSTANTS_H */
