#ifndef ZPSE_TYPES_H
#define ZPSE_TYPES_H

/* Packed wire types with explicit sizes. All integers little-endian on
 * the wire; these structs are host-side parsed representations.
 * Serialization goes through bits.h helpers, never raw struct casts. */

#include <stdint.h>
#include "constants.h"

typedef struct ZpseFrameHeader {
    uint32_t magic;
    uint8_t  version_major;
    uint8_t  version_minor;
    uint16_t frame_flags;
    uint32_t nominal_block_size;
    uint64_t content_size;
    uint32_t block_count;
    uint32_t frame_index_offset;
    uint32_t header_crc32c;
} ZpseFrameHeader;

typedef struct ZpseBlockHeader {
    uint16_t sync;
    uint8_t  block_version;
    uint8_t  block_flags;
    uint32_t uncompressed_size;
    uint32_t compressed_block_size;
    uint8_t  transform_id;
    uint8_t  segment_count;
    uint16_t transform_meta_size;
    uint16_t tile_size;
    uint8_t  predictor_profile;
    uint8_t  reserved0;
    uint32_t block_sequence_number;
    uint32_t decoded_crc32c;
    uint32_t header_crc32c;
} ZpseBlockHeader;

typedef struct ZpseSegmentDirEntry {
    uint32_t payload_offset; /* relative to payloads area start */
    uint32_t payload_size;   /* incl. 16-B segment header */
} ZpseSegmentDirEntry;

typedef struct ZpseSegmentHeader {
    uint32_t segment_decoded_size;
    uint16_t tile_count;
    uint8_t  initial_literal;
    uint8_t  segment_flags;
    uint32_t tile_directory_size; /* MUST be 0 in v1 */
    uint32_t reserved;            /* MUST be 0 */
} ZpseSegmentHeader;

typedef struct ZpseTileHeader {
    uint8_t  tile_flags;
    uint8_t  predictor_id;
    uint8_t  tile_length_minus_1;
    uint8_t  exception_count;
    uint16_t mask_payload_size;
    uint16_t residual_payload_size;
} ZpseTileHeader;

/* Transform identifiers (Sec. 5). */
typedef enum ZpseTransformId {
    ZPSE_T_RAW_BYTES          = 0,
    ZPSE_T_DELTA8             = 1,
    ZPSE_T_XOR8               = 2,
    ZPSE_T_TRANSPOSE_2        = 3,
    ZPSE_T_TRANSPOSE_4        = 4,
    ZPSE_T_TRANSPOSE_8        = 5,
    ZPSE_T_TRANSPOSE_16       = 6,
    ZPSE_T_DELTA_TRANSPOSE_2  = 7,
    ZPSE_T_DELTA_TRANSPOSE_4  = 8,
    ZPSE_T_DELTA_TRANSPOSE_8  = 9,
    ZPSE_T_DELTA_TRANSPOSE_16 = 10,
    ZPSE_T_STRIDE_TRANSPOSE   = 11
} ZpseTransformId;

/* Parsed transform metadata (Sec. 5). seeds valid iff has_seeds. */
typedef struct ZpseTransformMeta {
    uint8_t transform_id;
    uint8_t width_W;      /* interleave width; 1 for non-transposed */
    uint8_t stride_flags; /* ID 11 flags byte; 0 otherwise */
    uint8_t has_seeds;    /* boolean */
    uint8_t seeds[16];    /* per-segment seeds, index order */
} ZpseTransformMeta;

/* Parse/serialize entry points (implemented in .c, bounds-checked). */
int zpse_frame_header_parse(ZpseFrameHeader *out, const uint8_t *buf, uint32_t len);
void zpse_frame_header_serialize(uint8_t out[32], const ZpseFrameHeader *h);
int zpse_block_header_parse(ZpseBlockHeader *out, const uint8_t *buf, uint32_t len);
void zpse_block_header_serialize(uint8_t out[32], const ZpseBlockHeader *h);
int zpse_segment_header_parse(ZpseSegmentHeader *out, const uint8_t *buf, uint32_t len);
void zpse_segment_header_serialize(uint8_t out[16], const ZpseSegmentHeader *h);
int zpse_tile_header_parse(ZpseTileHeader *out, const uint8_t *buf, uint32_t len);
int zpse_transform_meta_parse(ZpseTransformMeta *out, uint8_t transform_id,
                              const uint8_t *buf, uint16_t len,
                              uint8_t segment_count);

#endif /* ZPSE_TYPES_H */
