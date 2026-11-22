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
    /* How many bytes the header occupies on the wire: ZPSE_BLOCK_HEADER_SIZE
     * for the long form, ZPSE_BLOCK_HEADER_SIZE_SHORT for a page's. Set by
     * the parser, and the offset the payload starts at. */
    uint8_t  header_size;
} ZpseBlockHeader;

typedef struct ZpseSegmentDirEntry {
    uint32_t payload_offset; /* relative to payloads area start */
    uint32_t payload_size;   /* incl. 8-B segment header */
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
int zpse_block_header_parse(ZpseBlockHeader *out, const uint8_t *buf, uint32_t len);
void zpse_block_header_serialize(uint8_t out[32], const ZpseBlockHeader *h);
/* 32, 16, or 0 when `buf` does not start with a recognised block sync. Only
 * the first two bytes are read, so a caller that has 16 bytes of a page's
 * block can tell which header it is looking at before reading the rest. */
uint32_t zpse_block_header_size(const uint8_t *buf, uint32_t len);
/* The 16-byte page header. Requires block_flags without ZPSE_B_CRC and an
 * uncompressed size of at most ZPSE_BLOCK_SHORT_MAX. */
void zpse_block_header_serialize_short(uint8_t out[16], const ZpseBlockHeader *h);

/* One-value page token (v1.1, sync ZPSE_BLOCK_SYNC_TOKEN): the 16-byte page
 * header with no payload, the fill byte in transform_id, segment_count 1,
 * meta 0 and comp == 16. LAST is the only optional flag bit.
 *
 * zpse_block_token_build validates EVERY field and the 16-bit CRC over the
 * first 14 bytes, then fills `out`. Returns ZPSE_OK on success, ZPSE_BAD_ARG
 * for a null pointer, ZPSE_TRUNCATED when fewer than 16 bytes are available,
 * ZPSE_BAD_HEADER_CRC on a CRC mismatch and ZPSE_BAD_HEADER for any illegal
 * field value. It is the single definition of what a legal token is: the
 * block parser dispatches to it, so the block layer's fast paths can call it
 * directly instead of repeating the field checks.
 *
 * zpse_block_header_serialize_token is the mirror of the short serializer:
 * it writes a legal token envelope from a (validated) parsed header and
 * recomputes the CRC. Fixed fields are written by the serializer, not taken
 * from `h`, so a token it emits always parses back cleanly. */
int zpse_block_token_build(ZpseBlockHeader *out, const uint8_t *buf, uint32_t len);
void zpse_block_header_serialize_token(uint8_t out[16], const ZpseBlockHeader *h);

/* The FLAT page form (sync ZPSE_BLOCK_SYNC_FLAT): a 16-byte header that names
 * a flat payload codec instead of a transform + segment directory.
 *
 * zpse_block_header_parse_flat validates the whole envelope -- the 16-bit CRC
 * over bytes [0..14), reserved byte 0, RESET (+optional LAST) with every other
 * flag bit refused, codec in 0..3, comp >= 16 and n in 1..4096 -- and fills
 * `out`. Returns ZPSE_OK, or ZPSE_BAD_ARG / ZPSE_TRUNCATED /
 * ZPSE_BAD_HEADER_CRC / ZPSE_BAD_HEADER / ZPSE_BAD_TRANSFORM /
 * ZPSE_BAD_BLOCK_SIZE. The block parser dispatches here on the FLAT sync.
 *
 * zpse_block_header_serialize_flat is the mirror serializer: it writes the
 * fixed sync and reserved byte, takes flags/codec/param/sizes/seq from `h`
 * (which must have been validated), and recomputes the CRC. */
int zpse_block_header_parse_flat(ZpseBlockHeader *out, const uint8_t *buf, uint32_t len);
void zpse_block_header_serialize_flat(uint8_t out[16], const ZpseBlockHeader *h);

/* The TEXT page form (sync ZPSE_BLOCK_SYNC_TEXT): a 16-byte header with no
 * codec fields, reserved bytes at [3..6) and n at [6..8) where 0 means 65536.
 * zpse_block_header_parse_text validates the CRC, reserved bytes, flags and
 * sizes and fills `out`; the block parser dispatches here on the TEXT sync.
 * zpse_block_header_serialize_text writes the fixed envelope from an
 * already-validated `h` and recomputes the CRC. */
int zpse_block_header_parse_text(ZpseBlockHeader *out, const uint8_t *buf, uint32_t len);
void zpse_block_header_serialize_text(uint8_t out[16], const ZpseBlockHeader *h);

int zpse_segment_header_parse(ZpseSegmentHeader *out, const uint8_t *buf, uint32_t len);
void zpse_segment_header_serialize(uint8_t out[8], const ZpseSegmentHeader *h);
int zpse_transform_meta_parse(ZpseTransformMeta *out, uint8_t transform_id,
                              const uint8_t *buf, uint16_t len,
                              uint8_t segment_count);

#endif /* ZPSE_TYPES_H */
