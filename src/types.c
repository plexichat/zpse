#include "zpse/types.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
#include "zpse/predictor.h"
#include "zpse/status.h"

static int zpse_is_valid_segment_count(uint8_t c)
{
    return c == 1 || c == 2 || c == 4 || c == 8 || c == 16;
}

int zpse_frame_header_parse(ZpseFrameHeader *out, const uint8_t *buf, uint32_t len)
{
    uint32_t magic;
    uint8_t major, minor;
    uint16_t flags;
    uint32_t nominal;
    uint64_t content_size;
    uint32_t block_count;
    uint32_t index_off;
    uint32_t stored_crc, calc_crc;

    if (out == 0 || buf == 0) {
        return PSE_BAD_ARG;
    }
    if (len < ZPSE_FRAME_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    magic = zpse_load_u32le(buf + 0);
    major = buf[4];
    minor = buf[5];
    flags = zpse_load_u16le(buf + 6);
    nominal = zpse_load_u32le(buf + 8);
    content_size = zpse_load_u64le(buf + 12);
    block_count = zpse_load_u32le(buf + 20);
    index_off = zpse_load_u32le(buf + 24);
    stored_crc = zpse_load_u32le(buf + 28);

    if (magic != ZPSE_MAGIC) {
        return PSE_BAD_MAGIC;
    }
    if (major != ZPSE_FRAME_MAJOR || minor != ZPSE_FRAME_MINOR) {
        return PSE_BAD_VERSION;
    }
    calc_crc = zpse_crc32c(buf, 28);
    if (calc_crc != stored_crc) {
        return PSE_BAD_HEADER_CRC;
    }
    if ((flags & ZPSE_F_RESERVED_MASK) != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (nominal < ZPSE_MIN_BLOCK_SIZE || nominal > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if ((flags & ZPSE_F_FRAME_INDEX) != 0) {
        if (index_off == 0) {
            return PSE_BAD_HEADER;
        }
    } else {
        if (index_off != 0) {
            return PSE_BAD_HEADER;
        }
    }

    out->magic = magic;
    out->version_major = major;
    out->version_minor = minor;
    out->frame_flags = flags;
    out->nominal_block_size = nominal;
    out->content_size = content_size;
    out->block_count = block_count;
    out->frame_index_offset = index_off;
    out->header_crc32c = stored_crc;
    return PSE_OK;
}

void zpse_frame_header_serialize(uint8_t out[32], const ZpseFrameHeader *h)
{
    uint32_t crc;

    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u32le(out + 0, h->magic);
    out[4] = h->version_major;
    out[5] = h->version_minor;
    zpse_store_u16le(out + 6, h->frame_flags);
    zpse_store_u32le(out + 8, h->nominal_block_size);
    zpse_store_u64le(out + 12, h->content_size);
    zpse_store_u32le(out + 20, h->block_count);
    zpse_store_u32le(out + 24, h->frame_index_offset);
    crc = zpse_crc32c(out, 28);
    zpse_store_u32le(out + 28, crc);
}

int zpse_block_header_parse(ZpseBlockHeader *out, const uint8_t *buf, uint32_t len)
{
    uint16_t sync;
    uint8_t ver, flags;
    uint32_t uncomp, comp;
    uint8_t tid, scount;
    uint16_t meta_size;
    uint16_t tile_size;
    uint8_t profile, r0;
    uint32_t seq;
    uint32_t dec_crc;
    uint32_t stored_crc, calc_crc;
    int has_meta;

    if (out == 0 || buf == 0) {
        return PSE_BAD_ARG;
    }
    if (len < ZPSE_BLOCK_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    sync = zpse_load_u16le(buf + 0);
    ver = buf[2];
    flags = buf[3];
    uncomp = zpse_load_u32le(buf + 4);
    comp = zpse_load_u32le(buf + 8);
    tid = buf[12];
    scount = buf[13];
    meta_size = zpse_load_u16le(buf + 14);
    tile_size = zpse_load_u16le(buf + 16);
    profile = buf[18];
    r0 = buf[19];
    seq = zpse_load_u32le(buf + 20);
    dec_crc = zpse_load_u32le(buf + 24);
    stored_crc = zpse_load_u32le(buf + 28);

    if (sync != ZPSE_BLOCK_SYNC) {
        return PSE_BAD_HEADER;
    }
    if (ver != ZPSE_BLOCK_VERSION) {
        return PSE_BAD_VERSION;
    }
    calc_crc = zpse_crc32c(buf, 28);
    if (calc_crc != stored_crc) {
        return PSE_BAD_HEADER_CRC;
    }
    if (uncomp < ZPSE_MIN_BLOCK_SIZE || uncomp > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (comp < ZPSE_BLOCK_HEADER_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (tid > 11) {
        return PSE_BAD_TRANSFORM;
    }
    if (!zpse_is_valid_segment_count(scount)) {
        return PSE_BAD_TRANSFORM;
    }
    if (tile_size != ZPSE_TILE_SIZE) {
        return PSE_BAD_TILE_SIZE;
    }
    if (profile != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (r0 != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    has_meta = (meta_size > 0) ? 1 : 0;
    if (((flags & ZPSE_B_HAS_META) != 0) != has_meta) {
        return PSE_BAD_HEADER;
    }
    if ((flags & ZPSE_B_RESET) == 0) {
        return PSE_BAD_HEADER; /* required-1 bit, cf. HAS_META agreement */
    }
    if ((flags & ZPSE_B_RESERVED_MASK) != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }

    out->sync = sync;
    out->block_version = ver;
    out->block_flags = flags;
    out->uncompressed_size = uncomp;
    out->compressed_block_size = comp;
    out->transform_id = tid;
    out->segment_count = scount;
    out->transform_meta_size = meta_size;
    out->tile_size = tile_size;
    out->predictor_profile = profile;
    out->reserved0 = r0;
    out->block_sequence_number = seq;
    out->decoded_crc32c = dec_crc;
    out->header_crc32c = stored_crc;
    return PSE_OK;
}

void zpse_block_header_serialize(uint8_t out[32], const ZpseBlockHeader *h)
{
    uint32_t crc;

    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u16le(out + 0, h->sync);
    out[2] = h->block_version;
    out[3] = h->block_flags;
    zpse_store_u32le(out + 4, h->uncompressed_size);
    zpse_store_u32le(out + 8, h->compressed_block_size);
    out[12] = h->transform_id;
    out[13] = h->segment_count;
    zpse_store_u16le(out + 14, h->transform_meta_size);
    zpse_store_u16le(out + 16, h->tile_size);
    out[18] = h->predictor_profile;
    out[19] = h->reserved0;
    zpse_store_u32le(out + 20, h->block_sequence_number);
    zpse_store_u32le(out + 24, h->decoded_crc32c);
    crc = zpse_crc32c(out, 28);
    zpse_store_u32le(out + 28, crc);
}

int zpse_segment_header_parse(ZpseSegmentHeader *out, const uint8_t *buf, uint32_t len)
{
    uint32_t dec_size;
    uint16_t tile_count;
    uint8_t init_lit, sflags;
    uint32_t dir_size;
    uint32_t reserved;

    if (out == 0 || buf == 0) {
        return PSE_BAD_ARG;
    }
    if (len < ZPSE_SEGMENT_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    dec_size = zpse_load_u32le(buf + 0);
    tile_count = zpse_load_u16le(buf + 4);
    init_lit = buf[6];
    sflags = buf[7];
    dir_size = zpse_load_u32le(buf + 8);
    reserved = zpse_load_u32le(buf + 12);

    if (tile_count == 0) {
        return PSE_BAD_TILE_SIZE;
    }
    if (dec_size == 0) {
        return PSE_BAD_TILE_SIZE;
    }
    if (sflags != ZPSE_S_LEN_PREFIX) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (dir_size != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (reserved != 0) {
        return PSE_UNSUPPORTED_FEATURE;
    }

    out->segment_decoded_size = dec_size;
    out->tile_count = tile_count;
    out->initial_literal = init_lit;
    out->segment_flags = sflags;
    out->tile_directory_size = dir_size;
    out->reserved = reserved;
    return PSE_OK;
}

void zpse_segment_header_serialize(uint8_t out[16], const ZpseSegmentHeader *h)
{
    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u32le(out + 0, h->segment_decoded_size);
    zpse_store_u16le(out + 4, h->tile_count);
    out[6] = h->initial_literal;
    out[7] = h->segment_flags;
    zpse_store_u32le(out + 8, h->tile_directory_size);
    zpse_store_u32le(out + 12, h->reserved);
}

int zpse_tile_header_parse(ZpseTileHeader *out, const uint8_t *buf, uint32_t len)
{
    uint8_t flags, mask_mode, res_mode, pid;

    if (out == 0 || buf == 0) {
        return PSE_BAD_ARG;
    }
    if (len < ZPSE_TILE_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    flags = buf[0];
    pid = buf[1];
    /* Structural mode validation at parse time (Sec. 6.2/11.1): the size
     * equation and K-derivation need cross-field context and stay with
     * zpse_tile_validate, but unknown modes and reserved bits fail here. */
    if ((flags & (uint8_t)0x40) != 0) {
        return PSE_UNSUPPORTED_FEATURE; /* bit 6 reserved-0 */
    }
    mask_mode = (uint8_t)(flags & (uint8_t)0x07);
    res_mode = (uint8_t)((flags & (uint8_t)0x38) >> 3u);
    if (mask_mode > 6u) {
        return PSE_BAD_MASK; /* mode 7 reserved */
    }
    if (res_mode > 4u) {
        return PSE_BAD_RESIDUAL; /* modes 5..7 reserved */
    }
    if (pid > (uint8_t)ZPSE_P_STRIDE_3) {
        return PSE_BAD_PREDICTOR;
    }
    out->tile_flags = flags;
    out->predictor_id = pid;
    out->tile_length_minus_1 = buf[2];
    out->exception_count = buf[3];
    out->mask_payload_size = zpse_load_u16le(buf + 4);
    out->residual_payload_size = zpse_load_u16le(buf + 6);
    return PSE_OK;
}

int zpse_transform_meta_parse(ZpseTransformMeta *out, uint8_t transform_id,
                              const uint8_t *buf, uint16_t len,
                              uint8_t segment_count)
{
    uint8_t i;

    if (out == 0) {
        return PSE_BAD_ARG;
    }
    if (len > 0 && buf == 0) {
        return PSE_BAD_ARG;
    }
    if (transform_id > 11) {
        return PSE_BAD_TRANSFORM;
    }
    if (!zpse_is_valid_segment_count(segment_count)) {
        return PSE_BAD_TRANSFORM;
    }

    out->transform_id = transform_id;
    out->width_W = 1;
    out->stride_flags = 0;
    out->has_seeds = 0;
    for (i = 0; i < 16; i++) {
        out->seeds[i] = 0;
    }

    switch (transform_id) {
    case ZPSE_T_RAW_BYTES:
        if (segment_count != 1) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 1;
        return PSE_OK;
    case ZPSE_T_DELTA8:
    case ZPSE_T_XOR8:
        if (segment_count != 1) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 1) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 1;
        out->has_seeds = 1;
        out->seeds[0] = buf[0];
        return PSE_OK;
    case ZPSE_T_TRANSPOSE_2:
        if (segment_count != 2) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 2;
        return PSE_OK;
    case ZPSE_T_TRANSPOSE_4:
        if (segment_count != 4) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 4;
        return PSE_OK;
    case ZPSE_T_TRANSPOSE_8:
        if (segment_count != 8) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 8;
        return PSE_OK;
    case ZPSE_T_TRANSPOSE_16:
        if (segment_count != 16) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 16;
        return PSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_2:
        if (segment_count != 2) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 2) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 2;
        out->has_seeds = 1;
        out->seeds[0] = buf[0];
        out->seeds[1] = buf[1];
        return PSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_4:
        if (segment_count != 4) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 4) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 4;
        out->has_seeds = 1;
        for (i = 0; i < 4; i++) {
            out->seeds[i] = buf[i];
        }
        return PSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_8:
        if (segment_count != 8) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 8) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 8;
        out->has_seeds = 1;
        for (i = 0; i < 8; i++) {
            out->seeds[i] = buf[i];
        }
        return PSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_16:
        if (segment_count != 16) {
            return PSE_BAD_TRANSFORM;
        }
        if (len != 16) {
            return PSE_BAD_HEADER;
        }
        out->width_W = 16;
        out->has_seeds = 1;
        for (i = 0; i < 16; i++) {
            out->seeds[i] = buf[i];
        }
        return PSE_OK;
    case ZPSE_T_STRIDE_TRANSPOSE: {
        uint8_t w, sflags;

        /* segment_count already validated to be in {1,2,4,8,16} */
        if (len != 2 && len != (uint16_t)(2u + (uint16_t)segment_count)) {
            return PSE_BAD_HEADER;
        }
        w = buf[0];
        sflags = buf[1];
        if (!zpse_is_valid_segment_count(w)) {
            return PSE_BAD_TRANSFORM;
        }
        if (w != segment_count) {
            return PSE_BAD_TRANSFORM;
        }
        if ((sflags & (uint8_t)0xFE) != 0) {
            return PSE_UNSUPPORTED_FEATURE;
        }
        if ((sflags & 0x01u) == 0) {
            if (len != 2) {
                return PSE_BAD_HEADER;
            }
            out->width_W = w;
            out->stride_flags = sflags;
            out->has_seeds = 0;
            return PSE_OK;
        }
        /* DELTA flag set: expect 2+W bytes */
        if (len != (uint16_t)(2u + (uint16_t)w)) {
            return PSE_BAD_HEADER;
        }
        out->width_W = w;
        out->stride_flags = sflags;
        out->has_seeds = 1;
        for (i = 0; i < w; i++) {
            out->seeds[i] = buf[2 + i];
        }
        return PSE_OK;
    }
    default:
        return PSE_BAD_TRANSFORM;
    }
}
