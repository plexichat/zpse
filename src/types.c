#include "zpse/types.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
#include "zpse/predictor.h"
#include "zpse/status.h"

static int zpse_is_valid_segment_count(uint8_t c)
{
    return c == 1 || c == 2 || c == 4 || c == 8 || c == 16;
}

uint32_t zpse_block_header_size(const uint8_t *buf, uint32_t len)
{
    uint16_t sync;

    if (buf == 0 || len < 2u) {
        return 0u;
    }
    sync = zpse_load_u16le(buf + 0);
    if (sync == ZPSE_BLOCK_SYNC) {
        return (uint32_t)ZPSE_BLOCK_HEADER_SIZE;
    }
    if (sync == ZPSE_BLOCK_SYNC_SHORT ||
        sync == ZPSE_BLOCK_SYNC_TOKEN ||
        sync == ZPSE_BLOCK_SYNC_FLAT ||
        sync == ZPSE_BLOCK_SYNC_TEXT) {
        return (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    }
    return 0u;
}

/* The 16-byte page header: [sync][flags][transform][segcount][meta8]
 * [uncomp16][comp16][seq32][crc16]. Everything the long header carries that
 * is not here is fixed by the format (version, tile size, predictor profile,
 * reserved), and a payload CRC has nowhere to live, so a block that wants
 * one is written long. */
static int zpse_block_header_parse_short(ZpseBlockHeader *out,
                                         const uint8_t *buf)
{
    uint8_t flags, tid, scount, meta8;
    uint32_t uncomp, comp, seq;
    uint16_t stored, calc;

    flags = buf[2];
    tid = buf[3];
    scount = buf[4];
    meta8 = buf[5];
    uncomp = (uint32_t)zpse_load_u16le(buf + 6);
    comp = (uint32_t)zpse_load_u16le(buf + 8);
    seq = zpse_load_u32le(buf + 10);
    stored = zpse_load_u16le(buf + 14);
    calc = (uint16_t)(zpse_crc32c(buf, 14) & UINT32_C(0xFFFF));

    if (calc != stored) {
        return ZPSE_BAD_HEADER_CRC;
    }
    if ((flags & ZPSE_B_CRC) != 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if ((flags & ZPSE_B_RESET) == 0u) {
        return ZPSE_BAD_HEADER;
    }
    if (((flags & ZPSE_B_HAS_META) != 0u) != (meta8 > 0u)) {
        return ZPSE_BAD_HEADER;
    }
    /* ZpseTransformMeta holds at most 16 seed bytes and no legal transform
     * needs more; a larger declaration would only drive an OOB access later. */
    if (meta8 > 16u) {
        return ZPSE_BAD_HEADER;
    }
    if ((flags & ZPSE_B_RESERVED_MASK) != 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (uncomp < (uint32_t)ZPSE_MIN_BLOCK_SIZE ||
        uncomp > (uint32_t)ZPSE_BLOCK_SHORT_MAX) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (comp < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (tid > 11u || !zpse_is_valid_segment_count(scount)) {
        return ZPSE_BAD_TRANSFORM;
    }

    out->sync = ZPSE_BLOCK_SYNC_SHORT;
    out->block_version = ZPSE_BLOCK_VERSION;
    out->block_flags = flags;
    out->uncompressed_size = uncomp;
    out->compressed_block_size = comp;
    out->transform_id = tid;
    out->segment_count = scount;
    out->transform_meta_size = (uint16_t)meta8;
    out->tile_size = (uint16_t)ZPSE_TILE_SIZE;
    out->predictor_profile = 0u;
    out->reserved0 = 0u;
    out->block_sequence_number = seq;
    out->decoded_crc32c = 0u;
    out->header_crc32c = stored;
    out->header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    return ZPSE_OK;
}

/* A one-value token (v1.1) is the 16-byte page header with a different sync
 * and the fill byte in transform_id, so the short parser's tid check cannot
 * be reused. Every other field has exactly one legal value and the 16-bit CRC
 * covers the first fourteen bytes. This is the one definition of a legal
 * token: the block parser dispatches here and the block layer's fast paths
 * call the same function, so the two cannot drift apart. */
int zpse_block_token_build(ZpseBlockHeader *out, const uint8_t *buf,
                           uint32_t len)
{
    uint8_t flags, scount, meta8;
    uint32_t uncomp, comp, seq;
    uint16_t stored, calc;

    if (out == 0 || buf == 0) {
        return ZPSE_BAD_ARG;
    }
    if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u16le(buf + 0) != (uint16_t)ZPSE_BLOCK_SYNC_TOKEN) {
        return ZPSE_BAD_HEADER;
    }

    flags = buf[2];
    scount = buf[4];
    meta8 = buf[5];
    uncomp = (uint32_t)zpse_load_u16le(buf + 6);
    comp = (uint32_t)zpse_load_u16le(buf + 8);
    seq = zpse_load_u32le(buf + 10);
    stored = zpse_load_u16le(buf + 14);
    calc = (uint16_t)(zpse_crc32c(buf, 14) & UINT32_C(0xFFFF));

    if (calc != stored) {
        return ZPSE_BAD_HEADER_CRC;
    }
    /* RAW and RESET are required-1 bits and LAST is the only optional one;
     * every other bit names a form or a feature this header has no room for.
     * The fixed fields (segcount, meta, comp) are checked here as well as
     * being forced by the serializer, so a hand-built token is held to the
     * exact same envelope. */
    if ((flags & (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET)) !=
            (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET) ||
        (flags & (uint8_t)(ZPSE_B_CRC | ZPSE_B_HAS_META | ZPSE_B_LZ |
                           ZPSE_B_LZ_TILE | ZPSE_B_LZ2 |
                           ZPSE_B_RESERVED_MASK)) != 0u ||
        scount != 1u || meta8 != 0u ||
        comp != (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT ||
        uncomp < (uint32_t)ZPSE_MIN_BLOCK_SIZE ||
        uncomp > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_HEADER;
    }
    out->sync = ZPSE_BLOCK_SYNC_TOKEN;
    out->block_version = ZPSE_BLOCK_VERSION;
    out->block_flags = flags;
    out->uncompressed_size = uncomp;
    out->compressed_block_size = comp;
    out->transform_id = buf[3]; /* the fill value */
    out->segment_count = 1u;
    out->transform_meta_size = 0u;
    out->tile_size = (uint16_t)ZPSE_TILE_SIZE;
    out->predictor_profile = 0u;
    out->reserved0 = 0u;
    out->block_sequence_number = seq;
    out->decoded_crc32c = 0u;
    out->header_crc32c = stored;
    out->header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    return ZPSE_OK;
}

/* The FLAT page form (BREAKING wire change): the 16-byte header names a flat
 * payload codec rather than a transform + segment directory, so the "transform
 * metadata" slot carries the codec's one parameter. CRC over bytes [0..14),
 * reserved byte 0, exactly RESET (+ optional LAST), codec 0..3, comp >= 16 and
 * n in 1..4096 are the whole envelope; the block parser dispatches here. */
int zpse_block_header_parse_flat(ZpseBlockHeader *out, const uint8_t *buf,
                                 uint32_t len)
{
    uint8_t flags, codec, param, reserved;
    uint32_t uncomp, comp, seq;
    uint16_t stored, calc;

    if (out == 0 || buf == 0) {
        return ZPSE_BAD_ARG;
    }
    if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u16le(buf + 0) != (uint16_t)ZPSE_BLOCK_SYNC_FLAT) {
        return ZPSE_BAD_HEADER;
    }

    flags = buf[2];
    codec = buf[3];
    param = buf[4];
    reserved = buf[5];
    uncomp = (uint32_t)zpse_load_u16le(buf + 6);
    comp = (uint32_t)zpse_load_u16le(buf + 8);
    seq = zpse_load_u32le(buf + 10);
    stored = zpse_load_u16le(buf + 14);
    calc = (uint16_t)(zpse_crc32c(buf, 14) & UINT32_C(0xFFFF));

    if (calc != stored) {
        return ZPSE_BAD_HEADER_CRC;
    }
    if (reserved != 0u) {
        return ZPSE_BAD_HEADER;
    }
    /* RESET is the only required-1 bit and LAST the only optional one; every
     * other bit names a form or feature this header has no room for. */
    if ((flags & (uint8_t)ZPSE_B_RESET) == 0u ||
        (flags & (uint8_t)(ZPSE_B_RAW | ZPSE_B_CRC | ZPSE_B_HAS_META |
                           ZPSE_B_LZ | ZPSE_B_LZ_TILE | ZPSE_B_LZ2 |
                           ZPSE_B_RESERVED_MASK)) != 0u) {
        return ZPSE_BAD_HEADER;
    }
    if (codec > (uint8_t)ZPSE_FLAT_SUBLELANE) {
        return ZPSE_BAD_TRANSFORM;
    }
    if (comp < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (uncomp < (uint32_t)ZPSE_MIN_BLOCK_SIZE ||
        uncomp > (uint32_t)ZPSE_BLOCK_SHORT_MAX) {
        return ZPSE_BAD_BLOCK_SIZE;
    }

    out->sync = ZPSE_BLOCK_SYNC_FLAT;
    out->block_version = ZPSE_BLOCK_VERSION;
    out->block_flags = flags;
    out->uncompressed_size = uncomp;
    out->compressed_block_size = comp;
    out->transform_id = codec;
    /* A FLAT payload has no segment directory; 1 is the minimal legal count
     * (as for a token) and the flat decoder does not consult it. */
    out->segment_count = 1u;
    out->transform_meta_size = (uint16_t)param;
    out->tile_size = (uint16_t)ZPSE_TILE_SIZE;
    out->predictor_profile = 0u;
    out->reserved0 = 0u;
    out->block_sequence_number = seq;
    out->decoded_crc32c = 0u;
    out->header_crc32c = stored;
    out->header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    return ZPSE_OK;
}

/* The TEXT page form: the 16-byte header names no codec -- bytes [3..6) are
 * reserved, and n at [6..8) is the uncompressed size with 0 meaning 65536.
 * CRC over bytes [0..14), reserved bytes 0, exactly RESET (+ optional LAST),
 * comp >= 16 and n in 1..65536 (or its 0 encoding) are the whole envelope; the
 * block parser dispatches here on the TEXT sync. */
int zpse_block_header_parse_text(ZpseBlockHeader *out, const uint8_t *buf,
                                 uint32_t len)
{
    uint8_t flags;
    uint16_t nstored;
    uint32_t uncomp, comp, seq;
    uint16_t stored, calc;

    if (out == 0 || buf == 0) {
        return ZPSE_BAD_ARG;
    }
    if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u16le(buf + 0) != (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
        return ZPSE_BAD_HEADER;
    }

    flags = buf[2];
    nstored = zpse_load_u16le(buf + 6);
    comp = (uint32_t)zpse_load_u16le(buf + 8);
    seq = zpse_load_u32le(buf + 10);
    stored = zpse_load_u16le(buf + 14);
    calc = (uint16_t)(zpse_crc32c(buf, 14) & UINT32_C(0xFFFF));

    if (calc != stored) {
        return ZPSE_BAD_HEADER_CRC;
    }
    if (buf[3] != 0u || buf[4] != 0u || buf[5] != 0u) {
        return ZPSE_BAD_HEADER;
    }
    /* RESET is the only required-1 bit and LAST the only optional one; every
     * other bit names a form or feature this header has no room for. */
    if ((flags & (uint8_t)ZPSE_B_RESET) == 0u ||
        (flags & (uint8_t)(ZPSE_B_RAW | ZPSE_B_CRC | ZPSE_B_HAS_META |
                           ZPSE_B_LZ | ZPSE_B_LZ_TILE | ZPSE_B_LZ2 |
                           ZPSE_B_RESERVED_MASK)) != 0u) {
        return ZPSE_BAD_HEADER;
    }
    if (comp < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    /* On the wire n is 16-bit: 0 is legal and spells 65536, not an error. */
    uncomp = (nstored == 0u) ? UINT32_C(65536) : (uint32_t)nstored;

    out->sync = ZPSE_BLOCK_SYNC_TEXT;
    out->block_version = ZPSE_BLOCK_VERSION;
    out->block_flags = flags;
    out->uncompressed_size = uncomp;
    out->compressed_block_size = comp;
    out->transform_id = 0u;
    out->segment_count = 1u;
    out->transform_meta_size = 0u;
    out->tile_size = (uint16_t)ZPSE_TILE_SIZE;
    out->predictor_profile = 0u;
    out->reserved0 = 0u;
    out->block_sequence_number = seq;
    out->decoded_crc32c = 0u;
    out->header_crc32c = stored;
    out->header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    return ZPSE_OK;
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
        return ZPSE_BAD_ARG;
    }
    if (len < 2u) {
        return ZPSE_TRUNCATED;
    }

    /* One sync load selects the header form and its length. Caching it here
     * avoids re-reading the first two bytes through the size helper and the
     * chain of form tests; the per-form parsers below are unchanged. */
    sync = zpse_load_u16le(buf + 0);
    switch (sync) {
    case ZPSE_BLOCK_SYNC_TOKEN:
        if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_TRUNCATED;
        }
        return zpse_block_token_build(out, buf, len);
    case ZPSE_BLOCK_SYNC_FLAT:
        if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_TRUNCATED;
        }
        return zpse_block_header_parse_flat(out, buf, len);
    case ZPSE_BLOCK_SYNC_TEXT:
        if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_TRUNCATED;
        }
        return zpse_block_header_parse_text(out, buf, len);
    case ZPSE_BLOCK_SYNC_SHORT:
        if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_TRUNCATED;
        }
        return zpse_block_header_parse_short(out, buf);
    case ZPSE_BLOCK_SYNC:
        if (len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
            return ZPSE_TRUNCATED;
        }
        break; /* the long header is parsed below */
    default:
        return ZPSE_BAD_HEADER;
    }

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
        return ZPSE_BAD_HEADER;
    }
    if (ver != ZPSE_BLOCK_VERSION) {
        return ZPSE_BAD_VERSION;
    }
    calc_crc = zpse_crc32c(buf, 28);
    if (calc_crc != stored_crc) {
        return ZPSE_BAD_HEADER_CRC;
    }
    if (uncomp < ZPSE_MIN_BLOCK_SIZE || uncomp > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (comp < ZPSE_BLOCK_HEADER_SIZE) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (tid > 11) {
        return ZPSE_BAD_TRANSFORM;
    }
    if (!zpse_is_valid_segment_count(scount)) {
        return ZPSE_BAD_TRANSFORM;
    }
    if (tile_size != ZPSE_TILE_SIZE) {
        return ZPSE_BAD_TILE_SIZE;
    }
    if (profile != 0) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (r0 != 0) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    /* ZpseTransformMeta holds at most 16 seed bytes and no legal transform
     * needs more; a larger declaration would only drive an OOB access later. */
    if (meta_size > 16u) {
        return ZPSE_BAD_HEADER;
    }
    has_meta = (meta_size > 0) ? 1 : 0;
    if (((flags & ZPSE_B_HAS_META) != 0) != has_meta) {
        return ZPSE_BAD_HEADER;
    }
    if ((flags & ZPSE_B_RESET) == 0) {
        return ZPSE_BAD_HEADER; /* required-1 bit, cf. HAS_META agreement */
    }
    if ((flags & ZPSE_B_RESERVED_MASK) != 0) {
        return ZPSE_UNSUPPORTED_FEATURE;
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
    out->header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE;
    return ZPSE_OK;
}

void zpse_block_header_serialize_short(uint8_t out[16], const ZpseBlockHeader *h)
{
    uint32_t crc;

    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u16le(out + 0, ZPSE_BLOCK_SYNC_SHORT);
    out[2] = h->block_flags;
    out[3] = h->transform_id;
    out[4] = h->segment_count;
    out[5] = (uint8_t)h->transform_meta_size;
    zpse_store_u16le(out + 6, (uint16_t)h->uncompressed_size);
    zpse_store_u16le(out + 8, (uint16_t)h->compressed_block_size);
    zpse_store_u32le(out + 10, h->block_sequence_number);
    crc = zpse_crc32c(out, 14);
    zpse_store_u16le(out + 14, (uint16_t)(crc & UINT32_C(0xFFFF)));
}

/* The token's envelope is fixed, so the serializer writes the fixed fields
 * rather than trusting them to `h`: a header built by a caller with, say, a
 * stale segment_count still emits the one layout every decoder reads. The
 * variable fields (flags, fill, size, seq) come from `h` and must have been
 * validated first; zpse_block_token_build is both the validator and the
 * reader that guarantees it. */
void zpse_block_header_serialize_token(uint8_t out[16], const ZpseBlockHeader *h)
{
    uint32_t crc;

    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u16le(out + 0, (uint16_t)ZPSE_BLOCK_SYNC_TOKEN);
    out[2] = h->block_flags;
    out[3] = h->transform_id; /* the fill value */
    out[4] = 1u;              /* one segment, always */
    out[5] = 0u;              /* no transform metadata, always */
    zpse_store_u16le(out + 6, (uint16_t)h->uncompressed_size);
    zpse_store_u16le(out + 8, (uint16_t)ZPSE_BLOCK_HEADER_SIZE_SHORT);
    zpse_store_u32le(out + 10, h->block_sequence_number);
    crc = zpse_crc32c(out, 14);
    zpse_store_u16le(out + 14, (uint16_t)(crc & UINT32_C(0xFFFF)));
}

/* Mirror of the FLAT parser: the sync and reserved byte are fixed, flags and
 * codec/param are taken from an already-validated `h`, and the 16-bit CRC is
 * recomputed over the first fourteen bytes. */
void zpse_block_header_serialize_flat(uint8_t out[16], const ZpseBlockHeader *h)
{
    uint32_t crc;

    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u16le(out + 0, (uint16_t)ZPSE_BLOCK_SYNC_FLAT);
    out[2] = h->block_flags;
    out[3] = h->transform_id;                  /* codec */
    out[4] = (uint8_t)h->transform_meta_size;  /* param */
    out[5] = 0u;                               /* reserved, always */
    zpse_store_u16le(out + 6, (uint16_t)h->uncompressed_size);
    zpse_store_u16le(out + 8, (uint16_t)h->compressed_block_size);
    zpse_store_u32le(out + 10, h->block_sequence_number);
    crc = zpse_crc32c(out, 14);
    zpse_store_u16le(out + 14, (uint16_t)(crc & UINT32_C(0xFFFF)));
}

/* Mirror of the TEXT parser: the sync and reserved bytes are fixed, flags,
 * n and c/seq are taken from an already-validated `h`, and the 16-bit CRC is
 * recomputed over the first fourteen bytes. A size of 65536 is stored as 0. */
void zpse_block_header_serialize_text(uint8_t out[16], const ZpseBlockHeader *h)
{
    uint32_t crc;
    uint16_t nstored;

    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u16le(out + 0, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
    out[2] = h->block_flags;
    out[3] = 0u;
    out[4] = 0u;
    out[5] = 0u;
    nstored = (h->uncompressed_size == UINT32_C(65536))
                  ? 0u
                  : (uint16_t)h->uncompressed_size;
    zpse_store_u16le(out + 6, nstored);
    zpse_store_u16le(out + 8, (uint16_t)h->compressed_block_size);
    zpse_store_u32le(out + 10, h->block_sequence_number);
    crc = zpse_crc32c(out, 14);
    zpse_store_u16le(out + 14, (uint16_t)(crc & UINT32_C(0xFFFF)));
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

    if (out == 0 || buf == 0) {
        return ZPSE_BAD_ARG;
    }
    if (len < ZPSE_SEGMENT_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    dec_size = zpse_load_u32le(buf + 0);
    tile_count = zpse_load_u16le(buf + 4);
    init_lit = buf[6];
    sflags = buf[7];

    if (tile_count == 0) {
        return ZPSE_BAD_TILE_SIZE;
    }
    if (dec_size == 0) {
        return ZPSE_BAD_TILE_SIZE;
    }
    if (sflags != ZPSE_S_LEN_PREFIX) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }

    /* The directory size and the reserved word are no longer carried: both
     * had exactly one legal value (zero), so the bytes that held them said
     * nothing. A reader that wants them still gets the only values a stream
     * can have produced. */
    out->segment_decoded_size = dec_size;
    out->tile_count = tile_count;
    out->initial_literal = init_lit;
    out->segment_flags = sflags;
    out->tile_directory_size = 0u;
    out->reserved = 0u;
    return ZPSE_OK;
}

void zpse_segment_header_serialize(uint8_t out[8], const ZpseSegmentHeader *h)
{
    if (out == 0 || h == 0) {
        return;
    }
    zpse_store_u32le(out + 0, h->segment_decoded_size);
    zpse_store_u16le(out + 4, h->tile_count);
    out[6] = h->initial_literal;
    out[7] = h->segment_flags;
}

int zpse_transform_meta_parse(ZpseTransformMeta *out, uint8_t transform_id,
                              const uint8_t *buf, uint16_t len,
                              uint8_t segment_count)
{
    uint8_t i;

    if (out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (len > 0 && buf == 0) {
        return ZPSE_BAD_ARG;
    }
    if (transform_id > 11) {
        return ZPSE_BAD_TRANSFORM;
    }
    if (!zpse_is_valid_segment_count(segment_count)) {
        return ZPSE_BAD_TRANSFORM;
    }

    out->transform_id = transform_id;
    out->width_W = 1;
    out->stride_flags = 0;
    out->has_seeds = 0;
    memset(out->seeds, 0, sizeof out->seeds);

    switch (transform_id) {
    case ZPSE_T_RAW_BYTES:
        if (segment_count != 1) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 1;
        return ZPSE_OK;
    case ZPSE_T_DELTA8:
    case ZPSE_T_XOR8:
        if (segment_count != 1) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 1) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 1;
        out->has_seeds = 1;
        out->seeds[0] = buf[0];
        return ZPSE_OK;
    case ZPSE_T_TRANSPOSE_2:
        if (segment_count != 2) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 2;
        return ZPSE_OK;
    case ZPSE_T_TRANSPOSE_4:
        if (segment_count != 4) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 4;
        return ZPSE_OK;
    case ZPSE_T_TRANSPOSE_8:
        if (segment_count != 8) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 8;
        return ZPSE_OK;
    case ZPSE_T_TRANSPOSE_16:
        if (segment_count != 16) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 0) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 16;
        return ZPSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_2:
        if (segment_count != 2) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 2) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 2;
        out->has_seeds = 1;
        out->seeds[0] = buf[0];
        out->seeds[1] = buf[1];
        return ZPSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_4:
        if (segment_count != 4) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 4) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 4;
        out->has_seeds = 1;
        for (i = 0; i < 4; i++) {
            out->seeds[i] = buf[i];
        }
        return ZPSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_8:
        if (segment_count != 8) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 8) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 8;
        out->has_seeds = 1;
        for (i = 0; i < 8; i++) {
            out->seeds[i] = buf[i];
        }
        return ZPSE_OK;
    case ZPSE_T_DELTA_TRANSPOSE_16:
        if (segment_count != 16) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (len != 16) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = 16;
        out->has_seeds = 1;
        for (i = 0; i < 16; i++) {
            out->seeds[i] = buf[i];
        }
        return ZPSE_OK;
    case ZPSE_T_STRIDE_TRANSPOSE: {
        uint8_t w, sflags;

        /* segment_count already validated to be in {1,2,4,8,16} */
        if (len != 2 && len != (uint16_t)(2u + (uint16_t)segment_count)) {
            return ZPSE_BAD_HEADER;
        }
        w = buf[0];
        sflags = buf[1];
        if (!zpse_is_valid_segment_count(w)) {
            return ZPSE_BAD_TRANSFORM;
        }
        if (w != segment_count) {
            return ZPSE_BAD_TRANSFORM;
        }
        if ((sflags & (uint8_t)0xFE) != 0) {
            return ZPSE_UNSUPPORTED_FEATURE;
        }
        if ((sflags & 0x01u) == 0) {
            if (len != 2) {
                return ZPSE_BAD_HEADER;
            }
            out->width_W = w;
            out->stride_flags = sflags;
            out->has_seeds = 0;
            return ZPSE_OK;
        }
        /* DELTA flag set: expect 2+W bytes */
        if (len != (uint16_t)(2u + (uint16_t)w)) {
            return ZPSE_BAD_HEADER;
        }
        out->width_W = w;
        out->stride_flags = sflags;
        out->has_seeds = 1;
        for (i = 0; i < w; i++) {
            out->seeds[i] = buf[2 + i];
        }
        return ZPSE_OK;
    }
    default:
        return ZPSE_BAD_TRANSFORM;
    }
}
