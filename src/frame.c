#include "zpse/frame.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "zpse/bits.h"
#include "zpse/block.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/status.h"
#include "zpse/types.h"

static void zpse_frame_pack28(uint8_t out[32], const ZpseFrameHeader *h)
{
    zpse_store_u32le(out + 0, h->magic);
    out[4] = h->version_major;
    out[5] = h->version_minor;
    zpse_store_u16le(out + 6, h->frame_flags);
    zpse_store_u32le(out + 8, h->nominal_block_size);
    zpse_store_u64le(out + 12, h->content_size);
    zpse_store_u32le(out + 20, h->block_count);
    zpse_store_u32le(out + 24, h->frame_index_offset);
}

int zpse_frame_header_build(ZpseFrameHeader *h, uint32_t nominal_block_size,
                            uint64_t content_size, int content_valid,
                            uint32_t block_count, int count_valid,
                            int with_index)
{
    uint8_t tmp[32];

    if (h == 0) {
        return PSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (with_index) {
        /* FrameIndex emission is unimplemented: accepting the flag would
         * emit PRESENT with offset 0, violating Sec. 3.1/3.2. The encoder
         * always passes 0. */
        return PSE_BAD_ARG;
    }
    h->magic = ZPSE_MAGIC;
    h->version_major = ZPSE_FRAME_MAJOR;
    h->version_minor = ZPSE_FRAME_MINOR;
    h->frame_flags = (uint16_t)(
        (content_valid ? (uint16_t)ZPSE_F_CONTENT_SIZE : (uint16_t)0) |
        (count_valid ? (uint16_t)ZPSE_F_BLOCK_COUNT : (uint16_t)0) |
        (with_index ? (uint16_t)ZPSE_F_FRAME_INDEX : (uint16_t)0));
    h->nominal_block_size = nominal_block_size;
    h->content_size = content_valid ? content_size : UINT64_MAX;
    h->block_count = count_valid ? block_count : UINT32_MAX;
    /* Index payload (if any) is patched by the caller afterwards. */
    h->frame_index_offset = 0u;
    zpse_frame_pack28(tmp, h);
    h->header_crc32c = zpse_crc32c(tmp, 28u);
    return PSE_OK;
}

int zpse_frame_header_write(const ZpseFrameHeader *h, uint8_t out[32])
{
    if (h == 0 || out == 0) {
        return PSE_BAD_ARG;
    }
    zpse_frame_pack28(out, h);
    zpse_store_u32le(out + 28, h->header_crc32c);
    return PSE_OK;
}

int zpse_frame_header_read(ZpseFrameHeader *h, const uint8_t in[32])
{
    ZpseFrameHeader tmp;
    uint32_t want;

    if (h == 0 || in == 0) {
        return PSE_BAD_ARG;
    }
    tmp.magic = zpse_load_u32le(in + 0);
    tmp.version_major = in[4];
    tmp.version_minor = in[5];
    tmp.frame_flags = zpse_load_u16le(in + 6);
    tmp.nominal_block_size = zpse_load_u32le(in + 8);
    tmp.content_size = zpse_load_u64le(in + 12);
    tmp.block_count = zpse_load_u32le(in + 20);
    tmp.frame_index_offset = zpse_load_u32le(in + 24);
    tmp.header_crc32c = zpse_load_u32le(in + 28);
    if (tmp.magic != ZPSE_MAGIC) {
        return PSE_BAD_MAGIC;
    }
    if (tmp.version_major != ZPSE_FRAME_MAJOR ||
        tmp.version_minor != ZPSE_FRAME_MINOR) {
        return PSE_BAD_VERSION;
    }
    if ((tmp.frame_flags & ZPSE_F_RESERVED_MASK) != 0u) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (tmp.nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        tmp.nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_HEADER;
    }
    if ((tmp.frame_flags & ZPSE_F_FRAME_INDEX) != 0u) {
        if (tmp.frame_index_offset == 0u) {
            return PSE_BAD_HEADER;
        }
    } else {
        if (tmp.frame_index_offset != 0u) {
            return PSE_BAD_HEADER;
        }
    }
    /* content_size / block_count sentinel-vs-validity mismatch is advisory
     * only (SPEC Sec. 3.1): no rejection here. */
    want = zpse_crc32c(in, 28u);
    if (want != tmp.header_crc32c) {
        return PSE_BAD_HEADER_CRC;
    }
    *h = tmp;
    return PSE_OK;
}

int zpse_frame_decode(ZpseDecoder *d, const uint8_t *in, size_t in_len,
                      uint8_t *output, size_t out_cap, size_t *out_len_out)
{
    ZpseFrameHeader fh;
    size_t pos;
    size_t out_pos;
    size_t nblocks;
    int have_last;
    int has_count;
    int has_content;
    int rc;

    if (d == 0 || in == 0 || output == 0 || out_len_out == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len < ZPSE_FRAME_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    rc = zpse_frame_header_read(&fh, in);
    if (rc != PSE_OK) {
        return rc;
    }
    has_count = (fh.frame_flags & ZPSE_F_BLOCK_COUNT) != 0;
    has_content = (fh.frame_flags & ZPSE_F_CONTENT_SIZE) != 0;

    pos = ZPSE_FRAME_HEADER_SIZE;
    out_pos = 0;
    nblocks = 0;
    have_last = 0;
    *out_len_out = 0;

    while (pos < in_len) {
        size_t rem = in_len - pos;
        uint32_t avail = rem > (size_t)UINT32_MAX ? UINT32_MAX : (uint32_t)rem;
        uint32_t comp_sz = 0;
        uint32_t uncomp_sz = 0;
        uint32_t blk_out = 0;
        size_t o_rem;
        uint32_t o_cap;
        int is_last;

        rc = zpse_block_probe(in + pos, avail, &comp_sz, &uncomp_sz);
        if (rc != PSE_OK) {
            return rc;
        }
        if (comp_sz < ZPSE_BLOCK_HEADER_SIZE) {
            return PSE_BAD_HEADER;
        }
        if ((size_t)comp_sz > rem) {
            return PSE_TRUNCATED;
        }
        /* comp_sz >= 32 and <= rem, so in[pos+3] is in bounds. */
        is_last = (in[pos + 3] & ZPSE_B_LAST) != 0;
        if (uncomp_sz < ZPSE_MIN_BLOCK_SIZE ||
            uncomp_sz > ZPSE_MAX_BLOCK_SIZE) {
            return PSE_BAD_BLOCK_SIZE;
        }
        /* Invariant: out_pos <= out_cap, so the subtraction cannot wrap. */
        if ((uint64_t)uncomp_sz > (uint64_t)(out_cap - out_pos)) {
            return PSE_OUTPUT_OVERFLOW;
        }
        o_rem = out_cap - out_pos;
        o_cap = o_rem > (size_t)UINT32_MAX ? UINT32_MAX : (uint32_t)o_rem;
        rc = zpse_block_decode(d, in + pos, comp_sz,
                              output + out_pos, o_cap, &blk_out);
        if (rc < 0) {
            return rc;
        }
        if (blk_out != uncomp_sz) {
            return PSE_BAD_HEADER;
        }
        /* SPEC Sec. 4.1: payload CRC MUST be verified when the block flag
         * OR the frame flag tags it. The block layer covers its own flag;
         * close the gap for foreign frames with F_BLOCK_CRC=1, B_CRC=0. */
        if ((fh.frame_flags & ZPSE_F_BLOCK_CRC) != 0u) {
            ZpseBlockHeader bh;
            rc = zpse_block_header_parse(&bh, in + pos, comp_sz);
            if (rc != PSE_OK) {
                return rc;
            }
            if ((bh.block_flags & ZPSE_B_CRC) == 0u) {
                uint32_t calc = zpse_crc32c(output + out_pos, blk_out);
                if (calc != bh.decoded_crc32c) {
                    return PSE_BAD_PAYLOAD_CRC;
                }
            }
        }
        out_pos += (size_t)blk_out;
        pos += (size_t)comp_sz;
        nblocks++;
        if (is_last) {
            have_last = 1;
            break;
        }
        if (has_count &&
            (uint64_t)nblocks > (uint64_t)fh.block_count) {
            return PSE_BAD_HEADER;
        }
    }

    if (!have_last) {
        /* Tolerated only when input ends exactly on a block boundary and
         * the declared block count matches. */
        if (pos != in_len) {
            return PSE_BAD_HEADER;
        }
        if (!has_count || (uint64_t)nblocks != (uint64_t)fh.block_count) {
            return PSE_BAD_HEADER;
        }
    }
    if (has_count && (uint64_t)nblocks != (uint64_t)fh.block_count) {
        return PSE_BAD_HEADER;
    }
    if (has_content && (uint64_t)out_pos != (uint64_t)fh.content_size) {
        return PSE_BAD_HEADER;
    }
    *out_len_out = out_pos;
    if (out_pos > (size_t)INT_MAX) {
        return PSE_OK;
    }
    return (int)out_pos;
}

int zpse_frame_encode(ZpseEncoder *e, const uint8_t *in, size_t n,
                      uint32_t nominal_block_size,
                      uint8_t *out, size_t out_cap, size_t *out_len_out)
{
    ZpseFrameHeader fh;
    size_t nblocks;
    size_t pos;
    size_t i;
    int rc;

    if (e == 0 || out == 0 || out_len_out == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0) {
        return PSE_BAD_ARG;
    }
    if (in == 0) {
        return PSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    *out_len_out = 0;
    nblocks = n / (size_t)nominal_block_size +
        ((n % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks == 0 || nblocks > (size_t)UINT32_MAX) {
        return PSE_BAD_ARG;
    }
    if (out_cap < ZPSE_FRAME_HEADER_SIZE) {
        return PSE_OUTPUT_OVERFLOW;
    }
    rc = zpse_frame_header_build(&fh, nominal_block_size, (uint64_t)n, 1,
                                 (uint32_t)nblocks, 1, 0);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_frame_header_write(&fh, out);
    if (rc != PSE_OK) {
        return rc;
    }
    pos = ZPSE_FRAME_HEADER_SIZE;
    for (i = 0; i < nblocks; i++) {
        size_t off = i * (size_t)nominal_block_size;
        size_t chunk;
        size_t rem;
        uint32_t cap;
        int wrc;

        if (off >= n) {
            return PSE_BAD_HEADER;
        }
        chunk = n - off;
        if (chunk > (size_t)nominal_block_size) {
            chunk = (size_t)nominal_block_size;
        }
        /* Invariant: pos <= out_cap, so the subtraction cannot wrap. */
        rem = out_cap - pos;
        if (rem < ZPSE_BLOCK_HEADER_SIZE) {
            return PSE_OUTPUT_OVERFLOW;
        }
        cap = rem > (size_t)UINT32_MAX ? UINT32_MAX : (uint32_t)rem;
        wrc = zpse_block_encode(e, in + off, (uint32_t)chunk, (uint32_t)i,
                               ((i + 1u) == nblocks) ? 1 : 0,
                               out + pos, cap);
        if (wrc < 0) {
            return wrc;
        }
        if (wrc == 0) {
            return PSE_BAD_HEADER;
        }
        if ((size_t)wrc > rem) {
            return PSE_BAD_HEADER;
        }
        pos += (size_t)wrc;
    }
    *out_len_out = pos;
    if (pos > (size_t)INT_MAX) {
        return PSE_OK;
    }
    return (int)pos;
}
