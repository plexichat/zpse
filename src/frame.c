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

static int zpse_frame_index_check(const uint8_t *in, size_t in_len,
                                  const ZpseFrameHeader *fh,
                                  const uint8_t **entries_out,
                                  uint32_t *count_out);

#if ZPSE_FRAME_PAGE_FAST
/* Read a page block's header once for the framing loop. A frame whose blocks
 * are at most ZPSE_BLOCK_SHORT_MAX normally carries a 16-byte header (short,
 * token, FLAT or TEXT). The full loop reads that header three times per block
 * -- zpse_block_probe, the LAST-flag parse and the one inside
 * zpse_block_decode -- so a page frame pays three 14-byte checksums and three
 * field walks to learn what one gives it. This reads it once and returns the
 * sizes and flags the loop needs; the block is then decoded straight from
 * these bytes with the one decoder the frame already reuses.
 *
 * The caller has already established `in` starts with a short, token, FLAT or
 * TEXT sync and that `avail` covers a 16-byte header, so this takes the lightweight
 * short parser path, not the generic dispatch. Every check and status here is
 * the one zpse_block_probe plus its header parse would return, in the same
 * order. */
static int zpse_frame_short_header(const uint8_t *in, uint32_t avail,
                                   uint32_t *comp_out, uint32_t *uncomp_out,
                                   uint8_t *flags_out)
{
    ZpseBlockHeader bh;
    uint16_t sync;
    int rc;

    if (avail < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    sync = zpse_load_u16le(in);
    if (sync != (uint16_t)ZPSE_BLOCK_SYNC_SHORT &&
        sync != (uint16_t)ZPSE_BLOCK_SYNC_TOKEN &&
        sync != (uint16_t)ZPSE_BLOCK_SYNC_FLAT &&
        sync != (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
        /* Same refusal as zpse_block_probe for an unrecognised sync. */
        return ZPSE_TRUNCATED;
    }
    if (sync == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN) {
        /* zpse_block_probe validates these bytes before its checksum and
         * before the ordinary parse. Keep that order so a token corrupt in
         * two ways is refused with the same status the full path returns. */
        if (in[4] != 1u || in[5] != 0u ||
            in[8] != (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT || in[9] != 0u) {
            return ZPSE_BAD_HEADER;
        }
    }
    /* For a 16-byte header this is the short/token/FLAT/TEXT parse: a 14-byte
     * checksum and the handful of page fields, no long-form work. */
    rc = zpse_block_header_parse(&bh, in, avail);
    if (rc != ZPSE_OK) {
        return rc;
    }
    if (bh.compressed_block_size < (uint32_t)bh.header_size) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (sync == (uint16_t)ZPSE_BLOCK_SYNC_FLAT &&
        bh.compressed_block_size == (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        /* FLAT has no empty payload: the probe refuses comp == header_size
         * as BAD_BLOCK_SIZE. Mirror that status so this fast path returns
         * exactly what zpse_block_probe would. */
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (sync == (uint16_t)ZPSE_BLOCK_SYNC_TEXT &&
        bh.compressed_block_size == (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        /* TEXT has no empty payload: the probe refuses comp == header_size
         * as BAD_BLOCK_SIZE. Mirror that status so this fast path returns
         * exactly what zpse_block_probe would. */
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (avail < bh.compressed_block_size) {
        return ZPSE_TRUNCATED;
    }
    *comp_out = bh.compressed_block_size;
    *uncomp_out = bh.uncompressed_size;
    *flags_out = bh.block_flags;
    return ZPSE_OK;
}

/* Read a long-form (32-byte) block header once for the framing loop. This is
 * the 64 KiB block case. zpse_block_probe would call zpse_block_header_size
 * (32 for this sync), refuse avail < 32 as TRUNCATED, call
 * zpse_block_header_parse, refuse compressed_block_size < 32 as BAD_BLOCK_SIZE
 * (which the parse already does) and avail < compressed_block_size as
 * TRUNCATED; the loop would then call zpse_block_header_parse a second time
 * for the LAST flag. One parse here supplies the sizes and the flag.
 *
 * The caller has established `in` starts with a ZPSE_BLOCK_SYNC sync and that
 * avail covers at least a short header. Every check and status is the one
 * zpse_block_probe plus its header parse would return, in the same order. */
static int zpse_frame_long_header(const uint8_t *in, uint32_t avail,
                                  uint32_t *comp_out, uint32_t *uncomp_out,
                                  uint8_t *flags_out)
{
    ZpseBlockHeader bh;
    int rc;

    /* zpse_block_probe refuses in_len < hdr (32) with TRUNCATED before it
     * parses. This subsumes its earlier in_len < 16 check for this sync. */
    if (avail < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    rc = zpse_block_header_parse(&bh, in, avail);
    if (rc != ZPSE_OK) {
        return rc;
    }
    /* The probe's comp < hdr -> BAD_BLOCK_SIZE is already enforced inside
     * zpse_block_header_parse (comp < ZPSE_BLOCK_HEADER_SIZE). What remains
     * is the probe's in_len < comp length bound, in the same position. */
    if (avail < bh.compressed_block_size) {
        return ZPSE_TRUNCATED;
    }
    *comp_out = bh.compressed_block_size;
    *uncomp_out = bh.uncompressed_size;
    *flags_out = bh.block_flags;
    return ZPSE_OK;
}
#endif /* ZPSE_FRAME_PAGE_FAST */

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
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
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
    return ZPSE_OK;
}

/* Re-seal the header after a field moved: the CRC covers the 28 bytes
 * before it, so patching frame_index_offset means recomputing it. */
static void zpse_frame_reseal(ZpseFrameHeader *h)
{
    uint8_t tmp[32];

    zpse_frame_pack28(tmp, h);
    h->header_crc32c = zpse_crc32c(tmp, 28u);
}

int zpse_frame_header_write(const ZpseFrameHeader *h, uint8_t out[32])
{
    if (h == 0 || out == 0) {
        return ZPSE_BAD_ARG;
    }
    zpse_frame_pack28(out, h);
    zpse_store_u32le(out + 28, h->header_crc32c);
    return ZPSE_OK;
}

int zpse_frame_header_read(ZpseFrameHeader *h, const uint8_t in[32])
{
    ZpseFrameHeader tmp;
    uint32_t want;

    if (h == 0 || in == 0) {
        return ZPSE_BAD_ARG;
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
        return ZPSE_BAD_MAGIC;
    }
    if (tmp.version_major != ZPSE_FRAME_MAJOR ||
        tmp.version_minor != ZPSE_FRAME_MINOR) {
        return ZPSE_BAD_VERSION;
    }
    if ((tmp.frame_flags & ZPSE_F_RESERVED_MASK) != 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (tmp.nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        tmp.nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_HEADER;
    }
    if ((tmp.frame_flags & ZPSE_F_FRAME_INDEX) != 0u) {
        if (tmp.frame_index_offset == 0u) {
            return ZPSE_BAD_HEADER;
        }
    } else {
        if (tmp.frame_index_offset != 0u) {
            return ZPSE_BAD_HEADER;
        }
    }
    /* content_size / block_count sentinel-vs-validity mismatch is advisory
     * only: no rejection here. */
    want = zpse_crc32c(in, 28u);
    if (want != tmp.header_crc32c) {
        return ZPSE_BAD_HEADER_CRC;
    }
    *h = tmp;
    return ZPSE_OK;
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
        return ZPSE_BAD_ARG;
    }
    if (in_len < ZPSE_FRAME_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    rc = zpse_frame_header_read(&fh, in);
    if (rc != ZPSE_OK) {
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
        uint8_t blk_flags = 0u;
        size_t o_rem;
        uint32_t o_cap;
        int is_last;
        int page_fast = 0;

#if ZPSE_FRAME_PAGE_FAST
        /* Read each block's header once here. A page frame's blocks carry the
         * 16-byte header (short/token/FLAT/TEXT); a 64 KiB block carries the
         * 32-byte long header. One parse supplies the sizes and the LAST flag
         * the loop needs, rather than the probe plus the second parse the
         * original path paid. An unrecognised sync still falls through to
         * zpse_block_probe unchanged. */
        if (avail >= (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            uint16_t sync = zpse_load_u16le(in + pos);

            if (fh.nominal_block_size <= (uint32_t)ZPSE_BLOCK_SHORT_MAX &&
                (sync == (uint16_t)ZPSE_BLOCK_SYNC_SHORT ||
                 sync == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN ||
                 sync == (uint16_t)ZPSE_BLOCK_SYNC_FLAT ||
                 sync == (uint16_t)ZPSE_BLOCK_SYNC_TEXT)) {
                rc = zpse_frame_short_header(in + pos, avail, &comp_sz,
                                             &uncomp_sz, &blk_flags);
                if (rc != ZPSE_OK) {
                    return rc;
                }
                page_fast = 1;
            } else if (sync == (uint16_t)ZPSE_BLOCK_SYNC) {
                rc = zpse_frame_long_header(in + pos, avail, &comp_sz,
                                            &uncomp_sz, &blk_flags);
                if (rc != ZPSE_OK) {
                    return rc;
                }
                page_fast = 1;
            }
            /* An unrecognised sync falls through to the original probe path. */
        }
#endif
        if (!page_fast) {
            rc = zpse_block_probe(in + pos, avail, &comp_sz, &uncomp_sz);
            if (rc != ZPSE_OK) {
                return rc;
            }
            /* Parse the header once here for the LAST flag, the only field
             * the framing loop still needs from it. */
            {
                ZpseBlockHeader bh;

                rc = zpse_block_header_parse(&bh, in + pos, comp_sz);
                if (rc != ZPSE_OK) {
                    return rc;
                }
                blk_flags = bh.block_flags;
            }
        }
        if (comp_sz < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_BAD_HEADER;
        }
        if ((size_t)comp_sz > rem) {
            return ZPSE_TRUNCATED;
        }
        is_last = (blk_flags & ZPSE_B_LAST) != 0;
        if (uncomp_sz < ZPSE_MIN_BLOCK_SIZE ||
            uncomp_sz > ZPSE_MAX_BLOCK_SIZE) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        /* Invariant: out_pos <= out_cap, so the subtraction cannot wrap. */
        if ((uint64_t)uncomp_sz > (uint64_t)(out_cap - out_pos)) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        o_rem = out_cap - out_pos;
        o_cap = o_rem > (size_t)UINT32_MAX ? UINT32_MAX : (uint32_t)o_rem;
        rc = zpse_block_decode(d, in + pos, comp_sz,
                              output + out_pos, o_cap, &blk_out);
        if (rc < 0) {
            return rc;
        }
        if (blk_out != uncomp_sz) {
            return ZPSE_BAD_HEADER;
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
            return ZPSE_BAD_HEADER;
        }
    }

    if (!have_last) {
        return ZPSE_BAD_HEADER;
    }
    if (has_count && (uint64_t)nblocks != (uint64_t)fh.block_count) {
        return ZPSE_BAD_HEADER;
    }
    if (has_content && (uint64_t)out_pos != (uint64_t)fh.content_size) {
        return ZPSE_BAD_HEADER;
    }
    /* A frame that claims an index has to carry one that describes
     * exactly the blocks just decoded: a table that disagrees with the
     * stream is a corrupt frame, and a reader that seeks by it would land
     * somewhere else than a sequential reader. */
    if ((fh.frame_flags & ZPSE_F_FRAME_INDEX) != 0u) {
        uint32_t count = 0u;

        rc = zpse_frame_index_check(in, in_len, &fh, 0, &count);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if ((uint64_t)count != (uint64_t)nblocks) {
            return ZPSE_BAD_HEADER;
        }
    }
    *out_len_out = out_pos;
    if (out_pos > (size_t)INT_MAX) {
        return ZPSE_OK;
    }
    return (int)out_pos;
}

/* Read and validate the index of a frame that declares one. On success
 * *entries_out (when asked) points at the first entry, inside `in`. */
static int zpse_frame_index_check(const uint8_t *in, size_t in_len,
                                  const ZpseFrameHeader *fh,
                                  const uint8_t **entries_out,
                                  uint32_t *count_out)
{
    uint32_t off;
    uint32_t count;
    uint32_t crc;
    size_t need;

    if ((fh->frame_flags & ZPSE_F_FRAME_INDEX) == 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    off = fh->frame_index_offset;
    if ((size_t)off < (size_t)ZPSE_FRAME_HEADER_SIZE || (size_t)off > in_len) {
        return ZPSE_BAD_HEADER;
    }
    if (in_len - (size_t)off < (size_t)ZPSE_INDEX_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u32le(in + off) != ZPSE_INDEX_MAGIC) {
        return ZPSE_BAD_HEADER;
    }
    count = zpse_load_u32le(in + off + 4u);
    /* Guard the size_t multiply against wrap on 32-bit hosts: a hostile
     * count near 2^32 would otherwise wrap `need` small and pass the check
     * below, then overrun the entry reads in zpse_frame_seek. */
    if ((uint64_t)count * (uint64_t)ZPSE_INDEX_ENTRY_SIZE >
        (uint64_t)((size_t)-1 - (size_t)ZPSE_INDEX_HEADER_SIZE)) {
        return ZPSE_TRUNCATED;
    }
    need = (size_t)ZPSE_INDEX_HEADER_SIZE +
           (size_t)count * (size_t)ZPSE_INDEX_ENTRY_SIZE;
    if (need > in_len - (size_t)off) {
        return ZPSE_TRUNCATED;
    }
    crc = zpse_load_u32le(in + off + 8u);
    if (zpse_crc32c(in + off + (size_t)ZPSE_INDEX_HEADER_SIZE,
                    (size_t)count * (size_t)ZPSE_INDEX_ENTRY_SIZE) != crc) {
        return ZPSE_BAD_HEADER_CRC;
    }
    if (zpse_load_u32le(in + off + 12u) != 0u) {
        return ZPSE_BAD_HEADER;
    }
    if ((fh->frame_flags & ZPSE_F_BLOCK_COUNT) != 0u &&
        (uint64_t)count != (uint64_t)fh->block_count) {
        return ZPSE_BAD_HEADER;
    }
    if (entries_out != 0) {
        *entries_out = in + off + (size_t)ZPSE_INDEX_HEADER_SIZE;
    }
    if (count_out != 0) {
        *count_out = count;
    }
    return ZPSE_OK;
}

/* Read and validate a frame's header and index in one pass. On success
 * *entries_out (when asked) points inside `in` at the first index entry and
 * *count_out holds the block count; a frame with no index reports count 0 and
 * a null entries pointer. This is the shared front half of the one-shot seek
 * and of the parallel decoder: the header CRC and the CRC over all count*8
 * index bytes are paid once per frame, not once per block. */
int zpse_frame_index_resolve(const uint8_t *in, size_t in_len,
                             ZpseFrameHeader *fh_out,
                             const uint8_t **entries_out,
                             uint32_t *count_out)
{
    ZpseFrameHeader fh;
    int rc;

    if (in == 0 || count_out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (in_len < (size_t)ZPSE_FRAME_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    rc = zpse_frame_header_read(&fh, in);
    if (rc != ZPSE_OK) {
        return rc;
    }
    if ((fh.frame_flags & ZPSE_F_FRAME_INDEX) == 0u) {
        *count_out = 0u;
        if (entries_out != 0) {
            *entries_out = 0;
        }
        if (fh_out != 0) {
            *fh_out = fh;
        }
        return ZPSE_OK;
    }
    rc = zpse_frame_index_check(in, in_len, &fh, entries_out, count_out);
    if (rc != ZPSE_OK) {
        return rc;
    }
    if (fh_out != 0) {
        *fh_out = fh;
    }
    return ZPSE_OK;
}

int zpse_frame_index_count(const uint8_t *in, size_t in_len,
                           uint32_t *count_out)
{
    return zpse_frame_index_resolve(in, in_len, 0, 0, count_out);
}

int zpse_frame_index_validate(const ZpseFrameHeader *fh, const uint8_t *idx,
                              size_t idx_len, uint32_t count,
                              const uint32_t *offsets, const uint32_t *sizes)
{
    size_t need;
    uint32_t k;

    if (fh == 0 || idx == 0 || offsets == 0 || sizes == 0) {
        return ZPSE_BAD_ARG;
    }
    if ((fh->frame_flags & ZPSE_F_FRAME_INDEX) == 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (count == 0u || idx_len < (size_t)ZPSE_INDEX_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u32le(idx) != ZPSE_INDEX_MAGIC) {
        return ZPSE_BAD_HEADER;
    }
    if ((uint64_t)count * (uint64_t)ZPSE_INDEX_ENTRY_SIZE >
        (uint64_t)((size_t)-1 - (size_t)ZPSE_INDEX_HEADER_SIZE)) {
        return ZPSE_TRUNCATED;
    }
    need = (size_t)ZPSE_INDEX_HEADER_SIZE +
           (size_t)count * (size_t)ZPSE_INDEX_ENTRY_SIZE;
    if (idx_len < need) {
        return ZPSE_TRUNCATED;
    }
    /* The count in the table and the reserved word both have to be the ones
     * the format writes, and the entries have to be the ones the CRC covers. */
    if (zpse_load_u32le(idx + 4u) != count) {
        return ZPSE_BAD_HEADER;
    }
    if (zpse_crc32c(idx + (size_t)ZPSE_INDEX_HEADER_SIZE,
                    (size_t)count * (size_t)ZPSE_INDEX_ENTRY_SIZE) !=
        zpse_load_u32le(idx + 8u)) {
        return ZPSE_BAD_HEADER_CRC;
    }
    if (zpse_load_u32le(idx + 12u) != 0u) {
        return ZPSE_BAD_HEADER;
    }
    if ((fh->frame_flags & ZPSE_F_BLOCK_COUNT) != 0u &&
        (uint64_t)count != (uint64_t)fh->block_count) {
        return ZPSE_BAD_HEADER;
    }
    for (k = 0; k < count; k++) {
        const uint8_t *ent = idx + (size_t)ZPSE_INDEX_HEADER_SIZE +
                             (size_t)k * (size_t)ZPSE_INDEX_ENTRY_SIZE;

        if (zpse_load_u32le(ent) != offsets[k] ||
            zpse_load_u32le(ent + 4u) != sizes[k]) {
            return ZPSE_BAD_HEADER;
        }
    }
    /* A table that names the right blocks but sits in the wrong place is
     * still a frame a seeker would read wrongly, so the blocks have to end
     * exactly where the table starts. */
    if ((uint64_t)offsets[count - 1u] + (uint64_t)sizes[count - 1u] !=
        (uint64_t)fh->frame_index_offset) {
        return ZPSE_BAD_HEADER;
    }
    return ZPSE_OK;
}

/* Decode one block addressed by a validated index entry. This is the tail of
 * zpse_frame_seek: the caller has already read the frame header and validated
 * the index as a whole, so all that remains is the per-entry bounds, block
 * header and identity checks, then the block decode. The parallel decoder
 * hands its workers entries resolved once by zpse_frame_index_resolve and
 * calls this directly, so the index CRC is not recomputed per block. */
int zpse_block_at(ZpseDecoder *d, const uint8_t *in, size_t in_len,
                  uint32_t off, uint32_t size, uint32_t seq,
                  uint8_t *output, uint32_t out_cap, uint32_t *out_len_out)
{
    uint32_t got = 0u;
    int rc;

    if (d == 0 || in == 0 || output == 0) {
        return ZPSE_BAD_ARG;
    }
    if ((size_t)off < (size_t)ZPSE_FRAME_HEADER_SIZE ||
        size < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT ||
        (uint64_t)off + (uint64_t)size > (uint64_t)in_len) {
        return ZPSE_BAD_HEADER;
    }
    {
        ZpseBlockHeader bh;

        rc = zpse_block_header_parse(&bh, in + off, size);
        if (rc != ZPSE_OK) {
            return rc;
        }
        /* The entry has to describe the block that is actually there,
         * and it has to be the block that was asked for: a table whose
         * entries point at the wrong blocks is worse than no table. */
        if (bh.compressed_block_size != size ||
            bh.block_sequence_number != seq) {
            return ZPSE_BAD_HEADER;
        }
        if (out_cap < bh.uncompressed_size) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
    }
    rc = zpse_block_decode(d, in + off, size, output, out_cap, &got);
    if (rc < 0) {
        return rc;
    }
    if (out_len_out != 0) {
        *out_len_out = got;
    }
    return ZPSE_OK;
}

int zpse_frame_seek(ZpseDecoder *d, const uint8_t *in, size_t in_len,
                    uint32_t block_index, uint8_t *output, uint32_t out_cap,
                    uint32_t *out_len_out)
{
    ZpseFrameHeader fh;
    const uint8_t *entries = 0;
    uint32_t count = 0u;
    uint32_t off;
    uint32_t size;
    int rc;

    if (d == 0 || in == 0 || output == 0) {
        return ZPSE_BAD_ARG;
    }
    rc = zpse_frame_index_resolve(in, in_len, &fh, &entries, &count);
    if (rc != ZPSE_OK) {
        return rc;
    }
    /* A frame without an index is an unsupported seek, not a seek to a
     * nonexistent block: the one-shot API has always reported that distinctly
     * and callers rely on it. */
    if ((fh.frame_flags & ZPSE_F_FRAME_INDEX) == 0u) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (block_index >= count) {
        return ZPSE_BAD_ARG;
    }
    off = zpse_load_u32le(entries + (size_t)block_index *
                                       (size_t)ZPSE_INDEX_ENTRY_SIZE);
    size = zpse_load_u32le(entries + (size_t)block_index *
                                        (size_t)ZPSE_INDEX_ENTRY_SIZE + 4u);
    return zpse_block_at(d, in, in_len, off, size, block_index, output,
                         out_cap, out_len_out);
}

int zpse_frame_write_index(uint8_t *out, size_t pos, size_t out_cap,
                           uint32_t nominal_block_size, uint64_t content_size,
                           uint32_t block_count, size_t *out_len_out)
{
    ZpseFrameHeader fh;
    size_t need;
    if ((uint64_t)block_count * (uint64_t)ZPSE_INDEX_ENTRY_SIZE >
        (uint64_t)((size_t)-1 - (size_t)ZPSE_INDEX_HEADER_SIZE)) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    need = (size_t)ZPSE_INDEX_HEADER_SIZE +
           (size_t)ZPSE_INDEX_ENTRY_SIZE * (size_t)block_count;
    uint32_t at;
    uint32_t off;
    uint32_t k;
    uint32_t crc;
    int rc;

    if (out == 0 || out_len_out == 0 || block_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (pos < (size_t)ZPSE_FRAME_HEADER_SIZE || pos > (size_t)UINT32_MAX) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    if (out_cap < pos || need > out_cap - pos) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    at = (uint32_t)pos;
    off = (uint32_t)ZPSE_FRAME_HEADER_SIZE;
    for (k = 0; k < block_count; k++) {
        ZpseBlockHeader bh;
        uint8_t *ent = out + at + ZPSE_INDEX_HEADER_SIZE +
                       (size_t)k * (size_t)ZPSE_INDEX_ENTRY_SIZE;

        if ((size_t)off + (size_t)ZPSE_BLOCK_HEADER_SIZE_SHORT > pos) {
            return ZPSE_BAD_HEADER;
        }
        rc = zpse_block_header_parse(&bh, out + off,
                                     (uint32_t)(pos - (size_t)off));
        if (rc != ZPSE_OK) {
            return rc;
        }
        if (bh.compressed_block_size <
                (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT ||
            (size_t)off + (size_t)bh.compressed_block_size > pos ||
            bh.block_sequence_number != k) {
            return ZPSE_BAD_HEADER;
        }
        zpse_store_u32le(ent, off);
        zpse_store_u32le(ent + 4u, bh.compressed_block_size);
        off += bh.compressed_block_size;
    }
    if ((size_t)off != pos) {
        return ZPSE_BAD_HEADER;
    }
    crc = zpse_crc32c(out + at + (size_t)ZPSE_INDEX_HEADER_SIZE,
                      (size_t)ZPSE_INDEX_ENTRY_SIZE * (size_t)block_count);
    zpse_store_u32le(out + at, ZPSE_INDEX_MAGIC);
    zpse_store_u32le(out + at + 4u, block_count);
    zpse_store_u32le(out + at + 8u, crc);
    zpse_store_u32le(out + at + 12u, 0u);
    /* The header is written last, because the field it carries is where
     * the table it describes begins. */
    rc = zpse_frame_header_build(&fh, nominal_block_size, content_size, 1,
                                 block_count, 1, 1);
    if (rc != ZPSE_OK) {
        return rc;
    }
    fh.frame_index_offset = at;
    zpse_frame_reseal(&fh);
    rc = zpse_frame_header_write(&fh, out);
    if (rc != ZPSE_OK) {
        return rc;
    }
    *out_len_out = pos + need;
    return ZPSE_OK;
}

int zpse_frame_stream_index(uint8_t hdr_out[32], uint8_t *idx_out,
                            size_t idx_cap, uint32_t nominal_block_size,
                            uint64_t content_size, uint32_t block_count,
                            uint32_t index_offset, const uint32_t *offsets,
                            const uint32_t *sizes, size_t *idx_len_out)
{
    ZpseFrameHeader fh;
    size_t need;
    if ((uint64_t)block_count * (uint64_t)ZPSE_INDEX_ENTRY_SIZE >
        (uint64_t)((size_t)-1 - (size_t)ZPSE_INDEX_HEADER_SIZE)) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    need = (size_t)ZPSE_INDEX_HEADER_SIZE +
           (size_t)ZPSE_INDEX_ENTRY_SIZE * (size_t)block_count;
    uint32_t crc;
    uint32_t k;
    int rc;

    if (hdr_out == 0 || idx_out == 0 || offsets == 0 || sizes == 0 ||
        idx_len_out == 0 || block_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (need > idx_cap) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    if ((size_t)index_offset < (size_t)ZPSE_FRAME_HEADER_SIZE) {
        return ZPSE_BAD_HEADER;
    }
    /* The blocks have to start right after the header and follow each other
     * with no gap, and the table has to start right after the last of them:
     * that is what the sequential reader assumes when it walks the stream, and
     * a table that says otherwise describes a different frame. */
    if (offsets[0] != (uint32_t)ZPSE_FRAME_HEADER_SIZE) {
        return ZPSE_BAD_HEADER;
    }
    for (k = 0; k < block_count; k++) {
        uint32_t off = offsets[k];
        uint32_t sz = sizes[k];
        uint8_t *ent = idx_out + (size_t)ZPSE_INDEX_HEADER_SIZE +
                       (size_t)k * (size_t)ZPSE_INDEX_ENTRY_SIZE;

        if (sz < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        if ((uint64_t)off + (uint64_t)sz > (uint64_t)index_offset) {
            return ZPSE_BAD_HEADER;
        }
        if (k + 1u < block_count && offsets[k + 1u] != off + sz) {
            return ZPSE_BAD_HEADER;
        }
        zpse_store_u32le(ent, off);
        zpse_store_u32le(ent + 4u, sz);
    }
    if ((uint32_t)(offsets[block_count - 1u] + sizes[block_count - 1u]) !=
        index_offset) {
        return ZPSE_BAD_HEADER;
    }
    crc = zpse_crc32c(idx_out + (size_t)ZPSE_INDEX_HEADER_SIZE,
                      (size_t)ZPSE_INDEX_ENTRY_SIZE * (size_t)block_count);
    zpse_store_u32le(idx_out, ZPSE_INDEX_MAGIC);
    zpse_store_u32le(idx_out + 4u, block_count);
    zpse_store_u32le(idx_out + 8u, crc);
    zpse_store_u32le(idx_out + 12u, 0u);
    rc = zpse_frame_header_build(&fh, nominal_block_size, content_size, 1,
                                 block_count, 1, 1);
    if (rc != ZPSE_OK) {
        return rc;
    }
    fh.frame_index_offset = index_offset;
    zpse_frame_reseal(&fh);
    rc = zpse_frame_header_write(&fh, hdr_out);
    if (rc != ZPSE_OK) {
        return rc;
    }
    *idx_len_out = need;
    return ZPSE_OK;
}

int zpse_frame_encode(ZpseEncoder *e, const uint8_t *in, size_t n,
                      uint32_t nominal_block_size,
                      uint8_t *out, size_t out_cap, size_t *out_len_out)
{
    return zpse_frame_encode_ex(e, in, n, nominal_block_size, out, out_cap,
                                out_len_out, 0);
}

int zpse_frame_encode_ex(ZpseEncoder *e, const uint8_t *in, size_t n,
                         uint32_t nominal_block_size,
                         uint8_t *out, size_t out_cap, size_t *out_len_out,
                         int with_index)
{
    ZpseFrameHeader fh;
    size_t nblocks;
    size_t pos;
    size_t i;
    int rc;

    if (e == 0 || out == 0 || out_len_out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0) {
        return ZPSE_BAD_ARG;
    }
    if (in == 0) {
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
    *out_len_out = 0;
    nblocks = n / (size_t)nominal_block_size +
        ((n % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks == 0 || nblocks > (size_t)UINT32_MAX) {
        return ZPSE_BAD_ARG;
    }
    if (out_cap < ZPSE_FRAME_HEADER_SIZE) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    rc = zpse_frame_header_build(&fh, nominal_block_size, (uint64_t)n, 1,
                                 (uint32_t)nblocks, 1, with_index);
    if (rc != ZPSE_OK) {
        return rc;
    }
    rc = zpse_frame_header_write(&fh, out);
    if (rc != ZPSE_OK) {
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
            return ZPSE_BAD_HEADER;
        }
        chunk = n - off;
        if (chunk > (size_t)nominal_block_size) {
            chunk = (size_t)nominal_block_size;
        }
        /* Invariant: pos <= out_cap, so the subtraction cannot wrap. */
        rem = out_cap - pos;
        if (rem < (size_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        cap = rem > (size_t)UINT32_MAX ? UINT32_MAX : (uint32_t)rem;
        wrc = zpse_block_encode(e, in + off, (uint32_t)chunk, (uint32_t)i,
                               ((i + 1u) == nblocks) ? 1 : 0,
                               out + pos, cap);
        if (wrc < 0) {
            return wrc;
        }
        if (wrc == 0) {
            return ZPSE_BAD_HEADER;
        }
        if ((size_t)wrc > rem) {
            return ZPSE_BAD_HEADER;
        }
        pos += (size_t)wrc;
    }
    /* The index rides at the end, because a block's size is not known
     * until it is written. Its entries are not stored while encoding:
     * each block already carries its own compressed size in its header,
     * so the table is a walk over the headers -- 32 bytes per block read
     * back, rather than an array of block offsets the encoder would have
     * to allocate. A parallel encoder finishes the same way, which is why
     * the table is a function of its own. */
    if (with_index) {
        rc = zpse_frame_write_index(out, pos, out_cap, nominal_block_size,
                                    (uint64_t)n, (uint32_t)nblocks, &pos);
        if (rc != ZPSE_OK) {
            return rc;
        }
    }
    *out_len_out = pos;
    if (pos > (size_t)INT_MAX) {
        return ZPSE_OK;
    }
    return (int)pos;
}
