#ifndef ZPSE_FRAME_H
#define ZPSE_FRAME_H

/* Frame-level helpers. Block framing loops live here;
 * per-block work delegates to block.h. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"
#include "types.h"

int zpse_frame_header_build(ZpseFrameHeader *h, uint32_t nominal_block_size,
                            uint64_t content_size, int content_valid,
                            uint32_t block_count, int count_valid,
                            int with_index);
int zpse_frame_header_write(const ZpseFrameHeader *h, uint8_t out[32]);
int zpse_frame_header_read(ZpseFrameHeader *h, const uint8_t in[32]);

/* Streaming decode: decode all blocks in [in, in+in_len) into output.
 * Returns bytes written or negative status. */
int zpse_frame_decode(ZpseDecoder *d, const uint8_t *in, size_t in_len,
                      uint8_t *output, size_t out_cap, size_t *out_len_out);

/* Streaming encode: split input into <= nominal_block_size blocks. */
int zpse_frame_encode(ZpseEncoder *e, const uint8_t *in, size_t n,
                      uint32_t nominal_block_size,
                      uint8_t *out, size_t out_cap, size_t *out_len_out);

/* Same, with the FrameIndex appended when with_index is nonzero. The
 * index costs 16 + 8 per block bytes and one cheap walk over the block
 * headers (their sizes are already in them), and it is what makes
 * zpse_frame_seek O(1) instead of a scan. */
int zpse_frame_encode_ex(ZpseEncoder *e, const uint8_t *in, size_t n,
                         uint32_t nominal_block_size,
                         uint8_t *out, size_t out_cap, size_t *out_len_out,
                         int with_index);

/* Number of entries in a frame's index, 0 when it carries none. A frame
 * whose index is present but does not validate returns a negative status
 * rather than a count. */
int zpse_frame_index_count(const uint8_t *in, size_t in_len,
                           uint32_t *count_out);

/* Validate a FrameIndex a sequential reader has already read out of the file
 * at fh->frame_index_offset, against the blocks that reader walked: `count`
 * entries, with offsets[k] and sizes[k] the file offset and compressed size of
 * block k. The table has to describe exactly the stream that was decoded, and
 * the blocks have to end where the table begins -- the same guarantee
 * zpse_frame_decode makes before it returns, for a decoder that never holds
 * the frame in memory. `idx` needs ZPSE_INDEX_HEADER_SIZE + 8 per entry. */
int zpse_frame_index_validate(const ZpseFrameHeader *fh, const uint8_t *idx,
                              size_t idx_len, uint32_t count,
                              const uint32_t *offsets, const uint32_t *sizes);

/* Decode exactly one block of a frame, using its FrameIndex. `output`
 * must hold that block's uncompressed size (at most 64 KiB); the frame
 * header's nominal_block_size bounds it, and the block's own header is
 * checked against the index entry. Random access is the reason the format
 * keeps every block independent; this is the entry point that uses it.
 *
 * Returns ZPSE_OK with *out_len_out set, or a negative status. A frame
 * without an index, a block index past the end, and an entry that does
 * not describe a block are all refusals, not scans. */
int zpse_frame_seek(ZpseDecoder *d, const uint8_t *in, size_t in_len,
                    uint32_t block_index, uint8_t *output, uint32_t out_cap,
                    uint32_t *out_len_out);

/* Finish a frame whose blocks are already written and end at `pos`: write
 * the header (sealed, carrying the index position) and append the
 * FrameIndex. The table is built by walking the block headers -- each
 * carries its own compressed size -- so the encoder never has to remember
 * an offset, and every entry is checked against the block it names.
 *
 * This is the shared tail of the serial and the multi-core encoder: both
 * have the blocks, and neither has the table until it is walked. `out`
 * needs ZPSE_INDEX_HEADER_SIZE + 8 per block free at `pos`. Returns ZPSE_OK
 * with *out_len_out = pos + table, or a negative status. */
int zpse_frame_write_index(uint8_t *out, size_t pos, size_t out_cap,
                           uint32_t nominal_block_size, uint64_t content_size,
                           uint32_t block_count, size_t *out_len_out);

/* The same tail for a writer that never holds the frame: blocks already
 * streamed to a file, and the table recorded as they went. `offsets` and
 * `sizes` hold block_count entries, `index_offset` is where the table will
 * be written (the byte just past the last block), and the two outputs are
 * the header to patch over the one already in the file and the table to
 * append. The entries are checked to be contiguous and inside the blocks
 * that precede the table, so a table built from the wrong offsets is
 * refused here rather than by the reader that seeks by it. */
int zpse_frame_stream_index(uint8_t hdr_out[32], uint8_t *idx_out,
                            size_t idx_cap, uint32_t nominal_block_size,
                            uint64_t content_size, uint32_t block_count,
                            uint32_t index_offset, const uint32_t *offsets,
                            const uint32_t *sizes, size_t *idx_len_out);

#endif /* ZPSE_FRAME_H */
