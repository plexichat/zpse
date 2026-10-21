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

/* Decode exactly one block of a frame, using its FrameIndex. `output`
 * must hold that block's uncompressed size (at most 64 KiB); the frame
 * header's nominal_block_size bounds it, and the block's own header is
 * checked against the index entry. Random access is the reason the format
 * keeps every block independent; this is the entry point that uses it.
 *
 * Returns PSE_OK with *out_len_out set, or a negative status. A frame
 * without an index, a block index past the end, and an entry that does
 * not describe a block are all refusals, not scans. */
int zpse_frame_seek(ZpseDecoder *d, const uint8_t *in, size_t in_len,
                    uint32_t block_index, uint8_t *output, uint32_t out_cap,
                    uint32_t *out_len_out);

#endif /* ZPSE_FRAME_H */
