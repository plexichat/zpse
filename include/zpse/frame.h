#ifndef ZPSE_FRAME_H
#define ZPSE_FRAME_H

/* Frame-level helpers. Block framing loops live here;
 * per-block work delegates to block.h. FrameIndex is opaque in v1. */

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

#endif /* ZPSE_FRAME_H */
