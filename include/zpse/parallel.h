#ifndef ZPSE_PARALLEL_H
#define ZPSE_PARALLEL_H

/* Multi-core frame encode.
 *
 * A frame's blocks are independent -- a block may not reference another,
 * which is what makes the format seekable -- so the only thing that forces
 * a frame to be encoded in order is the output layout: block k's offset
 * depends on the sizes of blocks 0..k-1. That layout is computed, not
 * searched, so it does not need the encoder: each worker encodes its block
 * into a reserved slot, and one serial compaction pass moves the blocks
 * into their final positions and writes the index. Encode is the slow side
 * of this codec (1-2 GB/s against 13 GB/s decode), so that is where the
 * cores go.
 *
 * The result is byte-identical to the serial call -- same blocks, same
 * header, same index -- because every block is encoded by the same encoder
 * with the same inputs. Threading changes when the work happens, never what
 * is written.
 *
 * Threading is compiled in only when ZPSE_MT is defined (POSIX: add
 * -pthread). Without it this is the serial encoder behind a different
 * name, so a caller can ship one build. The library still never allocates:
 * the caller owns the arenas, one per worker.
 */

#include <stddef.h>
#include <stdint.h>

#include "arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Encode `in` as one frame using up to `threads` workers.
 *
 * `enc_arenas` holds `arena_count` caller-owned 448 KiB encode arenas
 * (ZPSE_ENC_ARENA_SIZE, 64-B aligned); the number of workers is
 * min(threads, arena_count, blocks) and `threads` 0 or 1 means one worker.
 * With threads > 1 the caller's buffer must hold the reserved layout --
 * ZPSE_FRAME_HEADER_SIZE + blocks * (nominal_block_size + 32) bytes, plus
 * the index when `with_index` is set -- because each block is encoded into
 * a slot of that size before compaction; zpse_parallel_bound() computes it.
 * A smaller `out_cap` is not an error: the call falls back to the serial
 * encoder, which writes the same bytes in less room.
 *
 * `with_index` appends the FrameIndex, exactly as
 * zpse_compress_frame_indexed() does. Returns PSE_OK and sets
 * *out_len_out, or a negative status. */
int zpse_compress_frame_parallel(const uint8_t *in, size_t n,
                                 uint8_t *out, size_t out_cap,
                                 size_t *out_len_out,
                                 uint32_t nominal_block_size,
                                 uint32_t enc_flags, int with_index,
                                 unsigned threads,
                                 void *const enc_arenas[],
                                 unsigned arena_count);

/* The out_cap zpse_compress_frame_parallel() needs to run its workers: the
 * reserved slot layout plus the index, when `with_index` is set. A buffer
 * this size is enough for any encoder level, because a block never grows
 * past its input plus its 32-byte header. */
size_t zpse_parallel_bound(size_t n, uint32_t nominal_block_size,
                           int with_index);

/* Decode a frame's blocks on several workers.
 *
 * The read side is the same argument as the write side, one level lower:
 * blocks are independent, so block k decodes on its own -- but the decoder
 * can only find block k through the FrameIndex, because a block header
 * carries its own size and not its offset. So this needs a frame written
 * with the index (zpse_compress_frame_indexed or `with_index` above); a
 * frame without one falls back to the serial decoder, which is the same
 * bytes at one core. Block k lands at k * nominal_block_size in `output`,
 * so the decoded frame is the sequential one, byte for byte.
 *
 * `dec_arenas` holds `arena_count` caller-owned 320 KiB decode arenas
 * (ZPSE_DEC_ARENA_SIZE, 64-B aligned), and `staging` holds
 * threads * (nominal_block_size + 64) bytes, also caller-owned. The staging
 * block is not laziness: the LZ decoder's one-shot command paths write up to
 * sixteen bytes past the match they copy, on the argument that the next
 * command overwrites them. Inside one sequential buffer that is true and
 * costs nothing, but block k's tail is block k+1's head when both are decoded
 * in parallel, so each worker decodes into its own staging block (which has
 * the slack) and copies the bytes it produced into place. The copy is one per
 * block at memory speed against a block decode several times slower, and it
 * is what makes the parallel result the sequential result instead of a race.
 *
 * Returns PSE_OK with *out_len_out, or a negative status. */
int zpse_decompress_frame_parallel(const uint8_t *in, size_t in_len,
                                   uint8_t *output, size_t out_cap,
                                   size_t *out_len_out, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count,
                                   uint8_t *staging);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_PARALLEL_H */
