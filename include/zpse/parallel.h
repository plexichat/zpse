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
 * The blocks are the serial encoder's own at every level. From level 5 up the
 * encoder carries a cross-block history, and each worker carries its own, but
 * that history only chooses which survivor is attempted first: the winner, the
 * flat-block cap and the tied-field cut are all decided from prices the block
 * computes for itself, so a block is a pure function of its own bytes and the
 * worker count cannot change the stream. That used to be untrue -- with the
 * history hint promoted to the winner, the same 20 MB at level 6 packed to
 * 15,098,769 bytes on one worker and 15,101,233 on four -- and it is the one
 * property here that is checked by comparing bytes rather than by argument.
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

/* Called for each finished block, in block order, before the next one is
 * handed over. `seq` is the block's index in the frame. Return 0 to carry
 * on, or a negative status to abort the encode (it is returned as-is). */
typedef int (*ZpseBlockEmit)(void *ctx, uint32_t seq, const uint8_t *block,
                             size_t len);

/* The same encode, streamed: one slab of a frame at a time, with every
 * finished block handed to `emit` in order instead of being written into a
 * frame-sized buffer.
 *
 * `in` holds the bytes of blocks `first_block` .. `first_block + count - 1`,
 * where count covers `in_len` bytes at `nominal_block_size` and every block
 * but the last of the slab is a full one. `is_final_slab` marks the call that
 * owns the frame's last block, which is the one that carries the last-block
 * flag; a caller that has the whole input passes 1 with everything in one
 * call and gets the buffered encoder's stream, block for block.
 *
 * The workers write into `slot_mem`, which holds one (nominal_block_size + 32)
 * byte slot per worker, so only that much of the output is ever resident;
 * `ready` and `lens` are caller-owned `int` and `uint32_t` scratch of at least
 * that many entries, and `ready` is cleared here. The library still never
 * allocates. Worker `w` owns slot `w` and encodes blocks `w`, `w + n`,
 * `w + 2n`, ... in that order, so a slot has exactly one writer and the
 * writer thread frees a slot only after emitting it. Emitting blocks in order
 * is what makes the stream writable while the last blocks are still being
 * encoded; the static assignment is what makes the handoff provable, and an
 * earlier version that claimed blocks from a shared counter instead did hand
 * one block's bytes over under another block's length.
 *
 * `threads` above is the ceiling: the worker count is the smallest of it, the
 * arena count and the number of slots in `slot_mem`, and a slab shorter than
 * four blocks per worker is encoded serially. Returns PSE_OK, or a negative
 * status. The stream is block for block the one zpse_compress_frame_parallel()
 * writes at the same thread count, and that call's stream at one worker is the
 * serial one, so a caller may switch between the three without changing a
 * byte. */
int zpse_compress_frame_stream(const uint8_t *in, size_t in_len,
                               uint32_t nominal_block_size,
                               uint32_t enc_flags, unsigned threads,
                               uint32_t first_block, int is_final_slab,
                               void *const enc_arenas[], unsigned arena_count,
                               uint8_t *slot_mem, size_t slot_cap, int *ready,
                               uint32_t *lens, ZpseBlockEmit emit,
                               void *emit_ctx);

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
 * (ZPSE_DEC_ARENA_SIZE, 64-B aligned). A worker has no other state and no
 * buffer of its own: it decodes block k into `output + k * nominal` and
 * writes nothing outside it, so two workers cannot touch the same byte. That
 * bound is a property of the block decoder, not of this call -- the LZ
 * decoder's one-shot command paths used to write up to sixteen bytes past
 * the match they copied and rely on the next command to reclaim them, which
 * across two workers is a race between block k's tail and block k+1's head.
 * It now tests the room in front of every store, and this path decodes in
 * place instead of through a per-worker staging block.
 *
 * Returns PSE_OK with *out_len_out, or a negative status. */
int zpse_decompress_frame_parallel(const uint8_t *in, size_t in_len,
                                   uint8_t *output, size_t out_cap,
                                   size_t *out_len_out, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_PARALLEL_H */
