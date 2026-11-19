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

/* Control-slot stride for the streamed encode. `ready` and `lens` are indexed
 * one entry per 64 bytes, so two slots never share a cache line: a flag the
 * writer polls and a flag the next worker releases must not invalidate each
 * other, and with six workers on one line they did. */
#define ZPSE_MT_CTRL_STRIDE 16u
/* Most ring slots per worker the streamed encode will use. Four is where the
 * writer's own work stops being on the workers' critical path; deeper only
 * holds finished blocks longer without feeding anyone. */
#define ZPSE_MT_MAX_DEPTH 4u

/* Encode `in` as one frame using up to `threads` workers.
 *
 * `enc_arenas` holds `arena_count` caller-owned 464 KiB encode arenas
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
 * zpse_compress_frame_indexed() does. Returns ZPSE_OK and sets
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
 * The workers write into `slot_mem`, which holds (nominal_block_size + 32)
 * byte slots at a stride of that size, up to ZPSE_MT_MAX_DEPTH of them per
 * worker; `ready` and `lens` are caller-owned `int` and `uint32_t` scratch of
 * one entry per slot per ZPSE_MT_CTRL_STRIDE, and `ready` is cleared here.
 * The library still never allocates. Worker `w` owns slots `w`,
 * `w + workers`, ... `w + (depth - 1) * workers` and encodes blocks `w`,
 * `w + n`, `w + 2n`, ... into them in that order, so a slot has exactly one
 * writer and the writer thread frees a slot only after emitting it.
 * Emitting blocks in order is what makes the stream writable while the last
 * blocks are still being encoded; the static assignment is what makes the
 * handoff provable, and an earlier version that claimed blocks from a shared
 * counter instead did hand one block's bytes over under another block's
 * length.
 *
 * The depth is the pipeline's slack. With one slot per worker a worker can
 * be at most one block ahead of the writer, so every worker blocks on an
 * emit it has nothing to do with; with four it can run a block's worth of
 * work ahead per slot and the writer catches up in a burst. The depth is
 * whatever `slot_cap` holds, clamped to ZPSE_MT_MAX_DEPTH, so a caller that
 * passes one slot per worker gets the shallow ring and the same stream.
 *
 * Control entries are one per 64 bytes rather than one per slot: the whole
 * eightslot ring otherwise shares a single cache line, so every worker's
 * release invalidates every other worker's next poll. `slot_mem` holds at
 * least one slot per worker (or that many fewer workers are used).
 *
 * `threads` above is the ceiling: the worker count is the smallest of it, the
 * arena count and the number of slots in `slot_mem`, and a slab shorter than
 * four blocks per worker is encoded serially. Returns ZPSE_OK, or a negative
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
 * `dec_arenas` holds `arena_count` caller-owned 384 KiB decode arenas
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
 * Returns ZPSE_OK with *out_len_out, or a negative status.
 *
 * FLAT7 frames are never emitted by the parallel/stream APIs, and parallel
 * decode declines them with ZPSE_UNSUPPORTED_FEATURE (they carry no index). */
int zpse_decompress_frame_parallel(const uint8_t *in, size_t in_len,
                                   uint8_t *output, size_t out_cap,
                                   size_t *out_len_out, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count);

/* Decode a batch of independent blocks on several workers.
 *
 * This is the read side a streaming caller needs. The frame call above needs
 * the whole frame resident because a block's offset lives only in the
 * FrameIndex; a reader that is walking a file has just read the blocks it is
 * about to decode, so it can hand them over as a batch instead. `count`
 * blocks sit at `in + offsets[i]` with `lens[i]` bytes each, and block i
 * decodes into `out + i * stride`, whose length is written to `out_lens[i]`. A
 * block writes exactly its own bytes and nothing else, so two workers cannot
 * touch the same byte and the result is the sequential one, block for block.
 *
 * `stride` is the nominal block size; a short final block is fine, and any
 * block whose decoded length exceeds `stride` is refused rather than
 * truncated. The caller owns the batch buffers and reuses them, so nothing
 * here allocates and memory is bounded by the batch, not the frame.
 *
 * A batch shorter than four blocks per worker runs serially on the calling
 * thread: a block decodes in a few microseconds and a thread start costs more
 * than the blocks it would take. Without ZPSE_MT this is the serial loop.
 * Returns ZPSE_OK, or the first negative status any block reported. */
int zpse_decompress_batch_parallel(const uint8_t *in,
                                   const uint32_t *offsets,
                                   const uint32_t *lens, uint32_t count,
                                   uint8_t *out, uint32_t stride,
                                   uint32_t *out_lens, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count);

/* ---- persistent batch pool -------------------------------------------
 *
 * A streaming caller decodes one small batch after another. Sending each
 * batch through zpse_decompress_batch_parallel() creates and joins threads
 * every time -- a page stream of a few thousand batches pays that thousands
 * of times. These calls keep the workers alive between batches: the caller
 * owns one scratch buffer and one decode arena per worker, and the workers
 * sleep on a condition variable until the next batch is published. The
 * library still never allocates; threads are capped at arena_count and at
 * ZPSE_MT_MAX_THREADS. Without ZPSE_MT the pool is a serial decoder behind
 * the same API, so a caller can ship one build.
 *
 * ZPSE_POOL_SCRATCH_SIZE(threads) is the exact byte count of scratch needed
 * for `threads` participants (the calling thread plus threads-1 long-lived
 * workers). The scratch must be 64-byte aligned and stay alive until
 * zpse_batch_pool_free(). `threads` above is capped internally, so sizing
 * the scratch for the threads you intend to use is enough.
 */
size_t zpse_batch_pool_scratch_size(unsigned threads);
#define ZPSE_POOL_SCRATCH_SIZE(threads) zpse_batch_pool_scratch_size(threads)

/* Opaque pool; lives entirely inside the caller's scratch. */
typedef struct ZpseBatchPool ZpseBatchPool;

/* Carve a pool out of `scratch` (>= ZPSE_POOL_SCRATCH_SIZE(threads) bytes,
 * 64-B aligned), init one persistent decoder per worker from
 * dec_arenas[0..arena_count-1], and start the workers. *pool_out points into
 * `scratch`. Returns ZPSE_OK or a negative status. */
int zpse_batch_pool_init(ZpseBatchPool **pool_out, void *scratch,
                         size_t scratch_len, unsigned threads,
                         void *const dec_arenas[], unsigned arena_count);

/* Decode one batch. Arguments match zpse_decompress_batch_parallel() minus
 * the threading/arena ones, which were fixed at init; may be called any
 * number of times, reusing the same workers. Returns ZPSE_OK or the first
 * negative status any block reported. */
int zpse_batch_pool_run(ZpseBatchPool *pool, const uint8_t *in,
                        const uint32_t *offsets, const uint32_t *lens,
                        uint32_t count, uint8_t *out, uint32_t stride,
                        uint32_t *out_lens);

/* Wake and join the workers, destroy the sync objects, and leave `scratch`
 * reusable. Call once for every successful init. */
void zpse_batch_pool_free(ZpseBatchPool *pool);

/* ---- persistent encode pool -----------------------------------------
 *
 * The streamed encode above starts and joins threads for every slab. A caller
 * walking a page stream sends a few hundred slabs through it and pays that
 * start/join cost each time. These calls keep the same workers alive between
 * slabs: the caller owns one scratch buffer (which holds the pool and one
 * encoder per participant) and one encode arena per worker, and the workers
 * sleep on a condition variable until the next slab is published. The library
 * still never allocates; threads are capped at arena_count and at
 * ZPSE_MT_MAX_THREADS (internal). Without ZPSE_MT the pool is the serial
 * stream encoder behind the same API, so a caller can ship one build.
 *
 * ZPSE_ENCODE_POOL_SCRATCH_SIZE(threads) is the exact byte count of scratch
 * needed for `threads` participants (the calling thread plus threads-1
 * long-lived workers). The scratch must be 64-byte aligned and stay alive
 * until zpse_encode_pool_free(). `threads` is capped internally, so sizing
 * for the threads you intend to use is enough.
 */
size_t zpse_encode_pool_scratch_size(unsigned threads);
#define ZPSE_ENCODE_POOL_SCRATCH_SIZE(threads) \
    zpse_encode_pool_scratch_size(threads)

/* Opaque pool; lives entirely inside the caller's scratch. */
typedef struct ZpseEncodePool ZpseEncodePool;

/* Carve a pool out of `scratch` (>= ZPSE_ENCODE_POOL_SCRATCH_SIZE(threads)
 * bytes, 64-B aligned), init one persistent encoder per participant from
 * enc_arenas[0..arena_count-1], and start the workers. *pool_out points into
 * `scratch`. Returns ZPSE_OK or a negative status. */
int zpse_encode_pool_init(ZpseEncodePool **pool_out, void *scratch,
                          size_t scratch_len, unsigned threads,
                          void *const enc_arenas[], unsigned arena_count);

/* Encode one slab of a frame. The arguments are those of
 * zpse_compress_frame_stream() minus the threading/arena ones, which were
 * fixed at init: `in` holds blocks `first_block`... at `nominal_block_size`
 * (every block but the last a full one), `is_final_slab` marks the slab that
 * owns the frame's last block and carries the last-block flag, the workers
 * write into `slot_mem` (at least one slot per participant, up to
 * ZPSE_MT_MAX_DEPTH per participant) and `ready`/`lens` are caller-owned
 * scratch of one entry per slot per ZPSE_MT_CTRL_STRIDE, cleared here.
 * Finished blocks are handed to `emit` in block order. May be called any
 * number of times, reusing the same workers. The stream is block for block
 * the one zpse_compress_frame_stream() writes at the same thread count --
 * which is the serial stream at one worker -- so a caller may move between
 * them without changing a byte. Returns ZPSE_OK, a negative status, or the
 * negative status an `emit` callback returned. */
int zpse_encode_pool_run(ZpseEncodePool *pool, const uint8_t *in,
                         size_t in_len, uint32_t nominal_block_size,
                         uint32_t enc_flags, uint32_t first_block,
                         int is_final_slab, uint8_t *slot_mem, size_t slot_cap,
                         int *ready, uint32_t *lens, ZpseBlockEmit emit,
                         void *emit_ctx);

/* Wake and join the workers, destroy the sync objects, and leave `scratch`
 * reusable. Call once for every successful init. */
void zpse_encode_pool_free(ZpseEncodePool *pool);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_PARALLEL_H */
