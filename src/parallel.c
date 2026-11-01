/* Multi-core frame encode: blocks in parallel, layout in one serial pass.
 *
 * See include/zpse/parallel.h for the contract. The shape here is the
 * consequence of the format: a block's compressed size is not known before
 * it is encoded, so blocks cannot be written back to back directly. They
 * are written instead into fixed-size slots -- a block never exceeds its
 * input plus a 32-byte header, so block + 32 is a provable slot size -- and
 * one pass over the headers moves them together and builds the index. The
 * compaction is a memmove per block at memory speed; the encode is at
 * 1-2 GB/s, so the serial part is not what the wall clock is made of.
 */
#include "zpse/parallel.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "zpse/arena.h"
#include "zpse/block.h"
#include "zpse/constants.h"
#include "zpse/frame.h"
#include "zpse/status.h"
#include "zpse/types.h"

#if defined(ZPSE_MT)
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif
/* Worker count ceiling: beyond this the compaction pass, not the encode,
 * is the run. It also sizes the thread-handle array below. */
#define ZPSE_MT_MAX_THREADS 64u
#endif

/* A ring slot has three states, and all three are needed: a worker may only
 * fill a slot that has been emptied, which is not the same as one that no
 * writer is looking at. With only "free" and "ready" a worker could refill
 * the slot that holds the block the writer is about to emit -- the block is
 * older, so the counter has already handed out its successor -- and the two
 * would write over each other. */
#define ZPSE_MT_SLOT_EMPTY  0
#define ZPSE_MT_SLOT_FILLING 1
#define ZPSE_MT_SLOT_READY  2

/* A block's reserved slot: the block itself plus its 32-byte header, which
 * is what the raw form costs and therefore the most any form can cost. */
#define ZPSE_MT_SLOT_OVERHEAD ((size_t)ZPSE_BLOCK_HEADER_SIZE)

size_t zpse_parallel_bound(size_t n, uint32_t nominal_block_size,
                           int with_index)
{
    size_t blocks;

    if (n == 0u || nominal_block_size == 0u) {
        return 0u;
    }
    blocks = n / (size_t)nominal_block_size +
             ((n % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    return (size_t)ZPSE_FRAME_HEADER_SIZE +
           blocks * ((size_t)nominal_block_size + ZPSE_MT_SLOT_OVERHEAD) +
           (with_index ? (size_t)ZPSE_INDEX_HEADER_SIZE +
                             (size_t)ZPSE_INDEX_ENTRY_SIZE * blocks
                       : 0u);
}

/* One serial worker: the plain frame encoder. This is both the fallback and
 * the reference the threaded path is checked against. */
static int zpse_mt_serial(const uint8_t *in, size_t n, uint8_t *out,
                          size_t out_cap, size_t *out_len_out,
                          uint32_t block, uint32_t enc_flags, int with_index,
                          void *const arenas[])
{
    ZpseEncoder enc;
    int rc;

    rc = zpse_encoder_init(&enc, arenas[0], (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_frame_encode_ex(&enc, in, n, block, out, out_cap, out_len_out,
                              with_index);
    return rc < 0 ? rc : PSE_OK;
}

#if defined(ZPSE_MT)

typedef struct ZpseMtJob {
    const uint8_t *in;
    size_t n;
    uint8_t *out;      /* frame base: the header goes here */
    uint8_t *slots;    /* where the blocks are written, out + header */
    size_t out_cap;
    size_t nblocks;
    size_t stride;      /* reserved slot size per block */
    uint32_t block;     /* nominal block size */
    uint32_t enc_flags;
    /* The next block index no worker has claimed. A worker that finds it
     * past the end is finished; the counter is only ever incremented, so
     * no worker can be handed another's block. Volatile because on the
     * Win32 path it is the operand of an interlocked increment. */
    volatile long next;
    /* First failure, advisory: it stops the other workers early, and the
     * caller reads it after every thread has joined. */
    volatile int failed;
} ZpseMtJob;

typedef struct ZpseMtArg {
    ZpseMtJob *job;
    void *arena;
} ZpseMtArg;

/* The next unclaimed block index, or SIZE_MAX when there is none left.
 * One counter per job, incremented by whoever gets there first: no block can
 * be handed to two workers and no worker needs a lock. */
static size_t zpse_mt_claim(volatile long *next, size_t nblocks)
{
    long v;

#if defined(_WIN32)
    v = InterlockedIncrement((volatile LONG *)next) - 1;
#else
    v = __atomic_fetch_add(next, 1, __ATOMIC_RELAXED);
#endif
    if (v < 0 || (size_t)v >= nblocks) {
        return SIZE_MAX;
    }
    return (size_t)v;
}

static void zpse_mt_fail(ZpseMtJob *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : PSE_BAD_HEADER;
    }
}

static void zpse_mt_worker(ZpseMtJob *j, void *arena)
{
    ZpseEncoder enc;
    int rc;

    rc = zpse_encoder_init(&enc, arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        zpse_mt_fail(j, rc);
        return;
    }
    enc.enc_flags = j->enc_flags;
    for (;;) {
        size_t i;
        size_t off;
        size_t chunk;
        int wrc;

        if (j->failed != 0) {
            return;
        }
        i = zpse_mt_claim(&j->next, j->nblocks);
        if (i == SIZE_MAX) {
            return;
        }
        off = i * (size_t)j->block;
        if (off >= j->n) {
            zpse_mt_fail(j, PSE_BAD_HEADER);
            return;
        }
        chunk = j->n - off;
        if (chunk > (size_t)j->block) {
            chunk = (size_t)j->block;
        }
        /* Sequence number and last-block flag are the block's position in
         * the frame, not its arrival order, so a worker writes into the
         * slot the serial encoder would have written into. */
        wrc = zpse_block_encode(&enc, j->in + off, (uint32_t)chunk,
                                (uint32_t)i, (i + 1u) == j->nblocks,
                                j->slots + i * j->stride,
                                (uint32_t)j->stride);
        if (wrc <= 0 || (size_t)wrc > j->stride) {
            zpse_mt_fail(j, wrc);
            return;
        }
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_mt_entry(LPVOID p)
{
    ZpseMtArg *a = (ZpseMtArg *)p;

    zpse_mt_worker(a->job, a->arena);
    return 0;
}
#else
static void *zpse_mt_entry(void *p)
{
    ZpseMtArg *a = (ZpseMtArg *)p;

    zpse_mt_worker(a->job, a->arena);
    return 0;
}
#endif

/* Move the blocks from their slots to their final positions and finish the
 * frame: header, blocks back to back, then the index. */
static int zpse_mt_close(ZpseMtJob *j, int with_index, size_t *out_len_out)
{
    ZpseFrameHeader fh;
    size_t pos = (size_t)ZPSE_FRAME_HEADER_SIZE;
    size_t i;
    int rc;

    for (i = 0; i < j->nblocks; i++) {
        ZpseBlockHeader bh;
        uint8_t *src = j->slots + i * j->stride;

        rc = zpse_block_header_parse(&bh, src, (uint32_t)j->stride);
        if (rc != PSE_OK) {
            return rc;
        }
        if (bh.compressed_block_size < (uint32_t)ZPSE_BLOCK_HEADER_SIZE ||
            (size_t)bh.compressed_block_size > j->stride ||
            bh.block_sequence_number != (uint32_t)i) {
            return PSE_BAD_HEADER;
        }
        /* Slots start at the header offset and pos starts there too, so
         * pos <= header + i * stride throughout and writing block i's
         * `size <= stride` bytes ends at or before slot i+1 begins. A
         * forward pass therefore cannot overwrite a block it has not read
         * yet, which is the whole reason a block's slot is block + 32. */
        if (j->out + pos != src) {
            memmove(j->out + pos, src, (size_t)bh.compressed_block_size);
        }
        pos += (size_t)bh.compressed_block_size;
    }
    rc = zpse_frame_header_build(&fh, j->block, (uint64_t)j->n, 1,
                                 (uint32_t)j->nblocks, 1, with_index);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_frame_header_write(&fh, j->out);
    if (rc != PSE_OK) {
        return rc;
    }
    if (with_index) {
        rc = zpse_frame_write_index(j->out, pos, j->out_cap, j->block,
                                    (uint64_t)j->n, (uint32_t)j->nblocks,
                                    &pos);
        if (rc != PSE_OK) {
            return rc;
        }
    }
    *out_len_out = pos;
    return PSE_OK;
}

/* ---- streamed encode -----------------------------------------------
 *
 * The buffered worker path reserves a slot for every block of the frame in
 * the caller's buffer and compacts afterwards, which through a file means a
 * frame-sized buffer to fill and then walk -- more memory than the input, and
 * page faults for all of it. Nothing about the format needs it: a block is
 * independent, and the writer only needs them in order, not at once.
 *
 * So the workers write into a ring of a few slots, the calling thread emits
 * the blocks in order as they become ready, and the slot that has just been
 * written is released for the next block. A worker that finds its slot still
 * held waits for the writer, which is always making progress on an earlier
 * block, so no worker waits on the writer to wait on it.
 *
 * One encoder per worker, as in the buffered path, and the same block
 * content: the history an encoder carries chooses which trial runs first and
 * cannot change a byte (test_parallel's -DZPSE_FORCE_TRIAL_SWAP build holds
 * that down), so a slab boundary is invisible in the stream. */

typedef struct ZpseMtStream {
    const uint8_t *in;
    size_t n;
    size_t nblocks;
    size_t stride;
    uint32_t block;
    uint32_t enc_flags;
    uint32_t first_block;
    int final_slab;
    unsigned ring;
    uint8_t *slots;
    int *ready;
    uint32_t *lens;
    volatile long next;
    volatile int failed;
} ZpseMtStream;

typedef struct ZpseMtStreamArg {
    ZpseMtStream *job;
    void *arena;
} ZpseMtStreamArg;

/* A release/acquire pair over the slot state. The block's bytes and length
 * are written before the state says ready, and the writer reads the state
 * before it reads either. On x86 a volatile access plus a compiler barrier is
 * enough; the GCC builtins are used where they exist because they also make
 * the ordering explicit for a reader of this file. */
#if defined(__GNUC__)
#define ZPSE_MT_FLAG_GET(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define ZPSE_MT_FLAG_SET(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#else
#define ZPSE_MT_FLAG_GET(p) ((p)[0])
#define ZPSE_MT_FLAG_SET(p, v) ((p)[0] = (v))
#endif

/* Give the core up while waiting for a slot. A worker can only be waiting
 * for the writer, and the writer can only be waiting for a block, so the
 * wait is always short -- but with more workers than cores a tight spin
 * keeps the writer off a core and the ring stalls, which is a livelock, not
 * a slow path. */
static void zpse_mt_relax(void)
{
#if defined(_WIN32)
    SwitchToThread();
#else
    sched_yield();
#endif
}

static void zpse_mt_stream_fail(ZpseMtStream *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : PSE_BAD_HEADER;
    }
}

static void zpse_mt_stream_worker(ZpseMtStream *j, void *arena)
{
    ZpseEncoder enc;
    int rc;

    rc = zpse_encoder_init(&enc, arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        zpse_mt_stream_fail(j, rc);
        return;
    }
    enc.enc_flags = j->enc_flags;
    for (;;) {
        size_t i;
        size_t slot;
        size_t off;
        size_t chunk;
        int wrc;

        if (j->failed != 0) {
            return;
        }
        i = zpse_mt_claim(&j->next, j->nblocks);
        if (i == SIZE_MAX) {
            return;
        }
        slot = i % (size_t)j->ring;
        while (ZPSE_MT_FLAG_GET(&j->ready[slot]) != ZPSE_MT_SLOT_EMPTY) {
            if (j->failed != 0) {
                return;
            }
            zpse_mt_relax();
        }
        ZPSE_MT_FLAG_SET(&j->ready[slot], ZPSE_MT_SLOT_FILLING);
        off = i * (size_t)j->block;
        chunk = j->n - off;
        if (chunk > (size_t)j->block) {
            chunk = (size_t)j->block;
        }
        wrc = zpse_block_encode(&enc, j->in + off, (uint32_t)chunk,
                                j->first_block + (uint32_t)i,
                                (j->final_slab && (i + 1u) == j->nblocks) ? 1 : 0,
                                j->slots + slot * j->stride,
                                (uint32_t)j->stride);
        if (wrc <= 0 || (size_t)wrc > j->stride) {
            zpse_mt_stream_fail(j, wrc);
            return;
        }
        j->lens[slot] = (uint32_t)wrc;
        ZPSE_MT_FLAG_SET(&j->ready[slot], ZPSE_MT_SLOT_READY);
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_mt_stream_entry(LPVOID p)
{
    ZpseMtStreamArg *a = (ZpseMtStreamArg *)p;

    zpse_mt_stream_worker(a->job, a->arena);
    return 0;
}
#else
static void *zpse_mt_stream_entry(void *p)
{
    ZpseMtStreamArg *a = (ZpseMtStreamArg *)p;

    zpse_mt_stream_worker(a->job, a->arena);
    return 0;
}
#endif

/* Returns PSE_OK, a negative status, or 1 when no worker thread could be
 * started (nothing encoded, the caller runs it serially). */
static int zpse_mt_stream_run(const uint8_t *in, size_t in_len,
                              uint32_t block, uint32_t enc_flags,
                              unsigned threads, uint32_t first_block,
                              int final_slab, void *const arenas[],
                              uint8_t *slots, unsigned ring, int *ready,
                              uint32_t *lens, ZpseBlockEmit emit,
                              void *emit_ctx)
{
    ZpseMtStream job;
    ZpseMtStreamArg args[ZPSE_MT_MAX_THREADS];
#if defined(_WIN32)
    HANDLE handles[ZPSE_MT_MAX_THREADS];
#else
    pthread_t handles[ZPSE_MT_MAX_THREADS];
#endif
    unsigned started = 0u;
    unsigned w;
    size_t i;
    int rc = PSE_OK;

    job.in = in;
    job.n = in_len;
    job.nblocks = in_len / (size_t)block +
                  ((in_len % (size_t)block) != 0u ? 1u : 0u);
    job.stride = (size_t)block + ZPSE_MT_SLOT_OVERHEAD;
    job.block = block;
    job.enc_flags = enc_flags;
    job.first_block = first_block;
    job.final_slab = final_slab;
    job.ring = ring;
    job.slots = slots;
    job.ready = ready;
    job.lens = lens;
    job.next = 0;
    job.failed = 0;
    for (w = 0u; w < ring; w++) {
        ready[w] = ZPSE_MT_SLOT_EMPTY;
    }
    /* Every worker is spawned here: the calling thread is the writer, so it
     * cannot be one of them. */
    for (w = 0u; w < threads && w < (unsigned)ZPSE_MT_MAX_THREADS; w++) {
        args[w].job = &job;
        args[w].arena = arenas[w];
#if defined(_WIN32)
        handles[w] = CreateThread(NULL, 0, zpse_mt_stream_entry, &args[w], 0,
                                  NULL);
        if (handles[w] == NULL) {
            break;
        }
#else
        if (pthread_create(&handles[w], NULL, zpse_mt_stream_entry, &args[w]) !=
            0) {
            break;
        }
#endif
        started++;
    }
    if (started == 0u) {
        return 1;
    }
    for (i = 0; i < job.nblocks; i++) {
        size_t slot = i % (size_t)ring;

        while (ZPSE_MT_FLAG_GET(&ready[slot]) != ZPSE_MT_SLOT_READY) {
            if (job.failed != 0) {
                rc = job.failed < 0 ? job.failed : PSE_BAD_HEADER;
                goto join;
            }
            zpse_mt_relax();
        }
        rc = emit(emit_ctx, first_block + (uint32_t)i,
                  slots + slot * job.stride, (size_t)lens[slot]);
        if (rc != 0) {
            /* The emit failed (a short write, say): the workers have to stop
             * rather than wait for a writer that is not coming. */
            zpse_mt_stream_fail(&job, rc);
            goto join;
        }
        ZPSE_MT_FLAG_SET(&ready[slot], ZPSE_MT_SLOT_EMPTY);
    }
join:
    if (rc != PSE_OK) {
        zpse_mt_stream_fail(&job, rc);
    }
    for (w = 0u; w < started; w++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[w], INFINITE);
        CloseHandle(handles[w]);
#else
        pthread_join(handles[w], NULL);
#endif
    }
    if (rc == PSE_OK && job.failed != 0) {
        rc = job.failed;
    }
    return rc;
}

#endif /* ZPSE_MT */

/* Streamed encode: one slab at a time, blocks out in order. */
static int zpse_mt_stream_serial(const uint8_t *in, size_t in_len,
                                 uint32_t block, uint32_t enc_flags,
                                 uint32_t first_block, int final_slab,
                                 void *arena, uint8_t *slot, size_t stride,
                                 ZpseBlockEmit emit, void *emit_ctx)
{
    ZpseEncoder enc;
    size_t nblocks = in_len / (size_t)block +
                     ((in_len % (size_t)block) != 0u ? 1u : 0u);
    size_t i;
    int rc;

    rc = zpse_encoder_init(&enc, arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    for (i = 0; i < nblocks; i++) {
        size_t off = i * (size_t)block;
        size_t chunk = in_len - off;
        int wrc;

        if (chunk > (size_t)block) {
            chunk = (size_t)block;
        }
        wrc = zpse_block_encode(&enc, in + off, (uint32_t)chunk,
                                first_block + (uint32_t)i,
                                (final_slab && (i + 1u) == nblocks) ? 1 : 0,
                                slot, (uint32_t)stride);
        if (wrc <= 0 || (size_t)wrc > stride) {
            return wrc < 0 ? wrc : PSE_BAD_HEADER;
        }
        rc = emit(emit_ctx, first_block + (uint32_t)i, slot, (size_t)wrc);
        if (rc != 0) {
            return rc;
        }
    }
    return PSE_OK;
}

int zpse_compress_frame_stream(const uint8_t *in, size_t in_len,
                               uint32_t nominal_block_size,
                               uint32_t enc_flags, unsigned threads,
                               uint32_t first_block, int is_final_slab,
                               void *const enc_arenas[], unsigned arena_count,
                               uint8_t *slot_mem, size_t slot_cap, int *ready,
                               uint32_t *lens, ZpseBlockEmit emit,
                               void *emit_ctx)
{
    size_t stride;
    size_t nblocks;
    unsigned workers;

    if (in == 0 || in_len == 0u || emit == 0 || enc_arenas == 0 ||
        slot_mem == 0 || ready == 0 || lens == 0) {
        return PSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    nblocks = in_len / (size_t)nominal_block_size +
              ((in_len % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks > (size_t)UINT32_MAX ||
        (uint64_t)first_block + (uint64_t)nblocks > (uint64_t)UINT32_MAX) {
        return PSE_BAD_ARG;
    }
    stride = (size_t)nominal_block_size + ZPSE_MT_SLOT_OVERHEAD;
    if (slot_cap < stride) {
        return PSE_OUTPUT_OVERFLOW;
    }
    workers = threads;
    if (arena_count < workers) {
        workers = arena_count;
    }
    if (workers == 0u) {
        workers = 1u;
    }
    /* Multi-worker delivery is WITHHELD. The ring below gave byte-identical
     * streams to the serial encoder at two and four workers, then produced a
     * different stream on each of three consecutive six-worker runs of the
     * same 3 MB input (same length, three digests), and an 8 KiB-block run
     * differed at two, four and six workers as well while keeping the same
     * length. Same length and different bytes is a slot handed over twice --
     * one block's bytes going out under another's length -- so the handoff is
     * not proven and no caller should get it until it is. A one-worker slab
     * is the serial stream by construction, so that is all this serves; use
     * zpse_compress_frame_parallel() when several workers are wanted. */
    workers = 1u;
    /* A slab too short to keep the workers busy is encoded serially: a thread
     * start costs more than the block it would take. */
    if (workers > 1u && nblocks >= (size_t)workers * 4u) {
        unsigned ring = workers + 2u;
        size_t hold = slot_cap / stride;

        if (hold < ring) {
            ring = (unsigned)hold;
        }
        if (ring < 2u) {
            ring = 1u;
        }
#if defined(ZPSE_MT)
        if (ring >= 2u) {
            int rc = zpse_mt_stream_run(in, in_len, nominal_block_size,
                                        enc_flags, workers, first_block,
                                        is_final_slab, enc_arenas, slot_mem,
                                        ring, ready, lens, emit, emit_ctx);
            if (rc != 1) {
                return rc;
            }
        }
#endif
    }
    return zpse_mt_stream_serial(in, in_len, nominal_block_size, enc_flags,
                                 first_block, is_final_slab, enc_arenas[0],
                                 slot_mem, stride, emit, emit_ctx);
}

#if defined(ZPSE_MT)

/* ---- the read side -------------------------------------------------
 *
 * A decode worker needs no staging at all: block k's decoded bytes belong at
 * k * nominal in the caller's buffer, and the index already says where the
 * block starts, so the worker decodes straight into its final place. The
 * only per-worker state is the decoder arena. That is only true because a
 * block decode is bounded by its own output: the LZ decoder used to write a
 * few bytes past the end of a run and rely on the next command to reclaim
 * them, which in a frame means overwriting the next block's first bytes.
 * The literal copy now tests the room in front of where it stores
 * (lz_copy_literals), so exactly `nominal` bytes are written and no worker
 * can touch another's. */
typedef struct ZpseMtDecJob {
    const uint8_t *in;
    size_t in_len;
    uint8_t *out;
    size_t out_cap;
    size_t nominal;
    size_t nblocks;
    volatile long next;
    volatile int failed;
} ZpseMtDecJob;

typedef struct ZpseMtDecArg {
    ZpseMtDecJob *job;
    void *arena;
} ZpseMtDecArg;

static void zpse_mt_dec_fail(ZpseMtDecJob *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : PSE_BAD_HEADER;
    }
}

static void zpse_mt_dec_worker(ZpseMtDecJob *j, void *arena)
{
    ZpseDecoder d;
    int rc;

    rc = zpse_decoder_init(&d, arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != PSE_OK) {
        zpse_mt_dec_fail(j, rc);
        return;
    }
    for (;;) {
        size_t i;
        size_t off;
        size_t room;
        uint32_t got = 0u;

        if (j->failed != 0) {
            return;
        }
        i = zpse_mt_claim(&j->next, j->nblocks);
        if (i == SIZE_MAX) {
            return;
        }
        off = i * j->nominal;
        if (off > j->out_cap) {
            zpse_mt_dec_fail(j, PSE_OUTPUT_OVERFLOW);
            return;
        }
        /* Decoded straight into its final place: the seek writes the block's
         * own bytes and nothing else, so its room is exactly the room the
         * caller's buffer has left at this block. */
        room = j->out_cap - off;
        if (room > j->nominal) {
            room = j->nominal;
        }
        rc = zpse_frame_seek(&d, j->in, j->in_len, (uint32_t)i, j->out + off,
                             (uint32_t)room, &got);
        if (rc != PSE_OK) {
            zpse_mt_dec_fail(j, rc);
            return;
        }
        if (i + 1u < j->nblocks && (size_t)got != j->nominal) {
            zpse_mt_dec_fail(j, PSE_BAD_HEADER);
            return;
        }
        if ((size_t)got > j->out_cap - off) {
            zpse_mt_dec_fail(j, PSE_OUTPUT_OVERFLOW);
            return;
        }
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_mt_dec_entry(LPVOID p)
{
    ZpseMtDecArg *a = (ZpseMtDecArg *)p;

    zpse_mt_dec_worker(a->job, a->arena);
    return 0;
}
#else
static void *zpse_mt_dec_entry(void *p)
{
    ZpseMtDecArg *a = (ZpseMtDecArg *)p;

    zpse_mt_dec_worker(a->job, a->arena);
    return 0;
}
#endif

static int zpse_mt_dec_run(const uint8_t *in, size_t in_len, uint8_t *out,
                           size_t out_cap, size_t nblocks, size_t nominal,
                           size_t last_len, unsigned threads,
                           void *const arenas[])
{
    ZpseMtDecJob job;
    ZpseMtDecArg args[ZPSE_MT_MAX_THREADS];
#if defined(_WIN32)
    HANDLE handles[ZPSE_MT_MAX_THREADS];
#else
    pthread_t handles[ZPSE_MT_MAX_THREADS];
#endif
    unsigned started = 0u;
    unsigned w;

    job.in = in;
    job.in_len = in_len;
    job.out = out;
    job.out_cap = out_cap;
    job.nominal = nominal;
    job.nblocks = nblocks;
    job.next = 0;
    job.failed = 0;

    for (w = 1u; w < threads; w++) {
        args[w].job = &job;
        args[w].arena = arenas[w];
#if defined(_WIN32)
        handles[w] = CreateThread(NULL, 0, zpse_mt_dec_entry, &args[w], 0, NULL);
        if (handles[w] == NULL) {
            break;
        }
#else
        if (pthread_create(&handles[w], NULL, zpse_mt_dec_entry, &args[w]) != 0) {
            break;
        }
#endif
        started++;
    }
    zpse_mt_dec_worker(&job, arenas[0]);
    for (w = 1u; w <= started; w++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[w], INFINITE);
        CloseHandle(handles[w]);
#else
        pthread_join(handles[w], NULL);
#endif
    }
    (void)last_len;
    if (job.failed != 0) {
        return job.failed;
    }
    return PSE_OK;
}

static int zpse_mt_run(const uint8_t *in, size_t n, uint8_t *out,
                       size_t out_cap, size_t *out_len_out, uint32_t block,
                       uint32_t enc_flags, int with_index, unsigned threads,
                       void *const arenas[])
{
    ZpseMtJob job;
    ZpseMtArg args[ZPSE_MT_MAX_THREADS];
#if defined(_WIN32)
    HANDLE handles[ZPSE_MT_MAX_THREADS];
#else
    pthread_t handles[ZPSE_MT_MAX_THREADS];
#endif
    unsigned started = 0u;
    unsigned w;

    job.in = in;
    job.n = n;
    job.out = out;
    job.slots = out + ZPSE_FRAME_HEADER_SIZE;
    job.out_cap = out_cap;
    job.block = block;
    job.enc_flags = enc_flags;
    job.nblocks = n / (size_t)block + ((n % (size_t)block) != 0u ? 1u : 0u);
    job.stride = (size_t)block + ZPSE_MT_SLOT_OVERHEAD;
    job.next = 0;
    job.failed = 0;

    /* The calling thread is worker 0; the rest are spawned. A thread that
     * will not start leaves its share to the others rather than failing
     * the encode -- the work queue does not care how many join in. */
    for (w = 1u; w < threads; w++) {
        args[w].job = &job;
        args[w].arena = arenas[w];
#if defined(_WIN32)
        handles[w] = CreateThread(NULL, 0, zpse_mt_entry, &args[w], 0, NULL);
        if (handles[w] == NULL) {
            break;
        }
#else
        if (pthread_create(&handles[w], NULL, zpse_mt_entry, &args[w]) != 0) {
            break;
        }
#endif
        started++;
    }
    zpse_mt_worker(&job, arenas[0]);
    for (w = 1u; w <= started; w++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[w], INFINITE);
        CloseHandle(handles[w]);
#else
        pthread_join(handles[w], NULL);
#endif
    }
    if (job.failed != 0) {
        return job.failed;
    }
    return zpse_mt_close(&job, with_index, out_len_out);
}

#endif /* ZPSE_MT */

/* One serial decode: both the fallback and what a one-worker request means. */
static int zpse_mt_dec_serial(const uint8_t *in, size_t in_len,
                              uint8_t *output, size_t out_cap,
                              size_t *out_len_out, void *arena)
{
    ZpseDecoder d;
    int rc;

    rc = zpse_decoder_init(&d, arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_frame_decode(&d, in, in_len, output, out_cap, out_len_out);
    return rc < 0 ? rc : PSE_OK;
}

int zpse_decompress_frame_parallel(const uint8_t *in, size_t in_len,
                                   uint8_t *output, size_t out_cap,
                                   size_t *out_len_out, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count)
{
    ZpseFrameHeader fh;
    uint32_t count = 0u;
    size_t nominal;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arenas == 0 ||
        dec_arenas[0] == 0) {
        return PSE_BAD_ARG;
    }
    if (arena_count == 0u) {
        return PSE_BAD_ARG;
    }
    if (in_len < (size_t)ZPSE_FRAME_HEADER_SIZE) {
        return PSE_TRUNCATED;
    }
    *out_len_out = 0;
    rc = zpse_frame_header_read(&fh, in);
    if (rc != PSE_OK) {
        return rc;
    }
    /* The index is the whole reason a block can be found: a header carries
     * its size, not its offset. No index (or a bad one) means the serial
     * decoder, which is not slower, only single-threaded. */
    rc = zpse_frame_index_count(in, in_len, &count);
    if (rc != PSE_OK) {
        return rc;
    }
    nominal = (size_t)fh.nominal_block_size;
#if defined(ZPSE_MT)
    if (threads > 1u && count > 1u) {
        unsigned t = threads;

        if (t > arena_count) {
            t = arena_count;
        }
        if (t > ZPSE_MT_MAX_THREADS) {
            t = ZPSE_MT_MAX_THREADS;
        }
        if ((size_t)t > count) {
            t = (unsigned)count;
        }
        /* Thirty-two blocks per worker. A block decodes in about five
         * microseconds and a thread start costs a hundred, so a worker needs
         * a few hundred microseconds of queue before it is worth starting:
         * measured on this host, 1 MiB frames decode slower on two workers
         * (5.2 against 5.9 GB/s) and 4 MiB frames decode almost twice as fast
         * (8.3 against 4.3). */
        if ((size_t)t * 32u > count) {
            t = (unsigned)(count / 32u);
        }
        if (t > 1u) {
            size_t last_off = ((size_t)count - 1u) * nominal;
            size_t total;

            if (out_cap < last_off) {
                return PSE_OUTPUT_OVERFLOW;
            }
            if ((fh.frame_flags & ZPSE_F_CONTENT_SIZE) != 0u &&
                fh.content_size != UINT64_MAX &&
                fh.content_size > (uint64_t)out_cap) {
                return PSE_OUTPUT_OVERFLOW;
            }
            rc = zpse_mt_dec_run(in, in_len, output, out_cap, (size_t)count,
                                 nominal, 0u, t, dec_arenas);
            if (rc != PSE_OK) {
                return rc;
            }
            if ((fh.frame_flags & ZPSE_F_CONTENT_SIZE) != 0u &&
                fh.content_size != UINT64_MAX) {
                total = (size_t)fh.content_size;
            } else {
                /* No content size in the header: the last block's own
                 * length is the only thing left to ask, and it is the one
                 * block that may be short. */
                ZpseDecoder d;
                size_t room = out_cap - last_off;
                uint32_t got = 0u;

                rc = zpse_decoder_init(&d, dec_arenas[0],
                                       (size_t)ZPSE_DEC_ARENA_SIZE);
                if (rc != PSE_OK) {
                    return rc;
                }
                rc = zpse_frame_seek(&d, in, in_len, count - 1u,
                                     output + last_off,
                                     room > (size_t)UINT32_MAX
                                         ? UINT32_MAX
                                         : (uint32_t)room,
                                     &got);
                if (rc != PSE_OK) {
                    return rc;
                }
                total = last_off + (size_t)got;
            }
            *out_len_out = total;
            return PSE_OK;
        }
    }
#else
    (void)threads;
    (void)arena_count;
    (void)nominal;
#endif
    return zpse_mt_dec_serial(in, in_len, output, out_cap, out_len_out,
                              dec_arenas[0]);
}

int zpse_compress_frame_parallel(const uint8_t *in, size_t n,
                                 uint8_t *out, size_t out_cap,
                                 size_t *out_len_out,
                                 uint32_t nominal_block_size,
                                 uint32_t enc_flags, int with_index,
                                 unsigned threads,
                                 void *const enc_arenas[],
                                 unsigned arena_count)
{
    size_t nblocks;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arenas == 0 ||
        enc_arenas[0] == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0) {
        return PSE_BAD_ARG;
    }
    if (nominal_block_size == 0 ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (arena_count == 0u) {
        return PSE_BAD_ARG;
    }
    *out_len_out = 0;
    nblocks = n / (size_t)nominal_block_size +
              ((n % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks == 0u || nblocks > (size_t)UINT32_MAX) {
        return PSE_BAD_ARG;
    }
#if defined(ZPSE_MT)
    if (threads > 1u && nblocks > 1u &&
        out_cap >= zpse_parallel_bound(n, nominal_block_size, with_index)) {
        if (threads > arena_count) {
            threads = arena_count;
        }
        if (threads > ZPSE_MT_MAX_THREADS) {
            threads = ZPSE_MT_MAX_THREADS;
        }
        if ((size_t)threads > nblocks) {
            threads = (unsigned)nblocks;
        }
        /* Four blocks per worker, or the queue has nothing to balance and
         * the spawn costs more than the blocks it encodes: four 64 KiB
         * blocks are about a tenth of a millisecond of work, which a thread
         * start does not pay for. Measured on an 8 MiB frame three workers
         * take encode from 1.5 to 3.5 GB/s; measured on four blocks they are
         * slower than one. Below the line the serial encoder runs, and it
         * writes the same bytes. */
        if ((size_t)threads * 4u > nblocks) {
            threads = (unsigned)(nblocks / 4u);
        }
        if (threads > 1u) {
            return zpse_mt_run(in, n, out, out_cap, out_len_out,
                               nominal_block_size, enc_flags, with_index,
                               threads, enc_arenas);
        }
    }
#else
    (void)threads;
    (void)arena_count;
#endif
    /* One worker, one block, or not enough room for the reserved layout:
     * the serial encoder, which writes the same frame. */
    return zpse_mt_serial(in, n, out, out_cap, out_len_out,
                          nominal_block_size, enc_flags, with_index,
                          enc_arenas);
}
