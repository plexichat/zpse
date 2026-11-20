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
/* The persistent pool below uses condition variables (Vista+ on Win32). */
#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0600
#endif

#include "zpse/parallel.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "zpse/api.h"
#include "zpse/arena.h"
#include "zpse/block.h"
#include "zpse/constants.h"
#include "zpse/flat7.h"
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

/* A block's reserved slot: the block itself plus its header, which is what
 * the raw form costs and therefore the most any form can cost. A page's
 * header is 16 bytes rather than 32, and the encoder only writes the short
 * form for a block no larger than ZPSE_BLOCK_SHORT_MAX, so reserving 32 on
 * every page slot was 16 bytes of slack per block. */
#define ZPSE_MT_SLOT_OVERHEAD       ((size_t)ZPSE_BLOCK_HEADER_SIZE)
#define ZPSE_MT_SLOT_OVERHEAD_SHORT ((size_t)ZPSE_BLOCK_HEADER_SIZE_SHORT)

static size_t zpse_mt_slot_overhead(uint32_t nominal_block_size)
{
    return (nominal_block_size <= (uint32_t)ZPSE_BLOCK_SHORT_MAX)
               ? ZPSE_MT_SLOT_OVERHEAD_SHORT
               : ZPSE_MT_SLOT_OVERHEAD;
}

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
           blocks * ((size_t)nominal_block_size +
                     zpse_mt_slot_overhead(nominal_block_size)) +
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
    if (rc != ZPSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_frame_encode_ex(&enc, in, n, block, out, out_cap, out_len_out,
                              with_index);
    return rc < 0 ? rc : ZPSE_OK;
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
    unsigned workers;   /* static worker->block stride (see zpse_mt_worker) */
    /* First failure, advisory: it stops the other workers early, and the
     * caller reads it after every thread has joined. */
    volatile int failed;
} ZpseMtJob;

typedef struct ZpseMtArg {
    ZpseMtJob *job;
    void *arena;
    unsigned worker;
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

/* Decode workers take a run of blocks per claim instead of one at a time. A
 * decoded block is independent and lands at its own offset in `out`, so the
 * run size cannot change a byte -- it only trades atomic operations for a
 * little less load balance at the tail. Eight keeps the atomic off the hot
 * path without letting one slow block stall a whole worker's worth of work. */
#define ZPSE_MT_DEC_CHUNK 8u

static size_t zpse_mt_claim_chunk(volatile long *next, size_t nblocks,
                                  unsigned chunk)
{
    long v;

#if defined(_WIN32)
    v = InterlockedExchangeAdd((volatile LONG *)next, (LONG)chunk);
#else
    v = __atomic_fetch_add(next, (long)chunk, __ATOMIC_RELAXED);
#endif
    if (v < 0 || (size_t)v >= nblocks) {
        return SIZE_MAX;
    }
    return (size_t)v;
}

static void zpse_mt_fail(ZpseMtJob *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : ZPSE_BAD_HEADER;
    }
}

/* Worker w owns blocks w, w + workers, w + 2*workers, ... in that order --
 * the same static assignment the streamed encode uses. It has to be static:
 * with a shared claim counter the block -> worker pairing is decided by
 * timing, so the per-worker encoder history follows a timing-dependent
 * sequence -- and a history-confident stop (ZPSE_HIST_STOP_STREAK > 0) would
 * then let that timing change the bytes. Here a worker's blocks and their
 * order are fixed, and a block is a pure function of its own bytes at the
 * current defaults, so the frame is the same at every worker count. */
static void zpse_mt_worker(ZpseMtJob *j, void *arena, unsigned w)
{
    ZpseEncoder enc;
    unsigned g = 0u;
    int rc;

    rc = zpse_encoder_init(&enc, arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
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
        i = (size_t)w + (size_t)g * (size_t)j->workers;
        if (i >= j->nblocks) {
            return;
        }
        g++;
        off = i * (size_t)j->block;
        if (off >= j->n) {
            zpse_mt_fail(j, ZPSE_BAD_HEADER);
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

    zpse_mt_worker(a->job, a->arena, a->worker);
    return 0;
}
#else
static void *zpse_mt_entry(void *p)
{
    ZpseMtArg *a = (ZpseMtArg *)p;

    zpse_mt_worker(a->job, a->arena, a->worker);
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
        if (rc != ZPSE_OK) {
            return rc;
        }
        if (bh.compressed_block_size <
                (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT ||
            (size_t)bh.compressed_block_size > j->stride ||
            bh.block_sequence_number != (uint32_t)i) {
            return ZPSE_BAD_HEADER;
        }
        /* Slots start at the header offset and pos starts there too, so
         * pos <= header + i * stride throughout and writing block i's
         * `size <= stride` bytes ends at or before slot i+1 begins. A
         * forward pass therefore cannot overwrite a block it has not read
         * yet, which is the whole reason a block's slot is block plus its
         * header (32 bytes, or 16 for a page). */
        if (j->out + pos != src) {
            memmove(j->out + pos, src, (size_t)bh.compressed_block_size);
        }
        pos += (size_t)bh.compressed_block_size;
    }
    rc = zpse_frame_header_build(&fh, j->block, (uint64_t)j->n, 1,
                                 (uint32_t)j->nblocks, 1, with_index);
    if (rc != ZPSE_OK) {
        return rc;
    }
    rc = zpse_frame_header_write(&fh, j->out);
    if (rc != ZPSE_OK) {
        return rc;
    }
    if (with_index) {
        rc = zpse_frame_write_index(j->out, pos, j->out_cap, j->block,
                                    (uint64_t)j->n, (uint32_t)j->nblocks,
                                    &pos);
        if (rc != ZPSE_OK) {
            return rc;
        }
    }
    *out_len_out = pos;
    return ZPSE_OK;
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
    unsigned workers;   /* slots per ring group, and the block stride */
    unsigned depth;     /* ring groups: slots per worker */
    uint8_t *slots;
    int *ready;
    uint32_t *lens;
    volatile int failed;
} ZpseMtStream;

typedef struct ZpseMtStreamArg {
    ZpseMtStream *job;
    void *arena;
    unsigned worker;
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

/* Index of a slot's control entry. Every slot gets a whole cache line of
 * them: the writer polls block i's flag while worker i+1's release lands on
 * the neighbouring entry, and with one int per slot that release invalidates
 * the line the writer is spinning on -- six workers on two lines, so the flag
 * the writer waits for is evicted by five threads that are not touching it. */
#define ZPSE_MT_CTL(slot) ((size_t)(slot) * (size_t)ZPSE_MT_CTRL_STRIDE)

static void zpse_mt_stream_fail(ZpseMtStream *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : ZPSE_BAD_HEADER;
    }
}

/* One worker, one slot, every `workers`-th block: worker w owns slot w and
 * encodes blocks w, w + workers, w + 2*workers, ... in that order.
 *
 * The assignment has to be static. With a shared "who claims the next block"
 * counter, which block lands in a slot is decided by timing, so the pairing
 * of a slot's previous generation with its next one is decided by timing too --
 * and a slot released at the wrong moment hands one block's bytes to another
 * block's length. That was the defect that kept this path withheld: three
 * consecutive six-worker runs of one input produced three different streams
 * of the same length. Here slot w has exactly one writer, its generations
 * are strictly increasing, and the writer frees a generation only after
 * emitting it, so the only interleaving left is worker -> writer -> worker on
 * one slot, ordered by the acquire/release pair on `ready`. */
/* Encode this worker's `g`-th block into its ring slot, unless the slot is
 * still holding a block the writer has not emitted yet. Returns 1 when the
 * block was encoded, 0 when the slot is held, or a negative status. Both the
 * spawned workers and the calling thread go through this, which is what lets
 * the caller be a worker: the block index, the slot and the sequence number
 * are all a function of (worker, g), so which thread runs it changes nothing
 * about which slot it lands in. */
static int zpse_mt_stream_step(ZpseMtStream *j, ZpseEncoder *enc, unsigned w,
                               unsigned g)
{
    size_t slot = (size_t)w +
                  (size_t)(g % j->depth) * (size_t)j->workers;
    size_t ctl = ZPSE_MT_CTL(slot);
    size_t b = (size_t)w + (size_t)g * (size_t)j->workers;
    size_t off;
    size_t chunk;
    int wrc;

    if (b >= j->nblocks) {
        return 0;
    }
    if (ZPSE_MT_FLAG_GET(&j->ready[ctl]) != ZPSE_MT_SLOT_EMPTY) {
        return 0;
    }
    /* Sequence number and last-block flag are the block's position in the
     * frame, not its arrival order, so a worker writes into the slot the
     * serial encoder would have written into. */
    ZPSE_MT_FLAG_SET(&j->ready[ctl], ZPSE_MT_SLOT_FILLING);
    off = b * (size_t)j->block;
    chunk = j->n - off;
    if (chunk > (size_t)j->block) {
        chunk = (size_t)j->block;
    }
    wrc = zpse_block_encode(enc, j->in + off, (uint32_t)chunk,
                            j->first_block + (uint32_t)b,
                            (j->final_slab && (b + 1u) == j->nblocks) ? 1 : 0,
                            j->slots + slot * j->stride,
                            (uint32_t)j->stride);
    if (wrc <= 0 || (size_t)wrc > j->stride) {
        return wrc < 0 ? wrc : ZPSE_BAD_HEADER;
    }
    j->lens[ctl] = (uint32_t)wrc;
    ZPSE_MT_FLAG_SET(&j->ready[ctl], ZPSE_MT_SLOT_READY);
    return 1;
}

static void zpse_mt_stream_worker(ZpseMtStream *j, void *arena, unsigned w)
{
    ZpseEncoder enc;
    unsigned g = 0u;
    int rc;

    rc = zpse_encoder_init(&enc, arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        zpse_mt_stream_fail(j, rc);
        return;
    }
    enc.enc_flags = j->enc_flags;
    for (;;) {
        if (j->failed != 0) {
            return;
        }
        if ((size_t)w + (size_t)g * (size_t)j->workers >= j->nblocks) {
            return;
        }
        rc = zpse_mt_stream_step(j, &enc, w, g);
        if (rc < 0) {
            zpse_mt_stream_fail(j, rc);
            return;
        }
        if (rc == 0) {
            zpse_mt_relax();
            continue;
        }
        g++;
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_mt_stream_entry(LPVOID p)
{
    ZpseMtStreamArg *a = (ZpseMtStreamArg *)p;

    zpse_mt_stream_worker(a->job, a->arena, a->worker);
    return 0;
}
#else
static void *zpse_mt_stream_entry(void *p)
{
    ZpseMtStreamArg *a = (ZpseMtStreamArg *)p;

    zpse_mt_stream_worker(a->job, a->arena, a->worker);
    return 0;
}
#endif

/* Returns ZPSE_OK, a negative status, or 1 when no worker thread could be
 * started (nothing encoded, the caller runs it serially). */
static int zpse_mt_stream_run(const uint8_t *in, size_t in_len,
                              uint32_t block, uint32_t enc_flags,
                              unsigned workers, uint32_t first_block,
                              int final_slab, void *const arenas[],
                              uint8_t *slots, size_t slot_cap, int *ready,
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
    size_t i = 0u;
    int rc = ZPSE_OK;

    job.in = in;
    job.n = in_len;
    job.nblocks = in_len / (size_t)block +
                  ((in_len % (size_t)block) != 0u ? 1u : 0u);
    job.stride = (size_t)block + ZPSE_MT_SLOT_OVERHEAD;
    job.block = block;
    job.enc_flags = enc_flags;
    job.first_block = first_block;
    job.final_slab = final_slab;
    job.workers = workers;
    /* Slots per worker, from the room the caller left: one is the shallow
     * ring, and one is also what a caller that sized `slot_mem` for exactly
     * one slot per worker gets, so an old allocation is a legal call. */
    job.depth = (unsigned)(slot_cap / (job.stride * (size_t)workers));
    if (job.depth == 0u) {
        job.depth = 1u;
    }
    if (job.depth > (unsigned)ZPSE_MT_MAX_DEPTH) {
        job.depth = (unsigned)ZPSE_MT_MAX_DEPTH;
    }
    job.slots = slots;
    job.ready = ready;
    job.lens = lens;
    job.failed = 0;
    for (w = 0u; w < workers * job.depth; w++) {
        ready[ZPSE_MT_CTL(w)] = ZPSE_MT_SLOT_EMPTY;
    }
    /* The workers 1..workers-1 are spawned here; the calling thread is
     * worker 0 and also the writer. That is the whole point of the shape: a
     * dedicated writer thread is a seventh thread on a six-CPU machine, and
     * when it has nothing to emit it spins, so it takes a core from an
     * encoder for the whole run. Measured on 64 MiB at six workers, this path
     * with a spinning writer reached 2.5x one worker against the buffered
     * encoder's 4.6x -- the ring was not the cost, the thread was. */
    for (w = 1u; w < workers; w++) {
        args[w].job = &job;
        args[w].arena = arenas[w];
        args[w].worker = w;
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
    if (started + 1u < workers) {
        /* Not enough threads started to fill the ring: the blocks the missing
         * workers own would never be encoded, so the whole slab goes serial
         * instead of stalling the emit loop below. */
        for (w = 1u; w <= started; w++) {
#if defined(_WIN32)
            WaitForSingleObject(handles[w], INFINITE);
            CloseHandle(handles[w]);
#else
            pthread_join(handles[w], NULL);
#endif
        }
        return 1;
    }
    {
        ZpseEncoder enc;
        unsigned g = 0u;
        int erc = zpse_encoder_init(&enc, arenas[0],
                                    (size_t)ZPSE_ENC_ARENA_SIZE);

        if (erc != ZPSE_OK) {
            zpse_mt_stream_fail(&job, erc);
            rc = erc;
            goto join;
        }
        enc.enc_flags = enc_flags;
        for (;;) {
            size_t slot;
            size_t ctl;

            /* Everything the prefix of the frame has finished, in order. */
            while (i < job.nblocks) {
                slot = (i % (size_t)workers) +
                       ((i / (size_t)workers) % (size_t)job.depth) *
                           (size_t)workers;
                ctl = ZPSE_MT_CTL(slot);
                if (ZPSE_MT_FLAG_GET(&ready[ctl]) != ZPSE_MT_SLOT_READY) {
                    break;
                }
                rc = emit(emit_ctx, first_block + (uint32_t)i,
                          slots + slot * job.stride, (size_t)lens[ctl]);
                if (rc != 0) {
                    /* The emit failed (a short write, say): the workers have
                     * to stop rather than wait for a writer that is not
                     * coming. */
                    zpse_mt_stream_fail(&job, rc);
                    goto join;
                }
                ZPSE_MT_FLAG_SET(&ready[ctl], ZPSE_MT_SLOT_EMPTY);
                i++;
            }
            if (i == job.nblocks) {
                break;
            }
            if (job.failed != 0) {
                rc = job.failed < 0 ? job.failed : ZPSE_BAD_HEADER;
                goto join;
            }
            /* Nothing to emit yet: encode the next block this thread owns,
             * which is the block the emit loop will need first and therefore
             * the one worth doing here. */
            {
                int sr = zpse_mt_stream_step(&job, &enc, 0u, g);

                if (sr < 0) {
                    zpse_mt_stream_fail(&job, sr);
                    rc = sr;
                    goto join;
                }
                if (sr == 1) {
                    g++;
                    continue;
                }
            }
            zpse_mt_relax();
        }
    }
join:
    if (rc != ZPSE_OK) {
        zpse_mt_stream_fail(&job, rc);
    }
    /* The spawned threads are 1..started: the calling thread is worker 0 and
     * joins nobody but itself, so index 0 holds no handle. */
    for (w = 1u; w <= started; w++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[w], INFINITE);
        CloseHandle(handles[w]);
#else
        pthread_join(handles[w], NULL);
#endif
    }
    if (rc == ZPSE_OK && job.failed != 0) {
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
    if (rc != ZPSE_OK) {
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
            return wrc < 0 ? wrc : ZPSE_BAD_HEADER;
        }
        rc = emit(emit_ctx, first_block + (uint32_t)i, slot, (size_t)wrc);
        if (rc != 0) {
            return rc;
        }
    }
    return ZPSE_OK;
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
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
    nblocks = in_len / (size_t)nominal_block_size +
              ((in_len % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks > (size_t)UINT32_MAX ||
        (uint64_t)first_block + (uint64_t)nblocks > (uint64_t)UINT32_MAX) {
        return ZPSE_BAD_ARG;
    }
    stride = (size_t)nominal_block_size + ZPSE_MT_SLOT_OVERHEAD;
    if (slot_cap < stride) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    workers = threads;
    if (arena_count < workers) {
        workers = arena_count;
    }
    if (workers == 0u) {
        workers = 1u;
    }
#if defined(ZPSE_MT)
    if (workers > (unsigned)ZPSE_MT_MAX_THREADS) {
        workers = (unsigned)ZPSE_MT_MAX_THREADS;
    }
    /* One slot per worker. A worker that had to share a slot with another
     * would need the handoff ordered against a thread it did not start, which
     * is exactly what the static assignment exists to remove, so the slot
     * count caps the worker count rather than the other way round. */
    {
        size_t hold = slot_cap / stride;

        if (hold < (size_t)workers) {
            workers = (unsigned)hold;
        }
    }
#endif
    /* A slab too short to keep the workers busy is encoded serially: a thread
     * start costs more than the block it would take. */
    if (workers > 1u && nblocks >= (size_t)workers * 4u) {
#if defined(ZPSE_MT)
        int rc = zpse_mt_stream_run(in, in_len, nominal_block_size,
                                    enc_flags, workers, first_block,
                                    is_final_slab, enc_arenas, slot_mem,
                                    slot_cap, ready, lens, emit, emit_ctx);

        if (rc != 1) {
            return rc;
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
 * block decode is bounded by its own output: the LZ decoder writes a few
 * bytes past the end of a run and relies on the next command to reclaim
 * them, which in a frame would mean overwriting the next block's first
 * bytes. Its trusted region only runs while more than a command's worth of
 * output room remains (the RESERVE guard in lz.c/lz2.c), so every speculative
 * store stays inside this block's `nominal` bytes; the last RESERVE bytes are
 * decoded by the checked tail, which stores within the exact capacity. No
 * worker can touch another's output. */
typedef struct ZpseMtDecJob {
    const uint8_t *in;
    size_t in_len;
    /* Validated index entries, resolved once for the whole frame by
     * zpse_frame_index_resolve before any worker starts. Workers read only
     * their own entry, so the header/table CRCs are not re-run per block. */
    const uint8_t *entries;
    uint8_t *out;
    size_t out_cap;
    size_t nominal;
    size_t nblocks;
    volatile long next;
    volatile int failed;
    /* Decoded length of the last block, published by the single worker that
     * decodes it (see zpse_mt_dec_worker) and read after every thread joins.
     * The caller needs it to size a frame that carries no content size, and
     * decoding that block a second time is pure duplication. */
    volatile uint32_t last_got;
} ZpseMtDecJob;

typedef struct ZpseMtDecArg {
    ZpseMtDecJob *job;
    void *arena;
} ZpseMtDecArg;

static void zpse_mt_dec_fail(ZpseMtDecJob *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : ZPSE_BAD_HEADER;
    }
}

static void zpse_mt_dec_worker(ZpseMtDecJob *j, void *arena)
{
    ZpseDecoder d;
    int rc;

    rc = zpse_decoder_init(&d, arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        zpse_mt_dec_fail(j, rc);
        return;
    }
    for (;;) {
        size_t i;
        size_t end;

        if (j->failed != 0) {
            return;
        }
        i = zpse_mt_claim_chunk(&j->next, j->nblocks, ZPSE_MT_DEC_CHUNK);
        if (i == SIZE_MAX) {
            return;
        }
        end = i + (size_t)ZPSE_MT_DEC_CHUNK;
        if (end > j->nblocks) {
            end = j->nblocks;
        }
        for (; i < end; i++) {
            size_t out_off;
            size_t room;
            uint32_t blk_off;
            uint32_t blk_size;
            uint32_t got = 0u;

            if (j->failed != 0) {
                return;
            }
            out_off = i * j->nominal;
            if (out_off > j->out_cap) {
                zpse_mt_dec_fail(j, ZPSE_OUTPUT_OVERFLOW);
                return;
            }
            /* Decoded straight into its final place: the block writes its own
             * bytes and nothing else, so its room is exactly the room the
             * caller's buffer has left at this position. */
            room = j->out_cap - out_off;
            if (room > j->nominal) {
                room = j->nominal;
            }
            /* The index was validated once by the caller; read this block's
             * entry and let zpse_block_at do the per-entry checks and decode. */
            blk_off = zpse_load_u32le(j->entries +
                                      i * (size_t)ZPSE_INDEX_ENTRY_SIZE);
            blk_size = zpse_load_u32le(j->entries +
                                       i * (size_t)ZPSE_INDEX_ENTRY_SIZE + 4u);
            rc = zpse_block_at(&d, j->in, j->in_len, blk_off, blk_size,
                               (uint32_t)i, j->out + out_off, (uint32_t)room,
                               &got);
            if (rc != ZPSE_OK) {
                zpse_mt_dec_fail(j, rc);
                return;
            }
            if (i + 1u < j->nblocks && (size_t)got != j->nominal) {
                zpse_mt_dec_fail(j, ZPSE_BAD_HEADER);
                return;
            }
            if ((size_t)got > j->out_cap - out_off) {
                zpse_mt_dec_fail(j, ZPSE_OUTPUT_OVERFLOW);
                return;
            }
            /* Only the worker that gets the last block writes this slot and
             * the caller reads it after join. Room here is capped at
             * `nominal`, which is >= the block's own length, so `got` is the
             * same length a second decode would report. */
            if (i + 1u == j->nblocks) {
                j->last_got = got;
            }
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

static int zpse_mt_dec_run(const uint8_t *in, size_t in_len,
                           const uint8_t *entries, uint8_t *out,
                           size_t out_cap, size_t nblocks, size_t nominal,
                           size_t *last_len_out, unsigned threads,
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
    job.entries = entries;
    job.out = out;
    job.out_cap = out_cap;
    job.nominal = nominal;
    job.nblocks = nblocks;
    job.next = 0;
    job.failed = 0;
    job.last_got = 0u;

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
    *last_len_out = (size_t)job.last_got;
    if (job.failed != 0) {
        return job.failed;
    }
    return ZPSE_OK;
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
    job.stride = (size_t)block + zpse_mt_slot_overhead(block);
    job.workers = threads;
    job.failed = 0;

    /* The calling thread is worker 0; the rest are spawned. The assignment is
     * static (worker w owns w, w + threads, ...), so unlike a shared work
     * queue a worker that never starts leaves its blocks unencoded: rather
     * than close a frame with holes in it, hand the whole input to the serial
     * encoder once the threads that did start have been joined. */
    for (w = 1u; w < threads; w++) {
        args[w].job = &job;
        args[w].arena = arenas[w];
        args[w].worker = w;
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
    if (started + 1u < threads) {
        job.failed = 1; /* stop the workers that did start */
        for (w = 1u; w <= started; w++) {
#if defined(_WIN32)
            WaitForSingleObject(handles[w], INFINITE);
            CloseHandle(handles[w]);
#else
            pthread_join(handles[w], NULL);
#endif
        }
        return 1; /* caller runs the serial encoder over the same buffer */
    }
    zpse_mt_worker(&job, arenas[0], 0u);
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
    if (rc != ZPSE_OK) {
        return rc;
    }
    rc = zpse_frame_decode(&d, in, in_len, output, out_cap, out_len_out);
    return rc < 0 ? rc : ZPSE_OK;
}

int zpse_decompress_frame_parallel(const uint8_t *in, size_t in_len,
                                   uint8_t *output, size_t out_cap,
                                   size_t *out_len_out, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count)
{
    ZpseFrameHeader fh;
    const uint8_t *entries = 0;
    uint32_t count = 0u;
    size_t nominal;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arenas == 0 ||
        dec_arenas[0] == 0) {
        return ZPSE_BAD_ARG;
    }
    if (arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (in_len < (size_t)ZPSE_FRAME_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    *out_len_out = 0;
    /* FLAT7 has no block index, which is the only way a worker can find a
     * block to decode. Decline it here rather than letting the index resolve
     * below read its magic as a frame and report BAD_MAGIC. */
    if (zpse_flat7_is_frame(in, in_len)) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    /* The index is the whole reason a block can be found: a header carries
     * its size, not its offset. Validate header and index once here and hand
     * the workers the validated entries, rather than re-reading the header
     * and re-checking the table's CRC once for every block (O(N^2)). A frame
     * with no index reports count 0 and falls back to the serial decoder,
     * which is not slower, only single-threaded. */
    rc = zpse_frame_index_resolve(in, in_len, &fh, &entries, &count);
    if (rc != ZPSE_OK) {
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
            size_t last_len = 0u;

            if (out_cap < last_off) {
                return ZPSE_OUTPUT_OVERFLOW;
            }
            if ((fh.frame_flags & ZPSE_F_CONTENT_SIZE) != 0u &&
                fh.content_size != UINT64_MAX &&
                fh.content_size > (uint64_t)out_cap) {
                return ZPSE_OUTPUT_OVERFLOW;
            }
            rc = zpse_mt_dec_run(in, in_len, entries, output, out_cap,
                                 (size_t)count, nominal, &last_len, t,
                                 dec_arenas);
            if (rc != ZPSE_OK) {
                return rc;
            }
            if ((fh.frame_flags & ZPSE_F_CONTENT_SIZE) != 0u &&
                fh.content_size != UINT64_MAX) {
                total = (size_t)fh.content_size;
            } else {
                /* No content size in the header: the last block's length is
                 * the only thing left to ask, and it is the one block that
                 * may be short. The worker that decoded it published its
                 * length, so there is no need to decode the block a second
                 * time; that worker's room was capped at `nominal`, which is
                 * >= the block's own length, so the length is the same. */
                total = last_off + last_len;
            }
            *out_len_out = total;
            return ZPSE_OK;
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
        return ZPSE_BAD_ARG;
    }
    if (n == 0) {
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size == 0 ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
    if (arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    *out_len_out = 0;
    nblocks = n / (size_t)nominal_block_size +
              ((n % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks == 0u || nblocks > (size_t)UINT32_MAX) {
        return ZPSE_BAD_ARG;
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
            int rc = zpse_mt_run(in, n, out, out_cap, out_len_out,
                                 nominal_block_size, enc_flags, with_index,
                                 threads, enc_arenas);

            if (rc != 1) {
                return rc;
            }
            /* A worker could not be started; the serial encoder below writes
             * the same frame into the same buffer. */
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

/* ---- batch decode ---------------------------------------------------
 *
 * The frame-level parallel decode needs the whole frame resident, because it
 * finds block k through the FrameIndex. A caller that is streaming a file
 * out has the blocks in hand already -- it just read them -- so what it needs
 * is the decode side of the same pool over a batch it owns: `count`
 * independent compressed blocks at known offsets, decoded side by side into a
 * contiguous output buffer. Nothing here looks at a frame or an index, and
 * nothing allocates; the caller reuses one batch of buffers.
 *
 * A block writes exactly its own bytes and nothing else (that bound is the
 * block decoder's, not this loop's), so two workers cannot touch the same
 * byte of `out` at a stride of `stride`.
 */

/* One block of a batch, decoded with a caller-provided, already initialised
 * decoder. zpse_decompress_block would construct and init a ZpseDecoder for
 * every block; a worker owns one decoder for the whole batch (and, in the
 * persistent pool, across batches), so this is the same work without the per
 * block setup. The block decoder keeps no cross-block state that changes a
 * byte, which is why a frame decoder decodes every block through one context
 * too. Its own in_len floor check replaces the one-shot API's. */
static int zpse_mt_batch_one(const uint8_t *in, const uint32_t *offsets,
                             const uint32_t *lens, size_t i, uint8_t *out,
                             uint32_t stride, uint32_t *out_lens,
                             ZpseDecoder *d)
{
    uint32_t got = 0u;
    int rc;

    rc = zpse_block_decode(d, in + offsets[i], lens[i],
                           out + i * (size_t)stride, stride, &got);
    if (rc < 0) {
        return rc;
    }
    if (got > stride) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    out_lens[i] = got;
    return ZPSE_OK;
}

#if defined(ZPSE_MT)

typedef struct ZpseMtBatchJob {
    const uint8_t *in;
    const uint32_t *offsets;
    const uint32_t *lens;
    uint8_t *out;
    uint32_t stride;
    uint32_t *out_lens;
    size_t count;
    volatile long next;
    volatile int failed;
} ZpseMtBatchJob;

typedef struct ZpseMtBatchArg {
    ZpseMtBatchJob *job;
    void *arena;
} ZpseMtBatchArg;

static void zpse_mt_batch_fail(ZpseMtBatchJob *j, int rc)
{
    if (j->failed == 0) {
        j->failed = (rc < 0) ? rc : ZPSE_BAD_HEADER;
    }
}

static void zpse_mt_batch_worker(ZpseMtBatchJob *j, void *arena)
{
    ZpseDecoder d;
    int rc;

    rc = zpse_decoder_init(&d, arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        zpse_mt_batch_fail(j, rc);
        return;
    }
    for (;;) {
        size_t i;

        if (j->failed != 0) {
            return;
        }
        i = zpse_mt_claim(&j->next, j->count);
        if (i == SIZE_MAX) {
            return;
        }
        rc = zpse_mt_batch_one(j->in, j->offsets, j->lens, i, j->out,
                               j->stride, j->out_lens, &d);
        if (rc != ZPSE_OK) {
            zpse_mt_batch_fail(j, rc);
            return;
        }
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_mt_batch_entry(LPVOID p)
{
    ZpseMtBatchArg *a = (ZpseMtBatchArg *)p;

    zpse_mt_batch_worker(a->job, a->arena);
    return 0;
}
#else
static void *zpse_mt_batch_entry(void *p)
{
    ZpseMtBatchArg *a = (ZpseMtBatchArg *)p;

    zpse_mt_batch_worker(a->job, a->arena);
    return 0;
}
#endif

static int zpse_mt_batch_run(const uint8_t *in, const uint32_t *offsets,
                             const uint32_t *lens, uint32_t count,
                             uint8_t *out, uint32_t stride,
                             uint32_t *out_lens, unsigned threads,
                             void *const arenas[])
{
    ZpseMtBatchJob job;
    ZpseMtBatchArg args[ZPSE_MT_MAX_THREADS];
#if defined(_WIN32)
    HANDLE handles[ZPSE_MT_MAX_THREADS];
#else
    pthread_t handles[ZPSE_MT_MAX_THREADS];
#endif
    unsigned started = 0u;
    unsigned w;

    job.in = in;
    job.offsets = offsets;
    job.lens = lens;
    job.out = out;
    job.stride = stride;
    job.out_lens = out_lens;
    job.count = (size_t)count;
    job.next = 0;
    job.failed = 0;

    /* The calling thread is worker 0: a dedicated writer thread is a seventh
     * thread on a six-CPU host and takes a core from the decoders. */
    for (w = 1u; w < threads; w++) {
        args[w].job = &job;
        args[w].arena = arenas[w];
#if defined(_WIN32)
        handles[w] = CreateThread(NULL, 0, zpse_mt_batch_entry, &args[w], 0,
                                  NULL);
        if (handles[w] == NULL) {
            break;
        }
#else
        if (pthread_create(&handles[w], NULL, zpse_mt_batch_entry, &args[w]) !=
            0) {
            break;
        }
#endif
        started++;
    }
    zpse_mt_batch_worker(&job, arenas[0]);
    for (w = 1u; w <= started; w++) {
#if defined(_WIN32)
        WaitForSingleObject(handles[w], INFINITE);
        CloseHandle(handles[w]);
#else
        pthread_join(handles[w], NULL);
#endif
    }
    return job.failed != 0 ? job.failed : ZPSE_OK;
}

#endif /* ZPSE_MT */

int zpse_decompress_batch_parallel(const uint8_t *in,
                                   const uint32_t *offsets,
                                   const uint32_t *lens, uint32_t count,
                                   uint8_t *out, uint32_t stride,
                                   uint32_t *out_lens, unsigned threads,
                                   void *const dec_arenas[],
                                   unsigned arena_count)
{
    uint32_t i;

    if (in == 0 || offsets == 0 || lens == 0 || out == 0 || out_lens == 0 ||
        dec_arenas == 0 || dec_arenas[0] == 0 || arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (count == 0u || stride == 0u || stride > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
#if defined(ZPSE_MT)
    if (threads > 1u && (size_t)threads * 4u <= (size_t)count) {
        unsigned t = threads;

        if (t > arena_count) {
            t = arena_count;
        }
        if (t > ZPSE_MT_MAX_THREADS) {
            t = ZPSE_MT_MAX_THREADS;
        }
        if ((size_t)t > (size_t)count) {
            t = count;
        }
        /* Four blocks per worker before a thread is worth starting: a block
         * decodes in a few microseconds and a thread start costs a hundred,
         * so a small batch runs on the caller's thread and writes the same
         * bytes. */
        if (t > 1u) {
            return zpse_mt_batch_run(in, offsets, lens, count, out, stride,
                                     out_lens, t, dec_arenas);
        }
    }
#else
    (void)threads;
    (void)arena_count;
#endif
    {
        ZpseDecoder d;
        int rc = zpse_decoder_init(&d, dec_arenas[0],
                                   (size_t)ZPSE_DEC_ARENA_SIZE);

        if (rc != ZPSE_OK) {
            return rc;
        }
        for (i = 0u; i < count; i++) {
            rc = zpse_mt_batch_one(in, offsets, lens, (size_t)i, out, stride,
                                   out_lens, &d);

            if (rc != ZPSE_OK) {
                return rc;
            }
        }
    }
    return ZPSE_OK;
}

/* ---- persistent batch pool -------------------------------------------
 *
 * A long-lived set of decode workers. Between batches they sleep on cv_work;
 * a run publishes its batch, bumps the generation, broadcasts, and the
 * workers claim blocks through the same `next` counter the per-batch path
 * uses. The caller is participant 0 and needs no wakeup. A second condvar
 * lets the caller wait until every participant has finished before it may
 * reuse or publish the batch buffers. All control state is in caller-owned
 * scratch, so the library still never allocates.
 */

#if defined(ZPSE_MT)

#if defined(_WIN32)
#define ZPSE_POOL_LOCK_INIT(p)  InitializeSRWLock(&(p)->lock)
#define ZPSE_POOL_LOCK(p)       AcquireSRWLockExclusive(&(p)->lock)
#define ZPSE_POOL_UNLOCK(p)     ReleaseSRWLockExclusive(&(p)->lock)
#define ZPSE_POOL_CV_INIT(cv)   InitializeConditionVariable(cv)
#define ZPSE_POOL_WAIT(p, cv) \
    SleepConditionVariableSRW((cv), &(p)->lock, INFINITE, 0)
#define ZPSE_POOL_SIGNAL(cv)    WakeConditionVariable(cv)
#define ZPSE_POOL_BROADCAST(cv) WakeAllConditionVariable(cv)
#define ZPSE_POOL_LOCK_FINI(p)  ((void)0)
#define ZPSE_POOL_CV_FINI(cv)   ((void)0)
#else
#define ZPSE_POOL_LOCK_INIT(p)  pthread_mutex_init(&(p)->lock, 0)
#define ZPSE_POOL_LOCK(p)       pthread_mutex_lock(&(p)->lock)
#define ZPSE_POOL_UNLOCK(p)     pthread_mutex_unlock(&(p)->lock)
#define ZPSE_POOL_CV_INIT(cv)   pthread_cond_init((cv), 0)
#define ZPSE_POOL_WAIT(p, cv)   pthread_cond_wait((cv), &(p)->lock)
#define ZPSE_POOL_SIGNAL(cv)    pthread_cond_signal(cv)
#define ZPSE_POOL_BROADCAST(cv) pthread_cond_broadcast(cv)
#define ZPSE_POOL_LOCK_FINI(p)  pthread_mutex_destroy(&(p)->lock)
#define ZPSE_POOL_CV_FINI(cv)   pthread_cond_destroy(cv)
#endif

struct ZpseBatchPool;

typedef struct ZpsePoolWorker {
    struct ZpseBatchPool *pool;
    unsigned index;
    ZpseDecoder dec;
#if defined(_WIN32)
    HANDLE th;
#else
    pthread_t th;
#endif
} ZpsePoolWorker;

struct ZpseBatchPool {
#if defined(_WIN32)
    SRWLOCK lock;
    CONDITION_VARIABLE cv_work;
    CONDITION_VARIABLE cv_done;
#else
    pthread_mutex_t lock;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
#endif
    unsigned threads;   /* participants: caller + workers that actually run */
    unsigned started;   /* spawned workers */
    unsigned active;    /* participants still on the current batch */
    unsigned long gen;  /* bumped once per published batch */
    unsigned long seen[ZPSE_MT_MAX_THREADS];
    int stop;
    volatile int failed;
    const uint8_t *in;
    const uint32_t *offsets;
    const uint32_t *lens;
    uint8_t *out;
    uint32_t stride;
    uint32_t *out_lens;
    size_t count;
    volatile long next;
    ZpsePoolWorker w[ZPSE_MT_MAX_THREADS]; /* w[0] is the calling thread */
};

static void *zpse_pool_align(void *p)
{
    uintptr_t u = (uintptr_t)p;
    uintptr_t a = (uintptr_t)ZPSE_ARENA_ALIGN;

    return (void *)((u + a - 1u) & ~(a - 1u));
}

size_t zpse_batch_pool_scratch_size(unsigned threads)
{
    (void)threads; /* worker slots are a fixed-size array in the pool */
    return (size_t)ZPSE_ARENA_ALIGN + sizeof(struct ZpseBatchPool);
}

/* Claim and decode until the batch is exhausted or a worker fails. */
static void zpse_pool_do_work(struct ZpseBatchPool *p, ZpseDecoder *d)
{
    for (;;) {
        size_t i;
        int rc;

        if (p->failed != 0) {
            return;
        }
        i = zpse_mt_claim(&p->next, p->count);
        if (i == SIZE_MAX) {
            return;
        }
        rc = zpse_mt_batch_one(p->in, p->offsets, p->lens, i, p->out,
                               p->stride, p->out_lens, d);
        if (rc != ZPSE_OK) {
            if (p->failed == 0) {
                p->failed = (rc < 0) ? rc : ZPSE_BAD_HEADER;
            }
            return;
        }
    }
}

static void zpse_pool_worker(ZpsePoolWorker *self)
{
    struct ZpseBatchPool *p = self->pool;

    for (;;) {
        ZPSE_POOL_LOCK(p);
        while (!p->stop && p->gen == p->seen[self->index]) {
            ZPSE_POOL_WAIT(p, &p->cv_work);
        }
        if (p->stop) {
            ZPSE_POOL_UNLOCK(p);
            return;
        }
        /* seen[] starts at 0 with gen at 0, so a batch published before this
         * worker first locks is still picked up: no startup race. */
        p->seen[self->index] = p->gen;
        ZPSE_POOL_UNLOCK(p);

        zpse_pool_do_work(p, &self->dec);

        ZPSE_POOL_LOCK(p);
        if (--p->active == 0u) {
            ZPSE_POOL_SIGNAL(&p->cv_done);
        }
        ZPSE_POOL_UNLOCK(p);
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_pool_thread_entry(LPVOID arg)
{
    zpse_pool_worker((ZpsePoolWorker *)arg);
    return 0;
}
#else
static void *zpse_pool_thread_entry(void *arg)
{
    zpse_pool_worker((ZpsePoolWorker *)arg);
    return 0;
}
#endif

int zpse_batch_pool_init(ZpseBatchPool **pool_out, void *scratch,
                         size_t scratch_len, unsigned threads,
                         void *const dec_arenas[], unsigned arena_count)
{
    struct ZpseBatchPool *p;
    unsigned k;
    unsigned started = 0u;

    if (pool_out == 0 || scratch == 0 || dec_arenas == 0 || arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (threads == 0u) {
        threads = 1u;
    }
    if (threads > arena_count) {
        threads = arena_count;
    }
    if (threads > (unsigned)ZPSE_MT_MAX_THREADS) {
        threads = (unsigned)ZPSE_MT_MAX_THREADS;
    }
    if (scratch_len < zpse_batch_pool_scratch_size(threads)) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    p = (struct ZpseBatchPool *)zpse_pool_align(scratch);
    memset(p, 0, sizeof *p);
    ZPSE_POOL_LOCK_INIT(p);
    ZPSE_POOL_CV_INIT(&p->cv_work);
    ZPSE_POOL_CV_INIT(&p->cv_done);
    p->gen = 0;
    p->stop = 0;
    p->failed = 0;
    p->next = 0;
    /* All decoders are initialised before any thread starts, so a failure
     * here leaves no worker to join. */
    for (k = 0u; k < threads; k++) {
        int rc = zpse_decoder_init(&p->w[k].dec, dec_arenas[k],
                                   (size_t)ZPSE_DEC_ARENA_SIZE);

        if (rc != ZPSE_OK) {
            ZPSE_POOL_CV_FINI(&p->cv_done);
            ZPSE_POOL_CV_FINI(&p->cv_work);
            ZPSE_POOL_LOCK_FINI(p);
            return rc;
        }
        p->w[k].pool = p;
        p->w[k].index = k;
    }
    for (k = 1u; k < threads; k++) {
#if defined(_WIN32)
        p->w[k].th = CreateThread(NULL, 0, zpse_pool_thread_entry, &p->w[k],
                                  0, NULL);
        if (p->w[k].th == NULL) {
            break;
        }
#else
        if (pthread_create(&p->w[k].th, NULL, zpse_pool_thread_entry,
                           &p->w[k]) != 0) {
            break;
        }
#endif
        started++;
    }
    p->started = started;
    /* A thread that would not start leaves its share to the others, as in
     * the per-batch path; only the participants that exist count. */
    p->threads = started + 1u;
    *pool_out = p;
    return ZPSE_OK;
}

int zpse_batch_pool_run(ZpseBatchPool *pool, const uint8_t *in,
                        const uint32_t *offsets, const uint32_t *lens,
                        uint32_t count, uint8_t *out, uint32_t stride,
                        uint32_t *out_lens)
{
    struct ZpseBatchPool *p = pool;
    int rc;

    if (p == 0 || in == 0 || offsets == 0 || lens == 0 || out == 0 ||
        out_lens == 0) {
        return ZPSE_BAD_ARG;
    }
    if (count == 0u || stride == 0u || stride > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }

    ZPSE_POOL_LOCK(p);
    p->in = in;
    p->offsets = offsets;
    p->lens = lens;
    p->out = out;
    p->stride = stride;
    p->out_lens = out_lens;
    p->count = (size_t)count;
    p->next = 0;
    p->failed = 0;
    p->active = p->threads;
    p->gen++;
    ZPSE_POOL_BROADCAST(&p->cv_work);
    ZPSE_POOL_UNLOCK(p);

    /* Participant 0 needs no wakeup: it goes straight to the claim loop. */
    zpse_pool_do_work(p, &p->w[0].dec);

    ZPSE_POOL_LOCK(p);
    if (--p->active == 0u) {
        ZPSE_POOL_SIGNAL(&p->cv_done);
    }
    while (p->active > 0u) {
        ZPSE_POOL_WAIT(p, &p->cv_done);
    }
    rc = p->failed;
    ZPSE_POOL_UNLOCK(p);
    return rc != 0 ? rc : ZPSE_OK;
}

void zpse_batch_pool_free(ZpseBatchPool *pool)
{
    struct ZpseBatchPool *p = pool;
    unsigned k;

    if (p == 0) {
        return;
    }
    ZPSE_POOL_LOCK(p);
    p->stop = 1;
    ZPSE_POOL_BROADCAST(&p->cv_work);
    ZPSE_POOL_UNLOCK(p);
    for (k = 1u; k <= p->started; k++) {
#if defined(_WIN32)
        WaitForSingleObject(p->w[k].th, INFINITE);
        CloseHandle(p->w[k].th);
#else
        pthread_join(p->w[k].th, 0);
#endif
    }
    ZPSE_POOL_CV_FINI(&p->cv_done);
    ZPSE_POOL_CV_FINI(&p->cv_work);
    ZPSE_POOL_LOCK_FINI(p);
}

#else /* !ZPSE_MT: the pool is a serial decoder behind the same API */

struct ZpseBatchPool {
    ZpseDecoder dec;
};

size_t zpse_batch_pool_scratch_size(unsigned threads)
{
    (void)threads;
    return sizeof(struct ZpseBatchPool);
}

int zpse_batch_pool_init(ZpseBatchPool **pool_out, void *scratch,
                         size_t scratch_len, unsigned threads,
                         void *const dec_arenas[], unsigned arena_count)
{
    struct ZpseBatchPool *p;
    int rc;

    (void)threads;
    if (pool_out == 0 || scratch == 0 || dec_arenas == 0 || arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (scratch_len < sizeof(struct ZpseBatchPool)) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    p = (struct ZpseBatchPool *)scratch;
    rc = zpse_decoder_init(&p->dec, dec_arenas[0],
                           (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    *pool_out = p;
    return ZPSE_OK;
}

int zpse_batch_pool_run(ZpseBatchPool *pool, const uint8_t *in,
                        const uint32_t *offsets, const uint32_t *lens,
                        uint32_t count, uint8_t *out, uint32_t stride,
                        uint32_t *out_lens)
{
    struct ZpseBatchPool *p = pool;
    uint32_t i;

    if (p == 0 || in == 0 || offsets == 0 || lens == 0 || out == 0 ||
        out_lens == 0) {
        return ZPSE_BAD_ARG;
    }
    if (count == 0u || stride == 0u || stride > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
    for (i = 0u; i < count; i++) {
        int rc = zpse_mt_batch_one(in, offsets, lens, (size_t)i, out, stride,
                                   out_lens, &p->dec);

        if (rc != ZPSE_OK) {
            return rc;
        }
    }
    return ZPSE_OK;
}

void zpse_batch_pool_free(ZpseBatchPool *pool)
{
    (void)pool;
}

#endif /* ZPSE_MT */

/* ---- persistent encode pool -----------------------------------------
 *
 * The streamed encode above starts and joins threads once per slab. This pool
 * keeps those workers across slabs: between runs they sleep on cv_work; a run
 * publishes its slab, bumps the generation, broadcasts, and the workers run
 * the very same ring step the streamed path uses (same static worker->slot
 * assignment, same seq and last-block flag). cv_done lets the caller wait
 * until every participant has finished before it may reuse the slab buffers.
 *
 * One encoder per participant lives in the caller's scratch, initialised once
 * from a caller-owned arena. A block is a pure function of its own bytes; the
 * cross-block history an encoder carries only picks which survivor is tried
 * first, so keeping it across a slab boundary cannot change a byte (the same
 * property the stream's test pins by comparing every worker count). All
 * control state is in caller scratch, so the library still never allocates.
 */

#if defined(ZPSE_MT)

struct ZpseEncodePool;

typedef struct ZpseEncodePoolWorker {
    struct ZpseEncodePool *pool;
    unsigned index;
    ZpseEncoder enc;
#if defined(_WIN32)
    HANDLE th;
#else
    pthread_t th;
#endif
} ZpseEncodePoolWorker;

struct ZpseEncodePool {
#if defined(_WIN32)
    SRWLOCK lock;
    CONDITION_VARIABLE cv_work;
    CONDITION_VARIABLE cv_done;
#else
    pthread_mutex_t lock;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
#endif
    unsigned threads;   /* participants: caller + workers that actually run */
    unsigned started;   /* spawned workers */
    unsigned active;    /* participants still on the current slab */
    unsigned long gen;  /* bumped once per published slab */
    unsigned long seen[ZPSE_MT_MAX_THREADS];
    int stop;
    /* The current slab: the same job the streamed path drives. */
    ZpseMtStream job;
    ZpseEncodePoolWorker w[ZPSE_MT_MAX_THREADS]; /* w[0] is the calling thread */
};

static void *zpse_encode_pool_align(void *p)
{
    uintptr_t u = (uintptr_t)p;
    uintptr_t a = (uintptr_t)ZPSE_ARENA_ALIGN;

    return (void *)((u + a - 1u) & ~(a - 1u));
}

size_t zpse_encode_pool_scratch_size(unsigned threads)
{
    (void)threads; /* worker slots are a fixed-size array in the pool */
    return (size_t)ZPSE_ARENA_ALIGN + sizeof(struct ZpseEncodePool);
}

/* Participant 0's side of a slab: emit every finished block in block order,
 * encoding its own blocks in between. This is the same loop, over the same
 * static assignment and ring states, that zpse_mt_stream_run() drives; only
 * the encoder is long-lived. */
static int zpse_encode_pool_drive(struct ZpseEncodePool *p,
                                  ZpseBlockEmit emit, void *emit_ctx,
                                  uint32_t first_block)
{
    ZpseMtStream *j = &p->job;
    ZpseEncoder *enc = &p->w[0].enc;
    size_t i = 0u;
    unsigned g = 0u;
    int rc = ZPSE_OK;

    enc->enc_flags = j->enc_flags;
    for (;;) {
        size_t slot;
        size_t ctl;

        /* Everything the prefix of the slab has finished, in order. */
        while (i < j->nblocks) {
            slot = (i % (size_t)j->workers) +
                   ((i / (size_t)j->workers) % (size_t)j->depth) *
                       (size_t)j->workers;
            ctl = ZPSE_MT_CTL(slot);
            if (ZPSE_MT_FLAG_GET(&j->ready[ctl]) != ZPSE_MT_SLOT_READY) {
                break;
            }
            rc = emit(emit_ctx, first_block + (uint32_t)i,
                      j->slots + slot * j->stride, (size_t)j->lens[ctl]);
            if (rc != 0) {
                /* The emit failed: the workers must stop rather than wait for
                 * a writer that is no longer coming. */
                zpse_mt_stream_fail(j, rc);
                goto done;
            }
            ZPSE_MT_FLAG_SET(&j->ready[ctl], ZPSE_MT_SLOT_EMPTY);
            i++;
        }
        if (i == j->nblocks) {
            break;
        }
        if (j->failed != 0) {
            rc = j->failed < 0 ? j->failed : ZPSE_BAD_HEADER;
            goto done;
        }
        /* Nothing to emit yet: encode the next block this thread owns. */
        {
            int sr = zpse_mt_stream_step(j, enc, 0u, g);

            if (sr < 0) {
                zpse_mt_stream_fail(j, sr);
                rc = sr;
                goto done;
            }
            if (sr == 1) {
                g++;
                continue;
            }
        }
        zpse_mt_relax();
    }
done:
    if (rc != ZPSE_OK) {
        zpse_mt_stream_fail(j, rc);
    }
    return rc;
}

static void zpse_encode_pool_worker(ZpseEncodePoolWorker *self)
{
    struct ZpseEncodePool *p = self->pool;

    for (;;) {
        unsigned g = 0u;

        ZPSE_POOL_LOCK(p);
        while (!p->stop && p->gen == p->seen[self->index]) {
            ZPSE_POOL_WAIT(p, &p->cv_work);
        }
        if (p->stop) {
            ZPSE_POOL_UNLOCK(p);
            return;
        }
        /* seen[] starts at 0 with gen at 0, so a slab published before this
         * worker first locks is still picked up: no startup race. */
        p->seen[self->index] = p->gen;
        self->enc.enc_flags = p->job.enc_flags;
        ZPSE_POOL_UNLOCK(p);

        /* A slab can be narrower than the pool: participants past the slab's
         * worker count do no encoding but still count towards `active`. Their
         * block indices would otherwise overlap participant 0's. */
        if (self->index < p->job.workers) {
            for (;;) {
                int rc;

                if (p->job.failed != 0) {
                    break;
                }
                if ((size_t)self->index +
                        (size_t)g * (size_t)p->job.workers >=
                    p->job.nblocks) {
                    break;
                }
                rc = zpse_mt_stream_step(&p->job, &self->enc, self->index, g);
                if (rc < 0) {
                    if (p->job.failed == 0) {
                        p->job.failed = (rc < 0) ? rc : ZPSE_BAD_HEADER;
                    }
                    break;
                }
                if (rc == 0) {
                    zpse_mt_relax();
                    continue;
                }
                g++;
            }
        }

        ZPSE_POOL_LOCK(p);
        if (--p->active == 0u) {
            ZPSE_POOL_SIGNAL(&p->cv_done);
        }
        ZPSE_POOL_UNLOCK(p);
    }
}

#if defined(_WIN32)
static DWORD WINAPI zpse_encode_pool_thread_entry(LPVOID arg)
{
    zpse_encode_pool_worker((ZpseEncodePoolWorker *)arg);
    return 0;
}
#else
static void *zpse_encode_pool_thread_entry(void *arg)
{
    zpse_encode_pool_worker((ZpseEncodePoolWorker *)arg);
    return 0;
}
#endif

int zpse_encode_pool_init(ZpseEncodePool **pool_out, void *scratch,
                          size_t scratch_len, unsigned threads,
                          void *const enc_arenas[], unsigned arena_count)
{
    struct ZpseEncodePool *p;
    unsigned k;
    unsigned started = 0u;

    if (pool_out == 0 || scratch == 0 || enc_arenas == 0 || arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (threads == 0u) {
        threads = 1u;
    }
    if (threads > arena_count) {
        threads = arena_count;
    }
    if (threads > (unsigned)ZPSE_MT_MAX_THREADS) {
        threads = (unsigned)ZPSE_MT_MAX_THREADS;
    }
    if (scratch_len < zpse_encode_pool_scratch_size(threads)) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    p = (struct ZpseEncodePool *)zpse_encode_pool_align(scratch);
    memset(p, 0, sizeof *p);
    ZPSE_POOL_LOCK_INIT(p);
    ZPSE_POOL_CV_INIT(&p->cv_work);
    ZPSE_POOL_CV_INIT(&p->cv_done);
    p->gen = 0;
    p->stop = 0;
    /* All encoders are initialised before any thread starts, so a failure
     * here leaves no worker to join. */
    for (k = 0u; k < threads; k++) {
        int rc = zpse_encoder_init(&p->w[k].enc, enc_arenas[k],
                                   (size_t)ZPSE_ENC_ARENA_SIZE);

        if (rc != ZPSE_OK) {
            ZPSE_POOL_CV_FINI(&p->cv_done);
            ZPSE_POOL_CV_FINI(&p->cv_work);
            ZPSE_POOL_LOCK_FINI(p);
            return rc;
        }
        p->w[k].pool = p;
        p->w[k].index = k;
    }
    for (k = 1u; k < threads; k++) {
#if defined(_WIN32)
        p->w[k].th = CreateThread(NULL, 0, zpse_encode_pool_thread_entry,
                                  &p->w[k], 0, NULL);
        if (p->w[k].th == NULL) {
            break;
        }
#else
        if (pthread_create(&p->w[k].th, NULL, zpse_encode_pool_thread_entry,
                           &p->w[k]) != 0) {
            break;
        }
#endif
        started++;
    }
    p->started = started;
    /* A thread that would not start leaves its share to the others, as in the
     * streamed path; only the participants that exist count. */
    p->threads = started + 1u;
    *pool_out = p;
    return ZPSE_OK;
}

int zpse_encode_pool_run(ZpseEncodePool *pool, const uint8_t *in,
                         size_t in_len, uint32_t nominal_block_size,
                         uint32_t enc_flags, uint32_t first_block,
                         int is_final_slab, uint8_t *slot_mem, size_t slot_cap,
                         int *ready, uint32_t *lens, ZpseBlockEmit emit,
                         void *emit_ctx)
{
    struct ZpseEncodePool *p = pool;
    ZpseMtStream *j;
    size_t nblocks;
    size_t stride;
    size_t hold;
    unsigned workers;
    int rc;

    if (p == 0 || in == 0 || in_len == 0u || emit == 0 || slot_mem == 0 ||
        ready == 0 || lens == 0) {
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
    nblocks = in_len / (size_t)nominal_block_size +
              ((in_len % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks > (size_t)UINT32_MAX ||
        (uint64_t)first_block + (uint64_t)nblocks > (uint64_t)UINT32_MAX) {
        return ZPSE_BAD_ARG;
    }
    /* Same reserved slot layout as the streamed encode, so both paths write
     * the same bytes into the same slots and the index offsets agree. */
    stride = (size_t)nominal_block_size + ZPSE_MT_SLOT_OVERHEAD;
    if (slot_cap < stride) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    workers = p->threads;
    hold = slot_cap / stride;
    if (hold < (size_t)workers) {
        workers = (unsigned)hold; /* extra participants idle this slab */
    }
    if (workers == 0u) {
        workers = 1u;
    }

    ZPSE_POOL_LOCK(p);
    j = &p->job;
    j->in = in;
    j->n = in_len;
    j->nblocks = nblocks;
    j->stride = stride;
    j->block = nominal_block_size;
    j->enc_flags = enc_flags;
    j->first_block = first_block;
    j->final_slab = is_final_slab;
    j->workers = workers;
    j->depth = (unsigned)(slot_cap / (stride * (size_t)workers));
    if (j->depth == 0u) {
        j->depth = 1u;
    }
    if (j->depth > (unsigned)ZPSE_MT_MAX_DEPTH) {
        j->depth = (unsigned)ZPSE_MT_MAX_DEPTH;
    }
    j->slots = slot_mem;
    j->ready = ready;
    j->lens = lens;
    j->failed = 0;
    {
        unsigned s;

        for (s = 0u; s < workers * j->depth; s++) {
            ready[ZPSE_MT_CTL(s)] = ZPSE_MT_SLOT_EMPTY;
        }
    }
    p->active = p->threads;
    p->gen++;
    ZPSE_POOL_BROADCAST(&p->cv_work);
    ZPSE_POOL_UNLOCK(p);

    /* Participant 0 needs no wakeup: it drives the emit loop directly. */
    (void)zpse_encode_pool_drive(p, emit, emit_ctx, first_block);

    ZPSE_POOL_LOCK(p);
    if (--p->active == 0u) {
        ZPSE_POOL_SIGNAL(&p->cv_done);
    }
    while (p->active > 0u) {
        ZPSE_POOL_WAIT(p, &p->cv_done);
    }
    rc = j->failed;
    ZPSE_POOL_UNLOCK(p);
    return rc != 0 ? rc : ZPSE_OK;
}

void zpse_encode_pool_free(ZpseEncodePool *pool)
{
    struct ZpseEncodePool *p = pool;
    unsigned k;

    if (p == 0) {
        return;
    }
    ZPSE_POOL_LOCK(p);
    p->stop = 1;
    ZPSE_POOL_BROADCAST(&p->cv_work);
    ZPSE_POOL_UNLOCK(p);
    for (k = 1u; k <= p->started; k++) {
#if defined(_WIN32)
        WaitForSingleObject(p->w[k].th, INFINITE);
        CloseHandle(p->w[k].th);
#else
        pthread_join(p->w[k].th, 0);
#endif
    }
    ZPSE_POOL_CV_FINI(&p->cv_done);
    ZPSE_POOL_CV_FINI(&p->cv_work);
    ZPSE_POOL_LOCK_FINI(p);
}

#else /* !ZPSE_MT: the pool is the serial stream encoder behind the same API */

struct ZpseEncodePool {
    ZpseEncoder enc;
};

size_t zpse_encode_pool_scratch_size(unsigned threads)
{
    (void)threads;
    return sizeof(struct ZpseEncodePool);
}

int zpse_encode_pool_init(ZpseEncodePool **pool_out, void *scratch,
                          size_t scratch_len, unsigned threads,
                          void *const enc_arenas[], unsigned arena_count)
{
    struct ZpseEncodePool *p;
    int rc;

    (void)threads;
    if (pool_out == 0 || scratch == 0 || enc_arenas == 0 || arena_count == 0u) {
        return ZPSE_BAD_ARG;
    }
    if (scratch_len < sizeof(struct ZpseEncodePool)) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    p = (struct ZpseEncodePool *)scratch;
    rc = zpse_encoder_init(&p->enc, enc_arenas[0], (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    *pool_out = p;
    return ZPSE_OK;
}

int zpse_encode_pool_run(ZpseEncodePool *pool, const uint8_t *in,
                         size_t in_len, uint32_t nominal_block_size,
                         uint32_t enc_flags, uint32_t first_block,
                         int is_final_slab, uint8_t *slot_mem, size_t slot_cap,
                         int *ready, uint32_t *lens, ZpseBlockEmit emit,
                         void *emit_ctx)
{
    struct ZpseEncodePool *p = pool;
    size_t nblocks;
    size_t stride;
    size_t i;

    (void)ready;
    (void)lens;
    if (p == 0 || in == 0 || in_len == 0u || emit == 0 || slot_mem == 0) {
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size < ZPSE_MIN_BLOCK_SIZE ||
        nominal_block_size > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_ARG;
    }
    nblocks = in_len / (size_t)nominal_block_size +
              ((in_len % (size_t)nominal_block_size) != 0u ? 1u : 0u);
    if (nblocks > (size_t)UINT32_MAX ||
        (uint64_t)first_block + (uint64_t)nblocks > (uint64_t)UINT32_MAX) {
        return ZPSE_BAD_ARG;
    }
    stride = (size_t)nominal_block_size + ZPSE_MT_SLOT_OVERHEAD;
    if (slot_cap < stride) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    p->enc.enc_flags = enc_flags;
    for (i = 0u; i < nblocks; i++) {
        size_t off = i * (size_t)nominal_block_size;
        size_t chunk = in_len - off;
        int wrc;
        int rc;

        if (chunk > (size_t)nominal_block_size) {
            chunk = (size_t)nominal_block_size;
        }
        wrc = zpse_block_encode(&p->enc, in + off, (uint32_t)chunk,
                                first_block + (uint32_t)i,
                                (is_final_slab && (i + 1u) == nblocks) ? 1 : 0,
                                slot_mem, (uint32_t)stride);
        if (wrc <= 0 || (size_t)wrc > stride) {
            return wrc < 0 ? wrc : ZPSE_BAD_HEADER;
        }
        rc = emit(emit_ctx, first_block + (uint32_t)i, slot_mem, (size_t)wrc);
        if (rc != 0) {
            return rc;
        }
    }
    return ZPSE_OK;
}

void zpse_encode_pool_free(ZpseEncodePool *pool)
{
    (void)pool;
}

#endif /* ZPSE_MT */
