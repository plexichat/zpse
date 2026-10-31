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
#endif
/* Worker count ceiling: beyond this the compaction pass, not the encode,
 * is the run. It also sizes the thread-handle array below. */
#define ZPSE_MT_MAX_THREADS 64u
#endif

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

/* The block index this worker owns, or SIZE_MAX when the frame is done. */
static size_t zpse_mt_claim(ZpseMtJob *j)
{
    long v;

#if defined(_WIN32)
    v = InterlockedIncrement((volatile LONG *)&j->next) - 1;
#else
    v = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
#endif
    if (v < 0 || (size_t)v >= j->nblocks) {
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
        i = zpse_mt_claim(j);
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
