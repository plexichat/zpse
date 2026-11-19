/* Command-line binding over the public frame API.
 *
 * Usage:
 *   zpse_cli c [opts] IN OUT     compress a file
 *   zpse_cli d IN OUT            decompress a frame file
 *   zpse_cli x FILE BLOCK OUT    decode one block through the frame index
 *   zpse_cli dir FILE            print the frame's random-access directory
 *   zpse_cli col [opts] IN OUT   type a CSV's columns and compress the
 *                                typed, column-major blob (its own column
 *                                widths are what makes a text export
 *                                compress like the data it holds)
 *   zpse_cli uncol IN OUT        the inverse: rebuild the CSV, byte for byte
 *   zpse_cli colcheck [opts] IN  both, verified, with the text and columnar
 *                                ratios side by side
 *   zpse_cli bench [opts] FILE...  compress and decompress each file IN
 *                                MEMORY (nothing written), verify the
 *                                round trip, report ratio and throughput
 *
 * Options (c and bench):
 *   -l LEVEL   balanced (default) | faster | fastest | max
 *   -F         write the fast LZ form only (one match-finder pass, no
 *              shape decision); see ZPSE_ENC_LZ_FAST
 *   -F7, -7    bench only: run an in-memory FLAT7 arm (fused gate, ASCII scan
 *              and 7-bit pack) beside the normal block arm on eligible input,
 *              and print both rates and ratios with the flat7/block speedup
 *   -R         bench: run an in-memory whole-frame REPACK arm (fixed-width
 *              ASCII symbol repack, magic ZRPK) beside the normal block arm on
 *              eligible input, with both rates, ratios and the
 *              repack/block speedup. Replaces the -F7 A/B arm when both are
 *              given. c: try a ZRPK frame first, normal codec on decline;
 *              d: sniff ZRPK and call the repack decoder.
 *   -P         a page (block <= 4096) gets the identity tile stream and no
 *              shape decision; ignored above a page. See ZPSE_ENC_PAGE_FAST
 *   -c         emit the per-block payload CRC32-C (verified on decode)
 *   -b N       nominal block size, 1..65536 (default 65536)
 *   -D         decode-first: keep an intra-block LZ form only when it is at
 *              least 1/8 smaller than the best tile stream; see
 *              ZPSE_ENC_DECODE_FAST
 *   -x         append the FrameIndex (random access); bench also seeks
 *   -t N       use N worker threads (default 1: the single-core path every
 *              figure in the README was measured on). On decode the blocks
 *              are read in a batch and decoded side by side, written in
 *              order, with the read of the next batch and the write of the
 *              last one each on its own thread, so neither the frame nor the
 *              output is resident whole.
 *   -r N       bench repetitions per file (0 = pick from the file size)
 *   -n N       bench: time only the first N bytes of each file (0 = the
 *              whole file), for one case without the whole corpus
 *   -q         bench: summary line only, no per-file rows
 *
 * Exit 0 on success, 1 on an I/O, codec or verification failure, 2 on a
 * usage error. The arenas are the documented sizes (464 KiB encode /
 * 384 KiB decode), 64-byte aligned, allocated once per run -- the same
 * shape an embedder uses.
 *
 * Throughputs are decimal (1 MB = 1e6 bytes) and measure the codec only:
 * a bench reads each input once, before the clock starts, and keeps every
 * buffer in memory.
 *
 * A frame is streamed, not buffered: the input is read a slab at a time and
 * each block is written to the output as it is finished, so neither the file
 * nor the frame is ever resident whole -- and that holds for -x too, because
 * the table is recorded as the blocks go rather than walked off the file
 * afterwards. -t keeps that path and hands the slab to the worker ring, so
 * the workers hold a few slots between them rather than the frame. `bench`
 * is the one caller that still buffers, because it is measuring the codec in
 * memory rather than writing a file. */
/* clock_gettime is POSIX, and a strict -std=c11 hides it unless the
 * feature-test macro asks for it (the in-tree benchmarks do the same). */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> /* struct timespec + clock_gettime on the POSIX path */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h> /* encode maps its input instead of copying it */
#if defined(ZPSE_MT)
#include <pthread.h> /* the decode writer thread, beside the worker pool */
#endif
#endif

#include "zpse/zpse.h"
#include "zpse/block.h"
#include "zpse/flat7.h"
#include "zpse/frame.h"
#include "zpse/parallel.h"
#include "zpse/repack.h"
#include "zpse/rlefast.h"
#include "zpse/tile.h"

/* The persistent encode pool is an addition to parallel.h. Detect it by the
 * size macro it ships, so this file still builds against a parallel.h that
 * predates it: without the macro the pool path below compiles away and the
 * original per-slab zpse_compress_frame_stream() call (which is the serial
 * encoder at one worker / without ZPSE_MT) runs unchanged. The __has_include
 * branch is a fallback for a build that carries the pool in its own header. */
#if defined(ZPSE_MT) && defined(ZPSE_ENCODE_POOL_SCRATCH_SIZE)
#define ZPSE_CLI_HAVE_ENCODE_POOL 1
#elif defined(__has_include)
#if __has_include("zpse/encode_pool.h")
#include "zpse/encode_pool.h"
#if defined(ZPSE_MT) && defined(ZPSE_ENCODE_POOL_SCRATCH_SIZE)
#define ZPSE_CLI_HAVE_ENCODE_POOL 1
#endif
#endif
#endif

#if defined(_MSC_VER)
#define ZPSE_CLI_ALIGNED(align, size) _aligned_malloc((size), (align))
#define ZPSE_CLI_FREE(p) _aligned_free(p)
#else
/* Alignment by over-allocation: neither posix_memalign nor aligned_alloc
 * is declared for every toolchain under a strict -std=c11 (MinGW hides
 * both), and the arena only needs the pointer to be aligned. The original
 * block is kept in the word in front of the aligned one. */
static void *zpse_cli_aligned(size_t align, size_t size)
{
    uint8_t *raw = (uint8_t *)malloc(size + align - 1u + sizeof(void *));
    uintptr_t p;

    if (raw == NULL) {
        return NULL;
    }
    p = ((uintptr_t)raw + sizeof(void *) + align - 1u) &
        ~(uintptr_t)(align - 1u);
    ((void **)p)[-1] = (void *)raw;
    return (void *)(void *)p;
}

static void zpse_cli_aligned_free(void *p)
{
    if (p != NULL) {
        free(((void **)p)[-1]);
    }
}
#define ZPSE_CLI_ALIGNED(align, size) zpse_cli_aligned((align), (size))
#define ZPSE_CLI_FREE(p) zpse_cli_aligned_free(p)
#endif

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER f = {{0}};
    LARGE_INTEGER c;

    if (f.QuadPart == 0) {
        QueryPerformanceFrequency(&f);
    }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

static uint8_t *read_file(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    long end;
    uint8_t *buf;

    if (f == NULL) {
        fprintf(stderr, "zpse_cli: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "zpse_cli: cannot seek %s\n", path);
        fclose(f);
        return NULL;
    }
    end = ftell(f);
    if (end < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "zpse_cli: cannot size %s\n", path);
        fclose(f);
        return NULL;
    }
    buf = (uint8_t *)malloc((size_t)end + 1u);
    if (buf == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        fclose(f);
        return NULL;
    }
    if ((size_t)end > 0u && fread(buf, 1u, (size_t)end, f) != (size_t)end) {
        fprintf(stderr, "zpse_cli: short read on %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *len_out = (size_t)end;
    return buf;
}

static int write_file(const char *path, const uint8_t *buf, size_t len)
{
    FILE *f = fopen(path, "wb");

    if (f == NULL) {
        fprintf(stderr, "zpse_cli: cannot write %s\n", path);
        return 1;
    }
    if (len > 0u && fwrite(buf, 1u, len, f) != len) {
        fprintf(stderr, "zpse_cli: short write on %s\n", path);
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "zpse_cli: close failed on %s\n", path);
        return 1;
    }
    return 0;
}

static int parse_level(const char *s, uint32_t *flags)
{
    /* The level is a FIELD, not a flag: the caller arrives with the
     * default (BALANCED) already in it, so every branch below has to
     * replace those bits rather than OR into them. It used to OR, and the
     * result was a `-l` that silently encoded a different level than it
     * named -- 10|2 is 10, so `-l 1` ran BALANCED; 10|4 is 14, so
     * `-l 2` and `-l 3` both ran level 7; 10|18 is 26, above the field's
     * 9, so `-l 8` and `-l 9` ran whatever `(26 & mask) >> shift` is and
     * the header printed "level ?". Only `-l 5` and `-l 7` did what they
     * said. Clearing the field here is the whole fix, and it keeps the
     * other bits of the word (CHECKSUM) untouched. */
    *flags &= ~(uint32_t)ZPSE_ENC_SPEED_MASK;

    if (strcmp(s, "fastest") == 0 || strcmp(s, "1") == 0) {
        *flags |= ZPSE_ENC_SPEED_FASTEST;
    } else if (strcmp(s, "2") == 0) {
        *flags |= ZPSE_ENC_SPEED_L2;
    } else if (strcmp(s, "faster") == 0 || strcmp(s, "3") == 0) {
        *flags |= ZPSE_ENC_SPEED_FASTER;
    } else if (strcmp(s, "4") == 0) {
        *flags |= ZPSE_ENC_SPEED_L4;
    } else if (strcmp(s, "balanced") == 0 || strcmp(s, "5") == 0) {
        *flags |= ZPSE_ENC_SPEED_BALANCED;
    } else if (strcmp(s, "6") == 0) {
        *flags |= ZPSE_ENC_SPEED_L6;
    } else if (strcmp(s, "7") == 0) {
        *flags |= ZPSE_ENC_SPEED_L7;
    } else if (strcmp(s, "8") == 0) {
        *flags |= ZPSE_ENC_SPEED_L8;
    } else if (strcmp(s, "max") == 0 || strcmp(s, "9") == 0) {
        *flags |= ZPSE_ENC_SPEED_MAX;
    } else {
        fprintf(stderr, "zpse_cli: unknown level '%s'\n", s);
        return 0;
    }
    return 1;
}

static const char *level_name(uint32_t flags)
{
    switch (zpse_enc_speed(flags)) {
    case 1u:
        return "fastest";
    case 2u:
        return "2";
    case 3u:
        return "faster";
    case 4u:
        return "4";
    case 5u:
        return "balanced";
    case 6u:
        return "6";
    case 7u:
        return "7";
    case 8u:
        return "8";
    case 9u:
        return "max";
    default:
        return "?";
    }
}

/* Upper bound on one frame: every block can carry raw bytes plus its
 * header (32 B per ZPSE_MIN_BLOCK_SIZE would be absurd for tiny blocks,
 * so this is only used with the nominal size below), the frame header,
 * and the optional index. */
static size_t frame_bound(size_t n, uint32_t block, int indexed)
{
    size_t blocks = n / (size_t)block + 1u;

    return n + blocks * (ZPSE_BLOCK_HEADER_SIZE + 64u) +
           ZPSE_FRAME_HEADER_SIZE + 64u +
           (indexed ? ZPSE_INDEX_HEADER_SIZE +
                          blocks * ZPSE_INDEX_ENTRY_SIZE + 64u
                    : 0u);
}

static int compress_mem(const uint8_t *in, size_t n, uint8_t *out,
                        size_t out_cap, size_t *out_len, uint32_t block,
                        uint32_t enc_flags, int indexed,
                        void *const arenas[], unsigned threads)
{
    /* One entry point for both shapes: the parallel call is the serial one
     * when it is asked for one worker, or built without ZPSE_MT, so the
     * single-thread path measured here is the same code. */
    return zpse_compress_frame_parallel(in, n, out, out_cap, out_len, block,
                                        enc_flags, indexed, threads, arenas,
                                        threads);
}

/* One arena per worker, 64-byte aligned, allocated once. */
static void **alloc_arenas(unsigned count, size_t size)
{
    void **a = (void **)malloc(sizeof(void *) * (size_t)count);
    unsigned i;

    if (a == NULL) {
        return NULL;
    }
    for (i = 0; i < count; i++) {
        a[i] = ZPSE_CLI_ALIGNED(64u, size);
        if (a[i] == NULL) {
            while (i > 0u) {
                ZPSE_CLI_FREE(a[--i]);
            }
            free(a);
            return NULL;
        }
    }
    return a;
}

static void free_arenas(void **a, unsigned threads)
{
    unsigned i;

    if (a == NULL) {
        return;
    }
    for (i = 0; i < threads; i++) {
        ZPSE_CLI_FREE(a[i]);
    }
    free(a);
}

/* Decode a frame block by block, reading from the file and writing each block
 * out as it is finished.
 *
 * Neither the frame nor the output has to be in memory. The whole-frame path
 * holds the entire output before it can write a byte of it -- on this corpus
 * a 202 MiB allocation faulted in one page at a time -- and the path before
 * this one also read the whole frame into a second fresh allocation, which
 * measured as about half the decode's wall time: 202 MiB read with `cat`
 * costs 11 ms and the same bytes through `fread` into a new buffer cost
 * 213, because every 4 KiB of it is a zero-filled page the copy then
 * overwrites. Reading a block at a time reuses one buffer instead.
 *
 * The blocks are independent by construction (that is what the 64 KiB
 * framing is for) and each carries its own compressed size, so the loop reads
 * a 32-byte header, learns how long the block is, reads the rest behind it and
 * decodes it. Nothing is validated less: every block goes through
 * zpse_decompress_block, a block that does not decode to the length its header
 * claims is refused, the block count and content size still have to agree with
 * the frame header, and a frame that carries no index is the only kind that
 * takes this path (see the caller).
 *
 * A frame that carries a FrameIndex takes this path too. The table rides
 * behind the last block, so the walk never reads it as a block, and it is
 * validated against the blocks the walk actually visited (see the tail of
 * this function) instead of against a copy of the frame. That copy plus the
 * whole output in one fresh allocation is what `-x` used to cost: on the
 * 202 MiB corpus it measured 0.36 s against this path's 0.12 s, with the same
 * bytes decoded either way.
 *
 * Returns 0 on success, 1 on any failure, matching write_file. A failure
 * removes the partial output, because this path writes as it goes. */
/* Worst case for one compressed block: the long header plus a maximum-size
 * block. The streamed decoder no longer sizes every slot this way -- it uses
 * header_size + fh->nominal_block_size, which is 16x smaller for a 4 KiB page
 * frame -- but this is the bound a block can never exceed, and the reference
 * for the "variable-size block in a page frame" guard in reader_fill. */
#define ZPSE_CLI_DEC_SLOT \
    ((size_t)ZPSE_BLOCK_HEADER_SIZE + (size_t)ZPSE_MAX_BLOCK_SIZE)

/* The output write, on its own thread.
 *
 * A streaming decode is three phases per batch -- read the frame, decode the
 * batch, write the batch -- and they take turns on one thread. Measured on
 * the 202 MiB Silesia frame at six workers: 62 ms to read the frame, 35 ms in
 * the codec, and about 100 ms to write the output, a third of the wall clock
 * that no worker count touches. So the write is handed to this thread and the
 * read of the next batch starts at once.
 *
 * Two slots is the whole queue: the writer only has to stay one batch behind, 
 * which it does -- a batch is 3 MiB at six workers and the write of one takes
 * less time than the read and decode of the next.
 *
 * Every slot's write order follows the block order, so the file gets the
 * frame's bytes in the frame's order even though the write runs beside the
 * decode. A short write is remembered and reported by the finish call, which
 * is when all writes have completed; the caller removes the partial output.
 * With ZPSE_MT undefined, or if the thread cannot be started, the same two
 * buffers are used and the write happens on the calling thread, so this is
 * not a second code path for correctness. */
#define ZPSE_CLI_WRITE_SLOTS 2
/* One output batch: the decode batches at most this many blocks (see the
 * clamp in decode_stream), and the writer needs their lengths per slot. A
 * page-nominal frame can afford a longer batch: each of its blocks is a
 * few KiB rather than 64 KiB, so the same handful of MiB holds many more of
 * them and the handoff is paid for less often. */
#define ZPSE_CLI_DEC_BATCH_MAX 64u
#define ZPSE_CLI_DEC_BATCH_MAX_PAGE 256u

typedef struct {
    FILE *f;
    const char *path;
    uint8_t *buf[ZPSE_CLI_WRITE_SLOTS];
    /* One length array per slot, sized to the batch at writer_init. The old
     * fixed [2][64] array capped the batch and wasted space once the cap was
     * raised for pages; the slots are also the only place the lengths are
     * needed, so they live with the slot, not with the scratch buffer. */
    uint32_t *got[ZPSE_CLI_WRITE_SLOTS];
    unsigned nblk[ZPSE_CLI_WRITE_SLOTS];
    /* 0 free, 1 filled (the writer's to pick up), 2 being written. */
    int state[ZPSE_CLI_WRITE_SLOTS];
    /* The order the batches were handed over. The writer writes the oldest
     * filled slot, not the lowest-numbered one: a slot can be released and
     * filled again while an earlier batch is still waiting, and picking by
     * number would then write the newer batch first. */
    uint32_t seq[ZPSE_CLI_WRITE_SLOTS];
    uint32_t next_seq;
    unsigned cur;
    size_t stride; /* nominal block size: the spacing of blocks in each slot */
    unsigned batch; /* number of length entries allocated per slot */
    int quit;
    int error;
    int async;
#if defined(ZPSE_MT)
#if defined(_WIN32)
    CRITICAL_SECTION mu;
    CONDITION_VARIABLE cv;
    HANDLE th;
#else
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t th;
#endif
#endif
} ZpseCliWriter;

#if defined(ZPSE_MT)
#if defined(_WIN32)
#define ZPSE_CLI_WLOCK(w) EnterCriticalSection(&(w)->mu)
#define ZPSE_CLI_WUNLOCK(w) LeaveCriticalSection(&(w)->mu)
#define ZPSE_CLI_WWAIT(w) \
    SleepConditionVariableCS(&(w)->cv, &(w)->mu, INFINITE)
#define ZPSE_CLI_WSIGNAL(w) WakeAllConditionVariable(&(w)->cv)
#else
#define ZPSE_CLI_WLOCK(w) pthread_mutex_lock(&(w)->mu)
#define ZPSE_CLI_WUNLOCK(w) pthread_mutex_unlock(&(w)->mu)
#define ZPSE_CLI_WWAIT(w) pthread_cond_wait(&(w)->cv, &(w)->mu)
#define ZPSE_CLI_WSIGNAL(w) pthread_cond_broadcast(&(w)->cv)
#endif

/* The writer's loop: take a filled slot, write it, release it. A submitted
 * slot is always written, even after `quit` -- quit only ends the loop once
 * there is nothing left to write. */
static void zpse_cli_writer_run(ZpseCliWriter *w)
{
    ZPSE_CLI_WLOCK(w);
    for (;;) {
        unsigned s = ZPSE_CLI_WRITE_SLOTS;
        uint32_t best = 0u;
        unsigned k;
        int err = 0;
        int may_write;

        for (k = 0; k < ZPSE_CLI_WRITE_SLOTS; k++) {
            if (w->state[k] == 1 && (s == ZPSE_CLI_WRITE_SLOTS ||
                                     w->seq[k] < best)) {
                s = k;
                best = w->seq[k];
            }
        }
        if (s == ZPSE_CLI_WRITE_SLOTS) {
            if (w->quit) {
                break;
            }
            ZPSE_CLI_WWAIT(w);
            continue;
        }
        w->state[s] = 2;
        /* Read the error flag under the lock, beside the state change. It is
         * only ever set under this lock (below), so sampling it while holding
         * it keeps the write/skip decision well-defined against the finish
         * path instead of racing an unlocked read. */
        may_write = (w->error == 0);
        ZPSE_CLI_WUNLOCK(w);
        if (may_write) {
            size_t total = 0u;
            int dense = 1;

            /* Blocks are laid out at k * stride in the slot. When every full
             * block decoded to exactly `stride` (only the tail may be short)
             * the whole batch is contiguous and one fwrite does the work of
             * tens of thousands of page-sized ones. */
            for (k = 0u; k < w->nblk[s]; k++) {
                uint32_t n = w->got[s][k];

                if (k + 1u < w->nblk[s] && n != (uint32_t)w->stride) {
                    dense = 0;
                    break;
                }
                total += (size_t)n;
            }
            if (dense) {
                if (total > 0u && fwrite(w->buf[s], 1u, total, w->f) != total) {
                    err = 1;
                }
            } else {
                for (k = 0u; k < w->nblk[s]; k++) {
                    uint32_t n = w->got[s][k];

                    if (n > 0u && fwrite(w->buf[s] + (size_t)k * w->stride,
                                         1u, (size_t)n, w->f) != (size_t)n) {
                        err = 1;
                        break;
                    }
                }
            }
        }
        ZPSE_CLI_WLOCK(w);
        w->state[s] = 0;
        if (err != 0) {
            w->error = 1;
        }
        ZPSE_CLI_WSIGNAL(w);
    }
    ZPSE_CLI_WUNLOCK(w);
}

#if defined(_WIN32)
static DWORD WINAPI zpse_cli_writer_entry(LPVOID p)
{
    zpse_cli_writer_run((ZpseCliWriter *)p);
    return 0;
}
#else
static void *zpse_cli_writer_entry(void *p)
{
    zpse_cli_writer_run((ZpseCliWriter *)p);
    return NULL;
}
#endif
#endif /* ZPSE_MT */

/* Allocate the output batches and, when threads exist, start the writer.
 * `slot_bytes` is one whole batch of output, `batch` the block count in it
 * and `stride` the nominal block size (the spacing of blocks in a slot). */
static int writer_init(ZpseCliWriter *w, FILE *f, const char *path,
                       size_t slot_bytes, unsigned batch, size_t stride,
                       int async)
{
    unsigned i;

    memset(w, 0, sizeof *w);
    w->f = f;
    w->path = path;
    w->batch = batch;
    w->stride = stride;
    for (i = 0; i < ZPSE_CLI_WRITE_SLOTS; i++) {
        w->buf[i] = (uint8_t *)malloc(slot_bytes);
        w->got[i] = (uint32_t *)malloc((size_t)batch * sizeof *w->got[i]);
        if (w->buf[i] == NULL || w->got[i] == NULL) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            free(w->buf[i]); /* free(NULL) is safe: either side may have failed */
            free(w->got[i]);
            while (i-- > 0u) {
                free(w->buf[i]);
                free(w->got[i]);
            }
            return 1;
        }
    }
#if defined(ZPSE_MT)
    if (async != 0) {
#if defined(_WIN32)
        InitializeCriticalSection(&w->mu);
        InitializeConditionVariable(&w->cv);
        w->th = CreateThread(NULL, 0, zpse_cli_writer_entry, w, 0, NULL);
        if (w->th != NULL) {
            w->async = 1;
        } else {
            DeleteCriticalSection(&w->mu);
        }
#else
        if (pthread_mutex_init(&w->mu, NULL) == 0) {
            if (pthread_cond_init(&w->cv, NULL) == 0) {
                if (pthread_create(&w->th, NULL, zpse_cli_writer_entry, w) ==
                    0) {
                    w->async = 1;
                } else {
                    pthread_cond_destroy(&w->cv);
                    pthread_mutex_destroy(&w->mu);
                }
            } else {
                pthread_mutex_destroy(&w->mu);
            }
        }
#endif
    }
#else
    (void)async;
#endif
    return 0;
}

/* The buffer to decode the next batch into. Waits for the writer to release
 * it, which is one batch of slack: the writer is writing the previous one. */
static uint8_t *writer_reserve(ZpseCliWriter *w)
{
#if defined(ZPSE_MT)
    if (w->async != 0) {
        ZPSE_CLI_WLOCK(w);
        while (w->state[w->cur] != 0) {
            ZPSE_CLI_WWAIT(w);
        }
        ZPSE_CLI_WUNLOCK(w);
    }
#endif
    return w->buf[w->cur];
}

/* Hand the decoded batch over and pick the other slot for the next one. */
static int writer_commit(ZpseCliWriter *w, const uint32_t *got, unsigned n)
{
    unsigned k;

#if defined(ZPSE_MT)
    if (w->async != 0) {
        ZPSE_CLI_WLOCK(w);
        for (k = 0; k < n; k++) {
            w->got[w->cur][k] = got[k];
        }
        w->nblk[w->cur] = n;
        w->seq[w->cur] = w->next_seq++;
        w->state[w->cur] = 1;
        ZPSE_CLI_WSIGNAL(w);
        w->cur = (w->cur + 1u) % ZPSE_CLI_WRITE_SLOTS;
        ZPSE_CLI_WUNLOCK(w);
        return 0;
    }
#endif
    {
        size_t total = 0u;
        int dense = 1;

        for (k = 0u; k < n; k++) {
            if (k + 1u < n && got[k] != (uint32_t)w->stride) {
                dense = 0;
                break;
            }
            total += (size_t)got[k];
        }
        if (dense) {
            if (total > 0u && fwrite(w->buf[0], 1u, total, w->f) != total) {
                fprintf(stderr, "zpse_cli: short write on %s\n", w->path);
                return 1;
            }
        } else {
            for (k = 0u; k < n; k++) {
                if (got[k] > 0u && fwrite(w->buf[0] + (size_t)k * w->stride,
                                          1u, (size_t)got[k], w->f) !=
                                       (size_t)got[k]) {
                    fprintf(stderr, "zpse_cli: short write on %s\n", w->path);
                    return 1;
                }
            }
        }
    }
    return 0;
}

/* Wait for every submitted batch to be written and free the buffers. Returns
 * non-zero if a write failed at any point. */
static int writer_finish(ZpseCliWriter *w)
{
    unsigned i;
    int err;

#if defined(ZPSE_MT)
    if (w->async != 0) {
        ZPSE_CLI_WLOCK(w);
        w->quit = 1;
        ZPSE_CLI_WSIGNAL(w);
        ZPSE_CLI_WUNLOCK(w);
#if defined(_WIN32)
        WaitForSingleObject(w->th, INFINITE);
        CloseHandle(w->th);
        DeleteCriticalSection(&w->mu);
#else
        pthread_join(w->th, NULL);
        pthread_cond_destroy(&w->cv);
        pthread_mutex_destroy(&w->mu);
#endif
        w->async = 0;
    }
#endif
    err = w->error;
    if (err != 0) {
        fprintf(stderr, "zpse_cli: short write on %s\n", w->path);
    }
    for (i = 0; i < ZPSE_CLI_WRITE_SLOTS; i++) {
        free(w->buf[i]);
        free(w->got[i]);
        w->buf[i] = NULL;
        w->got[i] = NULL;
    }
    return err;
}

/* Decode the blocks held in the batch and hand them to the writer, in order.
 *
 * This is what makes -t mean something on a decode that is streaming: the
 * caller reads the blocks, the workers decode them side by side, and the
 * writer emits them in order. Memory is the batch, not the frame. Returns 0
 * on success, 1 after reporting (the caller removes the partial output). */
static int decode_flush(uint8_t *slot_mem, const uint32_t *off,
                        const uint32_t *len, const uint32_t *expect,
                        uint32_t *got, ZpseCliWriter *w, unsigned held,
                        unsigned threads, void *const arenas[],
                        size_t *out_pos, size_t stride, ZpseBatchPool *pool)
{
    uint8_t *out_buf;
    unsigned k;
    int rc;

    if (held == 0u) {
        return 0;
    }
    out_buf = writer_reserve(w);
    /* A page stream runs tens of thousands of batches; creating threads for
     * each one cost more than decoding the pages. The persistent pool keeps
     * the same workers alive for the whole stream. The one-shot call is the
     * fallback when a pool could not be started. */
    if (pool != NULL) {
        rc = zpse_batch_pool_run(pool, slot_mem, off, len, held, out_buf,
                                 (uint32_t)stride, got);
    } else {
        rc = zpse_decompress_batch_parallel(slot_mem, off, len, held, out_buf,
                                            (uint32_t)stride, got,
                                            threads, arenas, threads);
    }
    if (rc != ZPSE_OK) {
        fprintf(stderr, "zpse_cli: block does not decode\n");
        return 1;
    }
    for (k = 0u; k < held; k++) {
        if (got[k] != expect[k]) {
            fprintf(stderr, "zpse_cli: block does not decode\n");
            return 1;
        }
        *out_pos += (size_t)got[k];
    }
    return writer_commit(w, got, held);
}

/* The input side of a streamed decode, on its own thread.
 *
 * A streamed decode is three phases per batch -- read the frame, decode the
 * batch, write the batch -- and each used to take its turn on the caller. The
 * write runs beside the decode now (see the writer above); this does the same
 * for the read. The read is what is left once the write is hidden: on the
 * 202 MiB Silesia corpus the 4 KiB page frame streams at 500 ms and takes 421
 * with this reader (read and decode alone: 305), and the 64 KiB block frame
 * at one worker goes from 205 to 185 ms. Where the write is bigger than the
 * read -- 64 KiB at six workers, already on its floor -- it changes nothing,
 * which is why the fallback below costs nothing to keep.
 *
 * A reader cannot read ahead without parsing, because a block's length lives
 * in its header, so the reader parses as it reads and fills one of two batch
 * slots while the caller decodes the other. Each block is read straight into
 * its place in the slot, which also drops the copy the walking loop used to
 * make from a stack buffer.
 *
 * Two slots are enough: the reader is never more than one batch ahead, and
 * decoding a batch takes longer than reading the next. Without ZPSE_MT, or if
 * the thread cannot be started, the same fill runs on the calling thread
 * before each decode, so this is not a second code path for correctness.
 */
#define ZPSE_CLI_READ_SLOTS 2

typedef struct {
    uint8_t *mem;
    uint32_t *off;
    uint32_t *len;
    uint32_t *expect;
    unsigned n;
    /* The order the reader filled the slots: the caller decodes the oldest
     * filled one, not the lowest-numbered, because the reader can refill a
     * slot that was just released while an earlier batch is still waiting. */
    uint32_t seq;
    /* 0 free, 1 filled (the caller's to decode), 2 being filled or decoded. */
    int state;
} ZpseCliInBatch;

typedef struct {
    FILE *fi;
    const char *in_path;
    unsigned batch;
    uint32_t nominal; /* frame's nominal block size: the decode/output stride */
    /* Spacing of consecutive blocks inside a read slot: header_size + nominal
     * (16 + nominal for a page frame, 32 + nominal otherwise). */
    size_t slot_stride;
    ZpseCliInBatch b[ZPSE_CLI_READ_SLOTS];
    uint32_t *blk_off; /* the frame's index entries, as the walk meets them */
    uint32_t *blk_len;
    size_t blk_pos;
    size_t nblocks;
    uint64_t declared; /* the block count the frame declares, if it does */
    int has_count;
    int have_last;
    int eof;
    int done;
    int quit;
    int error;
    int async;
    uint32_t next_seq;
#if defined(ZPSE_MT)
#if defined(_WIN32)
    CRITICAL_SECTION mu;
    CONDITION_VARIABLE cv;
    HANDLE th;
#else
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t th;
#endif
#endif
} ZpseCliReader;

#if defined(ZPSE_MT)
#if defined(_WIN32)
#define ZPSE_CLI_RLOCK(r) EnterCriticalSection(&(r)->mu)
#define ZPSE_CLI_RUNLOCK(r) LeaveCriticalSection(&(r)->mu)
#define ZPSE_CLI_RWAIT(r) \
    SleepConditionVariableCS(&(r)->cv, &(r)->mu, INFINITE)
#define ZPSE_CLI_RSIGNAL(r) WakeAllConditionVariable(&(r)->cv)
#else
#define ZPSE_CLI_RLOCK(r) pthread_mutex_lock(&(r)->mu)
#define ZPSE_CLI_RUNLOCK(r) pthread_mutex_unlock(&(r)->mu)
#define ZPSE_CLI_RWAIT(r) pthread_cond_wait(&(r)->cv, &(r)->mu)
#define ZPSE_CLI_RSIGNAL(r) pthread_cond_broadcast(&(r)->cv)
#endif
#endif /* ZPSE_MT */

/* Read blocks into one slot until it is full, the frame ends, or a block is
 * bad. Runs on the reader thread when there is one, on the caller otherwise.
 * The checks and their messages are the ones the walking loop always made;
 * a slot is never decoded before this returns. */
static void reader_fill(ZpseCliReader *r, ZpseCliInBatch *b)
{
    b->n = 0u;
    /* The last block ended an earlier batch: an indexed frame has its table
     * behind it, and reading on would take that table for a block. */
    if (r->eof != 0 || r->have_last != 0) {
        return;
    }
    while (b->n < r->batch) {
        ZpseBlockHeader bh;
        uint32_t comp_sz;
        uint32_t hdr_sz;
        uint8_t *slot = b->mem + (size_t)b->n * r->slot_stride;
        size_t got =
            fread(slot, 1u, (size_t)ZPSE_BLOCK_HEADER_SIZE_SHORT, r->fi);

        /* No bytes at a block boundary is either a clean EOF -- legal only
         * for a frame whose declared count agrees, which the caller checks
         * -- or an I/O error. They must not be confused: treating an error
         * as EOF would hand a truncated frame to the tail checks as if it
         * were complete. */
        if (got == 0u) {
            if (ferror(r->fi) != 0) {
                fprintf(stderr, "zpse_cli: short read on %s\n", r->in_path);
                r->error = 1;
            } else {
                r->eof = 1;
            }
            return;
        }
        /* The sync says which header this is: a page's 16-byte form (short,
         * token, FLAT or TEXT) is accepted here too, everything else the 32
         * the format has always had, so the short read above is enough to tell
         * and the rest is read here. */
        hdr_sz = zpse_block_header_size(slot, (uint32_t)got);
        if (got < (size_t)ZPSE_BLOCK_HEADER_SIZE_SHORT && ferror(r->fi) != 0) {
            fprintf(stderr, "zpse_cli: short read on %s\n", r->in_path);
            r->error = 1;
            return;
        }
        if (hdr_sz == 0u || got < (size_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            fprintf(stderr, "zpse_cli: bad block at block %zu\n", r->nblocks);
            r->error = 1;
            return;
        }
        if (hdr_sz > (uint32_t)got &&
            fread(slot + got, 1u, (size_t)(hdr_sz - (uint32_t)got), r->fi) !=
                (size_t)(hdr_sz - (uint32_t)got)) {
            fprintf(stderr, "zpse_cli: frame ends mid-header\n");
            r->error = 1;
            return;
        }
        /* The slot holds header_size + nominal bytes, not the old fixed
         * 64 KiB ZPSE_CLI_DEC_SLOT: a page frame saves 16x here. Blocks whose
         * uncompressed size is below nominal (the short tail, or a frame of
         * variable-size blocks) still fit, so the per-block loop below keeps
         * working. A block that cannot fit is refused rather than allowed to
         * overrun the slot. */
        if (zpse_block_header_parse(&bh, slot, hdr_sz) != ZPSE_OK ||
            bh.compressed_block_size < hdr_sz ||
            bh.compressed_block_size > (uint32_t)r->slot_stride ||
            bh.uncompressed_size < (uint32_t)ZPSE_MIN_BLOCK_SIZE ||
            bh.uncompressed_size > r->nominal ||
            bh.uncompressed_size > (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
            fprintf(stderr, "zpse_cli: bad block at block %zu\n", r->nblocks);
            r->error = 1;
            return;
        }
        comp_sz = bh.compressed_block_size;
        if (r->blk_off != NULL) {
            r->blk_off[r->nblocks] = (uint32_t)r->blk_pos;
            r->blk_len[r->nblocks] = comp_sz;
            r->blk_pos += (size_t)comp_sz;
        }
        if (comp_sz > hdr_sz &&
            fread(slot + hdr_sz, 1u, (size_t)(comp_sz - hdr_sz), r->fi) !=
                (size_t)(comp_sz - hdr_sz)) {
            fprintf(stderr, "zpse_cli: frame ends mid-block\n");
            r->error = 1;
            return;
        }
        b->off[b->n] = (uint32_t)((size_t)b->n * r->slot_stride);
        b->len[b->n] = comp_sz;
        b->expect[b->n] = bh.uncompressed_size;
        b->n++;
        r->nblocks++;
        if ((bh.block_flags & (uint8_t)ZPSE_B_LAST) != 0u) {
            r->have_last = 1;
        }
        if (r->have_last) {
            return;
        }
        if (r->has_count != 0 && (uint64_t)r->nblocks > r->declared) {
            fprintf(stderr, "zpse_cli: more blocks than the frame declares\n");
            r->error = 1;
            return;
        }
    }
}

#if defined(ZPSE_MT)
/* The reader's loop: fill a free slot, publish it, wait for a free one. */
static void zpse_cli_reader_body(ZpseCliReader *r)
{
    ZPSE_CLI_RLOCK(r);
    for (;;) {
        unsigned k;
        int s = -1;

        for (k = 0; k < ZPSE_CLI_READ_SLOTS; k++) {
            if (r->b[k].state == 0) {
                s = (int)k;
                break;
            }
        }
        if (s < 0) {
            if (r->quit != 0) {
                break;
            }
            ZPSE_CLI_RWAIT(r);
            continue;
        }
        if (r->eof != 0 || r->have_last != 0) {
            break; /* nothing left to read */
        }
        r->b[s].state = 2;
        ZPSE_CLI_RUNLOCK(r);
        reader_fill(r, &r->b[s]);
        ZPSE_CLI_RLOCK(r);
        r->b[s].seq = r->next_seq++;
        r->b[s].state = (r->b[s].n > 0u && r->error == 0) ? 1 : 0;
        ZPSE_CLI_RSIGNAL(r);
        if (r->eof != 0 || r->error != 0 || r->have_last != 0) {
            break;
        }
    }
    r->done = 1;
    ZPSE_CLI_RSIGNAL(r);
    ZPSE_CLI_RUNLOCK(r);
}

#if defined(_WIN32)
static DWORD WINAPI zpse_cli_reader_entry(LPVOID p)
{
    zpse_cli_reader_body((ZpseCliReader *)p);
    return 0;
}
#else
static void *zpse_cli_reader_entry(void *p)
{
    zpse_cli_reader_body((ZpseCliReader *)p);
    return NULL;
}
#endif
#endif /* ZPSE_MT */

/* Allocate the batch slots and, when threads exist, start the reader. The
 * file is already positioned at the first block. `nominal` is the frame's
 * nominal block size: a page-sized one uses 16-byte block headers and 4 KiB
 * slots, everything else the 32-byte header and a slot per its nominal size. */
static int reader_start(ZpseCliReader *r, FILE *fi, const char *in_path,
                        unsigned batch, uint32_t nominal,
                        uint32_t *blk_off, uint32_t *blk_len,
                        uint64_t declared, int has_count, int async)
{
    unsigned i;
    /* Slots must hold the LARGEST header any block in the frame may carry,
     * not the first block's. A frame can mix forms: level-1 Silesia starts
     * with a 16-byte TEXT page and later has 32-byte raw blocks, so taking the
     * stride from block 0 gives 16 + nominal and refuses the first long-header
     * block (comp = 32 + nominal > stride). Sixteen bytes a page slot might
     * save are not worth a frame that will not decode. */
    uint32_t hdr = (uint32_t)ZPSE_BLOCK_HEADER_SIZE;

    memset(r, 0, sizeof *r);
    r->fi = fi;
    r->in_path = in_path;
    r->batch = batch;
    r->nominal = nominal;
    r->slot_stride = (size_t)hdr + (size_t)nominal;
    r->blk_off = blk_off;
    r->blk_len = blk_len;
    r->blk_pos = (size_t)ZPSE_FRAME_HEADER_SIZE;
    r->declared = declared;
    r->has_count = has_count;
    for (i = 0; i < ZPSE_CLI_READ_SLOTS; i++) {
        r->b[i].mem =
            (uint8_t *)malloc((size_t)batch * r->slot_stride);
        r->b[i].off = (uint32_t *)malloc((size_t)batch * sizeof *r->b[i].off);
        r->b[i].len = (uint32_t *)malloc((size_t)batch * sizeof *r->b[i].len);
        r->b[i].expect =
            (uint32_t *)malloc((size_t)batch * sizeof *r->b[i].expect);
        if (r->b[i].mem == NULL || r->b[i].off == NULL ||
            r->b[i].len == NULL || r->b[i].expect == NULL) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            free(r->b[i].mem); /* free(NULL) is safe: any one may have failed */
            free(r->b[i].off);
            free(r->b[i].len);
            free(r->b[i].expect);
            while (i-- > 0u) {
                free(r->b[i].mem);
                free(r->b[i].off);
                free(r->b[i].len);
                free(r->b[i].expect);
            }
            return 1;
        }
    }
#if defined(ZPSE_MT)
    if (async != 0) {
#if defined(_WIN32)
        InitializeCriticalSection(&r->mu);
        InitializeConditionVariable(&r->cv);
        r->th = CreateThread(NULL, 0, zpse_cli_reader_entry, r, 0, NULL);
        if (r->th != NULL) {
            r->async = 1;
        } else {
            DeleteCriticalSection(&r->mu);
        }
#else
        if (pthread_mutex_init(&r->mu, NULL) == 0) {
            if (pthread_cond_init(&r->cv, NULL) == 0) {
                if (pthread_create(&r->th, NULL, zpse_cli_reader_entry, r) ==
                    0) {
                    r->async = 1;
                } else {
                    pthread_cond_destroy(&r->cv);
                    pthread_mutex_destroy(&r->mu);
                }
            } else {
                pthread_mutex_destroy(&r->mu);
            }
        }
#endif
    }
#else
    (void)async;
#endif
    return 0;
}

/* The next filled slot, or -1 when the frame has been walked. With a thread
 * this waits for the reader; without one it fills slot 0 right here. */
static int reader_take(ZpseCliReader *r)
{
#if defined(ZPSE_MT)
    if (r->async != 0) {
        int s;
        ZPSE_CLI_RLOCK(r);
        for (;;) {
            int pick = -1;
            uint32_t best = 0u;

            for (s = 0; s < (int)ZPSE_CLI_READ_SLOTS; s++) {
                if (r->b[s].state == 1 &&
                    (pick < 0 || r->b[s].seq < best)) {
                    pick = s;
                    best = r->b[s].seq;
                }
            }
            if (pick >= 0) {
                r->b[pick].state = 2; /* being decoded */
                ZPSE_CLI_RUNLOCK(r);
                return pick;
            }
            if (r->done != 0) {
                ZPSE_CLI_RUNLOCK(r);
                return -1;
            }
            ZPSE_CLI_RWAIT(r);
        }
    }
#endif
    if (r->done != 0) {
        return -1;
    }
    reader_fill(r, &r->b[0]);
    if (r->b[0].n == 0u) {
        r->done = 1;
        return -1;
    }
    r->b[0].seq = r->next_seq++;
    return 0;
}

/* Give a decoded slot back to the reader. */
static void reader_release(ZpseCliReader *r, int s)
{
#if defined(ZPSE_MT)
    if (r->async != 0) {
        ZPSE_CLI_RLOCK(r);
        r->b[s].state = 0;
        ZPSE_CLI_RSIGNAL(r);
        ZPSE_CLI_RUNLOCK(r);
        return;
    }
#endif
    (void)r;
    (void)s;
}

/* Wait for the reader and free the slots. The walk's counts in `r` stay
 * readable, which is what the caller checks its frame against. */
static void reader_stop(ZpseCliReader *r)
{
    unsigned i;

#if defined(ZPSE_MT)
    if (r->async != 0) {
        ZPSE_CLI_RLOCK(r);
        r->quit = 1;
        ZPSE_CLI_RSIGNAL(r);
        ZPSE_CLI_RUNLOCK(r);
#if defined(_WIN32)
        WaitForSingleObject(r->th, INFINITE);
        CloseHandle(r->th);
        DeleteCriticalSection(&r->mu);
#else
        pthread_join(r->th, NULL);
        pthread_cond_destroy(&r->cv);
        pthread_mutex_destroy(&r->mu);
#endif
        r->async = 0;
    }
#endif
    for (i = 0; i < ZPSE_CLI_READ_SLOTS; i++) {
        free(r->b[i].mem);
        free(r->b[i].off);
        free(r->b[i].len);
        free(r->b[i].expect);
        r->b[i].mem = NULL;
    }
}

static int decode_stream(const char *in_path, const char *out_path,
                         const ZpseFrameHeader *fh,
                         void *const dec_arenas[], unsigned threads,
                         int indexed)
{
    FILE *f = NULL;
    FILE *fi = NULL;
    ZpseCliReader rd;
    int reader_ready = 0;
    ZpseCliWriter writer;
    int writer_ready = 0;
    ZpseBatchPool *pool = NULL;
    void *pool_scratch = NULL;
    uint32_t *slot_got = NULL;
    /* The stdio buffers at both ends; both have to outlive their close. */
    char *out_buf = NULL;
    char *in_buf = NULL;
    size_t io_buf_len = 256u << 10;
    /* Eight blocks a worker: a batch wants enough work per thread to pay for
     * the handoff, and the ring the workers fill is a few MiB either way. */
    unsigned batch = threads * 8u;
    /* The frame's nominal block size sizes every slot in the streamed path:
     * a page frame uses 16-byte headers and 4 KiB blocks, so the old fixed
     * 64 KiB DEC_SLOT per block was a 16x waste. `stride` is also the spacing
     * the parallel decoder uses for the output batch. */
    uint32_t nominal = fh->nominal_block_size;
    int page = nominal <= (uint32_t)ZPSE_BLOCK_SHORT_MAX;
    size_t stride = (size_t)nominal;
    size_t out_pos = 0u;
    int has_count = (fh->frame_flags & (uint16_t)ZPSE_F_BLOCK_COUNT) != 0u;
    int has_content = (fh->frame_flags & (uint16_t)ZPSE_F_CONTENT_SIZE) != 0u;
    /* Where each block this walk visits starts and how long it is, so the
     * table can be checked against the blocks that are really there. One
     * entry to spare, so a frame that holds more blocks than it declares is
     * recorded rather than overrun before it is refused. */
    uint32_t *blk_off = NULL;
    uint32_t *blk_len = NULL;

    fi = fopen(in_path, "rb");
    if (fi == NULL) {
        fprintf(stderr, "zpse_cli: cannot open %s\n", in_path);
        return 1;
    }
    /* The reader asks for a 16..32 byte header and then a payload behind it;
     * a page fills the default buffer exactly, so each one costs a read of
     * its own. A quarter mebibyte lets the payload come out of the buffer.
     * The encoding side measured that default at 107 / 93 ms against 99 / 85
     * for this size on mozilla / webster at eight workers, and the decode
     * reads the same bytes through the same stream: on the 170 MB page frame
     * this buffer alone took the streamed decode from 297 / 431 ms to 224 /
     * 360 at one worker and six, because the read is the slow side there. */
    in_buf = (char *)malloc(io_buf_len);
    if (in_buf != NULL) {
        setvbuf(fi, in_buf, _IOFBF, io_buf_len);
    }
    if (indexed) {
        size_t entries = (size_t)fh->block_count + 1u;

        blk_off = (uint32_t *)malloc(entries * sizeof *blk_off);
        blk_len = (uint32_t *)malloc(entries * sizeof *blk_len);
        if (blk_off == NULL || blk_len == NULL) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            goto fail;
        }
    }
    if (fseek(fi, (long)ZPSE_FRAME_HEADER_SIZE, SEEK_SET) != 0) {
        fprintf(stderr, "zpse_cli: cannot seek %s\n", in_path);
        goto fail;
    }
    f = fopen(out_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "zpse_cli: cannot write %s\n", out_path);
        goto fail;
    }
    /* The same 4 KiB default sits on the output: a 64 KiB block leaves as
     * sixteen writes and every page as one, from the writer thread. With the
     * output buffer alone the 170 MB page frame went from 420 / 514 to 297 /
     * 431 ms at one worker and six; the 102 MB block frame did not move,
     * because a 64 KiB write already bypassed the old buffer. The buffer must
     * outlive the close, which happens after the writer has finished, so it
     * is freed beside the stream. */
    out_buf = (char *)malloc(io_buf_len);
    if (out_buf != NULL) {
        setvbuf(f, out_buf, _IOFBF, io_buf_len);
    }
    /* A page batch is a handful of MiB even with a much larger cap, so raise
     * it there and keep the 64 KiB-frame cap that bounds the ring at a few
     * MiB. The default one worker gives batch = 8 either way, so the README's
     * single-core figures are unaffected. */
    if (batch > (page ? ZPSE_CLI_DEC_BATCH_MAX_PAGE : ZPSE_CLI_DEC_BATCH_MAX)) {
        batch = (page ? ZPSE_CLI_DEC_BATCH_MAX_PAGE : ZPSE_CLI_DEC_BATCH_MAX);
    }
    /* One scratch array for the parallel call's decoded lengths; the writer
     * keeps its own per-slot copy once a batch is committed. */
    slot_got = (uint32_t *)malloc((size_t)batch * sizeof *slot_got);
    if (slot_got == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        goto fail;
    }
    if (reader_start(&rd, fi, in_path, batch, nominal, blk_off, blk_len,
                     (uint64_t)fh->block_count, has_count, 1) != 0) {
        goto fail;
    }
    reader_ready = 1;
    if (writer_init(&writer, f, out_path, (size_t)batch * stride, batch,
                    stride, 1) != 0) {
        goto fail;
    }
    writer_ready = 1;
#if defined(ZPSE_MT)
    /* Keep one set of decode workers for the whole stream. Creating threads
     * per batch on a page frame costs more than decoding the pages. Failure
     * is not fatal: decode_flush falls back to the one-shot call. */
    {
        size_t scr = ZPSE_POOL_SCRATCH_SIZE(threads);

        pool_scratch = malloc(scr);
        if (pool_scratch != NULL &&
            zpse_batch_pool_init(&pool, pool_scratch, scr, threads,
                                 dec_arenas, threads) != ZPSE_OK) {
            pool = NULL;
        }
    }
#endif
    for (;;) {
        int s = reader_take(&rd);
        if (s < 0) {
            break;
        }
        if (decode_flush(rd.b[s].mem, rd.b[s].off, rd.b[s].len,
                         rd.b[s].expect, slot_got, &writer, rd.b[s].n, threads,
                         dec_arenas, &out_pos, stride, pool) != 0) {
            goto fail;
        }
        reader_release(&rd, s);
    }
    /* Stop (and join) the reader before reading its flags: rd.error, done and
     * the block counts are written by the reader thread, so sampling them
     * while it can still run is a data race. reader_stop joins it first. */
    reader_stop(&rd);
    reader_ready = 0;
    if (rd.error != 0) {
        goto fail;
    }
    if (!rd.have_last &&
        (!has_count || (uint64_t)rd.nblocks != (uint64_t)fh->block_count)) {
        fprintf(stderr, "zpse_cli: block count disagrees with the frame\n");
        goto fail;
    }
    if (has_count && (uint64_t)rd.nblocks != (uint64_t)fh->block_count) {
        fprintf(stderr, "zpse_cli: block count disagrees with the frame\n");
        goto fail;
    }
    if (has_content && (uint64_t)out_pos != (uint64_t)fh->content_size) {
        fprintf(stderr, "zpse_cli: content size disagrees with the frame\n");
        goto fail;
    }
    /* The table a -x frame carries has to describe exactly the blocks this
     * walk just decoded, and the blocks have to end where it begins: a table
     * that disagrees with the stream is a corrupt frame, and a reader that
     * seeks by it would land somewhere else than this one did. The check runs
     * here, on the blocks that were really there, rather than against a copy
     * of the whole frame. */
    if (indexed) {
        size_t count = (size_t)fh->block_count;
        size_t need = (size_t)ZPSE_INDEX_HEADER_SIZE +
                      count * (size_t)ZPSE_INDEX_ENTRY_SIZE;
        uint8_t *idx = (uint8_t *)malloc(need);
        int vrc;

        if (idx == NULL) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            goto fail;
        }
        if (fseek(fi, (long)fh->frame_index_offset, SEEK_SET) != 0 ||
            fread(idx, 1u, need, fi) != need) {
            fprintf(stderr, "zpse_cli: frame ends before its index\n");
            free(idx);
            goto fail;
        }
        vrc = zpse_frame_index_validate(fh, idx, need, (uint32_t)rd.nblocks,
                                        blk_off, blk_len);
        free(idx);
        if (vrc != ZPSE_OK) {
            fprintf(stderr, "zpse_cli: index disagrees with the frame\n");
            goto fail;
        }
    }
    fclose(fi);
    if (pool != NULL) {
        zpse_batch_pool_free(pool);
        pool = NULL;
    }
    free(pool_scratch);
    pool_scratch = NULL;
    writer_ready = 0;
    if (writer_finish(&writer) != 0) {
        fclose(f);
        remove(out_path);
        free(out_buf);
        free(in_buf);
        free(slot_got);
        free(blk_off);
        free(blk_len);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "zpse_cli: close failed on %s\n", out_path);
        remove(out_path);
        free(out_buf);
        free(in_buf);
        free(slot_got);
        free(blk_off);
        free(blk_len);
        return 1;
    }
    free(out_buf);
    free(in_buf);
    free(slot_got);
    free(blk_off);
    free(blk_len);
    return 0;

fail:
    if (reader_ready != 0) {
        reader_stop(&rd);
    }
    if (pool != NULL) {
        zpse_batch_pool_free(pool);
        pool = NULL;
    }
    free(pool_scratch);
    pool_scratch = NULL;
    if (writer_ready != 0) {
        (void)writer_finish(&writer);
    }
    if (f != NULL) {
        fclose(f);
        remove(out_path);
    }
    if (fi != NULL) {
        fclose(fi);
    }
    free(out_buf);
    free(in_buf);
    free(slot_got);
    free(blk_off);
    free(blk_len);
    return 1;
}

/* Encode a frame block by block, writing each block as it is finished.
 *
 * One encoder is initialised for the whole run and handed every block, as
 * the frame path does: levels 5 and up carry an order-n history of the shapes
 * that won the last blocks to decide which trial runs first, and a one-shot
 * call per block would reset it and write a different (larger) stream. The
 * stream here is the buffered path's, byte for byte.
 *
 * The buffered path holds the input and a frame-sized output buffer at once:
 * on the 202 MiB Silesia corpus that is a 202 MiB read into fresh pages plus
 * a 424 MiB allocation, and against the same encode with both already
 * resident it costs a quarter of the wall time. Nothing about the format
 * needs it -- each block is independent and carries its own compressed size
 * in its header -- so this walks the blocks and writes them out of one reused
 * buffer, exactly as the decoder's decode_stream already does. The input is
 * mapped rather than read into a slab, because the encoder only ever reads
 * it forwards and the copy was a third of the run on a corpus-sized file.
 *
 * The frame header is written first and is the same one zpse_frame_encode
 * writes: the content size and the block count are known from the input's
 * size, so nothing has to be patched afterwards. A frame written here is
 * byte for byte the frame the buffered path writes.
 *
 * Returns 0 on success, 1 on any failure, matching write_file. */
/* Where a streamed encode's blocks go: the output file and, when the frame
 * carries an index, the table the streamed writer records as each block is
 * written, so nothing has to be read back to build it. */
typedef struct StreamSink {
    FILE *f;
    const char *path;
    size_t pos;
    uint32_t *off;
    uint32_t *len;
    int failed;
} StreamSink;

static int stream_emit(void *ctx, uint32_t seq, const uint8_t *blk, size_t len)
{
    StreamSink *s = (StreamSink *)ctx;

    if (s->failed) {
        return ZPSE_BAD_ARG;
    }
    if (len > 0u && fwrite(blk, 1u, len, s->f) != len) {
        fprintf(stderr, "zpse_cli: short write on %s\n", s->path);
        s->failed = 1;
        return ZPSE_BAD_ARG;
    }
    if (s->off != NULL) {
        s->off[seq] = (uint32_t)s->pos;
        s->len[seq] = (uint32_t)len;
    }
    s->pos += len;
    return 0;
}

/* Cheap ratio-safety gate for the CLI's FLAT7 emit path. FLAT7 packs 7-bit
 * ASCII at a fixed ~1.143:1 and never runs the LZ/TEXT finder, so feeding it
 * redundant ASCII is a catastrophic ratio regression: 64 KiB of repeated
 * lines shrinks only to ~57 KiB where the normal frame reaches tens-to-one.
 * The API auto-gate was removed for determinism, so this decision has to be
 * made here, from the bytes alone.
 *
 * Sample three 4 KiB windows (front/middle/back) and count repeated 4-byte
 * grams in a 1024-entry direct-mapped cache that stores the full 32-bit key.
 * A repeat is counted only when the exact key returns to the same slot, so
 * collisions between different grams -- inevitable with 12 KiB of input in
 * 1024 slots -- are overwritten, not miscounted as repeats. Random printable
 * ASCII has ~95^4 possible grams and almost never repeats; redundant ASCII
 * repeats constantly. The probe reads at most 12 KiB, is a pure function of
 * the input bytes, and needs no floating point.
 *
 * Measured on this tree with this exact probe (repeat rate over the three
 * windows, 1024 slots):
 *   random printable 100 KiB     0.000  -> allow, 1.143:1 is the best there is
 *   random alphanumeric 100 KiB  0.000  -> allow
 *   webster (LZ 2.10:1)          0.339  -> decline
 *   nci     (LZ 5.43:1)          0.870  -> decline
 *   64 KiB repeated ASCII lines  0.901  -> decline
 *   English-like prose           0.789  -> decline
 * The 10% cutoff below sits ~3.4x under the least-redundant real case
 * (webster) and infinitely above every random case, so it separates them
 * cleanly. It is deliberately conservative about emitting FLAT7: a false
 * decline costs one normal frame, a false emit costs a ratio blowup.
 * Limitation, stated honestly: a file whose three sampled windows look
 * random but whose body is redundant can slip through -- no 12 KiB probe can
 * rule that out without the full match-finder pass this gate exists to skip.
 */
static int cli_flat7_ratio_gate(const uint8_t *in, size_t n)
{
    enum { WIN = 4096u, SLOTS = 1024u };
    uint32_t cache[SLOTS];
    uint8_t seen[SLOTS];
    size_t off[3], len[3], nwin, s, i, grams = 0u, reps = 0u;

    memset(seen, 0, sizeof seen);
    if (n >= 3u * WIN) {
        off[0] = 0u;                  len[0] = WIN;
        off[1] = n / 2u - WIN / 2u;   len[1] = WIN;
        off[2] = n - WIN;             len[2] = WIN;
        nwin = 3u;
    } else {
        off[0] = 0u; len[0] = n; nwin = 1u; /* short input: all of it */
    }
    for (s = 0u; s < nwin; s++) {
        for (i = 0u; i + 4u <= len[s]; i++) {
            const uint8_t *p = in + off[s] + i;
            uint32_t k = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            uint32_t h = (k * 0x9E3779B1u) >> 22u; /* top 10 bits -> 0..1023 */
            if (seen[h] && cache[h] == k) {
                reps++;
            }
            cache[h] = k;
            seen[h] = 1u;
            grams++;
        }
    }
    /* Decline above a 10% repeat rate; grams <= 3*4096, so no overflow. */
    return grams == 0u || reps * 10u <= grams;
}

static int encode_stream(const char *in_path, const char *out_path,
                         uint32_t block, uint32_t enc_flags, int indexed,
                         unsigned threads, void *const arenas[])
{
    ZpseFrameHeader fh;
    FILE *fi;
    FILE *fo = NULL;
    StreamSink sink;
    uint8_t *in_buf = NULL;
    const uint8_t *in_map = NULL;
    uint8_t *slot_mem = NULL;
    int *ready = NULL;
    uint32_t *lens = NULL;
#if defined(ZPSE_CLI_HAVE_ENCODE_POOL)
    /* One pool for the whole stream: the workers stay alive across slabs, so
     * a page stream does not pay a thread start/join per 64-block slab. Its
     * scratch and the per-worker arenas are caller-owned and live to the end
     * of this function. */
    ZpseEncodePool *enc_pool = NULL;
    void *enc_pool_scratch = NULL;
#endif
    uint8_t hdr[ZPSE_FRAME_HEADER_SIZE];
    /* The output buffer handed to setvbuf, freed after the close. */
    char *out_buf = NULL;
    size_t out_buf_len = 256u << 10;
    /* The ring is ZPSE_MT_MAX_DEPTH slots per worker, which is what lets a
     * worker run ahead of the writer instead of blocking on an emit it has
     * nothing to do with; `zpse_compress_frame_stream` clamps the depth to
     * whatever `slot_cap` actually holds, so a short allocation still runs
     * (shallower, same bytes). Control entries are padded to a cache line,
     * one group per slot: six workers on one line evicted each other's polls. */
    size_t stride = (size_t)block + (size_t)ZPSE_BLOCK_HEADER_SIZE;
    unsigned ring = (threads > 1u ? threads * ZPSE_MT_MAX_DEPTH : 1u);
    size_t slot_cap = stride * (size_t)ring;
    size_t ctl = (size_t)ring * (size_t)ZPSE_MT_CTRL_STRIDE;
    /* A slab is read, encoded and written before the next one is read: 64
     * blocks is 4 MiB at the default block size, enough to keep the workers
     * fed and small enough that the input stays in cache. */
    size_t slab_blocks = 64u;
    long end;
    size_t n;
    size_t off = 0u;
    size_t nblocks;
    /* Where each block landed and how long it is, so a frame that asks for
     * an index can carry one without the blocks still being in memory. */
    uint32_t *blk_off = NULL;
    uint32_t *blk_len = NULL;
    size_t out_pos = ZPSE_FRAME_HEADER_SIZE;
    int rc = 1;

    fi = fopen(in_path, "rb");
    if (fi == NULL) {
        fprintf(stderr, "zpse_cli: cannot open %s\n", in_path);
        return 1;
    }
    if (fseek(fi, 0, SEEK_END) != 0 || (end = ftell(fi)) < 0 ||
        fseek(fi, 0, SEEK_SET) != 0) {
        fprintf(stderr, "zpse_cli: cannot size %s\n", in_path);
        fclose(fi);
        return 1;
    }
    n = (size_t)end;
    if (n == 0u) {
        fprintf(stderr, "zpse_cli: %s is empty\n", in_path);
        fclose(fi);
        return 1;
    }
    nblocks = n / (size_t)block + ((n % (size_t)block) != 0u ? 1u : 0u);
    if (nblocks > (size_t)UINT32_MAX ||
        zpse_frame_header_build(&fh, block, (uint64_t)n, 1,
                                (uint32_t)nblocks, 1, indexed) != ZPSE_OK ||
        zpse_frame_header_write(&fh, hdr) != ZPSE_OK) {
        fprintf(stderr, "zpse_cli: cannot build the frame header\n");
        fclose(fi);
        return 1;
    }
    fo = fopen(out_path, "wb");
    if (fo == NULL) {
        fprintf(stderr, "zpse_cli: cannot write %s\n", out_path);
        fclose(fi);
        return 1;
    }
    /* Each finished block is handed straight to `fwrite`, and the default
     * stdio buffer is 4 KiB (Windows) -- so a corpus-sized frame leaves as
     * thousands of little WriteFile calls, one per buffer fill, with the
     * block copied into the buffer first. A quarter mebibyte turns that into
     * one write per four blocks at the default block size, and 1 MiB and
     * 4 MiB measured no better (256 KiB: 99 / 85 ms on mozilla / webster at
     * eight workers, against 107 / 93 ms at the 4 KiB default), so the
     * smallest of the three wins. The bytes and their order are the same at
     * any buffer size. */
    out_buf = (char *)malloc(out_buf_len);
    if (out_buf != NULL) {
        setvbuf(fo, out_buf, _IOFBF, out_buf_len);
    }
    /* Map the input rather than copying it in. The slab read this used to
     * do faults a page every 4 KiB on a fresh buffer and then memcpys the
     * whole input: on the 68 MiB columnar corpus that is 30 ms of a 96 ms
     * encode, against 10 ms for `cat` on the same file, and nothing about
     * the encoder needs the copy -- it reads its input once, front to back.
     * The mapping is read-only, so a stray write faults instead of quietly
     * editing the caller's file. Where mmap is not available (_WIN32) the
     * slab read below still runs unchanged. */
#if !defined(_WIN32)
    {
        int fd = fileno(fi);

        if (fd >= 0) {
            int fl = MAP_PRIVATE;
            void *m;

#if defined(MAP_POPULATE)
            /* Prefault the whole mapping in one syscall: the encoder is
             * about to walk all of it, and 17 000 individual faults on a
             * corpus-sized file cost more than the batched fill does. */
            fl |= MAP_POPULATE;
#endif
            m = mmap(NULL, n, PROT_READ, fl, fd, 0);

            if (m != MAP_FAILED) {
                in_map = (const uint8_t *)m;
            }
        }
    }
#endif
    if (in_map == NULL) {
        in_buf = (uint8_t *)malloc((size_t)slab_blocks * (size_t)block);
    }
    /* FLAT7 is a one-shot whole-input ASCII codec with no block index, so it
     * bypasses the slab writer entirely. Decline -x (nothing to append an
     * index to) and -c (no per-block CRC field). Only the packed form (~7/8
     * of n) plus its header is written, straight to `fo`. */
    if (!indexed && (enc_flags & ZPSE_ENC_CHECKSUM) == 0u) {
        const uint8_t *flat_in = in_map;
        uint8_t *flat_owned = NULL;
        size_t flat_cap = n + 256u; /* packed form < n; slack for the header */
        uint8_t *flat_out = NULL;
        size_t flat_len = 0u;
        int frc;

        if (flat_in == NULL) {
            /* No mmap here: stage the whole input, since this codec needs it
             * contiguous rather than a slab at a time. */
            flat_owned = (uint8_t *)malloc(n);
            if (flat_owned == NULL || fseek(fi, 0, SEEK_SET) != 0 ||
                fread(flat_owned, 1u, n, fi) != n) {
                fprintf(stderr, "zpse_cli: cannot stage input for FLAT7\n");
                free(flat_owned);
                goto done;
            }
            flat_in = flat_owned;
        }
        /* Cheap redundancy probe first: a decline skips the fused encoder's
         * whole-input pass entirely. */
        if (cli_flat7_ratio_gate(flat_in, n)) {
            flat_out = (uint8_t *)malloc(flat_cap);
            if (flat_out == NULL) {
                fprintf(stderr, "zpse_cli: out of memory\n");
                free(flat_owned);
                goto done;
            }
            /* ASCII check + redundancy gate + 7-bit pack in one pass over the
             * staged input. The 40-byte frame is committed only if every gate
             * passes; 0 is a normal decline and takes the block path below. */
            frc = zpse_flat7_encode_stream(flat_in, n, flat_out, flat_cap,
                                           &flat_len);
            if (frc > 0) {
                if (fwrite(flat_out, 1u, flat_len, fo) != flat_len) {
                    fprintf(stderr, "zpse_cli: short write on %s\n", out_path);
                    free(flat_out);
                    free(flat_owned);
                    goto done; /* done: closes fo */
                }
                if (fclose(fo) != 0) {
                    fo = NULL;
                    fprintf(stderr, "zpse_cli: close failed on %s\n", out_path);
                    free(flat_out);
                    free(flat_owned);
                    goto done;
                }
                fo = NULL;
                free(flat_out);
                free(flat_owned);
                rc = 0;
                goto done;
            }
            if (frc < 0) {
                fprintf(stderr, "zpse_cli: FLAT7 encode failed rc=%d\n", frc);
                free(flat_out);
                free(flat_owned);
                goto done;
            }
            /* frc == 0: declined -- drop the scratch and take the normal path. */
            free(flat_out);
        }
        /* Not a FLAT7 frame: the slab writer below reads `fi` from offset 0.
         * Where there is no mmap (Windows) the staging fread above left the
         * file at EOF, so put it back before the first slab is read -- otherwise
         * the first fread returns 0 and the loop reports "short read" after
         * emitting only the 32-byte frame header. On the mmap path the file
         * position was never advanced, and rewinding there is harmless, so
         * both paths end up identical here. clearerr() also drops the EOF flag
         * a full-length staging read can leave set. */
        clearerr(fi);
        if (fseek(fi, 0, SEEK_SET) != 0) {
            fprintf(stderr, "zpse_cli: cannot rewind input after FLAT7 probe\n");
            free(flat_owned);
            goto done;
        }
        free(flat_owned);
    }
    slot_mem = (uint8_t *)malloc(slot_cap);
    ready = (int *)malloc(sizeof(int) * ctl);
    lens = (uint32_t *)malloc(sizeof(uint32_t) * ctl);
    if ((in_map == NULL && in_buf == NULL) || slot_mem == NULL ||
        ready == NULL || lens == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        goto done;
    }
    if (indexed) {
        blk_off = (uint32_t *)malloc(sizeof(uint32_t) * nblocks);
        blk_len = (uint32_t *)malloc(sizeof(uint32_t) * nblocks);
        if (blk_off == NULL || blk_len == NULL) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            goto done;
        }
    }
#if defined(ZPSE_CLI_HAVE_ENCODE_POOL)
    /* One set of encode workers for the whole stream. Only build it when the
     * workers are actually the point: at one worker (or with a thin slot_mem
     * that can only hold one slot per worker) the streamed call already runs
     * the serial encoder, and there is nothing for a pool to keep alive. A
     * failed init is not fatal: enc_pool stays NULL and the slab loop falls
     * back to the original streamed call. */
    if (threads > 1u && slot_cap >= stride * (size_t)threads) {
        size_t scr = ZPSE_ENCODE_POOL_SCRATCH_SIZE(threads);

        enc_pool_scratch = ZPSE_CLI_ALIGNED(64u, scr);
        if (enc_pool_scratch != NULL &&
            zpse_encode_pool_init(&enc_pool, enc_pool_scratch, scr, threads,
                                  arenas, threads) != ZPSE_OK) {
            enc_pool = NULL;
        }
    }
#endif
    if (fwrite(hdr, 1u, sizeof hdr, fo) != sizeof hdr) {
        fprintf(stderr, "zpse_cli: short write on %s\n", out_path);
        goto done;
    }
    sink.f = fo;
    sink.path = out_path;
    sink.pos = ZPSE_FRAME_HEADER_SIZE;
    sink.off = blk_off;
    sink.len = blk_len;
    sink.failed = 0;
    {
        size_t done_blocks = 0u;

        while (done_blocks < nblocks) {
            size_t want = nblocks - done_blocks;
            size_t room = want * (size_t)block;
            size_t got = n - off;
            const uint8_t *src = in_buf;
            int final;

            if (want > slab_blocks) {
                want = slab_blocks;
                room = want * (size_t)block;
            }
            if (got > room) {
                got = room;
            }
            if (got == 0u) {
                fprintf(stderr, "zpse_cli: short read on %s\n", in_path);
                goto done;
            }
            if (in_map != NULL) {
                src = in_map + off;
            } else if (fread(in_buf, 1u, got, fi) != got) {
                fprintf(stderr, "zpse_cli: short read on %s\n", in_path);
                goto done;
            }
            final = (done_blocks + want) >= nblocks ? 1 : 0;
            {
                int erc;

#if defined(ZPSE_CLI_HAVE_ENCODE_POOL)
                /* The pool keeps the same workers (and the same static
                 * block-to-worker assignment) across slabs, so the stream is
                 * byte for byte the one the per-slab call writes. Fall back to
                 * that call when no pool was built -- one worker, ZPSE_MT off,
                 * or a failed init -- which is the serial path unchanged. */
                if (enc_pool != NULL) {
                    erc = zpse_encode_pool_run(enc_pool, src, got, block,
                                               enc_flags,
                                               (uint32_t)done_blocks, final,
                                               slot_mem, slot_cap, ready, lens,
                                               stream_emit, &sink);
                } else
#endif
                {
                    erc = zpse_compress_frame_stream(
                        src, got, block, enc_flags, threads,
                        (uint32_t)done_blocks, final, arenas, threads,
                        slot_mem, slot_cap, ready, lens, stream_emit, &sink);
                }
                if (erc != ZPSE_OK || sink.failed != 0) {
                    fprintf(stderr, "zpse_cli: blocks at %zu do not encode\n",
                            done_blocks);
                    goto done;
                }
            }
            off += got;
            done_blocks += want;
        }
        out_pos = sink.pos;
    }
    if (indexed) {
        size_t idx_len = 0u;
        uint8_t *idx = (uint8_t *)malloc((size_t)ZPSE_INDEX_HEADER_SIZE +
                                         (size_t)ZPSE_INDEX_ENTRY_SIZE *
                                             nblocks);

        if (idx == NULL) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            goto done;
        }
        if (zpse_frame_stream_index(hdr, idx,
                                    (size_t)ZPSE_INDEX_HEADER_SIZE +
                                        (size_t)ZPSE_INDEX_ENTRY_SIZE *
                                            nblocks,
                                    block, (uint64_t)n, (uint32_t)nblocks,
                                    (uint32_t)out_pos, blk_off, blk_len,
                                    &idx_len) != ZPSE_OK) {
            fprintf(stderr, "zpse_cli: cannot build the frame index\n");
            free(idx);
            goto done;
        }
        /* The header was written before the blocks, when the index offset
         * was not yet known; it is rewritten over itself here, and the table
         * is appended after the last block. Both are sealed by the helper. */
        if (fseek(fo, 0, SEEK_SET) != 0 ||
            fwrite(hdr, 1u, sizeof hdr, fo) != sizeof hdr) {
            fprintf(stderr, "zpse_cli: cannot rewrite the frame header\n");
            free(idx);
            goto done;
        }
        if (fseek(fo, 0, SEEK_END) != 0 ||
            fwrite(idx, 1u, idx_len, fo) != idx_len) {
            fprintf(stderr, "zpse_cli: short write on %s\n", out_path);
            free(idx);
            goto done;
        }
        free(idx);
    }
    if (fclose(fo) != 0) {
        fo = NULL;
        fprintf(stderr, "zpse_cli: close failed on %s\n", out_path);
        goto done;
    }
    fo = NULL;
    rc = 0;
done:
#if defined(ZPSE_CLI_HAVE_ENCODE_POOL)
    /* Stop and join the workers before anything they could touch (the output
     * stream, the mapped input) is released. This single block covers every
     * exit: slab errors, index failures, and a failed fclose all goto done. */
    if (enc_pool != NULL) {
        zpse_encode_pool_free(enc_pool);
        enc_pool = NULL;
    }
    ZPSE_CLI_FREE(enc_pool_scratch);
    enc_pool_scratch = NULL;
#endif
    if (fo != NULL) {
        fclose(fo);
    }
#if !defined(_WIN32)
    if (in_map != NULL) {
        munmap((void *)(uintptr_t)in_map, n);
    }
#endif
    free(in_buf);
    free(slot_mem);
    free(ready);
    free(lens);
    free(blk_off);
    free(blk_len);
    free(out_buf);
    fclose(fi);
    return rc;
}

/* Decode a FLAT7 frame to a file without holding the decoded frame resident.
 * The packed payload is a plain ASCII stream (ceil(7n/8) bytes), so it can be
 * expanded in chunks: 28672 packed bytes -> 32768 decoded bytes. `n` comes
 * from the frame header and bounds the output. */
static int decode_flat7(const char *in_path, const char *out_path)
{
    enum { PACK_CHUNK = 28672u, OUT_CHUNK = 32768u };
    FILE *fi, *fo = NULL;
    uint8_t head[256];
    uint8_t *whole = NULL;
    const uint8_t *hdr;
    size_t got, f_len, n = 0u, payload_off = 0u, packed_pos = 0u;
    size_t consumed = 0u;
    long end;
    uint8_t *pchunk = NULL, *ochunk = NULL;
    int raw, rc = 1;

    fi = fopen(in_path, "rb");
    if (fi == NULL) {
        fprintf(stderr, "zpse_cli: cannot open %s\n", in_path);
        return 1;
    }
    if (fseek(fi, 0, SEEK_END) != 0 || (end = ftell(fi)) < 0 ||
        fseek(fi, 0, SEEK_SET) != 0) {
        fprintf(stderr, "zpse_cli: cannot size %s\n", in_path);
        goto done;
    }
    f_len = (size_t)end;
    got = fread(head, 1u, sizeof head, fi);
    if (got < 4u || !zpse_flat7_is_frame(head, got)) {
        rc = -1; /* not FLAT7: let the caller report its usual error */
        goto done;
    }
    hdr = head;
    if (zpse_flat7_header_check(hdr, f_len, &n, &payload_off) != ZPSE_OK) {
        /* Header larger than the peek buffer: stage the packed frame and
         * retry. n <= UINT32_MAX, so this stays bounded. */
        whole = (uint8_t *)malloc(f_len);
        if (whole == NULL || fseek(fi, 0, SEEK_SET) != 0 ||
            fread(whole, 1u, f_len, fi) != f_len) {
            fprintf(stderr, "zpse_cli: cannot read %s\n", in_path);
            goto done;
        }
        hdr = whole;
        if (zpse_flat7_header_check(hdr, f_len, &n, &payload_off) != ZPSE_OK) {
            fprintf(stderr, "zpse_cli: bad FLAT7 header in %s\n", in_path);
            goto done;
        }
    }
    if (n == 0u || n > (size_t)UINT32_MAX || payload_off > f_len) {
        fprintf(stderr, "zpse_cli: implausible FLAT7 header in %s\n", in_path);
        goto done;
    }
    /* Region type sits at byte 20 (region base) + 12 (type field). The
     * encoder always writes ASCII7; RAW is a forward-compat form the header
     * check also accepts, carrying the bytes verbatim rather than packed. */
    raw = (head[20 + 12] == ZPSE_FLAT7_REGION_RAW);
    fo = fopen(out_path, "wb");
    if (fo == NULL) {
        fprintf(stderr, "zpse_cli: cannot write %s\n", out_path);
        goto done;
    }
    pchunk = (uint8_t *)malloc(PACK_CHUNK);
    /* The unpack kernels may store a few bytes past out_n; leave slack. */
    ochunk = (uint8_t *)malloc(OUT_CHUNK + 8u);
    if (pchunk == NULL || ochunk == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        goto done;
    }
    if (whole == NULL && fseek(fi, (long)payload_off, SEEK_SET) != 0) {
        fprintf(stderr, "zpse_cli: cannot seek %s\n", in_path);
        goto done;
    }
    while (consumed < n) {
        size_t left = n - consumed;
        uint32_t want_out = left > (size_t)OUT_CHUNK
                                ? (uint32_t)OUT_CHUNK : (uint32_t)left;
        uint32_t want_packed = raw ? want_out : zpse_ascii_len(want_out);
        size_t got_p;

        if (whole != NULL) {
            memcpy(pchunk, whole + payload_off + packed_pos, want_packed);
            got_p = want_packed;
        } else {
            got_p = fread(pchunk, 1u, want_packed, fi);
        }
        if (got_p != want_packed) {
            fprintf(stderr, "zpse_cli: short read on %s\n", in_path);
            goto done;
        }
        if (raw) {
            memcpy(ochunk, pchunk, want_out);
        } else {
            zpse_ascii_unpack_stream(pchunk, want_packed, ochunk, want_out);
        }
        if (fwrite(ochunk, 1u, want_out, fo) != want_out) {
            fprintf(stderr, "zpse_cli: short write on %s\n", out_path);
            goto done;
        }
        packed_pos += want_packed;
        consumed += want_out;
    }
    if (fclose(fo) != 0) {
        fo = NULL;
        fprintf(stderr, "zpse_cli: close failed on %s\n", out_path);
        goto done;
    }
    fo = NULL;
    rc = 0;
done:
    free(pchunk);
    free(ochunk);
    free(whole);
    if (fo != NULL) {
        fclose(fo);
    }
    fclose(fi);
    return rc;
}

static int cmd_codec(int argc, char **argv)
{
    int compress = (argv[1][0] == 'c');
    /* No level bits: the library's own default applies. This used to
     * hard-code BALANCED, which made the tool ignore ZPSE_ENC_SPEED_DEFAULT
     * entirely -- when the default moved from level 5 to level 6, `zpse`
     * without -l kept writing level 5 and the header kept saying "balanced"
     * while the codec's default was a rung deeper. One source of truth. */
    uint32_t enc_flags = 0u;
    uint32_t block = ZPSE_MAX_BLOCK_SIZE;
    int indexed = 0;
    int repack_opt = 0;
    unsigned threads = 1u;
    const char *in_path;
    const char *out_path;
    uint8_t *in;
    uint8_t *out;
    size_t in_len = 0u;
    size_t out_len = 0u;
    size_t out_cap;
    void **enc_arenas = NULL;
    int rc;
    int i = 2;

    for (; i < argc && argv[i][0] == '-'; i++) {
        switch (argv[i][1]) {
        case 'l':
            if (++i >= argc || !parse_level(argv[i], &enc_flags)) {
                return 2;
            }
            break;
        case 'F':
            /* The fast LZ form and nothing else: one match-finder pass per
             * block, no shape decision. The stream is the one MAX offers as
             * its last candidate, so any decoder reads it. */
            enc_flags |= ZPSE_ENC_LZ_FAST;
            break;
        case 'R':
            /* Whole-frame repack: try a ZRPK frame before the normal codec
             * on compress. On decompress the magic is sniffed from the
             * header below, so -R is not needed to read a ZRPK frame. */
            repack_opt = 1;
            break;
        case 'P':
            /* One shape per page: the identity tile stream, compared with a
             * verbatim copy and nothing else priced or tried. Ignored above
             * a page, where the flag has no meaning. */
            enc_flags |= ZPSE_ENC_PAGE_FAST;
            break;
        case 'D':
            /* Decode-first: an LZ form has to be 1/8 smaller than the tile
             * stream to be written at all. The streams are the sweep's own,
             * so every decoder reads them. */
            enc_flags |= ZPSE_ENC_DECODE_FAST;
            break;
        case 'T':
            /* Page turbo: one shape and the page decode-first margin, with
             * the LZ stages still armed behind that margin. */
            enc_flags |= ZPSE_ENC_PAGE_TURBO;
            break;
        case 'H':
            /* Huffman-wrap the LZ payload on a 4 KiB page at any LZ level. */
            enc_flags |= ZPSE_ENC_PAGE_HUF;
            break;
        case 'c':
            enc_flags |= ZPSE_ENC_CHECKSUM;
            break;
        case 'b': {
            char *end = NULL;
            unsigned long v;

            if (++i >= argc) {
                return 2;
            }
            v = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 1u ||
                v > (unsigned long)ZPSE_MAX_BLOCK_SIZE) {
                fprintf(stderr, "zpse_cli: bad block size '%s'\n", argv[i]);
                return 2;
            }
            block = (uint32_t)v;
            break;
        }
        case 'x':
            indexed = 1;
            break;
        case 't': {
            char *end = NULL;
            unsigned long v;

            if (++i >= argc) {
                return 2;
            }
            v = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 1u || v > 64u) {
                fprintf(stderr, "zpse_cli: bad thread count '%s'\n", argv[i]);
                return 2;
            }
            threads = (unsigned)v;
            break;
        }
        default:
            fprintf(stderr, "zpse_cli: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    if (i >= argc || argc - i != 2) {
        fprintf(stderr, "usage: zpse_cli c [opts] IN OUT\n"
                        "       zpse_cli d IN OUT\n"
                        "       zpse_cli x FILE BLOCK OUT\n"
                        "       zpse_cli dir FILE\n"
                        "       zpse_cli bench [opts] FILE...\n");
        return 2;
    }
    in_path = argv[i];
    out_path = argv[i + 1];
    /* The frame is written block by block straight to the output file, and
     * the index rides along because the writer records where each block went
     * as it writes it (see zpse_frame_stream_index). Nothing here holds the
     * input or the frame, which is what a whole-file encode spends a quarter
     * of its time on. */
    if (compress) {
        /* REPACK whole-frame form. One in-memory attempt (not a timed loop):
         * on commit the ZRPK frame is the whole output file and no block
         * stream is built; on decline fall through to the ordinary streamed
         * codec below. -x (index) and -c (per-block CRC) do not apply to a
         * ZRPK frame, so with either of those set we take the normal path. */
        if (repack_opt && !indexed && (enc_flags & ZPSE_ENC_CHECKSUM) == 0u) {
            uint8_t *rin;
            uint8_t *rout;
            size_t rn = 0u;
            size_t rlen = 0u;
            size_t rcap;

            rin = read_file(in_path, &rn);
            if (rin == NULL) {
                return 1;
            }
            rcap = zpse_repack_bound(rn);
            rout = (uint8_t *)malloc(rcap + 1u);
            if (rout == NULL) {
                fprintf(stderr, "zpse_cli: out of memory (REPACK)\n");
                free(rin);
                return 1;
            }
            if (zpse_repack_encode(rin, rn, rout, rcap, &rlen) == 1) {
                rc = write_file(out_path, rout, rlen);
                free(rout);
                free(rin);
                return rc;
            }
            /* 0: declined (non-ASCII or too many distinct symbols) -- free
             * the probe and take the normal path. encode_stream reopens the
             * input; this one-shot CLI read is not part of any timing. */
            free(rout);
            free(rin);
        }
        /* A worker count does not change the path: this one takes it too, as
         * it always could. The buffered worker call it used to take for -t
         * reserves a slot per block in the caller's buffer and compacts the
         * frame afterwards, so a 50 MiB file wanted the frame's size in
         * memory plus a serial memmove over all of it -- on exactly the runs
         * where the workers were supposed to be the point. The streamed
         * worker path writes the same bytes; test_parallel.c compares its
         * block region against the serial frame at every worker count and
         * both ring depths. */
        enc_arenas = alloc_arenas(threads, ZPSE_ENC_ARENA_SIZE);
        if (enc_arenas == NULL) {
            fprintf(stderr, "zpse_cli: arena allocation failed\n");
            return 1;
        }
        rc = encode_stream(in_path, out_path, block, enc_flags, indexed,
                           threads, (void *const *)enc_arenas);
        free_arenas(enc_arenas, threads);
        return rc;
    }
    if (!compress) {
        /* Peek the header straight from the file. Every frame decodes
         * sequentially block by block, indexed or not, and that needs no copy
         * of the frame in memory at all -- see decode_stream for what the copy
         * cost. An indexed frame also has to carry its block count, which is
         * what the table is checked against.
         *
         * A worker count keeps this path: the walk reads a batch of blocks
         * and hands it to the worker pool, which decodes them side by side
         * into one batch-sized buffer, and the writer emits them in order.
         * That is the streamed path with the blocks spread over the cores,
         * not the buffered worker call it used to be -- that one wanted the
         * whole frame and the whole output resident, so webster's 41 MiB
         * frame cost 382 ms through it against 122 ms from the file. */
        ZpseFrameHeader ph;
        /* A FLAT7 header is 40 bytes, longer than a frame header, so peek
         * enough to sniff its magic as well as read the ordinary header. */
        uint8_t head[64];
        int peek = 0;
        int indexed = 0;
        FILE *pf = fopen(in_path, "rb");

        if (pf != NULL) {
            size_t got = fread(head, 1u, sizeof head, pf);

            /* FLAT7 has its own magic and no block framing; the ordinary
             * header read below would reject it. Expand it to the output
             * file in chunks instead of buffering the decoded frame. */
            if (got >= 40u && zpse_flat7_is_frame(head, got)) {
                int frc;

                fclose(pf);
                frc = decode_flat7(in_path, out_path);
                if (frc < 0) {
                    fprintf(stderr,
                            "zpse_cli: not a zpse frame with a content size\n");
                    return 1;
                }
                return frc;
            }
            /* ZRPK is a self-delimiting whole-frame form with its own magic
             * and no block framing: [0..4) is "ZRPK", [8..16) is the original
             * length n (little-endian u64). The ordinary header read below
             * would reject it, so expand it with the repack decoder. Like
             * FLAT7 it needs the whole frame in memory; unlike a block frame
             * it carries its own content size, so it needs no frame header. */
            if (got >= 44u && head[0] == 'Z' && head[1] == 'R' &&
                head[2] == 'P' && head[3] == 'K') {
                uint8_t *frame;
                uint8_t *dec;
                size_t frame_len = 0u;
                size_t dn;
                uint64_t dn64 = 0u;
                unsigned b;

                fclose(pf);
                frame = read_file(in_path, &frame_len);
                if (frame == NULL) {
                    return 1;
                }
                if (frame_len < 44u) {
                    fprintf(stderr, "zpse_cli: truncated ZRPK frame\n");
                    free(frame);
                    return 1;
                }
                for (b = 0u; b < 8u; b++) {
                    dn64 |= (uint64_t)frame[8u + b] << (8u * b);
                }
                if (dn64 > (uint64_t)(size_t)-1) {
                    fprintf(stderr, "zpse_cli: ZRPK content size too large\n");
                    free(frame);
                    return 1;
                }
                dn = (size_t)dn64;
                /* A packed byte expands to at most 8 symbols (w >= 1) and the
                 * smallest ZRPK header is 44 bytes, so a frame cannot decode
                 * to more than this many bytes. Refuse a header that claims
                 * more before it can drive a huge malloc. */
                if (dn > (frame_len - 44u) * 8u) {
                    fprintf(stderr,
                            "zpse_cli: ZRPK content size exceeds frame data\n");
                    free(frame);
                    return 1;
                }
                dec = dn > 0u ? (uint8_t *)malloc(dn) : NULL;
                if (dn > 0u && dec == NULL) {
                    fprintf(stderr, "zpse_cli: out of memory (REPACK decode)\n");
                    free(frame);
                    return 1;
                }
                rc = zpse_repack_decode(frame, frame_len, dec, dn) == 1
                         ? write_file(out_path, dec, dn)
                         : 1;
                if (rc != 0) {
                    fprintf(stderr, "zpse_cli: ZRPK decode failed\n");
                }
                free(dec);
                free(frame);
                return rc;
            }
            /* RLE1 is another self-delimiting whole-frame form: [0..4) is
             * "RLE1", [4..8) is the original length n (u32 LE), [8..12) a
             * header CRC. Like ZRPK it carries its own content size and needs
             * the whole frame in memory, so expand it with rlefast. */
            if (got >= 12u && head[0] == 'R' && head[1] == 'L' &&
                head[2] == 'E' && head[3] == '1') {
                uint8_t *frame;
                uint8_t *dec;
                size_t frame_len = 0u;
                uint32_t n = 0u;
                unsigned b;

                fclose(pf);
                frame = read_file(in_path, &frame_len);
                if (frame == NULL) {
                    return 1;
                }
                if (frame_len < 12u) {
                    fprintf(stderr, "zpse_cli: truncated RLE1 frame\n");
                    free(frame);
                    return 1;
                }
                for (b = 0u; b < 4u; b++) {
                    n |= (uint32_t)frame[4u + b] << (8u * b);
                }
                /* Each (value, run-1) pair costs 2 bytes and expands to at
                 * most 256 bytes, so the payload bounds the claimed length.
                 * A uint64 comparison keeps the multiply from wrapping. */
                if ((uint64_t)n > (uint64_t)(frame_len - 12u) * 256u) {
                    fprintf(stderr,
                            "zpse_cli: RLE1 content size exceeds frame data\n");
                    free(frame);
                    return 1;
                }
                dec = n > 0u ? (uint8_t *)malloc(n) : NULL;
                if (n > 0u && dec == NULL) {
                    fprintf(stderr, "zpse_cli: out of memory (RLE1 decode)\n");
                    free(frame);
                    return 1;
                }
                rc = frame_len <= (size_t)UINT32_MAX &&
                             zpse_rlefast_decode(frame, (uint32_t)frame_len,
                                                 dec, n) == 0
                         ? write_file(out_path, dec, n)
                         : 1;
                if (rc != 0) {
                    fprintf(stderr, "zpse_cli: RLE1 decode failed\n");
                }
                free(dec);
                free(frame);
                return rc;
            }
            if (got >= (size_t)ZPSE_FRAME_HEADER_SIZE &&
                zpse_frame_header_read(&ph, head) == ZPSE_OK &&
                (ph.frame_flags & (uint16_t)ZPSE_F_CONTENT_SIZE) != 0u &&
                ph.content_size != UINT64_MAX &&
                ((ph.frame_flags & (uint16_t)ZPSE_F_FRAME_INDEX) == 0u ||
                 (ph.frame_flags & (uint16_t)ZPSE_F_BLOCK_COUNT) != 0u)) {
                peek = 1;
                indexed = (ph.frame_flags & (uint16_t)ZPSE_F_FRAME_INDEX) != 0u;
            }
            fclose(pf);
        }
        if (peek) {
            void **arenas = alloc_arenas(threads, ZPSE_DEC_ARENA_SIZE);
            int drc;

            if (arenas == NULL) {
                fprintf(stderr, "zpse_cli: arena allocation failed\n");
                return 1;
            }
            drc = decode_stream(in_path, out_path, &ph,
                                (void *const *)arenas, threads, indexed);
            free_arenas(arenas, threads);
            return drc;
        }
        /* No content size means no way to size the output, and every frame
         * this library writes carries one. */
        fprintf(stderr, "zpse_cli: not a zpse frame with a content size\n");
        return 1;
    }
    in = read_file(in_path, &in_len);
    if (in == NULL) {
        return 1;
    }
    {
        size_t ser = frame_bound(in_len, block, indexed);
        size_t par = zpse_parallel_bound(in_len, block, indexed);

        /* The worker path reserves a fixed slot per block and compacts
         * afterwards, so it wants the larger buffer; the compaction never
         * costs a byte of the result. */
        out_cap = par > ser ? par : ser;
        enc_arenas = alloc_arenas(threads, ZPSE_ENC_ARENA_SIZE);
    }
    if (enc_arenas == NULL) {
        fprintf(stderr, "zpse_cli: arena allocation failed\n");
        free(in);
        return 1;
    }
    out = (uint8_t *)malloc(out_cap + 1u);
    if (out == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        free_arenas(enc_arenas, threads);
        free(in);
        return 1;
    }

    rc = compress_mem(in, in_len, out, out_cap, &out_len, block, enc_flags,
                      indexed, enc_arenas, threads);
    if (rc != 0) {
        fprintf(stderr, "zpse_cli: codec failed rc=%d\n", rc);
        free_arenas(enc_arenas, threads);
        free(out);
        free(in);
        return 1;
    }
    rc = write_file(out_path, out, out_len);
    free_arenas(enc_arenas, threads);
    free(out);
    free(in);
    return rc;
}

/* Compare a seeked block against the slice of the original it holds: the
 * index and the block headers must agree with the bytes the full decode
 * produced. Cheap relative to the full decode, and it is the only thing
 * here that exercises zpse_seek_frame. */
static int check_seek(const uint8_t *frame, size_t frame_len,
                      const uint8_t *orig, size_t n, uint32_t block,
                      void *dec_arena)
{
    size_t nblocks = n / (size_t)block + (n % (size_t)block != 0u);
    uint8_t out[ZPSE_MAX_BLOCK_SIZE];
    uint32_t picks[3];
    int npick = 0;
    int i;

    if (nblocks == 0u) {
        return 1;
    }
    picks[npick++] = 0u;
    if (nblocks > 2u) {
        picks[npick++] = (uint32_t)(nblocks / 2u);
    }
    if (nblocks > 1u) {
        picks[npick++] = (uint32_t)(nblocks - 1u);
    }
    for (i = 0; i < npick; i++) {
        uint32_t got = 0u;
        size_t off = (size_t)picks[i] * (size_t)block;
        size_t want = n - off;

        if (want > (size_t)block) {
            want = (size_t)block;
        }
        if (zpse_seek_frame(frame, frame_len, picks[i], out,
                            (uint32_t)sizeof out, &got, dec_arena) != 0 ||
            (size_t)got != want || memcmp(out, orig + off, want) != 0) {
            fprintf(stderr, "zpse_cli: seek of block %u failed\n", picks[i]);
            return 0;
        }
    }
    return 1;
}

/* The A/B table's workload column names the file, not its directory: bench
 * paths are often whole corpus paths and would overrun the column. */
static const char *bench_label(const char *path)
{
    const char *slash = strrchr(path, '/');
    const char *bs = strrchr(path, '\\');
    const char *last = slash;

    if (bs != NULL && (last == NULL || bs > last)) {
        last = bs;
    }
    return last != NULL ? last + 1 : path;
}

static int cmd_bench(int argc, char **argv)
{
    /* No level bits: the library's own default applies. This used to
     * hard-code BALANCED, which made the tool ignore ZPSE_ENC_SPEED_DEFAULT
     * entirely -- when the default moved from level 5 to level 6, `zpse`
     * without -l kept writing level 5 and the header kept saying "balanced"
     * while the codec's default was a rung deeper. One source of truth. */
    uint32_t enc_flags = 0u;
    uint32_t block = ZPSE_MAX_BLOCK_SIZE;
    int indexed = 0;
    int quiet = 0;
    int flat7_bench = 0;
    int repack_bench = 0;
    long reps_arg = 0;
    long bytes_cap = 0;
    unsigned threads = 1u;
    int i = 2;
    int files;
    int f;
    int failures = 0;
    unsigned long long tot_in = 0u;
    unsigned long long tot_out = 0u;
    /* One rep's worth of bytes per file: the per-file rates are n/best_time,
     * so the totals sum n over files and divide by the sum of best times. */
    unsigned long long tot_passed = 0u;
    double tot_enc_ns = 0.0;
    double tot_dec_ns = 0.0;
    /* FLAT7 A/B totals: the flat7 side covers only eligible files, so the
     * block side used for its speedup covers exactly those same files. */
    unsigned long long tot_f_in = 0u;
    unsigned long long tot_f_out = 0u;
    double tot_f_enc_ns = 0.0;
    double tot_f_dec_ns = 0.0;
    unsigned long long tot_b_in = 0u;
    unsigned long long tot_b_out = 0u;
    double tot_b_enc_ns = 0.0;
    double tot_b_dec_ns = 0.0;
    /* REPACK A/B totals, same shape as the FLAT7 ones: the repack side
     * covers only eligible files, and tot_b_* is the block side over exactly
     * those same files. */
    unsigned long long tot_r_in = 0u;
    unsigned long long tot_r_out = 0u;
    double tot_r_enc_ns = 0.0;
    double tot_r_dec_ns = 0.0;
    void **enc_arenas;
    void **dec_arenas;

    for (; i < argc && argv[i][0] == '-'; i++) {
        switch (argv[i][1]) {
        case 'l':
            if (++i >= argc || !parse_level(argv[i], &enc_flags)) {
                return 2;
            }
            break;
        case 'F':
            /* -F7 is the FLAT7 A/B arm, not the fast LZ form: it selects a
             * second codec path for this bench and leaves the level bits (and
             * therefore the normal block arm) untouched. Every other -F still
             * means ZPSE_ENC_LZ_FAST, as in `c`. */
            if (argv[i][2] == '7' && argv[i][3] == '\0') {
                flat7_bench = 1;
            } else {
                enc_flags |= ZPSE_ENC_LZ_FAST;
            }
            break;
        case 'P':
            /* One shape per page: the identity tile stream, compared with a
             * verbatim copy and nothing else priced or tried. Ignored above
             * a page, where the flag has no meaning. */
            enc_flags |= ZPSE_ENC_PAGE_FAST;
            break;
        case 'D':
            /* Decode-first: an LZ form has to be 1/8 smaller than the tile
             * stream to be written at all. The streams are the sweep's own,
             * so every decoder reads them. */
            enc_flags |= ZPSE_ENC_DECODE_FAST;
            break;
        case 'T':
            /* Page turbo: one shape and the page decode-first margin, with
             * the LZ stages still armed behind that margin. */
            enc_flags |= ZPSE_ENC_PAGE_TURBO;
            break;
        case 'H':
            /* Huffman-wrap the LZ payload on a 4 KiB page at any LZ level. */
            enc_flags |= ZPSE_ENC_PAGE_HUF;
            break;
        case 'c':
            enc_flags |= ZPSE_ENC_CHECKSUM;
            break;
        case 'b': {
            char *end = NULL;
            unsigned long v;

            if (++i >= argc) {
                return 2;
            }
            v = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 1u ||
                v > (unsigned long)ZPSE_MAX_BLOCK_SIZE) {
                fprintf(stderr, "zpse_cli: bad block size '%s'\n", argv[i]);
                return 2;
            }
            block = (uint32_t)v;
            break;
        }
        case 'r': {
            char *end = NULL;

            if (++i >= argc) {
                return 2;
            }
            reps_arg = strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || reps_arg < 0 ||
                reps_arg > 1000) {
                fprintf(stderr, "zpse_cli: bad repetition count '%s'\n",
                        argv[i]);
                return 2;
            }
            break;
        }
        case 't': {
            char *end = NULL;
            unsigned long v;

            if (++i >= argc) {
                return 2;
            }
            v = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 1u || v > 64u) {
                fprintf(stderr, "zpse_cli: bad thread count '%s'\n", argv[i]);
                return 2;
            }
            threads = (unsigned)v;
            break;
        }
        case 'n': {
            char *end = NULL;

            if (++i >= argc) {
                return 2;
            }
            bytes_cap = strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || bytes_cap < 0) {
                fprintf(stderr, "zpse_cli: bad byte cap '%s'\n", argv[i]);
                return 2;
            }
            break;
        }
        case 'x':
            indexed = 1;
            break;
        case 'q':
            quiet = 1;
            break;
        case '7':
            /* Alias for -F7: select the in-memory FLAT7 arm. */
            flat7_bench = 1;
            break;
        case 'R':
            /* Select the in-memory whole-frame REPACK A/B arm. */
            repack_bench = 1;
            break;
        default:
            fprintf(stderr, "zpse_cli: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    /* -R and -F7 are both two-row A/B arms that share one table; keep only
     * one active, with REPACK taking precedence when both are requested. */
    if (repack_bench) {
        flat7_bench = 0;
    }
    /* The parallel decoder finds a block through the index, so a threaded
     * bench measures an indexed frame. It costs 8 bytes per block in the
     * ratio column, and the header line says which frame was measured. */
    if (threads > 1u) {
        indexed = 1;
    }
    files = argc - i;
    if (files < 1) {
        fprintf(stderr, "usage: zpse_cli bench [-l LEVEL] [-F7|-7|-R] [-F] [-P] "
                        "[-D] [-T] [-H] [-c] [-b N] [-r N] [-t N] [-n N] "
                        "[-x] [-q] FILE...\n"
                        "  -F7  add a FLAT7 arm beside the block arm on "
                        "eligible whole-ASCII input and print both rates,\n"
                        "       ratios and the flat7/block encode speedup\n"
                        "  -R   add a REPACK (ZRPK) arm beside the block arm "
                        "on eligible ASCII input and print both rates,\n"
                        "       ratios and the repack/block encode speedup\n");
        return 2;
    }
    enc_arenas = alloc_arenas(threads, ZPSE_ENC_ARENA_SIZE);
    dec_arenas = alloc_arenas(threads, ZPSE_DEC_ARENA_SIZE);
    if (enc_arenas == NULL || dec_arenas == NULL) {
        fprintf(stderr, "zpse_cli: arena allocation failed\n");
        free_arenas(enc_arenas, threads);
        free_arenas(dec_arenas, threads);
        return 1;
    }
    /* Pre-fault each worker's arena before any clock starts. The first write
     * into a fresh arena is a page fault every 4 KiB, which is the caller's
     * allocation cost, not the codec's work. The reps below reuse these
     * arenas, so touch them once here and keep that cost out of every rep. */
    {
        unsigned a;
        for (a = 0; a < threads; a++) {
            memset(enc_arenas[a], 0, ZPSE_ENC_ARENA_SIZE);
            memset(dec_arenas[a], 0, ZPSE_DEC_ARENA_SIZE);
        }
    }
    if (!quiet) {
        printf("level %s%s%s, block %u, %s, %u thread%s%s\n",
               level_name(enc_flags),
               (enc_flags & ZPSE_ENC_CHECKSUM) ? ", crc" : "",
               (enc_flags & ZPSE_ENC_PAGE_FAST) ? ", page-fast" : "",
               (unsigned)block, indexed ? "indexed" : "no index",
               threads, threads == 1u ? "" : "s",
               repack_bench
                    ? ", REPACK A/B (REPACK arm single-threaded)"
                    : (flat7_bench ? ", FLAT7 A/B (FLAT7 arm single-threaded)"
                                   : ""));
        if (repack_bench || flat7_bench) {
            printf("%-22s %-6s %10s %10s %10s %12s\n", "workload", "mode",
                   "enc MB/s", "dec MB/s", "ratio",
                   repack_bench ? "repack/block" : "flat7/block");
            printf("%-22s %-6s %10s %10s %10s %12s\n", "----------------------",
                   "------", "----------", "----------", "----------",
                   "------------");
        } else {
            printf("%-14s %11s %11s %7s %10s %10s %8s\n", "file", "bytes",
                   "packed", "ratio", "enc MB/s", "dec MB/s", "ms");
        }
    }

    for (f = 0; f < files; f++) {
        const char *path = argv[i + f];
        uint8_t *in;
        uint8_t *comp;
        uint8_t *out;
        uint8_t *flat = NULL;
        uint8_t *repack = NULL;
        size_t n = 0u;
        size_t comp_cap;
        size_t comp_len = 0u;
        size_t out_len = 0u;
        size_t flat_cap = 0u;
        size_t flat_len = 0u;
        size_t f_out_len = 0u;
        size_t repack_cap = 0u;
        size_t repack_len = 0u;
        long reps;
        long r;
        double t0;
        double dt;
        double enc_ns;
        double dec_ns;
        double f_enc_ns = 0.0;
        double f_dec_ns = 0.0;
        double r_enc_ns = 0.0;
        double r_dec_ns = 0.0;
        int flat7_eligible = 0;
        int f_ok = 0;
        int repack_eligible = 0;
        int r_ok = 0;
        int repack_canary_ok = 1;
        int ok = 1;

        in = read_file(path, &n);
        if (in == NULL) {
            failures++;
            continue;
        }
        /* A byte cap is what makes one case quick: the corpus members are
         * 4 MiB each and a whole-corpus run is minutes. */
        if (bytes_cap > 0 && n > (size_t)bytes_cap) {
            n = (size_t)bytes_cap;
        }
        comp_cap = frame_bound(n, block, indexed);
        {
            size_t par = zpse_parallel_bound(n, block, indexed);

            if (par > comp_cap) {
                comp_cap = par;
            }
        }
        comp = (uint8_t *)malloc(comp_cap + 1u);
        out = (uint8_t *)malloc(n + 1u);
        if (comp == NULL || out == NULL) {
            fprintf(stderr, "zpse_cli: out of memory on %s\n", path);
            free(in);
            free(comp);
            free(out);
            failures++;
            continue;
        }
        /* Both buffers are touched once before the clock starts. A codec
         * benchmark measures the codec, and the first write into a fresh
         * malloc is a page fault every 4 KiB of the buffer it produces -- on
         * the 51 MiB `mozilla` frame that was the difference between 1.64
         * GB/s into resident pages and 0.78 into untouched ones. That cost is
         * the caller's allocation, not the decoder's work, and an embedder
         * decoding into a buffer it has already used never pays it. The
         * repetitions below reuse these two, which is exactly that shape. */
        memset(out, 0, n);
        memset(comp, 0, comp_cap);
        /* FLAT7 A/B arm. A separate frame buffer keeps the block arm's `comp`
         * untouched, so the block row below is exactly the ordinary bench row
         * this file already produced. n + 256 leaves room for the 40-byte
         * header and the packer's trailing 8-byte scratch. The eligibility
         * call is untimed -- the timed encode below calls the same gate again,
         * so the FLAT7 number really is `zpse_flat7_should` + encode, not the
         * bare 8-to-7 kernel. */
        if (flat7_bench) {
            flat_cap = n + 256u;
            flat = (uint8_t *)malloc(flat_cap + 1u);
            if (flat == NULL) {
                fprintf(stderr, "zpse_cli: out of memory on %s (FLAT7)\n", path);
                free(in);
                free(comp);
                free(out);
                failures++;
                continue;
            }
            memset(flat, 0, flat_cap);
            flat7_eligible =
                zpse_flat7_should(in, n, enc_flags, indexed, flat_cap) != 0;
            if (!flat7_eligible) {
                fprintf(stderr,
                        "zpse_cli: bench -F7: %s: FLAT7 declined by "
                        "zpse_flat7_should (non-ASCII, redundant, indexed, "
                        "crc, or too small); skipping FLAT7 arm\n", path);
            }
        }
        /* REPACK A/B arm. A separate frame buffer keeps the block arm's
         * `comp` untouched, so the block row below stays exactly the ordinary
         * bench row. zpse_repack_bound sizes the whole ZRPK frame. There is no
         * separate `should` predicate in the API, so eligibility is settled
         * by one untimed probe encode here -- which also round-trips and
         * compares the whole input before any clock starts. The timed encode
         * below runs the same gate; a decline is reported and the arm skipped,
         * matching the -F7 arm. */
        if (repack_bench) {
            size_t probe_len = 0u;

            repack_cap = zpse_repack_bound(n);
            repack = (uint8_t *)malloc(repack_cap + 1u);
            if (repack == NULL) {
                fprintf(stderr, "zpse_cli: out of memory on %s (REPACK)\n",
                        path);
                free(in);
                free(comp);
                free(out);
                failures++;
                continue;
            }
            memset(repack, 0, repack_cap);
            /* The extra byte repack_cap is a canary: encode is bounded by
             * repack_cap and must never touch it. */
            repack[repack_cap] = 0xA5u;
            if (zpse_repack_encode(in, n, repack, repack_cap, &probe_len) == 1 &&
                zpse_repack_decode(repack, probe_len, out, n) == 1 &&
                memcmp(out, in, n) == 0) {
                repack_eligible = 1;
                repack_len = probe_len;
            } else if (probe_len != 0u) {
                fprintf(stderr, "zpse_cli: bench -R: %s: REPACK round trip "
                                "MISMATCH; skipping REPACK arm\n", path);
            }
            if (!repack_eligible) {
                fprintf(stderr,
                        "zpse_cli: bench -R: %s: REPACK declined (non-ASCII "
                        "or too many distinct symbols); skipping REPACK "
                        "arm\n", path);
            }
        }
        /* Enough repetitions for a 64 MiB sample per file, capped so the
         * big corpus members do not dominate the run. */
        reps = reps_arg;
        if (reps == 0) {
            reps = (long)((64u * 1024u * 1024u + n) / (n + 1u));
            if (reps < 1) {
                reps = 1;
            }
            if (reps > 20) {
                reps = 20;
            }
        }
        /* Best of N: time every rep on its own and keep the fastest. With
         * enough reps the first is a warmup and never enters the minimum. */
        enc_ns = 0.0;
        for (r = 0; r < reps; r++) {
            t0 = now_ns();
            if (compress_mem(in, n, comp, comp_cap, &comp_len, block,
                             enc_flags, indexed, enc_arenas, threads) != 0) {
                fprintf(stderr, "zpse_cli: compress failed on %s (n=%lu)\n",
                        path, (unsigned long)n);
                ok = 0;
                break;
            }
            if (reps >= 3 && r == 0) {
                continue;
            }
            dt = now_ns() - t0;
            if (enc_ns == 0.0 || dt < enc_ns) {
                enc_ns = dt;
            }
        }
        if (!ok) {
            free(in);
            free(comp);
            free(out);
            failures++;
            continue;
        }
        if (indexed &&
            !check_seek(comp, comp_len, in, n, block, dec_arenas[0])) {
            free(in);
            free(comp);
            free(out);
            failures++;
            continue;
        }
        dec_ns = 0.0;
        for (r = 0; r < reps; r++) {
            t0 = now_ns();
            if (zpse_decompress_frame_parallel(comp, comp_len, out, n,
                                               &out_len, threads, dec_arenas,
                                               threads) != 0) {
                fprintf(stderr, "zpse_cli: decompress failed on %s\n", path);
                ok = 0;
                break;
            }
            if (reps >= 3 && r == 0) {
                continue;
            }
            dt = now_ns() - t0;
            if (dec_ns == 0.0 || dt < dec_ns) {
                dec_ns = dt;
            }
        }
        if (ok) {
            /* Untimed verification pass: one decode, compared whole. */
            if (out_len != n || memcmp(out, in, n) != 0) {
                fprintf(stderr, "zpse_cli: %s: round trip MISMATCH\n", path);
                ok = 0;
            }
        }
        /* The FLAT7 arm, in memory. zpse_flat7_encode_stream is the fused
         * single-pass path: ASCII scan + both redundancy predicates + the
         * 7-bit pack, committing the header only if every gate passes. Timing
         * the bare pack would report the kernel's 17-26 GB/s and manufacture
         * a speedup the caller never sees. Best-of-reps matches the block arm. */
        if (flat7_bench && ok && flat7_eligible) {
            f_ok = 1;
            for (r = 0; r < reps; r++) {
                size_t fl = 0u;
                int frc;

                t0 = now_ns();
                frc = zpse_flat7_encode_stream(in, n, flat, flat_cap, &fl);
                if (frc <= 0) {
                    fprintf(stderr,
                            "zpse_cli: FLAT7 encode declined/failed on %s "
                            "(rc=%d)\n", path, frc);
                    f_ok = 0;
                    break;
                }
                dt = now_ns() - t0;
                if (reps >= 3 && r == 0) {
                    continue;
                }
                if (f_enc_ns == 0.0 || dt < f_enc_ns) {
                    f_enc_ns = dt;
                }
                flat_len = fl;
            }
            if (f_ok) {
                /* Untimed verification: header check, decode, whole compare. */
                size_t ck_n = 0u;
                size_t ck_off = 0u;

                if (zpse_flat7_header_check(flat, flat_len, &ck_n, &ck_off) !=
                        ZPSE_OK ||
                    ck_n != n ||
                    zpse_flat7_decode(flat, flat_len, out, n, &f_out_len) !=
                        ZPSE_OK ||
                    f_out_len != n || memcmp(out, in, n) != 0) {
                    fprintf(stderr,
                            "zpse_cli: %s: FLAT7 round trip MISMATCH\n", path);
                    f_ok = 0;
                }
            }
            /* zpse_flat7_decode validates the 40-byte header (magic, version,
             * CRC and lengths) before it unpacks, so the decode row already
             * includes the header check. It is single-threaded by design. */
            f_dec_ns = 0.0;
            if (f_ok) {
                for (r = 0; r < reps; r++) {
                    t0 = now_ns();
                    if (zpse_flat7_decode(flat, flat_len, out, n, &f_out_len) !=
                        ZPSE_OK) {
                        fprintf(stderr, "zpse_cli: FLAT7 decode failed on %s\n",
                                path);
                        f_ok = 0;
                        break;
                    }
                    dt = now_ns() - t0;
                    if (reps >= 3 && r == 0) {
                        continue;
                    }
                    if (f_dec_ns == 0.0 || dt < f_dec_ns) {
                        f_dec_ns = dt;
                    }
                }
            }
        }
        /* The REPACK arm, in memory: zpse_repack_encode writes a whole ZRPK
         * frame and zpse_repack_decode expands it, both over the same input
         * with no file I/O and no clock inside the setup. Best-of-reps matches
         * the block arm; the decode validates the frame (magic, version, CRC,
         * lengths) on every call, like FLAT7's decode. */
        if (repack_bench && ok && repack_eligible) {
            r_ok = 1;
            for (r = 0; r < reps; r++) {
                size_t rl = 0u;
                int rrc;

                t0 = now_ns();
                rrc = zpse_repack_encode(in, n, repack, repack_cap, &rl);
                if (rrc <= 0) {
                    fprintf(stderr,
                            "zpse_cli: REPACK encode declined/failed on %s "
                            "(rc=%d)\n", path, rrc);
                    r_ok = 0;
                    break;
                }
                dt = now_ns() - t0;
                if (reps >= 3 && r == 0) {
                    continue;
                }
                if (r_enc_ns == 0.0 || dt < r_enc_ns) {
                    r_enc_ns = dt;
                }
                repack_len = rl;
            }
            r_dec_ns = 0.0;
            if (r_ok) {
                for (r = 0; r < reps; r++) {
                    t0 = now_ns();
                    if (zpse_repack_decode(repack, repack_len, out, n) != 1) {
                        fprintf(stderr, "zpse_cli: REPACK decode failed on %s\n",
                                path);
                        r_ok = 0;
                        break;
                    }
                    dt = now_ns() - t0;
                    if (reps >= 3 && r == 0) {
                        continue;
                    }
                    if (r_dec_ns == 0.0 || dt < r_dec_ns) {
                        r_dec_ns = dt;
                    }
                }
            }
            if (r_ok) {
                /* One more untimed whole compare after the timed reps. */
                if (zpse_repack_decode(repack, repack_len, out, n) != 1 ||
                    memcmp(out, in, n) != 0) {
                    fprintf(stderr, "zpse_cli: %s: REPACK round trip "
                                    "MISMATCH\n", path);
                    r_ok = 0;
                }
            }
        }
        if (repack_bench && repack != NULL &&
            repack[repack_cap] != 0xA5u) {
            fprintf(stderr, "zpse_cli: bench -R: %s: REPACK encode wrote "
                            "past repack_cap\n", path);
            repack_canary_ok = 0;
        }
        if (!ok || (flat7_eligible && !f_ok) ||
            (repack_eligible && !r_ok) || !repack_canary_ok) {
            failures++;
        }
        if (repack_bench) {
            if (!quiet) {
                double be = enc_ns > 0.0 ? (double)n / enc_ns * 1e3 : 0.0;
                double bd = dec_ns > 0.0 ? (double)n / dec_ns * 1e3 : 0.0;

                if (repack_eligible) {
                    double re = r_enc_ns > 0.0 ? (double)n / r_enc_ns * 1e3
                                               : 0.0;
                    double rd = r_dec_ns > 0.0 ? (double)n / r_dec_ns * 1e3
                                               : 0.0;

                    printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %11.2fx%s\n",
                           bench_label(path), "repack", re, rd,
                           repack_len ? (double)n / (double)repack_len : 0.0,
                           be > 0.0 ? re / be : 0.0,
                           r_ok ? "" : "  FAIL");
                } else {
                    printf("%-22s %-6s %10s %10s %10s %12s\n",
                           bench_label(path), "repack", "-", "-", "-",
                           "declined");
                }
                printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %12s%s\n",
                       bench_label(path), "block", be, bd,
                       comp_len ? (double)n / (double)comp_len : 0.0, "",
                       ok ? "" : "  FAIL");
            }
        } else if (flat7_bench) {
            if (!quiet) {
                double be = enc_ns > 0.0 ? (double)n / enc_ns * 1e3 : 0.0;
                double bd = dec_ns > 0.0 ? (double)n / dec_ns * 1e3 : 0.0;

                if (flat7_eligible) {
                    double fe = f_enc_ns > 0.0 ? (double)n / f_enc_ns * 1e3
                                               : 0.0;
                    double fd = f_dec_ns > 0.0 ? (double)n / f_dec_ns * 1e3
                                               : 0.0;

                    printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %11.2fx%s\n",
                           bench_label(path), "flat7", fe, fd,
                           flat_len ? (double)n / (double)flat_len : 0.0,
                           be > 0.0 ? fe / be : 0.0,
                           f_ok ? "" : "  FAIL");
                } else {
                    printf("%-22s %-6s %10s %10s %10s %12s\n",
                           bench_label(path), "flat7", "-", "-", "-",
                           "declined");
                }
                printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %12s%s\n",
                       bench_label(path), "block", be, bd,
                       comp_len ? (double)n / (double)comp_len : 0.0, "",
                       ok ? "" : "  FAIL");
            }
        } else if (!quiet || !ok) {
            printf("%-14s %11lu %11lu %7.2f %10.1f %10.1f %8.1f%s\n", path,
                   (unsigned long)n, (unsigned long)comp_len,
                   comp_len ? (double)n / (double)comp_len : 0.0,
                   enc_ns > 0.0 ? (double)n / enc_ns * 1e3 : 0.0,
                   dec_ns > 0.0 ? (double)n / dec_ns * 1e3 : 0.0,
                   (enc_ns + dec_ns) / 1e6, ok ? "" : "  FAIL");
        }
        tot_in += (unsigned long long)n;
        tot_out += (unsigned long long)comp_len;
        tot_passed += (unsigned long long)n;
        tot_enc_ns += enc_ns;
        tot_dec_ns += dec_ns;
        /* The A/B totals cover only eligible files, on both sides, so the
         * flat7/block column stays an honest speedup over the same bytes. */
        if (flat7_bench && flat7_eligible && f_ok && ok) {
            tot_f_in += (unsigned long long)n;
            tot_f_out += (unsigned long long)flat_len;
            tot_f_enc_ns += f_enc_ns;
            tot_f_dec_ns += f_dec_ns;
            tot_b_in += (unsigned long long)n;
            tot_b_out += (unsigned long long)comp_len;
            tot_b_enc_ns += enc_ns;
            tot_b_dec_ns += dec_ns;
        }
        if (repack_bench && repack_eligible && r_ok && ok) {
            tot_r_in += (unsigned long long)n;
            tot_r_out += (unsigned long long)repack_len;
            tot_r_enc_ns += r_enc_ns;
            tot_r_dec_ns += r_dec_ns;
            tot_b_in += (unsigned long long)n;
            tot_b_out += (unsigned long long)comp_len;
            tot_b_enc_ns += enc_ns;
            tot_b_dec_ns += dec_ns;
        }
        free(in);
        free(comp);
        free(out);
        free(flat);
        free(repack);
    }
    if (repack_bench) {
        if (tot_r_in > 0u) {
            double rbe = (double)tot_r_in / tot_r_enc_ns * 1e3;
            double rbd = (double)tot_r_in / tot_r_dec_ns * 1e3;
            double bbe = (double)tot_b_in / tot_b_enc_ns * 1e3;
            double bbd = (double)tot_b_in / tot_b_dec_ns * 1e3;

            printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %11.2fx\n", "TOTAL",
                   "repack", rbe, rbd, (double)tot_r_in / (double)tot_r_out,
                   bbe > 0.0 ? rbe / bbe : 0.0);
            printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %12s\n", "TOTAL",
                   "block", bbe, bbd, (double)tot_b_in / (double)tot_b_out,
                   "");
        }
    } else if (flat7_bench) {
        if (tot_f_in > 0u) {
            double fbe = (double)tot_f_in / tot_f_enc_ns * 1e3;
            double fbd = (double)tot_f_in / tot_f_dec_ns * 1e3;
            double bbe = (double)tot_b_in / tot_b_enc_ns * 1e3;
            double bbd = (double)tot_b_in / tot_b_dec_ns * 1e3;

            printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %11.2fx\n", "TOTAL",
                   "flat7", fbe, fbd, (double)tot_f_in / (double)tot_f_out,
                   bbe > 0.0 ? fbe / bbe : 0.0);
            printf("%-22s %-6s %10.1f %10.1f %8.2f:1 %12s\n", "TOTAL",
                   "block", bbe, bbd, (double)tot_b_in / (double)tot_b_out,
                   "");
        }
    } else if (tot_in > 0u) {
        printf("%-14s %11llu %11llu %7.2f %10.1f %10.1f %8.1f\n", "TOTAL",
               tot_in, tot_out, (double)tot_in / (double)tot_out,
               (double)tot_passed / tot_enc_ns * 1e3,
               (double)tot_passed / tot_dec_ns * 1e3,
               (tot_enc_ns + tot_dec_ns) / 1e6);
    }
    free_arenas(enc_arenas, threads);
    free_arenas(dec_arenas, threads);
    if (failures) {
        fprintf(stderr, "zpse_cli: %d file(s) failed\n", failures);
        return 1;
    }
    return 0;
}

/* ---- typed-column path ------------------------------------------------
 * A CSV export is prose with commas in it: every number is decimal text, so
 * the codec sees digits where the data has integers and doubles and spends
 * its matches on the formatting instead of on the values. This path splits
 * such a file into columns, types each column, and stores the values in
 * their own form -- 8-byte integers, 8-byte IEEE doubles, exact bytes for
 * anything else -- column by column, behind a header that lets `uncol`
 * rebuild the file byte for byte.
 *
 * Exactness is a property of the typing rule rather than a hope. A column
 * is INT only when every field is the canonical text of its value, and
 * DOUBLE only when every field is exactly what "%.*f" at the column's own
 * decimal count prints from the stored double. Everything else stays STR
 * and keeps its bytes, so an empty field, a leading zero, "1e6" or
 * "33.97320" all survive the round trip. `colcheck` proves it with a
 * memcmp over the whole file rather than a sample.
 *
 * Blob layout (little-endian throughout, 16-byte header):
 *
 *   'Z','C','O','L'  u8 version  u8 ncols  u8 flags  u8 reserved
 *   u32 nrows        u32 header_len        header bytes
 *   per column: u8 type, u8 decimals, u16 reserved, u32 data_len, data[]
 *     INT:    nrows * 8, two's complement
 *     DOUBLE: nrows * 8, IEEE bits
 *     STR:    nrows * (u32 len + bytes)
 *
 * flags: 1 = row 0 is a header (its text is stored once), 2 = CRLF line
 * endings, 4 = the file ends with a newline. */
#define ZCOL_HDR_SIZE 16u
#define ZCOL_T_INT 0u
#define ZCOL_T_DBL 1u
#define ZCOL_T_STR 2u
#define ZCOL_FLAG_HEADER 1u
#define ZCOL_FLAG_CRLF 2u
#define ZCOL_FLAG_TRAIL 4u

typedef struct ZColBuf {
    uint8_t *p;
    size_t len;
    size_t cap;
    int oom;
} ZColBuf;

static void zcb_room(ZColBuf *b, size_t extra)
{
    size_t want;
    uint8_t *np;

    if (b->oom || b->len + extra <= b->cap) {
        return;
    }
    want = b->cap != 0u ? b->cap : (size_t)1u << 16;
    while (want < b->len + extra) {
        want *= 2u;
    }
    np = (uint8_t *)realloc(b->p, want);
    if (np == NULL) {
        b->oom = 1;
        return;
    }
    b->p = np;
    b->cap = want;
}

static void zcb_put(ZColBuf *b, const void *src, size_t n)
{
    zcb_room(b, n);
    if (b->oom) {
        return;
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
}

static void zcb_u8(ZColBuf *b, unsigned v)
{
    uint8_t t = (uint8_t)v;

    zcb_put(b, &t, 1u);
}

static void zcb_u16(ZColBuf *b, unsigned v)
{
    uint8_t t[2];

    t[0] = (uint8_t)v;
    t[1] = (uint8_t)(v >> 8);
    zcb_put(b, t, 2u);
}

static void zcb_u32(ZColBuf *b, unsigned long v)
{
    uint8_t t[4];
    int i;

    for (i = 0; i < 4; i++) {
        t[i] = (uint8_t)(v >> (8 * i));
    }
    zcb_put(b, t, 4u);
}

static void zcb_u64(ZColBuf *b, unsigned long long v)
{
    uint8_t t[8];
    int i;

    for (i = 0; i < 8; i++) {
        t[i] = (uint8_t)(v >> (8 * i));
    }
    zcb_put(b, t, 8u);
}

/* Bounds-checked readers for the unpacker: a blob that does not decode is
 * a corrupt frame, and it must be refused rather than read past. */
static int zcr_u32(const uint8_t *b, size_t len, size_t *pos,
                   unsigned long *out)
{
    if (*pos + 4u > len) {
        return 0;
    }
    *out = (unsigned long)b[*pos] | ((unsigned long)b[*pos + 1u] << 8) |
           ((unsigned long)b[*pos + 2u] << 16) |
           ((unsigned long)b[*pos + 3u] << 24);
    *pos += 4u;
    return 1;
}

static int zcr_u64(const uint8_t *b, size_t len, size_t *pos,
                   unsigned long long *out)
{
    unsigned long long v = 0u;
    int i;

    if (*pos + 8u > len) {
        return 0;
    }
    for (i = 7; i >= 0; i--) {
        v = (v << 8) | (unsigned long long)b[*pos + (size_t)i];
    }
    *pos += 8u;
    *out = v;
    return 1;
}

static int zc_digit(uint8_t c)
{
    return c >= (uint8_t)'0' && c <= (uint8_t)'9';
}

/* Canonical integer text: -?(0|[1-9][0-9]*), at most 18 digits (so the
 * value cannot overflow the int64 it is stored in). */
static int zc_parse_int(const uint8_t *p, uint32_t n, long long *out)
{
    uint32_t i = 0;
    int neg = 0;
    unsigned long long v = 0u;

    if (n == 0u) {
        return 0;
    }
    if (p[0] == (uint8_t)'-') {
        neg = 1;
        i = 1u;
    }
    if (i == n || n - i > 18u) {
        return 0;
    }
    if (p[i] == (uint8_t)'0' && n - i > 1u) {
        return 0;
    }
    for (; i < n; i++) {
        if (!zc_digit(p[i])) {
            return 0;
        }
        v = v * 10u + (unsigned long long)(p[i] - (uint8_t)'0');
    }
    *out = neg ? -(long long)v : (long long)v;
    return 1;
}

/* Decimal text with a fixed number of fraction digits, accepted only when
 * printing the parsed double back at that count reproduces the field. That
 * check is what makes DOUBLE lossless: a field that would print differently
 * (a trailing zero the file did not have, a value that needs 17 digits, a
 * rounded form) is refused and the column stays STR. */
static int zc_parse_dbl(const uint8_t *p, uint32_t n, uint32_t *dec_out,
                        double *out)
{
    char text[64];
    char back[512];
    uint32_t i = 0, dec = 0, digits = 0;
    int dot = 0;
    char *end = NULL;
    double v;
    size_t blen;

    if (n == 0u || n >= sizeof text) {
        return 0;
    }
    if (p[0] == (uint8_t)'-') {
        i = 1u;
    }
    for (; i < n; i++) {
        if (p[i] == (uint8_t)'.') {
            if (dot) {
                return 0;
            }
            dot = 1;
            continue;
        }
        if (!zc_digit(p[i])) {
            return 0;
        }
        digits++;
        if (dot) {
            dec++;
        }
    }
    if (!dot || digits == 0u || dec == 0u || dec > 17u) {
        return 0;
    }
    memcpy(text, p, n);
    text[n] = '\0';
    v = strtod(text, &end);
    if (end == NULL || *end != '\0') {
        return 0;
    }
    snprintf(back, sizeof back, "%.*f", (int)dec, v);
    blen = strlen(back);
    if (blen != (size_t)n || memcmp(back, p, n) != 0) {
        return 0;
    }
    *dec_out = dec;
    *out = v;
    return 1;
}

typedef struct ZCsvRow {
    size_t off;
    uint32_t len;
    size_t next; /* where the following row starts */
} ZCsvRow;

/* The row that starts at `off`, with its terminating CR (if any) already
 * removed from `len`, and the offset of the next row in `next`. `next` is
 * not `off + len + 1` on a CRLF file: `len` excludes the CR, so advancing
 * from it lands on the LF and the following row starts with an empty line.
 * That is what made every CRLF export read as ragged. */
static void zc_row_at(const uint8_t *p, size_t n, size_t off, ZCsvRow *r)
{
    size_t end = off;

    while (end < n && p[end] != (uint8_t)'\n') {
        end++;
    }
    r->off = off;
    r->len = (uint32_t)(end - off);
    r->next = (end < n) ? end + 1u : n;
    if (r->len > 0u && p[off + r->len - 1u] == (uint8_t)'\r') {
        r->len--;
    }
}

/* Field ranges of one row, in O(1) per field for the caller.
 *
 * Quote-aware: a comma inside a "..." field belongs to the field, and a
 * doubled quote inside one is an escaped quote. The quotes are KEPT in the
 * stored bytes, so a STR column reproduces them and `uncol` is byte-exact.
 * Without this, a real export -- a name like "Braund, Mr. Owen Harris" is
 * in the very first public CSV anyone tries -- reads as a different field
 * count on that row and the whole file is refused. */
static uint32_t zc_split(const uint8_t *p, const ZCsvRow *r, uint32_t *off,
                         uint32_t *flen, uint32_t maxcols)
{
    uint32_t c = 0, i = 0, start = 0;

    while (i <= r->len) {
        if (i < r->len && p[r->off + i] == (uint8_t)'"') {
            i++;
            while (i < r->len) {
                if (p[r->off + i] == (uint8_t)'"') {
                    if (i + 1u < r->len &&
                        p[r->off + i + 1u] == (uint8_t)'"') {
                        i += 2u;
                        continue;
                    }
                    i++;
                    break;
                }
                i++;
            }
            continue;
        }
        if (i == r->len || p[r->off + i] == (uint8_t)',') {
            if (c < maxcols) {
                off[c] = start;
                flen[c] = i - start;
            }
            c++;
            start = i + 1u;
        }
        i++;
    }
    return c;
}

typedef struct ZCsv {
    const uint8_t *p;
    size_t n;
    uint32_t ncols;
    uint32_t nrows; /* body rows, header excluded */
    int header;
    int crlf;
    int trail_nl;
    size_t body; /* offset of the first body row */
} ZCsv;

/* Shape of the file: constant field count, one line ending style, and
 * whether row 0 is a header (it is when any of its fields is not a number).
 */
static int zc_scan(const uint8_t *p, size_t n, ZCsv *z)
{
    size_t off = 0;
    uint32_t cols = 0, rows = 0;
    int first = 1, cr = 0, lf = 0;

    if (n == 0u) {
        return 0;
    }
    z->p = p;
    z->n = n;
    z->trail_nl = (p[n - 1u] == (uint8_t)'\n') ? 1 : 0;
    while (off < n) {
        uint32_t o[1], l[1];
        ZCsvRow r;
        uint32_t c;

        zc_row_at(p, n, off, &r);
        if (r.len > 0u && p[r.off + r.len] == (uint8_t)'\r') {
            cr = 1;
        } else {
            lf = 1;
        }
        if (cr && lf) {
            return 0;
        }
        /* maxcols 0: count only, the field ranges are not wanted here. */
        c = zc_split(p, &r, o, l, 0u);
        if (first) {
            cols = c;
            first = 0;
        } else if (c != cols) {
            return 0;
        }
        rows++;
        off = r.next;
    }
    if (cols == 0u || rows == 0u) {
        return 0;
    }
    z->ncols = cols;
    z->crlf = cr;
    z->header = 0;
    z->body = 0u;
    if (rows > 1u) {
        ZCsvRow r0;
        uint32_t fo[256], flr[256];
        uint32_t c;

        zc_row_at(p, n, 0u, &r0);
        if (cols > 256u) {
            return 0;
        }
        (void)zc_split(p, &r0, fo, flr, cols);
        for (c = 0; c < cols; c++) {
            const uint8_t *f = p + r0.off + fo[c];
            uint32_t fl = flr[c];
            long long iv;
            uint32_t dv;
            double fv;

            if (!zc_parse_int(f, fl, &iv) &&
                !zc_parse_dbl(f, fl, &dv, &fv)) {
                z->header = 1;
                break;
            }
        }
        if (z->header) {
            size_t e = 0u;

            while (e < n && p[e] != (uint8_t)'\n') {
                e++;
            }
            z->body = (e < n) ? e + 1u : n;
        }
    }
    z->nrows = rows - (uint32_t)(z->header ? 1 : 0);
    return 1;
}

/* Column type and, for DOUBLE, its decimal count. `n`/`body` come from the
 * scan. INT is tried first: it is the smaller form and the check is cheap. */
static int zc_type(const ZCsv *z, uint32_t col, uint32_t *dec_out)
{
    size_t off = z->body;
    int all_int = 1, all_dbl = 1;
    uint32_t dec = 0u;
    int have_dec = 0;
    uint32_t fo[256], flr[256];

    while (off < z->n) {
        ZCsvRow r;
        const uint8_t *f;
        uint32_t fl;
        long long iv;
        uint32_t dv = 0u;
        double fv;

        zc_row_at(z->p, z->n, off, &r);
        if (zc_split(z->p, &r, fo, flr, z->ncols) <= col) {
            return -1;
        }
        f = z->p + r.off + fo[col];
        fl = flr[col];
        off = r.next;
        if (all_int && !zc_parse_int(f, fl, &iv)) {
            all_int = 0;
        }
        if (!all_dbl) {
            continue;
        }
        if (!zc_parse_dbl(f, fl, &dv, &fv) ||
            (have_dec && dv != dec)) {
            all_dbl = 0;
        } else if (!have_dec) {
            dec = dv;
            have_dec = 1;
        }
    }
    if (all_int) {
        return ZCOL_T_INT;
    }
    if (all_dbl) {
        *dec_out = dec;
        return ZCOL_T_DBL;
    }
    return ZCOL_T_STR;
}

static int zc_build(const uint8_t *csv, size_t n, uint8_t **blob_out,
                    size_t *blob_len, ZCsv *z_out)
{
    ZCsv z;
    ZColBuf out;
    uint32_t c;
    uint32_t header_len = 0u;
    size_t header_off = 0u;
    uint32_t fo[256], flr[256];

    if (!zc_scan(csv, n, &z)) {
        fprintf(stderr, "zpse_cli: not a rectangular comma-separated file\n");
        return 1;
    }
    if (z.nrows == 0u) {
        fprintf(stderr, "zpse_cli: no data rows\n");
        return 1;
    }
    out.p = NULL;
    out.len = 0u;
    out.cap = 0u;
    out.oom = 0;
    if (z.header) {
        ZCsvRow r0;

        zc_row_at(csv, n, 0u, &r0);
        header_off = r0.off;
        header_len = r0.len;
    }
    zcb_put(&out, "ZCOL", 4u);
    zcb_u8(&out, 1u);
    zcb_u8(&out, z.ncols);
    zcb_u8(&out, (unsigned)((z.header ? ZCOL_FLAG_HEADER : 0u) |
                            (z.crlf ? ZCOL_FLAG_CRLF : 0u) |
                            (z.trail_nl ? ZCOL_FLAG_TRAIL : 0u)));
    zcb_u8(&out, 0u);
    zcb_u32(&out, z.nrows);
    zcb_u32(&out, header_len);
    if (header_len != 0u) {
        zcb_put(&out, csv + header_off, header_len);
    }
    for (c = 0; c < z.ncols; c++) {
        ZColBuf data;
        uint32_t dec = 0u;
        int type = zc_type(&z, c, &dec);
        size_t off = z.body;

        data.p = NULL;
        data.len = 0u;
        data.cap = 0u;
        data.oom = 0;
        if (type < 0) {
            fprintf(stderr, "zpse_cli: column %u is short\n", (unsigned)c);
            free(out.p);
            return 1;
        }
        while (off < n) {
            ZCsvRow r;
            const uint8_t *f;
            uint32_t fl;

            zc_row_at(csv, n, off, &r);
            if (zc_split(csv, &r, fo, flr, z.ncols) <= c) {
                break;
            }
            f = csv + r.off + fo[c];
            fl = flr[c];
            off = r.next;
            if (type == ZCOL_T_INT) {
                long long iv = 0;

                (void)zc_parse_int(f, fl, &iv);
                zcb_u64(&data, (unsigned long long)iv);
            } else if (type == ZCOL_T_DBL) {
                uint32_t dv = 0u;
                double fv = 0.0;
                unsigned long long bits = 0u;

                (void)zc_parse_dbl(f, fl, &dv, &fv);
                memcpy(&bits, &fv, 8u);
                zcb_u64(&data, bits);
            } else {
                zcb_u32(&data, (unsigned long)fl);
                zcb_put(&data, f, fl);
            }
        }
        if (data.oom || out.oom) {
            fprintf(stderr, "zpse_cli: out of memory\n");
            free(data.p);
            free(out.p);
            return 1;
        }
        zcb_u8(&out, (unsigned)type);
        zcb_u8(&out, dec);
        zcb_u16(&out, 0u);
        zcb_u32(&out, (unsigned long)data.len);
        zcb_put(&out, data.p, data.len);
        free(data.p);
    }
    if (out.oom) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        free(out.p);
        return 1;
    }
    *blob_out = out.p;
    *blob_len = out.len;
    if (z_out != NULL) {
        *z_out = z;
    }
    return 0;
}

/* Rebuild the CSV from a blob. Returns 0 on success. */
static int zc_unpack(const uint8_t *b, size_t len, uint8_t **csv_out,
                     size_t *csv_len)
{
    ZColBuf out;
    size_t pos = ZCOL_HDR_SIZE;
    unsigned long nrows = 0u, header_len = 0u;
    unsigned cols, flags, version;
    const uint8_t *header = NULL;
    const uint8_t *data[256];
    size_t dlen[256];
    uint32_t dec[256];
    uint8_t type[256];
    size_t cur[256];
    const char *nl;
    size_t nl_len;
    unsigned long row;
    unsigned c;

    if (len < ZCOL_HDR_SIZE || memcmp(b, "ZCOL", 4u) != 0) {
        fprintf(stderr, "zpse_cli: not a typed-column blob\n");
        return 1;
    }
    version = b[4];
    cols = b[5];
    flags = b[6];
    if (version != 1u || cols == 0u || cols > 256u || b[7] != 0u) {
        fprintf(stderr, "zpse_cli: unsupported blob version\n");
        return 1;
    }
    pos = 8u;
    if (!zcr_u32(b, len, &pos, &nrows)) {
        fprintf(stderr, "zpse_cli: truncated blob\n");
        return 1;
    }
    if (!zcr_u32(b, len, &pos, &header_len)) {
        fprintf(stderr, "zpse_cli: truncated blob\n");
        return 1;
    }
    if (header_len > len - pos) {
        fprintf(stderr, "zpse_cli: truncated header\n");
        return 1;
    }
    header = b + pos;
    pos += header_len;
    for (c = 0; c < cols; c++) {
        unsigned long d = 0u;

        if (pos + 8u > len) {
            fprintf(stderr, "zpse_cli: truncated column table\n");
            return 1;
        }
        type[c] = b[pos];
        dec[c] = b[pos + 1u];
        pos += 4u;
        if (!zcr_u32(b, len, &pos, &d) || d > len - pos) {
            fprintf(stderr, "zpse_cli: truncated column data\n");
            return 1;
        }
        data[c] = b + pos;
        dlen[c] = (size_t)d;
        cur[c] = 0u;
        pos += (size_t)d;
        if (type[c] > ZCOL_T_STR) {
            fprintf(stderr, "zpse_cli: bad column type\n");
            return 1;
        }
    }
    nl = (flags & ZCOL_FLAG_CRLF) ? "\r\n" : "\n";
    nl_len = (flags & ZCOL_FLAG_CRLF) ? 2u : 1u;
    out.p = NULL;
    out.len = 0u;
    out.cap = 0u;
    out.oom = 0;
    if ((flags & ZCOL_FLAG_HEADER) != 0u && header_len != 0u) {
        zcb_put(&out, header, (size_t)header_len);
        zcb_put(&out, nl, nl_len);
    }
    for (row = 0; row < nrows; row++) {
        for (c = 0; c < cols; c++) {
            if (c != 0u) {
                zcb_put(&out, ",", 1u);
            }
            if (type[c] == ZCOL_T_INT) {
                unsigned long long bits = 0u;
                long long iv;
                char text[24];

                size_t p2 = cur[c];

                if (!zcr_u64(data[c], dlen[c], &p2, &bits)) {
                    goto trunc;
                }
                cur[c] = p2;
                iv = (long long)bits;
                snprintf(text, sizeof text, "%lld", iv);
                zcb_put(&out, text, strlen(text));
            } else if (type[c] == ZCOL_T_DBL) {
                unsigned long long bits = 0u;
                double v;
                char text[512];

                size_t p2 = cur[c];

                if (!zcr_u64(data[c], dlen[c], &p2, &bits)) {
                    goto trunc;
                }
                cur[c] = p2;
                memcpy(&v, &bits, 8u);
                snprintf(text, sizeof text, "%.*f", (int)dec[c], v);
                zcb_put(&out, text, strlen(text));
            } else {
                unsigned long fl = 0u;

                size_t p2 = cur[c];

                if (!zcr_u32(data[c], dlen[c], &p2, &fl) ||
                    fl > dlen[c] - p2) {
                    goto trunc;
                }
                cur[c] = p2 + (size_t)fl;
                zcb_put(&out, data[c] + p2, (size_t)fl);
            }
        }
        if (row + 1u < nrows || (flags & ZCOL_FLAG_TRAIL) != 0u) {
            zcb_put(&out, nl, nl_len);
        }
    }
    if (out.oom) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        free(out.p);
        return 1;
    }
    *csv_out = out.p;
    *csv_len = out.len;
    return 0;

trunc:
    fprintf(stderr, "zpse_cli: column data ran out\n");
    free(out.p);
    return 1;
}

/* compress IN to OUT through the frame encoder, at the given level. */
static int col_compress(const uint8_t *in, size_t n, uint8_t **out,
                        size_t *out_len, uint32_t block, uint32_t enc_flags)
{
    size_t cap = frame_bound(n, block, 0);
    void **arenas = alloc_arenas(1u, ZPSE_ENC_ARENA_SIZE);
    uint8_t *buf = (uint8_t *)malloc(cap + 1u);
    int rc;

    if (arenas == NULL || buf == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        free_arenas(arenas, 1u);
        free(buf);
        return 1;
    }
    rc = compress_mem(in, n, buf, cap, out_len, block, enc_flags, 0, arenas,
                      1u);
    free_arenas(arenas, 1u);
    if (rc != 0) {
        fprintf(stderr, "zpse_cli: compress failed rc=%d\n", rc);
        free(buf);
        return 1;
    }
    *out = buf;
    return 0;
}

static int col_decompress(const uint8_t *in, size_t in_len, uint8_t **out,
                          size_t *out_len, size_t cap)
{
    void **arenas = alloc_arenas(1u, ZPSE_DEC_ARENA_SIZE);
    uint8_t *buf = (uint8_t *)malloc(cap + 1u);
    int rc;

    if (arenas == NULL || buf == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        free_arenas(arenas, 1u);
        free(buf);
        return 1;
    }
    rc = zpse_decompress_frame_parallel(in, in_len, buf, cap, out_len, 1u,
                                        arenas, 1u);
    free_arenas(arenas, 1u);
    if (rc != 0) {
        fprintf(stderr, "zpse_cli: decompress failed rc=%d\n", rc);
        free(buf);
        return 1;
    }
    *out = buf;
    return 0;
}

/* `col IN.csv OUT`: type the columns, then compress the typed blob. */
static int cmd_col(int argc, char **argv)
{
    uint32_t enc_flags = 0u;
    uint32_t block = ZPSE_MAX_BLOCK_SIZE;
    uint8_t *csv;
    uint8_t *blob = NULL;
    uint8_t *packed = NULL;
    size_t csv_len = 0u;
    size_t blob_len = 0u;
    size_t packed_len = 0u;
    ZCsv z;
    int i = 2;

    for (; i < argc && argv[i][0] == '-'; i++) {
        if (argv[i][1] == 'l' && ++i < argc) {
            if (!parse_level(argv[i], &enc_flags)) {
                return 2;
            }
        } else if (argv[i][1] == 'b' && ++i < argc) {
            char *end = NULL;
            unsigned long v = strtoul(argv[i], &end, 10);

            if (end == argv[i] || *end != '\0' || v < 1u ||
                v > (unsigned long)ZPSE_MAX_BLOCK_SIZE) {
                fprintf(stderr, "zpse_cli: bad block size '%s'\n", argv[i]);
                return 2;
            }
            block = (uint32_t)v;
        } else {
            fprintf(stderr, "zpse_cli: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    if (argc - i != 2) {
        fprintf(stderr, "usage: zpse_cli col [opts] IN.csv OUT\n");
        return 2;
    }
    csv = read_file(argv[i], &csv_len);
    if (csv == NULL) {
        return 1;
    }
    if (zc_build(csv, csv_len, &blob, &blob_len, &z) != 0) {
        free(csv);
        return 1;
    }
    if (col_compress(blob, blob_len, &packed, &packed_len, block,
                     enc_flags) != 0) {
        free(blob);
        free(csv);
        return 1;
    }
    printf("%s: %u rows x %u columns%s, csv %lu -> typed %lu -> packed %lu "
           "(%.2f:1 on the typed form, %.2f:1 against the csv)\n",
           argv[i], (unsigned)z.nrows, (unsigned)z.ncols,
           z.header ? ", with header" : "", (unsigned long)csv_len,
           (unsigned long)blob_len, (unsigned long)packed_len,
           packed_len ? (double)blob_len / (double)packed_len : 0.0,
           packed_len ? (double)csv_len / (double)packed_len : 0.0);
    i = write_file(argv[argc - 1], packed, packed_len);
    free(packed);
    free(blob);
    free(csv);
    return i;
}

/* `uncol IN OUT`: decompress, then rebuild the CSV from the typed blob. */
static int cmd_uncol(int argc, char **argv)
{
    uint8_t *packed;
    uint8_t *blob = NULL;
    uint8_t *csv = NULL;
    size_t packed_len = 0u;
    size_t blob_len = 0u;
    size_t csv_len = 0u;
    int rc;

    if (argc != 4) {
        fprintf(stderr, "usage: zpse_cli uncol IN OUT.csv\n");
        return 2;
    }
    packed = read_file(argv[2], &packed_len);
    if (packed == NULL) {
        return 1;
    }
    /* The frame header carries the uncompressed size, so the blob never has
     * to be sized by guesswork. */
    if (col_decompress(packed, packed_len, &blob, &blob_len,
                       ((packed_len * 64u) + ((size_t)1u << 20))) != 0) {
        free(packed);
        return 1;
    }
    rc = zc_unpack(blob, blob_len, &csv, &csv_len);
    free(blob);
    free(packed);
    if (rc != 0) {
        return 1;
    }
    rc = write_file(argv[3], csv, csv_len);
    free(csv);
    return rc;
}

/* `colcheck IN.csv`: the round trip and the two ratios, in one pass. The
 * text frame is the same bytes through the same encoder, so the difference
 * between the two packed sizes is the whole value of typing the columns. */
static int cmd_colcheck(int argc, char **argv)
{
    uint32_t enc_flags = 0u;
    uint32_t block = ZPSE_MAX_BLOCK_SIZE;
    uint8_t *csv;
    uint8_t *blob = NULL;
    uint8_t *packed = NULL;
    uint8_t *text_packed = NULL;
    uint8_t *back_blob = NULL;
    uint8_t *back_csv = NULL;
    size_t csv_len = 0u, blob_len = 0u, packed_len = 0u, text_len = 0u;
    size_t back_blob_len = 0u, back_csv_len = 0u;
    ZCsv z;
    int i = 2;
    int rc = 0;
    double t0, enc_ns = 0.0, dec_ns = 0.0;

    for (; i < argc && argv[i][0] == '-'; i++) {
        if (argv[i][1] == 'l' && ++i < argc) {
            if (!parse_level(argv[i], &enc_flags)) {
                return 2;
            }
        } else {
            fprintf(stderr, "zpse_cli: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    if (argc - i != 1) {
        fprintf(stderr, "usage: zpse_cli colcheck [opts] IN.csv\n");
        return 2;
    }
    csv = read_file(argv[i], &csv_len);
    if (csv == NULL) {
        return 1;
    }
    if (zc_build(csv, csv_len, &blob, &blob_len, &z) != 0) {
        free(csv);
        return 1;
    }
    t0 = now_ns();
    if (col_compress(blob, blob_len, &packed, &packed_len, block,
                     enc_flags) != 0) {
        free(blob);
        free(csv);
        return 1;
    }
    enc_ns = now_ns() - t0;
    if (col_compress(csv, csv_len, &text_packed, &text_len, block,
                     enc_flags) != 0) {
        free(packed);
        free(blob);
        free(csv);
        return 1;
    }
    t0 = now_ns();
    if (col_decompress(packed, packed_len, &back_blob, &back_blob_len,
                       blob_len + 4096u) != 0) {
        free(text_packed);
        free(packed);
        free(blob);
        free(csv);
        return 1;
    }
    dec_ns = now_ns() - t0;
    if (back_blob_len != blob_len || memcmp(back_blob, blob, blob_len) != 0) {
        fprintf(stderr, "zpse_cli: typed blob did not round-trip\n");
        rc = 1;
    } else if (zc_unpack(back_blob, back_blob_len, &back_csv,
                         &back_csv_len) != 0) {
        rc = 1;
    } else if (back_csv_len != csv_len || memcmp(back_csv, csv, csv_len) != 0) {
        fprintf(stderr, "zpse_cli: rebuilt csv differs from the input\n");
        rc = 1;
    }
    printf("%-24s %10lu %10lu %6.2f %10lu %6.2f %6.1f%% %8.1f %8.1f %s\n",
           argv[i], (unsigned long)csv_len, (unsigned long)text_len,
           text_len ? (double)csv_len / (double)text_len : 0.0,
           (unsigned long)packed_len,
           packed_len ? (double)csv_len / (double)packed_len : 0.0,
           text_len ? 100.0 * (1.0 - (double)packed_len / (double)text_len)
                    : 0.0,
           enc_ns > 0.0 ? (double)blob_len / enc_ns * 1e3 : 0.0,
           dec_ns > 0.0 ? (double)blob_len / dec_ns * 1e3 : 0.0,
           rc == 0 ? "ok" : "FAIL");
    free(back_csv);
    free(back_blob);
    free(text_packed);
    free(packed);
    free(blob);
    free(csv);
    return rc;
}

/* `dir`: print the frame's random-access directory. The index is the thing
 * that makes a zpse frame seekable -- block n's offset and packed size, one
 * entry each, checksummed as a table -- and nothing in the CLI surfaced it
 * before, so a caller had to link the library to see what a frame carries.
 * The decoded length comes from the entry's block header via the same
 * seek path a random read uses, so what this prints is what a seek gets. */
static int cmd_dir(int argc, char **argv)
{
    uint8_t *in;
    uint8_t *blk;
    size_t in_len = 0u;
    uint32_t count = 0u;
    uint32_t k;
    void *dec_arena;
    int rc;

    if (argc != 3) {
        fprintf(stderr, "usage: zpse_cli dir FILE\n");
        return 2;
    }
    in = read_file(argv[2], &in_len);
    if (in == NULL) {
        return 1;
    }
    if (zpse_flat7_is_frame(in, in_len)) {
        fprintf(stderr, "zpse_cli: FLAT7 has no frame index\n");
        free(in);
        return 1;
    }
    rc = zpse_frame_index_count(in, in_len, &count);
    if (rc != ZPSE_OK) {
        fprintf(stderr, "zpse_cli: %s carries no usable index (rc=%d); "
                        "encode it with -x\n", argv[2], rc);
        free(in);
        return 1;
    }
    blk = (uint8_t *)malloc(ZPSE_MAX_BLOCK_SIZE);
    dec_arena = ZPSE_CLI_ALIGNED(64u, ZPSE_DEC_ARENA_SIZE);
    if (blk == NULL || dec_arena == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        ZPSE_CLI_FREE(dec_arena);
        free(blk);
        free(in);
        return 1;
    }
    printf("%s: %u blocks\n", argv[2], (unsigned)count);
    printf("%-8s %10s\n", "block", "bytes");
    for (k = 0; k < count; k++) {
        uint32_t got = 0u;

        rc = zpse_seek_frame(in, in_len, k, blk, (uint32_t)ZPSE_MAX_BLOCK_SIZE,
                             &got, dec_arena);
        if (rc != ZPSE_OK) {
            fprintf(stderr, "zpse_cli: block %u does not seek (rc=%d)\n",
                    (unsigned)k, rc);
            ZPSE_CLI_FREE(dec_arena);
            free(blk);
            free(in);
            return 1;
        }
        printf("%-8u %10u\n", (unsigned)k, (unsigned)got);
    }
    ZPSE_CLI_FREE(dec_arena);
    free(blk);
    free(in);
    return 0;
}

/* `x`: decode one block through the index, without touching the others.
 * This is the CLI's random read; on a multi-GiB frame it is the difference
 * between decoding one block and decoding the file. */
static int cmd_seek(int argc, char **argv)
{
    uint8_t *in;
    uint8_t *out;
    size_t in_len = 0u;
    uint32_t got = 0u;
    unsigned long idx;
    char *end = NULL;
    void *dec_arena;
    int rc;

    if (argc != 5) {
        fprintf(stderr, "usage: zpse_cli x FILE BLOCK OUT\n");
        return 2;
    }
    idx = strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || idx > 0xFFFFFFFFu) {
        fprintf(stderr, "zpse_cli: bad block index '%s'\n", argv[3]);
        return 2;
    }
    in = read_file(argv[2], &in_len);
    if (in == NULL) {
        return 1;
    }
    if (zpse_flat7_is_frame(in, in_len)) {
        fprintf(stderr, "zpse_cli: FLAT7 has no frame index\n");
        free(in);
        return 1;
    }
    out = (uint8_t *)malloc(ZPSE_MAX_BLOCK_SIZE);
    dec_arena = ZPSE_CLI_ALIGNED(64u, ZPSE_DEC_ARENA_SIZE);
    if (out == NULL || dec_arena == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        ZPSE_CLI_FREE(dec_arena);
        free(out);
        free(in);
        return 1;
    }
    rc = zpse_seek_frame(in, in_len, (uint32_t)idx, out,
                         (uint32_t)ZPSE_MAX_BLOCK_SIZE, &got, dec_arena);
    if (rc != ZPSE_OK) {
        fprintf(stderr, "zpse_cli: seek of block %lu failed (rc=%d)\n", idx,
                rc);
        ZPSE_CLI_FREE(dec_arena);
        free(out);
        free(in);
        return 1;
    }
    rc = write_file(argv[4], out, (size_t)got);
    ZPSE_CLI_FREE(dec_arena);
    free(out);
    free(in);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: zpse_cli c [opts] IN OUT\n"
                        "       zpse_cli d IN OUT\n"
                        "       zpse_cli x FILE BLOCK OUT\n"
                        "       zpse_cli dir FILE\n"
                        "       zpse_cli col [opts] IN.csv OUT\n"
                        "       zpse_cli uncol IN OUT.csv\n"
                        "       zpse_cli colcheck [opts] IN.csv\n"
                        "       zpse_cli bench [opts] FILE...\n"
                        "       zpse_cli --version\n");
        return 2;
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("zpse %s\n", ZPSE_VERSION_STRING);
        return 0;
    }
    if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench(argc, argv);
    }
    if (strcmp(argv[1], "dir") == 0) {
        return cmd_dir(argc, argv);
    }
    if (strcmp(argv[1], "x") == 0) {
        return cmd_seek(argc, argv);
    }
    if (strcmp(argv[1], "col") == 0) {
        return cmd_col(argc, argv);
    }
    if (strcmp(argv[1], "uncol") == 0) {
        return cmd_uncol(argc, argv);
    }
    if (strcmp(argv[1], "colcheck") == 0) {
        return cmd_colcheck(argc, argv);
    }
    if (argc >= 2 && argv[1][1] == '\0' &&
        (argv[1][0] == 'c' || argv[1][0] == 'd')) {
        return cmd_codec(argc, argv);
    }
    fprintf(stderr, "zpse_cli: unknown command '%s'\n", argv[1]);
    return 2;
}
