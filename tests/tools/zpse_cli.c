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
 *   -c         emit the per-block payload CRC32-C (verified on decode)
 *   -b N       nominal block size, 1..65536 (default 65536)
 *   -x         append the FrameIndex (random access); bench also seeks
 *   -t N       encode with N worker threads (default 1: the single-core
 *              path every figure in the README was measured on)
 *   -r N       bench repetitions per file (0 = pick from the file size)
 *   -n N       bench: time only the first N bytes of each file (0 = the
 *              whole file), for one case without the whole corpus
 *   -q         bench: summary line only, no per-file rows
 *
 * Exit 0 on success, 1 on an I/O, codec or verification failure, 2 on a
 * usage error. The arenas are the documented sizes (448 KiB encode /
 * 320 KiB decode), 64-byte aligned, allocated once per run -- the same
 * shape an embedder uses.
 *
 * Throughputs are decimal (1 MB = 1e6 bytes) and measure the codec only:
 * a bench reads each input once, before the clock starts, and keeps every
 * buffer in memory.
 *
 * A single-threaded frame is streamed, not buffered: the input is read a slab
 * at a time and each block is written to the output as it is finished, so
 * neither the file nor the frame is ever resident whole -- and that holds for
 * -x too, because the table is recorded as the blocks go rather than walked
 * off the file afterwards. -t and `bench` use the buffered worker call, which
 * needs the whole frame: `bench` is measuring the codec rather than writing a
 * file, and the streamed worker path is withheld until its slot handoff is
 * proven (see src/parallel.c). */
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
#endif

#include "zpse/zpse.h"
#include "zpse/block.h"
#include "zpse/frame.h"
#include "zpse/parallel.h"

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

/* Decode a frame block by block, writing each block out as it is finished.
 *
 * The whole-frame path has to hold the entire output in memory before it can
 * write a byte of it, and on this corpus that is a 202 MiB allocation faulted
 * in one page at a time -- measured at about a third of the decode's wall
 * time, against an in-memory decode of the same bytes that reuses one buffer.
 * On a large file it also means a resident set as large as the file.
 *
 * The blocks are independent by construction (that is what the 64 KiB
 * framing is for), so a sequential decode can walk them and write each one
 * out of a single reused buffer. Nothing is validated less: every block goes
 * through zpse_decompress_block, a block that does not decode to the length
 * its header claims is refused, the block count and content size still have
 * to agree with the frame header, and a frame that carries no index is the
 * only kind that takes this path (see the caller).
 *
 * Returns 0 on success, 1 on any failure, matching write_file. */
static int decode_stream(const uint8_t *in, size_t in_len, const char *out_path,
                         const ZpseFrameHeader *fh, void *dec_arena)
{
    FILE *f;
    uint8_t out[ZPSE_MAX_BLOCK_SIZE];
    size_t pos = ZPSE_FRAME_HEADER_SIZE;
    size_t out_pos = 0u;
    size_t nblocks = 0u;
    int have_last = 0;
    int has_count = (fh->frame_flags & (uint16_t)ZPSE_F_BLOCK_COUNT) != 0u;
    int has_content = (fh->frame_flags & (uint16_t)ZPSE_F_CONTENT_SIZE) != 0u;

    f = fopen(out_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "zpse_cli: cannot write %s\n", out_path);
        return 1;
    }
    while (pos < in_len) {
        size_t rem = in_len - pos;
        uint32_t avail = rem > (size_t)UINT32_MAX ? UINT32_MAX : (uint32_t)rem;
        uint32_t comp_sz = 0u;
        uint32_t uncomp_sz = 0u;
        uint32_t blk_out = 0u;
        int is_last;

        if (zpse_block_probe(in + pos, avail, &comp_sz, &uncomp_sz) != PSE_OK ||
            comp_sz < ZPSE_BLOCK_HEADER_SIZE || (size_t)comp_sz > rem ||
            uncomp_sz < ZPSE_MIN_BLOCK_SIZE ||
            uncomp_sz > ZPSE_MAX_BLOCK_SIZE) {
            fprintf(stderr, "zpse_cli: bad block at offset %zu\n", pos);
            fclose(f);
            return 1;
        }
        /* comp_sz >= 32 and <= rem, so in[pos + 3] is in bounds. */
        is_last = (in[pos + 3] & ZPSE_B_LAST) != 0;
        if (zpse_decompress_block(in + pos, comp_sz, out,
                                  (uint32_t)sizeof out, &blk_out,
                                  dec_arena) != PSE_OK ||
            blk_out != uncomp_sz) {
            fprintf(stderr, "zpse_cli: block at offset %zu does not decode\n",
                    pos);
            fclose(f);
            return 1;
        }
        if (blk_out > 0u && fwrite(out, 1u, blk_out, f) != blk_out) {
            fprintf(stderr, "zpse_cli: short write on %s\n", out_path);
            fclose(f);
            return 1;
        }
        out_pos += (size_t)blk_out;
        pos += (size_t)comp_sz;
        nblocks++;
        if (is_last) {
            have_last = 1;
            break;
        }
        if (has_count && (uint64_t)nblocks > (uint64_t)fh->block_count) {
            fprintf(stderr, "zpse_cli: more blocks than the frame declares\n");
            fclose(f);
            return 1;
        }
    }
    if (!have_last) {
        if (pos != in_len) {
            fprintf(stderr, "zpse_cli: frame ends mid-block\n");
            fclose(f);
            return 1;
        }
        if (!has_count || (uint64_t)nblocks != (uint64_t)fh->block_count) {
            fprintf(stderr, "zpse_cli: block count disagrees with the frame\n");
            fclose(f);
            return 1;
        }
    }
    if (has_count && (uint64_t)nblocks != (uint64_t)fh->block_count) {
        fprintf(stderr, "zpse_cli: block count disagrees with the frame\n");
        fclose(f);
        return 1;
    }
    if (has_content && (uint64_t)out_pos != (uint64_t)fh->content_size) {
        fprintf(stderr, "zpse_cli: content size disagrees with the frame\n");
        fclose(f);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "zpse_cli: close failed on %s\n", out_path);
        return 1;
    }
    return 0;
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
 * buffer, exactly as the decoder's decode_stream already does.
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
        return PSE_BAD_ARG;
    }
    if (len > 0u && fwrite(blk, 1u, len, s->f) != len) {
        fprintf(stderr, "zpse_cli: short write on %s\n", s->path);
        s->failed = 1;
        return PSE_BAD_ARG;
    }
    if (s->off != NULL) {
        s->off[seq] = (uint32_t)s->pos;
        s->len[seq] = (uint32_t)len;
    }
    s->pos += len;
    return 0;
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
    uint8_t *slot_mem = NULL;
    int *ready = NULL;
    uint32_t *lens = NULL;
    uint8_t hdr[ZPSE_FRAME_HEADER_SIZE];
    /* One slot per worker plus two, so a worker always has somewhere to put
     * the block after its own while the writer is still on an earlier one;
     * that band is all the output this ever holds. */
    size_t stride = (size_t)block + (size_t)ZPSE_BLOCK_HEADER_SIZE;
    unsigned ring = (threads > 1u ? threads + 2u : 1u);
    size_t slot_cap = stride * (size_t)ring;
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
                                (uint32_t)nblocks, 1, indexed) != PSE_OK ||
        zpse_frame_header_write(&fh, hdr) != PSE_OK) {
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
    in_buf = (uint8_t *)malloc((size_t)slab_blocks * (size_t)block);
    slot_mem = (uint8_t *)malloc(slot_cap);
    ready = (int *)malloc(sizeof(int) * (size_t)ring);
    lens = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)ring);
    if (in_buf == NULL || slot_mem == NULL || ready == NULL || lens == NULL) {
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
            int final;

            if (want > slab_blocks) {
                want = slab_blocks;
                room = want * (size_t)block;
            }
            if (got > room) {
                got = room;
            }
            if (got == 0u || fread(in_buf, 1u, got, fi) != got) {
                fprintf(stderr, "zpse_cli: short read on %s\n", in_path);
                goto done;
            }
            final = (done_blocks + want) >= nblocks ? 1 : 0;
            if (zpse_compress_frame_stream(in_buf, got, block, enc_flags,
                                           threads, (uint32_t)done_blocks,
                                           final, arenas, threads, slot_mem,
                                           slot_cap, ready, lens, stream_emit,
                                           &sink) != PSE_OK ||
                sink.failed != 0) {
                fprintf(stderr, "zpse_cli: blocks at %zu do not encode\n",
                        done_blocks);
                goto done;
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
                                    &idx_len) != PSE_OK) {
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
    if (fo != NULL) {
        fclose(fo);
    }
    free(in_buf);
    free(slot_mem);
    free(ready);
    free(lens);
    free(blk_off);
    free(blk_len);
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
    unsigned threads = 1u;
    const char *in_path;
    const char *out_path;
    uint8_t *in;
    uint8_t *out;
    size_t in_len = 0u;
    size_t out_len = 0u;
    size_t out_cap;
    void **enc_arenas = NULL;
    void **dec_arenas = NULL;
    int rc;
    int stream_dec = 0;
    ZpseFrameHeader dhead;
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
    /* One thread streams: the frame is written block by block straight to the
     * output file, and the index rides along because the writer records where
     * each block went as it writes it (see zpse_frame_stream_index). Nothing
     * here holds the input or the frame, which is what a whole-file encode
     * spends a quarter of its time on. -t takes the buffered worker call: the
     * streamed worker path is withheld, see the note in src/parallel.c. */
    if (compress && threads == 1u) {
        enc_arenas = alloc_arenas(1u, ZPSE_ENC_ARENA_SIZE);
        if (enc_arenas == NULL) {
            fprintf(stderr, "zpse_cli: arena allocation failed\n");
            return 1;
        }
        rc = encode_stream(in_path, out_path, block, enc_flags, indexed, 1u,
                           (void *const *)enc_arenas);
        free_arenas(enc_arenas, 1u);
        return rc;
    }
    in = read_file(in_path, &in_len);
    if (in == NULL) {
        return 1;
    }
    if (compress) {
        size_t ser = frame_bound(in_len, block, indexed);
        size_t par = zpse_parallel_bound(in_len, block, indexed);

        /* The worker path reserves a fixed slot per block and compacts
         * afterwards, so it wants the larger buffer; the compaction never
         * costs a byte of the result. */
        out_cap = par > ser ? par : ser;
        enc_arenas = alloc_arenas(threads, ZPSE_ENC_ARENA_SIZE);
    } else {
        ZpseFrameHeader h;

        memset(&h, 0, sizeof h);
        /* The frame the encoder writes always carries the content size
         * (ZPSE_F_CONTENT_SIZE, sentinel UINT64_MAX when absent), so the
         * decompressor can size its output exactly instead of guessing. */
        if (in_len < 32u ||
            zpse_frame_header_read(&h, in) != 0 ||
            (h.frame_flags & (uint16_t)ZPSE_F_CONTENT_SIZE) == 0u ||
            h.content_size == UINT64_MAX ||
            h.content_size > (uint64_t)SIZE_MAX - 1u) {
            fprintf(stderr, "zpse_cli: not a zpse frame with a content size\n");
            free(in);
            return 1;
        }
        out_cap = (size_t)h.content_size;
        /* A frame without an index decodes sequentially, and that needs no
         * output buffer at all: decode_stream writes each block as it is
         * finished. A frame that declares an index still takes the
         * whole-frame path, which is what validates the index against the
         * stream it describes. */
        if (threads == 1u &&
            (h.frame_flags & (uint16_t)ZPSE_F_FRAME_INDEX) == 0u) {
            stream_dec = 1;
            dhead = h;
            out_cap = 1u;
        }
        dec_arenas = alloc_arenas(threads, ZPSE_DEC_ARENA_SIZE);
    }
    if (compress ? (enc_arenas == NULL) : (dec_arenas == NULL)) {
        fprintf(stderr, "zpse_cli: arena allocation failed\n");
        free_arenas(enc_arenas, threads);
        free_arenas(dec_arenas, threads);
        free(in);
        return 1;
    }
    out = (uint8_t *)malloc(out_cap + 1u);
    if (out == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        free_arenas(enc_arenas, threads);
        free_arenas(dec_arenas, threads);
        free(in);
        return 1;
    }

    if (compress) {
        rc = compress_mem(in, in_len, out, out_cap, &out_len, block, enc_flags,
                          indexed, enc_arenas, threads);
    } else if (stream_dec != 0) {
        /* decode_stream opens, writes and closes the output itself. */
        int drc = decode_stream(in, in_len, out_path, &dhead, dec_arenas[0]);

        free(out);
        free_arenas(dec_arenas, threads);
        free(in);
        return drc;
    } else {
        /* Frames written with -x decode block by block, so -t applies here
         * too; a frame without an index decodes sequentially inside the same
         * call, which is why the thread count is not an error there. */
        rc = zpse_decompress_frame_parallel(in, in_len, out, out_cap,
                                            &out_len, threads, dec_arenas,
                                            threads);
    }
    if (rc != 0) {
        fprintf(stderr, "zpse_cli: codec failed rc=%d\n", rc);
        free_arenas(enc_arenas, threads);
        free_arenas(dec_arenas, threads);
        free(out);
        free(in);
        return 1;
    }
    rc = write_file(out_path, out, out_len);
    free_arenas(enc_arenas, threads);
    free_arenas(dec_arenas, threads);
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
    long reps_arg = 0;
    long bytes_cap = 0;
    unsigned threads = 1u;
    int i = 2;
    int files;
    int f;
    int failures = 0;
    unsigned long long tot_in = 0u;
    unsigned long long tot_out = 0u;
    /* Bytes fed through each codec, counting every repetition: the
     * per-file rates are n*reps/time, so the totals must be too. */
    unsigned long long tot_passed = 0u;
    double tot_enc_ns = 0.0;
    double tot_dec_ns = 0.0;
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
            /* The fast LZ form and nothing else: one match-finder pass per
             * block, no shape decision. The stream is the one MAX offers as
             * its last candidate, so any decoder reads it. */
            enc_flags |= ZPSE_ENC_LZ_FAST;
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
        default:
            fprintf(stderr, "zpse_cli: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    /* The parallel decoder finds a block through the index, so a threaded
     * bench measures an indexed frame. It costs 8 bytes per block in the
     * ratio column, and the header line says which frame was measured. */
    if (threads > 1u) {
        indexed = 1;
    }
    files = argc - i;
    if (files < 1) {
        fprintf(stderr, "usage: zpse_cli bench [opts] FILE...\n");
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
    if (!quiet) {
        printf("level %s%s, block %u, %s, %u thread%s\n", level_name(enc_flags),
               (enc_flags & ZPSE_ENC_CHECKSUM) ? ", crc" : "",
               (unsigned)block, indexed ? "indexed" : "no index",
               threads, threads == 1u ? "" : "s");
        printf("%-14s %11s %11s %7s %10s %10s %8s\n", "file", "bytes",
               "packed", "ratio", "enc MB/s", "dec MB/s", "ms");
    }

    for (f = 0; f < files; f++) {
        const char *path = argv[i + f];
        uint8_t *in;
        uint8_t *comp;
        uint8_t *out;
        size_t n = 0u;
        size_t comp_cap;
        size_t comp_len = 0u;
        size_t out_len = 0u;
        long reps;
        long r;
        double t0;
        double enc_ns;
        double dec_ns;
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
        t0 = now_ns();
        for (r = 0; r < reps; r++) {
            if (compress_mem(in, n, comp, comp_cap, &comp_len, block,
                             enc_flags, indexed, enc_arenas, threads) != 0) {
                fprintf(stderr, "zpse_cli: compress failed on %s (n=%lu)\n",
                        path, (unsigned long)n);
                ok = 0;
                break;
            }
        }
        enc_ns = now_ns() - t0;
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
        t0 = now_ns();
        for (r = 0; r < reps; r++) {
            if (zpse_decompress_frame_parallel(comp, comp_len, out, n,
                                               &out_len, threads, dec_arenas,
                                               threads) != 0) {
                fprintf(stderr, "zpse_cli: decompress failed on %s\n", path);
                ok = 0;
                break;
            }
        }
        dec_ns = now_ns() - t0;
        if (ok) {
            /* Untimed verification pass: one decode, compared whole. */
            if (out_len != n || memcmp(out, in, n) != 0) {
                fprintf(stderr, "zpse_cli: %s: round trip MISMATCH\n", path);
                ok = 0;
            }
        }
        if (!quiet || !ok) {
            printf("%-14s %11lu %11lu %7.2f %10.1f %10.1f %8.1f%s\n", path,
                   (unsigned long)n, (unsigned long)comp_len,
                   comp_len ? (double)n / (double)comp_len : 0.0,
                   enc_ns > 0.0 ? (double)n * (double)reps / enc_ns * 1e3 : 0.0,
                   dec_ns > 0.0 ? (double)n * (double)reps / dec_ns * 1e3 : 0.0,
                   (enc_ns + dec_ns) / 1e6, ok ? "" : "  FAIL");
        }
        if (!ok) {
            failures++;
        }
        tot_in += (unsigned long long)n;
        tot_out += (unsigned long long)comp_len;
        tot_passed += (unsigned long long)n * (unsigned long long)reps;
        tot_enc_ns += enc_ns;
        tot_dec_ns += dec_ns;
        free(in);
        free(comp);
        free(out);
    }
    if (tot_in > 0u) {
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
    rc = zpse_frame_index_count(in, in_len, &count);
    if (rc != PSE_OK) {
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
        if (rc != PSE_OK) {
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
    if (rc != PSE_OK) {
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
