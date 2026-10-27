/* Command-line binding over the public frame API.
 *
 * Usage:
 *   zpse_cli c [opts] IN OUT     compress a file
 *   zpse_cli d IN OUT            decompress a frame file
 *   zpse_cli bench [opts] FILE...  compress and decompress each file IN
 *                                MEMORY (nothing written), verify the
 *                                round trip, report ratio and throughput
 *
 * Options (c and bench):
 *   -l LEVEL   balanced (default) | faster | fastest | max
 *   -c         emit the per-block payload CRC32-C (verified on decode)
 *   -b N       nominal block size, 1..65536 (default 65536)
 *   -x         append the FrameIndex (random access); bench also seeks
 *   -r N       bench repetitions per file (0 = pick from the file size)
 *   -q         bench: summary line only, no per-file rows
 *
 * Exit 0 on success, 1 on an I/O, codec or verification failure, 2 on a
 * usage error. The arenas are the documented sizes (448 KiB encode /
 * 320 KiB decode), 64-byte aligned, allocated once per run -- the same
 * shape an embedder uses.
 *
 * Throughputs are decimal (1 MB = 1e6 bytes) and measure the codec only:
 * a bench reads each input once, before the clock starts, and keeps every
 * buffer in memory. */
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
#include "zpse/frame.h"

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
                        uint32_t enc_flags, int indexed, void *arena)
{
    if (indexed) {
        return zpse_compress_frame_indexed(in, n, out, out_cap, out_len,
                                           block, enc_flags, arena);
    }
    return zpse_compress_frame_ex(in, n, out, out_cap, out_len, block,
                                  enc_flags, arena);
}

static int cmd_codec(int argc, char **argv)
{
    int compress = (argv[1][0] == 'c');
    uint32_t enc_flags = ZPSE_ENC_SPEED_BALANCED;
    uint32_t block = ZPSE_MAX_BLOCK_SIZE;
    int indexed = 0;
    const char *in_path;
    const char *out_path;
    uint8_t *in;
    uint8_t *out;
    size_t in_len = 0u;
    size_t out_len = 0u;
    size_t out_cap;
    void *arena;
    int rc;
    int i = 2;

    for (; i < argc && argv[i][0] == '-'; i++) {
        switch (argv[i][1]) {
        case 'l':
            if (++i >= argc || !parse_level(argv[i], &enc_flags)) {
                return 2;
            }
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
        default:
            fprintf(stderr, "zpse_cli: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    if (i >= argc || argc - i != 2) {
        fprintf(stderr, "usage: zpse_cli c [opts] IN OUT\n"
                        "       zpse_cli d IN OUT\n"
                        "       zpse_cli bench [opts] FILE...\n");
        return 2;
    }
    in_path = argv[i];
    out_path = argv[i + 1];
    in = read_file(in_path, &in_len);
    if (in == NULL) {
        return 1;
    }
    if (compress) {
        out_cap = frame_bound(in_len, block, indexed);
        arena = ZPSE_CLI_ALIGNED(64u, ZPSE_ENC_ARENA_SIZE);
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
        arena = ZPSE_CLI_ALIGNED(64u, ZPSE_DEC_ARENA_SIZE);
    }
    if (arena == NULL) {
        fprintf(stderr, "zpse_cli: arena allocation failed\n");
        free(in);
        return 1;
    }
    out = (uint8_t *)malloc(out_cap + 1u);
    if (out == NULL) {
        fprintf(stderr, "zpse_cli: out of memory\n");
        ZPSE_CLI_FREE(arena);
        free(in);
        return 1;
    }

    if (compress) {
        rc = compress_mem(in, in_len, out, out_cap, &out_len, block, enc_flags,
                          indexed, arena);
    } else {
        rc = zpse_decompress_frame(in, in_len, out, out_cap, &out_len, arena);
    }
    if (rc != 0) {
        fprintf(stderr, "zpse_cli: codec failed rc=%d\n", rc);
        ZPSE_CLI_FREE(arena);
        free(out);
        free(in);
        return 1;
    }
    rc = write_file(out_path, out, out_len);
    ZPSE_CLI_FREE(arena);
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
    uint32_t enc_flags = ZPSE_ENC_SPEED_BALANCED;
    uint32_t block = ZPSE_MAX_BLOCK_SIZE;
    int indexed = 0;
    int quiet = 0;
    long reps_arg = 0;
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
    void *enc_arena;
    void *dec_arena;

    for (; i < argc && argv[i][0] == '-'; i++) {
        switch (argv[i][1]) {
        case 'l':
            if (++i >= argc || !parse_level(argv[i], &enc_flags)) {
                return 2;
            }
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
    files = argc - i;
    if (files < 1) {
        fprintf(stderr, "usage: zpse_cli bench [opts] FILE...\n");
        return 2;
    }
    enc_arena = ZPSE_CLI_ALIGNED(64u, ZPSE_ENC_ARENA_SIZE);
    dec_arena = ZPSE_CLI_ALIGNED(64u, ZPSE_DEC_ARENA_SIZE);
    if (enc_arena == NULL || dec_arena == NULL) {
        fprintf(stderr, "zpse_cli: arena allocation failed\n");
        ZPSE_CLI_FREE(enc_arena);
        ZPSE_CLI_FREE(dec_arena);
        return 1;
    }
    if (!quiet) {
        printf("level %s%s%s, block %u, %s\n", level_name(enc_flags),
               (enc_flags & ZPSE_ENC_CHECKSUM) ? ", crc" : "", "",
               (unsigned)block, indexed ? "indexed" : "no index");
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
        comp_cap = frame_bound(n, block, indexed);
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
                             enc_flags, indexed, enc_arena) != 0) {
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
        if (indexed && !check_seek(comp, comp_len, in, n, block, dec_arena)) {
            free(in);
            free(comp);
            free(out);
            failures++;
            continue;
        }
        t0 = now_ns();
        for (r = 0; r < reps; r++) {
            if (zpse_decompress_frame(comp, comp_len, out, n, &out_len,
                                      dec_arena) != 0) {
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
    ZPSE_CLI_FREE(enc_arena);
    ZPSE_CLI_FREE(dec_arena);
    if (failures) {
        fprintf(stderr, "zpse_cli: %d file(s) failed\n", failures);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: zpse_cli c [opts] IN OUT\n"
                        "       zpse_cli d IN OUT\n"
                        "       zpse_cli bench [opts] FILE...\n");
        return 2;
    }
    if (strcmp(argv[1], "bench") == 0) {
        return cmd_bench(argc, argv);
    }
    if (argc >= 2 && argv[1][1] == '\0' &&
        (argv[1][0] == 'c' || argv[1][0] == 'd')) {
        return cmd_codec(argc, argv);
    }
    fprintf(stderr, "zpse_cli: unknown command '%s'\n", argv[1]);
    return 2;
}
