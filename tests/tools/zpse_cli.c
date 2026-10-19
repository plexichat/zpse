/* Minimal file-to-file CLI over the public frame API, so the cross-tool
 * benchmark times zpse through the same interface as the other tools:
 * one process, one input file, one output file.
 *
 * Usage: zpse_cli c IN OUT     compress
 *        zpse_cli d IN OUT     decompress
 *
 * Exit 0 on success, 1 on an I/O or codec failure, 2 on a usage error.
 * The arenas are the documented sizes (384 KiB encode / 256 KiB decode),
 * 64-byte aligned, allocated once per run -- the same shape an embedder
 * uses, and the same shape the in-tree bench measures. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv)
{
    int compress;
    uint8_t *in;
    uint8_t *out;
    size_t in_len = 0u;
    size_t out_len = 0u;
    size_t out_cap;
    void *arena;
    int rc;

    if (argc != 4 || (argv[1][0] != 'c' && argv[1][0] != 'd') ||
        argv[1][1] != '\0') {
        fprintf(stderr, "usage: zpse_cli c|d IN OUT\n");
        return 2;
    }
    compress = (argv[1][0] == 'c');
    in = read_file(argv[2], &in_len);
    if (in == NULL) {
        return 1;
    }
    if (compress) {
        /* A compressed block never exceeds its input by more than its
         * headers: 40 bytes per 64 KiB block plus the 32-byte frame
         * header, so input plus one eighth and slack is a strict bound. */
        out_cap = in_len + in_len / 8u + 65536u;
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
            h.content_size == UINT64_MAX) {
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
        rc = zpse_compress_frame(in, in_len, out, out_cap, &out_len, 65536u,
                                 arena);
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
    rc = write_file(argv[3], out, out_len);
    ZPSE_CLI_FREE(arena);
    free(out);
    free(in);
    return rc;
}
