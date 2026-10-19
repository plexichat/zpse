/* Write the ten bench patterns to files, for the cross-tool comparison.
 *
 * Usage: corpus_write DIR [SIZE]
 *
 * The generators are the ones in tests/bench.c, applied to the *absolute*
 * file position rather than to a 64 KiB block: every 64 KiB block then
 * holds exactly the bytes bench.c measures, while the file as a whole has
 * no block-to-block repetition (a repeated block would let a long-range
 * matcher like zstd or brotli compress the corpus in a way no real input
 * would allow, which is not what this comparison is for).
 *
 * SIZE defaults to 4 MiB and must be a multiple of 64 KiB. Exit 0 on
 * success.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK 65536u

static const char *const names[10] = {
    "zeros", "incr", "random", "sparse", "strided",
    "const5A", "alt00FF", "sawtooth", "lowentropy", "sparse2"
};

/* Fill out[0..n) for pattern idx, starting at absolute position base.
 * `rng` carries the PRNG state across calls; patterns that do not use it
 * leave it alone. */
static void make_pattern(uint8_t *out, size_t n, int idx, uint64_t base,
                         uint32_t *rng)
{
    size_t i;

    switch (idx) {
    case 0:
        memset(out, 0, n);
        break;
    case 1:
        for (i = 0; i < n; i++) out[i] = (uint8_t)((base + i) & 0xFFu);
        break;
    case 2:
        for (i = 0; i < n; i++) {
            *rng = *rng * 1664525u + 1013904223u;
            out[i] = (uint8_t)(*rng >> 24);
        }
        break;
    case 3:
        memset(out, 0, n);
        for (i = 0; i < n; i++) {
            if (((base + i) % 97u) == 0u) out[i] = (uint8_t)(base + i);
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            uint64_t p = base + i;
            out[i] = (uint8_t)((p / 4u + (p % 4u) * 64u) & 0xFFu);
        }
        break;
    case 5:
        memset(out, 0x5A, n);
        break;
    case 6:
        for (i = 0; i < n; i++)
            out[i] = (uint8_t)(((base + i) & 1u) ? 0xFFu : 0x00u);
        break;
    case 7:
        for (i = 0; i < n; i++) {
            uint64_t p = base + i;
            out[i] = (uint8_t)(p * 7u + (p >> 4));
        }
        break;
    case 8:
        for (i = 0; i < n; i++) {
            *rng = *rng * 1664525u + 1013904223u;
            out[i] = (uint8_t)((*rng >> 24) & 0x0Fu);
        }
        break;
    default:
        memset(out, 0, n);
        for (i = 0; i < n; i++) {
            uint64_t p = base + i;
            if ((p % 256u) == 0u) {
                out[i] = (uint8_t)p;
                if (i + 1u < n) out[i + 1u] = (uint8_t)(p >> 8);
            }
        }
        break;
    }
}

int main(int argc, char **argv)
{
    static uint8_t block[BLOCK];
    unsigned long size = 4u * 1024u * 1024u;
    char path[1024];
    int i;

    if (argc < 2) {
        fprintf(stderr, "usage: corpus_write DIR [SIZE]\n");
        return 2;
    }
    if (argc > 2) {
        size = strtoul(argv[2], NULL, 10);
        if (size == 0u || size % BLOCK != 0u) {
            fprintf(stderr, "SIZE must be a non-zero multiple of %u\n", BLOCK);
            return 2;
        }
    }
    for (i = 0; i < 10; i++) {
        uint64_t base = 0u;
        uint32_t rng = (i == 8) ? 0xBEEFu : 0x1234u;
        int n = snprintf(path, sizeof path, "%s/%s.bin", argv[1], names[i]);
        FILE *f;

        if (n <= 0 || (size_t)n >= sizeof path) {
            fprintf(stderr, "path too long\n");
            return 2;
        }
        f = fopen(path, "wb");
        if (f == NULL) {
            fprintf(stderr, "cannot write %s\n", path);
            return 1;
        }
        while (base < (uint64_t)size) {
            size_t chunk = (size_t)((uint64_t)size - base);

            if (chunk > BLOCK) chunk = BLOCK;
            make_pattern(block, chunk, i, base, &rng);
            if (fwrite(block, 1u, chunk, f) != chunk) {
                fprintf(stderr, "short write on %s\n", path);
                fclose(f);
                return 1;
            }
            base += chunk;
        }
        if (fclose(f) != 0) {
            fprintf(stderr, "close failed on %s\n", path);
            return 1;
        }
    }
    return 0;
}
