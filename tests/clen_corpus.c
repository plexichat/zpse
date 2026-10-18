/* Ratio guard: compress a fixed corpus of block shapes and print one line
 * per case with its compressed length. Two builds' outputs are diffed, so
 * any encoder decision that changes a size (or a byte) shows up as a diff
 * instead of hiding in an average.
 *
 * Usage: clen_corpus [--hex]
 *   default  "<case-index> <input-len> <clen>\n" per line
 *   --hex    additionally prints a CRC32-C of the compressed bytes, so a
 *            change that keeps the size but reorders the payload is caught
 *            too.
 *
 * Links src/zpse_unity.c only; exits 0 unless a case fails to round-trip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zpse/zpse.h"
#include "zpse/crc32c.h"

/* The corpus shapes and their generator, shared with tests/test_sizes.c. */
#include "corpus_gen.h"

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[ZPSE_MAX_BLOCK_SIZE];
static uint8_t comp[ZPSE_MAX_BLOCK_SIZE * 2];
static uint8_t out[ZPSE_MAX_BLOCK_SIZE];

int main(int argc, char **argv)
{
    const int nk = ZPSE_CORPUS_KINDS;
    const int ns = ZPSE_CORPUS_NSIZES;
    int hex = 0;
    int k, s, seed;
    long cases = 0;

    for (k = 1; k < argc; k++) {
        if (strcmp(argv[k], "--hex") == 0)
            hex = 1;
        else {
            fprintf(stderr, "usage: %s [--hex]\n", argv[0]);
            return 2;
        }
    }

    for (seed = 0; seed < ZPSE_CORPUS_SEEDS; seed++) {
        for (k = 0; k < nk; k++) {
            for (s = 0; s < ns; s++) {
                uint32_t n = corpus_sizes[s];
                uint32_t cl = 0, ol = 0;
                int rc;

                corpus_gen(k, n, corpus_seed(seed, k, n), in);
                rc = zpse_compress_block(in, n, comp, sizeof comp, &cl, 0, 1,
                                        E);
                if (rc != 0) {
                    printf("%ld kind=%d n=%u COMPRESS_FAIL rc=%d\n", cases, k,
                           n, rc);
                    cases++;
                    continue;
                }
                rc = zpse_decompress_block(comp, cl, out, sizeof out, &ol, D);
                if (rc != 0 || ol != n || memcmp(in, out, n) != 0) {
                    printf("%ld kind=%d n=%u ROUNDTRIP_FAIL rc=%d\n", cases, k,
                           n, rc);
                    cases++;
                    continue;
                }
                printf("%ld kind=%d n=%u clen=%u", cases, k, n, cl);
                if (hex)
                    printf(" crc=%08X", zpse_crc32c(comp, cl));
                printf("\n");
                cases++;
            }
        }
    }
    fprintf(stderr, "%ld cases\n", cases);
    return 0;
}
