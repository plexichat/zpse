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

static unsigned char E[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char D[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[ZPSE_MAX_BLOCK_SIZE];
static uint8_t comp[ZPSE_MAX_BLOCK_SIZE * 2];
static uint8_t out[ZPSE_MAX_BLOCK_SIZE];

static uint32_t rng_state;

static uint32_t rnd(void)
{
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return rng_state >> 8;
}

/* Each generator fills in[0..n) from `kind` and the (seeded) rng. */
static void gen(int kind, uint32_t n, uint32_t seed)
{
    uint32_t i;

    rng_state = seed;
    switch (kind) {
    case 0: /* zeros */
        memset(in, 0, n);
        break;
    case 1: /* constant */
        memset(in, (int)(seed & 0xFFu), n);
        break;
    case 2: /* increment */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)i;
        break;
    case 3: /* random */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)(rnd() >> 24);
        break;
    case 4: /* 4-bit random (the lowentropy shape) */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((rnd() >> 24) & 0x0Fu);
        break;
    case 5: /* sparse spikes */
        memset(in, 0, n);
        for (i = 0; i < n; i += 97)
            in[i] = (uint8_t)i;
        break;
    case 6: /* stride 4 */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((i / 4u + (i % 4u) * 64u) & 0xFFu);
        break;
    case 7: /* alternation */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((i & 1u) ? 0xFF : 0x00);
        break;
    case 8: /* sawtooth */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)(i * 7u + (i >> 4));
        break;
    case 9: /* text-ish: printable, repetitive words */
        for (i = 0; i < n; i++) {
            static const char *w = "the quick brown fox jumps over ";
            in[i] = (uint8_t)w[(i * 7u + (i >> 5)) % 30u];
        }
        break;
    case 10: /* two halves: noise then constant */
        for (i = 0; i < n; i++)
            in[i] = (i < n / 2u) ? (uint8_t)(rnd() >> 24) : 0x5A;
        break;
    case 11: /* two halves: constant then noise */
        for (i = 0; i < n; i++)
            in[i] = (i < n / 2u) ? 0x5A : (uint8_t)(rnd() >> 24);
        break;
    case 12: /* small alphabet, 3 values, biased */
        for (i = 0; i < n; i++) {
            uint32_t r = rnd() >> 24;
            in[i] = (uint8_t)(r < 200u ? 7u : (r < 250u ? 9u : 3u));
        }
        break;
    case 13: /* 16-value repeating palette, shuffled */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((rnd() >> 24) & 0x0Fu);
        if (n > 64u)
            for (i = 0; i + 1u < n; i += 2)
                in[i] = in[i + 1u];
        break;
    case 14: /* structured 2-byte records */
        for (i = 0; i < n; i += 2) {
            uint32_t v = (uint32_t)(i * 3u);
            in[i] = (uint8_t)v;
            if (i + 1u < n)
                in[i + 1u] = (uint8_t)(v >> 8);
        }
        break;
    default: /* 15: mixing of every fourth byte random */
        for (i = 0; i < n; i++)
            in[i] = (i % 4u == 0) ? (uint8_t)(rnd() >> 24)
                                  : (uint8_t)(i & 0x0Fu);
        break;
    }
}

int main(int argc, char **argv)
{
    static const uint32_t sizes[] = {
        1024u, 2048u, 3072u, 4096u, 5000u, 6144u, 8192u, 10000u,
        12288u, 16384u, 20000u, 24576u, 32768u, 40000u, 49152u, 65536u
    };
    const int nk = 16;
    const int ns = (int)(sizeof sizes / sizeof sizes[0]);
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

    for (seed = 1; seed <= 3; seed++) {
        for (k = 0; k < nk; k++) {
            for (s = 0; s < ns; s++) {
                uint32_t n = sizes[s];
                uint32_t cl = 0, ol = 0;
                int rc;

                gen(k, n, (uint32_t)(seed * 7919 + k * 104729 + (int)n));
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
