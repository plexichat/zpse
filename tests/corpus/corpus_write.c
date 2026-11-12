/* Write the benchmark corpora to files, for the cross-tool comparison.
 *
 * Usage: corpus_write DIR [SIZE] [--mixed | --probe]
 *
 * --mixed writes only `mixed.bin`, the generated workload the README's mixed
 * tables are measured on (see mix_fill); at the default SIZE it is 4 MiB, and
 * the page tables use 160 MiB.
 *
 * --probe writes only `probe.bin`, one 4 KiB page per shape (see probe_fill):
 * the page shapes the price's knobs are supposed to separate, which the mixed
 * workload does not contain.
 *
 * Two families of corpus live here.
 *
 * The byte patterns are the ones in tests/bench.c, applied to the
 * *absolute* file position rather than to a 64 KiB block: every 64 KiB
 * block then holds exactly the bytes bench.c measures, while the file
 * as a whole has no block-to-block repetition (a repeated block would
 * let a long-range matcher like zstd or brotli compress the corpus in a
 * way no real input would allow, which is not what this comparison is
 * for).
 *
 * The columnar corpora are the seven categories in columnar_gen.h, and
 * they are written as one continuous column: a real column file is a
 * single long column, not a repeated block, so the file crosses block
 * boundaries the way a real one would.
 *
 * SIZE defaults to 4 MiB and must be a multiple of 64 KiB. Exit 0 on
 * success.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "columnar_gen.h"

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

/* Write one columnar corpus, as one continuous column. The generator
 * carries its own state, so the chunks of a file concatenate into the
 * same stream the in-process harness produces in a single fill. */
static int write_column(FILE *f, int kind, size_t size)
{
    ZpseColGen g;
    uint8_t *chunk = (uint8_t *)malloc(BLOCK);
    size_t base = 0;

    if (chunk == NULL) {
        fprintf(stderr, "oom\n");
        return 1;
    }
    zpse_col_gen_init(&g, kind);
    while (base < size) {
        size_t n = size - base;
        if (n > BLOCK) n = BLOCK;
        zpse_col_gen_fill(&g, chunk, n);
        if (fwrite(chunk, 1u, n, f) != n) {
            fprintf(stderr, "short write on column %d\n", kind);
            free(chunk);
            return 1;
        }
        base += n;
    }
    free(chunk);
    return 0;
}

/* A mixed workload: text, source, executable-like bytes, library-like blobs,
 * records and opaque data, one kind per 64 KiB chunk and the kind indexed by
 * the chunk number. It stands in for the shape a browser's or a build tree's
 * footprint has -- programs, shared libraries, documentation and build
 * output -- but it is generated rather than scraped from the host, so the
 * table it feeds can be re-measured from this tree alone. Every chunk is a
 * pure function of its index and the chunk's bytes never depend on how the
 * file was handed to the writer, so the same SIZE always writes the same
 * bytes on any host. */
static const char *const mix_words[] = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "with", "as", "was", "on", "be", "by", "this", "are", "from", "or",
    "value", "buffer", "length", "offset", "header", "table", "block",
    "page", "match", "literal", "distance", "window", "stream", "code",
    "read", "write", "return", "static", "const", "uint32_t", "if", "while"
};

static uint32_t mix_rnd(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static void mix_fill(uint8_t *o, size_t n, uint32_t chunk)
{
    uint32_t s = 0x5EEDu ^ chunk * 2654435761u;
    size_t i = 0;

    switch (chunk % 6u) {
    case 0: /* prose: sentences of common words, broken into lines */
        while (i < n) {
            uint32_t col = 0u;
            uint32_t words = 12u + mix_rnd(&s) % 40u;
            uint32_t w;

            for (w = 0; w < words && i < n; w++) {
                const char *word = mix_words[mix_rnd(&s) %
                    (sizeof mix_words / sizeof mix_words[0])];
                size_t len = strlen(word);

                if (i + len + 1u >= n || col + len + 1u > 72u) break;
                memcpy(o + i, word, len);
                i += len;
                o[i++] = ' ';
                col += (uint32_t)len + 1u;
            }
            if (i < n) o[i++] = '\n';
        }
        break;
    case 1: /* source: indented lines over a small identifier set */
        while (i < n) {
            uint32_t ind = (mix_rnd(&s) % 5u) * 4u;
            uint32_t w = 2u + mix_rnd(&s) % 6u;
            uint32_t k;

            for (k = 0; k < ind && i < n; k++) o[i++] = ' ';
            for (k = 0; k < w && i < n; k++) {
                const char *word = mix_words[mix_rnd(&s) %
                    (sizeof mix_words / sizeof mix_words[0])];
                size_t len = strlen(word);
                size_t take = len;

                if (i + take > n) take = n - i;
                memcpy(o + i, word, take);
                i += take;
                if (i < n) o[i++] = (k + 1u == w) ? ';' : ' ';
            }
            if (i < n) o[i++] = '\n';
        }
        break;
    case 2: /* executable-like: opcode soup with the prologues repeated */
        for (i = 0; i < n; i++) {
            if ((i % 23u) == 0u && i + 16u < n) {
                static const uint8_t pro[16] = {
                    0x55u, 0x48u, 0x89u, 0xE5u, 0x41u, 0x57u, 0x41u, 0x56u,
                    0x53u, 0x48u, 0x83u, 0xECu, 0x08u, 0xE8u, 0x00u, 0x00u
                };
                memcpy(o + i, pro, 16u);
                i += 15u;
            } else {
                o[i] = (uint8_t)(mix_rnd(&s) >> 24);
            }
        }
        break;
    case 3: /* library-like: dense bytes with a string table every 96 */
        for (i = 0; i < n; i++) {
            if ((i % 96u) == 0u && i + 12u + 8u < n) {
                static const char *const sym[4] = {
                    "GLIBC_2.34", "libc.so.6", "_ITM_deregister", "malloc"
                };
                const char *name = sym[(i / 96u) % 4u];
                size_t len = strlen(name);

                memset(o + i, 0, 4u);
                memcpy(o + i + 4u, name, len);
                i += len + 4u;
            } else {
                o[i] = (uint8_t)(mix_rnd(&s) >> 24);
            }
        }
        break;
    case 4: /* records: id, small int, ramp, string offset */
        for (i = 0; i + 16u <= n; i += 16u) {
            uint32_t id = chunk * (uint32_t)(BLOCK / 16u) +
                          (uint32_t)(i / 16u);
            uint32_t v = 1u + mix_rnd(&s) % 9u;
            float f = (float)(id % 997u) / 8.0f;

            o[i] = (uint8_t)id;
            o[i + 1u] = (uint8_t)(id >> 8);
            o[i + 2u] = (uint8_t)(id >> 16);
            o[i + 3u] = 0u;
            o[i + 4u] = (uint8_t)v;
            o[i + 5u] = (uint8_t)(v >> 8);
            o[i + 6u] = 0u;
            o[i + 7u] = 0u;
            memcpy(o + i + 8u, &f, 4u);
            o[i + 12u] = (uint8_t)(id * 3u);
            o[i + 13u] = (uint8_t)(id >> 5);
            o[i + 14u] = 0u;
            o[i + 15u] = 0u;
        }
        break;
    default: /* opaque: embedded compressed resources */
        for (i = 0; i < n; i++) o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        break;
    }
}

/* A page workload aimed at the knobs the mixed corpus leaves inert: one
 * 4 KiB page per shape, the shape indexed by the page number, so a page run
 * meets every shape once per eight pages. The shapes are the ones the noise
 * probe, the flat trial cap and the single-plane tile count price -- a page
 * whose middle is noise inside an otherwise compressible block, a flat 4-bit
 * page, a record whose stride the identity cannot see, interleaved planes, a
 * text end against a noise end, sparse bytes with holes, a ramp with one
 * noisy tile per kilobyte, four quarters of different shapes, a tail-only
 * compressible page, stride-5 records, a 2-bit alphabet, alternating
 * compressible and noisy blocks, pointer words into zeros, and text with
 * ragged lines. Sixteen shapes, so a run meets each twice per 64 KiB. */
static void probe_fill(uint8_t *o, size_t n, uint32_t page)
{
    uint32_t s = 0x9E37u ^ page * 2654435761u;
    size_t i;
    static const uint8_t pro[16] = {
        0x55u, 0x48u, 0x89u, 0xE5u, 0x41u, 0x57u, 0x41u, 0x56u,
        0x53u, 0x48u, 0x83u, 0xECu, 0x08u, 0xE8u, 0x00u, 0x00u
    };
    static const char line[] =
        "the page encoder prices every plane of a page before it trials one "
        "of them, which is the point of the price; ";

    switch (page % 16u) {
    case 0: /* compressible ends, a noisy middle third */
        memset(o, 0, n);
        for (i = n / 3u; i < (2u * n) / 3u; i++)
            o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        break;
    case 1: /* noise, with the prologue repeated at the tail */
        for (i = 0; i < n; i++) o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        for (i = (n * 3u) / 4u; i + 16u <= n; i += 16u)
            memcpy(o + i, pro, 16u);
        break;
    case 2: /* flat: 4-bit noise */
        for (i = 0; i < n; i++)
            o[i] = (uint8_t)((mix_rnd(&s) >> 24) & 0x0Fu);
        break;
    case 3: /* records: every fourth byte random, the rest a ramp */
        for (i = 0; i + 4u <= n; i += 4u) {
            o[i] = (uint8_t)(mix_rnd(&s) >> 24);
            o[i + 1u] = (uint8_t)(page + i);
            o[i + 2u] = (uint8_t)((page + i) >> 8);
            o[i + 3u] = (uint8_t)((page + i) * 5u);
        }
        break;
    case 4: /* interleaved planes: sixteen ramped floats, then zeros */
        for (i = 0; i + 128u <= n; i += 128u) {
            uint32_t k;

            for (k = 0; k < 16u; k++) {
                float f = (float)((page * 16u + (i / 128u)) % 997u) / 8.0f +
                          (float)k;
                memcpy(o + i + k * 4u, &f, 4u);
            }
            memset(o + i + 64u, 0, 64u);
        }
        break;
    case 5: /* text at one end, noise at the other */
        for (i = 0; i < n / 2u; i++)
            o[i] = (uint8_t)line[i % (sizeof line - 1u)];
        for (; i < n; i++) o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        break;
    case 6: /* sparse with single-byte holes */
        memset(o, 0, n);
        for (i = 0; i < n; i += 251u) o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        break;
    case 7: /* a ramp with one noisy 256-byte tile per kilobyte */
        for (i = 0; i < n; i++) o[i] = (uint8_t)((page + i) * 7u);
        for (i = 0; i + 256u <= n; i += 1024u) {
            size_t k;

            for (k = 0; k < 256u; k++)
                o[i + k] = (uint8_t)(mix_rnd(&s) >> 24);
        }
        break;
    case 8: { /* four quarters: zeros, ramp, noise, records */
        size_t q = n / 4u;

        for (i = 0; i < n; i++) {
            size_t k = (q == 0u) ? 0u : i / q;

            if (k == 0u) o[i] = 0u;
            else if (k == 1u) o[i] = (uint8_t)(page + i);
            else if (k == 2u) o[i] = (uint8_t)(mix_rnd(&s) >> 24);
            else o[i] = (i % 16u == 0u) ? (uint8_t)(mix_rnd(&s) >> 24)
                                        : (uint8_t)(i * 3u);
        }
        break;
    }
    case 9: /* noise everywhere, one compressible tile at the tail */
        for (i = 0; i < n; i++) o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        for (i = (n > 512u) ? n - 512u : 0u; i < n; i++)
            o[i] = (uint8_t)(i & 0xFFu);
        break;
    case 10: /* stride-5 records */
        for (i = 0; i + 5u <= n; i += 5u) {
            o[i] = (uint8_t)(mix_rnd(&s) >> 24);
            o[i + 1u] = (uint8_t)(page + i);
            o[i + 2u] = (uint8_t)((page + i) >> 8);
            o[i + 3u] = (uint8_t)((page + i) * 3u);
            o[i + 4u] = (uint8_t)((page + i) * 7u);
        }
        break;
    case 11: /* a 2-bit alphabet */
        for (i = 0; i < n; i++)
            o[i] = (uint8_t)((mix_rnd(&s) >> 24) & 0x03u);
        break;
    case 12: /* alternating compressible and noisy 512-byte blocks */
        for (i = 0; i < n; i++)
            o[i] = ((i / 512u) & 1u) ? (uint8_t)(mix_rnd(&s) >> 24)
                                     : 0u;
        break;
    case 13: /* pointer-like words, then zeros */
        for (i = 0; i + 8u <= n; i += 8u) {
            uint64_t v = (uint64_t)(page * 4096u + i) * 16u + 0x5555u;

            if (i >= n / 2u) v = 0u;
            memcpy(o + i, &v, 8u);
        }
        break;
    case 14: { /* text with ragged line lengths */
        size_t col = 0u;

        for (i = 0; i < n; i++) {
            if (col > 40u + mix_rnd(&s) % 32u) {
                o[i] = '\n';
                col = 0u;
                continue;
            }
            o[i] = (uint8_t)line[mix_rnd(&s) % (sizeof line - 1u)];
            col++;
        }
        break;
    }
    default: /* a ramp with one random hole in thirty-two */
        for (i = 0; i < n; i++)
            o[i] = ((mix_rnd(&s) & 31u) == 0u)
                       ? (uint8_t)(mix_rnd(&s) >> 24)
                       : (uint8_t)((page + i) * 7u);
        break;
    }
}

int main(int argc, char **argv)
{
    static uint8_t block[BLOCK];
    unsigned long size = 4u * 1024u * 1024u;
    char path[1024];
    int i, mixed = 0, probe = 0, ai;

    if (argc < 2) {
        fprintf(stderr, "usage: corpus_write DIR [SIZE] [--mixed | --probe]\n");
        return 2;
    }
    for (ai = 2; ai < argc; ai++) {
        if (strcmp(argv[ai], "--mixed") == 0) {
            mixed = 1;
        } else if (strcmp(argv[ai], "--probe") == 0) {
            probe = 1;
        } else {
            size = strtoul(argv[ai], NULL, 10);
            if (size == 0u || size % BLOCK != 0u) {
                fprintf(stderr, "SIZE must be a non-zero multiple of %u\n",
                        BLOCK);
                return 2;
            }
        }
    }
    if (mixed && probe) {
        fprintf(stderr, "--mixed and --probe are separate corpora\n");
        return 2;
    }
    if (mixed || probe) {
        uint64_t base = 0u;
        int n = snprintf(path, sizeof path, "%s/%s.bin", argv[1],
                         mixed ? "mixed" : "probe");
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
            if (probe) {
                size_t off;

                for (off = 0; off < chunk; off += 4096u) {
                    size_t pn = chunk - off;
                    uint32_t page = (uint32_t)((base + off) / 4096u);

                    if (pn > 4096u) pn = 4096u;
                    probe_fill(block + off, pn, page);
                }
            } else {
                mix_fill(block, chunk, (uint32_t)(base / BLOCK));
            }
            if (fwrite(block, 1u, chunk, f) != chunk) {
                fprintf(stderr, "short write on %s\n", path);
                fclose(f);
                return 1;
            }
            base += chunk;
        }
        return fclose(f) != 0;
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
    /* The columnar corpora, under the same names the in-process harness
     * uses, so the two tables are about the same bytes. */
    for (i = 0; i < (int)ZPSE_COL_KINDS; i++) {
        int n = snprintf(path, sizeof path, "%s/%s.bin", argv[1],
                         zpse_col_names[i]);
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
        if (write_column(f, i, (size_t)size) != 0) {
            fclose(f);
            return 1;
        }
        if (fclose(f) != 0) {
            fprintf(stderr, "close failed on %s\n", path);
            return 1;
        }
    }
    return 0;
}
