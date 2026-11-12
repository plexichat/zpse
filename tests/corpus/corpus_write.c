/* Write the benchmark corpora to files, for the cross-tool comparison.
 *
 * Usage: corpus_write DIR [SIZE] [--mixed | --probe [--probe-shape=N]
 *                           | --pages]
 *
 * --mixed writes only `mixed.bin`, the generated workload the README's mixed
 * tables are measured on (see mix_fill); at the default SIZE it is 4 MiB, and
 * the page tables use 160 MiB.
 *
 * --pages writes the 4 KiB page corpus (see page_fill): one file per page
 * class (`page-<name>.bin`), `pages-nonzero.bin` interleaving every class but
 * zero, `pages-all.bin` interleaving all of them with one zero page in eight,
 * and `pages.txt`, a manifest naming each class and its zero-page count. The
 * zero class is kept out of the headline file because a zram workload's
 * untouched anonymous pages compress to nothing and would flatter any average
 * that included them.
 *
 * --probe writes only `probe.bin`, one 4 KiB page per shape (see probe_fill):
 * the page shapes the price's knobs are supposed to separate, which the mixed
 * workload does not contain.
 *
 * --probe-shape=N writes `probe-shapeN.bin` instead: every page the same
 * shape, which is how one shape is audited against the knob arms on its own
 * (a shape no arm moves is dead weight in probe.bin).
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
 * meets every shape once per twenty pages. The shapes are the ones the noise
 * probe, the flat trial cap, the candidate cap, the gate's sample and the
 * single-plane tile count price -- a page whose middle is noise inside an
 * otherwise compressible block, counters that restart every 32 bytes, a
 * stride-4 record, interleaved planes, ragged text in two quarters against a
 * 2-bit alphabet in the other two, sparse bytes with holes, a ramp with one
 * noisy tile per kilobyte, four quarters of different shapes, a tail-only
 * compressible page, records whose stride alternates four and eight bytes, a
 * 2-bit alphabet behind a text head, alternating compressible and noisy
 * blocks, pointer words into zeros, ragged text with a noisy tile at a
 * quarter, and four pages whose quarters alternate two shapes. Twenty shapes,
 * so a 64 MiB run meets each about eight hundred times.
 *
 * Every shape has to be one a page arm can flip. A shape every arm prices the
 * same is dead weight here, however realistic it looks: the five homogeneous
 * pages this file started with (a flat 4-bit page, a 2-bit alphabet, stride-5
 * records, text against a noise half, ragged text) were byte-identical under
 * all thirteen arms at level 6, and were replaced by the ambiguous pages
 * above. */
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

    switch (page % 20u) {
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
    case 2: /* 4-byte counters that restart every 32 bytes, so the delta
             * form fits inside a run and not across the restarts */
        memset(o, 0, n);
        for (i = 0; i + 4u <= n; i += 4u) {
            uint32_t v = ((i % 32u) == 0u) ? (mix_rnd(&s) >> 8)
                                           : (uint32_t)((i / 4u) & 7u);

            memcpy(o + i, &v, 4u);
        }
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
    case 5: { /* ragged text in two quarters, a 2-bit alphabet in the other
               * two: a page-level choice no single tile shape explains */
        size_t q = n / 4u;
        size_t col = 0u;

        for (i = 0; i < n; i++) {
            if (((i / (q == 0u ? 1u : q)) & 1u) == 0u) {
                if (col > 40u + mix_rnd(&s) % 32u) {
                    o[i] = '\n';
                    col = 0u;
                    continue;
                }
                o[i] = (uint8_t)line[mix_rnd(&s) % (sizeof line - 1u)];
                col++;
            } else {
                o[i] = (uint8_t)((mix_rnd(&s) >> 24) & 0x03u);
            }
        }
        break;
    }
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
    case 10: /* records whose stride changes between four and eight bytes
              * every 64 bytes, so no single width transposes them */
        for (i = 0; i + 8u <= n; i += 8u) {
            if (((i / 64u) & 1u) == 0u) {
                o[i] = (uint8_t)(page + i);
                o[i + 1u] = (uint8_t)((page + i) >> 8);
                o[i + 2u] = (uint8_t)(i * 3u);
                o[i + 3u] = 0u;
                memcpy(o + i + 4u, o + i, 4u);
            } else {
                size_t k;

                for (k = 0; k < 8u; k++)
                    o[i + k] = (k == 7u)
                                  ? 0u
                                  : (uint8_t)((page + i) * (k + 3u));
            }
        }
        break;
    case 11: /* a 2-bit alphabet with a kilobyte of text at the head */
        for (i = 0; i < n; i++)
            o[i] = (uint8_t)((mix_rnd(&s) >> 24) & 0x03u);
        for (i = 0; i < n / 4u; i++)
            o[i] = (uint8_t)line[i % (sizeof line - 1u)];
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
    case 14: { /* ragged text with a noisy 256-byte tile per kilobyte */
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
        for (i = n / 4u; i < n / 4u + 256u && i < n; i++)
            o[i] = (uint8_t)(mix_rnd(&s) >> 24);
        break;
    }
    case 16: /* quarters alternate zeros and noise: a two-tile sample sees
              * only the zeros (tiles 0 and 8) */
        for (i = 0; i < n; i++)
            o[i] = ((i / (n / 4u)) & 1u) ? (uint8_t)(mix_rnd(&s) >> 24) : 0u;
        break;
    case 17: /* the same, noise first: a two-tile sample sees only the noise */
        for (i = 0; i < n; i++)
            o[i] = ((i / (n / 4u)) & 1u) ? 0u : (uint8_t)(mix_rnd(&s) >> 24);
        break;
    case 18: /* quarters alternate a 2-bit alphabet and 16-byte records */
        for (i = 0; i < n; i++) {
            if (((i / (n / 4u)) & 1u) == 0u)
                o[i] = (uint8_t)((mix_rnd(&s) >> 24) & 0x03u);
            else
                o[i] = (i % 16u == 0u) ? (uint8_t)(mix_rnd(&s) >> 24)
                                       : (uint8_t)(i * 3u);
        }
        break;
    case 19: /* the same, records first */
        for (i = 0; i < n; i++) {
            if (((i / (n / 4u)) & 1u) == 0u)
                o[i] = (i % 16u == 0u) ? (uint8_t)(mix_rnd(&s) >> 24)
                                       : (uint8_t)(i * 3u);
            else
                o[i] = (uint8_t)((mix_rnd(&s) >> 24) & 0x03u);
        }
        break;
    default: /* a ramp with one random hole in thirty-two */
        for (i = 0; i < n; i++)
            o[i] = ((mix_rnd(&s) & 31u) == 0u)
                       ? (uint8_t)(mix_rnd(&s) >> 24)
                       : (uint8_t)((page + i) * 7u);
        break;
    }
}

/* ---- 4 KiB page corpus ------------------------------------------------
 *
 * The page corpus models what a zram device is actually handed: one 4 KiB
 * page at a time from a running process, not a file cut into pages. Each
 * class is a page shape that occurs often enough to matter in a real memory
 * footprint, and every byte is a pure function of (class, page number), so
 * the same SIZE writes the same corpus on any host.
 *
 * The zero class is separate on purpose. Untouched anonymous pages are a
 * large share of a real footprint and compress to a handful of bytes; mixing
 * them into a headline ratio measures the ratio of a process's *allocated*
 * pages, not of the pages it has content in. `pages.txt` records the count so
 * a reader can see both numbers.
 */
#define PAGE 4096u

enum {
    PZERO = 0, PCODE, PRODATA, PHEAPP, PHEAPO, PSLAB, PBTREE, PHASH,
    PJSON, PSTACK, PFLOAT, PBLOB, PMIXED, PKINDS
};

static const char *const page_names[PKINDS] = {
    "zero", "code", "rodata", "heap-pointers", "heap-objects", "slab-strings",
    "btree", "hash-table", "json-arena", "stack", "float-series", "blob",
    "mixed-heap"
};

static void put16(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void put32(uint8_t *p, uint32_t v)
{
    put16(p, v);
    put16(p + 2, v >> 16);
}

static void put64(uint8_t *p, uint64_t v)
{
    put32(p, (uint32_t)v);
    put32(p + 4, (uint32_t)(v >> 32));
}

static void page_code(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    static const uint8_t pro[16] = {
        0xF3, 0x0F, 0x1E, 0xFA, 0x55, 0x48, 0x89, 0xE5,
        0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x83, 0xEC
    };
    size_t i = 0;

    while (i + 16u <= n) {
        uint8_t b[16];
        uint32_t r = mix_rnd(s);
        uint32_t a = page * PAGE + (uint32_t)i;

        memset(b, 0x90, sizeof b);
        switch (r % 10u) {
        case 0:
            memcpy(b, pro, sizeof b);
            b[15] = (uint8_t)(8u + (r >> 8) % 120u);
            break;
        case 1: /* mov rax, [rip + disp32] */
            b[0] = 0x48; b[1] = 0x8B; b[2] = 0x05;
            put32(b + 3, a + (r >> 8));
            b[7] = 0x48; b[8] = 0x85; b[9] = 0xC0;
            break;
        case 2: /* call rel32 */
            b[0] = 0xE8;
            put32(b + 1, (r >> 8) | 0x1000u);
            break;
        case 3: /* lea rdi, [rip + disp32] */
            b[0] = 0x48; b[1] = 0x8D; b[2] = 0x3D;
            put32(b + 3, a + 0x40u + (r >> 12));
            break;
        case 4: /* mov [rsp + 8], rsi */
            b[0] = 0x48; b[1] = 0x89; b[2] = 0x74; b[3] = 0x24;
            b[4] = 0x08;
            break;
        case 5: /* cmp eax, imm32; jne rel8 */
            b[0] = 0x3D;
            put32(b + 1, r >> 8);
            b[5] = 0x75;
            b[6] = (uint8_t)(2u + (r >> 20) % 30u);
            break;
        case 6: /* endbr64; ret; padding */
            b[0] = 0xF3; b[1] = 0x0F; b[2] = 0x1E; b[3] = 0xFA;
            b[4] = 0xC3;
            memset(b + 5, 0xCC, sizeof b - 5u);
            break;
        case 7: /* jump table: eight absolute targets */
            for (uint32_t k = 0; k < 2u; k++)
                put64(b + k * 8u, 0x0000550000000000ull + a + k * 16u);
            break;
        case 8: /* movabs + movups */
            b[0] = 0x48; b[1] = 0xB8;
            put64(b + 2, 0x00007F0000000000ull | (a & 0xFFFFF000u));
            b[10] = 0x0F; b[11] = 0x11; b[12] = 0x07;
            break;
        default: /* int3 / nop run */
            memset(b, (r & 1u) ? 0xCC : 0x90, sizeof b);
            break;
        }
        memcpy(o + i, b, sizeof b);
        i += sizeof b;
    }
    if (i < n) memset(o + i, 0x90, n - i);
}

static void page_rodata(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    static const char *const words[] = {
        "GLIBC_2.34", "libc.so.6", "malloc", "free", "memcpy", "strlen",
        "__stack_chk_fail", "pthread_mutex_lock", "std::__cxx11", "operator new",
        "typeinfo", "vtable", "errno", "LC_ALL", "PATH", "HOME"
    };
    size_t i = 0;

    while (i < n) {
        uint32_t r = mix_rnd(s);

        if ((i % 256u) == 0u && i + 8u <= n) {
            put64(o + i, 0x00007F0000000000ull | (page * PAGE + i));
            i += 8u;
        } else if ((r & 3u) == 0u && i + 16u <= n) {
            /* A vtable-shaped run of pointers. */
            for (uint32_t k = 0; k < 2u; k++)
                put64(o + i + k * 8u,
                      0x0000550000000000ull + page * PAGE + i + k * 8u);
            i += 16u;
        } else {
            const char *w = words[r % (sizeof words / sizeof words[0])];
            size_t len = strlen(w);

            if (i + len + 1u > n) {
                memset(o + i, 0, n - i);
                break;
            }
            memcpy(o + i, w, len);
            i += len;
            o[i++] = 0;
        }
    }
}

static void page_heap_pointers(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i = 0;

    while (i + 8u <= n) {
        uint32_t r = mix_rnd(s);
        uint64_t v;

        if ((r % 10u) < 7u)
            v = 0x00007F0000000000ull | ((uint64_t)(page * PAGE + i) << 4);
        else if ((r % 10u) < 9u)
            v = (r >> 8) & 0xFFFFFFu;
        else
            v = 0u;
        put64(o + i, v);
        i += 8u;
    }
    if (i < n) memset(o + i, 0, n - i);
}

static void page_heap_objects(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i = 0;

    while (i + 64u <= n) {
        uint32_t r = mix_rnd(s);
        uint32_t type = r % 5u;

        put32(o + i, 1u + (r >> 8) % 7u);      /* refcount */
        put32(o + i + 4u, 0x10u + type * 0x20u); /* type tag */
        put64(o + i + 8u, 0x00007F0000000000ull | ((page * PAGE + i) << 4));
        put64(o + i + 16u, (r >> 12) & 0xFFFFu);
        if (type == 0u) {
            for (uint32_t k = 0; k < 6u; k++) put32(o + i + 24u + k * 4u, r + k);
            memset(o + i + 48u, 0, 16u);
        } else if (type == 1u) {
            memset(o + i + 24u, 0, 8u);
            memcpy(o + i + 32u, "name", 4u);
            put32(o + i + 36u, r >> 8);
            memset(o + i + 40u, 0, 24u);
        } else if (type == 2u) {
            memset(o + i + 24u, 0xFF, 8u);
            put64(o + i + 32u, 0x00007F0000000000ull | (r & 0xFFFFF000u));
            memset(o + i + 40u, 0, 24u);
        } else {
            memset(o + i + 24u, 0, 40u);
        }
        i += 64u;
    }
    if (i < n) memset(o + i, 0, n - i);
}

static void page_slab_strings(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    static const char *const names[] = {
        "identifier", "request_id", "user.name", "content-type", "session",
        "trace_id", "database", "connection", "timestamp", "locale"
    };
    size_t i = 0;

    while (i + 4u < n) {
        uint32_t r = mix_rnd(s);
        const char *w = names[r % (sizeof names / sizeof names[0])];
        size_t len = strlen(w);
        uint32_t take = (uint32_t)((len > n - i - 2u) ? n - i - 2u : len);

        if (take == 0u) break;
        put16(o + i, take);
        memcpy(o + i + 2u, w, take);
        i += 2u + take;
        while (i < n && (i & 3u) != 0u) o[i++] = 0;
    }
    if (i < n) memset(o + i, 0, n - i);
    (void)page;
}

static void page_btree(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t cell = n;
    size_t ptr = 100u;
    uint32_t count = 0;

    memset(o, 0, n);
    memcpy(o, "SQLite format 3", 15);
    o[16] = 0x10;                 /* page size 4096, in the file's units */
    o[17] = 0x00;
    o[20] = 0x0D;                 /* leaf table b-tree */
    put16(o + 105u, 0);

    while (count < 40u) {
        uint32_t r = mix_rnd(s);
        uint32_t payload = 13u + (r >> 8) % 27u;
        uint32_t rowid = page * 64u + count + 1u;
        if (cell < payload + 4u || ptr + 2u > cell) break;
        cell -= payload + 2u;
        o[cell] = (uint8_t)payload;
        put32(o + cell + 1u, rowid);
        put32(o + cell + 5u, r);
        put32(o + cell + 9u, rowid * 3u);
        memset(o + cell + 13u, 'a' + (int)(r % 26u), payload - 13u);
        put16(o + ptr, (uint32_t)cell);
        ptr += 2u;
        count++;
    }
    put16(o + 3u, count);
    put16(o + 5u, (uint32_t)cell);
    /* A few freeblocks in the unallocated middle. */
    for (uint32_t k = 0; k < 3u && cell > ptr + 8u; k++) {
        size_t fb = ptr + 4u + k * 36u;
        if (fb + 4u > cell) break;
        put16(o + fb, 32u);
        put16(o + fb + 2u, 0u);
    }
}

static void page_hash_table(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i = 0;

    while (i + 16u <= n) {
        uint32_t r = mix_rnd(s);
        uint32_t k = r % 10u;

        if (k < 2u) {
            memset(o + i, 0, 16u);          /* empty slot */
        } else if (k == 2u) {
            memset(o + i, 0xFF, 8u);        /* tombstone */
            memset(o + i + 8u, 0, 8u);
        } else {
            put32(o + i, r);                /* hash */
            put32(o + i + 4u, page * 64u + (uint32_t)(i / 16u));
            put64(o + i + 8u,
                  0x00007F0000000000ull | ((page * PAGE + i * 3u) & 0xFFFFFFF0ull));
        }
        i += 16u;
    }
}

static void page_json_arena(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    static const char *const keys[] = {
        "id", "type", "name", "value", "count", "timestamp", "ok", "items"
    };
    static const char *const vals[] = {
        "alpha", "beta", "gamma", "production", "us-east-1", "pending"
    };
    size_t i = 0;

    while (i + 48u <= n) {
        uint32_t r = mix_rnd(s);
        const char *k = keys[r % (sizeof keys / sizeof keys[0])];
        const char *v = vals[(r >> 8) % (sizeof vals / sizeof vals[0])];

        put32(o + i, 2u);                    /* array length */
        put64(o + i + 8u, 0x00007F0000000000ull | (r & 0xFFFFF000u));
        put64(o + i + 16u, 0x00007F0000000000ull | ((r >> 8) & 0xFFFFF000u));
        memset(o + i + 24u, 0, 8u);
        memcpy(o + i + 32u, k, strlen(k) < 8u ? strlen(k) : 8u);
        if (i + 40u + strlen(v) < n) {
            memcpy(o + i + 40u, v, strlen(v));
        }
        i += 48u;
    }
    if (i < n) memset(o + i, 0, n - i);
    (void)page;
}

static void page_stack(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i = 0;

    while (i + 8u <= n) {
        uint32_t r = mix_rnd(s);
        uint64_t v;

        switch (r % 8u) {
        case 0: case 1:                       /* return addresses */
            v = 0x0000550000000000ull | ((page * PAGE + i) & 0xFFFFFFull);
            break;
        case 2: case 3:                       /* frame / stack pointers */
            v = 0x00007FFE00000000ull | ((r >> 8) & 0xFFFFFFull);
            break;
        case 4:                               /* cooked pointer */
            v = 0x00007F0000000000ull | (r & 0xFFFFF000ull);
            break;
        case 5:                               /* locals */
            v = (r >> 8) & 0xFFFFu;
            break;
        case 6:                               /* stack canary */
            v = 0x00000A0000000000ull | (r & 0xFF00ull);
            break;
        default:                              /* redzone */
            v = 0xCCCCCCCCCCCCCCCCull;
            break;
        }
        put64(o + i, v);
        i += 8u;
    }
}

static void page_float_series(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i = 0;

    while (i + 8u <= n) {
        uint32_t r = mix_rnd(s);
        double v;

        if ((i % (97u * 8u)) == 0u) {
            v = 0.0 / 0.0;                    /* NaN sentinel */
        } else if ((r % 64u) == 0u) {
            v = -0.0;
        } else {
            v = (double)((int32_t)(r >> 8)) / 64.0 + (double)(page % 17u);
        }
        memcpy(o + i, &v, 8u);
        i += 8u;
    }
}

static void page_blob(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i;

    (void)page;
    for (i = 0; i < n; i++) o[i] = (uint8_t)(mix_rnd(s) >> 24);
    for (i = 0; i + 10u <= n; i += 512u) {
        static const uint8_t gz[10] = {
            0x1F, 0x8B, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03
        };
        memcpy(o + i, gz, sizeof gz);
    }
}

static void page_mixed_heap(uint8_t *o, size_t n, uint32_t page, uint32_t *s)
{
    size_t i = 0;

    while (i + 256u <= n) {
        uint32_t r = mix_rnd(s);
        uint32_t kind = (r >> 8) % 5u;

        if (kind == 0u) {
            memset(o + i, 0, 256u);           /* untouched / zeroed span */
        } else if (kind <= 2u) {
            for (uint32_t k = 0; k < 32u; k++) {
                uint64_t v = ((k & 1u) == 0u)
                    ? (0x00007F0000000000ull | ((page * PAGE + i + k * 8u) << 4))
                    : ((r + k) & 0xFFFFu);
                put64(o + i + k * 8u, v);
            }
        } else if (kind == 3u) {
            memset(o + i, 0, 8u);
            memcpy(o + i + 8u, "deadbeef", 8u);
            memset(o + i + 16u, 0xCC, 16u);
            memset(o + i + 32u, 0, 224u);
        } else {
            for (uint32_t k = 0; k < 256u; k++)
                o[i + k] = (uint8_t)(mix_rnd(s) >> 24);
        }
        i += 256u;
    }
    if (i < n) memset(o + i, 0, n - i);
}

static void page_fill(uint8_t *o, size_t n, uint32_t page, int cls)
{
    uint32_t s = 0x51ED2701u ^ (uint32_t)cls * 2654435761u ^ page * 40503u;

    switch (cls) {
    case PZERO:   memset(o, 0, n); break;
    case PCODE:   page_code(o, n, page, &s); break;
    case PRODATA: page_rodata(o, n, page, &s); break;
    case PHEAPP:  page_heap_pointers(o, n, page, &s); break;
    case PHEAPO:  page_heap_objects(o, n, page, &s); break;
    case PSLAB:   page_slab_strings(o, n, page, &s); break;
    case PBTREE:  page_btree(o, n, page, &s); break;
    case PHASH:   page_hash_table(o, n, page, &s); break;
    case PJSON:   page_json_arena(o, n, page, &s); break;
    case PSTACK:  page_stack(o, n, page, &s); break;
    case PFLOAT:  page_float_series(o, n, page, &s); break;
    case PBLOB:   page_blob(o, n, page, &s); break;
    default:      page_mixed_heap(o, n, page, &s); break;
    }
}

/* Schedule a page number onto a class. Non-zero pages cycle all twelve
 * content classes; the all-pages file puts one zero in every eight. */
static int page_class(uint32_t page, int include_zero)
{
    if (include_zero && (page % 8u) == 0u) return PZERO;
    return 1 + (int)(page % (PKINDS - 1));
}

static int write_page_stream(const char *dir, const char *name, size_t size,
                             int fixed_class, int include_zero)
{
    char path[1024];
    uint8_t *buf = (uint8_t *)malloc(BLOCK);
    uint64_t base = 0;
    FILE *f;
    int n;

    if (buf == NULL) {
        fprintf(stderr, "oom\n");
        return 1;
    }
    n = snprintf(path, sizeof path, "%s/%s", dir, name);
    if (n <= 0 || (size_t)n >= sizeof path) {
        fprintf(stderr, "path too long\n");
        free(buf);
        return 2;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "cannot write %s\n", path);
        free(buf);
        return 1;
    }
    while (base < (uint64_t)size) {
        size_t chunk = (size_t)((uint64_t)size - base);
        size_t off;

        if (chunk > BLOCK) chunk = BLOCK;
        for (off = 0; off < chunk; off += PAGE) {
            size_t pn = chunk - off;
            uint32_t page = (uint32_t)((base + off) / PAGE);
            int cls = (fixed_class >= 0) ? fixed_class
                                         : page_class(page, include_zero);

            if (pn > PAGE) pn = PAGE;
            page_fill(buf + off, pn, page, cls);
        }
        if (fwrite(buf, 1u, chunk, f) != chunk) {
            fprintf(stderr, "short write on %s\n", path);
            fclose(f);
            free(buf);
            return 1;
        }
        base += chunk;
    }
    free(buf);
    return fclose(f) != 0;
}

static int write_page_corpus(const char *dir, size_t size)
{
    char path[1024];
    FILE *m;
    int i;
    int n;

    if (size % PAGE != 0u) {
        fprintf(stderr, "SIZE must be a multiple of %u for --pages\n", PAGE);
        return 2;
    }
    for (i = 0; i < PKINDS; i++) {
        char name[64];

        snprintf(name, sizeof name, "page-%s.bin", page_names[i]);
        if (write_page_stream(dir, name, size, i, 0) != 0) return 1;
    }
    if (write_page_stream(dir, "pages-nonzero.bin", size, -1, 0) != 0)
        return 1;
    if (write_page_stream(dir, "pages-all.bin", size, -1, 1) != 0)
        return 1;

    n = snprintf(path, sizeof path, "%s/pages.txt", dir);
    if (n <= 0 || (size_t)n >= sizeof path) {
        fprintf(stderr, "path too long\n");
        return 2;
    }
    m = fopen(path, "w");
    if (m == NULL) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    fprintf(m, "# class pages bytes zero-page\n");
    for (i = 0; i < PKINDS; i++)
        fprintf(m, "%s %zu %zu %d\n", page_names[i],
                size / PAGE, size, i == PZERO);
    fprintf(m, "pages-nonzero %zu %zu 0\n", size / PAGE, size);
    fprintf(m, "pages-all %zu %zu 1\n", size / PAGE, size);
    return fclose(m) != 0;
}

int main(int argc, char **argv)
{
    static uint8_t block[BLOCK];
    unsigned long size = 4u * 1024u * 1024u;
    char path[1024];
    int i, mixed = 0, probe = 0, pages = 0, probe_shape = -1, ai;

    if (argc < 2) {
        fprintf(stderr,
                "usage: corpus_write DIR [SIZE] [--mixed | --probe "
                "[--probe-shape=N] | --pages]\n");
        return 2;
    }
    for (ai = 2; ai < argc; ai++) {
        if (strcmp(argv[ai], "--mixed") == 0) {
            mixed = 1;
        } else if (strcmp(argv[ai], "--probe") == 0) {
            probe = 1;
        } else if (strcmp(argv[ai], "--pages") == 0) {
            pages = 1;
        } else if (strncmp(argv[ai], "--probe-shape=", 14) == 0) {
            long v = strtol(argv[ai] + 14, NULL, 10);

            if (v < 0 || v > 19) {
                fprintf(stderr, "--probe-shape must be 0 to 19\n");
                return 2;
            }
            probe = 1;
            probe_shape = (int)v;
        } else {
            size = strtoul(argv[ai], NULL, 10);
            if (size == 0u || size % BLOCK != 0u) {
                fprintf(stderr, "SIZE must be a non-zero multiple of %u\n",
                        BLOCK);
                return 2;
            }
        }
    }
    if (mixed + probe + pages > 1) {
        fprintf(stderr,
                "--mixed, --probe and --pages are separate corpora\n");
        return 2;
    }
    if (pages) {
        return write_page_corpus(argv[1], (size_t)size);
    }
    if (mixed || probe) {
        uint64_t base = 0u;
        int n = (probe_shape >= 0)
                    ? snprintf(path, sizeof path, "%s/probe-shape%d.bin",
                               argv[1], probe_shape)
                    : snprintf(path, sizeof path, "%s/%s.bin", argv[1],
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

                    /* A pinned shape keeps every page on the same case: the
                     * shape is chosen by page % 20, so stepping the page by
                     * twenty holds it and still varies the bytes. */
                    if (probe_shape >= 0)
                        page = (uint32_t)probe_shape + 20u * page;
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
