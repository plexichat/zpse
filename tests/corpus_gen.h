/* The fixed corpus the ratio tools share.
 *
 * tests/clen_corpus.c prints one line per case (a diff between two builds
 * shows any decision that changed a size); tests/test_sizes.c pins the
 * same cases and fails when one grows. Both include this file, so the
 * shapes they measure cannot drift apart -- a corpus edited in one place
 * would otherwise stop guarding the other.
 *
 * A case is (seed, kind, size): 3 x 16 x 16 = 768 cases, deterministic.
 * corpus_gen() fills `in[0..n)` from `kind` using the seeded generator,
 * so every run of every tool sees the same bytes. */
#ifndef ZPSE_CORPUS_GEN_H
#define ZPSE_CORPUS_GEN_H

#include <stdint.h>
#include <string.h>

#define ZPSE_CORPUS_SEEDS 3
#define ZPSE_CORPUS_KINDS 16

static const uint32_t corpus_sizes[] = {
    1024u, 2048u, 3072u, 4096u, 5000u, 6144u, 8192u, 10000u,
    12288u, 16384u, 20000u, 24576u, 32768u, 40000u, 49152u, 65536u
};
#define ZPSE_CORPUS_NSIZES \
    ((int)(sizeof corpus_sizes / sizeof corpus_sizes[0]))
#define ZPSE_CORPUS_CASES \
    (ZPSE_CORPUS_SEEDS * ZPSE_CORPUS_KINDS * ZPSE_CORPUS_NSIZES)

static uint32_t corpus_rng_state;

/* The full 32-bit LCG state. The generators take the top byte of it
 * (`>> 24`); returning the state shifted right instead would make every
 * `>> 24` zero, and three of the sixteen shapes -- random, 4-bit random
 * and the 16-value palette -- would be constant planes that no candidate
 * can be ranked wrong on. They are pinned as the noisy shapes they are
 * meant to be. */
static uint32_t corpus_rnd(void)
{
    corpus_rng_state = corpus_rng_state * UINT32_C(1664525) +
                       UINT32_C(1013904223);
    return corpus_rng_state;
}

/* Each generator fills in[0..n) from `kind` and the (seeded) rng. */
static void corpus_gen(int kind, uint32_t n, uint32_t seed, uint8_t *in)
{
    uint32_t i;

    corpus_rng_state = seed;
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
            in[i] = (uint8_t)(corpus_rnd() >> 24);
        break;
    case 4: /* 4-bit random (the lowentropy shape) */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((corpus_rnd() >> 24) & 0x0Fu);
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
            in[i] = (i < n / 2u) ? (uint8_t)(corpus_rnd() >> 24) : 0x5A;
        break;
    case 11: /* two halves: constant then noise */
        for (i = 0; i < n; i++)
            in[i] = (i < n / 2u) ? 0x5A : (uint8_t)(corpus_rnd() >> 24);
        break;
    case 12: /* small alphabet, 3 values, biased */
        for (i = 0; i < n; i++) {
            uint32_t r = corpus_rnd() >> 24;
            in[i] = (uint8_t)(r < 200u ? 7u : (r < 250u ? 9u : 3u));
        }
        break;
    case 13: /* 16-value repeating palette, shuffled */
        for (i = 0; i < n; i++)
            in[i] = (uint8_t)((corpus_rnd() >> 24) & 0x0Fu);
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
            in[i] = (i % 4u == 0) ? (uint8_t)(corpus_rnd() >> 24)
                                  : (uint8_t)(i & 0x0Fu);
        break;
    }
}

/* The seed a case is generated from, matching clen_corpus's ordering. */
static uint32_t corpus_seed(int seed_index, int kind, uint32_t n)
{
    return (uint32_t)((seed_index + 1) * 7919 + kind * 104729 + (int)n);
}

#endif /* ZPSE_CORPUS_GEN_H */
