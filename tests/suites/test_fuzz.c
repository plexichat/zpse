/* Randomized differential corpus.
 *
 * tests/test_parity.c pins the encoder's OUTPUT over a fixed list of
 * patterns; this suite attacks the same invariant from the other side, over
 * thousands of shapes and sizes instead of dozens. It is built several ways
 * by tests/run.sh / run.ps1 and the printed summary line must be identical
 * in every build:
 *
 *   default (runtime SIMD dispatch)
 *   -DZPSE_FORCE_SCALAR      (the scalar reference paths)
 *   -DZPSE_FORCE_TRIAL_SWAP  (the block sweep takes the other attempt-order
 *                             decision wherever it has one, so its two width-1
 *                             candidates are tried in the opposite order)
 *
 * The last one is the interesting one for this suite: the block encoder's
 * candidate order is a free performance knob rather than a format decision,
 * so the two orders must produce byte-identical compressed blocks for EVERY
 * input, not just the bench corpus. A wide randomized corpus is what makes
 * that claim testable, so the digest below covers each block's size, its
 * compressed length and its compressed bytes.
 *
 * Nothing printed here may depend on the build: no addresses, no timings, no
 * sizeof-derived counts. Exit 0 = every round trip passed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t out[65536];
static uint8_t fcomp[300000];
static uint8_t fout[65536];

static int fails = 0;
static uint64_t dig = 1469598103934665603ull; /* FNV-1a 64 */

static void mix(const uint8_t *p, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        dig ^= (uint64_t)p[i];
        dig *= 1099511628211ull;
    }
}

static void mix_u32(uint32_t v)
{
    uint8_t b[4];
    b[0] = (uint8_t)v;
    b[1] = (uint8_t)(v >> 8);
    b[2] = (uint8_t)(v >> 16);
    b[3] = (uint8_t)(v >> 24);
    mix(b, 4u);
}

/* Deterministic PRNG: identical corpus on every run and platform. */
static uint32_t rs = 0x2468ACE0u;

static uint8_t nr(void)
{
    rs = rs * 1664525u + 1013904223u;
    return (uint8_t)(rs >> 24);
}

static uint32_t nr32(void)
{
    rs = rs * 1664525u + 1013904223u;
    return rs >> 5;
}

/* ------------------------------------------------------------------ *
 * Shapes. Each fills `dst[0..len)` with one regime; the block shapes
 * below either use one of these or concatenate several of them, so a
 * single block can hold more than one kind of content.
 * ------------------------------------------------------------------ */
static void region_fill(uint8_t *dst, uint32_t len, uint32_t kind,
                        uint32_t seed)
{
    uint32_t i;
    uint32_t acc = seed;

    rs = seed * 2654435761u + 1u;
    switch (kind % 8u) {
    case 0:
        memset(dst, 0, len);
        break;
    case 1:
        memset(dst, 0x5Au, len);
        break;
    case 2:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)((seed + i) & 0xFFu);
        }
        break;
    case 3:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)(nr() & 0x0Fu);
        }
        break;
    case 4:
        for (i = 0; i < len; i++) {
            dst[i] = nr();
        }
        break;
    case 5:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)((i % 251u) * 3u);
        }
        break;
    case 6:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)((i & 1u) ? 0xFFu : 0x00u);
        }
        break;
    default:
        for (i = 0; i < len; i++) {
            acc += 1u + (uint32_t)(nr() % 3u);
            dst[i] = (uint8_t)acc;
        }
        break;
    }
}

#define SHAPE_N 18

static const char *const shape_name[SHAPE_N] = {
    "zeros", "const5A", "constFF", "ramp", "ramp251", "inter2",
    "inter8", "inter16", "alt00FF", "sparse97", "sparse251", "lownib",
    "random", "walk", "walknoise", "regimes", "concat", "shifted"
};

static void fill(uint32_t shape, uint32_t n)
{
    uint32_t i;

    switch (shape) {
    case 0:
        memset(in, 0, n);
        break;
    case 1:
        memset(in, 0x5A, n);
        break;
    case 2:
        memset(in, 0xFF, n);
        break;
    case 3:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)(i & 0xFFu);
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)(i % 251u);
        }
        break;
    case 5:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i & 1u) ? 0xA5u : 0x5Au);
        }
        break;
    case 6:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i % 8u) * 31u);
        }
        break;
    case 7:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i % 16u) * 17u);
        }
        break;
    case 8:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i & 1u) ? 0xFFu : 0x00u);
        }
        break;
    case 9:
        memset(in, 0, n);
        for (i = 0; i < n; i += 97u) {
            in[i] = (uint8_t)i;
        }
        break;
    case 10:
        memset(in, 0, n);
        for (i = 0; i < n; i += 251u) {
            in[i] = (uint8_t)(i * 3u);
        }
        break;
    case 11:
        rs = 0xD1B54A32u + n;
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)(nr() & 0x0Fu);
        }
        break;
    case 12:
        rs = 0x94D049BBu + n;
        for (i = 0; i < n; i++) {
            in[i] = nr();
        }
        break;
    case 13:
        rs = 0x1B873593u + n;
        {
            uint32_t acc = 0u;
            for (i = 0; i < n; i++) {
                acc += 1u + (uint32_t)(nr() % 3u);
                in[i] = (uint8_t)acc;
            }
        }
        break;
    case 14:
        rs = 0x85EBCA6Bu + n;
        {
            uint32_t acc = 0u;
            for (i = 0; i < n; i++) {
                acc += 1u + (uint32_t)(nr() % 3u);
                if ((nr() % 17u) == 0u) {
                    in[i] = nr();
                } else {
                    in[i] = (uint8_t)acc;
                }
            }
        }
        break;
    case 15: { /* three regimes inside one block */
        uint32_t a = n / 3u;
        uint32_t b = (2u * n) / 3u;

        region_fill(in, a, 2u, 0x1234u);
        region_fill(in + a, b - a, 4u, 0x5678u);
        region_fill(in + b, n - b, 0u, 0x9ABCu);
        break;
    }
    case 16: { /* regions of random kinds at random boundaries */
        uint32_t p = 0u;
        uint32_t r = 0u;
        uint32_t keep;

        rs = 0x51ED270Bu + n;
        while (p < n) {
            uint32_t len = 1u + (nr32() % 3000u);

            if (len > n - p) {
                len = n - p;
            }
            /* region_fill reseeds `rs` for its own stream; keep the
             * boundary stream separate so the split stays its own
             * deterministic sequence. */
            keep = rs;
            region_fill(in + p, len, r, 0x1000u + r * 7u + n);
            rs = keep;
            r++;
            p += len;
        }
        break;
    }
    default: { /* the smooth walk, shifted one byte along */
        uint8_t prev = 0u;

        rs = 0x1B873593u + n;
        {
            uint32_t acc = 0u;
            for (i = 0; i < n; i++) {
                uint8_t cur;

                acc += 1u + (uint32_t)(nr() % 3u);
                cur = (uint8_t)acc;
                in[i] = prev;
                prev = cur;
            }
        }
        break;
    }
    }
}

int main(int argc, char **argv)
{
    /* Optional starting PRNG seed, for the nightly job that must not
     * restart from scratch each time. Pass the seed saved by the previous
     * nightly run and fuzzing continues from where it left off. */
    if (argc > 1) {
        rs = (uint32_t)strtoul(argv[1], NULL, 0);
    }
    /* Size spread: every size up to 64, the tile/segment boundaries, a
     * spread of larger powers of two with their neighbours, and
     * pseudorandom sizes across the whole legal range. */
    static const uint32_t edge[] = {
        127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u, 513u, 1023u, 1024u,
        1025u, 2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 8191u, 8192u,
        8193u, 16383u, 16384u, 16385u, 32768u, 65535u, 65536u
    };
    static uint32_t sizes[192];
    uint32_t nsizes = 0u;
    uint32_t si;
    uint32_t shape;
    uint32_t seq = 0u;
    uint32_t cases = 0u;
    uint32_t frames = 0u;
    int i;

    for (i = 1; i <= 64; i++) {
        sizes[nsizes++] = (uint32_t)i;
    }
    for (i = 0; i < (int)(sizeof edge / sizeof edge[0]); i++) {
        sizes[nsizes++] = edge[i];
    }
    rs = 0xF00Du;
    for (i = 0; i < 60; i++) {
        sizes[nsizes++] = 1u + (nr32() % ZPSE_MAX_BLOCK_SIZE);
    }

    for (si = 0u; si < nsizes; si++) {
        uint32_t n = sizes[si];

        for (shape = 0u; shape < SHAPE_N; shape++) {
            uint32_t clen = 0u;
            uint32_t olen = 0u;
            int rc;

            fill(shape, n);
            rc = zpse_compress_block(in, n, comp, sizeof comp, &clen, seq,
                                     1, enc_arena);
            if (rc != 0) {
                printf("FAIL case=%u shape=%s n=%u compress rc=%d(%s)\n",
                       (unsigned)cases, shape_name[shape], (unsigned)n, rc,
                       zpse_status_name(rc));
                fails++;
                seq++;
                continue;
            }
            /* zpse_compress_block reports 0 on success; the byte count
             * is the out parameter. What the encoder DOES guarantee is
             * that a coded block is only ever chosen while it beats the
             * RAW envelope, so no block can ever exceed it. */
            if (clen > n + (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
                printf("FAIL case=%u shape=%s n=%u clen=%u over envelope\n",
                       (unsigned)cases, shape_name[shape], (unsigned)n,
                       (unsigned)clen);
                fails++;
                seq++;
                continue;
            }
            memset(out, 0xA5, n);
            rc = zpse_decompress_block(comp, clen, out, n, &olen, dec_arena);
            if (rc < 0 || olen != n || memcmp(in, out, n) != 0) {
                printf("FAIL case=%u shape=%s n=%u decode rc=%d olen=%u\n",
                       (unsigned)cases, shape_name[shape], (unsigned)n, rc,
                       (unsigned)olen);
                fails++;
                seq++;
                continue;
            }
            mix_u32(n);
            mix_u32(clen);
            mix(comp, clen);
            cases++;
            seq++;
        }
    }

    /* Frames: same contract, over several nominal block sizes, including
     * one that splits a 64 KiB payload into 256 blocks. */
    {
        static const uint32_t fn[6] = { 20000u, 12345u, 65536u, 3000u,
                                        50000u, 900u };
        static const uint32_t fnbs[6] = { 1024u, 4096u, 65536u, 256u,
                                          8192u, 512u };
        static const uint32_t fshape[6] = { 12u, 13u, 11u, 16u, 14u, 15u };

        for (i = 0; i < 6; i++) {
            uint32_t n = fn[i];
            size_t flen = 0u;
            size_t olen = 0u;
            int rc;

            fill(fshape[i], n);
            rc = zpse_compress_frame(in, (size_t)n, fcomp, sizeof fcomp,
                                     &flen, fnbs[i], enc_arena);
            if (rc != 0) {
                printf("FAIL frame=%d shape=%s n=%u nbs=%u compress rc=%d\n",
                       i, shape_name[fshape[i]], (unsigned)n,
                       (unsigned)fnbs[i], rc);
                fails++;
                continue;
            }
            memset(fout, 0xA5, n);
            rc = zpse_decompress_frame(fcomp, flen, fout, (size_t)n, &olen,
                                       dec_arena);
            if (rc != 0 || olen != (size_t)n ||
                memcmp(in, fout, (size_t)n) != 0) {
                printf("FAIL frame=%d shape=%s n=%u nbs=%u decode rc=%d "
                       "olen=%u\n", i, shape_name[fshape[i]], (unsigned)n,
                       (unsigned)fnbs[i], rc, (unsigned)olen);
                fails++;
                continue;
            }
            mix_u32(n);
            mix_u32(fnbs[i]);
            mix_u32((uint32_t)flen);
            mix(fcomp, flen);
            mix(fout, (size_t)n);
            frames++;
        }
    }

    printf("FUZZ cases=%u frames=%u digest=%016llX seed=%08X\n",
           (unsigned)cases, (unsigned)frames, (unsigned long long)dig, rs);
    if (fails == 0) {
        printf("FUZZ: ALL OK\n");
    } else {
        printf("FUZZ: %d FAILURES\n", fails);
    }
    return fails != 0;
}
