/* Scalar/SIMD parity + roundtrip corpus.
 *
 * Builds the same corpus twice -- once with the default (SIMD) dispatch
 * and once with -DZPSE_FORCE_SCALAR -- and the digest line must match
 * byte for byte. tests/run.ps1 does that comparison; running this
 * binary on its own still verifies every roundtrip and prints the
 * digest to compare against.
 *
 * Exit 0 = all roundtrips ok and the digest is stable. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t out[65536];

static uint64_t digest = 1469598103934665603ull; /* FNV-1a 64 */
static int fails = 0;

static void mix(const uint8_t *p, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        digest ^= (uint64_t)p[i];
        digest *= 1099511628211ull;
    }
}

static uint32_t rng_state = 0xC0FFEEu;
static uint8_t rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (uint8_t)(rng_state >> 24);
}

static void run(const char *name, uint32_t n) {
    uint32_t clen = 0, olen = 0;
    int rc = zpse_compress_block(in, n, comp, sizeof comp, &clen, 7, 1,
                                 enc_arena);
    if (rc != 0) {
        printf("FAIL %-22s compress rc=%d(%s)\n", name, rc,
               zpse_status_name(rc));
        fails++;
        return;
    }
    memset(out, 0xA5, n);
    rc = zpse_decompress_block(comp, clen, out, n, &olen, dec_arena);
    if (rc < 0 || olen != n || memcmp(in, out, n) != 0) {
        printf("FAIL %-22s decode rc=%d olen=%u\n", name, rc, olen);
        fails++;
        return;
    }
    printf("ok   %-22s n=%5u clen=%6u\n", name, n, clen);
    /* Digest the length too, so a parity break in sizing is caught. */
    mix(comp, clen);
    {
        uint8_t l[4] = { (uint8_t)clen, (uint8_t)(clen >> 8),
                         (uint8_t)(clen >> 16), (uint8_t)(clen >> 24) };
        mix(l, 4);
    }
}

int main(void) {
    int i;
    static const uint32_t sizes[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32,
                                      33, 63, 127, 255, 256, 257, 511, 1000,
                                      4095, 4096, 4097, 12345, 65535, 65536 };
    const uint32_t nsizes = (uint32_t)(sizeof sizes / sizeof sizes[0]);
    char nm[64];

    /* Deterministic corpora, each exercised at every block size. */
    for (uint32_t s = 0; s < nsizes; s++) {
        uint32_t n = sizes[s];

        memset(in, 0, n);
        sprintf(nm, "zeros/%u", n);
        run(nm, n);

        memset(in, 0x5A, n);
        sprintf(nm, "const5A/%u", n);
        run(nm, n);

        for (i = 0; i < (int)n; i++) in[i] = (uint8_t)(i & 0xFF);
        sprintf(nm, "incr/%u", n);
        run(nm, n);

        for (i = 0; i < (int)n; i++) in[i] = (uint8_t)((i / 4 + (i % 4) * 64));
        sprintf(nm, "strided4/%u", n);
        run(nm, n);

        for (i = 0; i < (int)n; i++) in[i] = (i & 1) ? 0xFF : 0x00;
        sprintf(nm, "alt/%u", n);
        run(nm, n);

        memset(in, 0, n);
        for (i = 0; i < (int)n; i += 97) in[i] = (uint8_t)(i * 7);
        sprintf(nm, "sparse/%u", n);
        run(nm, n);

        rng_state = 0x1234u + s;
        for (i = 0; i < (int)n; i++) in[i] = rnd();
        sprintf(nm, "random/%u", n);
        run(nm, n);

        rng_state = 0x777u + s;
        for (i = 0; i < (int)n; i++) {
            in[i] = (uint8_t)(rnd() & 0x0F);
        }
        sprintf(nm, "lowentropy/%u", n);
        run(nm, n);
    }

    /* Frame path over a multi-block payload. */
    {
        size_t flen = 0, olen = 0;
        static uint8_t fcomp[300000];
        for (i = 0; i < 65536; i++) in[i] = (uint8_t)((i * 13) ^ (i >> 3));
        for (int bsz = 256; bsz <= 65536; bsz *= 8) {
            int rc = zpse_compress_frame(in, 65536, fcomp, sizeof fcomp,
                                         &flen, (uint32_t)bsz, enc_arena);
            if (rc != 0) {
                printf("FAIL frame/%d rc=%d\n", bsz, rc);
                fails++;
                continue;
            }
            memset(out, 0, 65536);
            rc = zpse_decompress_frame(fcomp, flen, out, 65536, &olen,
                                       dec_arena);
            if (rc != 0 || olen != 65536 || memcmp(in, out, 65536) != 0) {
                printf("FAIL frame/%d decode rc=%d olen=%u\n", bsz, rc,
                       (unsigned)olen);
                fails++;
                continue;
            }
            printf("ok   frame/%d flen=%u\n", bsz, (unsigned)flen);
            mix(fcomp, flen);
        }
    }

    printf("DIGEST %016llX\n", (unsigned long long)digest);
    printf(fails ? "PARITY: %d FAILURES\n" : "PARITY: ALL OK\n", fails);
    return fails != 0;
}
