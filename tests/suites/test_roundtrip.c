/* Block + frame roundtrip matrix.
 * Links zpse_unity.c. Exit 0 = all pass. */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t input[65536];
static uint8_t comp[131072];
static uint8_t out[65536];
static uint32_t rng = 0x12345678u;
static uint8_t rnd(void) { rng = rng * 1664525u + 1013904223u; return (uint8_t)(rng >> 24); }

static int fails = 0;

static void trial(const char *name, uint32_t n, uint32_t seq) {
    uint32_t clen = 0, olen = 0;
    int rc = zpse_compress_block(input, n, comp, sizeof comp, &clen, seq, 1, enc_arena);
    if (rc != 0) { printf("FAIL %-24s compress rc=%d(%s)\n", name, rc, zpse_status_name(rc)); fails++; return; }
    memset(out, 0, n);
    rc = zpse_decompress_block(comp, clen, out, n, &olen, dec_arena);
    if (rc < 0 || olen != n || memcmp(input, out, n) != 0) {
        printf("FAIL %-24s dec rc=%d olen=%u clen=%u\n", name, rc, olen, clen); fails++; return;
    }
    printf("ok   %-24s n=%5u clen=%5u\n", name, n, clen);
}

int main(void) {
    int i;
    uint32_t sizes[] = {1,2,7,8,9,15,16,17,255,256,257,1000,4096,65536};
    char nm[64];

    for (i = 0; i < 14; i++) {
        uint32_t n = sizes[i], k;
        for (k = 0; k < n; k++) input[k] = (uint8_t)(k & 0xFF);
        sprintf(nm, "incr/%u", n); trial(nm, n, (uint32_t)i);
    }
    for (i = 0; i < 14; i++) {
        memset(input, 0x5A, sizes[i]);
        sprintf(nm, "const5A/%u", sizes[i]); trial(nm, sizes[i], 100u + (uint32_t)i);
    }
    memset(input, 0x00, 1024); trial("const00/1024", 1024, 200);
    memset(input, 0xFF, 1024); trial("constFF/1024", 1024, 201);
    for (i = 0; i < 1024; i++) input[i] = (i & 1) ? 0xFF : 0x00;
    trial("alt00FF/1024", 1024, 202);
    rng = 42; for (i = 0; i < 5000; i++) input[i] = rnd();
    trial("random/5000", 5000, 300);
    rng = 7; for (i = 0; i < 65536; i++) input[i] = rnd();
    trial("random/65536", 65536, 301);
    memset(input, 0, 2048); for (i = 0; i < 2048; i += 97) input[i] = (uint8_t)i;
    trial("sparse/2048", 2048, 302);
    for (i = 0; i < 2048; i++) input[i] = (uint8_t)((i / 4 + (i % 4) * 64) & 0xFF);
    trial("strided/2048", 2048, 303);

    { /* frame level */
        size_t flen = 0, olen = 0;
        static uint8_t fout[70000];
        for (i = 0; i < 5000; i++) input[i] = (uint8_t)(i & 0xFF);
        int rc = zpse_compress_frame(input, 5000, fout, sizeof fout, &flen, 1024, enc_arena);
        if (rc != 0) { printf("FAIL frame compress rc=%d\n", rc); fails++; }
        else {
            memset(out, 0, 5000);
            rc = zpse_decompress_frame(fout, flen, out, 5000, &olen, dec_arena);
            if (rc != 0 || olen != 5000 || memcmp(input, out, 5000) != 0) {
                printf("FAIL frame dec rc=%d olen=%u\n", rc, (unsigned)olen); fails++;
            } else printf("ok   frame/5000 flen=%u\n", (unsigned)flen);
        }
    }
    printf(fails ? "ROUNDTRIP: %d FAILURES\n" : "ROUNDTRIP: ALL OK\n", fails);
    return fails != 0;
}
