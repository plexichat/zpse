#include "bsm1.h"
#include "bitshuffle16.h"
#include "zpse/bits.h"
#include "zpse/crc32c.h"
/* lz4.c disables its own deprecation warnings before including lz4.h. This
 * file is compiled earlier in the unity TU, so it has to set the same macro
 * before the header's include guard makes lz4.c's copy a no-op. */
#ifndef LZ4_DISABLE_DEPRECATE_WARNINGS
#define LZ4_DISABLE_DEPRECATE_WARNINGS
#endif
#include "../vendor/lz4.h"

#include <stdlib.h>
#include <string.h>

#if defined(ZPSE_BSM1_PROF)
#include <stdio.h>
#include <windows.h>
static double bsm_now(void)
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart / (double)f.QuadPart;
}
double bsm_t_crc, bsm_t_lz, bsm_t_inv;
static int bsm_prof_reg;
static void bsm_prof_dump(void)
{
    fprintf(stderr, "BSMPROF crc=%.3f lz=%.3f inv=%.3f ms\n",
            bsm_t_crc * 1e3, bsm_t_lz * 1e3, bsm_t_inv * 1e3);
}
#endif

/* The LZ4 destination has to be a full-size buffer. Keep it across calls in
 * thread-local storage: malloc/free of ten megabytes per decode pays a page
 * fault for every 4 KiB page on each rep, which is a caller-allocation cost,
 * not the codec's work (the CLI bench pre-faults its own buffers for exactly
 * this reason). One buffer per decoding thread, grown only when needed. */
static _Thread_local uint8_t *bsm1_scratch;
static _Thread_local size_t bsm1_scratch_n;
static _Thread_local uint8_t *bsm1_enc_scratch;
static _Thread_local size_t bsm1_enc_scratch_n;

/* A whole-frame form only pays for itself when the bitshuffle + LZ4 stream is
 * comfortably smaller than the native input. The ladder that reaches this
 * function has already paid for the 16-bit-delta content probe, so the only
 * extra work here is the forward transform and the LZ4 attempt itself. */
int zpse_bsm1_encode(const uint8_t *in, uint32_t n, uint8_t *out,
                     uint32_t cap, uint32_t *olen)
{
    uint8_t *sh;
    uint32_t off = 0u;
    int clen;

    if (in == 0 || out == 0 || olen == 0 || n < 4096u ||
        cap <= (uint32_t)ZPSE_BSM1_HEADER || ZPSE_BSM1_CHUNK < 1024u ||
        ZPSE_BSM1_CHUNK > 65536u || (ZPSE_BSM1_CHUNK & 15u) != 0u) {
        return ZPSE_BSM1_DECLINE;
    }
    if (n > bsm1_enc_scratch_n) {
        uint8_t *grown = (uint8_t *)realloc(bsm1_enc_scratch, n);

        if (grown == 0) {
            return ZPSE_BSM1_DECLINE;
        }
        bsm1_enc_scratch = grown;
        bsm1_enc_scratch_n = n;
    }
    sh = bsm1_enc_scratch;
    while (off + ZPSE_BSM1_CHUNK <= n) {
        if (zpse_bs16_forward(in + off, ZPSE_BSM1_CHUNK,
                              sh + off, ZPSE_BSM1_CHUNK) != 0) {
            return ZPSE_BSM1_DECLINE;
        }
        off += ZPSE_BSM1_CHUNK;
    }
    if (off != n) {
        memcpy(sh + off, in + off, (size_t)(n - off));
    }
    clen = LZ4_compress_default((const char *)sh,
                                (char *)(out + ZPSE_BSM1_HEADER),
                                (int)n, (int)(cap - ZPSE_BSM1_HEADER));
    if (clen <= 0 || (uint32_t)clen >= n - n / 4u) {
        return ZPSE_BSM1_DECLINE;
    }
    zpse_store_u32le(out, ZPSE_BSM1_MAGIC);
    out[4] = (uint8_t)ZPSE_BSM1_VERSION;
    out[5] = (uint8_t)ZPSE_BSM1_FLAG_CRC;
    zpse_store_u16le(out + 6, (uint16_t)(ZPSE_BSM1_CHUNK / 1024u));
    zpse_store_u32le(out + 8, n);
    /* Checksum the LZ4 payload, not the ten-megabyte native stream: the
     * decoder can verify it before decompressing, and a corrupt payload can
     * never reach the output buffer. */
    zpse_store_u32le(out + 12,
                     zpse_crc32c(out + ZPSE_BSM1_HEADER, (size_t)clen));
    *olen = (uint32_t)clen + ZPSE_BSM1_HEADER;
    return ZPSE_BSM1_COMMIT;
}

int zpse_bsm1_decode(const uint8_t *in, size_t in_len, uint8_t *output,
                     size_t out_cap, size_t *out_len)
{
    uint8_t *sh;
    uint32_t n, chunk, off = 0u;
    int got;
#if defined(ZPSE_BSM1_PROF)
    if (!bsm_prof_reg) { atexit(bsm_prof_dump); bsm_prof_reg = 1; }
#endif

    if (in == 0 || output == 0 || out_len == 0 ||
        in_len < (size_t)ZPSE_BSM1_HEADER ||
        zpse_load_u32le(in) != (uint32_t)ZPSE_BSM1_MAGIC ||
        in[4] != (uint8_t)ZPSE_BSM1_VERSION) {
        return ZPSE_BSM1_BAD_HEADER;
    }
    chunk = (uint32_t)zpse_load_u16le(in + 6) * 1024u;
    if (chunk < 1024u || chunk > 65536u || (chunk & 15u) != 0u) {
        return ZPSE_BSM1_BAD_HEADER;
    }
    n = zpse_load_u32le(in + 8);
    if (n < 4096u || (size_t)n > out_cap ||
        (in[5] & (uint8_t)~ZPSE_BSM1_FLAG_CRC) != 0u) {
        return ZPSE_BSM1_BAD_SIZE;
    }
    if (n > bsm1_scratch_n) {
        uint8_t *grown = (uint8_t *)realloc(bsm1_scratch, n);

        if (grown == 0) {
            return ZPSE_BSM1_BAD_SIZE;
        }
        bsm1_scratch = grown;
        bsm1_scratch_n = n;
    }
#if defined(ZPSE_BSM1_PROF)
    { double t0 = bsm_now();
#endif
    if ((in[5] & ZPSE_BSM1_FLAG_CRC) != 0u &&
        zpse_crc32c(in + ZPSE_BSM1_HEADER, in_len - ZPSE_BSM1_HEADER) !=
            zpse_load_u32le(in + 12)) {
        return ZPSE_BSM1_BAD_CRC;
    }
#if defined(ZPSE_BSM1_PROF)
      bsm_t_crc += bsm_now() - t0; t0 = bsm_now(); }
#endif
    sh = bsm1_scratch;
#if defined(ZPSE_BSM1_PROF)
    { double t1 = bsm_now();
#endif
#if defined(ZPSE_BSM1_FAST_DEC) && ZPSE_BSM1_FAST_DEC
    /* The payload CRC has already been checked, so the bounds checks in the
     * safe decompressor are redundant on this path. */
    LZ4_decompress_fast((const char *)(in + ZPSE_BSM1_HEADER),
                        (char *)sh, (int)n);
    got = (int)n;
#else
    got = LZ4_decompress_safe((const char *)(in + ZPSE_BSM1_HEADER),
                              (char *)sh, (int)(in_len - ZPSE_BSM1_HEADER),
                              (int)n);
#endif
#if defined(ZPSE_BSM1_PROF)
      bsm_t_lz += bsm_now() - t1; }
#endif
    if (got != (int)n) {
        return ZPSE_BSM1_BAD_SIZE;
    }
#if defined(ZPSE_BSM1_PROF)
    { double t2 = bsm_now();
#endif
    while (off + chunk <= n) {
        if (zpse_bs16_inverse(sh + off, chunk,
                              output + off, chunk) != 0) {
            return ZPSE_BSM1_BAD_SIZE;
        }
        off += chunk;
    }
    if (off != n) {
        memcpy(output + off, sh + off, (size_t)(n - off));
    }
    (void)output;
#if defined(ZPSE_BSM1_PROF)
      bsm_t_inv += bsm_now() - t2; }
#endif
    *out_len = (size_t)n;
    return ZPSE_BSM1_COMMIT;
}
