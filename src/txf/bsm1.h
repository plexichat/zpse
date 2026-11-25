#ifndef ZPSE_BSM1_H
#define ZPSE_BSM1_H

/* Whole-frame BSM1: BITSHUFFLE_16 followed by one raw LZ4 block.
 *
 *   [0..3]   magic "BSM1" (little-endian u32)
 *   [4]      version = 1
 *   [5]      flags: bit0 = CRC32C of the original bytes is stored
 *   [6..7]   bitshuffle chunk size in KiB (u16 LE; 64 = the default)
 *   [8..11]  original length n (u32 LE)
 *   [12..15] CRC32C of the original bytes (0 when bit0 is clear)
 *   [16..]   LZ4 block of the bitshuffled native bytes
 *
 * The bitshuffle is applied independently to each whole 64 KiB chunk of the
 * input; a short tail is copied verbatim. n is therefore length-preserving
 * and the decoder needs no side metadata beyond n. Decode is one LZ4 pass
 * into a scratch buffer followed by one bitshuffle inverse straight into the
 * caller's output -- no block directory, no per-segment transform machinery.
 */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_BSM1_MAGIC  UINT32_C(0x314D5342) /* "BSM1" LE */
#define ZPSE_BSM1_HEADER 16u
#define ZPSE_BSM1_VERSION 1u
#define ZPSE_BSM1_FLAG_CRC 1u
#ifndef ZPSE_BSM1_CHUNK
/* 32 KiB: measured faster than 64 KiB on mr (one inverse batch fits L2). */
#define ZPSE_BSM1_CHUNK 32768u
#endif

#define ZPSE_BSM1_COMMIT  1
#define ZPSE_BSM1_DECLINE 0
#define ZPSE_BSM1_BAD_HEADER (-1)
#define ZPSE_BSM1_BAD_SIZE   (-2)
#define ZPSE_BSM1_BAD_CRC    (-3)

int zpse_bsm1_encode(const uint8_t *in, uint32_t n, uint8_t *out,
                     uint32_t cap, uint32_t *olen);
int zpse_bsm1_decode(const uint8_t *in, size_t in_len, uint8_t *output,
                     size_t out_cap, size_t *out_len);

#endif /* ZPSE_BSM1_H */
