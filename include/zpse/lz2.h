#ifndef ZPSE_LZ2_H
#define ZPSE_LZ2_H

/* Intra-block LZ, fast form (block flag ZPSE_B_LZ2).
 *
 * This selector is kept for the level ladder and for ZPSE_ENC_LZ_FAST, but
 * its payload is the same raw LZ4 block as every other LZ form:
 *
 *   u32 uncompressed_bytes
 *   u32 command_count
 *   raw LZ4 block
 *
 * The fixed two-byte distance and single 16-bit load that this form once
 * defined are now LZ4's own; there is no separate fast-form codec left.
 * See src/lz4block.c.
 */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_LZ2_MIN_MATCH 4u
/* Retained for source compatibility; ignored by src/lz4block.c. */
#define ZPSE_LZ2_MAX_LIT   255u
#define ZPSE_LZ2_MAX_MATCH 273u
#define ZPSE_LZ2_MIN_PAYLOAD 10u

/* Encode into an LZ4 block with an eight-byte accounting header. Returns 0
 * when it cannot fit in cap; `table` and `hash_bits` are ignored. */
uint32_t zpse_lz2_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits);

/* Sample encoder: same LZ4 block, `sn` accepted and ignored. */
uint32_t zpse_lz2_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sn);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * ZPSE_OK, or a negative status for any stream that does not decode to exactly
 * n bytes with every reference in range. */
int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n);

#endif /* ZPSE_LZ2_H */
