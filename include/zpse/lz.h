#ifndef ZPSE_LZ_H
#define ZPSE_LZ_H

/* Intra-block LZ (block flag ZPSE_B_LZ).
 *
 * The payload is a raw LZ4 1.10.0 block preceded by two accounting words:
 *
 *   u32 uncompressed_bytes
 *   u32 command_count        number of LZ4 sequences in the block
 *   raw LZ4 block            LZ4_compress_default output
 *
 * LZ4's 2-byte distances give a 64 KiB window by definition, which is
 * exactly the block cap, so nothing here reads across a block boundary.
 * The declarations below are kept as the block layer's API; their table
 * and hash parameters are accepted and ignored. See src/lz4block.c. */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_LZ_MIN_MATCH       4u
#define ZPSE_LZ_MAX_MATCH       259u   /* 4 + 255 */
#define ZPSE_LZ_MAX_LITERALS    255u
/* Finder, hash-table, lazy-step and token-budget constants are retained for
 * source compatibility: the block layer's screens and arena sizing still
 * name them, but src/lz4block.c ignores them. LZ4 owns the match search and
 * needs no caller-supplied table. */
#define ZPSE_LZ_HASH_BITS       13u
#define ZPSE_LZ_HASH_BITS_FAST 11u
#define ZPSE_LZ_TABLE_ENTRIES   (1u << ZPSE_LZ_HASH_BITS)
#define ZPSE_LZ_TABLE_BYTES     (ZPSE_LZ_TABLE_ENTRIES * 2u)
#define ZPSE_LZ_HASH_BITS_WIDE  14u

#ifndef ZPSE_LZ_FAST4K
#define ZPSE_LZ_FAST4K 1
#endif

#define ZPSE_LZ_MIN_CMD         2u

#ifndef ZPSE_LZ_LAZY_MAX
#define ZPSE_LZ_LAZY_MAX        16384u
#endif

/* Minimum payload the block layer budgets for an LZ candidate. */
#define ZPSE_LZ_MIN_PAYLOAD     10u

/* Encode n <= 65536 bytes into dst[0..cap) as an LZ4 block with an eight-byte
 * accounting header. Returns the payload length, or 0 when LZ4 cannot fit it
 * in cap (a candidate that cannot fit cannot win, so 0 is not an error).
 * `table`, `hash_bits`, `lazy_max` and `sn` are accepted and ignored. */
uint32_t zpse_lz_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits);

uint32_t zpse_lz_encode_ex(const uint8_t *in, uint32_t n, uint8_t *dst,
                           uint32_t cap, void *table, uint32_t hash_bits,
                           uint32_t lazy_max);

uint32_t zpse_lz_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                               uint32_t cap, void *table, uint32_t hash_bits,
                               uint32_t sn);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * ZPSE_OK only when the LZ4 block decodes to exactly n bytes; otherwise a
 * negative status. out is only valid on ZPSE_OK. */
int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n);

#endif /* ZPSE_LZ_H */
