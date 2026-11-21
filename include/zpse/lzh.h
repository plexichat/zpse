#ifndef ZPSE_LZH_H
#define ZPSE_LZH_H

/* Intra-block LZ, hexadecimal form (block flags ZPSE_B_LZ | ZPSE_B_LZ2).
 *
 * This selector is still offered for page-sized blocks at level 7 and up and
 * when ZPSE_ENC_LZ_HEX is set, but its payload is the same raw LZ4 block as
 * the other LZ forms; the old nibble codebook and short distance code are
 * gone. See src/lz4block.c.
 */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_LZH_MIN_MATCH 4u
/* Longest runs one command carries, and 254 is the largest escape value: a
 * 255 byte is the decoder's "another escape byte follows", so a run of 270
 * (15 + 255) or a match of 273 (18 + 255) could not be written as one escape
 * and would desync the stream. The writer chunks longer runs instead. */
#define ZPSE_LZH_MAX_LIT   269u /* 15 + 254 */
#define ZPSE_LZH_MAX_MATCH 272u /* 18 + 254 */
/* The smallest payload the encoder can write: the two header words, a token
 * and a one-byte distance -- a block whose first match starts at offset one. */
#define ZPSE_LZH_MIN_PAYLOAD 10u

/* The fifteen nibble codes, in the order the corpus ranked them: {literal
 * run, match length}, with a length of zero meaning a literal-only command.
 * Measured over 24 234 323 commands (top of the list first: 7.15% down to
 * 1.57%, 48.53% of all commands in the fifteen).
 *
 * The table is part of the format: a decoder reads these as literals. It is
 * also fixed rather than fitted per corpus on purpose -- a per-block book
 * would cost the bytes it saves. */
typedef struct ZpseLzhCode {
    uint8_t run;
    uint8_t len;
} ZpseLzhCode;

#define ZPSE_LZH_CODES 15u

/* Encode into an LZ4 block with an eight-byte accounting header. Returns 0
 * when it cannot fit in cap; `table` and `hash_bits` are ignored. */
uint32_t zpse_lzh_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits);

/* Price a small sample without clearing the whole hash table. `sn` is the
 * sample length (the `n` this call encodes); only the slots a position in
 * [0,sn) can hash to are cleared. The stream is byte-identical to the one
 * zpse_lzh_encode writes for the same input. */
uint32_t zpse_lzh_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sn);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * ZPSE_OK, or a negative status for any stream that does not decode to exactly
 * n bytes with every reference in range. */
int zpse_lzh_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n);

#endif /* ZPSE_LZH_H */
