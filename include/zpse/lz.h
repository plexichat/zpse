#ifndef ZPSE_LZ_H
#define ZPSE_LZ_H

/* Intra-block LZ (block flag ZPSE_B_LZ, Sec. 4.6).
 *
 * Every reference is a distance back inside the same block, so a block
 * codec stays a block codec: nothing here reads across a block boundary,
 * the decoder needs no history window, and a block can still be decoded
 * on its own. Offsets fit an unsigned 16-bit field because a block is at
 * most 64 KiB.
 *
 * Payload layout (the block header is not part of it):
 *
 *   u32 literal_bytes        count of literal bytes that follow
 *   u8  literals[]           literal_bytes bytes, padded to a 4-byte
 *                            boundary with bytes the decoder never reads
 *   u32 commands[]           fixed 32-bit command words to the end
 *
 * A command word is
 *
 *   bits  0..15   offset, 1..65535; 0 means "no match, literals only"
 *   bits 16..23   literal bytes copied before the match, 0..255
 *   bits 24..31   match length minus four, 4..259 bytes
 *
 * There are no escape words and no variable-length tokens: every command
 * is one aligned 32-bit load however long the run it codes. A run longer
 * than 255 literals or 259 match bytes is written as repeated commands
 * with the same offset, which is what keeps the decoder's inner loop
 * branch-uniform (a 64 KiB run of one byte costs one command per 259
 * bytes, 1.6% of the input, and still decodes at copy speed).
 *
 * Decoding stops when the output is full, and requires that the literal
 * count, the literal stream, the command stream and the output length all
 * agree exactly: a stream that leaves any of them over is refused rather
 * than half-decoded. */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_LZ_MIN_MATCH       4u
#define ZPSE_LZ_MAX_MATCH       259u   /* 4 + 255 */
#define ZPSE_LZ_MAX_LITERALS    255u
/* The match finder's direct-mapped table: 2^hash_bits 16-bit block
 * positions. The table is the encoder's hottest random access, and a
 * 16 KiB table does not stay in L1 beside a 64 KiB block streaming past
 * it, so every position costs an L2 access. The size is a parameter of
 * the encode so an effort level can pick it: the deepest level wants the
 * matches and the big table, a level that trades size for speed wants
 * the cache. Measured on text: 11 bits (a 4 KiB table) encodes the LZ
 * level 35% faster for 1.2% more size on the worst corpus, and is
 * neutral everywhere else. */
#define ZPSE_LZ_HASH_BITS       13u
#define ZPSE_LZ_HASH_BITS_FAST 11u
#define ZPSE_LZ_TABLE_ENTRIES   (1u << ZPSE_LZ_HASH_BITS) /* 8192 */
#define ZPSE_LZ_TABLE_BYTES     (ZPSE_LZ_TABLE_ENTRIES * 2u) /* 16 KiB */



/* The smallest payload zpse_lz_encode can produce for an n-byte block:
 * the literal-count word and one command. */
#define ZPSE_LZ_MIN_PAYLOAD     8u

/* Encode n <= 65536 bytes into dst[0..cap). `table` is ZPSE_LZ_TABLE_BYTES
 * of caller-owned scratch (the encoder arena); it is filled and left
 * dirty, and nothing else is allocated.
 *
 * Returns the payload length, or 0 when the encoding does not fit in cap
 * (or the arguments are unusable). A 0 is not an error: the caller is
 * comparing sizes, and a payload that cannot fit is one that cannot win.
 * A block whose only content is literals still encodes, and comes out
 * slightly larger than the input -- that is what makes the raw fallback
 * win on incompressible data without a special case. */
uint32_t zpse_lz_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * PSE_OK, or a negative status for any stream that does not decode to
 * exactly n bytes with every reference in range. out is only valid on
 * PSE_OK; a refused stream may have had part of it written. */
int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n);

#endif /* ZPSE_LZ_H */
