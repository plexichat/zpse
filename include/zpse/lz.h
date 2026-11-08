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
 *   u32 command_count        number of tokens that follow
 *   u8  literals[]           literal_bytes bytes, padded to a 4-byte
 *                            boundary with bytes the decoder never reads
 *   token stream             to the end of the payload
 *
 * Each command is a token byte and an offset:
 *
 *   token       high nibble: literal bytes copied before the match,
 *               0..14 or 15 for "more follows"
 *               low nibble:  match length minus four, 0..14 or 15 for
 *               "more follows"
 *   escapes     each nibble that reads 15 is followed by 255-chained
 *               bytes: every 0xFF adds 255 and the first byte below 255
 *               ends the run
 *   offset      7-bit groups, least significant first, high bit set on
 *               every byte but the last:
 *                 0      no match, literals only
 *                 1..4   the i-th most recently used distance
 *                 5..    an explicit distance, value minus four
 *
 * The repeat distances are the four most recent matches, most recent
 * first, seeded to four ones. Using one moves it to the front; using an
 * explicit distance pushes it to the front and drops the oldest. That is
 * worth nothing on prose (almost every match is a fresh distance) and
 * 24% of the matches on executable and archive data, where a structure
 * repeats at a stride the finder keeps landing on.
 *
 * Four bytes per command, whatever the run, was the old form and it was
 * the codec's worst ratio decision: prose averages 5.6 bytes of output
 * per command, so commands cost 0.71 bytes per byte of input against
 * roughly 0.21 for a variable-length token. Moving to one bought 25% of
 * the payload on `dickens` (1.26:1 to 1.57:1), 22% on `webster` and 19%
 * on `mozilla`, measured by re-encoding the commands the finder already
 * produced. A run longer than 255 literals or 259 match bytes is still
 * several commands with the same offset, and after the first the repeat
 * code makes every one of them a byte shorter.
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



/* The smallest possible command: a token byte and one offset byte. A
 * stream can never code more than ZPSE_LZ_MAX_MATCH bytes with one of
 * them, which is the floor the block encoder budgets against. */
#define ZPSE_LZ_MIN_CMD         2u

/* The largest block the length-measured lazy step is ever offered. It was
 * tuned at a 4 KiB page and is not a no-op above one -- on a 16 KiB block it
 * can give up a byte for a later match that does not pay it back -- so the
 * only caller that asks for it past a page runs the pass twice and keeps the
 * smaller stream (src/block.c). 16 KiB is the next page size up. */
#define ZPSE_LZ_LAZY_MAX        16384u

/* The smallest payload zpse_lz_encode can produce for an n-byte block:
 * the two header words and one command -- a token byte and a one-byte
 * distance. A block whose first match starts at offset one is exactly
 * this, so anything larger would refuse payloads the encoder writes. */
#define ZPSE_LZ_MIN_PAYLOAD     10u

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

/* The same pass, with the length-measured lazy step enabled on blocks up to
 * `lazy_max` bytes rather than up to a page. `lazy_max == 0` is the pass with
 * the step off, which is what zpse_lz_encode is for every block above a page.
 * The stream such a pass writes is not ordered against the page one -- taking
 * the smaller of the two is the caller's job. */
uint32_t zpse_lz_encode_ex(const uint8_t *in, uint32_t n, uint8_t *dst,
                           uint32_t cap, void *table, uint32_t hash_bits,
                           uint32_t lazy_max);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * PSE_OK, or a negative status for any stream that does not decode to
 * exactly n bytes with every reference in range. out is only valid on
 * PSE_OK; a refused stream may have had part of it written. */
int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n);

#endif /* ZPSE_LZ_H */
