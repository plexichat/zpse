#ifndef ZPSE_LZ2_H
#define ZPSE_LZ2_H

/* Intra-block LZ, fast form (block flag ZPSE_B_LZ2).
 *
 * A second LZ codec, not a new version of the first one. src/lz.c spends its
 * per-position budget on three things a plain LZ finder does not need: a lazy
 * probe one byte ahead, a backward extension into the pending literals, and a
 * four-entry repeat-distance cache that turns a repeat into a one-byte
 * distance. They are worth size on structured data and they are paid for on
 * every position, and on prose they are most of the gap to lz4: the finder
 * runs at 256 MB/s on `dickens` against lz4's 429 on the same blocks, for a
 * payload within 2.6%.
 *
 * This form keeps two of them: a four-slot repeat cache, because a repeat is
 * a byte of distance where the token form spends one on its own code, and one
 * candidate per hash entry with no lookahead and no backward walk. A command
 * is a token, its literals and a distance that is one byte when it repeats
 * and a fixed little-endian 16-bit field otherwise, which is what lz4 writes
 * and why its decoder is a tight loop.
 *
 * The repeat code was added here after the varint that first tried it cost
 * this decoder a third of its rate: the four codes sit at the top of the
 * first distance byte, so the test is one compare against 0xFC and there is
 * no loop to run. That is the size the token form's cache buys, for a branch
 * instead of a parse -- and this form writes the blocks whose head it codes
 * smaller than the token form does (see zpse_lz_pick_fast in src/block.c),
 * so on prose its decode rate is the default level's decode rate.
 *
 * Payload layout, after the 32-byte block header:
 *
 *   u32 literal_bytes        total literal bytes in the stream
 *   u32 command_count        number of tokens
 *   token stream             to the end of the payload
 *
 * Each command is:
 *
 *   token       high nibble: literal bytes before the match, 0..14 or 15 for
 *               "more follows"; low nibble: match length minus four, same
 *               escape rule
 *   escapes     one byte per escaped field: 255-chained, every 0xFF adds 255
 *               and the first byte below 255 ends the run
 *   literals    the run the token names, inline
 *   distance    one byte 0xFC..0xFF for the first to fourth most recent
 *               distance (the four-slot cache, seeded at 1), else u16 little
 *               endian on whose low byte 0xFC..0xFF never appear; 0 means no
 *               match and literals only
 *
 * The streams are self-checking the same way src/lz.c's are: the decode stops
 * when the output is full and then requires that the literal total, the
 * command count, the stream position and the output length all agree exactly.
 * A stream that leaves any of them over is refused rather than half-decoded.
 */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_LZ2_MIN_MATCH 4u
/* Literal run and match length one token can carry before an escape, and the
 * longest match a single command codes: the encoder chunks longer runs so
 * that no field ever needs more than one escape byte. A match of 273 is the
 * ceiling because its escape value is (273 - 4) - 15 = 254; 274 would write
 * 255, which the decoder reads as "another escape byte follows" and the
 * chunks after it would be read from the wrong place. */
#define ZPSE_LZ2_MAX_LIT   255u
#define ZPSE_LZ2_MAX_MATCH 273u
#define ZPSE_LZ2_MIN_PAYLOAD 10u

/* Encode n <= 65536 bytes into dst[0..cap). `table` is the same
 * ZPSE_LZ_TABLE_BYTES scratch src/lz.c uses -- a plain 2^hash_bits array of
 * block positions, cleared per call. Returns the payload length, or 0 when the
 * encoding cannot fit in cap or the arguments are unusable. */
uint32_t zpse_lz2_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * ZPSE_OK, or a negative status for any stream that does not decode to exactly
 * n bytes with every reference in range. */
int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n);

#endif /* ZPSE_LZ2_H */
