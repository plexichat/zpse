#ifndef ZPSE_LZ2_H
#define ZPSE_LZ2_H

/* Intra-block LZ, fast form (block flag ZPSE_B_LZ2).
 *
 * A second LZ codec, not a new version of the first one. It spends its
 * per-position budget on the same things src/lz.c does -- a lazy probe one
 * byte ahead and a backward extension into the pending literals -- but codes
 * a match with a fixed two-byte little-endian distance instead of a
 * repeat-distance code. The finder takes one candidate per hash entry, so a
 * command is a token, its inline literals and a distance; the decoder's only
 * distance work is one 16-bit load, which is the shape lz4 writes.
 *
 * A four-slot repeat-distance cache used to turn a repeat into a one-byte
 * distance (0xFC | slot). Removing it measured +26.5% decode (2.31 -> 2.93
 * GB/s), +12.8% encode and +1.366% compressed size over twelve 64 KiB Silesia
 * blocks; seven of the twelve were within 0.2% of their old size. The rep
 * branch and the escape a match whose low distance byte was 0xFC..0xFF needed
 * are both gone, so the full 1..65535 range is usable. The size cost sits in
 * the spike-pattern cases (mozilla +4.02%, nci +3.89%), where a recent
 * distance is named often enough to be worth a byte.
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
 *   lit escape  one byte when the literal nibble is 15: 255-chained, every
 *               0xFF adds 255 and the first byte below 255 ends the run
 *   literals    the run the token names, inline
 *   distance    fixed u16 little endian, 1..65535; zero means no match and
 *               the command is literals only (no escape, no repeat code)
 *   len escape  one byte when the match-length nibble is 15, same 255-chain
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

/* Price a small sample without clearing the whole hash table. `sn` is the
 * sample length (the `n` this call encodes); only the slots a position in
 * [0,sn) can hash to are cleared. The stream is byte-identical to the one
 * zpse_lz2_encode writes for the same input. */
uint32_t zpse_lz2_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sn);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * ZPSE_OK, or a negative status for any stream that does not decode to exactly
 * n bytes with every reference in range. */
int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n);

#endif /* ZPSE_LZ2_H */
