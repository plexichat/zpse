#ifndef ZPSE_LZH_H
#define ZPSE_LZH_H

/* Intra-block LZ, hexadecimal form (block flags ZPSE_B_LZ | ZPSE_B_LZ2).
 *
 * The third codec in the LZ family. Two commands share a byte because a
 * command's two fields -- the literal run and the match length -- are written
 * as ONE HEX DIGIT when they are one of the fifteen pairs the corpus actually
 * asks for; the first two forms spend a whole byte on every command's token
 * (0xLR, one hex digit a field). The nibble book alone is a wash: half the
 * commands miss it and a miss costs the nibble it sat in plus the byte token,
 * which is exactly what a hit saved, and measured as a swap for the fast form
 * it was 0.73 to 1.56% LARGER at 4 KiB and 64 KiB.
 *
 * What makes the form pay is the distance code below it. The fast form spends
 * two bytes on every fresh distance because its first byte has to leave room
 * for four repeat codes; this form leaves only the top sixteen codes for them,
 * so a fresh distance of 1..224 is one byte and one of 225..4320 is two. A 4
 * KiB page's distances are mostly small, and there the form is 2.51% smaller
 * than the fast one over the dumps and 2.10% over Silesia on the same finder
 * pass; at 64 KiB the three-byte wide form costs more than the short codes
 * save (Silesia grows 3.2%), which is why a page is the only block size that
 * takes it -- from level 7 up, where size is the point (see
 * ZPSE_LZH_PAGE_LEVEL in src/block.c).
 *
 * The fifteen codes are not guessed: they are the most frequent (literal run,
 * match length) pairs in the fast form's own command stream, measured over
 * 24 234 323 commands of the dump corpora and nine Silesia files, and they
 * cover 48.53% of those commands. Fifteen and not sixteen because code 15 is
 * the escape (the sixteenth-ranked pair, 1.56%, is simply spelled out). A
 * pair outside the book is written as code 15 followed by the byte token,
 * which is the fast form's own (0xLR) code with the same escape rules, so the
 * form is never worse than one extra nibble for a command the book misses --
 * and a pair of hits is half a byte per command cheaper than either existing
 * form.
 *
 * Payload layout, after the block header:
 *
 *   u32 literal_bytes        total literal bytes in the stream
 *   u32 command_count        number of commands
 *   token pairs              ceil(command_count / 2) bytes, first command in
 *                            the high nibble; a nibble of 15 means the next
 *                            command byte is a byte token, so the book has
 *                            fifteen codes and not sixteen
 *   command bodies           in command order, immediately after the token
 *                            byte of the pair they belong to
 *
 * A command body is, in this order:
 *
 *   literal escape           one byte, only when the run code is 15:
 *                            run = 15 + byte
 *   length escape            one byte, only when the length code is 15:
 *                            match length = 18 + byte
 *   literals                 run bytes, verbatim
 *   distance                 one byte 0xFC..0xFF for one of the four most
 *                            recent distances (slot in the low two bits); one
 *                            byte 0x00..0xDF for a fresh distance of 1..224;
 *                            two bytes 0xE0..0xEF + low byte for 225..4320;
 *                            three bytes 0xF0 + u16 little endian above that.
 *                            0xF1..0xFB are reserved and refused. Absent
 *                            entirely when the command has no match.
 *
 * Field codes, shared by the nibble and the byte token:
 *
 *   run code    0..14: that many literals; 15: 15 + the escape byte (0..254),
 *               so one command carries at most 269.
 *   length code 0: no match (a literal-only command); 1..14: match length
 *               minus three, so 4..17; 15: 18 + the escape byte, at most 272.
 *
 * A code of 15 in the high nibble of a token byte means the next byte is a
 * byte token: high nibble the run code, low nibble the length code, same
 * rules. The byte token's own escapes follow it. The low nibble of a token
 * byte whose command does not exist (an odd command count) is written zero and
 * never read.
 *
 * Every reference is a distance back inside the same block, so a block stays
 * independently decodable. The streams are self-checking the same way the two
 * other forms' are: decoding stops when the output is full and then requires
 * that the literal total, the command count and the stream position all agree
 * exactly. A stream that leaves any of them over, or that resolves a distance
 * outside the bytes it has already written, is refused rather than
 * half-decoded. */

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

/* Encode n <= 65536 bytes into dst[0..cap). `table` is ZPSE_LZ_TABLE_BYTES of
 * caller-owned scratch, as src/lz2.c uses it. Returns the payload length, or 0
 * when the encoding does not fit in cap or the arguments are unusable. */
uint32_t zpse_lzh_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits);

/* Decode a payload of in_len bytes into exactly n bytes at out. Returns
 * ZPSE_OK, or a negative status for any stream that does not decode to exactly
 * n bytes with every reference in range. */
int zpse_lzh_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n);

#endif /* ZPSE_LZH_H */
