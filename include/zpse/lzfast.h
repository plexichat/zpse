#ifndef ZPSE_LZFAST_H
#define ZPSE_LZFAST_H

/* Minimal intra-block LZ finder: same command stream as an LZ4 block, with
 * the smallest useful search loop.
 *
 * This is a port of the measured prototype at
 * C:\Users\dboyn\AppData\Local\Temp\opencode\mf2\mf2_bench.c (see RESULTS.md
 * there). It is byte-exact round-trip verified and was measured in memory on
 * 48 x 64 KiB Silesia blocks, AVX2 compiled out for both sides of the A/B:
 *
 *   lz2-current   270.6 MB/s enc / 1632 dec / 1.830 ratio
 *   minimal, 14-bit table  356.6 enc (+32%) / 1935 dec (+19%) / 1.766
 *   minimal, 13-bit table  384.2 enc (+47%) / 1943 dec (+27%) / 1.745
 *
 * The 13-bit form is the default here: it is the same table size as lz2's
 * (8192 uint16_t slots = 16 KiB) and it measured the fastest. The finder's
 * hash width is NOT part of the wire format -- the decoder has no table at
 * all -- so calling this a "14-bit rolling multiply hash" would be about the
 * prototype's default build, not about anything a stream has to know.
 *
 * What was removed from src/lz2.c: the lazy one-byte probe, the backward
 * walk into pending literals, the varint distance and, later, the four-slot
 * repeat-distance cache. What remains: a 4-byte minimum match, one 32-bit load
 * hashed by a single multiply-shift into a direct-mapped table, one candidate
 * per slot, a skip that gallops after misses, and LZ4-style fixed-width
 * distances. No block header and no state across calls.
 *
 * Wire format (a raw LZ4-style token stream, no header, no framing):
 *
 *   token       high nibble = literal-run length (0..14; 15 => escapes follow)
 *               low nibble  = match length - 4 (0..14; 15 => escapes follow)
 *   escapes     for each field whose nibble is 15: zero or more 0xFF bytes,
 *               each adding 255, then one byte below 255 ending the run
 *   literals    the named run, inline
 *   distance    2-byte little endian; present on every command EXCEPT the
 *               final literal-only one. A zero length field means "no match"
 *               and that command ends the stream after its literals.
 *
 * Distances are absolute block positions back from the current output, so a
 * block is limited to 65536 bytes; call repeatedly for a larger input. The
 * decoder needs the exact expected block length, passed as out_cap.
 *
 * REGISTERED IN THE UNITY BUILD: src/lzfast.c is included by
 * src/zpse_unity.c, so callers need only link that translation unit.
 */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_LZFAST_MIN_MATCH 4u
#define ZPSE_LZFAST_HBITS     13u
#define ZPSE_LZFAST_HSIZE     (1u << ZPSE_LZFAST_HBITS)
#define ZPSE_LZFAST_TABLE_BYTES (ZPSE_LZFAST_HSIZE * (size_t)sizeof(uint16_t))
#define ZPSE_LZFAST_MAX_BLOCK 65536u

/* Worst-case payload size for n input bytes. Every match command covers at
 * least four bytes and costs at most three bytes plus escapes; escapes grow
 * as x/255, so even an all-literals stream fits in n + n/255 + a few bytes.
 * The generous constant also absorbs the encoder's cap checks. Returns 0 for
 * n == 0. */
size_t zpse_lzfast_bound(size_t n);

/* Encode n bytes into out[0..out_cap). n must be <= ZPSE_LZFAST_MAX_BLOCK.
 * Returns the payload length, or 0 on failure (unusable arguments, n too
 * large, or out_cap too small). n == 0 is a valid empty stream: nothing is
 * written and 0 is returned. For incompressible input the stream is a single
 * literal-only command. The hash table is caller-free scratch on the stack,
 * cleared per call, so concurrent calls do not share state. */
size_t zpse_lzfast_encode(const uint8_t *in, size_t n,
                          uint8_t *out, size_t out_cap);

/* Decode exactly out_cap bytes into out. out_cap is the known block length;
 * there is no length field in the stream. Returns ZPSE_OK-equivalent 0 on
 * success, a negative value on any stream that is truncated, malformed, or
 * does not produce exactly out_cap bytes. out_cap must be <= 65536. */
int zpse_lzfast_decode(const uint8_t *in, size_t in_len,
                       uint8_t *out, size_t out_cap);

#endif /* ZPSE_LZFAST_H */
