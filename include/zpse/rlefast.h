#ifndef ZPSE_RLEFAST_H
#define ZPSE_RLEFAST_H

/* RLE fast path for run-structured small-integer sequences.
 *
 * This is a port of DTYPE1's measured prototype at
 *   C:\Users\dboyn\AppData\Local\Temp\opencode\dtype1\bench_proto.c
 *   (enc_rle / dec_rle, "run-length prototype for small-integer sequences").
 * The prototype measured, on 8 MiB of generated small-integer data in memory
 * (single core, best-of-7), relative to the normal encoder:
 *
 *   ratio 8.241:1 -> 8.745:1 (+6.1%), encode 6.16x, decode 2.54x
 *
 * The prototype's decoder was a plain (value, run-1) loop with a 256-byte
 * run cap and no gate, no header CRC and no capacity checks. Everything in
 * this header beyond that loop -- the gate, the self-delimiting header, the
 * header CRC, the escape for runs longer than 255 and the bounds checking --
 * is the hardening needed to make the idea callable from the library.
 *
 * GATE. zpse_rlefast_encode inspects the first min(n, 65536) bytes and counts
 * the bytes that belong to a maximal run of at least four identical bytes
 * whose value is in 0..127 (narrow). It commits only if at least 90% of the
 * sampled bytes qualify; otherwise it returns 0 and writes nothing, and the
 * caller falls back to the normal encoder. n == 0 commits an empty stream
 * (the 90% condition is vacuous). Data with fewer than four bytes cannot
 * contain a qualifying run and is therefore declined.
 *
 * WIRE FORMAT (self-delimiting; all multi-byte fields little-endian):
 *
 *   offset  size  field
 *   0       4     magic = 0x31454C52, the four ASCII bytes "RLE1"
 *   4       4     n = exact uncompressed length in bytes
 *   8       4     CRC-32C (reflected, poly 0x82F63B78, init/final 0xFFFFFFFF)
 *                 of the eight header bytes at offsets 0..7
 *   12      ...   (value, run-1) pairs, described below
 *
 *   A pair is two bytes: the repeated byte value, then the run-length byte.
 *   run-1 in 0..254 means "this many+1 bytes of value, and the run ends
 *   here". run-1 == 255 is an escape meaning "255 bytes of value, and the
 *   next pair continues the SAME run" (the next pair must carry the same
 *   value). The encoder always terminates a run with a non-escape pair, so
 *   a run of length L is emitted as floor((L-1)/255) escapes plus one final
 *   pair; a run of length 256 is (v,255) followed by (v,0). Adjacent runs
 *   with the same value are merged by the encoder, so a decoder may reject
 *   a continuation pair whose value differs. The header's n is the stop
 *   condition; the decoder requires exactly n bytes of pairs and rejects a
 *   stream with trailing bytes, a truncated header/payload, a header whose
 *   CRC does not match, a length field that disagrees with the caller's n,
 *   or a run that would write past out[0..n).
 *
 *   Values are NOT restricted to 0..127 on the wire: the gate only decides
 *   whether a block is worth attempting, and the up-to-10% of bytes outside
 *   long narrow runs may be any byte value. This is why the format does not
 *   steal the value byte's high bit for the escape.
 *
 * No allocation, no state across calls. src/rlefast.c is self-contained: it
 * ships its own little-endian load/store and CRC-32C helpers and links no
 * other zpse translation unit.
 *
 * REGISTERED IN THE UNITY BUILD: src/rlefast.c is included by
 * src/zpse_unity.c, so callers need only link that translation unit.
 */

#include <stdint.h>

#define ZPSE_RLEFAST_MAGIC       0x31454C52u /* "RLE1" as four ASCII bytes */
#define ZPSE_RLEFAST_HEADER_SIZE 12u
#define ZPSE_RLEFAST_SAMPLE      65536u
#define ZPSE_RLEFAST_MIN_RUN     4u
#define ZPSE_RLEFAST_ESCAPE      255u        /* run-1 byte: continuation */

/* Upper bound on the encoded size of any n-byte input this encoder accepts:
 * 12-byte header plus at most one 2-byte pair per input byte (a run of L
 * bytes costs 2*ceil(L/255) <= 2*L bytes, escapes included). Saturates at
 * UINT32_MAX. Callers may use this to size out for zpse_rlefast_encode. */
uint32_t zpse_rlefast_bound(uint32_t n);

/* Try to encode in[0..n) into out[0..out_cap).
 *
 * Returns 1 on commit: *out_len receives the exact stream length, which is
 * <= out_cap. Returns 0 on decline: the input failed the gate, an argument
 * is unusable (out or out_len NULL, in NULL with n > 0), n/out_cap cannot be
 * represented, or out_cap is smaller than zpse_rlefast_bound(n). On decline
 * nothing is written and *out_len is set to 0. The caller then uses the
 * normal encoder. */
int zpse_rlefast_encode(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap, uint32_t *out_len);

/* Decode a stream produced by zpse_rlefast_encode into out[0..n). in_len
 * must equal the stream length exactly (see the trailing-byte rule above).
 *
 * Returns 0 on success, a negative value on any malformed input. The
 * decoder never writes outside out[0..n). */
int zpse_rlefast_decode(const uint8_t *in, uint32_t in_len,
                        uint8_t *out, uint32_t n);

#endif /* ZPSE_RLEFAST_H */
