#ifndef ZPSE_FLAT_H
#define ZPSE_FLAT_H

/* Position-aligned 4 KiB page wire form (sync 0x50B9).
 *
 * A FLAT page is a 16-byte header followed by a payload whose bytes are
 * stored in OUTPUT POSITION ORDER, so decode is one sequential pass with
 * no scatter and no per-page state. This header records only the constants
 * and the decoder entry point; the header parser/serializer and the encoder
 * live with their respective owners.
 *
 * Wire header at [0..16), payload at [16, c):
 *   [0..2)   u16 sync  = 0x50B9
 *   [2]      u8  flags = RESET | optional LAST (RAW/CRC/HAS_META must be 0)
 *   [3]      u8  codec (ZpseFlatCodec)
 *   [4]      u8  param (meaning depends on codec, see below)
 *   [5]      u8  reserved = 0
 *   [6..8)   u16 n
 *   [8..10)  u16 c
 *   [10..14) u32 seq
 *   [14..16) u16 crc16 of [0..14)
 *
 * Codecs (all store elements in output position order):
 *   0 RAW        param = 0;                 payload = n raw bytes
 *   1 NIBBLE4    param = nalpha-1 (0..15);  payload = u8 pal[16] then
 *                (n+1)/2 bytes of 4-bit indices (low nibble = even position)
 *   2 BIT2       param = nalpha-1 (0..3);   payload = u8 pal[8] then
 *                (n+3)/4 bytes of 2-bit indices (low pair = position 0)
 *   3 SUBLELANE  payload = [0] period in {1,2,4,8}, [1] keepmask,
 *                [2..10) u8 templ[8], [10..] sub[] (k=popcount(keepmask)
 *                bytes per group of `period` output bytes), then 4 zero
 *                pad bytes. Output[g*period+j] = keep bit j ?
 *                sub[g*k + rank(j)] : templ[j].
 */

#include <stdint.h>

/* Build knob: compile the AVX2 kernels (runtime-dispatched) in by default.
 * Set ZPSE_FLAT_AVX2=0 for a scalar-only build; both paths are bit-exact. */
#ifndef ZPSE_FLAT_AVX2
#define ZPSE_FLAT_AVX2 1
#endif

#define ZPSE_FLAT_MAX_N        ((uint32_t)4096u)

/* Header flag bits (byte [2]): RESET = 1<<3 and LAST = 1<<1 are the only
 * legal bits. RAW (0x01), CRC (0x04) and HAS_META (0x10) must all be 0. */

/* Codec ids are defined as ZPSE_FLAT_* macros in zpse/constants.h so the
 * header parser and the decoder agree on one definition. */

#ifdef __cplusplus
extern "C" {
#endif

/* Decode one FLAT payload into out[0..n). `codec` and `param` are the
 * already-parsed header fields; `payload`/`payload_len` describe the bytes
 * after the 16-byte header. n must be in [1,4096].
 *
 * Validates codec <= 3, the codec's parameter range, exact payload length
 * arithmetic, and n bounds. All pointers must be non-NULL for n > 0.
 * Returns ZPSE_OK, or one of ZPSE_BAD_ARG (bad pointer, n, param, or
 * payload length), ZPSE_TRUNCATED (payload shorter than required) or
 * ZPSE_UNSUPPORTED_FEATURE (codec > 3). No heap, no state, deterministic. */
int zpse_flat_decode(uint8_t codec, uint8_t param,
                     const uint8_t *payload, uint32_t payload_len,
                     uint8_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_FLAT_H */
