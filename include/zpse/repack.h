#ifndef ZPSE_REPACK_H
#define ZPSE_REPACK_H

#include <stddef.h>
#include <stdint.h>

/*
 * Fixed-width ASCII repack -- a whole-buffer, single-pass fast path.
 *
 * The encoder detects the alphabet width from a 64 KiB sample (distinct
 * low-7-bit symbols -> w = ceil(log2(distinct)), 1 <= w <= 7), maps the
 * symbols to dense w-bit codes, and packs groups of 8 codes into w bytes.
 * With w == 7 the map is the identity and the payload is byte-for-byte the
 * same 7-bit pack FLAT7 produces.  Non-ASCII input is declined.
 *
 * Wire format (all integers little-endian; "crc" covers bytes [0..24) and
 * the symbol map, with the crc field itself excluded):
 *
 *   [0..4)    magic = "ZRPK"  (0x4B50525A little-endian)
 *   [4]       version = 1
 *   [5]       w                (1..7)
 *   [6..8)    flags u16        (bit0 = MAP16: 16-byte map in use)
 *   [8..16)   n u64            (original length; frame is self-delimiting:
 *                               frame_len = payload_off + payload_len)
 *   [16..20)  payload_len u32  (packed payload byte count)
 *   [20..24)  distinct u32     (number of mapped symbols)
 *   [24..28)  header_crc32 u32 (CRC-32/IEEE over [0..24) + the map)
 *   [28..]    symbol map: inverse map code -> symbol, 16 bytes when w<=4
 *                         (MAP16 set), otherwise 128 bytes
 *   [payload_off..]  packed payload, payload_len bytes
 *
 * The map is stored at a fixed offset 28; payload_off is 44 (MAP16) or 156.
 * The decoder validates magic/version/flags/width/lengths, verifies the
 * header CRC, checks out_cap against n, then unpacks and expands via the map.
 *
 * Return convention: encode returns 1 when a frame was committed (*out_len
 * is set), 0 when it declined (non-ASCII, too-small cap, width overflow).
 * decode returns 1 on success, 0 on any malformed/truncated frame or when
 * out_cap < n.  On decline neither function writes past out_cap, and encode
 * leaves *out_len untouched.
 *
 * NOTE: this codec is REGISTERED IN THE UNITY BUILD: src/repack.c is included
 * by src/zpse_unity.c, so callers need only link that translation unit.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound on the encoded frame size for n input bytes. */
size_t zpse_repack_bound(size_t n);

/* Encode n bytes at in into out (capacity out_cap). 1 committed, 0 declined. */
int zpse_repack_encode(const uint8_t *in, size_t n, uint8_t *out,
                       size_t out_cap, size_t *out_len);

/* Decode a frame at in (in_len bytes) into out (capacity out_cap).
 * 1 on success, 0 on a malformed/truncated frame or insufficient out_cap. */
int zpse_repack_decode(const uint8_t *in, size_t in_len, uint8_t *out,
                       size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_REPACK_H */
