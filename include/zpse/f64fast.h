#ifndef ZPSE_F64FAST_H
#define ZPSE_F64FAST_H

/* f64fast -- a pinned DELTA_TRANSPOSE_8 fast path for float64 arrays.
 *
 * The normal encoder spends its time sweeping transform candidates and then
 * wrapping the winner in an optional LZ_TILE pass. DTYPE1's in-memory
 * measurement (8 MiB deterministic float64, single core, best of 7) found
 * that pinning transform 9 (ZPSE_T_DELTA_TRANSPOSE_8) directly, with no swap
 * and no LZ_TILE wrap, is enough: 2.29-2.51x encode and 1.29-1.42x decode for
 * +2.8% ratio (3.993 -> 4.104:1). The inner payload of every committed block
 * is a *normal* PSE1 frame, so the heavy lifting on decode is done by the
 * unmodified library's zpse_decompress_frame().
 *
 * REGISTERED IN THE UNITY BUILD: src/f64fast.c is included by
 * src/zpse_unity.c and built as part of the library. It calls only public
 * zpse_* entry points, so it also works as a standalone translation unit.
 *
 * Input shape gate. A byte buffer has no dtype tag, so the gate admits only
 * what the measured corpus looked like: n a nonzero multiple of 8, and the
 * high byte of each 8-byte lane (IEEE-754 sign+exponent byte 7) drawn from a
 * small alphabet with one value strongly dominant. Random data fails both
 * tests and is declined. A decline is a normal outcome: callers fall back to
 * the ordinary encoder.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return values for both encode and decode. No negative error codes: every
 * refusal is a decline, with *out_len left at 0 on the encode side. */
#define ZPSE_F64FAST_DECLINE 0
#define ZPSE_F64FAST_COMMIT  1

/* Container constants. */
#define ZPSE_F64FAST_MAGIC        UINT32_C(0x46343646) /* bytes "F64F" */
#define ZPSE_F64FAST_VERSION      UINT8_C(1)
#define ZPSE_F64FAST_HEADER_SIZE  UINT32_C(24)
#define ZPSE_F64FAST_TID_PINNED   UINT8_C(9)  /* ZPSE_T_DELTA_TRANSPOSE_8 */
#define ZPSE_F64FAST_TID_RAW      UINT8_C(0)  /* ZPSE_T_RAW_BYTES */
#define ZPSE_F64FAST_FLAG_RAW     UINT8_C(0x1)
#define ZPSE_F64FAST_FLAG_PINNED  UINT8_C(0x2)
#define ZPSE_F64FAST_BLOCK_SIZE   UINT32_C(65536)

/* WIRE FORMAT -- self-delimiting, all integers little-endian.
 *
 *   offset  size  field
 *   0       4     magic = "F64F" (u32 LE 0x46343646)
 *   4       1     format version (1)
 *   5       1     flags: bit0 RAW, bit1 PINNED, all other bits zero
 *   6       2     reserved, zero
 *   8       4     n, byte length of the original float64 array (multiple of 8)
 *   12      4     transform id: 9 for a pinned frame, 0 for a raw payload
 *   16      4     payload_len: bytes of payload following this header
 *   20      4     CRC32-C over bytes [0,20) followed by the payload
 *   24      ..    payload
 *
 * Total length is always ZPSE_F64FAST_HEADER_SIZE + payload_len. A PINNED
 * payload is byte-for-byte a normal PSE1 frame (32-byte frame header plus
 * one or more 64 KiB long blocks, each with transform 9, eight seed bytes of
 * transform metadata, one segment per plane, no LZ_TILE wrap); the
 * unmodified decoder accepts it. A RAW payload is the input verbatim and is
 * supported for readers that choose to store a declined block themselves;
 * the encoder here never emits one, it declines instead.
 *
 * Decoding validates magic, version, flags, reserved bytes, n, transform id,
 * payload length and the CRC before calling the library, and refuses a
 * truncated or corrupt header.
 */

/* Upper bound on the encoded length of an n-byte block. Includes the
 * envelope header and the worst-case segment expansion of the inner frame.
 * Returns 0 on overflow. */
uint32_t zpse_f64fast_bound(uint32_t n);

/* Encode one in-memory block.
 *  in       n bytes of input (n a nonzero multiple of 8).
 *  out      caller buffer, at least zpse_f64fast_bound(n) bytes.
 *  out_cap  capacity of out.
 *  out_len  receives the committed length, or 0 on decline.
 * Returns ZPSE_F64FAST_COMMIT (1) when a payload was written, else
 * ZPSE_F64FAST_DECLINE (0): bad shape, no output room, no arena, or the
 * pinned frame is not smaller than the raw envelope would be. No bytes of
 * out beyond *out_len are touched on commit. Callers must ignore out
 * whenever *out_len is 0: the shape/size gates refuse before the first
 * store, while the final raw-envelope check can refuse after the pinned
 * frame has been staged there. */
int zpse_f64fast_encode(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap, uint32_t *out_len);

/* Decode a committed container into exactly n bytes.
 * Returns ZPSE_F64FAST_COMMIT (1) on success, ZPSE_F64FAST_DECLINE (0) on a
 * malformed/truncated/corrupt container or a decode failure. */
int zpse_f64fast_decode(const uint8_t *in, uint32_t in_len,
                        uint8_t *out, uint32_t n);

/* Free this thread's cached encoder/decoder arenas. Call this before a thread
 * exits: the arenas are lazily allocated and reused for the life of the
 * thread, and no TLS destructor is registered, so an exiting thread that
 * skips this leaks its arenas until process exit. */
void zpse_f64fast_release(void);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_F64FAST_H */
