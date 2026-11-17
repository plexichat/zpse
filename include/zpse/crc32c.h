#ifndef ZPSE_CRC32C_H
#define ZPSE_CRC32C_H

/* CRC32-C (Castagnoli, reflected, poly 0x1EDC6F41).
 * init 0xFFFFFFFF, xorout 0xFFFFFFFF. The portable slice-by-8/slice-by-1
 * fallback and the run-time-dispatched SSE4.2 (crc32 instruction) path live
 * together in src/crc32c.c and are bit-identical: every entry point here
 * returns the same 32 bits whichever path ran, so callers may truncate that
 * value (a page header carries CRC & 0xFFFF) without knowing the path.
 * The software lookup tables are static const (no runtime/lazy init), so
 * results are byte-identical across worker counts and builds.
 * The hardware path can be disabled at build time with -DZPSE_CRC32C_HW=0. */

#include <stddef.h>
#include <stdint.h>

uint32_t zpse_crc32c_init(void); /* returns 0xFFFFFFFF */
uint32_t zpse_crc32c_finish(uint32_t crc); /* xorout */
uint32_t zpse_crc32c_update(uint32_t crc, const uint8_t *data, size_t len);
uint32_t zpse_crc32c(const uint8_t *data, size_t len);

/* Dedicated 14-byte page-header CRC (frame-encoder hot path). Returns the
 * FULL finished CRC; callers mask to 16 bits. Bit-identical to
 * zpse_crc32c(buf, 14) on both the hardware and software backends. */
uint32_t zpse_crc32c_hdr14(const uint8_t *buf);

/* CRC32-C of `len` copies of the byte `b`, in O(log len) matrix applies
 * instead of a pass over the data. Bit-identical to
 * zpse_crc32c() on a buffer of that shape. The decoder uses it when a
 * block's whole output is one byte value (the dense constant form). */
uint32_t zpse_crc32c_repeat(uint8_t b, size_t len);

/* Fused copy + checksum: identical in effect to
 *   memcpy(dst, src, n); return zpse_crc32c(src, n);
 * but the copy and the CRC share one pass over the source, so the
 * checksum's latency is hidden behind the store traffic instead of
 * costing a second read of already-evicted data. Bit-identical to
 * zpse_crc32c() for every n, including 0. */
uint32_t zpse_crc32c_copy(uint8_t *dst, const uint8_t *src, size_t n);

/* 256-entry software table exposed for table-driven paths. */
extern const uint32_t zpse_crc32c_table[256];

#endif /* ZPSE_CRC32C_H */
