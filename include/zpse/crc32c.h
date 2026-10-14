#ifndef ZPSE_CRC32C_H
#define ZPSE_CRC32C_H

/* CRC32-C (Castagnoli, reflected, poly 0x1EDC6F41).
 * init 0xFFFFFFFF, xorout 0xFFFFFFFF. Slice-by-1 software fallback;
 * SIMD/hardware variants live in vector.h and MUST be bit-identical. */

#include <stddef.h>
#include <stdint.h>

uint32_t zpse_crc32c_init(void); /* returns 0xFFFFFFFF */
uint32_t zpse_crc32c_finish(uint32_t crc); /* xorout */
uint32_t zpse_crc32c_update(uint32_t crc, const uint8_t *data, size_t len);
uint32_t zpse_crc32c(const uint8_t *data, size_t len);

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
