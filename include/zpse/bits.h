#ifndef ZPSE_BITS_H
#define ZPSE_BITS_H

/* Little-endian load/store + bit helpers. Portable C11, no UB on
 * unaligned access (byte-wise). Hot paths may use memcpy-based
 * variants in vector.h instead. */

#include <stdint.h>
#include <string.h>

static inline uint16_t zpse_load_u16le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t zpse_load_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t zpse_load_u64le(const uint8_t *p) {
    uint64_t lo = zpse_load_u32le(p);
    uint64_t hi = zpse_load_u32le(p + 4);
    return lo | (hi << 32);
}

static inline void zpse_store_u16le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}

static inline void zpse_store_u32le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static inline void zpse_store_u64le(uint8_t *p, uint64_t v) {
    zpse_store_u32le(p, (uint32_t)v);
    zpse_store_u32le(p + 4, (uint32_t)(v >> 32));
}

static inline uint32_t zpse_ceil_div_u32(uint32_t n, uint32_t d) {
    return (n + d - 1u) / d;
}

/* ceil(log2(x)) for x >= 1, small values (palette widths). */
static inline uint32_t zpse_ceil_log2_u32(uint32_t x) {
    uint32_t b = 0;
    uint32_t v = 1;
    while (v < x) { v <<= 1; b++; }
    return b;
}

static inline uint32_t zpse_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

/* The same for a 64-bit word, which is the width the match finder compares
 * in: the lowest set bit of an xor is the first byte that differs. */
static inline uint32_t zpse_ctz64(uint64_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctzll(w);
#else
    uint32_t c = 0;
    while ((w & 1u) == 0u) {
        w >>= 1;
        c++;
    }
    return c;
#endif
}

/* Index of the lowest set bit. `w` MUST be non-zero (the callers walk a
 * bitmask with `w &= w - 1`, so every iteration has one). */
static inline uint32_t zpse_ctz32(uint32_t w) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctz(w);
#else
    uint32_t c = 0;
    while ((w & 1u) == 0u) {
        w >>= 1;
        c++;
    }
    return c;
#endif
}

#endif /* ZPSE_BITS_H */
