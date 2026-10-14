#ifndef ZPSE_COMBINADIC_H
#define ZPSE_COMBINADIC_H

/* Macaulay/colex combinadics (SPEC Sec. 8.1-8.2). 256-bit ranks as
 * 4x u64 limbs, little-endian limb order. No division in hot paths;
 * ranking/unranking use u256 add/sub/compare only. */

#include <stddef.h>
#include <stdint.h>

typedef struct ZpseU256 { uint64_t w[4]; } ZpseU256;

int zpse_u256_gte(ZpseU256 a, ZpseU256 b); /* boolean */
ZpseU256 zpse_u256_add(ZpseU256 a, ZpseU256 b);
ZpseU256 zpse_u256_sub(ZpseU256 a, ZpseU256 b);
int zpse_u256_is_zero(ZpseU256 a);

/* Minimal payload bytes L = ceil(log2(C(nt,k))/8). Returns L (0..32)
 * or negative status. Uses 64-bit arithmetic with saturation: any
 * C(nt,k) >= 2^64 reports via *wide_out=1 and computes L via bit-length
 * of the exact big value using the scratch row builder. */
int zpse_comb_payload_len(uint16_t nt, uint16_t k, int *wide_out);

/* Exact C(n,k) as u256 (0 when n<k). Single value via multiplicative
 * recurrence with exact division — used OUTSIDE hot loops only
 * (length precomputation, range pre-check). */
ZpseU256 zpse_comb_choose_u256(uint16_t n, uint16_t k);

/* Rank K ascending positions c[0..k-1] -> u256 rank. */
ZpseU256 zpse_comb_rank(const uint16_t *c, uint16_t k);

/* Unrank: payload (payload_len LE bytes) -> dense words[8] (zeroed by
 * caller). Enforces rank < C(nt,k) pre-check + zero remainder.
 * comb_scratch: 8 KiB arena region (row workspace). */
int zpse_comb_unrank(uint16_t nt, uint16_t k, const uint8_t *payload,
                     uint16_t payload_len, uint32_t words[8],
                     uint8_t *comb_scratch);

/* Rank + serialize: positions -> payload bytes. Returns bytes written. */
int zpse_comb_rank_to_bytes(const uint16_t *c, uint16_t k, uint16_t nt,
                            uint8_t *payload_out, uint16_t payload_cap);

#endif /* ZPSE_COMBINADIC_H */
