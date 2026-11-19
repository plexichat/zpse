#ifndef ZPSE_TEXT_H
#define ZPSE_TEXT_H
#include <stdint.h>

/* Finder-free 7-bit ASCII pack for the TEXT block form.
 *
   Encode ASCII input (all bytes <128) by packing eight 7-bit literals into
   seven payload bytes (bit 7*i+k of a group is bit k of input byte i); a
   final group of r<8 bytes occupies r bytes masked to 7r bits. The payload
   length is always n - (n>>3). Returns that length (excluding the 16-byte
   block header), or 0 on failure (non-ASCII input, n==0, or cap exceeded). */
uint32_t zpse_text_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                          uint32_t cap);

/* Unpack a 7-bit payload to exactly n bytes. The only structural constraint
   is in_len == n - (n>>3); every bit pattern is legal payload. Returns 0 on
   success, negative on a length mismatch or a truncated final group. */
int zpse_text_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                     uint32_t n);

/* Exact packed payload length, n - (n>>3), for an eligible ASCII input;
   returns 0 if ineligible (any byte >=128 or n==0). For every eligible input
   this equals zpse_text_encode's return value. */
uint32_t zpse_text_price(const uint8_t *in, uint32_t n);

#endif /* ZPSE_TEXT_H */
