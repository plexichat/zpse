#ifndef ZPSE_FLAT7_H
#define ZPSE_FLAT7_H
#include <stdint.h>
#include <stddef.h>
/* Whole-ASCII fast frame: 40-byte header + 7-bit-packed payload.
   Returns ZPSE_OK / negative status. */
int zpse_flat7_should(const uint8_t *in, size_t n, uint32_t enc_flags, int indexed, size_t out_cap);
int zpse_flat7_is_frame(const uint8_t *in, size_t in_len);
int zpse_flat7_encode(const uint8_t *in, size_t n, uint8_t *out, size_t out_cap, size_t *out_len_out);
/* Fused single-pass variant: ASCII check + redundancy gate + 7-bit pack over
   a resident whole-input buffer, committing the 40-byte header only after
   every gate passes. Returns 1 committed (*out_len set), 0 declined (no
   frame written, *out_len untouched), or a negative ZPSE status on error.
   out_cap must leave 8 bytes of scratch past the packed frame, as for
   zpse_flat7_encode. The caller has already excluded indexed/checksummed
   input; this entry point assumes the basic size gate has passed. */
int zpse_flat7_encode_stream(const uint8_t *in, size_t n, uint8_t *out,
                             size_t out_cap, size_t *out_len);
int zpse_flat7_decode(const uint8_t *in, size_t in_len, uint8_t *out, size_t out_cap, size_t *out_len_out);
int zpse_flat7_header_check(const uint8_t *in, size_t in_len, size_t *out_n, size_t *out_payload_off);
#endif
