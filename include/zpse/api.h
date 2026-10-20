#ifndef ZPSE_API_H
#define ZPSE_API_H

/* High-level one-shot API. Contexts borrow caller-owned arenas
 * (256 KiB decode / 384 KiB encode, 64-B aligned). No allocation,
 * no threads, no I/O inside. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *zpse_version_string(void); /* "1.0.0" */

/* One-shot block calls.
 *
 * The plain calls emit no optional fields: in particular no payload
 * CRC32-C (ZPSE_ENC_CHECKSUM is off), which is the LZ4-style default --
 * a checksum nothing asked for should not cost either direction a pass
 * over the block. Pass the flag to the _ex form to have it emitted and
 * verified. */
int zpse_compress_block(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap,
                        uint32_t *out_len_out,
                        uint32_t seq, int is_last,
                        void *enc_arena_384k);

/* Same, with encoder options (ZPSE_ENC_*): ZPSE_ENC_CHECKSUM tags each
 * block with the CRC32-C of its decoded bytes, which every decoder here
 * then verifies. */
int zpse_compress_block_ex(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap,
                           uint32_t *out_len_out,
                           uint32_t seq, int is_last, uint32_t enc_flags,
                           void *enc_arena_384k);

int zpse_decompress_block(const uint8_t *in, uint32_t in_len,
                          uint8_t *output, uint32_t out_cap,
                          uint32_t *out_len_out,
                          void *dec_arena_256k);

/* One-shot frame calls. */
int zpse_compress_frame(const uint8_t *in, size_t n,
                        uint8_t *out, size_t out_cap,
                        size_t *out_len_out,
                        uint32_t nominal_block_size,
                        void *enc_arena_384k);

/* Same, with encoder options (ZPSE_ENC_*), applied to every block. */
int zpse_compress_frame_ex(const uint8_t *in, size_t n,
                           uint8_t *out, size_t out_cap,
                           size_t *out_len_out,
                           uint32_t nominal_block_size, uint32_t enc_flags,
                           void *enc_arena_384k);

int zpse_decompress_frame(const uint8_t *in, size_t in_len,
                          uint8_t *output, size_t out_cap,
                          size_t *out_len_out,
                          void *dec_arena_256k);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_API_H */
