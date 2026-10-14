#ifndef ZPSE_API_H
#define ZPSE_API_H

/* High-level one-shot API. Contexts borrow caller-owned arenas
 * (256 KiB decode / 384 KiB encode, 64-B aligned). No allocation,
 * no threads, no I/O inside. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *zpse_version_string(void); /* "1.0.0" */

/* One-shot block calls (SPEC Sec. 4, 11). */
int zpse_compress_block(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap,
                        uint32_t *out_len_out,
                        uint32_t seq, int is_last,
                        void *enc_arena_384k);

int zpse_decompress_block(const uint8_t *in, uint32_t in_len,
                          uint8_t *output, uint32_t out_cap,
                          uint32_t *out_len_out,
                          void *dec_arena_256k);

/* One-shot frame calls (SPEC Sec. 3). */
int zpse_compress_frame(const uint8_t *in, size_t n,
                        uint8_t *out, size_t out_cap,
                        size_t *out_len_out,
                        uint32_t nominal_block_size,
                        void *enc_arena_384k);

int zpse_decompress_frame(const uint8_t *in, size_t in_len,
                          uint8_t *output, size_t out_cap,
                          size_t *out_len_out,
                          void *dec_arena_256k);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_API_H */
