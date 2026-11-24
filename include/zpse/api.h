#ifndef ZPSE_API_H
#define ZPSE_API_H

/* High-level one-shot API. Contexts borrow caller-owned arenas
 * (384 KiB decode / 464 KiB encode, 64-B aligned). No allocation,
 * no threads, no I/O inside. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"
#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The library version, exactly ZPSE_VERSION_STRING (include/zpse/zpse.h is
 * the one place it is written down). */
const char *zpse_version_string(void);

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
                        void *enc_arena);

/* Same, with encoder options (ZPSE_ENC_*): ZPSE_ENC_CHECKSUM tags each
 * block with the CRC32-C of its decoded bytes, which every decoder here
 * then verifies. The effort level trades compressed size for encode time:
 * The default is level 6 (ZPSE_ENC_SPEED_L6), levels 1-4 buy
 * trials one at a time and never run the match finder, and levels 5-9 also
 * offer the intra-block LZ4 form as one more candidate -- one screen and one
 * decode bill shared by the whole tier, since LZ4 writes the same stream at
 * every level. The deep rungs differ only in their sweep trial cap and
 * tie_field cut, so adjacent rungs can tie. Levels run 1 = FASTEST ...
 * 9 = MAX (see include/zpse/constants.h). No level changes the format:
 * every level's stream is read by the same decoder, and a block is only
 * ever written in the form that came out smallest. */
int zpse_compress_block_ex(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap,
                           uint32_t *out_len_out,
                           uint32_t seq, int is_last, uint32_t enc_flags,
                           void *enc_arena);

int zpse_decompress_block(const uint8_t *in, uint32_t in_len,
                          uint8_t *output, uint32_t out_cap,
                          uint32_t *out_len_out,
                          void *dec_arena);

/* Same, with decoder hints (ZPSE_DEC_*). A hint is a property of the call,
 * not of the stream: compressed and decoded bytes are identical with and
 * without it. Hints outside ZPSE_DEC_VALID_MASK are refused. */
int zpse_decompress_block_ex(const uint8_t *in, uint32_t in_len,
                             uint8_t *output, uint32_t out_cap,
                             uint32_t *out_len_out,
                             uint32_t dec_flags, void *dec_arena);

/* The one-shot frame calls below always emit a normal PSE1 frame; they never
 * emit FLAT7 ("F7P1" magic), including the indexed form. FLAT7 frames are
 * produced only by the CLI's fast path for whole-ASCII inputs or by explicit
 * zpse_flat7_encode calls. zpse_decompress_frame decodes both normal and
 * FLAT7 frames transparently. zpse_seek_frame returns ZPSE_UNSUPPORTED_FEATURE
 * for FLAT7, which has no block index. */
/* One-shot frame calls. */
int zpse_compress_frame(const uint8_t *in, size_t n,
                        uint8_t *out, size_t out_cap,
                        size_t *out_len_out,
                        uint32_t nominal_block_size,
                        void *enc_arena);

/* Same, with encoder options (ZPSE_ENC_*), applied to every block. */
int zpse_compress_frame_ex(const uint8_t *in, size_t n,
                           uint8_t *out, size_t out_cap,
                           size_t *out_len_out,
                           uint32_t nominal_block_size, uint32_t enc_flags,
                           void *enc_arena);

int zpse_decompress_frame(const uint8_t *in, size_t in_len,
                          uint8_t *output, size_t out_cap,
                          size_t *out_len_out,
                          void *dec_arena);

/* Same, with decoder hints (ZPSE_DEC_*). The hint never changes the stream. */
int zpse_decompress_frame_ex(const uint8_t *in, size_t in_len,
                             uint8_t *output, size_t out_cap,
                             size_t *out_len_out,
                             uint32_t dec_flags, void *dec_arena);

/* Same as zpse_compress_frame_ex, with the FrameIndex appended so the
 * result can be read block by block in any order (zpse_frame_seek). The
 * index costs 16 + 8 bytes per block. */
int zpse_compress_frame_indexed(const uint8_t *in, size_t n,
                                uint8_t *out, size_t out_cap,
                                size_t *out_len_out,
                                uint32_t nominal_block_size,
                                uint32_t enc_flags,
                                void *enc_arena);

/* Decode one block of an indexed frame without touching the others.
 * `output` must hold one block (at most nominal_block_size, 64 KiB). */
int zpse_seek_frame(const uint8_t *in, size_t in_len, uint32_t block_index,
                    uint8_t *output, uint32_t out_cap, uint32_t *out_len_out,
                    void *dec_arena);

/* Same, with decoder hints (ZPSE_DEC_*). The hint never changes the stream. */
int zpse_seek_frame_ex(const uint8_t *in, size_t in_len, uint32_t block_index,
                       uint8_t *output, uint32_t out_cap,
                       uint32_t *out_len_out,
                       uint32_t dec_flags, void *dec_arena);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_API_H */
