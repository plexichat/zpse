#include "zpse/zpse.h" /* the version macros: the string is built from them */

#include "zpse/api.h"

#include "zpse/arena.h"
#include "zpse/block.h"
#include "zpse/frame.h"
#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>

const char *zpse_version_string(void)
{
    return ZPSE_VERSION_STRING;
}

/* Encoder options are ZPSE_ENC_CHECKSUM plus one of the nine levels in bits
 * 1-4, and an empty level field means the default level. Anything else is a
 * bad argument rather than a silent round to a neighbour: the level is the
 * caller's own speed/size trade, so a codec that cannot honour it must say
 * so. Bits outside the two fields are refused for the same reason. */
static int enc_flags_ok(uint32_t enc_flags)
{
    uint32_t lvl;

    if ((enc_flags & ~(uint32_t)(ZPSE_ENC_CHECKSUM | ZPSE_ENC_SPEED_MASK)) !=
        0u) {
        return 0;
    }
    lvl = (uint32_t)((enc_flags & ZPSE_ENC_SPEED_MASK) >>
                     ZPSE_ENC_SPEED_SHIFT);
    return lvl <= 9u;
}

int zpse_compress_block(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap,
                        uint32_t *out_len_out,
                        uint32_t seq, int is_last,
                        void *enc_arena_448k)
{
    /* The default stream carries no optional fields: no payload CRC. */
    return zpse_compress_block_ex(in, n, out, out_cap, out_len_out, seq,
                                  is_last, 0u, enc_arena_448k);
}

int zpse_compress_block_ex(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap,
                           uint32_t *out_len_out,
                           uint32_t seq, int is_last, uint32_t enc_flags,
                           void *enc_arena_448k)
{
    ZpseEncoder enc;
    int rc;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arena_448k == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0 || n > 65536u) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (!enc_flags_ok(enc_flags)) {
        return PSE_BAD_ARG;
    }
    rc = zpse_encoder_init(&enc, enc_arena_448k, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_block_encode(&enc, in, n, seq, is_last, out, out_cap);
    if (rc < 0) {
        return rc;
    }
    *out_len_out = (uint32_t)rc;
    return PSE_OK;
}

int zpse_decompress_block(const uint8_t *in, uint32_t in_len,
                          uint8_t *output, uint32_t out_cap,
                          uint32_t *out_len_out,
                          void *dec_arena_320k)
{
    ZpseDecoder dec;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arena_320k == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len < 32u) {
        return PSE_TRUNCATED;
    }
    rc = zpse_decoder_init(&dec, dec_arena_320k, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_block_decode(&dec, in, in_len, output, out_cap, out_len_out);
    if (rc < 0) {
        return rc;
    }
    return PSE_OK;
}

int zpse_compress_frame(const uint8_t *in, size_t n,
                        uint8_t *out, size_t out_cap,
                        size_t *out_len_out,
                        uint32_t nominal_block_size,
                        void *enc_arena_448k)
{
    /* The default stream carries no optional fields: no payload CRC. */
    return zpse_compress_frame_ex(in, n, out, out_cap, out_len_out,
                                  nominal_block_size, 0u, enc_arena_448k);
}

int zpse_compress_frame_ex(const uint8_t *in, size_t n,
                           uint8_t *out, size_t out_cap,
                           size_t *out_len_out,
                           uint32_t nominal_block_size, uint32_t enc_flags,
                           void *enc_arena_448k)
{
    ZpseEncoder enc;
    int rc;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arena_448k == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0) {
        return PSE_BAD_ARG;
    }
    if (nominal_block_size == 0 || nominal_block_size > 65536u) {
        return PSE_BAD_ARG;
    }
    if (!enc_flags_ok(enc_flags)) {
        return PSE_BAD_ARG;
    }
    rc = zpse_encoder_init(&enc, enc_arena_448k, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_frame_encode(&enc, in, n, nominal_block_size,
                           out, out_cap, out_len_out);
    if (rc < 0) {
        return rc;
    }
    return PSE_OK;
}

int zpse_compress_frame_indexed(const uint8_t *in, size_t n,
                                uint8_t *out, size_t out_cap,
                                size_t *out_len_out,
                                uint32_t nominal_block_size,
                                uint32_t enc_flags,
                                void *enc_arena_448k)
{
    ZpseEncoder enc;
    int rc;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arena_448k == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0) {
        return PSE_BAD_ARG;
    }
    if (nominal_block_size == 0 || nominal_block_size > 65536u) {
        return PSE_BAD_ARG;
    }
    if (!enc_flags_ok(enc_flags)) {
        return PSE_BAD_ARG;
    }
    rc = zpse_encoder_init(&enc, enc_arena_448k, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_frame_encode_ex(&enc, in, n, nominal_block_size, out, out_cap,
                              out_len_out, 1);
    if (rc < 0) {
        return rc;
    }
    return PSE_OK;
}

int zpse_seek_frame(const uint8_t *in, size_t in_len, uint32_t block_index,
                    uint8_t *output, uint32_t out_cap, uint32_t *out_len_out,
                    void *dec_arena_320k)
{
    ZpseDecoder dec;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arena_320k == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len < 32u) {
        return PSE_TRUNCATED;
    }
    rc = zpse_decoder_init(&dec, dec_arena_320k, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    return zpse_frame_seek(&dec, in, in_len, block_index, output, out_cap,
                           out_len_out);
}

int zpse_decompress_frame(const uint8_t *in, size_t in_len,
                          uint8_t *output, size_t out_cap,
                          size_t *out_len_out,
                          void *dec_arena_320k)
{
    ZpseDecoder dec;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arena_320k == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len == 0) {
        return PSE_BAD_ARG;
    }
    if (in_len < 32u) {
        return PSE_TRUNCATED;
    }
    rc = zpse_decoder_init(&dec, dec_arena_320k, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_frame_decode(&dec, in, in_len, output, out_cap, out_len_out);
    if (rc < 0) {
        return rc;
    }
    return PSE_OK;
}
