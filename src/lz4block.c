/* Intra-block LZ: the single raw-LZ4 block form, backed by vendored LZ4
 * 1.10.0. This file replaces src/lz.c, src/lz2.c, src/lzh.c and src/lzfast.c.
 *
 * Every LZ payload has one shape:
 *
 *   u32 uncompressed_bytes   (kept for the block layer's accounting)
 *   u32 command_count        (LZ4 sequences counted from the stream)
 *   raw LZ4 block            output of LZ4_compress_default
 *
 * The old fast (ZPSE_B_LZ2) and hexadecimal (ZPSE_B_LZ_HEX) selectors are
 * gone, along with their codecs: there is one encode and one decode path,
 * through LZ4_compress_default / LZ4_decompress_safe with the block's
 * declared output size. No custom token loop, match finder, repeat cache
 * or Huffman-trained LZ format is left in the codec.
 *
 * LZ4's raw-block window is 64 KiB by definition, which is exactly the
 * block cap here, so a block is self-contained: no history, no frame and
 * no cross-block reference is possible.
 */

#include "zpse/lz.h"
#include "zpse/bits.h"
#include "zpse/status.h"

#include "vendor/lz4.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ZPSE_LZ4_HDR 8u

/* Count the sequences in a raw LZ4 block. The block layer uses this as the
 * command count in its "bytes per command" decode-cost gate; it is derived
 * from the stream we just wrote, so it is exact for our own output. The
 * walk stops at the last five bytes, which LZ4 reserves for trailing
 * literals. A malformed stream cannot reach this function (only encoded
 * output is counted), but the bounds checks keep it safe anyway.
 *
 * `payload` is the raw block itself, without the [u32 n][u32 cmds] header.
 * Encoders only ever write a nonzero placeholder in the count word; the
 * caller that actually needs the count walks the stream once and writes the
 * real value back before committing it. */
uint32_t zpse_lz_cmd_count(const uint8_t *payload, uint32_t len)
{
    const uint8_t *p;
    const uint8_t *end;
    const uint8_t *last;
    uint32_t cmds = 0u;

    if (len <= 5u) {
        return 1u;
    }
    p = payload;
    end = p + len;
    last = end - 5u;
    while (p < last) {
        uint32_t ll;
        uint32_t ml;
        uint8_t tok = *p++;

        ll = (uint32_t)tok >> 4;
        if (ll == 15u) {
            uint8_t b;
            do {
                if (p >= end) {
                    return cmds ? cmds : 1u;
                }
                b = *p++;
                ll += (uint32_t)b;
            } while (b == 255u);
        }
        if (ll > (uint32_t)(end - p)) {
            return cmds ? cmds : 1u;
        }
        p += ll;
        if (p >= last) {
            break;
        }
        if ((size_t)(end - p) < 2u) {
            break;
        }
        p += 2u; /* little-endian match distance */
        ml = (uint32_t)(tok & 15u);
        if (ml == 15u) {
            uint8_t b;
            do {
                if (p >= end) {
                    return cmds ? cmds : 1u;
                }
                b = *p++;
                ml += (uint32_t)b;
            } while (b == 255u);
        }
        cmds++;
    }
    return cmds ? cmds : 1u;
}

static uint32_t lz4_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                           uint32_t cap)
{
    int written;

    if (in == 0 || dst == 0 || n == 0u ||
        cap <= ZPSE_LZ4_HDR || n > (uint32_t)INT32_MAX ||
        cap - ZPSE_LZ4_HDR > (uint32_t)INT32_MAX) {
        return 0u;
    }
    written = LZ4_compress_default((const char *)in,
                                   (char *)(dst + ZPSE_LZ4_HDR),
                                   (int)n, (int)(cap - ZPSE_LZ4_HDR));
    if (written <= 0) {
        return 0u;
    }
    zpse_store_u32le(dst, n);
    /* Nonzero placeholder: the decoder only tests for nonzero, and the block
     * layer overwrites this word with the real count once a candidate clears
     * its size gate. Streams that never get that far are discarded or keep a
     * harmless nonzero value. */
    zpse_store_u32le(dst + 4u, 1u);
    return (uint32_t)written + ZPSE_LZ4_HDR;
}

static int lz4_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                      uint32_t n)
{
    int got;

    if (in == 0 || out == 0 || n == 0u || in_len <= ZPSE_LZ4_HDR ||
        in_len - ZPSE_LZ4_HDR > (uint32_t)INT32_MAX ||
        n > (uint32_t)INT32_MAX) {
        return ZPSE_BAD_LZ;
    }
    if (zpse_load_u32le(in + 4u) == 0u) {
        return ZPSE_BAD_LZ;
    }
    got = LZ4_decompress_safe((const char *)(in + ZPSE_LZ4_HDR),
                              (char *)out, (int)(in_len - ZPSE_LZ4_HDR),
                              (int)n);
    if (got < 0 || (uint32_t)got != n) {
        return ZPSE_BAD_LZ;
    }
    return ZPSE_OK;
}

uint32_t zpse_lz4_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap)
{
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lz4_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap)
{
    return lz4_encode(in, n, dst, cap);
}

int zpse_lz4_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n)
{
    return lz4_decode(in, in_len, out, n);
}
