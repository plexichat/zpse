/* Intra-block LZ forms, backed by vendored LZ4 1.10.0.
 *
 * This file replaces src/lz.c, src/lz2.c, src/lzh.c and src/lzfast.c.
 * Every LZ block form now carries the same payload:
 *
 *   u32 uncompressed_bytes   (kept for the block layer's accounting)
 *   u32 command_count        (LZ4 sequences counted from the stream)
 *   raw LZ4 block            output of LZ4_compress_default
 *
 * The block layer still picks among ZPSE_B_LZ / ZPSE_B_LZ2 /
 * ZPSE_B_LZ_HEX by its existing screens and ladder; all three selectors
 * decode through LZ4_decompress_safe with the block's declared output
 * size. There is no custom token loop, match finder, repeat cache or
 * Huffman-trained LZ format left in the codec.
 *
 * LZ4's raw-block window is 64 KiB by definition, which is exactly the
 * block cap here, so a block is self-contained: no history, no frame and
 * no cross-block reference is possible.
 */

#include "zpse/lz.h"
#include "zpse/lz2.h"
#include "zpse/lzh.h"
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
 * output is counted), but the bounds checks keep it safe anyway. */
static uint32_t lz4_cmd_count(const uint8_t *payload, uint32_t len)
{
    const uint8_t *p;
    const uint8_t *end;
    const uint8_t *last;
    uint32_t cmds = 0u;

    if (len <= ZPSE_LZ4_HDR + 5u) {
        return 1u;
    }
    p = payload + ZPSE_LZ4_HDR;
    end = p + (len - ZPSE_LZ4_HDR);
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
    zpse_store_u32le(dst + 4u,
                     lz4_cmd_count(dst, (uint32_t)written + ZPSE_LZ4_HDR));
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

uint32_t zpse_lz_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                        uint32_t cap, void *table, uint32_t hash_bits)
{
    (void)table;
    (void)hash_bits;
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lz_encode_ex(const uint8_t *in, uint32_t n, uint8_t *dst,
                           uint32_t cap, void *table, uint32_t hash_bits,
                           uint32_t lazy_max)
{
    (void)table;
    (void)hash_bits;
    (void)lazy_max;
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lz_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                               uint32_t cap, void *table, uint32_t hash_bits,
                               uint32_t sn)
{
    (void)table;
    (void)hash_bits;
    (void)sn;
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lz2_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits)
{
    (void)table;
    (void)hash_bits;
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lz2_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sn)
{
    (void)table;
    (void)hash_bits;
    (void)sn;
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lzh_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits)
{
    (void)table;
    (void)hash_bits;
    return lz4_encode(in, n, dst, cap);
}

uint32_t zpse_lzh_encode_sample(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sn)
{
    (void)table;
    (void)hash_bits;
    (void)sn;
    return lz4_encode(in, n, dst, cap);
}

int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n)
{
    return lz4_decode(in, in_len, out, n);
}

int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n)
{
    return lz4_decode(in, in_len, out, n);
}

int zpse_lzh_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n)
{
    return lz4_decode(in, in_len, out, n);
}
