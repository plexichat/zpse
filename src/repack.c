/* Fixed-width ASCII repack -- self-contained encoder + decoder.
 *
 * See include/zpse/repack.h for the wire format and API contract.  This
 * translation unit deliberately depends on nothing but repack.h and the C
 * library so it can be compiled on its own (it is registered in the unity
 * build).  No file I/O, no reliance on other zpse objects.
 */

#include "zpse/repack.h"

#include <string.h>

/* ---- wire constants ------------------------------------------------- */
#define RP_MAGIC_B0 0x5Au /* 'Z' */
#define RP_MAGIC_B1 0x52u /* 'R' */
#define RP_MAGIC_B2 0x50u /* 'P' */
#define RP_MAGIC_B3 0x4Bu /* 'K' */
#define RP_VERSION  1u
#define RP_FLAG_MAP16 1u /* w <= 4: 16-byte inverse map; else 128-byte */

#define RP_OFF_MAGIC   0u
#define RP_OFF_VER     4u
#define RP_OFF_W       5u
#define RP_OFF_FLAGS   6u
#define RP_OFF_N       8u
#define RP_OFF_PAYLOAD 16u
#define RP_OFF_DISTINCT 20u
#define RP_OFF_HCRC    24u
#define RP_OFF_MAP     28u
#define RP_MAP16_BYTES 16u
#define RP_MAP128_BYTES 128u
#define RP_HDR_PREFIX  24u /* bytes [0..24) covered by the CRC */

#define RP_SAMPLE 65536u

/* ---- little-endian scalar helpers (no external dependency) --------- */
static void rp_put16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void rp_put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void rp_put64(uint8_t *p, uint64_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}

static uint16_t rp_get16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rp_get32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rp_get64(const uint8_t *p)
{
    return (uint64_t)rp_get32(p) | ((uint64_t)rp_get32(p + 4) << 32);
}

/* ---- CRC-32/IEEE ---------------------------------------------------- */
static uint32_t rp_crc_update(uint32_t crc, const uint8_t *p, size_t n)
{
    size_t i;
    unsigned k;

    for (i = 0; i < n; i++) {
        crc ^= p[i];
        for (k = 0; k < 8; k++) {
            uint32_t mask = (uint32_t)0 - (crc & 1u);
            crc = (crc >> 1) ^ (UINT32_C(0xEDB88320) & mask);
        }
    }
    return crc;
}

/* CRC over the fixed header prefix [0..24) plus the symbol map.  The four
 * CRC bytes at [24..28) are intentionally skipped (else the check is
 * circular).  Both encoder and decoder call this with identical inputs. */
static uint32_t rp_header_crc(const uint8_t *hdr, const uint8_t *map,
                              size_t map_bytes)
{
    uint32_t c = UINT32_C(0xFFFFFFFF);

    c = rp_crc_update(c, hdr, RP_HDR_PREFIX);
    c = rp_crc_update(c, map, map_bytes);
    return ~c;
}

/* ---- width / size helpers ------------------------------------------- */
/* smallest w with 2^w >= k, clamped to [1,7]; k==0 (empty input) -> 1 */
static int rp_width_from_distinct(unsigned k)
{
    int w = 0;

    while (w < 7 && (1u << w) < k) {
        w++;
    }
    return w < 1 ? 1 : w;
}

/* packed bytes for n symbols at width w (8 symbols -> w bytes per group,
 * tail of r<8 symbols -> ceil(r*w/8) bytes) */
static size_t rp_packed_size(size_t n, unsigned w)
{
    size_t groups = n / 8u;
    size_t rest = n - groups * 8u;
    size_t bytes = groups * (size_t)w;

    if (rest != 0u) {
        bytes += (rest * (size_t)w + 7u) / 8u;
    }
    return bytes;
}

size_t zpse_repack_bound(size_t n)
{
    /* Worst case: 128-byte map (payload_off 156) and w == 7. */
    return (size_t)RP_OFF_MAP + RP_MAP128_BYTES + rp_packed_size(n, 7u);
}

/* ---- pack/unpack kernels (RATIO1's group-of-8 bit accumulator) ------
 * For a full group the w-bit codes are accumulated little-endian
 * (code j at bit j*w) and emitted as w little-endian bytes.  The tail does
 * the same for r codes, emitting ceil(r*w/8) bytes.  w==7 with an identity
 * map is exactly FLAT7's 8x7 -> 7-byte pack.
 *
 * Both kernels take a `map` that is identity for w==7 (map[s] == s).
 * map==NULL means identity (used by the w==7 fast path).  For w<7 the caller
 * supplies a 128-byte dense map or NULL for identity; dynamic assignment is
 * performed by the caller before the kernel via rp_assign hook below.
 *
 * To keep the hot loop free of a high-bit branch, each group first OR-checks
 * the eight source bytes with 0x80..; any high bit aborts the encode.
 */
static uint32_t rp_pack_group(const uint8_t *s, int w, const uint8_t *map,
                              uint8_t *out, int check)
{
    uint64_t acc = 0;
    uint64_t raw;
    int j;

    memcpy(&raw, s, 8);
    /* `check` is false for groups wholly inside the 64 KiB sample, where the
     * detection pass already proved every byte < 0x80. Skipping the branch
     * there is safe and leaves the emitted bytes unchanged. */
    if (check && (raw & UINT64_C(0x8080808080808080)) != 0u) {
        return 0; /* high byte: caller declines */
    }
    for (j = 0; j < 8; j++) {
        uint8_t b = (uint8_t)(raw >> (8u * (unsigned)j));
        acc |= (uint64_t)(map != NULL ? map[b] : b) << (j * w);
    }
    for (j = 0; j < w; j++) {
        out[j] = (uint8_t)(acc >> (8u * (unsigned)j));
    }
    return (uint32_t)w;
}

static uint32_t rp_pack_tail(const uint8_t *s, unsigned r, int w,
                             const uint8_t *map, uint8_t *out, int check)
{
    uint64_t acc = 0;
    unsigned j;
    unsigned nb = (r * (unsigned)w + 7u) / 8u;
    unsigned k;

    for (j = 0; j < r; j++) {
        uint8_t b = s[j];
        if (check && (b & 0x80u)) {
            return 0;
        }
        acc |= (uint64_t)(map != NULL ? map[b] : b) << (j * (unsigned)w);
    }
    for (k = 0; k < nb; k++) {
        out[k] = (uint8_t)(acc >> (8u * k));
    }
    return (uint32_t)nb;
}

/* Dynamic-code group body used by the w<7 pack loops. The caller passes a
 * constant `check` (0 = source bytes already verified ASCII, 1 = must test),
 * so gcc specializes the two loops it is called from and the hot one carries
 * no high-bit branch or group-index arithmetic. */
static int rp_pack_body(const uint8_t *s, uint8_t *map, uint8_t *inv,
                        unsigned *next_code, uint8_t *out, int w, int check)
{
    uint64_t raw, acc = 0;
    size_t j;

    memcpy(&raw, s, 8);
    if (check && (raw & UINT64_C(0x8080808080808080)) != 0u) {
        return 0;
    }
    for (j = 0; j < 8; j++) {
        uint8_t b = (uint8_t)(raw >> (8u * j));
        uint8_t c = map[b];
        if (c == 0xFFu) {
            if (*next_code >= (1u << w)) {
                return 0;
            }
            c = (uint8_t)(*next_code)++;
            map[b] = c;
            inv[c] = b;
        }
        acc |= (uint64_t)c << (j * (unsigned)w);
    }
    for (j = 0; j < (size_t)w; j++) {
        out[j] = (uint8_t)(acc >> (8u * j));
    }
    return w;
}

/* Decode a group/tail back to raw bytes using inverse map inv[code]. */
static void rp_unpack_group(const uint8_t *p, int w, const uint8_t *inv,
                            uint8_t *out)
{
    uint64_t acc = 0;
    int j;

    for (j = 0; j < w; j++) {
        acc |= (uint64_t)p[j] << (8u * (unsigned)j);
    }
    for (j = 0; j < 8; j++) {
        unsigned code = (unsigned)((acc >> (j * w)) & ((1u << w) - 1u));
        out[j] = inv != NULL ? inv[code] : (uint8_t)code;
    }
}

static void rp_unpack_tail(const uint8_t *p, unsigned r, int w,
                           const uint8_t *inv, uint8_t *out)
{
    uint64_t acc = 0;
    unsigned nb = (r * (unsigned)w + 7u) / 8u;
    unsigned j;
    unsigned k;

    for (k = 0; k < nb; k++) {
        acc |= (uint64_t)p[k] << (8u * k);
    }
    for (j = 0; j < r; j++) {
        unsigned code = (unsigned)((acc >> (j * (unsigned)w)) & ((1u << w) - 1u));
        out[j] = inv != NULL ? inv[code] : (uint8_t)code;
    }
}

/* Every decline funnels through here.  The encoder writes the packed payload
 * into `out` starting at payload_off BEFORE the header is committed, so a
 * caller that reuses a buffer previously holding a valid frame would be left
 * with that frame's OLD magic and header CRC paired with a half-written
 * payload: decode would accept the header yet expand garbage.  Zeroing the
 * leading magic is O(1), stays inside the contract's "no writes past out_cap"
 * decline bound, and makes any stale frame fail decode's magic check outright.
 * A tiny-cap decline (out_cap < 4) simply has no magic to clear. */
static int rp_decline(uint8_t *out, size_t out_cap)
{
    if (out != NULL && out_cap >= 4u) {
        out[0] = 0u;
        out[1] = 0u;
        out[2] = 0u;
        out[3] = 0u;
    }
    return 0;
}

/* ---- encode --------------------------------------------------------- */
int zpse_repack_encode(const uint8_t *in, size_t n, uint8_t *out,
                       size_t out_cap, size_t *out_len)
{
    uint8_t present[128];
    uint8_t map[128];   /* symbol -> dense code, 0xFF = unassigned */
    uint8_t inv[128];   /* dense code -> symbol, 0xFF = unused */
    size_t sample;
    size_t i;
    unsigned distinct;
    unsigned next_code;
    int w, map16;
    size_t map_bytes, payload_off, payload, needed;
    size_t groups, rest, p, safe_groups;
    uint16_t flags;
    uint32_t hcrc;

    if (out_len == NULL) {
        return rp_decline(out, out_cap);
    }
    if (n != 0u && in == NULL) {
        return rp_decline(out, out_cap);
    }

    /* 1. Width detection from a 64 KiB sample. */
    memset(present, 0, sizeof present);
    sample = (n < (size_t)RP_SAMPLE) ? n : (size_t)RP_SAMPLE;
    for (i = 0; i < sample; i++) {
        uint8_t b = in[i];
        if (b & 0x80u) {
            return rp_decline(out, out_cap); /* non-ASCII in sample */
        }
        present[b] = 1;
    }
    distinct = 0;
    for (i = 0; i < 128; i++) {
        if (present[i]) {
            distinct++;
        }
    }
    w = rp_width_from_distinct(distinct);
    if (w > 7) {
        return rp_decline(out, out_cap); /* impossible for a 7-bit alphabet, defensive */
    }
    map16 = (w <= 4);
    map_bytes = map16 ? RP_MAP16_BYTES : RP_MAP128_BYTES;
    payload_off = (size_t)RP_OFF_MAP + map_bytes;
    payload = rp_packed_size(n, (unsigned)w);
    needed = payload_off + payload;
    if (out == NULL || payload > UINT32_MAX || out_cap < needed) {
        return rp_decline(out, out_cap); /* decline: capacity or representation limit */
    }

    /* 2. Build the symbol map.  Codes are pre-assigned to the sample's
     * symbols in symbol order; any symbol first seen later in the full pass
     * gets the next free code.  This keeps the detection pass 64 KiB but
     * still covers every byte actually present (NCI has 51 distinct symbols
     * in its first 64 KiB but 60 across 8 MiB) without a second full pass. */
    memset(map, 0xFF, sizeof map);
    memset(inv, 0xFF, sizeof inv);
    if (w == 7) {
        /* Identity map: bit-identical to FLAT7's 7-bit pack. */
        for (i = 0; i < 128; i++) {
            map[i] = (uint8_t)i;
            inv[i] = (uint8_t)i;
        }
        distinct = 128;
    } else {
        next_code = 0;
        for (i = 0; i < 128; i++) {
            if (present[i]) {
                map[i] = (uint8_t)next_code;
                inv[next_code] = (uint8_t)i;
                next_code++;
            }
        }
        distinct = 0; /* counted during the full pass below */
    }

    /* 3. Single fused pack pass. `sample` bytes at the head were already
     * verified ASCII by the detection pass, so the safe_groups groups wholly
     * inside that span need no second high-bit check. Later groups keep it --
     * a high byte past the sample must still be declined. Splitting the loop
     * at that boundary keeps the per-group cost identical to the old code. */
    groups = n / 8u;
    rest = n - groups * 8u;
    safe_groups = sample / 8u;
    if (safe_groups > groups) {
        safe_groups = groups;
    }
    p = (size_t)payload_off;
    if (w == 7) {
        for (i = 0; i < safe_groups; i++) {
            uint32_t got = rp_pack_group(in + i * 8u, w, NULL, out + p, 0);
            p += got;
        }
        for (i = safe_groups; i < groups; i++) {
            uint32_t got = rp_pack_group(in + i * 8u, w, NULL, out + p, 1);
            if (got == 0u) {
                return rp_decline(out, out_cap);
            }
            p += got;
        }
        if (rest != 0u) {
            uint32_t got = rp_pack_tail(in + groups * 8u, (unsigned)rest,
                                        w, NULL, out + p,
                                        (groups * 8u + rest > sample));
            if (got == 0u) {
                return rp_decline(out, out_cap);
            }
            p += got;
        }
    } else {
        for (i = 0; i < safe_groups; i++) {
            int got = rp_pack_body(in + i * 8u, map, inv, &next_code,
                                   out + p, w, 0);
            p += (size_t)got;
        }
        for (i = safe_groups; i < groups; i++) {
            int got = rp_pack_body(in + i * 8u, map, inv, &next_code,
                                   out + p, w, 1);
            if (got == 0) {
                return rp_decline(out, out_cap);
            }
            p += (size_t)got;
        }
        if (rest != 0u) {
            size_t j;
            uint64_t acc = 0;
            unsigned nb = (unsigned)((rest * (size_t)w + 7u) / 8u);
            unsigned k;

            for (j = 0; j < rest; j++) {
                uint8_t b = in[groups * 8u + j];
                uint8_t c;
                if ((groups * 8u + j >= sample) && (b & 0x80u)) {
                    return rp_decline(out, out_cap);
                }
                c = map[b];
                if (c == 0xFFu) {
                    if (next_code >= (1u << w)) {
                        return rp_decline(out, out_cap);
                    }
                    c = (uint8_t)next_code++;
                    map[b] = c;
                    inv[c] = b;
                }
                acc |= (uint64_t)c << (j * (unsigned)w);
            }
            for (k = 0; k < nb; k++) {
                out[p++] = (uint8_t)(acc >> (8u * k));
            }
        }
        distinct = next_code;
        if (distinct > (1u << w)) {
            return rp_decline(out, out_cap);
        }
    }

    if (p != needed) {
        return rp_decline(out, out_cap); /* internal pack/length disagreement: commit nothing */
    }

    /* 4. Header last; every earlier bail-out goes through rp_decline(), which
     * clears the magic, so a reused buffer can never expose a stale frame. */
    memset(out, 0, (size_t)RP_OFF_MAP);
    out[RP_OFF_MAGIC + 0] = RP_MAGIC_B0;
    out[RP_OFF_MAGIC + 1] = RP_MAGIC_B1;
    out[RP_OFF_MAGIC + 2] = RP_MAGIC_B2;
    out[RP_OFF_MAGIC + 3] = RP_MAGIC_B3;
    out[RP_OFF_VER] = RP_VERSION;
    out[RP_OFF_W] = (uint8_t)w;
    flags = map16 ? RP_FLAG_MAP16 : 0u;
    rp_put16(out + RP_OFF_FLAGS, flags);
    rp_put64(out + RP_OFF_N, (uint64_t)n);
    rp_put32(out + RP_OFF_PAYLOAD, (uint32_t)payload);
    rp_put32(out + RP_OFF_DISTINCT, distinct);
    /* Copy the inverse map into the frame (unused slots stored as 0). */
    {
        size_t k;
        for (k = 0; k < map_bytes; k++) {
            out[RP_OFF_MAP + k] = (k < 128u && inv[k] != 0xFFu) ? inv[k] : 0u;
        }
    }
    hcrc = rp_header_crc(out, out + RP_OFF_MAP, map_bytes);
    rp_put32(out + RP_OFF_HCRC, hcrc);

    *out_len = needed;
    return 1;
}

/* ---- decode --------------------------------------------------------- */
int zpse_repack_decode(const uint8_t *in, size_t in_len, uint8_t *out,
                       size_t out_cap)
{
    uint16_t flags;
    int w, map16;
    size_t map_bytes, payload_off, payload;
    uint64_t n64;
    size_t n, needed;
    uint32_t distinct, hcrc;
    const uint8_t *inv;
    const uint8_t *pay;
    size_t groups, rest, p;
    size_t i;

    if (in == NULL) {
        return 0;
    }
    if (in_len < (size_t)RP_OFF_MAP) {
        return 0;
    }
    if (in[RP_OFF_MAGIC + 0] != RP_MAGIC_B0 ||
        in[RP_OFF_MAGIC + 1] != RP_MAGIC_B1 ||
        in[RP_OFF_MAGIC + 2] != RP_MAGIC_B2 ||
        in[RP_OFF_MAGIC + 3] != RP_MAGIC_B3) {
        return 0;
    }
    if (in[RP_OFF_VER] != RP_VERSION) {
        return 0;
    }
    w = in[RP_OFF_W];
    if (w < 1 || w > 7) {
        return 0;
    }
    flags = rp_get16(in + RP_OFF_FLAGS);
    if ((flags & (uint16_t)~RP_FLAG_MAP16) != 0u) {
        return 0; /* reserved flag bits set */
    }
    map16 = (flags & RP_FLAG_MAP16) != 0;
    if (map16 != (w <= 4)) {
        return 0;
    }
    map_bytes = map16 ? RP_MAP16_BYTES : RP_MAP128_BYTES;
    payload_off = (size_t)RP_OFF_MAP + map_bytes;
    if (in_len < payload_off) {
        return 0;
    }

    n64 = rp_get64(in + RP_OFF_N);
    if (n64 > (uint64_t)SIZE_MAX) {
        return 0;
    }
    n = (size_t)n64;
    payload = (size_t)rp_get32(in + RP_OFF_PAYLOAD);
    distinct = rp_get32(in + RP_OFF_DISTINCT);
    if (distinct > 128u || distinct > (1u << w)) {
        return 0;
    }
    if (payload != rp_packed_size(n, (unsigned)w)) {
        return 0; /* not a canonical payload for this width/length */
    }
    needed = payload_off + payload;
    if (in_len < needed) {
        return 0; /* truncated */
    }
    if (n > out_cap || (n != 0u && out == NULL)) {
        return 0;
    }

    /* Header CRC (CRC bytes excluded from the computation). */
    hcrc = rp_header_crc(in, in + RP_OFF_MAP, map_bytes);
    if (hcrc != rp_get32(in + RP_OFF_HCRC)) {
        return 0; /* corrupt header or symbol map */
    }

    inv = in + RP_OFF_MAP;
    pay = in + payload_off;

    if (w == 7) {
        /* Identity map: skip the indirection, exactly like FLAT7 unpack. */
        groups = n / 8u;
        rest = n - groups * 8u;
        p = 0;
        for (i = 0; i < groups; i++) {
            rp_unpack_group(pay + p, w, NULL, out + i * 8u);
            p += (size_t)w;
        }
        if (rest != 0u) {
            rp_unpack_tail(pay + p, (unsigned)rest, w, NULL,
                           out + groups * 8u);
        }
    } else {
        groups = n / 8u;
        rest = n - groups * 8u;
        p = 0;
        for (i = 0; i < groups; i++) {
            rp_unpack_group(pay + p, w, inv, out + i * 8u);
            p += (size_t)w;
        }
        if (rest != 0u) {
            rp_unpack_tail(pay + p, (unsigned)rest, w, inv,
                           out + groups * 8u);
        }
    }
    return 1;
}
