#include "zpse/transform.h"
#include "zpse/constants.h"
#include "zpse/status.h"

#include <stdint.h>
#include <string.h>

static int zpse_valid_count(uint8_t v)
{
    return (v == 1 || v == 2 || v == 4 || v == 8 || v == 16) ? 1 : 0;
}

static int zpse_needs_delta(uint8_t tid, uint8_t flags)
{
    if (tid == 1) {
        return 1;
    }
    if (tid >= 7 && tid <= 10) {
        return 1;
    }
    if (tid == 11 && (flags & (uint8_t)1)) {
        return 1;
    }
    return 0;
}

static int zpse_needs_xor(uint8_t tid)
{
    return (tid == 2) ? 1 : 0;
}

static int zpse_ranges_overlap(const void *a, uint32_t alen,
                               const void *b, uint32_t blen)
{
    uintptr_t x;
    uintptr_t y;
    if (alen == 0 || blen == 0) {
        return 0;
    }
    x = (uintptr_t)a;
    y = (uintptr_t)b;
    return (x < y + (uintptr_t)blen && y < x + (uintptr_t)alen) ? 1 : 0;
}

int zpse_transform_width(uint8_t transform_id, uint8_t segment_count,
                         uint8_t *w_out)
{
    uint8_t w = 0;

    if (w_out == 0) {
        return PSE_BAD_ARG;
    }
    switch (transform_id) {
    case 0:
    case 1:
    case 2:
        w = 1;
        break;
    case 3:
    case 7:
        w = 2;
        break;
    case 4:
    case 8:
        w = 4;
        break;
    case 5:
    case 9:
        w = 8;
        break;
    case 6:
    case 10:
        w = 16;
        break;
    case 11:
        if (!zpse_valid_count(segment_count)) {
            return PSE_BAD_TRANSFORM;
        }
        w = segment_count;
        break;
    default:
        return PSE_BAD_TRANSFORM;
    }
    if (transform_id != 11 && segment_count != w) {
        return PSE_BAD_TRANSFORM;
    }
    *w_out = w;
    return PSE_OK;
}

int zpse_segment_lengths(uint32_t n, uint8_t w, uint32_t out_lens[16])
{
    uint32_t s;

    if (out_lens == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (!zpse_valid_count(w)) {
        return PSE_BAD_ARG;
    }
    for (s = 0; s < (uint32_t)w; s++) {
        out_lens[s] = (n + (uint32_t)w - 1u - s) / (uint32_t)w;
    }
    for (s = (uint32_t)w; s < 16u; s++) {
        out_lens[s] = 0;
    }
    return PSE_OK;
}

int zpse_transform_validate_sizes(uint8_t transform_id, uint8_t segment_count,
                                  uint32_t n, const uint32_t *seg_lens)
{
    uint8_t w = 0;
    uint8_t s;
    int rc;

    if (seg_lens == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_BLOCK_SIZE;
    }
    rc = zpse_transform_width(transform_id, segment_count, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    for (s = 0; s < w; s++) {
        uint32_t expect = (n + (uint32_t)w - 1u - (uint32_t)s) / (uint32_t)w;
        if (seg_lens[s] != expect) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
        if (seg_lens[s] < 1) {
            return PSE_BAD_SEGMENT_DIRECTORY;
        }
    }
    return PSE_OK;
}

int zpse_transform_forward(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *in, uint32_t n,
                           uint8_t *segs[16], const uint32_t seg_lens[16])
{
    uint8_t w = 0;
    uint8_t s;
    uint32_t j;
    int rc;
    int delta;
    int xorr;

    if (meta == 0 || in == 0 || segs == 0 || seg_lens == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (meta->transform_id > 11) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_width(meta->transform_id, seg_count, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    if (meta->width_W != w || seg_count != w) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_validate_sizes(meta->transform_id, seg_count, n,
                                       seg_lens);
    if (rc != PSE_OK) {
        return rc;
    }
    for (s = 0; s < w; s++) {
        if (segs[s] == 0) {
            return PSE_BAD_ARG;
        }
    }
    delta = zpse_needs_delta(meta->transform_id, meta->stride_flags);
    xorr = zpse_needs_xor(meta->transform_id);
    if ((delta || xorr) && !meta->has_seeds) {
        return PSE_BAD_ARG;
    }
    for (s = 0; s < w; s++) {
        if (zpse_ranges_overlap(in, n, segs[s], seg_lens[s])) {
            return PSE_BAD_ARG;
        }
    }
    for (s = 0; s < w; s++) {
        uint8_t t;
        for (t = (uint8_t)(s + 1u); t < w; t++) {
            if (zpse_ranges_overlap(segs[s], seg_lens[s],
                                    segs[t], seg_lens[t])) {
                return PSE_BAD_ARG;
            }
        }
    }
    for (s = 0; s < w; s++) {
        uint32_t len = seg_lens[s];
        uint8_t *dst = segs[s];
        if (delta) {
            uint8_t seed = meta->seeds[s];
            uint8_t prev = seed;
            for (j = 0; j < len; j++) {
                uint8_t raw = in[(uint32_t)s + j * (uint32_t)w];
                dst[j] = (uint8_t)(raw - prev);
                prev = raw;
            }
        } else if (xorr) {
            uint8_t seed = meta->seeds[s];
            uint8_t prev = seed;
            for (j = 0; j < len; j++) {
                uint8_t raw = in[(uint32_t)s + j * (uint32_t)w];
                dst[j] = (uint8_t)(raw ^ prev);
                prev = raw;
            }
        } else {
            for (j = 0; j < len; j++) {
                dst[j] = in[(uint32_t)s + j * (uint32_t)w];
            }
        }
    }
    return PSE_OK;
}

int zpse_transform_inverse(const ZpseTransformMeta *meta, uint8_t seg_count,
                           const uint8_t *segs[16], const uint32_t seg_lens[16],
                           uint8_t *out, uint32_t n)
{
    uint8_t w = 0;
    uint8_t s;
    uint32_t i;
    int rc;
    int delta;
    int xorr;
    uint8_t prev[16];

    if (meta == 0 || segs == 0 || seg_lens == 0 || out == 0) {
        return PSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return PSE_BAD_ARG;
    }
    if (meta->transform_id > 11) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_width(meta->transform_id, seg_count, &w);
    if (rc != PSE_OK) {
        return rc;
    }
    if (meta->width_W != w || seg_count != w) {
        return PSE_BAD_TRANSFORM;
    }
    rc = zpse_transform_validate_sizes(meta->transform_id, seg_count, n,
                                       seg_lens);
    if (rc != PSE_OK) {
        return rc;
    }
    for (s = 0; s < w; s++) {
        if (segs[s] == 0) {
            return PSE_BAD_ARG;
        }
    }
    delta = zpse_needs_delta(meta->transform_id, meta->stride_flags);
    xorr = zpse_needs_xor(meta->transform_id);
    if ((delta || xorr) && !meta->has_seeds) {
        return PSE_BAD_ARG;
    }
    for (s = 0; s < w; s++) {
        if (zpse_ranges_overlap(out, n, segs[s], seg_lens[s])) {
            return PSE_BAD_ARG;
        }
    }
    for (s = 0; s < w; s++) {
        uint8_t t;
        for (t = (uint8_t)(s + 1u); t < w; t++) {
            if (zpse_ranges_overlap(segs[s], seg_lens[s],
                                    segs[t], seg_lens[t])) {
                return PSE_BAD_ARG;
            }
        }
    }
    /* Per-segment loops instead of a single i loop with `i % w` and
     * `i / w`: those are two integer divisions per output byte when `w`
     * is not a compile-time constant, which dominated decode for the
     * single-segment transforms. Segment s owns exactly the output
     * positions s, s+w, s+2w, ... in ascending order, and the delta/xor
     * chains are segment-local, so this is bit-identical to the i loop. */
    (void)i;
    if (!delta && !xorr) {
        if (w == 1u) {
            memcpy(out, segs[0], (size_t)n);
            return PSE_OK;
        }
        for (s = 0; s < w; s++) {
            const uint8_t *src = segs[s];
            uint32_t len = seg_lens[s];
            uint32_t j;
            for (j = 0; j < len; j++) {
                out[(uint32_t)s + j * (uint32_t)w] = src[j];
            }
        }
        return PSE_OK;
    }
    for (s = 0; s < w; s++) {
        prev[s] = meta->seeds[s];
    }
    if (w == 1u) {
        const uint8_t *src = segs[0];
        uint8_t pv = prev[0];
        uint32_t j;
        for (j = 0; j < n; j++) {
            uint8_t e = src[j];
            uint8_t d = delta ? (uint8_t)(e + pv) : (uint8_t)(e ^ pv);
            pv = d;
            out[j] = d;
        }
        return PSE_OK;
    }
    for (s = 0; s < w; s++) {
        const uint8_t *src = segs[s];
        uint32_t len = seg_lens[s];
        uint8_t pv = prev[s];
        uint32_t j;
        for (j = 0; j < len; j++) {
            uint8_t e = src[j];
            uint8_t d = delta ? (uint8_t)(e + pv) : (uint8_t)(e ^ pv);
            pv = d;
            out[(uint32_t)s + j * (uint32_t)w] = d;
        }
    }
    return PSE_OK;
}

void zpse_delta8_encode(const uint8_t *in, uint8_t *enc, uint32_t n,
                        uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (in == 0 || enc == 0 || n == 0) {
        return;
    }
    prev = in[0];
    enc[0] = (uint8_t)(in[0] - seed);
    for (i = 1; i < n; i++) {
        uint8_t cur = in[i];
        enc[i] = (uint8_t)(cur - prev);
        prev = cur;
    }
}

void zpse_delta8_decode(const uint8_t *enc, uint8_t *out, uint32_t n,
                        uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (enc == 0 || out == 0 || n == 0) {
        return;
    }
    out[0] = (uint8_t)(enc[0] + seed);
    prev = out[0];
    for (i = 1; i < n; i++) {
        uint8_t e = enc[i];
        uint8_t d = (uint8_t)(e + prev);
        out[i] = d;
        prev = d;
    }
}

void zpse_xor8_encode(const uint8_t *in, uint8_t *enc, uint32_t n,
                      uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (in == 0 || enc == 0 || n == 0) {
        return;
    }
    prev = in[0];
    enc[0] = (uint8_t)(in[0] ^ seed);
    for (i = 1; i < n; i++) {
        uint8_t cur = in[i];
        enc[i] = (uint8_t)(cur ^ prev);
        prev = cur;
    }
}

void zpse_xor8_decode(const uint8_t *enc, uint8_t *out, uint32_t n,
                      uint8_t seed)
{
    uint32_t i;
    uint8_t prev;

    if (enc == 0 || out == 0 || n == 0) {
        return;
    }
    out[0] = (uint8_t)(enc[0] ^ seed);
    prev = out[0];
    for (i = 1; i < n; i++) {
        uint8_t e = enc[i];
        uint8_t d = (uint8_t)(e ^ prev);
        out[i] = d;
        prev = d;
    }
}
