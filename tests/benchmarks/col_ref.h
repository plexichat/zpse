/* Reference column codecs: what a purpose-built codec achieves on the
 * columnar corpora in tests/corpus/columnar_gen.h.
 *
 * The point of these is not to be fast. They are not competitors to
 * zpse and they are not a proposal to put a columnar stage inside it.
 * They answer one question the cross-tool table cannot: zpse loses to
 * zstd on a timestamp column -- but zstd is a *block* codec, so that
 * comparison only says "the data has structure zstd's match finder
 * likes". These encoders say what the structure is worth when it is
 * coded for what it is: a delta-of-delta for nanosecond timestamps, a
 * null bitmap for a nullable column, a frame of reference for
 * increasing offsets.
 *
 * The gap between a reference row and the zpse row on the same corpus
 * is therefore the price of zpse's design: no schema, no column
 * semantics, one byte-oriented model for everything. That is the
 * number that decides whether a typed stage is worth adding.
 *
 * Every encoder has a matching decoder, and tests/suites/test_colref.c
 * proves they round-trip, so a reference size is a size a real codec
 * reached and not an estimate.
 *
 * Each codec is applied to one 64 KiB block at a time, the same
 * framing zpse uses, so the sizes are comparable: a reference that
 * could see the whole column would be measuring something else.
 *
 * The formats are deliberately plain -- varint, bit packing, a bitmap.
 * They are reference points, not a proposal. */
#ifndef ZPSE_COL_REF_H
#define ZPSE_COL_REF_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---- bit writer / reader (LSB first within a byte) ----------------- */

typedef struct ZpseBitWriter {
    uint8_t *out;
    size_t cap;
    size_t pos;   /* bytes written */
    uint64_t acc; /* pending bits, in the low `nbits` */
    unsigned nbits;
    int overflow;
} ZpseBitWriter;

static inline void zpse_bw_init(ZpseBitWriter *w, uint8_t *out, size_t cap)
{
    w->out = out;
    w->cap = cap;
    w->pos = 0;
    w->acc = 0;
    w->nbits = 0;
    w->overflow = 0;
}

static inline void zpse_bw_put(ZpseBitWriter *w, uint64_t v, unsigned bits)
{
    if (bits == 0u) return;
    if (bits < 64u) v &= (UINT64_C(1) << bits) - 1u;
    w->acc |= v << w->nbits;
    w->nbits += bits;
    while (w->nbits >= 8u) {
        if (w->pos >= w->cap) {
            w->overflow = 1;
            w->nbits = 0;
            return;
        }
        w->out[w->pos++] = (uint8_t)(w->acc & 0xFFu);
        w->acc >>= 8;
        w->nbits -= 8u;
    }
}

static inline void zpse_bw_flush(ZpseBitWriter *w)
{
    if (w->nbits > 0u) {
        if (w->pos >= w->cap) {
            w->overflow = 1;
        } else {
            w->out[w->pos++] = (uint8_t)(w->acc & 0xFFu);
        }
        w->acc = 0;
        w->nbits = 0;
    }
}

typedef struct ZpseBitReader {
    const uint8_t *in;
    size_t len;
    size_t pos;
    uint64_t acc;
    unsigned nbits;
    int overrun;
} ZpseBitReader;

static inline void zpse_br_init(ZpseBitReader *r, const uint8_t *in, size_t len)
{
    r->in = in;
    r->len = len;
    r->pos = 0;
    r->acc = 0;
    r->nbits = 0;
    r->overrun = 0;
}

static inline uint64_t zpse_br_get(ZpseBitReader *r, unsigned bits)
{
    uint64_t v;

    if (bits == 0u) return 0;
    while (r->nbits < bits) {
        if (r->pos >= r->len) {
            r->overrun = 1;
            return 0;
        }
        r->acc |= (uint64_t)r->in[r->pos++] << r->nbits;
        r->nbits += 8u;
    }
    v = (bits >= 64u) ? r->acc : (r->acc & ((UINT64_C(1) << bits) - 1u));
    r->acc >>= bits;
    r->nbits -= bits;
    return v;
}

/* ---- varint (LEB128) ----------------------------------------------- */

static inline size_t zpse_varint_put(uint8_t *out, size_t cap, size_t pos,
                                     uint64_t v)
{
    while (v >= 0x80u) {
        if (pos >= cap) return pos;
        out[pos++] = (uint8_t)(v | 0x80u);
        v >>= 7;
    }
    if (pos >= cap) return pos;
    out[pos++] = (uint8_t)v;
    return pos;
}

static inline int zpse_varint_get(const uint8_t *in, size_t len, size_t *pos,
                                  uint64_t *v)
{
    uint64_t r = 0;
    unsigned shift = 0;

    for (;;) {
        uint8_t b;
        if (*pos >= len) return -1;
        b = in[(*pos)++];
        if (shift > 63u) return -1;
        r |= (uint64_t)(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0u) break;
        shift += 7u;
    }
    *v = r;
    return 0;
}

static inline uint64_t zpse_zigzag(int64_t v)
{
    return ((uint64_t)v << 1) ^ (uint64_t)(v >> 63);
}

static inline int64_t zpse_unzigzag(uint64_t v)
{
    return (int64_t)(v >> 1) ^ -(int64_t)(v & 1u);
}

/* ---- little-endian element access --------------------------------- */

static inline uint32_t zpse_ref_get32le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t zpse_ref_get64le(const uint8_t *p)
{
    uint64_t v = 0;
    int k;
    for (k = 0; k < 8; k++) v |= (uint64_t)p[k] << (8 * k);
    return v;
}

static inline void zpse_ref_put32le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static inline void zpse_ref_put64le(uint8_t *p, uint64_t v)
{
    int k;
    for (k = 0; k < 8; k++) p[k] = (uint8_t)(v >> (8 * k));
}

/* ---- ts_seq: delta of timestamps, varint --------------------------- */

/* The first value is stored absolute, the rest as the difference to
 * the previous one. A sorted microsecond column has a step of about a
 * millisecond, so a difference is a varint of two bytes and the whole
 * column shrinks by a factor of four. */
static inline int zpse_colref_ts_seq_encode(const uint8_t *in, size_t n,
                                             uint8_t *out, size_t cap,
                                             size_t *out_len)
{
    size_t pos = 0;
    size_t r;
    uint64_t prev = 0;

    for (r = 0; (r + 1u) * 8u <= n; r++) {
        uint64_t v = zpse_ref_get64le(in + r * 8u);
        uint64_t d = (r == 0u) ? v : (v - prev);
        pos = zpse_varint_put(out, cap, pos, d);
        if (pos >= cap) return -1;
        prev = v;
    }
    *out_len = pos;
    return 0;
}

static inline int zpse_colref_ts_seq_decode(const uint8_t *in, size_t n,
                                             uint8_t *out, size_t n_out)
{
    size_t pos = 0;
    size_t r = 0;
    uint64_t prev = 0;

    while ((r + 1u) * 8u <= n_out) {
        uint64_t d;
        uint64_t v;
        if (zpse_varint_get(in, n, &pos, &d) != 0) return -1;
        v = (r == 0u) ? d : (prev + d);
        zpse_ref_put64le(out + r * 8u, v);
        prev = v;
        r++;
    }
    return (pos == n) ? 0 : -1;
}

/* ---- ts_ns: double delta, varint ---------------------------------- */

/* The difference between two differences. A nanosecond feed's step is
 * nearly constant, so the second difference is a small signed number
 * -- usually zero -- and the column becomes a stream of one-byte
 * varints. */
static inline int zpse_colref_ts_ns_encode(const uint8_t *in, size_t n,
                                            uint8_t *out, size_t cap,
                                            size_t *out_len)
{
    size_t pos = 0;
    size_t r;
    int64_t prev = 0;
    int64_t prev_d = 0;

    for (r = 0; (r + 1u) * 8u <= n; r++) {
        int64_t v = (int64_t)zpse_ref_get64le(in + r * 8u);
        int64_t d = (r == 0u) ? v : (v - prev);
        int64_t dd = (r < 2u) ? d : (d - prev_d);
        pos = zpse_varint_put(out, cap, pos, zpse_zigzag(dd));
        if (pos >= cap) return -1;
        prev = v;
        prev_d = d;
    }
    *out_len = pos;
    return 0;
}

static inline int zpse_colref_ts_ns_decode(const uint8_t *in, size_t n,
                                            uint8_t *out, size_t n_out)
{
    size_t pos = 0;
    size_t r = 0;
    int64_t prev = 0;
    int64_t prev_d = 0;

    while ((r + 1u) * 8u <= n_out) {
        uint64_t z;
        int64_t d;
        int64_t v;
        if (zpse_varint_get(in, n, &pos, &z) != 0) return -1;
        d = (r < 2u) ? zpse_unzigzag(z) : (prev_d + zpse_unzigzag(z));
        v = (r == 0u) ? d : (prev + d);
        zpse_ref_put64le(out + r * 8u, (uint64_t)v);
        prev = v;
        prev_d = d;
        r++;
    }
    return (pos == n) ? 0 : -1;
}

/* ---- float_sparse: XOR of neighbours, bit packed (Gorilla) --------- */

/* The classic Gorilla float layout: XOR a value with the previous one,
 * then store the leading zeros, the length of the meaningful part and
 * the meaningful bits themselves. A value equal to its predecessor
 * costs a single bit, which is the whole point on a column of
 * readings that mostly repeat. */
static inline int zpse_colref_float_encode(const uint8_t *in, size_t n,
                                            uint8_t *out, size_t cap,
                                            size_t *out_len)
{
    ZpseBitWriter w;
    size_t r;
    uint32_t prev = 0;

    zpse_bw_init(&w, out, cap);
    for (r = 0; (r + 1u) * 4u <= n; r++) {
        uint32_t v = zpse_ref_get32le(in + r * 4u);
        uint32_t x = v ^ prev;
        unsigned lead;
        unsigned trail;
        unsigned len;

        if (x == 0u) {
            zpse_bw_put(&w, 0, 1);
            prev = v;
            continue;
        }
        lead = 0;
        while (lead < 32u && ((x >> (31u - lead)) & 1u) == 0u) lead++;
        trail = 0;
        while (trail < 32u && ((x >> trail) & 1u) == 0u) trail++;
        len = 32u - lead - trail;
        zpse_bw_put(&w, 1, 1);
        zpse_bw_put(&w, lead, 5);
        zpse_bw_put(&w, len, 6);
        zpse_bw_put(&w, (uint64_t)(x >> trail), len);
        prev = v;
    }
    zpse_bw_flush(&w);
    if (w.overflow) return -1;
    *out_len = w.pos;
    return 0;
}

static inline int zpse_colref_float_decode(const uint8_t *in, size_t n,
                                            uint8_t *out, size_t n_out)
{
    ZpseBitReader r;
    size_t rec = 0;
    uint32_t prev = 0;

    zpse_br_init(&r, in, n);
    while ((rec + 1u) * 4u <= n_out) {
        uint32_t v;
        if (zpse_br_get(&r, 1) == 0u) {
            v = prev;
        } else {
            unsigned lead = (unsigned)zpse_br_get(&r, 5);
            unsigned len = (unsigned)zpse_br_get(&r, 6);
            uint64_t mid;
            if (lead + len > 32u) return -1;
            mid = zpse_br_get(&r, len);
            if (r.overrun) return -1;
            v = (uint32_t)(mid << (32u - lead - len)) ^ prev;
        }
        zpse_ref_put32le(out + rec * 4u, v);
        prev = v;
        rec++;
    }
    return r.overrun ? -1 : 0;
}

/* ---- sparse_mix: null bitmap + values ----------------------------- */

/* One bit per value saying whether it is null, then the non-null
 * values. A column that is 92% null costs a bit per value and nothing
 * for the rest -- which is the whole point of null suppression, and
 * the reason a wide analytics table compresses to almost nothing. */
static inline int zpse_colref_sparse_encode(const uint8_t *in, size_t n,
                                            uint8_t *out, size_t cap,
                                            size_t *out_len)
{
    size_t nvals = n / 8u;
    size_t nbytes = (nvals + 7u) / 8u;
    size_t pos;
    size_t i;

    if (cap < nbytes) return -1;
    memset(out, 0, nbytes);
    pos = nbytes;
    for (i = 0; i < nvals; i++) {
        const uint8_t *v = in + i * 8u;
        int null = (v[0] | v[1] | v[2] | v[3] | v[4] | v[5] | v[6] | v[7]) == 0;
        if (!null) {
            out[i / 8u] |= (uint8_t)(1u << (i % 8u));
            if (pos + 8u > cap) return -1;
            memcpy(out + pos, v, 8);
            pos += 8u;
        }
    }
    *out_len = pos;
    return 0;
}

static inline int zpse_colref_sparse_decode(const uint8_t *in, size_t n,
                                            uint8_t *out, size_t n_out)
{
    size_t nvals = n_out / 8u;
    size_t nbytes = (nvals + 7u) / 8u;
    size_t pos = nbytes;
    size_t i;

    if (n < nbytes) return -1;
    for (i = 0; i < nvals; i++) {
        if ((in[i / 8u] >> (i % 8u)) & 1u) {
            if (pos + 8u > n) return -1;
            memcpy(out + i * 8u, in + pos, 8);
            pos += 8u;
        } else {
            memset(out + i * 8u, 0, 8);
        }
    }
    return 0;
}

/* ---- geo_point: zigzag deltas for the coordinates ----------------- */

/* The two coordinates are stored as the zigzag of their difference to
 * the previous point's, which for a moving point is a small number. The
 * cell id is a hash and does not delta, but it only changes every
 * thirty-two points, so a bitmap saying which points carry a new cell
 * costs a fraction of a byte per point. */
static inline int zpse_colref_geo_encode(const uint8_t *in, size_t n,
                                          uint8_t *out, size_t cap,
                                          size_t *out_len)
{
    size_t nrec = n / 16u;
    size_t nbytes = (nrec + 7u) / 8u;
    size_t pos;
    size_t r;
    int32_t plat = 0;
    int32_t plon = 0;
    uint64_t pcell = 0;

    if (cap < nbytes) return -1;
    memset(out, 0, nbytes);
    pos = nbytes;
    for (r = 0; r < nrec; r++) {
        const uint8_t *p = in + r * 16u;
        int32_t lat = (int32_t)zpse_ref_get32le(p);
        int32_t lon = (int32_t)zpse_ref_get32le(p + 4);
        uint64_t cell = zpse_ref_get64le(p + 8);
        uint8_t same;

        same = (r != 0u && cell == pcell) ? 0u : 1u;
        if (same) out[r / 8u] |= (uint8_t)(1u << (r % 8u));
        pos = zpse_varint_put(out, cap, pos, zpse_zigzag((int64_t)lat - plat));
        if (pos >= cap) return -1;
        pos = zpse_varint_put(out, cap, pos, zpse_zigzag((int64_t)lon - plon));
        if (pos >= cap) return -1;
        if (same) {
            if (pos + 8u > cap) return -1;
            zpse_ref_put64le(out + pos, cell);
            pos += 8u;
        }
        plat = lat;
        plon = lon;
        pcell = cell;
    }
    *out_len = pos;
    return 0;
}

static inline int zpse_colref_geo_decode(const uint8_t *in, size_t n,
                                          uint8_t *out, size_t n_out)
{
    size_t nrec = n_out / 16u;
    size_t nbytes = (nrec + 7u) / 8u;
    size_t pos = nbytes;
    size_t r;
    int32_t plat = 0;
    int32_t plon = 0;
    uint64_t pcell = 0;

    if (n < nbytes) return -1;
    for (r = 0; r < nrec; r++) {
        uint64_t zl;
        uint64_t zo;
        int32_t lat;
        int32_t lon;
        uint64_t cell;
        int newcell;

        if (zpse_varint_get(in, n, &pos, &zl) != 0) return -1;
        if (zpse_varint_get(in, n, &pos, &zo) != 0) return -1;
        newcell = (int)((in[r / 8u] >> (r % 8u)) & 1u);
        lat = plat + (int32_t)zpse_unzigzag(zl);
        lon = plon + (int32_t)zpse_unzigzag(zo);
        if (newcell) {
            if (pos + 8u > n) return -1;
            cell = zpse_ref_get64le(in + pos);
            pos += 8u;
        } else {
            cell = pcell;
        }
        zpse_ref_put32le(out + r * 16u, (uint32_t)lat);
        zpse_ref_put32le(out + r * 16u + 4, (uint32_t)lon);
        zpse_ref_put64le(out + r * 16u + 8, cell);
        plat = lat;
        plon = lon;
        pcell = cell;
    }
    return 0;
}

/* ---- struct_offsets: frame of reference, bit packed --------------- */

/* Offsets into a byte array only ever grow, and they grow by a small
 * amount. The classic frame of reference: take the first offset of a
 * group as the frame, and store the rest as bit-packed residuals
 * inside it. The width of the residual field is chosen per group from
 * the data, so a group of near-constant strides costs a few bits per
 * offset instead of four bytes.
 *
 * A group is byte-aligned in the output (the bit writer is flushed at
 * the end of each group), so the decoder can find the group boundaries
 * without a bit-level parse of the container. */
#define ZPSE_COLREF_FOR_GROUP 64u

static inline int zpse_colref_offsets_encode(const uint8_t *in, size_t n,
                                              uint8_t *out, size_t cap,
                                              size_t *out_len)
{
    size_t nvals = n / 4u;
    size_t pos = 0;
    size_t g;

    for (g = 0; g < nvals; g += ZPSE_COLREF_FOR_GROUP) {
        size_t cnt = nvals - g;
        uint32_t base;
        uint32_t maxres = 0;
        size_t k;
        unsigned width;
        ZpseBitWriter w;

        if (cnt > ZPSE_COLREF_FOR_GROUP) cnt = ZPSE_COLREF_FOR_GROUP;
        base = zpse_ref_get32le(in + g * 4u);
        for (k = 1; k < cnt; k++) {
            uint32_t v = zpse_ref_get32le(in + (g + k) * 4u);
            uint32_t res = (v >= base) ? (v - base) : 0u;
            if (res > maxres) maxres = res;
        }
        width = 0;
        while (width < 32u && (maxres >> width) != 0u) width++;
        pos = zpse_varint_put(out, cap, pos, base);
        if (pos >= cap) return -1;
        pos = zpse_varint_put(out, cap, pos, width);
        if (pos >= cap) return -1;
        zpse_bw_init(&w, out + pos, cap - pos);
        for (k = 1; k < cnt; k++) {
            uint32_t v = zpse_ref_get32le(in + (g + k) * 4u);
            uint32_t res = (v >= base) ? (v - base) : 0u;
            zpse_bw_put(&w, res, width);
        }
        zpse_bw_flush(&w);
        if (w.overflow) return -1;
        pos += w.pos;
    }
    *out_len = pos;
    return 0;
}

static inline int zpse_colref_offsets_decode(const uint8_t *in, size_t n,
                                              uint8_t *out, size_t n_out)
{
    size_t nvals = n_out / 4u;
    size_t pos = 0;
    size_t g = 0;

    while (g < nvals) {
        uint64_t base;
        uint64_t width;
        size_t cnt = nvals - g;
        size_t k;
        ZpseBitReader r;

        if (cnt > ZPSE_COLREF_FOR_GROUP) cnt = ZPSE_COLREF_FOR_GROUP;
        if (zpse_varint_get(in, n, &pos, &base) != 0) return -1;
        if (zpse_varint_get(in, n, &pos, &width) != 0) return -1;
        if (width > 32u) return -1;
        zpse_br_init(&r, in + pos, n - pos);
        for (k = 0; k < cnt; k++) {
            uint32_t v;
            if (k == 0u) {
                v = (uint32_t)base;
            } else {
                uint32_t res = (uint32_t)zpse_br_get(&r, (unsigned)width);
                if (r.overrun) return -1;
                v = (uint32_t)base + res;
            }
            zpse_ref_put32le(out + (g + k) * 4u, v);
        }
        pos += r.pos;
        g += cnt;
    }
    return 0;
}

/* ---- the table ---------------------------------------------------- */

typedef struct ZpseColRef {
    const char *name;
    int (*encode)(const uint8_t *in, size_t n, uint8_t *out, size_t cap,
                  size_t *out_len);
    int (*decode)(const uint8_t *in, size_t n, uint8_t *out, size_t n_out);
} ZpseColRef;

/* NULL for a column with no reference: var_string is high-cardinality
 * text, where the right tool is a block codec with a match finder --
 * zstd and brotli are already in the table -- and a column codec would
 * have nothing to add. */
static inline const ZpseColRef *zpse_colref_lookup(int kind)
{
    switch (kind) {
    case 0: /* ts_seq */
    {
        static const ZpseColRef r = { "delta+varint", zpse_colref_ts_seq_encode,
                                      zpse_colref_ts_seq_decode };
        return &r;
    }
    case 1: /* ts_ns */
    {
        static const ZpseColRef r = { "double-delta", zpse_colref_ts_ns_encode,
                                      zpse_colref_ts_ns_decode };
        return &r;
    }
    case 2: /* float_sparse */
    {
        static const ZpseColRef r = { "gorilla-xor", zpse_colref_float_encode,
                                      zpse_colref_float_decode };
        return &r;
    }
    case 4: /* sparse_mix */
    {
        static const ZpseColRef r = { "null-bitmap", zpse_colref_sparse_encode,
                                      zpse_colref_sparse_decode };
        return &r;
    }
    case 5: /* geo_point */
    {
        static const ZpseColRef r = { "zigzag-delta", zpse_colref_geo_encode,
                                      zpse_colref_geo_decode };
        return &r;
    }
    case 6: /* struct_offsets */
    {
        static const ZpseColRef r = { "frame-of-ref", zpse_colref_offsets_encode,
                                      zpse_colref_offsets_decode };
        return &r;
    }
    default:
        return NULL; /* var_string */
    }
}

#endif /* ZPSE_COL_REF_H */
