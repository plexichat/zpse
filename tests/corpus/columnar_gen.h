/* The columnar benchmark corpora, shared by every harness that measures
 * them.
 *
 * tests/benchmarks/bench_columnar.c fills them in RAM and compares zpse
 * against zstd/lz4/zlib/brotli in process; tests/corpus/corpus_write.c
 * writes them to files for the cross-tool comparison in
 * tests/benchmarks/bench_vs.sh; tests/benchmarks/bench_colref.c measures
 * what a purpose-built codec for each column achieves. All of them must
 * see the same bytes, or the tables are about different data -- so the
 * generators live here and each column's seed is fixed by the table
 * below rather than by whichever harness is running. A change to one
 * harness's PRNG must not move a column.
 *
 * Every generator is a pure function of the record index and the
 * column's seed, and never of how the bytes are handed to it: filling
 * 4 MiB in one call and filling 64 KiB sixty-four times produce the
 * same stream. The harnesses rely on that -- bench_columnar.c fills
 * one buffer, corpus_write.c fills one chunk at a time -- so they never
 * have to agree on a chunking, and a block boundary in the middle of a
 * record cannot change the corpus.
 *
 * The columns are the shapes a time-series and an analytics table are
 * made of, each one a case where a general-purpose block codec is the
 * wrong tool and a column codec is the right one:
 *
 *   ts_seq          i64 microsecond timestamps, strictly increasing
 *   ts_ns           i64 nanosecond timestamps, ~1 us apart
 *   float_sparse    f32 readings, mostly off a short ladder
 *   var_string      JSON-ish records with high-cardinality values
 *   sparse_mix      nullable i64 column, 92% null
 *   geo_point       lat/lon in 1e-7 degrees plus a cell id
 *   struct_offsets  i32 offsets into a byte array, strictly increasing
 *
 * All integers are little-endian, so a column's byte layout does not
 * depend on the host it was generated on. */
#ifndef ZPSE_COLUMNAR_GEN_H
#define ZPSE_COLUMNAR_GEN_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ZPSE_COL_KINDS 7u

enum ZpseColKind {
    ZPSE_COL_TS_SEQ = 0,
    ZPSE_COL_TS_NS = 1,
    ZPSE_COL_FLOAT_SPARSE = 2,
    ZPSE_COL_VAR_STRING = 3,
    ZPSE_COL_SPARSE_MIX = 4,
    ZPSE_COL_GEO_POINT = 5,
    ZPSE_COL_STRUCT_OFFSETS = 6
};

/* The name each column is filed under, in every harness. */
static const char *const zpse_col_names[ZPSE_COL_KINDS] = {
    "ts_seq", "ts_ns", "float_sparse", "var_string",
    "sparse_mix", "geo_point", "struct_offsets"
};

/* One seed per column. Fixed here, not derived from a harness's PRNG
 * state, so that every harness agrees on the bytes and a reordering of
 * the columns cannot change one of them. */
static const uint64_t zpse_col_seeds[ZPSE_COL_KINDS] = {
    UINT64_C(0x9E3779B97F4A7C15), /* ts_seq */
    UINT64_C(0xC2B2AE3D27D4EB4F), /* ts_ns */
    UINT64_C(0x165667B19E3779F9), /* float_sparse */
    UINT64_C(0x27D4EB2F165667C5), /* var_string */
    UINT64_C(0x85EBCA77C2B2AE63), /* sparse_mix */
    UINT64_C(0xD6E8FEB86659FD93), /* geo_point */
    UINT64_C(0xA0761D6478BD642F)  /* struct_offsets */
};

/* SplitMix64's finalizer: a bijection on 64 bits, so distinct record
 * indices give distinct hashes. A plain multiply-shift would leave the
 * low bits of a count poorly distributed, and the low bits are exactly
 * the ones a jitter term uses. */
static inline uint64_t zpse_col_mix(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static inline void zpse_col_put32le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static inline void zpse_col_put64le(uint8_t *p, uint64_t v)
{
    int k;
    for (k = 0; k < 8; k++) p[k] = (uint8_t)(v >> (8 * k));
}

/* ---- one record per column ---------------------------------------- */

/* ts_seq: microsecond timestamps, one millisecond apart, jittered by
 * up to half a millisecond. The jitter is bounded by half the step,
 * which keeps the series strictly increasing: the column is "sorted
 * timestamps", and a codec that assumes monotonicity has to be able to
 * rely on it. */
static inline unsigned zpse_col_rec_ts_seq(uint64_t idx, uint64_t seed,
                                            uint8_t *r)
{
    uint64_t step = 1000u;
    uint64_t jit = zpse_col_mix(seed ^ (idx * UINT64_C(0x2545F4914F6CDD1D))) %
                   (step / 2u);
    uint64_t t = UINT64_C(1700000000000000) + idx * step + jit;
    zpse_col_put64le(r, t);
    return 8u;
}

/* ts_ns: nanosecond timestamps, one microsecond apart, jittered by up
 * to 256 ns. The jitter is small next to the step on purpose: a
 * high-frequency feed's inter-arrival time is close to constant, and
 * that is the property a double-delta codec is built to exploit. */
static inline unsigned zpse_col_rec_ts_ns(uint64_t idx, uint64_t seed,
                                           uint8_t *r)
{
    uint64_t step = 1000u;
    uint64_t jit = zpse_col_mix(seed ^ (idx * UINT64_C(0x9E3779B97F4A7C15))) % 256u;
    uint64_t t = UINT64_C(1700000000000000000) + idx * step + jit;
    zpse_col_put64le(r, t);
    return 8u;
}

/* float_sparse: f32 readings, 78% of them off a ten-value ladder (a
 * sensor reporting a percentile or a plateau) and the rest unique. The
 * choice is made per RUN of eight records, not per record, so a
 * repeated value really is repeated: that is what a measurement does,
 * and it is what a XOR-of-consecutive-values codec is built for. */
static inline unsigned zpse_col_rec_float_sparse(uint64_t idx, uint64_t seed,
                                                 uint8_t *r)
{
    static const uint32_t ladder[10] = {
        0x00000000u, /* 0.0f */
        0x3F800000u, /* 1.0f */
        0x3F000000u, /* 0.5f */
        0x3E800000u, /* 0.25f */
        0x42C80000u, /* 100.0f */
        0x42280000u, /* 42.0f */
        0xBF800000u, /* -1.0f */
        0x3A83126Fu, /* 1e-3f */
        0x42C7C800u, /* 99.9f */
        0x40490FDBu  /* 3.14159274f */
    };
    uint64_t run = idx / 8u;
    uint64_t hr = zpse_col_mix(seed ^ (run * UINT64_C(0xA0761D6478BD642F)));
    uint32_t bits;

    if (hr % 100u < 78u) {
        bits = ladder[(hr >> 8) % 10u];
    } else {
        /* A unique reading: a random sign, an exponent in [0.125, 4),
         * and a full mantissa. */
        uint64_t h = zpse_col_mix(seed ^ (idx * UINT64_C(0xD6E8FEB86659FD93)));
        bits = 0x3E800000u | ((uint32_t)(h >> 32) & 0x007FFFFFu) |
               ((uint32_t)(h >> 8) & 0x80000000u);
    }
    zpse_col_put32le(r, bits);
    return 4u;
}

/* sparse_mix: a nullable i64 column, 92% of the values null (eight
 * zero bytes) and the rest a full-width id. The nulls are per value,
 * not per byte, so the column has runs of whole zero records -- which
 * is what a wide analytics table with 90% nulls looks like, and what a
 * null-suppression codec is built for. */
static inline unsigned zpse_col_rec_sparse_mix(uint64_t idx, uint64_t seed,
                                                uint8_t *r)
{
    uint64_t h = zpse_col_mix(seed ^ (idx * UINT64_C(0xBF58476D1CE4E5B9)));

    if (h % 100u < 92u) {
        memset(r, 0, 8);
    } else {
        zpse_col_put64le(r, h ^ UINT64_C(0x0123456789ABCDEF));
    }
    return 8u;
}

/* geo_point: latitude and longitude in units of 1e-7 degrees, plus a
 * cell id. The position holds still for sixteen records and then moves
 * to a new spot inside a small box, so the deltas a coordinate codec
 * would take are small and mostly zero. */
static inline unsigned zpse_col_rec_geo_point(uint64_t idx, uint64_t seed,
                                               uint8_t *r)
{
    uint64_t q = idx >> 4;
    uint64_t hp = zpse_col_mix(seed ^ (q * UINT64_C(0x94D049BB133111EB)));
    int32_t lat = 377000000 + (int32_t)(hp % 400000u) - 200000;
    int32_t lon = -1224000000 + (int32_t)((hp >> 20) % 400000u) - 200000;
    uint64_t cell = UINT64_C(0x8A2FB0C2) ^
                    (zpse_col_mix(seed ^ ((idx >> 5) * UINT64_C(0xC2B2AE3D27D4EB4F))) &
                     0xFFFFu);
    zpse_col_put32le(r, (uint32_t)lat);
    zpse_col_put32le(r + 4, (uint32_t)lon);
    zpse_col_put64le(r + 8, cell);
    return 16u;
}

/* struct_offsets: i32 offsets into a byte array, strictly increasing.
 * The step is 64 bytes and the jitter is below it, so the offsets
 * never repeat and never go backwards. */
static inline unsigned zpse_col_rec_struct_offsets(uint64_t idx, uint64_t seed,
                                                    uint8_t *r)
{
    uint64_t step = 64u;
    uint64_t jit = zpse_col_mix(seed ^ (idx * UINT64_C(0x2545F4914F6CDD1D))) % step;
    uint32_t off = (uint32_t)(1024u + idx * step + jit);
    zpse_col_put32le(r, off);
    return 4u;
}

/* var_string: a JSON-ish record per line. The keys repeat (a
 * dictionary encoder gets them for free) but the values are unique, so
 * the column is high-cardinality text: the case where a block codec
 * with a match finder wins and a predictor-based one does not. The
 * record length varies with the length of the hex id, so the record
 * boundaries are not on a grid. */
static inline unsigned zpse_col_rec_var_string(uint64_t idx, uint64_t seed,
                                                uint8_t *r)
{
    static const char hex[] = "0123456789abcdef";
    uint64_t h = zpse_col_mix(seed ^ (idx * UINT64_C(0xD6E8FEB86659FD93)));
    uint64_t h2 = zpse_col_mix(h);
    unsigned idlen = 12u + (unsigned)(h % 9u);
    unsigned n = 0;
    unsigned k;
    int written;

    written = snprintf((char *)r, 200u, "{\"id\":\"");
    if (written < 0) return 0u;
    n = (unsigned)written;
    for (k = 0; k < idlen && n < 190u; k++) {
        uint64_t v = ((k & 7u) == 0u) ? h2 : zpse_col_mix(h2 + k);
        r[n++] = (uint8_t)hex[v & 15u];
    }
    written = snprintf((char *)r + n, 200u - n,
                       "\",\"host\":\"node-%04u.example.net\","
                       "\"path\":\"/api/v2/tenants/%u/items/%u\","
                       "\"ms\":%u,\"ok\":%s}\n",
                       (unsigned)(h % 10000u), (unsigned)((h >> 8) % 100u),
                       (unsigned)((h >> 16) % 100000u), (unsigned)(h2 % 10000u),
                       ((h >> 40) & 1u) ? "true" : "false");
    if (written < 0) return 0u;
    n += (unsigned)written;
    return n;
}

/* ---- the generator ------------------------------------------------- */

/* A column in progress. The state is only the record counter and the
 * record currently being handed out, so a fill can be split anywhere
 * without changing the stream. */
typedef struct ZpseColGen {
    int kind;
    uint64_t seed;
    uint64_t ridx;   /* records produced so far */
    uint64_t rpos;   /* bytes of the current record already handed out */
    uint32_t rlen;   /* length of the current record */
    uint8_t rec[256];
} ZpseColGen;

static inline void zpse_col_gen_init(ZpseColGen *g, int kind)
{
    memset(g, 0, sizeof *g);
    g->kind = kind;
    g->seed = zpse_col_seeds[kind];
}

/* Produce the record `idx` of the column into `r`, returning its
 * length. */
static inline unsigned zpse_col_record(int kind, uint64_t seed, uint64_t idx,
                                       uint8_t *r)
{
    switch (kind) {
    case ZPSE_COL_TS_SEQ: return zpse_col_rec_ts_seq(idx, seed, r);
    case ZPSE_COL_TS_NS: return zpse_col_rec_ts_ns(idx, seed, r);
    case ZPSE_COL_FLOAT_SPARSE: return zpse_col_rec_float_sparse(idx, seed, r);
    case ZPSE_COL_VAR_STRING: return zpse_col_rec_var_string(idx, seed, r);
    case ZPSE_COL_SPARSE_MIX: return zpse_col_rec_sparse_mix(idx, seed, r);
    case ZPSE_COL_GEO_POINT: return zpse_col_rec_geo_point(idx, seed, r);
    default: return zpse_col_rec_struct_offsets(idx, seed, r);
    }
}

/* Append n bytes of the column to `out`. The stream depends on the
 * record index and the seed only, so the caller may split the fill
 * wherever it likes. */
static inline void zpse_col_gen_fill(ZpseColGen *g, uint8_t *out, size_t n)
{
    while (n > 0) {
        size_t take;

        if (g->rpos == (uint64_t)g->rlen) {
            g->rlen = zpse_col_record(g->kind, g->seed, g->ridx, g->rec);
            g->ridx++;
            g->rpos = 0;
            if (g->rlen == 0u) {
                /* A record of no bytes would make this loop spin. No
                 * column produces one, but the guard keeps a future
                 * shape from hanging the benchmark. */
                memset(out, 0, n);
                return;
            }
        }
        take = (size_t)((uint64_t)g->rlen - g->rpos);
        if (take > n) take = n;
        memcpy(out, g->rec + (size_t)g->rpos, take);
        g->rpos += take;
        out += take;
        n -= take;
    }
}

#endif /* ZPSE_COLUMNAR_GEN_H */
