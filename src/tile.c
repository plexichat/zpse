/* NOTE: tile.h names `struct ZpseTileHeader` without including types.h,
 * so the tag is forward-declared at file scope before it; types.h
 * defines the tag below. This makes all declarations agree. */
struct ZpseTileHeader;

#include "zpse/tile.h"

#include <stdint.h>
#include <string.h>

#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/mask.h"
#include "zpse/residual.h"
#include "zpse/status.h"
#include "zpse/types.h"
#include "zpse/vector.h"

/* Tile framing + reconstruct path (SPEC Sec. 6.2, 7-10).
 * Record layout (record_len == tile_record_size):
 *   [8-B header][1-B C iff predictor==9][mask][residual]
 *
 * The hot loops are organised by predictor *family* so the per-position
 * predictor dispatch happens once per tile, not once per byte:
 *
 *   FAM_CONST    0            x_hat == 0
 *   FAM_CONSTC   9            x_hat == C
 *   FAM_LANEHIST 1, 2, 10     x_hat from 8 lane histories (stateful)
 *   FAM_STRIDE   5, 6, 7, 8   x_hat == stream[abs - S]
 *   FAM_ORDER1   3            x_hat == order1[lane][last] (stateful)
 *   FAM_ORDER2   4            x_hat == o2val[lane][hash] (stateful)
 *
 * FAM_CONST, FAM_CONSTC and FAM_STRIDE never read the lane histories or
 * the ORDER tables, so their loops carry no cross-position state: the
 * history update mandated by SPEC Sec. 7 is applied by an O(8) epilogue
 * (tile_history_fixup) instead of an O(Nt) in-loop update. Every family
 * produces bit-identical predictions to the scalar formulas in Sec. 7.1.
 */

/* Predictor families. */
enum {
    TILE_FAM_CONST = 0,
    TILE_FAM_LANEHIST,
    TILE_FAM_STRIDE,
    TILE_FAM_ORDER1,
    TILE_FAM_ORDER2
};

static int tile_family(uint8_t predictor_id)
{
    switch (predictor_id) {
    case ZPSE_P_CONST_ZERO:
        return TILE_FAM_CONST;
    case ZPSE_P_LANE_PREVIOUS:
    case ZPSE_P_LANE_DELTA:
    case ZPSE_P_LANE_AVERAGE:
        return TILE_FAM_LANEHIST;
    case ZPSE_P_ORDER1:
        return TILE_FAM_ORDER1;
    case ZPSE_P_ORDER2:
        return TILE_FAM_ORDER2;
    case ZPSE_P_TILE_CONSTANT:
        return TILE_FAM_CONST; /* handled with constant_C */
    default:
        return TILE_FAM_STRIDE; /* 5..8 */
    }
}

static int tile_parse_header(const uint8_t *record, uint32_t record_len,
                             struct ZpseTileHeader *th)
{
    if (record_len < (uint32_t)ZPSE_TILE_HEADER_SIZE)
        return PSE_BAD_TILE_SIZE;
    th->tile_flags = record[0];
    th->predictor_id = record[1];
    th->tile_length_minus_1 = record[2];
    th->exception_count = record[3];
    th->mask_payload_size = zpse_load_u16le(record + 4);
    th->residual_payload_size = zpse_load_u16le(record + 6);
    return PSE_OK;
}

static uint32_t tile_record_need(const struct ZpseTileHeader *th)
{
    return (uint32_t)8 +
        (th->predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT ? (uint32_t)1 : (uint32_t)0) +
        (uint32_t)th->mask_payload_size +
        (uint32_t)th->residual_payload_size;
}

static uint16_t tile_cap16(uint32_t n)
{
    return n > (uint32_t)0xFFFF ? (uint16_t)0xFFFF : (uint16_t)n;
}

/* Encoder-side choice of C for TILE_LOCAL_CONSTANT: most frequent byte
 * (first-max wins ties). Any C is decodable; this maximizes hits. */
static uint8_t tile_pick_constant(const uint8_t *actual, uint16_t nt)
{
    uint16_t hist[256];
    uint16_t i;
    uint16_t best = 0;

    memset(hist, 0, sizeof hist);
    for (i = 0; i < nt; i++)
        hist[actual[i]]++;
    for (i = 1; i < 256; i++) {
        if (hist[i] > hist[best])
            best = i;
    }
    return (uint8_t)best;
}

static uint32_t tile_stride_of(uint8_t predictor_id)
{
    switch (predictor_id) {
    case ZPSE_P_STRIDE_1:
        return 1u;
    case ZPSE_P_STRIDE_2:
        return 2u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u;
    }
}

/* Epilogue for the stateless families (CONST, CONSTC, STRIDE): apply the
 * SPEC Sec. 7 history update for every lane in one O(8) pass. `src` holds
 * the Nt actual bytes (encoder input plane or decoder output plane);
 * ent_last/ent_second are the histories captured before the tile ran. */
static void tile_history_fixup(ZpseLaneState *lanes, const uint8_t *src,
                               uint16_t nt, const uint8_t ent_last[8],
                               const uint8_t ent_second[8])
{
    unsigned l;

    for (l = 0; l < 8u; l++) {
        if (l >= (unsigned)nt) {
            lanes->last[l] = ent_last[l];
            lanes->second_last[l] = ent_second[l];
            continue;
        }
        {
            unsigned lastp = l +
                (((unsigned)nt - 1u - l) / 8u) * 8u;
            lanes->last[l] = src[lastp];
            lanes->second_last[l] = (lastp >= 8u) ? src[lastp - 8u]
                                                  : ent_last[l];
        }
    }
}

static void tile_capture_hist(const ZpseLaneState *lanes, uint8_t last[8],
                              uint8_t second[8])
{
    memcpy(last, lanes->last, sizeof *last * 8u);
    memcpy(second, lanes->second_last, sizeof *second * 8u);
}

/* LSB-first bit reader (mirrors residual_get_bits in residual.c).
 *
 * The single caller reads a packed run of 6-bit signed deltas from a
 * payload whose length was validated against exactly those fields, so a
 * field spans at most one byte boundary and the byte it reaches is
 * always inside the payload: two shifts and an AND, no loop. */
static unsigned tile_get_bits(const uint8_t *s, uint32_t bitpos, unsigned w)
{
    uint32_t byte = bitpos >> 3u;
    unsigned off = (unsigned)(bitpos & 7u);
    unsigned v = (unsigned)s[byte] >> off;

    if (off + w > 8u) {
        v |= (unsigned)s[byte + 1u] << (8u - off);
    }
    return v & ((1u << w) - 1u);
}

/* Pred-free extraction of SIGNED_SMALL_DELTA units. Validation mirrors
 * residual_decode_mode4 exactly (length equation, submode range, zero
 * unit rejection, tail pad bits); only the pred-dependent x mapping is
 * deferred to the online loop, where the true online prediction is
 * known. */
static int tile_unpack_deltas(const uint8_t *payload, uint16_t payload_len,
                              uint16_t k, int8_t *deltas)
{
    uint8_t sub;
    uint32_t expect;
    uint16_t i;

    if (k > 0u && (payload == 0 || deltas == 0))
        return PSE_BAD_ARG;
    if (payload_len < 1u)
        return PSE_BAD_RESIDUAL;
    sub = payload[0];
    if (sub > 2u)
        return PSE_BAD_RESIDUAL;
    if (sub == 0u)
        expect = 1u + (((uint32_t)k + 1u) / 2u);
    else if (sub == 1u)
        expect = 1u + ((6u * (uint32_t)k + 7u) / 8u);
    else
        expect = 1u + (uint32_t)k;
    if ((uint32_t)payload_len != expect)
        return PSE_BAD_RESIDUAL;
    if (sub == 0u) {
        if (((k & 1u) != 0u) && ((payload[1u + (k >> 1u)] >> 4u) != 0u))
            return PSE_BAD_RESIDUAL;
        for (i = 0; i < k; i++) {
            uint8_t byte = payload[1u + (i >> 1u)];
            unsigned enc = ((i & 1u) == 0u) ? (byte & 0xFu) : (byte >> 4u);
            if (enc == 0u)
                return PSE_BAD_RESIDUAL;
            deltas[i] = (int8_t)((enc < 8u) ? (int)enc : (int)enc - 16);
        }
    } else if (sub == 1u) {
        const uint8_t *s = payload + 1u;
        uint32_t total_bits = 6u * (uint32_t)k;
        if (((total_bits & 7u) != 0u) && (expect > 1u) &&
            ((s[expect - 2u] >> (total_bits & 7u)) != 0u))
            return PSE_BAD_RESIDUAL;
        for (i = 0; i < k; i++) {
            unsigned enc = tile_get_bits(s, (uint32_t)i * 6u, 6u);
            if (enc == 0u)
                return PSE_BAD_RESIDUAL;
            deltas[i] = (int8_t)((enc < 32u) ? (int)enc : (int)enc - 64);
        }
    } else {
        for (i = 0; i < k; i++) {
            uint8_t enc = payload[1u + i];
            if (enc == 0u)
                return PSE_BAD_RESIDUAL;
            deltas[i] = (enc < 128u) ? (int8_t)enc :
                                       (int8_t)((int)enc - 256);
        }
    }
    return PSE_OK;
}

int zpse_tile_validate(const struct ZpseTileHeader *th, uint16_t nt_expected,
                       uint32_t bytes_remaining, uint16_t *nt_out,
                       uint16_t *k_out)
{
    uint8_t mask_mode;
    uint8_t res_mode;
    uint32_t nt;
    uint16_t k = 0;
    uint32_t need;
    int rc;

    if (th == 0)
        return PSE_BAD_ARG;
    mask_mode = zpse_tile_mask_mode(th->tile_flags);
    res_mode = zpse_tile_res_mode(th->tile_flags);
    if (th->tile_flags & ZPSE_TILE_FLAG_RESERVED_BIT)
        return PSE_UNSUPPORTED_FEATURE; /* bit 6 MUST be 0 */
    if (mask_mode > (uint8_t)ZPSE_M_ALL_EXCEPTION)
        return PSE_BAD_MASK;
    if (res_mode > (uint8_t)ZPSE_R_SIGNED_DELTA)
        return PSE_BAD_RESIDUAL; /* modes 5..7 reserved */
    if (th->predictor_id > (uint8_t)ZPSE_P_LANE_AVERAGE)
        return PSE_BAD_PREDICTOR;
    nt = (uint32_t)th->tile_length_minus_1 + (uint32_t)1;
    if (nt != (uint32_t)nt_expected)
        return PSE_BAD_TILE_SIZE;
    rc = zpse_mask_derive_k(mask_mode, th->exception_count, (uint16_t)nt, &k);
    if (rc != PSE_OK)
        return rc;
    need = tile_record_need(th);
    if (need > bytes_remaining)
        return PSE_BAD_TILE_SIZE;
    if (nt_out != 0)
        *nt_out = (uint16_t)nt;
    if (k_out != 0)
        *k_out = k;
    return PSE_OK;
}

/* ------------------------------------------------------------------ *
 * Exception classification (shared shape for both directions)         *
 * ------------------------------------------------------------------ */

/* Record an exception at position p: set the mask bit, remember the
 * prediction (the residual packers need it) and fold the position into
 * the residual statistics that drive mode selection. Doing this inline
 * keeps the mode search O(1) per tile instead of a second gather pass. */
/* Record one exception. Written to compile to straight-line code: no
 * branch for the distinct-residual count (cl->nu is the popcount of
 * cl->seen, computed once after the loop) and min/max guards that the
 * compiler turns into cmovs. dmin/dmax start at the extremes of the
 * reachable range so the first exception needs no special case; they
 * are only read by tile_res_len when k > 0, where the initial values
 * can no longer survive. */
#define TILE_TAKE_EXC(p_, xhat_, x_)                                    \
    do {                                                                \
        uint16_t _pp = (uint16_t)(p_);                                  \
        uint8_t _xh = (uint8_t)(xhat_);                                 \
        uint8_t _x = (uint8_t)(x_);                                     \
        uint8_t _r = zpse_exclusion_forward(_x, _xh);                   \
        uint8_t _db = (uint8_t)((((unsigned)_x - (unsigned)_xh + 128u)  \
                                 & 0xFFu));                             \
        words[_pp >> 5] |= (uint32_t)1u << (_pp & 31u);                 \
        cl->at_pred[k] = _xh;                                           \
        cl->res_at_exc[k] = _r;                                         \
        cl->seen[_r >> 5] |= (uint32_t)1u << (_r & 31u);                \
        if (_r > _rmax)                                                 \
            _rmax = _r;                                                 \
        if (_db < _dmin)                                                \
            _dmin = _db;                                                \
        if (_db > _dmax)                                                \
            _dmax = _db;                                                \
        k++;                                                            \
    } while (0)

/* Mark position p as an exception without touching the residual
 * statistics. The vector classifier uses this to build the mask in
 * position order across its prologue/vector/tail, then makes exactly one
 * ascending pass over the finished mask to do the per-exception
 * bookkeeping -- recording here as well would duplicate entries. */
#define TILE_SET_EXC_BIT(p_)                                            \
    do {                                                                \
        uint16_t _pp = (uint16_t)(p_);                                  \
        words[_pp >> 5] |= (uint32_t)1u << (_pp & 31u);                 \
    } while (0)

/* Number of one bits in a 32-bit word (same idiom as mask.c). */
static int tile_popcount32(uint32_t w)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(w);
#else
    int n = 0;

    while (w != 0u) {
        w &= w - 1u;
        n++;
    }
    return n;
#endif
}

/* ------------------------------------------------------------------ *
 * Vectorised exception discovery for the classification pass.
 *
 * The encoder spends most of its time here: for every (tile, transform
 * trial, candidate predictor) the scalar path walks all Nt positions to
 * compare a prediction. But the predictions of the CONST, LANEHIST and
 * STRIDE families are pure functions of the plane bytes, so the whole
 * comparison is 32 positions at a time -- and only the positions that
 * actually differ need the per-exception bookkeeping.
 *
 * The identities used below hold for p >= 16 because every tile resets
 * its lane state (SPEC Sec. 7), which makes the history at position p
 * exactly the tile's own bytes 8 and 16 back:
 *   last[l] = a[p-8+l],  second[l] = a[p-16+l]
 * Below 16 the entry state is `init`, not the plane, so [0,16) stays on
 * a scalar prologue that mirrors the scalar branches exactly. Note this
 * boundary is anchored by the reset and NOT by seg_pos: moving it was
 * tried and is wrong.
 *
 * Same per-function-target structure as the AVX2 prescreen in segment.c:
 * GCC/clang enable AVX2 for these routines alone, MSVC only when the
 * whole translation unit is built with /arch:AVX2 (which defines
 * __AVX2__), otherwise the SIMD version is compiled out and the scalar
 * reference below is used. Execution is additionally gated on
 * zpse_cpu_caps(), so the scalar path stays reachable and
 * tests/test_parity.c can require the two to agree.
 * ------------------------------------------------------------------ */
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_TILE_AVX2 __attribute__((target("avx2")))
#define ZPSE_TILE_HAVE_AVX2 1
#elif defined(__AVX2__)
#define ZPSE_TILE_AVX2
#define ZPSE_TILE_HAVE_AVX2 1
#else
#define ZPSE_TILE_AVX2
#define ZPSE_TILE_HAVE_AVX2 0
#endif
#else
#define ZPSE_TILE_HAVE_AVX2 0
#endif

#if ZPSE_TILE_HAVE_AVX2
#include <immintrin.h>

/* Index of the lowest set bit. Caller guarantees w != 0. */
static unsigned tile_ctz32(uint32_t w)
{
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_ctz(w);
#else
    unsigned n = 0;

    while ((w & 1u) == 0u) {
        w >>= 1;
        n++;
    }
    return n;
#endif
}

/* The families whose prediction is a pure function of the plane bytes.
 * ORDER1/ORDER2 read lane-indexed tables, so they stay scalar. */
static int tile_fam_is_vectorisable(int fam)
{
    return fam == TILE_FAM_CONST || fam == TILE_FAM_LANEHIST ||
           fam == TILE_FAM_STRIDE;
}

/* Staging buffer: 16 head + up to 256 tile + 32 pad. */
#define ZPSE_TILE_STG_LEN 304u

/* Accumulate one 32-byte chunk's exception bits into `words`. `p` is a
 * multiple of 32, so the 32 mask bits land in exactly one word and no
 * straddling shift is needed; `MASK` is all ones for a full chunk and the
 * validity vector for the last, partial one. */
#define TILE_CL_CHUNK(v_, pv_, MASK_)                                       \
    do {                                                                    \
        __m256i _eq = _mm256_cmpeq_epi8(v_, pv_);                           \
        __m256i _ex = _mm256_and_si256(                                     \
            _mm256_xor_si256(_eq, _mm256_set1_epi8((char)-1)), MASK_);      \
        words[(uint32_t)p >> 5] |= (uint32_t)_mm256_movemask_epi8(_ex);     \
    } while (0)

/* Classify one vectorisable tile: fills cl->words / cl->k / cl->at_pred /
 * cl->res_at_exc / cl->seen / cl->rmax / cl->dmin / cl->dmax / cl->nu and
 * advances the lane state to exactly the state the scalar path leaves.
 * Caller has already reset the lanes, zeroed words/seen and set
 * constant_C, and has captured ent_last/ent_second from AFTER that reset.
 * Requires nt >= 32, and for the STRIDE family seg_pos >= 16.
 *
 * The whole tile is one pass over a staging buffer, so there is no scalar
 * prologue, no scalar tail and no per-position lane state to track -- the
 * trap the previous prologue/body/tail split had to keep re-deriving:
 *
 *   head[0, 8)    second_last[] at tile entry
 *   head[8, 16)   last[] at tile entry
 *   body[16,..)   the tile's own bytes
 *   pad[..,+32)   zeros, so the last partial chunk can be read as full
 *
 * For the LANEHIST family those first 16 bytes are exactly the entry lane
 * state, which is what the scalar loop's `last[lane]` / `second_last[lane]`
 * hold for the positions below 8 and 16 (every tile resets its lanes, but
 * this works for a non-resetting tile too because the caller captures the
 * state this tile actually starts from). Past 16 the identities
 * last[lane] == a[p-8] and second_last[lane] == a[p-16] hold, which is the
 * same thing the previous scalar body relied on.
 *
 * For the STRIDE family the head instead holds the 16 real plane bytes
 * before the tile, because `seg_base[d - s]` is a plain `a[p - s]` that
 * reads them for p < s. That is why the caller only takes this path when
 * seg_pos >= 16: it makes the scalar `d < s` fallback to `init`
 * unreachable. This function therefore requires a == seg_base + seg_pos,
 * which is how zpse_segment_encode calls it (and the only caller). */
static ZPSE_TILE_AVX2 void tile_classify_avx2(ZpseLaneState *lanes,
                                              const uint8_t *actual,
                                              uint16_t nt,
                                              uint8_t predictor_id, int fam,
                                              uint8_t initial_literal,
                                              const uint8_t *seg_base,
                                              uint32_t seg_pos,
                                              ZpseTileClass *cl,
                                              const uint8_t ent_last[8],
                                              const uint8_t ent_second[8])
{
    uint32_t *words = cl->words;
    uint8_t stg[ZPSE_TILE_STG_LEN];
    uint32_t s = 0u;
    uint8_t c = 0u;
    uint16_t k = 0;
    uint16_t p = 0;
    uint8_t _rmax = 0;
    uint8_t _dmin = 255u;
    uint8_t _dmax = 0u;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i all = _mm256_cmpeq_epi8(zero, zero);

    (void)seg_base;
    (void)seg_pos;
    (void)initial_literal;

    if (fam == TILE_FAM_CONST) {
        c = (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
                ? cl->constant_C : 0u;
    } else if (fam == TILE_FAM_STRIDE) {
        s = tile_stride_of(predictor_id);
    }

    if (fam == TILE_FAM_STRIDE) {
        /* Real preceding plane bytes; requires seg_pos >= 16. */
        memcpy(stg, actual - 16, 16);
    } else {
        memcpy(stg, ent_second, 8);
        memcpy(stg + 8, ent_last, 8);
    }
    memcpy(stg + 16, actual, nt);
    memset(stg + 16 + nt, 0, 32);

    {
        const __m256i cvec = _mm256_set1_epi8((char)c);
        const __m256i idx = _mm256_setr_epi8(
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31);

        while (p + 32u <= (uint32_t)nt) {
            __m256i v = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + 16u + p));
            __m256i pv;

            if (fam == TILE_FAM_CONST) {
                pv = cvec;
            } else if (fam == TILE_FAM_STRIDE) {
                pv = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(stg + 16u + p - s));
            } else {
                const __m256i h8 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(stg + 8u + p));
                const __m256i h16 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(stg + p));

                if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
                    pv = h8;
                } else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
                    /* (2*h8 - h16) mod 256, byte-wise. */
                    pv = _mm256_sub_epi8(_mm256_add_epi8(h8, h8), h16);
                } else { /* ZPSE_P_LANE_AVERAGE rounds up, as scalar. */
                    pv = _mm256_avg_epu8(h8, h16);
                }
            }
            TILE_CL_CHUNK(v, pv, all);
            p = (uint16_t)(p + 32u);
        }
        if (p < nt) {
            uint32_t rem = (uint32_t)nt - (uint32_t)p;
            __m256i valid = _mm256_cmpgt_epi8(
                _mm256_set1_epi8((char)rem), idx);
            __m256i v = _mm256_loadu_si256(
                (const __m256i *)(const void *)(stg + 16u + p));
            __m256i pv;

            if (fam == TILE_FAM_CONST) {
                pv = cvec;
            } else if (fam == TILE_FAM_STRIDE) {
                pv = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(stg + 16u + p - s));
            } else {
                const __m256i h8 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(stg + 8u + p));
                const __m256i h16 = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(stg + p));

                if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
                    pv = h8;
                } else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
                    pv = _mm256_sub_epi8(_mm256_add_epi8(h8, h8), h16);
                } else {
                    pv = _mm256_avg_epu8(h8, h16);
                }
            }
            TILE_CL_CHUNK(v, pv, valid);
        }
    }

    /* Only the positions that differ need the bookkeeping, and the mask
     * already says which they are: walk it ascending so at_pred and
     * res_at_exc stay in position order, which the packers rely on. The
     * prediction is re-read from the staging buffer rather than stored for
     * every position during the vector pass -- the pass would otherwise
     * write 256 bytes per tile that are thrown away whenever k is small. */
    {
        unsigned wi;

        for (wi = 0; wi < 8u; wi++) {
            uint32_t w = words[wi];

            while (w != 0u) {
                uint16_t pos = (uint16_t)(wi * 32u + tile_ctz32(w));
                uint8_t xhat;

                if (fam == TILE_FAM_CONST) {
                    xhat = c;
                } else if (fam == TILE_FAM_STRIDE) {
                    xhat = stg[16u + (uint32_t)pos - s];
                } else {
                    uint8_t h8 = stg[8u + (uint32_t)pos];
                    uint8_t h16 = stg[pos];

                    if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
                        xhat = h8;
                    } else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
                        xhat = (uint8_t)((2u * (unsigned)h8 -
                                          (unsigned)h16) & 0xFFu);
                    } else {
                        xhat = (uint8_t)(((unsigned)h8 + (unsigned)h16 + 1u)
                                         >> 1);
                    }
                }
                TILE_TAKE_EXC(pos, xhat, actual[pos]);
                w &= w - 1u;
            }
        }
    }

    /* The O(8) epilogue is the same end state the scalar per-byte
     * updates reach, for every family and every nt. */
    tile_history_fixup(lanes, actual, nt, ent_last, ent_second);

    cl->k = k;
    cl->rmax = _rmax;
    cl->dmin = _dmin;
    cl->dmax = _dmax;
    {
        unsigned i;
        uint16_t nu = 0;

        for (i = 0; i < 8u; i++)
            nu = (uint16_t)(nu + (uint16_t)tile_popcount32(cl->seen[i]));
        cl->nu = nu;
    }
}
#endif /* ZPSE_TILE_HAVE_AVX2 */

/* Classify one tile: fill cl->words / cl->k / cl->at_pred / constant_C
 * and advance the lane state exactly as the decoder will. */
static int tile_classify(ZpseLaneState *lanes, const uint8_t *actual,
                         uint16_t nt, uint8_t predictor_id,
                         uint8_t initial_literal, int tile_reset,
                         const uint8_t *seg_base, uint32_t seg_pos,
                         ZpseTileClass *cl)
{
    uint32_t *words = cl->words;
    uint16_t k = 0;
    uint16_t p;
    /* Range accumulators live in locals, not in *cl: the exception loop
     * runs up to Nt times per tile and a struct store/reload through the
     * caller's pointer per exception is pure overhead when all three are
     * single-byte values the compiler can otherwise keep in registers. */
    uint8_t _rmax = 0;
    uint8_t _dmin = 255u;
    uint8_t _dmax = 0u;
    uint8_t ent_last[8];
    uint8_t ent_second[8];
    int fam;

    if (tile_reset)
        zpse_lane_reset_for(lanes, initial_literal, predictor_id);
    memset(words, 0, sizeof cl->words);
    memset(cl->seen, 0, sizeof cl->seen);
    cl->k = 0;
    cl->nu = 0;
    cl->rmax = 0;
    cl->dmin = 255u;
    cl->dmax = 0u;
    cl->constant_C = 0;
    if (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
        cl->constant_C = tile_pick_constant(actual, nt);

    fam = tile_family(predictor_id);
    if (fam == TILE_FAM_CONST || fam == TILE_FAM_STRIDE)
        tile_capture_hist(lanes, ent_last, ent_second);

#if ZPSE_TILE_HAVE_AVX2
    if (nt >= 32u && tile_fam_is_vectorisable(fam) &&
        /* STRIDE reads the 16 real plane bytes before the tile; */
        /* LANEHIST takes them from the entry lane state. */
        (fam != TILE_FAM_STRIDE || seg_pos >= 16u) &&
        (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        uint8_t vl[8];
        uint8_t vs[8];

        tile_capture_hist(lanes, vl, vs);
        tile_classify_avx2(lanes, actual, nt, predictor_id, fam,
                           initial_literal, seg_base, seg_pos, cl, vl, vs);
        return PSE_OK;
    }
#endif

    switch (fam) {
    case TILE_FAM_CONST:
        if (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT) {
            uint8_t c = cl->constant_C;
            for (p = 0; p < nt; p++) {
                if (actual[p] != c)
                    TILE_TAKE_EXC(p, c, actual[p]);
            }
        } else {
            for (p = 0; p < nt; p++) {
                if (actual[p] != 0u)
                    TILE_TAKE_EXC(p, 0u, actual[p]);
            }
        }
        tile_history_fixup(lanes, actual, nt, ent_last, ent_second);
        break;

    case TILE_FAM_STRIDE: {
        uint32_t s = tile_stride_of(predictor_id);
        for (p = 0; p < nt; p++) {
            uint32_t abs_pos = seg_pos + (uint32_t)p;
            uint8_t xhat = (abs_pos < s) ? initial_literal
                                         : seg_base[abs_pos - s];
            if (actual[p] != xhat)
                TILE_TAKE_EXC(p, xhat, actual[p]);
        }
        tile_history_fixup(lanes, actual, nt, ent_last, ent_second);
        break;
    }

    case TILE_FAM_LANEHIST:
        if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
            for (p = 0; p < nt; p++) {
                size_t lane = (size_t)p & 7u;
                uint8_t xhat = lanes->last[lane];
                uint8_t x = actual[p];
                if (x != xhat)
                    TILE_TAKE_EXC(p, xhat, x);
                lanes->second_last[lane] = xhat;
                lanes->last[lane] = x;
            }
        } else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
            for (p = 0; p < nt; p++) {
                size_t lane = (size_t)p & 7u;
                uint8_t last = lanes->last[lane];
                unsigned xd = (2u * (unsigned)last -
                               (unsigned)lanes->second_last[lane]) & 0xFFu;
                uint8_t xhat = (uint8_t)xd;
                uint8_t x = actual[p];
                if (x != xhat)
                    TILE_TAKE_EXC(p, xhat, x);
                lanes->second_last[lane] = last;
                lanes->last[lane] = x;
            }
        } else { /* LANE_AVERAGE */
            for (p = 0; p < nt; p++) {
                size_t lane = (size_t)p & 7u;
                unsigned last = lanes->last[lane];
                uint8_t xhat =
                    (uint8_t)((last + (unsigned)lanes->second_last[lane] + 1u)
                              >> 1);
                uint8_t x = actual[p];
                if (x != xhat)
                    TILE_TAKE_EXC(p, xhat, x);
                lanes->second_last[lane] = (uint8_t)last;
                lanes->last[lane] = x;
            }
        }
        break;

    case TILE_FAM_ORDER1: {
        uint8_t *o1 = lanes->order1;
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint8_t last = lanes->last[lane];
            uint8_t xhat = o1[lane * 256u + (size_t)last];
            uint8_t x = actual[p];
            if (x != xhat)
                TILE_TAKE_EXC(p, xhat, x);
            lanes->second_last[lane] = last;
            lanes->last[lane] = x;
            o1[lane * 256u + (size_t)last] = x;
        }
        break;
    }

    default: { /* TILE_FAM_ORDER2 */
        uint8_t *val = lanes->o2val;
        uint8_t *conf = lanes->o2conf;
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint8_t last = lanes->last[lane];
            uint8_t second = lanes->second_last[lane];
            size_t off = lane * 4096u + (size_t)zpse_order2_index(second, last);
            uint8_t xhat = val[off];
            uint8_t x = actual[p];
            if (x != xhat)
                TILE_TAKE_EXC(p, xhat, x);
            if (conf[off] == 0u) {
                val[off] = x;
                conf[off] = 1u;
            } else if (x == xhat) {
                if (conf[off] < 255u)
                    conf[off]++;
            } else {
                conf[off]--;
            }
            lanes->second_last[lane] = last;
            lanes->last[lane] = x;
        }
        break;
    }
    }

    cl->k = k;
    cl->rmax = _rmax;
    cl->dmin = _dmin;
    cl->dmax = _dmax;
    /* cl->nu is exactly the number of distinct exclusion residuals, so
     * it is the popcount of the 256-bit set the loop just built. Doing
     * it once here is much cheaper than a test-and-increment per
     * exception inside the byte loop, and gives the same value that
     * choose_modes' palette sizing depends on. */
    {
        unsigned l;
        uint16_t nu = 0;

        for (l = 0; l < 8u; l++)
            nu = (uint16_t)(nu + (uint16_t)tile_popcount32(cl->seen[l]));
        cl->nu = nu;
    }
    return PSE_OK;
}

int zpse_tile_classify(ZpseLaneState *lanes, const uint8_t *actual,
                       uint16_t nt, uint8_t predictor_id,
                       uint8_t initial_literal, int tile_reset,
                       const uint8_t *seg_base, uint32_t seg_pos,
                       ZpseTileClass *cl)
{
    if (lanes == 0 || actual == 0 || cl == 0)
        return PSE_BAD_ARG;
    if (nt == 0u || nt > ZPSE_MAX_TILE_LEN)
        return PSE_BAD_TILE_SIZE;
    if (predictor_id > (uint8_t)ZPSE_P_LANE_AVERAGE)
        return PSE_BAD_PREDICTOR;
    if (tile_family(predictor_id) == TILE_FAM_STRIDE && seg_base == 0)
        return PSE_BAD_ARG;
    return tile_classify(lanes, actual, nt, predictor_id, initial_literal,
                         tile_reset, seg_base, seg_pos, cl);
}

/* ------------------------------------------------------------------ *
 * Record packing                                                      *
 * ------------------------------------------------------------------ */

static int tile_check_modes(uint8_t mask_mode, uint8_t res_mode, uint16_t nt,
                            uint16_t k)
{
    if (mask_mode > (uint8_t)ZPSE_M_ALL_EXCEPTION)
        return PSE_BAD_MASK;
    if (res_mode > (uint8_t)ZPSE_R_SIGNED_DELTA)
        return PSE_BAD_RESIDUAL;
    if (k > nt)
        return PSE_BAD_MASK;
    if (mask_mode == (uint8_t)ZPSE_M_ALL_HIT) {
        if (k != 0u)
            return PSE_BAD_MASK;
    } else if (mask_mode == (uint8_t)ZPSE_M_ALL_EXCEPTION) {
        if (k != nt)
            return PSE_BAD_MASK;
    } else {
        if (k < 1u || k >= nt)
            return PSE_BAD_MASK;
        if (mask_mode == (uint8_t)ZPSE_M_ENUMERATIVE && nt < 2u)
            return PSE_BAD_MASK;
    }
    return PSE_OK;
}

/* Base-M packing group for the mode-3 length estimate. Mirrors the
 * normative Sec. 9.2 table via the shared helper in residual.c. */
static uint16_t tile_res_len(uint8_t res_mode, const ZpseTileClass *cl,
                             uint16_t nt, uint16_t k)
{
    uint32_t kk = k;

    (void)nt;
    switch (res_mode) {
    case ZPSE_R_EXCLUSION_U8:
    case ZPSE_R_RAW_ACTUAL:
        return (uint16_t)kk;
    case ZPSE_R_PALETTE_BITS: {
        unsigned m = cl->nu < 2u ? 2u : (unsigned)cl->nu;
        unsigned b = zpse_ceil_log2_u32(m);
        if (cl->nu > 16u)
            return 0xFFFFu;
        return (uint16_t)(1u + m + (((uint32_t)k * b + 7u) / 8u));
    }
    case ZPSE_R_PACKED_BASE_M: {
        uint8_t m;
        uint8_t g;
        if (k == 0u)
            return 1u;
        m = zpse_residual_mode3_pick_m(cl->rmax);
        if (m == 0u)
            return 0xFFFFu;
        g = zpse_base_m_group(m);
        return (uint16_t)(1u + (((uint32_t)k + g - 1u) / g) * 4u);
    }
    default: { /* SIGNED_DELTA */
        int dmin = (int)cl->dmin - 128;
        int dmax = (int)cl->dmax - 128;
        if (k == 0u)
            return 1u;
        if (dmin >= -8 && dmax <= 7)
            return (uint16_t)(1u + ((kk + 1u) / 2u));
        if (dmin >= -32 && dmax <= 31)
            return (uint16_t)(1u + ((6u * kk + 7u) / 8u));
        return (uint16_t)(1u + kk);
    }
    }
}

/* Pick the cheapest legal (mask_mode, res_mode) pair for a finished
 * classification. Mask and residual payload lengths are independent, so
 * each side is minimised on its own -- no encode-and-measure search.
 * Ties prefer the lower mode id, matching the historical exhaustive
 * search's first-wins rule. */
void zpse_tile_choose_modes(const ZpseTileClass *cl, uint16_t nt,
                            uint8_t *mask_mode_out, uint8_t *res_mode_out)
{
    uint16_t k = cl->k;
    uint32_t rb = ((uint32_t)nt + 7u) >> 3;
    uint8_t best_m;
    uint8_t best_r = 0u;
    uint16_t best_rlen;
    uint8_t rm;

    if (k == 0u) {
        best_m = (uint8_t)ZPSE_M_ALL_HIT;
    } else if (k == nt) {
        best_m = (uint8_t)ZPSE_M_ALL_EXCEPTION;
    } else if ((uint32_t)k <= rb) {
        best_m = (uint8_t)ZPSE_M_SPARSE_U8;
    } else {
        best_m = (uint8_t)ZPSE_M_RAW_BITMAP;
    }

    best_rlen = tile_res_len(0u, cl, nt, k);
    for (rm = 1u; rm <= (uint8_t)ZPSE_R_SIGNED_DELTA; rm++) {
        uint16_t l = tile_res_len(rm, cl, nt, k);
        if (l < best_rlen) {
            best_rlen = l;
            best_r = rm;
        }
    }

    *mask_mode_out = best_m;
    *res_mode_out = best_r;
}

int zpse_tile_pack(uint8_t predictor_id, const ZpseTileClass *cl,
                   uint16_t nt, const uint8_t *actual,
                   uint8_t mask_mode, uint8_t res_mode,
                   int tile_reset, uint8_t *record_out,
                   uint32_t record_cap, uint8_t *comb_scratch)
{
    const uint32_t *words;
    uint16_t k;
    uint8_t avals[256];
    uint8_t ec;
    uint32_t off;
    int need_vals;
    int need_act;
    int use_vals = 0;
    int use_act = 0;
    int rc;
    int mrc;
    int rrc;

    if (cl == 0 || actual == 0 || record_out == 0)
        return PSE_BAD_ARG;
    if (nt == 0u || nt > ZPSE_MAX_TILE_LEN)
        return PSE_BAD_TILE_SIZE;
    if (predictor_id > (uint8_t)ZPSE_P_LANE_AVERAGE)
        return PSE_BAD_PREDICTOR;
    if (record_cap < (uint32_t)ZPSE_TILE_HEADER_SIZE)
        return PSE_OUTPUT_OVERFLOW;

    words = cl->words;
    k = cl->k;
    rc = tile_check_modes(mask_mode, res_mode, nt, k);
    if (rc != PSE_OK)
        return rc;

    /* The two (mask, residual) pairs at the extremes of the exception
     * count have empty mask payloads and a residual payload that is
     * either empty or a verbatim copy, which is where a large share of
     * tiles lands on run-length-friendly data. Both arms below write the
     * bytes the generic path writes and report the same statuses, in the
     * same order, only without walking the tile again. */
    if (k == 0u && mask_mode == (uint8_t)ZPSE_M_ALL_HIT &&
        res_mode == (uint8_t)ZPSE_R_EXCLUSION_U8) {
        uint32_t total = (uint32_t)ZPSE_TILE_HEADER_SIZE +
                         (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT
                              ? 1u : 0u);

        if (total > record_cap)
            return PSE_OUTPUT_OVERFLOW;
        if (zpse_mask_count_bits(words, nt) != 0)
            return PSE_BAD_MASK;
        record_out[0] = zpse_tile_flags_make(mask_mode, res_mode,
                                             tile_reset ? 1 : 0);
        record_out[1] = predictor_id;
        record_out[2] = (uint8_t)(nt - (uint16_t)1);
        record_out[3] = 0u; /* ec: neither ALL_HIT nor ALL_EXCEPTION */
        zpse_store_u16le(record_out + 4, 0u); /* mask payload */
        zpse_store_u16le(record_out + 6, 0u); /* residual payload */
        if (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
            record_out[ZPSE_TILE_HEADER_SIZE] = cl->constant_C;
        return (int)total;
    }
    if (k == nt && mask_mode == (uint8_t)ZPSE_M_ALL_EXCEPTION &&
        res_mode == (uint8_t)ZPSE_R_EXCLUSION_U8) {
        uint32_t base = (uint32_t)ZPSE_TILE_HEADER_SIZE +
                        (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT
                             ? 1u : 0u);
        uint32_t total = base + (uint32_t)k;
        uint16_t i;

        if (total > record_cap)
            return PSE_OUTPUT_OVERFLOW;
        if (zpse_mask_count_bits(words, nt) != (int)nt)
            return PSE_BAD_MASK;
        for (i = 0; i < k; i++) {
            uint8_t v = cl->res_at_exc[i];

            if (v > 254u)
                return PSE_BAD_RESIDUAL;
            record_out[base + i] = v;
        }
        record_out[0] = zpse_tile_flags_make(mask_mode, res_mode,
                                             tile_reset ? 1 : 0);
        record_out[1] = predictor_id;
        record_out[2] = (uint8_t)(nt - (uint16_t)1);
        record_out[3] = 0u;
        zpse_store_u16le(record_out + 4, 0u);
        zpse_store_u16le(record_out + 6, (uint16_t)k);
        if (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
            record_out[ZPSE_TILE_HEADER_SIZE] = cl->constant_C;
        return (int)total;
    }

    if (mask_mode == (uint8_t)ZPSE_M_ALL_HIT) {
        ec = 0;
    } else if (mask_mode == (uint8_t)ZPSE_M_ALL_EXCEPTION) {
        ec = 0;
    } else {
        ec = (uint8_t)k; /* k <= nt-1 <= 255 here */
    }

    /* Gather the per-exception operands the chosen residual mode needs
     * (exact encoded length for the exclusion packers). */
    need_vals = (res_mode == (uint8_t)ZPSE_R_EXCLUSION_U8 ||
                 res_mode == (uint8_t)ZPSE_R_PALETTE_BITS ||
                 res_mode == (uint8_t)ZPSE_R_PACKED_BASE_M);
    need_act = (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL ||
                res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA);
    use_vals = need_vals && k > 0u;
    use_act = need_act && k > 0u;
    /* Exclusion-domain operands are exactly what classify recorded while
     * it walked the tile, so only the raw-actual operands (modes 1 and
     * 4) still need a gather pass here. */
    if (use_act) {
        uint16_t j = 0;
        uint16_t p;
        for (p = 0; p < nt; p++) {
            if ((words[p >> 5] >> (p & 31u)) & 1u) {
                avals[j++] = actual[p];
            }
        }
    }

    off = (uint32_t)ZPSE_TILE_HEADER_SIZE +
        (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT ? (uint32_t)1
                                                       : (uint32_t)0);
    if (off > record_cap)
        return PSE_OUTPUT_OVERFLOW;
    mrc = zpse_mask_encode(mask_mode, words, nt, k, record_out + off,
                           tile_cap16(record_cap - off), comb_scratch);
    if (mrc < 0)
        return mrc;
    off += (uint32_t)mrc;
    if (off > record_cap)
        return PSE_OUTPUT_OVERFLOW;
    rrc = zpse_residual_encode(res_mode, use_vals ? cl->res_at_exc : 0,
                               use_act ? avals : 0, cl->at_pred, k,
                               record_out + off,
                               tile_cap16(record_cap - off));
    if (rrc < 0)
        return rrc;
    off += (uint32_t)rrc;

    record_out[0] = zpse_tile_flags_make(mask_mode, res_mode,
                                         tile_reset ? 1 : 0);
    record_out[1] = predictor_id;
    record_out[2] = (uint8_t)(nt - (uint16_t)1);
    record_out[3] = ec;
    zpse_store_u16le(record_out + 4, (uint16_t)mrc);
    zpse_store_u16le(record_out + 6, (uint16_t)rrc);
    if (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
        record_out[8] = cl->constant_C;
    return (int)off;
}

int zpse_tile_encode(ZpseLaneState *lanes, const uint8_t *actual, uint16_t nt,
                     uint8_t predictor_id, uint8_t mask_mode,
                     uint8_t res_mode, uint8_t initial_literal,
                     int tile_reset, const uint8_t *seg_base,
                     uint32_t seg_pos, uint8_t *record_out,
                     uint32_t record_cap, uint8_t *comb_scratch)
{
    ZpseTileClass cl;
    int rc;

    if (lanes == 0 || actual == 0 || record_out == 0)
        return PSE_BAD_ARG;
    rc = tile_classify(lanes, actual, nt, predictor_id, initial_literal,
                       tile_reset, seg_base, seg_pos, &cl);
    if (rc != PSE_OK)
        return rc;
    return zpse_tile_pack(predictor_id, &cl, nt, actual, mask_mode, res_mode,
                          tile_reset, record_out, record_cap, comb_scratch);
}

/* ------------------------------------------------------------------ *
 * Decode                                                              *
 * ------------------------------------------------------------------ */

/* Move the mask register on to position p+1. Bits are consumed LSB
 * first, so a plain shift suffices except at a word boundary, where the
 * next word is loaded -- and where it is not loaded at all once the
 * tile has ended. */
#define ZPSE_MASK_ADVANCE(p_)                                                \
    do {                                                                     \
        if (((p_) & 31u) == 31u) {                                           \
            if ((uint32_t)(p_) + 1u < (uint32_t)nt)                          \
                mw = words[((uint32_t)(p_) >> 5) + 1u];                      \
        } else {                                                             \
            mw >>= 1;                                                         \
        }                                                                    \
    } while (0)

/* Reconstruct a stateless (CONST / STRIDE) tile. Two things the generic
 * byte loop pays for are hoisted out: the mask bit comes from a register
 * that shifts right rather than a reload plus a variable shift per byte,
 * and the residual polarity is a loop-invariant branch that is lifted by
 * generating one loop per case (modes 0, 2 and 3 all unpack to exclusion
 * residuals, so there are only three). XHAT_EXPR is the prediction; it
 * may read `p` and must be free of side effects. */
#define ZPSE_CONST_STRIDE_LOOP(XHAT_EXPR)                                    \
    do {                                                                     \
        j = 0;                                                               \
        mw = words[0];                                                       \
        if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {                        \
            for (p = 0; p < nt; p++) {                                       \
                uint8_t xhat = (uint8_t)(XHAT_EXPR);                         \
                out[p] = ((mw & 1u) != 0u) ? resid[j++] : xhat;              \
                ZPSE_MASK_ADVANCE(p);                                        \
            }                                                                \
        } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {               \
            for (p = 0; p < nt; p++) {                                       \
                uint8_t xhat = (uint8_t)(XHAT_EXPR);                         \
                out[p] = ((mw & 1u) != 0u)                                   \
                    ? (uint8_t)(((unsigned)xhat + (int)deltas[j++]) & 0xFFu) \
                    : xhat;                                                  \
                ZPSE_MASK_ADVANCE(p);                                        \
            }                                                                \
        } else {                                                             \
            for (p = 0; p < nt; p++) {                                       \
                uint8_t xhat = (uint8_t)(XHAT_EXPR);                         \
                out[p] = ((mw & 1u) != 0u)                                   \
                    ? zpse_exclusion_inverse(resid[j++], xhat) : xhat;       \
                ZPSE_MASK_ADVANCE(p);                                        \
            }                                                                \
        }                                                                    \
    } while (0)

/* Resolve one predicted/residual stream in position order, family by
 * family, with the predictor dispatch hoisted out of the byte loop. */
int zpse_tile_decode(ZpseLaneState *lanes, const uint8_t *record,
                     uint32_t record_len, uint8_t initial_literal,
                     int tile_reset, const uint8_t *seg_base,
                     uint32_t seg_pos, uint8_t *out, uint32_t out_cap,
                     uint8_t *comb_scratch)
{
    struct ZpseTileHeader th;
    uint16_t nt = 0;
    uint16_t k = 0;
    uint8_t mask_mode;
    uint8_t res_mode;
    uint8_t constant_C = 0;
    const uint8_t *mask_ptr;
    const uint8_t *res_ptr;
    uint32_t words[8];
    uint32_t mw;
    uint8_t resid[256];
    int8_t deltas[256];
    uint16_t p;
    uint16_t j;
    int rc;
    int fresh;
    int fam;

    if (lanes == 0 || record == 0 || out == 0)
        return PSE_BAD_ARG;
    rc = tile_parse_header(record, record_len, &th);
    if (rc != PSE_OK)
        return rc;
    nt = (uint16_t)((uint32_t)th.tile_length_minus_1 + (uint32_t)1);
    rc = zpse_tile_validate(&th, nt, record_len, &nt, &k);
    if (rc != PSE_OK)
        return rc;
    /* record_len IS tile_record_size here, so the size equation is exact. */
    if (tile_record_need(&th) != record_len)
        return PSE_BAD_TILE_SIZE;
    if ((uint32_t)nt > out_cap)
        return PSE_OUTPUT_OVERFLOW;

    mask_mode = zpse_tile_mask_mode(th.tile_flags);
    res_mode = zpse_tile_res_mode(th.tile_flags);

    mask_ptr = record + ZPSE_TILE_HEADER_SIZE;
    if (th.predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT) {
        constant_C = mask_ptr[0];
        mask_ptr += 1;
    }
    res_ptr = mask_ptr + th.mask_payload_size;

    fresh = (tile_reset || (th.tile_flags & ZPSE_TILE_FLAG_RESET)) ? 1 : 0;
    if (fresh)
        zpse_lane_reset_for(lanes, initial_literal, th.predictor_id);

    zpse_mask_clear(words);
    rc = zpse_mask_decode(mask_mode, mask_ptr, th.mask_payload_size,
                          nt, k, words, comb_scratch);
    if (rc != PSE_OK)
        return rc;

    /* Residual units are pred-independent except SIGNED_SMALL_DELTA, whose
     * payload unpacks to raw deltas here; the pred-dependent mapping runs
     * inside the online loop below. Other modes decode straight to
     * exclusion (or actual) bytes; their pred parameter is unused. */
    if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
        rc = tile_unpack_deltas(res_ptr, th.residual_payload_size, k,
                                deltas);
        if (rc != PSE_OK)
            return rc;
    } else if (k > 0 || th.residual_payload_size > 0) {
        rc = zpse_residual_decode(res_mode, res_ptr,
                                  th.residual_payload_size, k,
                                  0, resid, resid);
        if (rc != PSE_OK)
            return rc;
    }

    /* ---- All-hit fast path -------------------------------------- *
     * With no exceptions the whole tile is prediction. CONST and STRIDE
     * read no lane state at all (their state update is the O(8)
     * epilogue below), so this is valid on a non-resetting tile too;
     * only the history/table families need a freshly reset state to be
     * at a fixed point, where the first prediction reproduces itself
     * forever. Validation above still ran, so hostile streams are
     * rejected exactly as before. */
    if (k == 0u) {
        fam = tile_family(th.predictor_id);
        if (fam == TILE_FAM_CONST || fam == TILE_FAM_STRIDE ||
            (fresh && fam != TILE_FAM_ORDER1 && fam != TILE_FAM_ORDER2)) {
            uint8_t ent_last[8];
            uint8_t ent_second[8];
            if (fam == TILE_FAM_STRIDE) {
                uint32_t s = tile_stride_of(th.predictor_id);
                uint32_t filled = ((uint32_t)nt < s) ? (uint32_t)nt : s;
                /* out IS seg_base + seg_pos (the segment decoder writes
                 * every tile in place), so every prediction past the
                 * first s positions reads a byte this tile has already
                 * produced: the result is periodic with period s. Build
                 * the s-byte seed once, then expand it by repeated
                 * doubling instead of paying a store-to-load round trip
                 * per byte. */
                for (p = 0; p < (uint16_t)filled; p++) {
                    uint32_t abs_pos = seg_pos + (uint32_t)p;
                    out[p] = (abs_pos < s) ? initial_literal
                                           : seg_base[abs_pos - s];
                }
                while (filled < (uint32_t)nt) {
                    uint32_t chunk = filled;
                    if (chunk > (uint32_t)nt - filled) {
                        chunk = (uint32_t)nt - filled;
                    }
                    memcpy(out + filled, out, chunk);
                    filled += chunk;
                }
            } else if (th.predictor_id == (uint8_t)ZPSE_P_CONST_ZERO) {
                memset(out, 0, nt);
            } else if (th.predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT) {
                memset(out, constant_C, nt);
            } else {
                /* FAM_LANEHIST: a freshly reset state is a fixed point. */
                memset(out, initial_literal, nt);
            }
            tile_capture_hist(lanes, ent_last, ent_second);
            tile_history_fixup(lanes, out, nt, ent_last, ent_second);
            return (int)nt;
        }
    }

    /* ---- General path: dispatch once, then run the family loop ---- */
    fam = tile_family(th.predictor_id);
    if (fam == TILE_FAM_CONST || fam == TILE_FAM_STRIDE) {
        uint8_t ent_last[8];
        uint8_t ent_second[8];
        tile_capture_hist(lanes, ent_last, ent_second);
        if (fam == TILE_FAM_CONST) {
            uint8_t c = (th.predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
                            ? constant_C : 0u;
            ZPSE_CONST_STRIDE_LOOP(c);
        } else {
            uint32_t s = tile_stride_of(th.predictor_id);
            ZPSE_CONST_STRIDE_LOOP((seg_pos + (uint32_t)p < s)
                                       ? initial_literal
                                       : seg_base[seg_pos + (uint32_t)p - s]);
        }
        tile_history_fixup(lanes, out, nt, ent_last, ent_second);
        if (j != k)
            return PSE_BAD_MASK;
        return (int)nt;
    }

    if (fam == TILE_FAM_LANEHIST) {
        if (th.predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
            j = 0;
            for (p = 0; p < nt; p++) {
                size_t lane = (size_t)p & 7u;
                uint8_t xhat = lanes->last[lane];
                uint8_t x;
                if (((words[p >> 5] >> (p & 31u)) & 1u) == 0u) {
                    x = xhat;
                } else if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                    x = resid[j++];
                } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                    x = (uint8_t)(((unsigned)xhat + (int)deltas[j++]) & 0xFFu);
                } else {
                    x = zpse_exclusion_inverse(resid[j++], xhat);
                }
                out[p] = x;
                lanes->second_last[lane] = xhat;
                lanes->last[lane] = x;
            }
        } else if (th.predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
            j = 0;
            for (p = 0; p < nt; p++) {
                size_t lane = (size_t)p & 7u;
                uint8_t last = lanes->last[lane];
                uint8_t xhat = (uint8_t)((unsigned)(2u * (unsigned)last -
                    (unsigned)lanes->second_last[lane]) & 0xFFu);
                uint8_t x;
                if (((words[p >> 5] >> (p & 31u)) & 1u) == 0u) {
                    x = xhat;
                } else if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                    x = resid[j++];
                } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                    x = (uint8_t)(((unsigned)xhat + (int)deltas[j++]) & 0xFFu);
                } else {
                    x = zpse_exclusion_inverse(resid[j++], xhat);
                }
                out[p] = x;
                lanes->second_last[lane] = last;
                lanes->last[lane] = x;
            }
        } else {
            j = 0;
            for (p = 0; p < nt; p++) {
                size_t lane = (size_t)p & 7u;
                uint8_t last = lanes->last[lane];
                uint8_t xhat = (uint8_t)(((unsigned)last +
                    (unsigned)lanes->second_last[lane] + 1u) >> 1);
                uint8_t x;
                if (((words[p >> 5] >> (p & 31u)) & 1u) == 0u) {
                    x = xhat;
                } else if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                    x = resid[j++];
                } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                    x = (uint8_t)(((unsigned)xhat + (int)deltas[j++]) & 0xFFu);
                } else {
                    x = zpse_exclusion_inverse(resid[j++], xhat);
                }
                out[p] = x;
                lanes->second_last[lane] = last;
                lanes->last[lane] = x;
            }
        }
        if (j != k)
            return PSE_BAD_MASK;
        return (int)nt;
    }

    /* ORDER1 / ORDER2: table-updating online loop. */
    j = 0;
    if (fam == TILE_FAM_ORDER1) {
        uint8_t *o1 = lanes->order1;
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint8_t last = lanes->last[lane];
            uint8_t xhat = o1[lane * 256u + (size_t)last];
            uint8_t x;
            if (((words[p >> 5] >> (p & 31u)) & 1u) == 0u) {
                x = xhat;
            } else if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                x = resid[j++];
            } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                x = (uint8_t)(((unsigned)xhat + (int)deltas[j++]) & 0xFFu);
            } else {
                x = zpse_exclusion_inverse(resid[j++], xhat);
            }
            out[p] = x;
            lanes->second_last[lane] = last;
            lanes->last[lane] = x;
            o1[lane * 256u + (size_t)last] = x;
        }
    } else {
        uint8_t *val = lanes->o2val;
        uint8_t *conf = lanes->o2conf;
        for (p = 0; p < nt; p++) {
            size_t lane = (size_t)p & 7u;
            uint8_t last = lanes->last[lane];
            uint8_t second = lanes->second_last[lane];
            size_t off = lane * 4096u + (size_t)zpse_order2_index(second, last);
            uint8_t xhat = val[off];
            uint8_t x;
            if (((words[p >> 5] >> (p & 31u)) & 1u) == 0u) {
                x = xhat;
            } else if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                x = resid[j++];
            } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                x = (uint8_t)(((unsigned)xhat + (int)deltas[j++]) & 0xFFu);
            } else {
                x = zpse_exclusion_inverse(resid[j++], xhat);
            }
            out[p] = x;
            if (conf[off] == 0u) {
                val[off] = x;
                conf[off] = 1u;
            } else if (x == xhat) {
                if (conf[off] < 255u)
                    conf[off]++;
            } else {
                conf[off]--;
            }
            lanes->second_last[lane] = last;
            lanes->last[lane] = x;
        }
    }
    if (j != k)
        return PSE_BAD_MASK;
    return (int)nt;
}
