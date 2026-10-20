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

/* Tile framing + reconstruct path.
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
 * history update is applied by an O(8) epilogue
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
        return TILE_FAM_STRIDE; /* 5..8, 11 */
    }
}

void zpse_tile_read_header(const uint8_t *record, struct ZpseTileHeader *th)
{
    th->tile_flags = record[0];
    th->predictor_id = record[1];
    th->tile_length_minus_1 = record[2];
    th->exception_count = record[3];
    th->mask_payload_size = zpse_load_u16le(record + 4);
    th->residual_payload_size = zpse_load_u16le(record + 6);
}

static int tile_parse_header(const uint8_t *record, uint32_t record_len,
                             struct ZpseTileHeader *th)
{
    if (record_len < (uint32_t)ZPSE_TILE_HEADER_SIZE)
        return PSE_BAD_TILE_SIZE;
    zpse_tile_read_header(record, th);
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
    case ZPSE_P_STRIDE_3:
        return 3u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u;
    }
}

/* Epilogue for the stateless families (CONST, CONSTC, STRIDE): apply the
 * history update for every lane in one O(8) pass. `src` holds
 * the Nt actual bytes (encoder input plane or decoder output plane);
 * ent_last/ent_second are the histories captured before the tile ran. */
static void tile_history_fixup(ZpseLaneState *lanes, const uint8_t *src,
                               uint16_t nt, const uint8_t ent_last[8],
                               const uint8_t ent_second[8])
{
    unsigned l;

    /* A full tile is the shape of every tile but a segment's last, and
     * there the rule below collapses: lane l ends on position
     * l + 31*8 == 248 + l, and no lane's last position is below 8, so no
     * entry state is involved. That is two 8-byte loads and two stores
     * instead of eight iterations of a divide and two compares. */
    if (nt == (uint16_t)ZPSE_MAX_TILE_LEN) {
        memcpy(lanes->last, src + 248u, sizeof lanes->last);
        memcpy(lanes->second_last, src + 240u, sizeof lanes->second_last);
        return;
    }

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
    if (th->predictor_id > (uint8_t)ZPSE_P_STRIDE_3)
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
        cl->act_at_exc[k] = _x;                                         \
        cl->seen[_r >> 5] |= (uint32_t)1u << (_r & 31u);                \
        if (_r > _rmax)                                                 \
            _rmax = _r;                                                 \
        if (_db < _dmin)                                                \
            _dmin = _db;                                                \
        if (_db > _dmax)                                                \
            _dmax = _db;                                                \
        k++;                                                            \
    } while (0)

/* Same bookkeeping as TILE_TAKE_EXC for a position that the finished
 * mask ALREADY marks. Both callers below walk the mask words, so the
 * set-bit store of TILE_TAKE_EXC would write a bit that is already set:
 * dropping it leaves cl->words identical. What it also removes is the
 * only write the walk performs, and `words` is `cl->words`, a caller
 * pointer the compiler may not assume is disjoint from the tile bytes --
 * with the store gone it can keep `actual[]`/`stg[]` values in registers
 * across exceptions instead of reloading them after every store. */
#define TILE_TAKE_EXC_MASKED(p_, xhat_, x_)                             \
    do {                                                                \
        uint16_t _pp = (uint16_t)(p_);                                  \
        uint8_t _xh = (uint8_t)(xhat_);                                 \
        uint8_t _x = (uint8_t)(x_);                                     \
        uint8_t _r = zpse_exclusion_forward(_x, _xh);                   \
        uint8_t _db = (uint8_t)((((unsigned)_x - (unsigned)_xh + 128u)  \
                                 & 0xFFu));                             \
        (void)_pp;                                                      \
        cl->at_pred[k] = _xh;                                           \
        cl->res_at_exc[k] = _r;                                         \
        cl->act_at_exc[k] = _x;                                         \
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

/* Number of one bits in a 64-bit word. Same SWAR idiom as the 32-bit
 * form below, extended one step, so it stays clear of the libgcc helper
 * a __builtin would call into. */
static int tile_popcount64(uint64_t v)
{
    v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
    v = (v & UINT64_C(0x3333333333333333)) +
        ((v >> 2) & UINT64_C(0x3333333333333333));
    v = (v + (v >> 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    return (int)((v * UINT64_C(0x0101010101010101)) >> 56);
}

/* Number of one bits in a 32-bit word (same idiom as mask.c). */
static int tile_popcount32(uint32_t w)
{
    /* Branch-free SWAR popcount. __builtin_popcount lowers to a libgcc
     * call (__popcountdi2) unless the whole build is compiled with
     * -mpopcnt, which this portable codebase is not. */
    w = w - ((w >> 1) & 0x55555555u);
    w = (w & 0x33333333u) + ((w >> 2) & 0x33333333u);
    w = (w + (w >> 4)) & 0x0F0F0F0Fu;
    return (int)((w * 0x01010101u) >> 24);
}

/* The distinct-residual count is capped here, and that is exact where it
 * matters: the only consumer of cl->nu is the palette length in
 * tile_res_len(), which rejects the palette outright once nu exceeds 16
 * (a palette holds at most 16 entries). Reporting 17 for every tile above
 * the threshold therefore cannot change one packed byte, while making the
 * count knowable from a prefix of the exceptions -- which is what lets the
 * dense classify below stop recording residuals as soon as a seventeenth
 * distinct value is seen instead of walking all ~250 of them. Below the
 * threshold nu stays exact, and cl->seen is trimmed with it so the two
 * remain consistent (popcount(seen) == nu, which is what the scalar/SIMD
 * parity suite checks). */
#define ZPSE_TILE_NU_CAP 17u

/* Only a tile with more than ZPSE_TILE_NU_CAP distinct residuals reaches
 * the trim, so it is kept out of line: inlined into the classify epilogue
 * (which every tile runs) its register pressure measurably slowed the
 * sparse and nibble rows down. MSVC has no per-function attribute and
 * simply inlines it. */
#if defined(__GNUC__) || defined(__clang__)
#define ZPSE_TILE_RARE __attribute__((noinline, cold))
#else
#define ZPSE_TILE_RARE
#endif

/* Reduce cl->seen to its `keep` lowest set bits, so the set can never
 * disagree with a capped nu. Caller guarantees keep < popcount. */
static ZPSE_TILE_RARE void tile_seen_trim(uint32_t seen[8], unsigned keep)
{
    unsigned w;
    unsigned total = 0u;

    for (w = 0; w < 8u; w++)
        total += (unsigned)tile_popcount32(seen[w]);
    for (w = 0; w < 8u && total > keep; w++) {
        while (total > keep && seen[w] != 0u) {
            seen[w] &= seen[w] - 1u; /* drop the lowest set bit */
            total--;
        }
    }
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
 * its lane state, which makes the history at position p
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
/* Every kernel also pins its own 64-byte alignment. Their inner loops are
 * branch-light AVX2 blocks whose throughput depends on where the loop body
 * lands inside a 32-byte fetch window, and with plain -O2 the placement
 * moves with the size of any unrelated function in the translation unit: a
 * purely encoder-side edit of this file once moved palette16 decode from
 * 2.15 to 2.03 GB/s without touching a decode instruction. Aligning the
 * entries makes the kernels' cost a property of their own code. */
#define ZPSE_TILE_AVX2 __attribute__((target("avx2"), aligned(64)))
#define ZPSE_TILE_AVX2P __attribute__((target("avx2,popcnt"), aligned(64)))
#define ZPSE_TILE_HAVE_AVX2 1
#elif defined(__AVX2__)
#define ZPSE_TILE_AVX2
#define ZPSE_TILE_AVX2P
#define ZPSE_TILE_HAVE_AVX2 1
#else
#define ZPSE_TILE_AVX2
#define ZPSE_TILE_AVX2P
#define ZPSE_TILE_HAVE_AVX2 0
#endif
#else
#define ZPSE_TILE_HAVE_AVX2 0
#endif

/* Index of the lowest set bit. Caller guarantees w != 0. Plain integer
 * helper (no intrinsics), so every build has it: the exception walk in
 * zpse_tile_decode uses it with or without a vector unit. */
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

#if ZPSE_TILE_HAVE_AVX2
#include <immintrin.h>

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
        /* andnot is ~_eq & MASK, i.e. the same 32 exception bits as        \
         * XOR-against-all-ones followed by an AND, in one instruction. */  \
        words[(uint32_t)p >> 5] |= (uint32_t)_mm256_movemask_epi8(          \
            _mm256_andnot_si256(_eq, MASK_));                               \
    } while (0)

/* ------------------------------------------------------------------ *
 * Dense-tile bookkeeping (AVX2 + SSSE3 shuffle compaction).
 *
 * When most positions are exceptions (text, code, noise) the scalar
 * per-exception loop below costs ~10 cycles each and dominates the
 * encoder. Here the same quantities are derived 32 positions at a time:
 *   exclusion residual  r = (x < p) ? x : x - 1      (x != p on exceptions)
 *   signed-delta byte   db = x - p + 128
 * with masked max/min reductions for rmax/dmin/dmax, and the
 * ascending compaction of at_pred[] / res_at_exc[] done with a
 * 256-entry byte-shuffle table (8 mask bits -> 8 packed bytes).
 * The results are bit-identical to the scalar loop.
 * ------------------------------------------------------------------ */
static const uint8_t tile_cmp_lut[256][8] = {
    {0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x02,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x80,0x80,0x80,0x80,0x80},
    {0x03,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x80,0x80,0x80,0x80,0x80},
    {0x02,0x03,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x80,0x80,0x80,0x80},
    {0x04,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x80,0x80,0x80,0x80,0x80},
    {0x02,0x04,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x80,0x80,0x80,0x80},
    {0x03,0x04,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x80,0x80,0x80,0x80},
    {0x02,0x03,0x04,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x80,0x80,0x80},
    {0x05,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x05,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x05,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x02,0x05,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x05,0x80,0x80,0x80,0x80},
    {0x03,0x05,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x05,0x80,0x80,0x80,0x80},
    {0x02,0x03,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x05,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x05,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x05,0x80,0x80,0x80},
    {0x04,0x05,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x05,0x80,0x80,0x80,0x80},
    {0x02,0x04,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x05,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x05,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x05,0x80,0x80,0x80},
    {0x03,0x04,0x05,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x05,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x05,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x05,0x80,0x80,0x80},
    {0x02,0x03,0x04,0x05,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x05,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x05,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x80,0x80},
    {0x06,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x06,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x06,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x02,0x06,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x06,0x80,0x80,0x80,0x80},
    {0x03,0x06,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x06,0x80,0x80,0x80,0x80},
    {0x02,0x03,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x06,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x06,0x80,0x80,0x80},
    {0x04,0x06,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x06,0x80,0x80,0x80,0x80},
    {0x02,0x04,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x06,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x06,0x80,0x80,0x80},
    {0x03,0x04,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x06,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x06,0x80,0x80,0x80},
    {0x02,0x03,0x04,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x06,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x06,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x06,0x80,0x80},
    {0x05,0x06,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x05,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x05,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x02,0x05,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x05,0x06,0x80,0x80,0x80},
    {0x03,0x05,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x05,0x06,0x80,0x80,0x80},
    {0x02,0x03,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x05,0x06,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x05,0x06,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x05,0x06,0x80,0x80},
    {0x04,0x05,0x06,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x05,0x06,0x80,0x80,0x80},
    {0x02,0x04,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x05,0x06,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x05,0x06,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x05,0x06,0x80,0x80},
    {0x03,0x04,0x05,0x06,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x05,0x06,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x05,0x06,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x05,0x06,0x80,0x80},
    {0x02,0x03,0x04,0x05,0x06,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x05,0x06,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x05,0x06,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x80},
    {0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x02,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x07,0x80,0x80,0x80,0x80},
    {0x03,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x07,0x80,0x80,0x80,0x80},
    {0x02,0x03,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x07,0x80,0x80,0x80},
    {0x04,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x07,0x80,0x80,0x80,0x80},
    {0x02,0x04,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x07,0x80,0x80,0x80},
    {0x03,0x04,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x07,0x80,0x80,0x80},
    {0x02,0x03,0x04,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x07,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x07,0x80,0x80},
    {0x05,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x05,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x05,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x02,0x05,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x05,0x07,0x80,0x80,0x80},
    {0x03,0x05,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x05,0x07,0x80,0x80,0x80},
    {0x02,0x03,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x05,0x07,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x05,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x05,0x07,0x80,0x80},
    {0x04,0x05,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x05,0x07,0x80,0x80,0x80},
    {0x02,0x04,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x05,0x07,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x05,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x05,0x07,0x80,0x80},
    {0x03,0x04,0x05,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x05,0x07,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x05,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x05,0x07,0x80,0x80},
    {0x02,0x03,0x04,0x05,0x07,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x05,0x07,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x05,0x07,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x07,0x80},
    {0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x06,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x01,0x06,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x02,0x06,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x02,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x06,0x07,0x80,0x80,0x80},
    {0x03,0x06,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x03,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x06,0x07,0x80,0x80,0x80},
    {0x02,0x03,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x06,0x07,0x80,0x80,0x80},
    {0x01,0x02,0x03,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x06,0x07,0x80,0x80},
    {0x04,0x06,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x04,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x06,0x07,0x80,0x80,0x80},
    {0x02,0x04,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x06,0x07,0x80,0x80,0x80},
    {0x01,0x02,0x04,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x06,0x07,0x80,0x80},
    {0x03,0x04,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x06,0x07,0x80,0x80,0x80},
    {0x01,0x03,0x04,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x06,0x07,0x80,0x80},
    {0x02,0x03,0x04,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x06,0x07,0x80,0x80},
    {0x01,0x02,0x03,0x04,0x06,0x07,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x06,0x07,0x80},
    {0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80},
    {0x00,0x05,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x01,0x05,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x01,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x02,0x05,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x02,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x01,0x02,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x02,0x05,0x06,0x07,0x80,0x80},
    {0x03,0x05,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x03,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x01,0x03,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x03,0x05,0x06,0x07,0x80,0x80},
    {0x02,0x03,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x02,0x03,0x05,0x06,0x07,0x80,0x80},
    {0x01,0x02,0x03,0x05,0x06,0x07,0x80,0x80},
    {0x00,0x01,0x02,0x03,0x05,0x06,0x07,0x80},
    {0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80},
    {0x00,0x04,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x01,0x04,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x01,0x04,0x05,0x06,0x07,0x80,0x80},
    {0x02,0x04,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x02,0x04,0x05,0x06,0x07,0x80,0x80},
    {0x01,0x02,0x04,0x05,0x06,0x07,0x80,0x80},
    {0x00,0x01,0x02,0x04,0x05,0x06,0x07,0x80},
    {0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80},
    {0x00,0x03,0x04,0x05,0x06,0x07,0x80,0x80},
    {0x01,0x03,0x04,0x05,0x06,0x07,0x80,0x80},
    {0x00,0x01,0x03,0x04,0x05,0x06,0x07,0x80},
    {0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80},
    {0x00,0x02,0x03,0x04,0x05,0x06,0x07,0x80},
    {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x80},
    {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07}
};

/* Horizontal maximum of a 32-byte vector of unsigned bytes: the fold the
 * dense bookkeeping used to spell out as 32 scalar compares, in five
 * min/max steps. A zero-fill shift is safe here because this is a
 * maximum of unsigned bytes and a zero can never raise one -- which is
 * also why the SMALLEST delta is accumulated as the maximum of its
 * complement below. */
/* Compact eight exception lanes' prediction, residual and actual into
 * dst[o..]: the mask byte selects, through the 64-bit table above, which of
 * the eight source bytes land in the packed 64-bit store. */
#define TILE_CL_CMP3(tp_, tr_, ta_, o_, pd_, rd_, ad_, mask8_)               \
    do {                                                                     \
        const __m128i _c = _mm_loadl_epi64(                                   \
            (const __m128i *)(const void *)tile_cmp_lut[(mask8_)]);           \
        _mm_storel_epi64((__m128i *)(void *)((tp_) + (o_)),                   \
                         _mm_shuffle_epi8((pd_), _c));                        \
        _mm_storel_epi64((__m128i *)(void *)((tr_) + (o_)),                   \
                         _mm_shuffle_epi8((rd_), _c));                        \
        _mm_storel_epi64((__m128i *)(void *)((ta_) + (o_)),                   \
                         _mm_shuffle_epi8((ad_), _c));                        \
    } while (0)

static ZPSE_TILE_AVX2 uint8_t tile_hmax_u8x32(__m256i v)
{
    __m128i m = _mm_max_epu8(_mm256_castsi256_si128(v),
                             _mm256_extracti128_si256(v, 1));

    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
}

/* Fold the 256-byte residual-presence array into `seen`, returning how
 * many distinct values it holds. */
static ZPSE_TILE_AVX2P unsigned tile_pres_fold(const uint8_t *pres,
                                               uint32_t seen[8])
{
    const __m256i z = _mm256_setzero_si256();
    unsigned i;
    unsigned nu = 0u;

    for (i = 0; i < 8u; i++) {
        const __m256i q = _mm256_loadu_si256(
            (const __m256i *)(const void *)(pres + i * 32u));

        seen[i] = (uint32_t)~_mm256_movemask_epi8(_mm256_cmpeq_epi8(q, z));
        nu += (unsigned)_mm_popcnt_u32(seen[i]);
    }
    return nu;
}

/* `tile` is the tile's own first byte; the sixteen bytes BEFORE it are the
 * staged head (entry lane state, or the plane bytes preceding a STRIDE
 * tile), exactly as they sit behind the tile in the staging buffer. Passing
 * the tile pointer rather than the buffer makes the read offsets family-
 * independent and lets the caller hand in the input itself when the tile is
 * read in place (see tile_classify_avx2's `direct`). */
static ZPSE_TILE_AVX2P uint16_t tile_dense_bk_avx2(
    const uint8_t *tile, uint16_t nt, uint8_t predictor_id, int fam,
    uint8_t c, uint32_t s, const uint32_t words[8], ZpseTileClass *cl,
    uint8_t *rmax_o, uint8_t *dmin_o, uint8_t *dmax_o)
{
    /* The compacted exception arrays are written straight into the
     * classification: an 8-byte store per 16 lanes lands exactly where the
     * caller reads them from, and the three memcpys the staged form paid at
     * the end of every dense tile (up to 768 bytes of copy) disappear. The
     * stores may run 7 bytes past the last entry, which is why the arrays
     * carry 8 bytes of slack (tile.h). */
    uint8_t *const tp = cl->at_pred;
    uint8_t *const tr = cl->res_at_exc;
    uint8_t *const ta = cl->act_at_exc;
    uint32_t k = 0u;
    unsigned ch;
    unsigned i;
    const __m256i all = _mm256_set1_epi8((char)-1);
    const __m256i b128 = _mm256_set1_epi8((char)0x80);
    const __m256i cvec = _mm256_set1_epi8((char)c);
    const __m256i idx = _mm256_setr_epi8(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31);
    __m256i rmaxv = _mm256_setzero_si256();
    __m256i dmaxv = _mm256_setzero_si256();
    /* The smallest signed-delta byte is accumulated as the largest
     * COMPLEMENT: an excluded lane contributes 0x00, which is neutral for
     * a maximum, and the minimum comes back out as ~max at the end. All
     * three reductions are therefore maxima and the horizontal fold above
     * can zero-fill. When there is at least one exception -- which is why
     * this function was called -- the fold returns exactly
     * min(db over exceptions), the value the old all-ones-seeded minimum
     * returned. */
    __m256i ndminv = _mm256_setzero_si256();

    for (ch = 0; ch * 32u < (unsigned)nt; ch++) {
        unsigned p = ch * 32u;
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(tile + p));
        __m256i pv, eq, ex, lt, r, db;
        uint32_t m = words[ch];
        unsigned h;

        if (m == 0u)
            continue;
        if (fam == TILE_FAM_CONST) {
            pv = cvec;
        } else if (fam == TILE_FAM_STRIDE) {
            pv = _mm256_loadu_si256((const __m256i *)(const void *)(tile + p - s));
        } else {
            const __m256i h8 = _mm256_loadu_si256((const __m256i *)(const void *)(tile + p - 8u));
            const __m256i h16 = _mm256_loadu_si256((const __m256i *)(const void *)(tile + p - 16u));
            if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS)
                pv = h8;
            else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA)
                pv = _mm256_sub_epi8(_mm256_add_epi8(h8, h8), h16);
            else
                pv = _mm256_avg_epu8(h8, h16);
        }
        (void)eq; (void)idx;
        lt = _mm256_cmpeq_epi8(_mm256_min_epu8(v, pv), v);
        r = _mm256_add_epi8(v, _mm256_xor_si256(lt, all));
        db = _mm256_add_epi8(_mm256_sub_epi8(v, pv), b128);
        if (m == 0xFFFFFFFFu) {
            /* Fully set word: every lane is an exception, so the blend
             * mask the expansion below would build is all ones and both
             * uses of it collapse. The masked maxima take r and db
             * unmasked (`and` with an all-ones vector), the minimum over
             * ~db is ~db itself, and tile_cmp_lut[0xFF] is exactly
             * 00..07 (verified above), i.e. the compaction of a full word
             * is the sixteen bytes of pv / r in order -- so seventeen
             * words of vector work per chunk become four stores. A dense
             * tile has seven such chunks out of eight. Bits at or above
             * nt are never set by the chunk macro, so a full word implies
             * a full chunk and k + 32 <= 256 always holds here. */
            /* Eight lanes at a time, in position order: the low half of
             * each 128-bit lane holds positions h*8..h*8+7 and its high
             * half the next eight, so the four 8-byte stores cover the
             * chunk in order. (Storing a whole 128-bit lane with one
             * 8-byte store would put positions 16..23 after 0..7 and
             * drop 8..15 -- the unit-order trap this path has to avoid
             * just like the general one.) */
            const __m128i plo = _mm256_castsi256_si128(pv);
            const __m128i rlo = _mm256_castsi256_si128(r);
            const __m128i phi = _mm256_extracti128_si256(pv, 1);
            const __m128i rhi = _mm256_extracti128_si256(r, 1);

            rmaxv = _mm256_max_epu8(rmaxv, r);
            dmaxv = _mm256_max_epu8(dmaxv, db);
            ndminv = _mm256_max_epu8(ndminv, _mm256_xor_si256(db, all));
            _mm_storel_epi64((__m128i *)(void *)(tp + k), plo);
            _mm_storel_epi64((__m128i *)(void *)(tr + k), rlo);
            _mm_storel_epi64((__m128i *)(void *)(tp + k + 8u),
                             _mm_srli_si128(plo, 8));
            _mm_storel_epi64((__m128i *)(void *)(tr + k + 8u),
                             _mm_srli_si128(rlo, 8));
            _mm_storel_epi64((__m128i *)(void *)(tp + k + 16u), phi);
            _mm_storel_epi64((__m128i *)(void *)(tr + k + 16u), rhi);
            _mm_storel_epi64((__m128i *)(void *)(tp + k + 24u),
                             _mm_srli_si128(phi, 8));
            _mm_storel_epi64((__m128i *)(void *)(tr + k + 24u),
                             _mm_srli_si128(rhi, 8));
            {
                /* the actual bytes of a full word are the chunk itself */
                const __m128i vlo = _mm256_castsi256_si128(v);
                const __m128i vhi = _mm256_extracti128_si256(v, 1);

                _mm_storel_epi64((__m128i *)(void *)(ta + k), vlo);
                _mm_storel_epi64((__m128i *)(void *)(ta + k + 8u),
                                 _mm_srli_si128(vlo, 8));
                _mm_storel_epi64((__m128i *)(void *)(ta + k + 16u), vhi);
                _mm_storel_epi64((__m128i *)(void *)(ta + k + 24u),
                                 _mm_srli_si128(vhi, 8));
            }
            k += 32u;
            continue;
        }
        /* exception lanes straight from the finished mask word */
        {
            __m256i bits = _mm256_set1_epi32((int)m);
            __m256i sel = _mm256_setr_epi8(
                0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,
                2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
            __m256i bm = _mm256_setr_epi8(
                1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128,
                1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128);
            ex = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(bits, sel), bm), bm);
        }
        rmaxv = _mm256_max_epu8(rmaxv, _mm256_and_si256(r, ex));
        dmaxv = _mm256_max_epu8(dmaxv, _mm256_and_si256(db, ex));
        ndminv = _mm256_max_epu8(
            ndminv,
            _mm256_xor_si256(_mm256_or_si256(db, _mm256_xor_si256(ex, all)),
                             all));

        /* Eight exception lanes at a time. Unrolling the two 16-lane halves
         * and precomputing the four sub-mask popcounts was measured both
         * ways: it takes the popcount-to-store dependency off the 32 group
         * stores of a tile, but costs more live 128-bit registers than the
         * loop form on a nearly-all-exception tile (text: +15 cycles a tile
         * of classify), so the shorter form stays. */
        for (h = 0; h < 2u; h++) {
            const __m128i pd =
                h ? _mm256_extracti128_si256(pv, 1) : _mm256_castsi256_si128(pv);
            const __m128i rd =
                h ? _mm256_extracti128_si256(r, 1) : _mm256_castsi256_si128(r);
            const __m128i ad =
                h ? _mm256_extracti128_si256(v, 1) : _mm256_castsi256_si128(v);
            const unsigned m0 = (m >> (16u * h)) & 0xFFu;
            const unsigned m1 = (m >> (16u * h + 8u)) & 0xFFu;

            TILE_CL_CMP3(tp, tr, ta, k, pd, rd, ad, m0);
            k = (uint16_t)(k + _mm_popcnt_u32(m0));
            TILE_CL_CMP3(tp, tr, ta, k, _mm_srli_si128(pd, 8),
                         _mm_srli_si128(rd, 8), _mm_srli_si128(ad, 8), m1);
            k = (uint16_t)(k + _mm_popcnt_u32(m1));
        }
    }

    /* distinct-residual set. cl->seen is the complete set up to the point
     * where it is complete enough: word w bit j of cl->seen is "value
     * 32*w + j is present", which is the bit the scalar path sets, and
     * ZPSE_TILE_NU_CAP is where that stops being worth paying for.
     *
     * A 256-byte presence array needs one independent byte store per
     * exception and no read-modify-write, where the four-bitmap form paid
     * a load and a store through a data-dependent address for each unit.
     * The compare against zero marks the ABSENT values and the complement
     * of its movemask has a one exactly where a presence byte is nonzero.
     *
     * The pass is therefore done in two parts, and the first is a probe:
     * this function only runs with k >= 32, and a tile whose first 32
     * exceptions already hold ZPSE_TILE_NU_CAP distinct residuals has
     * already rejected the palette mode, so the remaining entries are
     * recorded, folded and thrown away. That is the common case for a
     * dense tile (a 256-byte tile of text, code or noise reaches 17
     * distinct residuals within its first ~20 exceptions), and it is the
     * single largest cost in this function. A tile that is still under
     * the cap after the probe -- the palettizable ones -- walks the rest
     * exactly as before, so its count stays exact. */
    {
        uint8_t pres[256];

        memset(pres, 0, sizeof pres);
        for (i = 0; i + 4u <= 32u; i += 4u) {
            pres[tr[i]] = 1u;
            pres[tr[i + 1u]] = 1u;
            pres[tr[i + 2u]] = 1u;
            pres[tr[i + 3u]] = 1u;
        }
        if (tile_pres_fold(pres, cl->seen) < ZPSE_TILE_NU_CAP) {
            for (; i + 4u <= (uint32_t)k; i += 4u) {
                pres[tr[i]] = 1u;
                pres[tr[i + 1u]] = 1u;
                pres[tr[i + 2u]] = 1u;
                pres[tr[i + 3u]] = 1u;
            }
            for (; i < k; i++)
                pres[tr[i]] = 1u;
            (void)tile_pres_fold(pres, cl->seen);
        }
    }

    /* Three horizontal folds instead of three 32-iteration byte scans.
     * rmax/dmax start from a zero accumulator and can only be raised, so
     * a maximum over all 32 lanes is the scalar result; dmin is the
     * complement of the maximum complement accumulated above. */
    *rmax_o = tile_hmax_u8x32(rmaxv);
    *dmax_o = tile_hmax_u8x32(dmaxv);
    *dmin_o = (uint8_t)~tile_hmax_u8x32(ndminv);
    return (uint16_t)k;
}

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
static ZPSE_TILE_AVX2P void tile_classify_avx2(ZpseLaneState *lanes,
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
    /* The bytes this tile is compared against and its exceptions read
     * from. `stg` exists for two reasons: the last chunk of a tile that
     * is not a whole number of 32-byte chunks is loaded past its end and
     * masked, and the LANEHIST/STRIDE predictions read bytes BEFORE the
     * tile. A CONST tile needs neither: its prediction is a single byte,
     * so the comparison never looks at the head, and a full 256-byte tile
     * has no partial last chunk -- every load in the loop below stays
     * inside [0, nt). Such a tile is therefore read straight from the
     * input and the 304-byte staging copy disappears. `stg` stays for the
     * other families and for short tiles, whose records are not the hot
     * case: a segment's tiles are all full except its last. */
    const int direct =
        (nt == (uint16_t)ZPSE_MAX_TILE_LEN &&
         (fam == TILE_FAM_CONST ||
          /* STRIDE reads only bytes at or before the tile, and a tile that
           * starts 16 bytes into the plane has all of them in place, so a
           * full STRIDE tile is read directly too: the 288-byte staging
           * copy below (16 head + 256 body + pad) is not needed for the
           * predictors the encoder actually screens, which is where most
           * of the tiles of a dense plane are classified. */
          (fam == TILE_FAM_STRIDE && seg_pos >= 16u)));
    const uint8_t *tile_bytes;

    (void)seg_base;
    (void)initial_literal;

    if (fam == TILE_FAM_CONST) {
        c = (predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
                ? cl->constant_C : 0u;
    } else if (fam == TILE_FAM_STRIDE) {
        s = tile_stride_of(predictor_id);
    }

    if (direct) {
        tile_bytes = actual;
    } else {
        if (fam == TILE_FAM_STRIDE) {
            /* Real preceding plane bytes; requires seg_pos >= 16. */
            memcpy(stg, actual - 16, 16);
        } else {
            memcpy(stg, ent_second, 8);
            memcpy(stg + 8, ent_last, 8);
        }
        memcpy(stg + 16, actual, nt);
        memset(stg + 16 + nt, 0, 32);
        tile_bytes = stg + 16;
    }

    {
        const __m256i cvec = _mm256_set1_epi8((char)c);
        const __m256i idx = _mm256_setr_epi8(
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31);

        while (p + 32u <= (uint32_t)nt) {
            __m256i v = _mm256_loadu_si256(
                (const __m256i *)(const void *)(tile_bytes + p));
            __m256i pv;

            if (fam == TILE_FAM_CONST) {
                pv = cvec;
            } else if (fam == TILE_FAM_STRIDE) {
                /* tile_bytes has the 16 preceding plane bytes right before
                 * it in the staged case, so this one expression covers both
                 * the in-place and the staged read. */
                pv = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(tile_bytes + p - s));
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
                (const __m256i *)(const void *)(tile_bytes + p));
            __m256i pv;

            if (fam == TILE_FAM_CONST) {
                pv = cvec;
            } else if (fam == TILE_FAM_STRIDE) {
                pv = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(tile_bytes + p - s));
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
        unsigned nexc;

        /* Population count of the whole mask, which is what decides
         * between the dense compaction and the per-exception walk below:
         * a nibble table and one SAD reduction instead of eight SWAR
         * popcounts. Bits at or above nt are never set (the chunk macro
         * masks them), so every set bit is an exception. */
        {
            const __m256i nib = _mm256_set1_epi8(0x0F);
            const __m256i plo = _mm256_setr_epi8(
                0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
            const __m256i wv = _mm256_loadu_si256(
                (const __m256i *)(const void *)words);
            const __m256i pl = _mm256_shuffle_epi8(
                plo, _mm256_and_si256(wv, nib));
            const __m256i ph = _mm256_shuffle_epi8(
                plo, _mm256_and_si256(_mm256_srli_epi16(wv, 4), nib));
            const __m256i ps = _mm256_sad_epu8(_mm256_add_epi8(pl, ph),
                                               _mm256_setzero_si256());

            nexc = (unsigned)((uint64_t)_mm256_extract_epi64(ps, 0) +
                              (uint64_t)_mm256_extract_epi64(ps, 1) +
                              (uint64_t)_mm256_extract_epi64(ps, 2) +
                              (uint64_t)_mm256_extract_epi64(ps, 3));
        }

        /* One walk per prediction formula, so the family and predictor
         * tests are made once per tile instead of once per exception.
         * Every arm computes the prediction and the position exactly as
         * the single mixed loop did, and visits the positions in the same
         * ascending order. */
        if (nexc >= 32u) {
            k = tile_dense_bk_avx2(tile_bytes, nt, predictor_id, fam, c, s,
                                   words, cl, &_rmax, &_dmin, &_dmax);
        } else if (fam == TILE_FAM_CONST) {
            /* Stop once every set bit of the mask has been consumed:
             * positions ascend across words and the mask holds exactly
             * `nexc` of them. */
            for (wi = 0; wi < 8u && k < (uint16_t)nexc; wi++) {
                uint32_t w = words[wi];

                while (w != 0u) {
                    uint16_t pos = (uint16_t)(wi * 32u + tile_ctz32(w));

                    TILE_TAKE_EXC_MASKED(pos, c, tile_bytes[pos]);
                    w &= w - 1u;
                }
            }
        } else if (fam == TILE_FAM_STRIDE) {
            /* Stop once every set bit of the mask has been consumed:
             * positions ascend across words and the mask holds exactly
             * `nexc` of them. */
            for (wi = 0; wi < 8u && k < (uint16_t)nexc; wi++) {
                uint32_t w = words[wi];

                while (w != 0u) {
                    uint16_t pos = (uint16_t)(wi * 32u + tile_ctz32(w));

                    /* tile_bytes, not stg: a full STRIDE tile is read in
                     * place (see `direct`), where stg holds nothing. The
                     * signed offset keeps a position below the stride
                     * reading before the tile, which is the staged head or
                     * real plane bytes. */
                    TILE_TAKE_EXC_MASKED(pos,
                                         tile_bytes[(ptrdiff_t)pos -
                                                    (ptrdiff_t)s],
                                         tile_bytes[pos]);
                    w &= w - 1u;
                }
            }
        } else if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
            /* Stop once every set bit of the mask has been consumed:
             * positions ascend across words and the mask holds exactly
             * `nexc` of them. */
            for (wi = 0; wi < 8u && k < (uint16_t)nexc; wi++) {
                uint32_t w = words[wi];

                while (w != 0u) {
                    uint16_t pos = (uint16_t)(wi * 32u + tile_ctz32(w));

                    TILE_TAKE_EXC_MASKED(pos, stg[8u + (uint32_t)pos],
                                         stg[16u + (uint32_t)pos]);
                    w &= w - 1u;
                }
            }
        } else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
            /* Stop once every set bit of the mask has been consumed:
             * positions ascend across words and the mask holds exactly
             * `nexc` of them. */
            for (wi = 0; wi < 8u && k < (uint16_t)nexc; wi++) {
                uint32_t w = words[wi];

                while (w != 0u) {
                    uint16_t pos = (uint16_t)(wi * 32u + tile_ctz32(w));
                    uint8_t h8 = stg[8u + (uint32_t)pos];
                    uint8_t h16 = stg[pos];

                    TILE_TAKE_EXC_MASKED(pos,
                                         (uint8_t)((2u * (unsigned)h8 -
                                                   (unsigned)h16) & 0xFFu),
                                         stg[16u + (uint32_t)pos]);
                    w &= w - 1u;
                }
            }
        } else { /* ZPSE_P_LANE_AVERAGE */
            /* Stop once every set bit of the mask has been consumed:
             * positions ascend across words and the mask holds exactly
             * `nexc` of them. */
            for (wi = 0; wi < 8u && k < (uint16_t)nexc; wi++) {
                uint32_t w = words[wi];

                while (w != 0u) {
                    uint16_t pos = (uint16_t)(wi * 32u + tile_ctz32(w));
                    uint8_t h8 = stg[8u + (uint32_t)pos];
                    uint8_t h16 = stg[pos];

                    TILE_TAKE_EXC_MASKED(pos,
                                         (uint8_t)(((unsigned)h8 + (unsigned)h16 +
                                                    1u) >> 1),
                                         stg[16u + (uint32_t)pos]);
                    w &= w - 1u;
                }
            }
        }
    }

    /* The O(8) epilogue is the same end state the scalar per-byte
     * updates reach, for every family and every nt. */
    tile_history_fixup(lanes, actual, nt, ent_last, ent_second);

    cl->k = k;
    cl->act_valid = 1u;
    cl->rmax = _rmax;
    cl->dmin = _dmin;
    cl->dmax = _dmax;
    {
        unsigned i;
        uint16_t nu = 0;

        for (i = 0; i < 8u; i++)
            nu = (uint16_t)(nu + (uint16_t)_mm_popcnt_u32(cl->seen[i]));
        if (nu > (uint16_t)ZPSE_TILE_NU_CAP) {
            tile_seen_trim(cl->seen, ZPSE_TILE_NU_CAP);
            nu = (uint16_t)ZPSE_TILE_NU_CAP;
        }
        cl->nu = nu;
    }
}
#endif /* ZPSE_TILE_HAVE_AVX2 */

/* Opt-in trial accounting (-DZPSE_TRIAL_STATS): a count of classify calls
 * per predictor id. Nothing about the encoder's candidates is observable
 * from outside it, so "which predictors does this corpus actually trial?"
 * cannot be measured without a counter -- and the answer decides where
 * encoder optimisation effort should go (a predictor that is never
 * prescreened costs nothing, however slow its classify is). The counters
 * exist only in a build that defines the macro; `tests/benchmarks/
 * bench_columnar.c stats` is the reader, and it reports the counts per
 * corpus case and per searched tile. A classify call is one candidate
 * attempt: one predictor, one tile, one pass. */
#if defined(ZPSE_TRIAL_STATS)
uint32_t zpse_trial_calls[12];
#endif

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
#if defined(ZPSE_TRIAL_STATS)
    zpse_trial_calls[predictor_id]++;
#endif
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
    cl->act_valid = 0u;
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
    cl->act_valid = 1u;
    cl->rmax = _rmax;
    cl->dmin = _dmin;
    cl->dmax = _dmax;
    /* cl->nu is exactly the number of distinct exclusion residuals up to
     * ZPSE_TILE_NU_CAP, so it is the popcount of the 256-bit set the loop
     * just built. Doing it once here is much cheaper than a
     * test-and-increment per exception inside the byte loop, and gives the
     * same value that choose_modes' palette sizing depends on. The scalar
     * path records every distinct residual (the dense path stops at the
     * cap), so the set is trimmed here to keep popcount(seen) == nu. */
    {
        unsigned l;
        uint16_t nu = 0;

        for (l = 0; l < 8u; l++)
            nu = (uint16_t)(nu + (uint16_t)tile_popcount32(cl->seen[l]));
        if (nu > (uint16_t)ZPSE_TILE_NU_CAP) {
            tile_seen_trim(cl->seen, ZPSE_TILE_NU_CAP);
            nu = (uint16_t)ZPSE_TILE_NU_CAP;
        }
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
    if (predictor_id > (uint8_t)ZPSE_P_STRIDE_3)
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

/* SWAR population count, spelled out rather than left to
 * __builtin_popcount: without -mpopcnt GCC turns the builtin into a call to
 * __popcountdi2, and this one runs once or twice per tile, in the encoder's
 * and the pricing path's mode choice. Six instructions inline beat two
 * calls (measured: the pricing path's extra mode choice cost 12-18% of the
 * `random` row through those calls alone). */
static unsigned tile_pop32(uint32_t w)
{
    w = w - ((w >> 1) & 0x55555555u);
    w = (w & 0x33333333u) + ((w >> 2) & 0x33333333u);
    w = (w + (w >> 4)) & 0x0F0F0F0Fu;
    return (unsigned)((w * 0x01010101u) >> 24);
}

/* Payload bytes ZPSE_M_RUNS needs for this mask: one byte per maximal run
 * of equal bits, where the run sequence starts with a HIT run (an
 * exception at position 0 is a zero-length hit run, which the sequence
 * carries as its own byte). A transition count gives that in eight word
 * operations -- bit p differs from bit p-1 counts as one transition, so
 * runs == transitions + 1. A run can never span the whole tile here (the
 * callers settle k == 0 and k == nt before looking at RUNS), so nt <= 256
 * bounds every run below the 255 split and no extra byte is due. */
static uint32_t tile_mask_runs_len(const uint32_t words[8], uint16_t nt,
                                   uint32_t limit)
{
    uint32_t runs = 1u;
    uint32_t prev = 0u;
    unsigned wi;

    for (wi = 0; wi < 8u; wi++) {
        uint32_t w = words[wi];
        uint32_t trans = w ^ ((w << 1) | prev);
        uint32_t keep = ~0u;

        if ((uint32_t)(wi + 1u) * 32u > (uint32_t)nt) {
            uint32_t rem = (uint32_t)nt - (uint32_t)wi * 32u;

            keep = (rem >= 32u) ? ~0u : ((1u << rem) - 1u);
        }
        runs += (uint32_t)tile_pop32(trans & keep);
        /* The count only grows: once it reaches the length RUNS has to
         * beat, this mask cannot win and the rest of the walk is dead
         * work. A noisy tile bails after a word or two; the clustered
         * masks the mode is for are the ones that walk all eight. */
        if (runs >= limit)
            return limit;
        prev = w >> 31;
    }
    return runs;
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

    /* RUNS is the one remaining mask mode that can undercut both of those,
     * and it does so exactly when the exceptions come in few long runs --
     * a burst of changed bytes rather than scattered noise: one byte per
     * run against one byte per exception (SPARSE_U8) or the fixed
     * ceil(nt/8) bitmap. It costs eight popcounts here to rule out and a
     * byte loop to decode, the same walk SPARSE_U8 already does, so it is
     * priced unconditionally rather than behind a heuristic. */
    if (k >= 4u && k != nt) {
        uint32_t cur = (best_m == (uint8_t)ZPSE_M_SPARSE_U8)
                           ? (uint32_t)k
                           : rb;

        if (tile_mask_runs_len(cl->words, nt, cur) < cur)
            best_m = (uint8_t)ZPSE_M_RUNS;
    }

    best_rlen = tile_res_len(0u, cl, nt, k);
    for (rm = 1u; rm <= (uint8_t)ZPSE_R_SIGNED_DELTA; rm++) {
        uint16_t l = tile_res_len(rm, cl, nt, k);
        /* RAW_ACTUAL costs exactly what EXCLUSION_U8 does (one byte per
         * exception), so on a tie it wins whenever a real mix of hits and
         * exceptions is present: the decoder then copies the byte instead
         * of undoing the exclusion against the previous output, which
         * removes a serial dependency from the reconstruction. The two
         * extremes keep EXCLUSION_U8 -- k == 0 has no residual at all and
         * k == nt is the verbatim fast path. */
        if (l < best_rlen ||
            (l == best_rlen && rm == (uint8_t)ZPSE_R_RAW_ACTUAL &&
             best_r == 0u && k != 0u && k != nt)) {
            best_rlen = l;
            best_r = rm;
        }
    }

    *mask_mode_out = best_m;
    *res_mode_out = best_r;
}


/* Gather actual[p] at every exception position, ascending (the RAW_ACTUAL
 * residual payload). Scalar reference first; the AVX2 form compacts 16
 * positions per step with the same byte-shuffle table the dense
 * classification uses and writes up to 7 bytes past the last real entry,
 * so `out` must have 8 spare bytes. */
static void tile_gather_act_scalar(const uint8_t *actual, uint16_t nt,
                                   const uint32_t words[8], uint8_t *out)
{
    uint16_t j = 0;
    uint16_t p;

    for (p = 0; p < nt; p++) {
        if ((words[p >> 5] >> (p & 31u)) & 1u)
            out[j++] = actual[p];
    }
}

#if ZPSE_TILE_HAVE_AVX2
static ZPSE_TILE_AVX2P void tile_gather_act_avx2(const uint8_t *actual,
                                                  uint16_t nt,
                                                  const uint32_t words[8],
                                                  uint8_t *out)
{
    uint8_t buf[272];
    unsigned k = 0u;
    unsigned p;

    memcpy(buf, actual, nt);
    memset(buf + nt, 0, sizeof buf - nt);
    for (p = 0; p < (unsigned)nt; p += 16u) {
        unsigned m = (words[p >> 5] >> (p & 31u)) & 0xFFFFu;
        unsigned m0 = m & 0xFFu;
        unsigned m1 = m >> 8;
        __m128i d = _mm_loadu_si128((const __m128i *)(const void *)(buf + p));
        __m128i c0 = _mm_loadl_epi64((const __m128i *)(const void *)tile_cmp_lut[m0]);
        __m128i c1 = _mm_loadl_epi64((const __m128i *)(const void *)tile_cmp_lut[m1]);

        _mm_storel_epi64((__m128i *)(void *)(out + k), _mm_shuffle_epi8(d, c0));
        k += (unsigned)_mm_popcnt_u32(m0);
        _mm_storel_epi64((__m128i *)(void *)(out + k),
                         _mm_shuffle_epi8(_mm_srli_si128(d, 8), c1));
        k += (unsigned)_mm_popcnt_u32(m1);
    }
}
#endif

int zpse_tile_pack(uint8_t predictor_id, const ZpseTileClass *cl,
                   uint16_t nt, const uint8_t *actual,
                   uint8_t mask_mode, uint8_t res_mode,
                   int tile_reset, uint8_t *record_out,
                   uint32_t record_cap, uint8_t *comb_scratch)
{
    const uint32_t *words;
    uint16_t k;
    uint8_t avals[272];
    const uint8_t *act_src = 0;
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
    if (predictor_id > (uint8_t)ZPSE_P_STRIDE_3)
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
        if (cl->act_valid) {
            /* classify already collected them */
            act_src = cl->act_at_exc;
        } else {
#if ZPSE_TILE_HAVE_AVX2
            if ((zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u)
                tile_gather_act_avx2(actual, nt, words, avals);
            else
#endif
                tile_gather_act_scalar(actual, nt, words, avals);
            act_src = avals;
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
    /* The classification counted the distinct exclusion residuals (the
     * mode search needed `nu` to size the palette), so it can certify the
     * palette packer's value -> index map and save the packer a full scan
     * of the tile. Only PALETTE_BITS reads the certificate. */
    rrc = zpse_residual_encode_hint(res_mode, use_vals ? cl->res_at_exc : 0,
                                    use_act ? act_src : 0, cl->at_pred, k,
                                    record_out + off,
                                    tile_cap16(record_cap - off),
                                    cl->nu, cl->seen);
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

/* ---- STRIDE_1: chain over the exceptions, then a vector fill ------ *
 * For stride 1 the prediction at p is out[p-1], so the reconstruction is
 * one recurrence over the tile's Nt positions and, on a tile with a dense
 * mask, every one of those positions sits on the critical path: measured
 * at ~5 cycles per byte through the byte-at-a-time loop below, which made
 * PALETTE16-style plates (STRIDE_1 + RAW_BITMAP, 118 exceptions in 256)
 * the slowest row of the whole decode corpus.
 *
 * The recurrence only MOVES at an exception. Between two of them
 * out[p] = out[p-1] holds the last exception's value, so the tile is
 * completely described by the sequence of values its K exceptions take,
 * plus the entry value (the segment byte before the tile). That splits
 * the work in two:
 *
 *   1. a chain over K values -- not Nt -- in exception order, each one a
 *      single step of the same predictor (r + (r >= x_hat) for the
 *      exclusion modes, x_hat + delta for SIGNED_DELTA, and the raw byte
 *      for RAW_ACTUAL, which needs no chain at all), and
 *   2. a fill: out[p] = w[number of exceptions at or before p], where
 *      w[0] is the entry value and w[i] is the value of exception i-1.
 *      That is a gather, not a recurrence: eight positions at a time, the
 *      index inside the group is the inclusive popcount of the group's
 *      mask byte (a three-step shift-and-add ladder), and one vpshufb
 *      picks the group's eight bytes out of the window w[E0..E0+15], E0
 *      being the number of exceptions before the group. Every index is
 *      0..8, which is why the window is 16 bytes and the group is 8 lanes
 *      rather than 16: a 16-lane ladder reaches 16, and a shuffle index
 *      of 16 wraps to 0.
 *
 * The bytes produced, the units consumed (ascending position order, one
 * per set mask bit) and the returned count are identical to the scalar
 * loop's; only the order in which independent work happens changes. Bits
 * at or above Nt are cleared from the mask first, exactly as the p < nt
 * loop bound ignored them, so a mask with stray high bits still consumes
 * its valid bits only.
 */

/* Fill `out` from the chain values, 32 positions per step.
 *
 * The index of a position is the number of exceptions at or before it,
 * counted inside the chunk, and the chunk is a 32-bit mask word: byte lane
 * i of the mask expansion tests bit (i & 7) of the word's byte (i >> 3),
 * and the same per-128-bit-lane ladder as tile_recon_const_avx2 turns
 * those bits into per-lane counts. Two vpshufb lookups then pick the
 * chunk's bytes: the low half from the window at w[E0], the high half from
 * w[E0 + popcount(low 16 bits)].
 *
 * A half whose 16 bits are all set is the one case a shuffle cannot
 * express -- its last lane's count is 16, and pshufb reads only four index
 * bits -- but that case needs no lookup at all: sixteen contiguous
 * exceptions ARE sixteen contiguous values, so the half is a straight
 * load. Both halves are tested with a compare on the word, which is what
 * keeps this branchless enough to be worth 32 lanes.
 *
 * The tail (fewer than 32 positions) drops to eight lanes, then to one,
 * with the same inclusive ladder: a 16-lane one is exactly what does not
 * fit a shuffle index. */
#if ZPSE_TILE_HAVE_AVX2
static ZPSE_TILE_AVX2 void tile_stride1_fill_avx2(uint8_t *out, uint32_t nt,
                                                  const uint32_t mw[8],
                                                  const uint8_t *w)
{
    const __m256i bitv32 = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80);
    const __m256i dup8 = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i one32 = _mm256_set1_epi8((char)1);
    const __m128i bitv = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)0x80,
                                       0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i one = _mm_set1_epi8((char)1);
    uint32_t at = 0u;
    uint32_t e0 = 0u;

    while (at + 32u <= nt) {
        uint32_t wi = mw[at >> 5u];
        /* 0/1 per lane, then the inclusive count within each 128-bit
         * half. Lane j of `inc` is the number of set bits at lanes
         * 0..j of its half. */
        __m256i v1 = _mm256_and_si256(
            _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(_mm256_set1_epi32((int)wi), dup8),
                    bitv32),
                bitv32),
            one32);
        __m256i inc = _mm256_add_epi8(v1, _mm256_slli_si256(v1, 1));
        __m256i win;
        __m256i r;
        uint32_t lo_cnt = (uint32_t)_mm_popcnt_u32(wi & UINT32_C(0xFFFF));
        __m128i lo128 = _mm_loadu_si128(
            (const __m128i *)(const void *)(w + e0));
        __m128i hi128 = _mm_loadu_si128(
            (const __m128i *)(const void *)(w + e0 + lo_cnt));

        inc = _mm256_add_epi8(inc, _mm256_slli_si256(inc, 2));
        inc = _mm256_add_epi8(inc, _mm256_slli_si256(inc, 4));
        inc = _mm256_add_epi8(inc, _mm256_slli_si256(inc, 8));
        win = _mm256_inserti128_si256(_mm256_castsi128_si256(lo128), hi128,
                                      1);
        r = _mm256_shuffle_epi8(win, inc);
        if ((wi & UINT32_C(0xFFFF)) == UINT32_C(0xFFFF)) {
            r = _mm256_inserti128_si256(r, _mm_loadu_si128(
                (const __m128i *)(const void *)(w + e0 + 1u)), 0);
        }
        if ((wi >> 16) == UINT32_C(0xFFFF)) {
            r = _mm256_inserti128_si256(r, _mm_loadu_si128(
                (const __m128i *)(const void *)(w + e0 + lo_cnt + 1u)), 1);
        }
        _mm256_storeu_si256((__m256i *)(void *)(out + (size_t)at), r);
        e0 += (uint32_t)_mm_popcnt_u32(wi);
        at += 32u;
    }

    while (at + 8u <= nt) {
        uint32_t m8 = (mw[at >> 5u] >> (at & 31u)) & UINT32_C(0xFF);
        __m128i b1 = _mm_and_si128(
            _mm_cmpeq_epi8(
                _mm_and_si128(_mm_set1_epi8((char)m8), bitv), bitv),
            one);
        __m128i idx;

        /* Inclusive prefix count of the group's set bits: after the three
         * steps lane j holds the number of set bits at lanes 0..j. */
        idx = _mm_add_epi8(b1, _mm_slli_si128(b1, 1));
        idx = _mm_add_epi8(idx, _mm_slli_si128(idx, 2));
        idx = _mm_add_epi8(idx, _mm_slli_si128(idx, 4));
        _mm_storel_epi64(
            (__m128i *)(void *)(out + (size_t)at),
            _mm_shuffle_epi8(
                _mm_loadu_si128(
                    (const __m128i *)(const void *)(w + e0)),
                idx));
        /* POPCNT: this function is built with the AVX2 target, which
         * implies it, and the SWAR form would cost more than the ladder. */
        e0 += (uint32_t)_mm_popcnt_u32(m8);
        at += 8u;
    }
    while (at < nt) {
        e0 += (uint32_t)((mw[at >> 5u] >> (at & 31u)) & 1u);
        out[at] = w[e0];
        at++;
    }
}
#endif /* ZPSE_TILE_HAVE_AVX2 */

#if ZPSE_TILE_HAVE_AVX2
/* Bit-parallel exclusion chain.
 *
 * The exclusion step is out[j] = r[j] + (r[j] >= out[j-1]): a byte-valued
 * recurrence that the scalar loop below walks one unit at a time, and one
 * cmp+add pair of dependent ALU ops per unit is its floor -- measured at
 * 1.6 cycles per exception, which on a palette16 tile (~118 exceptions in
 * 256 positions) is most of the reconstruction time.
 *
 * The comparison can be rewritten against r alone, so the chain becomes a
 * carry chain and the carry chains of adjacent units compose.
 *
 * Write b[j] = (r[j] >= out[j-1]) (so out[j] = r[j] + b[j]), g[j] =
 * (r[j] >= r[j-1]) and q[j] = (r[j] == r[j-1]). Then b[j] is g[j] when
 * b[j-1] is 0, and the strict comparison r[j] > r[j-1] when b[j-1] is 1:
 *
 *   b[j] = g[j] ^ (q[j] & b[j-1])
 *
 * (checked against the three cases: r[j] > r[j-1] gives g=1,q=0 -> 1;
 * r[j] == r[j-1] gives g=1,q=1 -> !b[j-1]; r[j] < r[j-1] gives g=0,q=0
 * -> 0). That is the carry recurrence of an adder with generate g,
 * propagate q and xor as the carry combine, so the usual Kogge-Stone
 * prefix applies: with (G,P) as the composed map x -> P*x ^ G of a span,
 *   combine(lo, hi) = (G_hi ^ (P_hi & G_lo), P_hi & P_lo).
 * Six shift-and steps over a 64-bit word give every b[0..63] at once.
 *
 * The chain's entry is r[-1] := the previous output value (w[0]), because
 * b[0] = (r[0] >= w[0]) is exactly the first scalar step. So the units
 * are copied into a padded local with that byte in front and everything
 * is uniform -- no special case for j == 0, and the class boundary is just
 * the word boundary (the carry out of one block is the carry in of the
 * next). The comparisons themselves are 32 bytes at a time: g from
 * max_epu8 + cmpeq, q from cmpeq, both to a 32-bit mask with movemask.
 * r is read from the untouched copy, never from the output, so the input
 * of the next block is still the residuals.
 *
 * out[j] is then r[j] plus a 0/1 byte built from bit j of b; two table
 * lookups per eight units expand the mask (no AVX2 instruction turns mask
 * bits into bytes before AVX-512). Both stores of a block are 32 bytes
 * wide, so the last block may write up to 63 bytes past the last unit;
 * the caller's `w` has that much slack and the fill never reads past
 * K + 16.
 *
 * The blocks are otherwise independent: b for one 64-unit block needs
 * only that block's comparisons and the previous block's final carry.
 * Two blocks are therefore scanned in one pass (see the loop), which
 * keeps 128 units' worth of chain at the latency of one scan; that is
 * exactly the two-block tile a palette16 plane produces.
 *
 * One shape the rewrite does NOT cover is a unit of 255. The format's
 * exclusion residuals are in [0,254] (every residual decoder enforces
 * that), where the recurrence never wraps; with a 255 the scalar loop's
 * out becomes 0 and the next comparison is against 0, which no comparison
 * of r[j] with r[j-1] can reproduce. So the block loop also flags any
 * 0xFF it sees and reports it, and the caller falls back to the scalar
 * loop -- which recomputes w[1..K] from scratch, so a partially written
 * w from this attempt is harmless. The check is one compare and one
 * movemask per 32 units on the vector already in a register.
 */
/* 32 mask bits -> 32 bytes, each 0 or 1: byte j = bit j of `m`.
 *
 * A byte of the mask broadcast to its eight positions, ANDed with the
 * per-position bit, compared against zero and subtracted from zero is
 * 0/1 in one pass: compare gives 0xFF where the bit was clear, and 0 -
 * 0xFF is 1. Four vector ops replace sixteen table loads (two per byte
 * of the mask) and the eight 64-bit inserts that merged them, which is
 * what this step used to cost. */
static ZPSE_TILE_AVX2 __attribute__((always_inline)) inline __m256i
tile_bits_to_bytes32(uint32_t m)
{
    const __m256i pick = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80);
    const __m256i z = _mm256_setzero_si256();
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)m), pick);

    v = _mm256_and_si256(v, bit);
    /* A byte is 0 where its bit was clear, so the compare clears exactly
     * the bytes whose bit was set: andnot with the 0/1 constant is the
     * mask itself. */
    return _mm256_andnot_si256(_mm256_cmpeq_epi8(v, z),
                               _mm256_set1_epi8(1));
}

/* Bit j = a[j] >= b[j], unsigned, for 32 byte pairs already loaded. */
static ZPSE_TILE_AVX2 __attribute__((always_inline)) inline uint32_t
tile_ge_mask_v(__m256i va, __m256i vb)
{
    return (uint32_t)_mm256_movemask_epi8(
        _mm256_cmpeq_epi8(_mm256_max_epu8(va, vb), va));
}

/* Bit j = a[j] == b[j], for 32 byte pairs already loaded. */
static ZPSE_TILE_AVX2 __attribute__((always_inline)) inline uint32_t
tile_eq_mask_v(__m256i va, __m256i vb)
{
    return (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb));
}

/* w[0..64) = r + b for one 64-unit block, `G` holding bit j = b[j] and
 * va / vah the two 32-byte halves of r. */
static ZPSE_TILE_AVX2 __attribute__((always_inline)) inline void
tile_chain_store64(uint8_t *w, __m256i va, __m256i vah, uint64_t G)
{
    _mm256_storeu_si256(
        (__m256i *)(void *)w,
        _mm256_add_epi8(va, tile_bits_to_bytes32((uint32_t)G)));
    _mm256_storeu_si256(
        (__m256i *)(void *)(w + 32u),
        _mm256_add_epi8(vah, tile_bits_to_bytes32((uint32_t)(G >> 32))));
}

/* Per-bit prefix AND of `q`: one for every bit below q's first zero.
 * q is a run of ones from bit 0 up to that zero, so q + 1 carries into
 * the zero and clears the run; q ^ (q + 1) marks the run and the zero,
 * and shifting right leaves exactly the run. Three dependent ALU ops,
 * against the ctz/shift/subtract form's five, and it sits on the
 * carry-in path. The all-ones word -- no zero to stop at -- is the one
 * input the identity misses, so it is patched with a compare. */
static inline uint64_t tile_prefix_and64(uint64_t q)
{
    return ((q ^ (q + UINT64_C(1))) >> 1) |
           ((uint64_t)0 - (uint64_t)(q == ~UINT64_C(0)));
}

/* w[0..n) = the exception values of the exclusion chain over
 * units[0..n) entered with `entry`. Returns non-zero (and leaves w
 * unspecified) if a unit is 255: see the note above. */
/* rbuf = [entry][units[0..n)][32 zero bytes]. The two libc calls this
 * replaces cost a fixed ~60 cycles per tile even when n is 8 (the
 * fortify variants add a check and a call each), which is more than the
 * scan they feed; the copy is at most 256 bytes and all of it is one
 * 32-byte store per step. A tail that is not a multiple of 32 is written
 * by one overlapping 32-byte store ending at n, so every byte of
 * units[0..n) is copied exactly once per byte and nothing past n is read.
 * The 64 bytes of zero padding cover the widest load the block loops do
 * (a second 32-byte vector past the last unit). */
static ZPSE_TILE_AVX2 void tile_chain_stage(uint8_t *rbuf,
                                            const uint8_t *units, uint32_t n,
                                            uint8_t entry)
{
    const __m256i z = _mm256_setzero_si256();
    uint8_t *d = rbuf + 1u;
    uint32_t i = 0u;

    rbuf[0] = entry;
    if (n >= 32u) {
        while (i + 32u <= n) {
            _mm256_storeu_si256(
                (__m256i *)(void *)(d + i),
                _mm256_loadu_si256((const __m256i *)(const void *)(units + i)));
            i += 32u;
        }
        if (i != n) {
            _mm256_storeu_si256(
                (__m256i *)(void *)(d + n - 32u),
                _mm256_loadu_si256(
                    (const __m256i *)(const void *)(units + n - 32u)));
        }
    } else {
        for (i = 0u; i < n; i++) {
            d[i] = units[i];
        }
    }
    _mm256_storeu_si256((__m256i *)(void *)(d + n), z);
    _mm256_storeu_si256((__m256i *)(void *)(d + n + 32u), z);
}

static ZPSE_TILE_AVX2 int tile_chain_excl_avx2(uint8_t *w,
                                               const uint8_t *units,
                                               uint32_t n, uint8_t entry)
{
    uint8_t rbuf[1u + ZPSE_MAX_TILE_LEN + 96u];
    const __m256i all_ff = _mm256_set1_epi8((char)0xFF);
    uint32_t bad = 0u;
    uint64_t carry = 0u;
    uint32_t i = 0u;

    tile_chain_stage(rbuf, units, n, entry);

    /* Two blocks per pass whenever a second block exists at all (more
     * than 64 units left). The two scans share no state -- only the
     * carry-in fixups that follow them do -- so both dependency chains
     * run in one body and cost the latency of one scan instead of two.
     * 118 exceptions (the palette16 shape) is two blocks, so the common
     * tile is one pass. The first block is always full here; the second
     * carries the tail mask when fewer than 64 units remain. The prefix
     * ANDs p0 / p1 depend on eq alone, so they are computed alongside
     * the scan and the fixup chain is just xor, shift, negate per block. */
    while (n - i > 64u) {
        uint32_t cnt1 = n - i - 64u;
        const uint8_t *a0 = rbuf + 1u + i;

        if (cnt1 > 64u) {
            cnt1 = 64u;
        }
        const uint8_t *a1 = a0 + 64u;
        uint64_t keep1;
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)a0);
        __m256i v0h =
            _mm256_loadu_si256((const __m256i *)(const void *)(a0 + 32u));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)a1);
        __m256i v1h =
            _mm256_loadu_si256((const __m256i *)(const void *)(a1 + 32u));
        __m256i b0 = _mm256_loadu_si256((const __m256i *)(const void *)(a0 - 1u));
        __m256i b0h =
            _mm256_loadu_si256((const __m256i *)(const void *)(a0 + 31u));
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(const void *)(a1 - 1u));
        __m256i b1h =
            _mm256_loadu_si256((const __m256i *)(const void *)(a1 + 31u));
        uint64_t eq0 = (uint64_t)tile_eq_mask_v(v0, b0) |
                       ((uint64_t)tile_eq_mask_v(v0h, b0h) << 32);
        uint64_t eq1 = (uint64_t)tile_eq_mask_v(v1, b1) |
                       ((uint64_t)tile_eq_mask_v(v1h, b1h) << 32);
        uint64_t G0 = (uint64_t)tile_ge_mask_v(v0, b0) |
                      ((uint64_t)tile_ge_mask_v(v0h, b0h) << 32);
        uint64_t G1 = (uint64_t)tile_ge_mask_v(v1, b1) |
                      ((uint64_t)tile_ge_mask_v(v1h, b1h) << 32);
        uint64_t P0;
        uint64_t P1;
        uint64_t p0;
        uint64_t p1;
        unsigned s;

        if (cnt1 < 64u) {
            /* Bits past the last unit would otherwise compare the zero
             * padding and come out as an endless run of generate bits. */
            keep1 = (UINT64_C(1) << cnt1) - 1u;
            G1 &= keep1;
            eq1 &= keep1;
        }
        p0 = tile_prefix_and64(eq0);
        p1 = tile_prefix_and64(eq1);
        P0 = eq0;
        P1 = eq1;
        bad |= (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, all_ff));
        bad |= (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0h, all_ff));
        bad |= (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, all_ff));
        bad |= (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1h, all_ff));

        for (s = 1u; s < 64u; s <<= 1u) {
            G0 ^= P0 & (G0 << s);
            P0 &= P0 << s;
            G1 ^= P1 & (G1 << s);
            P1 &= P1 << s;
        }
        G0 ^= p0 & carry;
        carry = (uint64_t)0 - (G0 >> 63);
        G1 ^= p1 & carry;
        carry = (uint64_t)0 - (G1 >> 63);
        tile_chain_store64(w + i, v0, v0h, G0);
        tile_chain_store64(w + i + 64u, v1, v1h, G1);
        i += 64u + cnt1;
    }
    while (i < n) {
        uint32_t cnt = n - i;
        const uint8_t *a = rbuf + 1u + i;
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)a);
        __m256i vah =
            _mm256_loadu_si256((const __m256i *)(const void *)(a + 32u));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(a - 1u));
        __m256i vbh =
            _mm256_loadu_si256((const __m256i *)(const void *)(a + 31u));
        uint64_t ge = (uint64_t)tile_ge_mask_v(va, vb) |
                      ((uint64_t)tile_ge_mask_v(vah, vbh) << 32);
        uint64_t eq = (uint64_t)tile_eq_mask_v(va, vb) |
                      ((uint64_t)tile_eq_mask_v(vah, vbh) << 32);
        uint64_t G;
        uint64_t P;
        unsigned s;

        bad |= (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, all_ff));
        bad |= (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(vah, all_ff));
        if (cnt < 64u) {
            /* Bits past the last unit would otherwise compare the zero
             * padding and come out as an endless run of generate bits. */
            uint64_t keep = (UINT64_C(1) << cnt) - 1u;
            ge &= keep;
            eq &= keep;
        }
        G = ge;
        P = eq;
        for (s = 1u; s < 64u; s <<= 1u) {
            G ^= P & (G << s);
            P &= P << s;
        }
        /* Carry in from the previous block. The scan's own P is not the
         * prefix AND -- `P &= P << s` zeroes bit j for j < s, and after
         * six steps it is all zero -- but it is exactly right *inside*
         * the loop, where G only ever reads P at bits >= s. */
        G ^= tile_prefix_and64(eq) & carry;
        carry = (uint64_t)0 - (G >> 63);
        tile_chain_store64(w + i, va, vah, G);
        i += (cnt < 64u) ? cnt : 64u;
    }
    /* The padding past the last unit is zero, so it can never set a bit
     * here; every bit counted is a real unit. */
    return (int)(bad != 0u);
}
#endif /* ZPSE_TILE_HAVE_AVX2 */

#if ZPSE_TILE_HAVE_AVX2
/* Prefix sum of int8 offsets, modulo 256: dst[j] = run + sum(d[-1..j]).
 *
 * This is the SIGNED_DELTA form of the STRIDE_1 chain, and unlike the
 * exclusion form it is associative -- so the serial recurrence the scalar
 * loop below walks two cycles at a time becomes four shift-and-add steps
 * per 16 units. PADDB is byte-wise with wraparound, which is exactly the
 * arithmetic the scalar loop does, and the running total is carried into
 * the next block by broadcasting the last byte of the block just
 * finished. A tile of 242 exceptions (the two-byte-record shape) is 15
 * blocks, so the chain stops being the tile's critical path. */
static ZPSE_TILE_AVX2 void tile_prefix_sum_avx2(uint8_t *dst,
                                                const uint8_t *deltas,
                                                uint32_t n, uint8_t run)
{
    const __m128i last = _mm_set1_epi8((char)15);
    __m128i carry = _mm_set1_epi8((char)run);
    uint32_t i = 0u;

    while (i + 16u <= n) {
        __m128i d = _mm_loadu_si128(
            (const __m128i *)(const void *)(deltas + i));

        d = _mm_add_epi8(d, _mm_slli_si128(d, 1));
        d = _mm_add_epi8(d, _mm_slli_si128(d, 2));
        d = _mm_add_epi8(d, _mm_slli_si128(d, 4));
        d = _mm_add_epi8(d, _mm_slli_si128(d, 8));
        d = _mm_add_epi8(d, carry);
        _mm_storeu_si128((__m128i *)(void *)(dst + i), d);
        carry = _mm_shuffle_epi8(d, last);
        i += 16u;
    }
    if (i != 0u) {
        run = (uint8_t)_mm_cvtsi128_si32(carry);
    }
    for (; i < n; i++) {
        run = (uint8_t)((unsigned)run + (unsigned)(int)(int8_t)deltas[i]);
        dst[i] = run;
    }
}
#endif /* ZPSE_TILE_HAVE_AVX2 */

/* Reconstruct a STRIDE_1 tile. Returns the number of residual units
 * consumed (the caller compares it against K). `units` is resid[] for the
 * exclusion modes and deltas[] for SIGNED_DELTA, whose entries are int8
 * offsets from the previous exception's value. */
static uint16_t tile_stride1_recon(uint8_t *out, uint16_t nt,
                                   const uint8_t *seg_base, uint32_t seg_pos,
                                   uint8_t init, const uint32_t words[8],
                                   uint32_t total, const uint8_t *units,
                                   uint8_t res_mode)
{
    /* w[0] is the entry value; w[1..total] are the exception values. The
     * tail is slack for the fill's 16-byte window reads plus (for the
     * bit-parallel chain) the 32-byte store granularity of its last
     * block, which writes up to 63 bytes past the final unit; it is
     * zeroed for the fill's reads.
     *
     * `total` is the exception count the mask decode already proved equal
     * to the bit count of `words`, so this reconstructor does not count
     * them again: eight SWAR popcounts were pure duplicate work on every
     * STRIDE_1 tile. */
    uint8_t w[1u + ZPSE_MAX_TILE_LEN + 96u];
    uint32_t mw[8];
    uint32_t nw = ((uint32_t)nt + 31u) >> 5u;
    uint32_t i;
    uint8_t run;

    memcpy(mw, words, sizeof mw);
    if ((nt & 31u) != 0u) {
        mw[nw - 1u] &= (UINT32_C(1) << (nt & 31u)) - 1u;
    }
    for (i = nw; i < 8u; i++) {
        mw[i] = 0u;
    }

    run = (seg_pos > 0u) ? seg_base[seg_pos - 1u] : init;
    w[0] = run;
    if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
        /* The residual IS the value, so every exception is independent. */
        if (total != 0u) {
            memcpy(w + 1u, units, (size_t)total);
        }
    } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
#if ZPSE_TILE_HAVE_AVX2
        if (total >= 16u &&
            (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
            tile_prefix_sum_avx2(w + 1u, units, total, run);
        } else
#endif
        {
            for (i = 0; i < total; i++) {
                run = (uint8_t)((unsigned)run +
                                (unsigned)(int)(int8_t)units[i]);
                w[i + 1u] = run;
            }
        }
    } else {
        /* Exclusion inverse against the previous exception's value. */
        int done = 0;
#if ZPSE_TILE_HAVE_AVX2
        if (total >= 16u &&
            (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
            done = (tile_chain_excl_avx2(w + 1u, units, total, run) == 0);
        }
#endif
        if (!done) {
            for (i = 0; i < total; i++) {
                uint8_t r = units[i];

                run = (uint8_t)(r + (uint8_t)(r >= run));
                w[i + 1u] = run;
            }
        }
    }
    memset(w + total + 1u, 0, 32u);

    /* An all-exception tile (a palette or noise plane under STRIDE_1) would
     * look like the one mask the fill has nothing to do for -- the rank of
     * position p is p + 1, so out is the exception values end to end. The
     * obvious shortcut, a copy in place of the ladder-and-shuffle kernel,
     * was measured and is a loss at every width tried (8-byte steps,
     * 32-byte vectors, even memcpy): that kernel is ~25 instructions per 32
     * positions and already runs out of the same registers the values
     * arrive in, while a 256-byte copy has to go through the store buffer.
     * So the fill runs for every mask and this case needs no branch. */

#if ZPSE_TILE_HAVE_AVX2
    if ((uint32_t)nt >= 16u &&
        (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        tile_stride1_fill_avx2(out, (uint32_t)nt, mw, w);
        return (uint16_t)total;
    }
#endif
    {
        uint32_t at = 0u;
        uint32_t e0 = 0u;

        while (at < (uint32_t)nt) {
            e0 += (uint32_t)((mw[at >> 5u] >> (at & 31u)) & 1u);
            out[at] = w[e0];
            at++;
        }
    }
    return (uint16_t)total;
}

/* Branchless STRIDE-family reconstruction for strides 1, 2, 4 and 8.
 *
 * Stride s predicts x[p] = x[p - s]; the serial dependence is through the
 * previous output, so the loop is latency-bound rather than throughput-
 * bound, and the old per-byte ternary/branch form paid a mispredict on
 * every irregular mask bit (15+ cycles per byte on text). Here the
 * exception select is an XOR/AND mask and the residual index advances by
 * the mask bit, so there is no data-dependent branch at all.
 *
 * For s > 1 the residue classes mod s are s INDEPENDENT chains: position
 * p's prediction reads out[p - s], which belongs to the same chain and to
 * no other position in between, so nothing couples the chains. On a DENSE
 * mask - the case that matters, text and noise at 200+ exceptions in 256
 * positions - reading out[p - s] back from memory puts the store-to-load
 * latency of s positions ago straight onto the critical path, and the loop
 * runs one chain at a time. The loop therefore unrolls by exactly s, one
 * chain kept in a register each: the s positions of a step are mutually
 * independent, so their latency overlaps instead of adding up. The
 * residual units are still consumed in ascending position order: each
 * step's expansions run in position order and the steps themselves are
 * ascending.
 *
 * For s == 1 there is one chain and no interleaving to be had; the chain
 * lives in `prev` and the loop stays as it was. Positions below the stride
 * read the segment's earlier bytes (or the initial literal at the very
 * start of the segment) exactly as before. Returns the number of residual
 * units consumed (compared against k by the caller). */
static uint16_t tile_stride_recon_scalar(uint8_t *out, uint16_t nt, uint32_t s,
                                         const uint8_t *seg_base,
                                         uint32_t seg_pos, uint8_t init,
                                         const uint32_t words[8],
                                         const uint8_t *resid,
                                         const int8_t *deltas,
                                         uint8_t res_mode)
{
    uint32_t j = 0u;
    uint32_t p = 0u;
    uint8_t prev = 0u;
    uint32_t head = (s < (uint32_t)nt) ? s : (uint32_t)nt;
    /* This loop reads a mask bit per position while storing that position's
     * byte through `out`, which is a uint8_t*, so the compiler has to treat
     * every store as a possible write to `words` and reload it: the mask
     * read comes back out of memory once per position even though nothing
     * here writes it. A local copy is an object no pointer in this function
     * can reach, so its eight words stay in registers (or at worst in a
     * stack slot the stores provably cannot touch). Same argument, and the
     * same hoist, in the two AVX2 kernels below. */
    uint32_t mw[8];

    memcpy(mw, words, sizeof mw);

    /* One position `PP`: take its prediction, invert the residual if the
     * mask marks the position, store, and leave the result as the front of
     * the chain (`CHAIN` is `prev` for the single-chain loops and one of
     * the interleaved registers otherwise). `mk_` is the mask bit expanded
     * to a byte mask, which is what makes the select branchless:
     * x_ = xh_ ^ ((xh_ ^ e_) & mk_). The position is a parameter rather
     * than an implicit `p` so the interleaved loops can expand several
     * positions of one step in ascending order without touching p, which
     * is what keeps the unit order the scalar loop had. */
#define ZPSE_STRIDE_STEP(PP, XHAT, CHAIN)                                   \
    do {                                                                    \
        uint32_t pp_ = (uint32_t)(PP);                                      \
        uint8_t xh_ = (uint8_t)(XHAT);                                      \
        uint32_t m_ = (mw[pp_ >> 5] >> (pp_ & 31u)) & 1u;                   \
        uint8_t mk_ = (uint8_t)(0u - m_);                                   \
        uint8_t x_;                                                         \
        if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {                       \
            uint8_t e_ = resid[j];                                          \
            x_ = (uint8_t)(xh_ ^ ((xh_ ^ e_) & mk_));                       \
        } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {              \
            uint8_t e_ = (uint8_t)((unsigned)xh_ + (int)deltas[j]);         \
            x_ = (uint8_t)(xh_ ^ ((xh_ ^ e_) & mk_));                       \
        } else {                                                            \
            uint8_t r_ = resid[j];                                          \
            uint8_t e_ = (uint8_t)(r_ + (uint8_t)(r_ >= xh_));              \
            x_ = (uint8_t)(xh_ ^ ((xh_ ^ e_) & mk_));                       \
        }                                                                   \
        out[pp_] = x_;                                                      \
        j += m_;                                                            \
        CHAIN = x_;                                                         \
    } while (0)

    /* Positions below the stride read the segment's earlier bytes (or the
     * initial literal at the very start of the segment). They also seed
     * the s chains: after this loop out[0 .. head-1] is final, and the
     * chain of position p is the one whose previous member is out[p - s]. */
    for (p = 0u; p < head; p++) {
        uint8_t xh = (seg_pos + p < s) ? init : seg_base[seg_pos + p - s];

        ZPSE_STRIDE_STEP(p, xh, prev);
    }

    if (s == 1u) {
        /* head == 1 here (nt >= 1); the chain is `prev`. */
        for (; p < (uint32_t)nt; p++)
            ZPSE_STRIDE_STEP(p, prev, prev);
    } else if (s == 2u && (uint32_t)nt >= 2u) {
        /* Two chains, one per residue class mod 2. out[0] and out[1] are
         * the current members of chain 0 and chain 1, which is exactly
         * what position 2 and position 3 predict from. */
        uint8_t c0 = out[0];
        uint8_t c1 = out[1];

        for (; p + 2u <= (uint32_t)nt; p += 2u) {
            ZPSE_STRIDE_STEP(p, c0, c0);
            ZPSE_STRIDE_STEP(p + 1u, c1, c1);
        }
        /* Fewer than s positions left: they read the finished bytes back,
         * which are the same values the chain registers hold. */
        for (; p < (uint32_t)nt; p++)
            ZPSE_STRIDE_STEP(p, out[p - s], prev);
    } else if (s == 3u && (uint32_t)nt >= 3u) {
        uint8_t c0 = out[0];
        uint8_t c1 = out[1];
        uint8_t c2 = out[2];

        for (; p + 3u <= (uint32_t)nt; p += 3u) {
            ZPSE_STRIDE_STEP(p, c0, c0);
            ZPSE_STRIDE_STEP(p + 1u, c1, c1);
            ZPSE_STRIDE_STEP(p + 2u, c2, c2);
        }
        for (; p < (uint32_t)nt; p++)
            ZPSE_STRIDE_STEP(p, out[p - s], prev);
    } else if (s == 4u && (uint32_t)nt >= 4u) {
        uint8_t c0 = out[0];
        uint8_t c1 = out[1];
        uint8_t c2 = out[2];
        uint8_t c3 = out[3];

        for (; p + 4u <= (uint32_t)nt; p += 4u) {
            ZPSE_STRIDE_STEP(p, c0, c0);
            ZPSE_STRIDE_STEP(p + 1u, c1, c1);
            ZPSE_STRIDE_STEP(p + 2u, c2, c2);
            ZPSE_STRIDE_STEP(p + 3u, c3, c3);
        }
        for (; p < (uint32_t)nt; p++)
            ZPSE_STRIDE_STEP(p, out[p - s], prev);
    } else if (s == 8u && (uint32_t)nt >= 8u) {
        uint8_t c0 = out[0];
        uint8_t c1 = out[1];
        uint8_t c2 = out[2];
        uint8_t c3 = out[3];
        uint8_t c4 = out[4];
        uint8_t c5 = out[5];
        uint8_t c6 = out[6];
        uint8_t c7 = out[7];

        for (; p + 8u <= (uint32_t)nt; p += 8u) {
            ZPSE_STRIDE_STEP(p, c0, c0);
            ZPSE_STRIDE_STEP(p + 1u, c1, c1);
            ZPSE_STRIDE_STEP(p + 2u, c2, c2);
            ZPSE_STRIDE_STEP(p + 3u, c3, c3);
            ZPSE_STRIDE_STEP(p + 4u, c4, c4);
            ZPSE_STRIDE_STEP(p + 5u, c5, c5);
            ZPSE_STRIDE_STEP(p + 6u, c6, c6);
            ZPSE_STRIDE_STEP(p + 7u, c7, c7);
        }
        for (; p < (uint32_t)nt; p++)
            ZPSE_STRIDE_STEP(p, out[p - s], prev);
    } else {
        for (; p < (uint32_t)nt; p++)
            ZPSE_STRIDE_STEP(p, out[p - s], prev);
    }
#undef ZPSE_STRIDE_STEP
    return (uint16_t)j;
}

/* ------------------------------------------------------------------ *
 * Vectorised reconstruction for tiles that have exceptions.
 *
 * A tile with K exceptions walks all Nt positions one byte at a time in
 * the loops at the end of zpse_tile_decode, and for the two families whose
 * prediction is a pure function of the tile's own lane histories -- CONST
 * and LANEHIST -- every position performs the same three steps: derive
 * x_hat, invert the residual if the mask marks an exception there, keep
 * x_hat otherwise.
 *
 * The recurrence is what makes this vectorisable at all. position p
 * reads last[p & 7] and second_last[p & 7], and those hold the bytes its
 * own lane wrote 8 and 16 positions back -- so the eight lanes are eight
 * *independent* chains, each advancing one step every eight positions.
 * A group of eight positions is therefore exactly one step of all eight
 * chains at once, and the group's predictions need nothing but the
 * previous two groups' results:
 *
 *   LANE_PREVIOUS   x_hat(g) = x(g-1)
 *   LANE_DELTA      x_hat(g) = 2*x(g-1) - x(g-2)
 *   LANE_AVERAGE    x_hat(g) = (x(g-1) + x(g-2) + 1) >> 1
 *
 * with x(-1) and x(-2) the entry lane histories, which is precisely the
 * state the scalar loops start from. That also means the entry state is
 * honoured for a non-resetting tile and the whole tile is covered -- no
 * prologue, no staging buffer, no head, and no assumption about where in
 * the segment the tile sits.
 *
 * The last, partial group is one more vector step with the mask bits past
 * Nt cleared, so exactly the same residual units are consumed in exactly
 * the same ascending position order as the scalar loop consumes them, and
 * its bytes past Nt are dropped by a byte copy rather than an 8-byte
 * store.
 *
 * The STRIDE family is deliberately NOT here: its prediction reads
 * out[p - s] for s in {1,2,4,8}, which for s < 8 is a byte the same group
 * is still producing, so a group's predictions would need the group's own
 * result. That lane-crossing recurrence is the scalar loop's business.
 *
 * Residual units reach this kernel already in the exclusion domain: modes
 * 0/2/3 give r with x = r + (r >= x_hat), mode 1 gives x itself, mode 4
 * gives an int8 delta with x = x_hat + delta. Those are the three branches
 * below.
 * ------------------------------------------------------------------ */
#if ZPSE_TILE_HAVE_AVX2

/* The two unit arrays in zpse_tile_decode -- resid[] and deltas[] -- are
 * exactly this many bytes, and that is the only buffer the kernels below
 * ever read units from. Knowing the length is what lets them take an
 * eight-byte window at `consumed` when that window is still inside the
 * array, and fall back to the per-bit walk when it is not. */
#define ZPSE_TILE_UNIT_BYTES 256u

/* Per-lane exclusive popcount of one group's eight mask bits, i.e. the
 * index, inside the group's eight-byte unit window, of the unit that
 * position consumes. Lane j of the result counts only the mask bits BELOW
 * j, so it is in 0..7 for every mask: that is why the window can be a
 * single aligned slice of eight units and every lane's fetch stays inside
 * it.
 *
 * `bits` is the broadcast mask byte already ANDed with `bitv`, which is
 * also what the caller's blend tests against `bitv`, so the compare here
 * costs nothing extra; the ladder is the usual byte prefix sum, three
 * shift-and-add steps. Deliberately not a 256x8 lookup table: a table
 * would be 2 KiB of constants whose only definition is "row m lists its
 * set bits' exclusive counts", and this form is about ten register ops
 * that the recurrence around it hides anyway.
 *
 * Lanes the mask does not mark get an index too. Their unit byte is
 * fetched and processed but never selected: the blend at the end of the
 * group takes `x_hat` for those lanes, and lanes past rem are cleared out
 * of `m8` by the caller's `valid` mask, so a zero there means "no
 * exception" just as it does in the middle of a tile. */
static ZPSE_TILE_AVX2 __m128i tile_unit_index(__m128i bits, __m128i bitv,
                                              __m128i one)
{
    __m128i x = _mm_and_si128(_mm_cmpeq_epi8(bits, bitv), one);

    x = _mm_slli_si128(x, 1);
    x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
    x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
    return x;
}

/* Reconstruct a whole CONST or LANEHIST tile eight positions at a time,
 * consuming residual units from units[unit_off..] and returning the new
 * unit count. ent_last/ent_second are the lane histories on entry, which
 * are the first two "groups" the recurrence needs. */
static ZPSE_TILE_AVX2 uint16_t tile_recon_avx2(uint8_t *out, uint16_t nt,
                                              int fam, uint8_t predictor_id,
                                              uint8_t constant_C,
                                              const uint32_t words[8],
                                              const uint8_t *units,
                                              uint8_t res_mode,
                                              const uint8_t ent_last[8],
                                              const uint8_t ent_second[8],
                                              uint16_t unit_off)
{
    /* Per-lane bit value of the group's mask byte. The upper eight lanes
     * carry zero, which the mask expansion below turns into "set"; only
     * the low eight lanes are ever stored, so that is harmless. */
    const __m128i bitv = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char)0x80,
                                       0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i one = _mm_set1_epi8((char)1);
    const __m128i cvec = _mm_set1_epi8((char)constant_C);
    __m128i prev1 = _mm_loadl_epi64((const __m128i *)(const void *)ent_last);
    __m128i prev2 =
        _mm_loadl_epi64((const __m128i *)(const void *)ent_second);
    uint16_t consumed = unit_off;
    uint32_t g;
    uint32_t ngroups = ((uint32_t)nt + 7u) >> 3u;
    /* The mask is read once per group and the group's bytes are stored
     * through `out` in between, so `words` has to be treated as aliased by
     * those stores unless it is copied somewhere unreachable from `out`. */
    uint32_t mw[8];

    memcpy(mw, words, sizeof mw);

    for (g = 0; g < ngroups; g++) {
        uint32_t at = g * 8u;
        uint32_t rem = (uint32_t)nt - at;
        uint32_t valid = (rem >= 8u) ? 0xFFu : ((UINT32_C(1) << rem) - 1u);
        uint32_t m8 = (mw[g >> 2u] >> ((g & 3u) * 8u)) & valid;
        uint32_t cnt = 0u;
        __m128i xhat;
        __m128i res;

        /* ---- Runs of exception-free 32-byte blocks ---------------- *
         * A block of four groups whose mask word is zero has no exception
         * in it at all, so every position is its own prediction and no
         * unit is consumed. Two of the three lane predictors then have a
         * closed form over such a run, and sparse tiles are mostly made of
         * these blocks (a 256-byte tile with fifteen exceptions has most
         * of its eight blocks clean):
         *
         *   LANE_PREVIOUS  every lane predicts the same byte, so the whole
         *                  run is the previous group repeated and the lane
         *                  state does not move;
         *   LANE_DELTA     the per-lane second difference is constant, so
         *                  the run is an arithmetic progression at one add
         *                  of that difference per group.
         *
         * Both are the same arithmetic the group-at-a-time chain performs
         * below, only hoisted out of the per-group mask extract, blend and
         * 8-byte store: one 32-byte store replaces four of them. The block
         * must be 32-byte aligned in the group index so the four mask
         * bytes are one word, and it must be whole, which is why a partial
         * final block stays on the general path. LANE_AVERAGE has no
         * closed form and keeps it too. */
        if (fam == TILE_FAM_LANEHIST && (g & 3u) == 0u &&
            (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS ||
             predictor_id == (uint8_t)ZPSE_P_LANE_DELTA)) {
            uint32_t g0 = g;

            while (g < ngroups && ((uint32_t)nt - g * 8u) >= 32u &&
                   mw[g >> 2u] == 0u) {
                uint8_t *dst = out + (size_t)g * 8u;

                if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
                    __m128i v = _mm_unpacklo_epi64(prev1, prev1);

                    _mm_storeu_si128((__m128i *)(void *)dst, v);
                    _mm_storeu_si128((__m128i *)(void *)(dst + 16), v);
                    prev2 = prev1;
                } else {
                    __m128i d = _mm_sub_epi8(prev1, prev2);
                    __m128i x0 = _mm_add_epi8(prev1, d);
                    __m128i x1 = _mm_add_epi8(x0, d);
                    __m128i x2 = _mm_add_epi8(x1, d);
                    __m128i x3 = _mm_add_epi8(x2, d);

                    _mm_storeu_si128((__m128i *)(void *)dst,
                                     _mm_unpacklo_epi64(x0, x1));
                    _mm_storeu_si128((__m128i *)(void *)(dst + 16),
                                     _mm_unpacklo_epi64(x2, x3));
                    prev2 = x2;
                    prev1 = x3;
                }
                g += 4u;
            }
            if (g != g0) {
                g -= 1u; /* the loop's own g++ */
                continue;
            }
        }

        if (fam == TILE_FAM_CONST) {
            xhat = cvec;
        } else if (predictor_id == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
            xhat = prev1;
        } else if (predictor_id == (uint8_t)ZPSE_P_LANE_DELTA) {
            xhat = _mm_sub_epi8(_mm_add_epi8(prev1, prev1), prev2);
        } else { /* LANE_AVERAGE rounds up, as scalar. */
            xhat = _mm_avg_epu8(prev1, prev2);
        }

        /* A group with no exception in it needs no units and no blend:
         * the blend below returns x_hat for every lane, which is what res
         * is set to here. Sparse tiles (a handful of exceptions in 256
         * positions) take this branch for almost every group. */
        if (m8 == 0u) {
            res = xhat;
        } else if (m8 == 0xFFu && (uint32_t)consumed + 8u <= ZPSE_TILE_UNIT_BYTES) {
            /* The other extreme, and the common group of a dense tile (235
             * exceptions in 256 positions): every lane is an exception, so
             * the indices the ladder would produce are 0..7 -- the units
             * are simply the eight contiguous bytes at `consumed` -- and
             * the byte mask the blend would build is all ones, so its
             * result is `x` for every lane. Skipping the expansion, the
             * index and the blend removes the whole mask side of the group,
             * which is what this kernel is short of: with a dense mask
             * every position costs the same handful of uops whether or not
             * anyone reads them. `m8 == 0xFF` also implies rem >= 8, since
             * `valid` is not all ones for a partial group. */
            __m128i r = _mm_loadl_epi64(
                (const __m128i *)(const void *)(units + consumed));
            __m128i x;

            cnt = 8u;
            if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                x = r;
            } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                x = _mm_add_epi8(xhat, r);
            } else {
                __m128i ge = _mm_cmpeq_epi8(_mm_max_epu8(r, xhat), r);

                x = _mm_add_epi8(r, _mm_and_si128(ge, one));
            }
            res = x;
        } else {
            const __m128i bits =
                _mm_and_si128(_mm_set1_epi8((char)m8), bitv);
            __m128i mb;
            __m128i r;
            __m128i x;

            /* The group's units, placed at the lane of the position they
             * belong to: lane j takes unit popcount(m8 below j), which is
             * what tile_unit_index() produces. Compacting them into the
             * low lanes instead would be wrong -- lane j must hold the
             * unit of position at + j, not the j-th unit. A fully set mask
             * byte is the identity case: the indices come out 0..7, so the
             * eight units are taken in order.
             *
             * The window is the eight bytes at `consumed`, which is the
             * most any group can consume, so every index is inside it. It
             * is only read when those eight bytes are still inside the
             * unit array; the last groups of a tile with more than 248
             * exceptions would run past its end, and those keep the per-bit
             * walk, which never reads beyond the units the group consumes
             * (consumed + popcount(m8) <= k <= ZPSE_TILE_UNIT_BYTES). */
            cnt = (uint32_t)tile_popcount32(m8);
            if ((uint32_t)consumed + 8u <= ZPSE_TILE_UNIT_BYTES) {
                r = _mm_shuffle_epi8(
                    _mm_loadl_epi64(
                        (const __m128i *)(const void *)(units + consumed)),
                    tile_unit_index(bits, bitv, one));
            } else {
                uint8_t tmp[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
                uint32_t got = 0u;
                uint32_t wg = m8;
                uint32_t bit = 0u;

                while (wg != 0u) {
                    if ((wg & 1u) != 0u) {
                        tmp[bit] = units[consumed + got];
                        got++;
                    }
                    wg >>= 1u;
                    bit++;
                }
                r = _mm_loadl_epi64((const __m128i *)(const void *)tmp);
            }

            if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                x = r;
            } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                x = _mm_add_epi8(xhat, r);
            } else {
                /* Exclusion inverse: r < x_hat ? r : r + 1, unsigned. */
                __m128i ge = _mm_cmpeq_epi8(_mm_max_epu8(r, xhat), r);

                x = _mm_add_epi8(r, _mm_and_si128(ge, one));
            }

            /* Expand the group's eight mask bits into a byte mask: bit j
             * of the byte is exactly the bit value of lane j, so
             * broadcasting the byte and comparing the AND against those
             * per-lane values leaves 0xFF where the bit is set. */
            mb = _mm_cmpeq_epi8(bits, bitv);
            res = _mm_or_si128(_mm_and_si128(mb, x),
                               _mm_andnot_si128(mb, xhat));
        }

        consumed = (uint16_t)(consumed + (uint16_t)cnt);
        if (rem >= 8u) {
            _mm_storel_epi64((__m128i *)(void *)(out + (size_t)at), res);
        } else {
            uint8_t ob[16];

            _mm_storeu_si128((__m128i *)(void *)ob, res);
            memcpy(out + (size_t)at, ob, (size_t)rem);
        }
        prev2 = prev1;
        prev1 = res;
    }
    return consumed;
}

/* The 32-lane counterpart of tile_unit_index(): the exclusive popcount of
the chunk's mask bits at each byte lane, i.e. the index, inside the
chunk's unit window, of the unit that lane consumes. `v1` is the chunk's
mask bits as 0/1 bytes (the caller already has the byte mask `mb`, so
this is just `mb & 1`), and the same three-step shift-and-add ladder
produces the exclusive prefix sum. _mm256_slli_si256 shifts each
128-bit lane independently, so each half of the chunk gets its own index
in 0..15 - exactly the range vpshufb needs while the two halves are
separate 16-byte windows. */
static ZPSE_TILE_AVX2 __m256i tile_unit_index32(__m256i v1)
{
    /* Inclusive prefix first, then one shift to make it exclusive. The
     * ladder has to span the whole half (16 lanes, so shifts 1/2/4/8); an
     * 8-wide ladder - which is what the 8-lane kernel needs - silently
     * undercounts from lane 9 on, and the index then points at an earlier
     * unit. The counts stay in a byte (at most 16). */
    __m256i x = v1;

    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 1));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 2));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 4));
    x = _mm256_add_epi8(x, _mm256_slli_si256(x, 8));
    return _mm256_slli_si256(x, 1);
}

/* Reconstruct a whole CONST tile 32 positions at a time.
 *
 * A CONST tile has no recurrence at all: every prediction is the same
 * byte (0 for CONST_ZERO, constant_C for TILE_CONSTANT), so unlike the
 * lane-history kernel above there is nothing to chain and a full AVX2
 * register can be produced per step. The 256 positions are eight steps of
 * the loop below with no dependency between them.
 *
 * Same contract as tile_recon_avx2: one residual unit per set mask bit,
 * consumed in ascending position order, each placed in the lane of ITS
 * position (not compacted into the low lanes - that mistake makes every
 * non-adjacent exception decode to the wrong byte), and the new unit
 * count returned. The horizontal boundary is handled by the caller: only
 * whole 32-byte chunks go through the vector loop, the remaining
 * nt % 32 positions are finished positionally below so the units keep
 * their order. */
static ZPSE_TILE_AVX2 uint16_t tile_recon_const_avx2(uint8_t *out, uint16_t nt,
                                                     uint8_t constant_C,
                                                     const uint32_t words[8],
                                                     const uint8_t *units,
                                                     uint8_t res_mode)
{
    /* Per-byte lane bit value of the chunk's mask word: byte lane i tests
     * bit (i & 7), so the pattern repeats four times across the register.
     * Lane 7's value must be (char)0x80, the sign-extended byte of the
     * cast, because that is what the AND keeps when bit 7 is set. */
    const __m256i bitv = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80,
        1, 2, 4, 8, 16, 32, 64, (char)0x80);
    /* Byte lane i must be tested against bit i of the 32-bit word, and
     * that bit lives in the word's byte i >> 3. Broadcasting the word as
     * 32-bit lanes gives byte i the word's byte (i & 3), which is the
     * wrong byte: four consecutive lanes have to hold the SAME byte of
     * the word, so duplicate each of its four bytes four times first.
     * (vpshufb is per 128-bit lane, and both lanes of the broadcast hold
     * the same four bytes, so one shuffle table serves both.)
     *
     * Eight consecutive lanes share a byte of the word, not four: lane i
     * holds byte i >> 3, and each 32-bit lane of the broadcast supplies
     * the word's bytes at source indices 0..3 of its own 128-bit half. */
    const __m256i dup8 = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i one = _mm256_set1_epi8((char)1);
    const __m256i xhat = _mm256_set1_epi8((char)constant_C);
    uint16_t consumed = 0u;
    uint32_t at = 0u;
    uint32_t full = ((uint32_t)nt / 32u) * 32u;
    /* Mask hoist, as in the 8-lane kernel: the chunk's bytes are stored
     * through `out` between mask reads, so a copy is what keeps them from
     * being reloaded. */
    uint32_t mw[8];

    memcpy(mw, words, sizeof mw);

    while (at < full) {
        uint32_t w = mw[at >> 5u];
        __m256i mb;
        __m256i r;
        __m256i x;
        __m256i res;

        /* No exceptions in this chunk: the whole result is the
         * prediction, with no units to consume and nothing to blend. */
        if (w == 0u) {
            _mm256_storeu_si256((__m256i *)(void *)(out + (size_t)at), xhat);
            at += 32u;
            continue;
        }

        /* The dense extreme: every position in the chunk is an exception,
         * so the byte mask is all ones (the blend would return `x` for all
         * 32 lanes) and the 32 contiguous bytes at `consumed` are already
         * the units in position order. A dense tile takes this branch for
         * seven of its eight chunks, and it removes the expansion, the
         * 16-lane index ladder, the two window loads and the blend from
         * all of them. */
        if (w == UINT32_C(0xFFFFFFFF) &&
            (uint32_t)consumed + 32u <= ZPSE_TILE_UNIT_BYTES) {
            __m256i r = _mm256_loadu_si256(
                (const __m256i *)(const void *)(units + consumed));
            __m256i x;

            consumed = (uint16_t)(consumed + 32u);
            if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                x = r;
            } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                x = _mm256_add_epi8(xhat, r);
            } else {
                __m256i ge = _mm256_cmpeq_epi8(_mm256_max_epu8(r, xhat), r);

                x = _mm256_add_epi8(r, _mm256_and_si256(ge, one));
            }
            _mm256_storeu_si256((__m256i *)(void *)(out + (size_t)at), x);
            at += 32u;
            continue;
        }

        /* Expand the chunk's 32 mask bits to a byte mask: byte lane i
         * holds the word's byte i >> 3, so ANDing with bitv (1 << (i & 7))
         * keeps exactly bitv's bit when bit i of the word is set. This is
         * also what the unit placement below indexes with, so it is built
         * before the units rather than after them. */
        mb = _mm256_cmpeq_epi8(
            _mm256_and_si256(
                _mm256_shuffle_epi8(_mm256_set1_epi32((int)w), dup8),
                bitv),
            bitv);

        /* The chunk's units, placed at the lane of the position they
         * belong to: lane i takes unit popcount(w below i), which is what
         * the ladder gives. Compacting them into the low lanes instead
         * would be wrong - lane i must hold the unit of position at + i,
         * not the i-th unit.
         *
         * The units of a chunk are contiguous, and the ladder is per
         * 128-bit lane, so the low half's units are the 16 bytes at
         * `consumed` and the high half's are the 16 bytes at `consumed +
         * popcount(w & 0xFFFF)`; two windows, one shuffle. (The old form
         * built the same vector in a 32-byte stack staging buffer with a
         * per-bit walk and loaded it back, a store-to-load round trip per
         * chunk, which is what sparse tiles spent their reconstruction on.)
         *
         * The two windows are only read when the whole chunk still fits in
         * the unit array. A chunk that does not (a tile with more than 224
         * exceptions, which is dense even by this kernel's standards) keeps
         * the per-bit walk, which never reads past the units it consumes
         * (consumed + popcount(w) <= k <= ZPSE_TILE_UNIT_BYTES). */
        if ((uint32_t)consumed + 32u <= ZPSE_TILE_UNIT_BYTES) {
            uint32_t lo = (uint32_t)tile_popcount32(w & UINT32_C(0xFFFF));
            __m128i win_lo = _mm_loadu_si128(
                (const __m128i *)(const void *)(units + consumed));
            __m128i win_hi = _mm_loadu_si128(
                (const __m128i *)(const void *)(units + consumed + lo));
            __m256i win = _mm256_castsi128_si256(win_lo);

            win = _mm256_inserti128_si256(win, win_hi, 1);
            r = _mm256_shuffle_epi8(
                win, tile_unit_index32(_mm256_and_si256(mb, one)));
            consumed = (uint16_t)(consumed + (uint16_t)tile_popcount32(w));
        } else {
            uint8_t tmp[32];
            uint32_t wg = w;
            uint32_t bit = 0u;
            uint16_t cnt = 0u;

            /* Lanes with no unit are never selected by mb below, but they
             * must still be defined: zero them rather than leave the
             * vector load reading an uninitialised buffer. */
            memset(tmp, 0, sizeof tmp);
            while (wg != 0u) {
                if ((wg & 1u) != 0u) {
                    tmp[bit] = units[consumed + cnt];
                    cnt++;
                }
                wg >>= 1;
                bit++;
            }
            consumed = (uint16_t)(consumed + cnt);
            r = _mm256_loadu_si256((const __m256i *)(const void *)tmp);
        }

        if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
            x = r;
        } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
            x = _mm256_add_epi8(xhat, r);
        } else {
            /* Exclusion inverse: r < x_hat ? r : r + 1, unsigned. */
            __m256i ge = _mm256_cmpeq_epi8(_mm256_max_epu8(r, xhat), r);

            x = _mm256_add_epi8(r, _mm256_and_si256(ge, one));
        }

        res = _mm256_or_si256(_mm256_and_si256(mb, x),
                              _mm256_andnot_si256(mb, xhat));
        _mm256_storeu_si256((__m256i *)(void *)(out + (size_t)at), res);
        at += 32u;
    }

    /* Tail: the same order of units, position by position, mirroring the
     * scalar CONST branch rather than duplicating its masking rules. */
    while (at < (uint32_t)nt) {
        uint8_t c = constant_C;
        uint8_t x;

        if (((mw[at >> 5u] >> (at & 31u)) & 1u) == 0u) {
            x = c;
        } else if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
            x = units[consumed++];
        } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
            x = (uint8_t)(((unsigned)c + (int)(int8_t)units[consumed++]) &
                          0xFFu);
        } else {
            x = zpse_exclusion_inverse(units[consumed++], c);
        }
        out[at] = x;
        at++;
    }
    return consumed;
}
#endif /* ZPSE_TILE_HAVE_AVX2 */

/* RAW_BITMAP mask for a full 256-position tile.
 *
 * This is mask.c's ZPSE_M_RAW_BITMAP branch with the three cases that
 * cannot happen here removed, because it is the mask shape of every dense
 * tile (a palette or noise plane: ~118 exceptions in 256 positions, which
 * is why the encoder picks the 32-byte bitmap over RUNS and the
 * enumerative code). Payload length 32 is exactly eight whole
 * little-endian words, so there is nothing to zero first and no partial
 * final word; nt == 256 means no tail bits can be set, so the tail check
 * is gone; and the k check is the same popcount. The bytes and the
 * statuses returned are identical to zpse_mask_decode's for this mode,
 * which is what the scalar/SIMD parity suite and the mask sweep cover. */
static int tile_mask_raw_bitmap_256(const uint8_t *payload, uint16_t k,
                                    uint32_t words[8])
{
    int i;

    int counted = 0;

    if (k < 1u || k > 255u)
        return PSE_BAD_MASK;
    /* Load and count in one pass. zpse_mask_count_bits trims its input
     * with a 32-byte copy and a zeroing loop that a full tile cannot use
     * (there is no tail), so it is not the right tool here: eight whole
     * words is four 64-bit loads and four SWAR popcounts. */
    for (i = 0; i < 4; i++) {
        uint64_t v = zpse_load_u64le(payload + 8u * (unsigned)i);

        counted += tile_popcount64(v);
        words[2 * i] = (uint32_t)v;
        words[2 * i + 1] = (uint32_t)(v >> 32);
    }
    if (counted != (int)k)
        return PSE_BAD_MASK;
    return PSE_OK;
}

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

    if (mask_mode == (uint8_t)ZPSE_M_RAW_BITMAP && nt == 256u &&
        th.mask_payload_size == 32u) {
        rc = tile_mask_raw_bitmap_256(mask_ptr, k, words);
    } else {
        zpse_mask_clear(words);
        rc = zpse_mask_decode(mask_mode, mask_ptr, th.mask_payload_size,
                              nt, k, words, comb_scratch);
    }
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

    /* ---- CONST with few exceptions ------------------------------ *
     * A constant prediction does not depend on any byte the tile writes,
     * so such a tile is that constant everywhere except at its
     * exceptions: fill it and then write them, instead of walking 256
     * mask bits. The exceptions are visited in mask order (ascending),
     * which is the order the residual units were packed in, and each one's
     * byte is resolved from the constant exactly as the byte loops below
     * resolve it. Below the crossover this is a memset plus k stores; the
     * 32-wide kernel it replaces costs a fixed ~150 cycles, so the limit
     * is where a few cycles per exception overtakes it. */
    if (fam == TILE_FAM_CONST && (uint32_t)k <= 24u) {
        uint8_t c = (th.predictor_id == (uint8_t)ZPSE_P_TILE_CONSTANT)
                        ? constant_C : 0u;
        uint8_t ent_last[8];
        uint8_t ent_second[8];
        uint16_t wi;

        tile_capture_hist(lanes, ent_last, ent_second);
        memset(out, c, nt);
        j = 0;
        for (wi = 0; wi < 8u && j < k; wi++) {
            uint32_t w = words[wi];

            while (w != 0u) {
                uint16_t pos =
                    (uint16_t)((uint32_t)wi * 32u + (uint32_t)tile_ctz32(w));
                uint8_t e;

                if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
                    e = resid[j];
                } else if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA) {
                    e = (uint8_t)(((unsigned)c + (int)deltas[j]) & 0xFFu);
                } else {
                    e = zpse_exclusion_inverse(resid[j], c);
                }
                out[pos] = e;
                j++;
                w &= w - 1u;
            }
        }
        tile_history_fixup(lanes, out, nt, ent_last, ent_second);
        if (j != k)
            return PSE_BAD_MASK;
        return (int)nt;
    }

#if ZPSE_TILE_HAVE_AVX2
    /* Vectorised reconstruction for the two families whose prediction is
     * a pure function of the tile's own lane histories (see
     * tile_recon_avx2). Only the byte-writing loops are replaced:
     * validation, the mask and residual decode above and every status they
     * produced are untouched, and the j != k test below is this path's
     * version of the scalar loops' own residual-consumption check, taken
     * only once the output bytes for the whole tile have been produced --
     * exactly as the scalar loops leave it. ORDER1/ORDER2 (stateful
     * tables), STRIDE (a lane-crossing recurrence), Nt < 32 and every
     * machine without AVX2 fall through to the scalar loops. */
    if ((uint32_t)nt >= 32u &&
        (fam == TILE_FAM_CONST || fam == TILE_FAM_LANEHIST) &&
        (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
        uint8_t ent_last[8];
        uint8_t ent_second[8];

        const uint8_t *units =
            (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA)
                ? (const uint8_t *)(const void *)deltas
                : resid;

        /* Captured before anything writes, for the O(8) history epilogue
         * that both the all-hit fast path and the scalar loops use as
         * their end state. */
        tile_capture_hist(lanes, ent_last, ent_second);
        /* CONST has no recurrence, so it takes the 32-wide kernel; the
         * lane-history family needs the 8-lane one, whose step is the
         * recurrence's own width. */
        j = (fam == TILE_FAM_CONST)
                ? tile_recon_const_avx2(out, nt, constant_C, words, units,
                                        res_mode)
                : tile_recon_avx2(out, nt, fam, th.predictor_id, constant_C,
                                  words, units, res_mode, ent_last,
                                  ent_second, 0u);
        tile_history_fixup(lanes, out, nt, ent_last, ent_second);
        if (j != k)
            return PSE_BAD_MASK;
        return (int)nt;
    }
#endif

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
            int done = 0;
#if ZPSE_TILE_HAVE_AVX2
            if (s == 8u && (uint32_t)nt >= 32u &&
                (zpse_cpu_caps() & (uint32_t)ZPSE_CPU_AVX2) != 0u) {
                /* Stride 8 is the 8-lane LANE_PREVIOUS recurrence with the
                 * tile's *real* preceding eight bytes as the entry lane
                 * state (init at the very start of the segment). */
                uint8_t el[8];
                unsigned l;
                const uint8_t *units =
                    (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA)
                        ? (const uint8_t *)(const void *)deltas : resid;
                for (l = 0; l < 8u; l++)
                    el[l] = (seg_pos + l < 8u) ? initial_literal
                                               : seg_base[seg_pos + l - 8u];
                j = tile_recon_avx2(out, nt, TILE_FAM_LANEHIST,
                                    (uint8_t)ZPSE_P_LANE_PREVIOUS, 0u, words,
                                    units, res_mode, el, el, 0u);
                done = 1;
            }
#endif
            if (!done) {
                if (s == 1u) {
                    /* One chain over the exceptions, then a vector fill:
                     * see tile_stride1_recon. */
                    j = tile_stride1_recon(
                        out, nt, seg_base, seg_pos, initial_literal, words,
                        (uint32_t)k,
                        (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA)
                            ? (const uint8_t *)(const void *)deltas
                            : resid,
                        res_mode);
                } else {
                    memset(resid + k, 0, (size_t)(256u - k) * sizeof resid[0]);
                    if (res_mode == (uint8_t)ZPSE_R_SIGNED_DELTA)
                        memset(deltas + k, 0,
                               (size_t)(256u - k) * sizeof deltas[0]);
                    j = tile_stride_recon_scalar(out, nt, s, seg_base, seg_pos,
                                                 initial_literal, words, resid,
                                                 deltas, res_mode);
                }
            }
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
