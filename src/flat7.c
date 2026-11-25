/* FLAT7 frame codec core: a whole-input ASCII fast path.
 *
 * Wire layout (all little-endian, 40-byte header then payload):
 *   [0..4)   u32 magic = ZPSE_FLAT7_MAGIC  ("F7P1" LE = 0x31503746)
 *   [4]      u8  version_major = 1
 *   [5]      u8  version_minor = 0
 *   [6..8)   u16 flags (must be 0 in v1)
 *   [8..16)  u64 total_out_size = n
 *   [16..20) u32 nregions = 1
 *   [20..36) region[0]: u32 offset=0, u32 packed_len, u32 out_len=n,
 *                       u8 type (ASCII7 / RAW), u8[3] reserved=0
 *   [36..40) u32 header_crc32c = zpse_crc32c(in[0..36))
 *   [40..]   payload: 7-bit packed, packed_len = n - n/8
 *
 * Encoding always uses the fused tile.c stream packer (via
 * zpse_flat7_encode -> zpse_flat7_encode_stream); decoding keeps a scalar
 * 7->8 fallback so this file remains self-contained when the integration
 * build does not define ZPSE_HAVE_ASCII_STREAM. Both unpackers are bit-
 * identical. */

#include "zpse/flat7.h"

#include "ntout.h"

#include <string.h>

#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/status.h"

#ifdef ZPSE_HAVE_ASCII_STREAM
#include "zpse/tile.h" /* zpse_ascii_pack_stream / unpacK_stream / ascii_scan */
#endif

/* The fused check+pack kernel lives in tile.c. Declared here too so this
 * translation unit stays independently compilable when tile.h is not on the
 * include chain (the unity build always pulls tile.h in first). */
uint32_t zpse_ascii_pack_checked(const uint8_t *in, uint32_t n, uint8_t *out,
                                 int *all_ascii);

/* Chunk size for the fused pass: a multiple of 8, so every chunk starts
 * 8-aligned and packs to exactly chunk - chunk/8 payload bytes. */
#define FLAT7_STREAM_CHUNK 65536u

/* ---- Constant fallbacks ------------------------------------------- *
 * The integration agent adds these to constants.h. Keeping local fallbacks
 * lets this translation unit compile on its own until that lands. */
#ifndef ZPSE_FLAT7_MAGIC
#define ZPSE_FLAT7_MAGIC        UINT32_C(0x31503746)
#endif
#ifndef ZPSE_FLAT7_VER_MAJOR
#define ZPSE_FLAT7_VER_MAJOR    ((uint8_t)1)
#endif
#ifndef ZPSE_FLAT7_VER_MINOR
#define ZPSE_FLAT7_VER_MINOR    ((uint8_t)0)
#endif
#ifndef ZPSE_FLAT7_REGION_ASCII7
#define ZPSE_FLAT7_REGION_ASCII7 ((uint8_t)0)
#endif
#ifndef ZPSE_FLAT7_REGION_RAW
#define ZPSE_FLAT7_REGION_RAW   ((uint8_t)1)
#endif
#ifndef ZPSE_FLAT7_MIN_SIZE
#define ZPSE_FLAT7_MIN_SIZE     4096u
#endif
#ifndef ZPSE_FLAT7_BASE_SIZE
#define ZPSE_FLAT7_BASE_SIZE    20u
#endif
#ifndef ZPSE_FLAT7_REGION_SIZE
#define ZPSE_FLAT7_REGION_SIZE  16u
#endif
#ifndef ZPSE_FLAT7_CRC_SIZE
#define ZPSE_FLAT7_CRC_SIZE     4u
#endif
#ifndef ZPSE_ENC_CHECKSUM
#define ZPSE_ENC_CHECKSUM       (UINT32_C(1) << 0)
#endif

/* Derived offsets: base(20) + region(16) = region end / CRC offset(36);
 * + crc(4) = header size(40). */
#define FLAT7_REGION_OFF        ((size_t)ZPSE_FLAT7_BASE_SIZE)              /* 20 */
#define FLAT7_CRC_OFF           ((size_t)(ZPSE_FLAT7_BASE_SIZE + ZPSE_FLAT7_REGION_SIZE)) /* 36 */
#define FLAT7_HEADER_SIZE       ((size_t)(ZPSE_FLAT7_BASE_SIZE + ZPSE_FLAT7_REGION_SIZE + ZPSE_FLAT7_CRC_SIZE)) /* 40 */

/* Region field offsets relative to the region base (20). */
#define FLAT7_R_OFFSET          0u  /* u32 offset relative to payload   */
#define FLAT7_R_PACKED          4u  /* u32 packed_len                   */
#define FLAT7_R_OUTLEN          8u  /* u32 out_len                      */
#define FLAT7_R_TYPE           12u  /* u8  region type                  */
#define FLAT7_R_RESERVED       13u  /* u8[3] reserved                   */

/* ---- Scalar 7->8 spread (exact tile.c:343-368 layout) ---------------- *
 * Encoding is always the fused tile.c packer now (zpse_flat7_encode
 * delegates to zpse_flat7_encode_stream); the 8->7 compact side is gone.
 * The scalar unpacker is the fallback for translation units built without
 * the ASCII stream kernels; the unity build uses those and has no use for it,
 * so keep the whole block out of that build (CI compiles with -Werror). */
#ifndef ZPSE_HAVE_ASCII_STREAM

static uint64_t flat7_spread8(uint64_t v)
{
    return (v & UINT64_C(0x000000000000007F)) |
           ((v & UINT64_C(0x0000000000003F80)) << 1) |
           ((v & UINT64_C(0x00000000001FC000)) << 2) |
           ((v & UINT64_C(0x000000000FE00000)) << 3) |
           ((v & UINT64_C(0x00000007F0000000)) << 4) |
           ((v & UINT64_C(0x000003F800000000)) << 5) |
           ((v & UINT64_C(0x0001FC0000000000)) << 6) |
           ((v & UINT64_C(0x00FE000000000000)) << 7);
}

/* r bytes (r <= 8) as a 64-bit little-endian word, zero above byte r-1. */
static uint64_t flat7_gather(const uint8_t *p, unsigned r)
{
    uint64_t w = 0u;
    unsigned i;
    for (i = 0; i < r; i++) {
        w |= (uint64_t)p[i] << (8u * i);
    }
    return w;
}

/* Bytes a group of r (1..8) ASCII bytes packs to: ceil(7r/8). */
static unsigned flat7_group_out(unsigned r)
{
    return (7u * r + 7u) >> 3;
}

static void flat7_unpack_scalar(const uint8_t *in, uint32_t in_len,
                                uint8_t *out, uint32_t n)
{
    uint32_t groups = n >> 3;
    uint32_t rest = n - groups * 8u;
    uint32_t g;
    uint32_t p = 0u;

    for (g = 0; g < groups; g++) {
        uint64_t v;
        uint64_t w;

        if (p + 8u <= in_len) {
            memcpy(&v, in + p, sizeof v);
            v &= UINT64_C(0x00FFFFFFFFFFFFFF);
        } else {
            v = flat7_gather(in + p, 7u);
        }
        w = flat7_spread8(v);
        memcpy(out + (size_t)g * 8u, &w, sizeof w);
        p += 7u;
    }
    if (rest != 0u) {
        uint64_t v = flat7_gather(in + p, flat7_group_out(rest));
        uint64_t w = flat7_spread8(v);
        uint32_t k;

        for (k = 0; k < rest; k++) {
            out[groups * 8u + k] = (uint8_t)(w >> (8u * k));
        }
    }
}

#endif /* !ZPSE_HAVE_ASCII_STREAM */

/* ---- Unpack dispatch ---------------------------------------------- *
 * Encode is the single fused stream path (see zpse_flat7_encode); only
 * decode keeps a scalar fallback for independently built translation units. */

#ifdef ZPSE_HAVE_ASCII_STREAM
static void flat7_unpack(const uint8_t *in, uint32_t packed_len,
                         uint8_t *out, uint32_t n)
{
    zpse_ascii_unpack_stream(in, packed_len, out, n);
}
#else
#define flat7_unpack flat7_unpack_scalar
#endif

/* ---- Redundancy gate: decline compressible all-ASCII input ----------- *
 * A whole-input ASCII file is not automatically a good FLAT7 file: the block
 * codec's LZ finder (and its entropy/tile path on low-alphabet data) packs
 * repetitive text far smaller than the fixed 7-bit pack. Measured here at
 * level 6 / 64 KiB blocks, webster is 19,772,044 bytes by blocks against
 * 36,276,406 by FLAT7 (+83%), nci 6,183,546 against 29,359,305 (+375%).
 * A false positive costs up to 23 MB of ratio; a false decline only costs
 * encode speed, so fire only on positive evidence of incompressibility.
 *
 * Two integer-only predicates, both pure functions of (in, n):
 *
 *   1. D4_local: distinct exact 4-byte grams over the head min(4096,n) bytes
 *      (8192-slot open-addressed table, 4093 grams -> load factor 0.5).
 *      Measured on the 4 KiB head: webster 0.594, nci 0.120, random printable
 *      ASCII 1.0000 (16 KiB averages are 0.405 / 0.075 / 0.9999); worst
 *      fire-side case is random lowercase at 0.9822. Require > 0.80: 0.21
 *      above the worst decline, 0.18 below the worst fire case.
 *
 *   2. boundary_dup: fraction of exact 4-byte grams at offsets 0, 256, 512,
 *      ... over the WHOLE input whose key was in a 1024-slot direct-mapped
 *      cache at an earlier offset. This catches globally repeated blocks a
 *      head sample cannot see (8 KiB random x 512: 0.998). Measured here
 *      webster 0.350, nci 0.852, random printable ASCII 0.0000 (an exact set
 *      gives 0.818 / 0.951 / 0.000; the cache is a lower bound). Require
 *      < 0.02, a ~40x margin over the worst random case.
 *
 * Both predicates are ANDed -- either alone has a hole -- and cost was
 * measured at 0.46% of block encode. Exact keys matter: a 1-bit-per-bucket
 * bitmap would manufacture collisions and decline random ASCII. All scratch
 * is stack and deterministic; no arena, no worker state, no floating point.
 * UINT32_MAX is an impossible key because the caller has already checked
 * every byte is ASCII (< 128), so it doubles as the empty-slot sentinel. */
#define FLAT7_D4_SAMPLE  4096u
#define FLAT7_D4_SLOTS   8192u
#define FLAT7_D4_SHIFT   19u              /* 32 - 13-bit index */
#define FLAT7_BD_STRIDE  256u
#define FLAT7_BD_SLOTS   1024u
#define FLAT7_BD_SHIFT   22u              /* 32 - 10-bit index */
#define FLAT7_EMPTY      UINT32_MAX

/* ---- Incremental form of the two gate predicates ------------------ *
 * The state holds both tables; feed helpers insert 4-byte keys and
 * flat7_gate_result reads the verdict. zpse_flat7_should walks the whole
 * buffer through these same helpers, so its decisions are byte-identical to
 * the fused stream path that feeds them a chunk at a time. The hash, table
 * sizes and thresholds are the ones measured and documented above; nothing
 * here changes what a decision is. */
typedef struct Flat7GateState {
    uint32_t d4[FLAT7_D4_SLOTS];
    uint32_t bd[FLAT7_BD_SLOTS];
    size_t distinct;
    size_t d4_grams;
    size_t bd_total;
    size_t bd_dup;
} Flat7GateState;

static void flat7_gate_init(Flat7GateState *g)
{
    memset(g->d4, 0xFF, sizeof g->d4);
    memset(g->bd, 0xFF, sizeof g->bd);
    g->distinct = 0u;
    g->d4_grams = 0u;
    g->bd_total = 0u;
    g->bd_dup = 0u;
}

/* Feed one exact 4-byte key from the D4 head region. */
static void flat7_gate_d4_feed(Flat7GateState *g, uint32_t k)
{
    uint32_t h = (k * UINT32_C(2654435761)) >> FLAT7_D4_SHIFT;

    while (g->d4[h] != FLAT7_EMPTY && g->d4[h] != k) {
        h = (h + 1u) & (FLAT7_D4_SLOTS - 1u);
    }
    if (g->d4[h] == FLAT7_EMPTY) {
        g->d4[h] = k;
        g->distinct++;
    }
    g->d4_grams++;
}

/* Feed the 4-byte key at a 256-strided boundary offset into the BD cache. */
static void flat7_gate_bd_feed(Flat7GateState *g, uint32_t k)
{
    uint32_t h = (k * UINT32_C(2654435761)) >> FLAT7_BD_SHIFT;

    if (g->bd[h] == k) {
        g->bd_dup++;
    } else {
        g->bd[h] = k;
    }
    g->bd_total++;
}

/* The D4 half of the gate on its own. The full verdict ANDs this with the
 * boundary_dup predicate, so once the D4 head has been fed a D4 decline is
 * final: no later BD gram (nor the ASCII result) can turn it into a fire.
 * Both early-out callers use this helper so the 0.80 threshold lives in one
 * place and cannot drift from the verdict it short-circuits. */
static int flat7_gate_d4_declines(const Flat7GateState *g)
{
    return g->distinct * 100u <= g->d4_grams * 80u;
}

static int flat7_gate_result(const Flat7GateState *g)
{
    if (flat7_gate_d4_declines(g)) {
        return 0; /* local D4 <= 0.80: redundant at the head */
    }
    if (g->bd_total != 0u && g->bd_dup * 100u >= g->bd_total * 2u) {
        return 0; /* boundary_dup >= 0.02: globally repeated blocks */
    }
    return 1;
}

/* 1 when FLAT7 is competitive (fire), 0 when the block codec should handle
 * this input (decline). Called only after the basic size/cap checks and the
 * whole-buffer ASCII scan. Walks the buffer through the incremental helpers
 * so it stays byte-identical to the fused single-pass gate. */
static int flat7_ascii_gate(const uint8_t *in, size_t n)
{
    Flat7GateState g;
    size_t sample = (n < (size_t)FLAT7_D4_SAMPLE) ? n : (size_t)FLAT7_D4_SAMPLE;
    size_t i;

    flat7_gate_init(&g);
    for (i = 0u; i + 4u <= sample; i++) {
        uint32_t k;

        memcpy(&k, in + i, sizeof k);
        flat7_gate_d4_feed(&g, k);
    }
    /* D4 is fed and final. The gate is an AND -- boundary_dup can only turn a
     * fire into a decline, never the reverse -- so a D4 decline decides the
     * verdict. Skip the whole-input BD scan (~n/256 dependent probes), which
     * dominates gate cost on large D4-declined input; the verdict is identical. */
    if (flat7_gate_d4_declines(&g)) {
        return 0;
    }
    for (i = 0u; i + 4u <= n; i += FLAT7_BD_STRIDE) {
        uint32_t k;

        memcpy(&k, in + i, sizeof k);
        flat7_gate_bd_feed(&g, k);
    }
    return flat7_gate_result(&g);
}

/* ---- Whole-buffer ASCII predicate --------------------------------- */

static int flat7_is_all_ascii(const uint8_t *in, size_t n)
{
    size_t off = 0u;

    while (n != 0u) {
        uint32_t chunk = (n > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)n;

#ifdef ZPSE_HAVE_ASCII_STREAM
        if (zpse_ascii_scan(in + off, chunk) == 0) {
            return 0;
        }
#else
        uint32_t i;
        for (i = 0; i < chunk; i++) {
            if (in[off + (size_t)i] >= 128u) {
                return 0;
            }
        }
#endif
        off += chunk;
        n -= chunk;
    }
    return 1;
}

/* ---- Public API --------------------------------------------------- */

int zpse_flat7_should(const uint8_t *in, size_t n, uint32_t enc_flags,
                      int indexed, size_t out_cap)
{
    size_t packed;

    if (in == NULL || n < (size_t)ZPSE_FLAT7_MIN_SIZE || n > (size_t)UINT32_MAX) {
        return 0;
    }
    if (indexed || (enc_flags & ZPSE_ENC_CHECKSUM) != 0u) {
        return 0;
    }
    packed = n - n / 8u;
    if (out_cap < FLAT7_HEADER_SIZE + packed + 8u) {
        return 0;
    }
    if (!flat7_is_all_ascii(in, n)) {
        return 0;
    }
    return flat7_ascii_gate(in, n);
}

int zpse_flat7_is_frame(const uint8_t *in, size_t in_len)
{
    if (in == NULL || in_len < FLAT7_HEADER_SIZE) {
        return 0;
    }
    return zpse_load_u32le(in) == ZPSE_FLAT7_MAGIC;
}

int zpse_flat7_encode(const uint8_t *in, size_t n, uint8_t *out,
                      size_t out_cap, size_t *out_len_out)
{
    size_t packed;
    size_t total;

    if (in == NULL || out == NULL || out_len_out == NULL) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0u || n > (size_t)UINT32_MAX) {
        return ZPSE_BAD_ARG;
    }

    packed = n - n / 8u;
    total = FLAT7_HEADER_SIZE + packed;
    /* The (BMI2 or scalar) stream packer's final overlapping 8-byte store can
     * reach up to 7 bytes past packed_len, so 8 bytes of scratch beyond the
     * frame must stay writable -- the same +8 zpse_flat7_should demands. */
    if (out_cap < total + 8u) {
        return ZPSE_OUTPUT_OVERFLOW;
    }

    /* Delegate to the fused single-pass path used by the CLI. It performs the
     * whole-buffer ASCII scan, both redundancy-gate predicates and the 7-bit
     * pack in one cache-resident walk, committing the 40-byte header only
     * after every gate passes. This gives library callers the same 16-20x
     * encode speedup the CLI measures, with frame bytes identical to the old
     * separate-scan path on every input both paths accept.
     *
     * The stream entry point returns 1 on commit and 0 on decline; translate
     * that to this API's 0 == ZPSE_OK / negative == error convention. */
    return zpse_flat7_encode_stream(in, n, out, out_cap, out_len_out)
               ? ZPSE_OK : ZPSE_BAD_HEADER;
}

/* Fused single-pass encoder. The old path scanned the whole input for ASCII,
 * then walked it twice more for the two gate predicates, then walked it again
 * to pack -- four passes over an input that may be larger than cache. Here a
 * single walk of 8-aligned chunks does all the work: each chunk is packed and
 * its high-bit verdict folded in by zpse_ascii_pack_checked (one walk, data
 * cache-resident), and the D4 head grams / 256-strided boundary grams are fed
 * to the gate from that same chunk before it leaves cache.
 *
 * The 40-byte header is deliberately built LAST, after every gate passes.
 * A high byte found mid-pack would be silently cleared by the 7-bit packer,
 * so no frame may be committed until the entire input is known ASCII. On any
 * decline we leave `out`'s header untouched and *out_len unchanged. */
int zpse_flat7_encode_stream(const uint8_t *in, size_t n, uint8_t *out,
                             size_t out_cap, size_t *out_len)
{
    size_t packed;
    size_t total;
    size_t sample;
    size_t off = 0u;
    size_t packed_done = 0u;
    Flat7GateState gate;

    if (in == NULL || out == NULL || out_len == NULL) {
        return ZPSE_BAD_ARG;
    }
    if (n < (size_t)ZPSE_FLAT7_MIN_SIZE || n > (size_t)UINT32_MAX) {
        return 0; /* declined: basic size gate (mirrors zpse_flat7_should) */
    }
    packed = n - n / 8u;
    total = FLAT7_HEADER_SIZE + packed;
    /* The stream packer's final 8-byte store can reach 7 bytes past the
     * payload, so keep the same +8 scratch zpse_flat7_should demands. */
    if (out_cap < total + 8u) {
        return 0; /* declined: capacity gate */
    }

    flat7_gate_init(&gate);
    sample = (n < (size_t)FLAT7_D4_SAMPLE) ? n : (size_t)FLAT7_D4_SAMPLE;

    while (off < n) {
        size_t remain = n - off;
        size_t chunk = (remain > (size_t)FLAT7_STREAM_CHUNK)
                           ? (size_t)FLAT7_STREAM_CHUNK
                           : remain;
        size_t chunk_end = off + chunk;
        uint32_t got;
        int chunk_ascii;
        size_t i;
        size_t hi;
        size_t b;

        /* D4 head: feed every 4-byte gram starting in [off, chunk_end) whose
         * end is inside sample. `sample - 3` is the last valid start. */
        hi = chunk_end;
        if (hi > sample - 3u) {
            hi = sample - 3u;
        }
        for (i = off; i < hi; i++) {
            uint32_t k;

            memcpy(&k, in + i, sizeof k);
            flat7_gate_d4_feed(&gate, k);
        }

        /* The D4 head is at most 4096 bytes; the first chunk is at least
         * ZPSE_FLAT7_MIN_SIZE (4096) and holds up to FLAT7_STREAM_CHUNK, so
         * the whole head lands in this (first) chunk and no later chunk feeds
         * a D4 gram. D4 is final here and -- unlike a general incremental
         * ratio, which a run of fresh grams could lift from decline to fire --
         * cannot later recover. The gate ANDs D4 with boundary_dup and the
         * per-chunk ASCII checks, so this decline is the final 0/decline
         * regardless of the rest of the input. Return now to skip the
         * whole-input BD scan and the pack; verdict and committed frame are
         * byte-identical, only the wasted work is gone. */
        if (off == 0u && flat7_gate_d4_declines(&gate)) {
            return 0;
        }

        /* boundary_dup: feed the grams at offsets 0, 256, 512, ... that fall
         * in this chunk and whose four bytes are inside the input. The chunk
         * size is a multiple of 256 so starts stay stride-aligned; `b` is
         * rounded up anyway, which covers the final short chunk. */
        b = ((off + FLAT7_BD_STRIDE - 1u) / FLAT7_BD_STRIDE) * FLAT7_BD_STRIDE;
        for (; b + 4u <= chunk_end && b + 4u <= n; b += FLAT7_BD_STRIDE) {
            uint32_t k;

            memcpy(&k, in + b, sizeof k);
            flat7_gate_bd_feed(&gate, k);
        }

        /* Fused: pack this chunk and fold its high-bit OR into chunk_ascii. */
        got = zpse_ascii_pack_checked(in + off, (uint32_t)chunk,
                                      out + FLAT7_HEADER_SIZE + packed_done,
                                      &chunk_ascii);
        if (chunk_ascii == 0) {
            return 0; /* a high byte: decline, commit nothing */
        }
        packed_done += got;
        off = chunk_end;
    }

    if (packed_done != packed || !flat7_gate_result(&gate)) {
        return 0; /* pack/length disagreement or gate declined */
    }

    /* All gates passed: build the 40-byte header (CRC zeroed, filled after
     * the payload is already in place) and commit the frame. */
    memset(out, 0, FLAT7_HEADER_SIZE);
    zpse_store_u32le(out + 0, ZPSE_FLAT7_MAGIC);
    out[4] = (uint8_t)ZPSE_FLAT7_VER_MAJOR;
    out[5] = (uint8_t)ZPSE_FLAT7_VER_MINOR;
    zpse_store_u16le(out + 6, 0u);
    zpse_store_u64le(out + 8, (uint64_t)n);
    zpse_store_u32le(out + 16, 1u); /* nregions */
    zpse_store_u32le(out + FLAT7_REGION_OFF + FLAT7_R_OFFSET, 0u);
    zpse_store_u32le(out + FLAT7_REGION_OFF + FLAT7_R_PACKED, (uint32_t)packed);
    zpse_store_u32le(out + FLAT7_REGION_OFF + FLAT7_R_OUTLEN, (uint32_t)n);
    out[FLAT7_REGION_OFF + FLAT7_R_TYPE] = (uint8_t)ZPSE_FLAT7_REGION_ASCII7;
    /* reserved bytes [33..36) already zero */

    zpse_store_u32le(out + FLAT7_CRC_OFF, zpse_crc32c(out, FLAT7_CRC_OFF));
    *out_len = total;
    return 1;
}

int zpse_flat7_header_check(const uint8_t *in, size_t in_len,
                            size_t *out_n, size_t *out_payload_off)
{
    uint32_t total;
    uint32_t nregions;
    uint32_t off;
    uint32_t packed_len;
    uint32_t out_len;
    uint8_t type;
    size_t expected;
    size_t need;

    if (in == NULL) {
        return ZPSE_BAD_ARG;
    }
    if (in_len < FLAT7_HEADER_SIZE) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u32le(in) != ZPSE_FLAT7_MAGIC) {
        return ZPSE_BAD_MAGIC;
    }
    if (in[4] != (uint8_t)ZPSE_FLAT7_VER_MAJOR ||
        in[5] != (uint8_t)ZPSE_FLAT7_VER_MINOR) {
        return ZPSE_BAD_VERSION;
    }
    if (zpse_load_u16le(in + 6) != 0u) {
        return ZPSE_BAD_HEADER;
    }

    /* total_out_size is u64 but the region's out_len is u32; they must agree
     * and the format only ever names inputs of at most UINT32_MAX bytes. */
    {
        uint64_t total64 = zpse_load_u64le(in + 8);
        if (total64 > (uint64_t)UINT32_MAX) {
            return ZPSE_BAD_HEADER;
        }
        total = (uint32_t)total64;
    }

    nregions = zpse_load_u32le(in + 16);
    if (nregions != 1u) {
        return ZPSE_BAD_HEADER;
    }

    off = zpse_load_u32le(in + FLAT7_REGION_OFF + FLAT7_R_OFFSET);
    if (off != 0u) {
        return ZPSE_BAD_HEADER;
    }
    packed_len = zpse_load_u32le(in + FLAT7_REGION_OFF + FLAT7_R_PACKED);
    out_len = zpse_load_u32le(in + FLAT7_REGION_OFF + FLAT7_R_OUTLEN);
    if (out_len != total) {
        return ZPSE_BAD_HEADER;
    }
    type = in[FLAT7_REGION_OFF + FLAT7_R_TYPE];
    if (type != (uint8_t)ZPSE_FLAT7_REGION_ASCII7 &&
        type != (uint8_t)ZPSE_FLAT7_REGION_RAW) {
        return ZPSE_BAD_HEADER;
    }
    if (in[FLAT7_REGION_OFF + FLAT7_R_RESERVED] != 0u ||
        in[FLAT7_REGION_OFF + FLAT7_R_RESERVED + 1u] != 0u ||
        in[FLAT7_REGION_OFF + FLAT7_R_RESERVED + 2u] != 0u) {
        return ZPSE_BAD_HEADER;
    }

    if (zpse_load_u32le(in + FLAT7_CRC_OFF) != zpse_crc32c(in, FLAT7_CRC_OFF)) {
        return ZPSE_BAD_HEADER_CRC;
    }

    /* A canonical ASCII7 payload is exactly out_len - out_len/8 bytes. A RAW
     * region (forward compat) carries out_len verbatim instead. */
    expected = (size_t)out_len - (size_t)(out_len / 8u);
    if (type == (uint8_t)ZPSE_FLAT7_REGION_ASCII7) {
        if ((size_t)packed_len != expected) {
            return ZPSE_BAD_HEADER;
        }
    } else {
        if (packed_len != out_len) {
            return ZPSE_BAD_HEADER;
        }
        expected = (size_t)out_len;
    }

    need = FLAT7_HEADER_SIZE + (size_t)packed_len;
    if (in_len < need) {
        return ZPSE_TRUNCATED;
    }

    if (out_n != NULL) {
        *out_n = (size_t)out_len;
    }
    if (out_payload_off != NULL) {
        *out_payload_off = FLAT7_HEADER_SIZE;
    }
    return ZPSE_OK;
}

int zpse_flat7_decode_ex(const uint8_t *in, size_t in_len, uint8_t *out,
                         size_t out_cap, size_t *out_len_out,
                         uint32_t dec_flags)
{
    size_t n;
    size_t payload_off;
    uint32_t packed_len;
    uint8_t type;
    int rc;

    if (out_len_out == NULL) {
        return ZPSE_BAD_ARG;
    }
    /* Validate the entire header (including CRC and lengths) before any
     * write to the output buffer. */
    rc = zpse_flat7_header_check(in, in_len, &n, &payload_off);
    if (rc != ZPSE_OK) {
        return rc;
    }
    if (out == NULL) {
        return ZPSE_BAD_ARG;
    }
    if (n > out_cap) {
        return ZPSE_OUTPUT_OVERFLOW;
    }

    packed_len = zpse_load_u32le(in + FLAT7_REGION_OFF + FLAT7_R_PACKED);
    type = in[FLAT7_REGION_OFF + FLAT7_R_TYPE];
    if (type == (uint8_t)ZPSE_FLAT7_REGION_RAW) {
        /* One verbatim forward copy into the caller's output, with no read
         * of those bytes before the frame completes: same shape as the RAW
         * block copy, so the same opt-in gate applies. */
        if (n >= (size_t)UINT32_MAX) {
            return ZPSE_BAD_HEADER;
        }
        if (zpse_ntout_ok(dec_flags, out, (uint32_t)n)) {
            zpse_ntout_copy(out, in + payload_off, (uint32_t)n);
        } else {
            memcpy(out, in + payload_off, n);
        }
    } else {
        flat7_unpack(in + payload_off, packed_len, out, (uint32_t)n);
    }

    *out_len_out = n;
    return ZPSE_OK;
}

int zpse_flat7_decode(const uint8_t *in, size_t in_len, uint8_t *out,
                      size_t out_cap, size_t *out_len_out)
{
    return zpse_flat7_decode_ex(in, in_len, out, out_cap, out_len_out, 0u);
}
