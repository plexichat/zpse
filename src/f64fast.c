/* f64fast -- pinned DELTA_TRANSPOSE_8 fast path for float64 arrays.
 *
 * Included by src/zpse_unity.c (it is built as part of the library, not as a
 * separate translation unit). Everything below is a thin, single-shape driver
 * over public zpse_* entry points; the block/segment/transform kernels
 * themselves are the unmodified library's.
 *
 * The shape of this file follows DTYPE1's measured prototype (bench_proto.c,
 * "forced tid=9, wrap=0"): one DELTA_TRANSPOSE_8 staging pass, eight segments
 * through the library's segment coder, one normal PSE1 frame. The ten-candidate
 * transform sweep and the LZ_TILE wrap are deliberately absent. Decode is the
 * unmodified zpse_decompress_frame() on the inner payload.
 */

#include "zpse/f64fast.h"

#include "zpse/api.h"
#include "zpse/arena.h"
#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/frame.h"
#include "zpse/predictor.h"
#include "zpse/segment.h"
#include "zpse/status.h"
#include "zpse/transform.h"
#include "zpse/types.h"

#include <stdlib.h>
#include <string.h>

/* The cached arenas must stay per-thread: a process-global fallback would let
 * two threads share (and free) the same arena. Every compiler the build
 * supports has a real TLS spelling, so never silently drop to empty. */
#if defined(_MSC_VER) && !defined(__clang__)
#define F64FAST_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define F64FAST_TLS _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
/* GCC and Clang support __thread as an extension well before C11, including
 * clang-cl (which defines _MSC_VER but is excluded from the branch above). */
#define F64FAST_TLS __thread
#else
#error "f64fast requires thread-local storage: build as C11, or use GCC/Clang (__thread) or MSVC (__declspec(thread))."
#endif

/* Cached, 64-byte aligned arenas. Kept per thread so neither direction pays
 * for a fresh mapping on every call; the library never clears these arenas
 * (it reinitialises every region before reading it).
 *
 * There is no portable TLS destructor here: __declspec(thread), _Thread_local
 * and __thread all have no C11/C++ destructor hook, and pthread/TSS keys would
 * require a threads library and edits outside this file. The caches therefore
 * leak on thread exit. Callers that create short-lived threads MUST call
 * zpse_f64fast_release() (ZPSE_F64FAST_RELEASE in the public header) once per
 * thread before it exits; for long-lived threads the leak is bounded at one
 * encoder and one decoder arena per thread. */
static F64FAST_TLS uint8_t *f64fast_enc_arena;
static F64FAST_TLS uint8_t *f64fast_dec_arena;

static uint8_t *arena_get(uint8_t **slot, size_t bytes)
{
    uint8_t *p = *slot;
    if (p == 0) {
        void *raw = malloc(bytes + (size_t)ZPSE_ARENA_ALIGN + sizeof(void *));
        uintptr_t from;
        uintptr_t aligned;
        if (raw == 0) return 0;
        from = (uintptr_t)raw + sizeof(void *);
        aligned = (from + (uintptr_t)ZPSE_ARENA_ALIGN - 1u) &
                  ~((uintptr_t)ZPSE_ARENA_ALIGN - 1u);
        ((void **)aligned)[-1] = raw; /* keep the malloc base for release */
        p = (uint8_t *)aligned;
        *slot = p;
    }
    return p;
}

void zpse_f64fast_release(void)
{
    if (f64fast_enc_arena) {
        free(((void **)f64fast_enc_arena)[-1]);
        f64fast_enc_arena = 0;
    }
    if (f64fast_dec_arena) {
        free(((void **)f64fast_dec_arena)[-1]);
        f64fast_dec_arena = 0;
    }
}

/* envelope + frame header + raw input + generous tile/palette expansion
 * + one block's fixed overhead. The normal encoder needs the same slack;
 * this is the same envelope the prototype benchmark allocated.
 *
 * Computed in 64 bits without truncation. zpse_f64fast_bound() must still
 * return a uint32 (0 on overflow) for its public API, but a caller that gates
 * a WRITE has to compare its capacity against this untruncated value: for
 * n > ~2.86 GiB the uint32 form is 0, which would make an `out_cap < bound`
 * check pass and let the encoder write past the caller's buffer. */
static uint64_t f64fast_bound64(uint32_t n)
{
    if (n == 0u || (n & 7u) != 0u) return 0u;
    return (uint64_t)ZPSE_F64FAST_HEADER_SIZE + (uint64_t)ZPSE_FRAME_HEADER_SIZE +
           (uint64_t)n + (uint64_t)n / 2u + (uint64_t)ZPSE_F64FAST_BLOCK_SIZE;
}

uint32_t zpse_f64fast_bound(uint32_t n)
{
    uint64_t b = f64fast_bound64(n);
    return b > UINT32_MAX ? 0u : (uint32_t)b;
}

/* The measured float64 corpus has a nearly constant top byte in every 8-byte
 * lane (v ~ 0.5 -> 0x3F in byte 7). Sample evenly across the whole block; a
 * random buffer gives ~256 distinct values and a ~0.4% dominant value. */
static int looks_like_f64(const uint8_t *in, uint32_t n)
{
    uint32_t lanes, sample, step;
    uint32_t hi[256];
    uint32_t i, b, max = 0u, distinct = 0u;

    /* Reject bad shapes and the zero-sample case BEFORE any division whose
     * divisor is derived from n. For n==0 or n in 1..7, lanes==0 and so
     * sample==0; computing step=lanes/sample first would be a 0/0 UB/SIGFPE.
     * n==0 and non-multiples of 8 are declined before even lanes=n/8. */
    if (in == 0 || n == 0u || (n & 7u) != 0u) return 0;
    lanes = n / 8u;
    sample = lanes < 8192u ? lanes : 8192u;
    if (sample == 0u) return 0;
    step = lanes / sample;
    memset(hi, 0, sizeof hi);
    for (i = 0u; i < sample; ++i) {
        uint32_t lane = i * step;
        ++hi[in[(size_t)lane * 8u + 7u]];
    }
    for (b = 0u; b < 256u; ++b) {
        if (hi[b] != 0u) {
            ++distinct;
            if (hi[b] > max) max = hi[b];
        }
    }
    if (distinct > 8u) return 0;
    /* A short smooth run can straddle one exponent boundary, so byte 7 may
     * be two adjacent values (0x3E/0x3F) split close to evenly. Random bytes
     * have ~256 distinct values and are rejected above. */
    if (distinct > 2u && max * 10u < sample * 6u) return 0;
    return 1;
}

int zpse_f64fast_encode(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap, uint32_t *out_len)
{
    ZpseEncoder e;
    ZpseLaneState lanes;
    ZpseFrameHeader fh;
    uint8_t meta[16];
    uint8_t *arena;
    uint8_t *fb;
    uint32_t fcap;
    size_t pos = (size_t)ZPSE_FRAME_HEADER_SIZE;
    uint32_t off, seq = 0u, flen, crc;
    uint32_t nb;

    if (out_len != 0) *out_len = 0u;
    if (in == 0 || out == 0 || out_len == 0) return ZPSE_F64FAST_DECLINE;
    if (!looks_like_f64(in, n)) return ZPSE_F64FAST_DECLINE;
    /* Capacity gate in 64 bits: a uint32 bound() of 0 (invalid shape or
     * 1.5n + 65592 overflowing uint32) must never be read as "no room
     * needed". Every n whose bound exceeds uint32 is refused here. */
    {
        uint64_t need = f64fast_bound64(n);
        if (need == 0u || (uint64_t)out_cap < need)
            return ZPSE_F64FAST_DECLINE;
    }

    nb = n / ZPSE_F64FAST_BLOCK_SIZE +
         ((n % ZPSE_F64FAST_BLOCK_SIZE) != 0u ? 1u : 0u);
    /* Even a perfectly compressed pinned frame has at least a frame header,
     * a block header, the eight seed bytes, eight directory entries and eight
     * segment headers per block. Refuse before writing a single byte when
     * that floor already reaches the raw envelope. */
    if ((uint64_t)ZPSE_FRAME_HEADER_SIZE +
            (uint64_t)nb * ((uint64_t)ZPSE_BLOCK_HEADER_SIZE + 8u +
                            8u * ZPSE_SEG_DIR_ENTRY_SIZE +
                            8u * ZPSE_SEGMENT_HEADER_SIZE) >= (uint64_t)n)
        return ZPSE_F64FAST_DECLINE;

    arena = arena_get(&f64fast_enc_arena, ZPSE_ENC_ARENA_SIZE);
    if (arena == 0 || zpse_encoder_init(&e, arena, ZPSE_ENC_ARENA_SIZE) != ZPSE_OK)
        return ZPSE_F64FAST_DECLINE;
    e.enc_flags = 0u;
    e.const_plane = 0;
    e.ascii_gate = 0u;
    lanes.order1 = e.order1;
    lanes.o2 = e.o2;
    lanes.gen = 0u;
    lanes.o2_ready = 0u;

    fb = out + ZPSE_F64FAST_HEADER_SIZE;
    fcap = out_cap - ZPSE_F64FAST_HEADER_SIZE;
    if (zpse_frame_header_build(&fh, ZPSE_F64FAST_BLOCK_SIZE, (uint64_t)n, 1,
                                nb, 1, 0) != ZPSE_OK)
        return ZPSE_F64FAST_DECLINE;
    zpse_frame_header_write(&fh, fb);

    /* Transform 9 is DELTA_TRANSPOSE_8: width 8, one seed byte per plane. */
    for (off = 0u; off < n; off += ZPSE_F64FAST_BLOCK_SIZE) {
        uint32_t chunk = (off + ZPSE_F64FAST_BLOCK_SIZE <= n)
                             ? ZPSE_F64FAST_BLOCK_SIZE : (n - off);
        uint32_t last = (off + chunk >= n) ? 1u : 0u;
        ZpseTransformMeta tm;
        uint8_t *planes[16];
        uint32_t lens[16], segsz[16];
        uint64_t base, cum = 0u;
        uint64_t sum = 0u;
        uint32_t o = 0u, total;
        unsigned s, k;
        ZpseBlockHeader h;
        const uint8_t sc = 8u;
        const uint8_t meta_len = 8u;

        /* The seeds are the first byte of each 8-byte lane of the block --
         * exactly what the normal encoder writes as transform metadata. */
        for (s = 0u; s < meta_len; ++s) meta[s] = in[off + s];
        if (zpse_transform_meta_parse(&tm, ZPSE_F64FAST_TID_PINNED, meta,
                                      meta_len, sc) != ZPSE_OK)
            return ZPSE_F64FAST_DECLINE;
        if (zpse_segment_lengths(chunk, sc, lens) != ZPSE_OK)
            return ZPSE_F64FAST_DECLINE;
        for (s = 0u; s < sc; ++s) {
            planes[s] = e.tstage + o;
            o += lens[s];
            sum += lens[s];
        }
        if (sum != (uint64_t)chunk ||
            zpse_transform_forward_planes(&tm, sc, in + off, chunk, planes,
                                          lens) != ZPSE_OK)
            return ZPSE_F64FAST_DECLINE;

        base = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + meta_len +
               (uint64_t)sc * ZPSE_SEG_DIR_ENTRY_SIZE;
        e.flat_covered = 0u;
        e.flat_paid = 0u;
        for (s = 0u; s < sc; ++s) {
            uint64_t at = base + cum;
            uint8_t lit = lens[s] != 0u ? planes[s][0] : 0u;
            int wl = zpse_segment_encode(&e, &lanes, planes[s], lens[s], lit,
                                         fb + pos + (size_t)at,
                                         fcap - (uint32_t)pos - (uint32_t)at,
                                         0u, 0u);
            if (wl < (int)ZPSE_SEGMENT_HEADER_SIZE)
                return ZPSE_F64FAST_DECLINE;
            segsz[s] = (uint32_t)wl;
            cum += (uint64_t)wl;
        }
        total = (uint32_t)(base + cum);

        memcpy(fb + pos + ZPSE_BLOCK_HEADER_SIZE, meta, meta_len);
        for (s = 0u; s < sc; ++s) {
            uint32_t rel = 0u;
            uint8_t *ent = fb + pos + ZPSE_BLOCK_HEADER_SIZE + meta_len +
                           (size_t)s * ZPSE_SEG_DIR_ENTRY_SIZE;
            for (k = 0u; k < s; ++k) rel += segsz[k];
            zpse_store_u32le(ent, rel);
            zpse_store_u32le(ent + 4u, segsz[s]);
        }

        memset(&h, 0, sizeof h);
        h.sync = ZPSE_BLOCK_SYNC;
        h.block_version = ZPSE_BLOCK_VERSION;
        h.block_flags = (uint8_t)(ZPSE_B_RESET | ZPSE_B_HAS_META |
                                (last ? ZPSE_B_LAST : 0u));
        h.uncompressed_size = chunk;
        h.compressed_block_size = total;
        h.transform_id = ZPSE_F64FAST_TID_PINNED;
        h.segment_count = sc;
        h.transform_meta_size = meta_len;
        h.tile_size = ZPSE_TILE_SIZE;
        h.block_sequence_number = seq++;
        h.header_size = ZPSE_BLOCK_HEADER_SIZE;
        zpse_block_header_serialize(fb + pos, &h);
        pos += total;
    }
    flen = (uint32_t)pos;

    /* Raw-envelope check: both forms carry the same 24-byte envelope, so the
     * pinned inner frame must simply be smaller than n to be worth keeping. */
    if (flen >= n) return ZPSE_F64FAST_DECLINE;

    zpse_store_u32le(out, ZPSE_F64FAST_MAGIC);
    out[4] = ZPSE_F64FAST_VERSION;
    out[5] = ZPSE_F64FAST_FLAG_PINNED;
    out[6] = 0u;
    out[7] = 0u;
    zpse_store_u32le(out + 8u, n);
    zpse_store_u32le(out + 12u, ZPSE_F64FAST_TID_PINNED);
    zpse_store_u32le(out + 16u, flen);
    crc = zpse_crc32c_init();
    crc = zpse_crc32c_update(crc, out, 20u);
    crc = zpse_crc32c_update(crc, fb, flen);
    zpse_store_u32le(out + 20u, zpse_crc32c_finish(crc));

    *out_len = ZPSE_F64FAST_HEADER_SIZE + flen;
    return ZPSE_F64FAST_COMMIT;
}

int zpse_f64fast_decode(const uint8_t *in, uint32_t in_len,
                        uint8_t *out, uint32_t n)
{
    uint32_t tid, plen, crc;
    size_t ol = 0u;
    uint8_t flags;
    uint8_t *arena;
    ZpseDecoder dec;
    int rc;

    if (in == 0 || out == 0 || in_len < ZPSE_F64FAST_HEADER_SIZE)
        return ZPSE_F64FAST_DECLINE;
    if (zpse_load_u32le(in) != ZPSE_F64FAST_MAGIC) return ZPSE_F64FAST_DECLINE;
    if (in[4] != ZPSE_F64FAST_VERSION) return ZPSE_F64FAST_DECLINE;
    flags = in[5];
    if ((flags & (uint8_t)~(ZPSE_F64FAST_FLAG_RAW | ZPSE_F64FAST_FLAG_PINNED)) != 0u)
        return ZPSE_F64FAST_DECLINE;
    if ((flags & ZPSE_F64FAST_FLAG_RAW) != 0u &&
        (flags & ZPSE_F64FAST_FLAG_PINNED) != 0u)
        return ZPSE_F64FAST_DECLINE;
    if ((in[6] | in[7]) != 0u) return ZPSE_F64FAST_DECLINE;
    if (n == 0u || (n & 7u) != 0u) return ZPSE_F64FAST_DECLINE;
    if (zpse_load_u32le(in + 8u) != n) return ZPSE_F64FAST_DECLINE;

    tid = zpse_load_u32le(in + 12u);
    plen = zpse_load_u32le(in + 16u);
    if (plen != in_len - ZPSE_F64FAST_HEADER_SIZE)
        return ZPSE_F64FAST_DECLINE;
    crc = zpse_crc32c_init();
    crc = zpse_crc32c_update(crc, in, 20u);
    crc = zpse_crc32c_update(crc, in + ZPSE_F64FAST_HEADER_SIZE, plen);
    if (zpse_crc32c_finish(crc) != zpse_load_u32le(in + 20u))
        return ZPSE_F64FAST_DECLINE;

    if ((flags & ZPSE_F64FAST_FLAG_RAW) != 0u) {
        if (tid != ZPSE_F64FAST_TID_RAW || plen != n)
            return ZPSE_F64FAST_DECLINE;
        memcpy(out, in + ZPSE_F64FAST_HEADER_SIZE, n);
        return ZPSE_F64FAST_COMMIT;
    }
    if ((flags & ZPSE_F64FAST_FLAG_PINNED) == 0u ||
        tid != ZPSE_F64FAST_TID_PINNED)
        return ZPSE_F64FAST_DECLINE;

    arena = arena_get(&f64fast_dec_arena, ZPSE_DEC_ARENA_SIZE);
    if (arena == 0) return ZPSE_F64FAST_DECLINE;
    if (zpse_decoder_init(&dec, arena, (size_t)ZPSE_DEC_ARENA_SIZE) != ZPSE_OK)
        return ZPSE_F64FAST_DECLINE;
    /* Decode the payload strictly as a normal PSE1 frame. Never route it back
     * through zpse_decompress_frame(): that dispatcher re-sniffs every
     * whole-frame magic (including F64F), so an outer F64F whose payload is
     * another F64F would recurse once per 24-byte envelope until the stack
     * overflows.
     * The internal frame decoder only understands PSE1, so a non-PSE1 payload
     * is refused as a bad header instead of re-entering the codec. */
    rc = zpse_frame_decode(&dec, in + ZPSE_F64FAST_HEADER_SIZE, (size_t)plen,
                           out, (size_t)n, &ol);
    if (rc < 0 || ol != (size_t)n) return ZPSE_F64FAST_DECLINE;
    return ZPSE_F64FAST_COMMIT;
}
