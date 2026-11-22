#include "zpse/zpse.h" /* the version macros: the string is built from them */

#include "zpse/api.h"

#include "zpse/arena.h"
#include "zpse/block.h"
#include "zpse/f64fast.h"
#include "zpse/dpcm1.h"
#include "zpse/mrd.h"
#include "zpse/flat7.h"
#include "zpse/frame.h"
#include "zpse/repack.h"
#include "zpse/rlefast.h"
#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>

/* Whole-frame codec flags. These five bits are defined here rather than in a
 * public header because this translation unit is the only consumer; the header
 * owner can promote them verbatim without changing any behaviour. They sit
 * above every allocated ZPSE_ENC_* bit (the highest live one is HUF at 8;
 * bits 9-13 are free or reserved), so no existing caller's flags change
 * meaning.
 *
 * Bits 14-16 (REPACK / RLEFAST / F64FAST) are still pure opt-ins: no bit means
 * no attempt. Bits 17-18 (DPCM1 / MRD) are DEPRECATED NO-OPs: the two 16-bit
 * codecs are content-gated and run by default on every call to
 * zpse_compress_frame_ex without CHECKSUM, so passing either bit produces the
 * same bytes as passing none. The bits remain accepted so an older caller
 * still compiles and round-trips. All five are stripped before the block
 * encoder sees them, and only zpse_compress_frame_ex acts on them. */
#ifndef ZPSE_ENC_REPACK
#define ZPSE_ENC_REPACK (UINT32_C(1) << 14)
#endif
#ifndef ZPSE_ENC_RLEFAST
#define ZPSE_ENC_RLEFAST (UINT32_C(1) << 15)
#endif
#ifndef ZPSE_ENC_F64FAST
#define ZPSE_ENC_F64FAST (UINT32_C(1) << 16)
#endif
#ifndef ZPSE_ENC_DPCM1
#define ZPSE_ENC_DPCM1 (UINT32_C(1) << 17)
#endif
#ifndef ZPSE_ENC_MRD
#define ZPSE_ENC_MRD (UINT32_C(1) << 18)
#endif
#define ZPSE_WHOLE_FRAME_FLAGS                                                \
    ((uint32_t)(ZPSE_ENC_REPACK | ZPSE_ENC_RLEFAST | ZPSE_ENC_F64FAST |      \
                ZPSE_ENC_DPCM1 | ZPSE_ENC_MRD))

/* Smallest valid ZRPK (repack whole-frame) header: magic/version/width/flags
 * (8), n u64 (8), payload_len/distinct/header_crc32 (12) and a 16-byte MAP16
 * symbol map = 44 bytes. repack.h does not export this, so name it here. */
#define API_ZRPK_MIN_HEADER 44u

/* Byte-assembled little-endian loads for the whole-frame headers. No unaligned
 * dereference and no dependence on the host's byte order. */
static uint32_t api_load32le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t api_load64le(const uint8_t *p)
{
    return (uint64_t)api_load32le(p) | ((uint64_t)api_load32le(p + 4) << 32);
}

/* Seven distinct four-byte magics: PSE1 = "PSE1", F7P1 = "F7P1",
 * "ZRPK", "RLE1", "F64F", "DCM1" and "MRD1". They differ in at least one byte,
 * so no sniff below can match a normal PSE1 frame or another whole-frame form. */
static int api_is_zrpk(const uint8_t *in, size_t in_len)
{
    return in_len >= 4u && in[0] == 'Z' && in[1] == 'R' &&
           in[2] == 'P' && in[3] == 'K';
}

static int api_is_rle1(const uint8_t *in, size_t in_len)
{
    return in_len >= 4u && in[0] == 'R' && in[1] == 'L' &&
           in[2] == 'E' && in[3] == '1';
}

static int api_is_f64f(const uint8_t *in, size_t in_len)
{
    return in_len >= 4u && in[0] == 'F' && in[1] == '6' &&
           in[2] == '4' && in[3] == 'F';
}

/* The sixth whole-frame magic: "DCM1", the 16-bit-sample DPCM form. */
static int api_is_dcm1(const uint8_t *in, size_t in_len)
{
    return in_len >= 4u && in[0] == 'D' && in[1] == 'C' &&
           in[2] == 'M' && in[3] == '1';
}

/* A seventh whole-frame magic: "MRD1", the DPCM + zero-run-RLE form. */
static int api_is_mrd1(const uint8_t *in, size_t in_len)
{
    return in_len >= 4u && in[0] == 'M' && in[1] == 'R' &&
           in[2] == 'D' && in[3] == '1';
}

/* ---- Combined cheap pre-gate for the whole-frame opt-ins --------------
 * Each whole-frame codec opens with a cheap accept/reject gate: repack needs
 * a 7-bit-ASCII sample, rlefast needs >=90% of that same sample inside runs of
 * >=4 narrow bytes, f64fast needs n%8==0 and a small, dominant alphabet in
 * byte 7 of a few evenly spaced lanes. Tried serially, non-target data pays
 * for every gate before falling back to PSE1. These two helpers gather the
 * same verdicts once, so the attempts in zpse_compress_frame_ex can skip a
 * codec whose gate is certainly false.
 *
 * Each test is a NECESSARY condition for its codec to commit: a clear flag
 * means that codec would decline on its own, so skipping it cannot change a
 * verdict or a byte of output. A set flag only means the codec is still worth
 * calling; its own gate and capacity checks remain the final authority. The
 * attempt order is deliberately left as repack, rlefast, f64fast -- several
 * codecs can accept the same input, and reordering would change which magic
 * the first commit writes. No pre-gate fact is computed for a codec that was
 * not opted in. DPCM1 and MRD are not part of this opt-in set: they run
 * unconditionally (subject only to CHECKSUM) in a separate block below, and
 * each opens with its own prefix probe and content gate, so neither needs a
 * shared pre-gate fact. */
static void api_wf_sample_gates(const uint8_t *in, size_t n,
                                int *ascii_ok, int *rle_ok)
{
    /* repack and rlefast both look at the first min(n, 64 KiB) bytes. Walk it
     * once: the high-bit probe is repack's sample gate, and the run grouping
     * reproduces rlefast's clipped gate contribution (a run straddling the
     * sample end counts only its in-sample prefix, exactly as rlefast does). */
    size_t m = (n < (size_t)65536u) ? n : (size_t)65536u;
    size_t i = 0;
    uint64_t good = 0;

    *ascii_ok = 1;
    while (i < m) {
        uint8_t v = in[i];
        size_t run = 1;

        if ((v & 0x80u) != 0u) {
            *ascii_ok = 0;
        }
        while (i + run < m && in[i + run] == v) {
            ++run;
        }
        if (v <= 127u && run >= 4u) {
            good += (uint64_t)run;
        }
        i += run;
    }
    /* rlefast declines when good*10 < m*9; commit needs the complement. */
    *rle_ok = (good * 10u >= (uint64_t)m * 9u);
}

static int api_wf_f64_gate(const uint8_t *in, size_t n)
{
    /* Byte-for-byte the shape predicate in zpse_f64fast_encode's gate: lanes
     * are sampled at i*step for i in [0,sample), so a different stride could
     * change a verdict near the distinct/dominance thresholds. */
    uint32_t hi[256] = {0};
    uint32_t lanes, sample, step, i, b, max = 0u, distinct = 0u;

    if (n == 0u || (n & 7u) != 0u || n > (size_t)UINT32_MAX) {
        return 0;
    }
    lanes = (uint32_t)n / 8u;
    sample = lanes < 8192u ? lanes : 8192u;
    if (sample == 0u) {
        return 0;
    }
    step = lanes / sample;
    for (i = 0u; i < sample; ++i) {
        ++hi[in[(size_t)(i * step) * 8u + 7u]];
    }
    for (b = 0u; b < 256u; ++b) {
        if (hi[b] != 0u) {
            ++distinct;
            if (hi[b] > max) {
                max = hi[b];
            }
        }
    }
    if (distinct > 8u) {
        return 0;
    }
    if (distinct > 2u && max * 10u < sample * 6u) {
        return 0;
    }
    return 1;
}

const char *zpse_version_string(void)
{
    return ZPSE_VERSION_STRING;
}

/* Encoder options are ZPSE_ENC_CHECKSUM, ZPSE_ENC_LZ_FAST and ZPSE_ENC_HUF,
 * plus one of the nine levels in bits 1-4 (an empty level field means the
 * default level). The five frame-level flags (ZPSE_ENC_REPACK /
 * ZPSE_ENC_RLEFAST / ZPSE_ENC_F64FAST / ZPSE_ENC_DPCM1 / ZPSE_ENC_MRD, bits
 * 14-18) are all accepted here so a caller may pass them without a BAD_ARG.
 * Only zpse_compress_frame_ex acts on them:
 * bits 14-16 are genuine opt-ins; bits 17-18 are deprecated no-ops because
 * those codecs now run by default. Anything else is a bad argument rather than
 * a silent round to a neighbour:
 * the level is the caller's own speed/size trade, so a codec that cannot
 * honour it must say so. Bits outside those fields are refused for the same
 * reason -- notably bit 11, which stays reserved. */
static int enc_flags_ok(uint32_t enc_flags)
{
    uint32_t lvl;

    if ((enc_flags &
         ~(uint32_t)(ZPSE_ENC_CHECKSUM | ZPSE_ENC_SPEED_MASK |
                     ZPSE_ENC_LZ_FAST | ZPSE_ENC_HUF |
                     ZPSE_ENC_REPACK | ZPSE_ENC_RLEFAST |
                      ZPSE_ENC_F64FAST | ZPSE_ENC_DPCM1 |
                      ZPSE_ENC_MRD)) != 0u) {
        return 0;
    }
    lvl = (uint32_t)((enc_flags & ZPSE_ENC_SPEED_MASK) >>
                     ZPSE_ENC_SPEED_SHIFT);
    return lvl <= 9u;
}

int zpse_compress_block(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap,
                        uint32_t *out_len_out,
                        uint32_t seq, int is_last,
                        void *enc_arena)
{
    /* The default stream carries no optional fields: no payload CRC. */
    return zpse_compress_block_ex(in, n, out, out_cap, out_len_out, seq,
                                  is_last, 0u, enc_arena);
}

int zpse_compress_block_ex(const uint8_t *in, uint32_t n,
                           uint8_t *out, uint32_t out_cap,
                           uint32_t *out_len_out,
                           uint32_t seq, int is_last, uint32_t enc_flags,
                           void *enc_arena)
{
    ZpseEncoder enc;
    int rc;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arena == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0 || n > 65536u) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (!enc_flags_ok(enc_flags)) {
        return ZPSE_BAD_ARG;
    }
    rc = zpse_encoder_init(&enc, enc_arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_block_encode(&enc, in, n, seq, is_last, out, out_cap);
    if (rc < 0) {
        return rc;
    }
    *out_len_out = (uint32_t)rc;
    return ZPSE_OK;
}

int zpse_decompress_block(const uint8_t *in, uint32_t in_len,
                          uint8_t *output, uint32_t out_cap,
                          uint32_t *out_len_out,
                          void *dec_arena)
{
    ZpseDecoder dec;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arena == 0) {
        return ZPSE_BAD_ARG;
    }
    /* A page's block is 16 bytes of header, so this floor is the short one:
     * the header itself is what says which form it is. */
    if (in_len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    rc = zpse_decoder_init(&dec, dec_arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    rc = zpse_block_decode(&dec, in, in_len, output, out_cap, out_len_out);
    if (rc < 0) {
        return rc;
    }
    return ZPSE_OK;
}

int zpse_compress_frame(const uint8_t *in, size_t n,
                        uint8_t *out, size_t out_cap,
                        size_t *out_len_out,
                        uint32_t nominal_block_size,
                        void *enc_arena)
{
    /* The default stream carries no optional fields: no payload CRC. */
    return zpse_compress_frame_ex(in, n, out, out_cap, out_len_out,
                                  nominal_block_size, 0u, enc_arena);
}

int zpse_compress_frame_ex(const uint8_t *in, size_t n,
                           uint8_t *out, size_t out_cap,
                           size_t *out_len_out,
                           uint32_t nominal_block_size, uint32_t enc_flags,
                           void *enc_arena)
{
    ZpseEncoder enc;
    int rc;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arena == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0) {
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size == 0 || nominal_block_size > 65536u) {
        return ZPSE_BAD_ARG;
    }
    if (!enc_flags_ok(enc_flags)) {
        return ZPSE_BAD_ARG;
    }
    /* Opt-in whole-frame forms: only bits 14-16 still control an attempt.
     * With none of them set this block is skipped entirely. A checksum cannot
     * be carried by a whole-frame stream, so those requests fall through to
     * the normal encoder too. The combined pre-gate above lets a codec be
     * skipped when its own cheap gate would certainly decline; the codecs that
     * remain are still tried in the original order (first commit wins). */
    if ((enc_flags & (ZPSE_ENC_REPACK | ZPSE_ENC_RLEFAST |
                      ZPSE_ENC_F64FAST)) != 0u &&
        (enc_flags & ZPSE_ENC_CHECKSUM) == 0u) {
        int repack_on = (enc_flags & ZPSE_ENC_REPACK) != 0u;
        int rle_on = (enc_flags & ZPSE_ENC_RLEFAST) != 0u;
        int f64_on = (enc_flags & ZPSE_ENC_F64FAST) != 0u;
        int ascii_ok = 1;
        int rle_ok = 0;
        size_t repack_len = 0;

        /* The serial-gate cost only exists when two or more codecs can be
         * tried. With a single opt-in, call that codec directly and pay no
         * pre-gate pass at all -- its own gate is already the first thing it
         * does. With several opt-ins, repack and rlefast share one contiguous
         * sample scan instead of two. */
        int use_gate = (repack_on + rle_on + f64_on) > 1;
        if (use_gate && (repack_on || rle_on)) {
            api_wf_sample_gates(in, n, &ascii_ok, &rle_ok);
        }

        if (repack_on && (!use_gate || ascii_ok) &&
            zpse_repack_encode(in, n, out, out_cap, &repack_len) == 1) {
            *out_len_out = repack_len;
            return ZPSE_OK;
        }
        if (rle_on && (!use_gate || rle_ok) &&
            n <= (size_t)UINT32_MAX) {
            uint32_t cap32 = (out_cap > (size_t)UINT32_MAX)
                                 ? UINT32_MAX : (uint32_t)out_cap;
            uint32_t rle_len = 0;

            if (zpse_rlefast_encode(in, (uint32_t)n, out, cap32,
                                    &rle_len) == 1) {
                *out_len_out = (size_t)rle_len;
                return ZPSE_OK;
            }
        }
        /* Evaluate the lane probe last and only if this codec can still be
         * reached; repack or rlefast committing above skips it entirely. */
        if (f64_on && n <= (size_t)UINT32_MAX &&
            (!use_gate || api_wf_f64_gate(in, n))) {
            uint32_t cap32 = (out_cap > (size_t)UINT32_MAX)
                                 ? UINT32_MAX : (uint32_t)out_cap;
            uint32_t f64_len = 0;

            if (zpse_f64fast_encode(in, (uint32_t)n, out, cap32,
                                    &f64_len) == ZPSE_F64FAST_COMMIT) {
                *out_len_out = (size_t)f64_len;
                return ZPSE_OK;
            }
        }
    }
    /* Default whole-frame forms. DPCM1 (16-bit rough deltas, x-ray shape) and
     * MRD (16-bit zero-heavy deltas, mr shape) are enabled at every level and
     * need no flag: each opens with its own prefix probe and content gate and
     * is the first thing that touches `out` on that attempt, so a decline
     * leaves the buffer untouched for the next codec or the PSE1 encoder. Bits
     * 17 and 18 are accepted but ignored -- passing them changes no byte. The
     * level field still controls every PSE1 file normally; these two codecs
     * are not levels and are tried in the fixed order DPCM1, then MRD. A
     * CHECKSUM request skips both, since neither whole-frame form can carry a
     * payload CRC. */
    if ((enc_flags & ZPSE_ENC_CHECKSUM) == 0u &&
        n <= (size_t)UINT32_MAX) {
        uint32_t cap32 = (out_cap > (size_t)UINT32_MAX)
                             ? UINT32_MAX : (uint32_t)out_cap;
        uint32_t wf_len = 0u;

        if (zpse_dpcm1_encode(in, (uint32_t)n, out, cap32, &wf_len) ==
            ZPSE_DPCM1_COMMIT) {
            *out_len_out = (size_t)wf_len;
            return ZPSE_OK;
        }
        if (zpse_mrd_encode(in, (uint32_t)n, out, cap32, &wf_len) ==
            ZPSE_MRD_COMMIT) {
            *out_len_out = (size_t)wf_len;
            return ZPSE_OK;
        }
    }
    rc = zpse_encoder_init(&enc, enc_arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    /* Strip the whole-frame opt-ins: the block encoder knows none of them and
     * normal PSE1 output must not depend on whether they were passed. */
    enc.enc_flags = enc_flags & ~ZPSE_WHOLE_FRAME_FLAGS;
    rc = zpse_frame_encode(&enc, in, n, nominal_block_size,
                           out, out_cap, out_len_out);
    if (rc < 0) {
        return rc;
    }
    return ZPSE_OK;
}

int zpse_compress_frame_indexed(const uint8_t *in, size_t n,
                                uint8_t *out, size_t out_cap,
                                size_t *out_len_out,
                                uint32_t nominal_block_size,
                                uint32_t enc_flags,
                                void *enc_arena)
{
    ZpseEncoder enc;
    int rc;

    if (in == 0 || out == 0 || out_len_out == 0 || enc_arena == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0) {
        return ZPSE_BAD_ARG;
    }
    if (nominal_block_size == 0 || nominal_block_size > 65536u) {
        return ZPSE_BAD_ARG;
    }
    if (!enc_flags_ok(enc_flags)) {
        return ZPSE_BAD_ARG;
    }
    rc = zpse_encoder_init(&enc, enc_arena, (size_t)ZPSE_ENC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    enc.enc_flags = enc_flags;
    rc = zpse_frame_encode_ex(&enc, in, n, nominal_block_size, out, out_cap,
                              out_len_out, 1);
    if (rc < 0) {
        return rc;
    }
    return ZPSE_OK;
}

int zpse_seek_frame(const uint8_t *in, size_t in_len, uint32_t block_index,
                    uint8_t *output, uint32_t out_cap, uint32_t *out_len_out,
                    void *dec_arena)
{
    ZpseDecoder dec;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arena == 0) {
        return ZPSE_BAD_ARG;
    }
    /* The whole-frame codecs carry no block index, so a block cannot be
     * addressed. Sniff their magics here, before the frame seek reads this
     * frame's magic and reports BAD_MAGIC, so the caller learns the operation
     * is unsupported. FLAT7 is handled the same way. Each sniff is length
     * guarded, so this must precede the 32-byte floor below: a short but valid
     * RLE1 stream (e.g. 14 bytes for n=1) is still recognised as unsupported
     * rather than being misreported as truncated. */
    if (zpse_flat7_is_frame(in, in_len) || api_is_zrpk(in, in_len) ||
        api_is_rle1(in, in_len) || api_is_f64f(in, in_len) ||
        api_is_dcm1(in, in_len) || api_is_mrd1(in, in_len)) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (in_len < 32u) {
        return ZPSE_TRUNCATED;
    }
    rc = zpse_decoder_init(&dec, dec_arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    return zpse_frame_seek(&dec, in, in_len, block_index, output, out_cap,
                           out_len_out);
}

int zpse_decompress_frame(const uint8_t *in, size_t in_len,
                          uint8_t *output, size_t out_cap,
                          size_t *out_len_out,
                          void *dec_arena)
{
    ZpseDecoder dec;
    int rc;

    if (in == 0 || output == 0 || out_len_out == 0 || dec_arena == 0) {
        return ZPSE_BAD_ARG;
    }
    if (in_len == 0) {
        return ZPSE_BAD_ARG;
    }
    /* Only rule out a truly empty stream here. The whole-frame magics below
     * can be shorter than a PSE1 frame (a valid RLE1 stream is 12 bytes with
     * n=0), so they must be sniffed before any PSE1-sized floor. PSE1 frames
     * are left to zpse_frame_decode, which enforces ZPSE_FRAME_HEADER_SIZE
     * (32 bytes) itself and reports ZPSE_TRUNCATED for a short magic match. */
    if (in_len < 4u) {
        return ZPSE_TRUNCATED;
    }
    /* A FLAT7 frame is a separate magic with its own whole-frame decoder. */
    if (zpse_flat7_is_frame(in, in_len)) {
        return zpse_flat7_decode(in, in_len, output, out_cap, out_len_out);
    }
    /* ZRPK and RLE1 are two more separate whole-frame magics with their own
     * decoders. Both formats spell their original length in a fixed header, so
     * this function can report it in *out_len_out just as the PSE1 path does.
     * Neither decoder accepts a PSE1 frame: the four magics are distinct. */
    if (api_is_zrpk(in, in_len)) {
        uint64_t n64;

        /* The magic sniff only needs 4 bytes; reading n and the minimum
         * header below needs API_ZRPK_MIN_HEADER bytes. */
        if (in_len < (size_t)API_ZRPK_MIN_HEADER) {
            return ZPSE_TRUNCATED;
        }
        n64 = api_load64le(in + 8); /* original length, u64 LE */
        if (n64 > (uint64_t)SIZE_MAX) {
            return ZPSE_BAD_HEADER;
        }
        /* repack_decode conflates a too-small out_cap with a malformed frame,
         * so bound n here; every other whole-frame path reports overflow too. */
        if (n64 > (uint64_t)out_cap) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        if (!zpse_repack_decode(in, in_len, output, out_cap)) {
            return ZPSE_BAD_HEADER;
        }
        *out_len_out = (size_t)n64;
        return ZPSE_OK;
    }
    if (api_is_rle1(in, in_len)) {
        uint32_t n;

        /* api_is_rle1 only needs 4 bytes for the magic, but reading the
         * length field below needs the full 12-byte RLE1 header. */
        if (in_len < (size_t)ZPSE_RLEFAST_HEADER_SIZE) {
            return ZPSE_TRUNCATED;
        }
        if (in_len > (size_t)UINT32_MAX) {
            return ZPSE_TRUNCATED;
        }
        n = api_load32le(in + 4); /* original length, u32 LE */
        /* rlefast_decode takes n but no out_cap, so bound it here. */
        if ((size_t)n > out_cap) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        if (zpse_rlefast_decode(in, (uint32_t)in_len, output, n) < 0) {
            return ZPSE_BAD_HEADER;
        }
        *out_len_out = (size_t)n;
        return ZPSE_OK;
    }
    if (api_is_f64f(in, in_len)) {
        uint32_t n;

        /* The magic sniff needs 4 bytes; the F64F envelope is 24 bytes and
         * carries its original length at offset 8 (u32 LE). */
        if (in_len < (size_t)ZPSE_F64FAST_HEADER_SIZE) {
            return ZPSE_TRUNCATED;
        }
        if (in_len > (size_t)UINT32_MAX) {
            return ZPSE_TRUNCATED;
        }
        n = api_load32le(in + 8); /* original length, u32 LE */
        /* f64fast_decode takes n but no out_cap, so bound it here. */
        if ((size_t)n > out_cap) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        if (zpse_f64fast_decode(in, (uint32_t)in_len, output, n) !=
            ZPSE_F64FAST_COMMIT) {
            return ZPSE_BAD_HEADER;
        }
        *out_len_out = (size_t)n;
        return ZPSE_OK;
    }
    /* DCM1 is a separate whole-frame magic with its own decoder. Its 16-byte
     * header carries the original length at offset 4 (u32 LE), and the decoder
     * validates every field before writing, so bound n against out_cap here
     * just as the other whole-frame paths do. */
    if (api_is_dcm1(in, in_len)) {
        uint32_t n;

        if (in_len < (size_t)ZPSE_DPCM1_HEADER_SIZE) {
            return ZPSE_TRUNCATED;
        }
        if (in_len > (size_t)UINT32_MAX) {
            return ZPSE_TRUNCATED;
        }
        n = api_load32le(in + 4); /* original length, u32 LE */
        if ((size_t)n > out_cap) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        if (zpse_dpcm1_decode(in, (uint32_t)in_len, output, n) !=
            ZPSE_DPCM1_COMMIT) {
            return ZPSE_BAD_HEADER;
        }
        *out_len_out = (size_t)n;
        return ZPSE_OK;
    }
    /* MRD1 is a separate whole-frame magic with its own decoder. Its 24-byte
     * header carries the original length at offset 4 (u32 LE). */
    if (api_is_mrd1(in, in_len)) {
        uint32_t n;

        if (in_len < (size_t)ZPSE_MRD_HEADER_SIZE) {
            return ZPSE_TRUNCATED;
        }
        if (in_len > (size_t)UINT32_MAX) {
            return ZPSE_TRUNCATED;
        }
        n = api_load32le(in + 4); /* original length, u32 LE */
        if ((size_t)n > out_cap) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        if (zpse_mrd_decode(in, (uint32_t)in_len, output, n) !=
            ZPSE_MRD_COMMIT) {
            return ZPSE_BAD_HEADER;
        }
        *out_len_out = (size_t)n;
        return ZPSE_OK;
    }
    rc = zpse_decoder_init(&dec, dec_arena, (size_t)ZPSE_DEC_ARENA_SIZE);
    if (rc != ZPSE_OK) {
        return rc;
    }
    rc = zpse_frame_decode(&dec, in, in_len, output, out_cap, out_len_out);
    if (rc < 0) {
        return rc;
    }
    return ZPSE_OK;
}
