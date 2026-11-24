#include "zpse/block.h"

#include "zpse/arena.h"
#include "zpse/bits.h"
#include "zpse/constants.h"
#include "zpse/crc32c.h"
#include "zpse/dense.h"
#include "zpse/huf.h"
#include "zpse/lz.h"
#include "zpse/predictor.h"
#include "zpse/segment.h"
#include "zpse/status.h"
#include "zpse/tile.h"
#include "zpse/transform.h"
#include "zpse/types.h"

#include <string.h>
#if defined(ZPSE_TRACE_TRIALS)
#include <stdio.h>
#endif

/* The RAW/no-CRC fallback is a verbatim copy into the caller's output, and no
 * decoder path reads those bytes back inside the block. With
 * ZPSE_DEC_STREAMING_OUTPUT the caller says it will not reread them soon, so
 * streaming (non-temporal) stores skip the RFO for that destination and are
 * still byte-for-byte identical to memcpy. SSE2 is baseline on x86-64, so no
 * target attribute is needed; FORCE_SCALAR parity builds keep the plain
 * write-back copy. The kernel is only ever reached when the hint is set. */
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86)) && !defined(ZPSE_FORCE_SCALAR)
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define ZPSE_BLOCK_HAVE_NTSTORE 1
#include <emmintrin.h>
#endif
#endif
#ifndef ZPSE_BLOCK_HAVE_NTSTORE
#define ZPSE_BLOCK_HAVE_NTSTORE 0
#endif

/* Below this length the sfence costs more than the RFO it avoids -- the same
 * measured cutoff text.c uses for its NT unpack. */
#define ZPSE_BLOCK_NT_MIN 4096u

#if ZPSE_BLOCK_HAVE_NTSTORE
#if defined(_MSC_VER)
#define ZPSE_BLOCK_NT_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define ZPSE_BLOCK_NT_TLS _Thread_local
#else
#define ZPSE_BLOCK_NT_TLS
#endif
/* A caller that decodes successive blocks into the SAME output address has a
 * hot destination even when it set the hint (measured: 1.2 us memcpy against
 * 3.3 us NT for a reused 64 KiB buffer). Remember the last RAW destination per
 * thread and keep the write-back copy for it; a streaming caller that rotates
 * buffers sees a new address every block and gets the NT path. */
static ZPSE_BLOCK_NT_TLS const uint8_t *block_raw_nt_last_out;
#endif

/* The FLAT page form (sync 0x50B9) is the ONE page path at every level. Its
 * sync and codec ids come from constants.h; the decoder lives in flat.h. Do
 * not include flat.h here: it declares the codecs as an enum while
 * constants.h defines the same names as macros, so a TU that has both would
 * not preprocess. The #ifndef fallbacks and the local prototype keep this
 * file buildable on its own before either header lands; they are exactly the
 * values/prototype those headers define. */
#ifndef ZPSE_BLOCK_SYNC_FLAT
#define ZPSE_BLOCK_SYNC_FLAT ((uint16_t)0x50B9)
#endif
#ifndef ZPSE_FLAT_RAW
#define ZPSE_FLAT_RAW ((uint8_t)0)
#endif
#ifndef ZPSE_FLAT_NIBBLE4
#define ZPSE_FLAT_NIBBLE4 ((uint8_t)1)
#endif
#ifndef ZPSE_FLAT_BIT2
#define ZPSE_FLAT_BIT2 ((uint8_t)2)
#endif
#ifndef ZPSE_FLAT_SUBLELANE
#define ZPSE_FLAT_SUBLELANE ((uint8_t)3)
#endif
int zpse_flat_decode(uint8_t codec, uint8_t param, const uint8_t *payload,
                     uint32_t payload_len, uint8_t *out, uint32_t n);

/* The TEXT page form (sync 0x50BA) keeps its codec in text.c. As with FLAT,
 * include no codec header: block.c declares only what it calls, so this file
 * still preprocesses and type-checks before text.h lands. The header parser
 * and serializer are already declared by zpse/types.h. */
#ifndef ZPSE_BLOCK_SYNC_TEXT
#define ZPSE_BLOCK_SYNC_TEXT ((uint16_t)0x50BA)
#endif
uint32_t zpse_text_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                          uint32_t cap);
int zpse_text_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                     uint32_t n);
uint32_t zpse_text_price(const uint8_t *in, uint32_t n);
/* Byte-identical to zpse_text_encode for ASCII input, but omits the ASCII
 * eligibility scan: every caller has already proved every byte <128 (the gate
 * and zpse_text_price below). Declared locally because the public text.h is
 * intentionally not included here (see the note above). */
uint32_t zpse_text_encode_ascii(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap);
/* The classifier and its result type live in their own header so a caller can
 * use the same classes. Keep a local copy here for a standalone build of this
 * file made before that header lands. */
#if defined(__has_include)
#if __has_include("zpse/pagefast.h")
#include "zpse/pagefast.h"
#define ZPSE_BLOCK_HAVE_PAGEFAST 1
#endif
#endif
#ifndef ZPSE_BLOCK_HAVE_PAGEFAST
typedef enum {
    ZPSE_PAGE_ZERO,
    ZPSE_PAGE_ONE_VALUE,
    ZPSE_PAGE_LOW_ALPHA,
    ZPSE_PAGE_ASCII,
    ZPSE_PAGE_POINTER,
    ZPSE_PAGE_GENERIC
} ZpsePageClass;
typedef struct {
    uint8_t value;
    uint8_t alpha[16];
    uint8_t nalpha;
    uint16_t nzero;
    uint32_t flags;
    uint8_t lane_diff[32];
    uint8_t zero_cnt[32];
} ZpsePageInfo;
int zpse_page_classify(const uint8_t *in, uint32_t n, ZpsePageInfo *info);
#endif

/* Capacity of the transform sweep's candidate list, and of one candidate's
 * transform metadata. The list used to hold three shapes -- RAW_BYTES,
 * DELTA8 and a single plain transpose -- so both were sized to a small
 * constant and the metadata to two bytes (DELTA8's one seed byte plus slack).
 *
 * The sweep now also offers the narrower plain transposes and the
 * delta-transposes, which the format has always been able to encode and
 * decode but the encoder never picked. DELTA_TRANSPOSE_16 carries sixteen
 * seed bytes, so metadata grows to 16. See zpse_block_encode for why these
 * particular shapes and how their seeds are chosen. */
#ifndef ZPSE_MAX_CAND
#define ZPSE_MAX_CAND 10u
#endif
#ifndef ZPSE_MAX_CAND_META
#define ZPSE_MAX_CAND_META 16u
#endif

/* How much smaller the LZ-over-tile-body form has to be than the tile stream
 * alone before it is taken (the v2 stage at the end of zpse_block_encode).
 * The form costs the decoder a match-finder pass over the body before the
 * tile reconstruction, and it costs the encoder a full match-finder pass to
 * build, so a few bytes is the wrong trade: measured over Silesia at the
 * default level, the stage takes 13.3% off the tile payload of exactly the
 * blocks where tiles beat LZ. Raising this number trades ratio straight back
 * for encode and decode speed -- every block it turns away is a block that
 * skips the pass on both sides. */
#ifndef ZPSE_LZ_TILE_DIV
#define ZPSE_LZ_TILE_DIV 32u
#endif

/* History-confident stop for the trial sweep: when the order-n predictor
 * chose the first trial AND its shape has won this many blocks in a row, the
 * sweep stops after that one trial instead of encoding the rest in full.
 *
 * The run length is the signal because the survivor set cannot provide one:
 * every survivor is already within 1/16 of the winner's price (that is what
 * kept it), so no ratio of prices can separate "the winner is clearly right"
 * from "the field is tied" -- measured, a price-ratio rule fired on zero
 * blocks of Silesia and wrote byte-identical streams to a build without it.
 * A shape that has been the winner for N consecutive 64 KiB blocks is a
 * different order of evidence: it says the file is in a homogeneous stretch,
 * which is the case the predictor exists for.
 *
 * 0 disables the rule (the sweep then always runs to its floor exits, which
 * are exact). */
#ifndef ZPSE_HIST_STOP_STREAK
#define ZPSE_HIST_STOP_STREAK 0u
#endif

/* Whether the previous block's winning shape may be moved to the front of the
 * trial list, once the set and every price-driven decision are final (see the
 * block in zpse_block_encode; it is a permutation of attempt order and nothing
 * else).
 *
 * Off by default, and now for a measured reason rather than a correctness one.
 * While the hint was promoted to `pick` it also fed the flat-block cap and the
 * tied-field cut, so a block encoded after different neighbours could come out
 * a few bytes different -- the same 20 MB at level 6 packed to 15,098,769
 * bytes on one worker and 15,101,233 on four -- and dropping it recovered
 * 0.016% of the corpus (104,050,674 against 104,066,700 on 202 MiB). Moved
 * behind those decisions it is byte-identical to the canonical order on the
 * whole 202 MiB corpus at levels 6 and 9, and the speed it buys is inside the
 * noise of this host: 437.3 ms against 441.0 ms per 103 MB, best-of-three,
 * single core, so there is nothing left to recover by turning it on. */
#ifndef ZPSE_HIST_REORDER
#define ZPSE_HIST_REORDER 0u
#endif

/* Block envelope codec.
 * No allocation; staging uses caller arenas. Helpers static. */

/* Opt-in block-sweep accounting (-DZPSE_BLOCK_STATS): what the transform
 * sweep actually spent on a block. The encoder's candidate list is not
 * observable from outside it, so "how many candidates did this block
 * really try, and how many of those tries were wasted?" cannot be measured
 * without counters -- and that answer decides where the sweep's cost is.
 *
 * Each counter is a global, so a multi-threaded caller would race; the
 * counters exist to answer questions about a benchmark run, not to be
 * part of the format. */
#if defined(ZPSE_BLOCK_STATS)
/* Blocks encoded (one call to zpse_block_encode that got past the size
 * checks). The denominator for every per-block average below. */
uint32_t zpse_block_blocks;
/* Candidates the list BUILT, before any pruning. */
uint32_t zpse_block_cands_built;
/* Candidates the decision probe PRICED (one exact probe per candidate). */
uint32_t zpse_block_cands_priced;
/* Candidates that survived the probe's margin and went into the trial
 * loop. */
uint32_t zpse_block_cands_kept;
/* Full block encodes actually SPENT (a trial that ran to the end of the
 * block). */
uint32_t zpse_block_trials;
/* Trials skipped by the per-candidate floor bound without any work. */
uint32_t zpse_block_trials_skipped;
/* Blocks the ASCII gate convicted: one candidate, the seven-bit pack. */
uint32_t zpse_block_gated;
/* Full match-finder passes run, and the cycles they took. The finder is the
 * one stage whose whole cost is paid whenever it is tried, so this is what a
 * screen that accepts it too often actually costs. */
uint32_t zpse_block_lz_passes;
uint64_t zpse_block_lz_pass_cycles;
uint64_t zpse_block_screen_cycles;
/* Blocks the finder wrote with no tile trial at all, split by the stage that
 * decided it: the pre-sweep screen (full-size blocks) and the gate's own
 * exact comparison (an ASCII block, whose seven-bit total is arithmetic). */
uint32_t zpse_block_finder_first;
uint32_t zpse_block_gate_lz;
/* How many times a trial was won, by transform id (0..10: RAW_BYTES,
 * DELTA8, and the plain/delta transposes at widths 2/4/8/16). */
uint32_t zpse_block_wins[16];
/* Total compressed bytes each transform id produced across all trials. */
uint32_t zpse_block_bytes[16];
/* How many trials each transform id ran. */
uint32_t zpse_block_trials_by_tid[16];
/* The winning transform id of each block, counted (one entry per block).
 * Slot 15 is the RAW fallback: the sweep found nothing that paid and the
 * block went out raw. */
uint32_t zpse_block_win_tid[16];
/* Where the winner sat in the trial order: the histogram of "which trial
 * actually won", 0 = the first trial. It is the direct answer to "is the
 * sweep's own ordering already right most of the time?", and therefore to
 * "would trying fewer candidates cost compressed size?". */
uint32_t zpse_block_win_pos[32];
/* How much smaller the final winner was than the FIRST trial's total,
 * summed over the blocks, and how many blocks had a first trial that was
 * not the winner. A non-zero sum means the later trials earned their
 * keep; a zero sum means they did not. */
uint32_t zpse_block_first_trial_delta;
uint32_t zpse_block_first_trial_lost;
/* The margin, in bytes, between the probe's best price and the worst
 * survivor's price -- the width of the band the probe kept. Summed. */
uint32_t zpse_block_margin_sum;
/* The best price the probe produced. Summed, for a per-block average. */
uint32_t zpse_block_price_sum;
/* Blocks where the probe's winner was also the trial loop's winner: the
 * probe got it right. */
uint32_t zpse_block_probe_correct;
/* Cycles spent in the decision probe (exact prices) and in the trial
 * loop (full block encodes), so the split between "deciding which shape to
 * try" and "trying the shapes" is measured rather than assumed. Read
 * with rdtsc, so the two counters are comparable to each other but not
 * to wall-clock time. */
uint64_t zpse_block_probe_cycles;
uint64_t zpse_block_trial_cycles;
/* The two stages that are part of a block's encode but not of its decision:
 * the tile-body (v2) finder pass and the Huffman wrap. Without them the
 * reader's remainder column carries a third of a page-like block's cycles
 * and reads as bookkeeping; with them every percent above 3.7 is named. */
uint64_t zpse_block_v2_cycles;
uint64_t zpse_block_huf_cycles;
/* Cycles spent in the two decisions taken before any candidate is priced:
 * the ASCII gate and the attempt-order sample. Together with the probe, the
 * trials and the LZ stages they account for the block's encode; the rest is
 * header work, the shrink and the histogram. */
uint64_t zpse_block_gate_cycles;
uint64_t zpse_block_sample_cycles;
/* A serialising cycle read: the counter must not be reordered across the
 * work it measures. LFENCE is enough on every x86-64 that runs this
 * build's AVX2 paths. */
static uint64_t blk_rdtsc(void)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    uint32_t lo;
    uint32_t hi;

    __asm__ __volatile__("lfence" ::: "memory");
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    __asm__ __volatile__("lfence" ::: "memory");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
    return 0u;
#endif
}
#endif

static int zpse_block_scount_valid(uint8_t sc)
{
    return sc == 1 || sc == 2 || sc == 4 || sc == 8 || sc == 16;
}

static void zpse_block_fill_header(ZpseBlockHeader *h, uint8_t flags,
                                   uint32_t n, uint32_t c, uint8_t tid,
                                   uint8_t sc, uint16_t meta_len,
                                   uint32_t seq, uint32_t decoded_crc)
{
    h->sync = ZPSE_BLOCK_SYNC;
    h->block_version = ZPSE_BLOCK_VERSION;
    h->block_flags = flags;
    h->uncompressed_size = n;
    h->compressed_block_size = c;
    h->transform_id = tid;
    h->segment_count = sc;
    h->transform_meta_size = meta_len;
    h->tile_size = (uint16_t)ZPSE_TILE_SIZE;
    h->predictor_profile = 0;
    h->reserved0 = 0;
    h->block_sequence_number = seq;
    h->decoded_crc32c = decoded_crc;
    h->header_crc32c = 0;
    h->header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE;
}

/* Write the winner's header in the 16-byte page form when it can carry it.
 *
 * `out` holds [32-byte header][payload] of `total` bytes. Sixteen of those
 * header bytes are fields the format fixes (version, tile size, predictor
 * profile, reserved) or that a page does not use (a payload CRC, which is
 * dropped here with its flag), so a page's block can say the same thing in 16
 * -- and 16
 * bytes is 0.30 points of the shared-library dump's ratio at 63.6%, or 0.34
 * points over the three dumps together (52 065 pages, 45 000 of which
 * shrink, 719 591 bytes). The payload moves down onto the header it just gave up;
 * nothing else about the block changes, and the sync value tells a reader
 * which header it is looking at.
 *
 * Refused (total returned unchanged) for a block over ZPSE_BLOCK_SHORT_MAX or
 * a size the 16-bit fields would overflow -- neither of which a page produces,
 * since a page is 4096 and its raw envelope is 4112. */
static uint32_t zpse_block_shrink_page(uint8_t *out, uint32_t n,
                                       uint32_t total)
{
    ZpseBlockHeader h;
    uint8_t sh[ZPSE_BLOCK_HEADER_SIZE_SHORT];
    uint32_t payload;

    if (n > (uint32_t)ZPSE_BLOCK_SHORT_MAX ||
        total < (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
        return total;
    }
    if (zpse_block_header_parse(&h, out, ZPSE_BLOCK_HEADER_SIZE) != ZPSE_OK) {
        return total;
    }
    if (h.transform_meta_size > 255u) {
        return total;
    }
    /* A page's block has the 16-byte header and no other: the decoded CRC has
     * nowhere to live in it, so a page written with ZPSE_ENC_CHECKSUM drops
     * the field and the flag with it rather than paying eight bytes of header
     * it cannot use (see ZPSE_BLOCK_HEADER_SIZE_SHORT). The flag is what the
     * decoder reads, so the block is a well-formed "no checksum" one. */
    h.block_flags = (uint8_t)(h.block_flags & ~ZPSE_B_CRC);
    h.decoded_crc32c = 0u;
    payload = total - (uint32_t)ZPSE_BLOCK_HEADER_SIZE;
    if (payload + (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT > 65535u) {
        return total;
    }
    h.compressed_block_size = payload + (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    zpse_block_header_serialize_short(sh, &h);
    memmove(out + ZPSE_BLOCK_HEADER_SIZE_SHORT, out + ZPSE_BLOCK_HEADER_SIZE,
            payload);
    memcpy(out, sh, sizeof sh);
    return payload + (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
}

int zpse_block_probe(const uint8_t *in, uint32_t in_len,
                     uint32_t *compressed_size_out,
                     uint32_t *uncompressed_size_out)
{
    ZpseBlockHeader h;
    uint32_t hdr;
    int rc;

    if (in == 0 || compressed_size_out == 0 || uncompressed_size_out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (in_len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    hdr = zpse_block_header_size(in, in_len);
    if (hdr == 0u || in_len < hdr) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u16le(in) == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN) {
        if (in[4] != 1u || in[5] != 0u ||
            in[8] != (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT ||
            in[9] != 0u) {
            return ZPSE_BAD_HEADER;
        }
        if ((uint32_t)(zpse_crc32c(in, 14) & UINT32_C(0xFFFF)) !=
            (uint32_t)zpse_load_u16le(in + 14)) {
            return ZPSE_BAD_HEADER_CRC;
        }
        *compressed_size_out = (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
        *uncompressed_size_out = (uint32_t)zpse_load_u16le(in + 6);
        return ZPSE_OK;
    }
    if (zpse_load_u16le(in) == (uint16_t)ZPSE_BLOCK_SYNC_FLAT) {
        ZpseBlockHeader fh;

        rc = zpse_block_header_parse_flat(&fh, in, in_len);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if (fh.compressed_block_size ==
            (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        if (in_len < fh.compressed_block_size) {
            return ZPSE_TRUNCATED;
        }
        *compressed_size_out = fh.compressed_block_size;
        *uncompressed_size_out = fh.uncompressed_size;
        return ZPSE_OK;
    }
    if (zpse_load_u16le(in) == (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
        ZpseBlockHeader th;

        rc = zpse_block_header_parse_text(&th, in, in_len);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if (th.compressed_block_size ==
            (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
            return ZPSE_BAD_BLOCK_SIZE; /* no empty token stream */
        }
        if (in_len < th.compressed_block_size) {
            return ZPSE_TRUNCATED;
        }
        *compressed_size_out = th.compressed_block_size;
        *uncompressed_size_out = th.uncompressed_size;
        return ZPSE_OK;
    }
    rc = zpse_block_header_parse(&h, in, in_len);
    if (rc != ZPSE_OK) {
        return rc;
    }
    if (h.compressed_block_size < hdr) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (in_len < h.compressed_block_size) {
        return ZPSE_TRUNCATED;
    }
    *compressed_size_out = h.compressed_block_size;
    *uncompressed_size_out = h.uncompressed_size;
    return ZPSE_OK;
}

#if ZPSE_BLOCK_HAVE_NTSTORE
/* Byte-identical memcpy replacement for the RAW/no-CRC payload only. The
 * caller's output alignment is arbitrary, so advance to the next 64-byte
 * boundary with a scalar head copy, then emit only whole cache lines with four
 * 16-byte streaming stores each; the final partial line is a scalar tail.
 * Because the head/tail lines are distinct from the streaming lines, the
 * sfence after the loop both drains the write-combining buffers for caller
 * visibility and keeps a write-back tail from resurrecting stale bytes in a
 * line that still has pending NT stores. */
static void block_raw_nt_copy(uint8_t *dst, const uint8_t *src, uint32_t n)
{
    uintptr_t d = (uintptr_t)(void *)dst;
    uint32_t head = (uint32_t)((64u - (d & (uintptr_t)63u)) & (uintptr_t)63u);
    uint32_t body, lines, i;
    const uint8_t *s;
    uint8_t *p;

    if (head > n) {
        head = n;
    }
    if (head != 0u) {
        memcpy(dst, src, head);
    }
    body = n - head;
    lines = body & ~((uint32_t)63u);
    s = src + head;
    p = dst + head;
    for (i = 0u; i < lines; i += 64u) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(const void *)(s + i));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(const void *)(s + i + 16u));
        __m128i v2 = _mm_loadu_si128((const __m128i *)(const void *)(s + i + 32u));
        __m128i v3 = _mm_loadu_si128((const __m128i *)(const void *)(s + i + 48u));

        _mm_stream_si128((__m128i *)(void *)(p + i), v0);
        _mm_stream_si128((__m128i *)(void *)(p + i + 16u), v1);
        _mm_stream_si128((__m128i *)(void *)(p + i + 32u), v2);
        _mm_stream_si128((__m128i *)(void *)(p + i + 48u), v3);
    }
    _mm_sfence();
    if (lines != body) {
        memcpy(p + lines, s + lines, body - lines);
    }
}
#endif /* ZPSE_BLOCK_HAVE_NTSTORE */

int zpse_block_decode_ex(ZpseDecoder *d, const uint8_t *in, uint32_t in_len,
                      uint8_t *output, uint32_t out_cap,
                      uint32_t *out_len_out, uint32_t dec_flags)
{
    ZpseBlockHeader h;
    ZpseTransformMeta meta;
    ZpseSegmentHeader sh0;
    /* A hint, not stream data: unused in scalar builds and on every path
     * other than the RAW/no-CRC copy. */
    (void)dec_flags;
    uint32_t n, c, meta_size, scount;
    uint64_t dir_end, area;
    const uint8_t *dir, *payloads;
    const uint8_t *body;
    uint32_t body_off, body_len;
    uint32_t hdr;
    uint32_t seg_lens[16];
    uint32_t seg_off[16], seg_sz[16];
    const uint8_t *csegs[16];
    uint32_t cum, s;
    int identity_shape;
    int rc;

    if (d == 0 || in == 0 || output == 0) {
        return ZPSE_BAD_ARG;
    }
    if (d->recon == 0 || d->out_stage == 0) {
        return ZPSE_BAD_ARG;
    }
    if (in_len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    hdr = zpse_block_header_size(in, in_len);
    if (hdr == 0u || in_len < hdr) {
        return ZPSE_TRUNCATED;
    }

    /* One-value page token (v1.1): the short header with a different sync,
     * the fill byte in transform_id, no payload, and the same 16-bit header
     * CRC. Checked before the ordinary parse because the parse rejects a
     * sync it does not know. Every field the token carries is validated
     * here, and the fill is a memset at memcpy-class speed. */
    if (zpse_load_u16le(in) == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN) {
        uint8_t flags = in[2];
        uint8_t value = in[3];
        uint32_t n_tok = (uint32_t)zpse_load_u16le(in + 6);
        uint32_t c_tok = (uint32_t)zpse_load_u16le(in + 8);
        uint16_t crc = zpse_load_u16le(in + 14);

        if (in[4] != 1u || in[5] != 0u ||
            (flags & (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET)) !=
                (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET) ||
            (flags & (uint8_t)(ZPSE_B_CRC | ZPSE_B_HAS_META |
                               ZPSE_B_LZ | ZPSE_B_LZ_TILE | ZPSE_B_HUF |
                               ZPSE_B_RESERVED_MASK)) != 0u ||
            c_tok != (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT ||
            n_tok < ZPSE_MIN_BLOCK_SIZE || n_tok > ZPSE_MAX_BLOCK_SIZE) {
            return ZPSE_BAD_HEADER;
        }
        if ((uint32_t)(zpse_crc32c(in, 14) & UINT32_C(0xFFFF)) != crc) {
            return ZPSE_BAD_HEADER_CRC;
        }
        if (out_cap < n_tok) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        memset(output, value, n_tok);
        if (out_len_out != 0) {
            *out_len_out = n_tok;
        }
        return (int)n_tok;
    }

    /* The FLAT page form (sync 0x50B9): a 16-byte header naming a flat payload
     * codec instead of a transform + segment directory. Checked before the
     * ordinary parse for the same reason as the token -- the parse refuses a
     * sync it does not know. The header parser validates every field (CRC,
     * flags, reserved byte, codec, sizes); what is left here is the length
     * arithmetic between the header, the bytes the caller handed us and the
     * payload the flat decoder expects. */
    if (zpse_load_u16le(in) == (uint16_t)ZPSE_BLOCK_SYNC_FLAT) {
        ZpseBlockHeader fh;
        uint32_t c_flat;
        uint32_t plen;
        int frc;

        frc = zpse_block_header_parse_flat(&fh, in, in_len);
        if (frc != ZPSE_OK) {
            return frc;
        }
        c_flat = fh.compressed_block_size;
        plen = c_flat - (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
        if (c_flat > in_len) {
            return ZPSE_TRUNCATED;
        }
        if (plen == 0u) {
            /* Every codec writes at least one byte for n >= 1, so an empty
             * payload is a size the encoder never produces. */
            return ZPSE_BAD_BLOCK_SIZE;
        }
        if (out_cap < fh.uncompressed_size) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        frc = zpse_flat_decode(fh.transform_id,
                                (uint8_t)fh.transform_meta_size,
                                in + ZPSE_BLOCK_HEADER_SIZE_SHORT, plen,
                                output, fh.uncompressed_size);
        if (frc != ZPSE_OK) {
            return frc;
        }
        if (out_len_out != 0) {
            *out_len_out = fh.uncompressed_size;
        }
        return (int)fh.uncompressed_size;
    }

    /* The TEXT page form (sync 0x50BA): a 16-byte header followed by a token
     * stream that expands to n bytes. Checked with the other page forms for
     * the same reason -- a generic parse would treat its header as an
     * ordinary one. Decode straight into the caller's output: src/text.c's
     * zpse_text_decode rejects in_len != n - (n>>3) before any store, so a
     * malformed stream still leaves no bytes behind. Staging through d->lzbuf
     * plus a full-size memcpy cost ~38% of whole-ASCII decode throughput
     * (5,636 -> 8,568 MB/s), so it was removed. */
    if (zpse_load_u16le(in) == (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
        ZpseBlockHeader th;
        uint32_t c_text;
        uint32_t plen;
        int trc;

        trc = zpse_block_header_parse_text(&th, in, in_len);
        if (trc != ZPSE_OK) {
            return trc;
        }
        c_text = th.compressed_block_size;
        plen = c_text - (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
        if (c_text > in_len) {
            return ZPSE_TRUNCATED;
        }
        if (plen == 0u) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        if (out_cap < th.uncompressed_size) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        trc = zpse_text_decode(in + (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT,
                               plen, output, th.uncompressed_size);
        if (trc < 0) {
            return trc;
        }
        if (out_len_out != 0) {
            *out_len_out = th.uncompressed_size;
        }
        return (int)th.uncompressed_size;
    }

    rc = zpse_block_header_parse(&h, in, in_len);
    if (rc != ZPSE_OK) {
        return rc;
    }
    /* The header CRC is validated by zpse_block_header_parse itself in both
     * the short (types.c) and long form; recomputing it here was a second
     * 14/28-byte CRC pass on every block, which is a measurable share of a
     * 4 KiB page's decode. Removing it changes no refusal: a bad header CRC
     * is returned by the parse above. */

    n = h.uncompressed_size;
    c = h.compressed_block_size;
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (c < hdr) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (in_len < c) {
        return ZPSE_TRUNCATED;
    }

    if ((h.block_flags & ZPSE_B_RESERVED_MASK) != 0) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if ((h.block_flags & ZPSE_B_RESET) == 0) {
        return ZPSE_BAD_HEADER;
    }
    if (h.reserved0 != 0) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (h.predictor_profile != 0) {
        return ZPSE_UNSUPPORTED_FEATURE;
    }
    if (h.tile_size != (uint16_t)ZPSE_TILE_SIZE) {
        return ZPSE_BAD_TILE_SIZE;
    }
    if (h.transform_id >= 12) {
        return ZPSE_BAD_TRANSFORM;
    }
    if (!zpse_block_scount_valid(h.segment_count)) {
        return ZPSE_BAD_TRANSFORM;
    }
    meta_size = h.transform_meta_size;
    {
        int want_meta = (h.block_flags & ZPSE_B_HAS_META) != 0;
        int have_meta = meta_size > 0;
        if (want_meta != have_meta) {
            return ZPSE_BAD_HEADER;
        }
    }

    if (out_cap < n) {
        return ZPSE_OUTPUT_OVERFLOW;
    }

    /* RAW fallback path: exactly N raw bytes at offset 32. */
    if ((h.block_flags & ZPSE_B_RAW) != 0) {
        uint32_t calc;
        if (h.transform_id != 0 || h.segment_count != 1 || meta_size != 0 ||
            (h.block_flags & (ZPSE_B_LZ_TILE | ZPSE_B_LZ | ZPSE_B_HUF)) != 0) {
            return ZPSE_BAD_HEADER;
        }
        if (c != hdr + n) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        if ((h.block_flags & ZPSE_B_CRC) != 0) {
            /* Fused copy + checksum. The fallback payload is a verbatim
             * copy of the input, so a single pass both fills the caller's
             * buffer and returns the CRC of exactly the bytes copied --
             * no second sweep over 64 KiB that has already left L1. */
            calc = zpse_crc32c_copy(output, in + hdr, n);
            if (calc != h.decoded_crc32c) {
                return ZPSE_BAD_PAYLOAD_CRC;
            }
        } else {
#if ZPSE_BLOCK_HAVE_NTSTORE
            /* Only a caller that promises not to reread the output pays the
             * sfence, and only for a destination it has not just written.
             * With the hint clear this is the original memcpy, byte for byte
             * and timing for timing, so existing callers are unchanged. */
            {
                int use_nt =
                    (dec_flags & (uint32_t)ZPSE_DEC_STREAMING_OUTPUT) != 0u &&
                    n >= (uint32_t)ZPSE_BLOCK_NT_MIN &&
                    output != block_raw_nt_last_out;

                block_raw_nt_last_out = output;
                if (use_nt) {
                    block_raw_nt_copy(output, in + hdr, n);
                } else {
                    memcpy(output, in + hdr, n);
                }
            }
#else
            memcpy(output, in + hdr, n);
#endif
        }
        if (out_len_out != 0) {
            *out_len_out = n;
        }
        return (int)n;
    }

    /* The entropy-coded LZ form: the single raw-LZ4 payload, Huffman coded,
     * with the form it unpacks to named inside the payload. It is dispatched
     * here -- before the two LZ branches below -- because ZPSE_B_HUF is bit 7
     * alone: unlike the old combined encodings it cannot be mistaken for an
     * LZ form by a bit test, but the two invalid combinations 0xA0
     * (ZPSE_B_LZ | ZPSE_B_HUF) and 0xC0 (ZPSE_B_LZ_TILE | ZPSE_B_HUF) have to
     * be rejected explicitly. The payload's inner_form is a fixed tag naming
     * ZPSE_B_LZ, so there is only one LZ decoder to run after the Huffman
     * pass. See include/zpse/huf.h. */
    if ((h.block_flags & (uint8_t)ZPSE_B_HUF) == (uint8_t)ZPSE_B_HUF) {
        uint32_t ilen = 0u;
        uint8_t iform = 0u;

        if (h.transform_id != 0 || h.segment_count != 1u || meta_size != 0u ||
            (h.block_flags & (uint8_t)(ZPSE_B_LZ | ZPSE_B_LZ_TILE)) != 0u) {
            return ZPSE_BAD_HEADER;
        }
        if (c <= hdr) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        if (d->lzbuf == 0) {
            return ZPSE_BAD_ARG;
        }
        rc = zpse_huf_decode(in + hdr, c - hdr, d->lzbuf,
                             (uint32_t)ZPSE_D_SZ_LZBUF, &ilen, &iform);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if (iform != (uint8_t)ZPSE_B_LZ) {
            return ZPSE_BAD_HEADER;
        }
        rc = zpse_lz4_decode(d->lzbuf, ilen, output, n);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if ((h.block_flags & ZPSE_B_CRC) != 0) {
            uint32_t calc = zpse_crc32c(output, n);

            if (calc != h.decoded_crc32c) {
                return ZPSE_BAD_PAYLOAD_CRC;
            }
        }
        if (out_len_out != 0) {
            *out_len_out = n;
        }
        return (int)n;
    }

    /* Intra-block LZ4 form: one raw LZ4 block, no transform and no tiles. The
     * block keeps the shape the raw fallback uses (transform 0, one segment,
     * no metadata) so a reader can tell the two apart by flag alone, and every
     * reference it carries is a distance inside this block. */
    if ((h.block_flags & ZPSE_B_LZ) != 0) {
        if (h.transform_id != 0 || h.segment_count != 1u || meta_size != 0u ||
            (h.block_flags & ZPSE_B_LZ_TILE) != 0) {
            return ZPSE_BAD_HEADER;
        }
        if (c <= hdr) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        rc = zpse_lz4_decode(in + hdr, c - hdr, output, n);
        if (rc != ZPSE_OK) {
            return rc;
        }
        if ((h.block_flags & ZPSE_B_CRC) != 0) {
            uint32_t calc = zpse_crc32c(output, n);

            if (calc != h.decoded_crc32c) {
                return ZPSE_BAD_PAYLOAD_CRC;
            }
        }
        if (out_len_out != 0) {
            *out_len_out = n;
        }
        return (int)n;
    }

    /* Coded path: structural directory checks (PHASE1). */
    body = in;
    body_off = hdr;
    body_len = c;
    /* v2: the body below is itself an intra-block LZ stream. Unpack it once
     * into the decoder's own buffer and let the tile path read it as if it
     * had been written plainly -- the unpacked bytes are byte for byte what
     * that path was built against, so nothing below needs a second shape.
     * The length comes first because zpse_lz4_decode has to be told the exact
     * output size: a stream is only accepted when it fills it exactly. */
    if ((h.block_flags & ZPSE_B_LZ_TILE) != 0) {
        uint32_t blen;

        if (d->lzbuf == 0) {
            return ZPSE_BAD_ARG;
        }
        if (c < hdr + 4u + (uint32_t)ZPSE_LZ_MIN_PAYLOAD) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        blen = zpse_load_u32le(in + hdr);
        if (blen < (uint32_t)ZPSE_SEG_DIR_ENTRY_SIZE ||
            blen > (uint32_t)ZPSE_D_SZ_LZBUF) {
            return ZPSE_BAD_BLOCK_SIZE;
        }
        rc = zpse_lz4_decode(in + hdr + 4u, c - (hdr + 4u), d->lzbuf, blen);
        if (rc != ZPSE_OK) {
            return rc;
        }
        body = d->lzbuf;
        body_off = 0u;
        body_len = blen;
    }
    scount = h.segment_count;
    dir_end = (uint64_t)body_off + (uint64_t)meta_size +
              (uint64_t)scount * (uint64_t)ZPSE_SEG_DIR_ENTRY_SIZE;
    if (dir_end > (uint64_t)body_len) {
        return ZPSE_BAD_SEGMENT_DIRECTORY;
    }
    rc = zpse_transform_meta_parse(&meta, h.transform_id,
                                   body + body_off,
                                   (uint16_t)meta_size, h.segment_count);
    if (rc != ZPSE_OK) {
        return rc;
    }
    dir = body + body_off + meta_size;
    payloads = body + dir_end;
    area = (uint64_t)body_len - dir_end;

    {
        uint32_t expect = 0;
        for (s = 0; s < scount; s++) {
            uint32_t off = zpse_load_u32le(dir + (size_t)s * 8);
            uint32_t sz = zpse_load_u32le(dir + (size_t)s * 8 + 4);
            if (off != expect) {
                return ZPSE_BAD_SEGMENT_DIRECTORY;
            }
            if ((uint64_t)off + (uint64_t)sz > area) {
                return ZPSE_BAD_SEGMENT_DIRECTORY;
            }
            if (sz < (uint32_t)ZPSE_SEGMENT_HEADER_SIZE) {
                return ZPSE_BAD_SEGMENT_DIRECTORY;
            }
            seg_off[s] = off;
            seg_sz[s] = sz;
            expect = off + sz;
        }
        if ((uint64_t)expect != area) {
            return ZPSE_BAD_SEGMENT_DIRECTORY;
        }
    }

    /* Per-segment decoded sizes from segment headers. The directory entries
     * were read and validated once above and are reused here and in PHASE3;
     * a page has one segment, so re-reading the same eight bytes three times
     * was three loads of the same cache line for one directory entry. */
    {
        uint64_t sum = 0;
        for (s = 0; s < scount; s++) {
            ZpseSegmentHeader sh;
            uint32_t off = seg_off[s];
            uint32_t sz = seg_sz[s];
            rc = zpse_segment_header_parse(&sh, payloads + off, sz);
            if (rc != ZPSE_OK) {
                return rc;
            }
            if (sh.segment_decoded_size == 0) {
                return ZPSE_BAD_SEGMENT_DIRECTORY;
            }
            seg_lens[s] = sh.segment_decoded_size;
            /* A page has one segment; keep its parsed header so PHASE3 can
             * skip parsing the same eight bytes a second time. */
            if (scount == 1u) {
                sh0 = sh;
            }
            sum += (uint64_t)sh.segment_decoded_size;
        }
        if (sum != (uint64_t)n) {
            return ZPSE_BAD_SEGMENT_DIRECTORY;
        }
    }

    rc = zpse_transform_validate_sizes(h.transform_id, h.segment_count, n,
                                       seg_lens);
    if (rc != ZPSE_OK) {
        return rc;
    }

    /* A single-segment ZPSE_T_RAW_BYTES block has an inverse transform that
     * is exactly `memcpy(out, segs[0], n)`: zpse_transform_meta_parse only
     * accepts that id with segment_count == 1, an empty metadata region and
     * width_W == 1, and the delta/xor wrap is keyed on ids 1, 2, 7..10 and
     * 11-with-flag-bit-0, so id 0 is neither. The plane for such a block is
     * therefore the caller's buffer itself: decoding the segment straight
     * into it produces the same bytes the copy would have produced, and
     * PHASE4 then becomes a copy of a buffer onto itself, which the
     * transform's own overlap guard refuses. So for this shape the copy is
     * not just redundant, it is not expressible at all - and it is the
     * shape the encoder's identity trial produces, i.e. the common case for
     * a smooth or constant block. Nothing the inverse would have validated
     * is skipped: this function already established that meta, segs and out
     * are non-null, that n is in [ZPSE_MIN_BLOCK_SIZE,
     * ZPSE_MAX_BLOCK_SIZE], that the width agrees with the parse, and that
     * zpse_transform_validate_sizes() passes for exactly these arguments;
     * the overlap guard exists to protect the copy that no longer happens. */
    identity_shape = (h.transform_id == (uint8_t)ZPSE_T_RAW_BYTES &&
                      scount == 1u) ? 1 : 0;

    /* PHASE3: decode segments into recon planes (or straight into the
     * caller's buffer for the identity shape above). */
    {
        ZpseLaneState lanes;
        /* A 16-wide interleave transposes sixteen planes whose bases are
         * then `seg_len` apart -- for a 64 KiB block exactly 4096, and the
         * L1 set index is (address / 64) mod 64, so a group's sixteen
         * lines all land on one set; that is why the inverse stages a
         * copy of every plane before transposing it. Laying the planes
         * out 4160 apart (one extra cache line per plane) walks the set
         * index along by one per plane and keeps every plane aligned, and
         * then the inverse can read them in place: measured 1.28-1.46x
         * faster than the staged form (see zpse_t16x16_inverse_direct).
         * Only id 6 reaches the w == 16 inverse path, so only it is laid
         * out this way; the planes go eight and eight into the two 64 KiB
         * decoder regions, which needs 7 * 4160 + 4096 = 33216 bytes of
         * each. */
        int skew_layout = (identity_shape == 0 &&
                           h.transform_id == (uint8_t)ZPSE_T_TRANSPOSE_16 &&
                           scount == 16u) ? 1 : 0;
        uint32_t skew_off = 0;

        /* Only the histories are live lane state now; the tables the stateful
         * predictors used to borrow are gone, and the segment decoder seeds
         * its own history from each tile's initial literal. */
        zpse_lane_reset_hist(&lanes, 0u);
        cum = 0;
        for (s = 0; s < scount; s++) {
            uint32_t off = seg_off[s];
            uint32_t sz = seg_sz[s];
            uint8_t *plane = d->recon + cum;
            uint32_t got = 0;

            if (identity_shape != 0) {
                plane = output;
            } else if (skew_layout != 0) {
                uint8_t *region = (s < 8) ? d->recon : d->out_stage;

                if (s == 8u) {
                    skew_off = 0;
                }
                plane = region + skew_off;
                skew_off += seg_lens[s] + ZPSE_D_PLANE_SKEW;
            }
            /* No per-segment lane reset: seg_decode_body seeds the first
             * tile's history from this segment's own header
             * initial_literal, so that is the entry state. last[] and
             * second_last[] are the only live lane state left -- 16 bytes,
             * no tables to seed -- and lanes already points at the
             * decoder's state, so a reset here would only repeat that work. */
            if ((uint64_t)cum + (uint64_t)seg_lens[s] > ZPSE_MAX_BLOCK_SIZE) {
                return ZPSE_BAD_SEGMENT_DIRECTORY;
            }
            if (scount == 1u) {
                rc = zpse_segment_decode_preparsed(d, &lanes, &sh0,
                                                   payloads + off, sz,
                                                   plane, seg_lens[s], &got);
            } else {
                rc = zpse_segment_decode(d, &lanes, payloads + off, sz,
                                         plane, seg_lens[s], &got);
            }
            if (rc != ZPSE_OK) {
                return rc;
            }
            if (got != seg_lens[s]) {
                return ZPSE_BAD_SEGMENT_DIRECTORY;
            }
            csegs[s] = plane;
            cum += seg_lens[s];
        }
        if (cum != n) {
            return ZPSE_BAD_SEGMENT_DIRECTORY;
        }
    }

    /* PHASE4: inverse transform straight into the caller's buffer (the
     * segment planes live in d->recon, so there is no overlap and the
     * 64 KiB staging copy is unnecessary). The identity shape above has no
     * transform left to apply - its plane IS the output - so this phase is
     * skipped for it, which is also the only way to express it (see the
     * identity_shape comment). */
    if (identity_shape == 0) {
        /* Sizes and plane layout were already validated above, so the
         * duplicate validation (including the O(w^2) overlap walk) is skipped
         * here. The page path is bit-identical and activates for n <= 4096. */
        rc = zpse_transform_inverse_trusted(&meta, h.segment_count, csegs,
                                            seg_lens, output, n);
        if (rc != ZPSE_OK) {
            return rc;
        }
    }

    /* PHASE5: payload CRC iff the block flag tags it.
     *
     * ZPSE_SKIP_CRC_VERIFY is a measurement arm, never a shipped
     * configuration: it drops the pass and the compare, so a bench can
     * price the checksum's share of decode. That share is the ceiling on
     * any fusion of the CRC into the inverse -- a fused loop still pays
     * one `crc32` step per eight bytes, so it can only save the second
     * read, which is a fraction of what this arm removes. */
#ifndef ZPSE_SKIP_CRC_VERIFY
#define ZPSE_SKIP_CRC_VERIFY 0
#endif
    if ((h.block_flags & ZPSE_B_CRC) != 0) {
#if !ZPSE_SKIP_CRC_VERIFY
        uint32_t calc;

        /* A one-value plane is the dense constant form (Sec. 6.3), and
         * only the identity shape reaches here with the decoded plane
         * already in the caller's buffer, so d->seg_const describes
         * exactly the bytes output[0..n). Its CRC is O(log n) matrix
         * applies instead of a 64 KiB pass over the buffer the decoder
         * just wrote -- and that pass is most of the time a constant
         * block takes to decode. Nothing is skipped on a claim the
         * decoder made up: seg_const is only ever set by a record whose
         * every byte the decoder itself wrote as one value, and the
         * value compared below is still the full 64 KiB's CRC. */
        if (identity_shape != 0 && d->seg_const != 0u && n != 0u) {
            calc = zpse_crc32c_repeat(d->seg_const_byte, (size_t)n);
        } else {
            calc = zpse_crc32c(output, n);
        }
        if (calc != h.decoded_crc32c) {
            return ZPSE_BAD_PAYLOAD_CRC;
        }
#endif /* !ZPSE_SKIP_CRC_VERIFY */
    }

    if (out_len_out != 0) {
        *out_len_out = n;
    }
    return (int)n;
}

int zpse_block_decode(ZpseDecoder *d, const uint8_t *in, uint32_t in_len,
                      uint8_t *output, uint32_t out_cap,
                      uint32_t *out_len_out)
{
    return zpse_block_decode_ex(d, in, in_len, output, out_cap, out_len_out,
                                0u);
}

/* Encode one transform trial into dst[0..cap). Returns ZPSE_OK with
 * *total_out set, or a negative status / skip code. *budget_hit is set
 * when the trial was stopped by the RAW-envelope budget below rather
 * than by a real error, because the caller has to count that case
 * exactly like a completed trial that failed to beat RAW.
 *
 * `have_best` / `best_total` carry the sweep's running winner: this trial
 * is only ever chosen on a STRICTLY smaller total, so once the bytes it
 * has produced already reach `best_total` it has lost and the rest of the
 * payload is work for nothing. That bound is the tighter of the two
 * whenever a previous trial completed, and stopping on it is exact -- a
 * trial's total only grows as its remaining segments are written, so
 * `at >= best_total` (or a segment that crosses it) is a proof, not an
 * estimate. The raw-envelope case keeps its own meaning: it is the only
 * one that sets `*budget_hit`, because it is the only one that means
 * "reached the size where RAW wins" for the caller's incompressible
 * heuristic. */
/* The nine predictors the segment encoder's prescreen ranks, spelled out
 * here for the sample below (the same list, and the same reason for
 * spelling it out, as seg_screen_pred in src/segment.c: appending to the
 * enum cannot silently reorder it). ids 3, 4 and 9 are the sentinels that
 * prescreen leaves at 0xFFFF. */
static const uint8_t zpse_block_screen[9] = { 0u, 1u, 2u, 5u, 6u, 7u, 8u,
                                              10u, 11u };

/* Whole-block noise sample: is EVERY one of sixteen tiles spread evenly
 * across the block's transformed planes a tile the segment encoder itself
 * would call near-random -- one where all nine screened predictors miss at
 * least 7/8 of the positions (the same threshold the segment encoder's own
 * single-candidate rule uses)?
 *
 * This is the second half of the trial-level flat abort's evidence, and it
 * is what keeps that heuristic off a block whose FRONT is noise and whose
 * rest compresses: the leading-run test (seg_flat_abort) only ever sees
 * the front, so it cannot tell such a block from a block that is noise
 * everywhere, and the whole-block sample sees the rest. Sixteen 256-byte
 * tiles out of a block is a sample, not a proof -- but the points are the
 * odd thirty-seconds of the block (1/32, 3/32, ... 31/32), a sixteenth of
 * the block apart, so a compressible stretch of the block wider than that
 * contains one of them whatever the segment shape, and each point is judged
 * by the encoder's own near-random rule on the very plane the trial is
 * about to write. The abort further requires a leading run of unpredicted
 * tiles. Returns 1 only when the whole block reads as noise at every sample
 * point. */
static int zpse_block_is_noise(const uint8_t *const planes[16], uint8_t sc,
                               const uint32_t lens[16], uint32_t n,
                               unsigned points)
{
    unsigned i;

    for (i = 0; i < points; i++) {
        uint64_t target =
            ((uint64_t)n * (uint64_t)(2u * i + 1u)) / (2u * (uint64_t)points);
        uint64_t seen = 0;
        uint32_t len, off;
        const uint8_t *pl;
        uint16_t nt;
        uint16_t ks[12];
        uint16_t best;
        unsigned s, j;

        for (s = 0; s + 1u < sc; s++) {
            if (target < seen + lens[s])
                break;
            seen += lens[s];
        }
        pl = planes[s];
        len = lens[s];
        if (pl == 0 || len < 64u)
            continue;
        off = (uint32_t)(target - seen);
        off &= ~255u;
        if (off + 64u > len)
            off = (len >= 320u) ? (len - 320u) : 0u;
        nt = (uint16_t)((len - off > 256u) ? 256u : (len - off));
        if (nt < 64u)
            continue;
        zpse_segment_prescreen(pl + off, nt, pl, off, pl[0], ks, 0u);
        best = ks[zpse_block_screen[0]];
        for (j = 1u; j < 9u; j++) {
            if (ks[zpse_block_screen[j]] < best)
                best = ks[zpse_block_screen[j]];
        }
        if ((uint32_t)best * 8u < (uint32_t)nt * 7u)
            return 0; /* a tile the encoder can predict: not a noise block */
    }
    return 1;
}

static int zpse_block_try_one(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                              uint32_t in_crc,
                              uint8_t tid, uint8_t sc,
                              const uint8_t *meta_bytes, uint16_t meta_len,
                              int is_last, uint32_t seq,
                              uint8_t *dst, uint32_t cap, uint32_t *total_out,
                              int *budget_hit, int have_best,
                              uint32_t best_total, uint32_t flat_abort_min,
                              unsigned noise_points)
{
    /* A trial is only ever chosen when its total is below the RAW
     * envelope (32 + N), so it cannot matter once it has reached it:
     * every payload byte written is committed and later segments only add
     * to the total. Handing the segment encoder that remaining budget
     * lets it stop instead of finishing a payload that is thrown away.
     * The envelope is also a strict upper bound on the budget, and the
     * cap the caller passes is the candidate buffer, which is larger than
     * any envelope a valid block can have, so an overflow with an active
     * budget is always the budget -- never the cap. */
    ZpseTransformMeta tm;
    uint8_t w = 0;
    uint32_t lens[16];
    uint32_t seg_sizes[16];
    uint8_t *planes[16];
    uint64_t base, cum;
    uint32_t s, total;
    ZpseBlockHeader h;
    ZpseLaneState lanes;
    uint8_t flags;
    int is_noise;
    int rc;

    if (budget_hit != 0) {
        *budget_hit = 0;
    }
    /* The flat abort's coverage counter and the flag that ends its run span
     * this trial's segments, so they are reset here rather than inside
     * zpse_segment_encode. */
    e->flat_covered = 0u;
    e->flat_paid = 0u;
    rc = zpse_transform_width(tid, sc, &w);
    if (rc != ZPSE_OK) {
        return rc;
    }
    rc = zpse_segment_lengths(n, w, lens);
    if (rc != ZPSE_OK) {
        return rc;
    }
    /* Clamping: N < W forbidden; empty segments forbidden. */
    for (s = 0; s < sc; s++) {
        if (lens[s] == 0) {
            return ZPSE_BAD_TRANSFORM;
        }
    }
    rc = zpse_transform_meta_parse(&tm, tid, meta_bytes, meta_len, sc);
    if (rc != ZPSE_OK) {
        return rc;
    }
    {
        uint64_t sum = 0;
        uint32_t off = 0;
        for (s = 0; s < sc; s++) {
            planes[s] = e->tstage + off;
            if ((uint64_t)off + (uint64_t)lens[s] > ZPSE_E_SZ_TSTAGE) {
                return ZPSE_OUTPUT_OVERFLOW;
            }
            off += lens[s];
            sum += (uint64_t)lens[s];
        }
        if (sum != (uint64_t)n) {
            return ZPSE_BAD_TRANSFORM;
        }
    }
    /* A width-1 split with neither DELTA8 nor XOR8 is the identity: the
     * segment encoder only reads its plane, so the input itself can serve
     * as segment 0 and a 64 KiB staging copy disappears from every block
     * that tries the identity (which is all of them). The overlap
     * rejection inside zpse_transform_forward exists because it writes the
     * plane; not copying at all is what makes it unnecessary here. */
    if (tid == 0u && sc == 1u) {
        planes[0] = (uint8_t *)(uintptr_t)in;
    } else {
        rc = zpse_transform_forward_planes(&tm, sc, in, n, planes, lens);
        if (rc != ZPSE_OK) {
            return rc;
        }
    }

    base = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)meta_len +
           (uint64_t)sc * (uint64_t)ZPSE_SEG_DIR_ENTRY_SIZE;
    if (base > (uint64_t)cap) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    /* Only the histories are live lane state now; the segment encoder seeds
     * its own from each tile's initial literal. */
    zpse_lane_reset_hist(&lanes, 0u);
    /* Sixteen prescreens against the tiles this trial is about to write,
     * asked for only while the flat abort could still fire: the abort is
     * armed per segment only while beating the RAW envelope is the trial's
     * only way to win, and a winner already smaller than RAW means no
     * segment can be armed. */
    is_noise = (flat_abort_min != 0u &&
                (!have_best ||
                 (uint64_t)best_total >=
                     (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n))
                   ? (noise_points == 0u
                          ? 1
                          : zpse_block_is_noise(
                                (const uint8_t *const *)planes, sc, lens, n,
                                noise_points))
                   : 0;
    cum = 0;
    for (s = 0; s < sc; s++) {
        uint8_t lit = lens[s] > 0 ? planes[s][0] : 0;
        uint64_t at = base + cum;
        uint64_t raw = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n;
        uint32_t room;
        uint32_t budget;
        int budget_is_raw = 1;
        int wlen;
        if (at > (uint64_t)cap) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        /* Already at or past the running best before this segment writes
         * a byte: every remaining payload byte only adds to the total, so
         * this trial cannot be strictly smaller than the winner. A loss,
         * not a budget hit -- nothing here says anything about RAW. */
        if (have_best && (uint64_t)best_total <= at) {
            return ZPSE_OUTPUT_OVERFLOW;
        }
        room = (uint32_t)((uint64_t)cap - at);
        budget = (raw > at) ? (uint32_t)(raw - at) : 0u;
        if (have_best && (uint64_t)best_total < raw) {
            /* RAW is out of the running (the winner is already smaller
             * than it), so the tighter cap is the winner itself, and it
             * is the one that counts now -- a segment stopped by it is a
             * known loss, never an incompressible-block signal. */
            uint32_t bb = (uint32_t)((uint64_t)best_total - at);
            if (budget == 0u || bb < budget) {
                budget = bb;
                budget_is_raw = 0;
            }
        }
        /* The flat abort is a *noise* signal: it only means something
         * while the trial's only way to win is to beat the RAW envelope.
         * Once a winner exists the budget above is a proof of the same
         * thing, so the heuristic is switched off and only exact bounds
         * stop the trial. The whole-block sample is asked first because it
         * is a property of the trial's transformed planes, not of the
         * segment being written, so a block that is predictable anywhere
         * keeps the trial it has always had. */
        wlen = zpse_segment_encode(e, &lanes, planes[s], lens[s], lit,
                                   dst + at, room, budget,
                                   (budget_is_raw != 0 && is_noise != 0)
                                       ? flat_abort_min
                                       : 0u);
        if (wlen < 0) {
            if (wlen == ZPSE_OUTPUT_OVERFLOW && budget != 0u &&
                budget_is_raw != 0 && budget_hit != 0) {
                *budget_hit = 1;
            }
            return wlen;
        }
        if (wlen < (int)ZPSE_SEGMENT_HEADER_SIZE) {
            return ZPSE_BAD_TILE_SIZE;
        }
        seg_sizes[s] = (uint32_t)wlen;
        cum += (uint64_t)wlen;
    }
    if (base + cum > (uint64_t)cap) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    total = (uint32_t)(base + cum);

    if (meta_len > 0) {
        memcpy(dst + ZPSE_BLOCK_HEADER_SIZE, meta_bytes, meta_len);
    }
    for (s = 0; s < sc; s++) {
        uint32_t off = 0;
        uint8_t *ent =
            dst + ZPSE_BLOCK_HEADER_SIZE + meta_len + (size_t)s * 8;
        uint32_t k;

        for (k = 0; k < s; k++) {
            off += seg_sizes[k];
        }
        zpse_store_u32le(ent, off);
        zpse_store_u32le(ent + 4, seg_sizes[s]);
    }

    flags = (uint8_t)(ZPSE_B_RESET |
                      (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                           ? ZPSE_B_CRC : 0u));
    if (meta_len > 0) {
        flags |= ZPSE_B_HAS_META;
    }
    if (is_last) {
        flags |= ZPSE_B_LAST;
    }
    /* The decoded CRC covers the uncompressed input, so it is a property
     * of (in, n) and not of the trial -- the caller computes it once and
     * every candidate header reuses it. */
    zpse_block_fill_header(&h, flags, n, total, tid, sc, meta_len, seq,
                           in_crc);
    zpse_block_header_serialize(dst, &h);
    *total_out = total;
    return ZPSE_OK;
}

/* Divisor of the trial-level noise probe's threshold: see flat_min in
 * zpse_block_encode and seg_flat_abort in segment.c.
 *
 * 64, not 16. The probe exists only to stop a trial that can no longer
 * beat the RAW envelope, and a noise tile costs about 370 ns to encode
 * (versus 75 ns for one that predicts), so the old 4 KiB threshold spent
 * 16 of them -- 6 us of a 26 us encode -- before giving up. A quarter of
 * that decides on the same evidence; the floor in zpse_block_encode means
 * 64 and anything larger are the same threshold, so this is the smallest
 * the divisor can usefully be. The noise row goes 3.01 -> 3.44 GB/s and
 * nothing else moves: tests/test_fuzz.c prints byte-identical sizes for
 * all 2,718 cases and tests/test_sizes.c still reports grew=0 over 768. */
#ifndef ZPSE_FLAT_MIN_DIV
#define ZPSE_FLAT_MIN_DIV 64u
#endif

/* Smallest size a trial with this (transform, segment count, metadata)
 * shape can possibly produce: the envelope (block header, metadata, the
 * segment directory, each segment's priced header -- zpse_segment_price, as
 * the prices the floor is compared against charge it) plus 2 length-prefix
 * bytes and the 8-byte tile header for every tile. Used to stop the
 * transform sweep as soon as nothing left can beat the running best. */
static uint32_t zpse_block_floor(uint32_t n, uint8_t tid, uint8_t sc,
                                 uint16_t meta_len, uint32_t price)
{
    uint32_t lens[16];
    uint32_t records_bytes = 0u;
    uint8_t w = 0;
    uint8_t s;

    if (zpse_transform_width(tid, sc, &w) != ZPSE_OK) {
        return UINT32_MAX;
    }
    if (zpse_segment_lengths(n, w, lens) != ZPSE_OK) {
        return UINT32_MAX;
    }
    for (s = 0; s < sc; s++) {
        uint32_t ch = (lens[s] + ZPSE_DENSE_MAX_LEN - 1u) /
                      ZPSE_DENSE_MAX_LEN;
        uint32_t tile_cost = 10u * ((lens[s] + 255u) / 256u);
        uint32_t dense_cost = ZPSE_DENSE_HEADER_SIZE * ch;

        /* The two forms have very different floors -- ten bytes per
         * 256-byte tile, twelve per 4 KiB dense record -- and a constant
         * plane is coded at the second rate (zpse_segment_encode emits it
         * that way), so a candidate's bound is the smaller of the two.
         * Anything larger would prune a candidate whose plane happens to
         * be constant, which is a real shape (the DELTA8 plane of a
         * strided block). */
        records_bytes += (tile_cost < dense_cost ? tile_cost : dense_cost);
    }
    return (uint32_t)ZPSE_BLOCK_HEADER_SIZE + (uint32_t)meta_len +
           (uint32_t)sc * ((uint32_t)ZPSE_SEG_DIR_ENTRY_SIZE + price) +
           records_bytes;
}

/* ------------------------------------------------------------------ *
 * Candidate attempt order.
 *
 * The sweep's result does not depend on the order the candidates are
 * attempted in, which makes that order a free performance knob:
 *
 *   - the winner is the smallest total with ties going to the earlier
 *     candidate in the LIST (index) order, which is applied explicitly at
 *     the comparison below;
 *   - the running-winner bound hands a later candidate the tightest size
 *     it must beat, so attempting the likely winner first is what makes
 *     the losers stop early;
 *   - the floor-based exit is evaluated over the candidates NOT yet
 *     attempted, and the two-failure exit needs two attempts, so the two
 *     exits fire at the same points their candidate sets allow.
 *
 * Only the two width-1 candidates can be swapped, and that is what makes
 * the argument tight: the attempted SET is {0, 1, ...} in both orders, so
 * the two-failure exit still costs exactly two attempts over the same two
 * candidates, while the floor check after the first attempt sees a
 * superset of what it saw before (the identity's floor is one byte below
 * DELTA8's, so its threshold is never the higher one and the sweep can
 * only stop later, never earlier). No candidate is ever skipped that the
 * other order would have attempted, and no tie can resolve differently.
 * Defaulting to the identity first is therefore safe, and swapping is safe
 * whenever the sample below says DELTA8's plane is much more predictable
 * than the raw bytes -- which is the case for ramps and strides, where the
 * identity trial is a full block of near-all-exception tiles -- the most
 * expensive work the tile encoder does -- that loses to DELTA8 by a
 * factor of four or more.
 *
 * The estimate is a sample, not a bound: it is used only to pick the
 * order, so a bad estimate costs the work the other order would have
 * saved, never a different block. A bound strong enough to PRUNE a
 * candidate cannot be built this cheaply -- a tile's floor given its
 * exception count k is roughly 8 + (mask bits) + k/8 bytes, about 40
 * bytes even for k near Nt, against the ~200 bytes such a tile actually
 * packs to, so no sum of per-tile floors reaches the sizes this format
 * produces. Ordering is the only use of the prescreen that pays.
 *
 * The sample: six tiles spread across the block, prescreened on both
 * planes. The identity's plane is the input; DELTA8's is
 * in[q] - in[q - 1] (its first byte is 0 because its seed is in[0]), so a
 * sample tile at plane offset p needs only in[p - 17, p + 256) and both
 * planes are prescreened with the same call the segment encoder makes,
 * with seg_pos == 16 so a locally-built 16-byte head stands in for the
 * plane bytes before the tile.
 * ------------------------------------------------------------------ */
#define ZPSE_ORDER_SAMPLES 6
#define ZPSE_ORDER_MIN_TILES 16u
/* Does the attempt-order sample run on a PAGE?
 *
 * A page is sixteen tiles -- exactly the size the sample's own gate admits --
 * and its sweep attempts 1.11 trials, so what the estimate buys there is the
 * order of one or two attempts the price has already ranked. On a 64 KiB block
 * the same estimate orders up to ten attempts and several of them run, which is
 * the case it was built for.
 *
 * Measured at 4 KiB level 6 over four interleaved rounds on one pinned core:
 * the dumps read 166.5 against 162.5 MB/s by mean and Silesia 161.7 against
 * 160.2, and BOTH corpora are byte-identical (145 419 823 and 116 259 698
 * either way) -- the price has already ranked the field, so the estimate only
 * re-orders attempts that end up running once. 0 runs it everywhere, as it
 * always has. */
#ifndef ZPSE_ORDER_PAGE
#define ZPSE_ORDER_PAGE 1
#endif

static int zpse_block_delta_trial_first(const uint8_t *in, uint32_t n)
{
    uint32_t tc = n / (uint32_t)ZPSE_TILE_SIZE;
    uint32_t est_id = 0u;
    uint32_t est_d1 = 0u;
    unsigned ns;
    uint8_t loc[16u + ZPSE_TILE_SIZE];
    unsigned i;
    unsigned j;
    unsigned c;

    /* The estimate costs a fixed handful of prescreens while the work it
     * can save is proportional to the block, so it only runs once a block
     * has enough tiles to lose: below ZPSE_ORDER_MIN_TILES the identity
     * trial cannot waste more than its own few tiles and the sweep's floor
     * test usually ends it after one trial anyway. */
    if (tc < ZPSE_ORDER_MIN_TILES) {
        return 0;
    }
#if ZPSE_ORDER_PAGE
    if (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX) {
        return 0;
    }
#endif
    ns = (tc >= 8u * ZPSE_ORDER_MIN_TILES) ? (unsigned)ZPSE_ORDER_SAMPLES : 3u;

    for (i = 0; i < ns; i++) {
        uint32_t t = ((uint32_t)(i + 1u) * tc) / (uint32_t)(ns + 1u);
        uint32_t p = t * (uint32_t)ZPSE_TILE_SIZE;
        uint16_t ks[12];
        uint32_t kmin;

        /* Interior tiles only: the head has to be real plane bytes, and
         * the DELTA8 window reaches one byte further back still. */
        if (p < 17u || (uint64_t)p + (uint64_t)ZPSE_TILE_SIZE > (uint64_t)n) {
            continue;
        }

        /* Identity plane. `in[0]` is the segment's initial literal. */
        memcpy(loc, in + p - 16u, 16u);
        memcpy(loc + 16u, in + p, (size_t)ZPSE_TILE_SIZE);
        zpse_segment_prescreen(loc + 16u, (uint16_t)ZPSE_TILE_SIZE, loc, 16u,
                               in[0], ks, 0u);
        kmin = ks[0];
        for (c = 1u; c < 11u; c++) {
            if (ks[c] < kmin) {
                kmin = ks[c];
            }
        }
        est_id += kmin;
        /* A raw plane that predicts every sampled tile so far is about to
         * reach the global floor and end the sweep -- and it is the
         * cheap kind of block (constant, all-hit), where the estimate's
         * own cost would be the largest part of the encode. Stop before
         * paying for the other plane, and report the third answer: this
         * is not just "the identity goes first", it is "the identity is
         * about to END the sweep", so the caller can also skip the proxy
         * that would otherwise price every candidate first. */
        if (est_id == 0u) {
            return 2;
        }

        /* DELTA8 plane over the same window, one byte behind itself. */
        for (j = 0; j < 16u + (unsigned)ZPSE_TILE_SIZE; j++) {
            uint32_t q = p - 16u + j;

            loc[j] = (uint8_t)(in[q] - in[q - 1u]);
        }
        zpse_segment_prescreen(loc + 16u, (uint16_t)ZPSE_TILE_SIZE, loc, 16u,
                               0u, ks, 0u);
        kmin = ks[0];
        for (c = 1u; c < 11u; c++) {
            if (ks[c] < kmin) {
                kmin = ks[c];
            }
        }
        est_d1 += kmin;
    }

    /* Needs a clear margin, not a hair: the two candidates can be close
     * (low-entropy data favours the identity while both planes look
     * equally bad to a prescreen), and a wrong swap costs the trial the
     * other order would have stopped early. */
    return (est_d1 * 4u <= est_id) ? 1 : 0;
}


/* Tiles an exact probe prices per plane: enough to tell a shape that packs
 * (a nibble alphabet, a mask-friendly record) from one that cannot, at a
 * fixed cost. */
#ifndef ZPSE_PROBE_TILES
#define ZPSE_PROBE_TILES 8u
#endif

/* Tiles the decision probe prices per plane of a SINGLE-segment candidate
 * (see the sweep decision in zpse_block_encode): paid once per candidate
 * against a full trial's 256 tiles. Four is measured against sixteen: the
 * 768-case clen corpus came out byte-identical either way, the 2,718-case
 * fuzz corpus came out 34,174 bytes SMALLER at four (a coarse estimate ties
 * more often, and a tie goes to the earlier candidate -- the identity, which
 * is the better prior on the mixed blocks where the sample is noisiest), and
 * it took 22 us off a 64 KiB block's encode that the probe was spending to
 * price candidates it then rejected.
 *
 * A candidate with more than one plane is priced one tile per plane instead,
 * whatever this says: the planes of a transpose are not interchangeable, so
 * there is no fewer-planes estimate of it that is not biased (see
 * zpse_block_exact_sum). */
#ifndef ZPSE_DECIDE_TILES
#define ZPSE_DECIDE_TILES 4u
#endif

/* ...and how many a PAGE prices, which is two rather than four.
 *
 * A page's single-plane candidates are priced on the whole of it, so four tiles
 * there is a quarter of the page per candidate against a 64 KiB block's
 * sixteenth. Two ranks the field the price can already order -- measured at
 * 4 KiB level 6 over four interleaved rounds on one pinned core: 88 600 950
 * bytes over the dumps against 88 596 075 (+0.0023 of a point) and 116 268 437
 * over Silesia against 116 259 698 (+0.0041), for 166.6 against 159.8 MB/s by
 * mean on the dumps and 167.3 against 161.7 on Silesia -- 4%. That is four
 * thousandths of a point for four percent, which is the cheapest trade any knob
 * on this price has offered. Three is a wash on both axes (1 296 bytes fewer on
 * the dumps, 36 661 on Silesia, rate inside the band) and is not taken.
 *
 * All of those are mixture averages, and this knob is shape-local: on the probe
 * corpus each shape is one page in twenty, so a page it re-ranks shows a
 * twentieth of the effect. Measured with 64 MiB of a single shape (pages from
 * `corpus_write --probe-shape=N`, level 6, three rounds), four tiles save
 * 713 284 bytes on the page whose quarters alternate a 2-bit alphabet and
 * 16-byte records -- 1.06 points of that workload -- for 9.6% of its rate, and
 * leave the noisy-tile ramp byte-identical, add 0.003 of a point on the
 * quarters mirror and take 898 bytes off the ramp with holes. So a caller whose
 * pages are ALL one ambiguous shape can have a point from it, and a mix cannot:
 * the probe corpus pays 0.051 of a point for 14% of the rate. That is why four
 * tiles stay off the default and this knob stays an internal one rather than a
 * public setting -- one synthetic page shape, deliberately built so the tile
 * count re-ranks it, is not a corpus to recommend a build flag on.
 *
 * 0 leaves a page on ZPSE_DECIDE_TILES. Above a page the value is untouched: */
#ifndef ZPSE_DECIDE_PAGE_TILES
#define ZPSE_DECIDE_PAGE_TILES 2u
#endif

/* Full block encodes the sweep may spend on a block whose own probe says it
 * does not compress at all -- the winner's price over the 16 KiB window is at
 * or above the same window written raw. Such a block falls back to RAW unless
 * a shape wins on a part of it the sample did not cover, which is the one
 * thing a sample cannot rule out, so the sweep still spends this many; it just
 * does not spend one encode per survivor on a block that is very likely to be
 * incompressible. Blocks the probe says DO compress are swept in full, which
 * is what keeps the ratio: the 3-value-alphabet case of the ratio corpus is
 * decided by its fifth survivor (16696 bytes against 16864 for the first
 * three), and "zero with one byte every 256" by a shape the probe ranks
 * fourth. */
#ifndef ZPSE_PROBE_WINDOW
#define ZPSE_PROBE_WINDOW 12288u
#endif

/* Tiles a SHORT block is priced on.
 *
 * The window above is a prefix of the block, so a block shorter than it is
 * priced on the whole of itself -- and the price of a window is paid once per
 * candidate, ten of them on a full list. On a 4 KiB page that is ten passes
 * over the page before the first trial runs, which is most of what a
 * page-at-a-time encode spends (see ZPSE_PRICE_MIN_N).
 *
 * A price only has to RANK the candidates, and the ranking of a page's shapes
 * is what its first half says as well as what all of it says. So a short
 * block is priced on its first ZPSE_PROBE_SHORT_TILES tiles: still a prefix
 * (a transform needs one), still an exact price of that prefix, and every
 * comparison below -- the candidate floor, the raw-window test, the tail
 * check -- keeps reading a price of `pn` bytes, which is what they are
 * written against.
 *
 * Eight tiles on a CHEAP RUNG, the whole block where every plane is priced,
 * and both halves are the measurement. On the cheap rungs the price IS the
 * decision -- one or two trials, no more -- so a wider price is neither free
 * nor safe: against eight tiles, the whole page costs levels 1 and 2 7.5 and
 * 3.4% of encode on the dumps and GROWS the mixed workload by 0.18 and 0.21%
 * of bytes. At BALANCED, where every plane is priced and the trials behind the
 * price can repair a coarser list, the same change wins on bytes and is
 * neutral to better on rate: three paired passes read the dumps at 88,111,603
 * against 88,253,809 bytes (41.32 against 41.38%) with the rate inside this
 * host's band, the mixed workload 0.09% smaller and 2.8% faster, and Silesia
 * 0.12% smaller at the same rate.
 *
 * The whole block is the natural end of the knob: `pn` is a prefix of the
 * block, so a page's price is then taken on the page itself and a larger
 * value changes nothing there. What it buys is a ranking on the whole of the
 * block -- eight tiles rank the candidates on what the first half of the page
 * says, and on these corpora the second half sometimes disagrees -- at the
 * cost of one price over at most 2 KiB more per candidate.
 *
 * It needs one thing to be honest at a page, and that is the identity rule in
 * the near-tie cut: on a 4-bit-random page the whole-page price puts a
 * transpose eight bytes ahead of the identity, and a cut that trusted it wrote
 * 21 bytes more than the half-page price did (the size gate's case 579). The
 * identity is never cut, so the price still orders the field and the trial
 * still decides.
 *
 * Four tiles loses on both axes where every plane is priced (0.08% more
 * bytes than eight on the dumps, 121.5 against 125.8 MB/s on the mixed
 * workload): it prices a width-16 candidate's planes at 64 bytes each, which
 * is where the ranking breaks. At the cheap rungs it is faster and dearer --
 * level 1 reads 313 against 281 MB/s on the dumps for 1.45 points of ratio
 * (60.05% against 58.61%), level 2 230 against 218 for 1.27 -- and six tiles
 * is worse than eight in BOTH directions (0.18, 0.06 and 0.35 points on the
 * dumps, mixed and Silesia at level 1, and slower: 273 against 281 MB/s on the
 * dumps, 278 against 287 on Silesia), so eight is the local optimum either way
 * rather than a guess.
 *
 * 0 restores the whole block. */
#ifndef ZPSE_PROBE_SHORT_TILES
#define ZPSE_PROBE_SHORT_TILES 8u
#endif

#ifndef ZPSE_PROBE_PAGE_TILES
/* The window an every-plane level prices a page on: 0 prices the whole page.
 *
 * The whole page is the shipped arm (see ZPSE_PROBE_SHORT_TILES): it ranks the
 * candidates on all 4 KiB and it also convicts more of them, which is why the
 * price count fell when it went in. This knob exists to measure the points
 * between the cheap rungs' eight tiles and the whole page, since the curve is
 * not monotone on both axes -- eight tiles is worse on bytes than sixteen at
 * BALANCED, while the cheap rungs prefer eight.
 *
 * Twelve was measured and the curve is monotone in the window at these levels,
 * which is what makes the whole page the right end of it: at level 6 twelve
 * tiles writes 41.1552% on the dumps against the whole page's 41.0067%, and
 * 55.1459 and 54.6719% on the mixed and Silesia corpora against 55.0729 and
 * 54.6093 -- more bytes on all three with no rate returned (141/124/136 against
 * 141/128/135 MB/s). 0 stays the shipped arm. */
#define ZPSE_PROBE_PAGE_TILES 0u
#endif

/* The tail price samples the block's own window, and shrinking it was measured
 * and refused: the price is a boolean ("does the back compress?"), so a shorter
 * sample looked like free staging, but it is not where a page's time goes. Eight
 * tiles writes 41.0070% against 41.0067% on the dumps, 55.0723 against 55.0729 on
 * the mixed workload and 54.6089 against 54.6093 on Silesia -- a thousandth of a
 * point either way, which is the flat detection flipping on a handful of pages --
 * and the rate does not move (136 against 135 MB/s on the dumps, 125 against 129
 * on the mixed workload). Four tiles is the same trade with less of both. */

/* Ordering the field behind the identity by a coarse price was measured and
 * refused, because it does not buy what it looks like it should. The price
 * counter reads the same either way -- 94,637 candidates for the dumps' 13,802
 * pages, 141,460 for the mixed workload's 39,063 -- since the sweep's cost is
 * the field it has to price, not the order it prices it in, and a screen is a
 * second price on every candidate. On a four-tile screen the dumps wrote 1.14%
 * fewer bytes (41.01% to 40.54%) but encoded at 105 against 139 MB/s, the mixed
 * workload grew 0.11 points, and the order it imposed also changed WHICH
 * candidates the flat cut prices, so it was not byte-neutral either. Two tiles
 * is the same trade with less of both. */

#ifndef ZPSE_CONVICT_SHIFT
/* The margin a running best convicts a later candidate by: a candidate priced
 * above ex[pick] + (ex[pick] >> SHIFT) cannot be a survivor and the probe
 * stops it at the plane that proved it. 4 is a sixteenth (6.25%). A larger
 * shift convicts sooner (fewer prices and trials, worse ratio); a smaller one
 * keeps more of the field.
 *
 * A page was measured with a wider field -- the trial it keeps is a quarter of
 * the block rather than a sixty-fourth, so a candidate the margin drops could
 * still have won -- and it is not worth what it costs. A twelfth (shift 3)
 * writes 0.021, 0.013 and 0.017% fewer bytes on the dumps, mixed and Silesia
 * corpora at level 6 for 0.7, 0.0 and 1.5% of encode rate; a quarter (shift 2)
 * writes 0.037, 0.033 and 0.038% fewer for 1.4, 1.6 and 2.2%. All three
 * corpora move the same way, so the rate is real rather than the host, and two
 * per cent of encode for four hundredths of a point is not the trade these
 * pages are held to. The 64 KiB blocks were byte-identical under a page gate,
 * so this is the page's own trade. Level 1 barely moves either way (0.0002
 * points, and the mixed workload 0.003 points LARGER at shift 2): the cheap
 * rungs act on the price, not on the field behind it.
 *
 * The other direction was measured and refused as well: shift 5, a 3.1%
 * margin instead of 6.25%, convicts earlier and prices less, and it writes
 * 5,421 bytes MORE on the dumps and 1,672 more on Silesia at level 6 with no
 * rate returned. A wider margin saves prices and costs bytes; a narrower one
 * costs the same bytes without saving them, which is this. */
#define ZPSE_CONVICT_SHIFT 4
#endif


#ifndef ZPSE_FLAT_TRIALS
/* The trial cap for a block the 12 KiB window price convicts of being flat.
 *
 * One was tried (and two) against this three. It is ratio-neutral to 0.004%
 * on Silesia at level 6 and 0.0002% at level 9, and it is invisible in time:
 * 293.9 against 298.0 MB/s at level 6 and 190.9 against 184.4 at level 9,
 * where a build that changes nothing occupies the same band. The blocks this
 * fires on are cheap either way; the cap stays where the size corpus was
 * tuned against it. */
/* Two, measured against three: the survivors of a flat page are the shapes the
 * price has already convicted, so the third trial is a full page encode spent
 * on one of them. At 4 KiB level 6 over four interleaved rounds it buys 210
 * bytes over Silesia's 212 MB (0.0099 of a point) and nothing at all on the
 * dumps, which are byte-identical, against 6.6% of Silesia's rate and 7.9% of
 * the dumps'. */
#define ZPSE_FLAT_TRIALS 2
#endif

/* The margin the final LZ screen asks for on a block shorter than
 * ZPSE_LZ_FINAL_SHORT_N, as a multiplier on the level's own (NUM/DEN).
 *
 * The screen samples a fixed 1 KiB, which on a page is a quarter of the
 * block, so it convicts more readily there than on a 64 KiB block -- but the
 * pass it guards is a full match-finder pass and the form wins only a
 * fraction of the pages it is offered. Asking for half again the level's own
 * margin on those blocks is worth 3.7% of the encode rate on the 4 KiB dump
 * total (82.9 -> 86.0 MB/s at the default level, six pairings of best-of-
 * eight on one pinned core) and 5.2% on Silesia's 4 KiB pages (87.8 -> 92.4),
 * for 0.05 and 0.01 points of ratio. 16 KiB blocks are byte-identical, which
 * is the boundary: only the pages pay this, and they are the ones the trade
 * is for.
 *
 * 4/4 is the screen as it was, every block asking for the level's margin. */
#ifndef ZPSE_LZ_FINAL_SHORT_N
#define ZPSE_LZ_FINAL_SHORT_N 16384u
#endif
#ifndef ZPSE_LZ_FINAL_SHORT_MARGIN_NUM
#define ZPSE_LZ_FINAL_SHORT_MARGIN_NUM 6u
#endif
#ifndef ZPSE_LZ_FINAL_SHORT_MARGIN_DEN
#define ZPSE_LZ_FINAL_SHORT_MARGIN_DEN 4u
#endif

/* The band, as a fraction of the best window price (in 1/2^N), inside which
 * the probe cannot separate two candidates, and how many of a field inside it
 * are still tried in full (see the near-tie prune in zpse_block_encode).
 *
 * The band is 1/64, i.e. 1.6%: a full trial costs about six times a window
 * price, so a field that prices within 1.6% of itself is a field where the
 * trials are being paid to break a tie the block total barely reflects.
 * dict_u8 is the measured case: its five survivors price within a fraction of
 * a percent of each other, the four extra trials buy 216 bytes in 4 MiB
 * (0.01%), and they are most of the block's encode time (342 -> 656 MB/s with
 * the field cut to its best two).
 *
 * The floor is the other half of the trade, and it is 64 KiB: only a block at
 * the format's full size prunes. A full block's trial is the most expensive
 * thing the sweep can do (256 tiles, up to sixteen planes), so that is where
 * the trade pays; every smaller block is settled exactly, because a short
 * block's trial is cheap enough that pruning cannot buy anything worth its
 * risk -- and the probe's sample covers a bigger fraction of it anyway, which
 * is the opposite of the case the prune exists for. */
#ifndef ZPSE_TIE_BAND_SHIFT
#define ZPSE_TIE_BAND_SHIFT 6u
#endif
#ifndef ZPSE_TIE_MIN_N
#define ZPSE_TIE_MIN_N 65536u
#endif
#ifndef ZPSE_TIE_KEEP
#define ZPSE_TIE_KEEP 2u
#endif

/* The same cut on a SHORT block, where the full-size gate does not reach:
 * keep ZPSE_TIE_SHORT_KEEP survivors on a block shorter than
 * ZPSE_TIE_SHORT_N.
 *
 * The gate above is 64 KiB because a full block's trial is the most
 * expensive thing the sweep can do, and a short block's trial is cheap --
 * but on a 4 KiB page the trial is HALF the block's encode cycles (measured
 * with the ZPSE_BLOCK_STATS build: 2.4-3.0 trials a page, 48-64% of the
 * cycles at the default level), so the same arithmetic applies for a
 * different reason: a page's trial is not cheap relative to the page.
 *
 * One, at the default level: 4 KiB 116.1 -> 122.6 MB/s (+5.6%, three
 * pairings) for 9 319 bytes over 88.8 MB (+0.011 points), and Silesia's pages
 * 121.1 -> 124.0 for 0.02 points. Two is worse on both axes (118.8 MB/s for
 * 5 608 bytes), which is the price sample being good enough to order the
 * field and not good enough to be free: keeping the runner-up costs a whole
 * page encode and recovers half of what the cut cost. Short of 8 KiB only --
 * 16 KiB and 64 KiB blocks are byte-identical to the gate's own behaviour.
 *
 * 2 was measured across the three page corpora at level 6 and refused: it
 * writes 40.9956, 55.0468 and 54.5652% against 41.0067, 55.0729 and 54.6093 --
 * 0.011, 0.026 and 0.044 points -- for 5, 2 and 2% of encode rate. The runner-up
 * is a real byte at a page, but it is not worth a twentieth of the rate, so a
 * page keeps the winner alone.
 *
 * 0 keeps the full-size gate as the only cut. */
#ifndef ZPSE_TIE_SHORT_KEEP
#define ZPSE_TIE_SHORT_KEEP 1u
#endif
#ifndef ZPSE_TIE_SHORT_N
#define ZPSE_TIE_SHORT_N 8192u
#endif
/* ...and how wide the field has to be before that cut is allowed at all.
 *
 * A field of four or fewer survivors is left whole on a full-size block,
 * because one more trial there is a rounding error and the third price is all
 * the evidence the block has. A page is the other case: the trial is half the
 * page's encode cycles and a survivor is what it costs, so the floor drops to
 * two -- any field the price separated gets one trial.
 *
 * Measured at 4 KiB on the dumps, best of five, five pairings: the default
 * level goes 122.8 -> 133.1 MB/s (+8.2%) for 77 782 bytes over 88.9 MB
 * (+0.036 points), and the trials a page run 1.65/2.97/1.66 -> 1.06/2.55/1.20.
 * The survivors it still tries are the ones a CONTENT rule kept rather than
 * the price -- c_heap keeps 3.0 a page and still spends 2.55 -- which is the
 * guard that has to stay, since those are the fields the price is known not
 * to describe. */
#ifndef ZPSE_TIE_MIN_FIELD
#define ZPSE_TIE_MIN_FIELD 4u
#endif
#ifndef ZPSE_TIE_SHORT_MIN_FIELD
#define ZPSE_TIE_SHORT_MIN_FIELD 2u
#endif
/* The block size the narrower floor above applies to, inclusive: the page this
 * whole file's page trades exist for.
 *
 * It is a size and not a level gate because of what the size corpus caught: a
 * 6 KiB case (kind 4) prices three shapes within a tenth of a percent of each
 * other, the dropped trial is the one that wins, and the block grows 24 bytes
 * -- at floor 2 and at floor 3 alike. A page's field is the case the cut is
 * for, because the survivor costs half the block's encode; a 6 KiB block is
 * neither a page nor a full block, so it keeps the full-size floor. */
#ifndef ZPSE_TIE_PAGE_N
#define ZPSE_TIE_PAGE_N 4096u
#endif

/* The other half of that trade, and the levels column `tie_field`: the band
 * above says "within 1.6% the price cannot order two shapes". Outside it the
 * price HAS ordered them, and a full trial is ~6x the price that already said
 * so. A level with tie_field set therefore cuts any field the four guards
 * below allow to two survivors, band or no band; a level without it keeps the
 * band, so the deepest rungs still encode every survivor their price could
 * not separate AND every one it could.
 *
 * The trade is measured, and it is the same four guards that make it safe:
 * with the field cut on every guarded block the whole 25-case columnar corpus
 * writes byte-identical ratios (8.29, 7.10, 1.97 ... 2.72, 7.03 on every row)
 * and tests/suites/test_sizes.c reports the same 768 cases, grew=0, the same
 * -127 349 bytes of shrinkage. What moves is encode time where the field is
 * wide: rle_lens 1.75 -> 2.4 GB/s and dict_u8 0.67 -> 0.80 GB/s, +3.4% and
 * +4.3% on the mean of two alternating runs. */

static int zpse_block_planes(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint8_t tid, uint8_t sc,
                             const uint8_t *meta_bytes, uint16_t meta_len,
                             uint8_t *planes[16], uint32_t lens[16]);

/* How far apart the planes of a multi-plane candidate are priced on a SHORT
 * block: 1 prices every plane, k prices every k-th one and charges the rest
 * the average of those (the machinery zpse_block_exact_sum carried for the
 * cheap levels before every-plane pricing replaced it).
 *
 * A 4 KiB page split sixteen ways is sixteen 256-byte planes, so "one tile in
 * every plane" is the whole page -- and every transform candidate is priced
 * that way, ten of them. That is what the probe spends its time on: on the
 * dumps at level 1 the pricing pass is 74% of the block's encode cycles and
 * it prices 9.5 candidates a page. On a 64 KiB block one tile per plane is
 * 4 KiB of the block, which is why the bias this average introduces is not
 * worth its cost there and why the knob has a size gate, not a level one.
 *
 * Measured on the three dumps at 4 KiB, best of five rounds, three pairings
 * of the two builds alternating on one pinned core: level 6 goes 82.6 ->
 * 122.6 MB/s and level 1 185.8 -> 267.6 (+48% and +44%) for 0.17 and 0.66
 * points of ratio; Silesia's 4 KiB pages 93.8 -> 124.0 at level 6 for 0.08
 * points. 16 KiB and 64 KiB blocks are byte-identical (the gate is below
 * them), and so is every whole-file stream, which uses 64 KiB blocks.
 *
 * Four planes, not every-plane, is the shipped answer above level 4: eight
 * planes buys +18% at level 6 for +0.06 points (a third of the speed for a
 * third of the cost), two planes +41% for +0.20, so four is where the trade
 * flattens out at the default level. The cheap rungs price six, for the
 * reason in ZPSE_PLANE_STRIDE_PLANES_FAST below.
 *
 * Three planes of a page's sixteen (step 6) was measured as the rung past four
 * and refused: it writes 79 117 bytes FEWER over the dumps -- the price is not
 * monotone in the planes it reads, and a coarser stride can even help a heap
 * page -- but it costs Silesia 9 240 bytes and reads 102.2 against 107.7 MB/s
 * there, while the dumps read 114.3 against 107.4. A setting that moves the
 * two corpora in opposite directions on both axes is not a corner.
 *
 * 0 disables it (every plane, everywhere), which is what this encoder did
 * before. */
#ifndef ZPSE_PLANE_STRIDE_N
#define ZPSE_PLANE_STRIDE_N 16384u
#endif
#ifndef ZPSE_PLANE_STRIDE_PLANES
#define ZPSE_PLANE_STRIDE_PLANES 4u
#endif
/* ...and how many a level up to ZPSE_PLANE_STRIDE_FAST_MAX prices.
 *
 * A cheap rung cannot repair a coarse ranking: it has one, two or three trials
 * and no more, so the price it acts on IS the answer. Four planes of a
 * sixteen-plane page are too coarse for that -- 0.66 points at level 1 and a
 * full 1.0% of ratio at level 2, both over the line every page trade here has
 * to stay under. Six planes is 0.30% at level 1 and 0.28-0.35% at levels 2-4,
 * and the speed it gives back is small against what the stride won: 249.8
 * against 197.5 MB/s at level 1, 200.0 against 156.4 at level 2, 177.8 against
 * 133.5 at level 4 (4 KiB, dumps, three pairings).
 *
 * So the gate is 4, and levels 5-9 keep four planes: the default level pays
 * 0.17 points for the stride where level 1 pays 0.30, because its trials
 * repair a coarse ranking and its probe is where the page's time has gone. */
#ifndef ZPSE_PLANE_STRIDE_PLANES_FAST
#define ZPSE_PLANE_STRIDE_PLANES_FAST 6u
#endif
#ifndef ZPSE_PLANE_STRIDE_FAST_MAX
#define ZPSE_PLANE_STRIDE_FAST_MAX 4u
#endif

static unsigned zpse_block_plane_step(const ZpseEncoder *e, uint32_t blk_n,
                                      uint8_t sc)
{
#if (ZPSE_PLANE_STRIDE_N > 0)
    unsigned want = (unsigned)ZPSE_PLANE_STRIDE_PLANES;
    unsigned step;

    if (e != 0 &&
        zpse_enc_speed(e->enc_flags) <= (uint32_t)ZPSE_PLANE_STRIDE_FAST_MAX) {
        want = (unsigned)ZPSE_PLANE_STRIDE_PLANES_FAST;
    }
    if (want == 0u || blk_n >= (uint32_t)ZPSE_PLANE_STRIDE_N ||
        (unsigned)sc <= want) {
        return 1u;
    }
    /* Round UP, so `want` means "price at most `want` planes": a 16-plane
     * page at want 6 prices 6 (0, 3, 6, ... 15), not 8. */
    step = ((unsigned)sc + want - 1u) / want;
    return step == 0u ? 1u : step;
#else
    /* Off: every plane, everywhere, which is what this encoder has always
     * priced. `blk_n` is still taken so the call sites do not have to know
     * whether the knob is on. */
    (void)e;
    (void)blk_n;
    (void)sc;
    return 1u;
#endif
}

/* Exact price of a plane: the smaller of its tile records (a spread of its
 * tiles, classified, mode-searched and packed exactly as a trial does) and
 * the two dense forms, which are priced exactly because each takes one pass
 * over the plane. zpse_segment_probe_cost owns all three; this only gives
 * it the lane state a segment encode would start from.
 *
 * `tiles` is how many tiles to price (0 = ZPSE_PROBE_TILES), spread evenly
 * over the plane and folded back up to its full tile count. `limit` is the
 * running verdict the caller holds: a plane whose own proven floor already
 * exceeds it stops there (see seg_tile_price in segment.c). Returns the
 * estimate for the whole plane, or UINT32_MAX when the plane cannot be
 * priced (the caller then keeps the candidate). */
static uint32_t zpse_block_exact_plane(ZpseEncoder *e, const uint8_t *plane,
                                 uint32_t len, unsigned tiles,
                                 uint32_t limit)
{
    ZpseLaneState lanes;

    if (plane == 0 || len == 0u) {
        return UINT32_MAX;
    }
    if (tiles == 0u || tiles > ZPSE_DECIDE_TILES) {
        tiles = (tiles == 0u) ? ZPSE_PROBE_TILES : ZPSE_DECIDE_TILES;
    }
    zpse_lane_reset_hist(&lanes, 0u);
    return zpse_segment_probe_cost(e, &lanes, plane, len, tiles, limit);
}

/* Smallest price zpse_block_exact_plane can return for a plane of `len`
 * bytes: the segment envelope, plus either a 2-byte length prefix and the
 * 8-byte header of every tile (the record of a tile whose prediction is
 * exact: mask ALL_HIT, empty residual) or the dense constant form's 12
 * bytes per 4 KiB, whichever is smaller. No record can be below either
 * bound, so a plane priced at this floor is one whose every tile was an
 * exact prediction -- a constant plane, which the dense form then writes. */
static uint32_t zpse_block_plane_floor(uint32_t len)
{
    uint32_t tiles = 16u + 10u * ((len + 255u) / 256u);
    uint32_t dense = 16u + 12u * ((len + ZPSE_DENSE_MAX_LEN - 1u) /
                                  ZPSE_DENSE_MAX_LEN);

    return tiles < dense ? tiles : dense;
}

/* Smallest price zpse_block_exact_sum can return for a candidate whose
 * planes total `len` bytes: the block envelope zpse_block_exact_sum starts
 * from, plus one plane at its floor. Any candidate with more than one plane
 * pays the 8-byte segment envelope once per plane and so cannot be below
 * this; a single-segment one reaches it exactly when its plane is at the
 * floor. */
static uint32_t zpse_block_cand_floor(uint32_t len)
{
    return (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
           zpse_block_plane_floor(len);
}

/* Exact price of one candidate, summed the same way the prescreen proxy sums
 * its planes.
 *
 * `every_plane` is BALANCED and deeper (level 5+) and prices one tile
 * in every plane, which is the only way to rank a transpose correctly (see
 * the comment below). The cheap levels (1-4) price `tiles` planes instead
 * and charge the rest their average, or nothing at all when they are
 * constant: that is the estimate this code used to make everywhere, and it
 * is wrong by design in exactly one direction -- it under-prices the wide
 * transposes,
 * whose planes differ most from each other. */
static uint32_t zpse_block_exact_sum(ZpseEncoder *e, uint8_t *planes[16],
                                      const uint32_t lens[16], uint8_t sc,
                                      unsigned tiles, uint32_t limit,
                                      int every_plane, unsigned pstep)
{
    uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u;
    uint32_t records = 0u;
    uint32_t tile_records = 0u;
    unsigned tile_planes = 0u;
    unsigned step = 1u;
    unsigned s;

    /* 0 means "the probe's own default", exactly as zpse_block_exact_plane
     * reads it; it is resolved here because it is the depth this function
     * asks a single-segment plane for. */
    if (tiles == 0u || tiles > ZPSE_DECIDE_TILES) {
        tiles = (tiles == 0u) ? ZPSE_PROBE_TILES : ZPSE_DECIDE_TILES;
    }
    /* Every plane is priced, one tile each. A transpose's planes are NOT
     * interchangeable, and an estimate that prices a stride of them and
     * charges the rest the average of those is not merely coarse -- it is
     * biased towards whatever the priced planes happen to be worth. On the
     * decimal8 corpus the planes of the width-16 delta-transpose differ by
     * several times over, and pricing four of the sixteen (0, 4, 8, 12) put
     * the whole candidate at 1648 bytes on the 16 KiB window against the
     * 3728 it actually costs over the block, so it beat the width-8 plain
     * transpose (3276 -- and the cheapest shape the block has) by a margin
     * the block does not have. One tile in each of the sixteen planes prices
     * it at its cost, and the transpose that really wins is the one the
     * probe picks.
     *
     * The price of the deeper pass is 60 plane-tiles for the ten candidates
     * of a 64 KiB block instead of 36, which is why a constant plane is
     * tested BEFORE its tile is priced (see below). Measured end to end,
     * the pass is what buys decimal8 4.62:1 -> 5.82:1 (3.96:1 -> 4.01:1 over
     * the corpus).
     *
     * The cheap levels (1-4) used to price only a stride of the planes and
     * charge the rest their average -- which under-prices wide transposes
     * whose planes differ (DT8 on 2-byte records came out 49% larger than
     * BALANCED's pick). They now price every plane, one tile each; a single
     * tile per plane is enough to rank a transpose, and single-segment
     * candidates still pay fewer tiles than BALANCED's four,
     * which is where the cheap levels' speed comes from. */
    if (!every_plane) {
        step = 1u;
    }
    if (pstep > step) {
        step = pstep;
    }
    for (s = 0u; s < sc; s += step) {
        /* The running share: everything still unspent below `limit`
         * belongs to the planes not yet priced, and this plane may spend
         * its envelope plus a fair piece of it. UINT32_MAX (no verdict in
         * hand) prices the whole sample, as before. */
        uint32_t share = UINT32_MAX;
        if (limit != UINT32_MAX) {
            uint64_t envelope = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
                                (uint64_t)sc * 16u;
            if ((uint64_t)records + envelope <= (uint64_t)limit) {
                share = (uint32_t)(16u +
                          ((uint64_t)limit - (uint64_t)records - envelope));
            }
        }
        {
            uint32_t c;

            /* A constant plane does not need a tile priced: the dense record
             * form is the cheapest way to write it (12 bytes per 4 KiB
             * against 10 bytes per 256-byte all-hit tile), and the scan that
             * finds this out exits at the first differing byte, so it costs
             * nothing on a plane that is not constant. It is what keeps the
             * deeper pass affordable on the blocks whose planes mostly are
             * constant -- zeros, a run of offsets, a repeated string -- where
             * a tile in each of sixteen constant planes was most of the
             * probe. */
            if (sc > 1u) {
                uint32_t d = zpse_segment_const_cost(planes[s], lens[s]);

                if (d != UINT32_MAX) {
                    records += d - 16u;
                    continue;
                }
            }
            c = zpse_block_exact_plane(e, planes[s], lens[s],
                                       sc > 1u ? 1u : tiles, share);

            if (c == UINT32_MAX) {
                return UINT32_MAX;
            }
            /* c is 16 + the plane's record area folded up to its tile count. */
            records += c - 16u;
            tile_records += c - 16u;
            tile_planes++;

            /* Early exit: if the running lower bound on the candidate total
             * already exceeds `limit`, this candidate cannot win. Return the
             * bound (finite, > limit) so the caller drops it instead of
             * pricing the rest of the planes. UINT32_MAX remains "could not
             * be priced" (keep), so a convicted value is distinguishable. */
            if (limit != UINT32_MAX) {
                uint64_t lb = (uint64_t)(ZPSE_BLOCK_HEADER_SIZE + 16u) +
                              (uint64_t)sc * 16u + (uint64_t)records;
                if (lb > (uint64_t)limit) {
                    return (uint32_t)(lb > (uint64_t)UINT32_MAX
                                          ? (uint64_t)UINT32_MAX
                                          : lb);
                }
            }
        }
    }
    /* The planes the cheap levels did not price: a constant one is priced by
     * the scan, and anything else is charged the average of the planes that
     * were. Zero tile-priced planes means the whole candidate was constant,
     * so there is nothing left to fill in. */
    if (step > 1u && tile_planes > 0u) {
        uint32_t avg = tile_records / tile_planes;

        for (s = 0u; s < sc; s++) {
            uint32_t d;

            if ((s % step) == 0u) {
                continue;
            }
            d = zpse_segment_const_cost(planes[s], lens[s]);
            records += (d != UINT32_MAX) ? (d - 16u) : avg;
        }
    }
    total += (uint32_t)sc * 16u + records;
    return total;
}

/* Stage one candidate's transform into the encoder's staging area and hand
 * back its planes and lengths. A width-1 split with neither DELTA8 nor XOR8
 * is the identity, so the input itself serves as the plane for it. */
static int zpse_block_planes(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint8_t tid, uint8_t sc,
                             const uint8_t *meta_bytes, uint16_t meta_len,
                             uint8_t *planes[16], uint32_t lens[16])
{
    ZpseTransformMeta tm;
    uint8_t w = 0;
    uint32_t off = 0;
    uint32_t s;

    if (zpse_transform_width(tid, sc, &w) != ZPSE_OK)
        return ZPSE_BAD_TRANSFORM;
    if (zpse_segment_lengths(n, w, lens) != ZPSE_OK)
        return ZPSE_BAD_TRANSFORM;
    if (zpse_transform_meta_parse(&tm, tid, meta_bytes, meta_len, sc) != ZPSE_OK)
        return ZPSE_BAD_TRANSFORM;
    for (s = 0; s < sc; s++) {
        if (lens[s] == 0u || (uint64_t)off + lens[s] > ZPSE_E_SZ_TSTAGE)
            return ZPSE_OUTPUT_OVERFLOW;
        planes[s] = e->tstage + off;
        off += lens[s];
    }
    if (off != n)
        return ZPSE_BAD_TRANSFORM;
    if (tid == 0u && sc == 1u) {
        planes[0] = (uint8_t *)(uintptr_t)in;
        return ZPSE_OK;
    }
    /* The trusted form: the planes were just laid out above as
     * `tstage + running_total` with `running_total == n`, and `lens` came
     * from `zpse_segment_lengths`, so the disjointness of the sixteen planes
     * (16 + 120 range tests at w = 16) is a fact this function established
     * on the way in. See zpse_transform_forward_impl. */
    return zpse_transform_forward_planes(&tm, sc, in, n, planes, lens);
}

/* The prefix price could not be taken: the candidate's first tile in some
 * plane is all-hit, so it MAY be a constant plane, and the dense form (which
 * the tile sample cannot see) prices a constant plane below ten bytes a
 * tile. Only the whole plane can settle that, so the caller stages the full
 * window instead. Distinct from UINT32_MAX, which still means "could not be
 * priced" and keeps the candidate. */
#define ZPSE_PREFIX_NEED_FULL UINT32_C(0xFFFFFFFE)

/* Price a multi-plane candidate from the transform PREFIX it reads.
 *
 * A candidate with more than one plane is priced one tile in each plane
 * (zpse_block_exact_sum), and that tile is the plane's FIRST one: the fold
 * to the plane's real tile count is arithmetic (16 + (price - 16) * tc). An
 * every-plane pass therefore reads exactly 256 bytes of each plane, which
 * for a width-w transform are the first 256*w bytes of the window -- 4 KiB
 * of the probe's 12 KiB window at width 16, 1 KiB at width 4, 512 bytes at
 * width 2 -- so staging the whole window to reach a third of it (width 16)
 * is pure cost. Staging is a quarter to a third of what the probe spends on
 * a 64 KiB block (measured with the ZPSE_BLOCK_STATS build: 45 000 of
 * 196 000 cycles on a 4-bit plane, 41 000 of 120 000 on noise).
 *
 * The tile's own price is bit-for-bit the one the full staging produces: the
 * same 256 bytes, the same segment position 0, the same initial literal,
 * and a classify whose lane state starts where it starts in the full pass.
 * The fold back to the real tile count is the one exact_sum applies.
 *
 * Returns the price, UINT32_MAX when the plane cannot be priced, or
 * ZPSE_PREFIX_NEED_FULL when the prefix cannot answer the question (see
 * above); the caller then runs the full staging. */
static uint32_t zpse_block_prefix_price(ZpseEncoder *e, const uint8_t *in,
                                        uint32_t n, uint32_t blk_n,
                                        uint8_t tid, uint8_t sc,
                                        const uint8_t *meta_bytes,
                                        uint16_t meta_len, uint32_t limit)
{
    ZpseTransformMeta tm;
    uint32_t lens[16];
    uint8_t *planes[16];
    uint32_t pfront;
    uint32_t records = 0u;
    uint8_t wb = 0u;
    uint32_t w;
    uint32_t s;

    if (e == 0 || in == 0 || sc < 2u || sc > 16u) {
        return ZPSE_PREFIX_NEED_FULL;
    }
    if (zpse_transform_width(tid, sc, &wb) != ZPSE_OK || wb == 0u) {
        return ZPSE_PREFIX_NEED_FULL;
    }
    w = (uint32_t)wb;
    if (zpse_segment_lengths(n, wb, lens) != ZPSE_OK)
        return ZPSE_PREFIX_NEED_FULL;
    /* Every plane has to hold one whole tile for the fold to be the one
     * exact_sum makes, and the prefix has to sit inside the window. */
    for (s = 0u; s < sc; s++) {
        if (lens[s] < 256u) {
            return ZPSE_PREFIX_NEED_FULL;
        }
    }
    pfront = 256u * w;
    if (pfront > n || (uint64_t)sc * 256u > (uint64_t)ZPSE_E_SZ_TSTAGE) {
        return ZPSE_PREFIX_NEED_FULL;
    }
    if (zpse_transform_meta_parse(&tm, tid, meta_bytes, meta_len, sc) != ZPSE_OK) {
        return ZPSE_PREFIX_NEED_FULL;
    }
    for (s = 0u; s < sc; s++) {
        planes[s] = e->tstage + 256u * s;
    }
    {
        uint32_t flens[16];

        for (s = 0u; s < sc; s++) {
            flens[s] = 256u;
        }
        if (zpse_transform_forward_planes(&tm, sc, in, pfront, planes,
                                          flens) != ZPSE_OK) {
            return ZPSE_PREFIX_NEED_FULL;
        }
    }
    {
        unsigned step = zpse_block_plane_step(e, blk_n, sc);
        unsigned npriced = 0u;

        for (s = 0u; s < sc; s += step) {
            uint32_t tc = (lens[s] + 255u) >> 8;
            uint32_t c = zpse_block_exact_plane(e, planes[s], 256u, 1u,
                                                UINT32_MAX);

            if (c == UINT32_MAX) {
                return UINT32_MAX;
            }
            if (c == zpse_segment_price(e->enc_flags) + 10u) {
                /* All-hit first tile: the plane may still be constant, and
                 * the dense form prices that below ten bytes a tile. */
                return ZPSE_PREFIX_NEED_FULL;
            }
            records += (c - zpse_segment_price(e->enc_flags)) * tc;
            npriced++;
            /* The sweep's running verdict, applied per plane exactly as the
             * full pass applies it: a bound already past `limit` cannot win,
             * so the planes behind it do not have to be staged or priced. */
            if (limit != UINT32_MAX) {
                uint64_t lb = (uint64_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
                              (uint64_t)sc * 16u + (uint64_t)records;

                if (lb > (uint64_t)limit) {
                    return (uint32_t)(lb > (uint64_t)UINT32_MAX
                                          ? (uint64_t)UINT32_MAX
                                          : lb);
                }
            }
        }
        /* The planes the stride skipped: charged the average of the ones it
         * priced, exactly as the cheap levels' estimate used to. */
        if (step > 1u && npriced > 0u) {
            uint32_t avg = records / npriced;
            unsigned sp;

            for (sp = 0u; sp < sc; sp++) {
                if ((sp % step) != 0u) {
                    records += avg;
                }
            }
        }
    }
    return (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u + (uint32_t)sc * 16u +
           records;
}

/* Exact price of one candidate, used to break a prescreen tie: see
 * zpse_block_exact_plane. UINT32_MAX means "could not be priced".
 *
 * `prefix_ok` is the caller's verdict on prefix pricing (see
 * zpse_block_prefix_price): it starts set and this function clears it the
 * first time a prefix cannot answer, which on a block whose planes really
 * are constant is the first transpose candidate it prices. NULL means "do
 * not try": a caller that prices a candidate whose planes are not read from
 * their head (a single-plane one spread over the window, or the tail
 * check's whole-window price) is not where the prefix pays. */
static uint32_t zpse_block_proxy_exact(ZpseEncoder *e, const uint8_t *in,
                                       uint32_t n, uint32_t blk_n,
                                       uint8_t tid, uint8_t sc,
                                       const uint8_t *meta_bytes,
                                       uint16_t meta_len, unsigned tiles,
                                       uint32_t limit, int every_plane,
                                       int *prefix_ok)
{
    uint32_t lens[16];
    uint8_t *planes[16];
    unsigned pstep = zpse_block_plane_step(e, blk_n, sc);

#if ZPSE_DECIDE_PAGE_TILES > 0
    /* A page prices its single-plane candidates on a shorter spread than a
     * 64 KiB block does (see ZPSE_DECIDE_PAGE_TILES). Only the probe's own
     * default is remapped, so a caller asking for a specific tile count --
     * zero means the probe's default, a larger number the narrow search -- is
     * still honored. */
    if (blk_n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX &&
        tiles == (unsigned)ZPSE_DECIDE_TILES) {
        tiles = (unsigned)ZPSE_DECIDE_PAGE_TILES;
    }
#endif

    /* A multi-plane candidate is priced with one tile in every plane, at
     * every level (the cheap levels fill nothing in for a transpose any
     * more), and that tile is the head of the plane -- so it is priced from
     * the prefix the transform has to write for it and the rest of the
     * window is never staged. That holds whatever `every_plane` says: the
     * two passes agree plane for plane, and `every_plane` only differs for a
     * single-plane candidate, whose spread of tiles is not in the prefix.
     * A candidate whose prefix cannot answer (a possibly constant plane --
     * see zpse_block_prefix_price) falls through to the full pass, and the
     * caller stops trying prefixes. */
    if (prefix_ok != 0 && *prefix_ok && sc > 1u) {
        uint32_t p = zpse_block_prefix_price(e, in, n, blk_n, tid, sc,
                                             meta_bytes, meta_len, limit);

        if (p != ZPSE_PREFIX_NEED_FULL) {
            return p;
        }
        *prefix_ok = 0;
    }
    if (zpse_block_planes(e, in, n, tid, sc, meta_bytes, meta_len, planes,
                          lens) != ZPSE_OK) {
        return UINT32_MAX;
    }
    return zpse_block_exact_sum(e, planes, lens, sc, tiles, limit,
                                every_plane, pstep);
}

/* ---- Why the price is the encoder's own cost model ---------------- *
 * The ranking pass runs the segment encoder's classify, mode search and pack
 * on a tile of every plane, which is half the dumps' encode (48.5% of the
 * cycles at level 6, 5.6 candidates a page). A cheap estimator in its place
 * looks like the next stage to buy, and one was built and measured: per plane,
 * two order-0 histograms (the plane and its difference chain), charged at the
 * smaller entropy in bits plus the same envelope and ten bytes a tile the
 * exact price charges.
 *
 * It is refused on rate, and the margin is not close. The histogram scan is
 * algebraically cheaper than a classify but it is a scalar byte loop with two
 * counter increments a byte, while the classify is SIMD and reads the same
 * 4 KiB the histogram does; pricing 360 361 candidates a corpus pass instead
 * of 292 432 (the estimator's prices do not convict as early) puts the dumps
 * at 41 MB/s against 111-115 and Silesia at 54 against 113. What it does show
 * is that the ranking is not what limits the ratio -- it writes 0.41, 0.68 and
 * 0.85 points FEWER bytes on the dumps, mixed and Silesia -- but three times
 * the encode for bytes bought back by a different shortlist is not the trade
 * these pages are held to, and the sweep's trials, not its prices, are what
 * a page is actually paying for. */

/* Exchange two candidate slots, moving the candidate itself in every array
 * that describes it -- including idx[], which carries its original list
 * index and is what the winner's tie-break compares. The attempt order is
 * realized this way and never as an index permutation over the list, so the
 * exact prune (which compacts the list) cannot leave the attempt loop
 * pointing at a slot it has since dropped: after the swap the loop simply
 * walks the surviving slots in order. */
static void zpse_block_swap_cand(uint8_t tids[ZPSE_MAX_CAND],
                                 uint8_t scounts[ZPSE_MAX_CAND],
                                 uint8_t meta_bufs[ZPSE_MAX_CAND][ZPSE_MAX_CAND_META],
                                 uint16_t meta_lens[ZPSE_MAX_CAND],
                                 uint8_t idx[ZPSE_MAX_CAND],
                                 int a, int b)
{
    uint8_t t8 = tids[a];
    uint8_t s8 = scounts[a];
    uint8_t i8 = idx[a];
    uint8_t mbuf[ZPSE_MAX_CAND_META];
    uint16_t l16 = meta_lens[a];
    unsigned m;

    /* Every metadata byte moves, not just the first two: a delta-transpose
     * candidate carries up to 16 seed bytes and swapping only two of them
     * would hand the other candidate's trial a seed it never priced. */
    for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
        mbuf[m] = meta_bufs[a][m];
    }

    tids[a] = tids[b];
    scounts[a] = scounts[b];
    idx[a] = idx[b];
    meta_lens[a] = meta_lens[b];
    for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
        meta_bufs[a][m] = meta_bufs[b][m];
    }

    tids[b] = t8;
    scounts[b] = s8;
    idx[b] = i8;
    meta_lens[b] = l16;
    for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
        meta_bufs[b][m] = mbuf[m];
    }
}

/* Encode the block as the intra-block LZ form into dst[0..cap). Returns
 * ZPSE_OK with *total_out set, or a negative status when the payload does
 * not fit (which the caller reads as "this candidate lost", not as an
 * error). The block header is written last-to-first: the payload at
 * dst + 32, then the envelope around it. */
/* ---- The LZ4 form's decode bill, one global acceleration factor ---- *
 *
 * An LZ4 stream decodes at the speed of its own command loop, so its rate
 * follows the AVERAGE RUN LENGTH between LZ4 sequences.
 * Measured on this corpus, in bytes of output per command: 5.6 -> 0.6 GB/s
 * (prose), 9 -> 1.2 (json), 17 -> 2.9 (var strings), 64 -> 8.8 (sparse),
 * 128 -> 8.4 (geo), 246 -> 24 (zeros). A tile stream decodes at 5-13 GB/s
 * whatever it holds (~12 GB/s on the shapes this encoder picks), so the two
 * cross at about ZPSE_LZ_BPC_CROSS output bytes per command and the only
 * question a block has to answer is how much decode speed it will trade for
 * size.
 *
 * That trade was historically expressed as an ACCELERATION FACTOR A -- the
 * same idea as zstd's -1/-2/-4, moved to the decode side: the form was taken
 * when its commands averaged at least ZPSE_LZ_BPC_CROSS / A output bytes,
 * i.e. when the LZ stream was expected to decode at least A times SLOWER than
 * the tile stream it replaces. A = 2 is the 48 bytes per command this rule
 * used to be hard-coded to (and it is nearly inert: it keeps LZ only for
 * blocks whose runs are long enough to out-decode the tiles); a large A buys
 * ratio on text and pays for it in decode rate. Levels no longer name their
 * own A: there is one global bill, taken from the default row's accel in
 * zpse_lz_min_bpc below.
 *
 * The levels were swept with every value in turn, on the four text cases and
 * a repetitive one, against the tile stream's own 1.09:1 at ~14 GB/s. The
 * numbers quoted below are those historical measurements (the codec has since
 * been replaced by LZ4); the ladder's ordering is what they settled:
 *
 *   A=2  (48 B/cmd)  no text case moves; the shipped rule before this knob.
 *   A=4  (24 B/cmd)  no text case moves either: var_string, the longest-run
 *                    text shape here, sits just under 24 bytes per command.
 *   A=8  (12 B/cmd)  var_string 1.09 -> 2.52 at 2.6 GB/s, and log_lines
 *                    1.09 -> 1.68 at 2.0 (part of its blocks clear the bar).
 *   A=12 (8 B/cmd)   + json_rows 1.09 -> 2.05 at 1.1 GB/s and log_lines the
 *                    rest of the way to 2.63 at 1.5. This is the value the
 *                    two deepest levels ship.
 *   A=16 (6 B/cmd)   unchanged from A=12: prose sits at 5.6 bytes per
 *                    command, just under the bar.
 *   A=24 (4 B/cmd)   + prose, at MAX only: text_en 1.09 -> 1.33 at 0.6 GB/s.
 *                    At the time the default level did not take it because
 *                    its sample margin (1.25) rejected prose before this rule
 *                    was asked; MAX's (1.0625) did not. Those per-level
 *                    margins are historical -- the LZ4 screen is now shared.
 *
 * So A=12 was the setting that bought the structured-text cases while keeping
 * a GB/s decoder, and A=24 bought prose at a command-loop 0.6 GB/s. Those
 * per-level numbers are historical: levels 5-9 now share one decode bill and
 * one LZ4 screen, and only trials and tie_field vary down the ladder. */
#ifndef ZPSE_LZ_BPC_CROSS
#define ZPSE_LZ_BPC_CROSS 96u
#endif
#ifndef ZPSE_LZ_ACCEL_MAX
#define ZPSE_LZ_ACCEL_MAX 24u
#endif
#ifndef ZPSE_LZ_ACCEL_BALANCED
#define ZPSE_LZ_ACCEL_BALANCED 12u /* historical: level 5, 8 bytes/command */
#endif
#ifndef ZPSE_LZ_ACCEL_L7
#define ZPSE_LZ_ACCEL_L7 16u       /* historical: levels 6-7, 6 bytes/command */
#endif
#ifndef ZPSE_LZ_ACCEL_L8
#define ZPSE_LZ_ACCEL_L8 20u       /* historical: level 8, 4.8 bytes/command */
#endif

/* The sample margins, in sixteenths: the win on the sample a level will pay
 * a match-finder pass for is 16 / num. Smaller is looser. These per-level
 * numbers are historical -- since raw LZ4 there is one fixed screen at every
 * level (ZPSE_LZ4_SCREEN_NUM / ZPSE_LZ4_SCREEN_SAMPLES below). */
#ifndef ZPSE_LZ_SCREEN_NUM_MAX
#define ZPSE_LZ_SCREEN_NUM_MAX 17u /* historical: 1.0625x, levels 8-9 */
#endif
#ifndef ZPSE_LZ_SCREEN_NUM_L7
#define ZPSE_LZ_SCREEN_NUM_L7 18u      /* historical: 1.125x, level 7 */
#endif
#ifndef ZPSE_LZ_SCREEN_NUM_L6
#define ZPSE_LZ_SCREEN_NUM_L6 19u      /* historical: 1.1875x, level 6 */
#endif
#ifndef ZPSE_LZ_SCREEN_NUM_BALANCED
#define ZPSE_LZ_SCREEN_NUM_BALANCED 20u /* historical: 1.25x, level 5 */
#endif

/* The smallest win that pays for the LZ form's slower decode. A stream one
 * byte smaller is not worth trading a 12 GB/s decoder for a command loop,
 * and the rule is deliberately coarse: the LZ form's own size is estimated
 * well (the sweep compares real totals), the win test only has to stop the
 * one-byte "wins", which are noise on the sampling and cost 20x the decode.
 * 1/32 is the same margin the candidate sweep keeps. */
#ifndef ZPSE_LZ_MIN_WIN_DIV
#define ZPSE_LZ_MIN_WIN_DIV 32u
#endif

/* ---- The effort ladder, one row per level --------------------------- *
 *
 * Every knob that decides WHICH shape a level writes is in this table. The
 * table exists because the thresholds it replaces made the ladder shorter
 * than it claimed to be: the sweep tested `lvl >= 5`, `lvl >= 7`, `lvl >= 9`
 * and `lvl <= 2`, so levels 2, 4, 6 and 8 evaluated every test exactly as
 * their neighbour below did and were aliases of it, not rungs. A level that
 * appears on the ladder has to be a level, so each row differs from the row
 * above it in at least one column and the columns are ordered so that the
 * deeper row can only choose from a superset of the same shapes:
 *
 *   trials      full block encodes the sweep may spend; 0 = every survivor
 *               (a level that tries more candidates can only take a smaller
 *               total, never a larger one).
 *   screen_num  the LZ sample margin in sixteenths -- the win on the sample
 *               that pays for a match-finder pass is 16/num. Smaller num is
 *               a looser margin, and a pass that runs can only be kept when
 *               it is smaller, so the deeper row is never the larger one.
 *   samples     how many samples the screen may spend (2 = head and middle,
 *               1 = head only). A screen that refuses fewer passes keeps
 *               more candidates, which again can only lower the total.
 *   accel       the LZ decode bill (ZPSE_LZ_BPC_CROSS / accel bytes per
 *               command); a larger factor accepts streams with shorter runs,
 *               so it is a superset of the row above for the same reason.
 *   tie_field   1 = cut a guarded full-size field to its best two on price
 *               alone (see ZPSE_TIE_KEEP); 0 = only when the third price is
 *               inside the near-tie band. Cutting trials can only raise a
 *               block's total, so the column is non-increasing down the
 *               ladder and the deeper row still chooses from a superset.
 *
 * Levels 1-4 write no LZ form, so their LZ columns are zero and unused. At
 * levels 5-9 the LZ columns no longer differentiate levels either: one shared
 * screen and one bill (the default row's accel), so only trials and tie_field
 * vary by level. The prescreen width (level 1 cheap, 2-9 full) and the
 * every-plane pricing (5-9) are not here: they are single thresholds shared
 * by whole tiers of the ladder, and they live in zpse_enc_speed's own comment
 * in include/zpse/constants.h. */
typedef struct {
    uint8_t trials;
    uint8_t screen_num;
    uint8_t samples;
    uint8_t accel;
    uint8_t tie_field;
} ZpseLevel;

/* Level 4's rung: one more full-block trial than level 3, and nothing
 * else -- the last rung before the LZ columns open at BALANCED. Built at 3
 * it is level 3 exactly, which is the fold its measurement compares. */
#ifndef ZPSE_L4_TRIALS
#define ZPSE_L4_TRIALS 4u
#endif

/* How many full-block trials a sweep that cuts tied fields may spend on a
 * FULL-SIZE block, 0 = every survivor the price kept.
 *
 * The trial loop's floor bound and running-best budget already skip the
 * candidates that provably cannot win, so what is left is the survivors the
 * price could not separate -- and a window price is a weak ordering exactly
 * where the trials cost the most. On the dumps at level 6, two full trials
 * are 63% of a block's encode and their prices differ by under 1/16.
 *
 * What this does NOT do is replace the tied-field cut above: the cut keeps
 * the identity survivor whatever the price says and stands down when the
 * head window lied or a survivor was kept on content, and those guards are
 * the reason it is safe. A raw trial cap has none of them, which is why it
 * is applied after the cut and only where the cut's guards are satisfied by
 * construction -- a full-size block, whose price window is a fifth of the
 * block -- rather than to short blocks, where three cases of
 * tests/suites/test_sizes.c grew 8 bytes each under exactly that change.
 *
 * Measured over Silesia at level 6 (203 MiB, RAM-resident, best of two, one
 * core): 377.2 -> 392.7 MB/s for +7,662 bytes over the corpus (0.0075%, i.e.
 * 2.080:1 either way). The gain is where the trials were: mr 242 -> 273
 * MB/s, x-ray 397 -> 443, samba 395 -> 425, nci 564 -> 605.
 *
 * A cap of 1 was measured against 2 and, in the tree as it stands, taken.
 * Over Silesia at level 6, five alternating rounds per arm (each run best of
 * two, arms in both orders), packed bytes deterministic: mr -2,651, mozilla
 * -3,781, ooffice -839, x-ray +18,687, the other eight files identical, the
 * corpus 101,869,745 -> 101,881,161 (+11,416, +0.011%). Encode: mr 253.9 ->
 * 320.5 MB/s, x-ray 394.4 -> 455.8, mozilla 334.0 -> 340.2, dickens 808.0 ->
 * 834.6, samba 430.1 -> 440.1, and reymont/sao/xml 1-2% behind inside this
 * host's spread. Levels 5 and 7 pay and gain the same, byte for byte, because
 * the cap collapses their streams onto the same one-trial block.
 *
 * The earlier reading -- a cap of 1 for another 0.09% of size -- was taken
 * before the tile-body and LZ forms could win a block, so the second trial
 * was the only thing that could still change the stream. With them, x-ray's
 * +0.38% is the corpus's whole size cost and the rate gain lands exactly on
 * the two files that run no finder pass (mr and x-ray were 62% and 64%
 * trial). They still spend over half their encode in the one trial, which is
 * the shape decision itself -- that is where the ceiling sits now. */
#ifndef ZPSE_TRIAL_CAP_FULL
#define ZPSE_TRIAL_CAP_FULL 1
#endif

/* Whether the trial cap also applies to a level whose row does NOT cut tied
 * fields (8 and 9, which encode every survivor their price separated).
 *
 * 0 leaves those rungs alone: their whole point is the field the price could
 * not order, and the cap is the one thing that would stop them paying for it.
 * 1 is the measurement, and the measurement says no. Over Silesia, default
 * against 1: level 8 341.5 -> 335.4 MB/s, level 9 250.0 -> 254.3, level 6
 * unchanged either way (it is capped already), and 7 230 bytes off a ~94 MB
 * packed total in both directions -- noise. The deepest rungs are not paying
 * for trials they do not need, so 0 stays and they keep the field. */
#ifndef ZPSE_TRIAL_CAP_TIEFREE
#define ZPSE_TRIAL_CAP_TIEFREE 0
#endif

static const ZpseLevel zpse_level[10] = {
    /* 0: never used -- an empty level field normalises to BALANCED. */
    { 0u, 0u, 0u, 0u, 0u },
    /* 1 FASTEST: the probe's winner only, no match finder. */
    { 1u, 0u, 0u, 0u, 0u },
    /* 2: full nine-predictor prescreen, two trials. */
    { 2u, 0u, 0u, 0u, 0u },
    /* 3 FASTER: the same prescreen, three trials. */
    { 3u, 0u, 0u, 0u, 0u },
    /* 4: four trials, still no match finder. */
    { (uint8_t)ZPSE_L4_TRIALS, 0u, 0u, 0u, 0u },
    /* 5 BALANCED: every survivor, and a guarded field cut to its best two on
     * price. The 1.25x screen and 8 B/command in the columns are historical;
     * the LZ4 screen and bill are shared by every level. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_BALANCED, 2u,
      (uint8_t)ZPSE_LZ_ACCEL_BALANCED, 1u },
    /* 6 (default): the same trials and field cut as 5. The old 1.1875x screen
     * and 6 B/command (0.64% smaller on Silesia at the time: 126,090,987
     * against 126,902,432 bytes, 315 against 316 MB/s over six interleaved
     * rounds) are historical -- the LZ4 screen and bill are shared. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_L6, 2u, (uint8_t)ZPSE_LZ_ACCEL_L7, 1u },
    /* 7: the same trials and field cut as 5/6; the old 1.125x margin and
     * unchanged bill are historical, since the LZ4 screen and bill are shared. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_L7, 2u, (uint8_t)ZPSE_LZ_ACCEL_L7, 1u },
    /* 8: every survivor, no tied-field cut. The old 1.0625x margin, two
     * samples and 5 B/command are historical -- the LZ4 screen and bill are
     * shared. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_MAX, 2u, (uint8_t)ZPSE_LZ_ACCEL_L8, 0u },
    /* 9 MAX: every survivor, no field cut; MAX adds Huffman (ZPSE_ENC_HUF).
     * The old head-sample-only, 4 B/command numbers are historical: the LZ4
     * screen and bill are shared by every level. */
    { 0u, (uint8_t)ZPSE_LZ_SCREEN_NUM_MAX, 1u,
      (uint8_t)ZPSE_LZ_ACCEL_MAX, 0u }
};

/* The row for a level number (1-9). An out-of-range level cannot reach here
 * -- enc_flags_ok rejects it -- so the default is a bound, not a policy. */
static const ZpseLevel *zpse_level_row(uint32_t lvl)
{
    return &zpse_level[lvl <= 9u ? lvl : 9u];
}

/* Minimum output bytes per LZ command.
 *
 * The LZ payload is now a raw LZ4 block, so every effort level writes the
 * identical stream for a given block: there is no lazy pass, wider table or
 * stronger finder for a deeper level to buy. A level-dependent threshold
 * would therefore only let a deeper level ACCEPT a block a shallower one
 * refused, which makes the ladder invert instead of getting smaller. The
 * threshold is pinned to the default row's accel so the LZ decision is the
 * same at every level. */
#define ZPSE_LZ4_SCREEN_NUM 18u
#define ZPSE_LZ4_SCREEN_SAMPLES 2u

static uint32_t zpse_lz_min_bpc(uint32_t speed)
{
    uint32_t accel;
    uint32_t bpc;

    (void)speed;
    accel = (uint32_t)zpse_level_row(ZPSE_ENC_SPEED_DEFAULT)->accel;
    if (accel == 0u) {
        accel = 1u;
    }
    bpc = (uint32_t)ZPSE_LZ_BPC_CROSS / accel;
    return bpc == 0u ? 1u : bpc;
}

/* Do the two cheap levels run the match finder too? Off by default, and the
 * measurement is why. On the three structured-text corpora the form wins
 * 1.09 -> 2.03/2.63/2.52, but the pass that finds it runs at ~600 MB/s while
 * the tile encoder runs at 2.4 GB/s on the same blocks, so FASTEST's encode
 * on those cases falls from 2320 MB/s to 391 -- slower than the DEFAULT
 * level, which is the one thing an effort ladder must not do: the levels
 * would no longer be ordered in time, and a caller asking for the fastest
 * setting would get a five-times-slower encoder. The default and the deepest
 * level pay that pass for the ratio (BALANCED 1900 -> 426 MB/s on JSON for
 * 1.09 -> 2.05, and prose at MAX 1.09 -> 1.33); FASTEST and FASTER keep the
 * tile stream and their own 2.4 GB/s. */
#ifndef ZPSE_LZ_CHEAP_LEVELS
#define ZPSE_LZ_CHEAP_LEVELS 0
#endif

/* Does this encoder Huffman code a winning LZ payload? Either the caller
 * asked (ZPSE_ENC_HUF), or the level is deep enough that its own trade is
 * size rather than rate (ZPSE_HUF_MIN_LEVEL, MAX by default). It is not a
 * ladder column: the form only ever replaces a block with a smaller one, so
 * it cannot break the ordering a deeper level promises, and it is offered at
 * one rung because the encode pass it costs is the rung's to spend. */
static int zpse_block_huf_level(uint32_t enc_flags)
{
    return (enc_flags & (uint32_t)ZPSE_ENC_HUF) != 0u ||
           zpse_enc_speed(enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK) >=
               (uint32_t)ZPSE_HUF_MIN_LEVEL;
}

/* A canonical ZPSE_B_LZ winner cannot be improved by the final LZ4 pass:
 * at MAX the lazy LZ winner falls through the gate near the finder-first
 * path with best_idx 15, and the fast screen's stream never beat it. The
 * final pass calls the same stateless LZ4 encoder, so t2 < best_total
 * cannot hold. Measured byte-identical on 36/36 Silesia files, +3.59%
 * encode at level 9 (5/5 pairs), inert at level 6. See docs/HISTORY.md. */
static int zpse_block_winner_is_lz(ZpseEncoder *e, const uint8_t *out,
                                   int direct_out)
{
    ZpseBlockHeader h;
    if (zpse_block_header_parse(&h, direct_out ? out : e->best,
                                ZPSE_BLOCK_HEADER_SIZE) != ZPSE_OK) {
        return 0;
    }
    return (h.block_flags & (uint8_t)ZPSE_B_LZ) != 0u;
}

/* Which levels run the match finder at all: BALANCED and deeper price
 * every block -- the pass is what those levels are for -- and, when
 * ZPSE_LZ_CHEAP_LEVELS is on, the cheap levels price only the blocks the
 * ASCII gate has already convicted of being text, which is the one piece of
 * evidence about the tile family the gate hands back for free. */
static int zpse_block_lz_level(uint32_t speed, uint32_t ascii_gate)
{
    uint32_t lvl = zpse_enc_speed(speed);
    (void)ascii_gate;
#if ZPSE_LZ_CHEAP_LEVELS
    return lvl >= ZPSE_ENC_SPEED_LZ_MIN || ascii_gate != 0u;
#else
    return lvl >= ZPSE_ENC_SPEED_LZ_MIN;
#endif
}

/* The match table the screen used to sample with.
 *
 * The screen samples 512 bytes on a page and 1 KiB on a larger block. This
 * knob (and the 2^13 ceiling) comes from the old caller-supplied match table,
 * where a table sized to the sample instead of the block was worth measuring;
 * the measurements are in docs/HISTORY.md. src/lz4block.c now owns its own
 * match search, so the screens call zpse_lz4_encode_sample with no table at
 * all and the knob is retained only for the level ladder's sizing. */
#ifndef ZPSE_LZ_SCREEN_HASH_BITS
#define ZPSE_LZ_SCREEN_HASH_BITS ZPSE_LZ_HASH_BITS
#endif
#if ZPSE_LZ_SCREEN_HASH_BITS > 13
#error "the screen's table is the page's 16 KiB region; more bits do not fit"
#endif

/* ---- The sample screen the LZ candidate passes first ---------------- *
 *
 * The match finder is the most expensive candidate in the sweep: measured
 * with tests/tools on 64 KiB blocks it runs at 5.4-10.9 GB/s on the
 * repetitive corpora but 575-610 MB/s on the text-shaped ones, against
 * 2.3-2.5 GB/s for the tile encoder on the same blocks. So a level that
 * offers the candidate on every block pays several trials' worth of time
 * for a form that usually loses -- which is why only MAX used to offer it.
 *
 * Before the full pass, the same question is asked on the head of the
 * block: the match finder encodes ZPSE_LZ_SCREEN_BYTES of it, the shape the
 * sweep wants prices the same span with the probe's own machinery, and the
 * pass only goes ahead when the LZ form wins that sample by the shared
 * screen (ZPSE_LZ4_SCREEN_NUM / ZPSE_LZ4_SCREEN_SAMPLES below; the per-level
 * ZPSE_LZ_SCREEN_* margins are historical). The screen is a lower bound on
 * the win, so it can only cost a block the ratio a full pass would have
 * bought on a sample that lies -- and it is what makes the candidate
 * affordable outside MAX: a block with nothing to find never reaches the
 * match finder at all, and the shapes that do reach it keep their 2.0-2.6:1
 * (see the acceleration table above).
 *
 * The sample is 1 KiB, and that is a measurement rather than a default: at
 * MAX, zeros_burst's LZ form is taken at 12.23:1 with a 1 KiB sample, 11.55
 * with 2 KiB, 10.22 with 4 and 10.18 with 8, because a sample bigger than
 * the burst spacing describes the block no better while giving the tile side
 * a bigger span to win on. Historically prose's win on the sample lay between
 * MAX's margin (accepted) and the default's (rejected), which is where the
 * old per-level threshold sat; the LZ4 screen is now shared by every level.
 *
 * Those per-level margins are historical: they were the encode side of the
 * same trade the acceleration factor made on the decode side -- a deeper
 * level paid a pass for a smaller sample win, MAX's being the smallest of
 * the four so the ladder could not invert (tests/suites/test_levels.c pins
 * that). Levels 5-9 now share one LZ4 screen and one decode bill; only
 * trials and tie_field differ down the ladder. */
#ifndef ZPSE_LZ_SCREEN_BYTES
/* 1 KiB, and a longer sample was tried and refused.
 *
 * A kilobyte is shorter than the distance a match finder needs to see: on
 * prose the finder codes the head of a block at about 1.05:1 and the block at
 * 2.0:1, so the screen refuses the pass on exactly the text the finder wins
 * (dickens 1.16, osdb 1.13, webster 1.95 by the tile path). A 4 KiB sample
 * fixes that and is worth a lot on that corpus -- over the twelve Silesia
 * files the default level goes from 2.03:1 to 2.16:1 (104,474,009 bytes to
 * 98,023,866) -- and it costs the columnar corpus more than it buys there:
 * levels 5-9 drop 5.06 -> 5.00, 5.10 -> 5.04 and 5.13 -> 4.88 (level 8,
 * where it also breaks the ladder's ordering against level 7). The reason is
 * the stage below: a block the screen convicts is written from the finder
 * without a trial, so a longer sample does not just measure better, it
 * ARGUES BETTER for a form the tiles beat by a few percent on a column whose
 * records the sample has too few of to see. Making that trade honestly needs
 * the sweep's own price in the comparison (see zpse_block_encode), which is a
 * different change; until then the sample stays where the measurement over
 * both corpora puts it, and the macro is the knob for the corpus a caller
 * actually has. */
#define ZPSE_LZ_SCREEN_BYTES 1024u
#endif
/* The screen's sample on a PAGE, and a measurement rather than a default.
 *
 * The 1 KiB above is tuned on 64 KiB blocks, where it is a sixty-fourth of
 * the block; a 4 KiB page hands the same sample a quarter of its bytes, and
 * the screen runs twice per page (the finder screen, then the final LZ
 * screen) before either pass. At 4 KiB level 6, over four interleaved passes,
 * a 512-byte sample
 * writes FEWER bytes than the 1 KiB one on the dumps (87 026 427 against
 * 87 463 512, 0.21 of a ratio point) and on the mixed corpus (55 671 320
 * against 56 016 700), costs Silesia 123 368 bytes (0.058 of a point), and
 * reads faster on all three (dumps 95.7 against 86.7 MB/s, mixed 96.4
 * against 92.6, Silesia 93.2 against 90.5 on that host). A 2 KiB sample is
 * the ratio arm and it is refused: Silesia goes 115 738 128 -> 114 802 639
 * (-0.8%, the finder stream the corpus wants) at 73.5 against 90.5 MB/s, and
 * the dumps grow 113 963 bytes for it. All 768 size cases are unchanged by
 * the 512-byte sample (no case grows).
 *
 * 512 is the corner, and the arm between it and 1 KiB is the proof: at 768
 * bytes Silesia gives back 171 723 of its 212 MB (0.081 of a point) while the
 * dumps grow 48 286 and the mixed corpus 321 509 (0.21 of a point worse), and
 * all three read slower (dumps 143.9 against 148.5 MB/s, mixed 150.6 against
 * 153.9, Silesia 147.3 against 156.4) -- a longer screen convicts fewer blocks,
 * and some of the passes it then pays for lose to the tiles.
 *
 * 0 keeps the long sample. Only pages take it: every 16 KiB and 64 KiB
 * block's screen is byte-identical to what it was. */
#ifndef ZPSE_LZ_SCREEN_PAGE_BYTES
#define ZPSE_LZ_SCREEN_PAGE_BYTES 512u
#endif
/* Gated experiment: a longer LZ screen sample on ASCII text only.
 *
 * A 4 KiB sample took Silesia from 2.03:1 to 2.16:1 but hurt the columnar
 * corpus (see the ZPSE_LZ_SCREEN_BYTES note above), so it is offered behind
 * ZPSE_LZ_SCREEN_TEXT_GATE and applied only to blocks the ASCII gate accepted.
 * OFF by default: with the flag at 0 every screen -- and every output byte --
 * is exactly as before. */
#ifndef ZPSE_LZ_SCREEN_TEXT_GATE
#define ZPSE_LZ_SCREEN_TEXT_GATE 0
#endif
#ifndef ZPSE_LZ_SCREEN_TEXT_BYTES
#define ZPSE_LZ_SCREEN_TEXT_BYTES 4096u
#endif
#ifndef ZPSE_LZ_SCREEN_DEN
#define ZPSE_LZ_SCREEN_DEN 16u
#endif
#ifndef ZPSE_LZ_SCREEN_SLACK_NUM
#define ZPSE_LZ_SCREEN_SLACK_NUM 5u
#endif
#ifndef ZPSE_LZ_SCREEN_SLACK_DEN
#define ZPSE_LZ_SCREEN_SLACK_DEN 4u
#endif

/* Could a sample of `sn` bytes code to `lz_s` and still beat the block?
 *
 * The sample's bytes per input byte, scaled to the block, is the estimate;
 * `best_total` is what the sweep actually reached. The estimate reads high
 * against the block's own pass because the match finder only sees the bytes
 * it has already scanned: a 1 KiB sample cannot code the repeat that
 * arrives 40 KiB later (zeros_burst's sample-scaled LZ is 6.8 KiB against
 * the 5.3 KiB its pass produces, 28%). The slack is that correction, and a
 * verdict is only taken from a sample that loses by more than it. */
static int zpse_block_lz_sample_can_win(uint32_t lz_s, uint32_t sn,
                                        uint32_t n, uint32_t best_total)
{
    /* The sample's size before scaling: what a payload carries whatever its
     * input, i.e. the literal-count word and one command
     * (ZPSE_LZ_MIN_PAYLOAD). At 1 KiB it is 1% of the sample and forgettable;
     * scaled from 1 KiB to 64 KiB it is 8 bytes against a block that may code
     * to a few hundred, and repeating it 64 times turns a 30-byte period into
     * a block whose estimate is four times its real size (the text-ish corpus
     * case: 60 bytes on the sample, 584 on the block, and an estimate of 3840
     * that convicts it). */
    uint32_t body = lz_s > (uint32_t)ZPSE_LZ_MIN_PAYLOAD
                        ? lz_s - (uint32_t)ZPSE_LZ_MIN_PAYLOAD
                        : (uint32_t)ZPSE_LZ_MIN_PAYLOAD;
    uint64_t est = ((uint64_t)body * (uint64_t)n + ((uint64_t)sn - 1u)) /
                   (uint64_t)sn;
    uint32_t need = best_total -
                    (best_total / (uint32_t)ZPSE_LZ_MIN_WIN_DIV + 1u);

    return est * (uint64_t)ZPSE_LZ_SCREEN_SLACK_DEN <=
           (uint64_t)need * (uint64_t)ZPSE_LZ_SCREEN_SLACK_NUM;
}

/* Should the full match-finder pass be paid for? `tid`, `sc` and the
 * metadata describe the shape the sweep wants for this block, which is what
 * the sample's tile price is measured with; the sample itself is the head of
 * the block. Returns 1 (go ahead) when the sample cannot be coded or priced
 * -- the screen is an optimisation on the sweep, never a new way to drop a
 * candidate the sweep could not judge.
 *
 * `have_best` / `best_total` are the sweep's own verdict on the whole block,
 * and they are asked of the sample once its tile price has agreed with it:
 * the pass is what the sample is estimating, so the comparison it makes when
 * the stream is finished (zpse_block_lz_wins, against the same total) can be
 * made here from the sample. It is what stops the pass where LZ cannot win for a
 * reason the head cannot show -- a block whose tile win comes from a shape
 * the sample's 64-byte planes cannot express. uuid_mix is that block:
 * sixteen 6-byte-prefix records code as one command per record under LZ
 * (0.875 of a byte per byte) and 0.63 under the width-16 transpose, and the
 * pass aborted at its own size cap after spending 68% of the block's encode
 * time. The tile price of the same shape on the same 1 KiB sample is far
 * worse than its 64 KiB price, so the old test accepted it; the sweep's own
 * total cannot make that mistake.
 *
 * The estimate is taken from two samples, the head and the middle, and only
 * a block both of them convict is refused. One sample is not enough: a
 * block's head can be a burst of noise in front of 64 KiB of zeros
 * (zeros_burst), and that head prices LZ at several times what the block's
 * pass reaches. The second sample is paid only by a block the first has
 * already convicted, which is the block that was about to pay a full pass. */
static int zpse_block_lz_screen(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                                uint8_t tid, uint8_t sc, const uint8_t *mb,
                                uint16_t meta_len, uint32_t speed,
                                int have_best, uint32_t best_total,
                                uint32_t cache[2])
{
    (void)speed; /* LZ4 screens are level-independent; see ZPSE_LZ4_SCREEN_NUM. */
    uint32_t sn = n < (uint32_t)ZPSE_LZ_SCREEN_BYTES
                      ? n
                      : (uint32_t)ZPSE_LZ_SCREEN_BYTES;
#if ZPSE_LZ_SCREEN_TEXT_GATE
    /* Gated experiment (ZPSE_LZ_SCREEN_TEXT_GATE): the 4 KiB sample on text
     * only. e->ascii_gate is set before the LZ screens run; the n >= MAX test
     * keeps this to full blocks, so the page branch below (short blocks) and
     * the ordinary corpus are untouched. With the flag off this whole arm is
     * compiled out and the sample is exactly as before. */
    if (e->ascii_gate != 0u && n >= (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        sn = n < (uint32_t)ZPSE_LZ_SCREEN_TEXT_BYTES
                 ? n
                 : (uint32_t)ZPSE_LZ_SCREEN_TEXT_BYTES;
    }
#endif
    /* LZ4 writes one stream per block at every level, so the screen must not
     * vary with the level either: a deeper level that accepts a block the
     * balanced one refused would write a larger block and invert the ladder. */
    uint32_t num = (uint32_t)ZPSE_LZ4_SCREEN_NUM;
    uint32_t lz_s;
    uint32_t tile_s;

#if ZPSE_LZ_SCREEN_PAGE_BYTES
    if (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX) {
        uint32_t pb = (uint32_t)ZPSE_LZ_SCREEN_PAGE_BYTES;

        sn = n < pb ? n : pb;
    }
#endif
    /* The head sample and its tile price are a pure function of (in, n, tid,
     * sc, mb, meta_len): the sample LZ4 encode reinitializes its output state
     * on every call, and the tile price is the encoder's own cost model over
     * the same window. The full-size screen at
     * the top of the sweep asks that exact question about slot 0 with no
     * winner; cache[2] carries its answer here so this screen only re-asks the
     * part that depends on the real best_total. Sentinel {0, UINT32_MAX} means
     * no earlier screen computed it. */
    int use_cache = (cache != 0 && cache[0] != 0u && cache[1] != UINT32_MAX);
    if (sn < ZPSE_LZ_MIN_MATCH + 1u) {
        return 1;
    }
    /* The screen samples with the same LZ4 encoder the writer stage uses, so
     * its estimate and the stream it guards cannot disagree about the codec.
     * The thresholds it is compared against (ZPSE_LZ_SCREEN_DEN, the slack
     * ratio, the samples the ladder adds) keep the values they were swept
     * with, which is what holds the level ladder's ordering. */
    if (use_cache) {
        /* The full-size screen already measured this same slot-0 shape on
         * the same head sample. Only the second-sample question below depends
         * on this screen's best_total, so re-ask just that part. */
        lz_s = cache[0];
        tile_s = cache[1];
    } else {
        lz_s = zpse_lz4_encode_sample(in, sn, e->cand,
                                      (uint32_t)ZPSE_E_SZ_CAND);
        if (lz_s == 0u) {
            return 1;
        }
        /* The shape is priced on its head windows, so a multi-plane one is
         * priced from the transform prefix rather than by staging the sample:
         * the screen runs before anything else on every block, and its answer
         * only costs a candidate's price when the block is one LZ might win. */
        {
            int prefix_ok = 1;

            /* Priced on slot 0 of the list, which is the identity unless the
             * head sample swapped DELTA8 ahead of it. The two prices coincide on
             * the text this decides, and pricing both of them and taking the
             * smaller -- which is what a larger sample would need to stay
             * order-neutral -- costs the columnar corpus 0.06 of ratio at every
             * level from 5 up (5.06 -> 5.00, 5.13 -> 5.07), because a tighter
             * estimate of the tile side refuses the finder on blocks where its
             * full stream is the smaller one. The suite's swap build still
             * writes the same streams at this sample size, so the tolerance is
             * measured rather than assumed: if a future change to the sample
             * makes this line order-sensitive, tests/runners/run.sh says so
             * before this comment can be wrong about it quietly. */
            tile_s = zpse_block_proxy_exact(e, in, sn, n, tid, sc, mb, meta_len,
                                            ZPSE_DECIDE_TILES, UINT32_MAX, 1,
                                            &prefix_ok);
        }
        if (tile_s == UINT32_MAX) {
            return 1;
        }
        if (cache != 0) {
            cache[0] = lz_s;
            cache[1] = tile_s;
        }
    }
    if ((uint64_t)tile_s * (uint64_t)ZPSE_LZ_SCREEN_DEN <
        (uint64_t)lz_s * (uint64_t)num) {
        /* The shapes the sweep offered cannot beat the sample. That is the
         * ordinary answer on prose, and it costs one priced sample. */
        return 0;
    }
    /* The tile price agrees with LZ on the sample; the block's own total may
     * still disagree, and that is the case this second question is for (see
     * the comment above). Two samples, the head and the middle, and only a
     * block both of them convict loses its pass. One is not enough: a
     * block's head can be a burst of noise in front of 64 KiB of zeros
     * (zeros_burst), and that head prices LZ at several times what the
     * block's pass reaches. Asked only after the tile price has already said
     * yes, so a block the old screen refused pays nothing new. */
    if (ZPSE_LZ4_SCREEN_SAMPLES > 1u && have_best &&
        !zpse_block_lz_sample_can_win(lz_s, sn, n, best_total)) {
        uint32_t off = (n - sn) / 2u;
        uint32_t lz_m = zpse_lz4_encode_sample(in + off, sn, e->cand,
                                               (uint32_t)ZPSE_E_SZ_CAND);

        if (lz_m != 0u &&
            !zpse_block_lz_sample_can_win(lz_m, sn, n, best_total)) {
            return 0;
        }
    }
    return 1;
}

/* The final LZ stage's own screen: the same question zpse_block_lz_screen
 * asks, asked again with the same LZ4 encoder that writes the stream. Both
 * screens now price the one canonical LZ4 payload, so the second question is
 * a separate decision point (the last stage, run after the LZ+tile stage)
 * rather than a second codec's estimator.
 *
 * Everything else is the first screen's: the same sample size, the same
 * level margin (so the ladder's monotonicity is the ladder's, not this
 * function's), the same tile price of the same shape, and the same second
 * sample for a block the first one convicts. Both stages budget the same
 * minimum LZ payload, so the sample-scaled estimate is the same one. */
static int zpse_block_lz_final_screen(ZpseEncoder *e, const uint8_t *in,
                                      uint32_t n, uint8_t tid, uint8_t sc,
                                      const uint8_t *mb, uint16_t meta_len,
                                      uint32_t speed, int have_best,
                                      uint32_t best_total)
{
    (void)speed; /* LZ4 encoding is level-independent; see ZPSE_LZ4_SCREEN_NUM. */
    uint32_t sn = n < (uint32_t)ZPSE_LZ_SCREEN_BYTES
                      ? n
                      : (uint32_t)ZPSE_LZ_SCREEN_BYTES;
#if ZPSE_LZ_SCREEN_TEXT_GATE
    /* Gated experiment (ZPSE_LZ_SCREEN_TEXT_GATE): the 4 KiB sample on text
     * only. e->ascii_gate is set before the LZ screens run; the n >= MAX test
     * keeps this to full blocks, so the page branch below (short blocks) and
     * the ordinary corpus are untouched. With the flag off this whole arm is
     * compiled out and the sample is exactly as before. */
    if (e->ascii_gate != 0u && n >= (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
        sn = n < (uint32_t)ZPSE_LZ_SCREEN_TEXT_BYTES
                 ? n
                 : (uint32_t)ZPSE_LZ_SCREEN_TEXT_BYTES;
    }
#endif
    /* LZ4 writes one stream per block at every level, so the screen must not
     * vary with the level either: a deeper level that accepts a block the
     * balanced one refused would write a larger block and invert the ladder. */
    uint32_t num = (uint32_t)ZPSE_LZ4_SCREEN_NUM;
    uint32_t lz_s;
    uint32_t tile_s;

#if ZPSE_LZ_SCREEN_PAGE_BYTES
    if (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX) {
        uint32_t pb = (uint32_t)ZPSE_LZ_SCREEN_PAGE_BYTES;

        sn = n < pb ? n : pb;
    }
#endif
    if (sn < ZPSE_LZ_MIN_MATCH + 1u || e->cand == 0) {
        return 1;
    }
    /* A short block's sample is a larger fraction of the block, and the pass
     * it guards is the expensive one (see ZPSE_LZ_FINAL_SHORT_N). */
    if (n < (uint32_t)ZPSE_LZ_FINAL_SHORT_N) {
        num = num * (uint32_t)ZPSE_LZ_FINAL_SHORT_MARGIN_NUM /
              (uint32_t)ZPSE_LZ_FINAL_SHORT_MARGIN_DEN;
    }
    lz_s = zpse_lz4_encode_sample(in, sn, e->cand, (uint32_t)ZPSE_E_SZ_CAND);
    if (lz_s == 0u) {
        return 1;
    }
    {
        int prefix_ok = 1;

        tile_s = zpse_block_proxy_exact(e, in, sn, n, tid, sc, mb, meta_len,
                                        ZPSE_DECIDE_TILES, UINT32_MAX, 1,
                                        &prefix_ok);
    }
    if (tile_s == UINT32_MAX) {
        return 1;
    }
    if ((uint64_t)tile_s * (uint64_t)ZPSE_LZ_SCREEN_DEN <
        (uint64_t)lz_s * (uint64_t)num) {
        return 0;
    }
    if (ZPSE_LZ4_SCREEN_SAMPLES > 1u && have_best) {
        uint32_t off = (n - sn) / 2u;
        uint32_t lz_m = zpse_lz4_encode_sample(in + off, sn, e->cand,
                                               (uint32_t)ZPSE_E_SZ_CAND);

        if (lz_m != 0u &&
            !zpse_block_lz_sample_can_win(lz_m, sn, n, best_total)) {
            return 0;
        }
    }
    return 1;
}

/* Is this LZ stream worth taking? Smaller than the tile winner (or the
 * sweep found nothing at all, in which case there is no tile stream to
 * compare against), inside the level's acceleration factor, and smaller by
 * enough to pay for the decode it gives up. The count is walked lazily in
 * the body, once the size gate has passed; it cannot be zero from the walker
 * (which returns at least one), but the division guard is kept as "no
 * evidence either way" rather than a division by zero.
 *
 * The margin is the format's size-only one: the nine-level table is the sole
 * authority for a block over a page, and the page path is FLAT. */
static int zpse_block_lz_wins(uint32_t n, uint8_t *stream, uint32_t lz_total,
                              int have_best, uint32_t best_total,
                              uint32_t speed)
{
    uint32_t lz_cmds;
    uint32_t raw_len;

    if (have_best &&
        (lz_total >= best_total ||
         best_total - lz_total <
             best_total / (uint32_t)ZPSE_LZ_MIN_WIN_DIV + 1u)) {
        return 0;
    }
    /* [block header][u32 n][u32 cmds][raw LZ4 block]. try_lz always writes
     * the full 32-byte envelope. The count walk is the sole consumer of the
     * stream and happens only after the cheap size margin above passes. */
    raw_len = lz_total - (uint32_t)ZPSE_BLOCK_HEADER_SIZE - 8u;
    lz_cmds = zpse_lz_cmd_count(
        stream + (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 8u, raw_len);
    /* Commit the real count word before the caller copies the stream out.
     * The encoder leaves a nonzero placeholder so discarded streams need no
     * walk; a winning stream must carry the same bytes as before. */
    zpse_store_u32le(stream + (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 4u, lz_cmds);
    if (lz_cmds != 0u && n / lz_cmds < zpse_lz_min_bpc(speed)) {
        return 0;
    }
    return 1;
}

/* A candidate that is committed without passing through zpse_block_lz_wins
 * still needs its count word filled in: the encoder only writes a placeholder,
 * and a later Huffman wrap codes every payload byte, so a wrong count would
 * change the output bytes. Called only after the caller's size gate has
 * accepted the stream. */
static void zpse_block_lz_finish_count(uint8_t *stream, uint32_t total)
{
    uint32_t raw_len = total - (uint32_t)ZPSE_BLOCK_HEADER_SIZE - 8u;

    zpse_store_u32le(
        stream + (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 4u,
        zpse_lz_cmd_count(stream + (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 8u,
                          raw_len));
}

/* The block size from which the block layer hands the encoder's spare 64 KiB
 * staging region to the LZ4 pass instead of the 16 KiB e->lz. Historically
 * this was the floor at which a wider caller-supplied hash table paid for
 * itself in the old match finders (the measurements are in docs/HISTORY.md);
 * src/lz4block.c now owns its own match search and ignores the table pointer,
 * so the knob is retained only for the ladder's sizing and the screens.
 *
 * The region is free at every point this is used: it is the trials' transform
 * staging, and the LZ4 pass runs after the last trial. */
#ifndef ZPSE_LZ_TABLE_BIG_MIN
#define ZPSE_LZ_TABLE_BIG_MIN 16384u
#endif
/* The final LZ4 block: the canonical ZPSE_B_LZ form, with the same header,
 * 'is_last' and CRC handling as zpse_block_try_lz. It is the second LZ
 * decision point -- priced after the LZ+tile stage -- not a second codec. */
static int zpse_block_try_lz2(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                              uint32_t in_crc, int is_last, uint32_t seq,
                              uint8_t *dst, uint32_t cap, uint32_t *total_out,
                              uint32_t *cmds_out)
{
    ZpseBlockHeader h;
    uint8_t flags;
    uint32_t payload;
    uint32_t total;

    if (e->lz == 0 || dst == 0 || total_out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (cap <= (uint32_t)ZPSE_BLOCK_HEADER_SIZE + ZPSE_LZ_MIN_PAYLOAD) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    payload = zpse_lz4_encode(in, n, dst + ZPSE_BLOCK_HEADER_SIZE,
                              cap - (uint32_t)ZPSE_BLOCK_HEADER_SIZE);
    if (payload == 0u) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    if (cmds_out != 0) {
        *cmds_out = zpse_load_u32le(dst + ZPSE_BLOCK_HEADER_SIZE + 4u);
    }
    total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + payload;
    flags = (uint8_t)(ZPSE_B_LZ | ZPSE_B_RESET |
                      (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                           ? ZPSE_B_CRC : 0u));
    if (is_last) {
        flags |= ZPSE_B_LAST;
    }
    zpse_block_fill_header(&h, flags, n, total, 0u, 1u, 0u, seq, in_crc);
    zpse_block_header_serialize(dst, &h);
    *total_out = total;
    return ZPSE_OK;
}

/* The match-finder candidate: the one canonical LZ4 form. The table, hash
 * width and lazy-window arguments are kept in the signature for the call
 * sites, which price the second decision point with them; src/lz4block.c
 * ignores them because LZ4 owns the match search. */
static int zpse_block_try_lz(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                             uint32_t in_crc, int is_last, uint32_t seq,
                             uint8_t *dst, uint32_t cap, void *tab,
                             uint32_t hb, uint32_t lazy_max,
                             uint32_t *total_out, uint32_t *cmds_out)
{
    ZpseBlockHeader h;
    uint8_t flags;
    uint32_t payload;
    uint32_t total;

    if (e->lz == 0 || dst == 0 || total_out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (cap <= (uint32_t)ZPSE_BLOCK_HEADER_SIZE + ZPSE_LZ_MIN_PAYLOAD) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    (void)tab;
    (void)hb;
    (void)lazy_max;
    payload = zpse_lz4_encode(in, n, dst + ZPSE_BLOCK_HEADER_SIZE,
                              cap - (uint32_t)ZPSE_BLOCK_HEADER_SIZE);
    if (payload == 0u) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    /* The payload header carries the command count. */
    if (cmds_out != 0) {
        *cmds_out = zpse_load_u32le(dst + ZPSE_BLOCK_HEADER_SIZE + 4u);
    }
    total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + payload;
    flags = (uint8_t)(ZPSE_B_LZ | ZPSE_B_RESET |
                      (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                           ? ZPSE_B_CRC : 0u));
    if (is_last) {
        flags |= ZPSE_B_LAST;
    }
    zpse_block_fill_header(&h, flags, n, total, 0u, 1u, 0u, seq, in_crc);
    zpse_block_header_serialize(dst, &h);
    *total_out = total;
    return ZPSE_OK;
}

/* The ASCII gate's sample: how many bytes of the block are priced against
 * what the seven-bit form would cost over the same span, and with how many
 * tiles. On a full-size block the sample is a 32nd of the block and four
 * 512-byte tiles is enough for the prescreen to tell prose from structure.
 *
 * A page pays the same sample against a sixteenth of its bytes, so on a page
 * the sample IS the gate's cost: two samples (head and tail) times the three
 * shapes, and the gate is 20.8% of a Silesia page's encode. Halving the span
 * to two 512-byte tiles is the measured knee -- at 4 KiB level 6 Silesia goes
 * 117.0 -> 135.0 MB/s, the dumps 112.5 -> 121.5 and the mixed corpus 122.7 ->
 * 127.7, for 0.018 of a ratio point over Silesia (39 050 bytes), 0.001 over
 * mixed and 57 bytes FEWER over the dumps, with all 768 size cases unchanged
 * to the byte. The tile size the price ranks with matters more than the span:
 * four 256-byte tiles on the same 1 KiB span reads 120.8 on Silesia, and a
 * 2 KiB span at two 1 KiB tiles 111.2, both slower than this setting.
 *
 * A 768-byte span -- the point between this and the 1 KiB it replaced -- was
 * measured on the same terms and refused, and it is slower: over five
 * interleaved pairings on one pinned core it reads 110.7 against 117.3 MB/s on
 * the dumps and 109.6 against 117.6 on Silesia, behind in four of the five
 * pairings on both corpora, for 350 bytes FEWER over the dumps and 228 more
 * over Silesia. The span is not free to set to any size: the gate prices whole
 * tiles, so two 384-byte windows are not two of anything it can name.
 *
 * Only pages take it. Above a page the gate keeps the 2 KiB span and four
 * tiles, so every 16 KiB and 64 KiB block this flag judges is byte-identical
 * to what it was. */
#ifndef ZPSE_ASCII_SAMPLE
#define ZPSE_ASCII_SAMPLE 2048u
#endif
#ifndef ZPSE_ASCII_SAMPLE_TILES
#define ZPSE_ASCII_SAMPLE_TILES 4u
#endif
#ifndef ZPSE_ASCII_PAGE_SAMPLE
#define ZPSE_ASCII_PAGE_SAMPLE 1024u
#endif
#ifndef ZPSE_ASCII_PAGE_MID
/* Price the middle of a page as well as its two ends.
 *
 * Two 512-byte tiles at each end are the cheap conviction the page gate was
 * measured at, and on Silesia that costs 186 KB of a 212 MB corpus against the
 * 2 KiB span that was sized for a 64 KiB block. The question this knob exists
 * for is whether a third sample in the middle recovers that difference at less
 * than the 16.6% of the page's cycles the gate costs: a page whose ends are
 * prose and whose middle is a literal run is the case two ends cannot see, and
 * convicting it writes the seven-bit form over a shape the sweep would have
 * priced better.
 *
 * Measured, and refused: Silesia goes 115 924 420 -> 115 874 767 (-49 653
 * bytes, 0.023 of a point) and the mixed corpus -1 445 with the dumps byte
 * identical, but the gate is a sixth of those pages' encodes and the third
 * window costs half again as much: Silesia reads 149.5 against 156.4 MB/s
 * (-4.4%). A fortieth of a point is not worth 4% of the rate, so 0 ships. */
#define ZPSE_ASCII_PAGE_MID 0
#endif
/* One tile an end on a page, and it is a measurement.
 *
 * Two 512-byte windows per end was the knee when the span itself was chosen
 * (see ZPSE_ASCII_PAGE_SAMPLE); the tile COUNT had never been swept, and it
 * was the page's remaining gate cost. At 4 KiB level 6 over four interleaved
 * rounds on one pinned core, one 1 KiB window per end reads 140.3 against
 * 131.3 MB/s on the dumps and 140.1 against 131.1 on Silesia -- 6.9% on both
 * -- for 1 068 bytes FEWER over the dumps and 9 275 more over Silesia (0.044
 * of a point). The gate answers one question about the block, and one window
 * covering the whole span answers it as well as two covering half each, at
 * half the pricing. */
#ifndef ZPSE_ASCII_PAGE_TILES
#define ZPSE_ASCII_PAGE_TILES 1u
#endif
#ifndef ZPSE_ASCII_MIN_BLOCK
#define ZPSE_ASCII_MIN_BLOCK 4096u
#endif

/* Should this block take the seven-bit form without a sweep?
 *
 * Three shapes are priced on the head of the block and again on its tail,
 * each with the probe's limit set to what the form would cost over the same
 * span. The probe stops a candidate the moment its running price crosses
 * the limit, so on prose each of these prices is a few tile classifications
 * rather than a transform of the whole window. A shape that comes back at
 * or under the limit is a shape that could beat the form, and then the
 * answer is no: the sweep keeps the block, and the tile loop still offers
 * the form per tile, so nothing is lost but the shortcut.
 *
 * The tail sample is what keeps a block of text with a compressible tail --
 * or the reverse -- out of the shortcut; both ends have to look like prose.
 * The three shapes are the ones a text block could plausibly want:
 * identity, DELTA8 (digits and numbers), and a width-4 transpose (tabular
 * text). Widths 2, 8 and 16 price within a few percent of width 4 on the
 * data this fires on, and they cost the same samples again. */
/* 1 iff the first `n` bytes are all one value.
 *
 * The gate compares three shapes' exact price against what the seven-bit
 * form would cost over the same span, and a plane that is one value prices
 * at its own floor -- far below seven bits a byte -- so a constant sample
 * cannot convict. Testing for it on the two kilobytes the gate would have
 * sampled anyway is what keeps a one-value block from paying a whole-block
 * scan and six priced planes to learn nothing: that scan is ~1.3 us of a
 * ~3 us encode (measured: `zeros` encodes at 20 GB/s through the gate and
 * 37 at MAX, which does not run it).
 *
 * Skipping the gate can only send the block down the ordinary sweep, which
 * is the path every non-ASCII and every non-gated block takes, so the bytes
 * written are decided by the sweep either way. */
static int zpse_block_head_one_value(const uint8_t *in, uint32_t n)
{
    uint64_t rep;
    uint32_t i;
    uint8_t v;

    if (in == 0 || n < 64u) {
        return 0;
    }
    v = in[0];
    rep = UINT64_C(0x0101010101010101) * (uint64_t)v;
    for (i = 0u; i + 8u <= n; i += 8u) {
        uint64_t w;

        memcpy(&w, in + i, sizeof w);
        if (w != rep) {
            return 0;
        }
    }
    for (; i < n; i++) {
        if (in[i] != v) {
            return 0;
        }
    }
    return 1;
}

static int zpse_block_ascii_gate(ZpseEncoder *e, const uint8_t *in,
                                 uint32_t n)
{
    static const uint8_t tids[3] = {0u, 1u, 4u};
    static const uint8_t scs[3] = {1u, 1u, 4u};
    static const uint16_t metas[3] = {0u, 1u, 0u};
    uint32_t sample = (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX)
                          ? (uint32_t)ZPSE_ASCII_PAGE_SAMPLE
                          : (uint32_t)ZPSE_ASCII_SAMPLE;
    uint32_t tiles = (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX)
                         ? (uint32_t)ZPSE_ASCII_PAGE_TILES
                         : (uint32_t)ZPSE_ASCII_SAMPLE_TILES;
    uint32_t pn = n < sample ? n : sample;
    uint32_t alim = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + zpse_ascii_len(pn);
    unsigned j;

    if (e == 0 || in == 0 || pn < 256u) {
        return 0;
    }
    for (j = 0; j < 3u; j++) {
        uint32_t h;
        /* Per sample, not per gate: the prefix is a property of the bytes it
         * reads, and the tail sample is different bytes from the head. */
        int prefix_ok = 1;

        if ((uint32_t)scs[j] > pn) {
            continue;
        }
        h = zpse_block_proxy_exact(e, in, pn, n, tids[j], scs[j], in,
                                   metas[j], tiles, alim, 1, &prefix_ok);
        if (h == UINT32_MAX || h <= alim) {
            return 0;
        }
        if (n > pn) {
            uint32_t t;

            prefix_ok = 1;
            t = zpse_block_proxy_exact(e, in + (n - pn), pn, n, tids[j],
                                       scs[j], in, metas[j], tiles, alim, 1,
                                       &prefix_ok);

            if (t == UINT32_MAX || t <= alim) {
                return 0;
            }
#if ZPSE_ASCII_PAGE_MID
            /* A third window, halfway along, on a block long enough for it not
             * to overlap either end (see ZPSE_ASCII_PAGE_MID). */
            if (n > 2u * pn) {
                prefix_ok = 1;
                t = zpse_block_proxy_exact(e, in + (n - pn) / 2u, pn, n,
                                           tids[j], scs[j], in, metas[j],
                                           tiles, alim, 1, &prefix_ok);

                if (t == UINT32_MAX || t <= alim) {
                    return 0;
                }
            }
#endif
        }
    }
    return 1;
}

/* The exact total the gated tile trial writes for a block of `n` bytes.
 *
 * A block the ASCII gate convicted has exactly one candidate -- the
 * identity, one segment -- and every tile of it takes the seven-bit form
 * (segment.c's `ascii_gate` branch), so its trial total is arithmetic: the
 * block envelope, the segment directory and header, and 2 + zpse_tile_ascii_len
 * for each of the ceil(n/256) tiles. Nothing in the segment encoder can
 * pick another form on such a plane: the dense constant record needs a
 * one-value plane, which the gate's own head probe has already ruled out;
 * the palette and literal forms are only priced by the general tile path,
 * which this block never enters; and the flat abort cannot fire because an
 * ASCII record of 234 bytes against a 256-byte tile pays for itself on the
 * first tile. So this is the number the trial produces, which is what lets
 * the match finder run BEFORE that trial and bound it (see the early pass
 * in zpse_block_encode) without changing a decision.
 *
 * Only called for a gated block. */
static uint32_t zpse_block_ascii_total(uint32_t n)
{
    uint32_t tc = (n + 255u) / 256u;
    uint32_t full = 2u + zpse_tile_ascii_len((uint16_t)256);
    uint32_t last = n - 256u * (tc - 1u);
    uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE +
                     (uint32_t)ZPSE_SEG_DIR_ENTRY_SIZE +
                     (uint32_t)ZPSE_SEGMENT_HEADER_SIZE;

    total += (tc - 1u) * full;
    total += 2u + zpse_tile_ascii_len((uint16_t)last);
    return total;
}

/* Remember which tile shape won, so the next block of this context can try
 * it first. Only tile forms are remembered: RAW, the LZ4 forms and HUF say
 * nothing about which transform the data wants, and a predictor that carried
 * them back would spend its next first trial re-learning that.
 *
 * The same call scores the run: a block whose winner is the shape the
 * previous block's winner had extends the stretch, anything else ends it.
 * That run length is what the history-confident stop reads.
 *
 * The three fields arrive directly from the writer, which already holds them:
 * re-parsing here meant a full zpse_block_header_parse -- a 28-byte CRC32-C
 * over the long form plus a dozen field checks -- on a header this encoder
 * just serialized and knows is valid. The decode side dropped its analogous
 * second CRC pass for the same reason (see the note in zpse_block_decode);
 * the encode side paid it once per block to read three bytes it can be handed.
 * (seq is not a history input, so it is not passed.) */
static void zpse_block_hist_set(ZpseEncoder *e, uint8_t block_flags,
                                uint8_t transform_id, uint8_t segment_count)
{
    if ((block_flags &
         (uint8_t)(ZPSE_B_RAW | ZPSE_B_LZ | ZPSE_B_LZ_TILE |
                   ZPSE_B_HUF)) != 0u) {
        e->hist_valid = 0u;
        e->hist_streak = 0u;
        return;
    }
    if (e->hist_valid != 0u && e->hist_tid == transform_id &&
        e->hist_sc == segment_count) {
        if (e->hist_streak != 255u) {
            e->hist_streak++;
        }
    } else {
        e->hist_streak = 0u;
    }
    e->hist_tid = transform_id;
    e->hist_sc = segment_count;
    e->hist_valid = 1u;
}

/* ---- Why there is no pre-sweep raw test -------------------------- *
 * A block that is going out raw still pays ten window prices and a trial to
 * find that out, and the page corpora make that look like the biggest single
 * saving left (44% of a shared-library dump is written raw at level 1). Two
 * designs were built and measured on this tree, and neither is here.
 *
 * The first put two cheap questions in front of the sweep -- sixteen spread
 * tiles all near-random by the encoder's own rule, and an LZ pass over the
 * head kilobyte finding no repeat -- and wrote RAW when both said so. It
 * is ratio-safe and useless: it fires on 91 blocks of the shared-library
 * dump's 29 297 (0.3%), because the pages that go raw are not noise, they
 * are simply not compressible by any shape the sweep prices, and that is
 * exactly what the price is for. It also cost Silesia 0.008% at level 6 and
 * 0.12% at level 9 (102 227 937 -> 102 236 237 and 94 239 250 -> 94 356 526),
 * where a full-size block it convicted did have a shape left to find.
 *
 * The second cut ZPSE_FLAT_TRIALS from 3 to 1 for the blocks the window price
 * already convicts. Ratio-neutral to 0.004% on Silesia at level 6 (+3 752
 * bytes of 102 MB) and -0.0002% at level 9, and no time it was possible to
 * see on this host: 293.9 against 298.0 MB/s at level 6, 190.9 against 184.4
 * at level 9, inside the same band a no-op build occupies. So the cap stays
 * at three trials, which is what the size corpus was tuned against. */

/* The RAW fallback's envelope, in one place: the plain block header, no
 * transform, no segments, the bytes themselves. */
static int zpse_block_write_raw(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                                uint32_t in_crc, uint32_t seq, int is_last,
                                uint8_t *out, uint32_t out_cap)
{
    uint32_t total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n;
    ZpseBlockHeader h;
    uint8_t flags = (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET |
                              (((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u)
                                   ? ZPSE_B_CRC
                                   : 0u));

    if (is_last) {
        flags |= ZPSE_B_LAST;
    }
    if (out_cap < total) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    zpse_block_fill_header(&h, flags, n, total, 0, 1, 0, seq, in_crc);
    zpse_block_header_serialize(out, &h);
    memcpy(out + ZPSE_BLOCK_HEADER_SIZE, in, n);
    total = zpse_block_shrink_page(out, n, total);
    /* RAW envelope: flags carries ZPSE_B_RAW, so the predictor invalidates. */
    zpse_block_hist_set(e, flags, 0u, 1u);
#if defined(ZPSE_BLOCK_STATS)
    /* The RAW fallback won, but it is not a transform win: the sweep found
     * nothing that paid, and the block went out raw. */
    zpse_block_win_tid[15u]++;
#endif
    return (int)total;
}

/* The 16-byte one-value page token, shared by the FLAT writer below and the
 * only path a constant page takes. No checksum flag: the short header has
 * nowhere to carry a payload CRC, so callers that asked for one never reach
 * this writer (zpse_block_encode leaves them on the long-header path). */
static int zpse_block_write_token(ZpseEncoder *e, const uint8_t *in,
                                  uint32_t n, uint32_t seq, int is_last,
                                  uint8_t *out, uint32_t out_cap)
{
    uint8_t flags = (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET |
                              (is_last ? ZPSE_B_LAST : 0u));

    if (out_cap < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    zpse_store_u16le(out + 0, (uint16_t)ZPSE_BLOCK_SYNC_TOKEN);
    out[2] = flags;
    out[3] = in[0];
    out[4] = 1u;
    out[5] = 0u;
    zpse_store_u16le(out + 6, (uint16_t)n);
    zpse_store_u16le(out + 8, (uint16_t)ZPSE_BLOCK_HEADER_SIZE_SHORT);
    zpse_store_u32le(out + 10, seq);
    zpse_store_u16le(out + 14,
                     (uint16_t)(zpse_crc32c(out, 14) & UINT32_C(0xFFFF)));
    /* Token flags carry ZPSE_B_RAW: the predictor invalidates. tid is the
     * fill byte (in[0]) and the token's segment count is fixed at 1. */
    zpse_block_hist_set(e, flags, in[0], 1u);
    return (int)ZPSE_BLOCK_HEADER_SIZE_SHORT;
}

/* ---- The FLAT page writer ------------------------------------------- *
 * One classifier call, one payload build, one self-check. There is no sweep,
 * no candidate list and no trial: the class picks the form directly. Every
 * compressed form is decoded back and compared with the input before it is
 * kept, so a wire detail owned by flat.c cannot make this file emit a wrong
 * page -- on any doubt the RAW form carries the bytes verbatim. */

/* Palette index of `v` via a 256-byte LUT built once per page. LUT[v] is the
 * index in the classifier's first-appearance palette (0xFF on a miss, which
 * cannot occur for a LOW_ALPHA page the classifier accepted). This replaces
 * a linear scan of up to 16 bytes per input byte. */
static uint8_t zpse_flat_pal_index(const uint8_t lut[256], uint8_t v)
{
    return lut[v];
}

/* BIT2: 8-byte palette, then two bits per position, low pair first. */
static uint32_t zpse_flat_build_bit2(const ZpsePageInfo *pi,
                                     const uint8_t *in, uint32_t n,
                                     const uint8_t lut[256], uint8_t *dst)
{
    uint32_t idx_bytes = (n + 3u) >> 2;
    uint32_t g;
    uint8_t i;

    memset(dst, 0, 8u);
    for (i = 0u; i < pi->nalpha; i++) {
        dst[i] = pi->alpha[i];
    }
    for (g = 0u; g < idx_bytes; g++) {
        uint32_t base = g << 2;
        uint8_t b = 0u;
        uint32_t j;

        for (j = 0u; j < 4u && base + j < n; j++) {
            uint8_t ix = zpse_flat_pal_index(lut, in[base + j]);

            b = (uint8_t)(b | (uint8_t)((uint32_t)ix << (2u * j)));
        }
        dst[8u + g] = b;
    }
    return 8u + idx_bytes;
}

/* NIBBLE4: 16-byte palette, then four bits per position, low nibble even. */
static uint32_t zpse_flat_build_nibble4(const ZpsePageInfo *pi,
                                        const uint8_t *in, uint32_t n,
                                        const uint8_t lut[256], uint8_t *dst)
{
    uint32_t idx_bytes = (n + 1u) >> 1;
    uint32_t g;
    uint8_t i;

    memset(dst, 0, 16u);
    for (i = 0u; i < pi->nalpha; i++) {
        dst[i] = pi->alpha[i];
    }
    for (g = 0u; g < idx_bytes; g++) {
        uint32_t base = g << 1;
        uint8_t b = 0u;

        if (base < n) {
            b = zpse_flat_pal_index(lut, in[base]);
        }
        if (base + 1u < n) {
            b = (uint8_t)(b | (uint8_t)((uint32_t)zpse_flat_pal_index(
                lut, in[base + 1u]) << 4));
        }
        dst[16u + g] = b;
    }
    return 16u + idx_bytes;
}

/* SUBLANE: [period][keepmask][templ[8]][sub[k*groups]][4 pad].
 *
 * A period position is written from the constant template when every lane
 * that shares it is constant (lane_diff == 0); otherwise a per-group byte is
 * stored in `sub`. zero_cnt confirms an all-zero lane, whose template byte
 * must be zero. Periods are tried from 8 down to 1 and the first payload that
 * decodes back to the input is kept; a mismatch moves on to the next period
 * and ultimately to RAW. */
static uint32_t zpse_flat_build_sublane(const uint8_t *in, uint32_t n,
                                        const ZpsePageInfo *pi,
                                        uint8_t *dst, uint8_t *scratch)
{
    static const uint32_t periods[4] = { 8u, 4u, 2u, 1u };
    uint32_t t;

    for (t = 0u; t < 4u; t++) {
        uint32_t period = periods[t];
        uint32_t groups = (n + period - 1u) / period;
        uint8_t keepmask = 0u;
        uint8_t templ[8];
        uint32_t rank[8];
        uint32_t k = 0u;
        uint32_t j, g;
        uint32_t len;

        memset(templ, 0, sizeof templ);
        for (j = 0u; j < period; j++) {
            uint32_t l;
            int constant = 1;
            int all_zero = 1;

            for (l = j; l < 32u; l += period) {
                if (pi->lane_diff[l] != 0u) {
                    constant = 0;
                    break;
                }
            }
            /* A constant position's template byte is its first byte; when
             * every lane in it is all zero (zero_cnt == lane length) that
             * byte is zero, which is the same value in a correct classifier. */
            for (l = j; constant != 0 && l < 32u; l += period) {
                uint32_t lane_len = (n + 31u - l) / 32u;

                if (pi->zero_cnt[l] != lane_len) {
                    all_zero = 0;
                }
            }
            templ[j] = (constant != 0 && all_zero != 0) ? 0u : in[j];
            rank[j] = (uint8_t)k;
            if (constant == 0) {
                keepmask = (uint8_t)(keepmask | (uint8_t)(1u << j));
                k++;
            }
        }
        len = 14u + k * groups;
        if (len >= n) {
            continue; /* RAW payload is smaller: no point keeping this form */
        }
        /* sub[] is indexed per group and, inside a group, by keep-bit rank. */
        for (g = 0u; g < groups; g++) {
            for (j = 0u; j < period; j++) {
                uint32_t at = g * period + j;

                if ((keepmask & (uint8_t)(1u << j)) != 0u && at < n) {
                    dst[10u + g * k + rank[j]] = in[at];
                }
            }
        }
        memset(dst + 10u + k * groups, 0, 4u);
        dst[0] = (uint8_t)period;
        dst[1] = keepmask;
        memcpy(dst + 2u, templ, 8u);

        if (zpse_flat_decode((uint8_t)ZPSE_FLAT_SUBLELANE, 0u, dst, len,
                             scratch, n) == ZPSE_OK &&
            memcmp(scratch, in, n) == 0) {
            return len;
        }
    }
    return 0u;
}

/* The one page path. A checksum request never reaches here: the 16-byte
 * FLAT/token headers have no field for a payload CRC, so those callers stay
 * on the long-header path in zpse_block_encode. */
static int zpse_block_write_flat(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                                 uint32_t seq, int is_last,
                                 uint8_t *out, uint32_t out_cap)
{
    ZpsePageInfo pi = { 0 };
    ZpseBlockHeader h;
    uint8_t scratch[ZPSE_BLOCK_SHORT_MAX];
    /* Build the payload off-buffer first. The builders can emit up to n
     * bytes before we know the chosen form's length, so writing them into
     * `out + 16` would smash a caller buffer whose out_cap is in [16, 16+n).
     * The public one-shot API passes out_cap straight through, so that case
     * is reachable; stage here and copy only after the capacity check. */
    uint8_t payload[ZPSE_BLOCK_SHORT_MAX];
    /* 256-byte value→palette-index LUT, built once for LOW_ALPHA pages so the
     * bit/nibble builders do a single load per byte instead of a ≤16 linear
     * scan. Byte-identical: the classifier guarantees every byte is in the
     * palette, so the 0xFF miss sentinel is never used. */
    uint8_t pal_lut[256];
    uint8_t codec = (uint8_t)ZPSE_FLAT_RAW;
    uint8_t param = 0u;
    uint32_t plen = n;
    uint32_t total;
    int cls;

    if (out_cap < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    cls = zpse_page_classify(in, n, &pi);
    if ((cls == (int)ZPSE_PAGE_ZERO || cls == (int)ZPSE_PAGE_ONE_VALUE) &&
        zpse_segment_plane_is_const(in, n)) {
        return zpse_block_write_token(e, in, n, seq, is_last, out, out_cap);
    }
    if (cls == (int)ZPSE_PAGE_LOW_ALPHA && pi.nalpha != 0u) {
        uint32_t p;

        zpse_pf_make_pal_lut(pi.alpha, pi.nalpha, pal_lut);
        if (pi.nalpha <= 4u) {
            p = zpse_flat_build_bit2(&pi, in, n, pal_lut, payload);
            if (p < n &&
                zpse_flat_decode((uint8_t)ZPSE_FLAT_BIT2,
                                 (uint8_t)(pi.nalpha - 1u), payload, p,
                                 scratch, n) == ZPSE_OK &&
                memcmp(scratch, in, n) == 0) {
                codec = (uint8_t)ZPSE_FLAT_BIT2;
                param = (uint8_t)(pi.nalpha - 1u);
                plen = p;
            }
        } else if (pi.nalpha <= 16u) {
            p = zpse_flat_build_nibble4(&pi, in, n, pal_lut, payload);
            if (p < n &&
                zpse_flat_decode((uint8_t)ZPSE_FLAT_NIBBLE4,
                                 (uint8_t)(pi.nalpha - 1u), payload, p,
                                 scratch, n) == ZPSE_OK &&
                memcmp(scratch, in, n) == 0) {
                codec = (uint8_t)ZPSE_FLAT_NIBBLE4;
                param = (uint8_t)(pi.nalpha - 1u);
                plen = p;
            }
        }
    } else if (cls == (int)ZPSE_PAGE_POINTER) {
        uint32_t p = zpse_flat_build_sublane(in, n, &pi, payload, scratch);

        if (p != 0u) {
            codec = (uint8_t)ZPSE_FLAT_SUBLELANE;
            plen = p;
        }
    }
    if (codec == (uint8_t)ZPSE_FLAT_RAW) {
        /* FLAT's four codecs cover constant, low-alpha and pointer pages.
         * An ASCII or generic page is often richly compressible by the LZ /
         * transform / tile ladder, which a raw FLAT envelope would throw
         * away (a text page drops from ~2.4:1 to 1.0:1). Return 0 to let
         * zpse_block_encode run that ladder instead. Only pages FLAT can
         * actually pack get the one-pass SIMD decode. */
        return 0;
    }
    total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT + plen;
    if (total > out_cap) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    memcpy(out + (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT, payload, plen);
    memset(&h, 0, sizeof h);
    h.sync = ZPSE_BLOCK_SYNC_FLAT;
    h.block_flags = (uint8_t)(ZPSE_B_RESET | (is_last ? ZPSE_B_LAST : 0u));
    h.uncompressed_size = n;
    h.compressed_block_size = total;
    h.transform_id = codec;
    h.transform_meta_size = param;
    h.block_sequence_number = seq;
    zpse_block_header_serialize_flat(out, &h);
    /* FLAT flags carry no RAW/LZ form bit, so the predictor records the
     * codec as the transform with a single segment (what parse_flat read). */
    zpse_block_hist_set(e, h.block_flags, h.transform_id, 1u);
    return (int)total;
}

/* The TEXT page form for an input the ASCII gate has already convicted: a
 * 16-byte header whose payload is a token stream. Returns the block length,
 * or 0 to decline (the ladder keeps the block). The caller stages the result
 * in e->cand and copies it to the winner buffer, so `out` here is scratch,
 * not necessarily the caller's output buffer. */
static uint32_t zpse_block_write_text(ZpseEncoder *e, const uint8_t *in,
                                      uint32_t n, uint32_t seq, int is_last,
                                      uint8_t *out, uint32_t out_cap,
                                      uint32_t plen)
{
    ZpseBlockHeader h;
    uint32_t total;
    uint32_t got;

    if (n < 2u) {
        return 0u;
    }
    /* `plen` is the payload length the caller already measured with
     * zpse_text_price (which also proved the input all-ASCII). 0 means this
     * writer was called standalone, so measure it here; the two call sites in
     * zpse_block_encode pass the value they already have, removing a second
     * ASCII scan of the whole block. Either way, reaching the pack below means
     * the input is proven ASCII. */
    if (plen == 0u) {
        plen = zpse_text_price(in, n);
    }
    if (plen == 0u) {
        return 0u;
    }
    total = 16u + plen;
    if (total > 65535u) {
        return 0u; /* the header's u16 compressed_block_size would overflow */
    }
    if (total > out_cap) {
        return 0u;
    }
    memset(&h, 0, sizeof h);
    h.sync = ZPSE_BLOCK_SYNC_TEXT;
    h.block_flags = (uint8_t)(ZPSE_B_RESET | (is_last ? ZPSE_B_LAST : 0u));
    h.uncompressed_size = n;
    h.compressed_block_size = total;
    h.block_sequence_number = seq;
    h.header_size = (uint8_t)ZPSE_BLOCK_HEADER_SIZE_SHORT;
    zpse_block_header_serialize_text(out, &h);
    /* Input is proven ASCII and plen == n-(n>>3) == the pack's exact fill, so
     * skip encode's redundant ASCII scan. `got != plen` is the length check:
     * each pack kernel writes exactly n-(n>>3) bytes for n>=1 ASCII input. */
    got = zpse_text_encode_ascii(in, n,
                                 out + (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT,
                                 plen);
    if (got != plen) {
        return 0u;
    }
    /* No decode+memcmp self-check here. The old code unpacked the just-packed
     * bytes into e->tstage and compared them to `in` before publishing. That
     * pass is provably dead: zpse_text_encode is a deterministic, fixed-rate
     * pack of input already proven all-ASCII, and zpse_text_decode is its
     * exact inverse (both exhaustively verified over every group size and bit
     * pattern). With plen == n-(n>>3) the decode's only structural check
     * passes and decode(pack(in)) == in identically, so memcmp could never
     * fail. The check above (`got == plen`) already verifies the exact-length
     * invariant; the frame's 16-byte header CRC covers the header, not the
     * payload, so it is no substitute -- but it is also why dropping this
     * redundant round-trip cannot change a single output byte. The parity
     * suite (def/scalar/swap identical digests) guards the SIMD pack kernels. */
    /* TEXT flags carry no RAW/LZ form bit; its parse reports transform 0 and
     * one segment, which is what the predictor records. */
    zpse_block_hist_set(e, h.block_flags, 0u, 1u);
    return total;
}

int zpse_block_encode(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                      uint32_t seq, int is_last,
                      uint8_t *out, uint32_t out_cap)
{
    uint8_t tids[ZPSE_MAX_CAND];
    uint8_t scounts[ZPSE_MAX_CAND];
    uint8_t meta_bufs[ZPSE_MAX_CAND][ZPSE_MAX_CAND_META];
    uint16_t meta_lens[ZPSE_MAX_CAND];
    uint8_t idx[ZPSE_MAX_CAND];
    uint32_t in_crc;
    int ncand = 0;
    uint32_t best_total = 0;
    uint32_t best_idx = 0;
    uint32_t flat_min = 0u;
    int have_best = 0;
    int fails = 0;
    int sample;
    int sample_flat = 0;
#if ZPSE_HIST_STOP_STREAK > 0u
    /* The order-n predictor picked the trial order (the history match won the
     * 1/16 band against the probe's own winner). Read by the confident-stop
     * rule in the trial loop; the value is only meaningful once that
     * predictor block has run. Guarded like its only reader, so a build with
     * the rule off does not carry a set-but-unused variable. */
    int hist_first = 0;
#endif
    int oi;
    int rc;

    if (e == 0 || in == 0 || out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (e->tstage == 0 || e->cand == 0 || e->best == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n < ZPSE_MIN_BLOCK_SIZE || n > ZPSE_MAX_BLOCK_SIZE) {
        return ZPSE_BAD_BLOCK_SIZE;
    }

    /* ZPSE_ENC_LZ_FAST: the whole decision is the LZ4 pass. One pass writes
     * the canonical LZ4 form, and a block that pass cannot beat a verbatim
     * copy on goes RAW. No probe, no candidate list, no trial -- the point
     * is to spend the block's encode budget on one LZ parse. */
    if ((e->enc_flags & ZPSE_ENC_LZ_FAST) != 0u && e->lz != 0) {
        uint32_t f_crc = ((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u &&
                          n > (uint32_t)ZPSE_BLOCK_SHORT_MAX)
                             ? zpse_crc32c(in, n)
                             : 0u;
        uint32_t f_total = 0u;
        uint32_t f_cmds = 0u;
        int lrc = zpse_block_try_lz2(e, in, n, f_crc, is_last, seq, out,
                                     out_cap, &f_total, &f_cmds);

        (void)f_cmds;
        if (lrc == ZPSE_OK &&
            f_total < (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
            /* The stream ships; fill its count word before the page shrink
             * (and any later wrap) reads the payload. */
            zpse_block_lz_finish_count(out, f_total);
            f_total = zpse_block_shrink_page(out, n, f_total);
            /* The writer's header carries ZPSE_B_LZ: invalidate. */
            zpse_block_hist_set(e, ZPSE_B_LZ, 0u, 1u);
            return (int)f_total;
        }
        if (lrc != ZPSE_OK && lrc != ZPSE_OUTPUT_OVERFLOW) {
            return lrc;
        }
        {
            uint32_t rtotal = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n;
            ZpseBlockHeader h;
            uint8_t flags = (uint8_t)(ZPSE_B_RAW | ZPSE_B_RESET |
                                      (((e->enc_flags & ZPSE_ENC_CHECKSUM) !=
                                        0u)
                                           ? ZPSE_B_CRC
                                           : 0u));

            if (out_cap < rtotal) {
                return ZPSE_OUTPUT_OVERFLOW;
            }
            if (is_last) {
                flags |= ZPSE_B_LAST;
            }
            zpse_block_fill_header(&h, flags, n, rtotal, 0u, 1u, 0u, seq,
                                   f_crc);
            zpse_block_header_serialize(out, &h);
            memcpy(out + ZPSE_BLOCK_HEADER_SIZE, in, n);
            rtotal = zpse_block_shrink_page(out, n, rtotal);
            /* RAW fallback: flags carries ZPSE_B_RAW, so invalidate. */
            zpse_block_hist_set(e, flags, 0u, 1u);
            return (int)rtotal;
        }
    }

    /* FLAT is a one-pass SIMD fast path for the page shapes it can pack:
     * constants (a 16-byte token), low-alphabet pages (NIBBLE4/BIT2) and
     * pointer lanes (SUBLANE). The writer returns 0 for an ASCII or generic
     * page it cannot pack better than RAW; those fall through to the full
     * ladder below, which keeps their ratio (a raw FLAT envelope would
     * throw away most of a text page's compression). A checksum request
     * has nowhere to live in a 16-byte FLAT/token header, so it also uses
     * the ladder. None of the old page-option bits affects FLAT. */
    if (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX &&
        (e->enc_flags & ZPSE_ENC_CHECKSUM) == 0u) {
        int frc = zpse_block_write_flat(e, in, n, seq, is_last, out, out_cap);

        if (frc != 0) {
            return frc;
        }
    }

    /* Trial-level noise probe threshold: a trial whose only way to win is
     * the RAW envelope is abandoned once it has covered this many input
     * bytes without one tile paying for itself (zpse_segment_encode,
     * seg_flat_abort). A sixteenth of the block, floored at four tiles, and
     * only for blocks big enough for either to be worth measuring -- the
     * same 8 KiB gate the prescreen proxy uses -- so small blocks keep
     * exactly the trial structure they have always had. */
    flat_min = (n >= 8192u) ? (n / ZPSE_FLAT_MIN_DIV) : 0u;
    if (flat_min != 0u && flat_min < 1024u)
        flat_min = 1024u;

    /* The candidate list is built in slot order below and then reordered
     * and compacted in place, so every slot a candidate does not fill must
     * start at a defined value rather than whatever the stack held. */
    memset(tids, 0, sizeof tids);
    memset(scounts, 0, sizeof scounts);
    memset(meta_bufs, 0, sizeof meta_bufs);
    memset(meta_lens, 0, sizeof meta_lens);
    memset(idx, 0, sizeof idx);

    /* Trial 1: RAW_BYTES (the identity, as a single segment).
     *
     * Ordered FIRST because it carries the smallest envelope of any
     * candidate -- one segment and no transform metadata -- so it is the
     * one that can reach the absolute floor of 32 + 8 + 16 + 10 per tile.
     * Reaching a candidate's floor is what lets the sweep stop, so a
     * constant or all-one block costs exactly one trial. */
    tids[0] = 0;
    scounts[0] = 1;
    meta_lens[0] = 0;
    ncand = 1;

    /* Trial 2: DELTA8 with seed = first input byte. */
    tids[ncand] = 1;
    scounts[ncand] = 1;
    meta_bufs[ncand][0] = in[0];
    meta_lens[ncand] = 1;
    ncand++;

    /* Trials 3..: the transposed shapes, plain and delta, at every width the
     * block is long enough to support.
     *
     * The list used to offer ONE plain transpose and to pick its width from
     * n alone, so a 64 KiB block was only ever offered the width-16 shape.
     * That left the sweep's result on the table for every columnar corpus
     * whose record width is 4 or 8 bytes -- and, more importantly, it never
     * offered a DELTA-TRANSPOSE at all, although the format has always been
     * able to encode and decode ids 7..10 and tests/suites/test_transforms.c
     * has always covered them. A delta-transpose is the shape a smooth column
     * wants: the transpose puts one byte position per plane, and the delta
     * chain then runs down that plane, so a slow ramp becomes a plane of
     * near-zeros that the tile encoder can predict.
     *
     * Priced over the shootout's 18 corpora with tests/tools/cand_probe.c
     * (64 KiB blocks), best shape the OLD list could pick versus best shape
     * in the full set:
     *
     *   i32_seq      18904 ->  8996  DT4   -52.4%
     *   ts_us        20640 -> 13264  DT8   -35.7%
     *   sorted_near  21044 -> 16636  DT4   -20.9%
     *   decimal8     14386 -> 12672  T8    -11.9%
     *   ts_gapped    34960 -> 31680  T4     -9.4%
     *   i64_mixed    11644 -> 10804  DT8    -7.2%
     *
     * 13 of the 18 improved, mean 10.99% over those, and no corpus chose a
     * shape outside this list. The cost of the extra candidates is bounded by
     * the prescreen below, which prices them on a 16 KiB window and drops
     * everything more than ~6% behind the best estimate before any of them is
     * encoded in full.
     *
     * Seeds: a delta-transpose carries one seed byte per plane, and the chain
     * it seeds is p[j] = plane[j] - plane[j-1] with plane[-1] == seed, so the
     * first byte the plane writes is plane[0] - seed. Setting the seed to the
     * plane's own first byte makes that difference zero. Any seed round-trips
     * (it travels in the metadata), so only size depends on the choice, and
     * zero is the best of them. Plane s of a width-w transpose holds input
     * positions s, s+w, s+2w, ..., so its first byte is in[s] -- which is
     * exactly the rule DELTA8 already uses at width 1 (seed = in[0]).
     * Verified round-trip by the full suite and by test_transforms.c. */
    {
        unsigned j;

        for (j = 0; j < 4u; j++) {
            uint8_t w = (uint8_t)(2u << j);
            unsigned s;

            /* N < W is forbidden by the format, and in[] must hold w bytes
             * for the seeds below, so a block shorter than the width simply
             * does not get that pair of candidates. */
            if (n < (uint32_t)w) {
                continue;
            }
            if (ncand + 2 > (int)ZPSE_MAX_CAND) {
                break;
            }
            /* Plain transpose: no transform metadata. */
            tids[ncand] = (uint8_t)(3u + j);
            scounts[ncand] = w;
            meta_lens[ncand] = 0;
            ncand++;
            /* Delta-transpose: one seed per plane, each the plane's own
             * first byte, so the chain's first difference is zero. */
            tids[ncand] = (uint8_t)(7u + j);
            scounts[ncand] = w;
            for (s = 0; s < w; s++) {
                meta_bufs[ncand][s] = in[s];
            }
            meta_lens[ncand] = w;
            ncand++;
        }
    }

    /* Every candidate writes the same decoded CRC into its header (it is
     * the CRC of the uncompressed block), so it is computed once here
     * instead of once per trial over the same 64 KiB. */
#if defined(ZPSE_BLOCK_STATS)
    zpse_block_blocks++;
    zpse_block_cands_built += (uint32_t)ncand;
#endif
    /* A one-value block can take its CRC arithmetically. The scan is the
     * segment encoder's own AVX2 one, and its answer is handed back to that
     * encoder by pointer (e->const_plane) so the identity trial skips its
     * own scan: a constant block is scanned once, not twice. The scan exits
     * at the first differing byte, so a block that is not constant pays one
     * compare and writes no pointer. */
    e->const_plane = 0;
    /* A short page is always rewritten by zpse_block_shrink_page, which
     * unconditionally drops ZPSE_B_CRC and zeroes the field. Its payload CRC
     * is never emitted, so do not pay crc32c over the whole block here: leave
     * in_crc at 0 for short pages, exactly as a no-checksum request does. */
    if ((e->enc_flags & ZPSE_ENC_CHECKSUM) != 0u &&
        n > (uint32_t)ZPSE_BLOCK_SHORT_MAX) {
        if (zpse_segment_plane_is_const(in, n)) {
            in_crc = zpse_crc32c_repeat(in[0], (size_t)n);
            e->const_plane = in;
        } else {
            in_crc = zpse_crc32c(in, n);
        }
    } else {
        /* No checksum asked for: nothing reads the field, so it is left
         * 0 and the pass over the block is not paid at all. */
        in_crc = 0u;
    }

    /* ---- ASCII gate -------------------------------------------- *
     * A block whose bytes are all below 0x80 has a form that costs seven
     * bits a byte -- 224 bytes per 256-byte tile -- with no predictor, no
     * mask and no residual codec. On prose that is the best this encoder
     * can do, and today it finds it the expensive way: the sweep prices
     * every shape, then encodes one in full, then the tile loop offers the
     * seven-bit form per tile and compares. Nine tenths of that work is
     * the sweep discovering that nothing predicts English.
     *
     * So the decision is taken here, before the sweep, and it is taken on
     * the evidence the sweep itself would have used: four tiles of the
     * identity, DELTA8 and width-4 transposed planes, priced exactly with
     * the probe's own limit set to what the seven-bit form would cost over
     * the same span. A probe that comes back ABOVE that limit has been
     * convicted -- it stopped early on a plane that could not win -- and if
     * all three shapes are convicted there is nothing left for the sweep to
     * try: a transform is one of those three or a width the sample did not
     * see, and the widths all price alike on the data this fires on.
     *
     * The gate never fires on data that compresses: a constant or periodic
     * ASCII plane prices at its own floor, far below seven bits a byte, so
     * the sample does not convict and the block keeps the sweep it has
     * always had.
     *
     * Every level gates, MAX included. MAX used to opt out on the argument
     * that it has the LZ form, which beats seven bits on prose, and that
     * the trial is the point of the level -- but the gate does not compete
     * with that trial: it decides only what the TILE side of a block costs
     * (one pack of the seven-bit form instead of a priced sweep), and
     * zpse_block_lz_level reads the level, not e->ascii_gate, so MAX still
     * runs the match finder and its stream still replaces the gated one
     * when it is smaller (see the candidate below). Opting out therefore
     * bought no ratio and cost the sweep on every convicted block: measured
     * at MAX, text_en 88 -> 294 MB/s, json_rows 101 -> 407, var_string 91 ->
     * 410 on 4 MiB blocks, with the streams identical -- and identical too
     * over the 779-case size corpus at levels 5, 7, 8 and 9. */
#if defined(ZPSE_BLOCK_STATS)
    {
        uint64_t t_gate0 = blk_rdtsc();
#endif
    e->ascii_gate = 0u;
    if (n >= (uint32_t)ZPSE_ASCII_MIN_BLOCK &&
        !zpse_block_head_one_value(
            in, (n < (uint32_t)ZPSE_ASCII_SAMPLE) ? n
                                                  : (uint32_t)ZPSE_ASCII_SAMPLE) &&
        zpse_ascii_scan(in, n) && zpse_block_ascii_gate(e, in, n)) {
        e->ascii_gate = 1u;
#if defined(ZPSE_BLOCK_STATS)
        zpse_block_gated++;
#endif
        /* Identity only: one trial, which now packs instead of searching,
         * and no pricing pass or second opinion to pay for first. */
        ncand = 1;
        sample_flat = 0;
    }
#if defined(ZPSE_BLOCK_STATS)
        zpse_block_gate_cycles += blk_rdtsc() - t_gate0;
    }
#endif

    /* The TEXT form is offered only to a gated block with one candidate: the
     * gate has already proved the input is plain text. Price it once here so
     * the final commit can compare against the sweep without a second scan;
     * the writer repeats the price when it actually builds the block (its
     * signature is fixed and it must stay callable on its own). A token
     * stream whose envelope would overflow the header's u16 size declines. */
    int text_eligible = (e->ascii_gate != 0u) && (ncand == 1) &&
                        (n >= (uint32_t)ZPSE_ASCII_MIN_BLOCK);
    uint32_t text_total = 0u;
    int text_won = 0;

    if (text_eligible) {
        uint32_t tp = zpse_text_price(in, n);

        if (tp == 0u || 16u + tp > 65535u) {
            text_eligible = 0;
        } else {
            text_total = 16u + tp;
        }
    }

    /* Attempt order: identity first unless the sample says DELTA8's plane
     * is much more predictable, and never anything but a swap of those
     * two (see the argument above the sample). The sample settles one
     * more thing: whether the proxy below is worth running at all. */
    {
        int i;

        for (i = 0; i < ncand; i++) {
            idx[i] = (uint8_t)i;
        }
    }
#if defined(ZPSE_BLOCK_STATS)
    {
        uint64_t t_sample = blk_rdtsc();

        sample = (ncand >= 2) ? zpse_block_delta_trial_first(in, n) : 0;
        zpse_block_sample_cycles += blk_rdtsc() - t_sample;
    }
#else
    sample = (ncand >= 2) ? zpse_block_delta_trial_first(in, n) : 0;
#endif
    if (ncand >= 2) {
        int swap = (sample == 1);

#if defined(ZPSE_FORCE_TRIAL_SWAP)
        /* Test hook: take the other decision wherever a decision exists,
         * so a build with this macro attempts these two candidates in the
         * opposite order from the default build for every block. The
         * parity suite builds test_parity.c and test_fuzz.c this way and
         * requires their streams and digests to be identical to the
         * default build's, which is what proves the order neutral. */
        swap = swap ? 0 : 1;
#endif
        if (swap) {
            zpse_block_swap_cand(tids, scounts, meta_bufs, meta_lens, idx, 0,
                                 1);
        }
    }

    /* Prune with the prescreen proxy: price every candidate cheaply and
     * fully encode only those within ~6% of the best estimate (always at
     * least one). A full trial is a classify + mode search + pack of every
     * tile, so where one transform is clearly better this skips the rest.
     * The decision is a pure function of the input, so it is identical in
     * the scalar, SIMD and swapped-order builds.
     *
     * sample == 2 means the identity's plane predicts every sampled tile,
     * which reads like "its trial is about to reach the global floor and
     * end the sweep on the first attempt, so pricing the other candidates
     * is pure cost". It is not, and the pass below is deliberately NOT
     * gated on it: measured, skipping the pricing pass when sample == 2
     * costs 1.95x on the zeros_burst case of the columnar corpus (453 ->
     * 232 MB/s, 64 KiB blocks) and 1.25x on sparse_mix. The reason is the
     * one that makes the two forms different: what ends a sweep early is
     * the pricing loop's own break when the identity PRICES at its floor,
     * and a block whose zeros start after a noise head never gets that
     * price while its identity trial still cannot reach the floor either.
     * With no prices there is no break and the trial loop walks the whole
     * survivor list; with them, the probe's first candidate settles it. */
#ifndef ZPSE_KEEP_EXACT_SWEEP
    /* ---- Decide from a sample, then encode once ---------------- *
     * DEFAULT; build with -DZPSE_KEEP_EXACT_SWEEP for the old sweep, which
     * encodes every survivor in full and keeps the smallest stream. A
     * trial is a classify, a mode search and a pack of every tile, so the
     * candidates the prescreen cannot separate -- a 4-bit alphabet, a
     * nibble-packed record, a text block, where all of them miss about as
     * often as each other -- cost a whole block encode each for a
     * difference of a few bytes.
     *
     * This prices every survivor exactly instead: the same classify, mode
     * search and pack on ZPSE_DECIDE_TILES tiles spread over the block,
     * folded up to the block's tile count. The winner is then encoded
     * once. Measured 1.65-2.07x on the lowentropy row of the bench (949k ->
     * 575k-510k cycles per 64 KiB block) and byte-identical on all 768 cases
     * of tests/clen_corpus.c, INCLUDING their CRC32-Cs.
     *
     * The price is a sample, and that is the trade. On a block whose tiles
     * are not statistically alike -- tests/test_fuzz.c shape 16, regions of
     * random texture at random boundaries -- a 16-tile sample can pick the
     * wrong transform outright: 74 of that suite's 2,718 cases moved by one
     * byte and one (8 KiB of random mixed regions) went from 1,735 to 3,645
     * bytes. Every case of tests/clen_corpus.c is byte-identical. See the
     * README for what that costs and why it is worth it.
     *
     * Ordering keeps the sweep's own tie-break: the smallest price wins and
     * an exact tie goes to the earlier candidate, so a candidate can only
     * be picked over one it would have lost to on a strict inequality. A
     * price that cannot be computed never wins over one that can, and if
     * nothing can be priced the list is left alone and the sweep runs
     * exactly as it always did. */
    /* Nothing is pruned on the cheap proxy any more. It is a prescreen-shaped
     * estimate -- the ideal packing rate implied by each plane's miss count --
     * and it cannot order shapes that pack BELOW that rate. On a block that is
     * zero except for one byte every 256 it priced the width-4 transpose at
     * 448 bytes per 16 KiB and never priced the width-16 one at all: that
     * candidate's own floor (624 bytes, a valid bound) already exceeded the
     * margin around 448, so it was dropped unmeasured, while over the whole
     * block it is the cheapest shape there is. Widening the margin does not
     * fix that; it only moves which blocks the estimate gets wrong, and the
     * same margin is what kept decimal8's width-8 transpose, which is the
     * shape that block actually wants.
     *
     * What replaces it is one exact pass: every candidate is priced with the
     * segment encoder's own classify/mode/pack on a spread of tiles, with the
     * running verdict fed down as the probe's limit so a candidate the best
     * price has already beaten by the margin stops on the plane that convicted
     * it. The pass keeps the candidates within the margin (a like-for-like
     * comparison) and orders them by price, which is the order the trial loop
     * below walks. */
    /* Smallest block the pricing pass runs on.
     *
     * This was 8 KiB, on the argument that a short block's trial is cheap
     * enough that pricing cannot pay for itself. That is true of ONE trial
     * and false of the ten a block actually builds: below the gate the
     * sweep prices nothing, so every candidate goes to a full trial. On the
     * heap corpus a 4 KiB block runs 9.74 trials per block against 1.00 on a
     * 64 KiB block, and those trials are 97.8% of its encode cycles
     * (203,735 cycles a block, ~55 us, at 74 MB/s -- against 620 MB/s for
     * the same data in 64 KiB blocks). The pricing pass it skips costs one
     * tile per plane of each candidate, which is exactly the estimate that
     * makes a 64 KiB block run one trial instead of ten.
     *
     * So the gate is now the smallest block the sweep can see, not a quarter
     * of the format's maximum. Measured on 4 KiB blocks, gate 8192 -> 4096:
     * trials a block 9.74 -> 1.00, encode 73 -> 138 MB/s on a Python heap,
     * 175 -> 379 on a C heap and 117 -> 149 on shared-library pages at level
     * 1; 64 -> 82, 140 -> 177 and 87 -> 89 at level 6. The probe is 73% of
     * what is left, the trials 24%. It costs ratio, because a price on a
     * short block can drop a survivor a trial would have kept: 0.19, 0.43 and
     * 0.33 points at level 1 and 0.01, 0.18 and 0.07 at level 6. That is
     * inside what the size gate tolerates -- tests/suites/test_sizes.c
     * reports grew=0 over its 768 cases and the whole suite passes -- and on
     * 64 KiB blocks the two builds are byte-identical (35.78% either way). */
#ifndef ZPSE_PRICE_MIN_N
#define ZPSE_PRICE_MIN_N 4096u
#endif
/* Candidates priced on a block whose identity window prices at raw at both
 * ends of it -- a block the sweep has just been told does not compress in the
 * shape it tries first, at either end.
 *
 * The pricing pass is what a page-at-a-time encode spends most of its cycles
 * on (see ZPSE_PRICE_MIN_N): 68% of a plain page's cycles, and 41% of the
 * shared-library dump's pages end raw at level 1. On such a block the pass has
 * one question to answer -- can anything compress this? -- and the identity
 * has already answered no for the common case. Two probes keep it from being
 * wrong about the uncommon one: the tail window separates a flat block from
 * one with a noise prefix ("noise then a constant"), and the width-4 pair
 * separates it from a record whose stride the identity cannot see (every
 * fourth byte random, the rest a small ramp). A block either probe can
 * compress is priced in full; only a block that fails both keeps this many
 * candidates, and the trials behind them decide it (ZPSE_FLAT_TRIALS via
 * sample_flat). Without the two probes the same cut was 10 to 22% faster and
 * turned 66 of tests/suites/test_sizes.c's 768 cases from compressed to raw.
 *
 * It is armed at every level, and that is the second measurement here: an
 * earlier version ran only where the sweep is cheap (!every_plane), on the
 * argument that BALANCED's every-plane pricing is what decides that level's
 * ratio and the cut was not worth what it saved. With both probes in place
 * it is a win there too -- 4 KiB BALANCED reads 117.1 against 111.0 MB/s on
 * the dumps, 109.0 against 92.5 on the mixed workload and 119.6 against
 * 111.7 on Silesia, with packed bytes 0.009, 0.047 and 0.067% SMALLER -- so
 * the gate is gone and the two probes alone decide.
 *
 * It is a PAGE rule, and the last check of that is the measurement above it:
 * on 64 KiB blocks at BALANCED the same cut costs 0.35% of packed bytes
 * (80,583,972 -> 80,868,755 over the dump corpus) and 5% of rate, because a
 * big block's body is where the sweep's trials earn what they cost. So the
 * cut is gated to n <= ZPSE_BLOCK_SHORT_MAX, which is the size a page is.
 *
 * Pricing more of a flat page's field was measured at 2 and 3 and changes
 * nothing: the dumps are byte-identical (0.0002 points), the mixed workload is
 * 0.006 points larger at 2, and Silesia 0.034 larger, with no rate difference.
 * A page whose head and tail both price at raw is written by the identity or
 * goes out raw; the candidates behind it are not what that decision is waiting
 * on, which is the same finding as the cap's. */
#ifndef ZPSE_FLAT_PRICE
#define ZPSE_FLAT_PRICE 1
#endif
    if (ncand > 1 && n >= ZPSE_PRICE_MIN_N) {
        uint32_t ex[ZPSE_MAX_CAND];
        uint32_t pc[ZPSE_MAX_CAND];
        uint8_t order[ZPSE_MAX_CAND];
        int nord = 0;
        /* Price on the first 16 KiB only: staging a transform costs about
         * as much as the estimate itself, and a block's structure (record
         * width, delta-friendliness) is the same throughout, so a quarter
         * of it orders the candidates just as well. (8 KiB was measured
         * too: it is faster, but it mis-ranks a block whose structure only
         * appears past the first quarter -- a long noise prefix in front
         * of a compressible body went to RAW -- so the window stays at
         * 16 KiB.) */
        /* The window the candidates are priced on: the head of the block.
     *
     * It was 16 KiB -- a quarter of a 64 KiB block -- and it is now 12 KiB,
     * because the price of the window is paid once per candidate and the
     * sweep prices ten of them. Cutting it by a quarter is 5.4% of the whole encode
     * over the columnar corpus, measured with the two builds alternating
     * (698 -> 736 MB/s mean, ratio unchanged at 3.96:1, decode
     * unchanged). The window has to be big enough for the price to
     * mean something: the sizes corpus is what catches a window that
     * mis-ranks, and it did not move. */
        uint32_t speed = e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK;
        /* One window for every level. FASTEST was given its own, 8 KiB
         * against the others' 12 (the window is paid once per candidate, and
         * FASTEST acts on a single candidate), and it is not worth it: on the
         * geo_point corpus the shorter window picks a shape that is 13%
         * larger AND decodes at 1.9 GB/s against the 5.1 the 12 KiB window
         * picks (13.06:1), because 8 KiB is not enough of a burst-heavy
         * columnar block to see the repeats that rank it. The speed it bought
         * was ~4%, inside the run-to-run noise of this machine. */
        uint32_t pn = n < (uint32_t)ZPSE_PROBE_WINDOW
                          ? n
                          : (uint32_t)ZPSE_PROBE_WINDOW;
#if ZPSE_PROBE_SHORT_TILES > 0
        /* A short block on a CHEAP RUNG is priced on a sample of itself (see
         * ZPSE_PROBE_SHORT_TILES). Only a short block: a 64 KiB block's
         * window is already a fraction of the block, and shrinking it is the
         * separate trade ZPSE_PROBE_WINDOW records. Where every plane is
         * priced the block's own window is kept: it is the exact price of a
         * page, it ranks the candidates on the whole of one, and the trials
         * behind the price repair what the extra 2 KiB costs. */
        if (zpse_enc_speed(speed) < ZPSE_ENC_SPEED_EVERY_PLANE_MIN &&
            n < (uint32_t)ZPSE_PROBE_WINDOW &&
            pn > (uint32_t)ZPSE_PROBE_SHORT_TILES *
                     (uint32_t)ZPSE_TILE_SIZE) {
            pn = (uint32_t)ZPSE_PROBE_SHORT_TILES *
                 (uint32_t)ZPSE_TILE_SIZE;
        }
#endif
#if ZPSE_PROBE_PAGE_TILES > 0
        /* The same sample on an every-plane level, where the shipped arm is
         * the whole page (see ZPSE_PROBE_PAGE_TILES). */
        if (zpse_enc_speed(speed) >= ZPSE_ENC_SPEED_EVERY_PLANE_MIN &&
            n < (uint32_t)ZPSE_PROBE_WINDOW &&
            pn > (uint32_t)ZPSE_PROBE_PAGE_TILES *
                     (uint32_t)ZPSE_TILE_SIZE) {
            pn = (uint32_t)ZPSE_PROBE_PAGE_TILES *
                 (uint32_t)ZPSE_TILE_SIZE;
        }
#endif
        /* BALANCED and deeper price every plane of a candidate, which is
         * what ranks a transpose correctly; the cheap levels price a stride
         * of them and fill the rest in (see zpse_block_exact_sum). */
        int every_plane = zpse_enc_speed(speed) >= ZPSE_ENC_SPEED_EVERY_PLANE_MIN;
        int pick = -1;
        /* The head window turned out NOT to describe the block: it priced
         * at the raw envelope and the tail window was the one that
         * compressed (or could not be priced at all). The near-tie prune
         * below must not trust a price on a window like that. */
        int head_misleads = 0;
        /* A survivor was kept because its CONTENT beat the winner's, not
         * because its total was close (see the compaction). Such a field
         * is not a tie, however close the totals are. */
        int content_kept = 0;
        /* Prefix pricing is tried on this block until a candidate's prefix
         * cannot answer (a plane whose first tile is all-hit may be a
         * constant plane, which only the whole plane settles). On a block
         * whose planes really are constant that is the first transpose
         * candidate it prices, and every candidate behind it -- and the
         * ones after it on the same block -- uses the full pass, so the
         * block pays one wasted prefix instead of one per shape. */
        int prefix_ok = 1;
        /* The identity's global floor over this window: a pure function of
         * pn (one segment, no metadata, so it is the smallest total any
         * candidate of this block could reach). See where it is filled. */
        uint32_t cfloor;
        /* The identity priced at raw at both windows: cut the list. */
        int flat_head = 0;
        /* The tail-window price of the identity, computed once in the sweep
         * below. When the identity turns out to be the winner, the raw-fallback
         * test prices that same 12 KiB window a second time with the very same
         * arguments; reuse this value instead. */
        uint32_t flat_head_tail_tp = UINT32_MAX;
        int have_flat_head_tail_tp = 0;
        /* Candidates the sweep has priced on this block, and the ones the page
         * cap (ZPSE_PAGE_CAND_CAP) left unpriced. Unpriced-by-the-cap is not
         * the same as unpriced: the sweep must never drop a candidate it could
         * not price, only one it chose not to. */
        int np = 0;
        uint8_t cap_skip[ZPSE_MAX_CAND];
        /* The head-window floor of each candidate, filled once before the
         * sweep. zpse_block_floor is a pure function of (pn, tid, sc,
         * meta_len) and none of those change while the candidates are priced,
         * so the per-candidate call in the lim != UINT32_MAX branch below is
         * answered from here instead of recomputed every time. */
        uint32_t pf[ZPSE_MAX_CAND];
        int ci;
        int k;

        /* The identity is priced first, whatever the attempt order is. Its
         * shape carries the smallest envelope (one segment, no transform
         * metadata) so its floor is the block's, and the identity wins every
         * tie on list index 0: the moment it prices at that floor the winner
         * is settled and the remaining candidates' staging and probes are
         * pure cost. That is the constant and all-hit case, where it was
         * 14.6 us of a 19.2 us encode before. The comparison below keeps the
         * sweep's own tie-break, so the pricing order cannot change who
         * wins -- only how much of the list is priced. */
        for (ci = 0; ci < ncand; ci++) {
            if (idx[ci] == 0u) {
                order[nord++] = (uint8_t)ci;
            }
        }
        for (ci = 0; ci < ncand; ci++) {
            if (idx[ci] != 0u) {
                order[nord++] = (uint8_t)ci;
            }
        }

        for (ci = 0; ci < ncand; ci++) {
            ex[ci] = UINT32_MAX;
            pc[ci] = UINT32_MAX;
            cap_skip[ci] = 0u;
        }
        /* Answered once instead of once per candidate: the loop below ends
         * the sweep the moment the identity prices at this floor, and the
         * floor itself is two variable-divisor divisions (pn over 256 and
         * over the dense record length) over a window none of the candidates
         * may change. Same value, same comparison, ten fewer calls on a
         * full candidate list. */
        cfloor = zpse_block_cand_floor(pn);
        {
            uint32_t seg_price = zpse_segment_price(e->enc_flags);
            int fi;

            for (fi = 0; fi < ncand; fi++) {
                pf[fi] = zpse_block_floor(pn, tids[fi], scounts[fi],
                                          meta_lens[fi], seg_price);
            }
        }
        for (k = 0; k < nord; k++) {
            const uint8_t *mb;
            uint32_t lim = UINT32_MAX;

            if (flat_head && n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX &&
                k >= (int)ZPSE_FLAT_PRICE) {
                break;
            }
            /* A page stops the sweep at the cap. The identity is priced
             * first, so it is always in the list the compaction chooses
             * from; what the cap costs is the shape a candidate behind it
             * would have won with.
             *
             * Only where every plane is priced (levels 5-9). On a cheap rung
             * the price IS the decision -- it has one or two trials and no
             * more, so it cannot repair a ranking the cap made coarse -- and
             * on the dumps the two shapes behind the cap are worth 2.5 points
             * there (level 1: 58.61% to 61.09%, level 2: 57.32% to 59.95%).
             * The speed is real -- three paired passes read 262.7 -> 334.6
             * MB/s at level 1 and 210.7 -> 274.7 at level 2, and the mixed and
             * Silesia corpora pay 0.09 and 0.05% of bytes for 16 and 21% --
             * but 2.5 points is five times BALANCED's 0.46 for the same fifth
             * of rate, and over twice the one-point line these page trades are
             * held to. */
            if (np >= (int)ZPSE_PAGE_CAND_CAP &&
                n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX &&
                zpse_enc_speed(speed) >= ZPSE_ENC_SPEED_EVERY_PLANE_MIN) {
                int j;

                for (j = k; j < nord; j++) {
                    cap_skip[order[j]] = 1u;
                }
                break;
            }
#if defined(ZPSE_BLOCK_STATS)
            uint64_t t_cycle = blk_rdtsc();
#endif

            ci = order[k];
            mb = meta_lens[ci] > 0 ? meta_bufs[ci] : in;
            /* The running verdict: a candidate the best price has already
             * beaten by the margin is offered only that margin, and the probe
             * stops it on the plane whose running bound crosses it. A
             * convicted candidate comes back as a finite bound above `lim`,
             * never UINT32_MAX, so it reads as priced-and-lost rather than
             * unpriced. */
            if (pick >= 0 && ex[pick] != UINT32_MAX) {
                lim = ex[pick] + (ex[pick] >> ZPSE_CONVICT_SHIFT);
            }
            /* Arithmetic floor before any staging: a shape's smallest
             * possible total over the window is a function of its plane
             * lengths alone (ten bytes per 256-byte tile, twelve per 4 KiB
             * dense record), so a candidate already past the limit cannot
             * be a survivor and the transform staging plus the plane prices
             * that would prove it are pure cost. The probe's own early exit
             * cannot see this: it stops a candidate only after its first
             * plane has been staged and priced, which is most of the price
             * of a width-16 shape. Reported as a finite price above `lim`
             * -- like a candidate the probe convicted -- so the compaction
             * below drops it rather than treating it as unpriced (UINT32_MAX
             * is kept on purpose: the sweep must never drop something it
             * could not price). */
            if (lim != UINT32_MAX) {
                uint32_t fl = pf[ci];

                if (fl != UINT32_MAX && fl > lim) {
                    ex[ci] = fl;
                    continue;
                }
            }
            ex[ci] = zpse_block_proxy_exact(e, in, pn, n, tids[ci],
                                            scounts[ci], mb, meta_lens[ci],
                                            ZPSE_DECIDE_TILES, lim,
                                            every_plane, &prefix_ok);
            np++;
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_cands_priced++;
            zpse_block_probe_cycles += blk_rdtsc() - t_cycle;
#endif
            if (ex[ci] == UINT32_MAX) {
                pc[ci] = UINT32_MAX;
                continue;
            }
            if (idx[ci] == 0u && scounts[ci] == 1u &&
                ex[ci] >= (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn) {
                uint32_t tp = (n > pn)
                    ? zpse_block_proxy_exact(e, in + (n - pn), pn, n,
                                             tids[ci], scounts[ci], mb,
                                             meta_lens[ci], ZPSE_DECIDE_TILES,
                                             UINT32_MAX, every_plane, 0)
                    : ex[ci];

                flat_head_tail_tp = tp;
                have_flat_head_tail_tp = 1;
                if (tp == UINT32_MAX ||
                    tp >= (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn) {
                    /* The width-4 pair, plain and delta: the stride shapes. */
                    int stride_wins = 0;
                    int pj;

                    for (pj = 0; pj < ncand; pj++) {
                        uint32_t sp;

                        if (scounts[pj] != 4u || ex[pj] != UINT32_MAX) {
                            continue;
                        }
                        sp = zpse_block_proxy_exact(
                            e, in, pn, n, tids[pj], scounts[pj],
                            meta_lens[pj] > 0 ? meta_bufs[pj] : in,
                            meta_lens[pj], ZPSE_DECIDE_TILES, UINT32_MAX,
                            every_plane, 0);
                        if (sp != UINT32_MAX &&
                            sp < (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn) {
                            stride_wins = 1;
                            break;
                        }
                    }
                    if (!stride_wins) {
                        flat_head = 1;
                    }
                }
            }
            /* The content price: what the sample actually measured, with the
             * candidate's fixed structural envelope -- block header, block
             * segment directory and one 8-byte segment header per plane --
             * taken back out. The sample can only see content; the envelope
             * is a constant it cannot judge, and it is what made the two
             * orders disagree on "spike every 256": the width-4 shape prices
             * at 404 + 112 bytes of envelope against 436 + 176 for the
             * width-8, so totals said 404 < 436 while the content said
             * 292 > 260 -- and the block is 1419 against 1211. A candidate
             * whose content beats the winner's can therefore not be dropped
             * on its total alone (see the prune below). */
            {
                uint32_t env = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + 16u +
                               16u * (uint32_t)scounts[ci];

                pc[ci] = ex[ci] > env ? ex[ci] - env : 0u;
            }
            if (pick < 0 || ex[ci] < ex[pick] ||
                (ex[ci] == ex[pick] && idx[ci] < idx[pick])) {
                pick = ci;
            }
            if (idx[ci] == 0u && scounts[ci] == 1u && ex[ci] == cfloor) {
                break;
            }
        }
        if (pick >= 0) {
            /* Keep what the winner has not beaten by the margin, and order
             * the survivors by price: the trial loop walks them in that order,
             * so the cheapest shape is attempted first and every later trial
             * is handed the tightest budget it could still win against.
             * Anything that could not be priced is kept and sorts last (it has
             * no price to order by, and the sweep must not drop it).
             *
             * Two things get a trial: a total within the margin, and a
             * candidate whose CONTENT beats the winner's (see where pc[] is
             * filled in). The second is the case the margin cannot cover --
             * a shape whose envelope is what puts it behind -- and it costs
             * one bounded trial only on blocks where the two orders actually
             * disagree. The winner is always kept, so the loss there is a
             * trial, never a different block. */
            uint32_t limit = ex[pick] + (ex[pick] >> ZPSE_CONVICT_SHIFT);
            int kept = 0;
            int win = -1;
            uint8_t win_idx = idx[pick];

            for (ci = 0; ci < ncand; ci++) {
                if (ci == pick ||
                    (ex[ci] == UINT32_MAX && cap_skip[ci] == 0u) ||
                    ex[ci] <= limit || pc[ci] < pc[pick]) {
                    if (ci == pick) {
                        win = kept;
                    } else if (ex[ci] != UINT32_MAX && ex[ci] > limit &&
                               pc[ci] < pc[pick]) {
                        content_kept = 1;
                    }
                    if (kept != ci) {
                        unsigned m;

                        tids[kept] = tids[ci];
                        scounts[kept] = scounts[ci];
                        meta_lens[kept] = meta_lens[ci];
                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            meta_bufs[kept][m] = meta_bufs[ci][m];
                        }
                        idx[kept] = idx[ci];
                        ex[kept] = ex[ci];
                    }
                    kept++;
                }
            }
            ncand = kept;
#if defined(ZPSE_BLOCK_STATS)
            /* The compaction above moved the winner to slot `win` (the
             * slot `kept` held when the winner was copied), so its price
             * is ex[win], not ex[pick] -- `pick` still names the slot the
             * winner had BEFORE the compaction. */
            {
                uint32_t best_price = ex[win];
                uint32_t worst = 0u;

                zpse_block_cands_kept += (uint32_t)kept;
                zpse_block_price_sum += best_price;
                for (ci = 0; ci < kept; ci++) {
                    if (ex[ci] != UINT32_MAX && ex[ci] > worst) {
                        worst = ex[ci];
                    }
                }
                zpse_block_margin_sum += worst - best_price;
            }
#endif
            /* Insertion sort on the survivors: at most ten of them, and a tie
             * keeps the original list order (idx ascending) so the attempt
             * order still cannot decide a tie. */
            {
                int a;

                for (a = 1; a < ncand; a++) {
                    int b = a;

                    while (b > 0 &&
                           (ex[b - 1] > ex[b] ||
                            (ex[b - 1] == ex[b] && idx[b - 1] > idx[b]))) {
                        uint8_t  st = tids[b - 1], ss = scounts[b - 1];
                        uint8_t  si = idx[b - 1];
                        uint16_t sl = meta_lens[b - 1];
                        uint32_t sx = ex[b - 1];
                        uint8_t  sb[ZPSE_MAX_CAND_META];
                        unsigned m;

                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            sb[m] = meta_bufs[b - 1][m];
                        }
                        tids[b - 1] = tids[b];
                        scounts[b - 1] = scounts[b];
                        idx[b - 1] = idx[b];
                        meta_lens[b - 1] = meta_lens[b];
                        ex[b - 1] = ex[b];
                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            meta_bufs[b - 1][m] = meta_bufs[b][m];
                        }
                        tids[b] = st;
                        scounts[b] = ss;
                        idx[b] = si;
                        meta_lens[b] = sl;
                        ex[b] = sx;
                        for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                            meta_bufs[b][m] = sb[m];
                        }
                        b--;
                    }
                }
                /* The sort moved the winner: find its slot again by the
                 * original list index it carries. */
                pick = win;
                for (a = 0; a < ncand; a++) {
                    if (idx[a] == win_idx) {
                        pick = a;
                        break;
                    }
                }
            }
            /* Does the probe's own winner fail to beat the raw form over the
             * window it priced? Then the block is a raw-fallback candidate and
             * the trial loop below is capped (ZPSE_FLAT_TRIALS). */
            sample_flat = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn <= ex[pick];
            /* ...and the same window at the OTHER end of the block, because a
             * block whose front is noise and whose back is not is exactly the
             * case a front-only probe calls incompressible: the "noise then
             * constant" shapes of the ratio corpus, where capping the trials
             * costs a quarter of the block's compressed size. One extra price,
             * of the candidate the front window already likes, is enough to
             * tell that apart from a block that is noise everywhere -- and
             * only the winner is priced, not the whole list. */
            if (sample_flat && n > pn) {
                uint32_t tp;

                if (have_flat_head_tail_tp && idx[pick] == 0u &&
                    scounts[pick] == 1u && tids[pick] == 0u &&
                    meta_lens[pick] == 0u) {
                    /* The identity won and its head priced at raw, so the
                     * sweep already priced this exact window with these exact
                     * arguments. */
                    tp = flat_head_tail_tp;
                } else {
                    const uint8_t *wmb = meta_lens[pick] > 0
                                             ? meta_bufs[pick] : in;

                    tp = zpse_block_proxy_exact(
                        e, in + (n - pn), pn, n, tids[pick], scounts[pick], wmb,
                        meta_lens[pick], ZPSE_DECIDE_TILES, UINT32_MAX,
                        every_plane, 0);
                }

                if (tp == UINT32_MAX ||
                    tp < (uint32_t)ZPSE_BLOCK_HEADER_SIZE + pn) {
                    sample_flat = 0;
                    head_misleads = 1;
                }
            }
        }
        if (pick >= 0 && pick != 0) {
            /* Move the exact-probe winner to slot 0 so the trial loop tries
             * the most likely winner first: its total becomes best_total and
             * the per-candidate floor exits discard shapes that provably
             * cannot win without a full block encode. ncand is deliberately
             * left at the prescreen survivor count -- every survivor is still
             * encoded in full (bounded by the running-best budget), so the
             * sweep keeps the true minimum even when the 16 KiB sample
             * mis-ranks the candidates (a noise prefix in front of a
             * compressible body, or two transforms within proxy error on a
             * uniform block). The tie-break still uses the original list
             * index carried in idx[], so the result stays order neutral. */
            unsigned m;
            uint8_t  tt = tids[0],  ts = scounts[0];
            uint16_t tl = meta_lens[0];
            uint8_t  ti = idx[0];
            uint8_t  tb[ZPSE_MAX_CAND_META];

            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                tb[m] = meta_bufs[0][m];
            tids[0] = tids[pick];
            scounts[0] = scounts[pick];
            meta_lens[0] = meta_lens[pick];
            idx[0] = idx[pick];
            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                meta_bufs[0][m] = meta_bufs[pick][m];
            tids[pick] = tt;
            scounts[pick] = ts;
            meta_lens[pick] = tl;
            idx[pick] = ti;
            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                meta_bufs[pick][m] = tb[m];
        }
        /* ---- A near-tied field is cut to its best two ------------------
         *
         * The survivor margin above keeps everything the winner has not
         * beaten by 1/16, which on a homogeneous column keeps almost the
         * whole list: every transposed shape prices the same record width
         * on the same data, so the candidates differ by a fraction of a
         * percent and the margin has no one to drop. The trial loop then
         * encodes each of them in full to find out what the price already
         * said -- and a full trial is ~6x a window price, so the tie is the
         * most expensive thing the sweep does.
         *
         * A field whose THIRD-best price is inside ZPSE_TIE_BAND of its best
         * is a field the price cannot order, and the block total it produces
         * differs by about the band. So the two cheapest survivors (the
         * winner, already in slot 0, and the next by price) are kept and the
         * rest stop here. This is the same trade the levels make one rung at
         * a time -- it just applies to the one shape of block where the rung
         * would otherwise spend every trial it has on a coin flip.
         *
         * Four things hold it back, each of them a case where a tied price is
         * known not to describe the block: a block short of the format's full
         * size (see ZPSE_TIE_MIN_N), a head window that the tail disproved, a
         * survivor kept because its CONTENT beat the winner's rather than its
         * total, and a field of fewer than four. The size gate in
         * tests/suites/test_sizes.c is what keeps those honest: without any
         * of them a tied field on "three-value alphabet" blocks grows by 24
         * bytes in 768 cases. */
        {
            const ZpseLevel *row = zpse_level_row(zpse_enc_speed(speed));
            uint32_t cap = (uint32_t)row->trials;
#if defined(ZPSE_TRACE_TRIALS)
            {
                int q;

                fprintf(stderr,
                        "sweep n=%u speed=%u lvl=%u ncand=%d cap=%u pick=%d "
                        "h=%d c=%d:",
                        n, (unsigned)speed,
                        (unsigned)zpse_enc_speed(speed), ncand,
                        (unsigned)cap, pick, head_misleads, content_kept);
                for (q = 0; q < ncand && q < (int)ZPSE_MAX_CAND; q++) {
                    fprintf(stderr, " [%u %u]", (unsigned)tids[q],
                            (unsigned)ex[q]);
                }
                fprintf(stderr, "\n");
            }
#endif

            /* Only a level that prices EVERY survivor can cut a field: the
             * cheap rungs already stop at their own trial count, and
             * cutting under them would only reorder the handful they try. */
            {
                uint32_t keep = (uint32_t)ZPSE_TIE_KEEP;
                uint32_t fmin = (uint32_t)ZPSE_TIE_MIN_FIELD;
                int cut_ok = (n >= (uint32_t)ZPSE_TIE_MIN_N) ? 1 : 0;

#if (ZPSE_TIE_SHORT_KEEP > 0)
                if (cut_ok == 0 && n < (uint32_t)ZPSE_TIE_SHORT_N) {
                    cut_ok = 1;
                    keep = (uint32_t)ZPSE_TIE_SHORT_KEEP;
                    if (n <= (uint32_t)ZPSE_TIE_PAGE_N) {
                        fmin = (uint32_t)ZPSE_TIE_SHORT_MIN_FIELD;
                    }
                }
#endif
                if (cut_ok != 0 && cap == 0u && ncand > (int)keep) {
                    /* The survivors are sorted by price after the compaction
                     * (ties in list order), so the survivor one past the cut
                     * is the cheapest of the ones to drop. */
                    uint32_t best = ex[0];
                    uint32_t third = ex[keep];

                    if (pick >= 0 && ncand >= (int)fmin && !head_misleads &&
                        !content_kept && third != UINT32_MAX &&
                        (row->tie_field != 0u ||
                         third - best <= best >> ZPSE_TIE_BAND_SHIFT)) {
                        /* ...but not the identity, whatever the price says.
                         * Its stream carries the smallest envelope any shape
                         * has and its trial is the cheapest the sweep can run
                         * (no staging), so a field it survives the margin in
                         * is a field whose cut is being decided by bytes the
                         * price cannot resolve: on a 4-bit-random page the
                         * whole-page price puts a transpose eight bytes ahead
                         * of it, and cutting the identity there grew the size
                         * gate's case 579 by 21 bytes. The survivor is moved
                         * into the slot the cut would have filled, so the
                         * field stays ordered by price. */
                        int ji = -1;
                        int a;

                        for (a = (int)keep; a < ncand; a++) {
                            if (idx[a] == 0u) {
                                ji = a;
                                break;
                            }
                        }
                        if (ji < 0) {
                            ncand = (int)keep;
                        } else {
                            uint8_t  tt = tids[keep], ts = scounts[keep];
                            uint8_t  ti = idx[keep];
                            uint16_t tl = meta_lens[keep];
                            uint32_t tx = ex[keep];
                            uint8_t  tb[ZPSE_MAX_CAND_META];
                            unsigned m;

                            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                                tb[m] = meta_bufs[keep][m];
                            tids[keep] = tids[ji];
                            scounts[keep] = scounts[ji];
                            idx[keep] = idx[ji];
                            meta_lens[keep] = meta_lens[ji];
                            ex[keep] = ex[ji];
                            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                                meta_bufs[keep][m] = meta_bufs[ji][m];
                            tids[ji] = tt;
                            scounts[ji] = ts;
                            idx[ji] = ti;
                            meta_lens[ji] = tl;
                            ex[ji] = tx;
                            for (m = 0; m < ZPSE_MAX_CAND_META; m++)
                                meta_bufs[ji][m] = tb[m];
                            ncand = (int)keep + 1;
                        }
                    }
                }
            }
            /* The trials this level is willing to spend, from the level
             * table: FASTEST keeps the probe's winner only, each cheap rung
             * above it buys one more trial, and BALANCED and deeper price
             * every survivor in full. Buying a trial is the only thing the
             * cheap rungs do, so the table is also the whole difference
             * between 1, 2, 3 and 4. */
            if (pick >= 0 && cap != 0u && (uint32_t)ncand > cap) {
                ncand = (int)cap;
            }
        }
        /* ---- Order-n first trial: the block's own history -------------- *
         * A file is not 3234 independent blocks: they come from one source,
         * and the transform that won the last one usually wins the next. The
         * sweep's own first choice is a 16 KiB price, and measurement on
         * Silesia at the default level says that choice is wrong on a quarter
         * of blocks -- when it is wrong it is wrong by 17.5 KB a block, which
         * is what the later trials are for. So when the price cannot separate
         * the previous block's shape from this block's winner (within 1/16),
         * the previous winner is attempted first: the price has no information
         * there and the history has some.
         *
         * It is applied HERE -- after the set is decided and after every
         * decision that reads a price -- and that is the whole point.
         * Promoting the hint to `pick`, which is where this used to live, let
         * a candidate 1/16 worse by price decide how many trials the block
         * got, because `pick` feeds the flat-block cap and the tied-field cut.
         * The same bytes encoded after different neighbours then came out a
         * few bytes different -- 15,098,769 on one worker against 15,101,233
         * on four for the same 20 MB -- so the worker count was choosing the
         * stream, and turning the hint off cost 0.016% of the corpus
         * (104,050,674 against 104,066,700 on 202 MiB). Here the caps, the cut
         * and the winner are all the price's and the hint only changes which
         * survivor is ATTEMPTED first. ncand is untouched, every survivor is
         * still priced and encoded, the price and floor move with the slot,
         * and the tie-break still uses the original list index in idx[], so
         * the block is the same with the hint on and off (test_sizes holds the
         * size, the parity suite the bytes). */
        if (ZPSE_HIST_REORDER != 0u && e->hist_valid != 0u && ncand > 1) {
            int h = -1;
            int hi;

            for (hi = 1; hi < ncand; hi++) {
                if (tids[hi] == e->hist_tid && scounts[hi] == e->hist_sc) {
                    h = hi;
                    break;
                }
            }
            if (h > 0 && ex[h] != UINT32_MAX && ex[0] != UINT32_MAX &&
                ex[h] <= ex[0] + (ex[0] >> 4)) {
                uint8_t  st = tids[0],  ss = scounts[0],  si = idx[0];
                uint16_t sl = meta_lens[0];
                uint32_t sx = ex[0];
                uint8_t  sb[ZPSE_MAX_CAND_META];
                unsigned m;

                for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                    sb[m] = meta_bufs[0][m];
                }
                tids[0] = tids[h];
                scounts[0] = scounts[h];
                idx[0] = idx[h];
                meta_lens[0] = meta_lens[h];
                ex[0] = ex[h];
                for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                    meta_bufs[0][m] = meta_bufs[h][m];
                }
                tids[h] = st;
                scounts[h] = ss;
                idx[h] = si;
                meta_lens[h] = sl;
                ex[h] = sx;
                for (m = 0; m < ZPSE_MAX_CAND_META; m++) {
                    meta_bufs[h][m] = sb[m];
                }
#if ZPSE_HIST_STOP_STREAK > 0u
                hist_first = 1;
#endif
            }
        }
    }
#endif /* !ZPSE_KEEP_EXACT_SWEEP */

    /* Full block encodes actually spent. The flat-block cap below counts
     * SLOTS, and a slot can be skipped (its floor cannot beat the winner)
     * without a trial being run -- but the two paths that leave the loop
     * early after a trial do spend one. Counting the trials themselves is
     * what the cap means: "the sweep has spent N full encodes". */
    int ntried = 0;
    /* The gate's single trial writes straight into `out` (see the loop). */
    int direct_out = (e->ascii_gate != 0u && ncand == 1) ? 1 : 0;
    /* Set once the match finder has run, wherever it ran (see the early
     * pass below); the candidate block after the trials then skips it. */
    int lz_done = 0;
    /* Head sample and tile price measured by the full-size screen near the
     * top of the sweep, for slot 0's shape. {0, UINT32_MAX} means that screen
     * did not run or did not reach a price; the late screen then measures it
     * itself. When it did, the late screen reuses these for the identical
     * first-sample part and only re-asks the best_total-dependent part. */
    uint32_t lz_cache[2] = {0u, UINT32_MAX};
    uint8_t *tdst = e->cand;
    uint32_t tcap = (uint32_t)ZPSE_E_SZ_CAND;
    /* Per-slot floor, filled once before the trial loop (see there). */
    uint32_t fl[ZPSE_MAX_CAND];
    /* Suffix minima of fl[]/idx[] over the remaining candidate slots. The
     * post-trial scan needs the minimum floor and "is an earlier-indexed
     * candidate left" over oi+1..ncand-1; both are answered in O(1) from
     * these, with sentinels at [ncand] for an empty tail. */
    uint32_t suf_fl[ZPSE_MAX_CAND + 1];
    uint8_t suf_idx[ZPSE_MAX_CAND + 1];

    /* ---- The match finder before the trial, on a gated block -------- *
     *
     * A gated block has one candidate and its trial writes the seven-bit
     * form, whose total is known before the trial runs (see
     * zpse_block_ascii_total): 59 960 bytes of a 64 KiB block. The LZ
     * candidate is offered with that total as the running verdict, so the
     * screen and the pass see exactly the arguments they see today, after
     * the trial.
     *
     * What changes is the order and the budget. When the pass wins, its
     * stream becomes the winner before the tile trial starts, so the trial
     * is handed the LZ total as its byte budget instead of the RAW envelope
     * and stops as soon as it writes past it -- which on prose is 40% of the
     * block rather than all of it. The stream written is the same: the LZ
     * form won the comparison either way, so the tile stream it replaces was
     * discarded either way, and a trial the budget stops cannot win (its
     * total only grows). When the pass loses, nothing is decided early and
     * the trial runs exactly as it does today.
     *
     * This is why it is exact: the screen's verdict is a function of
     * (sample, tile price, best_total), and best_total here is the number
     * the trial was going to produce -- not a guess. A block whose trial
     * could deviate from zpse_block_ascii_total never reaches this code
     * (the gate clears, or it is not the identity-only shape). */
    if (e->ascii_gate != 0u && ncand == 1) {
        uint32_t lz_speed = e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK;
        uint32_t atotal = zpse_block_ascii_total(n);
        uint32_t lz_floor = (uint32_t)ZPSE_BLOCK_HEADER_SIZE +
                            ZPSE_LZ_MIN_PAYLOAD +
                            (uint32_t)ZPSE_LZ_MIN_CMD *
                                ((n + ZPSE_LZ_MAX_MATCH - 1u) /
                                 ZPSE_LZ_MAX_MATCH);

        if (e->lz != 0 && atotal > lz_floor &&
            zpse_block_lz_level(lz_speed, e->ascii_gate)) {
            uint32_t lz_total = 0u;
            uint32_t lz_cap = (uint32_t)ZPSE_E_SZ_CAND;
            uint32_t win = atotal / (uint32_t)ZPSE_LZ_MIN_WIN_DIV + 1u;
            int lz_screen_win;

            /* This is the screen the candidate block below would run, on
             * the same sample, the same tile price and the same verdict
             * total (the trial's is atotal by construction), so its answer
             * IS that block's answer and it must not ask again: on a gated
             * block the screen is a 1 KiB match-finder pass plus a priced
             * sample, a tenth of a text block's encode. */
            lz_done = 1;
            lz_screen_win = zpse_block_lz_screen(
                e, in, n, tids[0], scounts[0],
                meta_lens[0] > 0 ? meta_bufs[0] : in,
                meta_lens[0], lz_speed, 1, atotal, 0);

            /* ---- The TEXT fast path: use the screen's verdict ---------- *
             *
             * When the cheap screen says the finder cannot win, a gated
             * block's only tile candidate is the arithmetic identity total
             * (0.875 bytes a byte plus the envelope). TEXT was priced once
             * above; when it beats that identity total by the same 1/32
             * margin the final commit would require, it is the winner and
             * there is no reason to run the tile trial or the sweep. Ship it
             * here. This is exactly the decision the final commit would make
             * for the same block, taken before paying for the ladder -- on
             * prose the screen refuses the finder on ~78% of text blocks,
             * which is where the encode time went. When the screen says LZ
             * can win, nothing changes: the finder runs and the final commit
             * keeps the exact TEXT-vs-LZ comparison. */
            if (text_eligible && !lz_screen_win &&
                text_total + atotal / 32u + 1u <= atotal &&
                text_total <= out_cap) {
                uint32_t tt = zpse_block_write_text(e, in, n, seq, is_last,
                                                    out, out_cap,
                                                    text_total - 16u);

                if (tt != 0u) {
#if defined(ZPSE_BLOCK_STATS)
                    zpse_block_gate_lz++;
#endif
                    return (int)tt;
                }
            }

            if (lz_screen_win) {
                if (atotal < (uint32_t)ZPSE_E_SZ_CAND && atotal > win) {
                    lz_cap = atotal - win;
                }
                {
#if defined(ZPSE_BLOCK_STATS)
                    uint64_t t_cyc = blk_rdtsc();
#endif
                    /* The finder writes e->best and a gated block never
                     * runs the sweep, so e->cand is free for its table. */
                    rc = zpse_block_try_lz(
                        e, in, n, in_crc, is_last, seq, e->best, lz_cap,
                        e->cand, 15u, (uint32_t)ZPSE_BLOCK_SHORT_MAX,
                        &lz_total, 0);
#if defined(ZPSE_BLOCK_STATS)
                    zpse_block_lz_passes++;
                    zpse_block_lz_pass_cycles += blk_rdtsc() - t_cyc;
#endif
                }
                if (rc == ZPSE_OK &&
                    zpse_block_lz_wins(n, e->best, lz_total, 1, atotal,
                                       lz_speed)) {
                    /* ---- The gate's total is arithmetic, so the trial is
                     * not needed to settle it ---------------------------
                     *
                     * A gated block's tile stream is the seven-bit pack of
                     * every tile, whose total zpse_block_ascii_total knows
                     * before anything runs (0.875 bytes a byte plus the
                     * envelope). The comparison this block would make after
                     * the trial is therefore already decided here: the
                     * finder's stream is smaller than the number the trial
                     * provably produces. Running the trial anyway -- it is
                     * handed `lz_total - win` as a byte budget, so it stops
                     * a while before the end -- was still most of a text
                     * block's encode spent on a stream that cannot win:
                     * `dickens` 174 -> 587 MB/s and `webster` 199 -> 246 once
                     * this returned here instead (screen at 4 KiB, one
                     * pinned core). The exception is the deepest rung, which
                     * offers the final LZ4 pass after the sweep: the trial is
                     * that candidate's `have_best`, so that rung keeps it and
                     * its stream is unchanged. */
                    /* When the cheap sample already said TEXT cannot win,
                     * the early TEXT return above handled the block. Here the
                     * finder's stream is the smaller one against the identity
                     * total, so ship it directly: the trial is arithmetically
                     * unable to beat it and the ladder would only re-derive
                     * that. The final TEXT comparison still runs on every
                     * block that reaches it (LZ not winning, or MAX), so this
                     * fast return gives up no ratio. */
                    if (lz_total <= out_cap &&
                        zpse_enc_speed(lz_speed) < 9u) {
#if defined(ZPSE_BLOCK_STATS)
                        zpse_block_gate_lz++;
#endif
                        memcpy(out, e->best, lz_total);
                        lz_total = zpse_block_shrink_page(out, n, lz_total);
                        /* try_lz's stream carries ZPSE_B_LZ: invalidate. */
                        zpse_block_hist_set(e, ZPSE_B_LZ, 0u, 1u);
                        return (int)lz_total;
                    }
                    /* The stream is already in e->best, which is where the
                     * winner is read from below; `out` must not be written
                     * by the trial any more, since the trial is about to
                     * lose. */
                    have_best = 1;
                    best_total = lz_total;
                    best_idx = 15u;
                    direct_out = 0;
                }
            }
        }
    }
#if defined(ZPSE_BLOCK_STATS)
    /* The total the first trial produced, and whether it succeeded at all.
     * The difference between it and the block's final winner is exactly
     * what the later trials bought. */
    uint32_t first_trial_total = 0u;
    int have_first_trial = 0;
    /* The slot (in trial order) that produced the current winner. */
    int win_pos = -1;
#endif

    /* The level row, for the full-size trial cap below (see
     * ZPSE_TRIAL_CAP_FULL): only a sweep that already cuts tied fields takes
     * the cap, so the deepest rungs keep every survivor their price
     * separated and the ladder stays monotone. */
    const ZpseLevel *cap_row = zpse_level_row(
        zpse_enc_speed(e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK));

    /* One floor per candidate, computed once.
     *
     * `zpse_block_floor` is a pure function of (n, transform, segment count,
     * metadata length), and none of those four is written between here and
     * the end of the sweep -- the candidate build and the pricing compaction
     * are both behind us. The loop below nevertheless asks for it twice per
     * REMAINING candidate: once at the top for the per-candidate skip, and
     * once per iteration for the post-trial scan, so a ten-candidate block
     * pays about fifty-five calls for ten distinct values. Each one
     * re-derives the width, runs `zpse_segment_lengths` (a variable-divisor
     * division per segment on the wide shapes) and walks the planes, which
     * is more arithmetic than the tiles a trial spends on a 16 KiB window.
     * The slot order is the one the loop uses, and it is fixed here: the
     * compaction that renumbers the slots ran already.
     *
     * Same arguments, same function, same value, and UINT32_MAX (a shape
     * that cannot be priced) compares exactly as it did at both sites. */
    {
        int fi;

        for (fi = 0; fi < ncand; fi++) {
            fl[fi] = zpse_block_floor(n, tids[fi], scounts[fi],
                                     meta_lens[fi],
                                     zpse_segment_price(e->enc_flags));
        }
    }
    /* Suffix minima over the same fixed slot order. idx[] values are the
     * original list indices (0..ZPSE_MAX_CAND), so UINT8_MAX is a sentinel
     * no real index can reach; UINT32_MAX does the same for the floors. */
    {
        int fi;

        suf_fl[ncand] = UINT32_MAX;
        suf_idx[ncand] = UINT8_MAX;
        for (fi = ncand - 1; fi >= 0; fi--) {
            suf_fl[fi] = fl[fi] < suf_fl[fi + 1] ? fl[fi] : suf_fl[fi + 1];
            suf_idx[fi] = idx[fi] < suf_idx[fi + 1] ? idx[fi] : suf_idx[fi + 1];
        }
    }

    /* ---- The match finder first, when the screen convicts --------------
     *
     * The screen below answers "will the match finder beat the shapes the
     * sweep offered" from a 1 KiB sample. When it says yes, the sweep is
     * about to spend a full block encode per survivor learning what the
     * sample already said -- and on the files where the two forms tie it is
     * the entire cost of the block: nci prices 5.33:1 from the finder at
     * 1166 MB/s and 5.34:1 from the tiles at 218, ooffice 1.47 at 315
     * against 1.48 at 180. Paying a tile pass first and a finder pass second
     * to choose between two streams a tenth of a percent apart is exactly
     * the case this is for.
     *
     * So the pass runs first, its verdict is the level's own screen margin
     * (the same one the later stage asks), and a block the screen convicts
     * is written from the finder without the sweep. The guard that is left is
     * zpse_block_lz_wins' acceleration factor: a stream whose commands are
     * too long to decode inside this level's bill is refused however small it
     * is, so the form still cannot trade decode for size here.
     *
     * What this cannot have is the sweep's own total to compare against, so
     * the trade is bounded by the screen's error -- a sample it gets wrong
     * costs a block the tile stream it would have written -- rather than by a
     * full trial. That is why the screen is asked with the level's margin and
     * not a looser one, and why it is asked only about a FULL-SIZE block: the
     * sample is a fixed 1 KiB, so on a 3 KiB block it is a third of the block
     * and the tiles still win those by a wide margin (the size suite showed
     * 142 such blocks growing, up to 2 KiB each, when this ran at every n).
     * tests/suites/test_sizes.c fails any case that grows and the Silesia
     * table is the rate this is worth.
     *
     * The screen prices slot 0 of the list, which here is the identity
     * shape, not the shape the price would have promoted. On text that is the
     * right question -- identity is what the tile encoder falls back to
     * there, and the finder wins by a third (dickens 1.16 to 1.57). On a ramp
     * or a sawtooth it is the wrong one: identity prices badly, the screen
     * convicts, and the finder writes 802 bytes where the transpose writes
     * 248. Those nine blocks are the measured cost of this rule, and the
     * trade was taken deliberately: over Silesia the corpus grows 0.7% and
     * the default level's encode rate goes from 258 to 369 MB/s (+41%), with
     * six files of it unchanged to the byte and the growth concentrated on
     * one (`mozilla` +2.1%), because that is a binary whose blocks the sample
     * gets wrong at the margins. tests/suites/test_sizes.c enforces the bound
     * this was accepted under (no case may grow by more than 1/64 of its
     * input) rather than the absolute "no case grows" it used to. */
    if (!direct_out && !lz_done && e->lz != 0 && ncand > 0 &&
        n >= (uint32_t)ZPSE_MAX_BLOCK_SIZE &&
        zpse_block_lz_level(e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK,
                            e->ascii_gate) &&
        zpse_block_lz_screen(e, in, n, tids[0], scounts[0],
                             meta_lens[0] > 0 ? meta_bufs[0] : in,
                             meta_lens[0],
                             e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK, 0,
                             0u, lz_cache)) {
        uint32_t lz_total = 0u;
        int lrc;

#if defined(ZPSE_BLOCK_STATS)
        {
            uint64_t t_cyc = blk_rdtsc();

            lrc = zpse_block_try_lz(e, in, n, in_crc, is_last, seq, e->cand,
                                    (uint32_t)ZPSE_E_SZ_CAND, e->best, 15u,
                                    (uint32_t)ZPSE_BLOCK_SHORT_MAX,
                                    &lz_total, 0);
            zpse_block_lz_passes++;
            zpse_block_lz_pass_cycles += blk_rdtsc() - t_cyc;
        }
#else
        lrc = zpse_block_try_lz(e, in, n, in_crc, is_last, seq, e->cand,
                                (uint32_t)ZPSE_E_SZ_CAND, e->best, 15u,
                                (uint32_t)ZPSE_BLOCK_SHORT_MAX, &lz_total, 0);
#endif

        /* ...and only when the block is not one the tiles own outright.
         * A stream under 1/64 of the block (zeros, an alternating pair, a
         * ramp the transpose flattens to nothing) means the data has a
         * structure the tile encoder expresses in tens of bytes, and the
         * finder's 800 is ten times worse: 69 bytes of tiles against 802 of
         * LZ on a block of zeros, 83 against 803 on 0x00/0xFF. Those blocks
         * are cheap in absolute terms, so the guard does not cost the rate
         * this stage is for -- nci prices 5.3:1, dickens 1.6:1, both far
         * above it -- and it keeps the size suite's contract exact. */
        if (lrc == ZPSE_OK && lz_total < (uint32_t)ZPSE_E_SZ_CAND &&
            lz_total <= out_cap && lz_total >= n / 64u &&
            zpse_block_lz_wins(n, e->cand, lz_total, 0, 0u,
                               e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK)) {
            memcpy(out, e->cand, lz_total);
            lz_total = zpse_block_shrink_page(out, n, lz_total);
            /* try_lz's stream carries ZPSE_B_LZ: invalidate. */
            zpse_block_hist_set(e, ZPSE_B_LZ, 0u, 1u);
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_finder_first++;
#endif
            return (int)lz_total;
        }
    }

    for (oi = 0; oi < ncand; oi++) {
        int i = oi;
        uint32_t t = 0;
        int budget_hit = 0;
        const uint8_t *mb;
        /* Exact bound before the work: the smallest total this candidate's
         * (transform, segment count, metadata) shape can produce is a
         * strictly greater size than the winner, so the trial cannot be
         * chosen or tie and the transform staging is pure cost. Unlike the
         * post-trial floor check, this one is per candidate, so a single
         * hopeless shape left in the list no longer keeps the others
         * alive. `>` (not `>=`) on purpose: a candidate whose floor equals
         * the winner could still tie it, and a tie is decided by list
         * order, so it has to be attempted.
         *
         * "Could still tie" is only true for a candidate EARLIER in the
         * list than the current winner: a tie goes to the lower index, so
         * a later one cannot win it. The list-order test is what keeps the
         * bound exact wherever the attempt order is not the list order
         * (ZPSE_FORCE_TRIAL_SWAP, and any future reordering): skipping an
         * earlier candidate because it merely ties would change the
         * stream, and the parity suite proves it does not. */
        if (have_best && (uint32_t)idx[i] > best_idx &&
            fl[i] > best_total) {
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_trials_skipped++;
#endif
            continue;
        }
        mb = meta_lens[i] > 0 ? meta_bufs[i] : in;
        ntried++;
        /* The gated block has exactly one trial and it is going to win:
         * the seven-bit form is smaller than the raw fallback by
         * construction. Write it straight into the caller's buffer instead
         * of staging it through cand/best and copying it out again -- two
         * full-block copies that were most of what the gate's own encode
         * cost (measured: dickens 841 -> 1.3 GB/s). */
        if (direct_out) {
            tdst = out;
            tcap = out_cap;
        } else {
            tdst = e->cand;
            tcap = (uint32_t)ZPSE_E_SZ_CAND;
        }
#if defined(ZPSE_BLOCK_STATS)
        zpse_block_trials++;
        zpse_block_trials_by_tid[tids[i] & 15u]++;
#endif
#if defined(ZPSE_BLOCK_STATS)
        {
            uint64_t t_cycle = blk_rdtsc();

            rc = zpse_block_try_one(e, in, n, in_crc, tids[i], scounts[i], mb,
                                    meta_lens[i], is_last, seq, tdst, tcap,
                                    &t, &budget_hit, have_best, best_total,
                                    flat_min, 16u);
            zpse_block_trial_cycles += blk_rdtsc() - t_cycle;
        }
#else
        rc = zpse_block_try_one(e, in, n, in_crc, tids[i], scounts[i], mb,
                                meta_lens[i], is_last, seq, tdst, tcap, &t,
                                &budget_hit, have_best, best_total, flat_min,
                                16u);
#endif
#if defined(ZPSE_TRACE_TRIALS)
        fprintf(stderr, "  trial oi=%d tid=%u rc=%d t=%u haven=%d bt=%u\n",
                oi, (unsigned)tids[i], rc, t, have_best, best_total);
#endif
        if (rc != ZPSE_OK) {
            /* A trial the budget stopped cannot win: its total only grows as
             * its remaining segments are written. That is a verdict on THIS
             * candidate and says nothing about the ones after it -- a shape
             * that packs worse on the front of a block can be the best one
             * over the whole of it -- so on its own it does not end the
             * sweep. The per-candidate floor test at the top of the loop
             * still skips a shape whose own bound cannot beat the winner
             * without spending the trial. */
            (void)budget_hit;
#if defined(ZPSE_BLOCK_STATS)
            if (!have_first_trial) {
                /* The first trial failed, so there is no first-trial total to
                 * compare against: the block's winner was found by a later
                 * trial with nothing to beat. */
                have_first_trial = 1;
                first_trial_total = UINT32_MAX;
            }
#endif
            /* A trial the budget stopped is a trial whose stream was already
             * bigger than the raw envelope before it reached the end of the
             * block: the budget IS the raw envelope and the trial ran into
             * it. On a block the sweep has priced as noise -- the winner's
             * price over the head window and over the tail window is at
             * least the raw envelope -- that is the verdict for the whole
             * sweep, because the shapes behind this one are priced no better
             * and run into the same budget. It saves two whole block encodes
             * on every block of noise, which is most of what such a block
             * costs to encode.
             *
             * The verdict is the trial's own, not the price's, which is what
             * keeps it honest: a block whose head and tail are noise but
             * whose middle is not -- the "noise then constant" shape of the
             * size corpus -- has a price that says flat and a trial that
             * says otherwise, and the trial wins here because it encodes the
             * whole block. A block that compresses still gets its
             * compression; only a block that really is noise stops after
             * one trial. */
            if (sample_flat) {
                break;
            }
            continue;
        }
        /* Smallest total wins; a tie goes to the candidate EARLIER in
         * the list, so the attempt order cannot change the winner. */
        if (!have_best || t < best_total ||
            (t == best_total && (uint32_t)idx[i] < best_idx)) {
            if (!direct_out && t > (uint32_t)ZPSE_E_SZ_BEST) {
                /* The trial ran and its result cannot be kept: the block is
                 * bigger than the best buffer, so the RAW fallback will
                 * carry it. That is a whole block encode spent for nothing,
                 * and on an incompressible block EVERY candidate lands
                 * here -- so this path has to reach the flat-block cap
                 * too, or the cap never fires for exactly the blocks that
                 * need it. */
                if (sample_flat && ntried >= ZPSE_FLAT_TRIALS) {
                    break;
                }
                continue;
            }
            if (!direct_out) {
                memcpy(e->best, tdst, t);
            }
            best_total = t;
            best_idx = (uint32_t)idx[i];
            have_best = 1;
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_bytes[tids[i] & 15u] += t;
            /* The winner changed: the previous holder's win is undone, so
             * the per-tid win count is tracked at the end of the block
             * rather than here. */
            win_pos = oi;
#endif
        }
#if defined(ZPSE_BLOCK_STATS)
        if (!have_first_trial) {
            have_first_trial = 1;
            first_trial_total = t;
        }
#endif
#if ZPSE_TRIAL_CAP_FULL > 0
        /* ---- Full-size trial cap ------------------------------------ *
         * A full block's price window is a fifth of the block, so the price
         * has already had its say; the survivors it could not separate are
         * the ones this stops paying for. A candidate EARLIER in the list
         * than the winner can still tie it on list order, so one of those in
         * the remaining field cancels the cap outright -- which is the same
         * rule the floor skip and the history stop use, and the reason the
         * stream cannot depend on attempt order. */
        if (!direct_out &&
            (cap_row->tie_field != 0u || ZPSE_TRIAL_CAP_TIEFREE != 0) &&
            ntried >= ZPSE_TRIAL_CAP_FULL &&
            n >= (uint32_t)ZPSE_MAX_BLOCK_SIZE) {
            int clear = 1;
            int oj;

            for (oj = oi + 1; oj < ncand; oj++) {
                if ((uint32_t)idx[oj] < best_idx) {
                    clear = 0;
                    break;
                }
            }
            if (clear) {
                break;
            }
        }
#endif
#if ZPSE_HIST_STOP_STREAK > 0u
        /* ---- History-confident stop -------------------------------- *
         * The user-facing shape of this rule: over a 200 MB file the encoder
         * knows what it is encoding, so when it is sure of the first trial it
         * should not spend three more full block encodes discovering that it
         * was right.
         *
         * "Sure" is a run, not a margin. The price model cannot supply one --
         * every survivor is already within 1/16 of the winner's price, since
         * that is what kept it -- so the only independent evidence left is
         * the file's own history: the shape that wins has been winning, block
         * after block, for ZPSE_HIST_STOP_STREAK blocks running. At 64 KiB a
         * block, an 8-block run is half a megabyte of one shape, which on a
         * 200 MB file is 1/400th of the input: a stretch the encoder has no
         * reason to keep re-deciding.
         *
         * The predictor has to have chosen the trial too (hist_first), because
         * the run itself says nothing about whether the PRICE would have found
         * the same shape -- and it is the predictor, not the run, that decided
         * what to try first.
         *
         * This is a size trade and not a bug: the winner is still the smallest
         * the sweep tried, the block it would have found is at most as good as
         * the field the price left tied, and tests/suites/test_sizes.c fails
         * any case that grows. A candidate EARLIER in the list than the winner
         * can still tie it on list order, so one in the remaining field
         * cancels the stop outright. */
        if (hist_first && have_best && !direct_out && ntried == 1 &&
            e->hist_streak >= (uint8_t)ZPSE_HIST_STOP_STREAK &&
            tids[0] != 0u &&
            best_total + 1u < (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
            int clear = 1;
            int oj;

            for (oj = oi + 1; oj < ncand; oj++) {
                if ((uint32_t)idx[oj] < best_idx) {
                    clear = 0;
                    break;
                }
            }
            if (clear) {
                break;
            }
        }
#endif
        /* Early exit over the candidates that are LEFT, not over all of
         * them: a trial whose own envelope is bigger (16 segments, or a
         * metadata byte) cannot reach a size the identity trial can, so
         * comparing the running best against each remaining candidate's
         * own floor stops the sweep as soon as nothing left could be
         * strictly smaller. That is why the identity runs first -- it is
         * the only candidate that can reach the global minimum, so for
         * constant/ramp blocks the sweep ends here after one trial while a
         * ramp whose winner is DELTA8 still gets its second trial. */
        {
            /* A candidate earlier in the list than the winner cannot
             * produce a smaller total, but it can produce the same one and
             * win the tie on order, so the bound does not cover it and the
             * sweep does not end while one is left. suf_idx[k] < best_idx is
             * exactly that condition over the remaining slots, and when it
             * is false every remaining slot contributes to suf_fl[k], which
             * is therefore the same minimum the scan computed. */
            int earlier_left = (suf_idx[oi + 1] < best_idx);
            uint32_t min_floor = suf_fl[oi + 1];

            if (have_best && !earlier_left && best_total <= min_floor) {
                break;
            }
            /* The probe says this block does not compress (see
             * ZPSE_FLAT_TRIALS): stop after that many full encodes. */
            if (sample_flat && ntried >= ZPSE_FLAT_TRIALS) {
                break;
            }
            /* A trial that finished above the raw envelope is the same
             * verdict the budget gives a trial that ran into it (see the
             * budget-stopped path above): the shape does not beat the raw
             * bytes, and on a block the sweep has priced as noise the shapes
             * behind it are priced no better, so the sweep stops here too. */
            if (sample_flat &&
                t + 1u > (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
                break;
            }
            /* Two candidates that both finish above the raw envelope are
             * evidence that the block is incompressible, and the sweep spends
             * two whole trials to find out -- the rest abort on the noise
             * probe long before they write the block out. RAW itself (id 0)
             * is the baseline -- it never beats itself -- so it does not count
             * as a failure; otherwise a block whose first candidates are RAW
             * and DELTA8 would abort before reaching a transpose that actually
             * pays. */
            if (tids[i] != 0u &&
                t + 1u > (uint32_t)ZPSE_BLOCK_HEADER_SIZE + n) {
                fails++;
                if (fails >= 2) {
                    break;
                }
            }
        }
    }

    /* One more candidate, competing on size alone: the intra-block LZ form.
     * A level that takes it can never write a block larger than the same
     * level without it, and a block it wins is a copy loop to decode instead
     * of a tile reconstruction. Which levels run the match finder at all is
     * zpse_block_lz_level (MAX and the default level price every block, the
     * two cheap levels only the blocks the ASCII gate convicted of being
     * text), and what they will trade for it is the level's acceleration
     * factor (see ZPSE_LZ_ACCEL_*): the match finder is a full pass over the
     * block, so it is the most expensive candidate in the sweep. */
    if (!lz_done &&
        zpse_block_lz_level(e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK,
                            e->ascii_gate) &&
        e->lz != 0) {
        /* Exact floor first: one command codes at most 259 bytes, so an
         * LZ payload can never be smaller than its 12-byte header plus one
         * 2-byte command per 259 bytes of output. A winner already under
         * that cannot be beaten, and on the blocks the tiles already
         * handle well -- a constant or all-hit plane is 248 bytes, far
         * below the ~1 KiB floor of a 64 KiB block -- the whole match
         * finder is skipped. */
        uint32_t lz_floor = (uint32_t)ZPSE_BLOCK_HEADER_SIZE +
                            ZPSE_LZ_MIN_PAYLOAD +
                            (uint32_t)ZPSE_LZ_MIN_CMD *
                                ((n + ZPSE_LZ_MAX_MATCH - 1u) /
                                 ZPSE_LZ_MAX_MATCH);

        if ((!have_best || best_total > lz_floor) &&
            zpse_block_lz_screen(e, in, n, tids[0], scounts[0],
                                 meta_lens[0] > 0 ? meta_bufs[0] : in,
                                 meta_lens[0],
                                 e->enc_flags &
                                     (uint32_t)ZPSE_ENC_SPEED_MASK,
                                 have_best, best_total, lz_cache)) {
            uint32_t lz_total = 0u;
            uint32_t lz_speed = e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK;

            /* Cap the pass at the size it could still win at. `lz_wins`
             * keeps the form only when its total is smaller than the
             * winner's by ZPSE_LZ_MIN_WIN_DIV, so a payload that reaches
             * that cap is one the sweep would reject anyway -- and a stream
             * only grows as it is written, so stopping there cannot change
             * the verdict, only the time spent reaching it. This is the
             * whole of the win on the blocks where the form LOSES after
             * its screen accepted it (the screen samples 1 KiB and can be
             * wrong about the block): measured on the level-5 corpus,
             * uuid_mix 254 -> 289 MB/s (+13.7%) and json_rows +1.1%, every
             * stream unchanged. */
            uint32_t lz_cap = (uint32_t)ZPSE_E_SZ_CAND;

            if (have_best && best_total < (uint32_t)ZPSE_E_SZ_CAND) {
                uint32_t win = best_total / (uint32_t)ZPSE_LZ_MIN_WIN_DIV +
                               1u;

                if (best_total > win) {
                    lz_cap = best_total - win;
                }
            }
            /* The table is the small one at a page and the 64 KiB staging
             * region from 16 KiB up: the trials have all finished, and the
             * lazy pass below writes to tstage itself, so it keeps e->lz.
             * See ZPSE_LZ_TABLE_BIG_MIN. */
            {
                void *ltab = e->lz;
                uint32_t lhb = (uint32_t)ZPSE_LZ_HASH_BITS;

                if (n >= (uint32_t)ZPSE_LZ_TABLE_BIG_MIN) {
                    ltab = e->tstage;
                    lhb = 15u;
                }
                rc = zpse_block_try_lz(e, in, n, in_crc, is_last, seq, e->cand,
                                       lz_cap, ltab, lhb,
                                       (uint32_t)ZPSE_BLOCK_SHORT_MAX,
                                       &lz_total, 0);
            }
            /* The lazy window is gone with the old finder. LZ4 owns the match
             * search and zpse_block_try_lz ignores its table/hash/lazy-window
             * arguments, so the second pass used to re-encode the same bytes
             * and could never produce a strictly smaller stream. */
            /* Smaller is not enough: an LZ4 stream and a tile stream decode
             * at speeds that have nothing to do with their ratio, so the win
             * also has to be big enough to pay for the decode it gives up and
             * the LZ4 command loop has to be inside this level's acceleration
             * factor. Both tests are in zpse_block_lz_wins, with the
             * measurements they came from. */
            if (rc == ZPSE_OK &&
                zpse_block_lz_wins(n, e->cand, lz_total, have_best,
                                   best_total, lz_speed)) {
                if (direct_out) {
                    /* In direct_out mode, `out` is the caller's buffer with
                     * capacity `tcap` (= out_cap). `lz_total` can be up to
                     * `lz_cap` (ZPSE_E_SZ_CAND = 64 KiB when !have_best),
                     * which may exceed `tcap`. Guard the copy so we never
                     * write past the caller's buffer; if LZ doesn't fit,
                     * skip it and fall through to RAW (which checks out_cap
                     * itself at line 2744). */
                    if (lz_total <= tcap) {
                        memcpy(out, e->cand, lz_total);
                        best_total = lz_total;
                        best_idx = 15u;
                        have_best = 1;
                    }
                } else {
                    memcpy(e->best, e->cand, lz_total);
                    best_total = lz_total;
                    best_idx = 15u;
                    have_best = 1;
                }
            }
        }
    }

    /* ---- v2: LZ *and* tiles ----------------------------------------- *
     * The two forms have been alternatives since v1: the encoder computes
     * both and keeps the smaller, so the tile stream of a block LZ wins was
     * built and thrown away. This keeps it and compresses it, which is the
     * arrangement that pays on both sides: the body is what the tiles won
     * with, and what is left in it is the repetition the tile records still
     * carry -- repeated records, mask and residual patterns that the tile
     * codecs write verbatim.
     *
     * Only taken when the result is at least 1/32 smaller than the tile
     * stream alone. The decode pays a match-finder pass over the body before
     * the tile reconstruction, so a few bytes is the wrong trade; measured
     * over Silesia at the default level, the stage takes 13.3% off the tile
     * payload of exactly the blocks where tiles beat LZ.
     *
     * Its cost on short blocks was measured on the way past: gating it to
     * blocks of 16 KiB and up moved the 4 KiB dump total from 41.45 to 42.42
     * points -- a full point of ratio -- for 2% of encode, which a build that
     * changes nothing also spends. The body it compresses is already a tile
     * stream, so the finder is fast on it and there is no pass to save. */
    if (!direct_out && have_best && e->lz != 0 &&
        zpse_block_lz_level(e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK,
                            e->ascii_gate) &&
        best_total > (uint32_t)(ZPSE_BLOCK_HEADER_SIZE + 4u + 64u) &&
        best_total <= (uint32_t)ZPSE_E_SZ_TSTAGE) {
        ZpseBlockHeader wh;

        if (zpse_block_header_parse(&wh, e->best, ZPSE_BLOCK_HEADER_SIZE) ==
                ZPSE_OK &&
            (wh.block_flags &
             (uint8_t)(ZPSE_B_RAW | ZPSE_B_LZ | ZPSE_B_LZ_TILE |
                       ZPSE_B_HUF)) == 0u) {
            uint32_t blen = best_total - (uint32_t)ZPSE_BLOCK_HEADER_SIZE;
#if defined(ZPSE_BLOCK_STATS)
            uint64_t t_v2 = blk_rdtsc();
#endif
            uint32_t pl = zpse_lz4_encode(e->best + ZPSE_BLOCK_HEADER_SIZE,
                                          blen, e->tstage,
                                          (uint32_t)ZPSE_E_SZ_TSTAGE);
#if defined(ZPSE_BLOCK_STATS)
            zpse_block_v2_cycles += blk_rdtsc() - t_v2;
#endif
            uint32_t tot = (uint32_t)(ZPSE_BLOCK_HEADER_SIZE + 4u) + pl;

            /* The tile body is read through a match finder before the
             * reconstruction, so a wrap has to pay for the pass: 1/32 of the
             * tile stream. Pricing it by the payload's share of the block
             * instead was tried (`pl <= n / 8`): it refused wraps that save a
             * third of a block and cost 1.8-3.1% of the frame on nci, xml and
             * dickens for no decode time this host could measure, so the
             * margin stayed. See docs/HISTORY.md. */
            if (pl != 0u && tot + tot / (uint32_t)ZPSE_LZ_TILE_DIV <=
                                best_total &&
                tot <= out_cap) {
                /* e->tstage holds [u32 n][u32 cmds][raw LZ4 block], with no
                 * block header. Fill the count word now, after the size gate
                 * and before the payload is copied to `out`. */
                zpse_store_u32le(e->tstage + 4u,
                                 zpse_lz_cmd_count(e->tstage + 8u, pl - 8u));
                wh.block_flags = (uint8_t)((wh.block_flags &
                                            ~(uint8_t)(ZPSE_B_RAW |
                                                       ZPSE_B_LZ)) |
                                           ZPSE_B_LZ_TILE);
                wh.compressed_block_size = tot;
                zpse_block_header_serialize(out, &wh);
                zpse_store_u32le(out + ZPSE_BLOCK_HEADER_SIZE, blen);
                memcpy(out + ZPSE_BLOCK_HEADER_SIZE + 4u, e->tstage, pl);
                best_total = tot;
                direct_out = 1;
            }
        }
    }

    /* ---- The final LZ4 pass, priced last ------------------------------ *
     *
     * A second LZ4 decision point, run after the finder pass and the LZ+tile
     * stage. Both write the same canonical raw-LZ4 block; keeping the second
     * screen and pass is a deliberate decision-point split, not a second
     * codec (the measurements behind it are in docs/HISTORY.md).
     *
     * Placement is the whole of the correctness here. This has to run after
     * the LZ-and-tiles stage, because that stage compresses the tile body and
     * on periodic data reaches a size LZ cannot touch; committing a merely
     * smaller stream before it would skip the better one. Committing last, the
     * candidate can only be reached when every other stage has had its say,
     * and `t2 < best_total` makes it a strict improvement over whatever they
     * produced.
     *
     * That stage can also write its winner straight into `out` (`direct_out`),
     * and it is not always the better stream. So the candidate competes in
     * that mode too, and overwrites the block already in `out` when it is
     * strictly smaller -- which it can only be by `t2 < best_total`, and
     * `t2 <= out_cap` keeps it inside the buffer.
     *
     * Levels 1-4 cannot take it: they write no LZ form at all, and offering
     * one would make a shallow rung write smaller blocks than the rung above
     * it. Every rung that pays a match finder offers it (ZPSE_LZ_FINAL_MIN_LEVEL),
     * and the screen in front of it is its own (zpse_block_lz_final_screen).
     * Asking the question with the encoder that writes the stream is what lets
     * the candidate reach the blocks it wins -- and the sizes can only fall,
     * because the stage is last and commits on a strict improvement.
     */
#ifndef ZPSE_LZ_FINAL_MIN_LEVEL
#define ZPSE_LZ_FINAL_MIN_LEVEL ZPSE_ENC_SPEED_LZ_MIN
#endif
    if (have_best && e->lz != 0 &&
        zpse_enc_speed(e->enc_flags & (uint32_t)ZPSE_ENC_SPEED_MASK) >=
            (uint32_t)ZPSE_LZ_FINAL_MIN_LEVEL &&
        best_total > (uint32_t)(ZPSE_BLOCK_HEADER_SIZE +
                                ZPSE_LZ_MIN_PAYLOAD) &&
        !zpse_block_winner_is_lz(e, out, direct_out)) {
        /* The screen prices its sample under ONE candidate's transform, and
         * which one must not depend on the order the sweep attempted them in:
         * both orders are required to write identical streams (test_fuzz
         * under ZPSE_FORCE_TRIAL_SWAP), and a few bytes of margin make the
         * screen answer differently -- a segment header's worth was enough.
         * List position 0 is the attempt order's first, so it is not that
         * candidate; the lowest list index is, and it is also the cheapest to
         * price -- the identity or the sample's favourite, whose plane needs
         * no staging copy. Pricing the WINNER instead was measured and
         * refused: on the 4 KiB dumps it took 3 691 bytes off for 3.3% of the
         * encode rate, because a winning transpose prices its sample through
         * a staged plane. */
        int wi = 0;
        int wj;

        for (wj = 1; wj < ncand; wj++) {
            if (idx[wj] < idx[wi]) {
                wi = wj;
            }
        }
        if (!zpse_block_lz_final_screen(e, in, n, tids[wi], scounts[wi],
                                        meta_lens[wi] > 0 ? meta_bufs[wi] : in,
                                        meta_lens[wi],
                                        e->enc_flags &
                                            (uint32_t)ZPSE_ENC_SPEED_MASK,
                                        1, best_total)) {
            goto lz_final_done;
        }
        uint32_t t2 = 0u;
        uint32_t cmds2 = 0u;

        if (zpse_block_try_lz2(e, in, n, in_crc, is_last, seq, e->cand,
                               (uint32_t)ZPSE_E_SZ_CAND, &t2, &cmds2) ==
                ZPSE_OK &&
            t2 < best_total &&
            (!direct_out || t2 <= out_cap)) {
            /* This path commits without zpse_block_lz_wins; fill the count
             * word now, after the strict size gate and before the copy. The
             * Huffman wrap below codes the payload, so it must see the same
             * bytes as the old eager count wrote. */
            zpse_block_lz_finish_count(e->cand, t2);
            if (direct_out) {
                memcpy(out, e->cand, t2);
            } else {
                memcpy(e->best, e->cand, t2);
            }
            best_total = t2;
            best_idx = 15u;
            have_best = 1;
        }
    }
lz_final_done:
    /* ---- Huffman coding of the winning LZ payload -------------------- *
     *
     * The last thing the sweep does, and the only stage that runs after the
     * decision rather than as part of it: it wraps the stream that already
     * won. That placement is the whole of the design. Wrapping inside either
     * LZ writer would be wrong: both stages can still be overwritten by a
     * later candidate, so a Huffman stream built there could be thrown away
     * after its cost was paid.
     *
     * Nothing above can see it, so it cannot change what the sweep chooses.
     * A block it takes is the same block with the same inner stream inside
     * it, gated by ZPSE_HUF_DIV: the decode pays a Huffman pass before the LZ
     * pass, and a few bytes is not worth a second pass over the payload.
     *
     * Only the single pure LZ4 form is wrapped. The tile form's transform and
     * segment count live in the block header, and this form's envelope is the
     * LZ form's own -- transform 0, one segment, no metadata -- so a tile body
     * has nowhere to put them. The RAW form is not wrapped either: it is the
     * fallback for bytes that did not compress, and coding it would make a
     * block that had already lost try again with a pass that cannot win.
     *
     * ZPSE_B_HUF is bit 7 on its own, and the payload's inner_form is fixed to
     * ZPSE_B_LZ, so the block header needs no combined LZ+HUF pattern. */
    if (have_best && zpse_block_huf_level(e->enc_flags) &&
        best_total > (uint32_t)(ZPSE_BLOCK_HEADER_SIZE + ZPSE_HUF_MIN_PAYLOAD)) {
        uint8_t *win = direct_out ? out : e->best;
        ZpseBlockHeader wh;

        if (zpse_block_header_parse(&wh, win, ZPSE_BLOCK_HEADER_SIZE) ==
            ZPSE_OK) {
            uint8_t form = (uint8_t)(wh.block_flags & (uint8_t)ZPSE_B_LZ);

            if (form == (uint8_t)ZPSE_B_LZ) {
                uint32_t plen = best_total - (uint32_t)ZPSE_BLOCK_HEADER_SIZE;
#if defined(ZPSE_BLOCK_STATS)
                uint64_t t_huf = blk_rdtsc();
#endif
                uint32_t hl = zpse_huf_encode(win + ZPSE_BLOCK_HEADER_SIZE,
                                              plen, e->tstage,
                                              (uint32_t)ZPSE_E_SZ_TSTAGE,
                                              form);
#if defined(ZPSE_BLOCK_STATS)
                zpse_block_huf_cycles += blk_rdtsc() - t_huf;
#endif

                /* Strictly smaller is not enough for a form that costs a
                 * decode pass: it has to be 1/ZPSE_HUF_DIV smaller, which at
                 * 64 is the point where the pass pays for itself on every
                 * corpus this was measured on. */
                if (hl != 0u &&
                    hl + plen / (uint32_t)ZPSE_HUF_DIV <= plen) {
                    uint8_t nh[ZPSE_BLOCK_HEADER_SIZE];

                    wh.block_flags =
                        (uint8_t)((wh.block_flags & ~(uint8_t)ZPSE_B_LZ) |
                                  (uint8_t)ZPSE_B_HUF);
                    wh.compressed_block_size =
                        (uint32_t)ZPSE_BLOCK_HEADER_SIZE + hl;
                    zpse_block_header_serialize(nh, &wh);
                    if (direct_out) {
                        memcpy(out, nh, sizeof nh);
                        memcpy(out + ZPSE_BLOCK_HEADER_SIZE, e->tstage, hl);
                    } else {
                        memcpy(e->best + ZPSE_BLOCK_HEADER_SIZE, e->tstage,
                               hl);
                        memcpy(e->best, nh, sizeof nh);
                    }
                    best_total = (uint32_t)ZPSE_BLOCK_HEADER_SIZE + hl;
                }
            }
        }
    }

    /* ---- The TEXT form, priced last against the sweep's winner ------- *
     *
     * The gate has already done the expensive work of proving this is text,
     * and zpse_text_price above already measured the token stream. Require a
     * 1/32 win over the LZ stream before paying for a second decode pass at
     * read time. The writer stages into e->cand (the LZ path's buffer) and
     * the copy below puts the bytes wherever the common copy-out expects the
     * winner -- `out` in direct_out mode, e->best otherwise. best_idx = 15 is
     * the sentinel the other non-candidate forms use. */
    if (text_eligible && have_best &&
        text_total + best_total / 32u + 1u <= best_total &&
        text_total <= out_cap) {
        uint32_t tt = zpse_block_write_text(e, in, n, seq, is_last,
                                            e->cand,
                                            (uint32_t)ZPSE_E_SZ_CAND,
                                            text_total - 16u);

        if (tt != 0u) {
            if (direct_out) {
                memcpy(out, e->cand, tt);
            } else {
                memcpy(e->best, e->cand, tt);
            }
            best_total = tt;
            best_idx = 15u;
            have_best = 1;
            text_won = 1;
        }
    }

    /* RAW fallback when coded does not pay: coded >= 32 + N. */
    if (!have_best || (uint64_t)best_total >=
                      (uint64_t)ZPSE_BLOCK_HEADER_SIZE + (uint64_t)n) {
        return zpse_block_write_raw(e, in, n, in_crc, seq, is_last, out,
                                    out_cap);
    }

    if (out_cap < best_total) {
        return ZPSE_OUTPUT_OVERFLOW;
    }
    if (!direct_out) {
        memcpy(out, e->best, best_total);
    }
    /* shrink_page rewrites a long header as a 16-byte SHORT one and memmoves
     * the payload from offset 32 to 16. A TEXT block is already a 16-byte
     * page form and its payload starts at 16, so that rewrite would destroy
     * it. The generic header parse now recognises the TEXT sync, so the
     * usual "parse fails, leave it alone" guard does not apply here. */
    best_total = text_won ? best_total
                          : zpse_block_shrink_page(out, n, best_total);
    /* The writer already recorded the TEXT block's history from the staging
     * copy; a second call here would count one block twice in hist_streak.
     *
     * The winner's three history fields are read straight out of the final
     * header: LONG keeps them at [3]/[12]/[13], SHORT at [2]/[3]/[4]. This is
     * the one site where they are not in a local -- the LZ_TILE / HUF
     * stages can rewrite the flags after the trial that won -- so read the
     * three bytes (no CRC, no validation) rather than the full header parse
     * the old hist_set ran over this same just-serialized header. */
    if (!text_won) {
        if (zpse_load_u16le(out) == (uint16_t)ZPSE_BLOCK_SYNC) {
            zpse_block_hist_set(e, out[3], out[12], out[13]);
        } else {
            zpse_block_hist_set(e, out[2], out[3], out[4]);
        }
    }
#if defined(ZPSE_BLOCK_STATS)
    {
        /* The winner's transform id: the candidate that carried best_total.
         * best_idx is the LIST index of the winner, and the list is sorted
         * by price, so the winner is found by scanning for it. */
        int wi;

        for (wi = 0; wi < ncand; wi++) {
            if ((uint32_t)idx[wi] == best_idx) {
                break;
            }
        }
        if (wi < ncand) {
            zpse_block_wins[tids[wi] & 15u]++;
            zpse_block_win_tid[tids[wi] & 15u]++;
        }
        /* Which trial won, and what the later trials bought over the first
         * one. A winner at position 0 means the sweep's own ordering was
         * right and every trial after it was spent for nothing. */
        if (win_pos >= 0 && win_pos < 32) {
            zpse_block_win_pos[win_pos]++;
        }
        if (have_first_trial && first_trial_total != UINT32_MAX &&
            first_trial_total > best_total) {
            zpse_block_first_trial_delta += first_trial_total - best_total;
            zpse_block_first_trial_lost++;
        }
    }
#endif
    return (int)best_total;
}

/* ---- Page dictionary ----------------------------------------------------
 *
 * The caller's half of the repeat-page trade (docs/page-dictionary.md): a
 * store that already holds a page's compressed bytes serves them again when
 * the same page comes back, instead of encoding it. What the codec can supply
 * is the 128-bit content tag and the header rewrite, because both are format
 * knowledge -- `seq` and the is-last flag live in the header, the header is
 * what its CRC covers, and a page's block may carry the 16-byte header while
 * everything else carries the 32-byte one. A caller writing that out by hand
 * would have to know all three.
 *
 * Nothing here allocates, reads the encoder, or changes the format: a hit's
 * block is an ordinary block, and a decoder cannot tell where it was written.
 * The measurements behind the design -- 16.4% of the dump corpus's pages
 * repeat, 109.8 -> 128.4 MB/s at 65 536 entries -- are in that document. */

/* murmur3's finaliser: every input bit is spread over the output, so the low
 * bits the index uses are as good as the high ones. */
static uint64_t zpse_page_fmix(uint64_t v)
{
    v ^= v >> 33;
    v *= UINT64_C(0xff51afd7ed558ccd);
    v ^= v >> 29;
    v *= UINT64_C(0x9E3779B97F4A7C15);
    v ^= v >> 32;
    return v;
}

void zpse_page_hash(const uint8_t *page, uint32_t n,
                    uint64_t *lo_out, uint64_t *hi_out)
{
    uint64_t a;
    uint64_t b;
    uint32_t i = 0;

    if (lo_out == 0 || hi_out == 0) {
        return;
    }
    if (page == 0 || n == 0u) {
        *lo_out = 0u;
        *hi_out = 0u;
        return;
    }
    /* Two independent 64-bit accumulators so one tag is 128 bits: the
     * index compares both halves, and a page that differs anywhere changes
     * both with the cost of one pass over the page.
     *
     * Four lanes, 32 bytes an iteration, because the pass is not free on a
     * corpus that does not repeat: a page's hash is paid before its lookup
     * can miss, and on the shared-library dump (0.9% of pages repeat) a
     * two-lane 8-byte loop cost 4.5% of the encode rate while returning
     * almost nothing. The lanes are independent multiply-xor chains, so the
     * four multiplies issue back to back instead of waiting on each other,
     * and the two output halves are folded from disjoint pairs -- a byte that
     * changes `x2` still moves `lo` without touching `x1`, so both halves
     * keep depending on every byte. */
    a = UINT64_C(0x9E3779B97F4A7C15) ^
        ((uint64_t)n * UINT64_C(0xff51afd7ed558ccd));
    b = UINT64_C(0xC2B2AE3D27D4EB4F) ^
        ((uint64_t)n * UINT64_C(0x9E3779B97F4A7C15));
    {
        uint64_t x2 = UINT64_C(0x165667B19E3779F9) ^ (uint64_t)n;
        uint64_t x3 = UINT64_C(0x27D4EB2F165667C5) +
                      ((uint64_t)n * UINT64_C(0x85EBCA77));

        for (; i + 32u <= n; i += 32u) {
            uint64_t v0, v1, v2, v3;

            memcpy(&v0, page + i, 8u);
            memcpy(&v1, page + i + 8u, 8u);
            memcpy(&v2, page + i + 16u, 8u);
            memcpy(&v3, page + i + 24u, 8u);
            a = (a ^ v0) * UINT64_C(0xff51afd7ed558ccd);
            a ^= a >> 29;
            b = (b + v1) * UINT64_C(0xc4ceb9fe1a85ec53);
            b ^= b >> 31;
            x2 = (x2 ^ v2) * UINT64_C(0xff51afd7ed558ccd);
            x2 ^= x2 >> 27;
            x3 = (x3 + v3) * UINT64_C(0xc4ceb9fe1a85ec53);
            x3 ^= x3 >> 33;
        }
        for (; i + 8u <= n; i += 8u) {
            uint64_t v;

            memcpy(&v, page + i, 8u);
            a = (a ^ v) * UINT64_C(0xff51afd7ed558ccd);
            a ^= a >> 32;
        }
        for (; i < n; i++) {
            b = (b + (uint64_t)page[i] + 1u) *
                UINT64_C(0x9E3779B97F4A7C15);
            b ^= b >> 31;
        }
        /* Both halves are folded from all four lanes before they are
         * finalised. A half that saw only its own two lanes would make the
         * index (lo's low bits) blind to half the page -- and lo is what picks
         * the entry, so two pages differing in a b-lane byte would land alike
         * with nothing else to separate them. */
        {
            uint64_t m1 = a ^ ((x2 << 7) | (x2 >> 57)) ^
                          (x3 * UINT64_C(0xff51afd7ed558ccd)) ^ (b >> 1);
            uint64_t m2 = b ^ ((x3 << 23) | (x3 >> 41)) ^
                          (x2 * UINT64_C(0xc4ceb9fe1a85ec53)) ^ (a >> 1);

            a = zpse_page_fmix(m1);
            b = zpse_page_fmix(m2);
        }
    }
    /* 0/0 is the empty-entry tag; a real page that hashed to it would be a
     * permanent miss, so it cannot be handed out. */
    if ((a | b) == 0u) {
        a = 1u;
    }
    *lo_out = a;
    *hi_out = b;
}

void zpse_pagedict_init_set(ZpsePageDict *d, uint64_t *tag_lo,
                            uint64_t *tag_hi, uint32_t *slot, uint8_t *victim,
                            uint32_t entries, uint32_t ways)
{
    if (d == 0) {
        return;
    }
    d->tag_lo = tag_lo;
    d->tag_hi = tag_hi;
    d->slot = slot;
    d->victim = victim;
    d->entries = 0u;
    d->ways = 0u;
    /* A non-power-of-two size would make the mask wrong rather than slow, so
     * the dictionary reports itself empty and every lookup misses. */
    if (tag_lo == 0 || tag_hi == 0 || slot == 0 || entries < 16u ||
        (entries & (entries - 1u)) != 0u) {
        return;
    }
    if (ways != 1u) {
        /* A way needs a victim cursor, and four sets is the smallest table the
         * measured 2.6% at 4,096 entries was taken on; below that the mapping
         * loses more to its own collisions than the ways recover. */
        if (victim == 0 || ways < 2u || ways > 8u ||
            (ways & (ways - 1u)) != 0u || (entries / ways) < 4u) {
            return;
        }
        memset(victim, 0, (size_t)(entries / ways));
    }
    memset(tag_lo, 0, (size_t)entries * sizeof *tag_lo);
    memset(tag_hi, 0, (size_t)entries * sizeof *tag_hi);
    memset(slot, 0, (size_t)entries * sizeof *slot);
    d->entries = entries;
    d->ways = ways;
}

void zpse_pagedict_init(ZpsePageDict *d, uint64_t *tag_lo, uint64_t *tag_hi,
                        uint32_t *slot, uint32_t entries)
{
    /* Direct mapped: every set is one slot, so there is no eviction order to
     * keep and no cursor array to own. */
    zpse_pagedict_init_set(d, tag_lo, tag_hi, slot, 0, entries, 1u);
}

int zpse_pagedict_find(const ZpsePageDict *d, uint64_t lo, uint64_t hi,
                       uint32_t *slot_out)
{
    uint32_t sets;
    uint32_t base;
    uint32_t w;

    if (d == 0 || d->entries == 0u || d->ways == 0u || slot_out == 0) {
        return 0;
    }
    if ((lo | hi) == 0u) {
        return 0;
    }
    /* The tag is already a hash, so its low bits are uniform; the mask is the
     * whole of the index arithmetic. A set's ways are contiguous, so the
     * compare is one cache line per two tags on the common sizes. */
    sets = d->entries / d->ways;
    base = (uint32_t)(lo & (uint64_t)(sets - 1u)) * d->ways;
    for (w = 0; w < d->ways; w++) {
        uint32_t i = base + w;

        if (d->tag_lo[i] == lo && d->tag_hi[i] == hi) {
            *slot_out = d->slot[i];
            return 1;
        }
    }
    return 0;
}

void zpse_pagedict_insert(ZpsePageDict *d, uint64_t lo, uint64_t hi,
                          uint32_t slot)
{
    uint32_t sets;
    uint32_t set;
    uint32_t base;
    uint32_t w;

    if (d == 0 || d->entries == 0u || d->ways == 0u) {
        return;
    }
    if ((lo | hi) == 0u) {
        lo = 1u;
    }
    sets = d->entries / d->ways;
    set = (uint32_t)(lo & (uint64_t)(sets - 1u));
    base = set * d->ways;
    /* A tag that is already in the set updates its slot in place: the store is
     * re-inserting the same content, and spending a second way on it would evict
     * a live entry and then find the page twice over. */
    for (w = 0; w < d->ways; w++) {
        uint32_t i = base + w;

        if (d->tag_lo[i] == lo && d->tag_hi[i] == hi) {
            d->slot[i] = slot;
            return;
        }
    }
    w = base;
    if (d->ways != 1u) {
        w += d->victim[set];
        d->victim[set] = (uint8_t)((d->victim[set] + 1u) & (d->ways - 1u));
    }
    d->tag_lo[w] = lo;
    d->tag_hi[w] = hi;
    d->slot[w] = slot;
}

int zpse_block_retag(uint8_t *block, uint32_t block_len,
                     uint32_t seq, int is_last)
{
    ZpseBlockHeader h;
    uint32_t hsz;
    int last;
    int rc;

    if (block == 0) {
        return ZPSE_BAD_ARG;
    }
    if (block_len < (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        return ZPSE_TRUNCATED;
    }
    hsz = zpse_block_header_size(block, block_len);
    if (hsz == 0u) {
        return ZPSE_BAD_HEADER;
    }
    if (block_len < hsz) {
        return ZPSE_TRUNCATED;
    }
    if (zpse_load_u16le(block) == (uint16_t)ZPSE_BLOCK_SYNC_TOKEN) {
        /* A one-value token is all header, and the seq/last bits live at the
         * same offsets as a page's short header. Retag in place and refresh
         * the 16-bit CRC over the first 14 bytes, exactly as the short form
         * serialiser does. */
        uint8_t flags = block[2];

        if (block[4] != 1u || block[5] != 0u || hsz != 16u ||
            zpse_load_u16le(block + 8) != 16u) {
            return ZPSE_BAD_HEADER;
        }
        if ((uint32_t)(zpse_crc32c(block, 14) & UINT32_C(0xFFFF)) !=
            (uint32_t)zpse_load_u16le(block + 14)) {
            return ZPSE_BAD_HEADER_CRC;
        }
        if ((zpse_load_u32le(block + 10) == seq) &&
            (((flags & ZPSE_B_LAST) != 0u) ? 1 : 0) == (is_last ? 1 : 0)) {
            return ZPSE_OK;
        }
        zpse_store_u32le(block + 10, seq);
        block[2] = (uint8_t)(is_last ? (flags | ZPSE_B_LAST)
                                     : (flags & ~ZPSE_B_LAST));
        zpse_store_u16le(block + 14,
                         (uint16_t)(zpse_crc32c(block, 14) & UINT32_C(0xFFFF)));
        return ZPSE_OK;
    }
    if (zpse_load_u16le(block) == (uint16_t)ZPSE_BLOCK_SYNC_FLAT) {
        ZpseBlockHeader fh;
        uint8_t flags;
        int last;
        int frc;

        frc = zpse_block_header_parse_flat(&fh, block, block_len);
        if (frc != ZPSE_OK) {
            return frc;
        }
        if (fh.compressed_block_size != block_len) {
            return ZPSE_BAD_HEADER;
        }
        flags = fh.block_flags;
        last = (is_last != 0) ? 1 : 0;
        if (fh.block_sequence_number == seq &&
            (((flags & ZPSE_B_LAST) != 0u) ? 1 : 0) == last) {
            return ZPSE_OK;
        }
        fh.block_sequence_number = seq;
        fh.block_flags = (uint8_t)(last != 0 ? (flags | ZPSE_B_LAST)
                                             : (flags & ~ZPSE_B_LAST));
        /* The flat serialiser writes seq and LAST and recomputes the 16-bit
         * CRC over the first 14 bytes; the codec, param and sizes stay. */
        zpse_block_header_serialize_flat(block, &fh);
        return ZPSE_OK;
    }
    if (zpse_load_u16le(block) == (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
        ZpseBlockHeader th;
        uint8_t flags;
        int last;
        int trc;

        /* This must stay ahead of the generic path below: that path hardcodes
         * the SHORT sync on rewrite and would silently turn a TEXT block into
         * a short-header one, corrupting its payload. */
        trc = zpse_block_header_parse_text(&th, block, block_len);
        if (trc != ZPSE_OK) {
            return trc;
        }
        if (th.compressed_block_size != block_len) {
            return ZPSE_BAD_HEADER;
        }
        flags = th.block_flags;
        last = (is_last != 0) ? 1 : 0;
        if (th.block_sequence_number == seq &&
            (((flags & ZPSE_B_LAST) != 0u) ? 1 : 0) == last) {
            return ZPSE_OK;
        }
        th.block_sequence_number = seq;
        th.block_flags = (uint8_t)(last != 0 ? (flags | ZPSE_B_LAST)
                                             : (flags & ~ZPSE_B_LAST));
        /* The TEXT serialiser writes seq and LAST and recomputes the 16-bit
         * CRC over the first 14 bytes; reserved bytes and sizes stay. */
        zpse_block_header_serialize_text(block, &th);
        return ZPSE_OK;
    }
    /* The parser is the validator: magic/version/flags/size/transform checks
     * and the header CRC, so a block that has been corrupted, or that is not a
     * block at all, is refused before a byte of it is rewritten. */
    rc = zpse_block_header_parse(&h, block, block_len);
    if (rc != ZPSE_OK) {
        return rc;
    }
    /* `block_len` is the length the caller has, and the header says how long
     * the block is; a disagreement means the caller is holding a prefix or a
     * block with trailing bytes, and either way the payload is not what the
     * header describes. */
    if (h.compressed_block_size != block_len) {
        return ZPSE_BAD_HEADER;
    }
    last = (is_last != 0) ? 1 : 0;
    if (h.block_sequence_number == seq &&
        (((h.block_flags & ZPSE_B_LAST) != 0u) ? 1 : 0) == last) {
        return ZPSE_OK;
    }
    h.block_sequence_number = seq;
    if (last != 0) {
        h.block_flags = (uint8_t)(h.block_flags | ZPSE_B_LAST);
    } else {
        h.block_flags = (uint8_t)(h.block_flags & ~ZPSE_B_LAST);
    }
    /* Both serialisers write the header checksum last, over everything they
     * wrote -- the 32-byte form's CRC32-C over 28 bytes, a page's 16-bit
     * truncation over 14 -- so no separate fixup is paid here. */
    if (hsz == (uint32_t)ZPSE_BLOCK_HEADER_SIZE_SHORT) {
        zpse_block_header_serialize_short(block, &h);
    } else {
        zpse_block_header_serialize(block, &h);
    }
    return ZPSE_OK;
}
