/* Multi-core frame encode: the same frame, from blocks encoded concurrently.
 *
 * The format lets blocks be encoded in any order -- a block may not
 * reference another -- and the only thing that forces a frame to be built
 * front to back is the output layout. So a worker encodes block k into a
 * reserved slot and one serial pass compacts the slots. That pass is where
 * a bug would quietly write a frame that decodes to the wrong bytes, which
 * is why this suite's central check is not "it round-trips" but "it is the
 * same frame the serial encoder writes, byte for byte".
 *
 * Every case asserts, for one (size, block size, level, checksum, index,
 * thread count) combination:
 *
 *   - the threaded frame equals the serial one, byte for byte, and both
 *     equal what the public serial entry points write;
 *   - it decodes to the input;
 *   - when it carries an index, the count is the block count and a seek to
 *     the first and last block returns their bytes;
 *   - a worker count larger than the block count is clamped, not an error.
 *
 * In a build without ZPSE_MT the parallel call is the serial encoder, so
 * this suite still runs and still checks the layout -- it just has no
 * second thread to disagree with.
 */
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"

/* The largest input any case uses. The streamed cases carry a short final
 * block past 400 KiB on purpose (a 409677-byte slab at 16 KiB blocks ends
 * mid-block), so the buffers are sized to hold that input, not 400 KiB. */
#define MAXN (416u * 1024u)
#define MAXTHREADS 6u
#define BLKMAX 65536u
/* Serial bound (input + 96 B per block + the index) and the reserved slot
 * layout of the worker path, whichever is larger. */
#define COMP_CAP (MAXN + MAXN / 8u + 65536u)

/* MSVC has no __attribute__; the arenas have to be 64-byte aligned either
 * way, so the spelling is the only thing that differs. */
#if defined(_MSC_VER)
#define ZPSE_ALIGN64 __declspec(align(64))
#else
#define ZPSE_ALIGN64 __attribute__((aligned(64)))
#endif

/* The alignment spelling is a prefix on MSVC and may be a prefix or a
 * suffix on gcc; putting it first is the form both accept. */
ZPSE_ALIGN64 static unsigned char ea[MAXTHREADS][ZPSE_ENC_ARENA_SIZE];
ZPSE_ALIGN64 static unsigned char da[ZPSE_DEC_ARENA_SIZE];
ZPSE_ALIGN64 static unsigned char dmt[MAXTHREADS][ZPSE_DEC_ARENA_SIZE];

static void *arenas[MAXTHREADS] = { ea[0], ea[1], ea[2], ea[3], ea[4], ea[5] };
static void *decs[MAXTHREADS] = { dmt[0], dmt[1], dmt[2],
                                  dmt[3], dmt[4], dmt[5] };

static uint8_t in[MAXN];
static uint8_t ref[COMP_CAP];
static uint8_t par[COMP_CAP];
static uint8_t plain[COMP_CAP];
static uint8_t out[MAXN];
static uint8_t blk[BLKMAX];

static int fails = 0;
static int checks = 0;
static uint32_t rng = 0x9E3779B9u;
static uint32_t digest = 2166136261u;

#define CHECK(cond, ...) do {                       \
    checks++;                                       \
    if (!(cond)) {                                  \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
        fails++;                                    \
    }                                               \
} while (0)

static uint32_t rnd(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void digest_bytes(const uint8_t *p, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++) {
        digest = (digest ^ p[i]) * 16777619u;
    }
}

/* The TEXT form's own kind: deterministic ASCII prose. A fixed-seed local
 * LCG picks words from a fixed vocabulary, so the byte stream is identical on
 * every call and on every platform, independent of the suite's shared rng.
 * Word-level repetition is what makes TEXT win the block. */
static void fill_text(uint32_t n)
{
    static const char *const w[24] = {
        "the", "of", "and", "a", "to", "in", "is", "that", "for",
        "it", "as", "was", "with", "be", "by", "on", "not", "he",
        "this", "are", "or", "his", "from", "at"
    };
    uint32_t seed = 0x13579BDFu;
    uint32_t at = 0u;
    uint32_t k = 0u;

    while (at < n) {
        const char *s;
        uint32_t j;

        seed = seed * 1103515245u + 12345u;
        s = w[(seed >> 16) % 24u];
        for (j = 0; s[j] != '\0' && at < n; j++) {
            in[at++] = (uint8_t)s[j];
        }
        if (at < n) {
            in[at++] = (uint8_t)((k % 13u == 12u) ? '.' : ' ');
        }
        k++;
    }
}

static void fill_input(uint32_t n, int kind)
{
    uint32_t i;

    if (kind == 4) {
        fill_text(n);
        return;
    }
    for (i = 0; i < n; i++) {
        switch (kind) {
        case 0: /* ramp in 8-byte lanes: transposes and stride kernels */
            in[i] = (uint8_t)(i / 8u);
            break;
        case 1: /* noise: the raw form, the widest per-block cost */
            in[i] = (uint8_t)rnd();
            break;
        case 2: /* prose-ish: the tile and LZ forms */
            in[i] = (uint8_t)("the quick brown fox jumps over the lazy dog. "
                              "[i]"[(i / 7u) % 48u]);
            break;
        default: /* runs of one value, then a gap */
            in[i] = (uint8_t)((i & 0x3FFu) < 900u ? 0u : (i & 0x3Fu));
            break;
        }
    }
}

static void run_case(uint32_t n, uint32_t block, unsigned threads,
                     uint32_t enc_flags, int kind, int indexed)
{
    size_t rlen = 0u;
    size_t plen = 0u;
    size_t qlen = 0u;
    size_t got = 0u;
    size_t pgot = 0u;
    uint32_t nblocks = n / block + (n % block != 0u ? 1u : 0u);
    int rc;

    fill_input(n, kind);

    /* The public serial entry points: the reference this suite is about. */
    if (indexed) {
        rc = zpse_compress_frame_indexed(in, n, ref, sizeof ref, &rlen,
                                         block, enc_flags, arenas[0]);
    } else {
        rc = zpse_compress_frame_ex(in, n, ref, sizeof ref, &rlen, block,
                                    enc_flags, arenas[0]);
    }
    CHECK(rc == ZPSE_OK, "serial encode (n %u, block %u, flags %u): %s",
          n, block, enc_flags, zpse_status_name(rc));
    if (rc != ZPSE_OK) {
        return;
    }

    /* threads == 1 is the documented fallback, and has to be the same
     * bytes as the reference, not merely a decodable frame. */
    rc = zpse_compress_frame_parallel(in, n, plain, sizeof plain, &qlen, block,
                                      enc_flags, indexed, 1u,
                                      (void *const *)arenas, 1u);
    CHECK(rc == ZPSE_OK, "one-worker encode (n %u, block %u)", n, block);
    CHECK(qlen == rlen && memcmp(plain, ref, rlen) == 0,
          "one worker wrote a different frame (n %u, block %u, flags %u, "
          "index %d): %lu vs %lu bytes",
          n, block, enc_flags, indexed, (unsigned long)qlen,
          (unsigned long)rlen);

    rc = zpse_compress_frame_parallel(in, n, par, sizeof par, &plen, block,
                                      enc_flags, indexed, threads,
                                      (void *const *)arenas, MAXTHREADS);
    CHECK(rc == ZPSE_OK, "%u workers encode (n %u, block %u): %s", threads,
          n, block, zpse_status_name(rc));
    if (rc != ZPSE_OK) {
        return;
    }
    CHECK(plen == rlen, "%u workers wrote %lu bytes, serial %lu (n %u, "
          "block %u, flags %u)", threads, (unsigned long)plen,
          (unsigned long)rlen, n, block, enc_flags);
    CHECK(plen == rlen && memcmp(par, ref, rlen) == 0,
          "%u workers wrote different bytes (n %u, block %u, flags %u, "
          "index %d)", threads, n, block, enc_flags, indexed);

    /* Form selection is a pure function of (input bytes, enc_flags), not of
     * how those bytes were split across workers. Parse the first block of
     * the reference, the one-worker fallback and the threaded frame and
     * require the same chosen form (sync) and the same first-block bytes in
     * all three -- not merely an equal frame length. The pattern is the one
     * the batch walk below uses (frame header, then one header per block). */
    if (rlen > (size_t)ZPSE_FRAME_HEADER_SIZE) {
        ZpseBlockHeader hb[3];
        const uint8_t *bp[3] = {
            ref + ZPSE_FRAME_HEADER_SIZE,
            plain + ZPSE_FRAME_HEADER_SIZE,
            par + ZPSE_FRAME_HEADER_SIZE
        };
        uint32_t avail = (uint32_t)(rlen - (size_t)ZPSE_FRAME_HEADER_SIZE);

        CHECK(zpse_block_header_parse(&hb[0], bp[0], avail) == ZPSE_OK &&
              zpse_block_header_parse(&hb[1], bp[1], avail) == ZPSE_OK &&
              zpse_block_header_parse(&hb[2], bp[2], avail) == ZPSE_OK,
              "form parse (n %u, block %u, flags %u, index %d)", n, block,
              enc_flags, indexed);
        CHECK(hb[0].sync == hb[1].sync && hb[0].sync == hb[2].sync,
              "form sync differs (n %u, block %u, flags %u, index %d): "
              "%04x/%04x/%04x", n, block, enc_flags, indexed,
              (unsigned)hb[0].sync, (unsigned)hb[1].sync,
              (unsigned)hb[2].sync);
        CHECK(hb[0].compressed_block_size >= (uint32_t)ZPSE_BLOCK_HEADER_SIZE &&
              (size_t)hb[0].compressed_block_size <=
                  rlen - (size_t)ZPSE_FRAME_HEADER_SIZE &&
              memcmp(bp[0], bp[1], hb[0].compressed_block_size) == 0 &&
              memcmp(bp[0], bp[2], hb[0].compressed_block_size) == 0,
              "first block bytes differ (n %u, block %u, flags %u, index %d)",
              n, block, enc_flags, indexed);
    }
    digest_bytes(par, plen);

    /* What the frame is for: it has to decode to the input, in one pass. */
    rc = zpse_decompress_frame(par, plen, out, sizeof out, &got, da);
    CHECK(rc == ZPSE_OK, "%u workers: decode (n %u)", threads, n);
    CHECK(rc == ZPSE_OK && got == n && memcmp(out, in, n) == 0,
          "%u workers: decoded bytes (n %u, block %u, flags %u)", threads, n,
          block, enc_flags);

    /* The read side, on the same frame. A frame without the index cannot be
     * decoded in parallel -- block k's offset is only in the table -- so the
     * call falls back to the sequential decoder there, and either way the
     * output has to be the input. */
    memset(out, 0, n);
    rc = zpse_decompress_frame_parallel(par, plen, out, sizeof out, &pgot,
                                        threads, (void *const *)decs,
                                        MAXTHREADS);
    CHECK(rc == ZPSE_OK, "%u workers: parallel decode (n %u, index %d): %s",
          threads, n, indexed, zpse_status_name(rc));
    CHECK(rc == ZPSE_OK && pgot == n && memcmp(out, in, n) == 0,
          "%u workers: parallel decoded bytes (n %u, block %u, index %d)",
          threads, n, block, indexed);

    /* The batch decoder a streaming caller uses: the frame's blocks handed
     * over in one call, decoded side by side, must be the sequential decode
     * block for block. This is the read side the CLI drives while it walks a
     * file, so it is checked against the bytes rather than against itself. */
    if (nblocks <= 128u) {
        static uint8_t bin[COMP_CAP];
        static uint8_t bout[MAXN];
        uint32_t boff[128], blen[128], bgot[128];
        size_t pos = (size_t)ZPSE_FRAME_HEADER_SIZE;
        uint32_t k;
        int brc = ZPSE_OK;

        for (k = 0u; k < nblocks && brc == ZPSE_OK; k++) {
            ZpseBlockHeader bh;

            if (zpse_block_header_parse(&bh, par + pos,
                                        (uint32_t)ZPSE_BLOCK_HEADER_SIZE) !=
                    ZPSE_OK ||
                bh.compressed_block_size < (uint32_t)ZPSE_BLOCK_HEADER_SIZE ||
                pos + (size_t)bh.compressed_block_size > plen) {
                brc = ZPSE_BAD_HEADER;
                break;
            }
            memcpy(bin + pos, par + pos, (size_t)bh.compressed_block_size);
            boff[k] = (uint32_t)pos;
            blen[k] = bh.compressed_block_size;
            pos += (size_t)bh.compressed_block_size;
        }
        if (brc == ZPSE_OK) {
            brc = zpse_decompress_batch_parallel(
                bin, boff, blen, nblocks, bout, block, bgot, threads,
                (void *const *)decs, MAXTHREADS);
        }
        CHECK(brc == ZPSE_OK,
              "%u workers: batch decode (n %u, block %u): %s", threads, n,
              block, zpse_status_name(brc));
        if (brc == ZPSE_OK) {
            for (k = 0u; k < nblocks; k++) {
                uint32_t want = (k + 1u == nblocks) ? (uint32_t)(n - (size_t)k * block)
                                                    : block;

                if (want > block) {
                    want = block;
                }
                if (bgot[k] != want ||
                    memcmp(bout + (size_t)k * block, in + (size_t)k * block,
                           want) != 0) {
                    CHECK(0, "%u workers: batch block %u (n %u, block %u)",
                          threads, k, n, block);
                    break;
                }
            }
        }
    }

    if (indexed) {
        uint32_t count = 0u;
        uint32_t last = nblocks - 1u;
        uint32_t want = 0u;
        uint32_t sgot = 0u;
        uint32_t off;

        CHECK(zpse_frame_index_count(par, plen, &count) == ZPSE_OK,
              "%u workers: index reads", threads);
        CHECK(count == nblocks, "%u workers: %u entries of %u blocks",
              threads, count, nblocks);
        CHECK(zpse_seek_frame(par, plen, 0u, blk, sizeof blk, &want, da) ==
                  ZPSE_OK && want == block && memcmp(blk, in, block) == 0,
              "%u workers: seek of block 0 (n %u, block %u)", threads, n,
              block);
        off = last * block;
        want = n - off;
        if (want > block) {
            want = block;
        }
        CHECK(zpse_seek_frame(par, plen, last, blk, sizeof blk, &sgot, da) ==
                  ZPSE_OK && sgot == want && memcmp(blk, in + off, want) == 0,
              "%u workers: seek of block %u (n %u, block %u)", threads, last,
              n, block);
    }
}

/* ---- the streamed encode --------------------------------------------
 *
 * The path the CLI ships: a slab in, finished blocks out through a callback,
 * a ring of slots between. Its contract is the buffered path's -- the bytes
 * the serial encoder writes -- and the ring is exactly where an earlier
 * version of this went wrong: when the assignment came from a shared claim
 * counter, a slot's previous generation could be handed out under the next
 * block's length, so six workers sometimes wrote a different stream of the
 * same size. So every worker count is compared against the serial frame's
 * block region, at the shallowest and the deepest ring, and the callback
 * refuses a block that arrives out of order.
 */

#define STRM_DEPTH_MAX 4u
#define STRM_CTL (MAXTHREADS * STRM_DEPTH_MAX * ZPSE_MT_CTRL_STRIDE)
static uint8_t slotmem[MAXTHREADS * STRM_DEPTH_MAX *
                       (BLKMAX + ZPSE_BLOCK_HEADER_SIZE)];
static int ready[STRM_CTL];
static uint32_t lens[STRM_CTL];
static uint8_t strm[COMP_CAP];

typedef struct {
    uint32_t next;
    size_t len;
    int bad;
} StrmSink;

static int stream_emit(void *ctx, uint32_t seq, const uint8_t *bytes,
                       size_t len)
{
    StrmSink *s = (StrmSink *)ctx;

    if (seq != s->next || s->len + len > sizeof strm) {
        s->bad = 1;
        return ZPSE_BAD_ARG;
    }
    memcpy(strm + s->len, bytes, len);
    s->len += len;
    s->next = seq + 1u;
    return 0;
}

static void run_stream_case(uint32_t n, uint32_t block, unsigned max_threads,
                            uint32_t enc_flags, int kind, int indexed,
                            unsigned depth)
{
    size_t rlen = 0u;
    size_t stride = (size_t)block + (size_t)ZPSE_BLOCK_HEADER_SIZE;
    uint32_t nblocks = n / block + (n % block != 0u ? 1u : 0u);
    size_t body;
    unsigned t;
    int rc;

    fill_input(n, kind);
    if (indexed) {
        rc = zpse_compress_frame_indexed(in, n, ref, sizeof ref, &rlen, block,
                                         enc_flags, arenas[0]);
    } else {
        rc = zpse_compress_frame_ex(in, n, ref, sizeof ref, &rlen, block,
                                    enc_flags, arenas[0]);
    }
    CHECK(rc == ZPSE_OK, "stream: serial reference (n %u, block %u)", n, block);
    if (rc != ZPSE_OK) {
        return;
    }
    /* The stream carries blocks and nothing else: no frame header and, when
     * the frame has one, no index. */
    body = rlen - (size_t)ZPSE_FRAME_HEADER_SIZE -
           (indexed ? (size_t)ZPSE_INDEX_HEADER_SIZE +
                          (size_t)ZPSE_INDEX_ENTRY_SIZE * (size_t)nblocks
                    : 0u);
    for (t = 1u; t <= max_threads; t++) {
        StrmSink sink;
        size_t slot_cap = stride * (size_t)t * (size_t)depth;

        sink.next = 0u;
        sink.len = 0u;
        sink.bad = 0;
        rc = zpse_compress_frame_stream(in, n, block, enc_flags, t, 0u, 1,
                                        (void *const *)arenas, MAXTHREADS,
                                        slotmem, slot_cap, ready, lens,
                                        stream_emit, &sink);
        CHECK(rc == ZPSE_OK,
              "stream encode (%u workers, depth %u, n %u): %s", t, depth,
              n, zpse_status_name(rc));
        CHECK(sink.bad == 0,
              "stream emitted out of order (%u workers, depth %u, n %u)", t,
              depth, n);
        CHECK(sink.len == body &&
                  memcmp(strm, ref + ZPSE_FRAME_HEADER_SIZE, body) == 0,
              "stream body differs (%u workers, depth %u, n %u, block %u, "
              "flags %u, index %d): %lu vs %lu bytes",
              t, depth, n, block, enc_flags, indexed,
              (unsigned long)sink.len, (unsigned long)body);
        if (t == max_threads) {
            digest_bytes(strm, sink.len);
        }
    }
}

/* ---- whole-frame streams on the parallel decoder -----------------------
 * DCM1/MRD1 streams are indivisible: they carry no FrameIndex, so the
 * parallel entry point has to hand them to the serial dispatcher. The two
 * builders below are copied from test_hostile.c so these valid streams are
 * written by hand, independent of the encoder. */
static uint32_t wf_dpcm1_build(uint8_t *b, uint32_t ns, uint16_t first,
                               const uint8_t *deltas,
                               const uint8_t *prefix, uint32_t plen,
                               const uint8_t *esctab, uint32_t ne)
{
    uint32_t nd = ns - 1u;
    uint32_t n = 2u * ns + plen;
    uint32_t i;

    memset(b, 0, 16u + plen + nd + 6u * ne);
    b[0] = 'D'; b[1] = 'C'; b[2] = 'M'; b[3] = '1';
    zpse_store_u32le(b + 4u, n);
    zpse_store_u16le(b + 8u, first);
    zpse_store_u16le(b + 10u, (uint16_t)plen);
    zpse_store_u32le(b + 12u, ne);
    for (i = 0u; i < plen; i++) b[16u + i] = prefix[i];
    for (i = 0u; i < nd; i++) b[16u + plen + i] = deltas[i];
    for (i = 0u; i < 6u * ne; i++) b[16u + plen + nd + i] = esctab[i];
    return 16u + plen + nd + 6u * ne;
}

static uint32_t wf_dpcm1_base(uint8_t *b, uint8_t *deltas)
{
    uint32_t i;

    for (i = 0u; i < 16u; i++) deltas[i] = 20u;
    return wf_dpcm1_build(b, 16u, 0x1000u, deltas, 0, 0u, 0, 0u);
}

static uint32_t wf_mrd_build(uint8_t *b, uint32_t n, uint16_t first,
                             const uint8_t *cs, uint32_t coff,
                             const uint8_t *esctab, uint32_t ne,
                             uint32_t nruns)
{
    uint32_t i;

    memset(b, 0, 24u + coff + 6u * ne);
    b[0] = 'M'; b[1] = 'R'; b[2] = 'D'; b[3] = '1';
    zpse_store_u32le(b + 4u, n);
    zpse_store_u16le(b + 8u, first);
    zpse_store_u16le(b + 10u, 0u);
    zpse_store_u32le(b + 12u, ne);
    zpse_store_u32le(b + 16u, coff);
    zpse_store_u32le(b + 20u, nruns);
    for (i = 0u; i < coff; i++) b[24u + i] = cs[i];
    for (i = 0u; i < 6u * ne; i++) b[24u + coff + i] = esctab[i];
    return 24u + coff + 6u * ne;
}

static void check_whole_frame_decode(const char *name, const uint8_t *b,
                                     size_t len, size_t want_n)
{
    static const unsigned thread_counts[] = { 1u, 4u };
    uint8_t serial_out[256u + 16u];
    uint8_t par_out[256u + 16u];
    size_t slen = 0u;
    unsigned k;

    memset(serial_out, 0xCC, sizeof serial_out);
    CHECK(zpse_decompress_frame(b, len, serial_out, sizeof serial_out, &slen,
                                da) == ZPSE_OK && slen == want_n,
          "%s: serial decode", name);
    for (k = 0u; k < sizeof thread_counts / sizeof thread_counts[0]; k++) {
        unsigned t = thread_counts[k];
        size_t got = 0u;
        size_t i;
        int clean;
        int rc;

        memset(par_out, 0xCC, sizeof par_out);
        rc = zpse_decompress_frame_parallel(b, len, par_out, sizeof par_out,
                                            &got, t, (void *const *)decs,
                                            MAXTHREADS);
        clean = (rc == ZPSE_OK && got == slen &&
                 memcmp(par_out, serial_out, slen) == 0);
        for (i = slen; i < sizeof par_out; i++) {
            if (par_out[i] != 0xCCu) clean = 0;
        }
        CHECK(clean, "%s: %u workers decode byte-exactly (%s)", name, t,
              zpse_status_name(rc));
    }
}

static void whole_frame_decode(void)
{
    uint8_t b[256u];
    uint8_t deltas[80u];
    uint32_t len;

    /* DCM1: 16 samples, all deltas +20 -> a 31-byte stream. */
    len = wf_dpcm1_base(b, deltas);
    CHECK(len == 31u, "dpcm1: hand-built stream is 31 bytes");
    check_whole_frame_decode("DCM1", b, len, 32u);

    /* MRD1: 81 samples, 80 literal +10 deltas -> a 104-byte stream. */
    {
        uint32_t i;

        for (i = 0u; i < 80u; i++) deltas[i] = 10u;
    }
    len = wf_mrd_build(b, 162u, 1000u, deltas, 80u, 0u, 0u, 0u);
    CHECK(len == 104u, "mrd1: hand-built stream is 104 bytes");
    check_whole_frame_decode("MRD1", b, len, 162u);

    /* The lower floor must not turn a genuinely short or corrupt PSE1 frame
     * into a false success on the serial delegation path. */
    {
        size_t got = 0u;
        uint8_t bad[32u];

        memset(bad, 'X', sizeof bad);
        CHECK(zpse_decompress_frame_parallel(bad, 4u, out, sizeof out, &got,
                                             4u, (void *const *)decs,
                                             MAXTHREADS) == ZPSE_TRUNCATED,
              "short PSE1 stream is still TRUNCATED");
        CHECK(zpse_decompress_frame_parallel(bad, sizeof bad, out,
                                             sizeof out, &got, 4u,
                                             (void *const *)decs,
                                             MAXTHREADS) == ZPSE_BAD_MAGIC,
              "corrupt PSE1 magic is still BAD_MAGIC");
    }
}

int main(void)
{
    static const unsigned counts[] = { 2u, 3u, 4u, MAXTHREADS };

    /* One block: the worker path has nothing to spread, so it has to fall
     * back without changing a byte. */
    run_case(65536u, 65536u, MAXTHREADS, 0u, 0, 0);
    run_case(65536u, 65536u, MAXTHREADS, (uint32_t)ZPSE_ENC_SPEED_MAX, 2, 1);

    /* The TEXT form's own kind: deterministic ASCII prose at both block
     * sizes it is written for (4 KiB pages and full 64 KiB blocks), indexed
     * and not, at the one-worker fallback and the many-worker path. run_case
     * requires every worker count to produce the serial frame byte for byte,
     * and the form-selection check above requires the same chosen form in the
     * reference, the fallback and the threaded frame. */
    {
        static const uint32_t blocks[] = { 4096u, 65536u };
        static const unsigned counts_text[] = { 1u, MAXTHREADS };
        const uint32_t n = 128u * 1024u; /* 32 pages or 2 full blocks */
        unsigned bi, ti;
        int indexed;

        for (bi = 0; bi < sizeof blocks / sizeof blocks[0]; bi++) {
            for (indexed = 0; indexed <= 1; indexed++) {
                for (ti = 0;
                     ti < sizeof counts_text / sizeof counts_text[0]; ti++) {
                    run_case(n, blocks[bi], counts_text[ti], 0u, 4, indexed);
                }
            }
        }
    }

    /* A short final block, at four block sizes, every worker count. */
    {
        static const uint32_t blocks[] = { 4096u, 16384u, 32768u, 65536u };
        unsigned bi;
        unsigned ci;

        for (bi = 0; bi < sizeof blocks / sizeof blocks[0]; bi++) {
            for (ci = 0; ci < sizeof counts / sizeof counts[0]; ci++) {
                run_case(262144u + 77u, blocks[bi], counts[ci], 0u, 0, 1);
            }
        }
    }

    /* The levels that write the LZ form, and the payload checksum, on the
     * two kinds of data that reach them. */
    run_case(400u * 1024u, 65536u, MAXTHREADS,
             (uint32_t)ZPSE_ENC_SPEED_MAX, 2, 1);
    run_case(400u * 1024u, 16384u, 3u,
             (uint32_t)ZPSE_ENC_SPEED_BALANCED, 1, 0);
    run_case(100000u, 4096u, 4u,
             (uint32_t)(ZPSE_ENC_CHECKSUM | ZPSE_ENC_SPEED_FASTER), 3, 1);
    run_case(300000u, 65536u, 2u,
             (uint32_t)(ZPSE_ENC_CHECKSUM | ZPSE_ENC_SPEED_FASTEST), 1, 0);

    /* More workers than blocks and more threads than arenas: both clamp. */
    run_case(4096u * 3u, 4096u, MAXTHREADS, 0u, 3, 1);
    {
        size_t plen = 0u;
        int rc;

        fill_input(4096u * 3u, 3);
        rc = zpse_compress_frame_parallel(in, 4096u * 3u, par, sizeof par,
                                          &plen, 4096u, 0u, 0, MAXTHREADS,
                                          (void *const *)arenas, 2u);
        CHECK(rc == ZPSE_OK, "fewer arenas than threads is accepted");
        digest_bytes(par, plen);
    }

    /* The streamed path, whole-slab calls: both ring depths, every worker
     * count, on a frame that reaches each of the forms the CLI can write. */
    run_stream_case(400u * 1024u, 65536u, MAXTHREADS, 0u, 2, 1, 1u);
    run_stream_case(400u * 1024u, 65536u, MAXTHREADS, 0u, 2, 1,
                    STRM_DEPTH_MAX);
    /* The TEXT kind, at both ring depths: the streamed path must select and
     * emit the same prose blocks as the buffered serial encoder. */
    run_stream_case(400u * 1024u, 65536u, MAXTHREADS, 0u, 4, 1, 1u);
    run_stream_case(400u * 1024u, 65536u, MAXTHREADS, 0u, 4, 1,
                    STRM_DEPTH_MAX);
    run_stream_case(400u * 1024u + 77u, 16384u, MAXTHREADS,
                    (uint32_t)ZPSE_ENC_SPEED_MAX, 1, 0, STRM_DEPTH_MAX);
    run_stream_case(192u * 1024u, 4096u, 4u,
                    (uint32_t)ZPSE_ENC_LZ_FAST, 2, 0, STRM_DEPTH_MAX);
    run_stream_case(96u * 1024u, 4096u, MAXTHREADS,
                    (uint32_t)(ZPSE_ENC_LZ_FAST | ZPSE_ENC_CHECKSUM), 3, 1,
                    3u);

    whole_frame_decode();

    printf("PARALLEL: %s (%d checks, digest %08x)\n",
           fails == 0 ? "ALL OK" : "FAILURES", checks, (unsigned)digest);
    return fails == 0 ? 0 : 1;
}
