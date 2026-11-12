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

static void fill_input(uint32_t n, int kind)
{
    uint32_t i;

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
    CHECK(rc == PSE_OK, "serial encode (n %u, block %u, flags %u): %s",
          n, block, enc_flags, zpse_status_name(rc));
    if (rc != PSE_OK) {
        return;
    }

    /* threads == 1 is the documented fallback, and has to be the same
     * bytes as the reference, not merely a decodable frame. */
    rc = zpse_compress_frame_parallel(in, n, plain, sizeof plain, &qlen, block,
                                      enc_flags, indexed, 1u,
                                      (void *const *)arenas, 1u);
    CHECK(rc == PSE_OK, "one-worker encode (n %u, block %u)", n, block);
    CHECK(qlen == rlen && memcmp(plain, ref, rlen) == 0,
          "one worker wrote a different frame (n %u, block %u, flags %u, "
          "index %d): %lu vs %lu bytes",
          n, block, enc_flags, indexed, (unsigned long)qlen,
          (unsigned long)rlen);

    rc = zpse_compress_frame_parallel(in, n, par, sizeof par, &plen, block,
                                      enc_flags, indexed, threads,
                                      (void *const *)arenas, MAXTHREADS);
    CHECK(rc == PSE_OK, "%u workers encode (n %u, block %u): %s", threads,
          n, block, zpse_status_name(rc));
    if (rc != PSE_OK) {
        return;
    }
    CHECK(plen == rlen, "%u workers wrote %lu bytes, serial %lu (n %u, "
          "block %u, flags %u)", threads, (unsigned long)plen,
          (unsigned long)rlen, n, block, enc_flags);
    CHECK(plen == rlen && memcmp(par, ref, rlen) == 0,
          "%u workers wrote different bytes (n %u, block %u, flags %u, "
          "index %d)", threads, n, block, enc_flags, indexed);
    digest_bytes(par, plen);

    /* What the frame is for: it has to decode to the input, in one pass. */
    rc = zpse_decompress_frame(par, plen, out, sizeof out, &got, da);
    CHECK(rc == PSE_OK, "%u workers: decode (n %u)", threads, n);
    CHECK(rc == PSE_OK && got == n && memcmp(out, in, n) == 0,
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
    CHECK(rc == PSE_OK, "%u workers: parallel decode (n %u, index %d): %s",
          threads, n, indexed, zpse_status_name(rc));
    CHECK(rc == PSE_OK && pgot == n && memcmp(out, in, n) == 0,
          "%u workers: parallel decoded bytes (n %u, block %u, index %d)",
          threads, n, block, indexed);

    if (indexed) {
        uint32_t count = 0u;
        uint32_t last = nblocks - 1u;
        uint32_t want = 0u;
        uint32_t sgot = 0u;
        uint32_t off;

        CHECK(zpse_frame_index_count(par, plen, &count) == PSE_OK,
              "%u workers: index reads", threads);
        CHECK(count == nblocks, "%u workers: %u entries of %u blocks",
              threads, count, nblocks);
        CHECK(zpse_seek_frame(par, plen, 0u, blk, sizeof blk, &want, da) ==
                  PSE_OK && want == block && memcmp(blk, in, block) == 0,
              "%u workers: seek of block 0 (n %u, block %u)", threads, n,
              block);
        off = last * block;
        want = n - off;
        if (want > block) {
            want = block;
        }
        CHECK(zpse_seek_frame(par, plen, last, blk, sizeof blk, &sgot, da) ==
                  PSE_OK && sgot == want && memcmp(blk, in + off, want) == 0,
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

static int stream_emit(void *ctx, uint32_t seq, const uint8_t *blk, size_t len)
{
    StrmSink *s = (StrmSink *)ctx;

    if (seq != s->next || s->len + len > sizeof strm) {
        s->bad = 1;
        return PSE_BAD_ARG;
    }
    memcpy(strm + s->len, blk, len);
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
    CHECK(rc == PSE_OK, "stream: serial reference (n %u, block %u)", n, block);
    if (rc != PSE_OK) {
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
        CHECK(rc == PSE_OK,
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

int main(void)
{
    static const unsigned counts[] = { 2u, 3u, 4u, MAXTHREADS };

    /* One block: the worker path has nothing to spread, so it has to fall
     * back without changing a byte. */
    run_case(65536u, 65536u, MAXTHREADS, 0u, 0, 0);
    run_case(65536u, 65536u, MAXTHREADS, (uint32_t)ZPSE_ENC_SPEED_MAX, 2, 1);

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
        CHECK(rc == PSE_OK, "fewer arenas than threads is accepted");
        digest_bytes(par, plen);
    }

    /* The streamed path, whole-slab calls: both ring depths, every worker
     * count, on a frame that reaches each of the forms the CLI can write. */
    run_stream_case(400u * 1024u, 65536u, MAXTHREADS, 0u, 2, 1, 1u);
    run_stream_case(400u * 1024u, 65536u, MAXTHREADS, 0u, 2, 1,
                    STRM_DEPTH_MAX);
    run_stream_case(400u * 1024u + 77u, 16384u, MAXTHREADS,
                    (uint32_t)ZPSE_ENC_SPEED_MAX, 1, 0, STRM_DEPTH_MAX);
    run_stream_case(192u * 1024u, 4096u, 4u,
                    (uint32_t)ZPSE_ENC_LZ_FAST, 2, 0, STRM_DEPTH_MAX);
    run_stream_case(96u * 1024u, 4096u, MAXTHREADS,
                    (uint32_t)(ZPSE_ENC_LZ_FAST | ZPSE_ENC_CHECKSUM), 3, 1,
                    3u);

    printf("PARALLEL: %s (%d checks, digest %08x)\n",
           fails == 0 ? "ALL OK" : "FAILURES", checks, (unsigned)digest);
    return fails == 0 ? 0 : 1;
}
