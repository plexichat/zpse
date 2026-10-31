#include "zpse/lz.h"

#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/status.h"
#include "zpse/vector.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* SSE2 is the baseline of every x86-64 build and of MSVC on x86; the guard
 * is the same one src/dense.c and src/transform.c use, so the kernel below
 * compiles everywhere and costs nothing where it cannot. */
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define ZPSE_LZ_HAVE_SSE2 1
#include <emmintrin.h>
#endif

/* An empty hash slot. Positions are 16-bit because a block is at most
 * 64 KiB, and the encoder never inserts a position past n - 4, so 0xFFFF
 * is not a legitimate entry. */
#define LZ_EMPTY 0xFFFFu

static uint32_t lz_hash(uint32_t v, uint32_t hash_bits)
{
    /* One multiply of the first four bytes at the cursor, top bits kept:
     * the multiply's high half is where the input bits are spread out, so
     * the index depends on all four bytes and the table needs no tag --
     * the four-byte compare at the candidate is the tag check. */
    return (v * UINT32_C(2654435761)) >> (32u - hash_bits);
}


static uint64_t lz_load64(const uint8_t *p)
{
    uint64_t v;

    memcpy(&v, p, 8u);
    return v;
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_LZ_AVX2 __attribute__((target("avx2")))
#define ZPSE_LZ_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_LZ_AVX2
#define ZPSE_LZ_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_LZ_HAVE_AVX2 0
#endif
#else
#define ZPSE_LZ_HAVE_AVX2 0
#endif

#if ZPSE_LZ_HAVE_AVX2
#include <immintrin.h>

static ZPSE_LZ_AVX2 uint32_t
lz_extend_avx2(const uint8_t *in, uint32_t pos, uint32_t cand, uint32_t len,
               uint32_t n)
{
    while (pos + len + 32u <= n && cand + len + 32u <= n) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(in + pos + len));
        __m256i b = _mm256_loadu_si256((const __m256i *)(in + cand + len));

        if ((uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)) !=
            0xFFFFFFFFu) {
            uint32_t k = 0u;

            while (in[pos + len + k] == in[cand + len + k]) {
                k++;
            }
            return len + k;
        }
        len += 32u;
    }
    return len;
}
#endif /* ZPSE_LZ_HAVE_AVX2 */

/* Output under construction. Literals grow forward from dst + 4; commands
 * grow backward from the end of the buffer, and are compacted behind the
 * literals once the block is done. The two cannot collide: every append
 * checks that the gap between them can still hold the other region. */
typedef struct LzOut {
    uint8_t *dst;
    uint32_t cap;
    uint32_t lit_total; /* literal bytes written */
    uint32_t cmd_hi;    /* first command byte; commands live at [cmd_hi,cap) */
    int fail;
} LzOut;

static void lz_put_literals(LzOut *o, const uint8_t *src, uint32_t len)
{
    if (o->fail || len == 0u) {
        return;
    }
    if (o->cmd_hi < 4u || o->lit_total > o->cmd_hi - 4u ||
        len > o->cmd_hi - 4u - o->lit_total) {
        o->fail = 1;
        return;
    }
    memcpy(o->dst + 4u + o->lit_total, src, len);
    o->lit_total += len;
}

static void lz_put_cmd(LzOut *o, uint32_t off, uint32_t lit, uint32_t mlen)
{
    if (o->fail) {
        return;
    }
    if (o->cmd_hi < 4u) {
        o->fail = 1;
        return;
    }
    o->cmd_hi -= 4u;
    if (o->cmd_hi < 4u + o->lit_total) {
        o->fail = 1;
        return;
    }
    zpse_store_u32le(o->dst + o->cmd_hi,
                     (off & 0xFFFFu) | ((lit & 0xFFu) << 16) |
                         (((mlen - ZPSE_LZ_MIN_MATCH) & 0xFFu) << 24));
}

/* The shape every command takes except a long run or a long match: a
 * literal run that fits the token's 8-bit field, a match that fits its
 * length field. Kept apart from lz_emit so the two appends share one
 * capacity test and neither goes through the out-of-line copy lz_emit
 * compiles to -- the tokens that follow a short match are mostly a few
 * bytes long, and a call per token is most of what that path costs on
 * hex, text and URL blocks, where a match is found every few bytes.
 *
 * A run below eight bytes is copied by one overlapping eight-byte store:
 * the bytes past the run land in the gap between the literal region and
 * the commands, which the end of the block zero-fills, and the read past
 * the run stays inside the block -- a run that reaches the end of the
 * block comes from the trailing-literal path, not from here. */
static void lz_emit_short(LzOut *o, const uint8_t *lit, uint32_t run,
                          uint32_t off, uint32_t len, const uint8_t *in_end)
{
    uint32_t room;
    uint8_t *ldst;

    if (o->fail || o->cmd_hi < 4u || o->lit_total > o->cmd_hi - 4u) {
        o->fail = 1;
        return;
    }
    room = o->cmd_hi - 4u - o->lit_total;
    /* Room for the run, and the command word behind it. */
    if (run > room || room - run < 4u) {
        o->fail = 1;
        return;
    }
    ldst = o->dst + 4u + o->lit_total;
    if (run < 8u && room - run >= 8u && (uint32_t)(in_end - lit) >= 8u) {
        uint64_t v;

        memcpy(&v, lit, 8u);
        memcpy(ldst, &v, 8u);
    } else if (run != 0u) {
        memcpy(ldst, lit, run);
    }
    o->lit_total += run;
    o->cmd_hi -= 4u;
    zpse_store_u32le(o->dst + o->cmd_hi,
                     (off & 0xFFFFu) | ((run & 0xFFu) << 16) |
                         (((len - ZPSE_LZ_MIN_MATCH) & 0xFFu) << 24));
}

/* One match and the literals that precede it. Runs longer than a token
 * can express are split into repeated commands: the literals go in front
 * of the first command and the rest carry zero literals, so the decoder's
 * loop never needs a length escape. */
static void lz_emit(LzOut *o, const uint8_t *lit_src, uint32_t run,
                    uint32_t off, uint32_t len)
{
    while (run > ZPSE_LZ_MAX_LITERALS) {
        lz_put_literals(o, lit_src, ZPSE_LZ_MAX_LITERALS);
        lz_put_cmd(o, 0u, ZPSE_LZ_MAX_LITERALS, 0u);
        lit_src += ZPSE_LZ_MAX_LITERALS;
        run -= ZPSE_LZ_MAX_LITERALS;
    }
    lz_put_literals(o, lit_src, run);
    while (len > ZPSE_LZ_MAX_MATCH) {
        uint32_t chunk = ZPSE_LZ_MAX_MATCH;

        /* Leave at least one minimum-length match for the last command,
         * so every command word it writes is a legal match length. */
        if (len - chunk < ZPSE_LZ_MIN_MATCH) {
            chunk = len - ZPSE_LZ_MIN_MATCH;
        }
        lz_put_cmd(o, off, run, chunk);
        run = 0u;
        len -= chunk;
    }
    lz_put_cmd(o, off, run, len);
}

/* Move the command stream into place behind the literals, and unreverse it
 * in the same pass.
 *
 * Commands were pushed backwards from the end of the buffer, so the region
 * [src, src + len) holds them in reverse order while the output wants them
 * forwards at [off, off + len). That used to be a memmove of the whole
 * command stream followed by an in-place four-byte swap per command: two
 * passes over up to 64 KiB, the second at 2 loads, 2 stores and four
 * address computations per command word.
 *
 * The destination is always below the source (the caller checks
 * off <= src), so when the two regions do not overlap the move and the
 * reversal are one pass: read the source from its end downwards, write the
 * destination upwards, and each step takes four commands out of one
 * 16-byte load with one dword shuffle -- _mm_shuffle_epi32 with 0x1B
 * reverses the order of the four words and leaves each word's own bytes
 * alone, which is exactly the permutation wanted. Nothing is read twice
 * and nothing is written twice.
 *
 * Overlap can only happen at the very end of a buffer the block nearly
 * fills (lz_emit_short refuses to go there in any compressible case), and
 * there the two passes are kept as they were: memmove is defined for an
 * overlapping move, and the reversal is a swap pair, not a copy. Same
 * bytes either way. */
static void lz_commands_out(uint8_t *dst, uint32_t src, uint32_t off,
                            uint32_t len)
{
    uint32_t words = len / 4u;

    if (off + len <= src) {
        uint32_t i = 0u;

#if ZPSE_LZ_HAVE_SSE2
        for (; i + 4u <= words; i += 4u) {
            __m128i v = _mm_loadu_si128(
                (const __m128i *)(const void *)(dst + src + len - 16u -
                                                (size_t)i * 4u));

            _mm_storeu_si128((__m128i *)(void *)(dst + off + (size_t)i * 4u),
                             _mm_shuffle_epi32(v, 0x1B));
        }
#endif /* ZPSE_LZ_HAVE_SSE2 */
        for (; i < words; i++) {
            zpse_store_u32le(dst + off + (size_t)i * 4u,
                             zpse_load_u32le(dst + src + len - 4u -
                                             (size_t)i * 4u));
        }
        return;
    }
    memmove(dst + off, dst + src, len);
    {
        uint32_t lo = off;
        uint32_t hi = off + len - 4u;

        while (lo < hi) {
            uint32_t a = zpse_load_u32le(dst + lo);
            uint32_t b = zpse_load_u32le(dst + hi);

            zpse_store_u32le(dst + lo, b);
            zpse_store_u32le(dst + hi, a);
            lo += 4u;
            hi -= 4u;
        }
    }
}

ZPSE_HOT uint32_t zpse_lz_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                        uint32_t cap, void *table, uint32_t hash_bits)
{
    uint16_t *tab = (uint16_t *)table;
    LzOut o;
    uint32_t pos, anchor, misses, i, cmd_off, cmd_len;
    uint32_t cur, h;
    uint32_t entries = 1u << hash_bits;

    if (in == 0 || dst == 0 || tab == 0) {
        return 0u;
    }
    /* Below two tokens' worth of input there is nothing to find, and the
     * payload could not be smaller than the block anyway. */
    if (n < ZPSE_LZ_MIN_MATCH + 1u || n > 65536u) {
        return 0u;
    }
    if (cap < ZPSE_LZ_MIN_PAYLOAD) {
        return 0u;
    }
    for (i = 0; i < entries; i++) {
        tab[i] = (uint16_t)LZ_EMPTY;
    }
    o.dst = dst;
    o.cap = cap;
    o.lit_total = 0u;
    o.cmd_hi = cap;
    o.fail = 0;

    pos = 0u;
    anchor = 0u;
    misses = 0u;
    /* The extension width is a property of the machine, not of the
     * position: the capability is read once per call rather than once per
     * match, which is one load and one test less on every match that gets
     * past the hash compare, and the answer cannot change inside the loop
     * (nothing here can change what the CPU is). */
    int have_avx2 = 0;
#if ZPSE_LZ_HAVE_AVX2
    have_avx2 = (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;
#endif
    /* The key of the current position -- the four bytes at the cursor and
     * its hash -- lives in `cur`/`h` across iterations instead of being
     * recomputed at the top of each one. The lazy skip is the path it pays
     * on: that path has just read the word at the position it advances to
     * (it is the probe's own key), so one 4-byte load and one multiply per
     * step of a walk come off it. The other two paths move the cursor to an
     * offset only they know and load there. Same decisions, same streams:
     * only which instructions issue the loads changed. Measured over the
     * columnar corpus, the match finder alone (tests/benchmarks/bench_lz on
     * the shootout's own generators): 1.03x geomean, `zeros_burst` 1.10x and
     * `sparse_mix` 1.09x, where a walk is most of the pass. */
    cur = zpse_load_u32le(in + pos);
    h = lz_hash(cur, hash_bits);

    while (pos + ZPSE_LZ_MIN_MATCH <= n) {
        uint32_t cand = tab[h];
        uint32_t len;

        tab[h] = (uint16_t)pos;
        /* No distance test: every entry the table can hold was written by
         * this call at a position strictly below the cursor, and the
         * cursor never passes n - 4 <= 65532, so no candidate is more than
         * 65532 back -- under the 65535 the field can express -- and
         * LZ_EMPTY (0xFFFF) is a value no position can take, because a
         * block is at most 64 KiB. That is also why the table is cleared
         * per call. Dropping the compare takes two operations out of the
         * loop the match finder spends its time in, and cannot change
         * which candidate is tested. */
        if (cand == LZ_EMPTY ||
            zpse_load_u32le(in + cand) != cur) {
            /* Two misses are a coincidence; a run of them means the data
             * has no match here, so the cursor accelerates. */
            pos += 1u + (misses >> 5);
            misses++;
            if (pos + ZPSE_LZ_MIN_MATCH > n) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lz_hash(cur, hash_bits);
            continue;
        }
        /* Lazy matching: if pos+1 has an 8-byte match, skip this position */
        if (pos + 9u <= n) {
            uint32_t v1 = zpse_load_u32le(in + pos + 1u);
            uint32_t h1 = lz_hash(v1, ZPSE_LZ_HASH_BITS_FAST);
            uint32_t cand1 = tab[h1];

            /* cand1 == pos is not a prior occurrence: when h1 == h the
             * entry the probe reads is the one this iteration wrote a few
             * lines up, so the test below would compare the cursor with
             * itself and pass unconditionally. On a run of one byte every
             * position then "has" an 8-byte match at pos+1, and the block
             * is walked a byte at a time -- 64 KiB of iterations on an
             * all-zero block, measured at 465 MB/s against const5A's
             * 28 GB/s for the same shape -- to reach a position the
             * backward extension below pulls straight back to the anchor.
             * A self-reference is no match at all, so it is refused here. */
            if (cand1 != LZ_EMPTY && cand1 != pos &&
                zpse_load_u32le(in + cand1) == v1 &&
                cand1 + 9u <= n &&
                lz_load64(in + pos + 1u) == lz_load64(in + cand1)) {
                pos++;
                misses = 0;
                cur = v1;
                h = lz_hash(cur, hash_bits);
                continue;
            }
        }

        /* Extend backward into the pending literals: an earlier start is a
         * longer match for the same four bytes. It can never cross the
         * last match's end, so the decoder's view of the stream does not
         * change. */
        while (pos > anchor && cand > 0u && in[pos - 1u] == in[cand - 1u]) {
            pos--;
            cand--;
        }
        len = ZPSE_LZ_MIN_MATCH;
        {
            /* The head of the match, compared eight bytes at a time. A
             * short match is the common one, and this is where its cost
             * is: a mismatch inside a word is located by the lowest set
             * bit of the xor instead of a byte loop, and all of it is in
             * the caller, so a match that ends inside the head pays no
             * call at all. That is most of the matches the LZ path emits
             * on hex, text and URL blocks. Only a match that runs past
             * the head reaches the wide loop, which is what long runs
             * want. `hit` says the length is already final. */
            uint32_t cap = (n - pos - len) < 24u ? (n - pos - len) : 24u;
            uint32_t m = 0u;
            int hit = 0;

            while (m + 8u <= cap) {
                uint64_t x = lz_load64(in + pos + len + m) ^
                             lz_load64(in + cand + len + m);

                if (x != 0u) {
                    m += zpse_ctz64(x) >> 3;
                    hit = 1;
                    break;
                }
                m += 8u;
            }
            len += m;
            if (!hit) {
                if (have_avx2) {
#if ZPSE_LZ_HAVE_AVX2
                    /* `have_avx2` is a constant 0 when the kernel is
                     * compiled out (see ZPSE_AVX2_ABI_OK), so this branch
                     * is unreachable -- but it still has to parse, and the
                     * kernel it names does not exist in that build. */
                    len = lz_extend_avx2(in, pos, cand, len, n);
#endif
                } else {
                    while (pos + len + 8u <= n && cand + len + 8u <= n &&
                           lz_load64(in + pos + len) ==
                               lz_load64(in + cand + len)) {
                        len += 8u;
                    }
                }
                while (pos + len < n && in[pos + len] == in[cand + len]) {
                    len++;
                }
            }
        }
        if (pos - anchor <= ZPSE_LZ_MAX_LITERALS &&
            len <= ZPSE_LZ_MAX_MATCH) {
            lz_emit_short(&o, in + anchor, pos - anchor, pos - cand, len,
                          in + n);
        } else {
            lz_emit(&o, in + anchor, pos - anchor, pos - cand, len);
        }
        if (o.fail) {
            return 0u;
        }
        pos += len;
        anchor = pos;
        misses = 0u;
        if (pos + ZPSE_LZ_MIN_MATCH > n) {
            break;
        }
        cur = zpse_load_u32le(in + pos);
        h = lz_hash(cur, hash_bits);
    }
    /* Trailing literals: no match left to code, so offset 0. What a run
     * of `run` literals does to both regions is known before a byte of it
     * is copied -- literal bytes plus one command per 255 of them -- and a
     * total that cannot fit cap is refused by the verification below
     * whether it is copied first or not. Computing it first is what keeps
     * an aborting call from copying up to 64 KiB of literals and 257
     * command words in 255-byte steps that the verification then throws
     * away: on an incompressible block every remaining byte is a literal,
     * and a capped pass (see zpse_block_try_lz) reaches here with one.
     * The arithmetic is 64-bit so the comparison cannot wrap. */
    if (anchor < n && !o.fail) {
        uint32_t run = n - anchor;
        uint64_t ncmd = ((uint64_t)run + ZPSE_LZ_MAX_LITERALS - 1u) /
                        (uint64_t)ZPSE_LZ_MAX_LITERALS;
        uint64_t lit_end = (uint64_t)o.lit_total + (uint64_t)run;
        uint64_t off_end = 4u + ((lit_end + 3u) & ~(uint64_t)3u);
        uint64_t len_end = (uint64_t)(cap - o.cmd_hi) + 4u * ncmd;

        if (off_end + len_end > (uint64_t)cap) {
            return 0u;
        }
    }
    if (anchor < n) {
        uint32_t run = n - anchor;
        const uint8_t *src = in + anchor;

        while (run > ZPSE_LZ_MAX_LITERALS) {
            lz_put_literals(&o, src, ZPSE_LZ_MAX_LITERALS);
            lz_put_cmd(&o, 0u, ZPSE_LZ_MAX_LITERALS, 0u);
            src += ZPSE_LZ_MAX_LITERALS;
            run -= ZPSE_LZ_MAX_LITERALS;
        }
        lz_put_literals(&o, src, run);
        lz_put_cmd(&o, 0u, run, 0u);
    }
    if (o.fail) {
        return 0u;
    }
    cmd_len = cap - o.cmd_hi;
    cmd_off = 4u + ((o.lit_total + 3u) & ~3u);
    if (cmd_off + cmd_len > cap || cmd_off > o.cmd_hi) {
        return 0u;
    }
    /* The commands were pushed backwards from the end of the buffer, so
     * the block holds them in reverse and behind the literals: put them
     * back in the order the decoder reads them, in one pass that copies as
     * it reverses. (The alternative -- growing them forwards into space
     * the literal stream has not reached yet -- cannot be done without a
     * second buffer, since neither stream's length is known until the
     * block is finished.) */
    lz_commands_out(dst, o.cmd_hi, cmd_off, cmd_len);
    /* Deterministic padding: a byte the decoder never reads is still a
     * byte of the stream, and the encoder promises identical output. */
    for (i = 4u + o.lit_total; i < cmd_off; i++) {
        dst[i] = 0u;
    }
    zpse_store_u32le(dst, o.lit_total);
    return cmd_off + cmd_len;
}

/* Literals are copied a 16-byte step at a time -- the size the compiler
 * inlines -- and the sub-16 remainder with two overlapping 8-byte copies,
 * which is one store pair instead of up to fifteen byte moves. The overlap
 * form writes past the literal run, so it is only taken when both the
 * source and the destination have eight readable/writable bytes of slack;
 * the exact byte loop covers the last few bytes of a block, where they do
 * not.
 *
 * Both slack tests are answers the caller already knows, which is why they
 * arrive as two scalars rather than as two more pointer differences here:
 * `lit_safe16` says the whole literal region has sixteen readable bytes of
 * input after it (computed once per call, before the command loop), and
 * `op_rem` is the output still unwritten, which the loop maintains anyway
 * for its own bounds tests. A 16-byte load that starts inside the literal
 * region and runs past its end still reads only bytes of the same input
 * buffer -- the padding and the first command words -- and only `len` of
 * them are stored, so the bytes written are the same ones the byte loop
 * writes. The original form tested the exact distance to the end of the
 * region, which made the wide path conditional on where in the region the
 * run happened to start and cost two 64-bit subtractions per literal run
 * to decide. */
static void lz_copy_literals(uint8_t *dst, const uint8_t *src, uint32_t len,
                             uint32_t op_rem, int lit_safe16)
{
    /* Two 16-byte copies per iteration rather than one. Both loads start at
     * or before src + len - 16, so they read inside the run exactly as the
     * single-copy form does and this needs no new slack test -- it is the
     * same two loads and two stores with one loop branch instead of two,
     * which is what a literal run of a few hundred bytes is bound by (a run
     * that reaches the LZ path at all is the > 8 case the one-shot command
     * path above does not take). The single 16-byte step stays for the
     * 16..31 remainder. */
    while (len >= 32u) {
        memcpy(dst, src, 16u);
        memcpy(dst + 16u, src + 16u, 16u);
        dst += 32u;
        src += 32u;
        len -= 32u;
    }
    if (len >= 16u) {
        memcpy(dst, src, 16u);
        dst += 16u;
        src += 16u;
        len -= 16u;
    }
    if (len != 0u) {
        if (len <= 8u && lit_safe16 != 0 && op_rem >= 8u) {
            memcpy(dst, src, 8u);
        } else if (len > 8u && lit_safe16 != 0 && op_rem >= 16u) {
            memcpy(dst, src, 16u);
        } else {
            while (len != 0u) {
                *dst++ = *src++;
                len--;
            }
        }
    }
}

/* Copy `len` bytes from `dst - off` to `dst`, which may overlap: an LZ
 * match repeats the bytes it just wrote. Loads and stores are wider than
 * one byte only when the distance is at least as wide, so no load ever
 * reads a byte the store in front of it has not written yet. */
/* A short match at a long distance is ONE store.
 *
 * Three things have to hold at once for a wide store to replace the copy
 * plus its byte tail: the width may be no larger than the distance, so the
 * load cannot read a byte this store has not written; it must be at least
 * the match length, so one store covers the match; and it must fit in the
 * output, so the bytes it writes past the match are ones a later command
 * overwrites instead of ones past the end of the block. The decoder fills
 * the block from zero upward, so every byte it has not reached yet is
 * written by a later command: the slack is always reclaimed. (Same slack
 * argument as lz_copy_literals, which writes past its literal run.)
 *
 * What it replaces is the tail: today a ten-byte match at distance 300 is
 * an eight-byte copy and then two byte moves, each with its own loop
 * branch -- and on the text corpora that shape is the common one (four
 * fifths of the matches are under sixteen bytes at a distance over
 * thirty-two), so the tail is most of the copy. */
ZPSE_HOT static void lz_copy_match(uint8_t *dst, uint32_t off, uint32_t len,
                          uint32_t room)
{
    if (len <= 32u) {
        if (off >= 32u && room >= 32u) {
            memcpy(dst, dst - off, 32u);
            return;
        }
        if (len <= 16u && off >= 16u && room >= 16u) {
            memcpy(dst, dst - off, 16u);
            return;
        }
        if (len <= 8u && off >= 8u && room >= 8u) {
            memcpy(dst, dst - off, 8u);
            return;
        }
    }
    /* A run of one byte is a fill, not a copy: off == 1 means every byte of
     * the match is the byte just written, so the whole match is that byte
     * repeated. It is the worst case for the doubling loop below -- which
     * spends four byte passes (1, 2, 4 and 8 bytes) before any wide store
     * runs, i.e. fifteen byte moves per sixteen bytes of a long run -- and
     * it is the shape every long run in a stream has. A fill writes exactly
     * `len` bytes, so the output bound and the reclaim argument are the ones
     * the caller already checked, and the doubling loop's own result for
     * off 1 is the same bytes. */
#if ZPSE_LZ_HAVE_SSE2
    if (off == 1u && len >= 16u) {
        const __m128i rep = _mm_set1_epi8((char)dst[-1]);

        do {
            _mm_storeu_si128((__m128i *)(void *)dst, rep);
            dst += 16u;
            len -= 16u;
        } while (len >= 16u);
    }
#endif /* ZPSE_LZ_HAVE_SSE2 */
    /* A short offset is a repeating pattern, not a slow copy. Copy it once
     * at its own width, which doubles the distance, and repeat: after one
     * pass the last `off` bytes hold two periods, so a copy at twice the
     * distance is the same bytes. Four passes take the worst case (off 1)
     * from a byte loop per 64 KiB run to a 16-byte loop, which is what
     * long runs of one value cost.
     *
     * Each pass is a byte loop of at most `off` bytes rather than a
     * memcpy of `off`: off is below 16 here, the sizes are 1..15, and a
     * call to memcpy for fifteen bytes costs far more than the fifteen
     * byte moves it replaces -- on prose, where most matches have a small
     * offset and a length of a few bytes, that call was the decode. */
    while (off < 16u && len > off) {
        uint8_t *d = dst;
        const uint8_t *s = dst - (ptrdiff_t)off;
        uint32_t k = off;

        do {
            *d++ = *s++;
        } while (--k != 0u);
        dst += off;
        len -= off;
        off <<= 1;
    }
    if (off >= 32u) {
        /* Two 32-byte copies per iteration. The second load reads at
         * dst + 32 - off, which is at or before dst because off >= 32, so
         * it can only read bytes the first store or an earlier iteration
         * already wrote -- the recurrence the single-step loop has, with one
         * loop branch per 64 bytes instead of one per 32. */
        while (len >= 64u) {
            memcpy(dst, dst - off, 32u);
            memcpy(dst + 32u, dst + 32u - off, 32u);
            dst += 64u;
            len -= 64u;
        }
        if (len >= 32u) {
            memcpy(dst, dst - off, 32u);
            dst += 32u;
            len -= 32u;
        }
    }
    if (off >= 16u) {
        while (len >= 16u) {
            memcpy(dst, dst - off, 16u);
            dst += 16u;
            len -= 16u;
        }
    }
    if (off >= 8u) {
        while (len >= 8u) {
            memcpy(dst, dst - off, 8u);
            dst += 8u;
            len -= 8u;
        }
    }
    while (len != 0u) {
        *dst = *(dst - (ptrdiff_t)off);
        dst++;
        len--;
    }
}

ZPSE_HOT int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n)
{
    uint32_t lit_bytes, cmd_off, cmd_len;
    const uint8_t *lp;
    const uint8_t *cp;
    uint8_t *op;

    if (in == 0 || out == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0u) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (in_len < ZPSE_LZ_MIN_PAYLOAD) {
        return PSE_TRUNCATED;
    }
    lit_bytes = zpse_load_u32le(in);
    if (lit_bytes > in_len - 4u) {
        return PSE_BAD_LZ;
    }
    cmd_off = 4u + ((lit_bytes + 3u) & ~3u);
    if (cmd_off > in_len) {
        return PSE_BAD_LZ;
    }
    cmd_len = in_len - cmd_off;
    if (cmd_len < 4u || (cmd_len & 3u) != 0u) {
        return PSE_BAD_LZ;
    }
    lp = in + 4u;
    cp = in + cmd_off;
    op = out;

    {
        /* Everything a command is checked against is a counter rather than
         * a pointer derived from a buffer end: `lit_rem` is the literal
         * stream still unread, `op_rem` the output still unwritten,
         * `op_done` the output already written, and `cmd_left` the command
         * words still unread -- one command per word, so it is the bound
         * `cp + 4 > cend` made. Four counters and the two cursors are the
         * whole live set; keeping the input end, the literal end and the
         * output end as pointers instead ran the loop three values short of
         * the register file, and the spill cost was two reloads of derived
         * command fields and one of the input end on every command (objdump:
         * the fast path opened by popping it back off the stack). The two
         * copies now take the room in front of their region, which is a
         * counter already in hand, in place of a pointer to its end. Same
         * tests, same order, same streams accepted and refused -- only the
         * arithmetic changed. Measured over the corpus, twelve interleaved
         * rounds against the tree before it: LZ decode alone comes out 24%
         * faster on `json_rows`, 13% on `log_lines` and 8% on `var_string`,
         * and the whole decoder 2.6% time-weighted across the 25 cases (1.5%
         * geomean). The encoder is untouched and reads as it did. */
        uint32_t lit_rem = lit_bytes;
        uint32_t op_rem = n;
        uint32_t op_done = 0u;
        uint32_t cmd_left = cmd_len / 4u;

        /* True when a 16-byte read that starts anywhere in the literal
         * region stays inside the input buffer: the region's end plus
         * sixteen cannot pass the end of the frame's block. It is false
         * only for a block whose literal region sits close enough to the
         * end of the payload that there is no padding behind it. */
        const int lit_safe16 =
            ((uint32_t)(in_len - 4u - lit_bytes) >= 16u) ? 1 : 0;
        /* One of the two one-shot bounds is about the block, not the
         * command: whether an eight-byte literal load stays inside the
         * input at all. Folding it into the output-room bound turns two
         * tests into one on every command of the fast path -- a block
         * without the padding asks for more output room than the block can
         * have, so its commands fall through to the general path, which
         * makes its own bounds and refuses the same streams. */
        const uint32_t fast_min_rem =
            (lit_safe16 != 0) ? 40u : 0xFFFFFFFFu;

        while (op_rem != 0u) {
            uint32_t cw;
            uint32_t lit;
            uint32_t off;
            uint32_t mlen;

            if (cmd_left == 0u) {
                return PSE_BAD_LZ;
            }
            cmd_left--;
            cw = zpse_load_u32le(cp);
            cp += 4;
            lit = (cw >> 16) & 0xFFu;
            off = cw & 0xFFFFu;
            mlen = (cw >> 24) + ZPSE_LZ_MIN_MATCH;
            /* One-shot path for the shape most commands of a text, JSON or
             * log block take: at most eight literal bytes and a match that
             * fits in sixteen, at a distance of at least sixteen. Both
             * copies become one store with no tail, and the whole command
             * becomes one branch instead of the two copies' own cascades of
             * width tests -- which is what the loop is worth, because it is
             * per-command bound, not per-byte bound: a command emits five to
             * eleven bytes and costs the same short chain of compares and
             * stores whatever its shape. Measured on the LZ decoder alone,
             * `json_rows` 2.37 -> 3.74 GB/s and `text_en` 1.54 -> 2.29.
             *
             * The bounds are the general path's, tested as one expression
             * and in the order those tests would run in: the literal count
             * is tested here rather than in front of the condition, `off <=
             * op_done + lit` is the distance test taken after the literal
             * bytes are placed, and `lit + mlen + 16 <= op_rem` covers both
             * the output bound and the sixteen bytes the match store writes
             * past its run, which a later command reclaims exactly as the
             * general path's slack does. Both stores over-copy and both
             * over-copies are overwritten -- the literal's tail by the match
             * store, the match's tail past the command -- so the bytes in
             * `out` are the ones the general path writes, and a stream the
             * general path would refuse is refused here too (every test it
             * makes is in this condition, or already made above). */
            /* The output test is `op_rem >= fast_min_rem`, not `lit + mlen
             * + 16 <= op_rem`: with lit <= 8 and mlen <= 16 the two are the
             * same bound, and a compare is one op where the sum is three.
             * `fast_min_rem` carries the block-wide half of the old
             * condition (`lit_safe16`) as well, so the padding question is
             * folded into that one compare rather than asked again. The
             * distance test is `off <= op_done`, one literal's worth
             * stricter than the general path's after the literals are
             * placed -- a match that starts within the first eight bytes of
             * the block takes the general path instead.
             *
             * `lit <= lit_rem` is part of this condition rather than a test
             * of its own before it: the general path below makes that test
             * and has to for every command that falls through, so a command
             * that fails it here is rejected there with the same status.
             * Folding it in is worth one less branch and one less min() of
             * the two counters per command, and the branch hint is worth the
             * layout: with seven terms gcc stops believing the block is
             * taken and moves it out of line, which costs more than the
             * tests it saves. */
            if (ZPSE_LIKELY(lit <= 8u && lit <= lit_rem && off >= 16u &&
                            mlen <= 16u && off <= op_done &&
                            op_rem >= fast_min_rem)) {
                memcpy(op, lp, 8u);
                memcpy(op + lit, op + lit - (ptrdiff_t)off, 16u);
                op += lit + mlen;
                lp += lit;
                lit_rem -= lit;
                op_rem -= lit + mlen;
                op_done += lit + mlen;
                continue;
            }
            /* The same one-shot shape, one width up: a match of 17..32
             * bytes at a distance of at least thirty-two is still one
             * overlapping store, and a command whose literal run is at most
             * eight bytes still fits it in the same forty bytes of output
             * the sixteen-byte form proves it has (8 + 32). It is a second
             * branch rather than a width chosen inside the first one
             * because the first one's shape -- eight literals and a match
             * that fits in sixteen -- is most commands of a JSON block, and
             * a select in front of it cost that path two compares it never
             * needed (json_rows 4570 -> 4340 MB/s). Here nothing is paid
             * unless the sixteen-byte form has already declined, which is
             * one command in twenty-five on json_rows and one in three on
             * log_lines -- where 2.33M of 4.98M commands are matches of
             * 17..32 bytes at a distance over 32 and used to fall through to
             * lz_copy_match's width ladder and doubling loop for them.
             *
             * The bytes are the ladder's own: at distance d >= w the w bytes
             * before a match position are exactly the bytes it repeats, and
             * a store past the match is reclaimed by the next command
             * (lz_copy_match's slack argument). The conditions are the first
             * branch's with the wider pair, tested in the same order, and a
             * command that fails them still reaches the general path below,
             * which accepts and rejects exactly what it did. */
            if (lit <= 8u && lit <= lit_rem && off >= 32u &&
                mlen <= 32u && off <= op_done && op_rem >= fast_min_rem) {
                memcpy(op, lp, 8u);
                memcpy(op + lit, op + lit - (ptrdiff_t)off, 32u);
                op += lit + mlen;
                lp += lit;
                lit_rem -= lit;
                op_rem -= lit + mlen;
                op_done += lit + mlen;
                continue;
            }
            if (lit > lit_rem || lit > op_rem) {
                return PSE_BAD_LZ;
            }
            lz_copy_literals(op, lp, lit, op_rem, lit_safe16);
            op += lit;
            lp += lit;
            lit_rem -= lit;
            op_rem -= lit;
            op_done += lit;
            if (off == 0u) {
                /* A literal-only command has to make progress. */
                if (lit == 0u) {
                    return PSE_BAD_LZ;
                }
                continue;
            }
            if (off > op_done || mlen > op_rem) {
                return PSE_BAD_LZ;
            }
            lz_copy_match(op, off, mlen, op_rem);
            op += mlen;
            op_rem -= mlen;
            op_done += mlen;
        }
        /* Every literal and every command has to be accounted for: a stream
         * that fills the output early and then stops is not the stream the
         * encoder wrote, and silently ignoring the rest would let a corrupt
         * count pass as valid. */
        if (cmd_left != 0u || lit_rem != 0u) {
            return PSE_BAD_LZ;
        }
    }
    return PSE_OK;
}
