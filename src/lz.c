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

/* The payload header: the literal count and the token count, then the
 * stream. Both counts are needed -- the stream is variable length, so its
 * length does not divide into commands. */
#define LZ_HDR 8u

/* Output under construction: one forward-growing stream, so a command is
 * written once, where it stays. A token, its literals, its distance and,
 * when the token asked for one, its length escape -- the same order the
 * decoder reads them in. Nothing is compacted or reversed at the end. */
typedef struct LzOut {
    uint8_t *dst;
    uint32_t cap;
    uint32_t pos;       /* next free byte of the stream */
    uint32_t lit_total; /* literal bytes written so far */
    uint32_t cmds;      /* tokens written */
    uint32_t rep[4];    /* recent distances, most recent first */
    /* The last position whose sixteen following bytes are all inside the
     * input. A literal run that starts at or before it can be copied with
     * one 16-byte load whose tail the copy never stores, which is what a
     * short run wants; past it the copy has to be exact. */
    const uint8_t *lit_wide;
    int fail;
} LzOut;

/* Extra bytes an escape adds after the token byte, for a field whose value
 * is `v` and whose nibble already carries `base` (fifteen). */
static uint32_t lz_esc_len(uint32_t v, uint32_t base)
{
    if (v < base) {
        return 0u;
    }
    v -= base;
    return 1u + v / 255u;
}

static uint32_t lz_varint_len(uint32_t v)
{
    uint32_t n = 1u;

    while (v >= 0x80u) {
        v >>= 7;
        n++;
    }
    return n;
}

static uint8_t *lz_put_esc(uint8_t *p, uint32_t v, uint32_t base)
{
    v -= base;
    while (v >= 255u) {
        *p++ = 255u;
        v -= 255u;
    }
    *p++ = (uint8_t)v;
    return p;
}

static uint8_t *lz_put_varint(uint8_t *p, uint32_t v)
{
    while (v >= 0x80u) {
        *p++ = (uint8_t)(v | 0x80u);
        v >>= 7;
    }
    *p++ = (uint8_t)v;
    return p;
}

/* One command: the literal run before the match (possibly none), the
 * distance and the match length. `off == 0` writes a literal-only command,
 * which is how a trailing run and each 255-byte literal chunk are coded.
 *
 * The distance goes through the four-entry repeat cache, and the decoder
 * updates it identically (see lz.h), so the two sides agree on which value
 * means which distance without a byte being spent on it. */
/* The largest number of bytes one command can write: the token, a
 * one-byte escape for the literal count, the run itself (at most
 * ZPSE_LZ_MAX_LITERALS), a three-byte distance and a one-byte escape for
 * the match length. With `lit` bytes of run the command needs at most
 * LZ_CMD_ROOM + lit, and the room test below is what lets the writer
 * below skip working out the exact size first. */
#define LZ_CMD_ROOM 8u

/* The four most recent distances, most recent first. This used to be a
 * `for` loop with a `break`, which is what the profiler found: on prose a
 * command repeats a distance often enough that the branch is a coin toss
 * and it mispredicts, and it is paid once per command -- about 4% of the
 * whole encode. The comparisons are branchless selects, so which index
 * wins is still the most recent one, byte for byte. */
static uint32_t lz_rep_index(const uint32_t *rep, uint32_t off)
{
    uint32_t m0 = (rep[0] == off);
    uint32_t m1 = (rep[1] == off);
    uint32_t m2 = (rep[2] == off);
    uint32_t m3 = (rep[3] == off);

    return m0 ? 0u : (m1 ? 1u : (m2 ? 2u : (m3 ? 3u : 4u)));
}

/* `lit` bytes from `lit_src` to `dst`. A run of two commands' worth takes
 * one 16-byte load and two 8-byte stores, which the caller has already
 * proved has the input and the output room it needs; the sub-9 tail is the
 * byte loop every run used to pay. */
static inline void lz_put_literals(uint8_t *dst, const uint8_t *lit_src,
                                   uint32_t lit, int wide)
{
    if (wide && lit <= 16u) {
        if (lit <= 8u) {
            memcpy(dst, lit_src, 8u);
        } else {
            memcpy(dst, lit_src, 8u);
            memcpy(dst + lit - 8u, lit_src + lit - 8u, 8u);
        }
        return;
    }
    memcpy(dst, lit_src, lit);
}

static inline void lz_emit_cmd(LzOut *o, const uint8_t *lit_src, uint32_t lit,
                               uint32_t off, uint32_t mlen)
{
    uint32_t offv;
    uint32_t ml;
    uint8_t *t;

    if (o->fail) {
        return;
    }
    if (off == 0u) {
        offv = 0u;
        ml = 0u; /* ignored by the decoder; keep the token legal */
    } else {
        uint32_t idx = lz_rep_index(o->rep, off);

        ml = mlen - ZPSE_LZ_MIN_MATCH;
        if (idx < 4u) {
            offv = idx + 1u;
            while (idx > 0u) {
                o->rep[idx] = o->rep[idx - 1u];
                idx--;
            }
        } else {
            offv = off + 4u;
            o->rep[3] = o->rep[2];
            o->rep[2] = o->rep[1];
            o->rep[1] = o->rep[0];
        }
        o->rep[0] = off;
    }
    /* No command this writer can be asked for is longer than LZ_CMD_ROOM
     * plus its literal run, so room for that much is room for the exact
     * size -- and the exact size, once computed, was only ever used for
     * this test. Working it out (a divide and a shifting loop) was ~9% of
     * the match finder on prose. The test stays for the boundary case, so
     * which commands fit is unchanged. */
    if (LZ_CMD_ROOM + lit > o->cap - o->pos) {
        uint32_t sz = 1u + lz_esc_len(lit, 15u) + lit +
                      lz_varint_len(offv) + lz_esc_len(ml, 15u);

        if (sz > o->cap - o->pos) {
            o->fail = 1;
            return;
        }
    }
    t = o->dst + o->pos;
    *t++ = (uint8_t)(((lit < 15u ? lit : 15u) << 4) |
                     (ml < 15u ? ml : 15u));
    if (lit >= 15u) {
        t = lz_put_esc(t, lit, 15u);
    }
    if (lit != 0u) {
        lz_put_literals(t, lit_src, lit,
                        lit_src <= o->lit_wide &&
                                o->cap - o->pos >= LZ_CMD_ROOM + lit + 16u);
        t += lit;
        o->lit_total += lit;
    }
    t = lz_put_varint(t, offv);
    if (ml >= 15u) {
        t = lz_put_esc(t, ml, 15u);
    }
    o->pos = (uint32_t)(t - o->dst);
    o->cmds++;
}

/* One match and the literals that precede it. Runs longer than a token
 * can express are split into repeated commands with the same distance:
 * the literals go in front of the first command and the rest carry none,
 * so the decoder's loop never needs an unbounded escape for a long run --
 * and after the first chunk the repeat code makes every command shorter. */
static inline void lz_emit(LzOut *o, const uint8_t *lit_src, uint32_t run,
                           uint32_t off, uint32_t len)
{
    while (run > ZPSE_LZ_MAX_LITERALS) {
        lz_emit_cmd(o, lit_src, ZPSE_LZ_MAX_LITERALS, 0u, 0u);
        lit_src += ZPSE_LZ_MAX_LITERALS;
        run -= ZPSE_LZ_MAX_LITERALS;
    }
    while (len > ZPSE_LZ_MAX_MATCH) {
        uint32_t chunk = ZPSE_LZ_MAX_MATCH;

        /* Leave at least one minimum-length match for the last command,
         * so every command it writes is a legal match length. */
        if (len - chunk < ZPSE_LZ_MIN_MATCH) {
            chunk = len - ZPSE_LZ_MIN_MATCH;
        }
        lz_emit_cmd(o, lit_src, run, off, chunk);
        lit_src += run;
        run = 0u;
        len -= chunk;
    }
    lz_emit_cmd(o, lit_src, run, off, len);
}

ZPSE_HOT uint32_t zpse_lz_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                        uint32_t cap, void *table, uint32_t hash_bits)
{
    uint16_t *tab = (uint16_t *)table;
    LzOut o;
    uint32_t pos, anchor, misses, i;
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
    /* Four entries per store. The loop this replaces was one 16-bit store per
     * iteration, and at -O2 gcc does not vectorise it: 8192 iterations of
     * store, increment, compare and branch, per block, on a table whose only
     * purpose at that moment is to be overwritten. The table is 16 KiB and
     * the pass around it reads it constantly, so the clear is paid every
     * block of every file. */
    {
        uint64_t fill = UINT64_C(0xFFFFFFFFFFFFFFFF);

        for (i = 0; i + 4u <= entries; i += 4u) {
            memcpy(tab + i, &fill, 8u);
        }
        for (; i < entries; i++) {
            tab[i] = (uint16_t)LZ_EMPTY;
        }
    }
    o.dst = dst;
    o.cap = cap;
    o.pos = (uint32_t)LZ_HDR;
    o.lit_total = 0u;
    o.cmds = 0u;
    o.rep[0] = 1u;
    o.rep[1] = 1u;
    o.rep[2] = 1u;
    o.rep[3] = 1u;
    o.lit_wide = (n >= 16u) ? in + n - 16u : in;
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
            /* One compare, not two: the eight-byte word at the candidate
             * carries the four bytes `v1` holds, so testing it makes the
             * four-byte test redundant -- and on prose this probe fires
             * almost never (it needs an eight-byte match one byte ahead;
             * the average match is 5.4 bytes), so every instruction here
             * is paid per hit for a decision that is usually "no". The
             * order is the cheap tests first, then the load. */
            if (cand1 != LZ_EMPTY && cand1 != pos && cand1 + 9u <= n &&
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
        lz_emit(&o, in + anchor, pos - anchor, pos - cand, len);
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
     * commands in 255-byte steps that the verification then throws
     * away: on an incompressible block every remaining byte is a literal,
     * and a capped pass (see zpse_block_try_lz) reaches here with one.
     * The arithmetic is 64-bit so the comparison cannot wrap. */
    if (anchor < n && !o.fail) {
        uint32_t run = n - anchor;
        uint64_t ncmd = ((uint64_t)run + ZPSE_LZ_MAX_LITERALS - 1u) /
                        (uint64_t)ZPSE_LZ_MAX_LITERALS;
        uint64_t need = 3u * ncmd + (uint64_t)run;

        if ((uint64_t)o.pos + need > (uint64_t)cap) {
            return 0u;
        }
    }
    if (anchor < n) {
        uint32_t run = n - anchor;
        const uint8_t *src = in + anchor;

        while (run > ZPSE_LZ_MAX_LITERALS) {
            lz_emit_cmd(&o, src, ZPSE_LZ_MAX_LITERALS, 0u, 0u);
            src += ZPSE_LZ_MAX_LITERALS;
            run -= ZPSE_LZ_MAX_LITERALS;
        }
        lz_emit_cmd(&o, src, run, 0u, 0u);
    }
    if (o.fail) {
        return 0u;
    }
    zpse_store_u32le(dst, o.lit_total);
    zpse_store_u32le(dst + 4u, o.cmds);
    return o.pos;
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
 * run and goes past its end still reads only bytes of the same input
 * buffer -- whatever the stream holds next -- and only `len` of them are
 * stored, so the bytes written are the same ones the byte loop writes. The original form tested the exact distance to the end of the
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
    uint32_t room = op_rem;

    while (len >= 32u) {
        memcpy(dst, src, 16u);
        memcpy(dst + 16u, src + 16u, 16u);
        dst += 32u;
        src += 32u;
        len -= 32u;
        room -= 32u;
    }
    if (len >= 16u) {
        memcpy(dst, src, 16u);
        dst += 16u;
        src += 16u;
        len -= 16u;
        room -= 16u;
    }
    if (len != 0u) {
        /* `op_rem` is the room in front of the run, not in front of `dst`:
         * the steps above have already spent part of it, so the tail tests
         * `room`. Testing `op_rem` let the wide store run up to fifteen
         * bytes past the end of a block whose literal run ended with a
         * 9..15-byte remainder -- harmless in a frame, where the next block
         * follows, and a buffer overflow in `zpse_seek_frame`, where it
         * does not. */
        if (len <= 8u && lit_safe16 != 0 && room >= 8u) {
            memcpy(dst, src, 8u);
        } else if (len > 8u && lit_safe16 != 0 && room >= 16u) {
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

/* One forward pass, no seeking: a token, its literals, its distance and,
 * when the token asked for one, its length escape. */
ZPSE_HOT int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n)
{
    uint32_t lit_total, cmd_count;
    const uint8_t *lp;
    const uint8_t *tend;
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
    lit_total = zpse_load_u32le(in);
    cmd_count = zpse_load_u32le(in + 4u);
    if (lit_total > in_len - LZ_HDR || cmd_count == 0u) {
        return PSE_BAD_LZ;
    }
    lp = in + LZ_HDR;
    tend = in + in_len;
    op = out;
    {
        uint32_t lit_seen = 0u;
        uint32_t op_rem = n;
        uint32_t op_done = 0u;
        uint32_t cmd_left = cmd_count;
        uint32_t rep0 = 1u, rep1 = 1u, rep2 = 1u, rep3 = 1u;

        while (op_rem != 0u) {
            uint32_t tok;
            uint32_t lit;
            uint32_t off;
            uint32_t mlen;
            uint32_t v;
            uint8_t b;

            if (cmd_left == 0u) {
                return PSE_BAD_LZ;
            }
            cmd_left--;
            if (lp >= tend) {
                return PSE_BAD_LZ;
            }
            /* ---- The escape-free fast region --------------------------
             * A token whose two nibbles are both below fifteen names a
             * literal run of at most fourteen bytes and a match of at most
             * eighteen, so a command of that shape needs no per-field bound
             * test at all -- only the room it is about to use, which one
             * compare against each buffer covers. That shape is four commands
             * in five on prose and most of them on the binaries, and it is
             * what the fast form (src/lz2.c) has always had and this loop
             * never did: measured over the twelve Silesia files, this form's
             * decode ran at 1.50 GB/s against the fast one's 2.22.
             *
             * The eighteen bytes of payload and thirty-two of output are
             * what makes every read and write below provably in bounds:
             * the token, fourteen literals, the two distance bytes and one
             * to spare; fourteen literals plus an eighteen-byte match and
             * nothing else. A command that does not fit either budget, or
             * a distance that needs a third group, falls through to the
             * careful path with `lp` untouched -- the token is read again
             * there, so nothing is consumed twice and no stream this
             * function refused before is accepted now. */
            if ((size_t)(tend - lp) >= 18u && op_rem >= 32u) {
                tok = *lp;
                if (((tok >> 4) != 15u) && ((tok & 15u) != 15u)) {
                    /* The distance is read BEFORE the literals are copied, so
                     * a third group -- which is the one shape this region
                     * cannot take -- leaves `lp` and the output exactly as
                     * the careful path expects to find them. `lit` is at most
                     * fourteen, so `dp[1]` is inside the eighteen bytes the
                     * region proved it has. */
                    const uint8_t *dp;
                    uint32_t b0;

                    lit = tok >> 4;
                    dp = lp + 1u + lit;
                    b0 = (uint32_t)dp[0];
                    /* The width of the distance is data, and the two forms
                     * are nearly a coin toss on the binaries, so decoding
                     * both and selecting measured WORSE than testing: on the
                     * twelve-file corpus at the default level this form went
                     * from 5.25 to 5.13 GB/s on dickens, 2.35 to 2.16 on
                     * webster and 1.79 to 1.78 on nci, i.e. the mispredicted
                     * branch is cheaper than the extra ALU work plus the
                     * third-group test it would still need. */
                    if (b0 < 0x80u) {
                        v = b0;
                        dp += 1;
                    } else if (dp[1] < 0x80u) {
                        v = (b0 & 0x7Fu) | ((uint32_t)dp[1] << 7u);
                        dp += 2;
                    } else {
                        goto careful;
                    }
                    /* The run as one 8-byte store, or two overlapping ones
                     * when it is nine to fourteen bytes: a memcpy of `lit`
                     * is a size dispatch per command, and the bytes past the
                     * run are the distance and the next token, both written
                     * before anything reads them. A single sixteen-byte store
                     * here was measured at 1.86 GB/s against this shape's
                     * 1.87 on text_en, so the roomier store buys nothing on
                     * this decoder and the narrower pair stays. */
                    if (lit <= 8u) {
                        memcpy(op, lp + 1u, 8u);
                    } else {
                        memcpy(op, lp + 1u, 8u);
                        memcpy(op + lit - 8u, lp + 1u + lit - 8u, 8u);
                    }
                    op += lit;
                    op_rem -= lit;
                    op_done += lit;
                    lit_seen += lit;
                    lp = dp;
                    if (v == 0u) {
                        if (lit == 0u) {
                            return PSE_BAD_LZ;
                        }
                        continue;
                    }
                    /* The distance cache, without a branch on the kind of
                     * distance. Whether this command names one of the four
                     * recent distances or an explicit one is data, and on the
                     * binaries -- where this form is what the gate writes --
                     * it is close to a coin toss, so the test mispredicted per
                     * command. Both candidates are computed and selected. */
                    {
                        uint32_t isrep = (v <= 4u) ? 1u : 0u;
                        uint32_t i = v - 1u;
                        uint32_t roff = (i == 0u)
                                            ? rep0
                                            : ((i == 1u) ? rep1
                                                         : ((i == 2u) ? rep2 : rep3));
                        uint32_t n1, n2, n3;

                        off = isrep != 0u ? roff : (v - 4u);
                        n1 = isrep != 0u ? ((i >= 1u) ? rep0 : rep1) : rep0;
                        n2 = isrep != 0u ? ((i >= 2u) ? rep1 : rep2) : rep1;
                        n3 = isrep != 0u ? ((i == 3u) ? rep2 : rep3) : rep2;
                        rep0 = off;
                        rep1 = n1;
                        rep2 = n2;
                        rep3 = n3;
                    }
                    mlen = (uint32_t)(tok & 15u) + ZPSE_LZ_MIN_MATCH;
                    if (off > op_done) {
                        return PSE_BAD_LZ;
                    }
                    /* A short match at a distance at least as wide as the
                     * store: one store covers the match, and the bytes it
                     * writes past it are output a later command overwrites.
                     * Eighteen is the longest match this region can see, so
                     * the two wide shapes are the only ones it needs -- past
                     * those the copier's own overlap handling runs. */
                    if (off >= 16u && mlen <= 16u) {
                        memcpy(op, op - (ptrdiff_t)off, 16u);
                    } else if (off >= 32u && mlen <= 32u && op_rem >= 32u) {
                        memcpy(op, op - (ptrdiff_t)off, 32u);
                    } else if (off >= 8u && mlen <= 8u && op_rem >= 8u) {
                        /* The short-distance match. All three of the tests
                         * above are false for it (off is under sixteen, so
                         * even the wide store's own guard could not have
                         * run), and the copier reaches the same eight-byte
                         * store through a call and its branch tree. One load
                         * and one store here instead -- the shape that
                         * dominates a tile body, where the LZ stream's
                         * distances are the short ones of a record layout. */
                        memcpy(op, op - (ptrdiff_t)off, 8u);
                    } else {
                        lz_copy_match(op, off, mlen, op_rem);
                    }
                    op += mlen;
                    op_rem -= mlen;
                    op_done += mlen;
                    continue;
                }
            }
        careful:
            tok = *lp++;
            lit = tok >> 4;
            if (lit == 15u) {
                do {
                    if (lp >= tend) {
                        return PSE_BAD_LZ;
                    }
                    b = *lp++;
                    lit += b;
                } while (b == 255u);
            }
            mlen = (tok & 15u) + ZPSE_LZ_MIN_MATCH;
            if (lit > op_rem || (uint32_t)(tend - lp) < lit) {
                return PSE_BAD_LZ;
            }
            if (lit != 0u) {
                /* `lit_safe16` is "there are sixteen readable bytes past
                 * this run", and the call used to pass 0 -- "the run ends
                 * at the end of the payload" -- which was true of the
                 * pre-inline layout and is false of this one: what follows
                 * a run here is its distance word plus the rest of the
                 * stream. Passing 0 made EVERY run take the byte loop, and
                 * the common run on text is under sixteen bytes, so the
                 * narrow path was the whole literal copy. The bytes a wide
                 * store writes past the run are output the next command
                 * overwrites (the decoder fills the block from zero up),
                 * and `room` inside the copy already proves they are inside
                 * the block. */
                {
                    uint32_t safe =
                        (uint32_t)(tend - lp) >= 16u ? 1u : 0u;

                    /* The run on prose is one or two bytes -- the block's
                     * own average is 0.9 literals per command -- and for
                     * those the copier's prologue is more work than the
                     * bytes. One 8-byte store is the whole run under
                     * exactly the test the copier makes for its own narrow
                     * tail (same three terms, same slack), so the bytes
                     * written are the same ones it writes. */
                    if (lit <= 8u && safe != 0u && op_rem >= 8u) {
                        memcpy(op, lp, 8u);
                    } else {
                        lz_copy_literals(op, lp, lit, op_rem, (int)safe);
                    }
                }
                op += lit;
                lp += lit;
                op_rem -= lit;
                op_done += lit;
                lit_seen += lit;
            }
            /* The distance: 7-bit groups, high bit means another byte
             * follows. Zero is a literal-only command, one to four name a
             * recent distance, and anything above that is the distance plus
             * four. The cache is updated exactly as lz_emit_cmd updates it,
             * which is the whole contract of the encoding. */
            /* One or two bytes on nearly every command: the groups are
             * seven bits, so two of them carry any distance below 16384,
             * and a 64 KiB block's distances sit under that. Reading those
             * two cases straight out of the stream skips the loop's
             * shift-and-test per byte, which measured as a fifth of what a
             * command costs. The set of streams accepted is unchanged:
             * the same bytes are consumed for the same values, a shorter
             * stream still reaches the loop and is still refused, and the
             * `shift > 21` bound still lives in the loop. */
            {
                uint32_t have2 = (uint32_t)(tend - lp) >= 2u;
                uint32_t b0 = (lp < tend) ? (uint32_t)lp[0] : 0xFFFFFFFFu;

                v = 0u;
                if (b0 < 0x80u) {
                    v = b0;
                    lp += 1;
                } else if (have2 && lp[1] < 0x80u) {
                    v = (b0 & 0x7Fu) | ((uint32_t)lp[1] << 7u);
                    lp += 2;
                } else {
                    uint32_t shift = 0u;

                    do {
                        if (lp >= tend || shift > 21u) {
                            return PSE_BAD_LZ;
                        }
                        b = *lp++;
                        v |= (uint32_t)(b & 0x7Fu) << shift;
                        shift += 7u;
                    } while ((b & 0x80u) != 0u);
                }
            }
            if (v == 0u) {
                /* A literal-only command has to make progress. */
                if (lit == 0u) {
                    return PSE_BAD_LZ;
                }
                continue;
            }
            if ((tok & 15u) == 15u) {
                do {
                    if (lp >= tend) {
                        return PSE_BAD_LZ;
                    }
                    b = *lp++;
                    mlen += b;
                } while (b == 255u);
            }
            if (v <= 4u) {
                if (v == 1u) {
                    off = rep0;
                } else if (v == 2u) {
                    off = rep1;
                    rep1 = rep0;
                    rep0 = off;
                } else if (v == 3u) {
                    off = rep2;
                    rep2 = rep1;
                    rep1 = rep0;
                    rep0 = off;
                } else {
                    off = rep3;
                    rep3 = rep2;
                    rep2 = rep1;
                    rep1 = rep0;
                    rep0 = off;
                }
            } else {
                off = v - 4u;
                rep3 = rep2;
                rep2 = rep1;
                rep1 = rep0;
                rep0 = off;
            }
            if (off > op_done || mlen > op_rem) {
                return PSE_BAD_LZ;
            }
            /* A short match at a long distance is one store: at distance d
             * the d bytes before the match are the bytes it repeats, and a
             * store past the match is reclaimed by the next command. */
            if (ZPSE_LIKELY(off >= 16u && mlen <= 16u && op_rem >= 16u)) {
                memcpy(op, op - (ptrdiff_t)off, 16u);
            } else if (off >= 32u && mlen <= 32u && op_rem >= 32u) {
                memcpy(op, op - (ptrdiff_t)off, 32u);
            } else {
                lz_copy_match(op, off, mlen, op_rem);
            }
            op += mlen;
            op_rem -= mlen;
            op_done += mlen;
        }
        /* Every literal and every command has to be accounted for, and the
         * stream has to end where the payload does. */
        if (cmd_left != 0u || lit_seen != lit_total || lp != tend) {
            return PSE_BAD_LZ;
        }
    }
    return PSE_OK;
}
