#include "zpse/lzh.h"

#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/status.h"
#include "zpse/vector.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(ZPSE_LZH_TRACE)
#include <stdio.h>
#endif

/* See include/zpse/lzh.h for the payload layout. The encoder below is the fast
 * form's finder (src/lz2.c) with a different writer: the same hash table, the
 * same backward walk into the pending literals, the same one-candidate lazy
 * probe. Keeping the parse identical is what makes a measurement of this form
 * a measurement of its command encoding rather than of two different parses. */

#define LZH_EMPTY 0xFFFFu
#define LZH_HDR 8u
#define LZH_REP_BASE 0xFCu
#define LZH_TOK_ESC 15u

/* Prefetch the finder's distance table (the hash slots holding candidate
 * positions). A hint only: it compiles to nothing where the toolchain has no
 * builtin, and it cannot change a byte of the stream. */
#ifndef ZPSE_LZH_DIST_PREFETCH
#define ZPSE_LZH_DIST_PREFETCH 1
#endif
#if ZPSE_LZH_DIST_PREFETCH && (defined(__GNUC__) || defined(__clang__))
#define LZH_PREFETCH(p) __builtin_prefetch((p), 0, 1)
#else
#define LZH_PREFETCH(p) ((void)0)
#endif

/* Hash slot access: one uint16_t array of caller-owned scratch, cleared per
 * call. LZH_EMPTY is the marker for a slot that holds no position. */
#define LZH_TAB_GET(h) ((tab)[(h)])
#define LZH_TAB_SET(h, v) do { (tab)[(h)] = (uint16_t)(v); } while (0)

/* The fifteen nibble codes: {literal run, match length}, length 0 meaning a
 * literal-only command. The order is the corpus ranking in
 * include/zpse/lzh.h; the index is what the nibble carries. Fifteen and not
 * sixteen because nibble 15 is the escape (the byte token follows), so the
 * sixteenth-ranked pair -- (3, 5), 1.56% of commands -- is spelled out
 * instead. */
static const ZpseLzhCode zpse_lzh_book[ZPSE_LZH_CODES] = {
    { 1u, 7u }, { 0u, 4u },  { 0u, 8u },  { 2u, 6u }, { 1u, 0u },
    { 0u, 5u }, { 1u, 15u }, { 1u, 5u },  { 0u, 7u }, { 0u, 6u },
    { 2u, 0u }, { 1u, 6u },  { 4u, 0u },  { 0u, 9u }, { 3u, 0u }
};

static uint32_t lzh_hash(uint32_t v, uint32_t hash_bits)
{
    return (v * UINT32_C(2654435761)) >> (32u - hash_bits);
}

static uint64_t lzh_load64(const uint8_t *p)
{
    uint64_t v;

    memcpy(&v, p, 8u);
    return v;
}

/* AVX2 forward extension for the main match, ported from src/lz.c's
 * lz_extend_avx2 (see src/lz2.c). Per-file macro and function names keep the
 * unity build's single translation unit free of collisions. The kernel stops
 * at the first mismatching byte -- the lowest clear bit of the equality mask,
 * the same offset the eight-byte xor loop finds with ctz -- so the length is
 * bit-for-bit the scalar one. */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_LZH_AVX2 __attribute__((target("avx2")))
#define ZPSE_LZH_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_LZH_AVX2
#define ZPSE_LZH_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_LZH_HAVE_AVX2 0
#endif
#else
#define ZPSE_LZH_HAVE_AVX2 0
#endif

#if ZPSE_LZH_HAVE_AVX2
#include <immintrin.h>

static ZPSE_LZH_AVX2 uint32_t
lzh_extend_avx2(const uint8_t *in, uint32_t a, uint32_t b, uint32_t len,
                uint32_t n)
{
    while (a + len + 32u <= n && b + len + 32u <= n) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(in + a + len));
        __m256i y = _mm256_loadu_si256((const __m256i *)(in + b + len));
        uint32_t eq = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y));

        if (eq != 0xFFFFFFFFu) {
            return len + (uint32_t)zpse_ctz64(~(uint64_t)eq);
        }
        len += 32u;
    }
    return len;
}
#endif /* ZPSE_LZH_HAVE_AVX2 */

/* Which of the four recent distances `off` is, or 4 when it is a fresh one. */
static uint32_t lzh_rep_index(const uint32_t *rep, uint32_t off)
{
    uint32_t m0 = (rep[0] == off);
    uint32_t m1 = (rep[1] == off);
    uint32_t m2 = (rep[2] == off);
    uint32_t m3 = (rep[3] == off);

    return m0 ? 0u : (m1 ? 1u : (m2 ? 2u : (m3 ? 3u : 4u)));
}

/* The distance cache's contract with the decoder: a hit is promoted to the
 * front, a fresh distance pushes the front down and drops the oldest. */
static void lzh_rep_move(uint32_t *rep, uint32_t off, uint32_t idx)
{
    if (idx < 4u) {
        while (idx > 0u) {
            rep[idx] = rep[idx - 1u];
            idx--;
        }
    } else {
        rep[3] = rep[2];
        rep[2] = rep[1];
        rep[1] = rep[0];
    }
    rep[0] = off;
}

/* The book above, flattened to one LUT with a sixteen-entry column per run
 * (run 0..4) plus an escape row and column. The writer used to pay a taken
 * branch per command for the range test and a two-dimensional load for the
 * answer; saturating the indices into the all-escape row/column makes it one
 * load and two conditional moves. The table is transposed from
 * zpse_lzh_book, which the decoder reads; a pair the book misses spells
 * itself out, so a disagreement between the two would change bytes but never
 * break a round trip, which is why the escaped and unescaped spellings'
 * packed totals are compared whenever either table is touched. */
#define LZH_CODE_COLS 17u /* sixteen lengths plus the escape column */
static const uint8_t lzh_code_lut[6u * LZH_CODE_COLS] = {
    /* run 0 */ 15u, 15u, 15u, 15u, 1u, 5u, 9u, 8u, 2u, 13u,
                15u, 15u, 15u, 15u, 15u, 15u, 15u,
    /* run 1 */ 4u, 15u, 15u, 15u, 15u, 7u, 11u, 0u, 15u, 15u,
                15u, 15u, 15u, 15u, 15u, 6u, 15u,
    /* run 2 */ 10u, 15u, 15u, 15u, 15u, 15u, 3u, 15u, 15u, 15u,
                15u, 15u, 15u, 15u, 15u, 15u, 15u,
    /* run 3 */ 14u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u,
                15u, 15u, 15u, 15u, 15u, 15u, 15u,
    /* run 4 */ 12u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u,
                15u, 15u, 15u, 15u, 15u, 15u, 15u,
    /* escape row */ 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u,
                     15u, 15u, 15u, 15u, 15u, 15u, 15u
};

static uint32_t lzh_code_of(uint32_t run, uint32_t len)
{
    /* One load, no taken branch: an out-of-book field saturates into a row
     * or column whose every entry is the escape nibble. */
    return (uint32_t)lzh_code_lut[(run < 5u ? run : 5u) * LZH_CODE_COLS +
                                  (len < 16u ? len : 16u)];
}

/* A command: the literal run in front of it (a byte offset into the input,
 * not a copy), the match, and the distance code. The code is decided when the
 * command is created and not when it is written, because the distance cache
 * has to move in command order: commands are written a pair at a time, and a
 * cache read at write time would see the previous command's move while the
 * encoder's own decision saw the state before it. Creation order is write
 * order, so deciding there keeps the two sides identical. */
typedef struct LzhCmd {
    uint32_t lit_off;
    uint32_t run;
    uint32_t off;
    uint32_t len;
    uint32_t ridx;
} LzhCmd;

/* The writer: commands arrive one at a time and are paired on the way out,
 * because the token byte of a pair has to carry both nibbles and it sits in
 * front of both bodies. Only one command is ever held, and it holds fields
 * rather than bytes -- the literals are still in the input. */
typedef struct LzhWr {
    uint8_t *dst;
    uint32_t cap;
    uint32_t pos;
    uint32_t lit_total;
    uint32_t cmds;
    LzhCmd pend;
    int have;
} LzhWr;

/* How many bytes this distance takes, by the code it will be written with.
 * A repeat is one byte, a distance that fits 240 values is one byte, and the
 * two wider forms are two and three. The split matters at a page, where the
 * distances are small: the fast form spends two bytes on every fresh distance
 * because its first byte has to leave room for the repeat codes, and this
 * form does not. */
static uint32_t lzh_dist_len(uint32_t off, uint32_t ridx)
{
    if (ridx < 4u || off <= 224u) {
        return 1u;
    }
    return (off <= 4320u) ? 2u : 3u;
}

/* One command's body: the byte token when the book did not have the pair, its
 * escapes, its literals and its distance. `code` is the nibble this command
 * was given in the pair token -- 15 means the decoder is waiting for the byte
 * token that spells the fields out. */
static int lzh_body(LzhWr *w, const uint8_t *in, const LzhCmd *c,
                    uint32_t code)
{
    uint32_t need = c->run + (c->run >= 15u ? 1u : 0u);
    uint32_t len_code = 0u;
    uint8_t *t;

    if (code == LZH_TOK_ESC) {
        need++;
        len_code = (c->len == 0u)
                       ? 0u
                       : (c->len < 18u ? c->len - 3u : 15u);
    }
    if (c->len != 0u) {
        need += (c->len >= 18u ? 1u : 0u);
        need += lzh_dist_len(c->off, c->ridx);
    }
    if (need > w->cap - w->pos) {
        return 0;
    }
    t = w->dst + w->pos;
    if (code == LZH_TOK_ESC) {
        /* The byte token: the two fields as the fast form writes them, with
         * 15 in either half meaning the escape byte below carries the rest. */
        *t++ = (uint8_t)(((c->run < 15u ? c->run : 15u) << 4) | len_code);
    }
    if (c->run >= 15u) {
        *t++ = (uint8_t)(c->run - 15u);
    }
    if (c->len >= 18u) {
        *t++ = (uint8_t)(c->len - 18u);
    }
    if (c->run != 0u) {
        memcpy(t, in + c->lit_off, c->run);
        t += c->run;
        w->lit_total += c->run;
    }
    if (c->len != 0u) {
        /* The distance: one byte for a repeat or for 1..224, two for 225..4320,
         * three above that. See the layout in include/zpse/lzh.h. */
        if (c->ridx < 4u) {
            *t++ = (uint8_t)(LZH_REP_BASE | c->ridx);
        } else if (c->off <= 224u) {
            *t++ = (uint8_t)(c->off - 1u);
        } else if (c->off <= 4320u) {
            uint32_t v = c->off - 225u;

            *t++ = (uint8_t)(0xE0u | (v >> 8));
            *t++ = (uint8_t)(v & 0xFFu);
        } else {
            uint32_t v = c->off - 4321u;

            *t++ = 0xF0u;
            *t++ = (uint8_t)(v & 0xFFu);
            *t++ = (uint8_t)((v >> 8) & 0xFFu);
        }
    }
    w->pos = (uint32_t)(t - w->dst);
    return 1;
}

/* Emit the pending command and `b` as one pair: the token byte first, then
 * each body in command order. */
static int lzh_pair(LzhWr *w, const uint8_t *in, const LzhCmd *b)
{
    uint32_t ca = lzh_code_of(w->pend.run, w->pend.len);
    uint32_t cb = lzh_code_of(b->run, b->len);

    if (w->pos >= w->cap) {
        return 0;
    }
    w->dst[w->pos++] = (uint8_t)((ca << 4) | cb);
    if (!lzh_body(w, in, &w->pend, ca)) {
        return 0;
    }
    return lzh_body(w, in, b, cb);
}

/* One command through the writer: the distance code is taken (and the cache
 * moved) here, in command order. */
static int lzh_cmd(LzhWr *w, const uint8_t *in, uint32_t *rep,
                   uint32_t lit_off, uint32_t run, uint32_t off,
                   uint32_t len)
{
    LzhCmd c;

    c.lit_off = lit_off;
    c.run = run;
    c.off = off;
    c.len = len;
    c.ridx = 4u;
    if (len != 0u) {
        c.ridx = lzh_rep_index(rep, off);
        lzh_rep_move(rep, off, c.ridx);
    }
    w->cmds++;
    if (!w->have) {
        w->pend = c;
        w->have = 1;
        return 1;
    }
    w->have = 0;
    return lzh_pair(w, in, &c);
}

/* Lazy probe helper, the fast form's: the length of the match at (a, b).
 * `max_len` caps the walk: the caller only asks whether this match is
 * strictly longer than the one in hand, so bytes past the cap cannot change
 * the answer -- a mismatch is reported at its exact offset by ctz, and
 * reaching the cap means every byte before it matched. */
static uint32_t lzh_extend(const uint8_t *in, uint32_t n, uint32_t a,
                           uint32_t b, uint32_t len, uint32_t max_len)
{
    while (len < max_len && a + len + 8u <= n) {
        uint64_t x = lzh_load64(in + a + len) ^ lzh_load64(in + b + len);

        if (x != 0u) {
            return len + (zpse_ctz64(x) >> 3);
        }
        len += 8u;
    }
    while (len < max_len && a + len < n && in[a + len] == in[b + len]) {
        len++;
    }
    return len;
}

/* Chunk a run into commands the coders can carry: a literal-only command holds
 * at most ZPSE_LZH_MAX_LIT literals, and a match is split so that every chunk
 * is at least ZPSE_LZH_MIN_MATCH long (the format codes a length as
 * (length - 3), so a chunk of one to three would come back as four). */
static int lzh_literals(LzhWr *w, const uint8_t *in, uint32_t *rep,
                        uint32_t lit_off, uint32_t run)
{
    while (run > ZPSE_LZH_MAX_LIT) {
        if (!lzh_cmd(w, in, rep, lit_off, ZPSE_LZH_MAX_LIT, 0u, 0u)) {
            return 0;
        }
        lit_off += ZPSE_LZH_MAX_LIT;
        run -= ZPSE_LZH_MAX_LIT;
    }
    return lzh_cmd(w, in, rep, lit_off, run, 0u, 0u);
}

/* Does the backward walk into the pending literals step eight bytes at a
 * time? Same walk, same bound (`anchor`): a word is stepped only when all
 * eight of its bytes match, and the byte loop below then resolves the byte a
 * mismatching word stopped on. See src/lz.c's ZPSE_LZ_BACK_WIDE: measured
 * and refused (the loop is nearly always short, so the extra bounds cost more
 * than the word steps save). 0 is the shipped arm. */
#ifndef ZPSE_LZH_BACK_WIDE
#define ZPSE_LZH_BACK_WIDE 0
#endif

/* Carry the candidate at the cursor across iterations, and read the next
 * position's key, hash and slot right after filing the current one, before the
 * candidate is tested (src/lz2.c's ZPSE_LZ2_PIPE_NEXT). The lookahead reads
 * are taken AFTER the table store, so a slot the store just wrote reads back
 * exactly what the next iteration would have found; the miss path's byte step
 * is otherwise the same, and so is the stream. 0 builds the finder without
 * the lookahead, which is what the measurement compares against. */
#ifndef ZPSE_LZH_PIPE_NEXT
#define ZPSE_LZH_PIPE_NEXT 1
#endif

/* Issue the lazy probe's two loads before the backward walk and the forward
 * extension, so their latency overlaps that work (src/lz2.c's
 * ZPSE_LZ2_PROBE_EARLY). With PIPE_NEXT the values are the lookahead's, already
 * in hand; without it they are read here, still before the walk. The probe
 * itself is unchanged: it runs only when the match starts at the cursor, with
 * the same candidate bound and the same 24-byte ceiling. */
#ifndef ZPSE_LZH_PROBE_EARLY
#define ZPSE_LZH_PROBE_EARLY 1
#endif

/* The one finder body behind every entry point. `sample_n == 0` is a full
 * pass; a nonzero `sample_n` prices a sample and clears only the slots that a
 * position in [0,sample_n) can hash to. Every decision and store is shared. */
static uint32_t lzh_encode_impl(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sample_n)
{
    uint16_t *restrict tab = (uint16_t *)table;
    uint32_t entries;
    uint32_t pos, anchor, misses;
    uint32_t cur, h, limit;
    uint32_t rep[4] = { 1u, 1u, 1u, 1u };
    LzhWr w;
    /* Ask for the extension width once per call, as src/lz.c and src/lz2.c
     * do; a constant 0 where the kernel is compiled out leaves the scalar
     * branch the only one taken. */
    int have_avx2 = 0;
#if ZPSE_LZH_HAVE_AVX2
    have_avx2 = (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;
#endif
    entries = 1u << hash_bits;

    if (in == 0 || dst == 0 || tab == 0) {
        return 0u;
    }
    if (n < ZPSE_LZH_MIN_MATCH + 1u || n > 65536u) {
        return 0u;
    }
    if (cap < ZPSE_LZH_MIN_PAYLOAD) {
        return 0u;
    }
    /* LZH_EMPTY is 0xFFFF and the table is uint16_t, so all-ones bytes are
     * exactly the empty marker. A full pass clears it with one memset; a
     * sample pass clears only the slots a position in [0, sample_n) can hash
     * to (a key hash, so the slots are found by hashing every sample key).
     * See src/lz.c's clear. */
    {
        uint32_t ce = (sample_n != 0u && sample_n < n) ? sample_n : n;

        if (sample_n == 0u) {
            memset(tab, 0xFF, (size_t)entries * sizeof *tab);
        } else if (ce >= ZPSE_LZH_MIN_MATCH) {
            uint32_t p;

            for (p = 0u; p + ZPSE_LZH_MIN_MATCH <= ce; p++) {
                LZH_TAB_SET(lzh_hash(zpse_load_u32le(in + p), hash_bits),
                            LZH_EMPTY);
            }
        }
    }
    w.dst = dst;
    w.cap = cap;
    w.pos = LZH_HDR;
    w.lit_total = 0u;
    w.cmds = 0u;
    w.have = 0;
    /* pend is only read once `have` says a command is waiting, but MSVC's
     * flow analysis cannot see that lzh_put fills it through a pointer. */
    memset(&w.pend, 0, sizeof w.pend);

    pos = 0u;
    anchor = 0u;
    misses = 0u;
    limit = n - ZPSE_LZH_MIN_MATCH;
    cur = zpse_load_u32le(in + pos);
    h = lzh_hash(cur, hash_bits);
#if ZPSE_LZH_PIPE_NEXT
    /* The candidate at the cursor, carried across iterations: the miss path
     * reads it while the word at the candidate is being compared (see the
     * knob above). */
    uint32_t cand = LZH_TAB_GET(h);
#endif

    while (pos <= limit) {
#if ZPSE_LZH_PIPE_NEXT
        uint32_t len = ZPSE_LZH_MIN_MATCH;
        uint32_t mpos = pos;
        uint32_t mcand = cand;
        uint32_t nx_cur = 0u, nx_h = 0u, nx_c = LZH_EMPTY;
        int nx_ok = 0;
#else
        uint32_t cand = LZH_TAB_GET(h);
        uint32_t len = ZPSE_LZH_MIN_MATCH;
        uint32_t mpos = pos;
        uint32_t mcand = cand;
#endif

        LZH_TAB_SET(h, pos);
#if ZPSE_LZH_PIPE_NEXT
        /* The next position's chain, read after the store above, exactly as
         * src/lz2.c does: a slot the store just wrote reads back what the next
         * iteration would have found. */
        if (pos + 1u + ZPSE_LZH_MIN_MATCH <= n) {
            nx_cur = zpse_load_u32le(in + pos + 1u);
            nx_h = lzh_hash(nx_cur, hash_bits);
            /* Issue the distance table's next slot before the probe reads it.
             * The hint is dead code when the toolchain has no prefetch
             * builtin. */
            LZH_PREFETCH((const void *)&tab[nx_h]);
            nx_c = LZH_TAB_GET(nx_h);
            nx_ok = 1;
        }
#endif
        if (ZPSE_LIKELY(cand == LZH_EMPTY ||
                        zpse_load_u32le(in + cand) != cur)) {
#if ZPSE_LZH_PIPE_NEXT
            /* One byte on, the ordinary miss: the key, hash and slot are
             * already in hand from the lookahead above. */
            if (nx_ok != 0 && (misses >> 5u) == 0u) {
                pos += 1u;
                misses++;
                cur = nx_cur;
                h = nx_h;
                cand = nx_c;
                continue;
            }
#endif
            pos += 1u + (misses >> 5u);
            misses++;
            if (pos > limit) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lzh_hash(cur, hash_bits);
#if ZPSE_LZH_PIPE_NEXT
            cand = LZH_TAB_GET(h);
#endif
            continue;
        }
#if ZPSE_LZH_PROBE_EARLY
        /* The lazy probe's two loads (and the slot they name), issued now so
         * their latency overlaps the backward walk and the forward extension
         * below, before the probe that uses them. Read after the table store
         * above, so a self-hit sees the position just filed, as before. */
        {
            uint32_t p_cur = 0u, p_h = 0u, p_c1 = LZH_EMPTY;
            int p_ok = 0;

#if ZPSE_LZH_PIPE_NEXT
            p_cur = nx_cur;
            p_h = nx_h;
            p_c1 = nx_c;
            p_ok = nx_ok;
#else
            if (pos + 1u + ZPSE_LZH_MIN_MATCH <= n) {
                p_cur = zpse_load_u32le(in + pos + 1u);
                p_h = lzh_hash(p_cur, hash_bits);
                p_c1 = LZH_TAB_GET(p_h);
                p_ok = 1;
            }
#endif
#endif /* ZPSE_LZH_PROBE_EARLY */
#if ZPSE_LZH_BACK_WIDE
            while (mpos - anchor >= 8u && mcand >= 8u &&
                   lzh_load64(in + mpos - 8u) == lzh_load64(in + mcand - 8u)) {
                mpos -= 8u;
                mcand -= 8u;
            }
#endif
            while (mpos > anchor && mcand > 0u && in[mpos - 1u] == in[mcand - 1u]) {
                mpos--;
                mcand--;
            }
            {
                int tail = 0;

                if (have_avx2) {
#if ZPSE_LZH_HAVE_AVX2
                    /* 32 bytes at a time where both cursors have the room
                     * (see ZPSE_LZH_AVX2). The kernel stops at the first
                     * mismatch and the byte loop below resolves the byte it
                     * stopped on -- the same length the xor loop finds. */
                    len = lzh_extend_avx2(in, mpos, mcand, len, n);
#endif
                } else {
                    while (mpos + len + 8u <= n) {
                        uint64_t x = lzh_load64(in + mpos + len) ^
                                     lzh_load64(in + mcand + len);

                        if (x != 0u) {
                            len += zpse_ctz64(x) >> 3;
                            tail = 1;
                            break;
                        }
                        len += 8u;
                    }
                }
                if (tail == 0) {
                    while (mpos + len < n && in[mpos + len] == in[mcand + len]) {
                        len++;
                    }
                }
            }
            /* One-candidate lazy probe at the next position, as the fast form
             * runs it: only when the match starts at the cursor, because a
             * walked-back match is already the better one and stepping to
             * mpos + 1 would move the scanning cursor backwards. */
#if ZPSE_LZH_PROBE_EARLY
            if (p_ok && mpos == pos && len <= 24u &&
                p_c1 != LZH_EMPTY && p_c1 <= mpos &&
                zpse_load_u32le(in + p_c1) == p_cur &&
                lzh_extend(in, n, mpos + 1u, p_c1, ZPSE_LZH_MIN_MATCH,
                           len + 1u) > len) {
                pos = mpos + 1u;
                misses = 0u;
                cur = p_cur;
                h = p_h;
#if ZPSE_LZH_PIPE_NEXT
                cand = LZH_TAB_GET(h);
#endif
                continue;
            }
        }
#else
            if (mpos == pos && len <= 24u && mpos + 1u + ZPSE_LZH_MIN_MATCH <= n) {
                uint32_t cur1 = zpse_load_u32le(in + mpos + 1u);
                uint32_t h1 = lzh_hash(cur1, hash_bits);
                uint32_t c1 = LZH_TAB_GET(h1);

                if (c1 != LZH_EMPTY && c1 <= mpos &&
                    zpse_load_u32le(in + c1) == cur1 &&
                    lzh_extend(in, n, mpos + 1u, c1, ZPSE_LZH_MIN_MATCH,
                               len + 1u) > len) {
                    pos = mpos + 1u;
                    misses = 0u;
                    cur = cur1;
                    h = h1;
#if ZPSE_LZH_PIPE_NEXT
                    cand = LZH_TAB_GET(h);
#endif
                    continue;
                }
            }
#endif
            {
                uint32_t run = mpos - anchor;
                uint32_t off = mpos - mcand;
                uint32_t rem = len;
                uint32_t lit_off = anchor;

                /* The literal run in front of the match is chunked here as well as
                 * in the tail: one command carries at most ZPSE_LZH_MAX_LIT of
                 * them, and a longer run is several literal-only commands before
                 * the one that starts the match. */
                while (run > ZPSE_LZH_MAX_LIT) {
                    if (!lzh_cmd(&w, in, rep, lit_off, ZPSE_LZH_MAX_LIT, 0u, 0u)) {
                        return 0u;
                    }
                    lit_off += ZPSE_LZH_MAX_LIT;
                    run -= ZPSE_LZH_MAX_LIT;
                }
                while (rem > ZPSE_LZH_MAX_MATCH) {
                    uint32_t take = ZPSE_LZH_MAX_MATCH;

                    if (rem - take < ZPSE_LZH_MIN_MATCH) {
                        take = rem - ZPSE_LZH_MIN_MATCH;
                    }
                    if (!lzh_cmd(&w, in, rep, lit_off, run, off, take)) {
                        return 0u;
                    }
                    lit_off += run;
                    run = 0u;
                    rem -= take;
                }
                if (!lzh_cmd(&w, in, rep, lit_off, run, off, rem)) {
                    return 0u;
                }
            }
            pos = mpos + len;
            anchor = pos;
            misses = 0u;
            if (pos + ZPSE_LZH_MIN_MATCH > n) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lzh_hash(cur, hash_bits);
#if ZPSE_LZH_PIPE_NEXT
            cand = LZH_TAB_GET(h);
#endif
    }
    if (anchor < n) {
        if (!lzh_literals(&w, in, rep, anchor, n - anchor)) {
            return 0u;
        }
    }
    /* A lone pending command: its token's low nibble is padding and is never
     * read (the command count says how many commands there are). */
    if (w.have) {
        uint32_t ca = lzh_code_of(w.pend.run, w.pend.len);

        if (w.pos >= w.cap) {
            return 0u;
        }
        w.dst[w.pos++] = (uint8_t)(ca << 4);
        if (!lzh_body(&w, in, &w.pend, ca)) {
            return 0u;
        }
        w.have = 0;
    }
    zpse_store_u32le(dst, w.lit_total);
    zpse_store_u32le(dst + 4u, w.cmds);
    return w.pos;
}

ZPSE_HOT uint32_t zpse_lzh_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                                  uint32_t cap, void *table, uint32_t hash_bits)
{
    return lzh_encode_impl(in, n, dst, cap, table, hash_bits, 0u);
}

/* The single hex pass for a caller that has already decided a page's form,
 * with none of the sweep around it. It writes exactly what zpse_lzh_encode
 * writes, so the two are interchangeable in a byte-identity build. */
ZPSE_HOT uint32_t zpse_lzh_encode_page(const uint8_t *in, uint32_t n,
                                       uint8_t *dst, uint32_t cap, void *table,
                                       uint32_t hash_bits)
{
    return lzh_encode_impl(in, n, dst, cap, table, hash_bits, 0u);
}

/* Price a sample of sn bytes (the length passed as `n`) with the hex form:
 * identical stream to zpse_lzh_encode, but only the slots a position in
 * [0,sn) can hash to are cleared. */
ZPSE_HOT uint32_t zpse_lzh_encode_sample(const uint8_t *in, uint32_t n,
                                         uint8_t *dst, uint32_t cap,
                                         void *table, uint32_t hash_bits,
                                         uint32_t sn)
{
    return lzh_encode_impl(in, n, dst, cap, table, hash_bits, sn);
}

/* Copy one match. The fast form's shapes, and for its reason: a distance under
 * sixteen is a repeating pattern rather than a slow copy, and each wide store
 * is only taken when all of it lands inside the block (a decoder hands a block
 * exactly the room its bytes need and decodes blocks side by side). */
static void lzh_copy_match(uint8_t *out, uint32_t op, uint32_t n, uint32_t off,
                           uint32_t mlen)
{
    uint8_t *d = out + op;
    uint32_t left = mlen;

    if (off < 16u && mlen != 0u) {
        static const uint8_t rle_off[16] = {
            0u, 0u, 0u, 1u, 0u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u
        };
        static const uint8_t rle_wide[16] = {
            0u, 1u, 2u, 2u, 4u, 3u, 2u, 1u, 8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u
        };
        const uint32_t need = (mlen + 7u) & ~7u;

        if (op + need <= n) {
            const uint32_t wide = 8u + off - rle_wide[off];
            const uint8_t *s = d - (ptrdiff_t)off;
            uint32_t rest = need;

            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = s[3];
            memcpy(d + 4u, d - (ptrdiff_t)off + rle_off[off], 4u);
            d += 8u;
            rest -= 8u;
            while (rest != 0u) {
                memcpy(d, d - wide, 8u);
                d += 8u;
                rest -= 8u;
            }
            return;
        }
    }
    if (off >= 16u && mlen <= 16u && op + 16u <= n) {
        memcpy(d, d - off, 16u);
        left = 0u;
    } else if (off >= 32u && mlen <= 32u && op + 32u <= n) {
        memcpy(d, d - off, 16u);
        memcpy(d + 16u, d - off + 16u, 16u);
        left = 0u;
    } else if (off >= 8u) {
        /* Step as wide as the distance allows: a step wider than the distance
         * would read bytes this same copy has not written yet, so the guard is
         * the distance, not the size left. Above 32 bytes back the widest
         * step is safe, and a long match is then eight bytes a store only for
         * its tail -- which is what the other corpora pay for: on the binary
         * members the medium-length match is the common one, and this loop
         * was copying it eight bytes at a time. */
        if (off >= 32u) {
            while (left >= 32u) {
                memcpy(d, d - off, 32u);
                d += 32u;
                left -= 32u;
            }
        }
        if (off >= 16u) {
            while (left >= 16u) {
                memcpy(d, d - off, 16u);
                d += 16u;
                left -= 16u;
            }
        }
        while (left >= 8u) {
            memcpy(d, d - off, 8u);
            d += 8u;
            left -= 8u;
        }
    }
    while (left != 0u) {
        *d = d[-(ptrdiff_t)off];
        d++;
        left--;
    }
}

/* A refused stream, with the source line under -DZPSE_LZH_TRACE: which
 * consistency check a stream fails is only visible from here. */
static int lzh_bad(int line)
{
#if defined(ZPSE_LZH_TRACE)
    fprintf(stderr, "lzh: bad stream at line %d\n", line);
#else
    (void)line;
#endif
    return ZPSE_BAD_LZ;
}

/* One chained escape: every 0xFF adds 255 and the first byte below it ends
 * the run. */
static int lzh_escape(const uint8_t *in, uint32_t in_len, uint32_t *pos,
                      uint32_t *add)
{
    uint8_t b;

    *add = 0u;
    do {
        if (*pos >= in_len) {
            return 0;
        }
        b = in[(*pos)++];
        *add += b;
    } while (b == 255u);
    return 1;
}

int zpse_lzh_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n)
{
    uint32_t lit_total, cmd_count, i, lit_seen = 0u, op = 0u;
    uint32_t rep[4] = { 1u, 1u, 1u, 1u };
    uint32_t pos = LZH_HDR;
    uint32_t tok = 0u;

    if (in == 0 || out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0u) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (in_len < ZPSE_LZH_MIN_PAYLOAD) {
        return ZPSE_TRUNCATED;
    }
    lit_total = zpse_load_u32le(in);
    cmd_count = zpse_load_u32le(in + 4u);
    if (lit_total > in_len - LZH_HDR || cmd_count == 0u) {
        return lzh_bad(__LINE__);
    }
    for (i = 0; i < cmd_count && op < n; i++) {
        uint32_t run, len, nib;

        /* Two commands share a token byte, first in the high nibble. */
        if ((i & 1u) == 0u) {
            if (pos >= in_len) {
                return lzh_bad(__LINE__);
            }
            tok = in[pos++];
            nib = tok >> 4;
        } else {
            nib = tok & 15u;
        }
        /* Fast region: every command whose fields are carried without an
         * escape -- a book hit, or a byte token whose two nibbles are below
         * fifteen -- at a distance the form carries in three bytes or fewer.
         * That is nearly every
         * command on a page, and this path needs no escape chain, no bounds
         * test of its own and no call per literal run: a sixteen-byte store
         * covers the longest run these forms carry, and the bytes past it are
         * output a later command overwrites. The two conditions are
         * conservative -- eighteen payload bytes is the widest body this path
         * reads (a byte token, fourteen literals and a three-byte distance),
         * thirty-two output bytes is the longest command it can take (fourteen
         * literals and a seventeen-byte match with the wide stores' slack).
         * Everything outside them falls through to the careful path, which is
         * what a truncated or hostile stream needs anyway. */
        if ((in_len - pos) >= 18u && (n - op) >= 32u) {
            uint32_t frun, flen, p = pos;
            int fast = 1;

            if (nib == LZH_TOK_ESC) {
                uint32_t bt = (uint32_t)in[p];

                frun = bt >> 4u;
                flen = bt & 15u;
                if (frun == 15u || flen == 15u) {
                    fast = 0; /* an escape byte: the careful path's */
                } else {
                    p += 1u;
                    flen = flen == 0u ? 0u : flen + 3u;
                }
            } else {
                frun = (uint32_t)zpse_lzh_book[nib].run;
                flen = (uint32_t)zpse_lzh_book[nib].len;
            }
            /* The cursor is local until the command is taken, because a byte
             * token whose distance is the three-byte form has already been
             * stepped over above: the careful path has to see it where it is.
             * Every branch below that mutates the output is past the distance
             * test, so a command this path will not take leaves no trace. */
            if (fast && (frun != 0u || flen != 0u)) {
                uint32_t b0 = (uint32_t)in[p + frun];

                /* Every distance form but the reserved codes: one byte,
                 * two bytes, the three-byte form and the four repeats. */
                if (flen == 0u || b0 <= 0xEFu || b0 == 0xF0u ||
                    b0 >= (uint32_t)LZH_REP_BASE) {
                    uint32_t off = 0u;

                    memcpy(out + op, in + p, 16u);
                    op += frun;
                    p += frun;
                    lit_seen += frun;
                    if (flen != 0u) {
                        if (b0 >= (uint32_t)LZH_REP_BASE) {
                            off = rep[b0 & 3u];
                            lzh_rep_move(rep, off, b0 & 3u);
                        } else if (b0 <= 0xDFu) {
                            off = b0 + 1u;
                            lzh_rep_move(rep, off, 4u);
                        } else if (b0 <= 0xEFu) {
                            off = 225u + (((b0 & 15u) << 8u) |
                                          (uint32_t)in[p + 1u]);
                            lzh_rep_move(rep, off, 4u);
                            p += 1u;
                        } else {
                            off = 4321u + ((uint32_t)in[p + 1u] |
                                           ((uint32_t)in[p + 2u] << 8u));
                            lzh_rep_move(rep, off, 4u);
                            p += 2u;
                        }
                        p += 1u;
                        if (off > op) {
                            return lzh_bad(__LINE__);
                        }
                        /* A match no longer than a store and at least a
                         * store away is one store; the bytes past it are
                         * overwritten later. */
                        if (off >= 16u && flen <= 16u) {
                            memcpy(out + op, out + op - off, 16u);
                        } else {
                            lzh_copy_match(out, op, n, off, flen);
                        }
                        op += flen;
                    }
                    pos = p;
                    continue;
                }
            }
        }
        if (nib == LZH_TOK_ESC) {
            uint32_t bt, lc;

            if (pos >= in_len) {
                return lzh_bad(__LINE__);
            }
            bt = in[pos++];
            run = bt >> 4;
            lc = bt & 15u;
            if (run == 15u) {
                uint32_t add;

                if (!lzh_escape(in, in_len, &pos, &add)) {
                    return lzh_bad(__LINE__);
                }
                run += add;
            }
            if (lc == 0u) {
                len = 0u;
            } else if (lc == 15u) {
                uint32_t add;

                if (!lzh_escape(in, in_len, &pos, &add)) {
                    return lzh_bad(__LINE__);
                }
                len = 18u + add;
            } else {
                len = lc + 3u;
            }
        } else {
            run = zpse_lzh_book[nib].run;
            /* Every book length is below the escape, 18, so it is taken as
             * written; the book never needs a second escape byte. */
            len = zpse_lzh_book[nib].len;
        }
        if (run > n - op || run > in_len - pos) {
            return lzh_bad(__LINE__);
        }
        if (run != 0u) {
            /* One store covers any run without an escape when both sides have
             * the room; the bytes past the run are output a later command
             * overwrites. */
            if (run < 15u && (n - op) >= 16u && (in_len - pos) >= 16u) {
                memcpy(out + op, in + pos, 16u);
            } else {
                memcpy(out + op, in + pos, run);
            }
            op += run;
            pos += run;
            lit_seen += run;
        }
        if (len == 0u) {
            if (run == 0u) {
                return lzh_bad(__LINE__);
            }
            continue;
        }
        if (pos >= in_len) {
            return lzh_bad(__LINE__);
        }
        {
            uint32_t b0 = (uint32_t)in[pos];
            uint32_t off;

            if (b0 >= (uint32_t)LZH_REP_BASE) {
                off = rep[b0 & 3u];
                lzh_rep_move(rep, off, b0 & 3u);
                pos += 1u;
            } else if (b0 <= 0xDFu) {
                off = b0 + 1u;
                lzh_rep_move(rep, off, 4u);
                pos += 1u;
            } else if (b0 <= 0xEFu) {
                if (in_len - pos < 2u) {
                    return lzh_bad(__LINE__);
                }
                off = 225u + (((b0 & 15u) << 8) | (uint32_t)in[pos + 1u]);
                lzh_rep_move(rep, off, 4u);
                pos += 2u;
            } else if (b0 == 0xF0u) {
                if (in_len - pos < 3u) {
                    return lzh_bad(__LINE__);
                }
                off = 4321u + ((uint32_t)in[pos + 1u] |
                               ((uint32_t)in[pos + 2u] << 8u));
                lzh_rep_move(rep, off, 4u);
                pos += 3u;
            } else {
                /* 0xF1..0xFB are reserved: a stream that names one is not
                 * one this encoder wrote. */
                return lzh_bad(__LINE__);
            }
            if (off == 0u || off > op || len > n - op) {
                return lzh_bad(__LINE__);
            }
            lzh_copy_match(out, op, n, off, len);
            op += len;
        }
    }
    if (i != cmd_count || op != n || lit_seen != lit_total || pos != in_len) {
        return lzh_bad(__LINE__);
    }
    return ZPSE_OK;
}
