#include "zpse/lz2.h"

#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* An empty hash slot. Positions are 16-bit because a block is at most
 * 64 KiB, and the encoder never inserts a position past n - 4, so 0xFFFF is
 * not a legitimate entry. */
#define LZ2_EMPTY 0xFFFFu

/* The payload header: literal total and token count. */
#define LZ2_HDR 8u

/* The most bytes one command can write: the token, a one-byte escape for the
 * literal count, the run itself (at most ZPSE_LZ2_MAX_LIT), the two-byte
 * distance and a one-byte escape for the match length. The writer tests for
 * that much room once instead of working out the exact size, and falls back
 * to the exact sum only at the cap boundary. */
#define LZ2_CMD_ROOM 260u

/* How fast the finder gallops once a run of positions has found nothing: the
 * step is 1 + (misses >> LZ2_SKIP_BITS). A parameter so the trade can be
 * measured -- a smaller shift finds more matches (better ratio) and costs
 * probes on data that has none -- and the default is the value the sweep of
 * 5..9 chose, see the README. */
#ifndef ZPSE_LZ2_SKIP_BITS
#define ZPSE_LZ2_SKIP_BITS 5u
#endif

/* Lazy matching: the longest match the finder will still spend a second
 * probe on to see whether the next byte starts a longer one. 4 disables the
 * probe (the match is at least four bytes, so nothing passes the test).
 *
 * This is what closes the size gap to the token form, which has always
 * looked ahead: a match that starts one byte later and runs longer is worth
 * more than the one in hand, because the byte given up becomes the first
 * byte of the next literal run rather than a byte of the match. Past a few
 * dozen bytes the probe almost never pays -- a long match at one position
 * is a long match at the next -- so the ceiling keeps the cost on the short
 * matches where the decision actually differs, which are also the ones the
 * probe can settle with a single xor. A parameter so the trade can be
 * measured; see the README for the sweep that chose the value. */
#ifndef ZPSE_LZ2_LAZY_MAX
#define ZPSE_LZ2_LAZY_MAX 24u
#endif

static uint32_t lz2_hash(uint32_t v, uint32_t hash_bits)
{
    /* One multiply at the cursor, top bits kept: the four-byte compare at the
     * candidate is the tag check, so no tag has to be stored. */
    return (v * UINT32_C(2654435761)) >> (32u - hash_bits);
}

static uint64_t lz2_load64(const uint8_t *p)
{
    uint64_t v;

    memcpy(&v, p, 8u);
    return v;
}

/* The four most recent distances, most recent first.
 *
 * A repeat is coded as ONE byte -- 0xFC plus the slot -- instead of the two a
 * fresh distance takes, which is the size the token form gets from its repeat
 * cache and the reason this form was a trade rather than a win without it: on
 * an executable, and on a block of spikes every 256 bytes, most matches land
 * on a distance one of the last four commands used.
 *
 * The four codes sit at the top of the first distance byte, so the encoder
 * cannot write an explicit distance whose low byte is 0xFC..0xFF. That is one
 * distance value in sixty-four, and a match that needs one is coded as a
 * literal instead (see the loop): a rare, bounded loss, rather than a second
 * width for every command to test.
 *
 * The decoder's test is one compare on one byte, which is what keeps this
 * form's decode loop the tight one the format exists for -- the varint this
 * replaced cost it a third of its rate. */
#define LZ2_REP_BASE 0xFCu

static uint32_t lz2_rep_index(const uint32_t *rep, uint32_t off)
{
    uint32_t m0 = (rep[0] == off);
    uint32_t m1 = (rep[1] == off);
    uint32_t m2 = (rep[2] == off);
    uint32_t m3 = (rep[3] == off);

    return m0 ? 0u : (m1 ? 1u : (m2 ? 2u : (m3 ? 3u : 4u)));
}

/* Move the cache for a distance whose slot `idx` was just looked up (4 = an
 * explicit distance). A hit promotes its slot to the front; an explicit
 * distance pushes the front down and drops the oldest. The decoder moves it
 * identically, which is the whole contract between the two sides. */
static void lz2_rep_move(uint32_t *rep, uint32_t off, uint32_t idx)
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

/* Decode side: resolve a distance value (1..4 = the k-th recent distance,
 * 0 = a literal-only command) and move the cache the way the encoder did. */
static inline uint32_t lz2_dist_take(uint32_t *rep, uint32_t v)
{
    uint32_t i, off, n1, n2, n3;

    if (v == 0u) {
        return 0u;
    }
    /* 1..4 name a slot, and the move is the encoder's: the hit goes to the
     * front and everything before it slides back one. Selects rather than a
     * shift loop, because which slot a command names is data and the slot
     * changes often enough -- this is the hot form on text -- that a loop
     * with a data-dependent trip count is a branch to mispredict per command
     * and three stores to a loop counter that has to be right. */
    i = v - 1u;
    off = (i == 0u) ? rep[0]
                    : ((i == 1u) ? rep[1] : ((i == 2u) ? rep[2] : rep[3]));
    n1 = (i >= 1u) ? rep[0] : rep[1];
    n2 = (i >= 2u) ? rep[1] : rep[2];
    n3 = (i == 3u) ? rep[2] : rep[3];
    rep[0] = off;
    rep[1] = n1;
    rep[2] = n2;
    rep[3] = n3;
    return off;
}

/* One command: the literal run before the match, the match length and the
 * distance. `off == 0` writes a literal-only command, which is how a run
 * longer than one token and a trailing run are coded. Returns 0 when the
 * command does not fit in cap. */
static int lz2_put(uint8_t *dst, uint32_t cap, uint32_t *out,
                   const uint8_t *lit, uint32_t run, uint32_t off,
                   uint32_t len, uint32_t *rep, uint32_t *lit_total,
                   uint32_t *cmds)
{
    uint32_t ml = len >= ZPSE_LZ2_MIN_MATCH ? len - ZPSE_LZ2_MIN_MATCH : 0u;
    uint32_t p = *out;
    uint32_t ridx = (off != 0u) ? lz2_rep_index(rep, off) : 4u;
    uint32_t dlen = (ridx < 4u) ? 1u : 2u;
    uint8_t *t;

    if (LZ2_CMD_ROOM > cap - p) {
        uint32_t need = 1u + run + dlen + (run >= 15u ? 1u : 0u) +
                        (ml >= 15u ? 1u : 0u);

        if (need > cap - p) {
            return 0;
        }
    }
    t = dst + p;
    *t++ = (uint8_t)(((run < 15u ? run : 15u) << 4) |
                     (ml < 15u ? ml : 15u));
    if (run >= 15u) {
        /* One escape byte: the chunker never asks for more than 255. */
        *t++ = (uint8_t)(run - 15u);
    }
    if (run != 0u) {
        memcpy(t, lit, run);
        t += run;
        *lit_total += run;
    }
    /* The distance: one byte for a recent one, two little endian for the
     * rest, and two zero bytes for a literal-only command (which names no
     * distance and leaves the cache alone). */
    if (off == 0u) {
        *t++ = 0u;
        *t++ = 0u;
    } else if (ridx < 4u) {
        *t++ = (uint8_t)(LZ2_REP_BASE | ridx);
        lz2_rep_move(rep, off, ridx);
    } else {
        *t++ = (uint8_t)(off & 0xFFu);
        *t++ = (uint8_t)((off >> 8) & 0xFFu);
        lz2_rep_move(rep, off, 4u);
    }
    if (ml >= 15u) {
        *t++ = (uint8_t)(ml - 15u);
    }
    *out = (uint32_t)(t - dst);
    (*cmds)++;
    return 1;
}

/* Forward extension for the lazy probe only: the main loop's copy is inline
 * because it runs at every match, and this one runs once per candidate the
 * probe asks about. Same shape, so a length means the same thing on both
 * sides of the decision. */
static uint32_t lz2_extend(const uint8_t *in, uint32_t n, uint32_t a,
                           uint32_t b, uint32_t len)
{
    while (a + len + 8u <= n) {
        uint64_t x = lz2_load64(in + a + len) ^ lz2_load64(in + b + len);

        if (x != 0u) {
            return len + (zpse_ctz64(x) >> 3);
        }
        len += 8u;
    }
    while (a + len < n && in[a + len] == in[b + len]) {
        len++;
    }
    return len;
}

/* Tagged with the rest of the hot set: this finder is the whole of the F
 * form's encode, and it was the one kernel in the tree left at the build's
 * own level. */
ZPSE_HOT uint32_t zpse_lz2_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits)
{
    uint16_t *tab = (uint16_t *)table;
    uint32_t entries = 1u << hash_bits;
    uint32_t pos, anchor, misses;
    uint32_t lit_total = 0u, cmds = 0u, out = LZ2_HDR;
    uint32_t cur, h, limit;
    uint32_t rep[4] = { 1u, 1u, 1u, 1u };

    if (in == 0 || dst == 0 || tab == 0) {
        return 0u;
    }
    if (n < ZPSE_LZ2_MIN_MATCH + 1u || n > 65536u) {
        return 0u;
    }
    if (cap < ZPSE_LZ2_MIN_PAYLOAD) {
        return 0u;
    }
    /* LZ2_EMPTY is 0xFFFF and the table is uint16_t, so all-ones bytes are
     * exactly the empty marker. One memset beats the 16-bit fill loop, which
     * gcc does not vectorise at -O2, for the same reason src/lz.c's clear
     * does (see the comment there). */
    memset(tab, 0xFF, (size_t)entries * sizeof *tab);
    pos = 0u;
    anchor = 0u;
    misses = 0u;
    limit = n - ZPSE_LZ2_MIN_MATCH;
    cur = zpse_load_u32le(in + pos);
    h = lz2_hash(cur, hash_bits);

    while (pos <= limit) {
        uint32_t cand = tab[h];
        uint32_t len = ZPSE_LZ2_MIN_MATCH;
        /* The match's own start: the walk below moves these instead of the
         * scanning cursor (see there). */
        uint32_t mpos = pos;
        uint32_t mcand = cand;

        tab[h] = (uint16_t)pos;
        if (ZPSE_LIKELY(cand == LZ2_EMPTY ||
                        zpse_load_u32le(in + cand) != cur)) {
            /* Two misses are a coincidence; a run of them means the data has
             * no match here, so the cursor accelerates. */
            pos += 1u + (misses >> ZPSE_LZ2_SKIP_BITS);
            misses++;
            if (pos > limit) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lz2_hash(cur, hash_bits);
            continue;
        }
        /* Extend backward into the pending literals: an earlier start is a
         * longer match for the same four bytes, and the run in front of it is
         * that many bytes shorter. That walk is where the token form earns
         * most of its size over this one, because it has always done it.
         *
         * It moves `mpos`/`mcand`, NOT the scanning cursor. `pos` was written
         * into the table above and every position the literal scan probed on
         * the way here is in it too; moving the cursor back over those would
         * leave entries AHEAD of it, and the first iteration that read one
         * would subtract to an enormous distance and desync the stream --
         * which is exactly how the first version of this broke dickens at
         * byte 2354 of a 3 KiB block. Coding the match from a copy cannot:
         * the entries stay behind the end of the match, `mpos + len`, which
         * is at least `pos + 4` because the walk only steps over bytes that
         * already compare equal.
         *
         * The `mpos > anchor` bound is what keeps it honest: the walk can
         * only take bytes this block has not coded yet, so the decoder never
         * sees a match that starts before the last one ended. `len` below is
         * measured from the new start, so the walked bytes are bytes the
         * match now covers rather than bytes it has to give back, and
         * `mcand > 0` keeps the reference inside the block. */
        while (mpos > anchor && mcand > 0u && in[mpos - 1u] == in[mcand - 1u]) {
            mpos--;
            mcand--;
        }
        /* Extend forward one eight-byte xor at a time. This is the only work
         * a match does: no lookahead at pos+1, no distance cache to update
         * beyond the one byte the command carries.
         *
         * The xor is the length as well as the test: its lowest set bit is
         * the first byte that differs, so one pair of loads answers both "is
         * this step equal" and "how far in does it stop". The loop used to
         * compare a pair and then load the same pair again to count its tail,
         * which on prose -- average match 5.4 bytes, so it exits on its first
         * test -- was two loads and a branch for one length.
         *
         * One four-byte compare for the candidate and one eight-byte xor for
         * the length is the cheapest shape tried so far: folding them into a
         * single eight-byte xor (so a miss pays a load pair and a ctz instead
         * of a compare) measured 12-17% SLOWER on prose, columnar XML and the
         * binary members, because misses outnumber hits and a miss is the
         * common case. Only the forward bound is tested below: the candidate
         * is behind the cursor, so a step that fits at pos fits at cand. */
        {
            int tail = 0;

            while (mpos + len + 8u <= n) {
                uint64_t x = lz2_load64(in + mpos + len) ^
                             lz2_load64(in + mcand + len);

                if (x != 0u) {
                    len += zpse_ctz64(x) >> 3;
                    tail = 1;
                    break;
                }
                len += 8u;
            }
            /* Only the end of the block reaches this: eight bytes are not
             * readable there and the length has to be counted one at a
             * time. */
            if (tail == 0) {
                while (mpos + len < n && in[mpos + len] == in[mcand + len]) {
                    len++;
                }
            }
        }
        /* Lazy matching, against the entry at the next position rather than
         * a fresh probe of the whole table: the candidate that hashes to the
         * position one byte on is the one a match there would use, so one
         * load, one compare and one xor settle whether this match is the one
         * to code. If the next position's is strictly longer, the byte at
         * `mpos` becomes a literal -- the cursor steps over it and the next
         * command's literal run picks it up, which is why nothing has to be
         * written here -- and the loop re-finds the longer match at its own
         * position, where it walks back into that run like any other.
         *
         * `c1 <= mpos` keeps the reference behind the new cursor: the entry
         * read is whatever the block coded before, never the position being
         * filed (this function writes `tab[h] = pos` at the loop top, and
         * `pos + 1` is not written until the next iteration files it).
         *
         * `mpos == pos` is the other half of that guarantee, and the one
         * that costs a stream when it is missing. The backward walk above
         * can start the match BEHIND the scanning cursor, and stepping to
         * `mpos + 1` would then move the cursor backwards, into positions
         * the literal scan has already filed: the next candidate read from
         * one of them sits ahead of the cursor, `pos - cand` underflows to a
         * sixty-thousand-byte distance, and the block no longer decodes
         * (mozilla, block 590, first bad byte 918 -- the first version of
         * this had no guard). A walked-back match is already the better
         * match, so there is nothing to look ahead for: the probe only runs
         * when the match starts where the cursor is. */
        if (mpos == pos && len <= (uint32_t)ZPSE_LZ2_LAZY_MAX &&
            mpos + 1u + ZPSE_LZ2_MIN_MATCH <= n) {
            uint32_t cur1 = zpse_load_u32le(in + mpos + 1u);
            uint32_t h1 = lz2_hash(cur1, hash_bits);
            uint32_t c1 = tab[h1];

            if (c1 != LZ2_EMPTY && c1 <= mpos &&
                zpse_load_u32le(in + c1) == cur1 &&
                lz2_extend(in, n, mpos + 1u, c1, ZPSE_LZ2_MIN_MATCH) > len) {
                pos = mpos + 1u;
                misses = 0u;
                cur = cur1;
                h = h1;
                continue;
            }
        }
        {
            uint32_t run = mpos - anchor;
            const uint8_t *ls = in + anchor;
            uint32_t off = mpos - mcand;
            uint32_t ridx = lz2_rep_index(rep, off);
            uint32_t rem = len;

            /* A distance whose low byte is one of the four repeat codes cannot
             * be written as itself, so this position is coded as a literal and
             * the search resumes one byte on: the bytes are still in the
             * stream, in the literal run that grows across them. */
            if (ridx == 4u && (off & 0xFFu) >= LZ2_REP_BASE) {
                pos++;
                misses++;
                if (pos + ZPSE_LZ2_MIN_MATCH > n) {
                    break;
                }
                cur = zpse_load_u32le(in + pos);
                h = lz2_hash(cur, hash_bits);
                continue;
            }

            /* The common command, written in place: a literal run and a match
             * that both fit their nibbles, so the whole command is a token,
             * the run and two distance bytes. On prose that is four commands
             * in five, and the call plus its room arithmetic was a fifth of
             * the writer's cost. The bytes are exactly lz2_put's, escapes and
             * all, which is what keeps the stream identical -- `rem` is left
             * at zero so the general path below does not emit a second
             * command for the same match. */
            if (run < 15u && rem < 15u + ZPSE_LZ2_MIN_MATCH &&
                cap - out >= run + (ridx < 4u ? 2u : 3u)) {
                dst[out] = (uint8_t)((run << 4) |
                                     (rem - ZPSE_LZ2_MIN_MATCH));
                if (run != 0u) {
                    /* A run of up to fourteen literals leaves as one
                     * 16-byte store when the buffer and the block both have
                     * the bytes: `run` is a register value, so a memcpy of
                     * it is a size dispatch per command, and this command
                     * shape is four in five of them on prose. The bytes
                     * past the run are the distance and the next token,
                     * both written before anything reads them -- the store
                     * is 16 bytes into a buffer with at least 17 left, so
                     * it cannot leave the payload either. */
                    if (cap - out >= 17u && anchor + 16u <= n) {
                        memcpy(dst + out + 1u, ls, 16u);
                    } else {
                        memcpy(dst + out + 1u, ls, run);
                    }
                }
                if (ridx < 4u) {
                    dst[out + 1u + run] = (uint8_t)(LZ2_REP_BASE | ridx);
                    out += run + 2u;
                } else {
                    dst[out + 1u + run] = (uint8_t)(off & 0xFFu);
                    dst[out + 2u + run] = (uint8_t)((off >> 8) & 0xFFu);
                    out += run + 3u;
                }
                lz2_rep_move(rep, off, ridx);
                lit_total += run;
                cmds++;
                run = 0u;
                rem = 0u;
            }
            while (run > ZPSE_LZ2_MAX_LIT) {
                if (!lz2_put(dst, cap, &out, ls, ZPSE_LZ2_MAX_LIT, 0u, 0u, rep, &lit_total, &cmds)) {
                    return 0u;
                }
                ls += ZPSE_LZ2_MAX_LIT;
                run -= ZPSE_LZ2_MAX_LIT;
            }
            /* Split a long match into commands of at most
             * ZPSE_LZ2_MAX_MATCH. Every chunk has to be at least
             * ZPSE_LZ2_MIN_MATCH long: the format codes a match as
             * (length - 4), so a chunk of one to three bytes would be written
             * as zero and read back as four, which desyncs the output by the
             * difference. A remainder that would be that short is avoided by
             * shortening the chunk before it and leaving exactly
             * MIN_MATCH at the end -- reachable only when the match ends four
             * bytes past a chunk boundary. */
            while (rem > ZPSE_LZ2_MAX_MATCH) {
                uint32_t take = ZPSE_LZ2_MAX_MATCH;

                if (rem - take < ZPSE_LZ2_MIN_MATCH) {
                    take = rem - ZPSE_LZ2_MIN_MATCH;
                }
                if (!lz2_put(dst, cap, &out, ls, run, off, take, rep,
                             &lit_total, &cmds)) {
                    return 0u;
                }
                ls += run;
                run = 0u;
                rem -= take;
            }
            if (rem != 0u &&
                !lz2_put(dst, cap, &out, ls, run, off, rem, rep, &lit_total,
                         &cmds)) {
                return 0u;
            }
        }
        pos = mpos + len;
        anchor = pos;
        misses = 0u;
        if (pos + ZPSE_LZ2_MIN_MATCH > n) {
            break;
        }
        cur = zpse_load_u32le(in + pos);
        h = lz2_hash(cur, hash_bits);
    }
    if (anchor < n) {
        uint32_t run = n - anchor;
        const uint8_t *ls = in + anchor;

        while (run > ZPSE_LZ2_MAX_LIT) {
            if (!lz2_put(dst, cap, &out, ls, ZPSE_LZ2_MAX_LIT, 0u, 0u, rep,
                         &lit_total, &cmds)) {
                return 0u;
            }
            ls += ZPSE_LZ2_MAX_LIT;
            run -= ZPSE_LZ2_MAX_LIT;
        }
        if (!lz2_put(dst, cap, &out, ls, run, 0u, 0u, rep, &lit_total,
                     &cmds)) {
            return 0u;
        }
    }
    zpse_store_u32le(dst, lit_total);
    zpse_store_u32le(dst + 4u, cmds);
    return out;
}

/* Copy one match: sixteen bytes for a short match at a long distance, two
 * stores for the next shape up, then eight-byte steps and a byte tail. Each
 * wide store is only taken when all of it lands inside the block, because a
 * frame decoder hands a block exactly the room its bytes need and decodes
 * blocks side by side, so a store past the end would land in a neighbour's
 * output. */
static void lz2_copy_match(uint8_t *out, uint32_t op, uint32_t n,
                           uint32_t off, uint32_t mlen)
{
    uint8_t *d = out + op;
    uint32_t left = mlen;

    /* A short offset is a repeating pattern, not a slow copy -- lzav's
     * shape. Write the first eight bytes of the pattern with four byte
     * moves and one four-byte store, then copy eight bytes at a time at a
     * distance that is a multiple of the period. The four-byte store reads
     * at `4 mod off` back into the bytes just written, so it is in phase,
     * and `wide` is the smallest multiple of the period that is at least
     * eight, i.e. the narrowest distance an eight-byte store may read from.
     * Without it, an offset under sixteen is a byte loop for the whole
     * match in this function, and on the binary corpora those offsets are
     * the common ones: a record layout repeats its 2- and 4-byte fields at
     * exactly those distances.
     *
     * Measured as decode only -- the streams built once, then each block
     * decoded repeatedly for about five seconds a reading, best of three,
     * two binaries built from the tree before and after, run in both
     * orders: webster 1 997 -> 2 030 MB/s (1.6%), reymont 1 713 -> 1 782
     * (4.0%), xml 2 281 -> 2 345 (2.8%), and dickens, osdb and ooffice
     * within 0.3%. Timing the encode passes in front of every decode
     * reading put the second binary on a hotter core and made xml and osdb
     * look 2-3% worse than they are; that bias is why this harness no
     * longer does it.
     *
     * The room test rounds the length up to the whole eight-byte stores
     * this writes, and a frame decoder hands a block exactly the room its
     * bytes need, so a match at the very end of a block falls through to
     * the byte loop rather than writing into a neighbour's output. */
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

int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n)
{
    uint32_t lit_total, cmd_count, cmd_left, lit_seen = 0u, op = 0u;
    uint32_t rep[4] = { 1u, 1u, 1u, 1u };
    const uint8_t *lp, *tend;

    if (in == 0 || out == 0) {
        return PSE_BAD_ARG;
    }
    if (n == 0u) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (in_len < ZPSE_LZ2_MIN_PAYLOAD) {
        return PSE_TRUNCATED;
    }
    lit_total = zpse_load_u32le(in);
    cmd_count = zpse_load_u32le(in + 4u);
    if (lit_total > in_len - LZ2_HDR || cmd_count == 0u) {
        return PSE_BAD_LZ;
    }
    lp = in + LZ2_HDR;
    tend = in + in_len;
    cmd_left = cmd_count;

    while (op < n) {
        uint32_t tok, lit, mlen, off;
        uint8_t b;

        if (cmd_left == 0u) {
            return PSE_BAD_LZ;
        }
        cmd_left--;
        /* Fast region: enough payload left for the largest command this path
         * can take (a token, fourteen literals and the two offset bytes) and
         * enough output for the longest match it can code, so a command whose
         * two nibbles are both literal needs no room test of its own -- and
         * that is four commands in five on prose. Everything else falls
         * through to the careful path, which is what a truncated or hostile
         * stream needs anyway. */
        if ((size_t)(tend - lp) >= 18u && (n - op) >= 19u) {
            tok = *lp;
            if (((tok >> 4) != 15u) && ((tok & 15u) != 15u)) {
                lit = tok >> 4;
                lp++;
                /* One store covers any run this token carries, and the bytes
                 * past it are output a later command overwrites. */
                memcpy(out + op, lp, 16u);
                op += lit;
                lp += lit;
                lit_seen += lit;
                /* The distance, with one branch on which kind it is.
                 *
                 * `lp[0] >= 0xFC` asks whether this command names one of the
                 * four recent distances or an explicit one. This used to
                 * compute both candidates and select, on the argument that
                 * the answer is data and the test mispredicts. It is data,
                 * but it is lopsided: over a census of the twelve Silesia
                 * files at 64 KiB, a rep code is 0-26% of commands and under
                 * 5% on nine of them, because this form is the one the cheap
                 * rungs take on every block and its finder only names a
                 * recent distance when the match is fresh. So the branch is
                 * almost never taken, while the select chain ran on every
                 * command. Measured decode-only, prebuilt -F frames, best of
                 * nine passes, alternated: dickens 2 080 -> 2 106 MB/s,
                 * mozilla 2 099 -> 2 162, x-ray 4 346 -> 4 521, nci
                 * 2 632 -> 2 645, xml 2 736 -> 2 752. The rep path keeps the
                 * select chain it needs. */
                {
                    uint32_t c0 = (uint32_t)lp[0];

                    if (c0 < (uint32_t)LZ2_REP_BASE) {
                        off = c0 | ((uint32_t)lp[1] << 8u);
                        lp += 2u;
                        /* A literal-only command (an explicit zero) names no
                         * distance and leaves the cache alone. */
                        if (off != 0u) {
                            rep[3] = rep[2];
                            rep[2] = rep[1];
                            rep[1] = rep[0];
                            rep[0] = off;
                        }
                    } else {
                        uint32_t ri = c0 & 3u;
                        uint32_t n1, n2, n3;

                        off = rep[ri];
                        n1 = (ri >= 1u) ? rep[0] : rep[1];
                        n2 = (ri >= 2u) ? rep[1] : rep[2];
                        n3 = (ri == 3u) ? rep[2] : rep[3];
                        rep[0] = off;
                        rep[1] = n1;
                        rep[2] = n2;
                        rep[3] = n3;
                        lp += 1u;
                    }
                }
                mlen = (uint32_t)(tok & 15u) + ZPSE_LZ2_MIN_MATCH;
                if (off == 0u) {
                    if (lit == 0u) {
                        return PSE_BAD_LZ;
                    }
                    continue;
                }
                if (off > op) {
                    return PSE_BAD_LZ;
                }
                /* A short match at a distance at least as wide as the store:
                 * the whole match is one store, and the bytes past it are
                 * output a later command overwrites. Thirty-two bytes covers
                 * the longest match a two-nibble token can name, which is
                 * what this region is restricted to. */
                if (off >= 32u && mlen <= 32u && op + 32u <= n) {
                    memcpy(out + op, out + op - off, 32u);
                } else if (off >= 16u && mlen <= 16u && op + 16u <= n) {
                    memcpy(out + op, out + op - off, 16u);
                } else {
                    lz2_copy_match(out, op, n, off, mlen);
                }
                op += mlen;
                continue;
            }
        }
        if (lp >= tend) {
            return PSE_BAD_LZ;
        }
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
        mlen = (tok & 15u) + ZPSE_LZ2_MIN_MATCH;
        if (lit > n - op || (uint32_t)(tend - lp) < lit) {
            return PSE_BAD_LZ;
        }
        if (lit != 0u) {
            if (lit < 15u && n - op >= 16u && (uint32_t)(tend - lp) >= 16u) {
                /* One store covers every run a token can carry without an
                 * escape, not just the eight-byte ones: the size is data, so
                 * the narrow form is a dispatch per command, and on prose the
                 * run is five bytes on average. The bytes past the run are
                 * output a later command overwrites, and both the output and
                 * the payload are checked to have the sixteen before the
                 * store is taken. */
                memcpy(out + op, lp, 16u);
            } else {
                memcpy(out + op, lp, lit);
            }
            op += lit;
            lp += lit;
            lit_seen += lit;
        }
        if (lp >= tend) {
            return PSE_BAD_LZ;
        }
        if (lp[0] >= LZ2_REP_BASE) {
            off = lz2_dist_take(rep, (uint32_t)(lp[0] & 3u) + 1u);
            lp++;
        } else {
            if ((uint32_t)(tend - lp) < 2u) {
                return PSE_BAD_LZ;
            }
            off = (uint32_t)lp[0] | ((uint32_t)lp[1] << 8u);
            lp += 2;
            if (off != 0u) {
                lz2_rep_move(rep, off, 4u);
            }
        }
        if (off == 0u) {
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
        if (off > op || mlen > n - op) {
            return PSE_BAD_LZ;
        }
        lz2_copy_match(out, op, n, off, mlen);
        op += mlen;
    }
    if (cmd_left != 0u || lit_seen != lit_total || lp != tend) {
        return PSE_BAD_LZ;
    }
    return PSE_OK;
}
