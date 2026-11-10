#include "zpse/lzh.h"

#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/status.h"

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

/* The nibble for this (run, length) pair, or 15 when the book does not have
 * it and the byte token has to spell the fields out. Only a handful of runs
 * and lengths are in the book, so the scan stops as soon as either field is
 * out of range rather than walking all sixteen. */
static uint32_t lzh_code_of(uint32_t run, uint32_t len)
{
    uint32_t i;

    if (run > 4u || len > 15u) {
        return LZH_TOK_ESC;
    }
    for (i = 0; i < ZPSE_LZH_CODES; i++) {
        if (zpse_lzh_book[i].run == run && zpse_lzh_book[i].len == len) {
            return i;
        }
    }
    return LZH_TOK_ESC;
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
        need += (c->ridx < 4u) ? 1u : 2u;
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
        if (c->ridx < 4u) {
            *t++ = (uint8_t)(LZH_REP_BASE | c->ridx);
        } else {
            *t++ = (uint8_t)(c->off & 0xFFu);
            *t++ = (uint8_t)((c->off >> 8) & 0xFFu);
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

/* Lazy probe helper, the fast form's: the length of the match at (a, b). */
static uint32_t lzh_extend(const uint8_t *in, uint32_t n, uint32_t a,
                           uint32_t b, uint32_t len)
{
    while (a + len + 8u <= n) {
        uint64_t x = lzh_load64(in + a + len) ^ lzh_load64(in + b + len);

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

ZPSE_HOT uint32_t zpse_lzh_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                                  uint32_t cap, void *table, uint32_t hash_bits)
{
    uint16_t *tab = (uint16_t *)table;
    uint32_t entries = 1u << hash_bits;
    uint32_t pos, anchor, misses, i;
    uint32_t cur, h, limit;
    uint32_t rep[4] = { 1u, 1u, 1u, 1u };
    LzhWr w;

    if (in == 0 || dst == 0 || tab == 0) {
        return 0u;
    }
    if (n < ZPSE_LZH_MIN_MATCH + 1u || n > 65536u) {
        return 0u;
    }
    if (cap < ZPSE_LZH_MIN_PAYLOAD) {
        return 0u;
    }
    {
        uint64_t fill = UINT64_C(0xFFFFFFFFFFFFFFFF);

        for (i = 0; i + 4u <= entries; i += 4u) {
            memcpy(tab + i, &fill, 8u);
        }
        for (; i < entries; i++) {
            tab[i] = (uint16_t)LZH_EMPTY;
        }
    }
    w.dst = dst;
    w.cap = cap;
    w.pos = LZH_HDR;
    w.lit_total = 0u;
    w.cmds = 0u;
    w.have = 0;

    pos = 0u;
    anchor = 0u;
    misses = 0u;
    limit = n - ZPSE_LZH_MIN_MATCH;
    cur = zpse_load_u32le(in + pos);
    h = lzh_hash(cur, hash_bits);

    while (pos <= limit) {
        uint32_t cand = tab[h];
        uint32_t len = ZPSE_LZH_MIN_MATCH;
        uint32_t mpos = pos;
        uint32_t mcand = cand;

        tab[h] = (uint16_t)pos;
        if (__builtin_expect(cand == LZH_EMPTY ||
                                 zpse_load_u32le(in + cand) != cur,
                             1)) {
            pos += 1u + (misses >> 5u);
            misses++;
            if (pos > limit) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lzh_hash(cur, hash_bits);
            continue;
        }
        while (mpos > anchor && mcand > 0u && in[mpos - 1u] == in[mcand - 1u]) {
            mpos--;
            mcand--;
        }
        {
            int tail = 0;

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
        if (mpos == pos && len <= 24u && mpos + 1u + ZPSE_LZH_MIN_MATCH <= n) {
            uint32_t cur1 = zpse_load_u32le(in + mpos + 1u);
            uint32_t h1 = lzh_hash(cur1, hash_bits);
            uint32_t c1 = tab[h1];

            if (c1 != LZH_EMPTY && c1 <= mpos &&
                zpse_load_u32le(in + c1) == cur1 &&
                lzh_extend(in, n, mpos + 1u, c1, ZPSE_LZH_MIN_MATCH) > len) {
                pos = mpos + 1u;
                misses = 0u;
                cur = cur1;
                h = h1;
                continue;
            }
        }
        {
            uint32_t run = mpos - anchor;
            uint32_t off = mpos - mcand;
            uint32_t ridx = lzh_rep_index(rep, off);
            uint32_t rem = len;
            uint32_t lit_off = anchor;

            /* A fresh distance whose low byte is one of the four repeat
             * codes cannot be written as itself, so the position becomes a
             * literal and the search resumes one byte on -- the same bounded
             * loss the fast form takes. */
            if (ridx == 4u && (off & 0xFFu) >= LZH_REP_BASE) {
                pos++;
                misses++;
                if (pos + ZPSE_LZH_MIN_MATCH > n) {
                    break;
                }
                cur = zpse_load_u32le(in + pos);
                h = lzh_hash(cur, hash_bits);
                continue;
            }
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
    return PSE_BAD_LZ;
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
        return PSE_BAD_ARG;
    }
    if (n == 0u) {
        return PSE_BAD_BLOCK_SIZE;
    }
    if (in_len < ZPSE_LZH_MIN_PAYLOAD) {
        return PSE_TRUNCATED;
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
            memcpy(out + op, in + pos, run);
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
            uint32_t c0 = (uint32_t)in[pos];
            uint32_t isrep = (c0 >= (uint32_t)LZH_REP_BASE) ? 1u : 0u;
            uint32_t ri = c0 & 3u;
            uint32_t n1, n2, n3, off;

            if (isrep == 0u) {
                if (in_len - pos < 2u) {
                    return lzh_bad(__LINE__);
                }
                off = c0 | ((uint32_t)in[pos + 1u] << 8u);
            } else {
                off = rep[ri];
            }
            n1 = isrep != 0u ? ((ri >= 1u) ? rep[0] : rep[1]) : rep[0];
            n2 = isrep != 0u ? ((ri >= 2u) ? rep[1] : rep[2]) : rep[1];
            n3 = isrep != 0u ? ((ri == 3u) ? rep[2] : rep[3]) : rep[2];
            pos += isrep != 0u ? 1u : 2u;
            if (off != 0u) {
                rep[0] = off;
                rep[1] = n1;
                rep[2] = n2;
                rep[3] = n3;
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
    return PSE_OK;
}
