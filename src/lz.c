#include "zpse/lz.h"

#include "zpse/bits.h"
#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* An empty hash slot. Positions are 16-bit because a block is at most
 * 64 KiB, and the encoder never inserts a position past n - 4, so 0xFFFF
 * is not a legitimate entry. */
#define LZ_EMPTY 0xFFFFu

static uint32_t lz_hash(const uint8_t *p, uint32_t hash_bits)
{
    /* One multiply of the first four bytes at the cursor, top bits kept:
     * the multiply's high half is where the input bits are spread out, so
     * the index depends on all four bytes and the table needs no tag --
     * the four-byte compare at the candidate is the tag check. */
    uint32_t v = zpse_load_u32le(p);

    return (v * UINT32_C(2654435761)) >> (32u - hash_bits);
}

static uint64_t lz_load64(const uint8_t *p)
{
    uint64_t v;

    memcpy(&v, p, 8u);
    return v;
}

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

uint32_t zpse_lz_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                        uint32_t cap, void *table, uint32_t hash_bits)
{
    uint16_t *tab = (uint16_t *)table;
    LzOut o;
    uint32_t pos, anchor, misses, i, cmd_off, cmd_len;
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
    while (pos + ZPSE_LZ_MIN_MATCH <= n) {
        uint32_t h = lz_hash(in + pos, hash_bits);
        uint32_t cand = tab[h];
        uint32_t len;

        tab[h] = (uint16_t)pos;
        if (cand == LZ_EMPTY || pos - cand > 65535u ||
            zpse_load_u32le(in + cand) != zpse_load_u32le(in + pos)) {
            /* Two misses are a coincidence; a run of them means the data
             * has no match here, so the cursor accelerates. */
            pos += 1u + (misses >> 5);
            misses++;
            continue;
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
        while (pos + len + 8u <= n && cand + len + 8u <= n &&
               lz_load64(in + pos + len) == lz_load64(in + cand + len)) {
            len += 8u;
        }
        while (pos + len < n && in[pos + len] == in[cand + len]) {
            len++;
        }
        lz_emit(&o, in + anchor, pos - anchor, pos - cand, len);
        if (o.fail) {
            return 0u;
        }
        pos += len;
        anchor = pos;
        misses = 0u;
    }
    /* Trailing literals: no match left to code, so offset 0. */
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
    memmove(dst + cmd_off, dst + o.cmd_hi, cmd_len);
    /* The commands were pushed backwards from the end of the buffer, so
     * the block now holds them in reverse: put them back in the order the
     * decoder reads them. (The alternative -- growing them forwards into
     * space the literal stream has not reached yet -- cannot be done
     * without a second buffer, since neither stream's length is known
     * until the block is finished.) */
    {
        uint32_t lo = cmd_off;
        uint32_t hi = cmd_off + cmd_len - 4u;

        while (lo < hi) {
            uint32_t a = zpse_load_u32le(dst + lo);
            uint32_t b = zpse_load_u32le(dst + hi);

            zpse_store_u32le(dst + lo, b);
            zpse_store_u32le(dst + hi, a);
            lo += 4u;
            hi -= 4u;
        }
    }
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
 * not. `lit_total` caps the read at the literal region and `oend` the
 * write at the output, so neither form can touch a byte the block does not
 * own. */
static void lz_copy_literals(uint8_t *dst, const uint8_t *src, uint32_t len,
                             const uint8_t *lit_end, const uint8_t *oend)
{
    if (len >= 16u) {
        do {
            memcpy(dst, src, 16u);
            dst += 16u;
            src += 16u;
            len -= 16u;
        } while (len >= 16u);
    }
    if (len != 0u) {
        if (len <= 8u && (uint32_t)(lit_end - src) >= 16u &&
            (uint32_t)(oend - dst) >= 8u) {
            memcpy(dst, src, 8u);
        } else if (len > 8u && (uint32_t)(lit_end - src) >= 16u &&
                   (uint32_t)(oend - dst) >= 16u) {
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
static void lz_copy_match(uint8_t *dst, uint32_t off, uint32_t len,
                          const uint8_t *oend)
{
    if (len <= 32u) {
        uint32_t room = (uint32_t)(oend - dst);

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
        while (len >= 32u) {
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

int zpse_lz_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                   uint32_t n)
{
    uint32_t lit_bytes, cmd_off, cmd_len;
    const uint8_t *lp;
    const uint8_t *lend;
    const uint8_t *cp;
    const uint8_t *cend;
    uint8_t *op;
    uint8_t *oend;

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
    lend = lp + lit_bytes;
    cp = in + cmd_off;
    cend = in + in_len;
    op = out;
    oend = out + n;

    while (op < oend) {
        uint32_t cw;
        uint32_t lit;
        uint32_t off;
        uint32_t mlen;

        if (cp + 4 > cend) {
            return PSE_BAD_LZ;
        }
        cw = zpse_load_u32le(cp);
        cp += 4;
        lit = (cw >> 16) & 0xFFu;
        off = cw & 0xFFFFu;
        mlen = (cw >> 24) + ZPSE_LZ_MIN_MATCH;
        if (lit > (uint32_t)(lend - lp)) {
            return PSE_BAD_LZ;
        }
        if (lit > (uint32_t)(oend - op)) {
            return PSE_BAD_LZ;
        }
        lz_copy_literals(op, lp, lit, lend, oend);
        op += lit;
        lp += lit;
        if (off == 0u) {
            /* A literal-only command has to make progress. */
            if (lit == 0u) {
                return PSE_BAD_LZ;
            }
            continue;
        }
        if (off > (uint32_t)(op - out)) {
            return PSE_BAD_LZ;
        }
        if (mlen > (uint32_t)(oend - op)) {
            return PSE_BAD_LZ;
        }
        lz_copy_match(op, off, mlen, oend);
        op += mlen;
    }
    /* Every literal and every command has to be accounted for: a stream
     * that fills the output early and then stops is not the stream the
     * encoder wrote, and silently ignoring the rest would let a corrupt
     * count pass as valid. */
    if (lp != lend || cp != cend) {
        return PSE_BAD_LZ;
    }
    return PSE_OK;
}
