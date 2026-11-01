#include "zpse/lz2.h"

#include "zpse/bits.h"
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

/* One command: the literal run before the match, the match length and the
 * distance. `off == 0` writes a literal-only command, which is how a run
 * longer than one token and a trailing run are coded. Returns 0 when the
 * command does not fit in cap. */
static int lz2_put(uint8_t *dst, uint32_t cap, uint32_t *out,
                   const uint8_t *lit, uint32_t run, uint32_t off,
                   uint32_t len, uint32_t *lit_total, uint32_t *cmds)
{
    uint32_t ml = len >= ZPSE_LZ2_MIN_MATCH ? len - ZPSE_LZ2_MIN_MATCH : 0u;
    uint32_t p = *out;
    uint8_t *t;

    if (LZ2_CMD_ROOM > cap - p) {
        uint32_t need = 1u + run + 2u + (run >= 15u ? 1u : 0u) +
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
    /* The distance, little endian, both bytes always: a fixed-width field is
     * what makes this form cheap to read back. */
    *t++ = (uint8_t)(off & 0xFFu);
    *t++ = (uint8_t)((off >> 8) & 0xFFu);
    if (ml >= 15u) {
        *t++ = (uint8_t)(ml - 15u);
    }
    *out = (uint32_t)(t - dst);
    (*cmds)++;
    return 1;
}

uint32_t zpse_lz2_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits)
{
    uint16_t *tab = (uint16_t *)table;
    uint32_t entries = 1u << hash_bits;
    uint32_t pos, anchor, misses, i;
    uint32_t lit_total = 0u, cmds = 0u, out = LZ2_HDR;
    uint32_t cur, h, limit;

    if (in == 0 || dst == 0 || tab == 0) {
        return 0u;
    }
    if (n < ZPSE_LZ2_MIN_MATCH + 1u || n > 65536u) {
        return 0u;
    }
    if (cap < ZPSE_LZ2_MIN_PAYLOAD) {
        return 0u;
    }
    for (i = 0; i < entries; i++) {
        tab[i] = (uint16_t)LZ2_EMPTY;
    }
    pos = 0u;
    anchor = 0u;
    misses = 0u;
    limit = n - ZPSE_LZ2_MIN_MATCH;
    cur = zpse_load_u32le(in + pos);
    h = lz2_hash(cur, hash_bits);

    while (pos <= limit) {
        uint32_t cand = tab[h];
        uint32_t len = ZPSE_LZ2_MIN_MATCH;

        tab[h] = (uint16_t)pos;
        if (__builtin_expect(cand == LZ2_EMPTY ||
                                 zpse_load_u32le(in + cand) != cur,
                             1)) {
            /* Two misses are a coincidence; a run of them means the data has
             * no match here, so the cursor accelerates. */
            pos += 1u + (misses >> 5);
            misses++;
            if (pos > limit) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lz2_hash(cur, hash_bits);
            continue;
        }
        /* Extend forward, eight bytes at a time. This is the only work a
         * match does: no lookahead at pos+1, no walk back into the pending
         * literals, and no distance cache to update. */
        while (pos + len + 8u <= n && cand + len + 8u <= n &&
               lz2_load64(in + pos + len) == lz2_load64(in + cand + len)) {
            len += 8u;
        }
        /* The tail is at most seven bytes, and a byte loop over it is the
         * most executed code in the finder on prose, where the average match
         * is 5.4 bytes: two loads, an xor and a count of trailing zeros give
         * the same length the loop would count, with no branch per byte. The
         * loop's exit condition guarantees the eight bytes are readable here
         * and that they differ, so the xor is never zero. The byte loop stays
         * for the end of the block, where eight bytes are not readable. */
        if (pos + len + 8u <= n && cand + len + 8u <= n) {
            uint64_t x = lz2_load64(in + pos + len) ^
                         lz2_load64(in + cand + len);

            len += (uint32_t)(__builtin_ctzll(x) >> 3);
        } else {
            while (pos + len < n && in[pos + len] == in[cand + len]) {
                len++;
            }
        }
        {
            uint32_t run = pos - anchor;
            const uint8_t *ls = in + anchor;
            uint32_t off = pos - cand;
            uint32_t rem = len;

            /* The common command, written in place: a literal run and a match
             * that both fit their nibbles, so the whole command is a token,
             * the run and two distance bytes. On prose that is four commands
             * in five, and the call plus its room arithmetic was a fifth of
             * the writer's cost. The bytes are exactly lz2_put's, escapes and
             * all, which is what keeps the stream identical -- `rem` is left
             * at zero so the general path below does not emit a second
             * command for the same match. */
            if (run < 15u && rem < 15u + ZPSE_LZ2_MIN_MATCH &&
                cap - out >= run + 3u) {
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
                dst[out + 1u + run] = (uint8_t)(off & 0xFFu);
                dst[out + 2u + run] = (uint8_t)((off >> 8) & 0xFFu);
                out += run + 3u;
                lit_total += run;
                cmds++;
                run = 0u;
                rem = 0u;
            }
            while (run > ZPSE_LZ2_MAX_LIT) {
                if (!lz2_put(dst, cap, &out, ls, ZPSE_LZ2_MAX_LIT, 0u, 0u,
                             &lit_total, &cmds)) {
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
                if (!lz2_put(dst, cap, &out, ls, run, off, take, &lit_total,
                             &cmds)) {
                    return 0u;
                }
                ls += run;
                run = 0u;
                rem -= take;
            }
            if (rem != 0u &&
                !lz2_put(dst, cap, &out, ls, run, off, rem, &lit_total,
                         &cmds)) {
                return 0u;
            }
        }
        pos += len;
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
            if (!lz2_put(dst, cap, &out, ls, ZPSE_LZ2_MAX_LIT, 0u, 0u,
                         &lit_total, &cmds)) {
                return 0u;
            }
            ls += ZPSE_LZ2_MAX_LIT;
            run -= ZPSE_LZ2_MAX_LIT;
        }
        if (!lz2_put(dst, cap, &out, ls, run, 0u, 0u, &lit_total, &cmds)) {
            return 0u;
        }
    }
    zpse_store_u32le(dst, lit_total);
    zpse_store_u32le(dst + 4u, cmds);
    return out;
}

int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t n)
{
    uint32_t lit_total, cmd_count, cmd_left, lit_seen = 0u, op = 0u;
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
        if ((uint32_t)(tend - lp) < 2u) {
            return PSE_BAD_LZ;
        }
        off = (uint32_t)lp[0] | ((uint32_t)lp[1] << 8u);
        lp += 2;
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
        {
            uint8_t *d = out + op;
            uint32_t left = mlen;

            if (off >= 16u && mlen <= 16u && op + 16u <= n) {
                /* A short match at a long distance is one store. It is only
                 * taken when all sixteen bytes land inside this block: a
                 * frame decoder hands a block the exact room its bytes need
                 * and decodes blocks side by side, so a store past the end
                 * would land in a neighbour's output. */
                memcpy(d, d - off, 16u);
                left = 0u;
            } else if (off >= 32u && mlen <= 32u && op + 32u <= n) {
                /* The next shape up is two stores rather than a four-step
                 * eight-byte loop. A distance of 32 is what keeps the second
                 * store's source clear of the first store's destination, so
                 * the two cannot feed each other. */
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
        op += mlen;
    }
    if (cmd_left != 0u || lit_seen != lit_total || lp != tend) {
        return PSE_BAD_LZ;
    }
    return PSE_OK;
}
