#include "zpse/lzfast.h"

#include <stdint.h>
#include <string.h>

/* Direct-mapped 13-bit table: 8192 uint16_t positions, 16 KiB, the same size
 * as lz2's. 0xFFFF is the empty marker; the encoder never inserts a position
 * past n - 4, and n <= 65536, so 65535 is never a legitimate entry. */
#define LF_EMPTY  0xFFFFu
#define LF_MINM   ZPSE_LZFAST_MIN_MATCH
#define LF_HSIZE  ZPSE_LZFAST_HSIZE
#define LF_SKIPB  5u
#define LF_MAXN   ZPSE_LZFAST_MAX_BLOCK

static inline uint32_t lf_ld32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t lf_ld64(const uint8_t *p)
{
    uint64_t v;
    memcpy(&v, p, 8u);
    return v;
}

static inline uint32_t lf_ctz64(uint64_t w)
{
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctzll(w);
#else
    uint32_t n = 0u;
    while ((w & 1u) == 0u) { w >>= 1; n++; }
    return n;
#endif
}

/* One multiply-shift, the same multiplicative family lz2 uses. Hash width is
 * not in the stream, so this is a pure speed/table-size knob. */
static inline uint32_t lf_hash(uint32_t v)
{
    return (v * 2654435761u) >> (32u - ZPSE_LZFAST_HBITS);
}

/* Bytes of 255-chaining a field of value v needs. */
static inline uint32_t lf_escapes(uint32_t v)
{
    return v < 15u ? 0u : 1u + (v - 15u) / 255u;
}

/* LZ4-style command: token, escapes, literals, then a 2-byte LE offset unless
 * this is the final literal-only command. */
static inline void lf_emit(uint8_t *dst, uint32_t *out, const uint8_t *lit,
                           uint32_t litlen, uint32_t off, uint32_t mlen,
                           int last)
{
    uint8_t *p = dst + *out;
    uint32_t ml = mlen ? mlen - LF_MINM : 0u;
    uint32_t v;

    *p++ = (uint8_t)(((litlen < 15u ? litlen : 15u) << 4) |
                     (ml < 15u ? ml : 15u));
    if (litlen >= 15u) {
        v = litlen - 15u;
        while (v >= 255u) { *p++ = 255u; v -= 255u; }
        *p++ = (uint8_t)v;
    }
    if (litlen != 0u) { memcpy(p, lit, litlen); p += litlen; }
    if (!last) {
        *p++ = (uint8_t)(off & 0xFFu);
        *p++ = (uint8_t)(off >> 8u);
        if (ml >= 15u) {
            v = ml - 15u;
            while (v >= 255u) { *p++ = 255u; v -= 255u; }
            *p++ = (uint8_t)v;
        }
    }
    *out = (uint32_t)(p - dst);
}

static uint32_t lf_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                          uint32_t cap, uint16_t *tab)
{
    uint32_t pos, anchor, misses, out = 0u, cur, h, c0;

    if (in == 0 || dst == 0 || tab == 0 || n > LF_MAXN) return 0u;
    memset(tab, 0xFF, (size_t)LF_HSIZE * sizeof *tab);

    /* n == 0 is the empty stream; n < 5 cannot form a 4-byte match plus a
     * lookahead byte, so emit the whole block as one literal-only command. */
    if (n < LF_MINM + 1u) {
        if (n == 0u) return 0u;
        if (cap < 1u + n + lf_escapes(n)) return 0u;
        lf_emit(dst, &out, in, n, 0u, 0u, 1);
        return out;
    }

    pos = 0u; anchor = 0u; misses = 0u;
    cur = lf_ld32(in);
    h = lf_hash(cur);
    c0 = tab[h];

    while (pos + LF_MINM <= n) {
        uint32_t c, len, off, litlen, need;

        /* Insert the position BEFORE deciding, as lz2 does; c0 was read
         * before the insert, so every candidate is strictly behind pos. */
        tab[h] = (uint16_t)pos;

        if (c0 != LF_EMPTY && c0 < pos && lf_ld32(in + c0) == cur) {
            c = c0;
        } else {
            pos += 1u + (misses >> LF_SKIPB);
            misses++;
            if (pos + LF_MINM > n) break;
            cur = lf_ld32(in + pos);
            h = lf_hash(cur);
            c0 = tab[h];
            continue;
        }

        /* Forward extension: four bytes already match; 8-byte xor + ctz,
         * then a byte tail only for the last < 8 bytes of the block. */
        {
            uint32_t p = pos + LF_MINM, q = c + LF_MINM;
            int tail = 1;
            len = LF_MINM;
            while (p + 8u <= n) {
                uint64_t x = lf_ld64(in + p) ^ lf_ld64(in + q);
                if (x != 0u) { len += lf_ctz64(x) >> 3; tail = 0; break; }
                p += 8u; q += 8u; len += 8u;
            }
            /* A mismatch already located the first differing byte via ctz;
             * re-walking from p would count those bytes a second time. */
            if (tail != 0) {
                while (p < n && in[p] == in[q]) { p++; q++; len++; }
            }
        }
        off = pos - c;                 /* <= 65532 for a <= 64 KiB block */
        litlen = pos - anchor;

        /* Exact size of this command; no conservative over-reserve, so a
         * caller passing zpse_lzfast_bound(n) is never refused. */
        need = 1u + litlen + lf_escapes(litlen) + 2u +
               lf_escapes(len - LF_MINM);
        if (out + need > cap) return 0u;
        lf_emit(dst, &out, in + anchor, litlen, off, len, 0);
        pos += len;
        anchor = pos;
        misses = 0u;
        if (pos + LF_MINM > n) break;
        cur = lf_ld32(in + pos);
        h = lf_hash(cur);
        c0 = tab[h];
    }

    if (anchor < n) {
        uint32_t run = n - anchor;
        if (cap - out < 1u + run + lf_escapes(run)) return 0u;
        lf_emit(dst, &out, in + anchor, run, 0u, 0u, 1);
    }
    return out;
}

/* Overlapping copy for off < 8: write the first period, then step by a
 * multiple of the period so every wide store reads bytes already written. */
static inline void lf_copy_short(uint8_t *d, uint32_t off, uint32_t len)
{
    static const uint8_t rle_off[8]  = { 0u, 0u, 0u, 1u, 0u, 4u, 4u, 4u };
    static const uint8_t rle_wide[8] = { 0u, 1u, 2u, 2u, 4u, 3u, 2u, 1u };
    const uint8_t *s = d - off;
    uint32_t wide;

    if (len >= 8u) {
        wide = 8u + off - rle_wide[off];
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        memcpy(d + 4u, d - off + rle_off[off], 4u);
        d += 8u; len -= 8u;
        while (len >= 8u) { memcpy(d, d - wide, 8u); d += 8u; len -= 8u; }
    }
    while (len-- != 0u) { *d = d[-(ptrdiff_t)off]; d++; }
}

/* Fast region uses fixed-width stores; escapes, the tail and hostile streams
 * go through the careful path. */
static int lf_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                     uint32_t n)
{
    const uint8_t *ip = in, *iend = in + in_len;
    uint32_t op = 0u;

    if (in == 0 || (n != 0u && out == 0)) return -1;
    while (op < n) {
        uint32_t tok, lit, off, ml;

        if (iend - ip >= 18 && n - op >= 40u) {
            tok = *ip;
            if (((tok >> 4) != 15u) && ((tok & 15u) != 15u)) {
                ip++;
                lit = tok >> 4;
                /* Two 8-byte stores cover any literal run a token can carry;
                 * the slack is reclaimed by the next command. The offset load
                 * deliberately follows the stores: loading it from ip[1+lit]
                 * first would put a dependent load in front of them and cost
                 * the fast path its throughput. */
                memcpy(out + op, ip, 8u);
                memcpy(out + op + 8u, ip + 8u, 8u);
                ip += lit; op += lit;
                off = (uint32_t)ip[0] | ((uint32_t)ip[1] << 8u);
                ip += 2;
                ml = (tok & 15u) + LF_MINM;
                /* One compare covers 1 <= off <= op: off - 1 wraps when
                 * off == 0, and off - 1 >= op when off > op. */
                if ((uint32_t)(off - 1u) < op) {
                    if (off >= 8u) {
                        memcpy(out + op, out + op - off, 8u);
                        memcpy(out + op + 8u, out + op + 8u - off, 8u);
                        if (ml > 16u) {
                            memcpy(out + op + 16u, out + op + 16u - off, 8u);
                        }
                    } else {
                        lf_copy_short(out + op, off, ml);
                    }
                } else {
                    /* Hostile stream: the distance does not point into the
                     * output written so far. Refuse it rather than read
                     * before the buffer. */
                    return -1;
                }
                op += ml;
                continue;
            }
        }
        /* Careful path: escapes, block tail, hostile streams. */
        if (ip >= iend) return -1;
        tok = *ip++;
        lit = tok >> 4;
        if (lit == 15u) {
            uint8_t b;
            do {
                if (ip >= iend) return -1;
                b = *ip++; lit += b;
            } while (b == 255u);
        }
        if (lit > n - op || lit > (uint32_t)(iend - ip)) return -1;
        memcpy(out + op, ip, lit);
        ip += lit; op += lit;
        if (op >= n) break;                 /* final literal-only command */

        if (iend - ip < 2) return -1;
        off = (uint32_t)ip[0] | ((uint32_t)ip[1] << 8u);
        ip += 2;
        if (off == 0u || off > op) return -1;

        ml = (tok & 15u) + LF_MINM;
        if ((tok & 15u) == 15u) {
            uint8_t b;
            do {
                if (ip >= iend) return -1;
                b = *ip++; ml += b;
            } while (b == 255u);
        }
        if (ml > n - op) return -1;
        if (off >= 8u && n - op >= 8u) {
            while (ml >= 8u) {
                memcpy(out + op, out + op - off, 8u); op += 8u; ml -= 8u;
            }
        }
        while (ml-- != 0u) { out[op] = out[op - off]; op++; }
    }
    return (ip == iend && op == n) ? 0 : -1;
}

size_t zpse_lzfast_bound(size_t n)
{
    if (n == 0u) return 0u;
    return n + n / 255u + 64u;
}

size_t zpse_lzfast_encode(const uint8_t *in, size_t n,
                          uint8_t *out, size_t out_cap)
{
    uint16_t tab[LF_HSIZE];             /* per-call scratch, 16 KiB on stack */

    if (n == 0u) return 0u;             /* valid empty stream */
    if (in == 0 || out == 0 || n > LF_MAXN) return 0u;
    if (out_cap > 0xFFFFFFFFu) out_cap = 0xFFFFFFFFu;
    return (size_t)lf_encode(in, (uint32_t)n, out, (uint32_t)out_cap, tab);
}

int zpse_lzfast_decode(const uint8_t *in, size_t in_len,
                       uint8_t *out, size_t out_cap)
{
    if (in == 0 || out_cap > LF_MAXN || in_len > 0xFFFFFFFFu) return -1;
    return lf_decode(in, (uint32_t)in_len, out, (uint32_t)out_cap);
}
