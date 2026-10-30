#include "zpse/combinadic.h"
#include "zpse/status.h"

#include <stddef.h>
#include <stdint.h>

/* Colexicographic (Macaulay) combinadics.
 * Ranks are 4-limb little-endian u256. Portable C11 (no __int128,
 * MSVC-safe). No malloc; no VLAs; stack use is O(1) limbs.
 *
 * The one-limb unrank below is the single exception: it needs an exact
 * 128-bit product to evaluate C(n,j) directly, and is compiled only where
 * the compiler provides one. `__extension__` keeps -Wpedantic quiet about
 * a type the standard does not name; without it the u256 path runs. */
#if defined(__SIZEOF_INT128__)
__extension__ typedef unsigned __int128 zpse_u128;
#endif

int zpse_u256_gte(ZpseU256 a, ZpseU256 b)
{
    int i;
    for (i = 3; i >= 0; i--) {
        if (a.w[i] != b.w[i])
            return a.w[i] > b.w[i];
    }
    return 1;
}

ZpseU256 zpse_u256_add(ZpseU256 a, ZpseU256 b)
{
    ZpseU256 r;
    uint64_t carry = 0;
    int i;
    for (i = 0; i < 4; i++) {
        uint64_t s = a.w[i] + carry;
        uint64_t c1 = (s < a.w[i]) ? 1u : 0u;
        uint64_t t = s + b.w[i];
        uint64_t c2 = (t < s) ? 1u : 0u;
        r.w[i] = t;
        carry = c1 | c2;
    }
    return r;
}

ZpseU256 zpse_u256_sub(ZpseU256 a, ZpseU256 b)
{
    ZpseU256 r;
    uint64_t borrow = 0;
    int i;
    for (i = 0; i < 4; i++) {
        uint64_t sub = b.w[i] + borrow;
        uint64_t c1 = (sub < b.w[i]) ? 1u : 0u;
        uint64_t t = a.w[i] - sub;
        uint64_t c2 = (a.w[i] < sub) ? 1u : 0u;
        r.w[i] = t;
        borrow = c1 | c2;
    }
    return r;
}

int zpse_u256_is_zero(ZpseU256 a)
{
    return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0u;
}

static ZpseU256 u256_zero(void)
{
    ZpseU256 z;
    z.w[0] = 0u; z.w[1] = 0u; z.w[2] = 0u; z.w[3] = 0u;
    return z;
}

static ZpseU256 u256_one(void)
{
    ZpseU256 o = u256_zero();
    o.w[0] = 1u;
    return o;
}

/* Multiply by a small factor (<= 256); wraps mod 2^256.
 * 32-bit half-limb schoolbook: no intermediate exceeds 2^41. */
static ZpseU256 u256_mul_small(ZpseU256 a, uint64_t m)
{
    ZpseU256 r;
    uint64_t carry = 0;
    int i;
    for (i = 0; i < 4; i++) {
        uint64_t alo = a.w[i] & 0xFFFFFFFFu;
        uint64_t ahi = a.w[i] >> 32;
        uint64_t c0 = carry & 0xFFFFFFFFu;
        uint64_t c1 = carry >> 32;
        uint64_t t0 = alo * m + c0;
        uint64_t t1 = ahi * m + c1 + (t0 >> 32);
        r.w[i] = (t0 & 0xFFFFFFFFu) | (t1 << 32);
        carry = t1 >> 32;
    }
    return r;
}

/* Divide by a small nonzero divisor; returns quotient, stores remainder.
 * Used only for exact divisions in choose (outside hot loops).
 * Invariant rem < d is preserved, so every 64-bit op is exact. */
static ZpseU256 u256_divmod_small(ZpseU256 a, uint64_t d, uint64_t *rem_out)
{
    ZpseU256 q;
    uint64_t rem = 0;
    int i;
    for (i = 3; i >= 0; i--) {
        uint64_t hi = a.w[i] >> 32;
        uint64_t lo = a.w[i] & 0xFFFFFFFFu;
        uint64_t cur = (rem << 32) | hi; /* rem < d <= 2^32: fits */
        uint64_t qhi = cur / d;
        uint64_t r1 = cur % d;
        uint64_t cur2 = (r1 << 32) | lo;
        uint64_t qlo = cur2 / d;
        rem = cur2 % d;
        q.w[i] = (qhi << 32) | qlo;
    }
    if (rem_out)
        *rem_out = rem;
    return q;
}

static uint64_t u256_mod_small(ZpseU256 a, uint64_t d)
{
    uint64_t rem = 0;
    int i;
    /* Requires d >= 1 (caller guarantees d <= 256 here, so
     * (rem<<32)|half never overflows). */
    for (i = 3; i >= 0; i--) {
        uint64_t hi = a.w[i] >> 32;
        uint64_t lo = a.w[i] & 0xFFFFFFFFu;
        rem = ((rem << 32) | hi) % d;
        rem = ((rem << 32) | lo) % d;
    }
    return rem;
}

static int u256_bitlen(ZpseU256 a)
{
    int i;
    for (i = 3; i >= 0; i--) {
        if (a.w[i]) {
            uint64_t x = a.w[i];
            int b = 0;
            while (x) {
                b++;
                x >>= 1;
            }
            return i * 64 + b;
        }
    }
    return 0;
}

static uint64_t u64_gcd(uint64_t a, uint64_t b)
{
    while (b) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

ZpseU256 zpse_comb_choose_u256(uint16_t n, uint16_t k)
{
    uint16_t kk, i;

    if (k > n)
        return u256_zero();
    kk = (k > (uint16_t)(n - k)) ? (uint16_t)(n - k) : k;
    if (kk == 0)
        return u256_one();

    /* Fast path: 64-bit multiplicative recurrence with gcd reduction.
     * acc_i = C(n-kk+i, i) is non-decreasing in i, so if the final
     * value fits in 64 bits every step does too. Each step computes
     * acc*num/den exactly: after dividing den by gcd(acc,den) and
     * gcd(num,den), the residual denominator is coprime to the
     * product yet still divides it, hence equals 1. */
    {
        uint64_t acc = 1u;
        int overflow = 0;
        for (i = 1; i <= kk; i++) {
            uint64_t num = (uint64_t)(n - kk + i);
            uint64_t den = (uint64_t)i;
            uint64_t g;
            g = u64_gcd(acc, den);
            acc /= g;
            den /= g;
            g = u64_gcd(num, den);
            num /= g;
            den /= g;
            if (den != 1u) {
                overflow = 1; /* unreachable for exact recurrence */
                break;
            }
            if (acc > (uint64_t)0xFFFFFFFFFFFFFFFFu / num) {
                overflow = 1;
                break;
            }
            acc *= num;
        }
        if (!overflow) {
            ZpseU256 r = u256_zero();
            r.w[0] = acc;
            return r;
        }
    }

    /* Fallback for values >= 2^64: same recurrence in u256. The raw
     * product r*num can exceed 2^256 even when the quotient fits, so
     * the denominator is cancelled BEFORE multiplying (first against
     * r, then against num); the residual denominator is then
     * provably 1 and the multiply cannot overflow. */
    {
        ZpseU256 r = u256_one();
        for (i = 1; i <= kk; i++) {
            uint64_t num = (uint64_t)(n - kk + i);
            uint64_t den = (uint64_t)i;
            uint64_t g, rem = 0;
            g = u64_gcd(u256_mod_small(r, den), den);
            if (g > 1u) {
                r = u256_divmod_small(r, g, &rem);
                if (rem != 0u)
                    return u256_zero(); /* unreachable */
                den /= g;
            }
            g = u64_gcd(num, den);
            num /= g;
            den /= g;
            if (den != 1u)
                return u256_zero(); /* unreachable for exact recurrence */
            r = u256_mul_small(r, num);
        }
        return r;
    }
}

int zpse_comb_payload_len(uint16_t nt, uint16_t k, int *wide_out)
{
    ZpseU256 c, cm1;
    int bits;

    if (wide_out)
        *wide_out = 0;
    if (nt < 1u || nt > 256u || k > nt)
        return PSE_BAD_ARG;
    c = zpse_comb_choose_u256(nt, k);
    if (c.w[1] | c.w[2] | c.w[3]) {
        if (wide_out)
            *wide_out = 1;
    }
    if (c.w[0] == 1u && c.w[1] == 0u && c.w[2] == 0u && c.w[3] == 0u)
        return 0; /* C == 1: k == 0 or k == nt; zero-length rank field */
    /* Bytes L with 256^L >= C, i.e. ceil(bitlen(C-1)/8). */
    cm1 = zpse_u256_sub(c, u256_one());
    bits = u256_bitlen(cm1);
    return (bits + 7) / 8;
}

ZpseU256 zpse_comb_rank(const uint16_t *c, uint16_t k)
{
    ZpseU256 rank = u256_zero();
    uint16_t t;
    /* No error channel on value return: caller MUST NOT pass NULL unless
     * k == 0, and MUST supply strictly ascending c[t] (subset order is a
     * precondition; rank_to_bytes enforces it where a code exists). */
    if (c == NULL)
        return rank;
    for (t = 0; t < k; t++) {
        /* j = t+1 is 1-based (rank = SUM C(c_j, j)). Cold
         * (encode-side) path: exact single-value recurrence; the
         * division-free MUST in Sec. 8.2 governs the hot unrank loop. */
        ZpseU256 term = zpse_comb_choose_u256(c[t], (uint16_t)(t + 1u));
        rank = zpse_u256_add(rank, term);
    }
    return rank;
}

int zpse_comb_rank_to_bytes(const uint16_t *c, uint16_t k, uint16_t nt,
                            uint8_t *payload_out, uint16_t payload_cap)
{
    ZpseU256 rank;
    int L;
    int byte_i;
    uint16_t t;

    if (c == NULL && k != 0)
        return PSE_BAD_ARG;
    if (nt < 1u || nt > 256u || k > nt)
        return PSE_BAD_ARG;
    /* Encoder-side subset validation: strictly ascending positions < nt.
     * (The decoder side re-verifies via the R < C(nt,k) pre-check, so an
     * invalid subset can never decode successfully.) */
    for (t = 0; t < k; t++) {
        if (c[t] >= nt)
            return PSE_BAD_ARG;
        if (t > 0u && c[t] <= c[t - 1u])
            return PSE_BAD_ARG;
    }
    L = zpse_comb_payload_len(nt, k, NULL);
    if (L < 0)
        return L;
    if (L > (int)payload_cap)
        return PSE_OUTPUT_OVERFLOW;
    if (L > 0 && payload_out == NULL)
        return PSE_BAD_ARG;
    rank = zpse_comb_rank(c, k);
    /* Minimal LE serialization of rank, zero-padded to exactly L bytes
     * (rank < C(nt,k) <= 256^L, so it always fits). */
    for (byte_i = 0; byte_i < L; byte_i++) {
        int li = byte_i >> 3;
        payload_out[byte_i] = (uint8_t)(rank.w[li] >> ((byte_i & 7) * 8));
    }
    return L;
}

int zpse_comb_unrank(uint16_t nt, uint16_t k, const uint8_t *payload,
                     uint16_t payload_len, uint32_t words[8],
                     uint8_t *comb_scratch)
{
    ZpseU256 cnk, r;
    int L;
    uint16_t t;
    int x;
    uint16_t take;

    if (payload == NULL || words == NULL)
        return PSE_BAD_ARG;
    if (nt < 1u || nt > 256u || k < 1u || k > (uint16_t)(nt - 1u))
        return PSE_BAD_ARG;
    if (payload_len < 1u || payload_len > 32u)
        return PSE_BAD_ENUMERATIVE_RANK;

    /* C(nt,k) is evaluated ONCE here and answers both of the questions
     * this function asks of it: the minimal payload length the format
     * requires, and the range pre-check R < C(nt,k) further down. It used
     * to be evaluated twice -- once inside zpse_comb_payload_len and again
     * for the pre-check -- and above 64 bits that evaluation is the whole
     * Pascal-row construction (nt-1 rounds of k u256 additions), so an
     * enumerative tile ranked in u256 paid for it twice. k is in
     * [1, nt-1] here, so C >= nt >= 2 and L is ceil(bitlen(C-1)/8) exactly
     * as zpse_comb_payload_len computes it, including the L >= 1 that the
     * caller has already required of payload_len. */
    cnk = zpse_comb_choose_u256(nt, k);
    {
        ZpseU256 cm1 = zpse_u256_sub(cnk, u256_one());

        L = (u256_bitlen(cm1) + 7) / 8;
    }
    if ((int)payload_len != L)
        return PSE_BAD_ENUMERATIVE_RANK;

    /* Payload is LE rank R. */
    r = u256_zero();
    for (take = 0; take < payload_len; take++) {
        int li = take >> 3;
        r.w[li] |= (uint64_t)payload[take] << ((take & 7u) * 8u);
    }

    /* Range pre-check BEFORE unranking: R < C(nt,k), from the value
     * computed at the top. */
    if (zpse_u256_gte(r, cnk))
        return PSE_BAD_ENUMERATIVE_RANK;

    /* The function owns the output: fully written, no caller zeroing
     * required. */
    for (take = 0; take < 8u; take++)
        words[take] = 0u;

#if defined(__SIZEOF_INT128__)
    /* ---- One-limb path: ranks that fit in 64 bits -------------------- *
     * C(nt,k) < 2^64 means every C(x,j) the descent can read is below
     * 2^64 as well, so the Pascal row fits u64 and the initial row
     * C(nt-1, 0..k) can be evaluated directly, in k steps, instead of
     * being grown from C(0,*) with nt-1 rounds of k u256 additions.
     *
     * That growth was the entire cost of a mode-2 mask with a handful of
     * exceptions -- 255 rounds, four limbs each, on every sparse tile --
     * which is the shape a sparse, sawtooth or stride-3 plane emits on
     * every tile. Three further savings, none of which changes a
     * decision:
     *
     *  - the row is only read at j <= t, and t only ever decreases, so
     *    the per-step update stops at the current t instead of running
     *    to k (the entries above t are dead);
     *  - t == 1 is closed form. The scan selects the first x (descending)
     *    with C(x,1) == x <= R, which is x == min(x_cur, R); that is the
     *    last selection, so the remaining descent is skipped;
     *  - the row never leaves u64.
     *
     * The u256 path below is unchanged and still handles every rank that
     * does not fit, including every k above 10 on a full tile. */
    if (k <= 32u && cnk.w[1] == 0u && cnk.w[2] == 0u && cnk.w[3] == 0u) {
        uint64_t row[33];
        uint64_t r64 = r.w[0];
        uint16_t tt = k;
        int j;
        int x;

        row[0] = 1u;
        for (j = 1; j <= (int)k; j++) {
            zpse_u128 prod = (zpse_u128)row[j - 1] *
                             (zpse_u128)((uint32_t)nt - (uint32_t)j);

            row[j] = (uint64_t)(prod / (uint32_t)j);
        }
        for (x = (int)nt - 1; x >= 0 && tt > 0u; x--) {
            if (tt == 1u) {
                unsigned pick = (r64 < (uint64_t)x) ? (unsigned)r64
                                                    : (unsigned)x;

                words[pick >> 5u] |= (uint32_t)1u << (pick & 31u);
                r64 -= (uint64_t)pick;
                tt = 0u;
                break;
            }
            if (r64 >= row[tt]) {
                r64 -= row[tt];
                words[(unsigned)x >> 5u] |= (uint32_t)1u << ((unsigned)x & 31u);
                tt--;
            }
            /* Step the row from C(x, *) down to C(x-1, *):
             * C(x-1,j) = C(x,j) - C(x-1,j-1), so the subtrahend is the
             * entry this same pass has already stepped -- ascending j,
             * which is the order the reference row update uses. */
            for (j = 1; j <= (int)tt; j++)
                row[j] -= row[j - 1];
        }
        if (tt != 0u || r64 != 0u)
            return PSE_BAD_ENUMERATIVE_RANK;
        return PSE_OK;
    }
#endif /* __SIZEOF_INT128__ */

    if (comb_scratch != NULL) {
        /* Division-free hot path): one Pascal
         * row C(x, 0..k) in scratch ((k+1) u256 <= 256*32 = 8 KiB),
         * maintained with u256 adds ascending and u256 subs descending.
         * Decisions are identical to the reference descending scan. */
        ZpseU256 *row = (ZpseU256 *)comb_scratch;
        uint16_t kk = k;
        int c, j;

        for (j = 0; j <= (int)kk; j++)
            row[j] = u256_zero();
        row[0] = u256_one();
        /* Phase 1: build row C(nt-1, *) ascending: C(c+1,j) =
         * C(c,j) + C(c,j-1), in-place descending j. The seed row
         * [1,0,...] is C(0,*), so nt-1 advances reach C(nt-1,*). */
        for (c = 0; c < (int)nt - 1; c++) {
            for (j = (int)kk; j >= 1; j--)
                row[j] = zpse_u256_add(row[j], row[j - 1]);
        }
        /* Phase 2: descend x = nt-1..0; row holds C(x, *). Step down via
         * C(x-1,j) = C(x,j) - C(x-1,j-1), ascending j (subtrahend already
         * stepped, hence always <= minuend: no underflow). */
        t = k;
        for (x = (int)nt - 1; x >= 0; x--) {
            if (t == 0u)
                break;
            /* row[k] is C(x,t): select x iff C(x,t) <= R. */
            if (zpse_u256_gte(r, row[t])) {
                r = zpse_u256_sub(r, row[t]);
                words[(unsigned)x >> 5] |= (uint32_t)1u << ((unsigned)x & 31u);
                t--;
            }
            for (j = 1; j <= (int)kk; j++)
                row[j] = zpse_u256_sub(row[j], row[j - 1]);
        }
        if (t != 0u)
            return PSE_BAD_ENUMERATIVE_RANK; /* ran out mid-loop */
        if (!zpse_u256_is_zero(r))
            return PSE_BAD_ENUMERATIVE_RANK; /* remainder must be zero */
        return PSE_OK;
    }

    /* Degraded path (scratch unavailable): same decisions via on-demand
     * exact choose. Functionally identical; uses division and SHOULD be
     * avoided in production (always pass the 8 KiB arena scratch). */
    t = k;
    for (x = (int)nt - 1; x >= 0; x--) {
        ZpseU256 cx;
        if (t == 0u)
            break;
        cx = zpse_comb_choose_u256((uint16_t)x, t);
        /* C(x,t) == 0 for x < t is <= any R; selecting then just fills
         * the lowest remaining positions without changing R. */
        if (zpse_u256_gte(r, cx)) {
            r = zpse_u256_sub(r, cx);
            words[(unsigned)x >> 5] |= (uint32_t)1u << ((unsigned)x & 31u);
            t--;
        }
    }
    if (t != 0u)
        return PSE_BAD_ENUMERATIVE_RANK; /* current_pos ran out mid-loop */
    if (!zpse_u256_is_zero(r))
        return PSE_BAD_ENUMERATIVE_RANK; /* final remainder must be zero */
    return PSE_OK;
}
