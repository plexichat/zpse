/* Self-contained RLE fast path; see include/zpse/rlefast.h for the wire
 * format, the gate and the return-code contract. Nothing here is shared
 * with any other zpse translation unit. */

#include "zpse/rlefast.h"

#include <stddef.h>
#include <string.h>

/* Thread-local staging for the first 64 KiB of payload. A local array this
 * large makes GCC emit __chkstk (a page-by-page stack probe) on every call,
 * which costs more than the scan the fusion saves. TLS has no probe and keeps
 * the encoder thread-safe; the buffer is scratch, fully overwritten each call. */
static _Thread_local uint8_t rlefast_staged[2u * ZPSE_RLEFAST_SAMPLE];

static void rlefast_store32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static uint32_t rlefast_load32(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* CRC-32C, reflected form: poly 0x82F63B78, init/final 0xFFFFFFFF. Only the
 * eight header bytes are covered, so the bit-at-a-time loop is plenty. */
static uint32_t rlefast_crc32c(const uint8_t *p, uint32_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    uint32_t i;
    for (i = 0; i < len; ++i) {
        int k;
        crc ^= (uint32_t)p[i];
        for (k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ ((crc & 1u) ? 0x82F63B78u : 0u);
    }
    return crc ^ 0xFFFFFFFFu;
}

uint32_t zpse_rlefast_bound(uint32_t n)
{
    uint64_t b = (uint64_t)ZPSE_RLEFAST_HEADER_SIZE + 2u * (uint64_t)n;
    return (b > (uint64_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)b;
}

int zpse_rlefast_encode(const uint8_t *in, uint32_t n,
                        uint8_t *out, uint32_t out_cap, uint32_t *out_len)
{
    /* Fused gate + emit. The old code walked the first 64 KiB twice: once to
     * score the 90% gate, then again (as part of the full emit walk) to find
     * the same runs. One walk now finds each run once and folds its gate
     * contribution into `good`. A run is emitted as canonical (v,count) pairs
     * in source order; a pair is staged while its source bytes are still
     * inside the sample and written straight to `out` afterwards. Because a
     * straddling pair is staged whole, the pair sequence -- and hence every
     * accepted frame -- is byte-identical to the two-walk version.
     *
     * Nothing is written to `out` until the gate verdict is known, preserving
     * the "decline leaves out completely untouched" contract: at most 2 bytes
     * per sampled byte are staged in thread-local scratch. */
    uint32_t staged_len = 0;
    uint32_t good = 0;
    uint32_t pos = 0;
    uint32_t m = (n < ZPSE_RLEFAST_SAMPLE) ? n : ZPSE_RLEFAST_SAMPLE;
    uint32_t i;
    int gate_done = 0;

    if (out_len == NULL)
        return 0;
    *out_len = 0;
    if (out == NULL || (n != 0u && in == NULL))
        return 0;
    if (n > (UINT32_MAX - ZPSE_RLEFAST_HEADER_SIZE) / 2u)
        return 0;
    /* Reject a too-small cap before writing anything, so a decline leaves
     * out completely untouched (including any caller canary past the cap). */
    if ((uint64_t)out_cap < (uint64_t)zpse_rlefast_bound(n))
        return 0;

    i = 0;
    while (i < n) {
        const uint8_t v = in[i];
        uint32_t run = 1;
        uint32_t left;
        uint32_t cur;

        while (i + run < n && in[i + run] == v)
            ++run;

        /* Gate contribution. The old gate clipped its run at the sample end;
         * clip the full-input run the same way so `good` -- and the 90%
         * verdict -- are identical, including for a run straddling m. */
        if (i < m) {
            uint32_t grun = run;
            if (grun > m - i)
                grun = m - i;
            if (v <= 127u && grun >= ZPSE_RLEFAST_MIN_RUN)
                good += grun;
        }

        /* Emit this run as its canonical sequence of pairs. `left` bytes of
         * value v remain; cur is the source position of the next pair. While
         * the gate is pending, pairs go to the staging buffer; once it has
         * passed, the loop below is the original tight emit loop (no per-pair
         * gate branch). */
        left = run;
        cur = i;
        if (!gate_done) {
            while (left != 0u && cur < m) {
                uint32_t take = (left > ZPSE_RLEFAST_ESCAPE)
                                    ? ZPSE_RLEFAST_ESCAPE : left;
                rlefast_staged[staged_len++] = v;
                rlefast_staged[staged_len++] = (left > ZPSE_RLEFAST_ESCAPE)
                                           ? (uint8_t)ZPSE_RLEFAST_ESCAPE
                                           : (uint8_t)(take - 1u);
                cur += take;
                left -= take;
            }
            if (left != 0u) {
                /* The sample is complete: decide before committing anything. */
                if ((uint64_t)good * 10u < (uint64_t)m * 9u)
                    return 0;
                rlefast_store32(out, ZPSE_RLEFAST_MAGIC);
                rlefast_store32(out + 4, n);
                rlefast_store32(out + 8, rlefast_crc32c(out, 8u));
                memcpy(out + ZPSE_RLEFAST_HEADER_SIZE, rlefast_staged, staged_len);
                pos = ZPSE_RLEFAST_HEADER_SIZE + staged_len;
                gate_done = 1;
            }
        }
        while (left != 0u) {
            uint32_t take = (left > ZPSE_RLEFAST_ESCAPE)
                                ? ZPSE_RLEFAST_ESCAPE : left;
            if (pos > out_cap - 2u)
                return 0;
            out[pos++] = v;
            out[pos++] = (left > ZPSE_RLEFAST_ESCAPE)
                             ? (uint8_t)ZPSE_RLEFAST_ESCAPE
                             : (uint8_t)(take - 1u);
            left -= take;
        }
        i += run;
    }

    if (!gate_done) {
        /* n <= m: the whole input was the sample; decide at the end. */
        if ((uint64_t)good * 10u < (uint64_t)m * 9u)
            return 0;
        rlefast_store32(out, ZPSE_RLEFAST_MAGIC);
        rlefast_store32(out + 4, n);
        rlefast_store32(out + 8, rlefast_crc32c(out, 8u));
        memcpy(out + ZPSE_RLEFAST_HEADER_SIZE, rlefast_staged, staged_len);
        pos = ZPSE_RLEFAST_HEADER_SIZE + staged_len;
    }

    *out_len = pos;
    return 1;
}

int zpse_rlefast_decode(const uint8_t *in, uint32_t in_len,
                        uint8_t *out, uint32_t n)
{
    uint32_t i, o;
    uint8_t pending_value = 0;
    int pending = 0;

    if (in == NULL || (n != 0u && out == NULL))
        return -1;
    if (in_len < ZPSE_RLEFAST_HEADER_SIZE)
        return -1;
    if (rlefast_load32(in) != ZPSE_RLEFAST_MAGIC)
        return -1;
    if (rlefast_load32(in + 4) != n)
        return -1;
    if (rlefast_load32(in + 8) != rlefast_crc32c(in, 8u))
        return -1;

    /* The loop reads a two-byte pair at i, and in_len was checked against the
     * (8-byte) header above. Make the in_len >= 2 invariant local so the
     * subtraction below cannot underflow on a short/truncated payload. */
    if (in_len < 2u)
        return -1;

    i = ZPSE_RLEFAST_HEADER_SIZE;
    o = 0;
    while (o < n) {
        uint8_t v, b;
        uint32_t count;
        if (i > in_len - 2u)
            return -1;
        v = in[i++];
        b = in[i++];
        /* An escape pair continues the previous run; its value must match. */
        if (pending && v != pending_value)
            return -1;
        count = (uint32_t)b + 1u;       /* b == 255 => 256, clamped below */
        if (b == ZPSE_RLEFAST_ESCAPE) {
            count = ZPSE_RLEFAST_ESCAPE;
            pending = 1;
        } else {
            pending = 0;
        }
        if (count > n - o)
            return -1;
        memset(out + o, (int)v, (size_t)count);
        o += count;
        pending_value = v;
    }
    /* The encoder always ends a run with a non-escape pair, and n is the
     * exact stop condition: no continuation may be left open, no payload
     * byte may remain unread. */
    if (pending || i != in_len)
        return -1;
    return 0;
}

