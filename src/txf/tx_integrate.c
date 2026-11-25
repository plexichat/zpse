#include "zpse_tx.h"
#include "zpse/arena.h"
#include "zpse/constants.h"

#include "zpse/pagefast.h"
#include "zpse/mrd.h"
#include "bitshuffle16.h"
#include "bitshuffle32.h"
#include "delta264.h"
#include "wordxor64.h"

/* Measurement arms. ZPSE_TX_FORCE=12|13|14 applies one transform to every
 * eligible block. Otherwise ZPSE_TX_SIGNAL selects which existing analysis
 * signal drives the one-shot choice (1 = MRD's 16-bit delta gate, 2 = the
 * encoder's previous-block transform history, 3 = the page classifier's
 * pointer-lane / zero-count signature). */
#ifndef ZPSE_TX_FORCE
#define ZPSE_TX_FORCE 0
#endif

int zpse_tx_apply(int tid, const uint8_t *src, uint32_t n, uint8_t *dst)
{
    if (src == 0 || dst == 0 || n == 0u) {
        return 0;
    }
    switch (tid) {
    case 15: {
        if (zpse_bs16_out_len(n) != n) {
            return 0;
        }
        return zpse_bs16_forward(src, n, dst, n) == 0;
    }
    case 12: {
        uint32_t olen = 0u;
        if (zpse_bs32_output_len(n, &olen) != ZPSE_BS32_OK || olen != n) {
            return 0; /* a partial word group would change the block length */
        }
        return zpse_bs32_forward(src, n, dst, n) == ZPSE_BS32_OK;
    }
    case 13:
        return zpse_d264_forward(src, n, dst, n) == ZPSE_D264_OK;
    case 14:
        return zpse_wx64_forward(src, n, dst, n) == WX64_OK;
    default:
        return 0;
    }
}

int zpse_tx_inverse(int tid, const uint8_t *src, uint32_t n, uint8_t *dst)
{
    if (src == 0 || dst == 0 || n == 0u) {
        return 0;
    }
    switch (tid) {
    case 15:
        return zpse_bs16_inverse(src, n, dst, n) == 0;
    case 12:
        return zpse_bs32_inverse(src, n, dst, n) == ZPSE_BS32_OK;
    case 13:
        return zpse_d264_inverse(src, n, dst, n) == ZPSE_D264_OK;
    case 14:
        return zpse_wx64_inverse(src, n, dst, n) == WX64_OK;
    default:
        return 0;
    }
}

static uint16_t tx_load16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int zpse_tx_mr_probe(const uint8_t *in, uint32_t n)
{
    const uint32_t sample = 256u;
    const uint32_t skip = 256u;
    uint32_t step, i, fit = 0u, sumabs = 0u, zeros = 0u;

    if (in == 0 || n < 4096u) {
        return 0;
    }
    step = ((n - skip) / 2u) / sample;
    if (step == 0u) {
        step = 1u;
    }
    for (i = 0u; i < sample; i++) {
        uint32_t at = skip + 2u * (i * step);
        uint16_t a, b;
        int32_t d;

        if (at + 4u > n) {
            break;
        }
        a = tx_load16(in + at);
        b = tx_load16(in + at + 2u);
        d = (int32_t)(int16_t)(uint16_t)(b - a);
        if (d >= -127 && d <= 127) {
            fit++;
        }
        sumabs += (uint32_t)(d < 0 ? -d : d);
        if (d == 0) {
            zeros++;
        }
    }
    return (fit * 20u >= sample * 19u &&
            sumabs >= 4u * sample && sumabs < 14u * sample &&
            zeros * 20u >= sample) ? 1 : 0;
}

int zpse_tx_select(const void *enc, const uint8_t *in, uint32_t n)
{
    if (n <= (uint32_t)ZPSE_BLOCK_SHORT_MAX || in == 0) {
        return 0;
    }
#if ZPSE_TX_FORCE != 0
    if (((ZPSE_TX_FORCE == 12) && ((n & 31u) != 0u)) ||
        ((ZPSE_TX_FORCE == 15) && ((n & 15u) != 0u))) {
        return 0;
    }
    return (int)ZPSE_TX_FORCE;
#else
#if ZPSE_TX_SIGNAL == 1
    /* Signal 1: the 16-bit zero-heavy-delta shape the whole-frame path
     * already recognises, sampled at 256 evenly spaced points (about half a
     * KiB instead of a second full scan). */
    if ((n & 15u) != 0u || enc == 0) {
        return 0;
    }
    /* The cheap rungs cannot price the transformed stream, so it loses there. */
    if (zpse_enc_speed(((const ZpseEncoder *)enc)->enc_flags) < 5u) {
        return 0;
    }
    return zpse_tx_mr_probe(in, n) ? 15 : 0;
#elif ZPSE_TX_SIGNAL == 2
    /* Signal 2: the transform shape the previous block of this context chose.
     * O(1), already carried in the encoder context, no scan of this block. */
    {
        const ZpseEncoder *e = (const ZpseEncoder *)enc;

        if (enc == 0 || e->hist_valid == 0u) {
            return 0;
        }
        if ((e->hist_tid >= 6u && e->hist_tid <= 10u) &&
            (n & 31u) == 0u) {
            return 12;
        }
        return 0;
    }
#elif ZPSE_TX_SIGNAL == 3
    /* Signal 3: the existing page classifier's pointer-lane class plus its
     * exact zero count, read over the block's first 4 KiB. */
    {
        ZpsePageInfo pi;
        uint32_t m = (n < 4096u) ? n : 4096u;

        if (zpse_page_classify(in, m, &pi) == (int)ZPSE_PAGE_POINTER &&
            (n & 31u) == 0u && pi.nzero >= 1000u && pi.nzero <= 2500u) {
            return 12;
        }
        return 0;
    }
#else
    (void)enc;
    return 0;
#endif
#endif
}
