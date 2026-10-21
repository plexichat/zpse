#include "zpse/arena.h"
#include "zpse/status.h"
#include <string.h>

int zpse_arena_is_aligned(const void *p)
{
    if (p == 0) {
        return 0;
    }
    return ((uintptr_t)p % (uintptr_t)ZPSE_ARENA_ALIGN) == 0 ? 1 : 0;
}

int zpse_decoder_init(ZpseDecoder *d, void *arena_mem, size_t arena_len)
{
    uint8_t *base;

    if (d == 0 || arena_mem == 0) {
        return PSE_BAD_ARG;
    }
    if (arena_len < (size_t)ZPSE_DEC_ARENA_SIZE) {
        return PSE_BAD_ARG;
    }
    if (!zpse_arena_is_aligned(arena_mem)) {
        return PSE_BAD_ARG;
    }
    base = (uint8_t *)arena_mem;
    /* No arena-wide clear. A 256 KiB memset on every block call was
     * pure overhead: every region that can be read before it is written
     * is re-initialised at the point of use instead --
     *   recon      fully written by the per-segment decodes before
     *              zpse_transform_inverse reads it (SUM(seg_len) == N)
     *   order1/o2            reset by zpse_lane_reset() at every segment
     *              start, and by the selective per-tile reset
     *   comb       zeroed by zpse_comb_unrank() before it is read
     *   out_stage  written before it is read: block decode puts the last
     *              eight planes of a 16-wide interleave here, skewed (see
     *              ZPSE_D_PLANE_SKEW), and the inverse transform reads
     *              them back; no other block shape touches it
     * The control-cushion region is reserved and never read. */
    d->arena = base;
    d->recon = base + ZPSE_D_OFF_RECON;
    d->out_stage = base + ZPSE_D_OFF_OUT;
    d->order1 = base + ZPSE_D_OFF_ORDER1;
    d->o2 = base + ZPSE_D_OFF_O2;
    d->comb = base + ZPSE_D_OFF_COMB;
    d->shuf = base + ZPSE_D_OFF_SHUF;
    d->seg_const = 0u;
    d->seg_const_byte = 0u;
    return PSE_OK;
}

int zpse_encoder_init(ZpseEncoder *e, void *arena_mem, size_t arena_len)
{
    uint8_t *base;

    if (e == 0 || arena_mem == 0) {
        return PSE_BAD_ARG;
    }
    if (arena_len < (size_t)ZPSE_ENC_ARENA_SIZE) {
        return PSE_BAD_ARG;
    }
    if (!zpse_arena_is_aligned(arena_mem)) {
        return PSE_BAD_ARG;
    }
    base = (uint8_t *)arena_mem;
    /* See zpse_decoder_init: same reasoning, and each encoder region is
     * written before it is read (tstage by the forward transform, cand
     * by the trial, best by the winning memcpy, input/hist by the tile
     * search, comb by zpse_comb_unrank). */
    e->arena = base;
    e->input = base + ZPSE_E_OFF_INPUT;
    e->tstage = base + ZPSE_E_OFF_TSTAGE;
    e->cand = base + ZPSE_E_OFF_CAND;
    e->best = base + ZPSE_E_OFF_BEST;
    e->order1 = base + ZPSE_E_OFF_ORDER1;
    e->o2 = base + ZPSE_E_OFF_O2;
    e->comb = base + ZPSE_E_OFF_COMB;
    e->hist = base + ZPSE_E_OFF_HIST;
    e->shuf = base + ZPSE_E_OFF_SHUF;
    e->lz = base + ZPSE_E_OFF_LZ;
    e->enc_flags = 0u;
    e->flat_covered = 0u;
    e->flat_paid = 0u;
    e->const_plane = 0;
    e->ascii_gate = 0u;
    return PSE_OK;
}
