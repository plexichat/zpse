#include "zpse/dense.h"

#include <string.h>

#include "zpse/bits.h"
#include "zpse/mask.h"
#include "zpse/predictor.h"
#include "zpse/residual.h"
#include "zpse/status.h"

/* Period of a STRIDE predictor (mirrors tile_stride_of in tile.c). */
static uint32_t dense_stride(uint8_t predictor_id)
{
    switch (predictor_id) {
    case ZPSE_P_STRIDE_1:
        return 1u;
    case ZPSE_P_STRIDE_2:
        return 2u;
    case ZPSE_P_STRIDE_3:
        return 3u;
    case ZPSE_P_STRIDE_4:
        return 4u;
    default:
        return 8u; /* ZPSE_P_STRIDE_8 */
    }
}

static unsigned dense_ctz(unsigned w)
{
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_ctz(w);
#else
    unsigned n = 0u;

    while ((w & 1u) == 0u) {
        w >>= 1;
        n++;
    }
    return n;
#endif
}

/* Number of set bits in a dense bitmap of `need` bytes. */
static uint32_t dense_popcount(const uint8_t *bits, uint32_t need)
{
    uint32_t n = 0u;
    uint32_t i;

    for (i = 0u; i < need; i++) {
        unsigned b = bits[i];

        while (b != 0u) {
            b &= b - 1u;
            n++;
        }
    }
    return n;
}

/* First exception position >= `from`, or nt when the mask has none left.
 * Positions only ever move forward, so this scan is entered once per
 * exception and costs O(nt/8 + K) over the whole record. */
static uint32_t dense_bitmap_next(const uint8_t *bits, uint32_t nt,
                                  uint32_t from)
{
    uint32_t p = from;

    while (p < nt) {
        uint32_t byte_i = p >> 3;
        unsigned w = (unsigned)bits[byte_i] >> (p & 7u);

        if (w != 0u) {
            return p + (uint32_t)dense_ctz(w);
        }
        p = (byte_i + 1u) << 3;
    }
    return nt;
}

/* Next exception index j >= 0 position, given the mask. */
static uint32_t dense_next(uint8_t mask_mode, const uint8_t *mask,
                           uint32_t nt, uint32_t j, uint32_t from)
{
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        return nt;
    case ZPSE_M_ALL_EXCEPTION:
        return j;
    case ZPSE_M_SPARSE_U8:
        return mask[j];
    default: /* ZPSE_M_RAW_BITMAP */
        return dense_bitmap_next(mask, nt, from);
    }
}

/* Actual byte at exception j: raw for mode 1, un-excluded for the rest. */
static uint8_t dense_exc_value(const uint8_t *resid, uint8_t res_mode,
                               uint32_t j, uint8_t xhat)
{
    if (res_mode == (uint8_t)ZPSE_R_RAW_ACTUAL) {
        return resid[j];
    }
    return zpse_exclusion_inverse(resid[j], xhat);
}

static int dense_check_mask(uint8_t mask_mode, const uint8_t *mask,
                            uint32_t nt, uint32_t k, uint32_t mask_len)
{
    switch (mask_mode) {
    case ZPSE_M_ALL_HIT:
        if (k != 0u || mask_len != 0u) {
            return PSE_BAD_MASK;
        }
        return PSE_OK;
    case ZPSE_M_ALL_EXCEPTION:
        if (k != nt || mask_len != 0u) {
            return PSE_BAD_MASK;
        }
        return PSE_OK;
    case ZPSE_M_RAW_BITMAP: {
        uint32_t need = (nt + 7u) >> 3;

        if (k < 1u || k >= nt) {
            return PSE_BAD_MASK;
        }
        if (mask_len != need) {
            return PSE_BAD_MASK;
        }
        if ((nt & 7u) != 0u && (mask[need - 1u] >> (nt & 7u)) != 0u) {
            return PSE_BAD_MASK;
        }
        if (dense_popcount(mask, need) != k) {
            return PSE_BAD_MASK;
        }
        return PSE_OK;
    }
    case ZPSE_M_SPARSE_U8: {
        uint32_t j;

        /* Positions are one byte each, so this form only exists for
         * planes whose positions all fit in a byte. */
        if (nt > 256u) {
            return PSE_BAD_MASK;
        }
        if (k < 1u || k >= nt) {
            return PSE_BAD_MASK;
        }
        if (mask_len != k) {
            return PSE_BAD_MASK;
        }
        for (j = 0u; j < k; j++) {
            if ((uint32_t)mask[j] >= nt) {
                return PSE_BAD_MASK;
            }
            if (j > 0u && mask[j] <= mask[j - 1u]) {
                return PSE_BAD_MASK;
            }
        }
        return PSE_OK;
    }
    default:
        /* Mask modes 3, 4, 5 and 7 are reserved inside a dense record. */
        return PSE_UNSUPPORTED_FEATURE;
    }
}

int zpse_dense_decode(const ZpseSegmentHeader *h, const uint8_t *payload,
                      uint32_t payload_len, uint8_t *out, uint32_t out_cap,
                      uint32_t *out_len, uint8_t *scratch)
{
    uint32_t nt;
    uint32_t k;
    uint32_t mask_len;
    uint32_t res_len;
    uint32_t rec_len;
    uint8_t flags;
    uint8_t mask_mode;
    uint8_t res_mode;
    uint8_t pred;
    uint8_t constant_c;
    const uint8_t *mask;
    const uint8_t *res;
    uint8_t last[8];
    uint8_t second[8];
    uint8_t literal;
    uint32_t p;
    uint32_t j;
    uint32_t next;
    int rc;
    unsigned l;

    if (h == 0 || payload == 0 || out == 0 || out_len == 0 || scratch == 0 ||
        h->tile_count != 1u) {
        return PSE_BAD_ARG;
    }
    nt = h->segment_decoded_size;
    if (nt == 0u || nt > ZPSE_DENSE_MAX_LEN) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (nt > out_cap) {
        return PSE_OUTPUT_OVERFLOW;
    }
    if (payload_len < ZPSE_SEGMENT_HEADER_SIZE + ZPSE_DENSE_HEADER_SIZE) {
        return PSE_BAD_TILE_SIZE;
    }

    rec_len = (uint32_t)zpse_load_u16le(payload + 16);
    flags = payload[18];
    pred = payload[19];
    constant_c = payload[20];
    mask_len = (uint32_t)zpse_load_u16le(payload + 24);
    res_len = (uint32_t)zpse_load_u16le(payload + 26);
    k = (uint32_t)zpse_load_u16le(payload + 22);
    mask_mode = (uint8_t)(flags & UINT8_C(0x07));
    res_mode = (uint8_t)((flags >> 3) & UINT8_C(0x07));

    if ((flags & ZPSE_DENSE_MARKER) == 0u) {
        return PSE_BAD_TILE_SIZE;
    }
    if ((flags & UINT8_C(0x80)) != 0u) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (payload[21] != 0u) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (rec_len != payload_len - 18u) {
        return PSE_BAD_TILE_SIZE;
    }
    if (mask_len + res_len != rec_len - 10u) {
        return PSE_BAD_TILE_SIZE;
    }
    /* Mode 4 (signed delta) is prediction-dependent, and a dense record
     * only knows a position's prediction while reconstructing it, so it
     * is reserved here and reported as an unsupported feature rather
     * than decoded wrongly. */
    if (res_mode > (uint8_t)ZPSE_R_PACKED_BASE_M) {
        return PSE_UNSUPPORTED_FEATURE;
    }
    if (pred > (uint8_t)ZPSE_P_STRIDE_3) {
        return PSE_BAD_PREDICTOR;
    }
    if (pred == (uint8_t)ZPSE_P_ORDER1 || pred == (uint8_t)ZPSE_P_ORDER2) {
        return PSE_UNSUPPORTED_FEATURE;
    }

    mask = payload + ZPSE_SEGMENT_HEADER_SIZE + ZPSE_DENSE_HEADER_SIZE;
    res = mask + mask_len;
    rc = dense_check_mask(mask_mode, mask, nt, k, mask_len);
    if (rc != PSE_OK) {
        return rc;
    }
    rc = zpse_residual_decode(res_mode, res, (uint16_t)res_len, (uint16_t)k,
                              0, scratch, scratch);
    if (rc != PSE_OK) {
        return rc;
    }

    literal = h->initial_literal;
    for (l = 0; l < 8u; l++) {
        last[l] = literal;
        second[l] = literal;
    }
    j = 0u;
    next = (k == 0u) ? nt : dense_next(mask_mode, mask, nt, 0u, 0u);

    if (pred == (uint8_t)ZPSE_P_CONST_ZERO ||
        pred == (uint8_t)ZPSE_P_TILE_CONSTANT) {
        uint8_t fill = (pred == (uint8_t)ZPSE_P_CONST_ZERO) ? 0u : constant_c;

        for (p = 0u; p < nt; p++) {
            uint8_t xhat = fill;
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k) ? dense_next(mask_mode, mask, nt, j, p + 1u) : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
        }
        /* The constant predictor reads no lane state; leave the
         * histories at their reset values, exactly as a CONST tile
         * does. */
    } else if (pred == (uint8_t)ZPSE_P_LANE_PREVIOUS) {
        for (p = 0u; p < nt; p++) {
            unsigned lane = p & 7u;
            uint8_t xhat = last[lane];
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k) ? dense_next(mask_mode, mask, nt, j, p + 1u) : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
            second[lane] = xhat;
            last[lane] = x;
        }
    } else if (pred == (uint8_t)ZPSE_P_LANE_DELTA) {
        for (p = 0u; p < nt; p++) {
            unsigned lane = p & 7u;
            uint8_t prev = last[lane];
            uint8_t xhat =
                (uint8_t)((2u * (unsigned)prev - (unsigned)second[lane]) &
                          UINT8_C(0xFF));
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k) ? dense_next(mask_mode, mask, nt, j, p + 1u) : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
            second[lane] = prev;
            last[lane] = x;
        }
    } else if (pred == (uint8_t)ZPSE_P_LANE_AVERAGE) {
        for (p = 0u; p < nt; p++) {
            unsigned lane = p & 7u;
            uint8_t prev = last[lane];
            uint8_t xhat = (uint8_t)(((unsigned)prev +
                                      (unsigned)second[lane] + 1u) >>
                                     1);
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k) ? dense_next(mask_mode, mask, nt, j, p + 1u) : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
            second[lane] = prev;
            last[lane] = x;
        }
    } else {
        uint32_t s = dense_stride(pred);

        for (p = 0u; p < nt; p++) {
            uint8_t xhat = (p < s) ? literal : out[p - s];
            uint8_t x;

            if (p == next) {
                x = dense_exc_value(scratch, res_mode, j, xhat);
                j++;
                next = (j < k) ? dense_next(mask_mode, mask, nt, j, p + 1u) : nt;
            } else {
                x = xhat;
            }
            out[p] = x;
        }
    }

    if (j != k) {
        return PSE_BAD_MASK;
    }
    *out_len = nt;
    return PSE_OK;
}
