/* Dense segment records (SPEC Sec. 6.3): the decoder must reconstruct a
 * plane from ONE predictor, ONE mask and ONE residual stream, and must
 * reject every malformed shape with an error rather than a decode.
 *
 * No emitter ships in this change, so the records here are built by the
 * test's own reference writer (predictions, exceptions, mask, residuals)
 * and then decoded through the real segment decoder. The lane-history
 * expectations are also written out by hand for two tiny planes, so a
 * formula that is wrong in BOTH the writer and the decoder still fails.
 *
 * Exits 0 with "DENSE:" summary lines, 1 on any failed check. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zpse/arena.h"
#include "zpse/dense.h"
#include "zpse/mask.h"
#include "zpse/predictor.h"
#include "zpse/residual.h"
#include "zpse/segment.h"
#include "zpse/status.h"

static unsigned char DARENA[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t seg[2 * ZPSE_DENSE_MAX_LEN];
static uint8_t out[ZPSE_DENSE_MAX_LEN];
static uint8_t xhat[ZPSE_DENSE_MAX_LEN];
static uint8_t exc[ZPSE_DENSE_MAX_LEN];
static uint8_t maskp[ZPSE_DENSE_MAX_LEN];
static uint8_t resp[ZPSE_DENSE_MAX_LEN];

static int failures;
static int cases;

#define CHECK(cond, what)                                                  \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s (%s:%d)\n", (what), __FILE__, __LINE__);        \
            failures++;                                                    \
        }                                                                  \
    } while (0)

/* Predictions with online history, exactly as the record is defined:
 * stateful families read the ACTUAL byte, so this is both what a writer
 * must send and what the decoder must produce. */
static void reference_pred(const uint8_t *plane, uint32_t nt, uint8_t literal,
                           uint8_t pred, uint8_t c)
{
    uint8_t last[8];
    uint8_t second[8];
    uint32_t p;
    int l;

    for (l = 0; l < 8; l++) {
        last[l] = literal;
        second[l] = literal;
    }
    for (p = 0; p < nt; p++) {
        unsigned lane = p & 7u;
        uint8_t ph;

        switch (pred) {
        case ZPSE_P_CONST_ZERO:
            ph = 0u;
            break;
        case ZPSE_P_TILE_CONSTANT:
            ph = c;
            break;
        case ZPSE_P_LANE_PREVIOUS:
            ph = last[lane];
            break;
        case ZPSE_P_LANE_DELTA:
            ph = (uint8_t)((2u * (unsigned)last[lane] -
                            (unsigned)second[lane]) & 0xFFu);
            break;
        case ZPSE_P_LANE_AVERAGE:
            ph = (uint8_t)(((unsigned)last[lane] +
                            (unsigned)second[lane] + 1u) >> 1);
            break;
        default: { /* STRIDE family */
            uint32_t s = 8u;

            if (pred == ZPSE_P_STRIDE_1) s = 1u;
            else if (pred == ZPSE_P_STRIDE_2) s = 2u;
            else if (pred == ZPSE_P_STRIDE_3) s = 3u;
            else if (pred == ZPSE_P_STRIDE_4) s = 4u;
            ph = (p < s) ? literal : plane[p - s];
            xhat[p] = ph;
            continue; /* no history update for STRIDE */
        }
        }
        xhat[p] = ph;
        second[lane] = (pred == ZPSE_P_LANE_PREVIOUS) ? ph : last[lane];
        last[lane] = plane[p];
    }
}

/* Build one dense segment payload for `plane` and return its length. */
static uint32_t build(const uint8_t *plane, uint32_t nt, uint8_t literal,
                      uint8_t pred, uint8_t c, uint8_t mask_mode,
                      uint8_t res_mode)
{
    uint32_t p;
    uint32_t k = 0;
    uint32_t need = 0;
    uint32_t mask_len = 0;
    uint32_t res_len = 0;
    uint32_t j;
    int n;
    ZpseSegmentHeader h;
    uint8_t *rec = seg + 16;

    reference_pred(plane, nt, literal, pred, c);
    for (p = 0; p < nt; p++) {
        if (plane[p] != xhat[p]) {
            exc[k++] = zpse_exclusion_forward(plane[p], xhat[p]);
        }
    }

    /* Only the mask the plane's exception count actually admits may be
     * sent: ALL_HIT iff K == 0, ALL_EXCEPTION iff K == nt, and the two
     * positional forms only in between. */
    if (k == 0u) {
        mask_mode = ZPSE_M_ALL_HIT;
    } else if (k == nt) {
        mask_mode = ZPSE_M_ALL_EXCEPTION;
    } else if (mask_mode == ZPSE_M_SPARSE_U8 && nt > 256u) {
        mask_mode = ZPSE_M_RAW_BITMAP; /* positions are bytes */
    } else if (mask_mode != ZPSE_M_RAW_BITMAP &&
               mask_mode != ZPSE_M_SPARSE_U8) {
        mask_mode = ZPSE_M_RAW_BITMAP;
    }

    if (mask_mode == ZPSE_M_RAW_BITMAP) {
        need = (nt + 7u) >> 3;
        memset(maskp, 0, need);
        for (p = 0; p < nt; p++) {
            if (plane[p] != xhat[p]) {
                maskp[p >> 3] |= (uint8_t)(1u << (p & 7u));
            }
        }
        mask_len = need;
    } else if (mask_mode == ZPSE_M_SPARSE_U8) {
        j = 0;
        for (p = 0; p < nt; p++) {
            if (plane[p] != xhat[p]) {
                maskp[j++] = (uint8_t)p;
            }
        }
        mask_len = k;
    }

    n = zpse_residual_encode(res_mode, exc, plane, 0, (uint16_t)k, resp,
                             (uint16_t)sizeof resp);
    if (n < 0) {
        return 0u;
    }
    res_len = (uint32_t)n;

    memset(&h, 0, sizeof h);
    h.segment_decoded_size = nt;
    h.tile_count = 1u;
    h.initial_literal = literal;
    h.segment_flags = ZPSE_S_LEN_PREFIX;
    zpse_segment_header_serialize(seg, &h);

    /* rec: u16 rec_len, flags, predictor, C, reserved, K, mask_len, res_len */
    rec[0] = (uint8_t)((10u + mask_len + res_len) & 0xFFu);
    rec[1] = (uint8_t)((10u + mask_len + res_len) >> 8);
    rec[2] = (uint8_t)((mask_mode & 7u) | ((res_mode & 7u) << 3) |
                       ZPSE_DENSE_MARKER);
    rec[3] = pred;
    rec[4] = c;
    rec[5] = 0u;
    rec[6] = (uint8_t)(k & 0xFFu);
    rec[7] = (uint8_t)(k >> 8);
    rec[8] = (uint8_t)(mask_len & 0xFFu);
    rec[9] = (uint8_t)(mask_len >> 8);
    rec[10] = (uint8_t)(res_len & 0xFFu);
    rec[11] = (uint8_t)(res_len >> 8);
    memcpy(rec + 12, maskp, mask_len);
    memcpy(rec + 12 + mask_len, resp, res_len);

    return 16u + 12u + mask_len + res_len;
}

static int decode(uint32_t payload_len, uint8_t *got, uint32_t *got_len)
{
    ZpseDecoder d;
    ZpseLaneState lanes;
    int rc;

    rc = zpse_decoder_init(&d, DARENA, sizeof DARENA);
    if (rc != PSE_OK) {
        return rc;
    }
    lanes.order1 = d.order1;
    lanes.o2val = d.o2val;
    lanes.o2conf = d.o2conf;
    return zpse_segment_decode(&d, &lanes, seg, payload_len, got,
                               ZPSE_DENSE_MAX_LEN, got_len);
}

static void fill_plane(uint8_t *plane, uint32_t nt, int kind)
{
    uint32_t i;
    uint32_t st = 12345u;

    for (i = 0; i < nt; i++) {
        switch (kind) {
        case 0:
            plane[i] = 0u;
            break;
        case 1:
            plane[i] = 0x5Au;
            break;
        case 2:
            plane[i] = (uint8_t)(i * 3u);      /* period 3, STRIDE_3 */
            break;
        case 3:
            plane[i] = (uint8_t)(i);           /* ramp, STRIDE_1 */
            break;
        case 4:                                /* lane-local, LANE_PREVIOUS */
            plane[i] = (uint8_t)((i & 7u) * 7u);
            break;
        case 5:                                /* sparse spikes */
            plane[i] = (i % 61u == 0u) ? (uint8_t)(i + 1u) : 0u;
            break;
        default: {                             /* low entropy */
            st = st * 1664525u + 1013904223u;
            plane[i] = (uint8_t)((st >> 24) & 0x0Fu);
            break;
        }
        }
    }
}

static void run_case(uint32_t nt, int kind, uint8_t pred, uint8_t c,
                     uint8_t mask_mode, uint8_t res_mode, const char *what)
{
    static uint8_t plane[ZPSE_DENSE_MAX_LEN];
    uint32_t len;
    uint32_t got_len = 0;
    int rc;

    fill_plane(plane, nt, kind);
    len = build(plane, nt, 0x11u, pred, c, mask_mode, res_mode);
    if (len == 0u) {
        return; /* residual packer declined this shape; not a failure */
    }
    cases++;
    rc = decode(len, out, &got_len);
    if (rc != PSE_OK || got_len != nt || memcmp(out, plane, nt) != 0) {
        printf("FAIL %s nt=%u pred=%u mask=%u res=%u rc=%s got=%u\n", what,
               nt, pred, mask_mode, res_mode, zpse_status_name(rc), got_len);
        failures++;
    }
}

/* Hand-computed predictions: the writer must send exactly these, and the
 * decoder must reproduce the plane from them. */
static void hand_checked(void)
{
    /* PLANE = 10 20 30 40 50 60 70 80 | 11 22 33 44, literal 0, nt 12.
     * LANE_DELTA at p=8: last=10, second=0 -> 20; plane has 11 (exception).
     * LANE_AVERAGE at p=8: (10 + 0 + 1) >> 1 = 5; plane has 11. */
    static uint8_t plane[12] = {10, 20, 30, 40, 50, 60, 70, 80, 11, 22, 33, 44};
    static uint8_t p2[12] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    uint32_t len;
    uint32_t got_len = 0;
    int rc;

    reference_pred(plane, 12u, 0u, ZPSE_P_LANE_DELTA, 0u);
    CHECK(xhat[8] == 20u && xhat[9] == 40u && xhat[10] == 60u,
          "lane-delta prediction is 2*last - second");
    reference_pred(p2, 12u, 0u, ZPSE_P_LANE_AVERAGE, 0u);
    CHECK(xhat[0] == 0u && xhat[1] == 0u && xhat[8] == 5u && xhat[9] == 5u,
          "lane-average prediction is (last + second + 1) / 2");

    len = build(plane, 12u, 0u, ZPSE_P_LANE_DELTA, 0u, ZPSE_M_SPARSE_U8,
                ZPSE_R_EXCLUSION_U8);
    cases++;
    rc = decode(len, out, &got_len);
    CHECK(rc == PSE_OK && got_len == 12u && memcmp(out, plane, 12u) == 0,
          "hand plane, lane-delta, sparse mask");
}

/* Every malformed shape must be rejected, and none may decode. */
static void hostile(void)
{
    static uint8_t plane[64];
    uint32_t len;
    uint32_t got_len = 0;
    uint8_t save;
    int rc;
    uint32_t base;

    fill_plane(plane, 64u, 5);
    len = build(plane, 64u, 0x11, ZPSE_P_LANE_PREVIOUS, 0u, ZPSE_M_RAW_BITMAP,
                ZPSE_R_EXCLUSION_U8);
    CHECK(len != 0u, "hostile base builds");
    cases++;
    rc = decode(len, out, &got_len);
    CHECK(rc == PSE_OK && memcmp(out, plane, 64u) == 0, "hostile base decodes");

    /* Reserved bit 7 of the flags byte. */
    save = seg[16 + 2];
    seg[16 + 2] = (uint8_t)(save | 0x80u);
    CHECK(decode(len, out, &got_len) != PSE_OK, "reserved flag bit rejected");
    seg[16 + 2] = save;

    /* Reserved header byte. */
    seg[16 + 5] = 1u;
    CHECK(decode(len, out, &got_len) != PSE_OK, "reserved header byte");
    seg[16 + 5] = 0u;

    /* ORDER1 prediction is reserved inside a dense record. */
    save = seg[16 + 3];
    seg[16 + 3] = ZPSE_P_ORDER1;
    CHECK(decode(len, out, &got_len) != PSE_OK, "ORDER1 rejected in dense");
    seg[16 + 3] = save;

    /* Predictor out of range. */
    seg[16 + 3] = 12u;
    CHECK(decode(len, out, &got_len) != PSE_OK, "predictor >= 12 rejected");
    seg[16 + 3] = save;

    /* Signed-delta residual mode is reserved in v1. */
    save = seg[16 + 2];
    seg[16 + 2] = (uint8_t)((save & 0xC7u) | (ZPSE_R_SIGNED_DELTA << 3));
    CHECK(decode(len, out, &got_len) != PSE_OK, "residual mode 4 rejected");
    seg[16 + 2] = save;

    /* Mask mode 3 is reserved inside a dense record. */
    seg[16 + 2] = (uint8_t)((save & 0xF8u) | 3u);
    CHECK(decode(len, out, &got_len) != PSE_OK, "mask mode 3 rejected");
    seg[16 + 2] = save;

    /* Mask mode 7. */
    seg[16 + 2] = (uint8_t)((save & 0xF8u) | 7u);
    CHECK(decode(len, out, &got_len) != PSE_OK, "mask mode 7 rejected");
    seg[16 + 2] = save;

    /* rec_len disagreeing with the span. */
    save = seg[16];
    seg[16] = (uint8_t)(save - 1u);
    CHECK(decode(len, out, &got_len) != PSE_OK, "short rec_len rejected");
    seg[16] = save;

    /* A trailing byte after the record. */
    seg[len] = 0u;
    CHECK(decode(len + 1u, out, &got_len) != PSE_OK,
          "trailing byte rejected");
    seg[len] = 0u;

    /* K larger than the plane (all-exception mask with a short plane). */
    base = (uint32_t)seg[16 + 6] | ((uint32_t)seg[16 + 7] << 8);
    seg[16 + 6] = (uint8_t)(base + 1u);
    CHECK(decode(len, out, &got_len) != PSE_OK, "K beyond mask rejected");
    seg[16 + 6] = (uint8_t)(base & 0xFFu);

    /* Bitmap with a set padding bit above nt.  Build one deliberately. */
    {
        static uint8_t pl[3] = {0, 0, 1}; /* one exception: a real bitmap */
        uint32_t l2 = build(pl, 3u, 0u, ZPSE_P_CONST_ZERO, 0u,
                            ZPSE_M_RAW_BITMAP, ZPSE_R_EXCLUSION_U8);
        uint32_t k0 = (uint32_t)seg[16 + 6] | ((uint32_t)seg[16 + 7] << 8);

        CHECK(l2 != 0u && k0 == 1u, "3-byte bitmap base");
        seg[16 + 12] |= 0x08u; /* bit 3 >= nt == 3: pad bit */
        CHECK(decode(l2, out, &got_len) != PSE_OK, "pad bit rejected");
    }

    /* Sparse positions not ascending. */
    {
        static uint8_t pl[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint32_t l3 = build(pl, 8u, 0u, ZPSE_P_CONST_ZERO, 0u,
                            ZPSE_M_SPARSE_U8, ZPSE_R_EXCLUSION_U8);
        uint32_t k3 = (uint32_t)seg[16 + 6] | ((uint32_t)seg[16 + 7] << 8);

        CHECK(l3 != 0u && k3 == 8u, "8-byte sparse base is all-exception");
        /* nt == 8 makes every position an exception, so the builder used
         * ALL_EXCEPTION; a sparse mask is validated further down. */
        (void)pl;
    }
    {
        static uint8_t pl[12];
        uint32_t l4;
        uint32_t k4;
        uint32_t q;

        for (q = 0; q < 12u; q++) {
            pl[q] = (q % 3u == 0u) ? 0u : 1u; /* 8 exceptions under CONST */
        }
        l4 = build(pl, 12u, 0u, ZPSE_P_CONST_ZERO, 0u, ZPSE_M_SPARSE_U8,
                   ZPSE_R_EXCLUSION_U8);
        k4 = (uint32_t)seg[16 + 6] | ((uint32_t)seg[16 + 7] << 8);
        CHECK(l4 != 0u && k4 >= 2u, "sparse base has >= 2 exceptions");
        if (k4 >= 2u) {
            /* positions are ascending: swap the first two. */
            uint8_t t = seg[16 + 12];

            seg[16 + 12] = seg[16 + 13];
            seg[16 + 13] = t;
            CHECK(decode(l4, out, &got_len) != PSE_OK,
                  "non-ascending sparse rejected");
        }
    }

    /* Marker clear: the same bytes must not be read as a dense record. */
    {
        static uint8_t pl[16];
        uint32_t l5;

        fill_plane(pl, 16u, 4);
        l5 = build(pl, 16u, 0u, ZPSE_P_LANE_PREVIOUS, 0u, ZPSE_M_RAW_BITMAP,
                   ZPSE_R_EXCLUSION_U8);
        seg[16 + 2] &= (uint8_t)~ZPSE_DENSE_MARKER;
        CHECK(decode(l5, out, &got_len) != PSE_OK, "unmarked record is no tile");
        seg[16 + 2] |= ZPSE_DENSE_MARKER;
        CHECK(decode(l5, out, &got_len) == PSE_OK, "marked record decodes");
    }
}

int main(void)
{
    static const uint8_t preds[] = {
        ZPSE_P_CONST_ZERO,     ZPSE_P_LANE_PREVIOUS, ZPSE_P_LANE_DELTA,
        ZPSE_P_LANE_AVERAGE,   ZPSE_P_STRIDE_1,     ZPSE_P_STRIDE_2,
        ZPSE_P_STRIDE_3,       ZPSE_P_STRIDE_4,     ZPSE_P_STRIDE_8,
        ZPSE_P_TILE_CONSTANT};
    static const uint32_t sizes[] = {1u, 2u, 7u, 8u, 12u, 17u, 256u, 1000u,
                                     4096u};
    static const uint8_t masks[] = {ZPSE_M_ALL_HIT, ZPSE_M_RAW_BITMAP,
                                    ZPSE_M_SPARSE_U8, ZPSE_M_ALL_EXCEPTION};
    static const uint8_t resmodes[] = {ZPSE_R_EXCLUSION_U8,
                                       ZPSE_R_PALETTE_BITS,
                                       ZPSE_R_PACKED_BASE_M};
    unsigned si, pi, mi, ri;
    int kind;

    hand_checked();

    for (si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        for (kind = 0; kind < 7; kind++) {
            for (pi = 0; pi < sizeof preds / sizeof preds[0]; pi++) {
                for (mi = 0; mi < sizeof masks / sizeof masks[0]; mi++) {
                    for (ri = 0; ri < sizeof resmodes / sizeof resmodes[0]; ri++) {
                        run_case(sizes[si], kind, preds[pi], 0x5Au, masks[mi],
                                 resmodes[ri], "roundtrip");
                    }
                }
            }
        }
    }

    hostile();

    printf("DENSE: cases=%d failures=%d\n", cases, failures);
    if (failures != 0) {
        printf("DENSE FAILURES\n");
        return 1;
    }
    printf("dense records ok\n");
    return 0;
}
