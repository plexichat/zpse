#ifndef ZPSE_PAGEFAST_H
#define ZPSE_PAGEFAST_H

/* One-pass page classifier. Given a 4 KiB page, decide which structure an
 * encoder can use WITHOUT running a predictor search or a second scan.
 *
 * The classes are ordered by specificity: the first one that applies wins.
 *   ZERO        every byte is 0
 *   ONE_VALUE   every byte is equal (info->value holds the byte)
 *   LOW_ALPHA   at most 16 distinct byte values (info->alpha in first-appearance
 *               order, info->nalpha entries)
 *   ASCII       every byte < 0x80
 *   POINTER     high bytes of 2/4/8-byte lanes are constant (or zero in at
 *               least 7/8 of the lanes) while low bytes vary
 *   GENERIC     none of the above
 *
 * info->value is the first byte of the page (0 for n == 0); it is the
 * definitive fill only for ZERO and ONE_VALUE. info->alpha/nalpha are filled
 * whenever at most 16 distinct values were found, regardless of class, and
 * are zeroed when the alphabet overflowed 16. info->nzero is the exact count
 * of zero bytes. info->flags is reserved and is always 0.
 *
 * info->lane_diff and info->zero_cnt expose the classifier's per-lane byte
 * statistics so a FLAT/SUBLANE encoder can build a 32-byte template from the
 * first period without a second scan. For lane j (0..31) the lane is the set
 * of byte positions j, j+32, j+64, ... whose offset modulo 32 is j:
 *   lane_diff[j] = OR of (byte XOR in[j]) over that lane. Bit k is set iff
 *                  some byte in the lane differed from the lane's first byte
 *                  in[j] in bit k; lane_diff[j] == 0 means position j held
 *                  the same byte in every period (a constant template byte).
 *   zero_cnt[j]  = number of zero bytes in that lane, in 0..128 for a 4 KiB
 *                  page (one byte per 32-byte period).
 * Both arrays are filled for every class and are bit-identical between the
 * scalar and AVX2 paths; for n == 0 they are all zero.
 *
 * n == 0 is reported as ZERO with nalpha == 0 and nzero == 0.
 *
 * No allocation, no state: the function is a pure function of (in, n). The
 * AVX2 kernel is selected at runtime on hosts where the ABI allows it; the
 * scalar fallback is bit-identical. Define ZPSE_PAGEFAST_AVX2=0 to compile
 * the kernel out (the scalar path is then the only one). */

#include <stdint.h>

#ifndef ZPSE_PAGEFAST_AVX2
#define ZPSE_PAGEFAST_AVX2 1
#endif

typedef enum ZpsePageClass {
    ZPSE_PAGE_ZERO        = 0,
    ZPSE_PAGE_ONE_VALUE   = 1,
    ZPSE_PAGE_LOW_ALPHA   = 2,
    ZPSE_PAGE_ASCII       = 3,
    ZPSE_PAGE_POINTER     = 4,
    ZPSE_PAGE_GENERIC     = 5
} ZpsePageClass;

typedef struct ZpsePageInfo {
    uint8_t  value;         /* ONE_VALUE / ZERO */
    uint8_t  alpha[16];     /* LOW_ALPHA, first-appearance order */
    uint8_t  nalpha;        /* entries in alpha, 0 if > 16 distinct */
    uint16_t nzero;         /* count of zero bytes in the page */
    uint32_t flags;         /* reserved, always 0 */
    uint8_t  lane_diff[32]; /* per-lane OR of (byte ^ lane's first byte) */
    uint8_t  zero_cnt[32];  /* per-lane number of zero bytes */
} ZpsePageInfo;

/* Returns one of the ZpsePageClass values. in may be unaligned. */
int zpse_page_classify(const uint8_t *in, uint32_t n, ZpsePageInfo *info);

/* Build a 256-byte value -> palette-index LUT for a FLAT low-alpha page.
 *
 * For every byte value v, lut[v] is the palette index of v: the smallest i
 * with alpha[i] == v (FIRST MATCH WINS), or 0xFF when v is absent from the
 * palette. The classifier's alpha holds distinct values in first-appearance
 * order, so the first match is also the only match and the mapping is
 * unambiguous. Every byte of a LOW_ALPHA page is in its palette, so the
 * 0xFF sentinel is never returned for input the classifier accepted; it is
 * only a safe default for values outside the palette.
 *
 * This is byte-identical to the linear first-match scan of alpha[0..nalpha-1]
 * for all values that can occur on a LOW_ALPHA page, but turns each lookup
 * into a single load instead of up to 16 compares.
 *
 * alpha must point to at least nalpha (<= 16) readable bytes and lut to 256
 * writable bytes. No allocation or retained state. */
void zpse_pf_make_pal_lut(const uint8_t alpha[16], uint8_t nalpha,
                          uint8_t lut[256]);

#endif /* ZPSE_PAGEFAST_H */
