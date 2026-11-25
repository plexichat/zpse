#ifndef ZPSE_TX_H
#define ZPSE_TX_H

/* One-shot whole-block transforms (verified standalone modules) glued into
 * the block codec. IDs 12..14 are length-preserving pre-transforms: the
 * encoder applies the forward map once, the ordinary block ladder encodes
 * the transformed bytes as an identity-shape block, and the decoder applies
 * the inverse once after the envelope has been decoded and checked. */

#include <stdint.h>

/* New transform IDs, kept here so block.c and types.c can name them. */
#define ZPSE_T_TX_BITSHUFFLE32 ((uint8_t)12)
#define ZPSE_T_TX_DELTA2_64    ((uint8_t)13)
#define ZPSE_T_TX_WORDXOR64    ((uint8_t)14)
#define ZPSE_T_TX_BITSHUFFLE16 ((uint8_t)15)

/* Selection arm, shared by the block wrapper and the whole-frame decision so
 * both agree on which blocks carry a one-shot transform.
 * 1 = MRD's existing 16-bit delta gate; 2 = previous-block transform history;
 * 3 = the page classifier's pointer-lane / zero-count signature. */
#ifndef ZPSE_TX_SIGNAL
#define ZPSE_TX_SIGNAL 1
#endif

/* O(1) selection from signals already computed by the encoder. Returns 0
 * (no transform) or one of the three IDs above. */
int zpse_tx_select(const void *enc, const uint8_t *in, uint32_t n);

/* Cheap O(1) content probe for the one-shot selector: 256 evenly spaced
 * adjacent 16-bit deltas after a fixed header skip. Mirrors MRD's gate shape
 * without its 8 K-sample pass. */
int zpse_tx_mr_probe(const uint8_t *in, uint32_t n);

/* Forward / inverse. Return 1 on success, 0 when the transform cannot be
 * applied to this n (e.g. a bitshuffle tail that would change the length).
 * `dst` must hold at least n bytes. */
int zpse_tx_apply(int tid, const uint8_t *src, uint32_t n, uint8_t *dst);
int zpse_tx_inverse(int tid, const uint8_t *src, uint32_t n, uint8_t *dst);

#endif /* ZPSE_TX_H */
