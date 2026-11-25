#ifndef ZPSE_ARENA_H
#define ZPSE_ARENA_H

/* Fixed arena layouts. Caller-owned, 64-byte
 * aligned, zeroed at context creation. No allocation afterwards.
 *
 * Both arenas open with a 4 KiB (0x01000) control block at base 0x00000.
 * The block-sized staging regions are 0x50000 (320 KiB) each: a 256 KiB
 * block plus the 64-byte-per-plane skew a 16-wide interleave needs.
 * The gaps between named regions are only padding, never part of a stream. */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_DEC_ARENA_SIZE (0x100000u) /* 1 MiB */
#define ZPSE_ENC_ARENA_SIZE (0x150000u)
#define ZPSE_ARENA_ALIGN    64u

/* Decoder offsets. */
#define ZPSE_D_OFF_RECON    0x01000u
#define ZPSE_D_OFF_OUT      0x51000u
#define ZPSE_D_OFF_COMB     0xA1000u
#define ZPSE_D_OFF_SHUF     0xA3000u
/* Unpacked body of an LZ_TILE block: a 256 KiB block body needs its own
 * region rather than borrowing from recon/out_stage. */
#define ZPSE_D_OFF_LZBUF    0xA4000u

#define ZPSE_D_SZ_RECON     0x50000u
#define ZPSE_D_SZ_OUT       0x50000u
/* 8 KiB generic scratch for dense/tile exception buffers. */
#define ZPSE_D_SZ_COMB      0x02000u
#define ZPSE_D_SZ_SHUF      0x01000u
#define ZPSE_D_SZ_LZBUF     0x50000u

/* Padding block decode leaves between the segment planes it writes for a
 * 16-wide interleave, so that the inverse transpose can read them in
 * place instead of staging a copy of each. Purely a placement inside
 * recon/out_stage, never part of the stream. */
#define ZPSE_D_PLANE_SKEW   64u

/* Encoder offsets. */
#define ZPSE_E_OFF_INPUT    0x01000u
#define ZPSE_E_OFF_TSTAGE   0x51000u
#define ZPSE_E_OFF_CAND     0xA1000u
#define ZPSE_E_OFF_BEST     0xF1000u
#define ZPSE_E_OFF_COMB     0x141000u
#define ZPSE_E_OFF_HIST     0x143000u
#define ZPSE_E_OFF_SHUF     0x144000u
#define ZPSE_E_OFF_LZ       0x145000u

#define ZPSE_E_SZ_INPUT     0x50000u
#define ZPSE_E_SZ_TSTAGE    0x50000u
#define ZPSE_E_SZ_CAND      0x50000u
#define ZPSE_E_SZ_BEST      0x50000u
/* 8 KiB generic scratch, as in the decoder arena (see ZPSE_D_SZ_COMB). */
#define ZPSE_E_SZ_COMB      0x02000u
#define ZPSE_E_SZ_HIST      0x01000u
#define ZPSE_E_SZ_SHUF      0x01000u
/* 32 KiB of spare staging for the LZ4 pass. */
#define ZPSE_E_SZ_LZ        0x08000u

typedef struct ZpseDecoder {
    uint8_t *arena;       /* base, 64-B aligned, 1 MiB (incl. 320 KiB lzbuf) */
    uint8_t *recon;       /* transformed reconstruction (320 KiB) */
    uint8_t *out_stage;   /* output staging (320 KiB) */
    uint8_t *comb;        /* 8 KiB generic scratch (dense/tile exceptions) */
    uint8_t *shuf;        /* 4 KiB shuffle/CRC consts */
    uint8_t *lzbuf;       /* 320 KiB: unpacked body of an LZ_TILE block */
    /* The plane the last zpse_segment_decode produced was one byte value
     * end to end, and which value. Set only by the dense constant form and
     * cleared by every segment decode, so a block whose single segment
     * reports it (the identity shape the encoder emits for a one-value
     * block) can checksum its output arithmetically instead of scanning
     * 320 KiB the decoder just wrote. */
    uint8_t seg_const;
    uint8_t seg_const_byte;
} ZpseDecoder;

typedef struct ZpseEncoder {
    uint8_t *arena;       /* base, 64-B aligned, 0x150000 */
    uint8_t *input;       /* 320 KiB */
    uint8_t *tstage;      /* 320 KiB */
    uint8_t *cand;        /* 320 KiB candidate staging */
    uint8_t *best;        /* 320 KiB best staging */
    uint8_t *comb;        /* 8 KiB generic scratch (dense/tile exceptions) */
    uint8_t *hist;        /* 4 KiB histogram/palette scratch */
    uint8_t *shuf;        /* 4 KiB shuffle LUT + CRC consts */
    uint8_t *lz;          /* 32 KiB LZ scratch/staging (legacy table name) */
    /* Which optional fields this encoder emits (ZPSE_ENC_* in
     * constants.h). The one-shot API leaves it 0, i.e. no payload
     * checksum; zpse_compress_block_ex / zpse_compress_frame_ex set it
     * from their flags argument after zpse_encoder_init. */
    uint32_t enc_flags;
    /* Trial-level progress for zpse_segment_encode's flat abort: the length
     * of the leading run of tiles whose record cost at least as much as the
     * bytes it codes, and the flag that ends that run (set by the first tile
     * that does pay for itself). Accumulated across the segments of one
     * trial and reset by the caller per trial. */
    uint32_t flat_covered;
    uint32_t flat_paid;
    /* The block input pointer, when zpse_block_encode has proved that whole
     * input is one byte value while choosing its CRC form. zpse_segment_encode
     * skips its own plane scan when it is handed this very pointer (the
     * identity trial's plane does not copy), so a constant block is scanned
     * once. NULL for every block that is not constant, and reset per block. */
    const uint8_t *const_plane;
    /* Set by zpse_block_encode when its decision sample has proved that the
     * tile path loses to the seven-bit ASCII form on this block. The segment
     * encoder then writes an ASCII record for every ASCII tile without a
     * prescreen, a classify, a mode search or a pack -- the form's price is
     * known before the first tile, which is the whole of why the gate exists.
     * Never read by the decoder, and reset per block. */
    uint8_t ascii_gate;
    /* The transform/segment shape the PREVIOUS block of this context chose,
     * when that block was a tile form. zpse_block_encode tries it first
     * whenever the probe cannot separate it from its own winner, which is
     * what makes a run of blocks from one file cheaper to encode: the first
     * block pays the full search and the rest start from its answer.
     * Only zpse_encoder_init clears it, so a one-shot call starts blank and
     * a frame keeps the history across its blocks. Never read by a decoder
     * and never part of a stream. */
    uint8_t hist_tid;
    uint8_t hist_sc;
    uint8_t hist_valid;
    /* Consecutive blocks whose winner was the same tile shape as the block
     * before them: the run length of the homogeneous stretch the order-n
     * predictor is extrapolating. Reset by any block the history cannot
     * describe (RAW, any LZ form) and by any change of shape. Saturates
     * rather than wrapping. See the history-confident stop in
     * src/block.c. */
    uint8_t hist_streak;
} ZpseEncoder;

int zpse_decoder_init(ZpseDecoder *d, void *arena_mem, size_t arena_len);
int zpse_encoder_init(ZpseEncoder *e, void *arena_mem, size_t arena_len);

int zpse_arena_is_aligned(const void *p);

#endif /* ZPSE_ARENA_H */
