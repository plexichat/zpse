#ifndef ZPSE_ARENA_H
#define ZPSE_ARENA_H

/* Fixed arena layouts. Caller-owned, 64-byte
 * aligned, zeroed at context creation. No allocation afterwards.
 *
 * Both arenas open with a 4 KiB (0x01000) control block at base 0x00000.
 * The decoder's live regions end at 0x44800, and 0x0B800 bytes of guard
 * carry that to the 0x50000 start of the 64 KiB LZ body buffer. The
 * encoder's live regions end at 0x6D800, and 0x06800 bytes of guard carry
 * that to the 0x74000 arena end. The guards are only padding between named
 * regions, never part of the stream. */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_DEC_ARENA_SIZE (384u * 1024u) /* 0x60000 */
#define ZPSE_ENC_ARENA_SIZE (464u * 1024u) /* 0x74000 */
#define ZPSE_ARENA_ALIGN    64u

/* Decoder offsets. 0x21000..0x41800 is a reserved hole where the ORDER1
 * and ORDER2 tables used to live; the live regions keep their offsets. */
#define ZPSE_D_OFF_RECON    0x01000u
#define ZPSE_D_OFF_OUT      0x11000u
#define ZPSE_D_OFF_COMB     0x41800u
#define ZPSE_D_OFF_SHUF     0x43800u
/* Where a ZPSE_B_LZ_TILE block's body is unpacked: the tile path reads it
 * from here exactly as it would read it from the block, so the 64 KiB a
 * block body can take has to be its own region rather than borrowed from
 * recon/out_stage, which the segment decoder is using at the same time. */
#define ZPSE_D_OFF_LZBUF    0x50000u

#define ZPSE_D_SZ_RECON     0x10000u
#define ZPSE_D_SZ_OUT       0x10000u
/* 8 KiB generic scratch for dense/tile exception buffers. */
#define ZPSE_D_SZ_COMB      0x02000u
#define ZPSE_D_SZ_SHUF      0x01000u
#define ZPSE_D_SZ_LZBUF     0x10000u

/* Padding block decode leaves between the segment planes it writes for a
 * 16-wide interleave, so that the inverse transpose can read them in
 * place instead of staging a copy of each. The L1 set index is
 * (address / 64) mod 64, so planes exactly 4096 bytes apart all land on
 * one set; 4096 + 64 is 65 lines, which advances the set by one per
 * plane, and -- what 16 bytes would not -- keeps every plane on a cache
 * line boundary, so the segment decoder's stores stay aligned. Purely a
 * placement inside recon/out_stage, never part of the stream. Eight
 * planes per region need 7 * 4160 + 4096 = 33216 bytes. */
#define ZPSE_D_PLANE_SKEW   64u

/* Encoder offsets. */
#define ZPSE_E_OFF_INPUT    0x01000u
#define ZPSE_E_OFF_TSTAGE   0x11000u
#define ZPSE_E_OFF_CAND     0x21000u
#define ZPSE_E_OFF_BEST     0x31000u
/* 0x41000..0x61800 is a reserved hole (the old ORDER1/ORDER2 tables). */
#define ZPSE_E_OFF_COMB     0x61800u
#define ZPSE_E_OFF_HIST     0x63800u
#define ZPSE_E_OFF_SHUF     0x64800u
#define ZPSE_E_OFF_LZ       0x65800u

#define ZPSE_E_SZ_INPUT     0x10000u
#define ZPSE_E_SZ_TSTAGE    0x10000u
#define ZPSE_E_SZ_CAND      0x10000u
#define ZPSE_E_SZ_BEST      0x10000u
/* 8 KiB generic scratch, as in the decoder arena (see ZPSE_D_SZ_COMB). */
#define ZPSE_E_SZ_COMB      0x02000u
#define ZPSE_E_SZ_HIST      0x01000u
#define ZPSE_E_SZ_SHUF      0x01000u
/* The intra-block LZ match finder's direct-mapped table, sized for the
 * widest pass that uses this region: 16,384 16-bit block positions, i.e.
 * a 2^14 table (ZPSE_LZ_HASH_BITS_WIDE). The screens and the page passes
 * still use the first 2^13 entries; only the 4-16 KiB lazy pass indexes
 * the whole region. Only the effort levels that write an LZ form touch
 * it, so it is the encoder arena that pays for it and the decoder never
 * carries it. */
#define ZPSE_E_SZ_LZ        0x08000u

typedef struct ZpseDecoder {
    uint8_t *arena;       /* base, 64-B aligned, 384 KiB (incl. 64 KiB lzbuf) */
    uint8_t *recon;       /* transformed reconstruction (64 KiB) */
    uint8_t *out_stage;   /* output staging (64 KiB) */
    uint8_t *comb;        /* 8 KiB generic scratch (dense/tile exceptions) */
    uint8_t *shuf;        /* 4 KiB shuffle/CRC consts */
    uint8_t *lzbuf;       /* 64 KiB: unpacked body of an LZ_TILE block */
    /* The plane the last zpse_segment_decode produced was one byte value
     * end to end, and which value. Set only by the dense constant form and
     * cleared by every segment decode, so a block whose single segment
     * reports it (the identity shape the encoder emits for a one-value
     * block) can checksum its output arithmetically instead of scanning
     * 64 KiB the decoder just wrote. */
    uint8_t seg_const;
    uint8_t seg_const_byte;
} ZpseDecoder;

typedef struct ZpseEncoder {
    uint8_t *arena;       /* base, 64-B aligned, 464 KiB */
    uint8_t *input;       /* 64 KiB */
    uint8_t *tstage;      /* 64 KiB */
    uint8_t *cand;        /* 64 KiB candidate staging */
    uint8_t *best;        /* 64 KiB best staging */
    uint8_t *comb;        /* 8 KiB generic scratch (dense/tile exceptions) */
    uint8_t *hist;        /* 4 KiB histogram/palette scratch */
    uint8_t *shuf;        /* 4 KiB shuffle LUT + CRC consts */
    uint8_t *lz;          /* 32 KiB LZ match-finder table (2^14 max) */
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
    /* The block input pointer, when        zpse_block_encode has proved that whole
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
     * describe (RAW, either LZ form) and by any change of shape. Saturates
     * rather than wrapping. See the history-confident stop in
     * src/block.c. */
    uint8_t hist_streak;
} ZpseEncoder;

int zpse_decoder_init(ZpseDecoder *d, void *arena_mem, size_t arena_len);
int zpse_encoder_init(ZpseEncoder *e, void *arena_mem, size_t arena_len);

int zpse_arena_is_aligned(const void *p);

#endif /* ZPSE_ARENA_H */
