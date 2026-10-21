#ifndef ZPSE_ARENA_H
#define ZPSE_ARENA_H

/* Fixed arena layouts. Caller-owned, 64-byte
 * aligned, zeroed at context creation. No allocation afterwards. */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_DEC_ARENA_SIZE (320u * 1024u) /* 0x50000 */
#define ZPSE_ENC_ARENA_SIZE (448u * 1024u) /* 0x70000 */
#define ZPSE_ARENA_ALIGN    64u

/* Decoder offsets. */
#define ZPSE_D_OFF_CTRL     0x00000u
#define ZPSE_D_OFF_RECON    0x01000u
#define ZPSE_D_OFF_OUT      0x11000u
#define ZPSE_D_OFF_ORDER1   0x21000u
#define ZPSE_D_OFF_O2       0x21800u
#define ZPSE_D_OFF_COMB     0x41800u
#define ZPSE_D_OFF_SHUF     0x43800u
#define ZPSE_D_OFF_GUARD    0x44800u

#define ZPSE_D_SZ_CTRL      0x01000u
#define ZPSE_D_SZ_RECON     0x10000u
#define ZPSE_D_SZ_OUT       0x10000u
#define ZPSE_D_SZ_ORDER1    0x00800u
#define ZPSE_D_SZ_O2        0x20000u
#define ZPSE_D_SZ_COMB      0x02000u
#define ZPSE_D_SZ_SHUF      0x01000u
#define ZPSE_D_SZ_GUARD     0x0B800u

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
#define ZPSE_E_OFF_CTRL     0x00000u
#define ZPSE_E_OFF_INPUT    0x01000u
#define ZPSE_E_OFF_TSTAGE   0x11000u
#define ZPSE_E_OFF_CAND     0x21000u
#define ZPSE_E_OFF_BEST     0x31000u
#define ZPSE_E_OFF_ORDER1   0x41000u
#define ZPSE_E_OFF_O2       0x41800u
#define ZPSE_E_OFF_COMB     0x61800u
#define ZPSE_E_OFF_HIST     0x63800u
#define ZPSE_E_OFF_SHUF     0x64800u
#define ZPSE_E_OFF_GUARD    0x65800u

#define ZPSE_E_SZ_CTRL      0x01000u
#define ZPSE_E_SZ_INPUT     0x10000u
#define ZPSE_E_SZ_TSTAGE    0x10000u
#define ZPSE_E_SZ_CAND      0x10000u
#define ZPSE_E_SZ_BEST      0x10000u
#define ZPSE_E_SZ_ORDER1    0x00800u
#define ZPSE_E_SZ_O2        0x20000u
#define ZPSE_E_SZ_COMB      0x02000u
#define ZPSE_E_SZ_HIST      0x01000u
#define ZPSE_E_SZ_SHUF      0x01000u
#define ZPSE_E_SZ_GUARD     0x0A800u

typedef struct ZpseDecoder {
    uint8_t *arena;       /* base, 64-B aligned, 320 KiB */
    uint8_t *recon;       /* transformed reconstruction (64 KiB) */
    uint8_t *out_stage;   /* output staging (64 KiB) */
    uint8_t *order1;      /* 2 KiB */
    uint8_t *o2;          /* 128 KiB: 8 x 4096 {value, confidence, gen} */
    uint8_t *comb;        /* 8 KiB combinadic scratch */
    uint8_t *shuf;        /* 4 KiB shuffle/CRC consts */
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
    uint8_t *arena;       /* base, 64-B aligned, 448 KiB */
    uint8_t *input;       /* 64 KiB */
    uint8_t *tstage;      /* 64 KiB */
    uint8_t *cand;        /* 64 KiB candidate staging */
    uint8_t *best;        /* 64 KiB best staging */
    uint8_t *order1;      /* 2 KiB */
    uint8_t *o2;          /* 128 KiB: 8 x 4096 {value, confidence, gen} */
    uint8_t *comb;        /* 8 KiB */
    uint8_t *hist;        /* 4 KiB histogram/palette scratch */
    uint8_t *shuf;        /* 4 KiB shuffle LUT + CRC consts */
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
} ZpseEncoder;

int zpse_decoder_init(ZpseDecoder *d, void *arena_mem, size_t arena_len);
int zpse_encoder_init(ZpseEncoder *e, void *arena_mem, size_t arena_len);
int zpse_arena_is_aligned(const void *p);

#endif /* ZPSE_ARENA_H */
