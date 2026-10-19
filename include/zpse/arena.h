#ifndef ZPSE_ARENA_H
#define ZPSE_ARENA_H

/* Fixed arena layouts (SPEC_V1.md Sec. 12). Caller-owned, 64-byte
 * aligned, zeroed at context creation. No allocation afterwards. */

#include <stddef.h>
#include <stdint.h>

#define ZPSE_DEC_ARENA_SIZE (256u * 1024u) /* 0x40000 */
#define ZPSE_ENC_ARENA_SIZE (384u * 1024u) /* 0x60000 */
#define ZPSE_ARENA_ALIGN    64u

/* Decoder offsets. */
#define ZPSE_D_OFF_CTRL     0x00000u
#define ZPSE_D_OFF_RECON    0x01000u
#define ZPSE_D_OFF_OUT      0x11000u
#define ZPSE_D_OFF_ORDER1   0x21000u
#define ZPSE_D_OFF_O2VAL    0x21800u
#define ZPSE_D_OFF_O2CONF   0x29800u
#define ZPSE_D_OFF_COMB     0x31800u
#define ZPSE_D_OFF_SHUF     0x33800u
#define ZPSE_D_OFF_GUARD    0x34800u

#define ZPSE_D_SZ_CTRL      0x01000u
#define ZPSE_D_SZ_RECON     0x10000u
#define ZPSE_D_SZ_OUT       0x10000u
#define ZPSE_D_SZ_ORDER1    0x00800u
#define ZPSE_D_SZ_O2VAL     0x08000u
#define ZPSE_D_SZ_O2CONF    0x08000u
#define ZPSE_D_SZ_COMB      0x02000u
#define ZPSE_D_SZ_SHUF      0x01000u
#define ZPSE_D_SZ_GUARD     0x0B800u

/* Encoder offsets. */
#define ZPSE_E_OFF_CTRL     0x00000u
#define ZPSE_E_OFF_INPUT    0x01000u
#define ZPSE_E_OFF_TSTAGE   0x11000u
#define ZPSE_E_OFF_CAND     0x21000u
#define ZPSE_E_OFF_BEST     0x31000u
#define ZPSE_E_OFF_ORDER1   0x41000u
#define ZPSE_E_OFF_O2VAL    0x41800u
#define ZPSE_E_OFF_O2CONF   0x49800u
#define ZPSE_E_OFF_COMB     0x51800u
#define ZPSE_E_OFF_HIST     0x53800u
#define ZPSE_E_OFF_SHUF     0x54800u
#define ZPSE_E_OFF_GUARD    0x55800u

#define ZPSE_E_SZ_CTRL      0x01000u
#define ZPSE_E_SZ_INPUT     0x10000u
#define ZPSE_E_SZ_TSTAGE    0x10000u
#define ZPSE_E_SZ_CAND      0x10000u
#define ZPSE_E_SZ_BEST      0x10000u
#define ZPSE_E_SZ_ORDER1    0x00800u
#define ZPSE_E_SZ_O2VAL     0x08000u
#define ZPSE_E_SZ_O2CONF    0x08000u
#define ZPSE_E_SZ_COMB      0x02000u
#define ZPSE_E_SZ_HIST      0x01000u
#define ZPSE_E_SZ_SHUF      0x01000u
#define ZPSE_E_SZ_GUARD     0x0A800u

typedef struct ZpseDecoder {
    uint8_t *arena;       /* base, 64-B aligned, 256 KiB */
    uint8_t *recon;       /* transformed reconstruction (64 KiB) */
    uint8_t *out_stage;   /* output staging (64 KiB) */
    uint8_t *order1;      /* 2 KiB */
    uint8_t *o2val;       /* 32 KiB */
    uint8_t *o2conf;      /* 32 KiB */
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
    uint8_t *arena;       /* base, 64-B aligned, 384 KiB */
    uint8_t *input;       /* 64 KiB */
    uint8_t *tstage;      /* 64 KiB */
    uint8_t *cand;        /* 64 KiB candidate staging */
    uint8_t *best;        /* 64 KiB best staging */
    uint8_t *order1;      /* 2 KiB */
    uint8_t *o2val;       /* 32 KiB */
    uint8_t *o2conf;      /* 32 KiB */
    uint8_t *comb;        /* 8 KiB */
    uint8_t *hist;        /* 4 KiB histogram/palette scratch */
    uint8_t *shuf;        /* 4 KiB shuffle LUT + CRC consts */
    /* Trial-level progress for zpse_segment_encode's flat abort: the length
     * of the leading run of tiles whose record cost at least as much as the
     * bytes it codes, and the flag that ends that run (set by the first tile
     * that does pay for itself). Accumulated across the segments of one
     * trial and reset by the caller per trial. */
    uint32_t flat_covered;
    uint32_t flat_paid;
} ZpseEncoder;

int zpse_decoder_init(ZpseDecoder *d, void *arena_mem, size_t arena_len);
int zpse_encoder_init(ZpseEncoder *e, void *arena_mem, size_t arena_len);
int zpse_arena_is_aligned(const void *p);

#endif /* ZPSE_ARENA_H */
