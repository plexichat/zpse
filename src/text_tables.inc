/* Constants for the finder-free 7-bit pack payload codec in text.c.
 *
 * The static-BPE tables this file used to hold are gone. The text form now
 * packs the low seven bits of every ASCII byte into a canonical group-of-8
 * stream, the same bit layout as src/tile.c's tile_ascii_compact8 /
 * tile_ascii_spread8. What remains here is the two pshufb masks the AVX2
 * bodies need to move seven-byte groups across 128-bit lane boundaries. */
#ifndef ZPSE_TEXT_TABLES_INC
#define ZPSE_TEXT_TABLES_INC

/* Applied to each 128-bit lane AFTER compact8. Each 64-bit lane then holds
 * seven packed bytes in bytes 0..6 and a zero in byte 7; this drops both
 * holes so the four groups' 28 payload bytes land contiguously. 0xFF zeroes
 * the two resulting don't-care positions. */
static const uint8_t ztext_shuf_compact[16] = {
     0,  1,  2,  3,  4,  5,  6,  8,
     9, 10, 11, 12, 13, 14, 255, 255
};

/* Applied to each 128-bit lane BEFORE spread8. Fourteen packed bytes (two
 * groups of seven) become two 64-bit lanes with a zero hole in byte 7 and
 * byte 15, i.e. one group in each lane's low 56 bits. */
static const uint8_t ztext_shuf_expand[16] = {
     0,  1,  2,  3,  4,  5,  6, 255,
     7,  8,  9, 10, 11, 12, 13, 255
};

#endif /* ZPSE_TEXT_TABLES_INC */
