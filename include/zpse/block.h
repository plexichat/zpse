#ifndef ZPSE_BLOCK_H
#define ZPSE_BLOCK_H

/* Block encode/decode. Envelope:
 * [32-B header][meta][segdir][payloads]. Includes RAW fallback. */

#include <stddef.h>
#include <stdint.h>
#include "arena.h"

/* Decode one block. in_len must be >= compressed_block_size taken from
 * the header; output must hold uncompressed_size bytes. Returns bytes
 * written (== uncompressed_size) or negative PseStatus. */
int zpse_block_decode(ZpseDecoder *d, const uint8_t *in, uint32_t in_len,
                      uint8_t *output, uint32_t out_cap,
                      uint32_t *out_len_out);

/* Encode one block (<= 65536 B). Selects transform + per-segment tiles,
 * falls back to RAW_BLOCK when compression does not pay. Returns block
 * bytes written or negative status. seq is block_sequence_number. */
int zpse_block_encode(ZpseEncoder *e, const uint8_t *in, uint32_t n,
                      uint32_t seq, int is_last,
                      uint8_t *out, uint32_t out_cap);

/* Header-only helpers (validation without full decode). */
int zpse_block_probe(const uint8_t *in, uint32_t in_len,
                     uint32_t *compressed_size_out,
                     uint32_t *uncompressed_size_out);

/* ---- Page dictionary ---------------------------------------------------- *
 *
 * A page store writes the same page more than once. 16.4% of the heap and
 * library dump pages this codec is measured on are byte-for-byte copies of a
 * page already written, and the copies are scattered, so only content
 * addressing finds them (docs/page-dictionary.md has the measurements and the
 * call sequence).
 *
 * None of it is part of the stream format: a hit is an ordinary block that a
 * decoder reads without knowing where it came from. The index is the caller's
 * memory because a page's encoded bytes are the store's -- an entry that held
 * them would be hundreds of megabytes, and that is the store's business, not
 * the encoder's. What the codec supplies is the content hash and the header
 * rewrite a block needs before it can stand at a new position. */

/* Caller-owned index, direct mapped: 20 bytes an entry, so 65,536 entries is
 * 1.25 MiB. A tag of 0/0 means "empty", so insertion maps it to 1/0. */
typedef struct ZpsePageDict {
    uint64_t *tag_lo;   /* caller memory: entries * 8 */
    uint64_t *tag_hi;   /* caller memory: entries * 8 */
    uint32_t *slot;     /* caller memory: entries * 4, the caller's own id */
    uint32_t entries;   /* power of two, at least 16 */
} ZpsePageDict;

/* 128-bit content tag of `n` bytes. 128 and not 64 on purpose: a tag
 * collision has to be a rounding error, not a wrong page, because nothing
 * downstream compares the decompressed bytes. `n == 0` or a NULL page gives
 * 0/0, which no lookup matches. */
void zpse_page_hash(const uint8_t *page, uint32_t n,
                    uint64_t *lo_out, uint64_t *hi_out);

/* Size the index. Zeroes it, so it starts empty. entries must be a power of
 * two of at least 16 for anything to be found; a smaller or non-power-of-two
 * `entries` leaves the dictionary permanently missing. */
void zpse_pagedict_init(ZpsePageDict *d, uint64_t *tag_lo, uint64_t *tag_hi,
                        uint32_t *slot, uint32_t entries);

/* 1 when (lo, hi) is present and its slot was stored; 0 on a miss, on an
 * empty dictionary, or when either pointer is NULL. */
int zpse_pagedict_find(const ZpsePageDict *d, uint64_t lo, uint64_t hi,
                       uint32_t *slot_out);

/* Record (lo, hi) -> slot. Direct mapped: the entry the tag's hash selects is
 * overwritten, and the slot it named is not freed or referenced again. */
void zpse_pagedict_insert(ZpsePageDict *d, uint64_t lo, uint64_t hi,
                          uint32_t slot);

/* Retag a stored block for a new position: rewrite block_sequence_number and
 * the is-last flag, in place, in whichever header form the block carries (the
 * 32-byte one or a page's 16-byte one), fixing the header CRC. The payload is
 * not touched: it decodes to the page's bytes, which are unchanged, so the
 * payload checksum a block is written with still holds.
 *
 * The block must be a complete block written by this library -- block_len is
 * what its header says its size is. Returns PSE_OK, PSE_BAD_ARG for a NULL
 * block, PSE_TRUNCATED for a block_len below a page header, PSE_BAD_HEADER
 * for a length that disagrees with the header or a header that is not one,
 * and PSE_BAD_HEADER_CRC when the header CRC does not match. Retagging a
 * block whose header already says (seq, is_last) is a no-op that still
 * returns PSE_OK. */
int zpse_block_retag(uint8_t *block, uint32_t block_len,
                     uint32_t seq, int is_last);

#endif /* ZPSE_BLOCK_H */
