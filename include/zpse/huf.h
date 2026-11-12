#ifndef ZPSE_HUF_H
#define ZPSE_HUF_H

/* Huffman coding of an intra-block LZ payload (block flag ZPSE_B_HUF, flag
 * pattern 110). See the ZPSE_B_HUF comment in constants.h for the flag
 * choice; this header is the payload format and the codec's contract.
 *
 * ---------------------------------------------------------------------------
 * Why a Huffman form at all
 * ---------------------------------------------------------------------------
 * The LZ forms leave the last few percent of every block on the table. A
 * token stream is bytes: a match-length nibble of 0 and a literal nibble of
 * 3 are the same eight bits to a reader, and on prose the distribution of
 * those bytes is nothing like flat. Huffman coding the payload is the oldest
 * answer to that, it costs one pass on each side, and it composes with any LZ
 * form because it wraps the stream rather than replacing it.
 *
 * What it is NOT is a replacement for the match finder. It runs on the bytes
 * an LZ form already wrote, so it can only ever take a block the LZ form
 * already won; a block whose tile stream beat LZ stays a tile stream.
 *
 * ---------------------------------------------------------------------------
 * Payload layout
 * ---------------------------------------------------------------------------
 *   u8  table_id             0..3: which of the four fixed tables
 *   u8  inner_form           ZPSE_B_LZ, ZPSE_B_LZ2 or ZPSE_B_LZ_HEX
 *   u32 inner_len            decoded length of the wrapped LZ payload
 *   u16 seg0, seg1, seg2     encoded byte lengths of streams 0, 1 and 2
 *   stream 0 .. stream 3     the four Huffman streams, back to back
 *
 * Twelve bytes of overhead and no table, because the tables are part of the
 * format rather than part of the block (see below). Stream 3's length is
 * inner_len's remainder and is not written down: the decoder knows the
 * payload runs to the end of the block.
 *
 * The block's envelope is the LZ forms' own -- transform_id 0, segment_count
 * 1, no transform metadata -- and the decoder requires it. `inner_form` says
 * which LZ decoder the unpacked payload belongs to; the flag byte says one
 * form and the payload says which stream is inside it.
 *
 * ---------------------------------------------------------------------------
 * The streams
 * ---------------------------------------------------------------------------
 * inner_len is split into four pieces at byte boundaries: share = inner_len/4
 * for streams 0, 1 and 2, and the remainder for stream 3 (so 0 to 3 bytes
 * more). Each piece is coded on its own, so the four bit streams are
 * independent and the decoder interleaves them -- that is the whole point of
 * the split. Serial bit-by-bit decoding has a loop-carried dependency of one
 * load, one shift and one store per symbol; four independent streams give the
 * core four of them to interleave, which is the parallelism this codec has
 * instead of SIMD lanes. Huffman bits are inherently serial -- one symbol's
 * length decides where the next one starts -- so an AVX2 kernel here would
 * have to find its width somewhere other than a lane, and there is none to
 * find: the four streams are that width, in scalar registers.
 *
 * Bit order is LSB-first: the first bit of the stream is bit 0 of the first
 * byte. A code is written by ORing its bits into a shift register whose low
 * end is the next bit out. The codes stored in the tables are therefore the
 * canonical Huffman codes with their bits REVERSED -- canonical codes are
 * prefix-free read from the top, and reading from the bottom needs them
 * suffix-free, which reversing makes them. That reversal is done once, by
 * tests/benchmarks/huf_train.c, so neither the encoder nor the decoder pays
 * for it.
 *
 * A stream's last byte is zero-padded to a byte boundary. The decoder decodes
 * exactly `share` symbols from each stream and then requires that it consumed
 * all but the last byte's worth of the stream's bits: a stream that runs out
 * early, or leaves more than seven bits over, is refused.
 *
 * ---------------------------------------------------------------------------
 * The tables, and why they are in the format
 * ---------------------------------------------------------------------------
 * Four tables, 11-bit lookup, 256 symbols each, trained offline on the
 * payloads the LZ forms actually produce (tests/benchmarks/huf_train.c) and
 * checked in as src/huf_tables.inc. The encoder picks one per block by
 * pricing all four against a strided sample of the block's bytes -- 4 x 256
 * multiply-adds -- and writes its index. The sample ranks the four books,
 * which is all the choice needs; the exact stream lengths come from the coding
 * pass, not from the sample.
 *
 * The alternative, a table per block, is what zstd's HUF does and it is the
 * right answer when blocks are independent and readers are interchangeable.
 * It is the wrong answer here for two reasons. The first is the rule this
 * codebase is built on: a block's bytes must be the same on every machine, at
 * every effort level and under every compiler, and a table built from the
 * block's own histogram is a second place a floating-point-free but still
 * subtle piece of code has to agree with itself -- the four tables are fixed
 * data, so a block's output depends only on its bytes. The second is the
 * decoder: a per-block table costs a table build (a sort over the alphabet)
 * before the first symbol is decoded, on every block, to save a few bytes of
 * header. At a 4 KiB page that trade is the wrong way round.
 *
 * Fixed tables cost: the encoder can miss the ideal code by a couple of
 * points on an unusual block. Measured over the corpora the tables were
 * trained on, the best of four is within 1.4% of a table built from the block
 * itself, and the choice of four rather than one is worth 2.1 points.
 *
 * ---------------------------------------------------------------------------
 * The gate
 * ---------------------------------------------------------------------------
 * The form is kept only when it makes the block at least 1/ZPSE_HUF_DIV
 * smaller than the LZ stream alone, because the decode pays a Huffman pass
 * before the LZ pass. The encoder also requires the inner payload to be at
 * least ZPSE_HUF_MIN_INNER bytes: below that the twelve-byte header and the
 * four-way split cost more than the entropy coding can find.
 */

#include <stddef.h>
#include <stdint.h>

/* Bits of one decode lookup. 11 covers a 256-symbol Huffman code with room to
 * spare (a 256-leaf tree needs 8 bits on average and 11 only for very skewed
 * distributions), so a lookup always yields a whole symbol. One symbol per
 * lookup, not two: see the dtab note below for why two does not fit this
 * data. */
#define ZPSE_HUF_TABLE_LOG   11u
#define ZPSE_HUF_TABLE_SIZE  (1u << ZPSE_HUF_TABLE_LOG)
/* Longest code a table may contain. Clamping to this is what makes the
 * lookup one level deep; the trainer rebalances to stay inside it. */
#define ZPSE_HUF_MAX_LEN     11u
#define ZPSE_HUF_TABLES      4u
#define ZPSE_HUF_SYMBOLS     256u
#define ZPSE_HUF_STREAMS     4u

/* table_id, inner_form, inner_len, and three stream lengths. */
#define ZPSE_HUF_HDR         12u
/* The smallest payload the form can write: the header and one byte per
 * stream, which is what a four-byte inner payload would cost -- far below
 * ZPSE_HUF_MIN_INNER and therefore not reachable, but a bound is a bound. */
#define ZPSE_HUF_MIN_PAYLOAD (ZPSE_HUF_HDR + ZPSE_HUF_STREAMS)
/* Smallest wrapped payload worth coding. */
#define ZPSE_HUF_MIN_INNER   64u
/* Largest wrapped payload. The decoder unpacks into the arena's 64 KiB LZ
 * buffer (ZPSE_D_OFF_LZBUF), so a payload that does not fit there cannot be
 * wrapped; a 64 KiB block's LZ payload is only below this when the block
 * compressed at all, which is the case the form is for. */
#define ZPSE_HUF_MAX_INNER   65536u
/* The form is kept only when it saves at least 1/this of the inner payload. */
#ifndef ZPSE_HUF_DIV
#define ZPSE_HUF_DIV         64u
#endif
/* Deepest rung that offers it without ZPSE_ENC_HUF. MAX is the level whose
 * trade is size over everything; the rungs below it are where the encode time
 * this pass costs is worth more than the bytes. */
#ifndef ZPSE_HUF_MIN_LEVEL
#define ZPSE_HUF_MIN_LEVEL   9u
#endif

/* zpse_huf_ctab[t][s] = (code << 4) | nbits, code bit-reversed, nbits 0 only
 * if the table has no code for the symbol (it always does). */
#define ZPSE_HUF_CTAB_BYTES (ZPSE_HUF_TABLES * ZPSE_HUF_SYMBOLS * 2u)
/* zpse_huf_dtab[t][i] = (nbBits << 8) | symbol: exactly one symbol per
 * lookup. A two-symbol entry needs both codes inside the eleven-bit window,
 * which an LZ payload (about seven bits per symbol) almost never has, so it
 * was measured and dropped -- see src/huf.c. */
#define ZPSE_HUF_DTAB_BYTES (ZPSE_HUF_TABLES * ZPSE_HUF_TABLE_SIZE * 2u)
/* What the tables add to the binary: 2 KiB of codes and 16 KiB of lookups,
 * as read-only data. */
#define ZPSE_HUF_TABLE_BYTES (ZPSE_HUF_CTAB_BYTES + ZPSE_HUF_DTAB_BYTES)

/* Code a payload of n bytes into dst[0..cap). Returns the payload length, or
 * 0 when the arguments are unusable, n is outside
 * [ZPSE_HUF_MIN_INNER, ZPSE_HUF_MAX_INNER], or the encoding does not fit in
 * cap. A 0 is not an error: the caller is comparing sizes, and an encoding
 * that cannot fit is one that cannot win. When cap can hold this payload's
 * worst case the four streams are written before the twelve-byte header, so
 * after a 0 dst may hold anything up to cap; dst is the caller's scratch
 * either way. */
uint32_t zpse_huf_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, uint8_t inner_form);

/* Decode a ZPSE_B_HUF payload of in_len bytes. Writes the wrapped LZ payload
 * to out[0..out_cap), its length to *inner_len_out and its form to
 * *inner_form_out. Returns ZPSE_OK, or
 *   ZPSE_BAD_HUF             the header, the stream lengths or the bit
 *                            accounting do not agree
 *   ZPSE_BAD_BLOCK_SIZE      the payload is too short to be one
 *   ZPSE_BAD_HEADER          inner_form is not an LZ form, or inner_len is
 *                            outside the range the encoder writes
 *   ZPSE_OUTPUT_OVERFLOW     inner_len would not fit in out_cap
 * out is only valid on ZPSE_OK. */
int zpse_huf_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                    uint32_t out_cap, uint32_t *inner_len_out,
                    uint8_t *inner_form_out);

#endif /* ZPSE_HUF_H */
