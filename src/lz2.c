#include "zpse/lz2.h"

#include "zpse/bits.h"
#include "zpse/hot.h"
#include "zpse/status.h"
#include "zpse/vector.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* SSE2 is baseline on every x86 target the build supports. The encoder's
 * forward extension uses one 16-byte compare where the scalar form loads and
 * xors two eight-byte words; it stops at the first mismatch with the exact
 * byte offset, so the length it returns is the scalar one. */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#define ZPSE_LZ2_HAVE_SSE2 1
#include <emmintrin.h>
#else
#define ZPSE_LZ2_HAVE_SSE2 0
#endif

/* Encode-for-decode (PUSH) configuration, on by default: hb17 + five-byte
 * hash + gallop step 3 (skip bits 3) + backward walk + reduced lazy probing
 * (PROBE_EARLY off, so no early speculative probe; LAZY_MAX=4 bounds the
 * probe to the common 4-byte matches -- probing is cut, not eliminated). The
 * caller's table is 64 KiB (ZPSE_E_SZ_CAND), enough for 15 hash bits, so 17
 * bits needs its own scratch; a thread-local array keeps simultaneous
 * encoders independent. Everything else in the file -- in particular the
 * wire format and the decoder -- is unchanged, so every configuration
 * decodes with the same shipped decoder.
 *
 * LZE1 measured, best of fifteen on 64 KiB Silesia blocks: default
 * 101 365 503 B at 403 MB/s, LZ_FAST 98 257 260 B at 387 MB/s, PUSH
 * 100 418 128 B (0.94% smaller than default) at ~530 MB/s in a build with
 * ~15% runtime-knob overhead, i.e. about 1.37x the shipped rate. The
 * five-byte hash is what makes dropping the lazy probe pay: it recovers the
 * ~1.7 MB the probe used to save.
 *
 * PUSH became the default and SKIP_BITS went 4 -> 3 on this measurement:
 * over all twelve Silesia files at 64 KiB blocks the lz2 kernel gains ~5%
 * (best-of passes, weighted by blocks) for +1.6% payload; at frame level the
 * incompressible pages go RAW, so the gain dilutes to ~1-2% encode for
 * +0.3% size (level 5 105 115 011 -> 105 495 055 B, level 6 101 591 982
 * -> 102 129 095 B, level 9 91 691 958 -> 93 506 764 B, worst +1.98% at
 * level 9 -- inside the 3% budget). */
#ifndef ZPSE_LZ2_PUSH
#define ZPSE_LZ2_PUSH 1
#endif

/* Finder-only minimum match. The wire format always uses ZPSE_LZ2_MIN_MATCH
 * (4) as its length bias, so raising this only makes the finder leave short
 * matches as literals; the shipped decoder is unchanged. Default 4 is today's
 * behaviour: a match is at least four bytes, so the gate below never fires.
 * Build with -DZPSE_LZ2_FIND_MIN=5 for the encode-for-decode variant.
 *
 * LZF2 measured 5 on 3,227 64 KiB Silesia blocks: commands per block fall
 * 5189 -> 4150 (-20.03%), compressed size +0.373%, decode +11.7% (range
 * +10.3..+15.6) through the unchanged decoder, encode -15.7%. */
#ifndef ZPSE_LZ2_FIND_MIN
#define ZPSE_LZ2_FIND_MIN 4u
#endif

#if ZPSE_LZ2_PUSH
#define ZPSE_LZ2_PUSH_HASH_BITS 17u
#undef ZPSE_LZ2_SKIP_BITS
#define ZPSE_LZ2_SKIP_BITS 3u
#undef ZPSE_LZ2_HASH5
#define ZPSE_LZ2_HASH5 1
/* Four does not disable the probe: the gate is len <= LAZY_MAX and len starts
 * at 4, so 4-byte matches -- the most common case -- still take the bounded
 * probe below. PROBE_EARLY=0 removes only the early speculative probe (its two
 * loads); the probe itself still runs on those 4-byte matches. */
#undef ZPSE_LZ2_LAZY_MAX
#define ZPSE_LZ2_LAZY_MAX 4u
#undef ZPSE_LZ2_PROBE_EARLY
#define ZPSE_LZ2_PROBE_EARLY 0

#if defined(_MSC_VER)
#define ZPSE_LZ2_TLS __declspec(thread)
#else
#define ZPSE_LZ2_TLS _Thread_local
#endif
static ZPSE_LZ2_TLS uint16_t
    lz2_push_tab[1u << ZPSE_LZ2_PUSH_HASH_BITS];
#endif /* ZPSE_LZ2_PUSH */

/* An empty hash slot. Positions are 16-bit because a block is at most
 * 64 KiB, and the encoder never inserts a position past n - 4, so 0xFFFF is
 * not a legitimate entry. */
#define LZ2_EMPTY 0xFFFFu

/* Hash slot access: one uint16_t array of caller-owned scratch, cleared per
 * call. LZ2_EMPTY is the marker for a slot that holds no position. */
#define LZ2_TAB_GET(h) ((tab)[(h)])
#define LZ2_TAB_SET(h, v) do { (tab)[(h)] = (uint16_t)(v); } while (0)

/* The payload header: literal total and token count. */
#define LZ2_HDR 8u

/* The most bytes one command can write: the token, a one-byte escape for the
 * literal count, the run itself (at most ZPSE_LZ2_MAX_LIT), the two-byte
 * distance and a one-byte escape for the match length. The writer tests for
 * that much room once instead of working out the exact size, and falls back
 * to the exact sum only at the cap boundary. */
#define LZ2_CMD_ROOM 260u

/* Output room the decode loop's trusted region needs before it takes a
 * command. The old 38-byte reserve covered only the 16-byte literal store
 * plus the eight-byte match steps of a single-nibble command; the tight loop
 * now trusts a command's escapes too, so the reserve has to cover the widest
 * command a corrupt byte can name: 270 literals (escape 255), a 274-byte
 * match, and the seven bytes the short-offset pattern copy rounds up. That
 * is 551, rounded up to 560. The reserve is what keeps every speculative
 * store inside the block -- a frame decoder hands a block exactly the room
 * its bytes need and decodes blocks side by side, so a store past the end
 * would land in a neighbour's output. The last 560 bytes always take the
 * fully checked tail. */
#define ZPSE_LZ2_FAST_ROOM 560u

/* Input back-off for the trusted loop's lp. The inline literal wildcopy (E)
 * copies whole 32-byte groups, so a 270-byte literal run reads 288 bytes from
 * lp; the two escape bytes that can precede the run bring it to 290. 296 keeps
 * every speculative input read at or before tend. The old 280 sufficed when
 * literals were copied with one exact-sized memcpy. */
#define LZ2_FAST_INPUT_BACK 296u

/* How fast the finder gallops once a run of positions has found nothing: the
 * step is 1 + (misses >> LZ2_SKIP_BITS). A parameter so the trade can be
 * measured -- a smaller shift finds more matches (better ratio) and costs
 * probes on data that has none -- and the default is the value the sweep of
 * 5..9 chose, see the README. */
#ifndef ZPSE_LZ2_SKIP_BITS
#define ZPSE_LZ2_SKIP_BITS 5u
#endif

/* Lazy matching: the longest match the finder will still spend a second
 * probe on to see whether the next byte starts a longer one. 4 disables the
 * probe (the match is at least four bytes, so nothing passes the test).
 *
 * This is what closes the size gap to the token form, which has always
 * looked ahead: a match that starts one byte later and runs longer is worth
 * more than the one in hand, because the byte given up becomes the first
 * byte of the next literal run rather than a byte of the match. Past a few
 * dozen bytes the probe almost never pays -- a long match at one position
 * is a long match at the next -- so the ceiling keeps the cost on the short
 * matches where the decision actually differs, which are also the ones the
 * probe can settle with a single xor. A parameter so the trade can be
 * measured; see the README for the sweep that chose the value.
 *
 * 12 rather than the 24 this shipped with first: the probe is a dependent
 * chain of two loads and on prose most matches are shorter than either
 * ceiling, so the cost is nearly flat in the ceiling while the ratio gain
 * is not. Measured on whole 64 KiB blocks, best of four alternating runs:
 * dickens 259.7 -> 267.7 MB/s (payload unchanged, 40474 bytes), webster
 * 307.9 -> 319.3 for +0.2%, mozilla 396.0 -> 408.1 for +0.15%, xml 666.2 ->
 * 695.4 for +0.4%. Below 12 the rate stops moving and the size starts to:
 * at 8 webster gains another 1.5% of rate for 0.4% of size, which is the
 * wrong way round. */
#ifndef ZPSE_LZ2_LAZY_MAX
#define ZPSE_LZ2_LAZY_MAX 12u
#endif

/* Whether the finder's key is five bytes rather than four.
 *
 * A four-byte key collides whenever two positions agree on four bytes and
 * differ on the fifth -- on prose that is a whole word family (" the", " ther",
 * " them"), so the slot keeps being overwritten by near-misses and the loop
 * pays a candidate load to find out. lz4 hashes five bytes on 64-bit builds for
 * this reason (`LZ4_hash5`). Here the extra byte is read from the block (the
 * four-byte key is already in a register), folded into a 64-bit multiply, and
 * the low `hash_bits` of the high half are the slot.
 *
 * Positions with fewer than five bytes left keep the four-byte key: the last
 * four of a block are never worth a guard on every iteration. A position
 * stored under one key and looked up under the other is simply a miss, so the
 * streams stay decodable -- what changes is which positions the finder offers,
 * and that is what the measurement is for. */
#ifndef ZPSE_LZ2_HASH5
/* Measured and refused: 0 ships.
 *
 * The idea is lz4's (`LZ4_hash5`): a five-byte key separates words that share
 * their first four bytes, so the slot stops being overwritten by near-misses
 * and the loop skips a candidate load. On whole 64 KiB Silesia blocks it is
 * not a win, and it is not consistent either -- 12 files, one pass each, the
 * payload moves `mr` 31 146 -> 26 418 and `nci` 12 816 -> 12 583 and `xml`
 * 25 449 -> 24 753, and moves `sao` 60 026 -> 63 491, `x-ray` 59 735 ->
 * 63 173 and `mozilla` 28 783 -> 29 488, for 383 015 bytes against 369 970.
 * The four-byte key is what the size corpus was built around and the near-miss
 * collisions it keeps are cheaper than the matches it finds. */
#define ZPSE_LZ2_HASH5 0
#endif

static uint32_t lz2_hash(uint32_t v, uint32_t hash_bits);

static inline uint32_t lz2_hash_at(const uint8_t *in, uint32_t pos, uint32_t n,
                                  uint32_t cur, uint32_t hash_bits)
{
#if ZPSE_LZ2_HASH5
    if (pos + 5u <= n) {
        uint64_t v = (uint64_t)cur | ((uint64_t)in[pos + 4u] << 32);

        return (uint32_t)((v * UINT64_C(0x9E3779B97F4A7C15)) >>
                          (64u - hash_bits));
    }
#else
    (void)in;
    (void)pos;
    (void)n;
#endif
    return lz2_hash(cur, hash_bits);
}

static uint32_t lz2_hash(uint32_t v, uint32_t hash_bits)
{
    /* One multiply at the cursor, top bits kept: the four-byte compare at the
     * candidate is the tag check, so no tag has to be stored. */
    return (v * UINT32_C(2654435761)) >> (32u - hash_bits);
}

static uint64_t lz2_load64(const uint8_t *p)
{
    uint64_t v;

    memcpy(&v, p, 8u);
    return v;
}

/* AVX2 forward extension for the main match, ported from src/lz.c's
 * lz_extend_avx2. Same guard and per-file macro names: the unity build puts
 * lz.c and this file in one translation unit. The first mismatching byte is
 * the lowest clear bit of the equality mask, which is exactly the offset the
 * eight-byte xor loop below finds with ctz on the xor -- so the resulting
 * length is bit-for-bit the scalar one. */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#if defined(__GNUC__) && !defined(_MSC_VER)
#define ZPSE_LZ2_AVX2 __attribute__((target("avx2")))
#define ZPSE_LZ2_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#elif defined(__AVX2__)
#define ZPSE_LZ2_AVX2
#define ZPSE_LZ2_HAVE_AVX2 ZPSE_AVX2_ABI_OK
#else
#define ZPSE_LZ2_HAVE_AVX2 0
#endif
#else
#define ZPSE_LZ2_HAVE_AVX2 0
#endif

#if ZPSE_LZ2_HAVE_AVX2
#include <immintrin.h>

/* Only needed on a target without SSE2: where SSE2 exists the encoder uses
 * lz2_extend_sse2 directly and this kernel would be dead code. */
#if !ZPSE_LZ2_HAVE_SSE2
static ZPSE_LZ2_AVX2 uint32_t
lz2_extend_avx2(const uint8_t *in, uint32_t a, uint32_t b, uint32_t len,
                uint32_t n)
{
    while (a + len + 32u <= n && b + len + 32u <= n) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(in + a + len));
        __m256i y = _mm256_loadu_si256((const __m256i *)(in + b + len));
        uint32_t eq = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y));

        if (eq != 0xFFFFFFFFu) {
            return len + (uint32_t)zpse_ctz64(~(uint64_t)eq);
        }
        len += 32u;
    }
    return len;
}
#endif /* !ZPSE_LZ2_HAVE_SSE2 */
#endif /* ZPSE_LZ2_HAVE_AVX2 */

#if ZPSE_LZ2_HAVE_SSE2
/* The main match extension, 16 bytes at a time from the four-byte starting
 * length. Replaces the out-of-line AVX2 call in the common short-match case:
 * the 32-byte kernel is only worth its call when the match is long enough to
 * reach a second step, and most matches are not. The movemask/ctz reports the
 * first differing byte, so this returns exactly the scalar length.
 *
 * B16 in the LZE3 measurement: compressed bytes unchanged on the corpus
 * (1 728 918 / 637 393), and, paired with the safe lazy gate below, 1.04x
 * all-files / 1.04x prose encode rate. A safe gate on its own is slower; the
 * two together are the win. */
static uint32_t lz2_extend_sse2(const uint8_t *in, uint32_t a, uint32_t b,
                                uint32_t len, uint32_t n)
{
    while (a + len + 16u <= n) {
        __m128i x = _mm_loadu_si128((const __m128i *)(in + a + len));
        __m128i y = _mm_loadu_si128((const __m128i *)(in + b + len));
        uint32_t eq = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(x, y));

        if (eq != 0xFFFFu) {
            return len + zpse_ctz32((~eq) & 0xFFFFu);
        }
        len += 16u;
    }
    while (a + len + 8u <= n) {
        uint64_t x = lz2_load64(in + a + len) ^ lz2_load64(in + b + len);

        if (x != 0u) {
            return len + (zpse_ctz64(x) >> 3);
        }
        len += 8u;
    }
    while (a + len < n && in[a + len] == in[b + len]) {
        len++;
    }
    return len;
}
#endif /* ZPSE_LZ2_HAVE_SSE2 */

/* Every match distance is coded as two little-endian bytes, 1..65535, with
 * zero reserved for a literal-only command. This is the shape lz4 writes and
 * why its decoder is a tight fixed-width loop: no rep slots to test, no
 * cache to move, no escape for a distance whose low byte happens to be high.
 *
 * A four-entry repeat cache used to sit here, coding a recent distance as one
 * byte (0xFC | slot). Measured (DEC1, in-memory, 64 KiB Silesia blocks, quiet
 * core): removing it is +29% decode (2.59 -> 3.33 GB/s mean) for +1.57%
 * compressed size, and on seven of twelve files the size cost is under 0.2%
 * because the finder rarely names a recent distance on prose. The rep branch
 * also forced a match whose low byte was 0xFC..0xFF to become a literal; that
 * restriction disappears with the cache, so the full 16-bit range is now
 * usable. The decoder's only distance work is one 16-bit little-endian load. */

/* One command: the literal run before the match, the match length and the
 * distance. `off == 0` writes a literal-only command, which is how a run
 * longer than one token and a trailing run are coded. Returns 0 when the
 * command does not fit in cap. */
static int lz2_put(uint8_t *dst, uint32_t cap, uint32_t *out,
                   const uint8_t *lit, uint32_t run, uint32_t off,
                   uint32_t len, uint32_t *lit_total,
                   uint32_t *cmds)
{
    uint32_t ml = len >= ZPSE_LZ2_MIN_MATCH ? len - ZPSE_LZ2_MIN_MATCH : 0u;
    uint32_t p = *out;
    uint8_t *t;

    if (LZ2_CMD_ROOM > cap - p) {
        uint32_t need = 1u + run + 2u + (run >= 15u ? 1u : 0u) +
                        (ml >= 15u ? 1u : 0u);

        if (need > cap - p) {
            return 0;
        }
    }
    t = dst + p;
    *t++ = (uint8_t)(((run < 15u ? run : 15u) << 4) |
                     (ml < 15u ? ml : 15u));
    if (run >= 15u) {
        /* One escape byte: the chunker never asks for more than 255. */
        *t++ = (uint8_t)(run - 15u);
    }
    if (run != 0u) {
        memcpy(t, lit, run);
        t += run;
        *lit_total += run;
    }
    /* The distance is always two little-endian bytes; zero names a
     * literal-only command. No rep slots, no escapes. */
    *t++ = (uint8_t)(off & 0xFFu);
    *t++ = (uint8_t)((off >> 8) & 0xFFu);
    if (ml >= 15u) {
        *t++ = (uint8_t)(ml - 15u);
    }
    *out = (uint32_t)(t - dst);
    (*cmds)++;
    return 1;
}

/* Forward extension for the lazy probe only: the main loop's copy is inline
 * because it runs at every match, and this one runs once per candidate the
 * probe asks about. Same shape, so a length means the same thing on both
 * sides of the decision. `max_len` caps the walk: the caller only asks
 * whether the match is strictly longer than the one in hand, so bytes past
 * the cap cannot change the answer -- a mismatch is reported at its exact
 * offset by ctz, and reaching the cap means every byte before it matched. */
static uint32_t lz2_extend(const uint8_t *in, uint32_t n, uint32_t a,
                           uint32_t b, uint32_t len, uint32_t max_len)
{
    while (len < max_len && a + len + 8u <= n) {
        uint64_t x = lz2_load64(in + a + len) ^ lz2_load64(in + b + len);

        if (x != 0u) {
            return len + (zpse_ctz64(x) >> 3);
        }
        len += 8u;
    }
    while (len < max_len && a + len < n && in[a + len] == in[b + len]) {
        len++;
    }
    return len;
}

/* Whether the lazy probe's two dependent loads are issued before the
 * backward walk and the forward extension rather than after them.
 *
 * The probe is a chain -- hash the four bytes at pos + 1, load the table
 * entry that hash names, then load the four bytes at that entry -- and both
 * loads are L2 hits whose latency the match extension below does not need.
 * Issued here, the chain overlaps the backward walk and the eight-byte
 * extension; issued where the probe used to load them, it stalls the loop
 * for its whole latency once per match, which is what a third of the
 * finder's rate on prose was.
 *
 * It reads the same two values the probe read before, from the same places
 * and in the same order relative to the table write at the top of the loop,
 * so the decisions -- and the streams -- are identical. The only cost is
 * that the work is wasted when the backward walk moves the match start (the
 * probe only applies when it did not) or when the match turns out longer
 * than ZPSE_LZ2_LAZY_MAX. */
#ifndef ZPSE_LZ2_PROBE_EARLY
#define ZPSE_LZ2_PROBE_EARLY 1
#endif

/* Whether the lazy probe may take its candidate from the slot the loop has
 * just filed for the current position. When the four bytes at pos and at
 * pos + 1 hash to the same slot, that entry IS the cursor, so the probe's
 * candidate is a self-reference at distance one rather than a real earlier
 * occurrence -- a run, where the match is long enough that the ceiling
 * usually refuses the probe anyway.
 *
 * 0 refuses those probes: a self-reference is not a candidate. Measured on
 * whole 64 KiB Silesia blocks against the shipped 1, best of two alternating
 * rounds: `dickens` 285 -> 276 MB/s, `webster` 332 -> 324, `mozilla` 421 ->
 * 415 with its payload 190 bytes LARGER, `nci` 886 -> 876, `osdb` 403 -> 405,
 * `xml` 685 -> 688. So it is neither faster nor the same size -- the probe is
 * right to take a distance-one candidate -- and 1 stays. */
#ifndef ZPSE_LZ2_PROBE_SELF
#define ZPSE_LZ2_PROBE_SELF 1
#endif

/* Whether the finder reads the key at pos + 1, its hash and that hash's entry
 * immediately after filing pos, before the candidate at pos is tested, and
 * carries the tested entry across iterations instead of re-reading it at the
 * top of the loop.
 *
 * The loop is one dependent chain -- load the four bytes at the cursor, hash
 * them, load the entry the hash names, load the four bytes at that entry --
 * and on prose the miss path, where the cursor steps a single byte, is four
 * iterations in five (the average match is 5.4 bytes). Issuing the next
 * position's chain before the candidate test puts it behind the one load the
 * iteration actually waits on instead of in front of the next one, and the
 * lazy probe then reads the same three values instead of issuing them a
 * second time (see ZPSE_LZ2_PROBE_EARLY).
 *
 * The reads are taken after the store above them, so a slot the store has
 * just overwritten reads back exactly what the next iteration would have
 * read, and the stream is unchanged -- the same argument, and the same
 * guarantee, as src/lz.c's ZPSE_LZ_PIPE_NEXT. Set to 0 to build the finder
 * without it, which is what the measurement compares against. */
#ifndef ZPSE_LZ2_PIPE_NEXT
#define ZPSE_LZ2_PIPE_NEXT 1
#endif

/* Does the backward walk into the pending literals step eight bytes at a
 * time? Same walk, same bound (`anchor`): a word is stepped only when all
 * eight of its bytes match, and the byte loop below then resolves the byte a
 * mismatching word stopped on. See src/lz.c's ZPSE_LZ_BACK_WIDE: measured
 * and refused (the loop is nearly always short, so the extra bounds cost more
 * than the word steps save, 3.6-5.5% of LZ encode). 0 is the shipped arm. */
#ifndef ZPSE_LZ2_BACK_WIDE
#define ZPSE_LZ2_BACK_WIDE 0
#endif

/* The one finder body behind both entry points. `sample_n == 0` is a full
 * pass; a nonzero `sample_n` prices a sample and clears only the slots that a
 * position in [0,sample_n) can hash to. Every decision and store is shared. */
static uint32_t lz2_encode_impl(const uint8_t *in, uint32_t n, uint8_t *dst,
                                uint32_t cap, void *table, uint32_t hash_bits,
                                uint32_t sample_n)
{
    uint16_t *restrict tab = (uint16_t *)table;
#if ZPSE_LZ2_PUSH
    /* The caller's table is a 64 KiB array (15 bits). PUSH uses its own
     * thread-local 17-bit scratch for both full and sample passes so a
     * sampled price names exactly the stream the full pass would write. */
    hash_bits = ZPSE_LZ2_PUSH_HASH_BITS;
    tab = lz2_push_tab;
#endif
    uint32_t entries;
    uint32_t pos, anchor, misses;
    uint32_t lit_total = 0u, cmds = 0u, out = LZ2_HDR;
    uint32_t cur, h, limit;
    /* The extension width is a property of the machine, not of the
     * position: ask once per call, as src/lz.c does. A constant 0 where the
     * kernel is compiled out keeps the scalar branch the only one taken. */
    int have_avx2 = 0;
#if ZPSE_LZ2_HAVE_AVX2
    have_avx2 = (ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) ? 1 : 0;
#endif

    if (in == 0 || dst == 0 || tab == 0) {
        return 0u;
    }
    if (n < ZPSE_LZ2_MIN_MATCH + 1u || n > 65536u) {
        return 0u;
    }
    if (cap < ZPSE_LZ2_MIN_PAYLOAD) {
        return 0u;
    }
    entries = 1u << hash_bits;
    /* LZ2_EMPTY is 0xFFFF and the table is uint16_t, so all-ones bytes are
     * exactly the empty marker. A full pass clears it (memset beats the
     * 16-bit fill loop gcc does not vectorise at -O2); a sample pass clears
     * only the slots a position in [0, sample_n) can hash to. The hash is a
     * key hash, so those slots are found by running the same hash over each
     * key in the sample. See src/lz.c's clear. */
    {
        uint32_t ce = (sample_n != 0u && sample_n < n) ? sample_n : n;

        if (sample_n == 0u) {
            memset(tab, 0xFF, (size_t)entries * sizeof *tab);
        } else if (ce >= ZPSE_LZ2_MIN_MATCH) {
            uint32_t p;

            for (p = 0u; p + ZPSE_LZ2_MIN_MATCH <= ce; p++) {
                uint32_t k = zpse_load_u32le(in + p);

                LZ2_TAB_SET(lz2_hash_at(in, p, n, k, hash_bits), LZ2_EMPTY);
            }
        }
    }
    pos = 0u;
    anchor = 0u;
    misses = 0u;
    limit = n - ZPSE_LZ2_MIN_MATCH;
    cur = zpse_load_u32le(in + pos);
    h = lz2_hash_at(in, pos, n, cur, hash_bits);
#if ZPSE_LZ2_PIPE_NEXT
    /* The entry the top of the loop tests, carried across iterations: the
     * miss path reads it while the candidate is being compared (see
     * ZPSE_LZ2_PIPE_NEXT), so re-reading it at the top would spend the
     * latency the lookahead just saved. */
    uint32_t cand = LZ2_TAB_GET(h);
#endif

    while (pos <= limit) {
#if ZPSE_LZ2_PIPE_NEXT
        uint32_t len = ZPSE_LZ2_MIN_MATCH;
        uint32_t mpos = pos;
        uint32_t mcand = cand;
        uint32_t nx_cur = 0u, nx_h = 0u, nx_c = LZ2_EMPTY;
        int nx_ok = 0;
#else
        uint32_t cand = LZ2_TAB_GET(h);
        uint32_t len = ZPSE_LZ2_MIN_MATCH;
        /* The match's own start: the walk below moves these instead of the
         * scanning cursor (see there). */
        uint32_t mpos = pos;
        uint32_t mcand = cand;
#endif

        LZ2_TAB_SET(h, pos);
#if ZPSE_LZ2_PIPE_NEXT
        /* The next position's chain, read BEFORE the candidate at the cursor
         * is tested: the word at pos + 1 is the byte the cursor is about to
         * step over on a miss, and the slot read is an L1 hit, while what
         * this iteration waits on is the word AT the candidate (see
         * ZPSE_LZ2_PIPE_NEXT). Read after the store above, so a self-hit
         * yields the `pos` just filed, exactly as the next iteration would
         * find it. */
        if (pos + 1u + ZPSE_LZ2_MIN_MATCH <= n) {
            nx_cur = zpse_load_u32le(in + pos + 1u);
            nx_h = lz2_hash_at(in, pos + 1u, n, nx_cur, hash_bits);
            nx_c = LZ2_TAB_GET(nx_h);
            nx_ok = 1;
        }
#endif
        /* A candidate must be a position already filed, i.e. strictly behind
         * the cursor. On a freshly cleared table every slot holds LZ2_EMPTY or
         * a position from an earlier iteration, so `cand >= pos` is exactly
         * the empty case and this test is unchanged. A sample pass clears only
         * slots reachable from [0,sn), yet this loop still runs to n, so a slot
         * past sn can carry a stale value from a previous call: if that value
         * is >= n the load below reads out of bounds, and if it is >= pos with
         * matching bytes `off = mpos - mcand` underflows. Rejecting cand >= pos
         * handles both. This is the main path's copy of the probe's guard
         * (`p_c1p <= mpos` / `c1p <= mpos` below). */
        if (ZPSE_LIKELY(cand >= pos ||
                        zpse_load_u32le(in + cand) != cur)) {
#if ZPSE_LZ2_PIPE_NEXT
            /* The ordinary step: one byte on, which is every miss until
             * ZPSE_LZ2_SKIP_BITS of them in a row accelerate it, and this
             * iteration has already read that position's key, slot and the
             * slot's entry. */
            if (nx_ok != 0 && (misses >> ZPSE_LZ2_SKIP_BITS) == 0u) {
                pos += 1u;
                misses++;
                cur = nx_cur;
                h = nx_h;
                cand = nx_c;
                continue;
            }
#endif
            /* Two misses are a coincidence; a run of them means the data has
             * no match here, so the cursor accelerates. */
            pos += 1u + (misses >> ZPSE_LZ2_SKIP_BITS);
            misses++;
            if (pos > limit) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lz2_hash_at(in, pos, n, cur, hash_bits);
#if ZPSE_LZ2_PIPE_NEXT
            cand = LZ2_TAB_GET(h);
#endif
            continue;
        }
#if ZPSE_LZ2_PROBE_EARLY
        /* The lazy probe's two loads, issued now so their latency overlaps
         * the backward walk and the extension below (see
         * ZPSE_LZ2_PROBE_EARLY). Read after the table write above, so a
         * self-hit reads exactly as it did before. */
        uint32_t p_cur = 0u, p_h = 0u, p_c1 = LZ2_EMPTY, p_c1p;
        int p_ok = 0;

#if ZPSE_LZ2_PIPE_NEXT
        /* The probe asks about pos + 1, and so does the lookahead above: on
         * a hit the three reads are already in hand. */
        p_cur = nx_cur;
        p_h = nx_h;
        p_c1 = nx_c;
        p_ok = nx_ok;
#else
        if (pos + 1u + ZPSE_LZ2_MIN_MATCH <= n) {
            p_cur = zpse_load_u32le(in + pos + 1u);
            p_h = lz2_hash_at(in, pos + 1u, n, p_cur, hash_bits);
            p_c1 = LZ2_TAB_GET(p_h);
            p_ok = 1;
        }
#endif
        p_c1p = p_c1;
#endif
        /* Extend backward into the pending literals: an earlier start is a
         * longer match for the same four bytes, and the run in front of it is
         * that many bytes shorter. That walk is where the token form earns
         * most of its size over this one, because it has always done it.
         *
         * It moves `mpos`/`mcand`, NOT the scanning cursor. `pos` was written
         * into the table above and every position the literal scan probed on
         * the way here is in it too; moving the cursor back over those would
         * leave entries AHEAD of it, and the first iteration that read one
         * would subtract to an enormous distance and desync the stream --
         * which is exactly how the first version of this broke dickens at
         * byte 2354 of a 3 KiB block. Coding the match from a copy cannot:
         * the entries stay behind the end of the match, `mpos + len`, which
         * is at least `pos + 4` because the walk only steps over bytes that
         * already compare equal.
         *
         * The `mpos > anchor` bound is what keeps it honest: the walk can
         * only take bytes this block has not coded yet, so the decoder never
         * sees a match that starts before the last one ended. `len` below is
         * measured from the new start, so the walked bytes are bytes the
         * match now covers rather than bytes it has to give back, and
         * `mcand > 0` keeps the reference inside the block. */
#if ZPSE_LZ2_BACK_WIDE
        while (mpos - anchor >= 8u && mcand >= 8u &&
               lz2_load64(in + mpos - 8u) == lz2_load64(in + mcand - 8u)) {
            mpos -= 8u;
            mcand -= 8u;
        }
#endif
        while (mpos > anchor && mcand > 0u && in[mpos - 1u] == in[mcand - 1u]) {
            mpos--;
            mcand--;
        }
        /* Extend forward one eight-byte xor at a time. This is the only work
         * a match does: no lookahead at pos+1, no distance cache to update
         * beyond the one byte the command carries.
         *
         * The xor is the length as well as the test: its lowest set bit is
         * the first byte that differs, so one pair of loads answers both "is
         * this step equal" and "how far in does it stop". The loop used to
         * compare a pair and then load the same pair again to count its tail,
         * which on prose -- average match 5.4 bytes, so it exits on its first
         * test -- was two loads and a branch for one length.
         *
         * One four-byte compare for the candidate and one eight-byte xor for
         * the length is the cheapest shape tried so far: folding them into a
         * single eight-byte xor (so a miss pays a load pair and a ctz instead
         * of a compare) measured 12-17% SLOWER on prose, columnar XML and the
         * binary members, because misses outnumber hits and a miss is the
         * common case. Only the forward bound is tested below: the candidate
         * is behind the cursor, so a step that fits at pos fits at cand. */
        {
#if ZPSE_LZ2_HAVE_SSE2
            (void)have_avx2;
            len = lz2_extend_sse2(in, mpos, mcand, len, n);
#else
            int tail = 0;

            if (have_avx2) {
#if ZPSE_LZ2_HAVE_AVX2
                len = lz2_extend_avx2(in, mpos, mcand, len, n);
#endif
            } else {
                while (mpos + len + 8u <= n) {
                    uint64_t x = lz2_load64(in + mpos + len) ^
                                 lz2_load64(in + mcand + len);

                    if (x != 0u) {
                        len += zpse_ctz64(x) >> 3;
                        tail = 1;
                        break;
                    }
                    len += 8u;
                }
            }
            if (tail == 0) {
                while (mpos + len < n && in[mpos + len] == in[mcand + len]) {
                    len++;
                }
            }
#endif
        }
        /* Finder-only minimum match: drop a match whose fully extended length
         * is below the knob. Its bytes stay in the pending literal run (anchor
         * untouched) and the cursor steps one byte on, exactly like an
         * ordinary miss. The wire bias stays ZPSE_LZ2_MIN_MATCH, so the stream
         * still decodes with the unchanged decoder. */
        if (len < (uint32_t)ZPSE_LZ2_FIND_MIN) {
            pos += 1u;
            misses++;
            if (pos > limit) {
                break;
            }
            cur = zpse_load_u32le(in + pos);
            h = lz2_hash_at(in, pos, n, cur, hash_bits);
#if ZPSE_LZ2_PIPE_NEXT
            cand = LZ2_TAB_GET(h);
#endif
            continue;
        }
        /* Lazy matching, against the entry at the next position rather than
         * a fresh probe of the whole table: the candidate that hashes to the
         * position one byte on is the one a match there would use, so one
         * load, one compare and one xor settle whether this match is the one
         * to code. If the next position's is strictly longer, the byte at
         * `mpos` becomes a literal -- the cursor steps over it and the next
         * command's literal run picks it up, which is why nothing has to be
         * written here -- and the loop re-finds the longer match at its own
         * position, where it walks back into that run like any other.
         *
         * `c1 <= mpos` keeps the reference behind the new cursor: the entry
         * read is whatever the block coded before, never the position being
         * filed (this function writes `tab[h] = pos` at the loop top, and
         * `pos + 1` is not written until the next iteration files it).
         *
         * `mpos == pos` is the other half of that guarantee, and the one
         * that costs a stream when it is missing. The backward walk above
         * can start the match BEHIND the scanning cursor, and stepping to
         * `mpos + 1` would then move the cursor backwards, into positions
         * the literal scan has already filed: the next candidate read from
         * one of them sits ahead of the cursor, `pos - cand` underflows to a
         * sixty-thousand-byte distance, and the block no longer decodes
         * (mozilla, block 590, first bad byte 918 -- the first version of
         * this had no guard). A walked-back match is already the better
         * match, so there is nothing to look ahead for: the probe only runs
         * when the match starts where the cursor is. */
#if ZPSE_LZ2_PROBE_EARLY
        if (p_ok && mpos == pos && len <= (uint32_t)ZPSE_LZ2_LAZY_MAX) {
            /* A_safe: the slot value is already in hand; no extra hash and no
             * candidate load. p_c1p == cand means the probe would re-test this
             * match's own source one byte on, which is one byte shorter and so
             * can never win. The probe only runs when the backward walk did
             * not move the start (mpos == pos), hence mcand == cand here and
             * the skip cannot change a decision -- measured byte-identical.
             *
             * LZE3: this gate ALONE is slower (0.97x all files); it is the
             * combination with the 16-byte extension above that wins. */
            if ((ZPSE_LZ2_PROBE_SELF != 0 || p_h != h) &&
                p_c1 != LZ2_EMPTY &&
                p_c1p <= mpos && p_c1p != cand &&
                zpse_load_u32le(in + p_c1p) == p_cur &&
                lz2_extend(in, n, mpos + 1u, p_c1p, ZPSE_LZ2_MIN_MATCH,
                           len + 1u) > len) {
                pos = mpos + 1u;
                misses = 0u;
                cur = p_cur;
                h = p_h;
#if ZPSE_LZ2_PIPE_NEXT
                cand = LZ2_TAB_GET(h);
#endif
                continue;
            }
        }
#else
        if (mpos == pos && len <= (uint32_t)ZPSE_LZ2_LAZY_MAX &&
            mpos + 1u + ZPSE_LZ2_MIN_MATCH <= n) {
            uint32_t cur1 = zpse_load_u32le(in + mpos + 1u);
            uint32_t h1 = lz2_hash_at(in, mpos + 1u, n, cur1, hash_bits);
            uint32_t c1 = LZ2_TAB_GET(h1);
            uint32_t c1p = c1;

            if (c1 != LZ2_EMPTY && c1p <= mpos &&
                zpse_load_u32le(in + c1p) == cur1 &&
                lz2_extend(in, n, mpos + 1u, c1p, ZPSE_LZ2_MIN_MATCH,
                           len + 1u) > len) {
                pos = mpos + 1u;
                misses = 0u;
                cur = cur1;
                h = h1;
#if ZPSE_LZ2_PIPE_NEXT
                cand = LZ2_TAB_GET(h);
#endif
                continue;
            }
        }
#endif
        {
            uint32_t run = mpos - anchor;
            const uint8_t *ls = in + anchor;
            uint32_t off = mpos - mcand;
            uint32_t rem = len;

            /* The common command, written in place: a literal run and a match
             * that both fit their nibbles, so the whole command is a token,
             * the run and two distance bytes. On prose that is four commands
             * in five, and the call plus its room arithmetic was a fifth of
             * the writer's cost. The bytes are exactly lz2_put's, escapes and
             * all, which is what keeps the stream identical -- `rem` is left
             * at zero so the general path below does not emit a second
             * command for the same match. */
            if (run < 15u && rem < 15u + ZPSE_LZ2_MIN_MATCH &&
                cap - out >= run + 3u) {
                dst[out] = (uint8_t)((run << 4) |
                                     (rem - ZPSE_LZ2_MIN_MATCH));
                if (run != 0u) {
                    /* A run of up to fourteen literals leaves as one
                     * 16-byte store when the buffer and the block both have
                     * the bytes: `run` is a register value, so a memcpy of
                     * it is a size dispatch per command, and this command
                     * shape is four in five of them on prose. The bytes
                     * past the run are the distance and the next token,
                     * both written before anything reads them -- the store
                     * is 16 bytes into a buffer with at least 17 left, so
                     * it cannot leave the payload either. */
                    if (cap - out >= 17u && anchor + 16u <= n) {
                        memcpy(dst + out + 1u, ls, 16u);
                    } else {
                        memcpy(dst + out + 1u, ls, run);
                    }
                }
                dst[out + 1u + run] = (uint8_t)(off & 0xFFu);
                dst[out + 2u + run] = (uint8_t)((off >> 8) & 0xFFu);
                out += run + 3u;
                lit_total += run;
                cmds++;
                run = 0u;
                rem = 0u;
            }
            while (run > ZPSE_LZ2_MAX_LIT) {
                if (!lz2_put(dst, cap, &out, ls, ZPSE_LZ2_MAX_LIT, 0u, 0u, &lit_total, &cmds)) {
                    return 0u;
                }
                ls += ZPSE_LZ2_MAX_LIT;
                run -= ZPSE_LZ2_MAX_LIT;
            }
            /* Split a long match into commands of at most
             * ZPSE_LZ2_MAX_MATCH. Every chunk has to be at least
             * ZPSE_LZ2_MIN_MATCH long: the format codes a match as
             * (length - 4), so a chunk of one to three bytes would be written
             * as zero and read back as four, which desyncs the output by the
             * difference. A remainder that would be that short is avoided by
             * shortening the chunk before it and leaving exactly
             * MIN_MATCH at the end -- reachable only when the match ends four
             * bytes past a chunk boundary. */
            while (rem > ZPSE_LZ2_MAX_MATCH) {
                uint32_t take = ZPSE_LZ2_MAX_MATCH;

                if (rem - take < ZPSE_LZ2_MIN_MATCH) {
                    take = rem - ZPSE_LZ2_MIN_MATCH;
                }
                if (!lz2_put(dst, cap, &out, ls, run, off, take,
                             &lit_total, &cmds)) {
                    return 0u;
                }
                ls += run;
                run = 0u;
                rem -= take;
            }
            if (rem != 0u &&
                !lz2_put(dst, cap, &out, ls, run, off, rem, &lit_total,
                         &cmds)) {
                return 0u;
            }
        }
        pos = mpos + len;
        anchor = pos;
        misses = 0u;
        if (pos + ZPSE_LZ2_MIN_MATCH > n) {
            break;
        }
        cur = zpse_load_u32le(in + pos);
        h = lz2_hash_at(in, pos, n, cur, hash_bits);
#if ZPSE_LZ2_PIPE_NEXT
        cand = LZ2_TAB_GET(h);
#endif
    }
    if (anchor < n) {
        uint32_t run = n - anchor;
        const uint8_t *ls = in + anchor;

        while (run > ZPSE_LZ2_MAX_LIT) {
            if (!lz2_put(dst, cap, &out, ls, ZPSE_LZ2_MAX_LIT, 0u, 0u,
                         &lit_total, &cmds)) {
                return 0u;
            }
            ls += ZPSE_LZ2_MAX_LIT;
            run -= ZPSE_LZ2_MAX_LIT;
        }
        if (!lz2_put(dst, cap, &out, ls, run, 0u, 0u, &lit_total,
                     &cmds)) {
            return 0u;
        }
    }
    zpse_store_u32le(dst, lit_total);
    zpse_store_u32le(dst + 4u, cmds);
    return out;
}

/* Tagged with the rest of the hot set: this finder is the whole of the F
 * form's encode, and it was the one kernel in the tree left at the build's
 * own level. */
ZPSE_HOT uint32_t zpse_lz2_encode(const uint8_t *in, uint32_t n, uint8_t *dst,
                         uint32_t cap, void *table, uint32_t hash_bits)
{
    return lz2_encode_impl(in, n, dst, cap, table, hash_bits, 0u);
}

/* Price a sample of sn bytes (the length passed as `n`) with the fast form:
 * identical stream to zpse_lz2_encode, but only the slots a position in
 * [0,sn) can hash to are cleared. */
ZPSE_HOT uint32_t zpse_lz2_encode_sample(const uint8_t *in, uint32_t n,
                                         uint8_t *dst, uint32_t cap,
                                         void *table, uint32_t hash_bits,
                                         uint32_t sn)
{
    return lz2_encode_impl(in, n, dst, cap, table, hash_bits, sn);
}

/* Copy one match: sixteen bytes for a short match at a long distance, two
 * stores for the next shape up, then eight-byte steps and a byte tail. Each
 * wide store is only taken when all of it lands inside the block, because a
 * frame decoder hands a block exactly the room its bytes need and decodes
 * blocks side by side, so a store past the end would land in a neighbour's
 * output. */
static void lz2_copy_match(uint8_t *out, uint32_t op, uint32_t n,
                           uint32_t off, uint32_t mlen)
{
    uint8_t *d = out + op;
    uint32_t left = mlen;

    /* A short offset is a repeating pattern, not a slow copy -- lzav's
     * shape. Write the first eight bytes of the pattern with four byte
     * moves and one four-byte store, then copy eight bytes at a time at a
     * distance that is a multiple of the period. The four-byte store reads
     * at `4 mod off` back into the bytes just written, so it is in phase,
     * and `wide` is the smallest multiple of the period that is at least
     * eight, i.e. the narrowest distance an eight-byte store may read from.
     * Without it, an offset under sixteen is a byte loop for the whole
     * match in this function, and on the binary corpora those offsets are
     * the common ones: a record layout repeats its 2- and 4-byte fields at
     * exactly those distances.
     *
     * Measured as decode only -- the streams built once, then each block
     * decoded repeatedly for about five seconds a reading, best of three,
     * two binaries built from the tree before and after, run in both
     * orders: webster 1 997 -> 2 030 MB/s (1.6%), reymont 1 713 -> 1 782
     * (4.0%), xml 2 281 -> 2 345 (2.8%), and dickens, osdb and ooffice
     * within 0.3%. Timing the encode passes in front of every decode
     * reading put the second binary on a hotter core and made xml and osdb
     * look 2-3% worse than they are; that bias is why this harness no
     * longer does it.
     *
     * The room test rounds the length up to the whole eight-byte stores
     * this writes, and a frame decoder hands a block exactly the room its
     * bytes need, so a match at the very end of a block falls through to
     * the byte loop rather than writing into a neighbour's output. */
    if (off < 16u && mlen != 0u) {
        static const uint8_t rle_off[16] = {
            0u, 0u, 0u, 1u, 0u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u
        };
        static const uint8_t rle_wide[16] = {
            0u, 1u, 2u, 2u, 4u, 3u, 2u, 1u, 8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u
        };
        const uint32_t need = (mlen + 7u) & ~7u;

        if (op + need <= n) {
            const uint32_t wide = 8u + off - rle_wide[off];
            const uint8_t *s = d - (ptrdiff_t)off;
            uint32_t rest = need;

            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = s[3];
            memcpy(d + 4u, d - (ptrdiff_t)off + rle_off[off], 4u);
            d += 8u;
            rest -= 8u;
            while (rest != 0u) {
                memcpy(d, d - wide, 8u);
                d += 8u;
                rest -= 8u;
            }
            return;
        }
    }
    if (off >= 16u && mlen <= 16u && op + 16u <= n) {
        memcpy(d, d - off, 16u);
        left = 0u;
    } else if (off >= 32u && mlen <= 32u && op + 32u <= n) {
        memcpy(d, d - off, 16u);
        memcpy(d + 16u, d - off + 16u, 16u);
        left = 0u;
    } else if (off >= 8u) {
        /* Step as wide as the distance allows, not as wide as the machine
         * could: a step wider than the distance reads bytes this copy has not
         * written yet. Past 32 bytes back, a 32-byte step is safe and a long
         * match stops paying eight bytes a store; the binary members are
         * where that shows, because their matches are the medium-length
         * ones at medium distances. */
        if (off >= 32u) {
            while (left >= 32u) {
                memcpy(d, d - off, 32u);
                d += 32u;
                left -= 32u;
            }
        }
        if (off >= 16u) {
            while (left >= 16u) {
                memcpy(d, d - off, 16u);
                d += 16u;
                left -= 16u;
            }
        }
        while (left >= 8u) {
            memcpy(d, d - off, 8u);
            d += 8u;
            left -= 8u;
        }
    }
    while (left != 0u) {
        *d = d[-(ptrdiff_t)off];
        d++;
        left--;
    }
}

/* One match in the trusted region. The three fixed stores for a short match
 * at off>=8 are LZ4's fast-loop copy (C in the mirror audit): for mlen<=18
 * they write 18 bytes with no loop and no length dispatch, and the next
 * command overwrites the surplus. Reading `m` is safe at every offset -- when
 * off<18 the later stores read bytes this copy has just written, which is
 * exactly the overlapping source. Longer matches and off<8 use eight-byte
 * steps to `cpy`, rounding the last store up to a whole word; off<8 starts
 * with LZ4's four-byte pattern fill. The caller's 560-byte output guard
 * covers the widest extent (270 literals + ceil(274/8)*8 = 550 bytes). */
static void lz2_copy_fast(uint8_t *out, uint32_t op, uint32_t n,
                          uint32_t off, uint32_t mlen)
{
    static const unsigned inc32[8] = {0u, 1u, 2u, 1u, 0u, 4u, 4u, 4u};
    static const int      dec64[8] = {0, 0, 0, -1, -4, 1, 2, 3};
    uint8_t *d = out + op;
    const uint8_t *m = d - (ptrdiff_t)off;
    uint8_t *cpy = d + mlen;

    (void)n;
    if (off >= 8u && mlen <= 18u) {
        memcpy(d,       m,       8u);
        memcpy(d + 8u,  m + 8u,  8u);
        memcpy(d + 16u, m + 16u, 2u);
        return;
    }
    if (off < 8u) {
        d[0] = m[0];
        d[1] = m[1];
        d[2] = m[2];
        d[3] = m[3];
        m += inc32[off];
        memcpy(d + 4u, m, 4u);
        m -= dec64[off];
        d += 8u;
    } else {
        memcpy(d, m, 8u);
        m += 8u;
        d += 8u;
    }
    while (d < cpy) {
        memcpy(d, m, 8u);
        d += 8u;
        m += 8u;
    }
}

/* B: LZ4's escape decoder, bounded for this loop's guard. LZ4's
 * read_variable_length chains 255 bytes until an input limit; feeding that
 * unbounded value to this decoder's trusted stores would break the 560-byte
 * output guard. A valid stream's escape is never 255 -- lz2_put caps a run at
 * ZPSE_LZ2_MAX_LIT (255) and a match at ZPSE_LZ2_MAX_MATCH (273) -- so the
 * common path is one byte and one well-predicted branch. A hostile 255
 * continues at most one more byte; a second 255 or a sum past 255 is
 * LZ2_RVL_ERR, which the caller turns into ZPSE_BAD_LZ before any store. Two
 * bytes is well inside the LZ2_FAST_INPUT_BACK input guard. */
#define LZ2_RVL_ERR UINT32_MAX
static inline uint32_t lz2_rvl(const uint8_t **ipp)
{
    const uint8_t *p = *ipp;
    uint32_t v = (uint32_t)*p++;

    if (ZPSE_UNLIKELY(v == 255u)) {
        uint32_t b = (uint32_t)*p++;

        v += b;
        if (ZPSE_UNLIKELY(b == 255u || v > 255u)) {
            v = LZ2_RVL_ERR;
        }
    }
    *ipp = p;
    return v;
}
int zpse_lz2_decode(const uint8_t *in, uint32_t in_len, uint8_t *out,
                     uint32_t n)
{
    uint32_t lit_total, cmd_count, cmd_left, lit_seen = 0u, op = 0u;
    const uint8_t *lp, *tend, *safe_lp;
    uint32_t low;

    if (in == 0 || out == 0) {
        return ZPSE_BAD_ARG;
    }
    if (n == 0u) {
        return ZPSE_BAD_BLOCK_SIZE;
    }
    if (in_len < ZPSE_LZ2_MIN_PAYLOAD) {
        return ZPSE_TRUNCATED;
    }
    lit_total = zpse_load_u32le(in);
    cmd_count = zpse_load_u32le(in + 4u);
    if (lit_total > in_len - LZ2_HDR || cmd_count == 0u) {
        return ZPSE_BAD_LZ;
    }
    lp = in + LZ2_HDR;
    tend = in + in_len;
    cmd_left = cmd_count;
    low = (n > ZPSE_LZ2_FAST_ROOM) ? (n - ZPSE_LZ2_FAST_ROOM) : 0u;
    safe_lp = (in_len > LZ2_FAST_INPUT_BACK) ?
                  (tend - LZ2_FAST_INPUT_BACK) : in;

    /* Trusted-command region. Two guards, each tested once per command,
     * bound every read and write below; there is no per-field room check,
     * no command-count test and no literal-total accumulation branch in the
     * body. LZD1 measured this shape (its harness gave the stream 512 B of
     * slack) at 3.64 GB/s against 3.07 for the old checked loop, +18.7%,
     * byte-exact on all twelve 64 KiB Silesia blocks.
     *
     * Output: op < n - ZPSE_LZ2_FAST_ROOM leaves room for the widest command
     * a corrupt escape can name -- 269 literals, a 273-byte match and the
     * seven bytes a short-offset pattern copy rounds up (549 in all). The
     * excess bytes the 16-byte literal store writes past a short run are
     * overwritten by the following commands, and the checked tail rewrites
     * everything from the last command it sees to the end of the block.
     *
     * Input: lp is kept at or before a pointer LZ2_FAST_INPUT_BACK (296)
     * bytes from the end. A command reads at most one token, two escapes,
     * 270 literals rounded up to 288 bytes by the inline wildcopy, and two
     * distance bytes. A corrupt 255 escape is continued at most one byte by
     * lz2_rvl and then refused, so it cannot name a run wider than the guard;
     * the final position/length/count check refuses any remaining misparse.
     * The checked tail keeps the full 255-chain semantics. */
    while (op < low && lp <= safe_lp) {
        uint32_t tok = *lp++;
        uint32_t lit = tok >> 4;
        uint32_t mn = tok & 15u;
        uint32_t mlen = mn + ZPSE_LZ2_MIN_MATCH;
        uint32_t off;

        /* B: both escapes use LZ4's helper. A valid stream's first byte is
         * never 255, so the chain runs once; the cap keeps a hostile chain
         * inside the 560-byte output guard. */
        if (lit == 15u) {
            uint32_t add = lz2_rvl(&lp);

            if (ZPSE_UNLIKELY(add == LZ2_RVL_ERR)) {
                return ZPSE_BAD_LZ;
            }
            lit += add;
        }
        /* One 16-byte store for every nibble-sized run, lit == 0 included:
         * skipping it measured -23% in LZD1, the worst single idea tried. The
         * bytes past the run are overwritten by the match or a later command.
         * E: a run past one nibble is copied with inlined 16-byte stores in
         * 32-byte steps (LZ4_wildCopy32's shape) instead of a libc memcpy call.
         * The two stores of a step are speculative but land in the 560-byte
         * output guard, and LZ2_FAST_INPUT_BACK keeps the reads in the payload. */
        if (lit <= 14u) {
            memcpy(out + op, lp, 16u);
        } else {
            uint8_t *d = out + op;
            const uint8_t *s = lp;
            uint8_t *e = d + lit;

            do {
                memcpy(d, s, 16u);
                memcpy(d + 16u, s + 16u, 16u);
                d += 32u;
                s += 32u;
            } while (d < e);
        }
        op += lit;
        lp += lit;
        lit_seen += lit;

        off = (uint32_t)lp[0] | ((uint32_t)lp[1] << 8u);
        lp += 2u;
        if (mn == 15u) {
            uint32_t add = lz2_rvl(&lp);

            if (ZPSE_UNLIKELY(add == LZ2_RVL_ERR)) {
                return ZPSE_BAD_LZ;
            }
            mlen += add;
        }
        /* Underflow is caught by the final cmd_left != 0 test. */
        cmd_left--;
        if (off > op) {
            return ZPSE_BAD_LZ;
        }
        if (off == 0u) {
            if (lit == 0u) {
                return ZPSE_BAD_LZ;
            }
            continue;
        }
        lz2_copy_fast(out, op, n, off, mlen);
        op += mlen;
    }

    /* Fully checked tail, for the last ZPSE_LZ2_FAST_ROOM bytes of output and
     * for a payload with fewer than 280 bytes left. This is also the path a
     * truncated or hostile stream takes. */
    while (op < n) {
        uint32_t tok, lit, mlen, off;
        uint8_t b;

        cmd_left--;
        if (lp >= tend) {
            return ZPSE_BAD_LZ;
        }
        tok = *lp++;
        lit = tok >> 4;
        if (lit == 15u) {
            do {
                if (lp >= tend) {
                    return ZPSE_BAD_LZ;
                }
                b = *lp++;
                lit += b;
            } while (b == 255u);
        }
        mlen = (tok & 15u) + ZPSE_LZ2_MIN_MATCH;
        if (lit > n - op || (uint32_t)(tend - lp) < lit) {
            return ZPSE_BAD_LZ;
        }
        if (lit != 0u) {
            if (lit < 15u && n - op >= 16u && (uint32_t)(tend - lp) >= 16u) {
                memcpy(out + op, lp, 16u);
            } else {
                memcpy(out + op, lp, lit);
            }
            op += lit;
            lp += lit;
            lit_seen += lit;
        }
        if (lp >= tend) {
            return ZPSE_BAD_LZ;
        }
        if ((uint32_t)(tend - lp) < 2u) {
            return ZPSE_BAD_LZ;
        }
        off = (uint32_t)lp[0] | ((uint32_t)lp[1] << 8u);
        lp += 2;
        if (off == 0u) {
            /* A literal-only command has to make progress. */
            if (lit == 0u) {
                return ZPSE_BAD_LZ;
            }
            continue;
        }
        if ((tok & 15u) == 15u) {
            do {
                if (lp >= tend) {
                    return ZPSE_BAD_LZ;
                }
                b = *lp++;
                mlen += b;
            } while (b == 255u);
        }
        if (off > op || mlen > n - op) {
            return ZPSE_BAD_LZ;
        }
        lz2_copy_match(out, op, n, off, mlen);
        op += mlen;
    }
    /* An extra command underflows cmd_left; a missing one leaves it positive.
     * Literal bytes, stream position and output length must all agree too. */
    if (cmd_left != 0u || lit_seen != lit_total || lp != tend) {
        return ZPSE_BAD_LZ;
    }
    return ZPSE_OK;
}
