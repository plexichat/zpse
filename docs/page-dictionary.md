# A page dictionary for the caller, not the encoder

*A page store writes the same 4 KiB page more than once. This document is what
the library supplies for that (include/zpse/block.h), what it deliberately does
not, and the measurements behind both. None of it is part of the stream format:
a reused encode is an ordinary block, and every decoder reads it unchanged.*

## What it buys

On the three dump corpora this codec is measured against, 17.9% of pages are
byte-for-byte copies of a page already written, and the C heap file on its own
is 78%. The repeats are scattered rather than consecutive: a cache of the last
64 pages hits 1.26%, direct-mapped 1,024 and 4,096-entry content tables hit
4.90% and 12.72%, and a 65,536-entry content table hits 17.94%. Only content
addressing gets them.

Measured through the shipped API (`build/pagedict.c`, level 6, 4 KiB pages, one
pinned core, four interleaved passes with the arm order rotated, best of four;
every arm's first hit in every file decoded and compared):

| corpus | none | dictionary | pages served |
| --- | ---: | ---: | ---: |
| Python heap | 143.5 MB/s | 149.4 | 7.74% |
| C heap | 170.0 | 731.8 | 78.32% |
| shared-library dump | 153.3 | 152.3 | 0.91% |
| mixed | 159.2 | 194.3 | 20.51% |
| **all** | **155.7** | **182.9** | **17.94%** |

Packed bytes are identical in every arm (142,699,989), because a hit rewrites
nothing but a header. That is the point of the feature: it is a rate trade with
no ratio side at all.

Silesia's pages repeat 0.21% of the time (110 of 51,746). A dictionary is a
heap-store feature; on prose it is a hash and a miss.

## Why it does not live in the codec

The encoder's state is a 448 KiB caller-owned arena, sized once and never
allocated from (include/zpse/arena.h). An entry has to hold a whole encoded
page -- 700 bytes for a text page, several kilobytes for a literal-heavy one --
so 65,536 entries is hundreds of megabytes of buffers. That is the page store's
memory and the page store's lifetime, and the codec has neither.

What the codec supplies is the two pieces a store cannot write itself: the
content tag and the header rewrite.

## The API

Caller-owned arrays, no allocation, no state in `ZpseEncoder`:

```c
typedef struct ZpsePageDict {
    uint64_t *tag_lo;   /* caller memory: entries * 8 */
    uint64_t *tag_hi;   /* caller memory: entries * 8 */
    uint32_t *slot;     /* caller memory: entries * 4 -- the caller's own id */
    uint8_t  *victim;   /* caller memory: (entries / ways) bytes, or NULL */
    uint32_t  entries;  /* total tags, a power of two, at least 16 */
    uint32_t  ways;     /* 1, 2, 4 or 8; 0 means the dictionary is empty */
} ZpsePageDict;

void zpse_page_hash(const uint8_t *page, uint32_t n,
                    uint64_t *lo_out, uint64_t *hi_out);
void zpse_pagedict_init(ZpsePageDict *d, uint64_t *tag_lo, uint64_t *tag_hi,
                        uint32_t *slot, uint32_t entries);
void zpse_pagedict_init_set(ZpsePageDict *d, uint64_t *tag_lo,
                            uint64_t *tag_hi, uint32_t *slot, uint8_t *victim,
                            uint32_t entries, uint32_t ways);
int  zpse_pagedict_find(const ZpsePageDict *d, uint64_t lo, uint64_t hi,
                        uint32_t *slot_out);
void zpse_pagedict_insert(ZpsePageDict *d, uint64_t lo, uint64_t hi,
                          uint32_t slot);
int  zpse_block_retag(uint8_t *block, uint32_t block_len,
                      uint32_t seq, int is_last);
```

20 bytes an entry plus one byte a set, so 65,536 entries is 1.25 MiB of index --
and the encoded pages themselves are already in the store, which is the whole
point: the page store keeps them to write them out. `zpse_pagedict_init` is the
direct-mapped case; `zpse_pagedict_init_set` takes `ways` and a cursor array
holding which way each set overwrites next, so a set's tags are replaced in
turn rather than the first one being sticky.

`zpse_block_retag` rewrites `block_sequence_number` and the is-last flag in
whichever header form the block carries -- the 32-byte one, or a page's 16-byte
one -- and re-emits the header checksum, which is a CRC32-C over 28 bytes in the
long form and a 16-bit truncation over 14 in the short one. The payload is not
touched: it decodes to the page's bytes, which did not change, so a payload
checksum written with the block still holds. It refuses a NULL block
(`PSE_BAD_ARG`), a length below a page header (`PSE_TRUNCATED`), a length the
header disagrees with or a block with no sync (`PSE_BAD_HEADER`) and a bad
header CRC (`PSE_BAD_HEADER_CRC`), so a store cannot turn a corrupt entry into a
stream that only fails at read time.

## The call sequence

```c
uint64_t lo, hi;
uint32_t slot, blen;

zpse_page_hash(page, 4096, &lo, &hi);
if (zpse_pagedict_find(&d, lo, hi, &slot) &&
    store_read(slot, blk, &blen) &&                  /* the stored block     */
    zpse_block_retag(blk, blen, seq, is_last) == PSE_OK) {
    emit(blk, blen);                                 /* no arena touched     */
} else {
    zpse_compress_block(page, 4096, blk, cap, &blen, seq, is_last, arena);
    if (blen < 4096) {                               /* only if it compressed */
        slot = store_write(blk, blen);
        zpse_pagedict_insert(&d, lo, hi, slot);
    }
    emit(blk, blen);
}
```

Two rules the sequence encodes:

* **Insert only what compressed.** A raw page's block is longer than the page,
  so an entry for it costs more than the hash it saves; the next copy of those
  bytes still pays a full encode, which is what a page store does anyway.
* **Retag, never reuse verbatim.** A block carries `seq` in its header and the
  header checksum covers it, so a stored block served for a new position
  without `zpse_block_retag` produces a stream that decodes correctly and
  indexes itself wrongly.

`tests/suites/test_pagedict.c` pins the whole loop: every served block is
byte-identical to a fresh encode of that page at that position, in both header
forms, and decodes to the page.

## What the tag costs on a corpus that does not repeat

A hash is paid before its lookup can miss, so on a store holding few repeats it
is pure cost. The first version of `zpse_page_hash` was a two-lane, 8-byte
loop, and on the shared-library dump (0.91% hits) it cost 4.5% of the encode
rate: 148.6 MB/s with a dictionary against 151.8 without.

The shipped version reads 32 bytes an iteration into four independent
multiply-xor lanes and folds all four into *both* output halves before the
finaliser -- a half that saw only its own two lanes would make the index (which
uses lo's low bits) blind to half the page, and the suite catches exactly that.
Against the same base it reads 153.3 -> 152.3 MB/s on that dump, i.e. the tag is
inside the run-to-run band, and the 4.5% is gone.

## How wide the table should be

Ways are worth having exactly where the table is small enough to collide, which
is why they are an option rather than the default. Measured at the same total
tag count, arm order rotated between rounds:

| entries | direct | 2-way | 4-way |
| ---: | ---: | ---: | ---: |
| 1,024 | 4.90% | 4.59% | 3.92% |
| 4,096 | 12.72% / 171.3 MB/s | 14.79% / 174.9 | 16.57% / 179.2 |
| 65,536 | 17.94% / 182.9 MB/s | 18.29% / 183.4 | 18.37% / 183.5 |

At 4,096 entries four ways serve 3.85 more points of pages for 4.6% of the rate
(171.3 -> 179.2 MB/s over the same pages), and on the C heap alone the difference
is 356 against 584 MB/s. At 65,536 entries the three extra tag compares buy 0.43
of a point and nothing measurable, and at 1,024 they lose a point because four
ways mean a quarter of the distinct sets. So: a store whose table is a few
thousand entries should ask for four ways, and a store at 65,536 should keep the
direct mapping -- which is what the two inits are for.

## What it deliberately is not

* **Not a decision cache.** Remembering only the winning shape and re-running
  the trial on a hit skips the price on 16% of the dumps' pages -- under 8% of
  their encode -- while still hashing every page. The win here comes from not
  encoding the page at all.
* **Not a decoder feature.** No new block type, no new header field, no format
  version: hits and misses are the same bytes a non-dictionary writer produces.
* **Not in the arena.** Growing `ZPSE_ENC_ARENA_SIZE` would make every caller
  pay for a heap corpus's repeats, including the ones compressing prose.

## Open questions

* **Hit verification.** The tag is 128 bits, so a collision over 2^32 pages is
  ~2^-64, which this design treats as a safety level rather than a check: nothing
  downstream compares the decompressed bytes. A store that needs an absolute
  guarantee should compare the page, or use the duplicate detection it already
  has (zram keeps one, for its own compression) and skip this API.
* **Where `seq` comes from.** A store that compresses before it knows a page's
  position needs the retag; one that knows the position up front can pass it to
  `zpse_compress_block` and store the block as written.
* **Sizing.** The dictionary pays from about 1,024 entries on a heap dump and
  never on prose. A caller with a mixed workload should size it by the fraction
  of heap pages it holds, not by the page count of the device.
* **The pages above the table.** At 65,536 entries the C heap is still 78% hits
  and the mixed corpus 20%; the remaining repeats are further apart than any
  table this size reaches. Bigger tables were not measured.
