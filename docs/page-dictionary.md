# A page dictionary for the caller, not the encoder

*Design note. Nothing here is part of the stream format: a reused encode is an
ordinary block, and every decoder reads it without changes.*

## What it buys

A page store writes the same 4 KiB page more than once. On the three dump
corpora this codec is measured against, 16.42% of pages (8,548 of 52,065,
35.0 MB of 213.3 MB) are byte-for-byte copies of a page already written, and
the C heap file on its own is 80.4% repeats. The repeats are scattered rather
than consecutive: a cache of the last 64 pages hits 1.26%, direct-mapped 1,024
and 4,096-entry content tables hit 2.33% and 5.02%, and a 65,536-entry content
table hits 16.05%. Only content addressing gets them.

Measured upper bound, level 6, 4 KiB pages, one pinned core: a 65,536-entry
table that serves the stored encode instead of encoding takes the dumps from
109.8 to 128.4 MB/s on a paired run (106.9 to 118.9 over four interleaved
passes), the C heap file from 128 to 571 MB/s, the Python heap from 101.7 to
106.3 and the library dump from 109.1 to 112.7. Packed bytes are identical in
every arm, because a hit rewrites nothing.

Silesia's pages repeat 0.21% of the time (110 of 51,746). A dictionary is a
heap-store feature; on prose it is a hash and a miss.

## Why it does not live in the codec

The encoder's state is a 448 KiB caller-owned arena, sized once and never
allocated from (include/zpse/arena.h). An entry has to hold a whole encoded
page -- 700 bytes for a text page, several kilobytes for a literal-heavy one --
so 65,536 entries is hundreds of megabytes of buffers. That is the page store's
memory and the page store's lifetime, and the codec has neither.

What the codec can offer is the two pieces a store cannot write itself: the
content hash and the header rewrite.

## The API

Caller-owned arrays, no allocation, no state in `ZpseEncoder`:

```c
/* 128-bit content tag: a 64-bit tag is cheaper (8 bytes an entry) but the
 * codec's contract is byte-exact, and a 64-bit tag makes a collision a wrong
 * page rather than a rounding error. */
typedef struct ZpsePageDict {
    uint64_t *tag_lo;   /* caller memory: entries * 8 */
    uint64_t *tag_hi;   /* caller memory: entries * 8 */
    uint32_t *slot;     /* caller memory: entries * 4: the store's own id */
    uint32_t  entries;  /* power of two, at least 16 */
} ZpsePageDict;

/* 0 iff the page is empty; a dictionary entry with tag 0/0 is unused. */
void zpse_page_hash(const uint8_t *page, uint32_t n,
                    uint64_t *lo_out, uint64_t *hi_out);

void zpse_pagedict_init(ZpsePageDict *d, uint64_t *tag_lo, uint64_t *tag_hi,
                        uint32_t *slot, uint32_t entries);

/* 1 when (lo, hi) is present and its slot is written to *slot_out. */
int zpse_pagedict_find(const ZpsePageDict *d, uint64_t lo, uint64_t hi,
                       uint32_t *slot_out);

/* Direct-mapped insert: the entry at hash % entries is overwritten. */
void zpse_pagedict_insert(ZpsePageDict *d, uint64_t lo, uint64_t hi,
                          uint32_t slot);

/* Retag a stored block for a new position: rewrites seq and is_last and fixes
 * the header CRC. The payload and its checksum are over the page's bytes and
 * do not move. Returns PSE_OK, or PSE_CORRUPT for a bad header CRC. */
int zpse_block_retag(uint8_t *block, uint32_t block_len,
                     uint32_t seq, int is_last);
```

20 bytes an entry, so 65,536 entries is 1.25 MiB of index -- and the encoded
pages themselves are already in the store, which is the whole point: the page
store keeps them to write them out.

## The call sequence

```c
uint64_t lo, hi;
uint32_t slot, blen;

zpse_page_hash(page, 4096, &lo, &hi);
if (zpse_pagedict_find(&d, lo, hi, &slot) &&
    store_read(slot, blk, &blen) &&                        /* the stored block */
    zpse_block_retag(blk, blen, seq, is_last) == PSE_OK) {
    emit(blk, blen);                                       /* no arena touched */
} else {
    zpse_compress_block(page, 4096, blk, cap, &blen, seq, is_last, arena);
    if (blen < 4096) {                                     /* only worth storing */
        store_write(new_slot, blk, blen);
        zpse_pagedict_insert(&d, lo, hi, new_slot);
    }
    emit(blk, blen);
}
```

Two rules the sequence encodes:

* **Insert only what compressed.** A raw page is not worth an index entry: it
  will be a hit for the same bytes next time and save the sweep, but its block
  is a full page and the store's slot is no cheaper than the page.
* **Retag, never reuse verbatim.** A block carries `seq` in its header and its
  header CRC covers it, so a stored block served for a new position without
  `zpse_block_retag` produces a stream that decodes correctly and checks its
  frame index wrongly. The prototype measured below copies blocks without the
  retag, which is why its numbers are an upper bound.

## What it deliberately is not

* **Not a decision cache.** Remembering only the winning shape and re-running
  the trial on a hit skips the price on 16% of the dumps' pages -- under 8% of
  their encode -- while still hashing every page. On Silesia it is a net loss.
  The win here comes from not encoding the page at all.
* **Not a decoder feature.** No new block type, no new header field, no format
  version: hits and misses are the same bytes a non-dictionary writer produces.
* **Not in the arena.** Growing `ZPSE_ENC_ARENA_SIZE` would make every caller
  pay for a heap corpus's repeats, including the ones compressing prose.

## Numbers this rests on

| table entries | hits (dumps) | dumps MB/s | note |
| ---: | ---: | ---: | --- |
| none | -- | 106.9-109.8 | level 6, 4 KiB, pinned |
| 1,024 | 4.85% | 115.0-115.4 | |
| 4,096 | 11.58% | 111.1-121.5 | |
| 65,536 | 16.08% | 118.9-128.4 | the arm the design needs |

Per file at 65,536 entries: Python heap 7.78% hits, 101.7 -> 106.3 MB/s; C
heap 78.41% hits, 128.3 -> 571.2 MB/s; library dump 0.91% hits, 109.1 -> 112.7
MB/s. The prototype is `build/pagecache.c` (scratch, not part of the tree): it
reserves one page-sized buffer per entry, copies a hit's stored bytes without
the retag, and does not decode hits, so its rate is an upper bound and its
memory (256 MB at 65,536 entries) is not the 1.25 MiB this design needs.

## Open questions

* **Hit verification.** A 128-bit tag makes a collision ~2^-64 over 2^32 pages,
  which is why the tag is 128 bits and not 64. A store that already detects
  duplicate pages itself (zram does, for its own compression) can use its own
  index and skip this API entirely.
* **Where `seq` comes from.** A store that compresses before it knows a page's
  position needs the retag; a store that knows the position up front can pass
  it to `zpse_compress_block` and store the block as written.
* **Eviction.** Direct mapping is what the hit-rate table above measures. A
  2-way or 4-way scheme costs a second tag compare and should raise the hit
  rate on the scattered dumps; it is unmeasured.
* **Sizing.** The dictionary pays from about 1,024 entries on a heap dump and
  never on prose. A caller with a mixed workload should size it by the fraction
  of heap pages it holds, not by the page count of the device.
