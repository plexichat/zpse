# zpse

A block-format compressor in C11. Fixed 64 KiB blocks, fixed-size
caller-owned arenas, no allocation, no I/O, no global state, and no threads in
the codec itself (the optional parallel frame encoder is one module, compiled
out unless it is asked for) — and an encoder whose output depends only on its
input, so a compressed stream is byte-identical on every build, every compiler,
every SIMD level and every thread count.

It is built for the case where a codec has to be *predictable*: decode ten
to seventy gigabytes a second out of cache, never touch the allocator, and
reject malformed input with a status code instead of a crash. It is not a
general-purpose replacement for zstd, and the numbers below say exactly where
it wins and where it loses.

```c
#include "zpse/zpse.h"

static _Alignas(64) uint8_t enc_arena[ZPSE_ENC_ARENA_SIZE];
static _Alignas(64) uint8_t dec_arena[ZPSE_DEC_ARENA_SIZE];
uint8_t in[65536], comp[65536 + 32], out[65536];
uint32_t clen = 0, olen = 0;

int rc = zpse_compress_block(in, sizeof in, comp, sizeof comp, &clen,
                             0u, 1, enc_arena);
rc = zpse_decompress_block(comp, clen, out, sizeof out, &olen, dec_arena);
```

Both arenas are 64-byte aligned and owned by the caller; nothing inside the
library ever allocates. They are 448 KiB (encode) and 384 KiB (decode) —
`ZPSE_ENC_ARENA_SIZE` and `ZPSE_DEC_ARENA_SIZE` say so, and the sizes are
in the names of the parameters the API takes. The decode arena grew by the
64 KiB a `ZPSE_B_LZ_TILE` block's body is unpacked into before the tile path
reads it. `rc` is negative on failure —
every status in `include/zpse/status.h` is reachable from a corrupt stream
and is tested that way.

## Performance

One core, 64 KiB blocks, the library linked directly into the benchmark
(`tests/benchmarks/bench.c`), best of its five reps over four runs. This is a
Ryzen 5 5600X under WSL2, decimal MB/s of uncompressed bytes (1e6 B/s).

| pattern | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| `zeros` | 0.001 | 6 130 | 64 490 |
| `const5A` | 0.001 | 6 210 | **66 070** |
| `random` | 1.000 | 2 410 | **66 410** |
| `lowentropy` (4-bit alphabet) | 0.508 | 740 | 51 000 |
| `incr` (0..255 ramp) | 0.002 | 3 240 | 13 440 |
| `strided` | 0.002 | 1 100 | 13 650 |
| `alt00FF` | 0.001 | 5 080 | 31 780 |
| `sparse2` | 0.016 | 4 990 | 21 340 |
| `sparse` (1 byte per 97) | 0.042 | 2 720 | 10 680 |
| `sawtooth` | 0.002 | 1 090 | 12 760 |

The encode column is the default level (6), which prices every plane of every
candidate and also offers the intra-block LZ form. On these patterns the price
buys the ratios and the finder finds nothing, so the sweep is most of the
encode: the one-value rows are 67 bytes out of 64 KiB and still cost 10.7 us
to encode against 64 GB/s to decode them, because nine planes still have to be
priced and the shapes that survive the price still have to be written. The
ASCII gate is not what costs them that: its whole-block scan is skipped when
the block's first two kilobytes are one value, and it stops at the first byte
with its high bit set, so a binary block pays almost nothing for it (`alt00FF`
is 12.9 us a block) while a block that really is ASCII reads every byte at 32
bytes an iteration in AVX2. Decode is the same at every level, because the
shapes written are. `random` is the row the tile classifier decides: every tile of every plane
is a dense one, and its encode roughly doubled once the vector classify covered
a plane's first tile too -- the tile whose stride predictor has no earlier
bytes in the plane to read, which used to drop the whole plane's first tile to
the byte loop.

What the rows are: `zeros`/`const5A` are one-byte-value blocks (coded in
248 bytes), `random` is incompressible (coded as a raw block, so decode is
exactly a `memcpy`), `lowentropy` is a random stream over sixteen byte values,
`incr`/`strided` are ramps, `alt00FF`/`sparse2`/`sawtooth` exercise the
16-wide interleave, and `sparse` is zeros with a byte every 97.

The interesting part is the ceilings: on the same machine and the same
working set, `memcpy` is **65 620 MB/s** and `memset` **127 370 MB/s**. The
one-value rows decode within two percent of `memcpy` -- wide stores and never
read back, but 64 KiB is too small for that to buy more than a couple of
percent -- and `random` decoding at 66 410 MB/s is the raw path doing a copy
plus a header check, i.e. right at the `memcpy` ceiling. Everything else is transform- and tile-bound, not memory-bound.

### 4 KiB pages

zram hands the codec one page at a time, so a page-sized block is a different
trade from the 64 KiB one above: the sweep has one tile per plane to price a
shape on, the trial it skips is worth half the block, and every structural
byte is charged against 4096 rather than 65536. Measured on three process
dumps -- a Python heap, a C heap and 120 MB of shared-library pages, 213 MB
together -- each page encoded as its own block and kept only when it comes out
smaller than it went in, one pinned core, best of five:

| level | Python heap | C heap | shared libraries | total | encode (py / c / lib) |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 `ZPSE_ENC_SPEED_FASTEST` | 37.77% | 10.36% | 83.11% | 58.56% | 213 / 456 / 218 MB/s |
| 6 (default) | 16.05% | 6.31% | 62.67% | **40.61%** | 108 / 160 / 113 MB/s |
| 9 `ZPSE_ENC_SPEED_MAX` | 16.01% | 6.31% | 62.00% | 40.22% | 94 / 157 / 102 MB/s |

The default level's page costs about 118 MB/s of encode for 40.61%, and
decodes at 1.8 to 5.6 GB/s (the C heap is all pointers, which the fast LZ form
codes as commands).The deepest level buys 0.39 points of ratio for about 10% of the rate, because on a page the tier's extra margin and decode bill reach
the blocks the default level leaves to the tiles rather than finding a
smaller shape for them. The shared-library dump is what holds every level's
total where it is: two pages in five go out raw at level 1, and 83% of the dump
survives there against 63% at the default level and 62% at the deepest.

Silesia's twelve files put the same shape on a general corpus:

| 4 KiB pages | level 1 | level 6 | level 9 |
| --- | ---: | ---: | ---: |
| ratio | 80.06% | 54.64% | 53.16% |
| encode | 247 MB/s | 118 MB/s | 104 MB/s |
| decode | 3.3 GB/s | 1.9 GB/s | 1.9 GB/s |

Against zstd on the same pages (one process, one reused context, the same
keep-it-if-it-is-smaller rule) the page size is where zstd is the better
default: zstd 6 writes 41.27% of Silesia at 88 MB/s and 30.93% of the dumps at
120 MB/s, against 54.64% at 118 and 40.61% at 118. What zpse keeps is decode --
1.5 to 4.2 GB/s against zstd's 0.7 to 2.9 -- and the deep rungs' encode rate,
twice zstd 9's at a ratio ten to twelve points worse. That gap is also the
reason no entropy stage sits on top: order-0 over zpse's own wire closes 6.95
of the 9.68 points to zstd 6, which would leave the dumps at 33.7% against
zstd's 30.9% while paying for the coder. At a page the format's problem is the
ratio it starts from, not the last few bits of it.A mixed workload at the same page size -- 160 MB of programs, shared
libraries, documentation and this project's own build outputs, which is the
shape a browser's or a build tree's footprint has:

| 4 KiB pages, mixed | level 1 | level 6 | level 9 | zstd 1 | zstd 6 | zstd 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ratio | 78.42% | 55.12% | 54.49% | 46.40% | 44.18% | 44.17% |
| encode | 242 MB/s | 106 MB/s | 98 MB/s | 290 MB/s | 87 MB/s | 46 MB/s |
| decode | 4.2 GB/s | 1.5 GB/s | 1.5 GB/s | 0.7 GB/s | 0.7 GB/s | 0.7 GB/s |

Where the dumps' 9.68-point gap to zstd sits, split by the form the default
level picked (zstd's column is its size over the same pages):

| form at level 6 | pages | zpse | zstd | gap |
| --- | ---: | ---: | ---: | ---: |
| token LZ | 32 925 | 18.42% | 13.13% | 5.29 |
| raw | 6 969 | 13.38% | 10.90% | 2.49 |
| LZ + tiles | 9 038 | 5.18% | 4.13% | 1.05 |
| tiles | 2 614 | 3.35% | 2.60% | 0.76 |
| fast LZ | 519 | 0.27% | 0.17% | 0.10 |

Half of it is the token LZ form's own payload on half the pages, and a quarter
is pages that go out raw; both are the same loss, bytes written as bytes. A
page cannot pay for a per-block entropy table -- a canonical Huffman header is
a hundred bytes against a literal stream of one or two thousand -- so zstd's
answer does not transfer to this size without tables baked for one corpus. The
one distance coding a page's format can carry was measured and refused:
delta-coding the explicit distances costs 2.5 points instead of saving them
(41.37% to 43.89% on the dumps, measured before the page parse below) and 7 points on the mixed workload, because a
match's distance has no relation to the last one on text, and zigzag doubles
every small distance that used to fit in one group.

A page's block carries a **16-byte header** rather than 32, always: the
version, tile size, predictor profile and reserved byte are fixed by the
format, and the decoded CRC has nowhere to live in sixteen bytes. A page
written with `ZPSE_ENC_CHECKSUM` therefore drops the CRC field and its flag
rather than spending eight header bytes it cannot use -- the block says it
carries no checksum, which is what the decoder reads, and the stream is the
same bytes otherwise. Sixteen bytes is 0.39% of a 4096-byte page, which is
0.34 points of ratio over the three dumps (a page that goes out raw has no
header left to save, so 45 000 of the 52 065 pages shrink; the checksummed
run saves the identical 719 591 bytes). The sync value (`0x50B6` or `0x50B7`)
is what a reader uses to tell the two apart, and nothing else about the block
changes. At 16 KiB and above the header stays 32 bytes.

The largest win the page size has had is in the match finder's *decision*, not
its search. The finder's lazy step asks one question -- does the position one
byte on start an eight-byte match? -- and that is the right question at 64 KiB,
where matches are long enough that a byte of literal never pays for a second
lookup. At a page it is the wrong one: the average match on these dumps is five
bytes, so a five-byte match here and a nine-byte match one byte on is a
four-byte win the eight-byte test throws away. At page size the match is now
measured first and the next position measured too, and the byte is given up
only when the later match is strictly longer by more than it costs. That is
1 118 460 bytes over the three dumps at the default level -- 41.13% to 40.61%,
0.52 points -- for about 9% of the encode rate (129 to 118 MB/s), and
1 025 625 bytes at level 9 (40.70% to 40.22%). The mixed workload gains 0.50
points at the default level (55.62% to 55.12%) and gives up 4% of its rate. The
streams are unchanged in form, so decode is untouched.

The decision itself is the last thing to pay for. At a page, levels 1 and 2
write no LZ form at all -- 0.0% of the dump corpus's bytes at either level --
and the pricing pass that chooses between tile shapes is 68% of their sweep
cycles, pricing nine or ten candidates to pick a stream that costs a third of
that to write. `ZPSE_ENC_PAGE_FAST` (bit 6 of `enc_flags`) cuts the candidate
list to the identity, which is the shape those rungs pick on these pages and
is always built, so the pricing pass, the attempt-order sample, the compaction
and the trial cap all fall away with it. What is left is one tile trial and
the raw comparison, and the rate follows:

| 4 KiB pages | plain L1 | `PAGE_FAST` L1 | plain L2 | `PAGE_FAST` L2 |
| --- | ---: | ---: | ---: | ---: |
| dumps | 58.56% at 244 MB/s | 66.17% at 917 MB/s | 57.28% at 196 MB/s | 66.04% at 903 MB/s |
| mixed | 78.42% at 243 MB/s | 81.25% at 805 MB/s | 77.33% at 213 MB/s | 80.73% at 790 MB/s |
| Silesia | 80.06% at 240 MB/s | 82.92% at 571 MB/s | 77.49% at 222 MB/s | 80.31% at 587 MB/s |

Two and a half to four and a half times the encode for 2.8 to 7.6 points of ratio, and
decode improves with it (3.4 GB/s on the dumps) because a page written as one
identity segment is one plane the decoder walks straight through.

What is left on a page that stays raw is the trial itself. On the library dump
40.7% of the page bytes go out raw, and each of those pages used to encode all
sixteen of its noise tiles to learn what the first four already said. The
trial-level noise abort -- armed only at 8 KiB and up, because a larger block
whose front is noise may still compress at the back -- now arms for a page
under this flag too, and the flag is what makes it safe there: a page has no
sweep behind its trial to pick a shape the abort could have missed. It also
probes one sample at the page's midpoint instead of all sixteen tiles, which
is 4.6 to 11.9% of the page rate at levels 1 and 2 (dumps 876 to 917, mixed
720 to 805, Silesia 536 to 571 MB/s at level 1) for 0.009 to 0.013% of the
packed bytes -- a hundredth of a ratio point -- because the page the sample can
misjudge is one whose midpoint reads as noise and whose rest does not. All
sixteen samples is the byte-identical setting, and it is the row above that
pays the rate for it. Pages of pure random bytes go from 688 to 1,473 MB/s,
and the abort on its own was worth 9% on the mixed corpus.
The plain encoder is untouched: without the flag the abort keeps its 8 KiB
gate, and 4 KiB level 1 writes the same 125,077,793 bytes it did before. What it
gives up is the predictors -- a width-16 transpose is 16.2% of the Python
heap's page bytes at level 1 -- so it is an option and not a level: every
level's own stream is unchanged, and so is every stream this flag writes for a
block larger than a page.

It is `-P` on the CLI. A page under it is one shape at *every* level: the four
LZ stages are decisions too, so the flag declines them as well -- before that,
levels 5 to 9 still offered the token and fast LZ forms to a page (the library
dump's 4 KiB total read 75,218,131 bytes with the flag at level 6 against
104,827,267 at level 2), which is not what one shape means. From level 2 up
the pages this flag writes are within 0.003% of each other -- byte-identical
on the dumps, 170,179,999 against 170,184,257 bytes over Silesia at 4 KiB
between level 2 and levels 5 and 9 -- and level 1 is 0.27% larger (105,106,058
bytes on that dump) because its trial does less per tile, not because it
chooses anything.

What a page cannot afford is the full-size block's trial structure: a 4 KiB
block below the probe's gate would run nine full trials, and the two knobs
that stop that are the ones this size was tuned with -- the pricing gate at
`ZPSE_PRICE_MIN_N` 4096 and the tie cut's field floor at two survivors for a
page against four for anything larger, and both stop at a page: 16 KiB and
64 KiB blocks are byte for byte what they were before either knob. The parse
step above is the one thing that is not free at those sizes -- a page-sized
body inside a larger block takes it too -- so a 16 KiB block is 34 779 bytes
smaller than it was and a 64 KiB block 52 785 bytes larger (0.07%), which is
why the step is worth shipping for pages and for nothing else.

### Methodology, and why this is not science

* In-process only. `tests/benchmarks/bench_vs.sh` measures other tools by
  running their command-line programs; those numbers include process startup
  and file I/O, so none of them appear here. The table above is a library call
  over the same buffer, best of five, single thread. `bench_pareto.sh` is the
  cross-tool comparison that *is* in this file, and it writes every output to
  `/dev/null` so that what it compares is the codec and not the filesystem.
* The host has two performance states — its `memcpy` ceiling wanders between
  57 000 and 66 000 MB/s depending on thermal state — so run-to-run spread on
  a busy row is around ±10%. Any comparison worth trusting here was done as
  two binaries alternating over the same rows, best of eight, never as one
  before-and-after pair.
* The patterns are synthetic, chosen to exercise each coding mode, not to
  model a file format. They are also highly redundant in ways a
  match-finding compressor eats alive, which is what the next section is
  about.
* One machine, one compiler, no confidence intervals. Treat the table as
  "this is roughly what it does" rather than a result.

### Measuring a change

`tests/benchmarks/ab.sh` is the harness behind the alternating-binary numbers
in this file. It builds `bench_columnar.c` against two source trees, runs them
back to back in the same round (order swapped every round) and reports the
per-case encode and decode rate ratio of the second tree to the first, with
two summaries: the geomean over cases, and the time-weighted total, which is
the ratio on the whole corpus and the one to quote -- a case running at
40 GB/s contributes almost no time and cannot outweigh one at 2 GB/s.

```
tests/benchmarks/ab.sh -e 6 -r 8 /path/to/before /path/to/after
```

`bench_columnar.c` takes a trailing `dec` argument (`... 64 0 dec`) that times
the decode alone: the repeated encode pass and its timing are skipped and the
encode/eff cells print `-`, so a decode profile or A/B is not charged for what
the encoder left in cache and in the branch predictors. The ratio and every
decoded byte come from the verified stream either way, which is what makes a
decode-only cell comparable with a full run's.

Four checks decide whether a change is worth keeping, in this order:

1. **The output has to be the same bytes.** `bench_levels.c` prints every
   (case, level, size) cell and memcmp-verifies each block's decode before it
   prints the row, so diffing its size column between the two trees is the
   format statement: an encoder-side change has to leave those numbers alone,
   or it is not a speed change.
2. **The suites have to pass**, in `tests/runners/run.sh` (roundtrip, fuzz,
   hostile, sizes, dense, decide, and the `-DZPSE_FORCE_SCALAR` and
   `-DZPSE_FORCE_TRIAL_SWAP` parity digests). A kernel that changes bytes on
   an input the corpus never generates is exactly what the parity builds are
   there to catch.
3. **A kernel needs a differential test of its own.** The suites pin the
   frame and block paths; a width-specific kernel needs every length and
   every row count that reaches it. Calling `zpse_transform_forward` and
   `zpse_transform_inverse` in a loop over `n = 1..1024`, with the digest of
   the plane bytes and the status of each call printed, and comparing that
   digest across the default and `-DZPSE_FORCE_SCALAR` builds, is the form
   used for the narrow-width kernels.
4. **The ratio has to reproduce**, twice, with the sign stable per case. A
   change that moves every case by 1% in a different direction is noise, and
   the harness will say 1.00 because it is.

Where to look first is decided by a profile, not by taste: `perf record` /
`perf annotate` over the columnar corpus, with the encoder or the decoder as
the only thing in the timed region. Which transform each corpus case selects
is the companion fact, because it says which half of the codec that case
exercises (this is the `transforms(...)` line `bench_columnar` prints):

| case | transform |
| --- | --- |
| `bools_nulls` | RAW |
| `decimal8` | T8 |
| `dict_u8` | RAW (51/64), T2 (13/64) |
| `dict_u16le` | T2 (55/64), DT2 |
| `dna_2bit` | RAW |
| `float_mix` | DT16 |
| `float_sparse` | RAW (56/64), T2 |
| `geo_point` | RAW |
| `i32_seq` | DT4 |
| `i64_mixed` | DT8 |
| `json_rows` | RAW |
| `log_lines` | RAW |
| `rle_lens` | T4 (58/64), DT4 |
| `sorted_near` | DT4 (62/64), DT8 |
| `sparse_mix` | RAW |
| `struct_offsets` | DT4 |
| `text_en` | RAW |
| `ts_gapped` | T4 |
| `ts_ns` | DT8 |
| `ts_seq` | DT8 |
| `ts_us` | DT8 |
| `utf8_multi` | RAW |
| `uuid_mix` | RAW |
| `var_string` | RAW |
| `zeros_burst` | RAW (61/64), T2 |

RAW means the block kept its bytes (the intra-block LZ form can still ride on
top of that at levels 5-9), T* is the plain width-* transpose, DT* the delta
transpose. The rows that dominate the decode time budget are the DT4/DT8 ones
-- their inverse is a chain per plane plus a merge -- and the rows that
dominate the encode budget are the RAW ones, where the sweep and the match
finder are the cost.

### Ratios against zstd

Same corpus, 4 MiB per pattern, `zstd -3` on the command line. Only the
compressed size is compared: it is the one thing a different harness cannot
change.

| pattern | zpse | zstd -3 | |
| --- | ---: | ---: | --- |
| `zeros` | 0.0010 | 0.00004 | zstd 29x smaller |
| `const5A` | 0.0010 | 0.00004 | zstd 29x smaller |
| `incr` | 0.0016 | 0.00016 | zstd 10x smaller |
| `strided` | 0.0017 | 0.00034 | zstd 5.1x smaller |
| `sparse` | 0.0464 | 0.00021 | zstd 219x smaller |
| `sparse2` | 0.0163 | 0.00023 | zstd 69x smaller |
| `alt00FF` | 0.0012 | 0.00010 | zstd 12.8x smaller |
| `sawtooth` | 0.0018 | 0.0626 | **zpse 35x smaller** |
| `lowentropy` | 0.5079 | 0.5196 | **zpse 2.3% smaller** |
| `random` | 1.0005 | 1.0000 | tie |

That is the honest picture, and read the table as the shape of the comparison
rather than as the default level's numbers: the intra-block LZ form below moves
`var_string` from 1.09 to about 0.34 at the default level. zpse has no entropy stage
and no cross-block window -- a match distance is 16 bits inside the block, so a
block stays independent -- and its tile model is "one of ten predictors over
256-byte tiles, code the exceptions enumeratively", with a single-probe hash
match finder (LZ4's shape, not a search) offered as one more candidate at
levels 5-9. On the ramp and one-value patterns zstd's finder still turns almost the
entire file into a handful of copies and beats zpse by one to two orders of
magnitude (5x on the stride, 29x on a constant); on the sawtooth and the
sixteen-value stream it is zpse that wins, by 35x and 2.3%, because a
repeating ramp is a prediction rather than a match. Where the data
has no redundancy to match but does have a small
alphabet, zpse's tile model is competitive, and on truly random data both are
at 1.000.

So: pick zpse when decode latency, allocation behaviour and output
determinism matter more than ratio, and pick zstd when the bytes matter more
than the clock.

### Columnar data

The ten patterns above are byte streams. Real columnar data is not: it is
typed records with a known width and a known order. Seven such corpora are in
the benchmark set — `ts_seq`, `ts_ns`, `float_sparse`, `var_string`,
`sparse_mix`, `geo_point`, `struct_offsets` — and they are the cases where
"pick zstd" is the wrong advice, because the right answer is not a block codec
at all.

Same harness as the table above (4 MiB per pattern, `zstd -3` on the command
line, only the compressed size compared):

| column | zpse | zstd -3 | |
| --- | ---: | ---: | --- |
| `ts_seq` (i64 µs timestamps) | 0.1770 | 0.2564 | **zpse 1.4x smaller** |
| `ts_ns` (i64 ns timestamps) | 0.1636 | 0.2564 | **zpse 1.6x smaller** |
| `float_sparse` (f32 readings) | 0.3239 | 0.2398 | zstd 1.4x smaller |
| `var_string` (JSON-ish records) | 0.3382 | 0.2123 | zstd 1.6x smaller |
| `sparse_mix` (nullable i64) | 0.1238 | 0.1020 | zstd 1.2x smaller |
| `geo_point` (lat/lon + cell) | 0.0538 | 0.0338 | zstd 1.6x smaller |
| `struct_offsets` (i32 offsets) | 0.2245 | 0.7808 | **zpse 3.5x smaller** |

Three of those rows are worth reading twice.

`struct_offsets` is a strictly increasing array of byte offsets — nothing in
it repeats, so zstd's match finder has nothing to match and lands at 0.78,
worse than storing the raw column. zpse's stride and lane-delta predictors
read the monotone sequence directly and get 0.25. *A block codec that predicts
instead of matching is not simply a worse match finder; on data with no
redundancy to match, it is a different codec, and it can win.*

`var_string` is the other end: high-cardinality JSON-ish text, where every
value is unique. The cheap levels fall back to raw (1.09) because none of the
ten tile predictors help, and zstd's match finder finds the repeated keys and
gets 0.21. This is the tile model's own limit showing up on realistic text
rather than on a synthetic ramp, and without the LZ form it is the row that
decides against zpse for this data. With the LZ form it is a different row:
`var_string` at the default level is about 0.34 (2.96:1), which is the effort
level table's number rather than the 1.09 a tile-only sweep reaches.

The timestamp rows are the interesting middle: both are narrower than zstd's
0.26, by about a third, which is what a delta-of-delta predictor buys over a
match finder on a monotone integer column.

### What the structure is worth

The zstd column above compares two block codecs. It cannot say what the
structure is *worth*, because zstd is not exploiting it as a column — it is
exploiting it as something a match finder happens to like.
`tests/benchmarks/bench_colref.c` answers that by running, for each column, a
codec built for that shape — a delta-of-delta for nanosecond timestamps, a
null bitmap for a nullable column, a frame of reference for increasing
offsets, Gorilla XOR for repeated floats — against zpse on the same bytes.
Both sides see 64 KiB chunks, so only the coding model differs.

| column | reference codec | reference | zpse | |
| --- | --- | ---: | ---: | --- |
| `ts_seq` | delta + varint | 4.00:1 | 5.65:1 | **zpse 1.4x smaller** |
| `ts_ns` | double delta | 4.56:1 | 6.11:1 | **zpse 1.3x smaller** |
| `float_sparse` | Gorilla XOR | 2.68:1 | 3.09:1 | **zpse 1.2x smaller** |
| `var_string` | *(none helps)* | — | 2.96:1 | block codec territory |
| `sparse_mix` | null bitmap | 10.44:1 | 8.08:1 | zpse 1.3x bigger |
| `geo_point` | zigzag + delta | 6.10:1 | 18.59:1 | **zpse 3.0x smaller** |
| `struct_offsets` | frame of reference | 2.56:1 | 4.45:1 | **zpse 1.7x smaller** |

The reference codecs are deliberately plain reference points, not proposals —
`tests/suites/test_colref.c` proves each one round-trips, so every size here
is one a real codec reached rather than an estimate.

Five rows deserve their own note. On `ts_seq`, `ts_ns`, `float_sparse`,
`geo_point` and `struct_offsets` zpse *beats* the purpose-built codec, and the reason is
visible in the stream: the encoder picks a delta transpose, which for a
fixed-width record is exactly the column split a columnar format does by
hand, and its stride and lane-delta predictors then replay the recurrence the
specialist hard-codes. zpse finds it without a schema, because splitting
bytes into planes and predicting each plane is what its model already does.
That is the strongest argument in the tree for the byte-oriented design, and
it is an accident worth naming.

On `sparse_mix` the gap is real and it is the one that would cost something:
null suppression is a typed feature zpse does not have, and a 92%-null
column is worth 10.44:1 to a codec that knows the nulls are whole records.
1.3x is what zpse pays for not knowing.

`float_sparse` used to be a tie -- 2.68 against 2.72 -- and is now 2.68
against 3.09: Gorilla's XOR and zpse's stride model still arrive in the same
place, but the stride model gets there with a fifth fewer bytes. The
`var_string` row is the honest limit: no column codec helps there either, and the answer
is a match finder -- which is what the intra-block LZ form gives it at levels
5-9, inside the block and nowhere else.

### Effort levels

`ZPSE_ENC_*` in the `enc_flags` word chooses how much encode time a block is
worth. There are nine levels, 1 = FASTEST to 9 = MAX, stored in bits 1-4 as
`level << 1`; a field with no level bits is the default, level 6, and an
unsupported level is rejected with `PSE_BAD_ARG` rather than rounded to
a neighbour. No level changes the format: every level's stream is read by the
same decoder, and a level only decides which shape gets written. All nine are
rungs, not aliases: 1 to 4 buy trials one at a time, and 5 to 9 loosen the
intra-block LZ screen and raise its decode bill a step at a time. Each
level's knobs are one row of a table in `src/block.c` (`zpse_level`), and
`tests/suites/test_levels.c` pins that no level writes a block larger than the
one above it. All nine were measured in one binary over the columnar corpus
(25 cases, 4 MiB each, 64 KiB blocks, best of three runs of the whole matrix,
each run the harness's own best of three timed reps, on one host,
`tests/benchmarks/bench_levels.c`):

| level | mean ratio | mean encode | mean decode | what changes |
| --- | ---: | ---: | ---: | --- |
| 1 `ZPSE_ENC_SPEED_FASTEST` | 4.01:1 | 2.17 GB/s | 18.4 GB/s | one trial per block: the probe's winner, on the cheap prescreen that tries the three uniform predictors only; no match finder |
| 2 | 4.09:1 | 1.90 GB/s | 18.5 GB/s | the full nine-predictor prescreen, two full trials, no match finder |
| 3 `ZPSE_ENC_SPEED_FASTER` | 4.09:1 | 1.81 GB/s | 19.0 GB/s | the same prescreen, three full trials |
| 4 | 4.09:1 | 1.78 GB/s | 19.2 GB/s | four full trials, still no match finder |
| 5 `ZPSE_ENC_SPEED_BALANCED` | 5.12:1 | 0.99 GB/s | 12.3 GB/s | every candidate priced on a tile of every plane, a guarded full-size field cut to its best two (see below), and the LZ form offered on a 1.25x sample win, 8 bytes per command |
| 6 (default) | 5.16:1 | 1.01 GB/s | 12.5 GB/s | the margin loosened to 1.1875x and the decode bill raised to 6 bytes per command — the bill is free on the encode side, so the default takes it; see the Silesia section for what the two knobs are each worth |
| 7 | 5.16:1 | 1.01 GB/s | 12.4 GB/s | the margin loosened to 1.125x, the decode bill unchanged from 6 |
| 8 | 5.16:1 | 0.96 GB/s | 13.0 GB/s | the margin loosened to 1.0625x, acceleration factor 20 (4.8 bytes per command), and the field cut back to a tied tail only |
| 9 `ZPSE_ENC_SPEED_MAX` | 5.16:1 | 0.92 GB/s | 12.6 GB/s | the same 1.0625x margin with one screen sample instead of two, acceleration factor 24 (4 bytes per command), and the fast LZ form as a last candidate |

Levels 5 to 9 are the v2 form's rows: `ZPSE_B_LZ_TILE` lifts the corpus mean
from 4.49:1 to 5.16:1 at the default level and costs about a fifth of the
encode rate, because every block it wins pays a full match-finder pass over
the tile body that the tile form alone never needed. Raising
`ZPSE_LZ_TILE_DIV` (the 1/32 threshold in `src/block.c`) trades straight back
along that line: at 16 the corpus grows 63 KB for no measurable speed.

Levels 3 and 4 read as the same mean ratio because on this corpus a third or
fourth candidate almost never wins: the extra trials cost encode time (1.81 to
1.78 GB/s) and buy the odd block, which is worth a rung to the caller who
wants it and worth nothing to the average. Level 2 is the visible step in that
tier, because the prescreen it switches to is what changes which shape the
winner is.

Levels 6 and 7 read the same mean encode rate as each other on this corpus
because it has almost nothing for the match finder to win: the accel knob that
level 6 took from 7 filters streams that were already produced, so it costs
nothing here, and the margin that is still 7's alone is measured on Silesia,
where there is text for it to buy.

One shape of block is the exception to "every survivor is encoded in full": a
full 64 KiB block whose field the window price has already ordered. The
survivor margin keeps everything within 1/16 of the best price, and on a
homogeneous column -- a byte dictionary, an id column -- every transposed
shape prices the same record width on the same data, so five survivors land
inside a percent of each other and the sweep would encode all five in full to
separate them by a couple of hundred bytes. A full trial costs about six times
a window price, so a price that separates two shapes by more than the 1/64 it
cannot resolve is evidence the trial only confirms: levels 5-7 keep the two
cheapest survivors of any such field, and levels 8-9 keep every survivor the
price could not separate and drop only a tied tail.

The cut is taken only when four things all hold, each of them a case where a
tied price is known not to describe the block: the block is a full-size one,
the head window was not contradicted by the tail window, no survivor was kept
for its content price rather than its total, and there are at least four to
choose from. Within those guards it changes no output: the 25-case columnar
corpus writes the same ratio on every row and the 768 cases of the size gate
grow by none. What it buys is encode time on the blocks with wide fields --
`rle_lens` 1.82 to 2.51 GB/s, `float_sparse` +9.5%, `zeros_burst` +8.0%,
`dict_u8` +4.6%.

Levels 5 to 9 are the ones to reach for when bytes matter more than the
clock, and the intra-block LZ form is what they buy. Its match finder is a
full pass over the block at roughly a quarter of the tile encoder's rate on
text, so no level pays for it blind: a sample of the block's head is coded
first, and the pass runs only when that sample shows LZ winning by the level's
margin. The sample's win is measured against the same head priced as tiles,
which is like for like but says nothing about the block: a 64-byte plane
carries 26 bytes of envelope and header, so a wide transpose that wins the
block can lose the sample by a wide margin, and on such a block LZ is offered
a pass it can only abort. So the sample is also scaled to the block and
compared with the size the sweep actually reached, and a block both the head
and the middle convict loses the pass -- which is what turns `uuid_mix` from
370 MB/s of aborted match finding into 1.2 GB/s. The margin is what separates
the levels: at 1.25x the default takes
the form on JSON, logs and variable strings; 1.1875x and 1.125x reach a little
further; and 1.0625x, which levels 8 and 9 share, is the only margin loose
enough to find it on prose. Level 9 also drops the screen's second sample, so
more blocks pay for a pass -- that is what keeps 8 and 9 apart. Where the tile
model has nothing to predict it is the
difference between raw and compression:

| case | `ZPSE_ENC_SPEED_BALANCED` | `ZPSE_ENC_MAX` | tile stream alone |
| --- | ---: | ---: | ---: |
| `json_rows` | 2.95:1 | 2.95:1 | 1.09:1 |
| `log_lines` | 3.40:1 | 3.40:1 | 1.09:1 |
| `var_string` | 2.96:1 | 2.96:1 | 1.09:1 |
| `utf8_multi` | 3.11:1 | 3.11:1 | 1.00:1 |
| `text_en` | 1.88:1 | 1.88:1 | 1.09:1 |

The form is kept only when it is smaller than the best tile stream, so no
level pays for it in size. Nor does a level pay for it in trials: the sweep
still prices every tile shape, and what a wide field would spend on full
encodes it gets back from the prune instead (see above). The
ASCII gate below replaces the tile-side sweep, not the match finder, which is
offered afterwards whatever the gate decided. What the form costs is decode
speed, and that is what the level's **acceleration factor** decides: an LZ stream decodes at the
speed of its command loop, one command per output run, so its rate
follows the average run length. Over 64 KiB blocks, at six bytes per command
prose decodes at about 2.5 GB/s, at eleven bytes JSON at 4.5, at twenty bytes
variable strings at 5.9, and at a hundred or more the long-run patterns at 5
to 6.5 -- against 5-13 GB/s for a tile stream. The match copy described below
is what keeps the short end of that range from costing more than one command
per match.

The acceleration factor belongs to the level: 12 at levels 5-6, i.e. a command
every 8 output bytes (a floor of about 1 GB/s at the command loop's cost per
command), 16 at level 7 (every 6 bytes), 20 at level 8 (every 4.8) and 24 at
level 9 (every 4); levels 1-4 write no LZ form, so they have none. That
is what lets `text_en` reach 1.33:1 at MAX while the default level, at its
1.25x screen and a factor of 12, leaves prose alone -- at a factor of 8 the
JSON and log cases would be given up as well and `var_string` alone would keep
its 2.52. Each level's factor and each level's sample margin are one constant
each in `src/block.c`, and lowering the default level's factor to 4 gives back
its ~14 GB/s decode at the cost of the whole text column -- the knob to turn
if the default level is being used for its decode rate rather than its ratio.

A block the form wins decodes as a copy loop instead of a tile
reconstruction, which is slower than the tile path on columnar data (hence
the lower mean decode above, and it is the default level that pays it) and
much faster than it on a long run: 5.8 GB/s on 1 KiB runs, 27 GB/s on a ramp,
measured over 64 KiB blocks. Levels 1-4 are the other side of the same trade:
they write the same tile shapes in less encode time, and they do not run the
match finder at all, so the factor above is theirs only if
`ZPSE_LZ_CHEAP_LEVELS` is turned on -- which the measurements argue against,
because FASTEST's text encode would fall from 2.2 GB/s to 390 MB/s, slower
than the default level it is supposed to be faster than.

### What a match costs

On the text corpora four fifths of the matches are under sixteen bytes at a
distance over thirty-two, so the copy a match is written with, not the match
length, sets the rate of the command loop. A ten-byte match at distance 300
has to cost one store, not a wide copy followed by two byte moves.

A decoder fills its block strictly forward, which is what allows a match to be
written with **one store**, as long as the store
covers the match, its width does not exceed the distance (so its load cannot
reach bytes it has not itself written), and the store stays inside the output.
The bytes it writes past the match are ones a later command overwrites, so the
slack is reclaimed rather than leaked, and the store is only taken when the
room before `dst` covers its width -- a match at the very end of a block gets
the exact byte loop instead. `lz_copy_literals` is the same shape with the
same bound, which is what lets a block be decoded into exactly its own bytes
and no more.

That is why a command costs about the same whatever its match length, and why
the stream's rate follows its bytes per command rather than its ratio: the
command loop is the ceiling on prose, JSON and variable strings, and only on
the long-run patterns -- where a match spans several stores -- does the copy
width take over as the cost.

### Sixteen positions at a time

The stride predictors reconstruct a tile by replaying a recurrence: a position
is its prediction -- the byte one stride back -- unless the mask marks it an
exception, in which case its residual inverts against that prediction. The
scalar loop is branchless and unrolled by the stride, but it is still one
position's work per position -- about twelve uops, three cycles per byte --
and it is most of the decode cost of every columnar row that depends on it:
`ts_gapped`, `float_sparse`, `ts_us`, `sorted_near`. Those rows are why the
decoder reconstructs sixteen positions at a time.

For a stride of two or more, the recurrence is a **shift** of the group's own
result, and a shift by s bytes of the group in progress concatenated with the
the sixteen bytes before it is exactly what `alignr` computes. So sixteen
positions go through one round of
`v = blend(alignr(v, prev, 16 - s), exception(alignr(v, prev, 16 - s)), mask)`
repeated `ceil(16 / s)` times: each round finalises one more hop of every
chain, and since the exception value is a pure function of the prediction, a
lane written early with a stale prediction is simply rewritten correctly once
the lane it reads is final. The group's fixed cost is the mask and unit
gather -- one shuffle turns the sixteen mask bits into a byte blend mask, a
byte prefix sum over those gives each lane its exclusive exception count, one
`pshufb` pulls that lane's residual unit out of the sixteen-unit window, and
the prefix sum's last lane is the exception count that advances the unit
index, so no popcount is needed.

Stride 1 and stride 8 have vector kernels of their own (one chain, and the
eight-lane recurrence respectively); strides 2, 3 and 4 take this one. The head before the first sixteen positions stays scalar, as does the rest of the
tile when the unit window would read past the 256-unit array. In the head the two
sources below the stride -- the initial literal, then the segment's earlier bytes
-- are split into ranges rather than selected per position, because the test that
picks between them is loop-invariant over each range. On the rows that
depend on it, decode over 64 KiB blocks:

| column | ratio | decode |
| --- | ---: | ---: |
| `ts_gapped` (timestamp with gaps) | 2.88:1 | 1.8 GB/s |
| `float_sparse` (f32 readings) | 3.09:1 | 3.7 GB/s |
| `ts_us` (i64 µs timestamps) | 5.92:1 | 3.7 GB/s |
| `sorted_near` | 4.80:1 | 2.1 GB/s |
| `ts_seq` (i64 µs timestamps) | 5.65:1 | 5.2 GB/s |

Two of the group's costs are the same lesson: a value the loop has already
produced should not be read back out of memory. A group's `prev` -- the sixteen
bytes it shifts against -- is the vector the group before it stored, so it
lives in a register and both branches of the step hand it forward; `out` is read
once, after the scalar head. The classify's mask walk has the same shape:
exceptions are walked with `ctz` and a clear-lowest-bit, so a sparse tile costs
one iteration per exception rather than sixteen per vector, and they still
arrive in ascending position, which is the only thing the record depends on.

`residual_copy_bytes` -- the mode-0/mode-1 payload copy, k bytes with k at most
256 -- moves thirty-two bytes a step, which is two 16-byte moves and the half
loop that implies, with the eight-byte loop finishing the last partial step.

The other half of the decode is the inverse transform: the width-4 and width-8
delta transposes come back through one merge kernel each, a plane per input
register, a chain per plane to undo the delta, then three levels of byte
interleave to rebuild the rows. The chain was the part paying for memory. It
ended every 32-byte chunk by extracting the chunk's total into a GPR, adding it
to one byte of `carry[]`, storing that and broadcasting it back at the top of
the next iteration -- a store-to-load round trip inside the loop, with two
shuffle-port ops either side of it. Keeping each carry as a broadcast in a
vector register instead, the running value in all 32 bytes and advanced by one
`pshufb` of the chunk's own last byte, removes the round trip and the GPR
entirely; the eight bytes are written back once, after the loop, for the 16-row
form that continues the chain. Width 4 already worked that way and width 8,
where `ts_us`, `ts_ns`, `ts_seq`, `i64_mixed` and `sorted_near` spend their
decode, did not, and neither did width 2. Both now hold each plane's carried
byte as a broadcast in a vector register and write it back once, after the
loop, rather than extracting the chunk's total to a GPR, adding it to
`carry[]`, storing that and broadcasting it back at the top of the next
iteration.

The chain's ladder was the next thing to measure, and on half the width-8 rows
it turned out to have nothing to do. A delta plane is all zeros wherever its
source plane is constant, and a constant plane is the common case for the high
bytes of a wide column: an i64's four high bytes, a padding or flag byte, an
offset's top bytes. Undoing the chain on a 32-byte chunk of zeros is the
identity -- the prefix is zero, so the plane is the incoming carry and the
carry does not move -- yet it paid the same five shuffle-port shift-adds,
cross-lane link and broadcast as any other chunk. One `testz` now answers that
question and returns the carry, so the zero chunks stop paying. Decode over
the corpus: 4.9% time-weighted (5.2% geomean), which is `i64_mixed` 45%,
`i32_seq` 23%, `struct_offsets` 20%, `ts_ns` 13%, `ts_seq` 12%, `sorted_near`
11%, `ts_us` 10% and `dict_u16le` 10%, against nothing measurable on the
columns whose delta planes are dense. That split is the statement that the
test is where the win is, not that a branch is free.

The encode side of the same transform is a per-plane delta chain that runs
after the parity split, and it now steps 32 bytes at a time instead of 16.
`vpslldq` is lane-local, so byte 16 of a chunk cannot come from the shift: one
permute puts lane 0's top byte into both lanes, one shift right fifteen moves
it down to byte 0, and a blend restores the carried byte in lane 0 -- three
ops, and the same value is what the next chunk's carry becomes after one more
shift. Text, JSON and the two-byte rows gain 1.5 to 3.7% of encode.
`dna_2bit` reads -1.6%, which is inside this host's drift floor but is also
why the change reads as 0.5% on the corpus rather than as the sum of its
parts.

Two attempts were measured and dropped. A stride-family group's mask as one
16-bit load instead of a shift and a mask of the 32-bit word, plus the group
loop's unit-window bound replaced by sixteen bytes of slack in the caller's
two arrays, remove one compare and one branch per group between them -- and
came out at -0.4% on `ts_gapped`, the case that kernel exists for. A
sixteen-position group there is about eight uops and the bound was never the
cost. Folding the copy into the CRC's three-chain pass, one 16-byte load per
region feeding a store and two `crc32` steps, measures zero: the checksum
saturates one port at eight bytes per instruction, so overlapping it with the
store traffic buys nothing and the extra code is not worth a tie.

The LZ command loop is bound per command, not per byte: a command emits five to
eleven bytes and costs about twelve cycles whatever its shape, so the copy
widths are not where its time is. Its shape is. Nine commands in ten on a text,
JSON or log block are at most eight literal bytes and a match that fits in
sixteen, at a distance of at least sixteen; those take one branch and two
unconditional stores, with no tail and no width cascade in either copy, and
fall through to the general path when a bound misses. Every bound the general
path tests is in that one condition, so a stream it refuses is refused there
too, and both stores over-copy into bytes a later command writes -- the slack
argument the literal and match copies already make.

The same shape one width up -- at most eight literals and a match of 17..32
bytes at a distance of at least thirty-two -- is a second branch rather than a
width selected inside the first, because a select in front of the first path
cost `json_rows` 5% of the rate it had (4.57 -> 4.34 GB/s) for a branch it used
to take one time in twenty-five, and nothing is paid for the second unless the
first has already declined. It is not a rare tail: on `log_lines`
2.33M of its 4.98M one-shot commands are matches of exactly that shape, a
third of the block, and each of them used to fall through to `lz_copy_match`'s
width ladder and doubling loop for what is still one overlapping store. Over
the corpus, LZ decode alone: `log_lines` 4.57 -> 4.99 GB/s, `var_string`
5.71 -> 5.95, `json_rows` 4.56 -> 4.60, and the whole decoder 1.0%
time-weighted (0.5% geomean), with no other case moved by more than noise.

Two of the one-shot bound's terms are questions about the block rather than
the command: whether an eight-byte literal load stays inside the input, and
whether the output has room for the widest pair of stores. The block-wide
half is answered once, before the loop, by folding it into the room bound --
a block without that padding asks for more room than the block can have, so
the test stays one compare instead of two. Between them the two branches then
ask five things where they asked six: `json_rows` 3.2%, `var_string` 3.1%,
`log_lines` 1.2% and `sorted_near` 0.7% of decode, 0.8% time-weighted over
the corpus. That is also the loop's whole budget. Keeping `min(lit_rem, 8)`
in a register instead of recomputing it -- one instruction cheaper, one value
more live, the same idea one step further -- measured 20-27% *slower* on those
same cases: at six live counters the loop spills, and the spill costs more
than the min did. The same arithmetic in the other direction (folding the
command-count test into the loop condition) is worth nothing on the JSON and
log rows, so the counter stays where it is.

The parity, fuzz and least-effort suites report identical digests in the
default and `-DZPSE_FORCE_SCALAR` builds, which is the statement that this
kernel writes exactly the bytes the scalar loop writes.

### Choosing the predictor

Every tile the encoder searches is asked one question first: of the nine
cheap predictors, which misses fewest? That is `seg_prescreen`, one vector
pass that counts all nine miss counts at once, and its answer decides which
one or two predictors are then classified exactly. A count is not a price --
two predictors that miss one position each usually pack to the same size --
so the encoder keeps the runner-up as a second candidate only when the two
counts tie exactly, and the pair of them is what the tile loop needs.

The pair comes out of one walk over a packed key per candidate -- miss count
above, priority below -- so the comparison is a single `<` whose result is the
decision a two-term test makes, ties included. The priority table is the winner
rule the format is defined by; the scan order (`seg_screen_rank`) is free to
change and changes nothing, because the key never reads it.

Two things in the tile loop follow from the fact that a plane is not always a
block. The repeat cache is keyed and zeroed per segment, 1.5 KiB of `memset`
that a 64 KiB plane does not notice and a 4 KiB plane pays 96 bytes a tile
for -- and a 4 KiB plane is what a width-16 transform hands the loop, sixteen
times a block. Below 32 tiles the cache is therefore not set up at all: a hit
needs a whole 256-byte tile to recur, sixteen tiles of a transposed plane are
different records, and a miss costs nothing but the reuse it did not find,
because the general path writes the record the copy would have written. And
the dense classify counts its distinct residuals as it folds, stopping at
seventeen -- the only value the count is ever compared against -- so the
bitmap it was just built from is never recounted. The
classification's capped count and its trimmed `seen` set agree with the
scalar path, which is what the parity and least-effort digests state.

### A block that does not compress

The sweep prices every candidate before it encodes one, and then encodes the
survivors in full to see which is smallest. On a block that compresses that is
money well spent. On a block that does not -- noise, an already-compressed
file, a column of random identifiers -- the survivors are all going to lose to
the raw form, and the sweep spends three whole block encodes to find out.

The sweep already knows. Its price over the head window of the block, and its
price over the tail window, both say that the block does not compress: no
shape of it writes fewer bytes than the block itself. When the first trial
then runs into the raw envelope -- the budget every trial carries, and the
reason a trial that cannot win stops instead of finishing a payload that is
thrown away -- that is the same verdict from the only side that gets to see
the whole block, and the sweep stops there.

The verdict is the trial's, not the price's, which is what keeps it honest: a
block whose head and tail are noise but whose middle is not -- the "noise then
constant" shape of the size corpus -- has a price that says flat and a trial
that says otherwise, and the trial wins because it encodes the whole block. A
block that compresses still gets its compression; only a block that really is
noise stops after one trial. No case of the size corpus (768 cases) grows a
byte for it, and the corpus mean encode rate is about 1.01 GB/s.

The rows it is worth most on are the ones whose head and tail are noise:

| case | encode |
| --- | ---: |
| `utf8_multi` | 477 MB/s |
| `bools_nulls` | 1 660 MB/s |
| `dna_2bit` | 2 000 MB/s |

At the deepest level, where the match finder is the candidate that can still
see such a block, `utf8_multi` encodes at 476 MB/s and `dna_2bit` at 1.47 GB/s,
with the same ratio on every case.

## What it does

A **frame** is a 32-byte header (magic `PSE1`, version, content size, block
count) followed by blocks. A **block** is at most 64 KiB: a 32-byte header, an
optional transform record, a segment directory, then one payload per segment.
A **segment** is one plane of the block — 1, 2, 4, 8 or 16 of them, split by a
coordinate transform — coded as a sequence of 256-byte **tiles**.

Each tile picks one of twelve predictors (`const zero`, lane previous, lane
delta, order-1, order-2, stride 1/2/3/4/8, tile constant, lane average),
records which bytes it failed to predict, and codes those positions with an
enumerative (combinadic) rank and their residuals through an exclusion map.
Seven mask modes and five residual modes are chosen per tile by a priced
search.

The transforms are identity, delta-8, xor-8, byte-transpose of width 2/4/8/16,
delta-plus-transpose at the same widths, and a stride variant for interleaved
data. A plane that is one value throughout has a dense form that codes 4 KiB
in twelve bytes.

A block may instead be coded as **intra-block LZ** (block flag `ZPSE_B_LZ`).
That form is a literal stream, a command stream and nothing else: every
reference is a 16-bit distance inside the same block, so a block is still
independent and still decodes on its own. A command is a token byte -- four bits of
literal count, four of match length minus four, a nibble of 15 meaning "more
follows" -- then the literals it names, then a variable-length distance
(7-bit groups, so a repeated distance costs about a byte). The four most
recent distances are numbered 1 to 4 and cost nothing to name again, which is
worth a quarter of the matches on executable and archive data and almost
nothing on prose; long runs are repeated commands, and every one after the
first is a byte shorter.

The decode is therefore a command loop, and measured over Silesia it costs
**23 cycles per command plus half a cycle per byte** (fitted from the streams
themselves: 0.084 commands per byte, 9.7 bytes per match, one stream or the
other on 156 of the corpus's 194 MB). At that command rate four fifths of an
LZ decode is per-command work rather than copying, which is why the loop is
written for the shape the commands actually take: 88% of them are a short
match at a distance over sixteen, which is one 16-byte store.

Writing a command costs the encoder the same per-command shape, and on prose
the commands are small enough that it is the whole hot path: at the default
level a Silesia prose block emits 10,440 commands for 64 KiB, one every 6.3
bytes, on an average match of 5.4 bytes. So the writer is written for the
common command rather than the largest one. A bounded room test -- no run can
need more than eight bytes of framing plus its literals -- replaces working
out the exact size first, which cost a divide and a shifting loop per command.
The four recent distances are four compares and selects rather than a loop
with a `break`, because on prose that branch is a coin toss and mispredicts;
it was 4% of the encode on its own. A run of up to sixteen literals leaves as
one 8-byte pair instead of a `memcpy` whose size dispatch costs more than the
bytes. All three are byte-exact: 254 to 280 MB/s over the corpus, same
103,921,535 bytes, stream for stream.

Levels 5 to 9 try that form as one more candidate and keep it only when it is
smaller than the best tile stream. All of them screen it on a sample of the
block first, because the match finder is a full pass at a quarter of the tile
encoder's rate; what separates the levels is how small a sample win is worth
that pass, so the default level takes it on JSON, logs and variable strings
and the deepest level also takes it on prose.
On high-cardinality text it is the difference between falling back to raw and
compressing at all -- the case the tile model cannot see, because nothing in
it repeats four bytes apart, only further back.

A block is written in one of five forms, and the block flags say which: the
tile form above (every form flag clear), a verbatim copy (`ZPSE_B_RAW`),
intra-block LZ (`ZPSE_B_LZ`), **LZ over the tile body** (`ZPSE_B_LZ_TILE`), or
the **fast LZ form** (`ZPSE_B_LZ2`) -- the same literal-and-command shape with
a fixed 16-bit distance, a four-entry repeat distance coded in one byte, no
lookahead and no distance varint, which is what lz4 writes and why its decoder
is a tight loop. MAX
offers it as a last candidate, after the LZ-over-tiles stage: a fixed-width
distance can beat the first form's variable-length codes when matches are
short and far (a 300-byte period codes to 23,318 bytes against 23,449), and it
is kept only when it is strictly smaller than everything already tried.

The match finder is also asked FIRST, before the sweep, on the blocks it can
be. The screen above compares the two forms on a 1 KiB sample and costs a
couple of prescreens; the sweep learns the same thing by encoding the whole
block, one survivor at a time. So when the screen convicts a full-size block
-- and the top of its list is the identity shape, which is what the tile
encoder falls back to on text -- the finder's pass runs there and then, and
the sweep behind it does not run at all: on `nci` and `ooffice` the two forms
land a tenth of a percent apart on size, so the sweep was only choosing
between two streams it could not separate. Over Silesia that is the default
level's encode rate from 0.27 to 0.37 GB/s for 0.5% more bytes, and it is a
measured trade rather than a free one: the sample prices the identity shape
and is wrong about a ramp, so a block whose finder stream comes out under
1/64 of its input keeps its sweep. That is exactly the set of blocks the
transpose owns outright -- a block of zeros is 69 bytes of tiles against 802
of finder -- and with the rule held to it the size suite's contract is still
"no case grows".

A gated block goes one step further, because its comparison needs no sample:
the seven-bit form's total is arithmetic (`zpse_block_ascii_total`), so when
the finder's stream beats that number the block is written from the finder
and the tile trial -- which is handed the LZ total as a byte budget and stops
before the end of the block -- does not run at all. That trial could not win
(the total it is compared against is the one it provably produces), so the
stream is unchanged and the work is not done. It is a small win on Silesia
because the 1 KiB screen refuses the finder on most prose in the first place
(`webster` 240 against 246 MB/s, within this host's run-to-run noise), and a
large one wherever that screen does convict.

Sizing the sample by the block instead of a kilobyte is the obvious next
lever there, and it is measured and *not* taken. At 4 KiB the finder wins on
exactly the text a kilobyte cannot see, and over the twelve Silesia files the
default level goes from 2.03:1 to 2.16:1 (104,474,009 bytes to 98,023,866;
`dickens` 1.16 -> 1.51, `osdb` 1.13 -> 1.82, level 9 2.18 -> 2.19): a 6%
smaller corpus, which is past `lz4 -1`'s 2.10:1 instead of behind it. It costs
the columnar corpus more than it buys there, though -- levels 5, 6 and 8 go
5.06 -> 5.00, 5.10 -> 5.04 and 5.13 -> 4.88, and at level 8 it breaks the
ladder's ordering against level 7 -- because a longer sample does not only
measure better, it argues better for a form the tiles beat by a few percent on
a column whose records a short sample has too few of. Taking that trade needs
the sweep's own price in the comparison rather than a bigger guess, so
`ZPSE_LZ_SCREEN_BYTES` (a compile-time knob) is where the experiment lives
and the default stays at the kilobyte both corpora agree on.

Where a block's encode time actually goes is counted rather than guessed --
`-DZPSE_BLOCK_STATS` carries the cycle counters -- and over the twelve files
the match finder is 1754M of the 2330M cycles the measured stages account for
(75%), the tile trials 316M (14%) and the decision probe 260M (11%). Nothing
inside the finder is a hotspot that can be taken out: the same counters put
the lazy probe at 11% of a pass, the literal copy at 7% and the backward walk
at 1%, and an in-place writer for the four-in-five command measured as noise.
That is why the fast form above is the only lever on this corpus worth
measuring, and why it stays an option: it walks a block at 397-517 MB/s
(`webster` 397, `mozilla` 517) against the token form's 222-257 (`dickens`
222, `webster` 259), and at the cheap levels it is worth +34% encode and
+40% decode over the corpus -- 0.41 and 1.84 GB/s at 2.00:1 -- for 1.66% more
bytes. The growth is not spread thin: `mozilla` pays 5% and a block of
spikes every 256 bytes codes to 1320 against the token form's 1066, which is
a ratio regression rather than a level ordering, so the deep rungs keep the
smaller stream and the trade stays where `ZPSE_ENC_LZ_FAST` puts it.

`ZPSE_ENC_LZ_FAST` (bit 5 of `enc_flags`, `-F` on the CLI) asks for that form
and nothing else: one match-finder pass per block, no probe, no trials, no
shape decision, with a verbatim copy as the only fallback. It is an option
rather than a level for the same reason it cannot be one: it gives up the tile
forms instead of trading against them, and on typed columns those forms are
where the ratio is. Over Silesia, one pinned core, 64 KiB blocks, best of two
runs of the CLI's own block bench, whose input and output buffers are touched
once before the clock starts and then reused -- the codec's rate, not the
caller's page faults:

| mode | ratio | encode | decode | packed |
| --- | ---: | ---: | ---: | ---: |
| level 9 `ZPSE_ENC_SPEED_MAX` | 2.25:1 | 0.23 GB/s | 1.70 GB/s | 94,248,419 |
| level 6 (default) | 2.08:1 | 0.36 GB/s | 1.72 GB/s | 101,858,449 |
| level 6 + `ZPSE_ENC_LZ_FAST` | 2.19:1 | 0.41 GB/s | 2.30 GB/s | 96,661,814 |

What it used to be was a ratio-for-rate trade; on the mixed corpus it is now
close to a free lunch, because the fast form is *smaller* than the default
level as well as about half again as fast (96,661,814 against
101,858,449). It is also the deepest rung's equal on ratio and three times
its encode (2.19:1 against 2.25:1, 414 against 233 MB/s), which is what the
lazy probe bought: the form looks one position ahead and gives up a short
match for a longer one that starts a byte later, and over the corpus that is
4.8% of the payload for 4% of the finder's rate. That is the
match table below: the fast path is the only shape that never touches the
encoder's 128 KiB of candidate staging, so the table lives there. On the prose
files it is still a real trade, because that is where the tile model has
nothing and a match finder has everything. It is `lz4 -1`'s operating point
on ratio -- 2.06:1 against 2.10:1 -- and behind it on rate, which is the
single-core comparison below; on this table's warm-buffer basis its decode is
the fastest row here, because the whole point of the form is that the command
loop is all that is left. The finder is most of the encode in this mode --
455 MB/s on `dickens`, 624 on `mozilla` and 1070 on `xml`, timed a block at a
time -- so what is left is the block wrapper and the read, not the command
writing. Three changes got it there, every one of them byte-identical:
a literal run of up to fourteen bytes leaves as one 16-byte store instead of
a `memcpy` of a runtime length, the CLI writes each block as it finishes
instead of holding a 202 MiB input and a 424 MiB frame, and the decoder's
literal and match copies widened from eight and sixteen bytes to sixteen and
thirty-two (1.7 to 1.98 GB/s a block). What that loop did get is
a branchless tail: a match's
last one to seven bytes used to be a compare per byte and are now one xor and
a count of trailing zeros, which is worth 36% on `dickens` (256 to 349 MB/s)
and 31% on `webster` with every stream identical. The weekly Silesia job runs
that benchmark with a floor, next to the ratio gate, so the next regression in
the finder is a row that moved rather than a claim nobody re-measured. The
table size is the lever that worked: the fast form runs a 2^16-entry table in
the 128 KiB of candidate staging it never touches, which is 9% faster *and*
2.4% smaller on `dickens` than the 16 KiB table every other caller keeps. A
small table is not a cheap one here: 16 KiB does not stay in L1 beside a 64 KiB
block streaming past, so it was paying L2 latency for the privilege of missing
more. Below that, 2^15 is 6% slower and 0.4% larger; above it, 2^17 buys 0.2%
for a 256 KiB table. Folding the candidate compare and the first length step
into one eight-byte xor is 12-17% *slower* on prose, columnar XML and the
binary members, because misses outnumber hits and a miss is the common case.

The frame layer costs a decode nothing measurable. Walking a frame's block
headers without decoding — parse, header CRC, size checks — runs at thousands
of GB/s, and decoding an LZ2 frame block by block lands within noise of the
bare command loop on the same payloads, so the envelope and the probe are not
where a decode's time goes. What the command loop did spend was validation:
three bounds tests, two escape chains and an output-room test per command, and
on prose four commands in five carry no escape at all. The decoder now runs a
fast region while the payload has eighteen bytes left and the output has
nineteen — no room test, no escape branch, one sixteen-byte literal store, and
the match copy's own wide stores still guarded — for 1162 to 2017 MB/s on
`dickens`, 1649 to 2521 on `mozilla` and 2391 to 2823 on `xml`, against a
1.98 GB/s ceiling per block. Above the loop's rate is the output write: the
51 MiB `mozilla` frame decodes at 1.64 GB/s into a reused buffer and 0.78 into
a fresh allocation, because the second one faults a page every 4 KiB of output.
That is why the CLI decodes block by block into one reused buffer.

The last one is the two forms together rather than a choice between them: the
payload is `[u32 body_len][LZ stream]`, the stream expands to exactly the body
the tile form would have written after its 32-byte header, and the header's
transform, segment count and metadata length stay the tile form's own, so the
decoder unpacks the body once and then runs the ordinary tile path over it. It
is taken only when it is at least 1/32 smaller than the tile stream alone,
because the decode pays a match-finder pass before the tile reconstruction.
On Silesia at the default level it takes 2.1% off the corpus (106,140,388 to
103,921,535 bytes) and 248 to 69 on a constant block, and no case in the size
corpus grows. All four forms carry the same 32-byte header, and all four decode
on their own, which is what makes a block the unit of random access.

A frame encoded with `zpse_compress_frame_indexed()` ends with a
**FrameIndex** -- a 16-byte header (magic `PXI1`, entry count, CRC32-C of the
entries) and one `{offset, size}` pair per block, both relative to the frame
-- and `zpse_seek_frame()` uses it to decode block *k* without touching block
*k-1*:

```c
size_t clen = 0;
uint8_t block[65536];
uint32_t blen = 0;
zpse_compress_frame_indexed(in, n, comp, sizeof comp, &clen, 65536, 0, enc_arena);
zpse_seek_frame(comp, clen, 7, block, sizeof block, &blen, dec_arena);
```

The table is checksummed and every entry is verified against the block it
points at, so a frame with a corrupt or truncated index is refused rather
than followed. Without the index a frame is still decodable sequentially;
what it cannot do is jump, because a block header carries its own size but
not its own offset.

The in-memory writer builds the table by walking the blocks it has just
written, which needs the whole frame in one buffer. A writer that streams
blocks to a file has no such buffer, so `zpse_frame_stream_index()` takes the
`{offset, size}` pairs it recorded as it went and returns the index and the
header that describes it -- the header is written last because the field it
carries is where the table begins. The entries are checked to start right
after the header, to follow each other with no gap, and to end exactly where
the table starts, so a table built from the wrong offsets is refused there
rather than by the reader that seeks by it. `zpse_cli c -x` is that path.

Everything is bounds-checked on the way in, and the checks are the interesting
half of the decoder: every length, count, mode and rank is validated before
use, and the payload CRC32-C is verified whenever a stream carries one.

The payload CRC32-C is **optional and off by default** (block flag
`ZPSE_B_CRC`). Pass `ZPSE_ENC_CHECKSUM` to `zpse_compress_block_ex()` /
`zpse_compress_frame_ex()` to tag a stream; the decoder verifies anything
tagged. Emitting it costs both directions a full pass over the block — a
quarter to a third of a decode on the fast rows, and on the raw path the
difference between `memcpy` and a third of `memcpy` — which is why the default
is to skip it, the same trade LZ4 makes for its frame checksum. When the flag
is not set the header's CRC field is written as 0 and no pass is made, in
every form: nothing reads the field without the flag, and the general encoder
has always skipped it (the fast form filled it regardless, which was a 64 KiB
pass per block for a number no decoder would look at; measured on this host it
was free either way, because its CRC32-C runs on the hardware instruction, but
a build without SSE4.2 pays about a quarter of the encode for it).

## Building and testing

No build system: the library is a set of `.c` files plus headers, and
`src/zpse_unity.c` compiles all of it as one translation unit.

```sh
gcc -std=c11 -O2 -Iinclude src/zpse_unity.c your_program.c -o your_program
```

The parallel frame encoder is the one part behind a switch: add `-DZPSE_MT
-pthread` (POSIX; the Win32 path uses `CreateThread` instead) to compile it in.
`tests/runners/run.sh` builds `test_parallel` both ways and requires the same
frames from each, and `tests/benchmarks/bench_parallel.c` -- the source of the
scaling table above -- refuses to run without the switch rather than print a
flat table that measures one core.

Portable C11, no compiler-specific code outside `#if defined(__GNUC__)`
guards; AVX2 paths are selected at runtime (`zpse_cpu_caps()`), and the scalar
fallbacks are bit-identical, which is what the parity suite exists to prove.

One toolchain does not get the AVX2 kernels: gcc on Windows. A 256-bit vector
passed to a function by value reaches the stack in a slot the compiler emits
`vmovdqa` for, and MinGW gcc cannot align that slot -- it caps
`-mpreferred-stack-boundary` at 4 and rejects 5, `-mstackrealign` and
`force_align_arg_pointer` realign only to the 16 bytes it accepts, and
`__m256i_u` does not change the ABI temporary, which is aligned for the
vector mode rather than the type. The kernels therefore faulted on the first
vector helper they called: `test_transforms`, `test_lz`, `test_ascii`,
`test_levels`, `test_sizes` and the fuzz driver all died on
`vmovdqa %ymm0,0x20(%rsp)` in `zpse_merge2_avx2` / `zpse_merge4_avx2`. So
`include/zpse/hot.h` answers `ZPSE_AVX2_ABI_OK 0` there and every kernel site
compiles to its SSE2 or scalar twin. MSVC keeps them (it aligns a 32-byte
type on the stack to 32) and so does clang. `-DZPSE_AVX2_ABI_OK=0` builds the
kernels out anywhere, which is how that configuration is tested on Linux: the
parity, fuzz and lseed suites print the same digests from it as from the
default build, and the whole suite passes on Windows gcc.

### ASCII tiles, packed four groups at a time

The ASCII tile form keeps seven bits per byte, so a group of eight characters
becomes seven bytes. The packer took them one group at a time: `pext` the
seven fields, copy seven bytes, and test on every group whether eight would
fit. Only the last group can answer no, so the bulk was paying a branch and a
narrow store to answer a question it had already settled. Four groups now go
out per iteration with four independent `pext`s, each stored as one eight-byte
write whose spare byte lands on the next group's first byte and is overwritten
by it; the loop stops while a full group still follows, so that spare byte is
always inside the payload, and the group that cannot prove it keeps the
exact-length form. `text_en`, the ASCII row of the columnar corpus, gains 35%
of encode and the corpus encode mean 1.6% geomean. Sizes are unchanged to the
byte -- this is a store pattern, not a coding decision -- which the size suite
checks across its 768 cases.

### The CLI's seekable directory

`zpse_cli dir FILE` prints the frame's random-access directory: one row per
block, taken from the index, with the decoded length the entry describes.
`zpse_cli x FILE BLOCK OUT` decodes exactly that block through the index,
which on a large frame is the difference between one block and the whole file.
Frames are seekable only when they were written with `-x`; without an index
`dir` says so rather than scanning, because a block header carries its own size
but not its offset.

`zpse_cli bench` reads whole files by default; `-n BYTES` times only the first
N bytes of each one, which is how a single case gets looked at without waiting
for the rest of the corpus.

### Typed columns from the command line

A CSV is prose with commas in it: every number is decimal text, so the codec
spends its matches on the formatting instead of on the values, and a column
that is a monotone counter reads like noise.

```
zpse_cli col IN.csv OUT        type the columns, compress the column-major blob
zpse_cli uncol IN OUT.csv      rebuild the CSV, byte for byte
zpse_cli colcheck IN.csv       both, verified, with the two ratios side by side
```

A column is `INT` only when every field is the canonical text of its value,
`DOUBLE` only when every field is exactly what `%.*f` at the column's own
decimal count prints back from the stored double, and `STR` otherwise -- so an
empty field, a leading zero, an exponent and a quoted name all survive, and
the round trip is exact rather than approximate. `colcheck` asserts that with
a `memcmp` over the whole file, not on a sample.

| file | bytes | text | typed columns | gain |
| --- | ---: | ---: | ---: | ---: |
| `titanic.csv` (891 × 12, real export) | 60 302 | 55 205 (1.09:1) | 39 999 (1.51:1) | 27.5% |
| `uber-rides-data1.csv` (1 511 443 × 3) | 58 604 196 | 16 324 704 (3.59:1) | 14 999 864 (3.91:1) | 8.1% |

The typed form itself is *larger* than the CSV (104 180 bytes on `titanic`:
eight bytes per int and per double, against `33.9732`). That is the point --
fixed-width columns of the same meaning compress like a columnar file instead
of like prose, which is why the packed size falls while the intermediate grows.
The gain shrinks as the file grows: `titanic` is 60 KiB, where the frame
overhead is a large share of a 40 KiB result, and 891 rows of a 12-column table
is not what the path is for; on the 58 MB export it is still 8.1% (and 1.4 s for
the whole typed-and-verified round trip), which is a narrow win on a format
whose numbers are dominated by the timestamp column staying text.

### Using more than one core

Blocks are independent, so a frame can be encoded by several workers at once.
What forced a frame to be built front to back was never the format, only the
output layout: block *k*'s offset depends on the sizes of blocks 0..k-1. That
layout is computed, not searched, so it does not need the encoder. With the
history reorder off (the default) a block is also a pure function of its own
bytes, so the worker count cannot reach the stream: `-t N` writes what `-t 1`
writes, at every level, byte for byte. Each worker
writes its block into a reserved slot of `nominal_block_size + 32` bytes -- the
raw form is the largest a block can be -- and one serial pass moves the blocks
together and appends the index:

```c
size_t clen = 0;
int rc = zpse_compress_frame_parallel(in, n, comp, sizeof comp, &clen,
                                      65536, 0u, 1, 4u, enc_arenas, 4u);
```

`enc_arenas` is one 448 KiB encode arena per worker, still caller-owned, so the
library still never allocates, and the compaction is a `memmove` per block at
memory speed against an encode that runs at one to two gigabytes a second. The
result is byte-identical to the serial call -- same blocks, same header, same
index -- which is what `tests/suites/test_parallel.c` asserts at every worker
count it tries.

The CLI streams every frame, workers included: `zpse_compress_frame_stream()`
hands finished blocks to a writer callback in order, so a file needs a slab of
input and one slot of output per worker rather than the input and the whole
frame. On the 202 MiB corpus at level 6, `zpse c -t 6` takes 688 ms wall --
1007 ms at one worker -- and 419 ms of that is the encode with the output sent
to `/dev/null`, the rest being the write. On `bench` the same file is 975 MB/s
of encode and 5.1 GB/s of decode at four workers against 281 and 1.58 at one.

The ring is what made that work, and the first version of it was wrong. It
claimed blocks from a shared counter, so which block landed in which slot was
decided by timing; it passed byte-identity at two and four workers, then wrote
three different streams in three consecutive six-worker runs of the same 3 MB
input, at the same length every time. Same length with different bytes means
one block's bytes were emitted under another block's length. The assignment is
now static -- worker `w` owns slot `w` and encodes blocks `w`, `w + n`,
`w + 2n` -- so a slot has one writer, its generations are strictly increasing,
and the writer frees a slot only after emitting it. `t=1,2,3,4,6` now produce
one digest at levels 1, 5, 6 and 9, on `-F`, and on 8 KiB blocks, and three
consecutive six-worker runs of the same input produce the same digest.

Threading is compiled in only under `ZPSE_MT`. Without it the call is the
serial encoder, so a caller ships one build and gets cores only where it asked
for them; a codec that never starts a thread is a different promise from one
that might, and the default keeps the first. A frame with fewer than four
blocks per worker is encoded serially too: spawning a thread costs more than
the tenth of a millisecond it would save, and the bytes are the same either
way.

`zpse_cli c -t N`, `zpse_cli d -t N` and `zpse_cli bench -t N` are that call,
one arena per worker. `dir` and `x` are unchanged, because they are the read
side and the index is built by the same pass whichever encoder wrote the
blocks.

The read side has its own function, `zpse_decompress_frame_parallel()`, and it
needs the index for the same reason `x` does: block *k*'s offset is only in the
table. A frame without one decodes sequentially inside the same call. What it
adds over the serial decoder is nothing per worker but an arena: block *k*'s
bytes belong at `k * nominal_block_size` in the caller's buffer and the seek
writes them straight there, so there is no staging block and no second copy.

That is only true because a block decode stays inside its own output. The LZ
decoder's one-shot command paths write up to sixteen bytes past the match they
copy, on the argument that the next command overwrites them, and
`lz_copy_literals` did the same with its tail -- while testing the room in
front of the *run* rather than in front of the store, which let a block whose
literal run ended with a 9-to-15-byte remainder write up to seven bytes past
its end. In one sequential buffer nobody notices; across workers, block *k*'s
tail is block *k+1*'s head. The literal copy now tests the room in front of
where it stores, and `tests/suites/test_frame.c` holds the bound down by
seeking every block into a heap buffer of exactly the bytes wanted, so a
sanitiser has nowhere to miss it. Before that fix the parallel path staged
each block and copied it into place; removing the copy is worth up to a third
of the parallel decode rate on frames too large for cache.

Both directions stay serial on small frames, because a thread start costs more
than the queue: encode wants four blocks per worker and decode thirty-two (a
block decodes in about five microseconds). The bytes are the same either way,
which is what makes the threshold a scheduling decision rather than a format
one.

### Scaling across workers

This is the one table in this file that is not one core and one thread. Six
worker counts on the mixed columnar corpus, 64 KiB blocks, indexed frames,
best of two, pinned to the host's six logical CPUs (`bench_parallel.c`); the
host reports three cores with two threads each, so the six-worker column is
three cores with SMT. GB/s of uncompressed bytes, at the default level.

| frame | encode 1 | 2 | 3 | 6 | decode 1 | 2 | 3 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 1.05 | 1.31 | 1.62 | 1.79 | 5.43 | 4.34 | 4.32 | 5.38 |
| 4 MiB | 1.04 | 1.76 | 2.36 | 2.97 | 5.19 | 7.63 | 7.16 | 5.77 |
| 16 MiB | 0.65 | 1.25 | 1.80 | 2.29 | 3.63 | 6.58 | 8.59 | 11.23 |
| 64 MiB | 0.77 | 1.48 | 1.98 | 3.32 | 3.99 | 7.16 | 7.84 | 11.79 |

Encode scales better than decode because it has more to do: the encode column
is the sweep and the match finder on four different column shapes, and the
decode column is a block reconstruction that already runs at four to six
gigabytes a second on one core. A 64 MiB frame goes from one worker to six at
four and a third times on encode and three times on decode (its last
row above), and a frame of a megabyte thin to 1.7x on encode and no more on
decode.

The row that flattens soonest is the 1 MiB frame's encode, and the cause is the
encoder's four-blocks-per-worker rule rather than the codec: sixteen blocks is
four per worker at four workers, so a fifth and a sixth have nothing left to
hand out and their thread start costs more than the block it splits. Decode
stays flat at that size (its rule asks for thirty-two blocks per worker),
which is why its row barely moves. The host CPU is a six-core Ryzen 5 5600X,
but the VM these were measured in is sliced to three physical cores with two
threads each (cpu0/1, 2/3, 4/5 siblings), so the six-worker column is three
cores with SMT and its useful ceiling is nearer four times than six -- the
one-worker column is still the number to trust for scaling, and every other
table in this file is a single-core measurement for that reason.

Every frame in that table was checked twice: the workers' frame is compared
byte for byte with the one-worker frame (that is the `identical frames` column
`bench_parallel` prints), and every decode is compared whole with the input.

Every **other** figure in this file is one core and one thread.

### Silesia

The synthetic corpus above is 25 generators and no real data. Silesia is the
other direction: twelve files, 202 MiB, from English prose to executables to
medical images, and every one of them a file somebody actually compressed. It
is also the corpus where this codec's weak spot shows, because half of it is
text:

| file | bytes | ratio | encode | decode |
| --- | ---: | ---: | ---: | ---: |
| dickens | 10,192,446 | 1.19 | 0.70 GB/s | 3.96 GB/s |
| mozilla | 51,220,480 | 2.01 | 0.30 GB/s | 1.20 GB/s |
| mr | 9,970,564 | 2.49 | 0.23 GB/s | 1.26 GB/s |
| nci | 33,553,445 | 5.43 | 0.54 GB/s | 1.84 GB/s |
| ooffice | 6,152,192 | 1.49 | 0.20 GB/s | 1.24 GB/s |
| osdb | 10,085,684 | 1.15 | 0.59 GB/s | 4.14 GB/s |
| reymont | 6,627,202 | 1.99 | 0.32 GB/s | 1.60 GB/s |
| samba | 21,606,400 | 2.61 | 0.37 GB/s | 1.57 GB/s |
| sao | 7,251,944 | 1.01 | 0.93 GB/s | 6.65 GB/s |
| webster | 41,458,703 | 2.10 | 0.29 GB/s | 2.19 GB/s |
| x-ray | 8,474,240 | 1.72 | 0.39 GB/s | 2.81 GB/s |
| xml | 5,345,280 | 4.27 | 0.39 GB/s | 2.55 GB/s |
| **total** | **211,938,580** | **2.08** | **0.35 GB/s** | **1.74 GB/s** |

`zpse_cli bench -r 2`, default level (6), 64 KiB blocks, one pinned core, best
of two runs. The whole table is the v2 form's: `ZPSE_B_LZ_TILE` moved the
corpus from 106,140,388 to the 101,858,449 bytes above, which cost ~6% of
encode and bought ~8% of decode (the tile bodies it compresses are shorter
than the plain LZ streams they replace, so the command loop runs over fewer
bytes).

The two LZ forms then split the work between them. The ASCII gate the encoder
has already computed says which one: a block it convicts is text, which is the
shape the fast form was built for, and every other block keeps the token form.
That pairing is what the last two columns moved on. Paired against the build
before it, the choice took the corpus from 104,474,009 to 102,852,453 bytes
and decode from 1,251.0 to 1,571.5 MB/s for nothing on the encode side (306.0
to 312.8 MB/s); the fast form's finder then got the 64 KiB of candidate
staging for a 2^15-entry table on the gated path and took another 0.5% of size
and 2.4% of encode (379.5 to 384.7 MB/s, same host, paired).

The `sao` row is the raw fallback doing its job at copy speed, and `osdb` is
close behind it. The prose rows are the ones the finder moved: `dickens` from
1.09 to 1.17, `webster` from 1.50 to 2.10 at the default, and `dickens` 1.65
with `webster` 2.14 at MAX, where both forms are priced against every other
shape in the sweep. `dickens` is the one file the default still refuses the
LZ form on -- the screen margin that would take it is level 7's.

### Against lz4 and zstd, on the same bytes

A ratio is only half a claim, so here are the two tools anyone would reach
for, on exactly the same 202 MiB, through `bench_pareto.sh`: one frame per
tool, one process per run, best of two, and every output written to
`/dev/null` so the rate is the codec's instead of the filesystem's. Every row
round-tripped and was compared whole with the input before it was timed, and
the `frontier` column is computed from the three measurements: a row is on it
when no other row is at least as good on all three.

| tool | ratio | encode | decode | frontier |
| --- | ---: | ---: | ---: | :---: |
| zpse 1 FASTEST | 1.25:1 | 0.78 GB/s | 2.74 GB/s | **yes** |
| zpse 3 FASTER | 1.29:1 | 0.55 GB/s | 2.43 GB/s | **yes** |
| zpse 5 BALANCED | 2.00:1 | 0.34 GB/s | 1.70 GB/s | no |
| zpse 6 (default) | 2.08:1 | 0.32 GB/s | 1.53 GB/s | no |
| zpse 9 MAX | 2.25:1 | 0.23 GB/s | 1.46 GB/s | no |
| zpse `LZ_FAST` | 2.19:1 | 0.40 GB/s | 1.82 GB/s | **yes** |
| lz4 -1 | 2.10:1 | 1.35 GB/s | 1.45 GB/s | **yes** |
| lz4 -3 | 2.60:1 | 0.37 GB/s | 1.52 GB/s | **yes** |
| lz4 -9 | 2.72:1 | 0.15 GB/s | 1.52 GB/s | **yes** |
| zstd -1 | 2.89:1 | 0.33 GB/s | 1.00 GB/s | **yes** |
| zstd -3 | 3.20:1 | 0.22 GB/s | 0.87 GB/s | **yes** |
| zstd -19 | 4.01:1 | 0.002 GB/s | 0.87 GB/s | **yes** |

Every row here was measured in one run, pinned to one core, lz4 and zstd
included. Levels 1 and 3, the fast form and every lz4 and zstd row are on the
frontier. Level 1 at 1.25:1 is reached by nothing else at that rate -- the
nearest competitor needs 2.89:1 before it gets to a comparable one -- and it
decodes at 2.74 GB/s, 1.9x `lz4 -1`; level 3 shares that corner. At the default
level the ratio is 2.08:1 against `lz4 -1`'s 2.10 and the decode 1.53 against
1.45 -- lz4 wins encode by four times and still decodes more slowly on a
slightly larger stream. MAX leaves the frontier: `lz4 -3` encodes at 369 MB/s
against its 233 with a better decode and only a bigger stream. `LZ_FAST` at
2.19:1 and 1.82 GB/s is the fastest decode of any row at or above lz4's ratio.

The one column zpse still loses on is encode: 0.32 GB/s at the default level
against `lz4 -1`'s 1.35, and the fast form's 0.40. That is the decision the
encoder makes on every block and lz4 makes on none, and it is what the
paragraphs below are about. `F` also dominates the default level on all three
columns (2.19:1, 0.39 GB/s, 1.71 GB/s against 2.07:1, 0.35 and 1.52): `F` is
one match-finder pass that may use the whole of the candidate staging as a
2^16-entry table, while a level's sweep needs that staging for its own
streams and the finder gets 2^13 -- 2^15 wherever a block has that much
staging free. MAX loses to `zstd -1` on ratio for the opposite reason: three
quarters of its budget goes on pricing shapes and it still lands 4% larger
than `zstd -1`, at twice its decode rate now.
`ZPSE_LEVELS="1 3 5 6 9 F"` reproduces the table, `F` being `ZPSE_ENC_LZ_FAST`.

The measurements below say where the encode gap is, and it is not the format.

The first is encode. One block at a time over the same 202 MiB, the match
finder runs at 554 MB/s -- 270 on `dickens`, 315 on `webster`, 405 on
`mozilla`, 858 on `nci` -- and the default level reaches 0.38 GB/s, so most of
its encode is the finder and the rest is the decision around it. Beyond that,
a Silesia prose block emits one command every 6.3 bytes, on an average match
of 5.4 bytes, so the finder pays per-command work at a rate the lz4 parse does
not, because that parse finds longer matches on the same bytes. Neither of the
finder's two knobs -- how fast it accelerates once a position has found
nothing, and how far ahead it will look before taking a match -- moves its rate
by more than 4% on any Silesia file, which is why they are where they are.

The second used to be decode, and the framing was most of it: the command loop
was running these twelve files at about 2 GB/s a block while the CLI's frame
decode of the same bytes read 0.97. The loop was not the problem. The CLI read
the whole 100 MiB frame into a fresh allocation before touching a block, and
that read is a page fault every 4 KiB -- 213 ms for bytes `cat` reads in 11 --
and then it decoded out of that. It now reads a block at a time, straight into
the buffer it decodes from, and writes each block out as it finishes: the same
frame decodes in 133 ms instead of 220, which is where the 1.52 GB/s in the
table comes from. The encode side of the CLI had the same defect and now has
the same fix: it copied its input into a slab before encoding it, which on the
68 MiB columnar corpus was 20 ms of a 96 ms run; it maps the input instead,
and the same encode takes 75 ms with an identical frame. What is left is the
codec, and an LZ block decodes at the rate of its own command loop.

*Encode.* A block here is decided, not merely encoded: the probe prices the
candidate shapes, the ASCII gate prices three of them on two samples, and a
block the gate convicts pays a screened match-finder pass.That decision is what caps the encoder at a quarter of `lz4 -1`'s rate, and lz4 makes none of
it -- one pass, one token per match, no comparison of shapes. The probe alone
is 14.7% of the encode time over the twelve files, weighted by each file's
share of the work: 2% on `dickens`, where the tiles win immediately, and 74%
on `sao`, every block of which ends raw. So the decision is not one cost but
ten small ones -- a full transform staging and an exact price per candidate --
and removing it entirely would lift the corpus from 0.27 to 0.32 GB/s, not to
`lz4`'s rate. The trials are the rest. `zstd -1` is the
same story at a smaller multiple.

*Decode.* An LZ block decodes at the rate of its own command loop, so the
tile stream's multi-gigabyte rate is not available on `dickens` or `webster`
however the block was chosen. The tile rows above are where the decode claim
lives.

So the strong claim is not on text. It is on typed columns, where the same
harness says the opposite.

### The same comparison on typed columns

`bench_pareto.sh` over the seventeen files `corpus_write` emits -- ten byte
patterns and the seven columnar kinds, 4 MiB each, concatenated into one
stream -- with the same three second-column measurements and the same
frontier rule:

| tool | ratio | encode | decode | frontier |
| --- | ---: | ---: | ---: | :---: |
| zpse 1 FASTEST | 4.36:1 | 1.25 GB/s | 3.94 GB/s | **yes** |
| zpse 5 BALANCED | 5.69:1 | 0.77 GB/s | 3.60 GB/s | no |
| zpse 6 (default) | 5.70:1 | 0.89 GB/s | 3.67 GB/s | **yes** |
| zpse 9 MAX | 5.70:1 | 0.77 GB/s | 3.61 GB/s | no |
| lz4 -1 | 3.29:1 | 0.95 GB/s | 1.16 GB/s | no |
| lz4 -3 | 3.54:1 | 0.42 GB/s | 1.07 GB/s | no |
| lz4 -9 | 3.61:1 | 0.27 GB/s | 1.06 GB/s | no |
| zstd -1 | 4.78:1 | 0.61 GB/s | 1.22 GB/s | no |
| zstd -3 | 4.91:1 | 0.43 GB/s | 1.39 GB/s | no |
| zstd -19 | 5.71:1 | 0.006 GB/s | 1.17 GB/s | **yes** |

This is the one corpus where zpse is on the frontier, and it is there at two
levels: level 6 at 5.70:1, 0.89 GB/s in and 3.67 GB/s out, and level 1 at
4.36:1 and 1.25 GB/s in. Level 6's stream is 45% smaller than `lz4 -1`'s on the
same bytes (5.70:1 against 3.29:1) and decodes 3.2x as fast (3.67 against
1.16), for 6% less encode. `lz4 -1` is off the frontier: level 1 encodes 32%
faster than lz4 does at 4.36:1 against 3.29, and decodes 3.4x as fast. Level 5
is marked off for the narrower reason that level 6 beats it on both ratio and
rate -- the ladder's extra trial buys its keep on this corpus. MAX at 0.77 GB/s
has the same 5.70:1 as the default and a slower decode, so this is the one
corpus where the deepest rung has nothing left to buy.

One check on that claim belongs here, because it is the difference between a
corpus and a format. A *real* table export in CSV form is not typed columns:
on 58 MB of ride-hailing trips (`uber-rides-data1.csv`, 1 511 443 rows, three
columns) zpse reads 3.58:1 at 236 MB/s against `lz4 -1`'s 3.69:1 at 1.25 GB/s
-- a hair behind on ratio and an order of magnitude behind on rate, because a
CSV is prose with separator punctuation. The typed-column shape the table above
measures is what a binary table format stores, not what a text export of the
same rows contains.

`zpse_cli col` is what closes the ratio part of that: the same export, typed
into columns first, packs to 3.91:1 -- past `lz4 -1`, 8.1% under the text form,
and rebuilt byte for byte by `uncol`. It does not close it to `zstd -3`, which
reads 5.47:1 on the same bytes.

Running the ladder over Silesia is how a CLI bug surfaced. `-l N` used to OR
the level into a word that already held the default, so `-l 1` ran BALANCED,
`-l 2` and `-l 3` both ran level 7, `-l 6` ran 7 as well, and `-l 8`/`-l 9`
left the field above its range (the header printed `level ?`). Only `-l 5`
and `-l 7` did what they said. `parse_level` now replaces the field instead of
OR-ing into it, and the ladder is what the level table documents:

| level | dickens + webster | ratio | encode |
| --- | ---: | ---: | ---: |
| 1 FASTEST | 47,250,579 | 1.09 | 3.41 GB/s |
| 2 | 47,250,549 | 1.09 | 3.55 |
| 3 FASTER | 47,250,549 | 1.09 | 3.21 |
| 4 | 47,250,549 | 1.09 | 3.02 |
| 5 BALANCED | 29,724,409 | 1.74 | 0.37 |
| 6 (default) | 28,278,012 | 1.83 | 0.32 |
| 7 | 26,179,799 | 1.97 | 0.30 |
| 8 | 25,572,783 | 2.02 | 0.29 |
| 9 MAX | 25,572,783 | 2.02 | 0.14 |

Sizes are non-increasing from level 2 up, which is the property the table
promises, and the step from level 4 to 5 is where the match finder is offered
to every block.

The default is level 6 rather than 5 because level 6 buys real size for a
small rate: 106,140,388 packed bytes against 113,246,135 over all twelve
files (2.00:1 against 1.87:1), encode 296 against 315 MB/s on the prose pair,
and decode 4% lower. MAX goes further -- 94,248,419 bytes, 2.25:1 -- for
about a third less encode rate, which is the rung to choose when size is the
only thing that matters.

The four cheap levels (1-4) write the same stream within 30 bytes of each
other because none of them runs the match finder, and on English prose the
tile stream is the same shape at one trial as at four -- their differences
show up on structured columns, which is what the 25-case corpus measures.

### What the compiler adds, and what the tags are worth

Every number in this README is an `-O2` number: the CI matrix,
`tests/runners/run.sh` and the released CLI all build at `-O2`. `-O3` was worth
a few percent of encode on the amalgamation when the release was set to use
it, and re-measured on the current tree it is slower -- that is the third row
below. The two ways to ask for a per-function level are not interchangeable.
Both were measured on the level corpus (`bench_levels.c`, 25
cases x 4 MiB in 64 KiB blocks; A/B/A/B runs interleaved with the order swapped
every round, median of the per-case throughput ratios, one pinned core):

| build | encode | notes |
| --- | ---: | --- |
| `-O2` | 1.000 | everything in this tree |
| `ZPSE_HOT` on the 18 hot functions (`include/zpse/hot.h`, gcc only) | 1.025 | 16 KB more text; decode and every compressed size unchanged |
| the amalgamation at `-O3` | 0.99 | re-measured at the counts above on gcc 15.2.0: median 0.9895 of 75 per-case ratios; text 244 KB -> 344 KB |
| that pragma scoped around the six hottest modules instead | 1.00 - 1.02 | 1.019, then 1.001 on a repeat |
| `optimize("O3")` on one function inside a hot chain | 0.93 | `zpse_transform_inverse` alone |

The `-O2`, `ZPSE_HOT` and `-O3` rows were re-run on the current tree (gcc
15.2.0, three interleaved rounds at the counts above); the pragma rows are
measurements of the tree that predates the LZ2 and page work.

Rows three to five are the same request made at three scopes, and why they
differ is worth knowing before adding a tag. At `-O3` the text of the
amalgamation grows from 244 KB to 344 KB: most of what `-O3` buys here is not
inside the loops -- those are hand-written SIMD already tuned at `-O2` -- it
is the inliner and the interprocedural passes deciding more, and those run at
the level the compilation was given, which a per-function attribute cannot
raise. The scope-every-module row confirms it: same object size as `-O2`,
same speed. Worse than merely ineffective, a tag makes gcc refuse to inline
that function with its `-O2` neighbours, which is why the last row is slower
than leaving the function untagged.

The `-O3` row was the one the release used, and it did not survive the tree it
was measured on. Re-run at the counts above -- 25 cases x 4 MiB in 64 KiB
blocks, 20 iterations and 3 repetitions per case, A/B interleaved with the
order swapped every round, one pinned core, gcc 15.2.0 -- the level corpus
reads a median of 0.9895 over 75 per-case ratios: `-O2` 887 MB/s against
`-O3` 843. The three rounds read 1.000, 0.989 and 0.972, so the 7% is gone
rather than buried in the noise. The 17-file corpus `zpse_cli bench` reads
agrees over five interleaved rounds: 1,209 MB/s at `-O2` against 1,191 at
`-O3`, with packed bytes identical (3,128,157 either way). Those earlier
figures measured a different tree and a different inliner, not a property of
this one.

So the release ships the `-O2` binary, like everything else in the tree: on
the current measurement `-O3` is about 1% of encode for 41% more text, with
every compressed size identical either way.

`ZPSE_HOT` is gcc-only and it is a set, not a sprinkle: the 18 functions it
marks were measured together, and a tag put on a function whose callees are
untagged reproduces that last row. The set is measured best of three
interleaved rounds with the order swapped every round, against the same tree
built with the macro defined away; the per-round medians were 1.008, 1.039 and
1.025, which is the spread the row above is quoted from. That measurement is also why the functions called from the
middle of a chain stay untagged: they come back with fewer instructions
retired and more cycles spent. Nor does a tag combine with a target
attribute -- gcc re-applies the command line's target to a function it has
been asked to optimize at another level -- so the AVX2 bodies stay untagged
however hot they are.

On clang and MSVC the macro is empty --
clang does not implement `optimize` and diagnoses it as an unknown attribute,
which `-Werror` would turn into a failed build, and MSVC has no per-function
optimization level at all -- so those builds compile the same sources with no
tag, and a build that wants gcc's `-O3` passes has to ask for them on the
command line rather than with this header.

```sh
bash tests/runners/run.sh          # POSIX / Git Bash / WSL
powershell -File tests/runners/run.ps1   # Windows
```

The runner builds and runs the suites (`roundtrip`, `transforms`, `combinadic`,
`hostile`, `units`, `stride3`, `dense`, `decide`, `fillpred`, `lz`, `ascii`,
`levels`, `frame`, `sizes`, `colref`, `colcorpus`), then builds `parity`, `fuzz`
and `lseed` three times — default,
`-DZPSE_FORCE_SCALAR` and `-DZPSE_FORCE_TRIAL_SWAP` — and requires identical
digests and per-case output across all three. The first two prove the SIMD
paths are bit-exact; the third permutes the encoder's candidate order and
proves that the order cannot change a single compressed byte. `test_hostile`
drives truncations, header and payload corruption, reserved bits, bad
transforms, output overflow and stream recomputation, and must see the
documented status for each. `test_colref` proves the reference column codecs
round-trip, and `test_colcorpus` proves the two benchmark harnesses generate
the same bytes for every columnar corpus.

Benchmarks live in `tests/benchmarks/`: `bench.c` for the in-process table
above, `bench_columnar.c` for columnar data, `bench_colref.c` for what a
purpose-built column codec gets on the same bytes, and `bench_vs.sh` for the
cross-tool comparison (zstd, lz4, brotli, gzip, xz, when installed).
`bench_pareto.sh` is that comparison with the I/O taken out and the frontier
computed: `bash tests/benchmarks/bench_pareto.sh` on Silesia, or a directory
of files such as the output of `corpus_write`. The
columnar corpora are generated by `tests/corpus/columnar_gen.h`, shared by
the in-process harness and the file-based corpus writer so both tables are
about the same bytes.

`bench_columnar stats` answers a different question: which predictors the
encoder actually trials, and how often per searched tile, over the same
corpora. Nothing about the encoder's candidate set is observable from
outside the library, so it needs a build that defines `ZPSE_TRIAL_STATS`
(compile-time only; the counters do not exist otherwise):

```
cc -std=c11 -O2 -DZPSE_TRIAL_STATS -Iinclude \
   tests/benchmarks/bench_columnar.c src/zpse_unity.c -ldl -o /tmp/bc && \
   taskset -c 0 /tmp/bc stats all
```

Use it before optimising a predictor: a predictor that is never prescreened
costs nothing however slow its classify is.

## Known limitations

* **No entropy coder.** The match finder is a single-probe hash one -- LZ4's
  shape, not zstd's search -- and it is screened on a sample before it runs,
  so ratio still loses to zstd on redundant data. Levels 1-4 do not run it at
  all, and levels 5-9 differ only in the sample win that is worth a full pass,
  how many samples that screen may spend and the decode bill it accepts. The
  columnar section shows the sharpest version: on high-cardinality
  text the cheap levels fall back to raw and zstd gets 1.9x, while levels 5-9
  get 2.5-3.4x.
* **The default level trades some decode speed for that ratio.** A block the
  LZ form wins decodes at 1.6-4.0 GB/s instead of the tile stream's 5-13, so
  the default level's mean decode over the corpus is 13.6 GB/s rather than the
  17.4 of level 4, which refuses the form (level 9's is 13.5, because it also
  takes the form on prose). If a caller is decoding-bound, the
  acceleration factor in `src/block.c` is the one constant to change.
* **No column semantics.** A nullable column costs 1.5x what null suppression
  would give it, and nothing in the format knows what a record boundary is.
  The byte transpose recovers the split for fixed-width records by accident;
  it cannot recover the *meaning* of a null.
* **64 KiB blocks, no cross-block state.** Each block is independent, which is
  what makes the transform and predictor code branch-light and allows random
  access, but it caps ratio on long-range redundancy.
* **Random access costs an index.** A frame is sequential unless it was
  encoded with one: `zpse_compress_frame_indexed` appends a 16 + 8-per-block
  table and `zpse_seek_frame` then decodes block *k* alone, in O(1). Without
  it the only way to block *k* is to walk the sizes in front of it, since a
  block header does not carry its own offset.
* **The decoder is hardened, not proven.** Bounds checks and the fuzz suite
  cover the malformed-input cases that are tested; there is no formal
  verification.
* **The wire format is defined by this tree.** There is no separate
  specification document: `include/zpse/constants.h`, `types.h` and the
  encoder/decoder pair are the definition, and the parity suite pins it.
* **Speed numbers above are one machine.** Do not quote them as a property of
  the codec.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
