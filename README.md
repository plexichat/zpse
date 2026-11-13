# zpse

A block-format compressor in C11. Fixed 64 KiB blocks, with a 4 KiB page mode
for zram-style stores, caller-owned arenas, no allocation, no I/O and no global
state. A compressed block is a pure function of its own bytes: the output is
byte-identical on every build, every compiler, every SIMD level and every
worker count.

It is built for the case where a codec has to be *predictable*: decode at
multi-gigabyte rates out of cache, never touch the allocator, and reject
malformed input with a status code instead of a crash. It is not a
general-purpose replacement for zstd, and the tables below say where it wins
and where it loses.

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
library allocates. They are 448 KiB (encode) and 384 KiB (decode), named
`ZPSE_ENC_ARENA_SIZE` and `ZPSE_DEC_ARENA_SIZE`. `rc` is negative on failure;
every status in `include/zpse/status.h` is reachable from a corrupt stream and
is tested that way.

The version is `ZPSE_VERSION_STRING`, standard semantic versioning -- the three
numbers of `MAJOR.MINOR.PATCH`, no stage letter and no build suffix -- and
`zpse_version_string()` returns exactly that string. It is written down in one
place, `include/zpse/zpse.h`, and the suite fails if the accessor and the macro
ever disagree.

## Performance

All ratios are **uncompressed:compressed** (`x:1`); all rates are decimal MB/s
of uncompressed bytes. Unless a table says otherwise: **one core**, 64 KiB
blocks, default level 6, clang 22.1 (LLVM-MinGW) on a Ryzen 5 5600X under
Windows 11. Every number here is single-core, on an otherwise idle host.

Every corpus is generated or fetched by a script under `tests/benchmarks/`,
and the harnesses stage it in memory and read the staged copy back before the
clock starts (`tests/benchmarks/lib_stage.sh`), so a timed row never waits on
the drive. The floor that staging reads at is printed with each table -- 1.7
GB/s for the 202 MiB Silesia stream, 0.73 GB/s for the 68 MiB columnar corpus
on this host -- and a whole-process row (a CLI invocation) still pays one read
of its input against that floor. The in-process harnesses (`bench.c`,
`bench_levels.c`, `bench_parallel.c`, the CLI's own `bench`) read once and then
clock the codec on their own copy, so no read is in their numbers at all.
Rates are best-of-N and move by roughly 10% between runs; ratios are exact
byte counts.

The compiler matters enormously here: gcc on Windows cannot align a 32-byte
stack slot for a by-value `__m256i`, so `hot.h` answers `ZPSE_AVX2_ABI_OK 0`
there and the tile kernels fall back to scalar. A gcc build of this tree is
two to four times slower on the tile path than the clang build every table
below uses; the same source, the same block, the same bytes out. Every
cross-codec row names the binary it came from, because an lz4 *library* call
and the lz4 *CLI* are not the same encoder.

### At a glance

| workload | codec | ratio | encode MB/s | decode MB/s |
| --- | --- | ---: | ---: | ---: |
| Silesia, 202 MiB | zpse 6 (default) | 2.08:1 | 296 | 1369 |
| | zpse 1 `FASTEST` | 1.24:1 | 816 | 2998 |
| | zpse 9 `MAX` | 2.31:1 | 183 | 1087 |
| 25-case columnar matrix, 100 MiB | zpse 6 (default) | 5.25:1 | 719 | 8562 |
| | zpse 1 `FASTEST` | 4.01:1 | 1716 | 14528 |
| | zpse 9 `MAX` | 5.28:1 | 677 | 9128 |
| 64 KiB block, all zeros | zpse 6 (default) | 978:1 | 10030 | 64680 |
| 64 KiB block, random | zpse 6 (default) | 1.00:1 | 1600 | 64700 |

The cross-codec rows are in each corpus's own section, labelled with the
binary they came from; a single table mixing them would mix hosts.

The short version:

* **Typed columns are the strong case.** Predictors and byte transposes beat a
  match finder when values are monotone, fixed-width or dictionary-coded.
* **Text is the weak case, and the match finder is why.** On Silesia the LZ
  pass is between 33% and 92% of a block's encode, and the finder itself runs
  at 278-881 MB/s on those files. The probe, the trials and the entropy pass
  are the minority of the time; the pass is the ceiling.
* **Decode is fast** because the tile path is a vectorised reconstruction
  rather than a serial command loop, and the LZ forms decode through a lookup.
* **Encode is the price.** The encoder chooses a shape per block; lz4 makes no
  such decision and is about four times faster on text.

### Silesia

Twelve real files, 211,938,580 bytes, fetched from
`mattmahoney.net/dc/silesia.zip` by `tests/benchmarks/silesia.sh`. Per-file
rows are the CLI's own block bench at the default level, best of two, one
pinned core, output written normally.

| file | bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| `dickens` | 10,192,446 | 1.19:1 | 510 | 3005 |
| `mozilla` | 51,220,480 | 2.01:1 | 259 | 1007 |
| `mr` | 9,970,564 | 2.50:1 | 173 | 1011 |
| `nci` | 33,553,445 | 5.43:1 | 423 | 1362 |
| `ooffice` | 6,152,192 | 1.49:1 | 178 | 867 |
| `osdb` | 10,085,684 | 1.15:1 | 467 | 3058 |
| `reymont` | 6,627,202 | 1.99:1 | 197 | 1493 |
| `samba` | 21,606,400 | 2.61:1 | 312 | 1461 |
| `sao` | 7,251,944 | 1.01:1 | 776 | 4381 |
| `webster` | 41,458,703 | 2.10:1 | 269 | 1495 |
| `x-ray` | 8,474,240 | 1.72:1 | 295 | 1950 |
| `xml` | 5,345,280 | 4.22:1 | 461 | 1982 |
| **total** | **211,938,580** | **2.08:1** | **296** | **1369** |

Half of Silesia is English prose, the shape the tile model is weakest on.
`sao` and `osdb` are close to the raw path doing a copy; `nci`, `samba` and
`xml` are the structured files the predictors handle well.

### Where a Silesia encode actually goes

`build/blkstat` (a scratch tool, not part of the suite) reads the library's own
`ZPSE_BLOCK_STATS` cycle counters and attributes a block's encode to the
decision probe, the full trials, the match-finder pass and the two pre-passes.
Level 6, 64 KiB blocks, same run as the table above:

| file | finder | probe | trials | gate + order | rest | block MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `webster` | 91.6% | 0.0% | 0.1% | 5.7% | 2.6% | 322 |
| `reymont` | 90.4% | 0.2% | 0.0% | 6.0% | 3.3% | 245 |
| `xml` | 81.2% | 1.5% | 1.5% | 9.7% | 6.1% | 573 |
| `ooffice` | 78.9% | 8.9% | 5.5% | 0.5% | 6.2% | 176 |
| `mozilla` | 68.2% | 9.9% | 8.1% | 0.7% | 13.1% | 319 |
| `samba` | 67.8% | 6.3% | 8.6% | 5.5% | 11.7% | 328 |
| `nci` | 63.4% | 22.1% | 2.5% | 5.2% | 6.9% | 570 |
| `dickens` | 58.4% | 1.0% | 9.6% | 12.7% | 18.3% | 786 |
| `osdb` | 43.6% | 21.5% | 20.3% | 1.2% | 13.4% | 603 |
| `sao` | 28.3% | 36.3% | 12.3% | 1.6% | 21.4% | 594 |
| `mr` | 0.4% | 9.7% | 63.5% | 0.7% | 25.8% | 251 |
| `x-ray` | 0.0% | 16.2% | 61.9% | 0.8% | 21.1% | 439 |

That is the answer to "why is the whole path not 500 MB/s": on the text and
executable files the block is one match-finder pass, and the finder runs at
220-790 MB/s on them (measured standalone over whole 64 KiB blocks, best of
three passes: `dickens` 219, `webster` 271, `mozilla` 309, `osdb` 326,
`xml` 552, `nci` 787 MB/s). A pass at that rate cannot produce a block faster
than that rate, and the rest of the pipeline cannot win back what the pass
spends. The two files that are not finder-bound are bound by the trial loop
instead -- `x-ray` (medical image) and `mr` spend 62% and 64% of a block in two
full shape trials. Raising the ceiling means either a faster finder or fewer
finder passes, and fewer passes costs ratio: level 1 writes no LZ form at all
and runs at 816 MB/s for 1.24:1, and `LZ_FAST` is one pass at 2.19:1.

The finder is one hash probe per position over a 2^15-entry table, with a
one-byte lazy lookahead limited to short matches: the lookahead is a dependent
chain of two loads, so past a dozen bytes it costs the same and buys nothing.
The trial loop is the second ceiling. A full-size block's price window is a
fifth of the block, so a block runs two full shape trials, and the survivors
the sweep abandons after them are the ones the price could not separate by more
than a sixteenth.

The cross-tool comparison uses `tests/benchmarks/bench_pareto.sh`, which
encodes one frame per tool with every output sent to `/dev/null`, round-trips
and compares every row before timing it, and pins one core. Rates here are the
codec's, not the filesystem's; the read floor is printed by the script.

| tool | packed bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 170,031,147 | 1.25:1 | 577 | 1190 |
| zpse 5 `BALANCED` | 105,756,601 | 2.00:1 | 286 | 936 |
| zpse 6 (default) | 102,043,806 | 2.08:1 | 269 | 870 |
| zpse 9 `MAX` | 91,665,498 | 2.31:1 | 184 | 709 |
| lz4 CLI -1 | 100,910,487 | 2.10:1 | 433 | 1258 |
| lz4 CLI -3 | 81,374,580 | 2.60:1 | 89 | 1311 |
| lz4 CLI -9 (HC) | 77,916,351 | 2.72:1 | 31 | 1342 |
| lz4 CLI -12 (HC) | 77,293,783 | 2.74:1 | 9 | 1364 |
| zstd CLI -1 | 73,307,280 | 2.89:1 | 809 | 879 |
| zstd CLI -3 | 66,209,100 | 3.20:1 | 464 | 791 |
| zstd CLI -19 | 52,855,117 | 4.01:1 | 4 | 572 |

Every row was round-tripped and compared whole with the input before it was
timed, with the input staged in memory and the output sent to `/dev/null`. The
read floor that staging measures here is 2179 MB/s, so a whole-process row
still pays a share of its read; it is the same share for every row. All three
tools are the real binaries: `lz4` 1.10.0 and `zstd` from their CLIs.

On this corpus the default level is not the frontier: lz4 writes a slightly
smaller stream at the same ratio and encodes 1.6x faster. zpse 1 is the encode
row and `MAX` is the only row under 2.4:1. That is the honest summary of
Silesia -- zpse is a decode-and-determinism choice, not a text-ratio choice.

### Typed columns

`tests/corpus/corpus_write.c` emits 17 files -- ten byte patterns and seven
columnar kinds -- four MiB each. `bench_pareto.sh` runs the same three-tool
comparison over the concatenation:

| tool | packed bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 16,355,993 | 4.36:1 | 548 | 931 |
| zpse 5 `BALANCED` | 12,529,172 | 5.69:1 | 455 | 939 |
| zpse 6 (default) | 12,514,963 | 5.70:1 | 476 | 877 |
| zpse 9 `MAX` | 12,505,979 | 5.70:1 | 455 | 992 |
| zpse 6 `LZ_FAST` | 20,431,878 | 3.49:1 | 523 | 834 |
| lz4 CLI -1 | 21,744,005 | 3.28:1 | 474 | 750 |
| lz4 CLI -3 | 20,141,108 | 3.54:1 | 117 | 732 |
| lz4 CLI -9 (HC) | 19,786,464 | 3.60:1 | 54 | 771 |
| lz4 CLI -12 (HC) | 19,728,335 | 3.61:1 | 5 | 667 |
| zstd CLI -1 | 14,930,140 | 4.78:1 | 634 | 666 |
| zstd CLI -3 | 14,531,238 | 4.91:1 | 425 | 708 |
| zstd CLI -19 | 12,482,868 | 5.71:1 | 7 | 604 |

The default stream is 42% smaller than lz4 -1's and decodes 1.17x as fast, for
the same encode rate. zstd -19 matches the ratio at 1/68th of the encode rate
and 69% of the decode rate. The read floor on this corpus is 731 MB/s, so the
whole-process rows here pay a larger share of their time to the read than the
Silesia rows above do; the ratio and packed-byte columns are unaffected. This
is the corpus where the design pays off: the win is not "a better match
finder", it is predicting a monotone or fixed-width column instead of trying
to match it.

The 25-case effort matrix in `tests/benchmarks/bench_levels.c` shows the same
thing per case. Selected default-level rows:

| case | ratio | encode MB/s | decode MB/s | what it is |
| --- | ---: | ---: | ---: | --- |
| `i32_seq` | 9.63:1 | 1010 | 5780 | monotone i32 |
| `i64_mixed` | 7.93:1 | 821 | 9690 | mixed-width i64 |
| `geo_point` | 18.59:1 | 1390 | 5200 | lat/lon plus cell id |
| `struct_offsets` | 4.45:1 | 729 | 10490 | increasing byte offsets |
| `text_en` | 1.88:1 | 292 | 1920 | English prose |
| `uuid_mix` | 1.33:1 | 252 | 2660 | high-cardinality hex |

A CSV export of typed data is prose with punctuation, not a typed column; the
`zpse_cli col` path types it first and rebuilds it byte for byte, but the gain
over compressing the text depends on the column mix and is much smaller than
the typed-column rows above. It is not a substitute for a binary columnar
format.

### 4 KiB pages

zram hands the codec one 4 KiB page at a time. The page corpus is generated by
`tests/corpus/corpus_write.c --pages`, not captured from a host process, so it
is byte-reproducible from a checkout. It models twelve page shapes a real
footprint contains: code, rodata, pointer-heavy heap, object heap, slab
strings, a b-tree page, a hash table, a JSON arena, a thread stack, a float
series, an incompressible blob and a mixed heap.

`tests/benchmarks/page_table.sh` runs every page as its own block for every
codec. zstd and lz4/lz4hc are called once per 4 KiB page through their shared
libraries, so no frame spans two pages; packed bytes include each codec's
framing. That harness needs `dlopen`, which this host does not have, so the
cross-codec page comparison cannot be built here; the rows below are zpse's
own, measured with the CLI at `-b 4096` over the 16 MiB non-zero corpus.

| codec | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 1.74:1 | 244 | 2514 |
| zpse 6 (default) | 2.40:1 | 146 | 1290 |
| zpse 9 `MAX` | 2.57:1 | 124 | 1196 |
| zpse 6 `PAGE_FAST` | 1.52:1 | 807 | 2222 |
| zpse 6 `LZ_FAST` | 2.22:1 | 537 | 1912 |

`LZ_FAST` is the cheap way to a page: one finder pass, no decision, 537 MB/s
of encode, and it gives up a fifth of the ratio (2.22:1 against 2.40:1) for
nearly four times the rate. `PAGE_FAST` is the other end of that trade, 807
MB/s for 1.52:1. Level 9's entropy pass is what turns 2.40:1 into 2.57:1, and
it costs 15% of the encode rate.

**Zero pages are a separate class.** Untouched anonymous pages are a large
share of a real footprint and compress to almost nothing. On the same stream
with one zero page in eight, zpse 6 scores 2.67:1 and decodes at 1760 MB/s;
dropping the zero pages from that same file leaves 2.40:1. The headline table
uses the non-zero file, so the ratio describes content, not untouched memory.

The per-class split needs the same `dlopen` harness (`bench_pages`), so it is
not in this table. What the corpus is for still holds: on a 4 KiB
page the encoder has a sixteenth of the input to decide on, so the page path is
where `PAGE_FAST` and `LZ_FAST` exist at all, and the ones placed above are the
rates a page store has to choose between.

### 64 KiB byte patterns

`tests/benchmarks/bench.c`, library linked in, default level, best of five
reps. These are codec exercises, not workloads; they show the breadth of the
tile model and the decode ceilings.

| pattern | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| `zeros` | 978:1 | 10030 | 64680 |
| `const5A` | 964:1 | 10440 | 60260 |
| `random` | 1.00:1 | 1600 | 64700 |
| `lowentropy` (4-bit alphabet) | 1.97:1 | 540 | 53160 |
| `incr` (0..255 ramp) | 624:1 | 3390 | 10460 |
| `strided` | 580:1 | 830 | 11050 |
| `alt00FF` | 830:1 | 5950 | 30610 |
| `sparse2` | 61:1 | 4550 | 15380 |
| `sparse` (1 byte per 97) | 24:1 | 2370 | 8140 |
| `sawtooth` | 565:1 | 860 | 9570 |

On the same machine `memcpy` measures 49,220 MB/s and `memset` 86,620 MB/s.
The one-value rows decode within a couple of percent of `memcpy`; the raw row is
a copy plus a header check. The rest are transform-bound, not memory-bound.

## Effort levels

Nine levels, 1 `FASTEST` to 9 `MAX`, selected by `level << 1` in `enc_flags`.
No level changes the format: every level's stream is read by the same decoder,
and a level only decides which shape is written. The default is level 6; a
field with no level bits normalises to it, and an unsupported level is rejected
with `ZPSE_BAD_ARG`.

Measured over the 25-case columnar matrix (four MiB per case, 64 KiB blocks,
best of three):

| level | mean ratio | mean encode MB/s | mean decode MB/s | what changes |
| --- | ---: | ---: | ---: | --- |
| 1 `FASTEST` | 4.01:1 | 1716 | 14528 | one trial, cheap prescreen, no match finder |
| 2 | 4.09:1 | 1472 | 13424 | full prescreen, two trials |
| 3 `FASTER` | 4.09:1 | 1345 | 13619 | three trials |
| 4 | 4.09:1 | 1304 | 12877 | four trials, still no match finder |
| 5 `BALANCED` | 5.20:1 | 708 | 8427 | price every plane, guarded field cut, LZ at 1.25x |
| 6 (default) | 5.25:1 | 719 | 8562 | margin 1.1875x, decode bill 6 bytes/command |
| 7 | 5.25:1 | 739 | 7533 | margin 1.125x |
| 8 | 5.25:1 | 716 | 8182 | margin 1.0625x, accel 20, tied-tail cut |
| 9 `MAX` | 5.28:1 | 677 | 9128 | margin 1.0625x, one sample, accel 24, fast LZ candidate, Huffman over the LZ payload |

The jump from 4 to 5 is the match finder and the LZ form; the jump from 1 to 2
is the full predictor prescreen. Levels 3-4 and 6-9 buy little on this corpus,
because most cases have nothing the extra trial or margin can separate. The
suite pins the ladder on its own sizes; on this 25-case corpus a few rows tie
or differ by a handful of bytes rather than strictly shrinking, and four cases
(`dict_u16le`, `rle_lens`, `uuid_mix`, `zeros_burst`) are a few bytes LARGER at
5-9 than at 4, which is the same property the suite's own sizes pin in the
other direction. Level 9 is the only rung that offers the entropy pass, and it
costs about 6% of the level-8 encode rate on this corpus.

## Parallel frames

Blocks are independent, so a frame can be encoded by several workers at once.
The output layout is computed (block *k*'s offset is the sum of the sizes before
it), so workers write into reserved slots and one serial pass compacts them.
The assignment is static -- worker *w* owns slot *w* and blocks *w*, *w+n*, ...
-- which is what makes the frame byte-identical at every worker count. Threads
are compiled in only with `-DZPSE_MT -pthread`.

`tests/benchmarks/bench_parallel.c`, mixed columnar corpus, 64 KiB blocks,
indexed frames, best of two, default level, a 4 MiB frame. MB/s of
uncompressed bytes; this host has six physical cores and no SMT to disable.

| frame | workers | encode | speedup | decode | speedup | identical frames |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| 4 MiB | 1 | 0.79 GB/s | 1.00x | 4.36 GB/s | 1.00x | yes |
| 4 MiB | 2 | 1.33 GB/s | 1.68x | 5.60 GB/s | 1.28x | yes |
| 4 MiB | 3 | 1.93 GB/s | 2.44x | 4.65 GB/s | 1.07x | yes |
| 4 MiB | 4 | 1.83 GB/s | 2.31x | 5.23 GB/s | 1.20x | yes |
| 4 MiB | 5 | 3.08 GB/s | 3.90x | 4.03 GB/s | 0.92x | yes |
| 4 MiB | 6 | 1.83 GB/s | 2.32x | 5.26 GB/s | 1.21x | yes |
| 16 MiB | 1 | 0.50 GB/s | 1.00x | 2.38 GB/s | 1.00x | yes |
| 16 MiB | 2 | 0.87 GB/s | 1.75x | 2.79 GB/s | 1.17x | yes |
| 16 MiB | 3 | 1.31 GB/s | 2.65x | 7.36 GB/s | 3.10x | yes |
| 16 MiB | 4 | 1.77 GB/s | 3.58x | 7.37 GB/s | 3.10x | yes |
| 16 MiB | 5 | 2.00 GB/s | 4.04x | 10.47 GB/s | 4.40x | yes |
| 16 MiB | 6 | 1.98 GB/s | 4.00x | 10.96 GB/s | 4.61x | yes |
| 64 MiB | 1 | 0.56 GB/s | 1.00x | 2.72 GB/s | 1.00x | yes |
| 64 MiB | 2 | 1.24 GB/s | 2.21x | 5.33 GB/s | 1.96x | yes |
| 64 MiB | 3 | 1.81 GB/s | 3.23x | 6.86 GB/s | 2.53x | yes |
| 64 MiB | 4 | 2.19 GB/s | 3.91x | 8.97 GB/s | 3.30x | yes |
| 64 MiB | 5 | 2.68 GB/s | 4.78x | 9.95 GB/s | 3.67x | yes |
| 64 MiB | 6 | 2.62 GB/s | 4.67x | 9.07 GB/s | 3.34x | yes |

Encode scales better than decode because it has more work to divide; decode
already reconstructs a block in a few microseconds, and at 4 MiB a whole frame
decodes inside this host's run-to-run spread. Every frame in the table was
checked byte-for-byte against the one-worker frame and every decode was
compared whole with the input.

A worker decodes straight into the caller's buffer at its block's offset, so a
block decode writes exactly its own bytes and nothing else; that is what makes
every row above byte-identical at every worker count.

## What it does

A **frame** is a 32-byte header (magic `PSE1`, version, content size, block
count) followed by blocks, optionally ending in a 16-byte index header plus one
`{offset, size}` pair per block. A **block** is at most 64 KiB of input: a
32-byte header (16 bytes for a page-sized block), an optional transform record,
a segment directory, then one payload per segment.

A **segment** is one plane of the block -- 1, 2, 4, 8 or 16 of them, split by a
coordinate transform -- coded as a sequence of 256-byte **tiles**. Each tile
picks one of twelve predictors (`const zero`, lane previous, lane delta,
order-1, order-2, stride 1/2/3/4/8, tile constant, lane average), records the
bytes it failed to predict, and codes those positions with an enumerative rank
and their residuals through an exclusion map. Seven mask modes and five
residual modes are chosen per tile by a priced search.

The transforms are identity, delta-8, xor-8, byte-transpose of width 2/4/8/16,
delta-plus-transpose at the same widths, and a stride variant for interleaved
data. A plane that is one value throughout has a dense form that codes 4 KiB in
twelve bytes. An all-ASCII block has a seven-bit form that packs eight
characters into seven bytes.

A block may instead be coded as **intra-block LZ**: a literal stream and a
command stream, with every reference a 16-bit distance inside the block. The
four most recent distances are numbered 1 to 4 and cost nothing to name again.
The **fast LZ** form (`ZPSE_ENC_LZ_FAST`) uses a fixed 16-bit distance, a
one-byte repeat code and no distance varint -- lz4's shape -- and is offered at
the deepest levels or requested explicitly. The **hexadecimal LZ** form packs
common literal/length pairs into a nibble and is used for page-sized blocks
from level 7 up. Every form is kept only when it is strictly smaller than
everything already tried.

At the deepest level a block's LZ payload may be coded a second time, by
**Huffman**: four fixed eleven-bit tables, fitted offline on real LZ payloads
(`tests/benchmarks/huf_train.c`) and compiled in, with each block naming the
one it used. The payload is split into four independent bit streams, so the
entropy layer has four chains to run side by side and no dictionary to carry,
and it wraps a payload that already won rather than replacing the match
finder -- which is why it composes with every LZ form. It is offered from
level 9 up, or at any level with `ZPSE_ENC_HUF`, and kept only when it saves at
least a sixty-fourth of the payload it wraps. Over `dickens`, `nci` and
`mozilla` at level 9 it takes 71% of blocks and cuts the payloads it wraps by
6.9%, for 18 KiB of read-only tables. The writer runs at 1.66-1.70 cycles per
byte of payload (2.2 GB/s) on real Silesia payloads and the reader at 2.4
(1.5 GB/s), both scalar, with no per-block state and no allocation.

All block forms carry the same header shape and decode independently. That is
what makes a block the unit of random access, and an optional `FrameIndex` lets
`zpse_seek_frame()` decode block *k* without touching block *k-1*. The index is
checksummed and every entry is verified against the block it points at; a frame
with a corrupt or truncated index is refused rather than followed.

Everything is bounds-checked on the way in. Every length, count, mode and rank
is validated before use. The payload CRC32-C is optional and off by default
(`ZPSE_ENC_CHECKSUM`), because a checksum pass is a large fraction of a fast
decode; the decoder verifies anything tagged. When the flag is not set the
header's CRC field is written as 0 and no pass is made.

## Building and testing

No build system: the library is a set of `.c` files plus headers, and
`src/zpse_unity.c` compiles all of it as one translation unit.

```sh
gcc -std=c11 -O2 -Iinclude src/zpse_unity.c your_program.c -o your_program
```

### Binary size

The unity object is the whole library, so the figures below are what an
executable carries when it links `zpse_unity.c`. Measured on x86-64 with
gcc 15.2 on the Performance host; `tests/benchmarks/binary_size.sh`
regenerates the table.

| build | text | data | bss | object file |
| --- | ---: | ---: | ---: | ---: |
| `-O2` | 264,512 B (258 KiB) | 36 B | 12,352 B (12 KiB) | 296,259 B |
| `-Os` | 189,746 B (185 KiB) | 36 B | 12,352 B (12 KiB) | 219,695 B |

The unity build links as one object, so a program that uses any entry point
gets all of it. A per-file build, or `-ffunction-sections -fdata-sections
-Wl,--gc-sections`, drops the paths a caller never touches. The reference CLI
is larger than the library alone: `tests/tools/zpse_cli.c` links to 280,752 B
stripped with the same compiler.

The parallel frame encoder is the one part behind a switch: add `-DZPSE_MT
-pthread` (POSIX; Win32 uses `CreateThread`). `tests/runners/run.sh` builds
`test_parallel` both ways and requires the same frames from each.

Portable C11, no compiler-specific code outside `#if defined(__GNUC__)` guards.
AVX2 paths are selected at runtime by `zpse_cpu_caps()`, and the scalar
fallbacks are bit-identical; the parity suite proves that across every length
and row count. One toolchain does not get the AVX2 kernels: gcc on Windows
cannot align a 32-byte stack slot for a by-value `__m256i`, so `hot.h` answers
`ZPSE_AVX2_ABI_OK 0` there and every kernel compiles to its SSE2 or scalar
twin. `-DZPSE_AVX2_ABI_OK=0` builds the kernels out anywhere, which is how that
configuration is tested on Linux.

The suite is `tests/runners/run.sh`: parity, fuzz, sizes, levels, LZ, ASCII,
frames, the page dictionary, the Huffman form and the parallel encoder. The
benchmarks are all under `tests/benchmarks/`:

| script | table |
| --- | --- |
| `bench.c` | 64 KiB byte patterns and machine ceilings |
| `bench_levels.c` | nine levels over the 25-case columnar matrix |
| `bench_columnar.c` | per-case transforms and cross-codec comparison |
| `bench_pareto.sh` | ratio/encode/decode frontier against lz4 and zstd |
| `silesia.sh` | Silesia per-file table (downloads the corpus) |
| `page_table.sh` | 4 KiB page corpus, zero pages separated, page dictionary |
| `bench_pagedict.c` | page-dictionary hit rate and encode rate on repeat pages |
| `bench_parallel.c` | worker scaling and byte-identity |
| `huf_train.c` | refits the four Huffman tables into `src/huf_tables.inc` |
| `binary_size.sh` | compiled library size |
| `refresh_tables.sh` | regenerates the reproducible tables in one pass |
| `lib_stage.sh` | memory staging and the read floor, sourced by the scripts above |

None of them writes into the repository; corpora and tools that are not already
on the host are fetched into a temporary directory.

## CLI

`tests/tools/zpse_cli.c` is the reference front end:

```
zpse_cli c [opts] IN OUT       compress a file to a frame
zpse_cli d [opts] IN OUT       decompress a frame
zpse_cli bench [opts] FILE...  compress/decompress each file, report rates
zpse_cli col IN.csv OUT        type the columns, compress the column-major blob
zpse_cli uncol IN OUT.csv      rebuild the CSV byte for byte
zpse_cli colcheck IN.csv       both, verified, ratios side by side
zpse_cli dir FILE              print the frame's random-access directory
zpse_cli x FILE BLOCK OUT      decode one block through the index
```

Useful options: `-l N` selects a level, `-b BYTES` the block size (4096 for
pages), `-t N` workers, `-x` writes an index, `-F` requests `LZ_FAST`, and
`-n BYTES` limits how much of each input `bench` reads.

## Documentation

* [`docs/HISTORY.md`](docs/HISTORY.md) — why the codec looks the way it does,
  which experiments were rejected, and what measurement changed.
* [`docs/page-dictionary.md`](docs/page-dictionary.md) — the caller-side page
  dictionary API for page stores that write the same page repeatedly.

## Known limitations

* **No cross-block window, one optional entropy stage.** The match finder is a
  single-probe hash one (lz4's shape, not zstd's search), and each block is
  independent. The Huffman form wraps a winning LZ payload and is offered at
  level 9 or on request; it is not a general entropy coder for the tile forms.
  Ratio loses to zstd on redundant text and on memory pages.
* **Encode is slower than lz4's** because every block is decided, not merely
  encoded, and because the match finder runs once per block at 220-790 MB/s on
  real data. On Silesia the default level is about 62% of the lz4 CLI's encode
  rate at the same ratio; on typed columns the two run at the same rate and
  zpse's stream is 42% smaller.
* **The default level trades decode speed for ratio.** A block the LZ form wins
  decodes at 1–4 GB/s instead of the tile path's 5–13 GB/s. If decode latency
  is the only metric, level 4 or `PAGE_FAST` is the lever.
* **No column semantics.** The format does not know record boundaries or null
  meanings; the byte transpose recovers the split for fixed-width records by
  accident. A nullable column costs more than null suppression would.
* **Random access costs an index.** A frame is sequential unless encoded with
  one; `zpse_compress_frame_indexed` appends 16 + 8 bytes per block and
  `zpse_seek_frame` then decodes block *k* in O(1). Without it, block *k*'s
  offset is the sum of the sizes before it.
* **The decoder is hardened, not proven.** Bounds checks and the fuzz suite
  cover the malformed-input cases that are tested; there is no formal
  verification.
* **The wire format is defined by this tree.** `include/zpse/constants.h`,
  `types.h` and the encoder/decoder pair are the definition, and the parity
  suite pins it.
* **Speed numbers above are one machine.** Do not quote them as a property of
  the codec.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
