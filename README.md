# zpse

A block-format compressor in C11. 64 KiB blocks by default (up to 256 KiB via
`-b`), with a 4 KiB page mode for zram-style stores, caller-owned arenas, no
allocation, no I/O and no global
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

The buffers above size one default 64 KiB block. A caller using the full
`ZPSE_MAX_BLOCK_SIZE` (256 KiB) sizes them to that instead.

Both arenas are 64-byte aligned and owned by the caller; nothing inside the
library allocates. They are 1344 KiB (encode) and 1024 KiB (decode), named
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
blocks (the default; `-b` accepts up to 256 KiB), default level 6, clang 22.1
(LLVM-MinGW) on a Ryzen 5 5600X under Windows 11. Every number here is
single-core, on an otherwise idle host. Blocks larger than the default trade
ratio against speed by an amount that depends on the host; these tables are
the 64 KiB default, so changing it would require regenerating every one of
them on an idle machine.

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
| Silesia, 202 MiB | zpse 6 (default) | 2.14:1 | 520 | 3578 |
| | zpse 1 `FASTEST` | 1.27:1 | 814 | 3225 |
| | zpse 9 `MAX` | 2.21:1 | 439 | 2439 |
| 25-case columnar matrix, 100 MiB | zpse 6 (default) | 5.24:1 | 719 | 8562 |
| | zpse 1 `FASTEST` | 4.02:1 | 1716 | 14528 |
| | zpse 9 `MAX` | 5.28:1 | 677 | 9128 |
| 64 KiB block, all zeros | zpse 6 (default) | 978:1 | 10030 | 64680 |
| 64 KiB block, random | zpse 6 (default) | 1.00:1 | 1600 | 64700 |

The cross-codec rows are in each corpus's own section, labelled with the
binary they came from; a single table mixing them would mix hosts.

The short version:

* **Typed columns are the strong case.** Predictors and byte transposes beat a
  match finder when values are monotone, fixed-width or dictionary-coded.
* **Text is the weak case, and the block decision is why.** LZ compression is
  now done by vendored LZ4 1.10.0, so the cost is the per-block shape sweep --
  the screens, the tile trials and the candidate pricing -- rather than a
  custom match finder.
* **Decode is fast** because the tile path is a vectorised reconstruction and
  the LZ forms are raw LZ4 blocks decoded by `LZ4_decompress_safe` into the
  block's declared output size.
* **Encode is the price.** The encoder still chooses a shape per block; lz4
  makes no such decision. The LZ work inside that decision is LZ4's, so the
  remaining gap is the decision itself.

### Silesia

Twelve real files, 211,938,580 bytes, fetched from
`mattmahoney.net/dc/silesia.zip` by `tests/benchmarks/silesia.sh`. Per-file
rows are the CLI's own block bench at the default level, best of two, one
pinned core, output written normally. These rows were regenerated with
`SILESIA_DIR=... CC=clang bash tests/benchmarks/silesia.sh 6 2` on this host
(clang 22.1, LLVM-MinGW). PGO is on by default (`ZPSE_PGO=1`): each run trains
a fresh profile on this corpus and uses the PGO binary only after its
compressed streams match a plain `-O2` build byte for byte.

| file | bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| `dickens` | 10,192,446 | 1.35:1 | 552 | 4236 |
| `mozilla` | 51,220,480 | 1.92:1 | 498 | 3026 |
| `mr` | 9,970,564 | 2.60:1 | 656 | 6104 |
| `nci` | 33,553,445 | 5.21:1 | 718 | 4675 |
| `ooffice` | 6,152,192 | 1.47:1 | 396 | 3305 |
| `osdb` | 10,085,684 | 1.90:1 | 490 | 4289 |
| `reymont` | 6,627,202 | 1.85:1 | 404 | 2012 |
| `samba` | 21,606,400 | 2.54:1 | 515 | 3162 |
| `sao` | 7,251,944 | 1.09:1 | 459 | 3368 |
| `webster` | 41,458,703 | 2.01:1 | 412 | 3339 |
| `x-ray` | 8,474,240 | 1.93:1 | 1415 | 8644 |
| `xml` | 5,345,280 | 3.91:1 | 725 | 4465 |
| **total** | **211,938,580** | **2.14:1** | **520** | **3578** |

Half of Silesia is English prose, the shape the tile model is weakest on.
`sao` and `osdb` are close to the raw path doing a copy; `nci`, `samba` and
`xml` are the structured files the predictors handle well.

### Where a Silesia encode goes now

The old per-stage counter table is gone with the custom match finder: there is
no finder pass to attribute time to any more. A block still pays for the
screens, the tile trials and the final candidate pricing, but the LZ candidate
itself is now a single `LZ4_compress_default` call. The per-file totals above
are the current figures; the stage breakdown was produced by a scratch tool
that is not part of the suite and was not re-run for this change.

That is the answer to "why is the whole path not 500 MB/s": on the text and
executable files the block spends its time in the match search, and the rest
of the pipeline cannot win back what those passes cost. Before LZ4 took over
the search, the custom finder ran at 220-790 MB/s on those files (measured
standalone over whole 64 KiB blocks, best of three passes: `dickens` 219,
`webster` 271, `mozilla` 309, `osdb` 326, `xml` 552, `nci` 787 MB/s) -- a
historical figure for the old finder, not the current path. The two files
that were not finder-bound were bound by the trial loop instead -- `x-ray`
(medical image) and `mr` spent 54% and 51% of a block in their one full shape
trial. Raising the ceiling means a faster search or fewer passes, and fewer
passes costs ratio: level 1 writes no LZ form at all and runs at 816 MB/s for
1.27:1, and `LZ_FAST` is one pass at 2.19:1.

That old finder was one hash probe per position over a 2^15-entry table, with
a one-byte lazy lookahead limited to short matches: the lookahead is a
dependent chain of two loads, so past a dozen bytes it cost the same and
bought nothing. LZ4 now owns the match search in a single
`LZ4_compress_default` call. The trial loop is the second ceiling. A
full-size block's price window is a fifth of the block, so a block runs one
full shape trial, and the survivors the sweep abandons after it are the ones
the price could not separate by more than a sixteenth.

The cross-tool comparison uses `tests/benchmarks/bench_pareto.sh`, which
encodes one frame per tool with every output sent to `/dev/null`, round-trips
and compares every row before timing it, and pins one core. Rates here are the
codec's, not the filesystem's; the read floor is printed by the script.

| tool | packed bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 170,031,147 | 1.25:1 | 721 | 1638 |
| zpse 5 `BALANCED` | 105,307,341 | 2.01:1 | 363 | 1197 |
| zpse 6 (default) | 101,781,169 | 2.08:1 | 363 | 1143 |
| zpse 9 `MAX` | 91,646,686 | 2.31:1 | 245 | 990 |
| lz4 1 | 100,924,480 | 2.10:1 | 1333 | 1541 |
| lz4 3 | 81,440,500 | 2.60:1 | 441 | 1518 |
| lz4 9 | 78,001,045 | 2.72:1 | 208 | 1450 |
| lz4 12 | 77,381,328 | 2.74:1 | 51 | 1228 |
| zstd 1 | 73,307,280 | 2.89:1 | 732 | 842 |
| zstd 3 | 66,209,100 | 3.20:1 | 507 | 818 |
| zstd 19 | 52,855,117 | 4.01:1 | 5 | 715 |

Every row was round-tripped and compared whole with the input before it was
timed, with the input staged in memory and the output sent to `/dev/null`. The
read floor that staging measures here is 2744 MB/s, so a whole-process row
still pays a share of its read; it is the same share for every row. All three
tools are the real binaries: `lz4` 1.10.0 and `zstd` from their CLIs.

On this corpus the default level is not the frontier: lz4 encodes about 3.7
times as fast at the same ratio, and zstd writes 28% fewer bytes at a fifth of
the rate. zpse 1 is the encode row and `MAX` is the only row under 2.4:1. That
is the honest summary of Silesia -- zpse is a decode-and-determinism choice,
not a text-ratio choice.

### Typed columns

`tests/corpus/corpus_write.c` emits 17 files -- ten byte patterns and seven
columnar kinds -- four MiB each. `bench_pareto.sh` runs the same three-tool
comparison over the concatenation:

| tool | packed bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 16,190,105 | 4.40:1 | 548 | 931 |
| zpse 5 `BALANCED` | 12,533,517 | 5.69:1 | 455 | 939 |
| zpse 6 (default) | 12,517,932 | 5.70:1 | 476 | 877 |
| zpse 9 `MAX` | 12,508,891 | 5.70:1 | 455 | 992 |
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

`tests/benchmarks/page_table_static.sh` runs every page as its own block for
every codec. It links lz4 1.10.0 in instead of loading it through `dlopen`,
so the cross-codec comparison builds on hosts without shared-library loading;
packed bytes include each codec's framing (two length bytes are charged to a
raw lz4 block). `tests/benchmarks/refresh_tables.sh` still regenerates the
`dlopen` variant. Rates below are best-of-five, one process, 16 MiB non-zero
corpus, and the corpus sha256 is pinned by the generator.

| codec | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 1.66:1 | 321 | 2646 |
| zpse 6 (default) | 2.28:1 | 215 | 1891 |
| zpse 9 `MAX` | 2.42:1 | 160 | 1387 |
| lz4 1 | 2.03:1 | 1047 | 3761 |
| lz4hc 4 | 2.26:1 | 190 | 4372 |
| lz4hc 9 | 2.30:1 | 70 | 4489 |
| lz4hc 12 | 2.31:1 | 22 | 4660 |

Pages now take a single one-pass path. A 16-byte **FLAT** block carries a
constant page as a token, an irregular low-alphabet page as packed NIBBLE4 or
BIT2 indices, and a pointer-lane page as SUBLANE; these decode in one SIMD pass
at roughly 9 GB/s *on the shapes they cover*. Structured pages (ramps, strides,
alternation) fall through to the same 9-level ladder larger blocks use, because
FLAT's fixed-width packing loses badly to the ladder's DELTA and tile forms on
them. Text is no longer part of that fallthrough: a whole-ASCII page may use
the TEXT form, and a stream of ASCII pages may take the FLAT7 whole-frame fast
path, which packs eight characters into seven bytes and decodes in one spread
pass. A redundancy gate keeps repetitive ASCII on the ladder, where the match
finder earns its ratio; high-entropy printable ASCII takes the pack. The old
`PAGE_FAST`, `TURBO`, `PAGE_INSTANT`, `DECODE_FAST` and `LZ_HEX` flags are gone:
every level now picks the same page form for the same bytes, so the ladder is
monotonic on pages as it is elsewhere. `LZ_FAST` (`-F`) and `ZPSE_ENC_HUF` remain
live encoder options. The ratios above reflect that split -- level 9's
entropy pass turns 2.28:1 into 2.42:1.

The honest decode result: because most content pages take the ladder, the
**mixed-corpus decode rate is about 1.9 GB/s, not 5 GB/s**. The 9 GB/s FLAT
kernels are reached on constant, zero and low-alphabet pages (the zram pages
that are cheapest to begin with); on those classes decode runs at 20-75 GB/s.
lz4 1 still has the faster mixed-corpus decode loop (3.8 GB/s) and, at 2.03:1,
slightly less ratio than the default. No single page mode leads on all three
axes at once.

**Zero pages are a separate class.** Untouched anonymous pages are a large
share of a real footprint and compress to almost nothing: a zero page is a
16-byte token and decodes at about 75 GB/s. Dropping the zero pages from the
stream leaves the 2.28:1 in the headline table, so the ratio describes content,
not untouched memory.

`page_table_static.sh` also prints a per-class decode table. It is where the
FLAT kernels show their speed: the zero and constant classes run at 20-75 GB/s
while content pages that take the ladder run at 1.4-2.8 GB/s. That split is the
reason the mixed-corpus rate sits around 1.9 GB/s even though the FLAT path by
itself reaches about 9 GB/s.

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
| 1 `FASTEST` | 4.02:1 | 1716 | 14528 | one trial, cheap prescreen, no match finder |
| 2 | 4.09:1 | 1472 | 13424 | full prescreen, two trials |
| 3 `FASTER` | 4.09:1 | 1345 | 13619 | three trials |
| 4 | 4.09:1 | 1304 | 12877 | four trials, still no match finder |
| 5 `BALANCED` | 5.19:1 | 708 | 8427 | every plane priced, guarded tied-field cut, LZ form on |
| 6 (default) | 5.24:1 | 719 | 8562 | same as 5 |
| 7 | 5.24:1 | 739 | 7533 | same as 5 |
| 8 | 5.25:1 | 716 | 8182 | tied-field cut off, all tied survivors kept |
| 9 `MAX` | 5.28:1 | 677 | 9128 | same as 8 plus Huffman over the LZ payload |

The jump from 4 to 5 is the LZ4 search and the LZ form; the jump from 1 to 2
is the full predictor prescreen. Levels 3-4 and 6-9 buy little on this corpus,
because most cases have nothing the extra trial can separate. The old LZ
margin and acceleration rungs (1.25x down to 1.0625x, accel 12-24) no longer
change a byte: LZ4 owns the search with one `LZ4_compress_default` call, so
only the trial count, the tied-field cut and level 9's Huffman pass still
vary down the ladder. The
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

That call wants the frame and its output resident, because a block's offset
lives only in the FrameIndex. A reader walking a file already holds the blocks
it just read, so `zpse_decompress_batch_parallel` decodes a caller-owned batch
of independent blocks instead: the CLI's `d -t N` reads a batch, spreads it
over the pool and writes the results in order, keeping memory at the batch
rather than the frame. With the payloads already in memory
(`tests/benchmarks/bench_batch_decode.c`) that batch decodes at 4.3x on six
workers of a 202 MiB corpus. The read and the write used to take turns with
the decode and cost about a third of a streamed decode between them, so the
CLI reads the next batch and writes the last one on their own threads: the
same frame streamed takes 174 ms on one worker and 131 ms at six, where the
decode thread did both itself at 217-234 ms at six; discarding the output (to
`NUL`) takes 87-98 ms. Pages gain less from the workers because they start
faster -- 4 KiB pages decode at 3.4 GB/s on one worker and 7.9 GB/s on six
(2.3x), where 64 KiB blocks go from 1.8 to 7.7 GB/s (4.3x).

## What it does

A **frame** is a 32-byte header (magic `PSE1`, version, content size, block
count) followed by blocks, optionally ending in a 16-byte index header plus one
`{offset, size}` pair per block. A **block** is at most 256 KiB of input
(64 KiB by default): a 32-byte header (16 bytes for a page-sized block), an
optional transform record,
a segment directory, then one payload per segment.

A **segment** is one plane of the block -- 1, 2, 4, 8 or 16 of them, split by a
coordinate transform -- coded as a sequence of 256-byte **tiles**. Each tile
picks one of nine predictors: `const zero` (id 0), lane previous (1), lane
delta (2), stride 1/2/4/8 (ids 5-8), lane average (10) and stride 3 (11). Ids
3, 4 and 9 are reserved gaps in the format, not live predictors. A tile records
the bytes it failed to predict, and codes those positions with an enumerative
rank and their residuals through an exclusion map. Seven mask modes and five
residual modes are chosen per tile by a priced search.

The transforms are identity, delta-8, xor-8, byte-transpose of width 2/4/8/16,
delta-plus-transpose at the same widths, and a stride variant for interleaved
data. A plane that is one value throughout has a dense form that codes 4 KiB in
twelve bytes, with the value carried in the record.
An all-ASCII block has a seven-bit form that packs eight
characters into seven bytes.

A block may instead be coded as **intra-block LZ**, backed by vendored
**LZ4 1.10.0** (`src/vendor/lz4.c`, BSD-2, notice in
`src/vendor/LICENSE.lz4`). There are two live LZ wire forms. `ZPSE_B_LZ`
(bit 5) is a raw LZ4 block: `LZ4_compress_default` on the way in and
`LZ4_decompress_safe` on the way out, with the uncompressed size as the only
parameter the decoder needs. `ZPSE_B_LZ_TILE` (bit 6) is an LZ4-compressed
tile body: the decoder reads a 4-byte length, unpacks the LZ4 payload into its
own buffer, then runs the ordinary tile path over the unpacked bytes. It is
mutually exclusive with the raw form, and a v1 reader refuses its bit.
`ZPSE_ENC_LZ_FAST` (`-F`) is a live encoder option but does not name a third
wire form. The old token, varint-distance, repeat-cache, hexadecimal, `lzfast`
and LZ2 codecs are gone. An LZ block is kept only when it is smaller than
everything already tried.

At the deepest level a block's LZ4 payload may be coded a second time, by
**Huffman**: four fixed ten-bit tables compiled in from `src/huf_tables.inc`,
with each block naming the one it used. The `ZPSE_B_HUF` form is bit 7 on its
own; its 12-byte payload header carries a fixed `inner_form` of `ZPSE_B_LZ`.
The payload is split into four independent bit streams and wraps a payload that
already won, so it composes with the LZ4 form. It is offered from level 9 up,
or at any level with `ZPSE_ENC_HUF`, and kept only when it saves at least a
sixty-fourth of the payload it wraps. It is a size-only option: if the entropy
layer does not pay for itself on an LZ4 stream the block stays as raw LZ4.

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

**FLAT7** (`F7P1`) is a whole ASCII frame: a 40-byte header and one 7-bit-packed
region, with a redundancy gate that leaves repetitive ASCII on the ladder.

**REPACK** (`ZRPK`) maps a low-alphabet ASCII buffer to dense w-bit codes
(w = 1..7), with a 16-byte inverse map for w <= 4 and a 128-byte one otherwise;
it declines non-ASCII input, and at w = 7 its payload is byte-for-byte the FLAT7
pack.

**RLE1** codes run-structured small integers as (value, run-1) pairs under a
CRC'd 12-byte header, committing only when 90% of the sampled bytes fall in
maximal runs of at least four values in 0..127.

**f64fast** (`F64F`) pins DELTA_TRANSPOSE_8 for float64 arrays whose lane-high
bytes come from a small dominant alphabet, wrapping a normal PSE1 frame; it is
a library entry point with no CLI flag.

These four are opt-in containers the ladder decoder never chooses: a caller
asks for them explicitly. The CLI's `-F7`/`-7` and `-R` arms time FLAT7 and
REPACK against the block codec on the same input. DPCM1 and MRD are two more
whole-frame paths that need no flag: they auto-fire by default on gated 16-bit
content at every level, and bits 17 and 18 are accepted as no-ops. None of the
six is a new block form; each declines anything outside its gate back to the
ordinary block codec.

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
frames, the page dictionary, the Huffman form, the parallel encoder, hostile
inputs, fill prediction, page forms, FLAT7, and the column reference and
corpus cases. The `parity` and `fuzz` suites are
each built in default, forced-scalar and trial-swap modes and must print the
same digest in all three. The benchmarks are all under
`tests/benchmarks/`:

| script | table |
| --- | --- |
| `bench.c` | 64 KiB byte patterns and machine ceilings |
| `bench_levels.c` | nine levels over the 25-case columnar matrix |
| `bench_columnar.c` | per-case transforms and cross-codec comparison |
| `bench_lz.c` | the single intra-block LZ4 entry point: encode rate and payload size |
| `bench_pareto.sh` | ratio/encode/decode frontier against lz4 and zstd |
| `silesia.sh` | Silesia per-file table (downloads the corpus) |
| `page_table.sh` | 4 KiB page corpus, zero pages separated, page dictionary |
| `page_table_static.sh` | 4 KiB page corpus against statically linked lz4 (no `dlopen`) |
| `bench_pagedict.c` | page-dictionary hit rate and encode rate on repeat pages |
| `bench_parallel.c` | worker scaling and byte-identity |
| `bench_batch_decode.c` | batch decode in memory, one worker to N, 64 KiB blocks or 4 KiB pages (codec only) |
| `decode_round.sh` | the decode tables in one pass: batch scaling for blocks and pages, streaming decode |
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

Useful options: `-l N` selects a level, `-b BYTES` the block size (default
65536, up to 262144; 4096 for pages), `-t N` workers (encode streams through
the worker ring; decode reads a batch of blocks on one thread, decodes them
side by side and writes them in
order on another, so a frame is still never resident whole), `-x` writes an
index, `-F` requests `LZ_FAST`, `-F7` (or `-7`) adds an in-memory FLAT7 arm
to `bench`, `-R` adds an in-memory ZRPK arm to
`bench` (and tries a ZRPK frame first in `c`, sniffing it in `d`), and
`-n BYTES` limits how much of each input `bench` reads.

## Documentation

* [`docs/HISTORY.md`](docs/HISTORY.md) — why the codec looks the way it does,
  which experiments were rejected, and what measurement changed.
* [`docs/page-dictionary.md`](docs/page-dictionary.md) — the caller-side page
  dictionary API for page stores that write the same page repeatedly.

## Known limitations

* **No cross-block window, one optional entropy stage.** LZ blocks are raw LZ4
  blocks, whose 2-byte distances give a 64 KiB window by definition, so each
  block is independent. The Huffman form wraps a winning LZ4 payload and is
  offered at level 9 or on request; it is not a general entropy coder for the
  tile forms. Ratio loses to zstd on redundant text and on memory pages.
* **Encode is slower than lz4's** because every block is decided, not merely
  encoded: the screens, tile trials and candidate pricing dominate. The LZ
  candidate itself is a single LZ4 call. On Silesia the default level encodes
  at about 506 MB/s against lz4 1's ~830 MB/s in the same CLI harness.
* **The default level trades decode speed for ratio.** A block the LZ form wins
  decodes at 1–4 GB/s instead of the tile path's 5–13 GB/s. If decode latency
  is the only metric, level 4 is the lever.
* **The whole-frame paths are mostly caller-driven.** Six paths sit outside
  the ladder. FLAT7 (`F7P1`), REPACK (`ZRPK`), RLE1 (`RLE1`) and f64fast
  (`F64F`) are opt-in self-delimiting containers with a magic and a narrow
  input gate; each declines anything outside its gate back to the ordinary
  block codec. FLAT7 and REPACK have in-memory A/B arms in `bench`
  (`-F7`/`-7` and `-R`); RLE1 and f64fast are library entry points with no CLI
  flag. DPCM1 and MRD are the other two: they are enabled at every level and
  auto-fire on gated 16-bit content with no flag -- bits 17 and 18 are accepted
  but ignored. None is a general-purpose replacement for the ladder. REPACK is
  a ratio codec, not a speed path: its decode is much slower than the ordinary
  block decoder.
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
