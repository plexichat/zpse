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

## Performance

All ratios are **uncompressed:compressed** (`x:1`); all rates are decimal MB/s
of uncompressed bytes. Unless a table says otherwise: **one pinned core**,
64 KiB blocks, default level 6, gcc 15.2, on a Ryzen 5 5600X under WSL2 (three
physical cores, six threads). Every number in this section is single-core; the
only multi-core measurement is the parallel-frames table, which names its
worker counts. Every corpus is generated or fetched by a script under
`tests/benchmarks/`, so the rows can be re-measured from a checkout. Rates move
by roughly 10% between runs on this host; ratios are exact byte counts.

### At a glance

| workload | codec | ratio | encode MB/s | decode MB/s |
| --- | --- | ---: | ---: | ---: |
| Silesia, 202 MiB | zpse 6 (default) | 2.08:1 | 366 | 1984 |
| | zstd -3 | 3.20:1 | 225 | 883 |
| | lz4hc -12 | 2.74:1 | 57 | 1570 |
| Typed columns, 68 MiB | zpse 6 (default) | 5.70:1 | 817 | 3465 |
| | zstd -3 | 4.91:1 | 373 | 1295 |
| | lz4hc -12 | 3.62:1 | 32 | 1164 |
| 4 KiB pages, non-zero, 16 MiB | zpse 6 (default) | 2.40:1 | 146 | 1860 |
| | zstd -6, per page | 2.98:1 | 96 | 855 |
| | lz4hc -12, per page | 2.31:1 | 25 | 4222 |
| 4 KiB pages, 12.5% zero, 16 MiB | zpse 6 (default) | 2.67:1 | 155 | 2128 |
| | zstd -6, per page | 3.25:1 | 105 | 909 |
| | lz4hc -12, per page | 2.54:1 | 24 | 4757 |

The short version:

* **Typed columns are the strong case.** Predictors and byte transposes beat a
  match finder when values are monotone, fixed-width or dictionary-coded. zpse
  has both the smallest stream and the fastest decode of the three here.
* **Text and memory pages are the weak case.** zstd's entropy stage and search
  buy 18–35% smaller streams on Silesia and pages. zpse only beats lz4hc on
  ratio, and lz4hc decodes pages more than twice as fast.
* **Decode is fast against zstd** almost everywhere, because the tile path is a
  vectorised reconstruction rather than a serial command loop. It is not fast
  against lz4hc on literal-heavy page data.
* **Encode is the price.** The encoder chooses a shape per block; lz4 makes no
  such decision and is an order of magnitude faster on text.

### Silesia

Twelve real files, 211,938,580 bytes, fetched from
`mattmahoney.net/dc/silesia.zip` by `tests/benchmarks/silesia.sh`. Per-file
rows are the CLI's own block bench at the default level, best of two, one
pinned core, output written normally.

| file | bytes | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| `dickens` | 10,192,446 | 1.19:1 | 717 | 4538 |
| `mozilla` | 51,220,480 | 2.01:1 | 297 | 1477 |
| `mr` | 9,970,564 | 2.50:1 | 256 | 1272 |
| `nci` | 33,553,445 | 5.43:1 | 544 | 1905 |
| `ooffice` | 6,152,192 | 1.49:1 | 223 | 1289 |
| `osdb` | 10,085,684 | 1.15:1 | 611 | 4963 |
| `reymont` | 6,627,202 | 1.99:1 | 301 | 1982 |
| `samba` | 21,606,400 | 2.61:1 | 394 | 2050 |
| `sao` | 7,251,944 | 1.01:1 | 1024 | 10973 |
| `webster` | 41,458,703 | 2.10:1 | 313 | 2271 |
| `x-ray` | 8,474,240 | 1.72:1 | 407 | 3057 |
| `xml` | 5,345,280 | 4.27:1 | 528 | 2550 |
| **total** | **211,938,580** | **2.08:1** | **366** | **1984** |

Half of Silesia is English prose, the shape the tile model is weakest on.
`sao` and `osdb` are the raw path doing a copy; `nci`, `samba` and `xml` are
the structured files the predictors handle well.

The cross-tool comparison uses `tests/benchmarks/bench_pareto.sh`, which
encodes one frame per tool with every output sent to `/dev/null`, round-trips
and compares every row before timing it, and pins one core. Rates here are the
codec's, not the filesystem's; the read floor is printed by the script.

| tool | ratio | encode MB/s | decode MB/s | frontier |
| --- | ---: | ---: | ---: | :---: |
| zpse 1 `FASTEST` | 1.25:1 | 829 | 2709 | **yes** |
| zpse 5 `BALANCED` | 2.01:1 | 344 | 1674 | no |
| zpse 6 (default) | 2.08:1 | 347 | 1604 | no |
| zpse 9 `MAX` | 2.25:1 | 216 | 1499 | no |
| zpse `LZ_FAST` | 2.19:1 | 384 | 1879 | **yes** |
| lz4 -1 | 2.10:1 | 1318 | 1600 | **yes** |
| lz4 -3 | 2.60:1 | 394 | 1518 | **yes** |
| lz4 -9 (HC) | 2.72:1 | 162 | 1613 | **yes** |
| lz4 -12 (HC) | 2.74:1 | 57 | 1570 | **yes** |
| zstd -1 | 2.89:1 | 338 | 983 | **yes** |
| zstd -3 | 3.20:1 | 225 | 883 | **yes** |
| zstd -19 | 4.01:1 | 2 | 635 | **yes** |

A row is on the frontier when no other row is at least as good on ratio,
encode and decode together. On this corpus the default level is *not* on it:
lz4 -1 writes a slightly smaller stream, encodes four times faster and decodes
at the same rate. zpse level 1 and `LZ_FAST` are on it, and `LZ_FAST` has the
fastest decode of any row at or above lz4's ratio. That is the honest summary
of Silesia: zpse is a decode-and-determinism choice, not a text-ratio choice.

### Typed columns

`tests/corpus/corpus_write.c` emits 17 files -- ten byte patterns and seven
columnar kinds -- four MiB each. `bench_pareto.sh` runs the same three-tool
comparison over the concatenation:

| tool | ratio | encode MB/s | decode MB/s | frontier |
| --- | ---: | ---: | ---: | :---: |
| zpse 1 `FASTEST` | 4.36:1 | 1151 | 4017 | **yes** |
| zpse 5 `BALANCED` | 5.69:1 | 875 | 3747 | **yes** |
| zpse 6 (default) | 5.70:1 | 817 | 3465 | **yes** |
| zpse 9 `MAX` | 5.70:1 | 758 | 3572 | **yes** |
| zpse `LZ_FAST` | 3.49:1 | 889 | 2563 | no |
| lz4 -1 | 3.29:1 | 909 | 1037 | no |
| lz4 -3 | 3.54:1 | 452 | 1090 | no |
| lz4 -9 (HC) | 3.61:1 | 310 | 1105 | no |
| lz4 -12 (HC) | 3.62:1 | 32 | 1164 | no |
| zstd -1 | 4.78:1 | 647 | 1353 | no |
| zstd -3 | 4.91:1 | 373 | 1295 | no |
| zstd -19 | 5.71:1 | 5 | 1153 | **yes** |

The default stream is 42% smaller than lz4 -1's and decodes 3.3x as fast, for
10% less encode. zstd -19 matches the ratio at 1/160th of the encode rate and
a third of the decode rate. This is the corpus where the design pays off: the
win is not "a better match finder", it is predicting a monotone or fixed-width
column instead of trying to match it.

The 25-case effort matrix in `tests/benchmarks/bench_levels.c` shows the same
thing per case. Selected default-level rows:

| case | ratio | encode MB/s | decode MB/s | what it is |
| --- | ---: | ---: | ---: | --- |
| `i32_seq` | 9.63:1 | 1160 | 10200 | monotone i32 |
| `i64_mixed` | 7.93:1 | 1150 | 11750 | mixed-width i64 |
| `geo_point` | 18.59:1 | 2060 | 7000 | lat/lon plus cell id |
| `struct_offsets` | 4.45:1 | 985 | 11010 | increasing byte offsets |
| `text_en` | 1.88:1 | 371 | 1890 | English prose |
| `uuid_mix` | 1.33:1 | 258 | 3610 | high-cardinality hex |

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
framing, with two length bytes charged to a raw lz4 block.

| codec | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| zpse 1 `FASTEST` | 1.74:1 | 254 | 2912 |
| zpse 6 (default) | 2.40:1 | 146 | 1860 |
| zpse 9 `MAX` | 2.41:1 | 128 | 1839 |
| zpse 6 `PAGE_FAST` | 1.52:1 | 988 | 3503 |
| zpse 6 `LZ_FAST` | 2.22:1 | 605 | 1826 |
| zpse 6 `LZ_HEX` | 2.40:1 | 150 | 1870 |
| zstd 1 | 2.87:1 | 315 | 719 |
| zstd 6 | 2.98:1 | 96 | 855 |
| zstd 19 | 3.14:1 | 5 | 705 |
| lz4 1 | 2.03:1 | 1010 | 3429 |
| lz4hc 4 | 2.26:1 | 191 | 4314 |
| lz4hc 9 | 2.30:1 | 76 | 4202 |
| lz4hc 12 | 2.31:1 | 25 | 4222 |

`LZ_FAST` is the cheap way to a page: one finder pass, no decision, 605 MB/s
of encode, and it gives up a fifth of the ratio (2.22:1 against 2.40:1) for
four times the rate. `LZ_HEX` is the same finder with the hexadecimal command
encoding, which is what level 7 and up write for a page anyway; it costs
nothing in ratio against the default and decodes at the same rate, so it is a
row for the codec comparison rather than a suggestion.

**Zero pages are a separate class.** Untouched anonymous pages are a large
share of a real footprint and compress to almost nothing. On the same stream
with one zero page in eight, zpse 6 scores 2.67:1 and decodes at 2128 MB/s;
dropping the zero pages from that same file leaves 2.35:1. The headline table
uses the non-zero file, so the ratio describes content, not untouched memory.

Per-class ratios at the default level:

| class | zpse 6 | zstd 6 | lz4hc 12 |
| --- | ---: | ---: | ---: |
| `code` | 3.00:1 | 3.67:1 | 3.27:1 |
| `rodata` | 4.00:1 | 6.48:1 | 3.90:1 |
| `heap-pointers` | 2.36:1 | 3.36:1 | 2.03:1 |
| `heap-objects` | 2.98:1 | 4.26:1 | 3.13:1 |
| `slab-strings` | 4.93:1 | 9.54:1 | 6.43:1 |
| `btree` | 4.64:1 | 5.78:1 | 4.81:1 |
| `hash-table` | 2.10:1 | 2.55:1 | 1.89:1 |
| `json-arena` | 2.40:1 | 3.27:1 | 2.56:1 |
| `stack` | 2.03:1 | 3.29:1 | 1.95:1 |
| `float-series` | 1.77:1 | 1.86:1 | 1.47:1 |
| `blob` | 1.00:1 | 1.00:1 | 1.01:1 |
| `mixed-heap` | 3.10:1 | 3.53:1 | 2.46:1 |

The page story is not a zpse win on ratio. zstd is smaller on every
compressible class, by 5–48% depending on how much structure the page has;
lz4hc is close on ratio and much faster to decode. `PAGE_FAST` exists for
callers who want the page path near 1 GB/s of encode, but lz4 1 offers 33% more
ratio at the same encode rate and essentially the same decode. Use zpse here
for deterministic bytes and bounded memory, not for the smallest page.

### 64 KiB byte patterns

`tests/benchmarks/bench.c`, library linked in, default level, best of five
reps. These are codec exercises, not workloads; they show the breadth of the
tile model and the decode ceilings.

| pattern | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| `zeros` | 978:1 | 9610 | 65660 |
| `const5A` | 964:1 | 9570 | 66900 |
| `random` | 1.00:1 | 2180 | 67650 |
| `lowentropy` (4-bit alphabet) | 1.97:1 | 730 | 51430 |
| `incr` (0..255 ramp) | 624:1 | 3530 | 13970 |
| `strided` | 580:1 | 1100 | 13760 |
| `alt00FF` | 830:1 | 5150 | 32300 |
| `sparse2` | 61:1 | 5000 | 21690 |
| `sparse` (1 byte per 97) | 24:1 | 3080 | 11270 |
| `sawtooth` | 565:1 | 1080 | 13030 |

On the same machine `memcpy` measures about 65,600 MB/s and `memset` about
127,400 MB/s. The one-value rows decode within a couple of percent of
`memcpy`; the raw row is a copy plus a header check. The rest are
transform-bound, not memory-bound.

## Effort levels

Nine levels, 1 `FASTEST` to 9 `MAX`, selected by `level << 1` in `enc_flags`.
No level changes the format: every level's stream is read by the same decoder,
and a level only decides which shape is written. The default is level 6; a
field with no level bits normalises to it, and an unsupported level is rejected
with `PSE_BAD_ARG`.

Measured over the 25-case columnar matrix (four MiB per case, 64 KiB blocks,
best of three):

| level | mean ratio | mean encode MB/s | mean decode MB/s | what changes |
| --- | ---: | ---: | ---: | --- |
| 1 `FASTEST` | 4.01:1 | 2160 | 17880 | one trial, cheap prescreen, no match finder |
| 2 | 4.09:1 | 1870 | 18380 | full prescreen, two trials |
| 3 `FASTER` | 4.09:1 | 1710 | 18650 | three trials |
| 4 | 4.09:1 | 1750 | 17560 | four trials, still no match finder |
| 5 `BALANCED` | 5.12:1 | 1030 | 12090 | price every plane, guarded field cut, LZ at 1.25x |
| 6 (default) | 5.16:1 | 1040 | 12040 | margin 1.1875x, decode bill 6 bytes/command |
| 7 | 5.16:1 | 1020 | 11620 | margin 1.125x |
| 8 | 5.16:1 | 980 | 11700 | margin 1.0625x, accel 20, tied-tail cut |
| 9 `MAX` | 5.16:1 | 950 | 13360 | margin 1.0625x, one sample, accel 24, fast LZ candidate |

The jump from 4 to 5 is the match finder and the LZ form; the jump from 1 to 2
is the full predictor prescreen. Levels 3-4 and 6-9 buy little on this corpus,
because most cases have nothing the extra trial or margin can separate. The
suite pins the ladder on its own sizes; on this 25-case corpus a few rows tie
or differ by a handful of bytes rather than strictly shrinking.

## Parallel frames

Blocks are independent, so a frame can be encoded by several workers at once.
The output layout is computed (block *k*'s offset is the sum of the sizes before
it), so workers write into reserved slots and one serial pass compacts them.
The assignment is static -- worker *w* owns slot *w* and blocks *w*, *w+n*, ...
-- which is what makes the frame byte-identical at every worker count. Threads
are compiled in only with `-DZPSE_MT -pthread`.

`tests/benchmarks/bench_parallel.c`, mixed columnar corpus, 64 KiB blocks,
indexed frames, best of two, default level. MB/s of uncompressed bytes; the
six workers are three physical cores with SMT.

| frame | enc 1 | enc 3 | enc 6 | dec 1 | dec 3 | dec 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 1200 | 1840 | 2480 | 5440 | 4440 | 5360 |
| 4 MiB | 930 | 2430 | 3350 | 4280 | 7100 | 5210 |
| 16 MiB | 690 | 1750 | 2890 | 3830 | 8790 | 13640 |
| 64 MiB | 840 | 1970 | 3070 | 3860 | 9170 | 11100 |

Encode scales better than decode because it has more work to divide; decode
already reconstructs a block in a few microseconds. Every frame in the table
was checked byte-for-byte against the one-worker frame and every decode was
compared whole with the input. Small frames stay serial: encode wants four
blocks per worker and decode thirty-two, because a thread start costs more than
the microseconds it saves.

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
| `-O2` | 229,678 B (224 KiB) | 48 B | 12,384 B (12 KiB) | 273,088 B |
| `-Os` | 172,601 B (169 KiB) | 40 B | 12,384 B (12 KiB) | 215,896 B |

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
frames, the page dictionary and the parallel encoder. The benchmarks are all
under `tests/benchmarks/`:

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
| `binary_size.sh` | compiled library size |
| `refresh_tables.sh` | regenerates the reproducible tables in one pass |

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

* **No entropy coder and no cross-block window.** The match finder is a
  single-probe hash one (lz4's shape, not zstd's search), and each block is
  independent. Ratio loses to zstd on redundant text and on memory pages; the
  page section shows the gap.
* **Encode is slower than lz4's** because every block is decided, not merely
  encoded. On Silesia the default level is about a quarter of `lz4 -1`'s encode
  rate; on typed columns it is close.
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
