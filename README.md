# zpse

A block-format compressor in C11. Fixed 64 KiB blocks, fixed-size
caller-owned arenas, no allocation, no I/O, no threads, no global state, and
an encoder whose output depends only on its input — so a compressed stream is
byte-identical on every build, every compiler and every SIMD level.

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
library ever allocates. They are 448 KiB (encode) and 320 KiB (decode) —
`ZPSE_ENC_ARENA_SIZE` and `ZPSE_DEC_ARENA_SIZE` say so, and the sizes are
in the names of the parameters the API takes. `rc` is negative on failure —
every status in `include/zpse/status.h` is reachable from a corrupt stream
and is tested that way.

## Performance

One core, 64 KiB blocks, the library linked directly into the benchmark
(`tests/benchmarks/bench.c`), best of six runs. This is a Ryzen 5 5600X under
WSL2, decimal MB/s of uncompressed bytes (1e6 B/s).

| pattern | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| `zeros` | 0.004 | 43 900 | **74 600** |
| `const5A` | 0.004 | 44 100 | **74 600** |
| `random` | 1.000 | 4 650 | **66 400** |
| `lowentropy` (4-bit alphabet) | 0.508 | 2 500 | 47 300 |
| `incr` (0..255 ramp) | 0.040 | 8 200 | 17 100 |
| `strided` | 0.040 | 4 360 | 16 400 |
| `alt00FF` | 0.009 | 9 420 | 19 300 |
| `sparse2` | 0.019 | 5 820 | 16 000 |
| `sparse` (1 byte per 97) | 0.060 | 2 350 | 15 100 |
| `sawtooth` | 0.116 | 3 840 | 11 200 |

What the rows are: `zeros`/`const5A` are one-byte-value blocks (coded in
248 bytes), `random` is incompressible (coded as a raw block, so decode is
exactly a `memcpy`), `lowentropy` is a random stream over sixteen byte values,
`incr`/`strided` are ramps, `alt00FF`/`sparse2`/`sawtooth` exercise the
16-wide interleave, and `sparse` is zeros with a byte every 97.

The interesting part is the ceilings: on the same machine and the same
working set, `memcpy` is **65 400 MB/s** and `memset` **132 000 MB/s**. The
one-value rows decode *above* `memcpy` because the decoder writes them with
wide stores and never reads them back; `random` decoding at 66 400 MB/s is
the raw path doing a copy plus a header check, i.e. right at the `memcpy`
ceiling. Everything else is transform- and tile-bound, not memory-bound.

### Methodology, and why this is not science

* In-process only. `tests/benchmarks/bench_vs.sh` measures other tools by
  running their command-line programs; those numbers include process startup
  and file I/O, so none of them appear here. The table above is a library call
  over the same buffer, best of six, single thread.
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

### Ratios against zstd

Same corpus, 4 MiB per pattern, `zstd -3` on the command line. Only the
compressed size is compared: it is the one thing a different harness cannot
change.

| pattern | zpse | zstd -3 | |
| --- | ---: | ---: | --- |
| `zeros` | 0.0038 | 0.00004 | zstd 95x smaller |
| `const5A` | 0.0038 | 0.00004 | zstd 95x smaller |
| `incr` | 0.0400 | 0.00016 | zstd 250x smaller |
| `strided` | 0.0400 | 0.00034 | zstd 118x smaller |
| `sparse` | 0.0605 | 0.00021 | zstd 288x smaller |
| `sparse2` | 0.0192 | 0.00023 | zstd 83x smaller |
| `alt00FF` | 0.0093 | 0.00010 | zstd 93x smaller |
| `sawtooth` | 0.1157 | 0.0626 | zstd 1.8x smaller |
| `lowentropy` | 0.5079 | 0.5196 | **zpse 2% smaller** |
| `random` | 1.0005 | 1.0000 | tie |

That is the honest picture. zpse has no block matching and no entropy stage:
its whole model is "one of ten predictors over 256-byte tiles, code the
exceptions enumeratively". On the ramp/sparse patterns zstd's match finder
turns almost the entire file into a handful of copies and beats it by two
orders of magnitude. Where the data has no redundancy to match but does have a
small alphabet, zpse's tile model is competitive, and on truly random data
both are at 1.000.

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
| `ts_seq` (i64 µs timestamps) | 0.2638 | 0.2564 | zstd 3% smaller |
| `ts_ns` (i64 ns timestamps) | 0.2605 | 0.2564 | zstd 2% smaller |
| `float_sparse` (f32 readings) | 0.3957 | 0.2398 | zstd 1.7x smaller |
| `var_string` (JSON-ish records) | 1.0005 | 0.2123 | zstd 4.7x smaller |
| `sparse_mix` (nullable i64) | 0.1479 | 0.1020 | zstd 1.5x smaller |
| `geo_point` (lat/lon + cell) | 0.0766 | 0.0338 | zstd 2.3x smaller |
| `struct_offsets` (i32 offsets) | 0.2955 | 0.7808 | **zpse 2.6x smaller** |

Three of those rows are worth reading twice.

`struct_offsets` is a strictly increasing array of byte offsets — nothing in
it repeats, so zstd's match finder has nothing to match and lands at 0.78,
worse than storing the raw column. zpse's stride and lane-delta predictors
read the monotone sequence directly and get 0.30. *A block codec with no
match finder is not simply a worse match finder; on data with no redundancy
to match, it is a different codec, and it can win.*

`var_string` is the other end: high-cardinality JSON-ish text, where every
value is unique. zpse falls back to raw (1.0005) because none of its ten
predictors help, and zstd's match finder finds the repeated keys and gets
0.21. This is the "no match finder" limitation showing up on realistic text
rather than on a synthetic ramp, and it is the row that decides against zpse
for this data.

The timestamp rows are the interesting middle: zstd is 2–3% ahead of zpse on
both, which is to say they are within noise of each other on data a
delta-of-delta codec is made for.

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
| `ts_seq` | delta + varint | 4.00:1 | 3.79:1 | zpse 1.1x bigger |
| `ts_ns` | double delta | 4.56:1 | 3.84:1 | zpse 1.2x bigger |
| `float_sparse` | Gorilla XOR | 2.68:1 | 2.53:1 | zpse 1.1x bigger |
| `var_string` | *(none helps)* | — | 1.00:1 | block codec territory |
| `sparse_mix` | null bitmap | 10.44:1 | 6.76:1 | zpse 1.5x bigger |
| `geo_point` | zigzag + delta | 6.10:1 | 13.06:1 | **zpse 2.1x smaller** |
| `struct_offsets` | frame of reference | 2.56:1 | 3.38:1 | **zpse 1.3x smaller** |

The reference codecs are deliberately plain reference points, not proposals —
`tests/suites/test_colref.c` proves each one round-trips, so every size here
is one a real codec reached rather than an estimate.

Two results deserve their own note. On `geo_point` and `struct_offsets` zpse
*beats* the purpose-built codec, and the reason is visible in the stream: the
encoder picks the 16-wide byte transpose, which for a fixed-width record is
exactly the column split a columnar format does by hand. zpse finds it
without a schema, because splitting bytes into planes and predicting each
plane is what its model already does. That is the strongest argument in the
tree for the byte-oriented design, and it is an accident worth naming.

On `sparse_mix` the gap is real and it is the one that would cost something:
null suppression is a typed feature zpse does not have, and a 92%-null
column is worth 10.44:1 to a codec that knows the nulls are whole records.
1.5x is what zpse pays for not knowing.

The other four are within 20% of a purpose-built codec, without any of the
schema that would make such a codec possible to ship. The `var_string` row
is the honest limit: no column codec helps there either, and the answer is a
block codec with a match finder.

### Effort levels

`ZPSE_ENC_*` in the `enc_flags` word chooses how much encode time a block is
worth. No level changes the format: every level's stream is read by the same
decoder, and a level only decides which shape gets written. The levels are
adjacent rungs, not aliases -- each one costs encode time and returns ratio,
and the levels on both sides of the default were measured in one binary
over the columnar corpus (25 cases, 4 MiB each, 64 KiB blocks):

| level | mean ratio | mean encode | mean decode | what changes |
| --- | ---: | ---: | ---: | --- |
| `ZPSE_ENC_MAX` | 4.49:1 | 348 MB/s | 10.7 GB/s | the default sweep, plus intra-block LZ as one more candidate |
| `ZPSE_ENC_SPEED_BALANCED` (default) | 4.01:1 | 720 MB/s | 23.1 GB/s | every candidate priced on a tile of every plane; survivors encoded in full |
| `ZPSE_ENC_FASTER` | 4.01:1 | 916 MB/s | 22.8 GB/s | one tile per plane, then at most two full trials |
| `ZPSE_ENC_FASTEST` | 4.00:1 | 1.19 GB/s | 22.1 GB/s | the cheap price, then one trial: the shape it likes best |

The deepest level is the one to reach for when bytes matter more than the
clock. It never writes a larger block than the default level -- the LZ stream
is kept only when it is smaller -- and it roughly doubles encode time because
the match finder is a full pass over the block. Where the tile model has
nothing to predict, it is the difference between raw and compression:
`text_en` 1.00:1 -> 1.33:1, `json_rows` 1.00:1 -> 2.05:1, `utf8_multi`
1.00:1 -> 1.95:1, `log_lines` 1.00:1 -> 2.63:1, `var_string` 1.00:1 ->
2.52:1, and on data the tiles do compress it still adds a little:
`float_sparse` 2.50:1 -> 3.10:1, `geo_point` 13.06:1 -> 15.54:1,
`zeros_burst` 9.29:1 -> 12.23:1. A block it wins decodes as an LZ stream
instead of a tile reconstruction, which is slower than the tile path on
columnar data (hence the lower mean decode above) and much faster than it on
a long run: 5.8 GB/s on 1 KiB runs, 27 GB/s on a ramp, measured over 64 KiB
blocks. The cheap levels are the other side of the same trade: they write
the same shapes in less encode time.

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
independent and still decodes on its own. A command is one 32-bit word --
offset, literal count, match length minus four -- and there are no escape
words, so the decoder's inner loop is a copy loop with no length parsing. Long
runs are written as repeated commands (4 bytes per 259 bytes of run).

The encoder's deepest level, `ZPSE_ENC_MAX`, tries that form as one more
candidate and keeps it only when it is smaller than the best tile stream. On
the columnar corpus that is worth +12% ratio over the default level, and on
high-cardinality text it is the difference between falling back to raw and
compressing at all -- the case the tile model cannot see, because nothing in
it repeats four bytes apart, only further back.

A block is written in one of three forms, and the block flags say which: the
tile form above (`ZPSE_B_RAW` and `ZPSE_B_LZ` clear), a verbatim copy
(`ZPSE_B_RAW`), or intra-block LZ (`ZPSE_B_LZ`). All three carry the same
32-byte header, and all three decode on their own, which is what makes a
block the unit of random access.

Everything is bounds-checked on the way in, and the checks are the interesting
half of the decoder: every length, count, mode and rank is validated before
use, and the payload CRC32-C is verified whenever a stream carries one.

The payload CRC32-C is **optional and off by default** (block flag
`ZPSE_B_CRC`). Pass `ZPSE_ENC_CHECKSUM` to `zpse_compress_block_ex()` /
`zpse_compress_frame_ex()` to tag a stream; the decoder verifies anything
tagged. Emitting it costs both directions a full pass over the block — a
quarter to a third of a decode on the fast rows, and on the raw path the
difference between `memcpy` and a third of `memcpy` — which is why the default
is to skip it, the same trade LZ4 makes for its frame checksum.

## Building and testing

No build system: the library is a set of `.c` files plus headers, and
`src/zpse_unity.c` compiles all of it as one translation unit.

```sh
gcc -std=c11 -O2 -Iinclude src/zpse_unity.c your_program.c -o your_program
```

Portable C11, no compiler-specific code outside `#if defined(__GNUC__)`
guards; AVX2 paths are selected at runtime (`zpse_cpu_caps()`), and the scalar
fallbacks are bit-identical, which is what the parity suite exists to prove.

```sh
bash tests/runners/run.sh          # POSIX / Git Bash / WSL
powershell -File tests/runners/run.ps1   # Windows
```

The runner builds and runs the suites (`roundtrip`, `combinadic`, `hostile`,
`units`, `stride3`, `dense`, `decide`, `fillpred`, `sizes`, `colref`,
`colcorpus`), then builds `parity`, `fuzz` and `lseed` three times — default,
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
cross-tool comparison (zstd, lz4, brotli, gzip, xz, when installed). The
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

* **No entropy coder.** The default level has no match finder either: ratio
  loses to zstd on redundant data, as shown above, and that is a design
  choice. `ZPSE_ENC_MAX` adds one (a single-probe hash match finder, LZ4's
  shape, not zstd's search), which is what makes the ratio table's text rows
  compress at all; the default level still does not run it.
  The columnar section shows the sharpest version: on high-cardinality text
  the default level falls back to raw and zstd gets 4.7x, while the deepest
  level gets 2.5x.
* **No column semantics.** A nullable column costs 1.5x what null suppression
  would give it, and nothing in the format knows what a record boundary is.
  The byte transpose recovers the split for fixed-width records by accident;
  it cannot recover the *meaning* of a null.
* **64 KiB blocks, no cross-block state.** Each block is independent, which is
  what makes the transform and predictor code branch-light and allows random
  access, but it caps ratio on long-range redundancy.
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
