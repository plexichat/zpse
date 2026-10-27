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
(`tests/benchmarks/bench.c`), best of five runs. This is a Ryzen 5 5600X under
WSL2, decimal MB/s of uncompressed bytes (1e6 B/s).

| pattern | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| `zeros` | 0.004 | 35 600 | **74 000** |
| `const5A` | 0.004 | 35 500 | **74 000** |
| `random` | 1.000 | 1 020 | **65 900** |
| `lowentropy` (4-bit alphabet) | 0.508 | 210 | 45 900 |
| `incr` (0..255 ramp) | 0.040 | 2 020 | 16 800 |
| `strided` | 0.040 | 840 | 16 800 |
| `alt00FF` | 0.004 | 10 950 | 22 700 |
| `sparse2` | 0.018 | 1 090 | 12 500 |
| `sparse` (1 byte per 97) | 0.030 | 240 | 21 200 |
| `sawtooth` | 0.041 | 720 | 7 900 |

The encode column is the default level (5), which prices every plane of every
candidate and also offers the intra-block LZ form. On these patterns the price
buys the ratios and the finder finds nothing, so the sweep is most of the
encode: the one-value rows are 35 GB/s against 74 to decode them, and they are
that fast only because the ASCII gate's whole-block scan is skipped when the
block's first two kilobytes are one value -- a constant sample cannot convict
the gate, and the scan is a third of such a block's encode. That scan stops at
the first byte with its high bit set, so a binary block pays almost nothing
for it: it was 1.25 us of the 7.3 us `alt00FF` takes to encode, which is why
that row is 10 950 MB/s here and not the 8 960 the accumulating form gave it;
a block that really is ASCII still reads every byte, 1.19 us against 1.56 for
the scalar form, because that path is 32 bytes an iteration in AVX2. Decode is the same
at every level, because the shapes written are.

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
  over the same buffer, best of five, single thread.
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

That is the honest picture, and read the table as the shape of the comparison
rather than as the default level's numbers: the intra-block LZ form below moves
`var_string` from 1.0005 to about 0.40 at level 5. zpse has no entropy stage
and no cross-block window -- a match distance is 16 bits inside the block, so a
block stays independent -- and its tile model is "one of ten predictors over
256-byte tiles, code the exceptions enumeratively", with a single-probe hash
match finder (LZ4's shape, not a search) offered as one more candidate at
levels 5-9. On the ramp/sparse patterns zstd's finder still turns almost the
entire file into a handful of copies and beats zpse by two orders of
magnitude. Where the data has no redundancy to match but does have a small
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
read the monotone sequence directly and get 0.30. *A block codec that predicts
instead of matching is not simply a worse match finder; on data with no
redundancy to match, it is a different codec, and it can win.*

`var_string` is the other end: high-cardinality JSON-ish text, where every
value is unique. The cheap levels fall back to raw (1.0005) because none of the
ten tile predictors help, and zstd's match finder finds the repeated keys and
gets 0.21. This is the tile model's own limit showing up on realistic text
rather than on a synthetic ramp, and without the LZ form it is the row that
decides against zpse for this data. It is the row the screened match finder
changes: `var_string` at the default level is about 0.40 (2.5:1) now rather
than 1.0005, which is what the effort-level table above shows -- so treat the
table above as the shape of the comparison rather than as the current default
level's numbers.

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
match finder -- which is what the intra-block LZ form gives it at levels 5-9,
inside the block and nowhere else.

### Effort levels

`ZPSE_ENC_*` in the `enc_flags` word chooses how much encode time a block is
worth. There are nine levels, 1 = FASTEST to 9 = MAX, stored in bits 1-4 as
`level << 1`; a field with no level bits is the default, BALANCED at level 5,
and an unsupported level is rejected with `PSE_BAD_ARG` rather than rounded to
a neighbour. No level changes the format: every level's stream is read by the
same decoder, and a level only decides which shape gets written. The levels
are adjacent rungs, not aliases -- each one costs encode time and returns
ratio -- and all nine were measured in one binary over the columnar corpus
(25 cases, 4 MiB each, 64 KiB blocks, best of three runs of the whole matrix
on one host, `tests/benchmarks/bench_levels.c`):

| level | mean ratio | mean encode | mean decode | what changes |
| --- | ---: | ---: | ---: | --- |
| 1 `ZPSE_ENC_SPEED_FASTEST` | 3.99:1 | 1.43 GB/s | 14.1 GB/s | one trial per block, the three uniform predictors screened only, no match finder |
| 2 | 3.99:1 | 1.44 GB/s | 14.1 GB/s | same as level 1 |
| 3 `ZPSE_ENC_SPEED_FASTER` | 4.06:1 | 1.09 GB/s | 14.3 GB/s | the full nine-predictor prescreen, at most two full trials, no match finder |
| 4 | 4.06:1 | 1.11 GB/s | 14.8 GB/s | same as level 3 |
| 5 `ZPSE_ENC_SPEED_BALANCED` (default) | 4.49:1 | 669 MB/s | 12.1 GB/s | every candidate priced on a tile of every plane, survivors encoded in full, and the LZ form offered on a 1.25x sample win |
| 6 | 4.49:1 | 661 MB/s | 12.4 GB/s | same as level 5 |
| 7 | 4.51:1 | 600 MB/s | 11.9 GB/s | the LZ sample margin tightened to 1.125x, acceleration factor 16 |
| 8 | 4.51:1 | 600 MB/s | 12.5 GB/s | same as level 7 |
| 9 `ZPSE_ENC_SPEED_MAX` | 4.52:1 | 586 MB/s | 10.9 GB/s | the LZ sample margin at its loosest, 1.0625x, acceleration factor 24 |

Levels 5 to 9 are the ones to reach for when bytes matter more than the
clock, and the intra-block LZ form is what they buy. Its match finder is a
full pass over the block at roughly a quarter of the tile encoder's rate on
text, so no level pays for it blind: a sample of the block's head is coded
first, and the pass runs only when that sample shows LZ winning by the level's
margin. The margin is what separates the levels: at 1.25x the default takes
the form on JSON, logs and variable strings, at 1.125x it also reaches beyond
those, and at 1.0625x MAX is the only level whose screen is loose enough to
find it on prose. Where the tile model has nothing to predict it is the
difference between raw and compression:

| case | `ZPSE_ENC_SPEED_BALANCED` | `ZPSE_ENC_MAX` |
| --- | --- | --- |
| `json_rows` | 1.09:1 -> 2.05:1 | 1.09:1 -> 2.05:1 |
| `log_lines` | 1.09:1 -> 2.63:1 | 1.09:1 -> 2.63:1 |
| `var_string` | 1.09:1 -> 2.52:1 | 1.09:1 -> 2.52:1 |
| `utf8_multi` | 1.00:1 -> 1.64:1 | 1.00:1 -> 2.00:1 |
| `text_en` | 1.09:1 | 1.09:1 -> 1.33:1 |

The form is kept only when it is smaller than the best tile stream, so no
level pays for it in size, and no level gives up the trial to save time: the
ASCII gate below replaces the tile-side sweep, not the match finder, which is
offered afterwards whatever the gate decided. What the form costs is decode
speed, and that is what the level's **acceleration factor** decides: an LZ stream decodes at the
speed of its command loop, one 4-byte command per output run, so its rate
follows the average run length. Measured over 64 KiB blocks, in bytes per
command: 5.6 -> 1.6 GB/s (prose), 9 -> 2.2 (JSON), 17 -> 4.0 (var strings),
64 -> 8.8 (sparse), 246 -> 24 (zeros), against 5-13 GB/s for a tile stream.
The match copy described below is what keeps the short end of that range from
costing more than one command per match.

The acceleration factor belongs to the level: 12 at levels 1-6, i.e. a command
every 8 output bytes (a floor of about 1 GB/s at the command loop's cost per
command), 16 at levels 7-8 (every 6 bytes) and 24 at level 9 (every 4). That
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
slack is reclaimed rather than leaked -- the same argument `lz_copy_literals`
already made for literals, applied to the other half of the command.

That is what the run lengths above report: a command costs about the same
whatever its match length, so prose at 5.6 bytes per command decodes at
1.6 GB/s, JSON at 2.2, variable strings at 4.0, and the long-run cases at 8.8
and 24 GB/s, where a match is several stores wide and the copy width, not the
command, is the cost.

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
eight-lane recurrence respectively); strides 2, 3 and 4 take this one. The
head before the first sixteen positions stays scalar, as does the rest of the
tile when the unit window would read past the 256-unit array. On the rows that
depend on it, decode over 64 KiB blocks:

| column | ratio | decode |
| --- | ---: | ---: |
| `ts_gapped` (timestamp with gaps) | 2.06:1 | 1.7 GB/s |
| `float_sparse` (f32 readings) | 2.72:1 | 2.6 GB/s |
| `ts_us` (i64 µs timestamps) | 5.11:1 | 2.5 GB/s |
| `sorted_near` | 3.82:1 | 3.2 GB/s |
| `ts_seq` (i64 µs timestamps) | 5.38:1 | 3.6 GB/s |

The parity, fuzz and least-effort suites report identical digests in the
default and `-DZPSE_FORCE_SCALAR` builds, which is the statement that this
kernel writes exactly the bytes the scalar loop writes.

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
noise stops after one trial. The size corpus (768 cases) grows by nothing, and
the corpus mean encode rate goes from about 830 MB/s to about 900 MB/s.

What it buys is the rows that were slowest for the wrong reason:

| case | before | after | |
| --- | ---: | ---: | --- |
| `utf8_multi` | 230 MB/s | 590 MB/s | **2.5x** |
| `bools_nulls` | 380 MB/s | 900 MB/s | **2.4x** |
| `dna_2bit` | 460 MB/s | 920 MB/s | **2.0x** |

At the deepest level, where the match finder is the candidate that can still
see such a block, `utf8_multi` goes 172 MB/s -> 307 MB/s and `dna_2bit`
424 MB/s -> 949 MB/s, with the same ratio on every case.

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

Levels 5 to 9 try that form as one more candidate and keep it only when it is
smaller than the best tile stream. All of them screen it on a sample of the
block first, because the match finder is a full pass at a quarter of the tile
encoder's rate; what separates the levels is how small a sample win is worth
that pass, so the default level takes it on JSON, logs and variable strings
and the deepest level also takes it on prose.
On high-cardinality text it is the difference between falling back to raw and
compressing at all -- the case the tile model cannot see, because nothing in
it repeats four bytes apart, only further back.

A block is written in one of three forms, and the block flags say which: the
tile form above (`ZPSE_B_RAW` and `ZPSE_B_LZ` clear), a verbatim copy
(`ZPSE_B_RAW`), or intra-block LZ (`ZPSE_B_LZ`). All three carry the same
32-byte header, and all three decode on their own, which is what makes a
block the unit of random access.

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

* **No entropy coder.** The match finder is a single-probe hash one -- LZ4's
  shape, not zstd's search -- and it is screened on a sample before it runs,
  so ratio still loses to zstd on redundant data. Levels 1-4 do not run it at
  all and levels 5-9 differ only in how small a sample win is worth a full
  pass. The columnar section shows the sharpest version: on high-cardinality
  text the cheap levels fall back to raw and zstd gets 4.7x, while levels 5-9
  get 2.5-3.4x.
* **The default level trades some decode speed for that ratio.** A block the
  LZ form wins decodes at 1.6-4.0 GB/s instead of the tile stream's 5-13, so
  the default level's mean decode over the corpus is about 12.1 GB/s rather
  than the 14.8 of level 4, which refuses the form (level 9's is 10.9, because
  it also takes the form on prose). If a caller is decoding-bound, the
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
