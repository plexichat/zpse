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
library ever allocates. `rc` is negative on failure — every status in
`include/zpse/status.h` is reachable from a corrupt stream and is tested that
way.

## Performance

One core, 64 KiB blocks, the library linked directly into the benchmark
(`tests/benchmarks/bench.c`), best of six runs. This is a Ryzen 5 5600X under
WSL2, decimal MB/s of uncompressed bytes (1e6 B/s).

| pattern | ratio | encode MB/s | decode MB/s |
| --- | ---: | ---: | ---: |
| `zeros` | 0.004 | 42 500 | **73 000** |
| `const5A` | 0.004 | 42 700 | **70 100** |
| `random` | 1.000 | 4 150 | **64 200** |
| `lowentropy` (4-bit alphabet) | 0.508 | 2 180 | 43 000 |
| `incr` (0..255 ramp) | 0.040 | 7 500 | 17 000 |
| `strided` | 0.040 | 3 650 | 15 200 |
| `alt00FF` | 0.009 | 9 310 | 17 600 |
| `sparse2` | 0.019 | 5 540 | 14 100 |
| `sparse` (1 byte per 97) | 0.060 | 2 150 | 13 100 |
| `sawtooth` | 0.116 | 3 480 | 9 820 |

What the rows are: `zeros`/`const5A` are one-byte-value blocks (coded in
248 bytes), `random` is incompressible (coded as a raw block, so decode is
exactly a `memcpy`), `lowentropy` is a random stream over sixteen byte values,
`incr`/`strided` are ramps, `alt00FF`/`sparse2`/`sawtooth` exercise the
16-wide interleave, and `sparse` is zeros with a byte every 97.

The interesting part is the ceilings: on the same machine and the same
working set, `memcpy` is **57 000 MB/s** and `memset` **115 000 MB/s**. The
one-value rows decode *above* `memcpy` because the decoder writes them with
wide stores and never reads them back; `random` decoding at 64 200 MB/s is
the raw path doing a copy plus a header check, i.e. right at the `memcpy`
ceiling. Everything else is transform- and tile-bound, not memory-bound.

### Methodology, and why this is not science

* In-process only. `tests/benchmarks/bench_vs.sh` measures other tools by
  running their command-line programs; those numbers include process startup
  and file I/O, so none of them appear here. The table above is a library call
  over the same buffer, best of six, single thread.
* The host has two performance states — its `memcpy` ceiling wanders between
  53 000 and 70 000 MB/s depending on thermal state — so run-to-run spread on
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
`units`, `stride3`, `dense`, `decide`, `sizes`), then builds `parity`, `fuzz`
and `lseed` three times — default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` — and requires identical digests and per-case output
across all three. The first two prove the SIMD paths are bit-exact; the third
permutes the encoder's candidate order and proves that the order cannot change
a single compressed byte. `test_hostile` drives truncations, header and payload
corruption, reserved bits, bad transforms, output overflow and stream
recomputation, and must see the documented status for each.

Benchmarks live in `tests/benchmarks/`: `bench.c` for the in-process table
above, `bench_columnar.c` for columnar data, and `bench_vs.sh` for the
cross-tool comparison (zstd, lz4, brotli, gzip, xz, when installed).

## Known limitations

* **No entropy coder and no match finder.** Ratio loses badly to zstd on
  redundant data, as shown above. This is a design choice, not a to-do.
* **Frame index is not implemented.** The `ZPSE_F_FRAME_INDEX` flag is
  refused rather than emitted with a wrong offset.
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
