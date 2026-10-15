# zpse — PSE Block Format v1.0.0

`zpse` is a lossless block codec for binary data. It is single-threaded, does
no allocation after context creation, and has no external dependencies. The
wire format is normative in [SPEC_V1.md](SPEC_V1.md); this repository is the
reference implementation of it.

The codec uses no LZ77 back-references and no Huffman/arithmetic/range/ANS/FSE
coding. Instead each 64 KiB block is transformed, split into 256-byte tiles,
and every tile is coded by *predicting* it, *selecting* the cheapest way to
describe where the prediction is wrong, and *enumerating* those positions
(hence Predict-Select-Enumerate, PSE).

## What it is good at

Per block the encoder tries a small set of coordinate transforms (identity,
delta8, xor8, transpose 2/4/8/16, delta+transpose, stride transpose) and keeps
the smallest result. Within a segment, each tile independently picks from 11
predictors (`CONST_ZERO`, `LANE_PREVIOUS`, `LANE_DELTA`, `LANE_AVERAGE`,
`ORDER1`, `ORDER2`, `STRIDE_1/2/4/8`, `TILE_LOCAL_CONSTANT`), 7 exception-mask
modes and 5 residual modes, so a single stream can serve both run-like and
structured data. Blocks that do not compress are stored verbatim behind the
same envelope (the RAW fallback), and every block carries a CRC32-C of its
decoded bytes.

`tests/bench.c` measures ten patterns, which is a fair summary of the shapes
the format is designed for:

| pattern | shape |
| --- | --- |
| `zeros`, `const5A` | constant planes |
| `incr` | byte ramp |
| `alt00FF` | two-symbol alternation |
| `strided` | period-4/period-64 interleave |
| `sparse`, `sparse2` | mostly zero with isolated nonzero bytes |
| `sawtooth`, `lowentropy` | low-entropy but noisy |
| `random` | incompressible (exercises the RAW fallback) |

## Repository layout

```
include/zpse/   public and internal headers (one per stage: crc32c, arena,
                transform, predictor, combinadic, mask, residual, tile,
                segment, block, frame, vector, api)
src/            the implementation, plus src/zpse_unity.c
tests/          suites, throughput bench and micro-profilers
SPEC_V1.md      the normative format specification
LICENSE         Apache License 2.0
```

Compiling `src/zpse_unity.c` puts the whole library in one translation unit
(maximum inlining, no cross-TU calls); each `src/*.c` also compiles on its
own, and every internal symbol is `static` or `ZPSE_`-prefixed so the
amalgamation has no duplicate names.

## Requirements and build

* A C11 compiler. The suites expect `gcc` on `PATH` and either a PowerShell
  host (`tests/run.ps1`) or a POSIX shell (`tests/run.sh`); the same
  translation units build with clang or MSVC.
* Nothing else: no libc beyond `string.h`/`stdint.h`/`stddef.h` in the library
  itself, no allocation, no threads, no I/O.
* x86-64 gets runtime-dispatched SIMD (SSE4.2 CRC32-C, SSSE3, AVX2) with the
  scalar paths always reachable; other architectures use the scalar code.
  Under MSVC the AVX2 intrinsics compile only when the translation unit is
  built with `/arch:AVX2` — without it the verified-equivalent scalar paths
  are used instead. Defining `ZPSE_FORCE_SCALAR` disables SIMD dispatch
  entirely (this is what the parity test builds).

Build the library together with any program of yours:

```sh
gcc -std=c11 -O2 -Wall -Wextra -I include your_program.c src/zpse_unity.c -o your_program.exe
```

## Quick start

The one-shot API is in `include/zpse/api.h`. Contexts borrow caller-owned
arenas: 384 KiB for the encoder, 256 KiB for the decoder, 64-byte aligned
(`ZPSE_ARENA_ALIGN`), initialised once with `zpse_encoder_init` /
`zpse_decoder_init`.

```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "zpse/zpse.h"

static uint8_t enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

int main(void)
{
    uint8_t in[65536], comp[70000], out[65536];
    uint32_t clen = 0, olen = 0;
    int rc;

    /* fill `in` with whatever you want to compress */

    rc = zpse_compress_block(in, sizeof in, comp, sizeof comp, &clen,
                             /*seq=*/0, /*is_last=*/1, enc_arena);
    if (rc != 0) {
        fprintf(stderr, "compress: %s\n", zpse_status_name(rc));
        return 1;
    }

    rc = zpse_decompress_block(comp, clen, out, sizeof out, &olen, dec_arena);
    if (rc < 0 || olen != sizeof in || memcmp(in, out, sizeof in) != 0) {
        fprintf(stderr, "decode: %s\n", zpse_status_name(rc));
        return 1;
    }
    printf("%u -> %u bytes\n", (unsigned)sizeof in, clen);
    return 0;
}
```

`zpse_compress_frame` / `zpse_decompress_frame` wrap the same block codec in
the 32-byte frame envelope (`SPEC_V1.md` Sec. 3) and split the input into
blocks of `nominal_block_size`. Every entry point returns bytes written on
success and a negative `PseStatus` (see `include/zpse/status.h`) on failure.

## Testing

```sh
powershell -NoProfile -File tests/run.ps1   # Windows
bash tests/run.sh                           # Linux / macOS / Git Bash
```

The two runners are twins: same suites, same flags, same checks, and each
captures every build's and every program's exit status directly (never
through a pipe), so a failing suite cannot be filtered into a passing one.
They build and run each suite against `src/zpse_unity.c` and stop with a
non-zero exit code if any of them fails:

| suite | coverage |
| --- | --- |
| `test_roundtrip.c` | block and frame round trips over degenerate and constant inputs at many sizes |
| `test_combinadic.c` | exact binomial values, rank/unrank round trips, hostile ranks |
| `test_hostile.c` | truncations, bit flips, reserved bits, bad transform ids, bad mask/residual payloads, output-overflow, random garbage — all must return a status, never crash |
| `test_units.c` | CRC, arena layout, header parse/serialize, K derivation, tables, tile round trips over every predictor and residual mode |
| `test_parity.c` | the full corpus plus four per-primitive sweeps (uniform, small-`nt`/mask-word boundary, stateful-table ids, and the state a classification leaves) |

The parity step is the important one: `test_parity.c` is compiled twice, once
with the default runtime SIMD dispatch and once with `-DZPSE_FORCE_SCALAR`,
and the runner requires the two runs to print byte-identical output and the
same `DIGEST` line. The SIMD paths are therefore checked to be bit-exact
replacements for the scalar reference, and the per-primitive sweeps in the same
file fold `k`, `nu`, `rmax`, `dmin`, `dmax`, the mask words, the packed
record and the chosen modes into a second digest for every predictor id, so a
divergence names the predictor that caused it. The four sweeps are:

| sweep | what it drives |
| --- | --- |
| `tileclass` | every predictor id over twelve tile shapes at `Nt = 256`, three literals, `tile_reset` 0/1 and two `seg_pos` offsets |
| `tilesmall` | `Nt` from 1 to 256 (so the 32-wide chunking's masked partial tail is exercised, and `Nt < 32` stays on the scalar classifier) over shapes built to land on the dense path's 31/32/33-exception boundary, a completely set first/last/only mask word, and pseudorandom fills; it also asserts that no mask bit at or above `Nt` is ever set |
| `tiletable` | the ids the encoder's prescreen never ranks (`ORDER1`, `ORDER2`, `TILE_LOCAL_CONSTANT`) at a full chunk, a partial tail and a sub-chunk length |
| `tilechain` | two tiles classified back to back on ONE lane state, the second without a reset - the histories and every byte of the table the predictor reads are folded in, and both records are then decoded in sequence, the second from the state the first decode leaves. Nothing else observes what a classification LEAVES behind, which is what made a grouped `ORDER1` scan look good and measure badly; see the performance note below |

A successful run ends with:

```
parity ok: DIGEST <value> (identical streams)
ALL SUITES PASS
```

`tests/profile.c` and `tests/profile2.c` are micro-profilers for individual
primitives, not part of the suite.

## Benchmarking

```sh
gcc -std=c11 -O2 -Wall -Wextra -I include tests/bench.c src/zpse_unity.c -o bench.exe
./bench.exe 300 5      # [iters] [reps], defaults 200 and 5
```

The harness compresses and decompresses a 64 KiB block of each pattern,
single-core, and reports the best of `reps` rounds for each direction (decode
is measured ten times per round) as GB/s plus the ratio and compressed size.
It then prints the machine's `memcpy`, `memset` and CRC32-C rates for the same
working set: those bound any codec on this format, because every block is
verified with a CRC32-C and the RAW fallback is a copy. The bench times
itself with `QueryPerformanceCounter` on Windows and `clock_gettime`
(`CLOCK_MONOTONIC`) elsewhere, so the same command works on any platform
the library builds on.

### Performance

Measured with `./bench.exe 60 5` on a desktop x86-64 machine (best of four
such runs, single core, 64 KiB blocks). Reproduce with the command above; the
absolute numbers move with the machine and the clock, the shape does not.

| pattern | ratio | encoded bytes | encode GB/s | decode GB/s |
| --- | --- | --- | --- | --- |
| zeros | 0.040 | 2616 | 12.8 | 16.5 |
| incr | 0.040 | 2619 | 1.2 | 8.6 |
| random | 1.000 | 65568 | 0.45 | 20.8 |
| sparse | 0.060 | 3962 | 0.38 | 4.7 |
| strided | 0.040 | 2625 | 1.1 | 8.6 |
| const5A | 0.040 | 2616 | 12.3 | 16.8 |
| alt00FF | 0.040 | 2618 | 2.5 | 13.9 |
| sawtooth | 0.163 | 10656 | 0.26 | 1.2 |
| lowentropy | 0.688 | 45058 | 0.13 | 0.9 |
| sparse2 | 0.048 | 3126 | 0.54 | 5.2 |

This round's encode change is in the block-level trial sweep, and it moved
three rows: measured by alternating the two binaries in the same clock
state (best of three rounds of 900 compressions, min per pattern), the
`sparse` row is 1.12x, `lowentropy` 1.12x and `sparse2` 1.06x, with every
other row within 1% (including `random`, whose incompressible path is
deliberately untouched) and all ten compressed sizes byte-identical. The
decode column is unchanged code this round and measures within 2%.

The same run reports this machine's ceilings for the same 64 KiB working
set: memcpy 69 GB/s, memset 123 GB/s and CRC32-C 30.5 GB/s. The CRC32-C
rate matters because every block carries a checksum of its decoded bytes,
so on the fastest paths - 16.2 GB/s against a 30.5 GB/s checksum - the
checksum is about half of the decode time, and it is now the floor under
this table: `crc32 r64,r64` retires one instruction per cycle here, which
is 8 B/cycle = 30 GB/s, and a PCLMULQDQ fold cannot beat that on a core
that retires one carry-less multiply per cycle (a fold-by-4 needs eight of
them per 64 bytes, the same 8 B/cycle). Holding the checksum down would
take a machine with two clmul per cycle or AVX-512 `VPCLMULQDQ`.

What sets each direction: encode is dominated by the per-tile predictor
search - the AVX2 prescreen computes the exception count of eight
predictors in one whole-tile pass, and a run of all-hit tiles is written
as its 8-byte record directly. Tiles that are mostly exceptions (text,
noise: the `random`, `sawtooth` and `lowentropy` rows) no longer pay a
per-exception scalar loop at all: the exception mask, the exclusion
residuals and the signed-delta range come out of one AVX2 pass, and the
predictions and residuals are compacted into position order with a
256-entry byte-shuffle table, a 32-exception chunk whose mask word is full
is placed with four stores instead of the expansion and the shuffle, and a
dense palette residual payload packs sixteen values per step through a
two-half byte shuffle (the isolated dense pack measured 293 -> 191 ns per
245 values; the earlier round's 16-wide unpack covers the decode side of
the same path). Decode is dominated by reconstruction
(exception tiles), the segment transforms, and the CRC; all-hit runs of a
segment coalesce into a single fill (and a run's records are now compared
two at a time), the delta8/xor8 inverses are AVX2 prefix scans whose carry
never leaves the vector domain, strided reconstruction is branchless and
interleaves its residue classes, and a single-segment raw-bytes block
decodes straight into the caller's buffer. On the exception-heavy rows the
work moved into wide code: a dense tile's residual units are placed with a
shuffle instead of a per-bit walk, and palette residual payloads unpack
sixteen values at a time (which is most of the `lowentropy` row's 3.2x).

That trial sweep is where this round's encode win comes from, and it is
worth recording what it can and cannot do. Every block runs the candidate
transforms in turn and keeps the smallest; an instrumented build shows
`lowentropy` running all three to completion (identity 45058 bytes,
delta8 57241, transpose-16 45392 - the identity wins, and the other two
produce a payload that is thrown away). A candidate's total only grows as
its remaining segments are written, so the moment the bytes it has already
produced reach the running winner it has lost by the same strict-`<` test
the sweep applies at the end, and the rest of that payload is work for
nothing. The sweep now hands each trial that bound (`min(RAW envelope,
running winner)`) instead of just the RAW envelope, which stops `delta8`
and the transpose trial at the winner's size rather than at the envelope;
the two cases stay distinct, because only the RAW bound still means "this
block is incompressible" for the sweep's own early exit. That is an exact
bound, not an estimate: it skips only payloads the sweep could not have
chosen. What it cannot do is prune a trial *early* on a large block - a
losing candidate still has to be encoded up to the winner's size to prove
it loses, and no data-derived lower bound does better, since an
enumerative mask could in principle take any single tile down to ~25
bytes.

The `ORDER1` and `ORDER2` predictor arms are the only classification paths
that stay scalar, because their prediction reads a per-lane table whose
chains cannot be evaluated 32 positions at a time (`ORDER1` 709 ns and
`ORDER2` 420 ns per 256-position tile, against 180-240 ns for the vector
arms). Both were written as grouped 8-lane scans and both were measured
before being kept: `ORDER1`'s scan is 1.15-1.23x faster in isolation, but
adding it to the unity build slowed the incompressible encode path by 4.5%
through code generation, and no codec path selects either id - the tile
encoder's prescreen ranks the other eight predictors and the encoder never
passes `ORDER1`/`ORDER2` - so the speedup would have been paid for by the
hot path and collected by nobody. `ORDER2`'s scan was 35% *slower* than its
per-position loop (its two tables cost the eight lanes more live state
than a group can hold), which the resetting regime hides: with the per-tile
reset that regime spends ~1300 of the arm's ~1740 ns clearing 2 x 8 KiB of
tables, so the memset, not the loop, is what `ORDER2` classification costs.
Neither scan shipped; `tilechain` exists to keep the state they would have
had to reproduce honest.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
