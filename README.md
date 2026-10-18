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
the smallest result. Within a segment, each tile independently picks from 12
predictors (`CONST_ZERO`, `LANE_PREVIOUS`, `LANE_DELTA`, `LANE_AVERAGE`,
`ORDER1`, `ORDER2`, `STRIDE_1/2/3/4/8`, `TILE_LOCAL_CONSTANT`), 7 exception-mask
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
                dense, segment, block, frame, vector, api)
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
  entirely (this is what the parity test builds). `ZPSE_FORCE_TRIAL_SWAP`
  is the other test hook: it makes the block encoder attempt its width-1
  candidates in the opposite order on every block, which changes no output
  (see the testing and performance sections).

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
| `test_stride3.c` | `STRIDE_3` (id 11): interleaved 3-byte records must round-trip *and* be recognised — the size has to collapse on data whose only redundancy is "same as 3 bytes back", at run boundaries, odd lengths and sub-tile sizes, and pure noise must not be hurt by the extra candidate |
| `test_parity.c` | the full corpus plus four per-primitive sweeps (uniform, small-`nt`/mask-word boundary, stateful-table ids, and the state a classification leaves) |
| `test_fuzz.c` | a randomized differential corpus: every size 1 to 64, every tile and segment boundary and its neighbours up to 64 KiB, 60 pseudo-random sizes, 18 shapes each (2718 blocks) plus 6 frames, each block round-tripped and its size, compressed length and compressed bytes folded into one digest |
| `test_lseed.c` | hand-built foreign segments whose ORDER1/ORDER2 tiles deliberately do NOT reset (non-resetting first tile, chains, mixed ORDER/STRIDE), each decoded by the library and by a reference in the file that hardcodes the documented segment-start state — the shape the lane-state seeding above exists for, and one this encoder cannot produce |
| `test_dense.c` | dense segment records (SPEC Sec. 6.3): 6044 records built by a reference writer over every predictor family, mask form and residual mode at nine plane sizes, plus hand-computed lane-history predictions, plus 15 malformed shapes (reserved bits/fields/modes, `ORDER1`, wrong spans, pad bits, non-ascending positions, trailing bytes) that must be rejected and never decoded, plus 96 emitter and sequence cases (a constant plane must come out dense exactly when the form is smaller, with the cheapest of the three constant predictors, through `zpse_segment_encode` and through the public block API, a near-constant plane must stay tiles, the palette form must be emitted only for alphabets of at most sixteen values with the record's prediction byte absent from the plane, and a palette plane that *contains* 255 must be filled with the largest byte it lacks and decoded through the exclusion-mapped residual alphabet) |
| `test_decide.c` | the block decision itself, pinned in bytes: alternating 0x00/0xFF must stay at 608 (not the 2618 identity cost), two-byte spikes at 1255, 4-bit noise at 33288 (the palette form, emitted as soon as it is priced) and zeros at 248 (the floor) — a regression to any of these still round-trips, so nothing else in the suite would notice |
| `test_sizes.c` | the ratio corpus (`tests/corpus_gen.h`, shared with `tests/clen_corpus.c`) with all 768 compressed sizes pinned: 3 seeds x 16 shapes x 16 sizes, every case decoded and byte-compared, and the suite fails if any case's size grows. A shrink is reported in the summary and does not fail. This is the gate that catches "the encoder still round-trips, it just stopped choosing the smaller form" — it caught a proxy-sampling change at +34,314 bytes over 21 cases while every other suite stayed green |

The parity step is the important one, and it is two checks over three builds
of the same three suites: `test_parity.c`, `test_fuzz.c` and `test_lseed.c`
are each compiled with the default runtime SIMD dispatch, with
`-DZPSE_FORCE_SCALAR` and with `-DZPSE_FORCE_TRIAL_SWAP`, and the runner
requires all three builds of each to exit 0, to print the same digest line,
and to be byte-identical everywhere else too, so a divergence also names the
case that caused it. The first two builds prove the SIMD paths are bit-exact
replacements for the scalar reference. The third takes the encoder's
candidate-attempt order and permutes it on every block (see the performance
note below), which proves that order cannot change a single compressed byte:
`test_parity.c` does that over the fixed corpus, `test_fuzz.c` over the wide
randomized one, so the invariant is not a property of the patterns someone
happened to list. The per-primitive sweeps in `test_parity.c`
fold `k`, `nu`, `rmax`, `dmin`, `dmax`, the mask words, the packed
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
parity ok (scalar): DIGEST <value> identical streams
parity ok (swap): DIGEST <value> identical streams
fuzz ok (scalar): FUZZ cases=2718 frames=6 digest=<value> identical streams
fuzz ok (swap): FUZZ cases=2718 frames=6 digest=<value> identical streams
lseed ok (scalar): LSEED: ALL OK (10 cases) identical streams
dense records ok   (DENSE: cases=6136 failures=0)
lseed ok (swap): LSEED: ALL OK (10 cases) identical streams
ALL SUITES PASS
```

`tests/profile.c` and `tests/profile2.c` are micro-profilers for individual
primitives, not part of the suite.

## Benchmarking

```sh
gcc -std=c11 -O2 -Wall -Wextra -I include tests/bench.c src/zpse_unity.c -o bench.exe
./bench.exe 300 5                  # [iters] [reps], defaults 200 and 5
./bench.exe 60 5 sawtooth lowentropy   # optionally just these patterns
```

Naming patterns restricts the run to them, in the order given, which is what
makes a profiler run or a single-row check concentrate on one path; an unknown
name is refused with a non-zero exit rather than silently benchmarking nothing.

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
| zeros | 0.004 | 248 | 14.5 | 21.2 |
| incr | 0.040 | 2619 | 5.2 | 10.2 |
| random | 1.000 | 65568 | 3.1 | 20.6 |
| sparse | 0.060 | 3962 | 1.9 | 5.3 |
| strided | 0.040 | 2625 | 3.1 | 10.3 |
| const5A | 0.004 | 248 | 14.3 | 21.7 |
| alt00FF | 0.009 | 608 | 3.5 | 7.6 |
| sawtooth | 0.116 | 7584 | 1.1 | 5.5 |
| lowentropy | 0.508 | 33288 | 2.1 | 13.0 |
| sparse2 | 0.019 | 1255 | 2.8 | 6.8 |

The table is one session (both columns re-measured together) and the deltas
below are paired measurements, so the absolute columns move between machines
and runs while the sizes and the ratios do not. The encode column reflects the
encoder rounds at the end of this file; before the first of them, `zeros` and
`const5A` were 3.4 GB/s, `alt00FF` 2.3, `sparse` 0.98, `lowentropy` 0.33 and
the mean across the ten rows was 2.5 GB/s. `alt00FF` now spends encode time on
a stream 4.3x smaller than the one it used to write (608 bytes, not 2618), and
its decode rate is for that stream: 608 bytes in, 64 KiB out, as sixteen dense
4 KiB records instead of one 2618-byte plane.

### `lowentropy`: what the sweep costs

(The decode half of the 1 GB/s target this section reasons about was later
reached by a format change rather than by the sweep -- see "Dense palette
records" at the end of this file.)

That row is the sweep's worst case, and it is now the one place where the
encoder deliberately trades a little ratio for a lot of time. Measured in
`__rdtsc` reference cycles per 64 KiB block (best of many, so clock drift
does not enter):

| | cycles/blk | GB/s |
| --- | --- | --- |
| encode, full sweep (`-DZPSE_KEEP_EXACT_SWEEP`) | 949,000 | 0.26 |
| encode, as shipped (sampled decision) | 470,000-510,000 | 0.49 |
| decode | 260,000-275,000 | 0.94 |
| budget for 1 GB/s | 242,000 | 1.00 |

On 4-bit noise the prescreen prices the identity, DELTA8 and the transpose
within 0.02% of each other (45,020 vs 45,012 bytes), so no prune can drop
either and the full sweep encodes a trial per survivor. A trial, per tile:
prescreen 190 cycles, classify 850 (of which ~640 is the dense compaction and
~350 the distinct-residual count), mode search 50, pack 650 (palette discovery
~300, palette packing ~250, mask 60, validation 30) -- 1,740 cycles per
256-byte tile, times 256 tiles, plus the transform stagings and probes.

The sampled decision is therefore the default now: `zpse_block_encode` prices
every survivor exactly -- the same classify, mode search and pack, on 16 tiles
spread over the block -- and encodes only the winner. That is the 1.9-2.0x
above, and the ratio it costs is measured rather than hoped for: identical
sizes *and* CRCs on all 768 cases of `tests/clen_corpus.c`, 74 of
`tests/test_fuzz.c`'s 2,718 cases one byte larger, and one 8 KiB block of
random mixed regions 1,735 -> 3,645 bytes, because a 16-tile sample cannot see
a block whose regions differ within it. `-DZPSE_KEEP_EXACT_SWEEP` restores the
exact sweep for anyone who would rather pay the time.

What is left between 510,000 and the 242,000 that 1 GB/s needs is not a
micro-optimisation: it is the per-tile framing itself. Every tile carries its
own 8-byte header, its own mask and its own residual mode, so classify, the
mode search and the pack all run once per 256 bytes. A segment-level dense
record -- one predictor, one mask over the whole 4 KiB plane and one residual
stream, with the palette discovered once instead of sixteen times -- would
amortise all three over sixteen times the data. That record is now defined
(SPEC Sec. 6.3), decoded and emitted: `src/dense.c` handles one record per
plane, with its own mask cursor, all three lane-history predictors, the STRIDE
family over absolute positions and residual modes 0-3, under the same
hostile-stream validation as tiles, and `tests/test_dense.c` covers it. A
record AREA is a sequence of records (4096 + 4096 + ... + remainder), so a
64 KiB block encoded as one segment uses the form too.

The emitter takes exactly the shapes the form wins on in v1, and no other: a
plane that is a single byte value end to end, and (see "Dense palette records"
below) a plane of at most sixteen distinct values. A record with K == 0 is a
fill, so the first costs 12 bytes per 4 KiB against ten per 256-byte all-hit
tile. Everything else keeps the tiles it had, byte for byte. The decision costs no scan: the all-hit run path has already proved each
tile is constant, so one byte per tile settles the whole plane, and the
records are replaced only when the dense form is smaller (a plane holding one
tile keeps its 10-byte tile record, which beats 12). Three consequences the
sweep had to learn. Its per-candidate floor is the smaller of 10 bytes per
tile and 12 per 4 KiB, or it would prune a candidate whose transformed plane
happens to be constant. A candidate that can only *tie* the winner may no
longer be skipped when it comes earlier in the list, because a tie goes to the
lower index -- and the sweep may not stop while such a candidate is left.
Without those two rules the swapped attempt order produced a different stream
for a 4 KiB constant block; `ZPSE_FORCE_TRIAL_SWAP` now proves it does not.

The block sweep prices candidates with `zpse_segment_proxy_cost`, and a
constant plane has to be priced at the dense rate there too, or the estimate
is ten times the truth and the candidate is dropped before it can win. That
was a real miss, not a hypothetical: the 16-way split of `sparse2` (one nonzero
two-byte header every 256 bytes) has fifteen all-zero planes, so it costs 16
segment headers, fifteen 28-byte dense payloads and one 675-byte tile segment
-- 1255 bytes -- and the sweep never tried it, because the estimate said 16 x
~2560. Pricing the constant planes correctly lets it through: `sparse2`
3126 -> 1255 bytes, ratio 0.048 -> 0.019, and encode 1.13 -> 2.89 GB/s because
the winner is the cheap candidate.

Measured, five alternating rounds of `bench 80 3`, best per row: `zeros` and
`const5A` 2616 -> 248 bytes (ratio 0.040 -> 0.004), encode 3.21 -> 3.23 and
3.24 -> 3.23 GB/s, decode 22.4 -> 21.8 and 22.5 -> 22.0 GB/s on this machine;
`sparse2` 3126 -> 1255 bytes with encode 1.13 -> 2.89 and decode 5.81 -> 5.42
GB/s; every other row unchanged in size and within +-3% in both directions.
`tests/clen_corpus.c`:
324 of 768 cases are smaller, none is larger, and every one of the 324 is a
plane that is constant end to end. The `zeros`/`const5A`/`sparse2` columns in
the table above are those paired ratios applied to the session's numbers.

Four cheaper things were tried on the per-tile path and measured, and are not
in the tree: widening the dense classifier's compaction stores from 8 to 16
bytes (0.99x), writing the per-exception arrays in place instead of copying
them out of the classifier's staging buffers (neutral), a branchless
block-at-a-time palette discovery scan (slower -- the branch it removes is
why the repeats cost nothing, since a repeat writes no store at all), and
skipping the actual-byte compaction on the grounds that the packer can gather
those bytes itself (0.98x: modes 1 and 4 then pay the gather, and the two
stores per chunk it saves are cheaper than that). The per-tile path is tight;
what is left is the framing.

This round adds a twelfth predictor and lets the block sweep skip candidates,
and it is the first round in some time to move a compressed size: nine of the
ten `clen` values are the ones this format has always produced, and
`lowentropy` is 46 bytes smaller. On 610 KB of this repository's own sources
(the implementation, the headers and the suites) the new build produces 1,824
fewer bytes: 576,543 -> 574,719, ratio 0.9441 -> 0.9412.

`STRIDE_3` (predictor id 11) is `x_hat = x[p - 3]`: the STRIDE family at S = 3,
which the format had at S = 1, 2, 4 and 8 but not at the one width an
interleaved 3-byte record (an RGB pixel, an xyz vector) wants. It is appended
LAST in the encoder's prescreen order, deliberately: the winner is the first
predictor with the strictly smallest exception count, so a new entrant can only
take a tile where it beats every candidate that precedes it. That is a change
in the *choice*, not automatically in the bytes -- the tile is packed from
its two best candidates and the smaller record wins -- and the measured effect
is the one above: every bench `clen` is unchanged except `lowentropy`, and the
real corpus is 0.32% smaller. `tests/test_stride3.c` is the suite for it: pure
period-3 data collapses from 65536 to 2883 bytes, 8-pixel RGB runs to 18826,
and pseudorandom data stays at exactly RAW size, which is the check that a
candidate which cannot win costs no ratio.

The sweep change is a prescreen proxy. `zpse_block_proxy` prices a candidate by
staging its transform over the first 16 KiB and summing the prescreen's own
per-tile estimate, and `zpse_block_encode` then fully encodes only the
candidates whose estimate is within ~6% of the best (always at least one). A
full trial is a classify plus mode search plus pack of every tile, so on a ramp
or a stride -- where one candidate is four or more times more predictable than
the other -- this replaces three trials with one. Measured against the previous
commit, same source, alternating binaries, min per pattern over nine rounds of
`60 3`: encode `incr` 3.65x, `sawtooth` 3.38x, `strided` 2.36x, `sparse` 2.31x,
`sparse2` 1.76x, `lowentropy` 1.17x, and the rows whose first trial already wins
(`zeros`, `const5A`, `random`, `alt00FF`) at 1.00-1.03x. Decode is untouched by
this round and measures 0.92-1.06x across the ten rows, i.e. noise. Two costs
are worth stating rather than hiding: the proxy is pure overhead when nothing
is droppable -- `random` is incompressible and its sweep is ended by the
RAW-envelope budget instead -- and the ninth prescreen count is work on every
tile of every encode. With the proxy removed entirely the same build still
measures `random` 0.93x and `strided` 0.90x, so that part is the added
candidate's price, not the proxy's.

The proxy is gated as cheaply as it can be: it runs only for blocks of at least
8 KiB with more than one candidate, and it is skipped entirely when the
order sample already reports that the identity's plane predicts every sampled
tile, because such a block reaches the global floor on its first trial and
pricing the others would then be most of its encode. That gate is not
cosmetic: without it `zeros` measured 0.50x and `const5A` 0.47x against the
previous build, because the old sweep ends those blocks after one trial.

Pruning compacts the candidate list, which is why the attempt order stopped
being an index permutation over it: candidates are moved -- transform id,
segment count, metadata and the original list index the winner's tie-break
compares -- and the sweep walks the surviving slots in order. The permutation
form could point the loop at a slot the pruning had already dropped, and on
`sawtooth` that ran a stale DELTA8 (12345 bytes) against the transpose that
should win (10656, +16% ratio) -- invisible to the parity step, because the
staleness is deterministic in every build. Swapping before pruning means the
sweep is exactly the surviving list, and `-DZPSE_FORCE_TRIAL_SWAP` still
permutes it on every block, so both the proxy's gate and its ordering are
covered by the parity suite rather than by inspection.

Two smaller encoder changes ride along. The classifier now records the actual
bytes at the exception positions while it walks the tile (`cl->act_at_exc`),
including in the AVX2 dense path, so the packer reads them instead of gathering
them a second time for the `RAW_ACTUAL` and `SIGNED_DELTA` payloads. And
`zpse_tile_choose_modes` now prefers `RAW_ACTUAL` over `EXCLUSION_U8` when the
two cost exactly the same (one byte per exception) and the tile has a real mix
of hits and exceptions: the decoder then copies the byte instead of rebuilding
it from the exclusion-domain residual and the prediction, which removes a
serial dependency from reconstruction. Same length, different bytes -- both
legal and the two extremes keep `EXCLUSION_U8`, because k == 0 has no payload
at all and k == Nt is the verbatim path. `SPEC_V1.md` documents id 11 and
`ids 12..255` as invalid, and `zpse_tile_header_parse` now accepts 0..11 like
the tile validator does: the change left that parser at 0..10, which would have
rejected a legal id-11 stream at the very boundary it exists to check.

The previous round's change was to the lane state, and it is a speed-only
change that touches no encoded byte. `zpse_lane_reset` used to fill the three ORDER
tables -- 66 KiB of memset, `order1` with the literal and `o2val`/`o2conf`
with the literal and zero -- at the start of every segment, and the block
decoder did it once per segment in front of a segment decoder that did it
again. Only predictors 3 and 4 (ORDER1/ORDER2) ever read those tables, this
encoder never emits them (they are not in the tile prescreen), and a tile
whose reset bit is set re-seeds the table it reads inside `zpse_tile_decode`
-- which is every tile the encoder produces. So the tables were being
zeroed, twice, for tables that no tile in the stream would read: 132 KiB per
segment on the decode side and 66 KiB per segment per candidate on the encode
side (16 segments in the transpose candidate, 3 candidates per block). The
reset is now split into the histories (16 bytes, established as before) and
the tables, and the tables are seeded lazily -- in the segment decode loop,
once, just before the first tile that can read one without resetting it. A
segment whose tiles all reset pays nothing; a segment that contains a
non-resetting ORDER tile sees exactly the state the eager reset would have
given it. That equivalence is checked directly: a throwaway harness builds
segment payloads whose tiles deliberately do not reset (a non-resetting
*first* tile, ORDER1 chains, ORDER2 chains, mixed ORDER1/ORDER2/STRIDE
chains) and requires the two policies to decode them to identical bytes -
they do, and disabling the new seeding makes the same harness report
different output, so the check has teeth. Only a foreign stream can reach
that path at all, since this encoder resets every tile; the shipped
`tilechain` sweep covers the neighbouring question with the encoder's own
records.
Measured against the previous build from the same source with only the reset
policy switched (paired, alternating binaries, min per pattern): decode
`const5A` 1.61x, `sawtooth` 1.49x, `zeros` 1.41x, and encode `const5A` 1.40x,
`zeros` 1.10x, `sawtooth` 1.07x, with the rows whose blocks are one segment
of incompressible or code-heavy tiles unchanged. Every `clen` and both corpus
digests are identical to the previous round, so no stream moved.

Three size levers were measured in an earlier round and none of them shipped,
which is worth recording because each looks attractive from the code:

* The tile predictor search tries the best prescreened predictor and a second
one only when its exception count is within 4 of the best. Widening that
window to 12 (where it saturates) changes 570 bytes of 523 KB of this
repository's own sources and C headers -- 0.12% -- and no bench block at all,
while costing a second classify and pack on more tiles. The heuristic is not
the limiter.
* The block sweep tries three candidates (identity, DELTA8, and the
width-appropriate plain transpose). Adding XOR8, DELTA_TRANSPOSE_4 and
DELTA_TRANSPOSE_16 changes the same 523 KB corpus by exactly 0 bytes.
* Every coded tile in that corpus uses mask mode 1 (RAW_BITMAP, 32 bytes); an
ENUMERATIVE mask for the same masks would need 38,120 bytes against 53,956,
i.e. **3.43% of the compressed output** on real source, and would win on 1577
of 1693 tiles. It is not taken because the codec is a reference path, not an
affordable one: for a 256-bit mask the enumerative rank costs 0.8 us at k=8
and 368 us at k=128, and the unrank 34 us at k=8 and 1.0 ms at k=240, against
15-27 ns for the raw bitmap in both directions. The mask-mode rule is
choosing the only mode whose codec is affordable, not leaving a cheap 3% on
the table.

An earlier round's encode change is a candidate-attempt order picked from the
tile encoder's own prescreen. The sweep's result does not depend on the order
it tries its candidates in - the winner is the smallest total, ties going to
the earlier candidate in the list, and that is applied explicitly - so the
order is a free knob, and it pays because a later candidate is handed the
running winner as a size bound: attempting the likely winner first is what
makes the losers stop early. Only the two width-1 candidates can be swapped,
which keeps the argument tight: the attempted set is the same in both orders,
so the floor-based exit sees a superset of what it saw before and can only
stop later (never earlier), the two-failure exit still costs exactly two
attempts over the same two candidates, and no tie can resolve differently.
The estimate that picks the order samples six tiles spread across the block,
prescreens each on both planes with the same call the segment encoder makes,
and swaps only when DELTA8's plane is at least four times more predictable
than the raw bytes - a clear margin, because a wrong swap costs the trial the
other order would have stopped, and low-entropy data can look equally bad on
both planes. It is a sample, not a bound: it is used only to pick an order, so
a bad estimate costs time, never a different block. On a block too small to
lose a tile's worth of work it is not run at all, and it bails out as soon as
a plane looks like it will hit the global floor. Measured the same way as the
trial bound (paired, alternating binaries, min per pattern): `incr` 1.24x,
`strided` 1.16x, `sawtooth` 1.27x, every other row within the 1% noise floor
and all ten compressed sizes identical - the rows it moves are exactly the
ramps and strides, where the identity trial is a full block of the most
expensive tiles the encoder has and loses to DELTA8 by a factor of four or
more. `-DZPSE_FORCE_TRIAL_SWAP` is the test hook that takes the other
decision on every block, so the difference between the two orders is covered
by the parity step above rather than by inspection.

An earlier round's encode change is in the block-level trial sweep, and it
moved three rows: measured by alternating the two binaries in the same clock
state (best of three rounds of 900 compressions, min per pattern), the
`sparse` row is 1.12x, `lowentropy` 1.12x and `sparse2` 1.06x, with every
other row within 1% (including `random`, whose incompressible path is
deliberately untouched) and all ten compressed sizes byte-identical. The
decode column is unchanged code this round and measures within 2%.

The same run reports this machine's ceilings for the same 64 KiB working
set: memcpy 50 GB/s, memset 77 GB/s and CRC32-C 24 GB/s. The CRC32-C rate
matters because every block carries a checksum of its decoded bytes, so on
the fastest paths - 22.9 GB/s of decode against a 24 GB/s checksum - almost
all of that decode time is the checksum, and it is the floor under this table: `crc32 r64,r64` retires one instruction per cycle
here, which is 8 B/cycle = ~31 GB/s, and a PCLMULQDQ fold cannot beat that
on a core
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

That trial sweep is where the previous round's encode win came from, and it is
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

A dense tile that cannot be palettized no longer counts its distinct
exclusion residuals to the end. The residual-presence pass is the largest
single cost inside a dense classification (about 105 of its 205 ns for a
256-position tile with k~250, and it is that size because it is ~250
independent byte stores into a 256-byte array plus the fold that turns them
into `cl->seen`), and its whole product is `cl->nu` -- the count the mode
search compares against 16, the most entries a palette can hold. Once a
seventeenth distinct residual is known the palette mode is rejected and no
larger exact count can change one packed byte, so the pass now records the
first 32 exceptions, folds them, and stops there when that prefix already
holds 17 distinct values; a tile still under 17 after the probe walks the
rest exactly as before and stays exact. `cl->seen` is trimmed together with
`nu` so the two can never disagree (`popcount(seen) == nu`, which the parity
sweeps fold), and `tile.h` documents the cap: `nu` is exact up to 17 and 17
above it. Measured paired against the previous build, alternating binaries,
best of six rounds of `200 6` per pattern: encode `random` 1.21x and
`lowentropy` 1.03x, every other row 1.00-1.01x, both decode columns
unchanged, all ten compressed sizes byte-identical, and the parity, fuzz and
lseed digests identical to the previous build's in all three builds
(default, `-DZPSE_FORCE_SCALAR`, `-DZPSE_FORCE_TRIAL_SWAP`). An instrumented
build counts 444 dense classifications on `random` -- every one of them
retired by the probe -- and 699 on `lowentropy`, 520 of which run the full
pass because their tiles stay under the cap. The trim itself is kept out of
line: inlined into the classify epilogue, which every tile runs, its
register pressure cost the `sparse` row 7-9% without ever being called
there, the same code-generation failure mode the `ORDER1` note above
records.

This round abandons a hopeless transform-sweep trial as soon as the front of
the block says it is hopeless. On a block whose every candidate has to beat
the RAW envelope -- incompressible data, where the answer is RAW and the only
question left is how many trials it takes to say so -- `zpse_block_encode`
now also passes `zpse_segment_encode` a flat-abort threshold, and a trial that
has covered a sixteenth of the block (floored at four tiles) with not one tile
paying for itself is dropped with the same status the byte budget returns.
The probe is one compare per tile (`seg_flat_abort`), and the run it measures
ends at the first tile that does pay, so a block with any compressible front
pays nothing for the test.

Abandoning a *hopeless* trial early is only unsound in one direction: the
front of a block can be noise while the rest of it compresses. A 64 KiB block
of 16 KiB of noise and then zeros codes at 20883 bytes, and a test that only
ever looks forward cannot tell it apart from a block that is noise all the
way through. Ungated, the probe measured `random` 2.4x and turned exactly that
block into RAW (65568 against 20883, -68%), which is why it ships gated by a
whole-block sample: sixteen tiles at the odd thirty-seconds of the block are
prescreened on the trial's own transformed planes, and the abort is armed only
when every one of them is a tile the segment encoder itself would call
near-random -- all nine screened predictors miss at least 7/8 of it, the same
threshold its single-candidate rule uses -- so a compressible stretch wider
than a sixteenth of the block is seen whatever the segment shape. The sample
costs sixteen prescreens per trial and is skipped entirely once a winner
smaller than RAW exists, which is the only condition under which no segment
could be armed anyway.

Measured paired, alternating binaries, best of four rounds of `200 6` per
pattern, against the previous build: encode `random` 1.90 -> 0.33 ns/B (5.76x),
the other nine rows 1.000-1.10x, both decode columns unchanged, all ten
compressed sizes byte-identical, and the parity, fuzz and lseed digests
identical in all three builds. The ratio check is the larger one, because this
is a heuristic and the sample is a sample: on 110,000 adversarial shapes built
for exactly the case above -- a noise block with a compressible stripe of
random length at a random offset, 1 byte to 64 KiB, five fill patterns, plus
the mirror shape of a compressible block with a random noise stripe, at block
sizes 8, 16, 32 and 64 KiB -- every compressed size is byte-identical to the
previous build's. A stretch thinner than a sixteenth of the block that the
sampled tiles straddle can still be aborted, and the cost is bounded by that
stretch; the sweep did not find one. `ZPSE_FLAT_MIN_DIV` (default 16) sets the
fraction of the block, and 0 disables the probe.

The next round removes two pieces of work from the encoder's critical path
without touching a single compressed byte. First, the identity trial no longer
stages anything: a width-1 split with neither DELTA8 nor XOR8 is the input, and
the segment encoder only reads its plane, so `planes[0]` points straight at the
caller's buffer and every block that tries the identity (which is all of them)
saves the 64 KiB copy that `zpse_transform_forward` would have made. The
overlap rejection inside the transform exists because it writes the plane;
not copying at all is what makes it unnecessary. Second, a candidate whose
*floor* -- block header, metadata, segment directory, every segment header,
plus 10 bytes per tile -- already exceeds the running winner is dropped before
its transform is staged, instead of after its first segment has been encoded.
The comparison is strict (`floor > best`), because a candidate that could tie
still has to be attempted: ties go to the earlier candidate in the list, so
skipping one on equality could change which of two equal-size encodings wins.

The measured effect, alternating binaries over six rounds of `80 3`, min per
pattern: encode `zeros` 1.26x, `const5A` 1.24x, `random` 1.04x, `strided`
1.03x, `sawtooth` 1.03x, `alt00FF` 1.02x, the rest within +-3% (i.e. noise),
decode untouched, and all ten `clen` values byte-identical. The parity digest
(`BB15773EC15BF841`), the fuzz digest (`B23313EB9E83B797`) and every compressed
size in the adversarial sweeps -- 40,000 noise-with-a-stripe and
compressible-with-a-noise-stripe shapes, 4,000 mixed-offset shapes, the
threshold sweep and the 8/12/32 KiB small-block sweeps -- are unchanged in the
default, `-DZPSE_FORCE_SCALAR` and `-DZPSE_FORCE_TRIAL_SWAP` builds.

Two cheaper-looking variants lost on the ratio guard and were not kept. Pricing
the proxy on the first 8 KiB instead of 16 KiB is ~10% faster on incompressible
blocks, but a block whose structure only appears past the first quarter -- a
long noise prefix in front of a compressible body -- is then mis-ranked:
sawtooth behind 8 KiB of noise went from 13,855 to 65,568 bytes (RAW). Pricing
a fixed sixteen tiles per candidate instead of one in every two is worse on
*both* axes: the coarser estimate prunes badly, so more full trials run and the
whole encode slows down (`random` 0.29 -> 0.43 ns/B). Shrinking
`ZPSE_FLAT_MIN_DIV` from 16 to 32 or 48 also fails the guard: on 20,000
compressible-with-a-noise-stripe shapes, 308 of them fall back to RAW
(46,526 -> 65,568) because two flat-aborted trials in a row end the sweep
before the transform that would have coded them is attempted.

The sweep's prescreen has one blind spot, and this round prices around it.
`zpse_segment_proxy_cost` sees nine predictors; data on a small alphabet (a
4-bit sensor, a nibble-packed record, text) is invisible to all nine, so every
candidate prices the same "no gain", the ~6% prune keeps them all, and every
one is then encoded in full. `lowentropy` -- uniform 4-bit bytes -- was the
worst row in the table for exactly that reason: three full trials per block,
468 us of tile work, where one trial would produce the same 45,012 bytes.

`zpse_block_prune_exact` closes it with a different *estimator* rather than a
cheaper one: classify, choose modes and pack eight spread tiles per candidate
(the same three calls a real tile makes, so nibble packing and the mask modes
are visible), fold the sample back up to the candidate's tile count, and keep
only what is within the same ~6% of the best. It runs only where the sweep has
already decided to spend another trial (`best_total > min_floor` over the
candidates left) and only after the first winner exists, which is what keeps
it free everywhere else: a constant or all-hit block ends before it, a block
whose first trial aborted (noise) never reaches it, and a block the floor test
stops pays nothing. The probe budget is per candidate, not per plane, so a
16-segment transpose prices sixteen tiles in total rather than sixteen apiece.

Measured against the previous commit, alternating binaries, best of six rounds
of `80 3`: encode `lowentropy` 0.16 -> 0.22 GB/s (1.38x), `sparse2` 1.05x, the
other eight rows 0.98-1.02x, both decode columns unchanged -- and all ten
`clen` values byte-identical, because a candidate the probe drops could not
have won: 6 of 6 adversarial sweeps (the two noise/stripe families, the mixed
offset family, the threshold sweep and the 8/12 KiB small-block sweeps, 66,000
shapes) reproduce every compressed size byte for byte, and the parity
(`BB15773EC15BF841`), fuzz (`B23313EB9E83B797`) and lseed digests are unchanged
in the default, `-DZPSE_FORCE_SCALAR` and `-DZPSE_FORCE_TRIAL_SWAP` builds.
`ZPSE_PROBE_TILES` (default 8) sets the per-candidate budget; 16 and 48 were
measured too and prune no further, so the smaller budget is the one that ships.

### Where the 4-bit noise row spends its time

`lowentropy` (uniform 4-bit bytes) is the slow row, and it is worth knowing
why. Its tiles are dense: ~235 of 256 positions are exceptions, coded the same
way every time -- `RAW_BITMAP` (32 bytes) plus a fifteen-entry palette and four
bits per exception (~174 bytes per 256-byte tile). Two candidates survive the
sweep and both are encoded in full, because the split candidate ties the
identity to within 0.02% (45,020 vs 45,012 bytes) and a tie goes to the
earlier one, so the challenger can only be dropped by *knowing* it loses.

What ships this round is a certificate. The classification already counts the
distinct exclusion residuals (the mode search needs `nu` to size the palette),
so the palette packer no longer rediscovers them: it walks the tile only as far
as the last NEW value -- about fifty values instead of all ~235 -- and the
first-appearance order, and therefore the payload, is unchanged. The hint is
verified before use (the prefix must hold `nu` distinct values and each must be
set in the classification's set), so a stale certificate costs time, never
bytes. Decode gets the same treatment from the other end: `RAW_BITMAP` has to
prove its 32 payload bytes hold exactly `k` bits, and eight SWAR popcounts
(~95 cycles of a tile) became one SSE2 fold-and-SAD pass.

Paired binaries, best of nine, cycles per 64 KiB block: encode 1,152,000 ->
900,000 (1.28x, 0.22 -> 0.28 GB/s), decode 274,000 -> 259,000 (1.06x). All ten
bench `clen` values, all 768 `tests/clen_corpus` cases and all 4,000 random
multi-texture blocks of the sweep below are byte-identical to the previous
commit, as are the parity (`BB15773EC15BF841`), fuzz (`B23313EB9E83B797`) and
lseed streams in the default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` builds. `tests/clen_corpus.c` is the tool those
numbers come from: it prints one line per case with the compressed size and a
CRC of the compressed bytes, so two builds can be diffed instead of averaged.

Three larger heuristics were measured and rejected, and they are the reason
this round is a 1.28x and not the 4.5x that "above 1 GB/s" would need:

* **Drop the near-tie candidate.** Pruning every candidate that is not
  strictly cheaper than the best on the existing 16 KiB prescreen proxy is
  worth 2.8x on this row (0.22 -> 0.62 GB/s, `clen` unchanged at 45,012). It
  also changed 55 of the 4,000 random multi-texture blocks, all worse,
  20,596 bytes in total and up to +3,591 bytes on one block: a prefix proxy
  mis-ranks exactly the heterogeneous blocks the +6% margin exists for.
* **"Noise everywhere means no split can help."** Filling the block's count
  from the sixteen spread noise samples `zpse_block_is_noise` already takes
  reads well and is false: a 30-character permutation cycle is noise to all
  nine predictors, yet a 16-way split exposes a period the raw plane does not
  have (3,070 -> 65,568 bytes, i.e. fallen back to RAW). The abort keeps that
  evidence because there the trial has already been attempted and is losing.
* **Price the challenger exactly on spread tiles and drop it if the estimate
  is not strictly smaller.** Whole-block staging, 8 and 64 sampled tiles: the
  corpus is clean, but 30-55 of the 4,000 random blocks still get worse
  (one by 3,591 bytes). Sampling cannot resolve a 0.02% tie without also
  resolving blocks whose structure sits between two samples.

For the arithmetic of the target: 1 GB/s is ~216,000 cycles per 64 KiB block,
which -- with the challenge above -- is 422 cycles per 256-byte tile over 511
tiles, staging and CRC included. A tile costs ~1,760 today, of which ~1,300 is
classify plus pack at about one cycle per byte per pass. Reaching the target
therefore needs roughly 4x on both trials at once, which is a format-level
change (a whole-segment dense record instead of one bitmap and one palette per
256 bytes) rather than more micro-optimisation of the per-tile path.

### Dense palette records: `lowentropy` decode 0.9 -> 4.9 GB/s

The per-tile path is the wrong shape for a plane on a small alphabet. Every
256-byte tile pays a header, a mask and a residual mode, and the decoder then
runs a predictor chain to reconstruct 256 bytes of which ~235 were sent
literally. SPEC Sec. 6.3's dense record exists to amortise that over 4 KiB, and
`src/dense.c` already accepted any conforming record; what was missing was an
emitter for anything but a constant plane.

A plane whose bytes span at most sixteen distinct values is now emitted as one
ALL_EXCEPTION + PALETTE_BITS record per 4 KiB, predicted by a TILE_CONSTANT the
plane does not contain (otherwise the exclusion mapping has no residual for the
bytes equal to it; 255 is preferred when absent, where the residual of every
value below it is the value itself). ALL_EXCEPTION makes every position an
exception, so the record carries no mask and the decoder runs no prediction at
all: it is one palette expansion per 4 KiB, decoded straight into the output
with the exclusion inverse applied 16 bytes at a time.

`lowentropy` (uniform 4-bit bytes), 64 KiB blocks:

| | before | after |
| --- | --- | --- |
| bytes | 45012 (0.687) | 33288 (0.508) |
| encode GB/s | 0.52 | 0.33 |
| decode GB/s | 0.94 | 12.8 |

Decode is 13.6x and the stream is 26% smaller; encode pays 36% because the row
now runs the tile search *and* the palette pass -- the tile loop stops as soon
as the tiles can no longer win (at ~74% of the plane here), but it still has to
reach that point. The form is priced only where it can plausibly be used: the
first tile is probed for its alphabet (256 bytes, and more than sixteen
distinct values settles it outright), the whole plane is scanned only when that
tile needed a mask or a residual at all, and the form is priced only when that
tile cost at least an eighth of its own bytes. All-hit planes -- every periodic
and near-constant block, nine of the ten bench rows -- never pay for the scan
and keep their tile coding byte for byte: `incr`, `strided`, `alt00FF`, `zeros`,
`const5A` and `sparse2` are unchanged in size and within noise in both speeds.

Two decode changes were needed to reach that number, and both are shape-level
rather than micro-optimisation:

* **`residual.c`, mode-2 palette unpack.** Four-bit indices tile a byte
  exactly, so one 16-byte load carries 32 values: two nibble extractions, two
  `pshufb` palette lookups and two interleaved stores cover 32 values where the
  sixteen-value loop covered 16. Every plane of 9-16 distinct values lands on
  that width, tiles and dense records alike.
* **`dense.c`, the exclusion inverse.** The encoder picks `C == 255` whenever
  the plane does not contain 255, and `zpse_exclusion_inverse(r, 255) == r` for
  every residual a payload can hold -- so the reconstruction IS the expansion
  and the pass that used to invert 4 KiB byte by byte disappears. (A `C` the
  plane does contain is still inverts through the byte test.)

`tile.c` got one more of the same kind, for the lane-history family: a 32-byte
block of four groups whose mask word is zero consumes no units and, for
`LANE_PREVIOUS`, repeats the previous group's eight bytes, and for
`LANE_DELTA` is an arithmetic progression at one add per group. Taking such a
run as one 32-byte store instead of four blended 8-byte ones lifts `sawtooth`
(16-way transposed ramp, `LANE_DELTA`, 15 exceptions per tile) 1.83 -> 2.53
GB/s decode with every `clen` unchanged. `LANE_AVERAGE` has no closed form and
keeps the per-group path.

And `transform.c` had the same shape of waste on the 16-way transposed
plane: the 16x16 AVX2 transpose was called with two freshly assembled
sixteen-entry pointer arrays per 256-byte group, which cost a third of the
kernel. Both the gather and the inverse now hoist those bases out of the loop
and pass the group's byte offset instead, which is what lifts `sawtooth` to
2.76 and `sparse2` 5.14 -> 6.8 GB/s decode.

The decoder needs no new rules: the prefix equation, K == Nt for ALL_EXCEPTION
and the residual payload are validated by the checks every dense record already
goes through (`dense_check_mask`, `zpse_residual_decode`), so a hostile record
is rejected by the same statuses as before, and a byte string that is neither
emitted shape decodes exactly as it did. `tests/test_dense.c` drives the
emitter directly now: sixteen values must come out as one 2077-byte record with
TILE_CONSTANT / ALL_EXCEPTION / PALETTE_BITS and a prediction byte the plane
lacks, two values as one bit per byte (543 bytes), seventeen values as tiles,
and four tampered records (wrong K, a bitmap claimed without one, a residual
span that overruns, a truncated payload) that must be rejected rather than
decoded. All 768 cases of `tests/clen_corpus.c` come out byte-identical to the
previous commit -- none of those shapes has an alphabet that small -- and
parity, fuzz and lseed match across the default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` builds under both gcc and clang. That parity run is
also what proves the two decode rewrites above: the `-DZPSE_FORCE_SCALAR` build
runs the original scalar loops over the same corpus, and the two digests are
still identical (`8694766F1D93FB27` parity, `C1931AD9F42A615B` fuzz).

### Encoder: what the sampled decision still paid for

The encoder was 4-30x slower than the decoder on the compressible rows, and
none of that gap was in the tile search itself: measured with region timers
inside `zpse_segment_encode` and the block sweep (a throwaway instrumented
copy, 64 KiB blocks, per region), the three largest items were work the
encoder did and then threw away. Each is now removed, with the decision it
feeds left bit-for-bit as it was.

**The decision still staged every candidate.** `zpse_block_encode` picks the
surviving candidate from an exact price on 16 spread tiles, and it priced the
identity, DELTA8 and the 16-way transpose whether or not the answer was
already known. On a block whose identity plane predicts every sampled tile the
identity's price lands exactly on its own floor (`zpse_block_plane_floor`: the
segment envelope plus ten bytes per tile, the record of a tile with no
exceptions), and then nothing can beat it -- a price is that envelope plus, per
plane, at least that floor, so a single-segment candidate can only tie, and a
tie goes to the earlier list entry, which the identity always is. So the
pricing loop now stops there (`zpse_block_cand_floor`), and the remaining
candidates' transform staging -- a 64 KiB transpose on a constant block -- is
not paid at all: 14.6 us of the 19.2 that 64 KiB of zeros used to take.

**The palette form scanned the plane twice, scalar.** Pricing the dense palette
record needs the plane's alphabet (`seg_plane_alphabet`), and discovering it
was a load-modify-store of the same bitmap word per byte -- a dependent chain
through L1 that cost 46 us per 64 KiB on four-bit data, more than the palette
pack itself. The first 1024 bytes are still gathered scalar (they hold a
palette plane's whole alphabet several times over), and the rest of the plane
is only TESTED for membership: the set becomes two sixteen-entry nibble tables
and a byte is a member iff bit `(v >> 4) & 7` of `A[v & 15]` (or of `B`, above
high nibble 7), which is three `pshufb` lookups and a masked select per 32
bytes, independent of the alphabet size. A plane with more values still leaves
that pass the first time it meets one, so the early exit that keeps text and
noise cheap is unchanged. 46 -> ~4 us.

**The palette packer rediscovered its palette per record.** `dense.c`'s
emitter packs one 4 KiB record at a time and each call scanned all 4096 bytes
to find the first-appearance palette. The plane-wide alphabet is already known
at that point, so the emitter now derives the residual alphabet for its `fill`
(the exclusion map is a function of one byte, so distinct values can only
merge) and hands it to `zpse_residual_encode_hint` as the certificate the tile
path already uses. The packer walks only as far as the last new value -- a few
dozen bytes -- and the payload is unchanged, because the hint is verified
before use (the prefix must hold every value of the certificate, at which point
its palette order and size are the ones the exhaustive scan produces). 41 ->
~10 us per block.

Paired binaries, `bench 300 5`, best of five rounds, all ten `clen` values
unchanged (248 / 2619 / 65568 / 3962 / 2625 / 248 / 2618 / 10656 / 33288 /
1255):

| row | encode before | after |
| --- | --- | --- |
| zeros | 3.4 GB/s | 12.2 |
| const5A | 3.4 | 11.8 |
| alt00FF | 2.3 | 4.5 |
| lowentropy | 0.33 | 0.58 |
| the other six | -- | 0.98-1.07x (no code of theirs changed) |

Mean encode across the ten rows 2.49 -> 4.49 GB/s. The `lowentropy` number in
the dense-record table above (0.33 GB/s) is superseded by this one; its two
additions there still stand, and the row is still the slowest because the tile
loop has to run until `pos` reaches the palette's price before the palette form
can be chosen -- 79 us of the 115 it now takes is tile search whose records are
thrown away, and only a change to that decision (or a cheaper classify+pack)
can remove it. Verified: `tests/run.sh` ALL SUITES PASS under gcc and clang,
`DENSE: cases=6140 failures=0`, and the parity (`8694766F1D93FB27`) and fuzz
(`C1931AD9F42A615B`) digests identical to before the round in the default,
`-DZPSE_FORCE_SCALAR` and `-DZPSE_FORCE_TRIAL_SWAP` builds -- the
`FORCE_SCALAR` build is the reference the fast paths are diffed against, and
the alphabet pass was additionally checked against its scalar form on 4,000
randomized planes (alphabets of 1-17 values, values near 240-255, stragglers
past the head, both `limit` settings) with no divergence.

What is left on the encode side, measured rather than guessed: the per-tile
`classify` (74-150 ns per 256-byte tile, dense tiles the expensive ones) and
the nine-predictor prescreen (~15 us per 64 KiB, 30% of `sawtooth`'s tile
time). Both are per-256-byte framing costs, the same ones the dense-record form
was added to amortise, and they are now what dominates `sparse` (1.5 GB/s) and
`sawtooth` (1.1).

### Encoder: three decisions that were pricing the wrong thing

The three rounds before this one measured, in order: the palette form
replacing tile records that had already been written, the block decision
pricing sixteen tiles per candidate, and the tile path pricing a second
predictor that was one prescreen miss behind. This one is the decision layer
itself.

**The palette form was priced last.** A small-alphabet plane was priced by
writing tile records until they reached the palette's size; on 64 KiB of
four-bit data that meant classifying, mode-searching and packing 190 tiles
whose bytes the palette then threw away -- 110 us of the block's 114 us of
encode. The palette's price is exact and costs one alphabet scan, so once it
is known and the tiles have not caught up, eight spread tiles go through the
trial's own pipeline and the result is scaled up; if the palette is an eighth
smaller than that, it is emitted now and the loop returns. Otherwise the
tiles run exactly as before. `lowentropy` 0.57 -> 1.47 GB/s with all 768
`clen_corpus` cases byte-identical.

**A candidate was priced per tile, which cannot see a dense form.** A plane of
one byte value costs twelve bytes per 4 KiB as a dense record against ten
bytes per 256-byte tile, and a palette plane about four bits per byte, so a
transpose whose phases are constant -- alternating bytes, a spike every 256 --
looked like the worst candidate on the list. Alternating 0x00/0xFF compressed
to 2618 bytes that way; transposed, all sixteen phases are constant and it is
608. Two-byte spikes were ranked behind the identity: 3126 bytes instead of
1255. `zpse_segment_probe_cost` now owns a plane's price -- the sampled tile
records, or the dense constant form when the plane is constant (tested only
when the tiles priced at their all-hit floor, so the scan is free on every
other plane) -- and the block decision prices the identity first, since its
envelope is the smallest and a hit on its floor settles the winner.

**The sample was priced on more tiles than it needed, and on the wrong
ones.** Sixteen tiles per candidate decided the same corpus cases as four and
lost 34 KB on the fuzz suite, because a coarse estimate ties more often and a
tie goes to the identity, the better prior where the sample is noisiest. A
transpose's sixteen planes are sixteen phases of the same data, so pricing a
tile in each measured sixteen tiles to buy four tiles' worth of information
(12.5 us of a 27 us block); the budget now covers four planes and every
unpriced plane still gets the constant test, which is the one form that can
hide in a phase the sample skipped. The tile path likewise stopped pricing a
second predictor unless it ties the best one's prescreen miss count exactly:
an exact tie is the only case where the second record can be smaller, and
`sparse` was paying for two of everything (1.03 -> 1.63 GB/s).

The prescreen proxy stays in front of all of this. It is broad and cheap
(every other tile of the first 16 KiB, no packing) and the exact sample is
narrow, so on a block whose structure changes inside that window the proxy is
what ranks it correctly: removing it made the fuzz suite's mixed-region shapes
400 KB worse.

Paired measurements, `bench 300 5`, best of five, this machine (memcpy 61 GB/s,
memset 122, CRC32-C 28.5 during the runs):

| row | encode before these three rounds | now | encode bytes |
| --- | --- | --- | --- |
| zeros | 12.2 GB/s | 14.5 | 248 |
| const5A | 11.8 | 14.3 | 248 |
| incr | 5.1 | 5.2 | 2619 |
| alt00FF | 4.5 | 3.5 | 2618 -> 608 |
| random | 3.0 | 3.1 | 65568 |
| strided | 3.2 | 3.1 | 2625 |
| lowentropy | 0.57 | 2.1 | 33288 |
| sparse2 | 2.8 | 2.8 | 1255 |
| sparse | 0.98 | 1.5 | 3962 |
| sawtooth | 1.1 | 1.1 | 10656 -> 7584 |
| mean | 4.53 | 5.12 | |

`alt00FF` trades encode rate for a stream 4.3x smaller, which is the right way
round for a compressor, and its decode follows the same stream.

Verified on the final source: `tests/run.sh` ALL SUITES PASS under gcc and
clang, `DENSE: cases=6140 failures=0`, and the parity, fuzz and lseed digests
identical across the default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` builds (that build pair is what proves the output is
independent of the fast paths and of the attempt order). Ratio: the 768-case
`clen_corpus` loses 18,864 bytes on 27 cases and gains 300 bytes on 6 (three
of them the alternation shape at 12 KiB, where 538 -> 608 follows from the
transpose now being priced as what it writes); the 2,718-case fuzz corpus is
283 KB smaller than before the three rounds. Nothing here changes the wire
format.

### Where a `sawtooth` tile spends its cycles, and two changes that measured worse

`sawtooth` (1.1 GB/s) and `sparse` (1.5) are the slow encode rows, and both
are dominated by per-256-byte-tile work rather than by the block decision.
Attribution matters here because three plausible "make the classifier faster"
ideas turned out to be worth nothing or worse once measured. Numbers are
`__rdtsc` reference cycles per 64 KiB block, `best of 300`, on the development
machine, with the tile loop instrumented per stage (the rdtsc pair itself
inflates each stage by ~40 cycles per call, so read the shares, not the
absolute values):

| stage | calls/block | cycles/call | share of block |
| --- | --- | --- | --- |
| `seg_prescreen` (nine predictors) | 365 | ~143 | 18% |
| `zpse_tile_classify` | 256 | ~342 | 30% |
| `zpse_tile_pack` | 256 | ~170 | 15% |
| `zpse_tile_choose_modes` | 256 | ~40 | 3% |
| block proxy (128 prescreens over 16 KiB) | 1 | — | 8% |
| rest (transform staging of the winning 16-plane transpose, CRC32-C, copies) | — | — | ~26% |

The 365 prescreen calls are the 256 tiles plus the 109 the pricing phase
spends; the exact-price pass is skipped on this row because only one
candidate survives the proxy prune. Two of the three calls per tile
(`classify`, `pack`) are irreducible short of a different record form: the
encoder must derive the mask and the payloads for the predictor the prescreen
picked.

What was tried and measured:

- **Rank fewer than nine predictors per tile.** A one-pass "is this tile one
  repeated byte" test that settles the winner from `a[0]` was implemented,
  proved equivalent (same predictor, same k) and measured: no change on any of
  the sixteen corpus shapes or the ten bench rows, within noise. The reason is
  that uniform tiles already skip the prescreen — they are all-hit tiles, and
  the all-hit run path proves those three cases from the tile's own bytes
  (`seg_all_hit_cheap_ok`) before asking. Reverted rather than kept as dead
  code. The prescreen's 18% is the *non-uniform* tiles, where all nine counts
  are still candidates for the minimum.
- **Coarser proxy sampling** (every 4th tile instead of every 2nd, halving the
  18% pricing share): `test_sizes.c` failed on 21 of 768 cases at +34,314
  bytes, because a 16 KiB sample ranked the transpose against the identity
  wrongly on mixed-region shapes. Reverted; the 2-tile stride and the 16 KiB
  window are load-bearing.
- **Lowering the dense-classify threshold** from 32 exceptions to 16, to send
  `sawtooth`'s ~16-exception tiles through the vectorised path: `sparse` fell
  from 1.56 to 1.19 GB/s and nothing else moved, because that path's fixed
  cost (a 256-byte presence array plus its fold) dominates below ~32. The
  scalar per-exception walk is the right side of that crossover.

The remaining headroom is the per-exception walk itself (about 25 cycles per
exception, including a data-dependent branch per mask word) and the transform
staging, which is a full pass over the block that no pricing change removes.

### Run continuation without the prescreen

The all-hit run path (a stretch of tiles whose prediction is exact, 10 bytes
per tile) could only prove a tile that is *one repeated byte*. Every periodic
plane therefore failed that proof on every tile and paid the full
nine-predictor prescreen per 256 bytes: `strided` (period-4) paid 365 of them
per 64 KiB block, and `incr`'s DELTA8 plane — constant except for its first
tile, where the seed lands — paid 111.

A run may continue with the same predictor when two things are proved:

1. that predictor misses nothing in this tile, and
2. every predictor *before* it in the prescreen's scan order misses somewhere
   — otherwise one of them would win the tile instead, because the winner is
   the first predictor with the strictly smallest count.

(1) is one prediction pass over the tile (a shifted compare for LANE_DELTA,
LANE_AVERAGE and STRIDE_2/3/4/8). (2) needs one witness each, and a witness
is usually the first few bytes, so the search is bounded to eight positions
per predictor; when it cannot find one that fast the prescreen is asked,
exactly as before. Being wrong towards "ask" costs time, never bytes: the
outcome is the same predictor the prescreen would name, with the same
k == 0, so the record written is identical.

That claim is verified rather than asserted: every parity, fuzz and lseed
digest is unchanged (`FC79D5E365BA40E1`, `DFB9C23F6259A686`, `LSEED`), and
all 768 ratio-corpus sizes are unchanged, in the default, `-DZPSE_FORCE_SCALAR`
and `-DZPSE_FORCE_TRIAL_SWAP` builds.

Deterministic effect, from counters rather than a clock — prescreen calls per
64 KiB block:

| row | before | after |
| --- | --- | --- |
| strided | 365 | **111** |
| incr | 111 | **80** |
| sparse2 | 110 | 108 |
| zeros, const5A, alt00FF | 7, 7, 30 | unchanged (already all-hit or uniform) |
| sparse, sawtooth, lowentropy, random | 365, 365, 145, 152 | unchanged (no run tiles) |

Each of those prescreens is a nine-predictor pass over 256 bytes replaced by
a one-predictor pass plus a couple of witness scans, so the work removed is
real on the rows that take it. The measured wall-clock effect could not be
pinned down better than that on this session's machine, which was at 90-100%
external load throughout: process CPU time (30,000 blocks, min of three runs)
gave `strided` -10%, `sparse` -3%, and `incr`/`sparse2` within ±7% with the
sign flipping between runs — inside the noise band of a saturated host. The
one window measured on an idle machine showed `strided` 3.12 -> 3.57 GB/s and
`incr` 5.22 -> 5.41 GB/s.

### Encoding a repeated tile without encoding it

The per-tile pipeline above is what a `sawtooth` block spends almost all of
its time in, and the numbers said it could not be shrunk much further per
tile. It can be skipped instead. A tile record is a pure function of four
things: the tile's bytes, the eight plane bytes before it, its length and the
segment literal. Every tile the encoder writes is classified with
`tile_reset == 1` (nothing carries across a boundary), and `seg_pos` reaches
nothing else -- the STRIDE predictors are its only readers, and they read
exactly those eight bytes. So when a tile repeats an earlier one with the same
head, its record *is* that record, and copying it writes byte for byte what
the pipeline would have produced.

That matters for periodic data. A `sawtooth` block's winning candidate puts
one period of the block in each 256-byte tile, so 224 of its 256 tiles are
repeats of an earlier tile; `sparse`'s period is 97 tiles, which a 128-entry
table catches part of.

The cache is 128 direct-mapped slots (12 bytes each, zeroed once per segment,
under 3 cycles per tile on a 4 KiB plane), keyed by four words of the tile
(offsets 0, 64, 128, 192, XORed into one product) and confirmed by a 264-byte
compare of the tile and its head before anything is copied. The key is only a
hint, so a collision costs a compare and can never change a byte. Tile 0 is
neither a source nor a target: its STRIDE reads fall back to the literal, and
it has no head. The palette form is the one piece of tile-loop state a copy
would step over, so a copy is only taken when that form could not have been
priced at this tile either (already priced, or a record too small to trigger
it) and never while it is still in play.

Deterministic effect, from counters per 64 KiB block:

| row | prescreens | classifies | tiles served from the cache |
| --- | --- | --- | --- |
| sawtooth | 365 -> **141** | 256 -> **32** | **224 / 256** |
| sparse | 365 -> 341 | 256 -> 232 | 24 / 256 |
| incr, strided, alt00FF, sparse2 | 111, 111, 30, 110 | 1, 1, 2, 16 | 0 (no general-path tiles) |
| zeros, const5A | 7, 7 | 0, 0 | 0 |
| random, lowentropy | 152, 145 | 28, 21 | 0 (nothing repeats) |

Thread-cycle time per 64 KiB block (min of 3, alternating build order, on the
host this was developed on -- see the note at the end of "Run continuation"
about its load): `sawtooth` 97.7 -> 34.3 us (2.85x, and every post-change
measurement was below every pre-change one, 24.5-39.9 against 88.6-109.7),
`sparse` 73.8 -> 63.1 us. The rows the table above says do no new work moved
by less than the noise band of this host (zeroes, `const5A` and `strided`
within it in both directions); the hash they pay is about eight cycles a tile,
and the only other new cost is the per-segment clear.

Two rejected variants, both measured:

- **Keying on the tile's first eight bytes.** `sparse` lost 25%, because its
tiles are mostly zeroes and every one of them hashed to the same slot, so the
confirming compare ran all the way to the spike that distinguishes them. The
four-word key costs two more loads and removes that. The first version of the
model also predicted 150 `sawtooth` hits keyed on the leading bytes where the
truth is 0: those were false keys, caught by the compare (a ramp repeats its
opening bytes every 16 tiles, its whole tile every 256).
- **16 slots.** `sparse` kept 8 of its 24 hits; 128 slots is one per 2 KiB of
the 64 KiB block, which is what a block of fixed-size records needs.

Every digest is unchanged (`FC79D5E365BA40E1`, `DFB9C23F6259A686`, `LSEED`)
and all 768 ratio-corpus sizes are unchanged, in the default,
`-DZPSE_FORCE_SCALAR` and `-DZPSE_FORCE_TRIAL_SWAP` builds -- the shortcut
cannot change what the encoder means to write, only how it gets there.

### Decoding a repeated tile without decoding it

The same argument runs the other way. A reset tile's reconstruction is a
function of its record and, for the STRIDE family, of the eight output bytes
before it -- a reset tile starts from the segment literal, and CONST's byte,
the lane histories and those eight bytes are everything the predictors here
consult. So a record that repeats the previous tile's, at a position whose
preceding eight output bytes repeat too, decodes to the bytes already written
one tile back, and a 256-byte copy stands in for the whole tile. A full
tile's epilogue is the O(8) history update, which for `nt == 256` reads the
tile's own bytes, so the lane state after the previous tile is the state after
this one and nothing needs repairing.

The test is one 8-byte key compare -- the record's leading word, the same
filter the encoder's cache uses -- and only a tile that matches it pays the
full `memcmp`. ORDER1/ORDER2 are excluded (stateful tables) and so are tiles
that are not full length. A hostile stream cannot notice: the bytes compared
are at the same length and already passed validation at that position, so any
rejection they earn has already happened.

On `sawtooth` (224 of 256 tiles) decode goes **21.5-23.7 -> 12.2 us per
64 KiB block, 1.8-1.9x**, and part of that is the 29% smaller stream this
round's mask change produces. Every other row stayed within +-2% across five
alternating rounds, in both directions, so no claim is made for them.

### A constant tile with few exceptions, filled instead of predicted

A CONST tile's prediction does not depend on any byte the tile writes, so with
`k` exceptions the tile is the constant everywhere else: `memset` it and write
the `k` exceptions in mask order (ascending, which is the order the residual
units were packed in). Under 24 exceptions that is a fill plus a handful of
stores against the fixed ~150 cycles of the 32-wide CONST kernel, so the
crossover is where a few cycles per exception overtakes the kernel. The
window is small: `sparse` is the row with CONST tiles and a small `k`, and its
decode delta sits in the noise band above.

### Mask mode: RUNS, and what it is worth

`zpse_tile_choose_modes` minimised the mask payload over ALL_HIT,
SPARSE_U8 and RAW_BITMAP, and the residual payload over all five residual
modes. RUNS was never chosen -- and it is the mode that wins when a tile's
exceptions come in a few long runs rather than scattered: one byte per run,
against one byte per exception (SPARSE_U8) or the fixed `ceil(nt/8)` bitmap.
The run count is eight word operations, not a 256-bit scan: with `prev` the
bit below the word, bit `p` of `w ^ (w << 1 | prev)` is set exactly when
position `p` differs from `p - 1`, so `runs == transitions + 1` and the walk
stops as soon as it cannot beat the incumbent.

Ratio effect on the 768-case corpus: **249 cases shrink, none grow, -160,862
bytes (-3.95%)**, by shape:

| shape | before | after | delta |
| --- | --- | --- | --- |
| sawtooth (`i*7 + i>>4`) | 156,858 | 118,293 | **-24.6%** |
| structured 2-byte records | 207,252 | 189,132 | -8.7% |
| noise then constant | 540,344 | 496,397 | -8.1% |
| constant then noise | 604,880 | 566,449 | -6.4% |
| palette, shuffled | 286,371 | 264,620 | -7.6% |
| the other eleven shapes | -- | -- | 0.0% |

The `sawtooth` number is the one that explains why the mode was missing: its
winning candidate leaves ~15 exceptions per tile in a few runs, so the mask
cost 15 bytes where 3-5 will do. Encode and decode pay nothing measurable for
this: the run encoder is the transition walk above (byte-for-byte the payload
the old bit scan produced -- 21 corpus cases with a partial last tile caught
the difference while it was still being written, which is how the `nt` edge
bit got masked), the choice is eight popcounts on tiles that could win, and a
RUNS decode is one byte loop, the same per-exception walk SPARSE_U8 always did.

A cheaper variant was measured and rejected: gating RUNS to `k <= 3*nt/4`
removed the encode cost entirely on the `random` row (whose records are
discarded anyway, since the block is stored raw) but gave back 100 KB of the
ratio win, because the shapes above are mostly-exception tiles. Making the run
encoder a transition walk was the better trade: with it, encode on all ten
rows is within +-4% of where it was, and `sawtooth` is 3.2% faster on a stream
29% smaller.

Two guards for the new encoder path: `tests/test_units.c` now round-trips
`zpse_mask_encode` -> `zpse_mask_decode` for modes 1-4 over 4,000 masks of
every tile length (dense, sparse, one run, alternating -- so exact multiples of
32, partial last words and every run length are covered), and the tile
roundtrip loop now includes mask modes 3 and 4. Checked to have teeth: a
one-run corruption injected into the RUNS encoder fails that test with 1,017
bad masks.

Verified: `tests/run.sh` ALL SUITES PASS under gcc and clang and
`tests/run.ps1` under PowerShell. The parity and fuzz digests move with the
stream this round shrinks (`FC79D5E365BA40E1` -> `7FC51D6C47053813`,
`DFB9C23F6259A686` -> `A23B414B92409BD6`), `LSEED` is unchanged, and all
three are still identical across the default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` builds -- which is what proves the new mode choice
is a decision the scalar and vector paths and the attempt order agree on. The
768 pinned
sizes in `tests/test_sizes.c` are the new ones, so the guard now sits at the
improved baseline (`grew=0 shrank=0 delta=+0`). The wire format is unchanged:
RUNS was already a legal mask mode that the decoder has always implemented,
and the encoder simply never wrote it.

### Not pricing nine predictors when the first one is nearly exact

`sparse` was the slowest encode row and the reason was the prescreen: nine
predictors evaluated over every 256-byte tile, ~30 ns of a tile that costs
~170 ns to encode. Its tiles are 250-odd zeroes plus a spike or two, and for
that shape there is a shortcut. CONST_ZERO's miss count *is* the number of
nonzero bytes, so one 256-byte compare pass answers it, and CONST_ZERO is
the first predictor in `seg_screen_pred[]` -- it wins every tie, so the other
eight can only beat it by a strictly lower count. At three misses or fewer
the record they would buy differs by a byte or two (a predictor that also
predicts one spike), so the tile is classified against CONST_ZERO alone.

Armed by the previous tile, not measured on every tile: `zero_cheap` says the
last tile's prescreen found CONST_ZERO within the threshold, so a plane whose
tiles are mostly not near-zero -- random, alternation, a ramp -- never pays
for the zero count at all. Measured unconditionally it cost 1.5-5.8% on every
row it does not help; armed, those rows stay inside the host's +-3.5% band
and `sparse` keeps the whole win.

| row | before | after |
| --- | --- | --- |
| sparse | 42.7 us | **33.6 us (-21%)** |
| the other nine rows | -- | within +-3.5%, both directions |
| total of the ten | 206.8 us | 196.2 us (-5.1%) |

Ratio is untouched: all 768 pinned corpus sizes come out byte-identical
(`delta=+0`), as does the parity corpus. The fuzz corpus's digest moves, which
is the honest statement of what the shortcut is -- a heuristic, not a proof:
on a tile where CONST_ZERO is within three misses of the best, the encoder
now takes CONST_ZERO. The fuzz frames are random blocks that no pinned size
covers; the 768 structured cases are the guard that says nothing grew.

One methodological note, because it cost a wasted round: the two binaries must
alternate within a round (`B N N B`), not run as `B N` each time. Always
measuring the candidate second made every change look 3-6% slower than it was
-- the host drifts over a session, and the second binary always got the later,
slower slot. The first version of this change looked like a 4.7% regression
that way, and like a 21% win with the order alternated.

### Palette widths below four bits: a two-valued plane decodes 1.4 -> 14 GB/s

`residual.c` had a vector path for four-bit palette indices and three-bit
tiles, and nothing for one or two bits per value, so those widths fell through
to the generic sixteen-value loop: extract 16*b bits from a shifting window,
validate, `pshufb`, store. Measured on one 64 KiB plane, that loop costs
**2.3 cycles per value at one bit and 1.8 at two bits**, against 0.15 at four
bits -- a 1-bit palette was the slowest shape in the file by 13x, and it is the
shape a two-valued plane produces (a binary mask, a near-binary sensor, the
residual of data that is one bit deep).

Both widths now have their own block loop. The order they have to undo is the
one `residual_get_bits` writes: value `8i + j` lives in bit `j` of payload byte
`i`, so the eight bit planes of an input byte become eight 16-byte vectors
(`cmp` against a single-bit constant, then `pal[0] ^ (mask & (pal[0] ^
pal[1]))`) and the usual 2-, 4- and 8-way unpack ladder interleaves them back
into byte order: **128 values per 64 operations, 2.3 -> 0.12 cycles per
value**. Two-bit indices tile a byte four ways instead: three shifts and masks
give four index vectors, one validation, four lookups, and an 8- then 16-bit
unpack puts 64 values back in order. Validation stays where it was -- before
the first store, so a rejected payload writes nothing and the generic loop
re-decides from the same position and reports at the same byte, which is what
keeps a hostile record's status identical.

The encode side of the same widths had already fallen behind: the packer's
vector path covered four-bit palettes only, so one- and two-bit data ran the
eight-value scalar group loop at 3.7 cycles per value against 2.1. Both widths
now pack vector-to-vector at the same 2.1.

End to end, `zpse_compress_block` / `zpse_decompress_block` on 64 KiB planes of
`n` distinct uniformly random byte values (so the dense palette record is the
form chosen at 2 and at 3-4), paired binaries, best of three rounds, all `clen`
identical:

| distinct values | width | encode GB/s | decode GB/s |
| --- | --- | --- | --- |
| 2 | 1 bit | 1.16 -> 2.6 (2.2x) | 1.45 -> 14.2 (9.8x) |
| 3 | 2 bits | 0.38 -> 0.55 | 1.85 -> 13.2 (7.1x) |
| 4 | 2 bits | 1.14 -> 1.97 (1.7x) | 1.80 -> 13.3 (7.4x) |
| 9, 16 | 4 bits | 1.9, 1.9 (unchanged) | 10-13 (unchanged) |
| 17 | tiles | 0.48 (unchanged) | 0.9 (unchanged) |

Decode now costs the same at every palette width, which is the point: the
width is a property of the data, not a tax on it. The ten bench rows are
unaffected -- `alt00FF` is 608 bytes as tiles, not a palette record, so it
never enters these loops, and every row stayed within noise in both
directions.

No byte moves: the ten `clen` values, all 768 `tests/clen_corpus` cases and the
parity / fuzz / lseed digests across the default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` builds are identical to the previous commit, because
the packers write the same bits and the decoders only change how fast the same
bytes are expanded. The scalar build is the reference the vector paths are
checked against: `tests/test_units.c`'s palette roundtrip now runs m = 2..16 at
25 lengths from 1 to 1024, chosen so 64/65, 127/128/129, 255/256/259, 511/512
and 1023/1024 each land a whole number of the new blocks plus a tail; a
differential run of all 16 widths against all 707 lengths from 1 to 8192, built
both ways, produced the same digest (`7FC8C571E3168DC9`) and the same bytes.

### Classifying a full tile where it lies, and three smaller fixed costs

`tile_classify_avx2` compares a tile against its prediction in a 304-byte
staging buffer: 16 bytes of head (the entry lane state for LANEHIST, the
preceding plane bytes for STRIDE), the tile's own 256 bytes, and 32 bytes of
zero pad so a partial last chunk can be read as one full 32-byte load and
masked. The copy is there for the two reads a 32-byte vector loop cannot make
from the input -- the head, and the over-read past a short tile.

A CONST tile needs neither. Its prediction is a single byte, so the comparison
never reads the head, and a full 256-byte tile is eight whole chunks with no
tail at all: every load stays inside `[0, nt)`. Such a tile is now compared
straight from the input, and the exception walk that follows reads its bytes
there too. Measured on one 256-byte tile with three spikes, 200,000 calls per
build: **classify 48.1 -> 37.5 ns (-22%)**. The segment's last tile (the only
short one) and the LANEHIST/STRIDE families keep the staging path, because
their predictions genuinely read the 16 bytes before the tile; the dense
classifier takes a tile pointer rather than the buffer, so the same code serves
both, with the head behind the tile either way.

Two more fixed costs went with it. `tile_history_fixup` derives lane l's last
position as `l + ((nt-1-l)/8)*8`, which for `nt == 256` is `248 + l` for every
lane and never reaches below 8 -- so a full tile's epilogue is two 8-byte loads
and two stores instead of eight divides and compares, on the way in (every
tile) and on the way out (every decoded tile).

And the transform sweep no longer prices a candidate it can prove the prune
will drop: the proxy price of each candidate is compared against a 6% margin
around the running best, and a candidate whose own floor (`zpse_block_floor`,
arithmetic on its transform and segment shape, no plane touched) already
exceeds that margin cannot be kept whatever its proxy would have said. The
identity is priced first, so for a block whose identity plane is already
near-perfect -- a ramp, a constant, an all-hit plane -- the transpose
candidate's 16 KiB staging disappears from the encode. The floor is taken over
the proxy's 16 KiB window, not the block: it is being compared with a proxy
price, which is what makes it a bound and not an estimate.

Paired binaries, alternating `B N N B` within a round, best of four rounds,
ten 64 KiB bench rows, every `clen` identical:

| row | encode | decode |
| --- | --- | --- |
| sparse | 1.98 -> **2.12 GB/s (+7.1%)** | 4.96 -> **5.52 GB/s (+11.3%)** |
| sawtooth | 2.47 -> 2.50 | 5.47 -> 5.73 (+4.8%) |
| strided | 3.74 -> 3.79 | 10.07 -> 10.35 (+2.8%) |
| incr | 5.42 -> 5.50 | 10.05 -> 10.27 (+2.2%) |
| alt00FF | 3.61 -> 3.62 | 7.48 -> 7.63 (+2.0%) |
| sparse2 | 2.87 -> 2.90 | 6.77 -> 6.82 |
| random, lowentropy, zeros, const5A | within +-3% | within +-3% |
| total of the ten | 195.4 -> 192.9 us (-1.3%) | 70.8 -> 68.3 us (-3.5%) |

`sparse` is the row the change is aimed at -- its 232 CONST tiles per block are
the ones that stop copying themselves -- and the decode side lands on every row
that decodes general tiles at all. No byte moves: the ten `clen` values, all
768 `tests/clen_corpus` cases (`delta=+0`), and the parity / fuzz / lseed
digests across the default, `-DZPSE_FORCE_SCALAR` and
`-DZPSE_FORCE_TRIAL_SWAP` builds are identical to the previous commit, and every
suite passes under gcc, clang and `tests/run.ps1`. Dropping a candidate before
its proxy is a bound, not a heuristic: a proxy is never below its floor, so the
survivor list is the same one the prune would have produced, which is why the
pinned corpus does not move by a byte.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
