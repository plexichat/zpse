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
| `test_parity.c` | the full corpus plus four per-primitive sweeps (uniform, small-`nt`/mask-word boundary, stateful-table ids, and the state a classification leaves) |
| `test_fuzz.c` | a randomized differential corpus: every size 1 to 64, every tile and segment boundary and its neighbours up to 64 KiB, 60 pseudo-random sizes, 18 shapes each (2718 blocks) plus 6 frames, each block round-tripped and its size, compressed length and compressed bytes folded into one digest |
| `test_lseed.c` | hand-built foreign segments whose ORDER1/ORDER2 tiles deliberately do NOT reset (non-resetting first tile, chains, mixed ORDER/STRIDE), each decoded by the library and by a reference in the file that hardcodes the documented segment-start state — the shape the lane-state seeding above exists for, and one this encoder cannot produce |
| `test_dense.c` | dense segment records (SPEC Sec. 6.3): 6042 records built by a reference writer over every predictor family, mask form and residual mode at nine plane sizes, plus hand-computed lane-history predictions, plus 15 malformed shapes (reserved bits/fields/modes, `ORDER1`, wrong spans, pad bits, non-ascending positions, trailing bytes) that must be rejected and never decoded |

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
dense records ok   (DENSE: cases=6042 failures=0)
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
| zeros | 0.040 | 2616 | 17.9 | 23.5 |
| incr | 0.040 | 2619 | 5.2 | 10.4 |
| random | 1.000 | 65568 | 3.1 | 20.9 |
| sparse | 0.060 | 3962 | 1.0 | 5.4 |
| strided | 0.040 | 2625 | 3.1 | 10.3 |
| const5A | 0.040 | 2616 | 17.3 | 23.9 |
| alt00FF | 0.040 | 2618 | 2.6 | 19.3 |
| sawtooth | 0.163 | 10656 | 1.1 | 1.9 |
| lowentropy | 0.687 | 45012 | 0.5 | 0.9 |
| sparse2 | 0.048 | 3126 | 1.1 | 5.9 |

The table is one session and the deltas below are paired measurements, so the
absolute columns move between machines and runs while the sizes and the ratios
do not.

### `lowentropy`: what the sweep costs, and why 1 GB/s is still out of reach

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
(SPEC Sec. 6.3) and decoded: `src/dense.c` handles one record per plane, with
its own mask cursor, all three lane-history predictors, the STRIDE family over
absolute positions and residual modes 0-3, under the same hostile-stream
validation as tiles, and `tests/test_dense.c` covers it. What is NOT in the
tree is the emitter: `zpse_segment_encode` still writes tile records, so the
format exists, round-trips and is tested, but no compressed stream uses it yet
and the speed above is unchanged. Emitting it requires a plane-level candidate
search that replaces the tile loop for homogeneous planes rather than running
after it, which is the next change and the one that has to be measured on
`lowentropy`.

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

## License

Apache License 2.0 — see [LICENSE](LICENSE).
