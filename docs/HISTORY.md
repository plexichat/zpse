# History

This is the design history of zpse: why the codec looks the way it does, which
ideas survived measurement, and which were removed again. It is not a release
log. The README describes the codec as it is; this file describes how it got
there.

The one rule that shaped everything else arrived with the first commit: a block
is a pure function of its own bytes. No allocation, no I/O, no global state, no
cross-block window, and no dependence on the machine, the compiler, the SIMD
level or the thread count. Every optimisation below is therefore an encoding
decision or a faster way to make the same decision -- never a change to what a
stream means.

## The starting point

The first checkpoint was already a complete codec: fixed 64 KiB blocks built
from 256-byte tiles, twelve predictors, seven mask modes and five residual
modes, a per-tile priced search, an optional width-2/4/8/16 byte transpose, a
frame layer, and a parity suite that compared every SIMD path against its
scalar twin. That shape never changed. What changed was how cheaply the
encoder could choose a tile form and how fast the decoder could rebuild one.

## Making decode fast

The decode work came first, because the tile model's inverse is where the time
goes. The order was:

* One AVX2 pass that counts the misses of all nine cheap predictors at once,
  replacing nine scalar classify passes.
* A single-pass classifier that fills the mask and residual arrays while it
  walks the tile, instead of staging predicted bytes first.
* A 32-wide stride reconstruction, with the stride-family mask walked by
  `ctz` so a sparse tile pays per exception rather than per position.
* Wide residual and literal copies, and the transform inverses rebuilt around
  registers rather than through a carry array in memory.
* Delta-transpose merges that keep each plane's carry in a vector instead of
  extracting it to a general register every 32 bytes. A `testz` then skips the
  chain for the all-zero planes that wide integer columns produce.
* A three-way interleaved CRC32-C, and an opt-in payload checksum rather than
  a default one, because a checksum pass is a large fraction of a fast decode.

The scalar and SIMD paths stayed bit-identical throughout; that is what the
parity and fuzz builds are for.

## Paying for decisions

The encoder's candidate search was always the expensive half. Its evolution
was a sequence of ways to avoid encoding a block twice:

* Price candidates on windows rather than on the whole block, then encode only
  the survivors.
* Stop the sweep as soon as nothing left can beat the running winner.
* Prescreen predictors by miss count, keep a runner-up only on an exact tie,
  and rank the scan without letting the ranking affect the bytes.
* Prune a field of tied prices to its best few survivors, guarded by the cases
  where a tied price is known not to describe the block.
* Abandon a trial that has already run into the raw envelope, and abandon a
  noisy block from its front sample when the tail says the same.
* Let cheap levels act on the price alone, with one, two or three trials, and
  let the deeper levels pay for exact pricing and the match finder.

Each of those was measured against the alternative, and several were kept only
for a limited set of block sizes or levels because the trade did not travel.

## Forms the tile model cannot express

Three forms were added when the tile model was structurally the wrong answer:

* A **dense** form for a plane that is one value, and a palette record for a
  plane with a tiny alphabet.
* A **seven-bit ASCII** form for text, which stores eight characters in seven
  bytes and is decided before the sweep.
* **Intra-block LZ**, a literal-and-command stream whose distances stay inside
  the block. It is offered as one more candidate at the deeper levels and is
  the difference between raw and compression on high-cardinality text.
* **LZ over the tile body**, which compresses the body the tile path would
  have written and then reconstructs it through the ordinary tile decoder.
* A fixed-distance **fast LZ** form (LZ4's shape) for the cases where the
  variable-length distance costs more than it saves.
* A **hexadecimal LZ** form for page-sized blocks, which packs common
  literal/length pairs into a nibble and shortens fresh distances.

The LZ work forced a second rule into the design: a form is kept only when it
is strictly smaller than everything already tried, so a level never pays for a
candidate in ratio. The price is paid in decode speed, which is why the effort
levels expose how small a sample win is worth a full match-finder pass.

## Effort levels

The first API exposed a handful of presets. Measurement made it clear that the
useful axis was not a single "quality" number but a ladder of decisions:

1. one trial on a cheap prescreen, no match finder;
2. the full prescreen and two trials;
3. three trials;
4. four trials;
5. exact pricing, a guarded field cut, and the LZ form on a conservative
   sample win;
6-9. progressively looser LZ margins, smaller acceleration factors and a
   wider candidate field.

The ladder became nine levels, with level 6 as the default. The suite pins
that no level writes a block larger than the level below it. Levels 1-4 never
run the match finder; levels 5-9 differ mainly in how much evidence they need
before paying for it, and in the decode bill they accept.

## Frames, seeking and threads

A block stayed the unit of random access. A frame is a header plus blocks; an
optional `FrameIndex` records one offset/size pair per block so `zpse_seek_frame`
can decode block *k* directly. A streaming writer can build the same index from
the offsets it recorded while writing.

Parallel encoding was possible from the start because blocks are independent;
the only serial part was the output layout, which is arithmetic. Workers write
into reserved slots and one pass compacts them. The first ring implementation
was wrong: it assigned slots dynamically, so the byte stream depended on
timing. It passed at two and four workers and failed at six. The fix was a
static slot assignment, and the test now requires identical frames at every
worker count and repeated runs. Threading remains compiled out unless
`ZPSE_MT` is requested.

## The page size

zram hands a codec one 4 KiB page at a time, and a page is a different problem
from a 64 KiB block: the sweep has one tile per plane, structural bytes are
charged against 4096 rather than 65536, and the average match is short. The
page work therefore changed decisions rather than the format:

* A 16-byte header for pages, with the version and profile fields fixed by the
  format and the checksum field dropped rather than spent.
* A lazy step that measures the current match and the next position instead of
  asking only whether an eight-byte match starts one byte on.
* `ZPSE_ENC_PAGE_FAST`, which cuts the candidate list to the identity on the
  cheap rungs and removes the pricing pass, the sample and the trial cap.
* A candidate cap on the levels that price every plane, chosen because the
  cap's ratio cost crosses the one-point line only there.
* Shorter ASCII and match-finder samples at page size, and pricing a page
  whole once every plane is priced.
* The noise abort extended to pages, but only under the fast flag, because a
  page has no sweep behind its trial to repair a wrong verdict.
* The hexadecimal LZ form from level 7 up, where distances above 4320 are
  rare enough that its shorter distance codes win.

Many page ideas were measured and refused: a middle gate window, a larger gate
sample, four page tiles, a third flat trial, extra noise samples, delta coding
of explicit distances, and a per-page dictionary. Each is recorded in the
README only where it still explains a shipped choice; the numbers live with
the harness that produced them.

## What measurement changed

The project converged on a small set of rules:

* An encoder-side change must leave the compressed sizes identical, or it is
  not a speed change.
* Every kernel needs a differential test over every length and row count it
  can see, compared against the scalar build.
* A ratio change must reproduce twice with a stable sign per case.
* A rate is only trusted when two binaries alternate over the same rows; a
  single before/after pair is not evidence.
* A change that is below the one-point ratio line on a corpus, or inside the
  host's rate spread, is not kept just because it is clever.

The result is a codec with a narrow promise: deterministic bytes, no
allocation, predictable decode, and a documented place on the ratio/rate
frontier rather than a claim to beat general-purpose compressors everywhere.
