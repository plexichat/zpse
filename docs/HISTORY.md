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
* A page-sized hash table for `ZPSE_ENC_LZ_FAST`. The fast form's 2^16 entries
  were chosen on 64 KiB blocks, where they hold a load factor of one; on a
  page that cannot insert 4,092 positions the same table is 128 KiB of clear,
  or 32 bytes of store per input byte. Thirteen bits buys 9.6% of encode on
  the mixed page stream and 6.5% on the repeated-page stream for 0.073% of
  packed bytes there and at most 0.19% on any single class; twelve bits, the
  load factor the 64 KiB arm runs at, was slower and larger.

Many page ideas were measured and refused: a middle gate window, a larger gate
sample, four page tiles, a third flat trial, extra noise samples, delta coding
of explicit distances, and a per-page dictionary. Each is recorded in the
README only where it still explains a shipped choice; the numbers live with
the harness that produced them.

## The fast LZ decoder, and where its time actually went

`LZ_FAST` is the form the page table quotes first, and on prose it decoded
30-45% slower than lz4 per 64 KiB block while issuing 5-15% *fewer* commands --
about 11.5 cycles per command against 7.5. Tile blocks on the same file decode
at 5 782 MB/s against 2 115 for the command stream, so the gap was in the shape
of the commands, not in the frame around them. Three attempts followed, and
only the last one paid.

* The distance's kind was a select chain (`c0 < 0xFC`), on the argument that
  its answer is data and therefore mispredicts. A census said otherwise: a
  repeat code is 0-26% of commands over the twelve Silesia files, under 10% on
  nine of them and at most 5% on eight, so the branch is almost never taken
  while the select chain ran on every command. Making it a branch measured
  2 080 -> 2 106 MB/s on dickens, 2 099 -> 2 162 on mozilla, 4 346 -> 4 521 on
  x-ray, and flat on nci and xml.

* The next plan was the branch count itself: fold the payload and output room
  tests into one budget per run, merge the two nibble tests into one with
  `(((tok & 15) + 1) | ((tok >> 4) + 1)) & 16`, drop the match-length tests the
  region's own restriction makes dead, merge "zero distance" and "distance
  past the output" into one unsigned compare, and hand every escape to a
  single caretaker -- five branches per command. It was 5-14% slower: nci
  -14%, xml -13%, x-ray -11%, mozilla -5%, dickens +4%. Keeping the loop and
  taking only the tests that could go without restructuring it was still 1-3%
  slower. The decoder is not front-end bound, and the branch count was the
  wrong metric for it.

* What remained was the shape of the stores. The region wrote a match with one
  32-byte store (a 16-byte one at nearer distances) and its literals with one
  16-byte store, and a few bytes later another store landed in the same cache
  line -- the pattern store-to-load forwarding handles worst. Eight-byte steps
  instead, two for an ordinary match and a third only for the seventeen and
  eighteen byte matches the region can still name, plus two eight-byte literal
  stores, with the room test raised from nineteen bytes to the twenty-four the
  last step reaches. Best of ten alternated runs on prebuilt blocks: dickens
  2 134 -> 2 247 MB/s, mozilla 2 133 -> 2 226, xml 2 673 -> 2 723, x-ray
  5 675 -> 5 707, nci 2 636 -> 2 606 (its commands are 40% escapes and take the
  careful path). The 4 KiB page row moved from 1 826 to 1 994 MB/s, and the
  parity and fuzz digests did not move at all.

The three together say something worth keeping: on this decoder the
per-command tests were never the cost, so an idea that removes them is worth
nothing until the copy changes. The remaining gap to lz4 on prose is about a
quarter, and it sits on the escape-heavy side, where the careful path still
uses the rle-table copy helper.

## Coding the token stream

The LZ forms left a residue that was easy to see and hard to take: the command
stream is bytes, and on a page of heap pointers the distribution of those bytes
is nothing like flat. Huffman coding the payload is the oldest answer to that.
It went in as a block form of its own -- flag pattern 110 -- rather than as
another LZ encoder, so it wraps a stream the match finder already won and never
has to re-decide a match.

Three of its choices were settled by measurement, and two of them by being
wrong first.

* Two symbols per lookup is the wrong table for this data. A 2 048-entry table
  only packs two codes into one entry when their lengths fit inside eleven
  bits, and an LZ command stream averages 7.41 bits per symbol, so a pair
  needs fourteen. Measured over the page and mixed corpora, only 10.4% of
  eleven-bit windows hold two complete codes: The first tables were
  built that way and produced a 101% payload: 32 KiB of lookup to decode one
  symbol per lookup. One symbol in two bytes of entry is what the payload
  wants, and it halves the tables to 16 KiB.
  Measured over 29.8 MB of real payloads at 7.41 bits per symbol, 10.4% of
  windows hold two complete codes, so a pair table would remove one chain
  step in ten, not one in two.

* The decoder had to stop keeping its state in memory. A first version held
  seven values per stream -- accumulator, bits left, bit position, cursor,
  cursor end, stream start, stream end -- which does not fit in sixteen
  registers, and the compiler's answer to that is to spill all of them: 122
  stack moves in the decoding function and 470 MB/s. Three values per stream,
  five lookups between refills on a fixed cadence instead of a test, and no
  bounds check inside the loop is 1 450 MB/s, or 2.40 cycles per byte. That is the same shape huff0
  (Zstd's entropy coder) uses, and the shape is the reason, not a coincidence.
  It is also 2.40 cycles per byte, which is where the shape lands: one chain
  step is a masked window, a five-cycle L1 load and a shift, and four streams
  at three live values each is what the register file allows.

* Interleaving the four encode streams is also wrong here. It needs twenty live
  values -- four cursors, four sources, four ends, four accumulators, four bit
  offsets -- and spills. Coding them one after another needs six, and the
  overrun that each stream's eight-byte flush leaves in the next stream's
  region is repaired by that stream being written afterwards. Positions within
  a five-code step come from a prefix sum rather than a running total, so the
  five shifts issue in parallel instead of queueing behind each other's table
  loads. The result is 2.2 GB/s of encode on real Silesia payloads, 1.66 to
  1.70 cycles per byte, against 500 MB/s for the version that accumulated a
  position per symbol.

Two more passes went after the first version shipped, and both by asking what
each one was really for.

The four per-stream histograms went first. They were doing two jobs -- picking
the table, and making each stream's exact length arithmetic instead of a
scan -- and the measurement said the two jobs want different things. A scatter
histogram is a chain of read-modify-writes to a counter, and an LZ payload
hits the same bin often enough that the chain is what the pass waits on: 2.52
cycles per byte with the four streams counted one after another, 2.00 with
them interleaved, 1.95 with sixteen copies of the counters instead of four.
That is close to a hard floor, and it was the largest single item left in the
encoder. Ranking four fixed code books does not need every byte, so the table
is now chosen from one byte in eight -- 0.25 cycles per byte -- and the
ranking is the only thing that was ever allowed to be approximate. It is still
deterministic: every build, every SIMD level and every worker samples the same
bytes of the same block, so a block is still a function of its own bytes.

The exact lengths then came free from the coding loop. That loop already forms
the bit total of every five symbols it stores, so a stream's encoded length is
what the write itself produced. When the caller's capacity can absorb eleven
bits per symbol and the store slop, the four streams are written first and the
twelve-byte header last, and no bit count runs at all. When it cannot -- a
payload near the 64 KiB staging buffer, where eleven bits a symbol could not
be stored -- the lengths are counted first and nothing is written until the
size is known. Both paths were checked to produce identical bytes.

What it is worth, measured over 16 MiB of 4 KiB pages at level 9 -- HUF on
against a build with it switched off, the two binaries alternating: the frame
goes from ratio 0.3887 to 0.3649, so 6.1% smaller. 63.4% of blocks take the
form, and it cuts the payloads it wraps by 11.3%.

Choosing the table from one byte in eight rather than every byte was the one
approximation in the design, so it was measured the same way, against a build
with the stride at one: 9 534 bytes out of 24.5 MB, which is four hundredths
of a percent of the output. That is what makes the sampled ranking the right
trade -- the exact version of it costs a cycle per byte and buys nothing a
reader can see.

It costs 18 KiB of read-only tables and a pass on each side: under two percent
of encode time at level 9 end to end, because the match finder still dominates
the block, and about 1 430 MB/s of decode end to end on Silesia. Level 6 does
not offer it: the rung below MAX is where encode time is worth more than seven
percent of bytes. `ZPSE_ENC_HUF` asks for it at any level.

Four shapes of the writer were built and timed against each other on real LZ
payloads (`build/hufvar`), and the shipped one won every time. One 32-bit table
holding code and length together -- one load per symbol instead of two -- is
2.25 to 2.50 cycles per byte against the shipped 1.66 to 1.70: the loop is not
waiting on load ports, it is waiting on the shifts and the prefix adds, so
trading a load for a shift and a mask costs more than it saves. A tree of ORs
instead of a chain is 1.71 and 1.82; ten symbols per iteration is 1.74 and
1.87. Interleaving two streams needs two output cursors whose starts depend on
each other's final lengths, which is the same reason four at once does not
work.

The form is kept only when it saves at least a sixty-fourth of the payload it
wraps. Without that gate, a payload that is already close to uniform still
finds a better code sometimes, and a block would pay for a second header and a
second pass to gain four bytes.

## Why a Silesia encode is not five hundred megabytes a second

The entropy pass is not the gate. It writes at 2.2 GB/s and reads at 1.5, and
it only ever wraps a payload the block encoder has already produced, so a
Silesia block cannot be waiting on it. The gate is the match finder, and the
library's own cycle counters say so (`ZPSE_BLOCK_STATS`, read by
`build/blkstat`): at level 6 the finder is 33% to 92% of a block's encode on
every file where it runs at all, and the two files where it does not run
(`mr`, `x-ray`) spend 62% and 64% in the shape trials instead.

The arithmetic a reader can check in one line: the finder runs at 278 MB/s on
`dickens`, 315 on `webster`, 409 on `mozilla`, and a 64 KiB block cannot leave
that pass faster than the pass runs. The rest of the pipeline -- the candidate
probe, the trials, the header -- is the other half of the budget and cannot buy
back what the finder spends. The ladder makes the same trade visible from the
other end: level 1, which writes no LZ form at all, encodes Silesia at 983 MB/s
for 1.24:1, and every rung that offers the finder pays roughly two and a half
times the time for two thirds of the bytes.

Three changes came out of that profile. Together they are about 4.2% of
geometric mean over six Silesia files for 0.3% of packed size.

The first is `restrict` on the table pointer: the store of the cursor into the
hash table cannot alias the candidate load, so the compiler is free to keep
that load from queueing behind the store. Three interleaved rounds over eight
files, best of each, byte-identical payloads throughout: `nci` +6.1%, `sao`
+5.1%, `dickens` +2.8%, `webster` +2.2%, `osdb` +1.5%, `mozilla` -0.4%, `mr`
-0.6%, `ooffice` -2.5% -- about 1.9% geometric mean, and the same stream.

The second is the lazy probe's ceiling, 24 bytes down to 12. The probe is a
dependent chain -- hash the four bytes at the next position, load the table
entry that hash names, load the four bytes there -- so its cost is nearly flat
in the ceiling while the size it buys is not. Best of four alternating runs per
configuration, whole 64 KiB blocks: `dickens` 260 -> 274 MB/s with its payload
byte-identical, `webster` 308 -> 315 for +0.2% of size, `mozilla` 396 -> 408,
`xml` 666 -> 695 for +0.4%. Below 12 the rate stops moving and the size does
not: at 8, `webster` gains another 1.5% of rate for 0.4% of size, which is the
wrong way round.

The third is where the probe's two loads are issued. They used to be loaded
where the decision is made, after the backward walk and the eight-byte
forward extension, which stalls the loop for the whole latency of the chain
once per match. Issued before that work, they overlap it. The values read are
the same, from the same places, in the same order relative to the table write
at the top of the loop, so the decisions and the streams are identical: the
same payload bytes in every one of the six files, and `xml` +4%, `dickens`
+2%, `mozilla` +1%, `nci` +0.5%, `webster` and `osdb` to the noise.

A fourth experiment was rejected. Sweeping `hash_bits` 11..16 over whole
blocks showed 16 bits both faster and smaller than the 13 the benchmark binary
uses (`dickens` 292 against 274 MB/s, 39 477 against 40 474 bytes), which looks
like a free win until you check the encoder: the block path already passes 15
(`e->tstage` is the 64 KiB region that holds 2^15 entries, and
`ZPSE_LZ_TABLE_BIG_MIN` is where it starts being used), so the only thing the
sweep was measuring is the benchmark's own default. Sixteen bits would need a
128 KiB staging region for a few percent that 15 already has most of.

The trial loop was the other half of the profile, and it is harder to cut than
it looks, because a window price cannot separate the survivors it keeps -- that
is what keeping them means. The one thing it can answer is "how many trials is
a full-size block worth": its price window is a fifth of the block, and on the
dumps two full trials are 63% of a block's encode. So a full-size block runs at
most two. Measured over Silesia at level 6, best of three alternating rounds:
`mr` 257 -> 272 MB/s, `samba` 409 -> 424, `ooffice` 225 -> 232, `mozilla` 316
-> 323, and the corpus total 382 -> 385 MB/s -- inside this host's run-to-run
noise -- for no measurable size.

The first attempt at this was wrong and the suite said so. Setting the level
row's trial count to 2 replaces the tied-field cut rather than supplementing
it, and that cut is what keeps the identity survivor whatever the price says
and what stands down when the head window lied or a survivor was kept on
content. Three cases of `tests/suites/test_sizes.c` grew by 8 bytes each. The
shipped rule leaves the cut alone, applies the cap after it, and fires only
where the cut's guards already hold by construction: a full-size block. A cap
of one trial is another 2.5% of rate for another 0.09% of size, and it is not
taken.

Two more were measured and rejected, and both are left in the tree as switches
so the measurement can be repeated.

The first is a lazy probe that refuses a self-referencing candidate -- the slot
the loop has just filed for the current position, which happens when the bytes
at pos and pos + 1 hash alike, i.e. in a run. It is not faster (`dickens` 285
-> 276 MB/s, `webster` 332 -> 324) and it does not hold the ratio: `mozilla`
grows 190 bytes, which is exactly the distance-one run the probe was taking.
`ZPSE_LZ2_PROBE_SELF` stays 1.

The second is the full-size trial cap applied to the tied-free rows as well --
levels 8 and 9, whose whole design is to encode every survivor their price
separated. Default against `ZPSE_TRIAL_CAP_TIEFREE=1` over Silesia: level 8
341.5 -> 335.4 MB/s, level 9 250.0 -> 254.3, level 6 unchanged in both (it is
capped already), and 7 230 bytes off a ~94 MB packed total in each direction.
The deepest rungs are not paying for trials they do not need, so the flag stays
0 and they keep the field.

A ceiling re-sweep on the early-load build says the knee has not moved: 10, 12
and 14 are within 1% of each other in rate (`dickens` 271/285/284, `webster`
338/332/330, `xml` 707/685/689), while 14 is a few bytes smaller on four of the
six files and 10 is a few bytes larger on two. 12 stays, because the difference
is inside the host's noise and moving it would re-pin every stream in the suite
for nothing.

What would move the ceiling rather than the noise is a finder that makes fewer
or better-placed probes per byte, and every version of that changes which
matches are found, so it changes the stream, the ratio and every pinned size in
the suite. That is a different change from this one and it is not taken here.

## The fast LZ decoder wrote past its block

A frame's workers decode straight into the caller's buffer, each block at its
own offset, so a block decode must write exactly its own bytes. That invariant
was false for one class of block, and it took a shuffled decode to see it.

The fast LZ decoder's fast region reserved 24 bytes of output before it took a
command, counting the match store. But the match store does not start at the
token: it starts at the token plus the literal run, which the same region has
already advanced `op` over. The literal store itself touches 16 bytes, so the
true reach was 14 + 24 = 38. A command carrying a literal run and a 17- or
18-byte match could therefore write up to 14 bytes past the end of its block --
straight into the next block's slot.

In order, nothing showed. The next block is decoded immediately afterwards and
writes those bytes again, so every sequential decode and every whole-frame
decode of the 202 MiB Silesia corpus was correct. The frame benchmark found
it because it decodes blocks out of order and compares each one against the
input: two bytes at the front of block 99 and block 128 of a 16 MiB columnar
frame, and only sometimes, because whether a poisoned block was decoded before
or after its neighbour depended on which worker claimed it.

Finding it took a harness that names the failing side rather than guessing at
it. A canary decode -- each block into a slot with eight guard bytes after it
-- reported the overrunning blocks directly: three of a 16 MiB frame, all of
them fast-LZ blocks. That killed the first hypothesis (decoder arena state
carried between blocks: re-initialising the decoder before every block changed
nothing) and the second (a bad predecessor: no single predecessor reproduced
it). The two differing bytes were the block's first two, which is what a
neighbour's overrun leaves.

The reserve is `ZPSE_LZ2_FAST_ROOM`, 38, with the arithmetic written down
beside it. After the fix: zero overruns across every block of a 16 MiB frame,
zero bad blocks in forward, shuffled and fresh-decoder order, and identical
frames and decode output at 2, 3, 4, 5 and 6 workers. The two other decoders
were checked for the same shape and are not affected: the token form reserves
`LZ_CMD_ROOM + lit + 16`, which includes the run, and the hex form's 32 bytes
cover its 14 + 16.

This is why the README's parallel section quotes 16 MiB again, and why the
invariant the README opens with -- a block's output is a function of its own
bytes -- is now tested by an out-of-order decode and not only an in-order one.

## The finder's next key, read before the candidate

The match finder spends its time on one dependent chain: hash the four bytes
at the cursor, load the table entry that hash names, then load the four bytes
at that entry and compare. The third step cannot start until the second
returns, and the second is an L2 hit, so on prose -- where a match turns up
every five or six bytes and the miss path advances the cursor one byte at a
time -- the loop is waiting on that chain more than it is doing arithmetic.

The lever is not a shorter chain but work already in flight. `ZPSE_LZ_PIPE_NEXT`
reads the key at pos + 1, hashes it and reads its slot immediately after the
store for pos, before the candidate at pos is tested. That is the position the
cursor steps to on a miss, and a miss is the common case, so the read sits
behind the compare instead of in front of it. It is taken after the table
store, which is what makes the change byte-identical: a position whose slot the
store has just overwritten reads back exactly the value the next iteration
would have read, and every other read is unchanged.

Measured with `tests/benchmarks/bench_lz2.c` over six Silesia files, two
binaries alternating over the same blocks, six rounds: ahead in five of the
six, median 312 against 305 MB/s, payload 176,670 bytes either way, and the
suite's stream digest unchanged. Identical bytes for a measured rate is the
shape of change this project takes.

Two knobs beside it were measured and left where they are. Dropping the fast
form's lazy ceiling to 4 or 8 is 2-3% slower, not faster: the probe is what
keeps the number of positions the loop walks down, so removing it costs more
than the probe does. Moving the miss acceleration from `misses >> 5` to 4 or 6
moves the weighted rate by 1.5% in one direction and the payload by 0.19% in
the other, which is the host's spread rather than a result.

## The drive, kept out of the tables

Every cross-tool harness here times a process that reads its own input, and
the corpus lives on a disk: a cold 202 MiB read is the size of a whole Silesia
encode pass. Reading each file twice to warm the page cache was the first
answer to that, and it is not sufficient on a host whose reads go through a
compatibility layer at 0.6-1.7 GB/s -- on a 68 MiB corpus that is the same
order as the codec's own rate, so the read could still be the column.

`tests/benchmarks/lib_stage.sh` now stages every corpus file in a
memory-backed directory when the host has one, reads the staged copy back
before the clock starts, and reports the rate that read ran at under the
table. The in-process harnesses (`bench.c`, `bench_levels.c`,
`bench_parallel.c`, the CLI's own `bench`) never needed it: they read once and
then clock the codec on their own copy, so their numbers carry no read at all.
The scripts that run a tool as a process print the floor, and the tables are
read against it rather than against an assumption about the drive. The lz4
rows in the cross-codec tables are the real `lz4` CLI built from 1.10.0
source, driven the way a user drives it.

The other use of staging is measurement itself. The same tree builds here
twice -- clang 22.1 (LLVM-MinGW) for the tables, gcc 15 under WSL2 -- and
the two produce identical streams (the parity digest and every packed size
in the suite match), which is the repository's byte-identity invariant seen
from a second compiler on a second OS, not a claim about it. An A/B that is
close to the host's noise band is run on both, and the quieter one decides;
the fast-form lookahead below is one of those, and it agreed on both.

## The fast form's finder, the same lookahead

The token form's finder (`src/lz.c`) got its next-key lookahead first. This is
the same lever in the fast form (`src/lz2.c`), which is the one text blocks
actually run, and it paid twice over because the lazy probe asks about the
same position the lookahead reads.

The loop files `pos` and then tests the candidate at `pos`. On a miss the
cursor steps a single byte -- four iterations in five on prose, where the
average match is 5.4 bytes -- and the old code loaded the word at the new
cursor, hashed it and read its slot only after that test had resolved. The
lookahead issues those three reads right after the store, so the miss path
starts with its key, hash and entry in registers, and the entry is carried
across iterations instead of being re-read at the top of the loop. On a hit
the probe was about to read the same three values for `pos + 1` anyway, so it
now takes them from the lookahead instead of issuing the chain a second time.

The reads are taken after the table store, so a slot the store has just
overwritten reads back exactly what the next iteration would read, and every
other read is unchanged. Two binaries alternating over six Silesia files
(176,670 bytes of payload either way), eight rounds on one core: ahead in all
eight, median 367.5 against 356.1 MB/s. Through the whole block encoder at
level 6 over the twelve-file corpus, ahead in three rounds of four (median 392
against 376 MB/s) with the payload identical to the byte, 169,502,562 bytes
both ways.

## Wider steps in the match copy

Both LZ decoders copy a long match eight bytes at a store once the distance is
past sixteen, because a step wider than the distance would read bytes the same
copy has not written yet. That is the right reason, but the wrong guard: the
condition is the distance, not the size left. Past 32 bytes back a 32-byte step
is safe and past 16 a 16-byte one, so the long match stops paying eight bytes a
store and only its tail does. A match that ends mid-step is copied by the same
steps and byte loop as before, and the bytes are identical by construction --
this changes how the copy runs, not what it copies.

Prose does not move: its matches average 5.4 bytes and never reach that loop.
The columnar corpus does, and decode-only over its 17 files (two binaries
alternating, four rounds) the mean is 11.1 against 10.3 GB/s, ahead in four of
four rounds -- though the per-file spread on that host is wider than the
difference on the files whose matches are short.

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
