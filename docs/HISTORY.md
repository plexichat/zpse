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
  the block. It was offered as one more candidate at the deeper levels and was
  the difference between raw and compression on high-cardinality text.
* **LZ over the tile body**, which compresses the body the tile path would
  have written and then reconstructs it through the ordinary tile decoder.
* A fixed-distance **fast LZ** form (LZ4's shape) for the cases where the
  variable-length distance costs more than it saves (later removed).
* A **hexadecimal LZ** form for page-sized blocks, which packs common
  literal/length pairs into a nibble and shortens fresh distances (later removed).

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
* `ZPSE_ENC_PAGE_FAST` (later removed), which cut the candidate list to the
  identity on the cheap rungs and removed the pricing pass, the sample and the
  trial cap.
* A candidate cap on the levels that price every plane, chosen because the
  cap's ratio cost crosses the one-point line only there.
* Shorter ASCII and match-finder samples at page size, and pricing a page
  whole once every plane is priced.
* The noise abort extended to pages, but only under the fast flag (later
  removed), because a page has no sweep behind its trial to repair a wrong
  verdict.
* The hexadecimal LZ form (later removed) from level 7 up, where distances
  above 4320 were rare enough that its shorter distance codes won.
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

The repeat-distance cache itself did not survive that line of reasoning. It
coded a recent distance as one byte (0xFC | slot), which saved a byte wherever
the finder named one, and the test for it was a branch on every command. What
the branch cost was not in the original argument: removing the cache leaves a
fixed two-byte little-endian distance -- lz4's shape, and one 16-bit load as
the decoder's only distance work. Measured in memory on twelve 64 KiB Silesia
blocks, decode went 2.31 -> 2.93 GB/s (+26.5%), encode +12.8%, packed size
+1.366%. Seven of the twelve files stayed within 0.2% of their old size,
because the finder rarely names a recent distance on prose; the cost lands on
the spike-pattern cases in test_sizes, mozilla +4.02% and nci +3.89%, where a
recent distance is worth a byte often enough to miss it. Dropping the rep
slots also frees the 0xFC..0xFF low-byte range that had to be escaped around,
so the full 1..65535 distance range is usable.

## Coding the token stream

The LZ forms left a residue that was easy to see and hard to take: the command
stream is bytes, and on a page of heap pointers the distribution of those bytes
is nothing like flat. Huffman coding the payload is the oldest answer to that.
It went in as a block form of its own -- the old flag pattern 110, later
replaced by the single bit-7 `ZPSE_B_HUF` form -- rather than as another LZ
encoder, so it wraps a stream the match finder already won and never has to
re-decide a match.

Three of its choices were settled by measurement, and two of them by being
wrong first.

* One symbol per lookup is the right table for this data. A two-symbol entry
  only fires when both codes fit inside the ten-bit window, and an LZ command
  stream averages about seven bits per symbol, so a pair needs fourteen and
  almost never has both codes in ten bits. A two-symbol table was measured on
  this corpus and produced a 101% payload: it paid for a wider table and still
  decoded one symbol per lookup. Two bytes per entry and a 1 024-entry decode
  table give one symbol per lookup for 8 KiB of lookups, half what the
  16 KiB two-symbol tables took.

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
what the write itself produced. When the caller's capacity can absorb ten
bits per symbol and the store slop, the four streams are written first and the
twelve-byte header last, and no bit count runs at all. When it cannot -- a
payload near the 64 KiB staging buffer, where ten bits a symbol could not
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

It costs 10 KiB of read-only tables -- 2 KiB of codes and 8 KiB of lookups --
and a pass on each side: under two percent of encode time at level 9 end to
end, because the match finder still dominates
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
`ZPSE_LZ2_PROBE_SELF` (later removed) stayed at 1.

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

The reserve was `ZPSE_LZ2_FAST_ROOM` (later removed), 38, with the arithmetic
written down beside it. After the fix: zero overruns across every block of a 16 MiB frame,
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

## The fast form's finder, the same lookahead (later removed)

The token form's finder (`src/lz.c`, later removed) got its next-key lookahead
first. This was the same lever in the fast form (`src/lz2.c`, later removed),
which was the one text blocks actually ran, and it paid twice over because the
lazy probe asked about the same position the lookahead reads.

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

## lz4's one-byte shift in the fast form's probe

The fast form's lazy probe asks one question: would a match starting one byte
later run longer than the one in hand? It answered by hashing the four bytes at
`pos + 1`, loading the candidate they name and extending it -- two dependent
loads, a hash and a table read, on every match the lazy ceiling lets through.
lz4 answers it for free in the common case: if the eight bytes after the match's
own start equal the eight after its candidate, the same match shifted by one is
at least eight long, which is already longer than any match the ceiling (12)
would spend a probe on.

Measured alone on whole 64 KiB Silesia blocks with the table bits held equal
(2^13, best of three): `dickens` 280.9 -> 345.9 MB/s (+23.1%) for 40,474 ->
42,141 bytes (+4.1%), `mozilla` 422.2 -> 461.2 (+9.2%) for +3.6%, `webster`
316.0 -> 376.4 (+19.1%) for +3.8%. That is the right trade for a rung whose
promise is rate, and the wrong one for a rung whose promise is size: through
the whole block encoder at the default level, prose grew about 4%, which is the
shape `test_sizes` exists to refuse.

Every shape that keeps the probe was measured at BALANCED and none of them pays.
The shift in place of the probe for matches under eight, gated by the caller's
level field so that only BALANCED and below take it, is +2.6% of encode on
`dickens` (491.9 -> 504.9 MB/s) and +5.5% on `webster` (283.4 -> 298.9) for
+0.25% and +5.9% of packed bytes, and over all twelve Silesia files at 64 KiB it
is 105,451,298 -> 107,445,043 bytes (+1.89%) for a mean +6.3% of encode that is
better on seven files and worse on five: `reymont` +12.4% and `xml` +7.5% of
payload, `webster` +5.9%, `samba` +2.6%, then a tenth of a point or nothing on
the rest; `xml` -9.4%, `x-ray` -8.0%, `nci` -5.6%, `mr` -2.7% and `mozilla`
-1.1% of rate. As a pre-filter for matches under six, with the probe still
deciding everything else, BALANCED is byte-identical to the baseline on all
twelve files -- 105,451,298 bytes either way -- and the rate is inside this
host's spread, because the only probes it can skip are ones whose answer it
already agrees with; the eight-byte gate with the probe still running is
identical again. The shift's gain IS the probe it replaces, so a rung that keeps
the probe buys a compare per short match and nothing else.

That is why the arm was removed rather than shipped dormant. 1.89% of payload
for a rate this host cannot resolve outside prose is not what BALANCED is for,
and 0.25% of payload for no rate is not a feature: the code, the two knobs and
the level argument on `zpse_lz2_encode` are gone, and this entry is the record
of what the shift is worth and where it stops paying.

The default level got its own attempts and kept none of them. Inlining
`lz2_hash`, `lz2_load64` and `lz2_extend` leaves the payload unchanged, and the
finder reads 287.1 / 401.7 / 332.3 MB/s against 280.9 / 422.2 / 316.0 -- inside
this host's spread, both signs. Changing its decisions instead, over all twelve
files, best of three rounds each: the skip shift from 5 to 6 writes 101,843,154
bytes against 101,881,161 (-0.04%) with the rate mixed-sign and summing
negative (+2.9% on `osdb`, -4.3% on `sao`, -3.8% on `xml`); the lazy ceiling at
10 and at 8 costs 101,920,094 (+0.04%) and 101,967,508 (+0.09%) for no rate; the
two together is 101,882,391 (+0.001%) and again inside the spread. Level 6 pays
for the finder pass itself, not for the decisions it makes inside it, so the
rung keeps its parse.

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

## Fourteen bits for the lazy pass, level 4's fourth trial, and the CRC's place

Three follow-ups were measured in one pass. Two changed the tree; one did not.

**The 4-16 KiB lazy pass got all fourteen bits (later removed).** The arena's LZ region was
16 KiB -- a 2^13 table -- and the lazy second pass on blocks between 4 and
16 KiB inserts up to 16,384 positions into it, a load factor of two. The region
was widened to 32 KiB (`ZPSE_E_SZ_LZ 0x08000`, `ZPSE_LZ_HASH_BITS_WIDE 14`) and
the arena to 464 KiB. Every other caller of the finder kept 13 bits, so the
screens and the page passes were byte-for-byte untouched -- at the time only
the lazy pass indexed all fourteen bits (that custom finder was later removed
when LZ4 took over the search). Measured per file over two rounds at level 6
with `BLOCK=16384 bash tests/benchmarks/ab_silesia.sh`, the rounds agreeing on
every byte: the twelve files go 106,180,882 -> 106,079,560 bytes,
-101,322 (-0.095%). mozilla -58,844, nci -13,251, ooffice -12,510, osdb -9,901,
samba -6,371, sao -217, xml -124, dickens -92, mr and x-ray unchanged, and
reymont +25 -- the one case the wider table loses, in both rounds. At the
default 64 KiB block the same build is identical on eleven files and 4 bytes
smaller on reymont, and the 768-case size gate reports no growth. The price is
16 KiB of every caller's encode arena and the win is confined to blocks of
4-16 KiB.

**Level 4's rung is one trial, and it is worth its rate.** Level 4 exists only
to spend a fourth full-block trial over level 3, so the rung is now
`ZPSE_L4_TRIALS` and the fold is a build. Over Silesia at level 4, two rounds,
identical sizes in both: level 3's three trials write 163,955,885 against
163,631,549 (+324,336, +0.198%), and seven of the twelve files do not move at
all -- the rung is mozilla's (+293,583), with samba +14,008, nci +12,535,
ooffice +2,881 and x-ray +1,329. Encode is faster folded on every file in both
rounds (mozilla 352/426 against 298/349 MB/s, dickens 1764/2193 against
1321/1463), so the trial buys 0.198% for its rate and level 4 keeps it.

**Fusing the CRC into the inverse buys nothing it can reach.** The decoder's
checksum pass was priced with `ZPSE_SKIP_CRC_VERIFY=1`, which drops PHASE5
entirely: at level 1 on the fastest blocks the pass is a large share of decode
(xml 7.2 -> 12.7 GB/s, reymont 5.4 -> 10.2, ooffice 5.1 -> 6.9), while every
other cell on this host sits inside a 2-3x per-file spread. The share is
already the `crc32` instruction's throughput limit -- three chains at 8 B per
cycle is 24 GB/s, and a fused loop still executes one `crc32` per eight bytes,
so it can only save the second L1 read, a fraction of what that arm removes.
The tree had already measured the two forms against each other
(`zpse_crc32c_copy_hw`: ~11 GB/s fused against ~3x for memcpy plus the
three-chain hasher, which is why the fused form only serves blocks under
8 KiB), so no fusion was landed. The headroom is real but it is not in the
fusion: a folding CRC (PCLMULQDQ, and 4-8x again with AVX-512 VPCLMULQDQ)
would have to replace the instruction chain, which is a different change with
its own detection and test surface.

`ab_silesia.sh` grew the `BLOCK` variable for the first measurement; without
it the default-block numbers are the only ones the script can produce.

## The page lazy probe measured its match a byte at a time

The lazy pass on a page-sized block weighs the match one byte ahead against the
one it already has, and weighing it means walking its whole length from four
bytes on -- the only place in either LZ finder that still measured a match a
byte at a time (`src/lz2.c` and `src/lzh.c` ask `lz2_extend` / `lzh_extend`,
which have stepped eight bytes at a time since they were written). On prose
that walk is one or two bytes; on a structured page it is fifty and more, and
every byte of it was a dependent load, compare and branch.

The wide form compares eight bytes and takes the low set bit of the mismatching
word as the length, which is where the byte loop would have stopped, so `l1`,
the decision and the stream are unchanged. `ZPSE_LZ_LAZY_WIDE 0` puts the byte
loop back for re-measuring the two.

`ZPSE_LZ2_BLOCK=4096 bench_lz2 FILE...` (the harness and switch were later
removed) then measured the page path (the lz1 column passed the block size as
the lazy ceiling, which is the rule the block encoder applied; the default was
still the fast form's 64 KiB). Two runs per arm
-- each run best-of-three passes over the file, arms alternated in both orders
-- LZ encode MB/s, 4 KiB blocks then 16 KiB: `dickens` 256.6 -> 274.4 and 217.4
-> 246.3, `mozilla` 383.3 -> 433.1 and 336.2 -> 373.7, `nci` 514.7 -> 666.4 and
532.0 -> 725.8, `xml` 423.7 -> 523.2 and 462.1 -> 573.8, `reymont` 256.0 ->
301.4 and 255.6 -> 319.7, `ooffice` 311.2 -> 338.2 and 244.7 -> 266.6,
`samba` 405.5 -> 486.8 and 387.4 -> 470.3. Every file, both block sizes, both
orders.

The whole encoder sees a smaller share of it, because the LZ pass is one of the
things a page block does: the CLI at `-b 4096` over the same 16 MiB stream,
two rounds, packed bytes identical at every level, level 6 361.1 -> 414.8 and
363.0 -> 410.9 MB/s (+14%), level 9 288.4 -> 324.7 and 288.5 -> 327.8 (+13%),
level 1 535.2 -> 542.3 and 546.0 -> 553.1 (inside the host's spread, which is
what a page block that writes no LZ form should do).

Sizes did not move, which is the whole claim: the 768-case size gate reports the
same `delta=-435636 bytes`, `LZ: ALL OK (1191 checks)` pins the payloads this
pass writes, and the parallel digest is unchanged.

## A wide walk that lost, one trial on a full block, and a page table re-read

Three follow-ups in one pass. One changed which blocks the encoder writes; two
change nothing in the library at all.

**The backward walk stays bytewise.** The walk into the pending literals is the
last loop in the LZ finders that steps one byte at a time, so widening it to
compare eight bytes and step only while they all match looked free: it is the
same walk, and the byte loop still resolves the byte a mismatching word stopped
on, so it cannot change a payload. It is not free. Five alternating runs per
arm, payload bytes identical in every cell, LZ encode MB/s bytewise -> wide:
`dickens` 321.5 -> 310.0, `mozilla` 493.1 -> 470.1, `nci` 793.4 -> 749.9, `xml`
585.1 -> 576.8 at 4 KiB blocks; 299.6 -> 282.3, 435.0 -> 422.3, 861.2 -> 843.5
and 670.8 -> 653.4 at 16 KiB; the 16 MiB page corpus 605.0 -> 576.5. The walk
is bounded by `anchor`, the end of the previous match, so on prose it takes one
or two steps and the word loop almost never runs -- and its two bounds plus the
loop setup are paid on every match. `ZPSE_LZ_BACK_WIDE`, `ZPSE_LZ2_BACK_WIDE`
and `ZPSE_LZH_BACK_WIDE` stayed in the tree at 0 so the measurement could be
repeated (the LZ2 and LZH knobs were later removed).

**A full-size block now spends one trial, not two.** The cap was two, on the
argument that a full block's price window is a fifth of the block and the two
survivors behind it are the ones the price could not separate. Re-measured at
level 6 over Silesia, five alternating rounds per arm (each run best of two,
arms in both orders), packed bytes deterministic: `mr` -2,651, `mozilla`
-3,781, `ooffice` -839, `x-ray` +18,687, the other eight files identical, the
corpus 101,869,745 -> 101,881,161 (+11,416, +0.011%). Encode: `mr` 253.9 ->
320.5 MB/s, `x-ray` 394.4 -> 455.8, `mozilla` 334.0 -> 340.2, `dickens` 808.0
-> 834.6, `samba` 430.1 -> 440.1; `reymont`, `sao` and `xml` are 1-2% behind,
inside this host's spread. Levels 5 and 7 pay and gain the same bytes, because
the cap collapses their streams onto the same one-trial block.

The earlier reading of a one-trial cap -- 2.5% of rate for 0.09% of size -- was
taken before the tile-body and LZ forms could win a block, so the second trial
was the only thing left that could change the stream. With them, `x-ray`'s
+0.38% is the whole size cost, and the rate gain lands where the trials were:
the two files that run no finder pass. `mr` and `x-ray` still spend 51% and
54% of a block in the one trial, which is the shape decision itself; that is
where their ceiling sits now.

The gates agree with the claim. The 768-case size gate still reports no case
grown (`grew=0` over 768, `roundtrip_bad=0`) and its total is 88 bytes smaller
than before (`delta=-435548` against `-435636`); `LZ: ALL OK (1191 checks)` and
the parallel digest `cec73f05` are unchanged, because neither form is affected
by the tile sweep; and the scalar/swap parity builds still produce identical
digests.

**The 4 KiB page rows come from the tree's own corpus.** They are read from a
generated `pages-nonzero.bin` (16 MiB, sha256 `c9c0e860…`, `corpus_write
--pages`) with the CLI at `-b 4096`, best of five, one process, median
reported. Packed bytes are a property of the stream: 1.74:1, 2.40:1, 2.56:1
for levels 1, 6 and 9, 1.52:1 for `PAGE_FAST` (later removed) and 2.21:1 for
`LZ_FAST`. The rates are 296/2879 for level 1, 175/1841 for level 6, 138/1323
for level 9, 995/3365 for `PAGE_FAST` (later removed) and 624/1967 for
`LZ_FAST` (encode/decode MB/s). On
the same stream with one zero page in eight, level 6 reads 2.67:1 and decodes
at 1954 MB/s.

`LZ_FAST` gives up 8.6% of the packed size for about 3.6 times the encode rate,
and level 9's entropy pass costs 19% of its encode. The per-file profile table
in the README is regenerated with `build/blkstat` and reports the trial share,
which is what the cap change moved.

## The two macro tables, re-read on the shipped scripts

Nothing in the library changed here. `tests/benchmarks/silesia.sh` and
`tests/benchmarks/bench_pareto.sh` were run on the tree as it stands -- with
lz4 v1.10.0 built from source for the pareto rows, because this host has no
`make` and the script's source build needs one; CMake produced the same
binary -- and their output replaces the README rows.

Silesia at level 6, 202 MiB, two reps: the corpus is 101,881,161 bytes packed
(2.08:1) at 408 MB/s encode and 1908 MB/s decode. Every row moved up on this
host relative to the recorded table (`dickens` 510 -> 830 MB/s encode,
`webster` 269 -> 345, `x-ray` 295 -> 475), which is a host difference as much
as a tree difference: the same binary produces the same bytes, and the ratio
column moved only where the encoder did (mozilla 2.01 -> 2.02, x-ray 1.72 ->
1.71, from the one-trial full block).

The cross-tool table now carries every row the script measures instead of the
single lz4 point the old README quoted. On the same 202 MiB and one pass each,
`lz4 1` is 1333/1541 MB/s at 2.10:1 against zpse 6's 363/1143 at 2.08:1 --
about four times the encode rate for the same ratio, which is the trade the
README's short version names -- and `zstd 19` is 4.01:1 at 5 MB/s. The
frontier column is computed by the script: at these settings `zpse 1`, `lz4 1`,
`lz4 3`, `lz4 9`, `lz4 12` and the three zstd rows are on it, and zpse 5, 6 and
9 are not -- lz4 1 is at least as good as zpse 6 on all three columns. The
README's table prints the bytes and rates and leaves the verdict in prose.

## What is left in a trial, and in the page finder

The one-trial full block makes the trial the shape decision and nothing else,
and on the two files with no finder pass it is still over half the encode:
`build/blkstat` at level 6 reads `mr` at 51.3% trial / 12.3% probe / 34.9%
rest and `x-ray` at 54.1% / 18.4% / 26.7%, with one trial per block and the
remainder spread over the transform staging, the candidate list and the
writer. Those two are the ceiling now, and what saturates them is the per-tile
pipeline itself: a nine-predictor AVX2 prescreen (about 30 ns a tile), a
classify and mode search for the winner, and a pack. The all-hit run already
skips all three for tiles whose cheap proof holds, which is why the cost is
where it is rather than higher.

On the page path the finder is the whole of a `LZ_FAST` page and a fraction of
the rest. The finder measured alone (`bench_lz2`, `ZPSE_LZ2_BLOCK=4096`, both
later removed) read 576-600 MB/s over the page corpus after the lazy probe's
wide measurement,
with the 16 KiB table clear, the per-position hash and the probe's dependent
chain accounting for what is left. The clear is 16 KiB per 4 KiB page at
thirteen bits, but shrinking it means fewer hash bits and therefore different
matches: that is a ratio trade, not a speed change.

It was measured rather than assumed. Twelve bits on the 16 MiB page corpus,
five alternating rounds, `-b 4096`, best of each arm: `LZ_FAST` (the arm that
actually uses the page table) reads 603.1 against 558.7 MB/s at the best
round and about 569 against 551 by median, and writes 7,599,177 bytes against
7,591,788 (+0.097%); level 6, which does not take the fast form on every
page, is byte-identical and inside the noise either way. Twelve bits is
therefore both slower and larger, which is the same verdict the tree already
recorded against the 16-bit arm, and the page table stays at thirteen.The clear is not the cost it looks like: the finder is bound by the dependent
chain of hash, table load and candidate load, so halving the footprint does
not buy back the collisions it adds.

## There is no scalar classify left to vectorise

The next idea on the list was to give the tile trial an AVX2 classify for the
nine screened predictors, on the theory that the prescreen counts misses with
SIMD and the winner is then re-classified a byte at a time. The premise is
wrong, and the source says so: `tile_classify` already dispatches to
`tile_classify_avx2` whenever `nt >= 32` and the family is CONST, LANEHIST or
STRIDE (`tile_fam_is_vectorisable`), and `zpse_tile_fill_at_pred_reset` has an
AVX2 twin for the SIGNED_DELTA operand. That is every screened family, and
every tile but the last of a plane is 256 bytes. What is left of the scalar
loop is the short tail tile and the `-DZPSE_FORCE_SCALAR` reference build.

Measured rather than assumed, level 6, 64 KiB blocks, `build/blkstat` built
twice: the scalar build runs `mr` at 73.2 MB/s against the shipped 332.1 and
`x-ray` at 100.2 against 454.2, with the trial share going 71.7% -> 51.1% and
65.7% -> 53.5%. So the vector path is already worth 4.5x on the two files the
trial dominates, and the remaining trial time is the pack, the mode search and
the prescreen -- not a scalar classify that could be widened. Reusing the
prescreen's counters for the mask cannot work either: the AVX2 prescreen
accumulates nine byte-lane COUNTS, and a count does not carry where the misses
are, which is exactly what `cl->words` stores.

## The tile pipeline's split, measured

`build/tpack` (classify, then choose_modes, then pack, each step timed on the
same tiles) says the pack is the bigger half of a searched tile: on seven
columnar shapes classify is 406-698 cycles, choose_modes adds 38-72, and the
pack adds another 490-1,380 -- 49-65% of the three. `build/ucost` puts the
same split in absolute terms: on a random 256-byte tile classify is 269
cycles and pack 141, of which the mask encoder is 67 and the residual encoder
29; on a four-value tile pack is 31.

The mode histogram is the target. Over the columnar corpus, SPARSE_U8 +
RAW_ACTUAL is 17,140 packs at 364 cycles each and RAW_BITMAP + RAW_ACTUAL is
12,480 at 794 -- the second is 8 stores and a `k`-byte copy of data the
classify already gathered (`cl->act_at_exc`, `act_valid == 1`), so 794 is the
next thing to explain rather than the cost of the bytes it writes. Nothing
was changed on that reading: it is a profile note for the next pass, taken so
the pack path is entered with a number rather than an assumption.

## The sparse mask's narrowing loop was not what cost

`zpse_mask_encode`'s SPARSE_U8 branch collected exception positions into a
`uint16_t c[256]` and then narrowed them to the payload one byte at a time.
Since a tile holds at most 256 positions, every position already fits a byte,
so the array can be `uint8_t` and the write one `memcpy`. It is the same
payload, in the same order, from the same ctz walk -- and it is not faster.
Five alternating rounds per file at level 6, byte-identical output in every
cell: `nci` 589.2 -> 580.4 MB/s, `ooffice` 242.6 -> 235.0, `xml` 554.3 ->
536.2, `reymont` 350.5 -> 345.2, `x-ray` 451.7 -> 448.3, `mozilla` 335.2 ->
333.7, `mr` unchanged at 320.5, `samba` 426.2 -> 434.7. That is the host's
spread with no sign, so the change was reverted rather than kept as a
tidy-up: the mask branch was never where the pack time went, and the 794-cycle
RAW_BITMAP figure belongs to a different composition than the one this
touched.

## What mr and x-ray actually pack

`build/modestat` (scratch, `-DZPSE_TRIAL_STATS`) prints the encoder's own
(mask, residual) histogram for a file. Level 6, 64 KiB blocks:

* `mr`: 44,089 searched tiles. RAW_BITMAP-family + EXCLUSION_U8 20,402 (46%),
  a bit-packed pair 12,481, SIGNED_DELTA buckets 7,919, the rest under 2,800.
* `x-ray`: 26,165 tiles. 10,153 in one bit-packed pair and 8,371 in another,
  so 70% of its tiles land in two modes, with SIGNED_DELTA at 4,400.
* `samba` is the same shape as `mr` (7,393 of 9,784 in one mode); `dickens`
  and `xml` never leave EXCLUSION_U8 on the tiles they search at all.

The predictors say the same thing from the other side: `p0` (CONST_ZERO) and
`p5` account for 43,440 of `mr`'s 44,089 tiles and 30,942 of `x-ray`'s 26,165
-- so the trial on these files is sparse masks over a constant or a stride,
not a search among shapes. There is no single mode loop to fix: the cost is
spread across four packers, which is why the sparse-mask narrowing measured
flat and was reverted.

## The RUNS payload difference, widened and refused

RUNS is the hot mask on the image files: 20,402 of `mr`'s 44,089 packed tiles
are (RUNS, EXCLUSION_U8), 46% of them, and `tpack` prices that pair at 794
cycles. Its payload is the first difference of the edge list, so the write can
be done eight runs at a time with one 16-bit subtract and one unsigned pack
(`packus` cannot saturate: no run exceeds nt-1 <= 255). Same bytes, verified
equal in every cell, and it is not faster: five alternating rounds at level 6
gave `mr` 322.4 -> 320.0 MB/s, `x-ray` 471.5 -> 454.2, `xml` 558.6 -> 536.5,
`nci` 591.9 -> 578.6, `ooffice` 239.0 -> 236.9, `reymont` 332.7 -> 327.4,
`mozilla` 333.3 -> 332.3, `samba` 426.1 -> 433.9.

So the per-run loop was not the cost either. With the sparse-mask narrowing
above it, both edits that targeted the mask payload came back flat or behind,
which says the time in these packs is in the edge scan (one ctz per transition)
and in the classify that produced the mask, not in the byte write. The RUNS
edit is reverted; the measurement stays here so the next pass does not retry
it.

## The RUNS edge list, removed and still flat

The RUNS payload is a list of run lengths, and each length is the distance
from the previous transition to the one the `ctz` just found -- so the 258-entry
uint16 edge list existed only so a second pass could subtract neighbours into
the payload. Removing it (write the byte in the walk, with `last` carried)
deletes one store, one load and one subtract per run and the whole array.

Stream-identical: all four files, every cell, byte for byte. Flat on the clock.
Five alternating rounds per file at level 6, then thirty more of the minimum
(the min is the noise-free estimator on this host): `mr` 320.8 -> 320.6 MB/s,
`x-ray` 451.8 -> 453.3, `mozilla` 337.0 -> 335.8, `samba` 434.7 -> 428.6. The
suite passed on the modified build (`ALL SUITES PASS`, parallel digest
cec73f05), so the edit was correct and merely free of gain; it is reverted.

That is the third stream-identical edit on this path to measure flat, and
together they locate the cost: not the payload write, not the edge list, but
the walk itself and the classify above it. `mr` runs ~296 packed tiles per
64 KiB block (256 trial + 40 price), the trial is 51% of its encode, and its
tiles are smooth -- few transitions, so there was little list to save. On these
files classify is ~400-700 cycles per 256-byte tile, i.e. roughly half the
trial, and that, not any packer, is what the next pass has to attack.

## The Huffman LUT, priced

Asked whether the Huffman stage could be compiled without its lookup table.
It cannot be made faster that way, and the numbers below are why.

`build/hufbench` times the stage alone (32 KiB payloads, decode clocked on
payloads encoded before the clock starts):

| file      | bits/byte | encode      | decode      |
|-----------|-----------|-------------|-------------|
| dickens   | 7.86      | 1593 MB/s   | 1350 MB/s   |
| samba     | 7.82      | 1492 MB/s   | 1263 MB/s   |
| mr        | 5.29      | 1610 MB/s   | 1439 MB/s   |
| mozilla   | 7.05      | 1517 MB/s   | 1345 MB/s   |
| xml       | 8.00      | 1757 MB/s   | 1377 MB/s   |
| ooffice   | 7.60      | 1623 MB/s   | 1391 MB/s   |

At 7.9 bits per byte that is about 1.06 symbols per byte, so decode is ~2.8
cycles per symbol, and the table load is the head of that chain: index, load,
shift, next index. The entry is two bytes and the tables are static read-only
data -- there is no per-block table build to remove, which is the cost zstd's
HUF pays and this format does not. A bit-by-bit canonical walk would replace
one load with a loop of up to ten shifts and compares, which is the shape
every Huffman decoder had before LUTs for exactly this reason.

The window is already the right size. The four books have a ten-bit longest
code, so 2^10 = 1 024 entries cover every code in one level; a 15-bit window,
which is what a two-symbol entry needs (an LZ payload averages ~7 bits a
symbol, so a pair needs ~14), is 32768 four-byte entries per book, 512 KiB for
four books, and leaves L1. The two-symbol version was measured before the
format was frozen: 101% payload, because the pair almost never fits.

For scale, HUF is only reachable at level 9 and above (ZPSE_HUF_MIN_LEVEL) or
with ZPSE_ENC_HUF, and at 1.6 GB/s its encode is a few percent of a level-9
block whose LZ parse runs at 100-250 MB/s. There is no speed in this stage to
find; the level-9 and level-6 time is in the parse and the tile classify.

## Two ways to break the Huffman decode's LUT chain, both refused

The decode loop is four streams, five lookups per refill, one symbol per
lookup (`src/huf.c`). Its serial chain is index -> table load -> shift -> next
index, so the only speed left is more independent work per cycle. Two
rewrites were tried on `build/hufbench` (32 KiB payloads, decode clocked on
payloads encoded before the clock starts: 1.35-1.44 GB/s, ~2.8 cycles a
symbol).

1. Hoist the run bound. The loop recomputed its affordable-run count every
   iteration: eight integer divisions by five, four of them on output room and
   four on input room. A run consumes exactly five symbols and at most seven
   bytes, so the bound can be taken once. It is valid and it is SLOWER:
   `dickens` 1388 -> 785 MB/s, `mr` 1407 -> 613, `xml` 1405 -> 849 -- a 45%
   loss on every file, uniformly, from nothing but the loop's shape. The
   counted loop makes the compiler generate different register allocation for
   the twenty lookups, and it loses far more than the divisions cost.

2. Interleave the four streams a symbol at a time: refill all four windows,
   then twenty lookups in stream order (LOOK4) instead of five per stream
   (RUN_DO). Same symbols, same order per stream, bytes identical. Measured
   within this host's spread both ways: `ooffice` 1417 -> 1508, `samba` 1362 ->
   1408, `dickens` 1414 -> 1440, `mr` 1434 -> 1407, `xml` 1405 -> 1358,
   `mozilla` unchanged. No sign, so it is not kept.

Both are reverted. What the two together say is that the compiler already
overlaps the four chains as well as the register file allows, and that the
table load is unavoidable: at 2.8 cycles a symbol the stage is within reach of
the load-to-use latency of the lookup itself, which is the floor for any
table-driven decoder.

## The parked fast-form diff: why gating it is not a one-line fix

The parked work (tagged table epochs, the greedy fast form, page headers)
was taken apart again to answer three questions, and two of the three
answers are "no" for a reason worth writing down.

**The ASCII gate has to fire at every level.** The parked diff had narrowed it
to the full-size block at level 6 and deeper (a level term) to protect the
level ladder's time check. That is backwards: the gate does not compete with
the match finder, it only decides what the TILE side of a block costs, and
`zpse_block_lz_level` still reads the level. With
the narrowing, level 5 kept the sweep and wrote 116 bytes of tiles on a
block of two runs where level 9 took the gated path and wrote 476 of LZ2 --
a deeper rung writing a larger block, which the size ladder forbids. Every
level gated, as it did before the diff, restored that shape to 116/116.

**It is necessary and not sufficient.** Two 64 KiB shapes still inverted
after the gate was restored (balanced/deepest, bytes): the sparse
marker-every-97 shape 816/1236, and the two-run block 116/476. The trace
says why, and it is stage ORDER, not the gate: on the two-run block the
sweep's own total is 2610 at both levels, the tile-body stage compresses it
to 116, and at level 9 the token LZ stage commits 804 bytes first -- which
overwrites the tile winner, so the tile-body stage has nothing to compress
and the best the deep rung can do afterwards is 1518 from the fast form
(1236 after the Huffman wrapper). At level 5 the same 804-byte stream is
refused by the level's decode-speed guard, the tile body survives, and 116
ships. A decode-speed guard that differs by level can therefore reorder
which stage leads, and the ladder inherits the difference. Making the
tile-body stage compete after the token LZ stage instead of before it is a
real change to the block encoder's decision order, not a switch, and it
belongs in its own diff with its own measurements -- which is why this one
stays parked.

**The fast form was already a flag; the only open question was its default.**
The fast wire form was later removed; `ZPSE_ENC_LZ_FAST` remains a live option.
The diff's block-level wiring was what made the default level take its trade
implicitly. The trade is real and it
is large on prose: `dickens` 8 560 336 -> 9 109 041 bytes at 878 -> 1240
MB/s (one run each on this host, level 6, 64 KiB), i.e. +6.4% of bytes for
1.4x the rate on the one file the gate owns. `webster` is the other side:
19 772 044 -> 19 897 137 and slower in the same run (346 -> 300 MB/s). Until
the ladder and the stage order above were settled, the default level kept
the token form and the fast form stayed behind its flag (both wire forms were
later removed).

The diff itself is unchanged and still parked: the gate fix and the stage
probes used here were scratch and are not landed.

## Where a default-level block spends its cycles

The block stats carried five brackets -- the ASCII gate, the attempt-order
sample, the pricing pass, the trials and the match-finder stages -- and a
remainder that looked like bookkeeping: 26% of a level-6 `x-ray` block and
34.5% of an `mr` block. It was not bookkeeping. The tile-body pass ("v2", the
finder over the winning tile stream) and the Huffman wrap had no bracket, and
the tile-body pass is most of that remainder. With both counted, the remainder
is in the single digits on every file measured.

Level 6, 64 KiB blocks, share of encode cycles:

| file    | probe | trial | finder |  v2  | gate  | rest |
|---------|-------|-------|--------|------|-------|------|
| dickens |  1.1% |  7.4% |  60.0% | 8.3% | 13.1% | 10.0% |
| samba   |  7.0% |  6.1% |  71.0% | 7.0% |  5.0% |  3.7% |
| mozilla | 10.7% |  6.7% |  68.4% | 9.1% |  0.3% |  4.2% |
| mr      | 12.3% | 51.7% |   0.4% |28.1% |  0.4% |  6.5% |
| x-ray   | 19.1% | 54.0% |   0.0% |20.4% |  0.0% |  5.5% |

That splits what is left in two, and neither half still has a lever the
entries above have not already measured and refused. Text is the finder: one
or two left-to-right passes over the block, where prefetching the next
position and resizing the table were both tried and lost. Page-like binary is
the trial -- one or two full block encodes, with the probe's window already
cut from 16 KiB to 12 -- plus the tile-body finder, which is the same finder
over a body that has to compress by 1/32 to be kept at all, so capping it at
the size it could still win at saves a percentage point of a stage that
already stops near there. The gate is 13.1% of prose and is already an AVX2
scan over the block plus the 2 KiB sample that prices it; the scan is not
where that 13% goes.

The point of the table is the negative that was missing before: the share
that looked unexplained is named, it is necessary work, and the earlier hunt
for a cut in it was chasing a stage that was already counted as something
else.

## The CLI's own I/O is part of the wall clock

Every figure in the README measures the codec in memory, so the tool's I/O
never came up until the streamed path made the wall clock legible. Two
candidates were tried on a 12-core Windows host. One survived.

The first was to give Windows the input mapping the POSIX path has had since
the streamed rewrite. Mapping removes the copy and the fault storm the POSIX
comment blames -- but only where the buffer being filled is fresh and
frame-sized. This path reuses one 4 MiB slab buffer, so `fread` faults it once
and then serves it from the page cache; `CreateFileMapping` + `MapViewOfFile`
trades that for soft faults taken inside the encoder, which is worse. Best of
five at eight workers, mozilla / webster: 111 / 95 ms mapped against
111 / 93 ms read. Refused, and the mapping stays POSIX-only.

The second was the output buffer. Each finished block goes straight to
`fwrite`, and the default stdio buffer is 4 KiB on Windows -- a corpus-sized
frame left as thousands of `WriteFile` calls with a copy in front of each.
Handing `setvbuf` a quarter mebibyte took eight-worker mozilla from 107 to
99 ms and webster from 93 to 85 ms (minimum of seven, alternating), and
single-worker mozilla from 228 to 221 ms, webster from 192 to 181 ms. 1 MiB
and 4 MiB measured the same or a little worse on the means, so the smallest
wins. The frame is byte-identical at any buffer size: 32 comparisons over
eight corpus files at -t 1, -t 8 and -x agreed exactly.

Decode writes its blocks the same way, so it got the same treatment and
refused it: 90 -> 110 ms on webster and 111 -> 110 ms on mozilla, minimum of
seven. The decode sink keeps the default buffer, which is the useful half of
the result -- the shape of the loop predicts the win, and the measurement is
what decides it.

## Decode is I/O, and the CLI decodes one way

The encode side's remaining cost is inside the match finder; the decode side
turned out to have no such cost at all, and getting to that answer meant
correcting a figure first. Time the decode call alone, with the frame and the
output already resident, and the codec decodes 41.5 MB of webster in 20.4 ms on
one core -- 2.0 GB/s -- and 4.5 ms on twelve; mozilla's 51.2 MB goes 36.2 to
6.6 ms. The workers scale the decode 4.5 to 5.5x, so the "460 MB/s decode"
quoted earlier in this file was a disk: a CLI decode's wall clock is the read
of the frame and the write of the output, not the codec.

That measurement also settled a suspicion about the worker path. Its workers
hand the decoder the nominal block size as the room for the block, where the
serial path hands it the rest of the frame, and a decoder that tests the room
in front of every store could be much slower tight than loose -- which would
explain text blocks (match-heavy) losing far more than binary ones. Removing
the clamp measured 4.4 to 7.0 ms at twelve workers against 4.5 to 6.9 clamped,
inside the host's band at both ends, and it would let a worker write into the
block next to it, so the clamp stays and the anomaly was the host.

The CLI had a real regression on top of this: `d -t N` read the whole frame,
allocated the whole output and decoded into it, so a file paid a page fault per
4 KiB of frame and of output before the first block came out -- while the
streamed path at one worker already writes blocks out of one reused buffer.
Minimum of nine alternating runs over webster: `-t 8` went from 406 to 96 ms on
a plain frame and 353 to 95 ms on an indexed one; the binary frames were a wash
(mozilla 134 to 109, indexed mozilla 99 to 110). The workers are not the
problem -- 5 ms against 20 -- the frame held twice is. So decode now always
streams, `-t` is a compression option, and the buffered decode machinery is
gone from the tool. Decoded output was compared against the corpus for all
four frames.

## What is left in a CLI encode, and what was ruled out of it

The streamed encoder's ring depth and the tool's slab size were the two
assumptions left unmeasured, and both turned out to be no-ops. Eight slots per
worker instead of four, on webster and mozilla at eight and twelve workers:
77/79 against 79/79 ms and 98/93 against 94/97, inside the host's band, with
the frame byte-identical to the four-slot one at one, eight and twelve
workers. A slab of 128 blocks (8 MiB) rather than 64 measured the same way and
also wrote the same bytes. Both refused -- the header's claim that four slots
are enough because the writer stops being on the critical path survives
because nothing measured beat it, not because it was argued well.

With those two gone, the question was what the rest of the wall clock is. Read
a corpus file in the tool's 4 MiB slabs and the 41 MB of webster arrives in
4.9 ms (8.5 GB/s), so the input side is nothing. Stream the same resident
bytes through `zpse_compress_frame_stream` and discarding the blocks, at
1/2/3/4/6/8/10/12 workers, gives 116.5 / 61.6 / 42.7 / 31.9 / 28.1 / 23.7 /
21.2 / 25.3 ms for webster and 148.6 / 80.3 / 58.9 / 48.1 / 41.8 / 33.9 /
31.5 / 35.1 ms for mozilla. Ten workers is the peak at 5.5x; twelve is worse
because the writer makes thirteen threads on twelve cores. That peak does not
survive the round trip through the tool -- `c -t 8`, `-t 10` and `-t 12` come
out 75/75/75 ms on webster and 92/89/90 on mozilla -- which is the useful
part: the tool no longer exposes a difference the library has.

What it does expose is the process. A hello-world takes 38 ms on this host,
the CLI with no arguments takes 39, and a compress of one 64 KiB block takes
37, so about 38 ms of every CLI figure in this file is the OS starting a
process, not the codec. Subtract it and a level-6 webster encode at eight
workers is roughly 24 ms of encoder and 5 ms of reading, with writing the
19.7 MB frame costing about 2 (76 ms to `nul` against 78 to a file). There is
no lever left there either -- which is the answer this round produced, and the
reason the last four candidates were refused rather than shipped.

## The decode round that produced no second speedup

The goal for this round was twice the single-stream decode rate. The profile
that opened it said the tile path was not where any of it could come from:
tile blocks decode at 12-15 GB/s on real Silesia files, against 64 GB/s for a
block of zeros and 51 GB/s for `memcpy`, so the vectorised reconstruction is
already at the machine. What is left is the intra-block LZ command loops, at
2.0-2.4 GB/s (`lz2` on webster 2 383 MB/s over 622 of its 633 blocks, the token
form 3 833 on nci), and they set the file's rate rather than share it: 98% of
webster's bytes are in one `lz2` stream, so doubling the file means doubling
that loop. It is 1.25x behind lz4's decoder as it stands.

Three changes to the loop were tried and all three were reverted. Eliding the
dead stores a command overwrites, `restrict` on the decode pointers, and
reading the distance byte before the literal stores: each builds, each passes
`test_lz`, `test_lzh`, `test_roundtrip` and `test_hostile`, and none survives
measurement. Six alternated rounds of two binaries, medians per file, put the
decode ratios at webster 1.12, mozilla 0.83, nci 0.96, xml 0.96, dickens 0.75,
reymont 1.28 -- ±28% with no consistent sign. That is this host's run-to-run
spread, not a result, and it is far too wide to keep a change on.

The only single-core idea left with a 2x ceiling was interleaving two
independent block decodes, since a command loop is one dependency chain and a
superscalar window with two of them has more to overlap. A prototype ran the
same single-command step over two prebuilt payloads two ways, drained
sequentially and advanced alternately, so the only difference was the order:
0.854 webster, 0.817 nci, 0.851 xml, 0.892 dickens, 0.806 mozilla. Interleaving
lost on every file, by 11-19%, because the second state's bookkeeping and
branches cost more than the extra chain buys. The prototype is not the shipped
decoder -- no fast-region split, and a 16-byte match head with a byte tail --
but both drivers share that code, so the ratio is the interleaving's and the
idea is refused.

What did come out of the round is the read side. `zpse_decompress_batch_parallel`
decodes a caller-owned batch of independent blocks, and the CLI's `d -t N`
drives it while streaming a file: read a batch, decode it across the pool,
write in order. On the 202 MiB Silesia frame decoded to `/dev/null`, three
alternated rounds measured 266/273/235 ms on one worker against 209/183/175 ms
on six. The wall clock is I/O-bound -- the frame read is most of it -- so that
is a 1.35x process, not a codec ratio, but it is the first decode path here
that uses more than one core without holding the frame. A later run of the
same frame measured 201 ms against 128 ms, 1.57x; the process figures move
with the drive, which is why the question below is asked in memory.

With the payloads packed once and held (`tests/benchmarks/bench_batch_decode.c`),
the batch is pure codec, and there it scales to 4.0x -- 3 234 blocks of the
202 MiB corpus, three reps best of:

| workers | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MB/s | 1 707 | 3 362 | 4 820 | 6 213 | 6 851 | 6 840 |
| speedup | 1.00x | 1.97x | 2.82x | 3.64x | 4.01x | 4.01x |

Five workers is the peak and the sixth adds nothing: this host has six
physical cores and the caller is one of them. So the codec does take the
cores -- 7.4 GB/s of tile and command-loop decode -- and the file figure is
the read plus the process, as the `bench` numbers in the other direction have
been all along.

The other addition was `ZPSE_ENC_DECODE_FAST` (later removed), the decode-first
preset: an LZ form was kept only when it was an eighth smaller than the tile
stream, and the stage that wrapped a tile body in LZ had to clear the same
margin. Over the whole Silesia corpus at level 6 it moved blocks in four of the
twelve files (`tests/benchmarks/bench_decode_first.c`, later removed):

| file | blocks | changed | level 6 | -D | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| dickens | 156 | 0 | 8 560 304 | 8 560 304 | 0.000% |
| mozilla | 782 | 87 | 25 417 832 | 24 969 579 | -1.764% |
| mr | 153 | 135 | 3 988 520 | 4 256 995 | +6.731% |
| nci | 512 | 0 | 6 183 514 | 6 183 514 | 0.000% |
| ooffice | 94 | 4 | 4 133 008 | 4 128 235 | -0.115% |
| osdb | 154 | 0 | 8 804 209 | 8 804 209 | 0.000% |
| reymont | 102 | 0 | 3 335 151 | 3 335 151 | 0.000% |
| samba | 330 | 9 | 8 273 879 | 8 302 823 | +0.350% |
| sao | 111 | 0 | 7 194 251 | 7 194 251 | 0.000% |
| webster | 633 | 0 | 19 772 012 | 19 772 012 | 0.000% |
| x-ray | 130 | 0 | 4 952 647 | 4 952 647 | 0.000% |
| xml | 82 | 0 | 1 265 450 | 1 265 450 | 0.000% |

The frame total moves 101 881 161 -> 101 725 554 bytes, -0.15%: nearly a wash,
but not a ratio lever, because `mr` pays 6.7% of a file and samba 0.35%. The
movement goes both ways, which is what a margin gate does: 58 803 -> 65 305 on
one mozilla block, where the tile form replaces a command loop, and 30 451 ->
24 388 on another, where refusing the LZ form frees the tile stream enough for
the LZ-over-tile stage to win. Decode rates stayed inside the host's band on
every file. The flag is for a caller whose only metric is decode, not a
default.

The preset was subsequently removed from the encoder, so the table above
records what the flag once did, not what any current build does. The bit and
its `-D` CLI option were later removed outright; the per-block decode-first
trade it describes is not available today.

### Pricing the wrapped tile by its payload

The stage that wraps a tile body in LZ was the one form whose decode cost
depends on how much it wraps, so a variant priced it directly instead of by
the total: the match finder walks the payload `pl`, so accept the wrap when
`pl <= n / 8` and it is smaller at all. At the measured 2.4 GB/s command loop
and 12 GB/s tile kernel that is at most 62% more decode for the block, against
the plain tile stream. The model is right and the rule still lost: it refuses
wraps that save a third of a block. Over five files it moved 9 nci blocks
(20 293 -> 32 565 bytes, +1.84% of the file), one xml block (+3.11%), 4
dickens (+0.52%), 174 mozilla (+2.74%) -- +2.7% across the five frames. Decode
at one worker on mozilla, three rounds in each order to take the order out:
medians 100 / 98 / 97 ms (level 6 / margin / payload) with a 94-116 ms spread,
so there was nothing to buy the bytes with. Refused; the margin stayed, and
the comment beside it records why.

### Pages in a batch

The same in-memory measurement on 4 KiB pages (`ZPSE_BD_PAGE=1`: level 6 with
`PAGE_FAST`, later removed, the shape a page frame had then) covers the same
202 MiB:

| workers | 64 KiB blocks | 4 KiB pages |
| ---: | ---: | ---: |
| 1 | 1 765 | 3 203 |
| 2 | 3 308 | 4 532 |
| 3 | 4 312 | 5 498 |
| 4 | 5 347 | 6 470 |
| 5 | 6 064 | 7 289 |
| 6 | 7 297 | 7 700 |
| speedup at 6 | 4.13x | 2.40x |

Pages scale worse because they start better: a page is decoded by the tile
kernel alone, so one worker already runs at 3.2 GB/s against 1.8 for a 64 KiB
block, whose LZ blocks decode through a command loop. At six workers the page
table still ends ahead (7.7 against 7.3 GB/s), so the batching is worth the
same on a page frame -- there is just less room to divide. The 64 KiB row here
reads 4.13x where the table above says 4.01x: the same frame measured again at
best-of-3, and this host varies about 20% run to run. Read both as 4x, and
pages as 2.3-2.5x.

### The write beside the decode

The streamed decode is three phases per batch -- read the frame, decode the
batch, write the batch -- and they used to take turns on one thread. On the
202 MiB frame at six workers the split is 62 ms reading, 35 ms in the codec,
and about 100 ms writing the output (`d ... NUL` against a file): a third of
the wall clock that no worker count divides. The CLI hands each decoded batch
to a writer thread and reads the next one behind it; two batch slots are enough
because the writer is never more than one batch behind.

| frame | workers | synchronous writes | write on its own thread | read+decode floor (NUL) |
| --- | ---: | ---: | ---: | ---: |
| 202 MiB Silesia, 102 MB frame | 6 | 217-234 ms | 154-172 ms | 139-152 ms |
| mozilla, 51 MiB content | 6 | 95 ms (median) | 81 ms (median) | 74 ms (median) |
| mozilla, 51 MiB content | 1 | 116 ms (median) | 100 ms (median) | -- |

The pipelined run sits on the read-plus-decode floor in both cases, so what is
left is I/O the codec cannot hide. `decode_round.sh` reproduces it end to end:
174 ms at one worker and 131 ms at six on the same frame, best of three with
the write included. Measuring the 202 MiB frame repeatedly needed pacing:
without it the dirty pages back up and single runs reach a
second (one measured 1 107 ms, its partner 336 ms), which is writeback and not
the decoder. The table above is paced runs in alternating order, every output
compared with the corpus byte for byte. The pipeline is kept, and the comment
beside it in `zpse_cli.c` records the decomposition.

### The read beside the decode

The write moved off the decode thread; the read is what was left. On the same
202 MiB corpus the reader parses each block as it reads it -- a block's length
lives in its header, so reading ahead means parsing -- and fills one of two
batch slots while the caller decodes the other. A block is read straight into
its place in the slot, which drops the copy the walking loop made from a stack
buffer as well.

| frame | workers | read inline | read on its own thread | read+decode floor (NUL) |
| --- | ---: | ---: | ---: | ---: |
| 202 MiB Silesia, 170 MB page frame | 6 | 500 ms | 421 ms | 300-317 ms |
| 202 MiB Silesia, 170 MB page frame | 1 | 331 ms | 339 ms | -- |
| 202 MiB Silesia, 102 MB block frame | 1 | 205 ms | 185 ms | -- |
| 202 MiB Silesia, 102 MB block frame | 6 | 137 ms | 134 ms | 87-98 ms |

Where the write is bigger than the read the change is a wash, which is the
block frame at six workers: it already sits on its floor there, and the floor
is write, not read. It pays where the read dominates the run. The page frame
is the case that matters -- 170 MB of frame for 202 MiB of content, decoding at
7.7 GB/s, so the disk is the slow side of that pipeline -- and six workers go
from 500 to 421 ms there. Paced runs in alternating order, every output
compared with the corpus.

That frame is also what exposed a bug in the write pipeline. The streamed page
decode came back with two adjacent batches swapped in five runs out of ten.
The writer wrote the lowest-numbered filled slot, and
a page decode is fast enough that the caller can release a slot and refill it
before the writer looks for work, so the newer batch went out first. The
reader had the same shape: it refills a just-released slot, and the caller took
the lowest-numbered filled one, which is the newer batch when an older one is
still waiting. Both sides now take the oldest by a sequence number stamped at
handover. Forty-eight byte-identical runs at one and six workers on both
frames, plus the serial build, followed; the swap only ever appeared with pages, because a 64 KiB block
decodes slowly enough that the writer is always waiting for work. Every output
in this document is compared with its corpus, which is why a swapped batch
surfaced as a failed run instead of a wrong file nobody read.

### The same buffer at both ends

A streamed decode writes each finished block through stdio, and the default
stdio buffer is 4 KiB. That is exactly one page, so a 4 KiB page frame leaves
the writer as 51 743 separate writes; it is a sixteenth of a 64 KiB block, so
the block path copies every block through the buffer on its way out. The
encoder had already found the size that works -- 256 KiB took its output from
107 / 93 ms to 99 / 85 on mozilla / webster at eight workers -- and the reader
has the same shape: it asks for a 16 or 32 byte header and then the payload,
which for a page fills the buffer exactly.

Both streams in the streamed decode now carry 256 KiB. Pages are the case that
showed it. Medians of five alternating, paced runs, every output compared with
the corpus: the 170 MB page frame goes from 420 to 222 ms at one worker and
from 514 to 380 ms at six, while the 102 MB block frame moves 204 to 201 and
163 to 163 -- a wash, as it should be, since a 64 KiB write is larger than the
old buffer and already went straight to the file. Splitting the change says
which end did what: the output buffer alone gets pages to 297 and 431 ms, and
the input buffer is what turns that into 224 and 360, because on the page frame
the read is the slow side and every page was costing a read of its own.

The host was visibly busy during that run -- the output-to-`NUL` floor moved
between 300 and 460 ms -- but the direction held in every pair.

The tables in this section are regenerated by
`bash tests/benchmarks/decode_round.sh` with a Silesia tree in `SILESIA_DIR`:
the in-memory scaling tables for 64 KiB blocks and 4 KiB pages, the per-block
decode-first table, the level 6 against level 6|-D frame tables, and the
streaming decode with its write included.

### Why the page encode is cheaper and what is still expensive

A page spends most of its encode on deciding, not on writing, so the work that
landed here is work removed from the decision. The price probe used to pack
every sampled tile only to count the bytes it had written; the length is
already known exactly from the classifier's residual count and the mask mode's
arithmetic, so the pack is gone and the stream is byte-identical. The LZ screen
cleared a 16 KiB hash table to price a 512-byte sample, as many as four times a
page; it now clears the slots that sample can hash to. The page candidate list
built all ten transposes before cutting itself to one, so the flag that asked
for one shape (`PAGE_FAST`, later removed) was read before the list was built.
On the generated page
corpus these together buy about a quarter of FASTEST's encode and a twentieth
of the default level's, with the packed bytes unchanged to the byte.

Two other forms were tried and refused with measurements. Predictor 9 (later
removed from the format) as an extra trial on tiles the cheap screen misses
wrote about 900 fewer bytes over 16 MiB of pages but halved the PAGE_FAST encode
rate (715 to 355 MB/s), because a page has mostly noisy tiles and each one paid
a second classify. The generation-stamped hash table replaced the per-page
memset with an epoch compare and was slower still (418 against 476 MB/s at
LZ_FAST): the clear is 16 KiB that stays in L1, and the extra array costs more
than it saves. Both knobs were left in the tree, defaulted off, with the
numbers beside them (and were later removed).

Decode lost two CRC passes on every block: the header parse already validates
the header, and the page's single segment directory entry was read three times
in one decode. The parallel frame path used to re-CRC the whole index for every
block it sought, which on a page frame is tens of gigabytes of redundant CRC;
it now resolves the index once and hands workers a block-at helper. These are
byte-identical changes. Their effect is visible in the all-page decode column:
FASTEST gains about a quarter and the default level is inside the host's band,
because the remaining cost there is the inverse transform, not the envelope.

A one-value page is now a 16-byte token rather than 44 bytes of dense record,
and the page dictionary stores incompressible pages too: a repeated raw page
was a permanent miss even though its first block was still in the store. Hit
rate on the repeat corpus rises from about 18 to about 19.5% with the packed
stream unchanged.

The remaining gap to lz4 is not a missed micro-optimization. On this corpus
lz4 1 writes 2.03:1 at roughly 981 MB/s and our cheapest page mode at the time
(`PAGE_FAST`, later removed) wrote 1.52:1 at 995, while the default writes
2.40:1 at 175. A
page's decision is still a larger bill than one greedy match pass, and beating
lz4 on encode, ratio and decode at once needs a cheaper shape decision than
the sweep -- not a faster version of it.

## Profiles belong to the build, not the tree

Profile-guided optimization is the one compiler lever that cannot be reviewed
in the source: it changes the code emitted for decisions the encoder already
makes, so the packed stream has to stay byte-identical and any gain has to
come from better layout and inlining on branch-dense paths.
`tests/benchmarks/pgo_build.sh` trains the 4 KiB path on the generated page,
probe and mixed corpora, rebuilds with the profile, and with `--verify` checks
the result in the default, forced-scalar and swapped-attempt-order parity
modes. A rate table alone would not tell whether the profile had crossed from
code generation into a decision.

A profile is a build artifact, never a shipped asset. clang's `.profraw` and
gcc's `.gcda` are not interchangeable, and even the same compiler's counters
describe branch probabilities for the source they were taken from: a stale
profile spends the compiler's inlining and layout budget on a shape the code
no longer has, which is worse than no profile at all. Regenerating it from the
checked-in generators on every run is what keeps the counters tied to the tree
that consumes them; neither the format nor a block's bytes depend on them.

The build has to keep the unity translation unit and the attributes inside it
intact. `ZPSE_HOT` on gcc is a per-function `-O3` request whose 1.053x was
measured on that whole-unit shape, and the AVX2 target attributes beside it
are what let the vector kernels obey the ABI rules recorded in `hot.h`.
Adding `-fprofile-*` to the same command changes only what the compiler knows
about branch frequencies. Replacing the unity build with per-file objects, or
stripping those attributes to make a profile match, would trade a measured
gain for one nobody has measured; the parity digests are the guard that the
profile stayed a code-generation choice.

Training on generated pages left the headline workload unprobed. The original
harness prints only a 4 KiB before/after table, and no Silesia PGO delta was
ever published. `tests/benchmarks/pgo_silesia.sh` now trains the same two-stage
clang build on the twelve named Silesia files at 64 KiB and level 6, and runs
the base and profile binaries in alternating rounds on one pinned core. On this
host the profile build encoded the same packed bytes for every file at every
level and cut encode time by roughly a tenth to a quarter across twenty
alternated runs; decode improved as well. The spread between sessions was wide
enough that the honest claim is a double-digit encode win with byte-identical
output, not a precise ratio, and the A/A floor of the host still has to be
measured before any figure moves into a README row. The profile remains a build
artifact: keeping it tied to the source and regenerating it for each compiler
version is the cost of the win, and a stale profile is worse than none.

## The one-pass page form and why it is a fast path, not the whole path

The page path grew a 16-byte FLAT block precisely because the page is where
decode rates are hardest to lift: one finder pass cannot amortise a full sweep
over 4 KiB, and the tile reconstruction that buys ratio on a large block costs
more than the whole page. FLAT gives the cheap page shapes a payload that one
SIMD pass can decode -- a token for a constant page, packed 4-bit or 2-bit
indices for an irregular low-alphabet page, and a per-lane template for a
pointer page. On those shapes its kernels run at roughly 9 GB/s, and a zero or
constant page, being a 16-byte token, runs at 20-75 GB/s.

The first version made FLAT the only page path. Measured on the pinned
non-zero page corpus that collapsed the default ratio from 2.40:1 to 1.06:1:
every content page that FLAT's four codecs could not pack became a verbatim
RAW payload, and the 9 GB/s was the rate of a memcpy, not of compression.
Pure FLAT also erased the level ladder on pages -- all nine levels wrote the
same bytes. Structured low-alphabet pages made it starkest: an alternating
page that the ladder writes in 72 bytes took 1 048 as a 2-bit FLAT payload,
and a stride-record page grew from 1 565 to 2 080. Fixed-width packing throws
away exactly the run and DELTA structure the ladder prices.

The resolution is to treat FLAT as a fast path for the shapes it packs well
and fall through to the ladder for the rest. The classifier's one pass already
separates constants, low-alphabet and pointer pages from ASCII and generic
ones; a generic or ASCII page goes to the ladder with its ratio intact, and a
constant, low-alphabet or pointer page keeps the one-pass decode. This keeps
the ladder monotonic on pages (705 of 768 corpus cases shrank under the new
token and packed forms, 27 structured low-alphabet cases are re-pinned as the
cost), and it keeps the 2.28:1 default ratio close to the old 2.40:1.

The honest cost is that the mixed-corpus decode target of 5 GB/s is not met:
because most content pages take the ladder, the stream decodes at about 1.9
GB/s. Reaching 5 GB/s on those pages would need a compressible codec that is
also one SIMD pass, which FLAT's four fixed-width forms are not. lz4 1 still
has a faster mixed-corpus decode loop (3.8 GB/s). The retired `PAGE_FAST`,
`TURBO`, `PAGE_INSTANT`, `DECODE_FAST` and `LZ_HEX` flags were the old way of
buying page rate with a cruder shape decision; folding them into the level
ladder removes near-identical modes without changing what any level writes.
`ZPSE_ENC_LZ_FAST` (`-F`) remains a live encoder option.


## What a wave of small speedups bought on the ladder

A broad audit of the encoder hot paths produced a long list of individually
sound, byte-identical changes: precomputed suffix minima and pricing floors in
the trial sweep, a palette LUT in the FLAT low-alpha packers, `restrict` and
branch hints in the LZ finders, AVX2 match extension there, staged deltas in
the mode-4 residual packer, precomputed Huffman code tables, a cheaper CRC tail,
a single header parse in the frame loop, and deterministic static block
assignment in the multithreaded encoder. Every change kept the compressed
stream identical -- the def/scalar/swap digests and the no-growth size gate
all still pass.

Measured together on the pinned non-zero page corpus, they did **not** move
the rate: level-6 encode and decode stayed within the roughly 10% run-to-run
noise of the harness (median 164 -> 162 MB/s encode, 1558 -> 1568 MB/s
decode). This is the honest result, and it settles the question of where the
remaining speed has to come from: the cost is not repeated small work in the
decision loop, it is the match finder's dependent hash chain on text and the
ladder's mandatory full passes on content pages. Micro-optimizing the wiring
around those paths is dominated by the noise floor.

The one lever that did move a number is a build-time experiment,
`ZPSE_LZ_SCREEN_TEXT_GATE`, which is off by default. Enabling it lengthens the
LZ decision sample to 4 KiB on ASCII-gated 64 KiB blocks. That sample was
measured earlier at +0.13 ratio points on Silesia but was rejected because it
hurt columnar data; the ASCII gate now separates the two. With the gate on,
Silesia level 6 goes from 2.08:1 to 2.14:1 (dickens 1.19 -> 1.63, webster 2.10
-> 2.13), ladder monotonicity still holds and no size-gate case grows. The
cost is encode rate: 408 -> 294 MB/s on Silesia, because more blocks reach the
finder. It stays off by default because the shipped posture favours encode
rate; it is available for a caller whose workload is text and whose budget is
decode or storage rather than encode.

## The whole-ASCII fast path, and the ratio wall around it

Everything above about text comes back to one measurement: the match finder is
59% to 91% of a level-6 block's encode on the text files (dickens 59.3%,
mozilla 68.5%, samba 70.5%, xml 79.3%, ooffice 80.2%, reymont 90.9%,
webster 91.0%), and what it spends is a serial T<->V dependency -- each key's
table load feeds the next value comparison, so a wider datapath does not make
the chain shorter. The decoder has the opposite shape: it is already at the
machine on the tile forms, and the LZ command loops are the only thing holding
it back. That is why the place to look for more speed is a form that never
enters either pipeline: an input whose only structure is that every byte is
seven bits wide.

The finder-free kernel is cheap. Timed alone over printable ASCII, the 8-to-7
compact runs at about 17 GB/s through the AVX2 path and about 26 GB/s through
the BMI2 path on encode; it is shifts and masks with no table and no branch per
byte. FLAT7 is that kernel wrapped in a whole-input frame -- a 40-byte header
and one 7-bit-packed region -- with the inverse spread on decode. Eight bytes
always pack to seven, so its ratio is fixed at 8/7 = 1.143:1 on a full group
however the bytes are distributed. That is both its promise and its ceiling:
it cannot take advantage of redundancy, which is exactly what the finder's
gate is for, and on a low-alphabet or repeated ASCII file the ladder writes a
fraction of the bytes at a small extra cost.

The cost that is not in the kernel is the hierarchy around it. Reaching the
payload through the whole-frame path -- magic sniff, frame header, block
directory, segment and tile bookkeeping -- measures about 2.3x the decode of
calling the flat kernel on a resident payload. On a FLAT7 frame, whose only
payload is one packed region, all of that is paid to inherit the ordinary
decoder's transparent support; a direct call skips it. The speed claim has to
be made against the whole path a caller actually runs, not against the kernel.

Measured one worker at a time against the normal frame path (forced by
ZPSE_ENC_CHECKSUM, which FLAT7 declines), best-of-N, codec only: on ASCII that
clears the redundancy gate FLAT7 encodes at least 3x as fast as the normal
frame, and the whole-input speedup lands between 4.45x and 10.1x on the
eligible inputs. Decode is 1.1x to 1.4x faster there, because the pack is
already close to the kernel and the frame is the larger share of both sides.
Across a mixed corpus the total is about 1x -- the eligible frames are a small
share of the bytes, and on everything with structure the block codec wins the
ratio by too much for the pack's rate to matter. The honest claim is a niche:
take FLAT7 for high-entropy printable ASCII that nothing else compresses, and
take the ladder for the rest.

What is left is the ratio-versus-speed corner. FLAT7's 1.143:1 cannot be
lowered without giving up the fixed-width pack, and every attempt to make the
ladder decode like a flat form has run into a wall: on the whole frame, decode
does not reach about 3x while the ratio stays good, because the serial chain
in the pack and the branchy per-command decoder both sit on the critical path.
The change that would break it is not a faster version of either -- it is a
fast static-dictionary coder to supply the ratio, paired with a branchless
AVX2 decoder with one straight-line chain to issue. Until that pair exists the
two forms stay separated by the gate, and the numbers above say which side of
it a caller should land on.

## The fixed-distance form, its tight decoder, and the whole-frame fast paths

Removing the repeat-distance cache from the fast form settled a debate that had
run through the page work: the cache coded a recent distance as one byte, but
the test for it sat on every command and its escape had to be carved out of the
low byte's 1..65535 range. With it gone the fast form was one LZ4-style command
-- a token, its literals and a fixed two-byte distance -- written at every
level that offered it, from the explicit LZ_FAST option to the deepest level's
last candidate (the form was later removed). Over twelve 64 KiB Silesia blocks
decode went 2.31 -> 2.93
GB/s (+26.5%), encode +12.8% and packed size +1.366%; seven files stayed within
0.2% of their old size and the cost landed on the spike-pattern cases where a
recent distance was worth a byte often enough to miss it. That change is
recorded in full earlier; what it set up is the loop below.

The decoder then stopped being a chain of checked fields. Its trusted region
runs three pieces of LZ4's own decode shape: C, three fixed stores for a short
match at off >= 8, writing 18 bytes with no loop and no length dispatch; E,
inlined 16-byte literal stores in 32-byte steps instead of a libc call; and B,
a bounded escape decoder that chains at most one byte on a hostile 255. Two
guards, one on output room and one on input position, replace the per-field
tests, and the last 560 bytes take the fully checked tail. The streams decode
byte-exact on all twelve Silesia blocks and on the hostile corpus. The C+E+B
loop measures faster than the checked loop and narrows the gap to lz4 on this
host; an earlier harness that gave the stream 512 bytes of slack measured an
even larger gain, and the shipped loop trades part of that for the 560-byte
output guard. A same-host in-memory comparison put lz4 only modestly ahead of
the decoder, not by the factor of two that published numbers from other
machines suggested -- that "twice as fast" reading came from comparing
different builds. B is inside the measurement noise; the gain is C and E, with
B kept because a hostile escape must not break the guard. Precise A/B rates for
these mirrors live in the temp harness logs, not in a script under
tests/benchmarks/, so they are not quoted here as project figures.

Beside the block LZ forms, four whole-input fast paths sat outside the ladder
at the time (a fifth, lzfast, was later removed), each with a gate and a magic
of its own, and each declining back to the ordinary codec. FLAT7 (F7P1) is the
finder-free 8-to-7 ASCII pack described above: its 1.143:1 ratio is a ceiling,
so the redundancy gate keeps repetitive ASCII on the ladder. lzfast was the
smallest useful finder (later removed) -- the fast form's command shape with the
lazy probe, backward walk and repeat slots removed -- and it traded a little
ratio for a cheaper encode; the 13-bit table was the default because it was the
faster of the two sizes tried. repack (ZRPK) maps a
low-alphabet ASCII buffer to dense w-bit codes for w = 1..7, uses a 16-byte
inverse map at w <= 4 and a 128-byte one above that, and at w = 7 is
byte-for-byte the FLAT7 pack; non-ASCII input is declined. rlefast (RLE1)
commits only when at least 90% of the first 64 KiB fall in maximal runs of four
or more bytes whose value is 0..127, so it is a narrow codec for run-structured
data, not for small integers in general. f64fast (F64F) pins transform 9
(DELTA_TRANSPOSE_8) with no sweep and no LZ_TILE wrap for float64 arrays whose
lane-high bytes come from a small, strongly dominated alphabet. The gates are
what keep these from being general codecs: each declines the input it was not
designed for, and the ladder writes the stream for everything else. The precise
A/B rates that decided these gates live in temp harnesses, not in a script under
tests/benchmarks/, so this section records the format, the gate and the reason
rather than quoting rates as project figures.

## The token decoder gets a trusted region too

The token form's decoder (`zpse_lz_decode`, then in `src/lz.c`, later removed)
was the last command loop still written as a chain of per-field checks. Its
earlier fast path was an escape-free region inside the careful loop: a command
qualified only when both token nibbles were under fifteen, which fit in
eighteen payload and thirty-two
output bytes, and even then it re-tested its room on every command. The fast
form (`src/lz2.c`) had already shown the better shape a trusted region ahead of
the checked loop, with two bounds tested once per command and a fully checked
tail for everything the region cannot take; that change narrowed the gap to
LZ4 and, with the fixed-distance command, made its decode markedly faster.

The token decoder was given that shape. `op < n - 576` (RESERVE, room for the
widest command -- a 269-byte literal run and a 274-byte match plus store slack)
bounds the output and `lp <= tend - 352` (BACK) bounds the speculative reads.
A 255-escape continuation, a fourth distance group, or a run wider than the
guard bails to the checked tail with `lp` rewound to the command's own start, so
full escape semantics and every rejection still live in one place. The stream is
unchanged by construction; only how fast the same bytes are rebuilt moved.

An in-memory A/B on the twelve 64 KiB Silesia blocks found token-heavy decode
-- mozilla, ooffice, samba and nci, the files whose blocks the token form
dominates -- faster by a clear margin, byte-exact on every block and on a corpus
of hostile, truncated and corrupt inputs, with an A/A null comparison to bound
the host's noise. That is the region removing per-command tests from the common
command, not a ratio trade: no packed size moved. The harness was temporary and
was never checked in as a script under `tests/benchmarks/`, so the measurement
is recorded here as a design note rather than as a project figure for the
README.

## A one-byte-per-delta form for the x-ray's 16-bit samples

The 16-bit medical image in Silesia has a shape the ladder cannot exploit
cheaply: after a 240-byte header, successive samples differ by a small signed
amount, so the information is in the deltas rather than in the sample values.
The new whole-frame form stores exactly that -- a fixed previous-sample
predictor, one signed byte for every delta in [-127,127], and a sorted escape
table (u32 delta index, i16 true delta) for the rare deltas outside that range.
The leading raw byte prefix is copied verbatim instead of being escaped as
samples, so prediction starts at the first real sample and the 240-byte header
survives byte for byte. Decode is a byte read, an SSE2 inclusive scan over eight
16-bit lanes and a scatter of the escapes; there is no mask, residual,
transform or Huffman stage behind it. That is why it is a fast form for one
input shape rather than a general codec.

Nothing is emitted unless the caller opts in with ZPSE_ENC_DPCM1 (bit 17) on
zpse_compress_frame_ex. Even then two gates decide. A content gate accepts only
data whose sampled deltas are at least 90% int8 fits with mean |delta| at least
18: smooth input such as Silesia "mr" is already cheaper under PSE1 and must
fall through to the ladder. The exact encoded length is computed in a first
pass, before any output byte is written, and the form declines unless that
length is no larger than the raw input and fits the caller's buffer. The
decoder validates magic, length, parity, prefix and escape count before
touching output, requires the byte count to match exactly, and rejects escape
positions that are out of range or not strictly increasing, so a hostile stream
cannot steer the scatter past the delta window.

Measured on the x-ray file, the frame went from 4,952,679 B (1.711:1) to
4,390,381 B (1.930:1). On a same-host alternating A/B with core 5 pinned,
decode measured 2.711x faster than the normal frame path, and the result was
byte-exact on all twelve Silesia files: the x-ray is the only input this gate
accepts, and the other eleven decline and keep their existing streams. The
measurement came from a temporary harness and was never checked in as a script
under tests/benchmarks/, so no figure from it is quoted in the README.

## Committing the fast form in the finder-first pass

The finder-first commit in `block.c` (around line 5729) runs before the sweep and
the tile stages have priced anything, so whatever it commits wins the block
outright. On `nci` the finder and the tile forms price within 0.2% of each
other, which means that pass is deciding the block's form on a rounding error --
and the token LZ form it committed there was the mistake. When two forms are
that close, committing the one whose decoder is a command loop instead of the
tile kernel pays a large decode bill for almost no ratio. The experiment
committed LZ2, the fixed-distance fast form (later removed), in that pass instead, on the theory
that a form already priced inside the noise could be swapped for the one that
decodes about twice as fast at no ratio cost.

On `nci` it did what the theory said. On the tree at that commit (HEAD 83217f7),
in memory on core 7 with QPC, best of nine, arms alternating, and an A/A null
comparison putting the noise floor at 0.12%: packed goes 6,184,688 -> 6,008,734
bytes (-2.84%, a better ratio), encode +9.88% and decode +114.99% -- about
2.15x. The round trip was byte-exact on all twelve Silesia files.

That is the whole win, and it does not travel. The toggle was global rather
than content-gated, so eight of the twelve files changed size: `mozilla` +6.19%
(+1.58 MB) and `ooffice` +7.78% (+321 KB) regressed, the same prose trade the
earlier parked fast-form discussion settled by keeping the token form as the
default and the fast form behind its flag. Worse, the columnar `sparse_mix`
ladder broke: level 5 -> 6 grew 8,516 -> 9,106 bytes where it is supposed to
shrink. That is the documented stage-order and guard-bypass hazard from that
same discussion -- a decode-speed guard that differs by level lets the
committed form step in front of the tile-body stage and decide first, and the
level ladder inherits the difference.

So the idea is discarded rather than parked as a switch. Shipping it needs two
separate things: a content gate that fires only on the 16-bit structures `nci`
is made of, so the other eleven files keep their existing streams; and the
stage-order fix, so committing a form early can no longer bypass the tile-body
stage and invert the ladder. With both in place it may be worth revisiting,
because the `nci` number is real. The A/B was run in a temporary harness that
was never checked in as a script under `tests/benchmarks/`, so the figures above
are a design note, not a project number; no table in the README was updated.

## Skipping the final LZ pass on a canonical-LZ winner

The final LZ stage in `block.c` is priced last and commits only on a strict
`t2 < best_total`. When the current winner is already a canonical `ZPSE_B_LZ`
stream, that pass cannot win: at MAX the lazy LZ winner falls through the gate
near the finder-first path with `best_idx` 15, and the fast screen's stream
never beat it, so the stage calls the same stateless LZ4 encoder on the same
bytes and can only reproduce the size already in hand. The guard parses the
winner's block header and skips the stage when its flag is plain LZ. Short
circuiting it as the last conjunct keeps the parse off blocks that never reach
the stage at all.

The skip is byte-identical: all 36 Silesia outputs (twelve files at levels 1,
6 and 9) hash the same with and without it, and the L6 and L9 totals are
unchanged. On a saturated host in one environment it bought +3.59% encode at
level 9 across 5/5 timing pairs, against 3-4x noise, and was inert at level 6
where the guard fires on zero or one block per file. An earlier form keyed on
`best_idx == 15` was rejected: that index also marks other commit paths, so it
would skip the stage for blocks that were not canonical LZ. Testing the flag
precisely leaves the `LZ_TILE` and `HUF` forms untouched, which is why the
outputs stay identical.

## Zero-run RLE on the smooth 16-bit image (MRD)

DPCM1 made Silesia's x-ray a one-byte-per-delta stream, but its gate was drawn
so that `mr` -- whose 16-bit samples are smooth enough that the ladder's sparse
masks already price them cheaply -- would decline: a fixed byte per sample buys
nothing there. MRD keeps DPCM1's previous-sample predictor, one signed byte for
each delta in [-127,127] and a sorted escape table (u32 delta index, i16 true
delta) for the rare larger ones, and adds the one thing `mr` has that x-ray
does not: long stretches of *equal* samples. A run of at least sixteen
consecutive zero deltas is written as a single three-byte token (marker 0x80
plus a u16 run length) and rebuilt by splatting the held sample, so a flat
stretch costs three bytes rather than one per sample and no bitmap at all. The
marker is int8 -128, outside the fitting range, so neither a literal delta nor
an escape placeholder (0x00) can collide with it. Decode keeps DPCM1's SSE2
inclusive scan over stretches of literal deltas and adds a 16-bit splat for the
runs; there is no mask, residual, transform or Huffman stage behind it.

Nothing is emitted unless the caller sets ZPSE_ENC_MRD (bit 18) on
zpse_compress_frame_ex, and MRD is tried last, after DPCM1. Its content gate is
the first thing it does: at least 95% of a sample of deltas must fit int8, the
mean absolute delta must lie in [4,14), and at least 5% of the sampled deltas
must be zero -- the room the RLE token needs. `mr` clears it (mean |delta|
about 11), the x-ray declines at 25, and the other ten files decline on the fit
or mean test. A decline writes nothing: the whole-frame flag bits are stripped
before the ordinary frame encoder runs, so its PSE1 stream is byte-for-byte the
one produced without the flag. The experiment used a runtime arm toggle so one
binary could emit both arms for measurement; the landed code has no toggle --
selection is the flag plus the content gate, as with f64fast and DPCM1.

The encoder measures the exact control-stream length in a first pass and
declines unless it is no larger than the input and fits the caller's buffer.
The decoder validates magic, original length, parity, prefix, escape count,
control length and RLE-token count before touching output, requires the byte
count to match exactly, checks every escape position is in range and strictly
increasing, and bounds each run by the samples still to be written. Hostile
frames -- bad magic, truncation, a run past the end or below the minimum, an
out-of-range or non-monotonic escape, an escape count above ns-1, odd n, a
short or overflowing control length, and an RLE count above coff/3 -- are all
rejected cleanly with a guard page after the output buffer, so no write can
escape the caller's length.

Measured on `mr`, the frame went from 3,988,552 B (2.50:1) to 3,832,738 B
(2.60:1). On a same-host alternating decode A/B (QPC, one pinned core, best of
nine, with an A/A control) the MRD frame decoded 2.4-2.8x faster than the PSE1
frame, and the round trip was byte-exact on all twelve Silesia files: `mr` is
the only input this gate accepts, and the other eleven keep their existing
streams. The measurement came from a temporary harness and was never checked
in as a script under `tests/benchmarks/`, so no figure from it is quoted in the
README.

## The ratio ceiling, and what it would cost to reach it

The target that keeps resurfacing -- 4.28:1 on Silesia, about 49.5 MB from the
211,938,580-byte corpus, at roughly 520 MB/s encode and better than 3.6 GB/s
decode -- is not a tuning problem. It is a point on the frontier this codec's
shape cannot occupy, and a campaign of measurements said why in several
different ways.

Start with the ratio in isolation. The best packing any general codec achieves
on this corpus is 47.56 MB (4.456:1), and LZMA-9e alone reaches 48.46 MB
(4.374:1), so 49.5 MB is physically reachable -- but only in the 5 to 30 MB/s
encode band those codecs live in. The target is a ratio that belongs at the
other end of the frontier. At the speed end, around 500 MB/s, the state of the
art is zstd-3 at 3.20:1 (66.2 MB); zpse today writes about 2.14:1 (99.1 MB)
with its text sample gate on. Closing that gap means cutting packed bytes by
about 35 percent at constant speed, and none of the obvious levers found a
third of the stream.

Entropy coding the vendored LZ4 parse was first, because it keeps the parse and
attacks only the bytes. Structural bytes are 55.7 percent of LZ4's output and
the offset stream carries H0 = 12.18 and H1 = 6.76 bits of entropy, so there is
real skew to exploit. The floors say how much: order-0 coding tops out at 2.48:1,
a 16.6 percent saving, and order-1 at 2.87:1, a 27.9 percent saving -- both
short of 35. The literals alone would have to reach 4.67:1 before the tokens
stopped mattering, and that is a ceiling on the literal stream, not a result.
A perfect entropy coder over the existing parse therefore cannot deliver the
third that is needed.

A word/token coder was the attempt to do better by modelling words rather than
bytes. Its best measured rates were 2.73 bits per character on dickens (2.93:1)
and 3.72 on webster (2.15:1). The ratio is attractive on the first file and
ordinary on the second, but the decoder is the disqualifier: word-ID indirection
is a serial chain of pointer chases, and it runs at 200 to 600 MB/s, six to
eighteen times short of the 3.6 GB/s target. The ratio is bought with exactly
the dependency structure the decode target forbids.

Static and frozen context models failed from a different direction. They land
between 1.5:1 and 3.4:1 depending on how well the frozen model matches the
genre, so the good number is a property of the corpus, not of the model. The
64K-context table that produces it is 32 MiB, which does not stay resident on
this host; the ratio arrives with a table walk that misses at every level. A
static dictionary of general English words was the cheap version of the same
idea, and its net saving was 2.58 percent on dickens, 0.71 percent on webster
and 0.19 percent on mozilla -- noise on all but the most prose-heavy file.

The `sao` assumption deserves its own correction. It is sometimes treated as
the JPEG-like entry in Silesia and assigned a 1.7 MB target. It is raw
float-like data, not JPEG: its order-0 entropy is 7.525 bits per byte and the
best LZMA result on it is 4.43 MB. It cannot reach 1.7 MB under any parse, and
it is not the binding constraint on the corpus anyway. The honest conclusion
is a trade, not a near miss: every mechanism that reaches 4.3:1 here is serial
and cache-hostile and runs in the tens of MB/s, while the decode target wants a
wide, branch-light kernel. They are different machines.

## More byte-identical speed candidates, refused at the noise floor

Four more candidates joined the file's existing record of changes that were
correct and not worth keeping. Each left the stream identical; none cleared
this host's run-to-run spread.

`>>5` for the noise divisor was proposed as a cheaper stand-in for a divide by
32. clang's interprocedural constant propagation had already folded that divide
into a shift, so the emitted assembly was identical and there was nothing to
win. The candidate was dropped at the assembly check rather than on the clock.

`seg_prescreen` marked `always_inline` genuinely inlined at all four call sites,
produced byte-identical output in 48 of 48 comparisons and passed the suite. It
measured +1.4 percent encode, but the sign was mixed across ABBA quartets and
the magnitude sits inside the host's noise; a change that needs a sign to be
believed does not have one.

Screen-sample dedup, reusing a sample encode already computed when the same
sample came round again, hit only 196 of 3,164 sample encodes at level 6. The
+0.28 percent encode it bought is the arithmetic of that hit rate and nothing
more; the other 2,968 samples still pay for their own encode.

The 64-byte encoder staging stagger shifted where staging writes land to avoid
a store conflict. Its deltas were inside the noise with inconsistent signs, and
the decode rate -- which the change cannot touch, since it is encoder-side --
moved by plus or minus 1.8 percent. That ±1.8 percent on a path the edit cannot
reach is the host's noise, quoted here so the other ±1 to 2 percent figures in
this file can be read against it.

## The transform extension prototypes, verified but not integrated

The one substantive deliverable of this campaign was a set of typed transforms,
built and checked but never wired into the codec. Fourteen length-preserving
transforms were implemented and assigned IDs 12 through 25: BITSHUFFLE at 16,
32 and 64 bits, ZIGZAG at 16, 32 and 64, WORD_XOR at 32 and 64, DELTA2 at 32 and
64, NIBBLE_TRANSPOSE, and BSWAP at 16, 32 and 64. Alongside them, six
variable-length block-form prototypes were written: FOR, patched FOR, Gorilla
float, front-coding, general RLE, and a combined zigzag-transpose form. Every
one was verified byte-for-byte across the scalar, SSE2 and AVX2 paths under
sanitizers, so the work is correct; it is also separate from the shipped codec.

The artifacts live in the temp directory rather than the tree:
`C:\Users\dboyn\AppData\Local\Temp\opencode\txf_*` for the length-preserving
transforms and `C:\Users\dboyn\AppData\Local\Temp\opencode\proto_*` for the
block-form prototypes. Each directory carries a `README.txt` recording the
exact wire layout, the gate that decides when the form is offered, and the
measured results, so a future implementer can integrate a form by reading its
own directory instead of reconstructing the format from this file. They are not
integrated: nothing in `src/` offers these IDs, and the frame format does not
carry them.

The reason they stayed out is the same reason the ratio ceiling above holds.
These forms target typed integer, timestamp and float data, where a zigzag or a
frame-of-reference step turns an awkward high-entropy stream into a narrow one.
Silesia is text-heavy, and its files do not have that shape, so the transforms
are unlikely to move the headline ratio even once integrated. Their value is
conditional -- they are the right tools if a typed-data workload appears --
which is a different bet from the one the README's tables measure.

## BSM1: one whole-frame bitshuffle form for the MR shape

The block-level BITSHUFFLE_16 trial did lower mr's packed size, but it also
forced the MR-shaped data away from its whole-frame codec and into a 64 KiB,
256-tile block envelope. The envelope was the design choice, not a property of
the transform, so the final form moves the transform to the frame level: an
eighth whole-frame magic, BSM1, carries a 16-byte header and one LZ4 block of
bitshuffled bytes, and decoding is one inverse straight into the caller's
buffer. Its gate needs level 5 or above, the existing 16-bit zero-heavy delta
probe, and a frame of at least 1 MiB, so the other eleven Silesia files keep
their exact compressed streams and decoded bytes.

The packed-size change is deterministic and does not depend on the clock: mr
goes from 3,832,738 to 3,349,842 bytes (2.60:1 to 2.98:1), and the twelve-file
total from 99,099,878 to 98,616,982 bytes. That is 482,896 bytes, or 0.487
percent of the corpus. The README tables are deliberately left untouched by
this change: every published rate has to come from a regeneration by
`tests/benchmarks/silesia.sh` on an idle host, and this host is shared with
other work, so the rates measured alongside the new form are not publishable.
Byte identity was checked directly instead: 36 of 36 corpus round trips and
all twelve decoded Silesia frames match, and eleven compressed streams are
unchanged.
