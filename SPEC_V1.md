# PSE Block Format v1.0.0

Status: Normative Implementation Specification (zpse)
Codec Family: Predict–Select–Enumerate (PSE)
Supersedes: v0.1.1 (Corrected & Hardened)

Primary Objective: General-purpose, block-parallel lossless compression
without LZ77 back-references and without Huffman, arithmetic, range, ANS, or FSE coding.

Normative Language: MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, MAY per RFC 2119.

## 0. Changelog from v0.1.1 (normative corrections)

v0.1.1 contained load-bearing inaccuracies. v1.0.0 corrects them; where
v0.1.1 and v1.0.0 disagree, v1.0.0 governs.

1. **Decoder arena size (Section 12): was false.** v0.1.1 claimed
   "192 KiB decoder arena" but the tabulated layout summed to
   256 KiB (0x00000–0x3FFFF). v1 fixes the claim to **256 KiB** and
   provides a contiguous, exactly-summing layout (Sec. 12).
2. **Encoder arena undefined.** v0.1.1 claimed "384 KiB encoder arena"
   but never defined it. v1 defines it (Sec. 12.2).
3. **Combinadic C-table size: was false.** v0.1.1 allocated 20 KiB for a
   "Precomputed Pascal Triangle". Full `C(n,k)` for `n <= 255`,
   `k <= 256` as `u256` (32 B) is `256*257*32 = 2,105,344 B (~2 MiB)`,
   which cannot fit in 20 KiB and cannot live in a per-block arena
   without blowing the memory bound. v1 removes the full table from the
   arena. Decoders MUST either (a) use a global read-only table (not
   counted against the arena), or (b) compute the needed row on demand
   with `u256` add/sub only in 8 KiB of arena scratch (Sec. 8.2, 12).
   Reference on-demand algorithm is given.
4. **ORDER2 confidence update: was non-functional.** v0.1.1 rule
   `conf <- hit ? min(conf+1,255) : (conf==0 ? 1 : conf-1)` with
   replacement condition `(conf==0 & !hit)` never fires after the first
   miss (0 -> 1 -> stays >= 1), so `pred` freezes after init. v1 fixes
   the rule to a standard saturating-confidence replacement (Sec. 7.1,
   ID 4).
5. **Segment Directory format: was missing.** v0.1.1 required
   `segment_count * 8` bytes of directory and a
   `cumulative segment sizes == uncompressed_size` check, but never
   defined the 8-byte entry. v1 defines it: `{ uint32_le offset;
   uint32_le comp_size }` (Sec. 4.4).
6. **Tile flags bit layout: was ambiguous.** v0.1.1 wrote
   `[mask_mode:3, residual_mode:3, all_exc:1, reset:1]` without bit
   positions, and `all_exc` duplicated mask modes 0/6. v1 fixes bit
   positions (bits 0–2 mask, 3–5 residual, 6 reserved-0, 7 reset) and
   deletes `all_exc`; the 256-exception case is encoded via
   `mask_mode == 6` with `exception_count == 0` (Sec. 6.2, 8).
7. **`exception_count == 256` encoding: was contradictory.** v0.1.1 said
   "0 with all_exc=1 means 256" while also allowing `exception_count`
   0..255 as a count. v1 normatively defines: `K = exception_count==0 ?
   (mask_mode==0 ? 0 : mask_mode==6 ? Nt : INVALID) : exception_count`,
   with cross-checks against `Nt` (Sec. 6.2, 8).
8. **TileRecord size accounting: was ambiguous.** v1 defines:
   `tile_record_size == 8 + (predictor==9 ? 1 : 0) + mask_payload_size +
   residual_payload_size`, and the 2-byte prefix is excluded (Sec. 6.2).
9. **TILE_LOCAL_CONSTANT (predictor 9) payload: unaccounted.** v1 counts
   the 1-byte `C` as part of the TileRecord (Sec. 6.2, 7.1).
10. **STRIDE_TRANSPOSE segment_count: was contradictory.**
    Table said `1..16`, block header said `(1,2,4,8,16)`. v1 requires
    `segment_count IN {1,2,4,8,16}` for all transforms, and
    `W == segment_count` for transpose family (Sec. 5).
11. **Enumerative rank range check: was insufficient.** Checking only the
    final remainder `R == 0` accepts out-of-range ranks that underflow
    mid-loop. v1 additionally requires the input rank `< C(Nt,K)` and
    rejects overlong (non-canonical) encodings and non-zero padding
    (Sec. 8.1, 11.1).
12. **Residual sub-modes / parameters: were unencodable.** SIGNED_SMALL_DELTA
    cited 4/6/8-bit sub-modes with no signal; PALETTE_FIXED_BITS cited
    `M` with no encoding; PACKED_BASE_M cited `M` and chunk group `G`
    with no table. v1 defines explicit parameter bytes and packing order
    (Sec. 9.1–9.2).
13. **SECONDARY_MASK (residual mode 5): was unimplementable**
    ("filtered through a secondary predictor layer"). v1 reserves
    residual modes 5–7: decoders MUST reject them (Sec. 9.1).
14. **Degenerate N=0 test: was illegal.** Block size is 1..65536, so
    `N=0` is not a valid block. v1 conformance matrix starts at N=1
    (Sec. 13).
15. **CRC32-C definition: was missing.** v1 defines reflected CRC-32C
    (Castagnoli, poly 0x1EDC6F41), init 0xFFFFFFFF, xorout 0xFFFFFFFF
    (Sec. 3.1, 4.1).
16. **Version handling.** v1 bumps `version_major` to 1,
    `version_minor` to 0, `block_version` to 1. Major 0 decoders MUST
    reject v1 frames and vice versa (Sec. 14).
17. **ORDER1 table size waste.** 8*256 = 2048 B was charged 4 KiB.
    v1 charges exactly 2 KiB (plus alignment pad absorbed into the next
    region's alignment, Sec. 12).

---

## 1. Scope, Design Constraints, Execution Philosophy

PSE models structured redundancy deterministically using finite-context
linear lanes, physical stride predictors, coordinate transforms, and
combinadic/radix packing. It saturates memory bandwidth on superscalar
SIMD architectures while enforcing hard safety guarantees.

### 1.1 Non-Functional Requirements

- **Bounded Memory:** working memory allocated strictly at context
  creation: **256 KiB decoder arena, 384 KiB encoder arena** (Sec. 12).
- **Zero Allocation:** no dynamic allocation (malloc, mmap, new) during
  block encode or decode.
- **Independent Blocks:** every block decodes independently; no
  inter-block state. `RESET_PREDICTORS` (block flag bit 3) MUST be 1.
- **Deterministic 8-Lane Model:** predictions advance in 8 interleaved
  logical streams `p(mod 8)`, matching vector register topologies.
- **Safe Untrusted Decoding:** complete bounds checking and bitstream
  validation before entering unrolled vector kernels (Sec. 11).

### 1.2 Prohibited Primitives

- No LZ77/LZSS sliding-window matches or arbitrary backwards copies.
- No entropy-coding state machines (Huffman, FSE, tANS, rANS, CABAC,
  Arithmetic, Range).
- No C++ exception handling (throw/catch), no stack unwinding, no
  dynamic dispatch in inner loops.

## 2. Zero-Overhead Execution Model & Microarchitectural Invariants

Tile/packet loops MUST NOT incur function-call overhead, register
clobbering, or stack-frame allocation.

```
TRADITIONAL DISPATCH (SLOW):
Block -> Tile -> [Call Predictor] -> [Call Mask] -> [Call Residual]
~3-5 calls per 256 B tile = 15-25 cycles/tile (10-20% throughput loss)

PSE MONOLITHIC PIPELINE (FAST):
Header validated -> monolithic unrolled kernel -> branchless SIMD write
0 calls inside segment hot loop. Registers pinned in AVX/Neon banks.
```

### 2.1 Invocation ABI & Register Pinning

- Inner decoders MUST be compiled with `-fno-exceptions`,
  `-fno-asynchronous-unwind-tables`, `-fno-rtti`, `-fomit-frame-pointer`.
- Inner tile reconstructors, predictors, mask parsers, unpackers MUST be
  `static inline __attribute__((always_inline))` (`__forceinline` on MSVC).
- No virtual dispatch or function-pointer tables in tile loops.
  Specializations over `(predictor_id, mask_mode, residual_mode)` MUST use
  flattened `switch` / contiguous jump tables over cache-aligned blocks.
- x86-64/AVX-512 pinning (advisory): ymm0/zmm0–ymm3/zmm3 lane histories;
  ymm4/zmm4–ymm7/zmm7 active predictions; k1–k4 exception masks;
  r12–r15 payload-in / output / residual / arena-base pointers.
- Hot loops MUST NOT use `alloca`/VLAs/large stack structs. All scratch
  lives in the 64-byte-aligned arena.

## 3. Top-Level Binary Layout

```
+---------------------------+
| FrameHeader (32 bytes)    |
+---------------------------+
| Block 0 Envelope          |
+---------------------------+
| ...                       |
+---------------------------+
| Block N-1 Envelope        |
+---------------------------+
| [Optional FrameIndex]     |
+---------------------------+
```

All multi-byte integers are little-endian.

### 3.1 FrameHeader (32 bytes)

| Off | Size | Field              | Type   | Description                              |
|-----|------|--------------------|--------|------------------------------------------|
| 0   | 4    | magic              | u32    | 0x31455350 ("PSE1")                      |
| 4   | 1    | version_major      | u8     | MUST be 1                                |
| 5   | 1    | version_minor      | u8     | MUST be 0                                |
| 6   | 2    | frame_flags        | u16    | Bitmask (Sec. 3.2)                       |
| 8   | 4    | nominal_block_size | u32    | 1..65536                                 |
| 12  | 8    | content_size       | u64    | Total bytes, or UINT64_MAX if unknown    |
| 20  | 4    | block_count        | u32    | Total blocks, or UINT32_MAX if unknown   |
| 24  | 4    | frame_index_offset | u32    | Offset to FrameIndex, 0 if absent        |
| 28  | 4    | header_crc32c      | u32    | CRC32-C (Castagnoli, see below) of 0..27 |

CRC32-C: reflected, polynomial 0x1EDC6F41, init 0xFFFFFFFF,
xorout 0xFFFFFFFF, check("123456789") = 0xE3069283.

### 3.2 Frame Flags

- Bit 0 `BLOCK_CRC32C_PRESENT`: every block carries payload CRC.
- Bit 1 `CONTENT_SIZE_VALID`: `content_size` valid.
- Bit 2 `BLOCK_COUNT_VALID`: `block_count` exact.
- Bit 3 `FRAME_INDEX_PRESENT`: index record present at
  `frame_index_offset` (relative to frame start).
- Bits 4..15 reserved, MUST be 0; non-zero => `PSE_UNSUPPORTED_FEATURE`.

FrameIndex format is out of scope for block v1 (opaque to block
decoder); if present, `frame_index_offset` MUST point inside the frame
and MUST NOT overlap any block envelope.

## 4. Block Envelope & Header Invariants

```
+----------------------------------------------+
| BlockHeader (32 bytes)                       |
+----------------------------------------------+
| Transform Metadata (transform_meta_size B)   |
+----------------------------------------------+
| Segment Directory (segment_count * 8 bytes)  |
+----------------------------------------------+
| Concatenated Segment Payloads                |
+----------------------------------------------+
```

### 4.1 BlockHeader (32 bytes)

| Off | Size | Field                 | Type | Description                          |
|-----|------|-----------------------|------|--------------------------------------|
| 0   | 2    | sync                  | u16  | 0x50B6                               |
| 2   | 1    | block_version         | u8   | MUST be 1                            |
| 3   | 1    | block_flags           | u8   | Bitmask (Sec. 4.2)                   |
| 4   | 4    | uncompressed_size     | u32  | 1..65536                             |
| 8   | 4    | compressed_block_size | u32  | Total block bytes incl. header       |
| 12  | 1    | transform_id          | u8   | 0..11 (12..255 MUST be rejected)     |
| 13  | 1    | segment_count         | u8   | MUST be in {1,2,4,8,16}              |
| 14  | 2    | transform_meta_size   | u16  | Bytes following header               |
| 16  | 2    | tile_size             | u16  | MUST be 256                          |
| 18  | 1    | predictor_profile     | u8   | MUST be 0                            |
| 19  | 1    | reserved0             | u8   | MUST be 0                            |
| 20  | 4    | block_sequence_number | u32  | SHOULD increase by 1 from 0; decoder SHOULD verify when BLOCK_COUNT_VALID but MUST NOT abort on gap alone |
| 24  | 4    | decoded_crc32c        | u32  | CRC32-C of decoded bytes (if flag)   |
| 28  | 4    | header_crc32c         | u32  | CRC32-C of bytes 0..27               |

`decoded_crc32c` MUST be verified when block flag bit 2 or frame flag
bit 0 is set; otherwise decoders SHOULD verify but MUST NOT abort
solely on mismatch of an untagged field — instead treat as mismatch
only when a CRC-present flag is set.

### 4.2 Block Flags

- Bit 0 `RAW_BLOCK`: raw fallback (Sec. 4.3).
- Bit 1 `LAST_BLOCK`: final block of frame.
- Bit 2 `BLOCK_CRC32C_PRESENT`: verify `decoded_crc32c`.
- Bit 3 `RESET_PREDICTORS`: MUST be 1 in v1.
- Bit 4 `HAS_TRANSFORM_METADATA`: == (`transform_meta_size > 0`).
  Mismatch => `PSE_BAD_HEADER`.
- Bits 5..7 reserved, MUST be 0.

### 4.3 Raw Block Fallback Invariant

For input size N: `compressed_block_size <= 32 + N`. If the compressed
form would be `>= 32 + N`, the encoder MUST emit `RAW_BLOCK`
(bit 0 = 1, `transform_id` = 0, `segment_count` = 1,
`transform_meta_size` = 0, no segment directory, no segment header —
exactly N raw bytes at offset 32). Decoders MUST accept this shape and
MUST still verify CRC/length invariants.

### 4.4 Segment Directory (new in v1; was undefined)

Immediately after transform metadata, `segment_count` entries of 8 bytes:

```
struct PseSegmentDirEntry { uint32_le payload_offset; uint32_le payload_size; };
```

- `payload_offset`: byte offset of this segment's payload relative to
  the start of the Concatenated Segment Payloads area (i.e. first
  payload is offset 0). MUST be 8-byte aligned? No — byte-aligned; MUST
  satisfy `payload_offset + payload_size <= payloads_area_size`.
- `payload_size`: bytes of the segment payload (incl. its 16-byte
  SegmentHeader).
- Entries MUST be in segment index order, non-overlapping, gapless, and
  sorted by offset: `entry[0].payload_offset == 0`, and for `s > 0`
  `entry[s].payload_offset == entry[s-1].payload_offset +
  entry[s-1].payload_size`.
- Sum of `payload_size` MUST equal
  `compressed_block_size - 32 - transform_meta_size -
  segment_count*8`. Violation => `PSE_BAD_SEGMENT_DIRECTORY`.
- Sum over segments of `segment_decoded_size` (from each SegmentHeader)
  MUST equal `uncompressed_size`. Violation => `PSE_BAD_SEGMENT_DIRECTORY`.
- RAW_BLOCK blocks have no directory (`segment_count == 1` but directory
  is omitted; payload is raw bytes).

## 5. Transform Pipeline

| ID | Name               | segment_count | Metadata                                  |
|----|--------------------|---------------|-------------------------------------------|
| 0  | RAW_BYTES          | 1             | none (0 B)                                |
| 1  | DELTA8             | 1             | 1-B seed                                  |
| 2  | XOR8               | 1             | 1-B seed                                  |
| 3  | TRANSPOSE_2        | 2             | none                                      |
| 4  | TRANSPOSE_4        | 4             | none                                      |
| 5  | TRANSPOSE_8        | 8             | none                                      |
| 6  | TRANSPOSE_16       | 16            | none                                      |
| 7  | DELTA_TRANSPOSE_2  | 2             | 2-B seeds (1 per segment, index order)    |
| 8  | DELTA_TRANSPOSE_4  | 4             | 4-B seeds                                 |
| 9  | DELTA_TRANSPOSE_8  | 8             | 8-B seeds                                 |
| 10 | DELTA_TRANSPOSE_16 | 16            | 16-B seeds                                |
| 11 | STRIDE_TRANSPOSE   | {1,2,4,8,16} | 2 B `[W:u8, flags:u8]` + K seeds iff DELTA flag |

- `segment_count` MUST match the table row; any other combination
  (e.g. ID 3 with count 8) => `PSE_BAD_TRANSFORM`.
- `transform_meta_size` MUST equal the table's byte length; mismatch =>
  `PSE_BAD_HEADER`. `HAS_TRANSFORM_METADATA` MUST agree (Sec. 4.2).
- ID 11: `W` MUST equal `segment_count` and be in {1,2,4,8,16}; `W==1`
  means no interleave (equivalent to RAW_BYTES segmentation).
  `flags`: bit 0 `DELTA` (if 1, `W` seed bytes follow the 2-byte prefix);
  bits 1..7 reserved-0. If `DELTA==0`, `transform_meta_size==2`; if
  `DELTA==1`, `transform_meta_size==2+W`.
- DELTA8: byte `i` encodes `out[i] = (in[i] + seed) mod 256` chained?
  No — v1 normative: `t[0]=in[0]^seed`-family is per-segment initial
  delta: `enc[0] = (in[0]-seed) mod 256`, `enc[i]=(in[i]-in[i-1]) mod 256`.
  XOR8: `enc[0]=in[0]^seed`, `enc[i]=in[i]^in[i-1]`. Decode inverts.
  DELTA_TRANSPOSE_* applies the same per segment with that segment's seed.

### 5.1 Stride Transpose Invariants & Clamping

For width `W in {2,4,8,16}`: input byte `i in [0,N-1]` maps to segment
`s = i mod W` at position `j = floor(i/W)`. Exact segment length:

```
segment_length[s] = floor((N + W - 1 - s) / W)
```

- If `N < W`, encoder MUST NOT select TRANSPOSE_W / DELTA_TRANSPOSE_W
  (equivalently: every emitted segment MUST satisfy
  `segment_uncompressed_size >= 1`; empty segments are forbidden).
- Decoder MUST verify each `segment_decoded_size` against this formula
  given `N` and `W` (for IDs 3–11 with W>1); mismatch =>
  `PSE_BAD_SEGMENT_DIRECTORY`.
- For IDs 0–2, 11-with-W=1: single segment length MUST equal N.

### 5.2 Hot-Loop Inverse Transpose (reference; scalar tail required)

8-way inverse interleave consumes 8 B from each of 8 lanes and emits 64
interleaved bytes (AVX2 unpack matrix, see v0.1.1 reference routine —
unchanged). Conforming decoders MUST handle tails where
`N mod W != 0` (or `N mod 64 != 0`) with a scalar/partial-vector epilogue
and MUST bounds-check `segs[s] + seg_offset` per lane. The AVX2 kernel
alone is not a complete implementation.

## 6. Segment Payloads & Tile Framing

```
+----------------------------------------------------+
| SegmentHeader (16 bytes)                           |
+----------------------------------------------------+
| u16 tile_record_size[0] + TileRecord_0 (variable)   |
+----------------------------------------------------+
| ...                                                |
+----------------------------------------------------+
| u16 tile_record_size[M-1] + TileRecord_M-1         |
+----------------------------------------------------+
```

`tile_record_size` is the byte length of the immediately following
TileRecord (excluding the 2-byte prefix itself).

### 6.1 SegmentHeader (16 bytes)

| Off | Size | Field                | Type | Description                          |
|-----|------|----------------------|------|--------------------------------------|
| 0   | 4    | segment_decoded_size | u32  | Decoded bytes for this segment       |
| 4   | 2    | tile_count           | u16  | MUST equal ceil(decoded_size / 256)  |
| 6   | 1    | initial_literal      | u8   | Predictor seed literal               |
| 7   | 1    | segment_flags        | u8   | bit0 LEN_PREFIX=1, bit1 DIR=0, 2..7=0|
| 8   | 4    | tile_directory_size  | u32  | MUST be 0                            |
| 12  | 4    | reserved             | u32  | MUST be 0                            |

- `segment_flags != 0x01` => `PSE_UNSUPPORTED_FEATURE`.
- `tile_count == 0` => `PSE_BAD_TILE_SIZE`. `tile_count` MUST equal
  `ceil(segment_decoded_size/256)`; `segment_decoded_size==0` is
  forbidden (no empty segments, Sec. 5.1).

### 6.2 TileRecord Header (8 bytes fixed)

| Off | Size | Field                 | Type | Description                          |
|-----|------|-----------------------|------|--------------------------------------|
| 0   | 1    | tile_flags            | u8   | bits 0–2 mask_mode, 3–5 residual_mode, bit 6 reserved-0, bit 7 reset |
| 1   | 1    | predictor_id          | u8   | 0..11 (12..255 => PSE_BAD_PREDICTOR) |
| 2   | 1    | tile_length_minus_1   | u8   | 0..255 => length 1..256              |
| 3   | 1    | exception_count       | u8   | See K-derivation below               |
| 4   | 2    | mask_payload_size     | u16  | Mask section bytes                   |
| 6   | 2    | residual_payload_size | u16  | Residual section bytes               |

Bit layout of `tile_flags` (v1, normative):

```
bit 0..2 : mask_mode      (0..6 valid, 7 => PSE_BAD_MASK)
bit 3..5 : residual_mode  (0..4 valid, 5..7 => PSE_BAD_RESIDUAL)
bit 6    : reserved, MUST be 0
bit 7    : reset — SHOULD be 1 (lane histories re-seeded from
           initial_literal at tile start when predictor needs history);
           decoder MUST honor 0/1 as stated (0 = continue lane histories
           across tiles within the segment; 1 = re-seed). Block-level
           RESET_PREDICTORS=1 forces reset at segment start regardless.
```

TileRecord total layout:

```
[8-B header][1-B C iff predictor_id==9][mask_payload_size B mask][residual_payload_size B residual]
```

Normative size equation (decoder MUST enforce):

```
tile_record_size == 8 + (predictor_id==9 ? 1 : 0) + mask_payload_size + residual_payload_size
```

Normative length derivation: `Nt = tile_length_minus_1 + 1`.

- Interior tiles `t in [0, tile_count-2]`: `Nt MUST == 256`.
- Final tile: `Nfinal = (segment_decoded_size mod 256 == 0) ? 256 :
  (segment_decoded_size mod 256)`; `tile_length_minus_1 MUST ==
  Nfinal - 1`.
- Sum of `Nt` over tiles MUST equal `segment_decoded_size`, else
  `PSE_BAD_TILE_SIZE`.
- Decoder MUST verify `tile_record_size <= segment bytes remaining`
  before parsing (Sec. 11).

Exception-count derivation (v1, normative). Let `ec = exception_count`:

```
if (mask_mode == 0) { K = 0; REQUIRE(ec == 0); }
else if (mask_mode == 6) { K = (ec == 0 ? Nt : ec); REQUIRE(K == Nt); }
else { K = ec; REQUIRE(1 <= K && K <= Nt-1
         -- note K==0 and K==Nt are forbidden under modes 1..5); }
```

Additionally: mode 0 with `K>0`, mode 6 with `K != Nt`, modes 1..5 with
`K==0` or `K==Nt` => `PSE_BAD_MASK`. The legacy `all_exc` flag bit does
not exist in v1; bit 6 is reserved-0.

### 6.3 Dense Segment Record (defined in v1.1)

Tile framing pays one header, one mask section and one residual section
per `Nt` bytes. A segment whose plane is homogeneous is cheaper as ONE
record covering up to 4 KiB of the plane: one predictor, one mask over
that stretch, one residual stream. Since `tile_count` is 1 for such a
segment, the record shape is:

| Off | Size | Field                 | Type | Description                        |
|-----|------|-----------------------|------|------------------------------------|
| 0   | 2    | record_size           | u16  | Bytes AFTER this field (header tail + payloads) |
| 2   | 1    | dense_flags           | u8   | bit 6 marker, bits 0–2 mask_mode, bits 3–5 residual_mode, bit 7 reserved-0 |
| 3   | 1    | predictor_id          | u8   | 0..11 (12..255 => PSE_BAD_PREDICTOR) |
| 4   | 1    | constant_C            | u8   | used iff predictor_id == 9        |
| 5   | 1    | reserved              | u8   | MUST be 0                          |
| 6   | 2    | exception_count       | u16  | K                                  |
| 8   | 2    | mask_payload_size     | u16  | Mask section bytes                 |
|10   | 2    | residual_payload_size | u16  | Residual section bytes             |
|12   | ...  | mask payload, then residual payload                  |

Identification (normative): a segment payload whose byte at offset 18
has bit 6 set carries dense records; its `tile_count` MUST be 1 and the
record area MUST consist of them and nothing else. The records are laid
out back to back, and each one's length is fixed by the segment size
rather than carried separately: record `i` codes exactly
`min(4096, segment_decoded_size - 4096*i)` bytes, so a 4 KiB plane is one
record, a 64 KiB segment is sixteen 4 KiB records followed by nothing,
and a 4796-byte segment is a 4096-byte record and a 700-byte one. A
decoder MUST derive the length that way and MUST reject a payload whose
records do not add up to `segment_decoded_size` or leave a byte over
(`PSE_BAD_TILE_SIZE`). Bit 6 is reserved-zero in a TileRecord
(Sec. 6.2), so no byte string is both a legal tile record and a legal
dense record: a decoder that accepts the tile stream keeps accepting
exactly the same streams, and old decoders reject dense records as
reserved-bit violations rather than mis-reading them.

Size equation (decoder MUST enforce):

```
record_size == mask_payload_size + residual_payload_size + 10
record_size >= 10
the next record starts at 2 + record_size
```

Prediction. Positions are RECORD-relative (`d = p`), and every record is
self-contained: it restarts from the entry state, so a plane of several
records is coded 4 KiB at a time and no record reads a byte a previous
one reconstructed. The predictor formulas are those of Sec. 7 with `d`
in place of `seg_pos + p`. In particular `STRIDE_S` uses
`initial_literal` for `d < S` and the record's own `out[d - S]`
otherwise, and LANE_DELTA / LANE_AVERAGE read the two PREVIOUS values of
the lane, both `initial_literal` at the record start. Histories are
updated with the reconstructed (actual) byte.
`ORDER1` and `ORDER2` are reserved inside a dense record in v1
(`PSE_UNSUPPORTED_FEATURE`): they are the only predictors that read the
segment tables, and no encoder emits them.

Mask: modes 0 (ALL_HIT), 1 (RAW_BITMAP), 2 (SPARSE_U8) and 6
(ALL_EXCEPTION) only; modes 3, 4, 5, 7 are reserved
(`PSE_UNSUPPORTED_FEATURE`). K derivation is as in Sec. 6.2 with
`Nt = record length`, except that SPARSE_U8 (one byte per
position) exists only when `record length <= 256`
(`PSE_BAD_MASK` otherwise), and the bitmap is `ceil(Nt/8)` bytes with
pad bits above `Nt` zero.

Residual: modes 0..3. Mode 4 (SIGNED_DELTA) is reserved
(`PSE_UNSUPPORTED_FEATURE`) because its decoding needs the prediction at
each exception before the reconstruction reaches it.

Limits: each record codes between 1 and 4096 bytes, and a plane longer
than that is a sequence of records (see Identification). A record whose
length would be 0 or above 4096 is not reachable, because the length is
derived from `segment_decoded_size`.

What this encoder emits (non-normative, and the only use of the form in
v1): a plane that is a single byte value `v` end to end, where the dense
record is strictly smaller than the all-hit tile records it replaces
(12 bytes per 4 KiB against 10 per 256 bytes, so one-tile planes keep
their tile). The record is then `K == 0` with mask ALL_HIT, no residual,
and predictor CONST_ZERO when `v == 0`, STRIDE_1 when
`v == initial_literal`, TILE_CONSTANT with `C == v` otherwise -- all
three reconstruct as a fill of `v`. Every other plane is coded as tiles,
byte for byte as before this form existed.

Status mapping for a malformed dense record: size/spans
`PSE_BAD_TILE_SIZE`; mask shape `PSE_BAD_MASK`; residual payload
`PSE_BAD_RESIDUAL`; reserved bit, field, mode, predictor or limit
`PSE_UNSUPPORTED_FEATURE`.

## 7. Predictor Execution Engine (Profile 0)

8 parallel lanes; transformed position `p in [0,Nt-1]` maps to
`lane = p mod 8`, `lane_idx = floor(p/8)`.

```
Pos : 0  1  2  3  4  5  6  7  8  ...
Lane: L0 L1 L2 L3 L4 L5 L6 L7 L0 ...
```

Lane histories `lane_last[8]`, `lane_second_last[8]`, seeded from
`initial_literal` at each reset point (segment start, and tile start iff
tile reset bit = 1): both histories `<= initial_literal` for all lanes.
`ORDER1`/`ORDER2` tables likewise reset to `initial_literal`-derived
state (ORDER1 entries = `initial_literal`; ORDER2 pred entries =
`initial_literal`, conf = 0) at the same points.

STRIDE_PHYSICAL predictors (IDs 5–8 and 11) use already-decoded bytes at
`p - S` (`S in {1,2,3,4,8}`) within the current segment's transformed
stream; for `p < S`, prediction = `initial_literal`. (This is a
forward-referenced causal read, not an LZ back-reference: offset set is
fixed {1,2,3,4,8}.)

### 7.1 Predictor IDs & Formulas

- **0 CONST_ZERO:** `x_hat = 0`.
- **1 LANE_PREVIOUS:** `x_hat = lane_last[lane]`.
- **2 LANE_DELTA:** `x_hat = (2*lane_last[lane] - lane_second_last[lane]) mod 256`.
- **3 ORDER1_TABLE:** `x_hat = order1[lane][lane_last[lane]]`;
  after decoding actual `x`: `order1[lane][lane_last] <- x`.
- **4 ORDER2_HASH (corrected):** per-lane 4096-entry tables `pred`, `conf`:
  `idx = ((lane_second_last*251) XOR (lane_last*17)) AND 0xFFF`.
  `x_hat = pred[lane][idx]`; `hit = (x == x_hat)`;
  `if (hit) conf[lane][idx] = min(conf+1, 255);`
  `else { if (conf[lane][idx]==0) { pred[lane][idx]=x; conf[lane][idx]=1; } else conf[lane][idx]--; }`
  (Replacement happens exactly when confidence is exhausted.)
- **5..8 STRIDE_PHYSICAL S=1,2,4,8:** `x_hat = (p>=S) ?
  transformed[p-S] : initial_literal`. ID5 S=1, ID6 S=2, ID7 S=4, ID8 S=8.
- **9 TILE_LOCAL_CONSTANT:** `x_hat = C`, where `C` is the 1 byte after
  the 8-byte tile header (counts toward `tile_record_size`, Sec. 6.2).
- **10 RESERVED_ZERO?** v0.1.1 allowed 0..10 but defined only 0–9.
  v1 defines **10 CONST_LITERAL_0x00? No — v1 defines 10 as
  LANE_AVERAGE:** `x_hat = (lane_last[lane] + lane_second_last[lane] + 1) >> 1`
  (integer average, byte domain).
- **11 STRIDE_PHYSICAL S=3 (STRIDE_3):** `x_hat = (p>=3) ?
  transformed[p-3] : initial_literal`. Added after v1's first draft for
  interleaved 3-byte records (RGB pixels, xyz vectors); it behaves exactly
  like IDs 5–8 with S=3, including the tile-start and history rules above.
  IDs 12..255 => `PSE_BAD_PREDICTOR`.

History update (all predictors): after decoding actual `x` at lane L:
`lane_second_last[L] <- lane_last[L]; lane_last[L] <- x`. ORDER1/ORDER2
updates additionally as above. Updates apply to hit and exception lanes
alike, in position order.

### 7.2 Vectorized Prediction Hot Loop (reference)

LANE_DELTA over 64 B: `pred = last + (last - prev) mod 256` via
`_mm256_sub_epi8` / `_mm256_add_epi8` (unchanged from v0.1.1 reference).
All 8-lane predictors MUST produce bit-identical output to the scalar
formulas in Sec. 7.1.

## 8. Exception Mask Engines

`mask_mode` (tile_flags bits 0–2):

| ID | Name            | Description                                              |
|----|-----------------|----------------------------------------------------------|
| 0  | ALL_HIT         | K==0, zero mask payload bytes                            |
| 1  | RAW_BITMAP      | `ceil(Nt/8)` bytes; bit `p` (LSB-first, byte `p>>3` bit `p&7`)=1 => exception |
| 2  | SPARSE_U8       | K ascending u8 positions (REQUIRE Nt<=256; each < Nt; strictly ascending) |
| 3  | SPARSE_GAPS     | Gap-encoded: u8 `W` in {1,2} width flag? See below; then gap stream |
| 4  | RUNS            | Alternating hit/miss run lengths (see below)             |
| 5  | ENUMERATIVE     | Combinadic rank (Sec. 8.1)                               |
| 6  | ALL_EXCEPTION   | K==Nt, zero mask payload bytes                           |
| 7  | Reserved        | MUST be rejected (`PSE_BAD_MASK`)                        |

- Mode 1: `mask_payload_size MUST == ceil(Nt/8)`; unused high bits of
  the last byte MUST be 0, else `PSE_BAD_MASK`. Total set bits MUST
  equal K, else `PSE_BAD_MASK`.
- Mode 2: `mask_payload_size MUST == K`; positions strictly ascending
  and `< Nt`, else `PSE_BAD_MASK`.
- Mode 3: payload = `u8 gap_width_flag` (0 => gaps are u8, 1 => gaps are
  u16le) followed by K gap values. First position = gap[0], subsequent
  `pos[i] = pos[i-1] + 1 + gap[i]` (so gap = number of intervening hits).
  All positions `< Nt`, ascending by construction; `mask_payload_size`
  MUST equal `1 + K*(flag?2:1)`. New in v1 (was underspecified).
- Mode 4: payload = alternating `u8` run lengths starting with a hit run:
  `runs[0]=hits, runs[1]=misses, ...`, summing to Nt; number of misses
  MUST equal K; each run in 1..255 with a 0-escape? v1 normative: runs
  are u8, value 0 is forbidden except a single terminating 0 is
  forbidden too — instead runs longer than 255 MUST be split
  (e.g. 300 hits = [255, 0-miss, 45-hit, ...] with explicit zero-length
  alternate run). Decoder MUST verify sum == Nt and miss-sum == K.
- Modes 0/6: `mask_payload_size MUST == 0`.

### 8.1 Mode 5: Enumerative Combinadics (Colexicographic Rank)

For K-subset `0 <= c1 < c2 < ... < cK < Nt`:

```
Rank = SUM_{j=1..K} C(c_j, j),  with C(n,k)=0 for n<k.  (Macaulay/colex)
```

- `K==0` or `K==Nt`: forbidden under mode 5 (use modes 0/6).
- Payload length: `L = ceil(log2(C(Nt,K)) / 8)` bytes, little-endian
  unsigned rank. `L==0` never occurs under mode 5 (K in 1..Nt-1 implies
  C>=Nt>=2 for Nt>=2; for Nt==1 no valid K exists under mode 5 —
  encoder MUST NOT emit mode 5 when Nt==1).
- Decoder MUST enforce: `mask_payload_size == L`; numeric rank `<
  C(Nt,K)` (range check BEFORE unranking); minimal-length encoding is
  NOT required, but any payload bytes at or above the minimal length
  that encode a value `>= C(Nt,K)` => `PSE_BAD_ENUMERATIVE_RANK`;
  trailing zero bytes beyond L are impossible (size check covers it).
- After unranking loop, remainder MUST be exactly 0, else
  `PSE_BAD_ENUMERATIVE_RANK` (Sec. 8.2).

Maximum rank width for Nt<=256 is C(256,128) (~2^252), i.e. <= 32 bytes
(4x u64 limbs). No allocation: unranking works in-place on the payload
plus 8 KiB arena scratch.

### 8.2 Combinadic Unranking (corrected hot loop)

v0.1.1's in-arena full Pascal table is removed. v1 requires ONE of:

- (a) global static const table (not arena-counted), or
- (b) on-demand row computation in arena scratch (reference below):
  iteratively build `C(n, j)` for the running `j` with `u256` additions
  only, scanning `current_pos` downward. Only O(K) `u256` values
  (<= 256*32 B = 8 KiB) are live at once.

Reference `u256` ops (`gte`, `sub` over 4 limbs) are unchanged from
v0.1.1. Unranking loop is unchanged except for the ADDED pre-check
`R < C(N,K)` and rejection of `current_pos < 0` mid-loop as
`PSE_BAD_ENUMERATIVE_RANK` (not a generic failure). Implementations MUST
not use division in the hot loop.

## 9. Residual Mapping & Packing

Exception at `p` guarantees `x != x_hat`. Exclusion mapping projects to
[0,254]:

```
forward:  residual = (x < p_hat) ? x : x - 1
inverse:  x        = (residual < p_hat) ? residual : residual + 1
```

A decoded residual value 255 is forbidden in every residual mode except
RAW_ACTUAL_BYTES (mode 1). Decoder MUST abort with `PSE_BAD_RESIDUAL`
on `residual >= 255` elsewhere (note: multi-bit packings already bound
values < 255 by construction; the check applies after unpacking).

`residual_payload_size` MUST equal the exact encoded length defined per
mode below; overlong/short => `PSE_BAD_RESIDUAL`.

### 9.1 Residual Modes (tile_flags bits 3–5)

- **0 EXCLUSION_U8:** K sequential exclusion bytes ([0,254] each).
  Length MUST be K.
- **1 RAW_ACTUAL_BYTES:** K raw actual bytes `x` ([0,255] all legal).
  Length MUST be K. (The 255-forbidden rule does NOT apply.)
- **2 PALETTE_FIXED_BITS (new spec):** payload =
  `u8 M` (2..16; 0,1,17..255 => error) + `M` palette bytes (each an
  exclusion residual in [0,254]; 255 => error) + packed indices
  `ceil(K * ceil(log2(M)) / 8)` bytes, LSB-first bit order (index 0
  occupies the least-significant bits of byte 0), zero-padded tail bits
  MUST be 0. Total length MUST be `1 + M + packed`. Indices `>= M` =>
  error. (Palette holds exclusion residuals, not raw bytes; reconstruct
  via Sec. 9 inverse.)
- **3 PACKED_BASE_M (new spec):** payload = `u8 M` with
  `M in {3,5,6,7,9,10,11,12,13,14,15}` + packed words per Sec. 9.2.
  Indices are exclusion residuals scaled? No — indices index a sorted
  implicit alphabet? v1 normative: values ARE the exclusion residuals
  and REQUIRE `M == max_residual_used + 1`? No — simpler and checkable:
  each unpacked digit MUST be `< M` AND `<= 254`; digit `== 255`
  impossible since M<=15. Encoder SHOULD choose smallest valid M
  covering its residuals; decoder enforces `digit < M` only.
- **4 SIGNED_SMALL_DELTA (new spec):** payload = `u8 submode` (0 => 4-bit,
  1 => 6-bit, 2 => 8-bit; else error) + packed deltas.
  `d = ((x - x_hat + 128) mod 256) - 128`; `d==0` is invalid (hit lanes
  carry no residual) => `PSE_BAD_RESIDUAL`.
  Packing: 4-bit mode packs two deltas per byte (low nibble first);
  each nibble encodes `d in [-8,7]\\{0}` via `enc = d & 0xF`
  (so 0x0 encoding => error). 6-bit mode packs 4 deltas per 3 bytes
  LSB-first bit stream; each 6-bit unit `enc = d & 0x3F` must decode to
  `d in [-32,31]\\{0}` (0x00 => error). 8-bit mode stores K raw bytes
  `enc = d mod 256` (0x00 => error). Tail padding bits MUST be 0.
  Lengths: 4-bit `ceil(K/2)`, 6-bit `ceil(6*K/8)`, 8-bit `K`, plus 1
  submode byte.
- **5..7 reserved:** decoder MUST return `PSE_BAD_RESIDUAL`
  (`PSE_UNSUPPORTED_FEATURE` also acceptable but implementations SHOULD
  use `PSE_BAD_RESIDUAL` for tile-level errors).

### 9.2 Base-M Tail Packing

Chunk size G per M (chosen so `M^G <= 2^32`; words are u32le):

```
M:3->G=20 (3^20<2^32) | 5->13 | 6->11 | 7->10 | 9->9 | 10->9 | 11->8
12->8 | 13->8 | 14->7 | 15->7   -- encoder MUST use this table.
```

`num_groups = ceil(K / G)`. Full groups pack `G` digits LE-digit-first:
`word = SUM_{i=0..G-1} d_i * M^i`. Tail group with `g = K mod G`
(`1 <= g < G`) packs `word_tail = SUM_{i=0..g-1} d_i * M^i` and MUST
satisfy `floor(word_tail / M^g) == 0` (upper bits zero); non-zero =>
`PSE_BAD_RESIDUAL`. Payload after the `M` byte is `num_groups * 4` bytes.

## 10. Monolithic Reconstruction Kernels (reference)

AVX-512 `RAW_BITMAP + EXCLUSION_U8` kernel and AVX2 `pshufb` compaction
kernel are unchanged from v0.1.1 as non-normative illustrations. They
require VBMI2 (`vpexpandb`) / AVX2+POPCNT respectively with scalar
fallbacks. Normative behavior is defined by Secs. 7–9; SIMD paths MUST
be bit-exact to scalar references.

## 11. End-to-End Decoder State Machine & Validation

```
PHASE 1 STRUCTURAL: header CRC -> bounds/offsets -> transpose constraints
PHASE 2 ARENA INIT: base pointers; histories/tables <- initial_literal
PHASE 3 KERNELS: per segment, per tile: length check -> mask -> predict ->
                 reconstruct -> update tables (all in L1/arena, no alloc)
PHASE 4 INVERSE TRANSFORM: de-interleave planes (SIMD matrix + tail)
PHASE 5 CHECKSUM: CRC32-C over decoded bytes; compare iff tagged; return status
```

### 11.1 Verification Checklist (abort codes)

- `magic != 0x31455350` => `PSE_BAD_MAGIC`.
- Header CRC mismatch => `PSE_BAD_HEADER_CRC`.
- Decoded-payload CRC mismatch (when block flag bit 2 or frame flag bit 0
  tags it) => `PSE_BAD_PAYLOAD_CRC` (v1.0.1; previously aliased to
  `PSE_BAD_HEADER_CRC`).
- `transform_id >= 12` => `PSE_BAD_TRANSFORM`; `segment_count` not in
  {1,2,4,8,16} or mismatched to transform => `PSE_BAD_TRANSFORM`.
- `uncompressed_size > 65536` or `< 1` => `PSE_BAD_BLOCK_SIZE`;
  `compressed_block_size` inconsistent with actual input span =>
  `PSE_BAD_BLOCK_SIZE`.
- Segment directory overlaps/gaps/sum mismatch, or
  `SUM(segment_decoded_size) != uncompressed_size`, or transpose-length
  formula violation => `PSE_BAD_SEGMENT_DIRECTORY`.
- Tile length derivation violation or `SUM(Nt) != segment_decoded_size`
  or `tile_record_size` equation/overrun => `PSE_BAD_TILE_SIZE`.
- `predictor_id >= 12` => `PSE_BAD_PREDICTOR` (dense record: Sec. 6.3).
- Dense record with `tile_count != 1`, wrong `record_size`, or
  `mask_payload_size + residual_payload_size` mismatch =>
  `PSE_BAD_TILE_SIZE`.
- `mask_mode == 7`, K-derivation failure, bitmap popcount mismatch,
  non-ascending/OO-range sparse positions, bad gaps/runs, non-zero pad
  bits => `PSE_BAD_MASK`.
- Rank `>= C(Nt,K)` pre-check failure, mid-loop underflow, or final
  remainder `!= 0` => `PSE_BAD_ENUMERATIVE_RANK`.
- Residual 255 outside mode 1, `d==0` in mode 4, bad palette/M/G/digit,
  non-zero tail words/bits, length mismatch => `PSE_BAD_RESIDUAL`.
- Any reserved bit/field non-zero (frame/block/segment/tile reserved,
  `reserved0`, `tile_directory_size != 0`, STRIDE flags reserved bits) =>
  `PSE_UNSUPPORTED_FEATURE`.
- New-reserved residual modes 5..7 => `PSE_BAD_RESIDUAL`.

Status-code enum (C ABI, negative values are errors; 0 = OK) is defined
in `pse_status.h` (to be created with the implementation); the names
above are normative strings.

## 12. Working Memory Layout & Stack Invariants

No allocation during block encode/decode. Arenas are caller-owned,
64-byte aligned.

### 12.1 Decoder Arena (256 KiB = 0x40000; contiguous, exact)

| Offset            | Size  | Purpose                                              |
|-------------------|-------|------------------------------------------------------|
| 0x00000–0x00FFF   | 4 KiB | Context control block & arena pointers               |
| 0x01000–0x10FFF   | 64 KiB| Transformed block reconstruction buffer (max 65536)  |
| 0x11000–0x20FFF   | 64 KiB| Inverse-transpose / output staging (max 65536)       |
| 0x21000–0x217FF   | 2 KiB | ORDER1 table (8 lanes x 256 B, exact)                |
| 0x21800–0x297FF   | 32 KiB| ORDER2 value table (8 x 4096 B)                      |
| 0x29800–0x317FF   | 32 KiB| ORDER2 confidence table (8 x 4096 B)                 |
| 0x31800–0x337FF   | 8 KiB | Combinadic on-demand scratch (<= 256 u256 values)    |
| 0x33800–0x347FF   | 4 KiB | Shuffle masks & CRC32-C constants                    |
| 0x34800–0x3FFFF   | 46 KiB| Guard & alignment cushion (MUST be addressable)      |
| **Total**         | **256 KiB (262144 B)** |                                          |

Check: 4096+65536+65536+2048+32768+32768+8192+4096+47104 = 262144.
All region bases are 64-byte aligned (0x21800 % 64 == 0, etc.).
The v0.1.1 "192 KiB / 20 KiB Pascal / 4 KiB ORDER1 / 32 KiB guard"
tabulation was arithmetically inconsistent and is superseded by this table.

### 12.2 Encoder Arena (384 KiB = 0x60000; contiguous, exact)

| Offset            | Size   | Purpose                                             |
|-------------------|--------|-----------------------------------------------------|
| 0x00000–0x00FFF   | 4 KiB  | Context control block & arena pointers              |
| 0x01000–0x10FFF   | 64 KiB | Input window (max block 65536)                      |
| 0x11000–0x20FFF   | 64 KiB | Transform staging (transposed planes)               |
| 0x21000–0x30FFF   | 64 KiB | Candidate/output staging (worst-case expansion)     |
| 0x31000–0x40FFF   | 64 KiB | Best-output staging (kept across transform trials)  |
| 0x41000–0x417FF   | 2 KiB  | ORDER1 working table                                |
| 0x41800–0x497FF   | 32 KiB | ORDER2 value working table                          |
| 0x49800–0x517FF   | 32 KiB | ORDER2 confidence working table                     |
| 0x51800–0x537FF   | 8 KiB  | Combinadic rank-computation scratch                 |
| 0x53800–0x547FF   | 4 KiB  | Histogram / palette / radix frequency scratch       |
| 0x54800–0x557FF   | 4 KiB  | Shuffle LUT (pshufb 256x8 = 2 KiB) + CRC constants  |
| 0x55800–0x5FFFF   | 42 KiB | Guard & alignment cushion                           |
| **Total**         | **384 KiB (393216 B)** |                                         |

Check: 4096+4*65536+2048+32768+32768+8192+4096+4096+43008 = 393216.

### 12.3 Stack Usage Bounds

- Max stack per block execution frame MUST NOT exceed 2048 bytes.
- All large structures reside in the arena.
- Call depth from API entry to SIMD inner loop MUST NOT exceed 2 frames.

## 13. Conformance Test Matrix

Scalar-vs-SIMD bit-exactness REQUIRED. Minimum cases:

| Category         | Invariants (corrected)                                        |
|------------------|---------------------------------------------------------------|
| Degenerate sizes | N = 1, 2, 7, 8, 9, 15, 16, 17, 255, 256, 257, 65536 (N=0 removed: illegal) |
| Constant vectors | repeating 0x00, 0xFF, 0x5A, alternating 0x00/0xFF             |
| Pathological     | K=0 (ALL_HIT), K=Nt (ALL_EXCEPTION), K=1, K=Nt-1 per mask mode 1..5 |
| Combinadic edges | rank 0, rank C(N,K)-1, rank C(N,K) (MUST fail), rank with non-zero pad (MUST fail) |
| Base-M boundary  | K mod G == 0 vs 1, 2, G-1; tail-word non-zero high bits (MUST fail) |
| Delta overflow   | wrap across 0/255 for LANE_DELTA, SIGNED_SMALL_DELTA, DELTA8  |
| Hostile streams  | mutated lengths, bad CRCs, invalid tail bits, bad directory gaps/overlaps, transpose-clamp violations, reserved-bit violations |

## 14. Versioning & Extensibility

- Major 1: v1 decoders MUST accept `version_major==1,
  version_minor==0`, `block_version==1` and MUST reject major 0
  bitstreams (`PSE_BAD_VERSION`). Minor revisions (1.x) MUST preserve
  decodability of 1.0 bitstreams; a breaking change REQUIRES major bump.
- Reserved fields MUST be zero on encode; non-zero on decode =>
  `PSE_UNSUPPORTED_FEATURE` (not best-effort decode).
- Reserved IDs (transforms 12..255, predictors 11..255, mask 7,
  residual 5..7) are future-standardization space; decoders MUST reject
  them today.

End of PSE Block Format Specification v1.0.0.
