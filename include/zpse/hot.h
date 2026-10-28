#ifndef ZPSE_HOT_H
#define ZPSE_HOT_H

/* Per-function -O3 for the loops the encoder spends its milliseconds in.
 *
 * Every build of this tree is -O2 -- the CI matrix, tests/runners/run.sh and
 * the release CLI all say so -- and the hot paths were tuned at that level.
 * This tag asks gcc for the -O3 passes on one function inside that -O2
 * build: on the 18 functions it marks it is worth a measured 1.029x of
 * encode throughput on tests/benchmarks/bench_levels.c over the 25-case
 * corpus (best of seven interleaved rounds with the order swapped every
 * round, against the same tree built with the macro empty; two builds of
 * identical sources read 1.000 in that harness), for a slightly larger
 * binary, with every size in that corpus unchanged.  It is not a
 * substitute for building the amalgamation at -O3 (+7% measured), and it
 * cannot be: gcc runs its inliner and its interprocedural passes at the
 * compilation's own level, so a per-function
 * tag only reaches that function's passes, not the cross-function
 * decisions where a unity build spends its -O3 budget (the object file's
 * text grows 152 KB -> 166 KB under these tags, against 152 KB -> 249 KB
 * for -O3 on the whole translation unit).  See "Building and testing" in
 * README.md for the table.
 *
 * The tag is gcc-only, so it is guarded, and neither guard is incidental:
 *
 *   - clang does not implement `optimize`.  To clang it is an *unknown*
 *     attribute, which -Wunknown-attributes (on by default) diagnoses, and
 *     this tree builds with -Werror, so emitting it would fail the clang
 *     build rather than merely doing nothing.  clang's own knob is the
 *     build flag: compile with -O3 or above.
 *   - MSVC has no per-function optimization level at all.
 *
 * Where it expands to nothing, the sources and the format are identical and
 * only the code generation differs, so every suite in the tree has to pass
 * either way -- which is why the tags are placed by measurement rather than
 * by taste.  One measurement to keep in mind before adding one: gcc will
 * not inline across an optimization mismatch, so tagging a function in the
 * middle of a hot call chain *costs* -- tagging zpse_transform_inverse
 * alone measured -7% encode, because its -O2 callees stopped being inlined
 * into it.  The tags below are a set that was measured together; adding one
 * to a function whose callees are untagged reproduces that mistake.  The
 * set is therefore leaves and functions that call only other tagged code;
 * the functions that are called from the middle of a chain (tile_classify,
 * zpse_segment_encode, seg_scan_best, zpse_mask_encode) are left untagged
 * on that rule, and measuring them confirms it costs rather than gains.
 *
 * It also cannot be combined with a target attribute: gcc re-applies the
 * command line's target to a function it has been asked to optimize at a
 * different level, which would compile AVX2 intrinsics for a baseline ISA.
 * tile_classify_avx2, seg_prescreen_avx2 and the rest of the AVX2 bodies
 * are therefore untagged even where they are hot. */
#if defined(__GNUC__) && !defined(__clang__) && !defined(_MSC_VER)
#define ZPSE_HOT __attribute__((optimize("O3")))
#else
#define ZPSE_HOT
#endif

#endif /* ZPSE_HOT_H */
