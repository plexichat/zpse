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

/* A block that is taken on almost every iteration. gcc's estimate of a
 * branch chain is a product of per-test guesses, so a condition with enough
 * terms to describe a whole fast path lands under the 50% that decides
 * fall-through: the LZ decoder's seven-term one was laid out as a jump to a
 * cold block instead of a straight-line run, which cost more than the
 * condition itself. The hint says what the code means -- "this is the
 * iteration" -- and is only a layout instruction. It cannot change which
 * streams are accepted, and it compiles to nothing where it is unknown. */
#if defined(__GNUC__) || defined(__clang__)
#define ZPSE_LIKELY(X) __builtin_expect(!!(X), 1)
#else
#define ZPSE_LIKELY(X) (X)
#endif

/* Whether this toolchain can be handed the 256-bit kernels at all.
 *
 * A `__m256i` stack slot has to be 32-byte aligned, because that is what
 * gcc emits for it: `vmovdqa`, not `vmovdqu`. On the SysV ABI the alignment
 * is there. On Windows x64 the compiler cannot provide it -- MinGW gcc caps
 * -mpreferred-stack-boundary at 4 (it rejects 5) -- so every 256-bit value
 * that reaches the stack is a fault waiting for its address. Ordinary
 * locals are not the problem: gcc spills them with `vmovdqu` when it can
 * see they are only vector registers. The problem is a vector argument
 * passed BY VALUE to a function gcc chose not to inline (and a vector
 * return through the hidden pointer): the caller materialises the argument
 * in an aligned temporary and the callee reads it as aligned, and nothing
 * in the sources can change either assumption.
 *
 * Measured on this tree, with gcc 16.1 (MinGW-w64) at -O2 and the same
 * sources passing under Linux gcc: test_transforms, test_lz, test_ascii,
 * test_levels, test_sizes and the fuzz driver died in zpse_merge2_avx2 /
 * zpse_merge4_avx2 on `vmovdqa %ymm0,0x20(%rsp)` -- the argument block of
 * a 256-bit helper passed by value (`zpse_chain_undo32`; its arithmetic now
 * lives in two always-inlined twins, so that argument block has gone from
 * the tree but the shape it demonstrated has not). `__m256i_u` (alignment
 * 1) does not help, because
 * the ABI temporary is aligned for the vector MODE, not the type, and
 * -mstackrealign / force_align_arg_pointer do not either, because both
 * realign to the 16-byte boundary that is the most MinGW accepts.
 *
 * So on that toolchain the runtime gate answers "no AVX2" and every kernel
 * site falls back to its SSE2 or scalar twin, which are bit-exact (that is
 * what tests/test_parity.c proves on every platform) and are what a CPU
 * without AVX2 runs anyway. MSVC keeps the kernels: it aligns a 32-byte
 * type on the stack to 32. clang keeps them too. The macro can be set on
 * the command line (-DZPSE_AVX2_ABI_OK=0) to build the kernels out
 * anywhere, which is how the fallback build is tested on Linux. */
#ifndef ZPSE_AVX2_ABI_OK
#if defined(__GNUC__) && !defined(__clang__) && \
    (defined(_WIN32) || defined(__CYGWIN__))
#define ZPSE_AVX2_ABI_OK 0
#else
#define ZPSE_AVX2_ABI_OK 1
#endif
#endif

#endif /* ZPSE_HOT_H */
