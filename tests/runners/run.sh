#!/usr/bin/env bash
# Expects a C compiler on PATH (CC, default gcc). POSIX twin of
# tests/run.ps1: same suites, same flags, same checks, same exit
# convention -- build each suite against src/zpse_unity.c and run it,
# then build tests/test_parity.c and tests/test_fuzz.c
# in three modes (default (runtime SIMD dispatch), -DZPSE_FORCE_SCALAR and
# -DZPSE_FORCE_TRIAL_SWAP) and require every mode's digest and per-case
# output to match. The first two modes prove the SIMD paths are bit-exact
# replacements for the scalar reference; the third permutes the encoder's
# candidate-attempt order and proves that order cannot change a single
# compressed byte.
#
# Run from anywhere. Exit 0 with "ALL SUITES PASS" only when every suite
# passed; otherwise prints "FAILURES PRESENT" and exits 1.
set -u
# A failing program inside a pipeline must fail the pipeline: a filtered
# build or test log must never be able to pass by being filtered.
set -o pipefail

# Two levels up: this script lives in tests/runners/, so the repository
# root is its grandparent. (It sat in tests/ itself before the tree was
# reorganised; one level up now points at tests/ and every build fails.)
repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
tmp="${TMPDIR:-/tmp}/zpse-tests"
mkdir -p "$tmp"

CC="${CC:-gcc}"

# A bare name is what a POSIX toolchain produces; Git Bash / Cygwin / MSYS
# want the .exe suffix. Either way the same file runs.
exe=""
case "$(uname -s 2>/dev/null)" in
MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac

# Compare two files without diffutils. A minimal MSYS2 install (what the
# Windows CI job has) ships bash and grep but no cmp, and `cmp: command not
# found` used to be reported as a digest mismatch on identical digests.
# `$(<file)` is a bash builtin, so this needs nothing on PATH.
same() {
    [ "$(<"$1")" = "$(<"$2")" ]
}

fail=0

for t in test_roundtrip test_transforms test_hostile test_units test_stride3 test_dense test_decide test_fillpred test_lz test_huf test_ascii test_levels test_frame test_parallel test_sizes test_colref test_colcorpus test_pagedict test_pageforms test_flat7 test_nt_alignment; do
    src="$repo/tests/suites/$t.c"
    bin="$tmp/$t$exe"
    echo "=== build $t ==="
    # tests/corpus is on the include path for the suites that share the
    # sized corpus header (test_sizes); it moved there with the corpus.
    if ! "$CC" -std=c11 -O2 -Wall -Wextra -I "$inc" \
        -I "$repo/tests/corpus" "$src" "$unity" -o "$bin"; then
        echo "BUILD FAILED: $t"
        fail=1
        continue
    fi
    "$bin"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "SUITE FAILED: $t"
        fail=1
    fi
done

# def/scalar/swap determinism gate: the three modes below must produce
# byte-identical compressed streams for every corpus case -- def is the
# runtime SIMD dispatch, scalar forces the reference path
# (-DZPSE_FORCE_SCALAR), and swap permutes the encoder's candidate-attempt
# order (-DZPSE_FORCE_TRIAL_SWAP). This is what keeps the new static-const
# CRC32C slice/GF(2) tables and the restored fixed-offset arena carves
# honest: no lazy table init, no position-dependent scratch, and no trial
# order may leak into the output. Any digest or per-case line that differs
# across the three modes fails the run.
echo "=== scalar/SIMD/order parity ==="

# `parity` compares whole compressed corpora; `fuzz` does the same over a
# randomized corpus. Each prints one summary line (DIGEST / FUZZ cases=)
# that must match across the three builds, and the rest of their output must
# match too, so a divergence names the case that caused it.
for suite in parity fuzz; do
    src="$repo/tests/suites/test_$suite.c"
    built=1
    for mode in def scalar swap; do
        case "$mode" in
        def) fl="" ;;
        scalar) fl="-DZPSE_FORCE_SCALAR" ;;
        swap) fl="-DZPSE_FORCE_TRIAL_SWAP" ;;
        esac
        echo "=== build $suite ($mode) ==="
        # shellcheck disable=SC2086
        if ! "$CC" -std=c11 -O2 -Wall -Wextra $fl -I "$inc" "$src" "$unity" \
            -o "$tmp/$suite.$mode$exe"; then
            echo "BUILD FAILED: $suite ($mode)"
            fail=1
            built=0
        fi
    done
    [ "$built" -eq 1 ] || continue

    for mode in def scalar swap; do
        "$tmp/$suite.$mode$exe" >"$tmp/$suite.$mode.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "SUITE FAILED: $suite ($mode)"
            fail=1
        fi
    done

    key='^DIGEST'
    [ "$suite" = fuzz ] && key='^FUZZ cases='
    grep "$key" "$tmp/$suite.def.log" >"$tmp/$suite.def.dig" || :
    grep -v "$key" "$tmp/$suite.def.log" >"$tmp/$suite.def.body" || :
    for mode in scalar swap; do
        grep "$key" "$tmp/$suite.$mode.log" >"$tmp/$suite.$mode.dig" || :
        grep -v "$key" "$tmp/$suite.$mode.log" >"$tmp/$suite.$mode.body" || :
        if ! same "$tmp/$suite.def.dig" "$tmp/$suite.$mode.dig"; then
            echo "$suite FAILED: digest $(cat "$tmp/$suite.def.dig") vs $mode $(cat "$tmp/$suite.$mode.dig")"
            fail=1
        elif ! same "$tmp/$suite.def.body" "$tmp/$suite.$mode.body"; then
            echo "$suite FAILED: per-case output differs in $mode"
            fail=1
        else
            echo "$suite ok ($mode): $(cat "$tmp/$suite.def.dig") identical streams"
        fi
    done
done

# The parallel frame encoder is compiled out unless ZPSE_MT is defined, so
# the loop above ran test_parallel against its serial fallback. Build the
# same suite once with threads in and require it to pass: it asserts the
# threaded frame equals the serial one byte for byte, at every worker count
# it tries. POSIX needs -pthread; the Win32 path uses CreateThread, so the
# flag would only be an unused library there.
echo "=== multi-core encode (ZPSE_MT) ==="
mt_flags="-DZPSE_MT"
case "$(uname -s 2>/dev/null)" in
MINGW* | MSYS* | CYGWIN*) ;;
*) mt_flags="$mt_flags -pthread" ;;
esac
# shellcheck disable=SC2086
if ! "$CC" -std=c11 -O2 -Wall -Wextra $mt_flags -I "$inc" \
    "$repo/tests/suites/test_parallel.c" "$unity" \
    -o "$tmp/test_parallel.mt$exe"; then
    echo "BUILD FAILED: test_parallel ($mt_flags)"
    fail=1
elif ! "$tmp/test_parallel.mt$exe"; then
    echo "SUITE FAILED: test_parallel ($mt_flags)"
    fail=1
fi

# The page kernels have AVX2 and scalar/SSE2 twins (see hot.h and the page
# kernel sources). Build test_pageforms with the kernels compiled out and
# require every byte of its output to match the default build, so the SIMD
# path cannot drift from the scalar reference on the new page forms. This is
# the same kind of guard the parity/fuzz loop applies to the older
# kernels; it is kept separate so those two digest checks stay unchanged.
echo "=== page kernels without AVX2 (ZPSE_AVX2_ABI_OK=0) ==="
pf_src="$repo/tests/suites/test_pageforms.c"
pf_built=1
for mode in def noavx; do
    case "$mode" in
    def) fl="" ;;
    noavx) fl="-DZPSE_AVX2_ABI_OK=0" ;;
    esac
    echo "=== build pageforms ($mode) ==="
    # shellcheck disable=SC2086
    if ! "$CC" -std=c11 -O2 -Wall -Wextra $fl -I "$inc" \
        -I "$repo/tests/corpus" "$pf_src" "$unity" \
        -o "$tmp/pageforms.$mode$exe"; then
        echo "BUILD FAILED: pageforms ($mode)"
        fail=1
        pf_built=0
    fi
done
if [ "$pf_built" -eq 1 ]; then
    for mode in def noavx; do
        "$tmp/pageforms.$mode$exe" >"$tmp/pageforms.$mode.log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "SUITE FAILED: pageforms ($mode)"
            fail=1
        fi
    done
    if ! same "$tmp/pageforms.def.log" "$tmp/pageforms.noavx.log"; then
        echo "pageforms FAILED: output differs with ZPSE_AVX2_ABI_OK=0"
        fail=1
    else
        echo "pageforms ok (noavx): $(cat "$tmp/pageforms.def.log") identical streams"
    fi
fi

# End-to-end CLI smoke check. The block-level TEXT form (sync 0x50BA) is also
# exercised through the public tool, not just the in-tree suites: build the CLI,
# feed it deterministic all-ASCII prose at -l 6 with 64 KiB blocks, ask `dir`
# to walk the indexed frame, then decompress and compare byte for byte. `dir`
# needs a FrameIndex, hence -x. Every failure prints a FAILURE line so the
# final summary reports it.
echo "=== CLI smoke check (ASCII prose, -l 6 -b 65536) ==="
cli="$tmp/zpse_cli$exe"
cli_src="$repo/tests/tools/zpse_cli.c"
if [ ! -f "$cli" ] || [ "$cli_src" -nt "$cli" ]; then
    if ! "$CC" -std=c11 -O2 -Wall -Wextra -I "$inc" \
        "$cli_src" "$unity" -o "$cli"; then
        echo "BUILD FAILED: zpse_cli (smoke check)"
        fail=1
    fi
fi
if [ "$fail" -eq 0 ]; then
    ascii_in="$tmp/cli_ascii.in"
    ascii_frame="$tmp/cli_ascii.zpse"
    ascii_out="$tmp/cli_ascii.out"
    # Deterministic English-like prose: fixed sentences with a varying counter
    # on each line so there are no long exact repeats, comfortably past the
    # 4096-byte TEXT gate and inside one 64 KiB block.
    : >"$ascii_in"
    k=0
    while [ "$k" -lt 256 ]; do
        printf '%s %u %s\n' \
            "the quick brown fox jumps over the lazy dog and then runs through the green field near the old village" \
            "$k" \
            "english prose with common words and ordinary punctuation is what the static table expects to see here" \
            >>"$ascii_in"
        k=$((k + 1))
    done
    if ! "$cli" c -l 6 -b 65536 -x "$ascii_in" "$ascii_frame"; then
        echo "CLI FAILURE: compress (c -l 6 -b 65536 -x)"
        fail=1
    elif ! "$cli" dir "$ascii_frame" >"$tmp/cli_ascii.dir"; then
        echo "CLI FAILURE: dir on frame"
        fail=1
    elif ! "$cli" d "$ascii_frame" "$ascii_out"; then
        echo "CLI FAILURE: decompress (d)"
        fail=1
    elif ! same "$ascii_in" "$ascii_out"; then
        echo "CLI FAILURE: round-trip mismatch (ascii prose)"
        fail=1
    fi
fi

if [ "$fail" -eq 0 ]; then
    echo "ALL SUITES PASS"
else
    echo "FAILURES PRESENT"
fi
exit "$fail"
