#!/usr/bin/env bash
# Expects a C compiler on PATH (CC, default gcc). POSIX twin of
# tests/run.ps1: same suites, same flags, same checks, same exit
# convention -- build each suite against src/zpse_unity.c and run it,
# then build tests/test_parity.c, tests/test_fuzz.c and tests/test_lseed.c
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

repo="$(cd "$(dirname "$0")/.." && pwd)"
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

fail=0

for t in test_roundtrip test_combinadic test_hostile test_units test_stride3 test_dense test_decide; do
    src="$repo/tests/$t.c"
    bin="$tmp/$t$exe"
    echo "=== build $t ==="
    if ! "$CC" -std=c11 -O2 -Wall -Wextra -I "$inc" "$src" "$unity" -o "$bin"; then
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

echo "=== scalar/SIMD/order parity ==="

# `parity` compares whole compressed corpora; `fuzz` does the same over a
# randomized corpus; `lseed` decodes hand-built foreign segments whose
# ORDER tiles do not reset, against a reference that hardcodes the
# documented segment-start state. Each prints one summary line (DIGEST / FUZZ
# cases= / LSEED:) that must match across the three builds, and the rest of
# their output must match too, so a divergence names the case that caused it.
for suite in parity fuzz lseed; do
    src="$repo/tests/test_$suite.c"
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
    [ "$suite" = lseed ] && key='^LSEED:'
    grep "$key" "$tmp/$suite.def.log" >"$tmp/$suite.def.dig" || :
    grep -v "$key" "$tmp/$suite.def.log" >"$tmp/$suite.def.body" || :
    for mode in scalar swap; do
        grep "$key" "$tmp/$suite.$mode.log" >"$tmp/$suite.$mode.dig" || :
        grep -v "$key" "$tmp/$suite.$mode.log" >"$tmp/$suite.$mode.body" || :
        if ! cmp -s "$tmp/$suite.def.dig" "$tmp/$suite.$mode.dig"; then
            echo "$suite FAILED: digest $(cat "$tmp/$suite.def.dig") vs $mode $(cat "$tmp/$suite.$mode.dig")"
            fail=1
        elif ! cmp -s "$tmp/$suite.def.body" "$tmp/$suite.$mode.body"; then
            echo "$suite FAILED: per-case output differs in $mode"
            fail=1
        else
            echo "$suite ok ($mode): $(cat "$tmp/$suite.def.dig") identical streams"
        fi
    done
done

if [ "$fail" -eq 0 ]; then
    echo "ALL SUITES PASS"
else
    echo "FAILURES PRESENT"
fi
exit "$fail"
