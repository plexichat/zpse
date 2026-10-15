#!/usr/bin/env bash
# Expects a C compiler on PATH (CC, default gcc). POSIX twin of
# tests/run.ps1: same suites, same flags, same checks, same exit
# convention -- build each suite against src/zpse_unity.c and run it,
# then build tests/test_parity.c twice (default (runtime SIMD dispatch)
# and -DZPSE_FORCE_SCALAR) and require the two compressed corpora
# digests and the per-case output to match, proving the SIMD paths are
# bit-exact replacements for the scalar reference.
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

for t in test_roundtrip test_combinadic test_hostile test_units; do
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

echo "=== scalar/SIMD parity ==="
parity="$repo/tests/test_parity.c"
binS="$tmp/parity_simd$exe"
binR="$tmp/parity_scalar$exe"
if ! "$CC" -std=c11 -O2 -Wall -Wextra -I "$inc" "$parity" "$unity" -o "$binS"; then
    echo "BUILD FAILED: parity (simd)"
    fail=1
fi
if ! "$CC" -std=c11 -O2 -Wall -Wextra -DZPSE_FORCE_SCALAR -I "$inc" \
    "$parity" "$unity" -o "$binR"; then
    echo "BUILD FAILED: parity (scalar)"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    "$binS" >"$tmp/parity_simd.log" 2>&1
    rcS=$?
    "$binR" >"$tmp/parity_scalar.log" 2>&1
    rcR=$?
    if [ "$rcS" -ne 0 ]; then
        echo "SUITE FAILED: parity (simd)"
        fail=1
    fi
    if [ "$rcR" -ne 0 ]; then
        echo "SUITE FAILED: parity (scalar)"
        fail=1
    fi
    grep '^DIGEST' "$tmp/parity_simd.log" >"$tmp/parity_simd.dig" || :
    grep '^DIGEST' "$tmp/parity_scalar.log" >"$tmp/parity_scalar.dig" || :
    grep -v '^DIGEST' "$tmp/parity_simd.log" >"$tmp/parity_simd.body" || :
    grep -v '^DIGEST' "$tmp/parity_scalar.log" >"$tmp/parity_scalar.body" || :
    digS="$(cat "$tmp/parity_simd.dig")"
    digR="$(cat "$tmp/parity_scalar.dig")"
    if [ "$fail" -eq 0 ]; then
        if ! cmp -s "$tmp/parity_simd.dig" "$tmp/parity_scalar.dig"; then
            echo "PARITY FAILED: digest $digS vs $digR"
            fail=1
        elif ! cmp -s "$tmp/parity_simd.body" "$tmp/parity_scalar.body"; then
            echo "PARITY FAILED: per-case output differs"
            fail=1
        else
            echo "parity ok: $digS (identical streams)"
        fi
    fi
fi

if [ "$fail" -eq 0 ]; then
    echo "ALL SUITES PASS"
else
    echo "FAILURES PRESENT"
fi
exit "$fail"
