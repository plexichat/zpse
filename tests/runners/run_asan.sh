#!/usr/bin/env bash
# Opt-in AddressSanitizer + UndefinedBehaviorSanitizer harness. It is NOT
# wired into the normal test runners: invoke it explicitly when you want to
# hunt memory bugs (e.g. in the new FLAT code). It builds src/zpse_unity.c
# together with each suite in tests/suites/ using
#   -fsanitize=address,undefined -fno-omit-frame-pointer -g
# and runs every resulting binary, reporting a clear pass/fail summary.
#
# Mirrors tests/runners/run.sh: same suite list, same repo-relative paths,
# same tmp layout (a sibling "zpse-tests-asan" dir so sanitizer binaries do
# not clobber the normal ones). Only the temp dir is written; the repo is
# never modified.
#
# Run from anywhere. Uses $CC (default gcc). Exits 0 only when every suite
# passes; 1 when any suite/build fails; 2 when sanitizer support is missing.
set -u
# A failing program inside a pipeline must fail the pipeline: a filtered
# build or test log must never be able to pass by being filtered.
set -o pipefail

# Two levels up: this script lives in tests/runners/, so the repository
# root is its grandparent.
repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
tmp="${TMPDIR:-/tmp}/zpse-tests-asan"
mkdir -p "$tmp"

CC="${CC:-gcc}"

if ! command -v "$CC" >/dev/null 2>&1; then
    echo "ASAN UNAVAILABLE: compiler '$CC' not found on PATH."
    echo "Set CC to a gcc/clang with ASan/UBSan support and retry."
    exit 2
fi

# A bare name is what a POSIX toolchain produces; Git Bash / Cygwin / MSYS
# want the .exe suffix. Either way the same file runs.
exe=""
case "$(uname -s 2>/dev/null)" in
MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac

# -fno-sanitize-recover makes a UBSan diagnostic abort (nonzero exit);
# otherwise UBSan only prints and the binary still exits 0, which would be
# reported as a false PASS. -O1 keeps stack traces more usable than -O2.
SAN_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -fno-sanitize-recover=all"

# Respect a caller-supplied UBSAN_OPTIONS; otherwise make UBSan halt on the
# first diagnostic so it cannot be silently reported as a pass.
: "${UBSAN_OPTIONS:=halt_on_error=1}"
export UBSAN_OPTIONS

# Probe once: some toolchains accept the flag but cannot link or run a
# sanitized binary (missing libasan/libubsan, unsupported platform, ...).
# Fail early with a clear message instead of a wall of misleading errors.
probe_src="$tmp/.asan_probe.c"
probe_bin="$tmp/.asan_probe$exe"
printf 'int main(void) { return 0; }\n' > "$probe_src"
# shellcheck disable=SC2086
if ! "$CC" -std=c11 -O1 $SAN_FLAGS "$probe_src" -o "$probe_bin" >/dev/null 2>&1; then
    echo "ASAN UNAVAILABLE: $CC cannot build a -fsanitize=address,undefined binary."
    echo "Install a compiler with ASan/UBSan support (gcc/clang + libasan/libubsan) and retry."
    exit 2
fi
if ! "$probe_bin" >/dev/null 2>&1; then
    echo "ASAN UNAVAILABLE: sanitized binary built but could not run (missing ASan/UBSan runtime?)."
    echo "Check that the compiler's sanitizer runtime libraries are on PATH."
    exit 2
fi

# Union of the suite lists parsed from run.ps1 and run.sh (run.sh is the
# superset, adding test_parallel and test_pagedict). Build all of them so a
# memory bug in any suite -- including the new FLAT paths -- is surfaced.
suites="test_roundtrip test_transforms test_hostile test_units test_stride3 test_dense test_decide test_fillpred test_lz test_huf test_ascii test_levels test_frame test_parallel test_sizes test_colref test_colcorpus test_pagedict test_pageforms"

pass=0
fail=0
total=0
for t in $suites; do
    total=$((total + 1))
    src="$repo/tests/suites/$t.c"
    bin="$tmp/$t.asan$exe"
    log="$tmp/$t.asan.log"
    echo "=== build $t (asan+ubsan) ==="
    # tests/corpus is on the include path for suites that share the sized
    # corpus header, matching the main loop of run.sh.
    # shellcheck disable=SC2086
    if ! "$CC" -std=c11 -O1 -Wall -Wextra $SAN_FLAGS -I "$inc" \
        -I "$repo/tests/corpus" "$src" "$unity" -o "$bin"; then
        echo "FAIL: $t (build)"
        fail=$((fail + 1))
        continue
    fi
    "$bin" >"$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS: $t"
        pass=$((pass + 1))
    else
        echo "FAIL: $t (exit $rc)"
        cat "$log"
        fail=$((fail + 1))
    fi
done

echo "=== sanitizer summary ==="
echo "passed: $pass  failed: $fail  total: $total"
if [ "$fail" -eq 0 ]; then
    echo "ALL SANITIZER SUITES PASS"
    exit 0
else
    echo "SANITIZER FAILURES PRESENT"
    exit 1
fi
