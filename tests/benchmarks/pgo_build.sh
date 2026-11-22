#!/usr/bin/env bash
# Reproducible two-stage profile-guided build of the 4 KiB path.
#
# Usage: bash tests/benchmarks/pgo_build.sh [--verify] [SIZE]
#        CC=gcc bash tests/benchmarks/pgo_build.sh --verify
#
# SIZE is the bytes written per corpus class by tests/corpus/corpus_write.c
# (default 4 MiB, must be a multiple of 64 KiB). --verify additionally builds
# the parity and fuzz suites with the PGO flags in the default,
# forced-scalar and swapped-attempt-order modes, and requires each mode to
# print the same digest and the same per-case output as a plain build and as
# its two siblings.
#
# Stage 1 compiles the unity source and tests/tools/zpse_cli.c with profile
# instrumentation, trains `zpse_cli bench -q` on the 4 KiB page, probe and mixed
# corpora at `-b 4096 -l 6` and on the byte-pattern and columnar corpora at
# `-b 65536`, levels 1, 6 and 9, and keeps the counters. Stage 2 recompiles the
# SAME sources to the SAME relative object paths with -fprofile-use, which is
# what makes gcc's .gcda names resolve; clang needs llvm-profdata to merge its
# .profraw files first; automatic selection falls back to gcc without it.
#
# Every source is copied into a mktemp -d directory and every binary, corpus,
# profile and log lives there: nothing is written into the repository. Prints
# markdown. KEEP=1 leaves the work directory behind for inspection.
set -uo pipefail

die() { echo "pgo_build: $*" >&2; exit 1; }

repo="$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
inc="$repo/include"

verify=0
size=""
for arg in "$@"; do
    case "$arg" in
    --verify) verify=1 ;;
    -h | --help) sed -n '2,24p' "$0"; exit 0 ;;
    *) [ -z "$size" ] || die "usage: pgo_build.sh [--verify] [SIZE]"; size="$arg" ;;
    esac
done
size="${size:-4194304}"
case "$size" in *[!0-9]*) die "SIZE must be a multiple of 65536 bytes" ;; esac
[ $((size % 65536)) -eq 0 ] || die "SIZE must be a multiple of 65536 bytes"

work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pgo.XXXXXX")" || exit 1
cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "pgo_build: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT
mkdir -p "$work/src" "$work/corpus" "$work/prof" || exit 1

# The unity file #includes its siblings by relative name, so copy the whole
# src/ directory, not just the amalgamation. The CLI, the two parity suites
# and the corpus writer are copied too, so every compiled source path is
# relative to $work (gcc embeds that path in its .gcda file name and resolves
# a relative -fprofile-dir against the run's current directory).
cp -R "$repo"/src/. "$work/src/" || exit 1
cp "$repo/tests/tools/zpse_cli.c" "$work/src/cli.c" || exit 1
cp "$repo/tests/corpus/corpus_write.c" "$work/src/corpus_write.c" || exit 1
cp "$repo/tests/corpus/columnar_gen.h" "$work/src/columnar_gen.h" || exit 1
for s in parity fuzz; do
    cp "$repo/tests/suites/test_$s.c" "$work/src/test_$s.c" || exit 1
done

exe=""
case "$(uname -s 2>/dev/null)" in
MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac

profdata_for() {
    [ -n "${LLVM_PROFDATA:-}" ] && { echo "$LLVM_PROFDATA"; return 0; }
    if command -v llvm-profdata >/dev/null 2>&1; then
        command -v llvm-profdata
        return 0
    fi
    p="$(command -v "$1" 2>/dev/null)" || return 1
    d="$(dirname "$p")"
    for c in "$d/llvm-profdata$exe" "$d/llvm-profdata"; do
        [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}

cc="${CC:-}"
if [ -z "$cc" ]; then
    if command -v clang >/dev/null 2>&1 && profdata="$(profdata_for clang)"; then
        cc=clang
    elif command -v gcc >/dev/null 2>&1; then
        cc=gcc
        echo "pgo_build: clang or llvm-profdata absent, using gcc" >&2
    else
        die "need clang (with llvm-profdata) or gcc"
    fi
fi
command -v "$cc" >/dev/null 2>&1 || die "no compiler '$cc'"
ver="$("$cc" --version 2>/dev/null | head -n 1)"
case "$ver" in
*clang* | *Clang*) kind=clang ;;
*gcc* | *GCC* | *GNU*) kind=gcc ;;
*) die "unrecognised compiler: $ver" ;;
esac
if [ "$kind" = clang ]; then
    profdata="$(profdata_for "$cc")" || die "clang PGO needs llvm-profdata"
fi
echo "pgo_build: $ver" >&2

base=(-std=c11 -O2 -I "$inc" -I "$work/src")
if [ "$kind" = clang ]; then
    # clang emits .profraw counters; merge them into one indexed profile.
    # -fprofile-correction is a gcc flag that clang accepts and ignores, so
    # silence that one warning instead of hiding it behind -Werror elsewhere.
    gen=(-fprofile-generate)
    # A suite built with different macros has different function hashes, so
    # clang discards those counters and warns from the backend. The warning is
    # expected for --verify's extra builds; suppress it there and here.
    use=(-fprofile-use="$work/pgo.profdata" -fprofile-correction
         -Wno-ignored-optimization-argument -Wno-profile-instr-out-of-date
         -Wno-profile-instr-unprofiled -Wno-backend-plugin)
else
    # Relative profile directory: gcc would otherwise put the drive letter in
    # the .gcda name and fail to create the directory on Windows hosts.
    gen=(-fprofile-generate=prof -fprofile-dir=prof)
    use=(-fprofile-use=prof -fprofile-correction -fprofile-dir=prof
         -Wno-missing-profile)
fi

same() { [ "$(<"$1")" = "$(<"$2")" ]; }

# Object names are fixed by stage, not by output binary, so gcc reads back
# exactly the counters stage 1 wrote. Every build runs from $work.
build_cli() {
    local ob="$1" out="$2"
    shift 2
    (cd "$work" &&
        "$cc" "${base[@]}" "$@" -c src/cli.c -o "$ob.cli.o" &&
        "$cc" "${base[@]}" "$@" -c src/zpse_unity.c -o "$ob.unity.o" &&
        "$cc" "${base[@]}" "$@" "$ob.cli.o" "$ob.unity.o" -o "$out")
}

build_suite() {
    local suite="$1" ob="$2" out="$3"
    shift 3
    (cd "$work" &&
        "$cc" "${base[@]}" "$@" -c "src/test_$suite.c" -o "$ob.suite.o" &&
        "$cc" "${base[@]}" "$@" -c src/zpse_unity.c -o "$ob.unity.o" &&
        "$cc" "${base[@]}" "$@" "$ob.suite.o" "$ob.unity.o" -o "$out")
}

# ---- reproducible corpora -------------------------------------------------
(cd "$work" && "$cc" "${base[@]}" src/corpus_write.c -o "corpus_write$exe") ||
    die "corpus_write build failed"
(cd "$work" &&
    ./corpus_write$exe corpus "$size" --pages &&
    ./corpus_write$exe corpus "$size" --probe &&
    ./corpus_write$exe corpus "$size" --mixed &&
    ./corpus_write$exe corpus "$size") ||
    die "corpus generation failed"
if command -v sha256sum >/dev/null 2>&1; then
    for f in pages-nonzero.bin probe.bin mixed.bin; do
        d="$(cd "$work" && sha256sum "corpus/$f" | cut -d' ' -f1)"
        echo "pgo_build: $f sha256 ${d:0:16}…" >&2
    done
fi

# ---- stage 1: instrument, train, collect ----------------------------------
build_cli base "cli.base$exe" || die "baseline CLI build failed"
build_cli pg "cli.pg$exe" "${gen[@]}" || die "instrumented CLI build failed"

train=(corpus/pages-nonzero.bin corpus/pages-all.bin corpus/pages-repeat.bin
       corpus/page-code.bin corpus/probe.bin corpus/mixed.bin)
# The default corpus_write output: the ten byte patterns (one 64 KiB block of
# each per file) and the seven columnar classes. Trained at the headline block
# size so size-gated code is covered: the >=8 KiB trial caps, the big LZ table
# and the wide backward walk, plus the level 1 and level 9 branches.
train64=(corpus/zeros.bin corpus/incr.bin corpus/random.bin corpus/sparse.bin
         corpus/strided.bin corpus/const5A.bin corpus/alt00FF.bin
         corpus/sawtooth.bin corpus/lowentropy.bin corpus/sparse2.bin
         corpus/ts_seq.bin corpus/ts_ns.bin corpus/float_sparse.bin
         corpus/var_string.bin corpus/sparse_mix.bin corpus/geo_point.bin
         corpus/struct_offsets.bin)
echo "pgo_build: training over ${#train[@]} 4 KiB and ${#train64[@]} 64 KiB corpora (${size} B/class)" >&2
if [ "$kind" = clang ]; then
    (cd "$work" && LLVM_PROFILE_FILE="prof/pgo-%p.profraw" \
        "./cli.pg$exe" bench -q -b 4096 -l 6 "${train[@]}" >/dev/null) ||
        die "PGO training run failed"
    (cd "$work" && LLVM_PROFILE_FILE="prof/pgo-%p.profraw" \
        "./cli.pg$exe" bench -q -b 65536 -l 1 "${train64[@]}" >/dev/null) ||
        die "PGO training run failed"
    (cd "$work" && LLVM_PROFILE_FILE="prof/pgo-%p.profraw" \
        "./cli.pg$exe" bench -q -b 65536 -l 6 "${train64[@]}" >/dev/null) ||
        die "PGO training run failed"
    (cd "$work" && LLVM_PROFILE_FILE="prof/pgo-%p.profraw" \
        "./cli.pg$exe" bench -q -b 65536 -l 9 "${train64[@]}" >/dev/null) ||
        die "PGO training run failed"
    shopt -s nullglob
    raw=("$work"/prof/*.profraw)
    shopt -u nullglob
    [ "${#raw[@]}" -gt 0 ] || die "training produced no .profraw files"
    "$profdata" merge -sparse "${raw[@]}" -o "$work/pgo.profdata" ||
        die "llvm-profdata merge failed"
else
    (cd "$work" && "./cli.pg$exe" bench -q -b 4096 -l 6 "${train[@]}" >/dev/null) ||
        die "PGO training run failed"
    (cd "$work" && "./cli.pg$exe" bench -q -b 65536 -l 1 "${train64[@]}" >/dev/null) ||
        die "PGO training run failed"
    (cd "$work" && "./cli.pg$exe" bench -q -b 65536 -l 6 "${train64[@]}" >/dev/null) ||
        die "PGO training run failed"
    (cd "$work" && "./cli.pg$exe" bench -q -b 65536 -l 9 "${train64[@]}" >/dev/null) ||
        die "PGO training run failed"
    got="$(find "$work/prof" -name '*.gcda' -print -quit)"
    [ -n "$got" ] || die "training produced no .gcda files"
fi

# ---- stage 2: consume ------------------------------------------------------
build_cli pg "cli.pgo$exe" "${use[@]}" || die "PGO CLI build failed"

# ---- before / after table -------------------------------------------------
bench_total() {
    (cd "$work" && "$1" bench -q -b 4096 -l 6 "$2") |
        awk '/^TOTAL/ { print; exit }'
}
field() { awk -v n="$2" 'NR == 1 { print $n }' <<<"$1"; }
max2() { awk -v a="$1" -v b="$2" 'BEGIN { print (a+0 > b+0) ? a+0 : b+0 }'; }
pct() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        if (a+0 == 0) print "n/a"; else printf "%+.1f%%", (b-a)/a*100
    }'
}

echo
echo "## before / after: 4 KiB pages, level 6"
echo
echo "| corpus | ratio | enc MB/s base | enc MB/s PGO | enc delta | dec MB/s base | dec MB/s PGO | dec delta |"
echo "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
ratio_ok=1
for f in pages-nonzero.bin pages-repeat.bin probe.bin mixed.bin; do
    p="$work/corpus/$f"
    b1="$(bench_total "$work/cli.base$exe" "$p")" || die "baseline bench failed on $f"
    q1="$(bench_total "$work/cli.pgo$exe" "$p")" || die "PGO bench failed on $f"
    q2="$(bench_total "$work/cli.pgo$exe" "$p")" || die "PGO bench failed on $f"
    b2="$(bench_total "$work/cli.base$exe" "$p")" || die "baseline bench failed on $f"
    be="$(max2 "$(field "$b1" 5)" "$(field "$b2" 5)")"
    bd="$(max2 "$(field "$b1" 6)" "$(field "$b2" 6)")"
    qe="$(max2 "$(field "$q1" 5)" "$(field "$q2" 5)")"
    qd="$(max2 "$(field "$q1" 6)" "$(field "$q2" 6)")"
    br="$(field "$b1" 4)"
    qr="$(field "$q1" 4)"
    [ "$br" = "$qr" ] || ratio_ok=0
    printf '| `%s` | %s | %.1f | %.1f | %s | %.1f | %.1f | %s |\n' \
        "$f" "$qr" "$be" "$qe" "$(pct "$be" "$qe")" \
        "$bd" "$qd" "$(pct "$bd" "$qd")"
done
echo
if [ "$ratio_ok" -eq 1 ]; then
    echo "Packed bytes are identical in both columns: PGO changes the code"
    echo "generation, not a block decision. Rates are the best of two"
    echo "alternating runs per corpus on the same generated bytes."
else
    echo "WARNING: a packed size changed between the two builds" >&2
fi

# ---- optional parity verification ----------------------------------------
if [ "$verify" -eq 1 ]; then
    echo
    echo "## parity: PGO flags in all three modes"
    echo
    echo "The parity binaries reuse the CLI's profile. A compiler that cannot"
    echo "match counters for a differently compiled suite drops the counts, so"
    echo "what is checked here is byte identity, not a second rate table."
    echo
    fail=0
    for suite in parity fuzz; do
        key='^DIGEST'
        [ "$suite" = fuzz ] && key='^FUZZ cases='

        build_suite "$suite" ref "ref.$suite$exe" ||
            { echo "BUILD FAILED: $suite (plain)"; fail=1; continue; }
        (cd "$work" && "./ref.$suite$exe") >"$work/$suite.ref.log" 2>&1 ||
            { echo "SUITE FAILED: $suite (plain)"; fail=1; }

        for mode in def scalar swap; do
            case "$mode" in
            def) mf=() ;;
            scalar) mf=(-DZPSE_FORCE_SCALAR) ;;
            swap) mf=(-DZPSE_FORCE_TRIAL_SWAP) ;;
            esac
            build_suite "$suite" "pgo.$mode" "pgo.$suite.$mode$exe" \
                "${mf[@]}" "${use[@]}" ||
                { echo "BUILD FAILED: $suite ($mode)"; fail=1; continue; }
            (cd "$work" && "./pgo.$suite.$mode$exe") \
                >"$work/$suite.$mode.log" 2>&1 ||
                { echo "SUITE FAILED: $suite ($mode)"; fail=1; }
        done

        grep "$key" "$work/$suite.ref.log" >"$work/$suite.ref.dig" || :
        grep -v "$key" "$work/$suite.ref.log" >"$work/$suite.ref.body" || :
        for mode in def scalar swap; do
            grep "$key" "$work/$suite.$mode.log" >"$work/$suite.$mode.dig" || :
            grep -v "$key" "$work/$suite.$mode.log" >"$work/$suite.$mode.body" || :
            if ! same "$work/$suite.ref.dig" "$work/$suite.$mode.dig"; then
                echo "$suite FAILED: $(cat "$work/$suite.ref.dig") vs $mode $(cat "$work/$suite.$mode.dig")"
                fail=1
            elif ! same "$work/$suite.ref.body" "$work/$suite.$mode.body"; then
                echo "$suite FAILED: per-case output differs in $mode"
                fail=1
            else
                echo "$suite ok ($mode): $(cat "$work/$suite.$mode.dig") identical streams"
            fi
        done
    done
    echo
    if [ "$fail" -eq 0 ]; then
        echo "ALL PARITY PASS"
    else
        echo "PARITY FAILURES"
        exit 1
    fi
fi
