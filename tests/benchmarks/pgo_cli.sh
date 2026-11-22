#!/usr/bin/env bash
# Reproducible two-stage profile-guided build of the CLI, for the benchmark
# numbers the README publishes. Unlike pgo_build.sh (which trains on generated
# corpora and also verifies the test suites), this one trains on the exact
# files a caller is about to measure, so the profile covers the input mix the
# published table actually reports.
#
# Usage: bash tests/benchmarks/pgo_cli.sh OUTPUT [TRAIN_FILE...]
#        CC=clang bash tests/benchmarks/pgo_cli.sh /tmp/zpse-cli silesia/*
#
# Stage 1 compiles the unity source and tests/tools/zpse_cli.c with profile
# instrumentation, trains `zpse_cli bench -q` on the supplied files at 64 KiB
# blocks at levels 1, 6 and 9 and at 4 KiB blocks at level 6, and keeps the
# counters. Stage 2 recompiles the SAME sources to the SAME relative object
# paths with -fprofile-use, which is what makes gcc's .gcda names resolve;
# clang needs llvm-profdata to merge its .profraw files first.
#
# OUTPUT is not touched until a BYTE-IDENTITY GATE passes: a plain -O2 build
# and the PGO build both compress every training file with `c -l 6` and the
# two streams are compared whole. PGO may change code generation, never the
# bytes a reader sees. A single difference is a hard failure and no OUTPUT is
# written, so callers can fall back to a plain build.
#
# Every source and binary lives in a mktemp -d directory: nothing is written
# into the repository. KEEP=1 leaves the work directory behind for inspection.
set -uo pipefail

die() { echo "pgo_cli: $*" >&2; exit 1; }

[ "$#" -ge 1 ] || die "usage: pgo_cli.sh OUTPUT [TRAIN_FILE...]"
out="$1"
shift
[ "$#" -gt 0 ] || die "no training files supplied"

repo="$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
inc="$repo/include"

work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pgo-cli.XXXXXX")" || exit 1
cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "pgo_cli: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT
mkdir -p "$work/src" "$work/prof" || exit 1

# The unity file #includes its siblings by relative name, so copy the whole
# src/ directory. The CLI is copied to the same relative path pgo_build.sh
# uses; gcc embeds that path in its .gcda file name and resolves a relative
# -fprofile-dir against the run's current directory.
cp -R "$repo"/src/. "$work/src/" || exit 1
cp "$repo/tests/tools/zpse_cli.c" "$work/src/cli.c" || exit 1

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

cc="${CC:-clang}"
command -v "$cc" >/dev/null 2>&1 || die "no compiler '$cc'"
ver="$("$cc" --version 2>/dev/null | sed -n '1p')"
case "$ver" in
*clang* | *Clang*) kind=clang ;;
*gcc* | *GCC* | *GNU*) kind=gcc ;;
*) die "unrecognised compiler: $ver" ;;
esac
if [ "$kind" = clang ]; then
    profdata="$(profdata_for "$cc")" || die "clang PGO needs llvm-profdata"
fi
echo "pgo_cli: $ver" >&2

base=(-std=c11 -O2 -I "$inc" -I "$work/src")
if [ "$kind" = clang ]; then
    # clang emits .profraw counters; merge them into one indexed profile.
    # -fprofile-correction is a gcc flag that clang accepts and ignores, so
    # silence that one warning instead of hiding it behind -Werror elsewhere.
    gen=(-fprofile-generate)
    # Counter hashes can go out of date when the tree changes between the
    # training and use builds. The byte-identity gate below is the real check;
    # these keep the backend warnings from drowning it.
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

# Object names are fixed by stage, not by output binary, so gcc reads back
# exactly the counters stage 1 wrote. Every build runs from $work.
build_cli() {
    local ob="$1" dst="$2"
    shift 2
    (cd "$work" &&
        "$cc" "${base[@]}" "$@" -c src/cli.c -o "$ob.cli.o" &&
        "$cc" "${base[@]}" "$@" -c src/zpse_unity.c -o "$ob.unity.o" &&
        "$cc" "${base[@]}" "$@" "$ob.cli.o" "$ob.unity.o" -o "$dst") ||
        die "build failed: $dst"
}

train=("$@")
echo "pgo_cli: training on ${#train[@]} file(s), 64 KiB L1/L6/L9 and 4 KiB L6" >&2
build_cli base "cli.base$exe"
build_cli pg "cli.pg$exe" "${gen[@]}"

train_at() { # nominal block size, level
    local b="$1" l="$2"
    if [ "$kind" = clang ]; then
        (cd "$work" && LLVM_PROFILE_FILE="prof/pgo-%p.profraw" \
            "./cli.pg$exe" bench -q -b "$b" -l "$l" "${train[@]}" >/dev/null)
    else
        (cd "$work" && "./cli.pg$exe" bench -q -b "$b" -l "$l" "${train[@]}" >/dev/null)
    fi
}
train_at 65536 1 || die "PGO training failed at -b 65536 -l 1"
train_at 65536 6 || die "PGO training failed at -b 65536 -l 6"
train_at 65536 9 || die "PGO training failed at -b 65536 -l 9"
train_at 4096 6 || die "PGO training failed at -b 4096 -l 6"

if [ "$kind" = clang ]; then
    shopt -s nullglob
    raw=("$work"/prof/*.profraw)
    shopt -u nullglob
    [ "${#raw[@]}" -gt 0 ] || die "training produced no .profraw files"
    "$profdata" merge -sparse "${raw[@]}" -o "$work/pgo.profdata" ||
        die "llvm-profdata merge failed"
else
    got="$(find "$work/prof" -name '*.gcda' -print -quit)"
    [ -n "$got" ] || die "training produced no .gcda files"
fi

build_cli pg "cli.pgo$exe" "${use[@]}"

# ---- byte-identity gate ---------------------------------------------------
# The PGO binary is only useful to a published table if it writes exactly
# what a plain build writes. Compress every training file with both binaries
# and compare the whole frame. Do not rename anything to $out until here.
i=0
for f in "${train[@]}"; do
    a="$work/plain.$i.zpse"
    b="$work/pgo.$i.zpse"
    "$work/cli.base$exe" c -l 6 "$f" "$a" >/dev/null 2>&1 ||
        die "plain -O2 CLI could not compress $f"
    "$work/cli.pgo$exe" c -l 6 "$f" "$b" >/dev/null 2>&1 ||
        die "PGO CLI could not compress $f"
    if ! cmp -s "$a" "$b"; then
        echo "pgo_cli: BYTE-IDENTITY GATE FAILED on $f" >&2
        echo "pgo_cli: PGO and plain -O2 wrote different bytes; no output written" >&2
        exit 1
    fi
    rm -f "$a" "$b"
    i=$((i + 1))
done
echo "pgo_cli: byte identity verified on $i file(s)" >&2

cp "$work/cli.pgo$exe" "$out" || die "cannot write $out"
chmod +x "$out" 2>/dev/null || :
echo "pgo_cli: wrote PGO CLI to $out" >&2
