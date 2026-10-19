#!/usr/bin/env bash
# Cross-tool benchmark: the in-tree codec against the compressors a CI
# runner has, on one corpus, through one interface.
#
# Every tool is driven as a process over a file: compress IN to OUT, then
# decompress OUT to /dev/null. That is the only shape in which a library
# and a command-line compressor can be compared honestly -- zpse's own
# numbers here therefore include file I/O and process startup, and are
# lower than tests/bench.c, which measures the codec in-process on 64 KiB
# blocks.
#
# Reported per row: ratio (compressed / input) and MB/s of *uncompressed*
# bytes for both directions. Best of REPS runs, page cache warm.
#
# Usage: bash tests/bench_vs.sh [pattern ...]
#   SIZE=1048576  corpus size per pattern (default 4 MiB, multiple of 64 KiB)
#   REPS=3        timed runs per measurement
#   CC=gcc        corpus/CLI compiler
#
# Prints a markdown table; also appends it to $GITHUB_STEP_SUMMARY when set
# (which is what makes it show up on the workflow run page). Exit 0 unless
# the corpus, the zpse CLI or the harness itself failed -- a missing
# third-party tool is skipped, not an error.
set -uo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
SIZE="${SIZE:-4194304}"
REPS="${REPS:-3}"
CC="${CC:-cc}"
# /dev/shm is not always usable: MSYS reports it as a writable-looking
# directory and then refuses mktemp, which used to leave `work` empty and
# turn every path below into a relative one (the build wrote next to the
# repo). Probe, then fall back, and refuse to run with an empty work dir.
work=""
if [ -d /dev/shm ] && [ -w /dev/shm ]; then
    work="$(mktemp -d /dev/shm/zpse-bench.XXXXXX 2>/dev/null)" || work=""
fi
if [ -z "$work" ]; then
    work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-bench.XXXXXX" 2>/dev/null)" || work=""
fi
[ -n "$work" ] || { echo "bench_vs: no writable temp directory" >&2; exit 1; }
trap 'rm -rf "$work"' EXIT
exe=""
case "$(uname -s 2>/dev/null)" in
MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac

fail() {
    echo "bench_vs: $*" >&2
    exit 1
}

command -v "$CC" >/dev/null 2>&1 || fail "no C compiler ($CC)"

# ---- Build the corpus writer and the zpse CLI -------------------------
"$CC" -std=c11 -O2 -DNDEBUG -I "$inc" "$repo/tests/corpus/corpus_write.c" \
    -o "$work/corpus_write$exe" || fail "building corpus_write"
"$CC" -std=c11 -O2 -DNDEBUG -I "$inc" "$repo/tests/tools/zpse_cli.c" "$unity" \
    -o "$work/zpse_cli$exe" || fail "building zpse_cli"

mkdir -p "$work/corpus"
"$work/corpus_write$exe" "$work/corpus" "$SIZE" || fail "generating corpus"

# ---- Timing -----------------------------------------------------------
# Integer nanoseconds in and out; bash arithmetic is 64-bit on every
# platform this runs on, and 4 MiB never overflows it.
now_ns() {
    date +%s%N
}

# Best-of-REPS wall time in ns for a command; empty when it failed.
timeit() {
    local best="" t0 t1 dt r
    for ((r = 0; r < REPS; r++)); do
        t0=$(now_ns)
        "$@" >/dev/null 2>&1 || return 1
        t1=$(now_ns)
        dt=$((t1 - t0))
        if [ -z "$best" ] || [ "$dt" -lt "$best" ]; then
            best="$dt"
        fi
    done
    echo "$best"
}

mbs() { # bytes ns -> MB/s
    awk -v b="$1" -v n="$2" 'BEGIN { if (n > 0) printf "%.1f", b * 1000.0 / n; else printf "-" }'
}

have() { command -v "$1" >/dev/null 2>&1; }

# Compress IN to OUT, decompress OUT to /dev/null. Each entry sets the two
# command lines as functions so the quoting stays readable.
declare -a TOOLS=()
if [ -x "$work/zpse_cli$exe" ]; then TOOLS+=("zpse"); fi
have zstd && TOOLS+=("zstd -3")
have lz4 && TOOLS+=("lz4 -1")
have brotli && TOOLS+=("brotli -q 5")
have gzip && TOOLS+=("gzip -6")
have xz && TOOLS+=("xz -1")
[ "${#TOOLS[@]}" -gt 0 ] || fail "no compressors found"

tool_name() { echo "${1%% *}"; }

tool_compress() { # tool label IN OUT
    local t="$1" in="$2" out="$3"
    case "$(tool_name "$t")" in
    zpse) "$work/zpse_cli$exe" c "$in" "$out" ;;
    zstd) zstd -q -f "$(echo "$t" | cut -d' ' -f2)" "$in" -o "$out" ;;
    lz4) lz4 -q -f "$(echo "$t" | cut -d' ' -f2)" "$in" "$out" ;;
    brotli) brotli -f -q "${t##* }" "$in" -o "$out" ;;
    gzip) gzip -c "${t##* }" "$in" >"$out" ;;
    xz) xz -c "${t##* }" "$in" >"$out" ;;
    *) return 1 ;;
    esac
}

tool_decompress() { # tool label IN OUT
    local t="$1" in="$2" out="$3"
    case "$(tool_name "$t")" in
    zpse) "$work/zpse_cli$exe" d "$in" "$out" ;;
    zstd) zstd -d -q -f "$in" -o "$out" ;;
    lz4) lz4 -d -q -f "$in" "$out" ;;
    brotli) brotli -d -f "$in" -o "$out" ;;
    gzip) gzip -d -c "$in" >"$out" ;;
    xz) xz -d -c "$in" >"$out" ;;
    *) return 1 ;;
    esac
}

report="$work/report.md"
printf '| pattern | bytes | tool | level | ratio | comp MB/s | decomp MB/s |\n' >"$report"
printf '|---|---:|---|---:|---:|---:|---:|\n' >>"$report"

shopt -s nullglob
patterns=("$@")
if [ "${#patterns[@]}" -eq 0 ]; then
    for f in "$work/corpus"/*.bin; do
        patterns+=("$(basename "$f" .bin)")
    done
fi
shopt -u nullglob
[ "${#patterns[@]}" -gt 0 ] || fail "empty corpus"

for pat in "${patterns[@]}"; do
    in="$work/corpus/$pat.bin"
    [ -f "$in" ] || fail "missing corpus file $in"
    insize=$(wc -c <"$in")
    for t in "${TOOLS[@]}"; do
        name=$(tool_name "$t")
        level="${t#* }"
        [ "$level" = "$t" ] && level="-"
        out="$work/out.$name"
        : >"$out"
        if ! tool_compress "$t" "$in" "$out"; then
            printf '| %s | %s | %s | %s | FAILED | - | - |\n' "$pat" "$insize" "$name" "$level" >>"$report"
            continue
        fi
        outsize=$(wc -c <"$out")
        ratio=$(awk -v a="$outsize" -v b="$insize" 'BEGIN { printf "%.5f", a / b }')
        cns=$(timeit tool_compress "$t" "$in" "$out")
        dns=$(timeit tool_decompress "$t" "$out" /dev/null)
        printf '| %s | %s | %s | %s | %s | %s | %s |\n' \
            "$pat" "$insize" "$name" "$level" "$ratio" \
            "$(mbs "$insize" "${cns:-0}")" "$(mbs "$insize" "${dns:-0}")" \
            >>"$report"
    done
done

cat "$report"
echo
echo "levels: zstd -3, lz4 -1, brotli -q 5, gzip -6, xz -1, zpse default."
echo "MB/s counts uncompressed bytes; best of $REPS; process per run."
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo '### zpse vs other compressors'
        echo
        cat "$report"
        echo
        echo 'Levels: zstd -3, lz4 -1, brotli -q 5, gzip -6, xz -1, zpse default.'
        echo 'MB/s counts uncompressed bytes; best of '"$REPS"'; one process per run.'
    } >>"$GITHUB_STEP_SUMMARY"
fi
