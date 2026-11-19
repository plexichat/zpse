#!/usr/bin/env bash
# Focused A/B harness for FLAT7: encode rate, decode rate and ratio against the
# ordinary frame path, on deterministic ASCII generated in C (nothing is read
# from disk). Two corpora of the same size: word-repetitive prose, which the
# redundancy gate declines, and high-entropy random printable ASCII, which it
# accepts.
#
# One build, one table. The harness calls zpse_flat7_encode/decode for the
# FLAT7 row and zpse_compress_frame_parallel (one worker, ZPSE_ENC_CHECKSUM to
# force a normal frame) / zpse_decompress_frame for the normal row. Every row
# is round-tripped and compared whole before it is timed; rates are best of N
# with no I/O inside the timed loops, pinned to one CPU where available.
# Ratios are exact and written uncompressed:compressed as x:1.
#
# Usage:
#   bash tests/benchmarks/flat7_ab.sh [options]
#
# Options:
#   -r REPS  best-of-N encode/decode reps per row (default 5)
#   -n SIZE  corpus bytes per workload, a multiple of 65536 (default 4 MiB)
#   -c CPU   taskset CPU for timed runs (default 0; empty string disables pin)
#   -h       this help
#
# Nothing is deleted: the temp build dir is kept and its path printed to stderr.
# No files in the repository are modified.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
harness="$repo/tests/benchmarks/bench_flat7.c"

CC="${CC:-cc}"
REPS="${REPS:-5}"
CPU="${CPU:-0}"
SIZE="${SIZE:-}"

usage() { echo "usage: flat7_ab.sh [-r reps] [-n size] [-c cpu]" >&2; }
die() { echo "flat7_ab: $*" >&2; exit 1; }

while getopts "r:n:c:h" opt; do
    case "$opt" in
    r) REPS=$OPTARG ;;
    n) SIZE=$OPTARG ;;
    c) CPU=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

case "$REPS" in
    ''|*[!0-9]*) die "REPS must be a positive integer" ;;
esac
[ "$REPS" -ge 1 ] || die "REPS must be a positive integer"
if [ -n "$SIZE" ]; then
    case "$SIZE" in
        ''|*[!0-9]*) die "SIZE must be a positive multiple of 65536" ;;
    esac
    [ $((SIZE % 65536)) -eq 0 ] || die "SIZE must be a positive multiple of 65536"
fi

[ -f "$repo/src/zpse_unity.c" ] || die "$repo: src/zpse_unity.c missing"
[ -f "$harness" ] || die "$harness missing"

work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-flat7-ab.XXXXXX")" || die "no temp dir"
echo "flat7_ab: build dir $work (kept)" >&2

pin=()
if [ -n "$CPU" ] && command -v taskset >/dev/null 2>&1; then
    pin=(taskset -c "$CPU")
    echo "flat7_ab: pinning timed runs to CPU $CPU" >&2
else
    echo "flat7_ab: no taskset or empty CPU; running unpinned" >&2
fi

echo "flat7_ab: building bench_flat7 from $repo with $CC" >&2
"$CC" -std=c11 -O2 -I"$repo/include" "$harness" "$repo/src/zpse_unity.c" \
    -o "$work/bench_flat7" || die "harness build failed"

args=("--reps=$REPS")
[ -n "$SIZE" ] && args+=("--size=$SIZE")

rev=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || printf 'worktree')
echo
echo "## FLAT7 A/B table — single build ($rev)"
echo
echo "Source \`$repo\`, CC \`$CC\`, best of $REPS, one worker, CPU ${CPU:-none}."
echo "Corpora are generated in C and byte-identical across builds; ratios are"
echo "exact (x:1) and rates are best of $REPS in MB/s of uncompressed bytes."
echo "normal is forced with ZPSE_ENC_CHECKSUM, which FLAT7 declines."
echo
"${pin[@]}" "$work/bench_flat7" "${args[@]}" || die "harness run failed"
exit 0
