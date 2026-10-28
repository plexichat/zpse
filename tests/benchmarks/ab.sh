#!/usr/bin/env bash
# Interleaved A/B for one change: build tests/benchmarks/bench_columnar.c
# against two source trees, run them alternately on the same CPU, and report
# the per-case encode and decode rate ratio of the second tree to the first.
#
# Why a script and not two timed runs: a single before/after pair on a shared
# or virtualised machine carries whatever the machine was doing in between.
# Here each round runs both binaries back to back (order swapped every
# round), the estimator is the best of the rounds per (tree, case), and the
# table prints every case so a change that helps three columns and hurts ten
# cannot be summarised away by an average.
#
# Two summaries close the table:
#   geomean       the mean of the per-case ratios -- every case counts once
#   time-weighted total time of tree A over total time of tree B, i.e. the
#                 ratio you would see on the whole corpus (a case that runs
#                 at 40 GB/s contributes almost no time, so it cannot
#                 outweigh one that runs at 2 GB/s)
#
# Usage: ab.sh [options] TREE_A TREE_B
#   -e ITERS   inner iterations per block, passed to the benchmark (default 6)
#   -r ROUNDS  interleaved rounds (default 8)
#   -b KB      block size in KiB (default 64)
#   -c CPU     taskset CPU (default 2; empty disables the pin)
# A tree is a directory holding include/, src/ and tests/ (a git worktree or a
# plain copy of this one). Build both from the same recipe, on the same host,
# in the same configuration -- the script does that part.
set -uo pipefail

ITERS=6
ROUNDS=8
BLK=64
CPU=2
while getopts "e:r:b:c:" opt; do
    case "$opt" in
    e) ITERS=$OPTARG ;;
    r) ROUNDS=$OPTARG ;;
    b) BLK=$OPTARG ;;
    c) CPU=$OPTARG ;;
    *) echo "usage: ab.sh [-e iters] [-r rounds] [-b kb] [-c cpu] A B" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [ $# -ne 2 ]; then
    echo "usage: ab.sh [-e iters] [-r rounds] [-b kb] [-c cpu] A B" >&2
    exit 2
fi
A=$1
B=$2

pin() {
    if [ -n "$CPU" ]; then taskset -c "$CPU" "$@"; else "$@"; fi
}

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
for t in "$A" "$B"; do
    name=$(basename "$t")
    if ! ${CC:-gcc} -std=c11 -O2 -I "$t/include" -I "$t/tests/corpus" \
        "$t/tests/benchmarks/bench_columnar.c" "$t/src/zpse_unity.c" \
        -o "$OUT/bin_$name" -ldl; then
        echo "build failed for $t" >&2
        exit 1
    fi
done

n=0
for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then order=("$A" "$B"); else order=("$B" "$A"); fi
    for t in "${order[@]}"; do
        name=$(basename "$t")
        pin "$OUT/bin_$name" "$ITERS" 1 all zpse "$BLK" 0 \
            > "$OUT/$name.$r.txt" 2>&1
        n=$((n + 1))
    done
done
if [ "$n" -ne $((2 * ROUNDS)) ]; then
    echo "a round did not produce output" >&2
    exit 1
fi

awk -v a="$(basename "$A")" -v b="$(basename "$B")" '
function rate(s,   p, u) {
    # "812.3 MB/s (790/830)" or "6.1 GB/s (5.9/6.3)" -> MB/s
    split(s, p, " ")
    u = (p[2] == "GB/s") ? 1000.0 : 1.0
    return (p[1] + 0.0 >= 0) ? p[1] * u : 0
}
/^\| [a-z0-9_]+ \| zpse/ {
    # each row carries a rate; which tree it belongs to is in the file name
    t = FILENAME; sub(/.*\//, "", t); sub(/\.[0-9]+\.txt$/, "", t)
    split($0, f, "|")
    c = f[2]; gsub(/ /, "", c)
    enc = rate(f[5]); dec = rate(f[6])
    if (enc > be[t, c]) be[t, c] = enc
    if (dec > bd[t, c]) bd[t, c] = dec
    if (!(c in seen)) { seen[c] = 1; ord[++nc] = c }
}
END {
    if (nc == 0) { print "no zpse rows parsed"; exit 1 }
    printf "%-16s %12s %12s %8s   %12s %12s %8s\n", "case", \
        a " enc", b " enc", "ratio", a " dec", b " dec", "ratio"
    for (i = 1; i <= nc; i++) {
        c = ord[i]
        printf "%-16s %12.1f %12.1f %8.4f   %12.1f %12.1f %8.4f\n", c, \
            be[a, c], be[b, c], (be[a, c] > 0 ? be[b, c] / be[a, c] : 0), \
            bd[a, c], bd[b, c], (bd[a, c] > 0 ? bd[b, c] / bd[a, c] : 0)
    }
    for (i = 1; i <= nc; i++) {
        c = ord[i]
        if (be[a, c] > 0) { ge += log(be[b, c] / be[a, c]); te1 += 1 / be[a, c]; te2 += 1 / be[b, c] }
        if (bd[a, c] > 0) { gd += log(bd[b, c] / bd[a, c]); td1 += 1 / bd[a, c]; td2 += 1 / bd[b, c] }
    }
    printf "ENCODE geomean %s/%s = %.5f   time-weighted = %.5f\n", b, a, exp(ge / nc), te1 / te2
    printf "DECODE geomean %s/%s = %.5f   time-weighted = %.5f\n", b, a, exp(gd / nc), td1 / td2
}
' "$OUT"/*.txt
