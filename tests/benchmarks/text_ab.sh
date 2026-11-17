#!/usr/bin/env bash
# Focused A/B harness for TEXT: encode rate, decode rate and ratio on
# deterministic, word-repetitive prose generated in C (nothing is read from
# disk). Two workloads: 4 KiB pages and 64 KiB blocks over a 4 MiB corpus each.
#
# Usage:
#   bash tests/benchmarks/text_ab.sh [options]          # one build, one table
#   bash tests/benchmarks/text_ab.sh [options] A B      # A/B, alternating rounds
#
# Options:
#   -r REPS  best-of-N encode/decode reps per run (default 5)
#   -l LIST  comma-separated levels, e.g. 1,6,9 (default 6)
#   -c CPU   taskset CPU for timed runs (default 0; empty string disables pin)
#   -d DIR   lz4 source dir (needs lib/lz4.c, lib/lz4hc.c, lib/lz4.h); enables
#            lz4 and lz4hc rows in both builds
#   -s DIR   source tree to build from in single mode (default repo root)
#   -R N     alternating rounds in A/B mode (default 8; env ROUNDS also works)
#
# In A/B mode A and B are source-tree directories (e.g. a worktree of the old
# commit and the current checkout). The harness source always comes from the
# current repo; only include/ and src/zpse_unity.c come from A and B. Order is
# swapped each round (a,b,b,a,...) so host drift does not favour one build,
# and each round's number is a best-of-REPS inside that build. "B/A" > 1 means
# B is faster. Build logs go to stderr; the table goes to stdout.
#
# Nothing is deleted: the temp build dir is kept and its path is printed to
# stderr. Set KEEP=0 has no effect. No files in the repository are modified.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
harness="$repo/tests/benchmarks/bench_text.c"

CC="${CC:-cc}"
REPS="${REPS:-5}"
LEVELS="${LEVELS:-6}"
CPU="${CPU:-0}"
LZ4_DIR="${LZ4_DIR:-}"
SRC="${SRC:-$repo}"
ROUNDS="${ROUNDS:-8}"

usage() { echo "usage: text_ab.sh [-r reps] [-l levels] [-c cpu] [-d lz4dir] [-s src] [-R rounds] [A B]" >&2; }
die() { echo "text_ab: $*" >&2; exit 1; }

while getopts "r:l:c:d:s:R:h" opt; do
    case "$opt" in
    r) REPS=$OPTARG ;;
    l) LEVELS=$OPTARG ;;
    c) CPU=$OPTARG ;;
    d) LZ4_DIR=$OPTARG ;;
    s) SRC=$OPTARG ;;
    R) ROUNDS=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

for v in REPS ROUNDS; do
    eval val=\$$v
    case "$val" in
    ''|*[!0-9]*) die "$v must be a positive integer" ;;
    esac
    [ "$val" -ge 1 ] || die "$v must be a positive integer"
done

# Optional lz4: a header/source dir adds the two rows and charges 2 length
# bytes per block, the same fairness rule as page_table_static.sh.
lz4_def=""; lz4_inc=""; lz4_src=""
if [ -n "$LZ4_DIR" ]; then
    lz4lib="$LZ4_DIR/lib"
    for f in lz4.c lz4hc.c lz4.h; do
        [ -f "$lz4lib/$f" ] || die "LZ4_DIR/lib/$f is missing"
    done
    lz4_def="-DZPSE_TEXT_HAVE_LZ4=1"
    lz4_inc="-I$lz4lib"
    lz4_src="$lz4lib/lz4.c $lz4lib/lz4hc.c"
fi

check_tree() {  # dir
    [ -f "$1/include/zpse/zpse.h" ] || die "$1: include/zpse/zpse.h missing"
    [ -f "$1/src/zpse_unity.c" ]    || die "$1: src/zpse_unity.c missing"
}
build_harness() {  # src_tree out
    check_tree "$1" || return 1
    # shellcheck disable=SC2086  # lz4_def/lz4_inc/lz4_src are intentional words
    "$CC" -std=c11 -O2 $lz4_def -I"$1/include" $lz4_inc \
        "$harness" $lz4_src "$1/src/zpse_unity.c" -o "$2"
}

work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-text-ab.XXXXXX")" || die "no temp dir"
echo "text_ab: build dir $work (kept)" >&2

pin=()
if [ -n "$CPU" ] && command -v taskset >/dev/null 2>&1; then
    pin=(taskset -c "$CPU")
    echo "text_ab: pinning timed runs to CPU $CPU" >&2
else
    echo "text_ab: no taskset or empty CPU; running unpinned" >&2
fi

run_harness() {  # bin outfile
    "${pin[@]}" "$1" --reps="$REPS" --levels="$LEVELS" > "$2"
}

if [ "$#" -eq 0 ]; then
    # ---- single build ----------------------------------------------------
    check_tree "$SRC"
    echo "text_ab: building bench_text from $SRC with $CC" >&2
    build_harness "$SRC" "$work/bench_text" || die "harness build failed"
    rev=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || printf 'worktree')
    echo "## text table — single build ($rev)"
    echo
    echo "Source \`$SRC\`, CC \`$CC\`, reps $REPS, levels $LEVELS, lz4 ${LZ4_DIR:-off}."
    echo "Corpus is generated in C and is byte-identical across builds. Ratios are"
    echo "exact; rates are best of $REPS, pinned to CPU ${CPU:-none} where available."
    echo
    run_harness "$work/bench_text" "$work/table.txt" || die "harness run failed"
    cat "$work/table.txt"
    exit 0
fi

# ---- A/B: two source trees, alternating rounds --------------------------
[ "$#" -eq 2 ] || { usage; exit 2; }
A="$1"; B="$2"
check_tree "$A"; check_tree "$B"
echo "text_ab: building A from $A" >&2
build_harness "$A" "$work/harness_a" || die "harness build failed for A"
echo "text_ab: building B from $B" >&2
build_harness "$B" "$work/harness_b" || die "harness build failed for B"

: > "$work/data.tsv"
for ((r = 1; r <= ROUNDS; r++)); do
    if (( r % 2 )); then order=(a b); else order=(b a); fi
    for nm in "${order[@]}"; do
        out="$work/run.$nm.$r.txt"
        run_harness "$work/harness_$nm" "$out" || die "run failed ($nm round $r)"
        awk -F'|' -v nm="$nm" -v rd="$r" '
            /^\| text-/ {
                w=$2; m=$3; gsub(/^ +| +$/,"",w); gsub(/^ +| +$/,"",m)
                e=$5; d=$6; gsub(/ /,"",e); gsub(/ /,"",d)
                if (e+0>0 && d+0>0)
                    printf "%s\t%d\t%s\t%s\t%s\t%s\n", nm, rd, w, m, e, d
            }' "$out" >> "$work/data.tsv" || die "parse failed ($nm round $r)"
    done
done
[ -s "$work/data.tsv" ] || die "no data rows collected"

echo
echo "## text A/B: $A (A) vs $B (B)"
echo
echo "Best of $REPS inside each round, best across $ROUNDS alternating rounds"
echo "(order a,b,b,a,...). enc/dec B/A > 1 means B is faster. Ratios are exact"
echo "and are identical across builds when the codec is unchanged."
echo
printf '| workload | mode | A enc | B enc | enc B/A | A dec | B dec | dec B/A |\n'
printf '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
awk -v R="$ROUNDS" '
BEGIN { FS="\t" }
{
    nm=$1; rd=$2+0; w=$3; m=$4; e=$5+0; d=$6+0
    key=w SUBSEP m
    seen[key]=1
    val[key SUBSEP nm SUBSEP "e" SUBSEP rd]=e
    val[key SUBSEP nm SUBSEP "d" SUBSEP rd]=d
}
function best(key,nm,metric,   r,v,mx) {
    mx=0
    for (r=1; r<=R; r++) {
        k=key SUBSEP nm SUBSEP metric SUBSEP r
        if (k in val) { v=val[k]; if (v>mx) mx=v }
    }
    return mx
}
function sortstr(a,n,   i,j,x) {
    for (i=2; i<=n; i++) { x=a[i]; j=i-1
        while (j>0 && a[j]>x) { a[j+1]=a[j]; j-- } a[j+1]=x }
}
END {
    n=0
    for (k in seen) keys[++n]=k
    sortstr(keys,n)
    for (i=1; i<=n; i++) {
        k=keys[i]
        split(k,p,SUBSEP)
        ae=best(k,"a","e"); be=best(k,"b","e")
        ad=best(k,"a","d"); bd=best(k,"b","d")
        if (ae<=0 || be<=0 || ad<=0 || bd<=0) continue
        printf "| %-9s | %-7s | %8.0f | %8.0f | %7.3f | %8.0f | %8.0f | %7.3f |\n", \
            p[1], p[2], ae, be, be/ae, ad, bd, bd/ad
    }
}' "$work/data.tsv"

echo
echo "Corpus: deterministic prose, 4 MiB per workload. Same harness source,"
echo "compiler and flags in both builds; only include/ and src/ differ."
exit 0
