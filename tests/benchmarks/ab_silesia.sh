#!/usr/bin/env bash
# A/B two builds of the codec on Silesia, same corpus, same host, same runs.
#
# Every optimisation in this repository is a claim about a rate, and a rate
# compared against a number measured last week on a warm laptop is not a
# measurement. This runs the two builds back to back in opposite orders and
# prints both rates and the byte delta between them.
#
# Usage: bash tests/benchmarks/ab_silesia.sh [base-ref] [reps] [level] [flags]
#   base-ref   git ref the "before" build is taken from; default HEAD. The
#              "after" build is always the working tree, including uncommitted
#              changes, which is what a change under test is.
#   reps       timed runs per file inside one bench call, default 1
#   level      1-9, default the CLI's own default (6)
#   flags      extra CLI option letters for both builds, e.g. "F" for the
#              fast LZ form, "c" for the payload checksum, "Fc" for both
#   SILESIA_DIR  an already-extracted corpus; downloaded when unset
#   CC         compiler for both builds, default cc
#
# Both builds go through `git archive`, so the base revision is built from the
# source it was committed with rather than from whatever the tree holds now,
# and nothing is written to the repository. The corpus is the same twelve
# files silesia.sh uses: a stray file in SILESIA_DIR is ignored, because a
# denominator that depends on the directory is not a denominator.
#
# Prints a markdown table with each build's packed bytes, ratio and rates, then
# the deltas. Exits 0 when both builds ran and each produced the same byte
# count on both of its runs; 1 otherwise, because an unstable byte count means
# something other than the code changed between runs.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
CC="${CC:-cc}"
base="${1:-HEAD}"
reps="${2:-1}"
level="${3:-}"
extra="${4:-}"

work="$(mktemp -d)" || { echo "ab: no temp dir" >&2; exit 1; }
trap 'rm -rf "$work"' EXIT

# ---- corpus ---------------------------------------------------------------
corpus="${SILESIA_DIR:-}"
if [ -z "$corpus" ]; then
    corpus="$work/silesia"
    echo "ab: downloading the corpus" >&2
    if ! curl -fsSL -o "$work/silesia.zip" \
            "http://mattmahoney.net/dc/silesia.zip"; then
        echo "ab: download failed; set SILESIA_DIR" >&2
        exit 1
    fi
    unzip -q -o "$work/silesia.zip" -d "$corpus" ||
        { echo "ab: unzip failed" >&2; exit 1; }
fi

files=""
for n in dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml; do
    [ -f "$corpus/$n" ] && files="$files $corpus/$n"
done
if [ -z "$files" ]; then
    echo "ab: none of the twelve corpus files are in $corpus" >&2
    exit 1
fi

# ---- the two builds -------------------------------------------------------
build() {   # build <source-root> <output>
    "$CC" -std=c11 -O2 -I"$1/include" "$1/tests/tools/zpse_cli.c" \
        "$1/src/zpse_unity.c" -o "$2"
}

echo "ab: building $base" >&2
mkdir -p "$work/base" || exit 1
git -C "$repo" archive "$base" | tar -x -C "$work/base" ||
    { echo "ab: cannot archive $base" >&2; exit 1; }
build "$work/base" "$work/before" ||
    { echo "ab: base build failed" >&2; exit 1; }

echo "ab: building the working tree" >&2
build "$repo" "$work/after" ||
    { echo "ab: tree build failed" >&2; exit 1; }

pin=""
command -v taskset >/dev/null 2>&1 && pin="taskset -c 0"

opt=""
[ -n "$level" ] && opt="-l $level"
for ((i = 0; i < ${#extra}; i++)); do
    opt="$opt -${extra:$i:1}"
done

# One bench call per build. The TOTAL row is the last line of the CLI's table
# and its columns are: bytes, packed, ratio, encode MB/s, decode MB/s, ms.
total() {   # total <binary>
    # shellcheck disable=SC2086
    $pin "$1" bench $opt -r "$reps" $files 2>/dev/null | grep '^TOTAL'
}

# Opposite orders, so a warming page cache or a drifting clock lands on the
# pair rather than on one of the builds.
a1=$(total "$work/before")
b1=$(total "$work/after")
b2=$(total "$work/after")
a2=$(total "$work/before")

if [ -z "$a1" ] || [ -z "$b1" ]; then
    echo "ab: a bench produced no TOTAL row" >&2
    exit 1
fi

# The better of each build's two runs, independently per rate: the byte count
# is deterministic and is checked separately below.
best() {    # best <col> <line> <line>
    printf '%s\n%s\n' "$2" "$3" | awk -v c="$1" '{ if ($c > m) m = $c } END { print m }'
}
col() { printf '%s' "$1" | awk -v c="$2" '{ print $c }'; }

echo ""
echo "| build | packed | ratio | encode | decode |"
echo "| --- | ---: | ---: | ---: | ---: |"
for pair in "$base:$a1:$a2" "tree:$b1:$b2"; do
    who="${pair%%:*}"
    rest="${pair#*:}"
    one="${rest%%:*}"
    two="${rest#*:}"
    enc=$(best 5 "$one" "$two")
    dec=$(best 6 "$one" "$two")
    printf '| %s | %s | %s:1 | %.2f GB/s | %.2f GB/s |\n' "$who" \
        "$(col "$one" 3)" "$(col "$one" 4)" \
        "$(awk -v v="$enc" 'BEGIN { print v / 1000 }')" \
        "$(awk -v v="$dec" 'BEGIN { print v / 1000 }')"
done

pa=$(col "$a1" 3)
pb=$(col "$b1" 3)
printf '\npacked: %s -> %s (%+d bytes)\n' "$pa" "$pb" "$((pb - pa))"
printf 'encode: %s -> %s MB/s (best of two runs each)\n' \
    "$(best 5 "$a1" "$a2")" "$(best 5 "$b1" "$b2")"
printf 'decode: %s -> %s MB/s\n' "$(best 6 "$a1" "$a2")" "$(best 6 "$b1" "$b2")"

rc=0
for pair in "before:$a1:$a2" "after:$b1:$b2"; do
    who="${pair%%:*}"
    rest="${pair#*:}"
    one="${rest%%:*}"
    two="${rest#*:}"
    if [ "$(col "$one" 3)" != "$(col "$two" 3)" ]; then
        echo "ab: $who wrote $(col "$one" 3) then $(col "$two" 3) bytes" >&2
        rc=1
    fi
done
exit "$rc"
