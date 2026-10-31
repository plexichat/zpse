#!/usr/bin/env bash
# Is zpse on the Pareto frontier?
#
# ratio, encode and decode, for zpse at several levels and for the tools
# anyone would reach for, on the same bytes, with every output sent to
# /dev/null.
#
# Sending the output to /dev/null is the whole point. `bench_vs.sh` and the
# ad-hoc runs behind an earlier version of the README wrote each output to a
# real file, and on this host that write caps EVERY tool at roughly a
# gigabyte a second -- /dev/shm copies at about 1.15 GB/s -- so lz4 and zstd
# were measured at the filesystem's speed and the table said zpse's decode
# was on the frontier when it was not. Here an encoder writes 202 MiB to
# /dev/null (free) and a decoder writes its output there too, so the rate is
# the codec's. The read floor is measured with `cat` on the same input and
# printed under the table, because that is what the numbers still pay for.
#
# Every tool is also round-tripped once and compared byte for byte against
# the input before it is timed: a ratio that does not decode is not a ratio.
#
# A row is on the frontier when no other row is at least as good on all
# three of ratio, encode and decode rate. That is computed below from the
# measurements, not asserted.
#
# Usage: bash tests/benchmarks/bench_pareto.sh [FILE...]
#   No arguments: $SILESIA_DIR if set, otherwise Silesia is downloaded to a
#                 temporary directory (202 MiB, mattmahoney.net).
#   A directory: every regular file in it, by name.
#   Files:       those files, concatenated into one stream.
#
#   REPS=2              timed runs per tool (slow tools are timed once)
#   ZPSE_LEVELS="1 5 6 9"  zpse levels to put on the table
#   LZ4=path ZSTD=path  binaries that are not on PATH (built from source)
#   CC=cc               compiler for the CLI
#   KEEP=1              keep the download directory and print its path
#
# Prints a markdown table; appends it to $GITHUB_STEP_SUMMARY when set.
# Nothing is written to the repository. Exit 0 only when every tool ran and
# round-tripped.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
cli_src="$repo/tests/tools/zpse_cli.c"
CC="${CC:-cc}"
REPS="${REPS:-2}"
ZPSE_LEVELS="${ZPSE_LEVELS:-1 5 6 9}"
work=""
tmp=""
scratch=""
cli=""
rc=0

cleanup() {
    [ -n "$cli" ] && rm -f "$cli"
    [ -n "$scratch" ] && rm -rf "$scratch"
    if [ -n "$tmp" ] && [ "${KEEP:-0}" != "1" ]; then
        rm -rf "$tmp"
    fi
}
trap cleanup EXIT

fail() {
    echo "bench_pareto: $*" >&2
    exit 1
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---- Corpus -----------------------------------------------------------
# The twelve Silesia files, by name: a reused directory can hold a stray
# gmon.out or a half-extracted file, and a denominator that depends on what
# else is lying around is not a denominator.
names="dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml"

work="${SILESIA_DIR:-}"
if [ -z "$work" ] && [ "$#" -gt 0 ] && [ -d "$1" ]; then
    work="$1"
    shift
fi
if [ -z "$work" ] && [ "$#" -eq 0 ]; then
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pareto.XXXXXX")" ||
        fail "cannot create a temp directory"
    work="$tmp"
    url="http://mattmahoney.net/dc/silesia.zip"
    echo "bench_pareto: fetching $url" >&2
    if have curl; then
        curl -sSLo "$work/silesia.zip" "$url" || fail "download failed"
    elif have wget; then
        wget -q -O "$work/silesia.zip" "$url" || fail "download failed"
    else
        fail "needs curl or wget, or a SILESIA_DIR"
    fi
    if have unzip; then
        unzip -q -o "$work/silesia.zip" -d "$work" || fail "unzip failed"
    elif have python3; then
        python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
            "$work/silesia.zip" "$work" || fail "extraction failed"
    else
        fail "needs unzip or python3"
    fi
    rm -f "$work/silesia.zip"
elif [ -n "$work" ] && [ ! -d "$work" ]; then
    fail "SILESIA_DIR=$work is not a directory"
fi

# Files as arguments win; then the twelve Silesia names; then whatever a
# directory holds, which is how the 25-case columnar corpus is passed in.
inputs=()
if [ "$#" -gt 0 ]; then
    for f in "$@"; do
        [ -f "$f" ] && inputs+=("$f")
    done
elif [ -n "$work" ]; then
    for n in $names; do
        [ -f "$work/$n" ] && inputs+=("$work/$n")
    done
    if [ "${#inputs[@]}" -eq 0 ]; then
        for f in "$work"/*; do
            [ -f "$f" ] && inputs+=("$f")
        done
    fi
fi
[ "${#inputs[@]}" -gt 0 ] || fail "no input files"

# One stream, so a ratio is the ratio of the corpus and every tool sees the
# same bytes in the same order. Scratch (the joined input, the packed stream
# and the round-trip output) goes in its own directory: writing a 202 MiB
# file into the corpus directory would leave the corpus different from what
# it was, and a later run could pick it up.
scratch="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pareto-run.XXXXXX")" ||
    fail "cannot create a scratch directory"
in="$scratch/in.bin"
if [ "${#inputs[@]}" -eq 1 ] && [ -f "${inputs[0]}" ] &&
    [ "$(wc -c <"${inputs[0]}")" -gt 1048576 ]; then
    in="${inputs[0]}"
else
    cat "${inputs[@]}" >"$in" || fail "cannot concatenate the inputs"
fi
insize=$(wc -c <"$in")
[ "$insize" -gt 0 ] || fail "empty input"

# ---- Tools ------------------------------------------------------------
command -v "$CC" >/dev/null 2>&1 || fail "no C compiler ($CC)"
cli="$(mktemp "${TMPDIR:-/tmp}/zpse-cli.XXXXXX")" || fail "cannot make a path"
rm -f "$cli"
echo "bench_pareto: building the CLI ($CC)" >&2
"$CC" -std=c11 -O2 -I"$inc" "$cli_src" "$unity" -o "$cli" ||
    fail "CLI build failed"

# $LZ4 / $ZSTD point at a build that is not on PATH, which is the case in a
# sandbox where the binary was compiled from source beside the corpus.
lz4=""
if [ -n "${LZ4:-}" ] && [ -x "${LZ4}" ]; then
    lz4="$LZ4"
elif have lz4; then
    lz4="$(command -v lz4)"
fi
zstd=""
if [ -n "${ZSTD:-}" ] && [ -x "${ZSTD}" ]; then
    zstd="$ZSTD"
elif have zstd; then
    zstd="$(command -v zstd)"
fi

tools=()
for l in $ZPSE_LEVELS; do tools+=("zpse:$l"); done
[ -n "$lz4" ] && tools+=("lz4:1" "lz4:3" "lz4:9")
[ -n "$zstd" ] && tools+=("zstd:1" "zstd:3" "zstd:19")
[ "${#tools[@]}" -gt 0 ] || fail "no compressors found"
[ -n "$lz4" ] || echo "bench_pareto: no lz4, its rows are missing" >&2
[ -n "$zstd" ] || echo "bench_pareto: no zstd, its rows are missing" >&2

# `-f` on every one: /dev/null is not a regular file and both CLIs refuse it
# without the flag.
enc_cmd() { # tool in out
    case "${1%%:*}" in
    zpse) "$cli" c -l "${1##*:}" "$2" "$3" ;;
    lz4) "$lz4" "-${1##*:}" -q -f "$2" "$3" ;;
    zstd) "$zstd" -q -f "-${1##*:}" "$2" -o "$3" ;;
    esac
}

dec_cmd() { # tool in out
    case "${1%%:*}" in
    zpse) "$cli" d "$2" "$3" ;;
    lz4) "$lz4" -d -q -f "$2" "$3" ;;
    zstd) "$zstd" -q -d -f "$2" -o "$3" ;;
    esac
}

now_ns() { date +%s%N; }

best_ns() { # reps cmd...
    local r="$1" best="" t0 t1 dt
    shift
    for ((r = r; r > 0; r--)); do
        t0=$(now_ns)
        "$@" >/dev/null 2>&1 || return 1
        t1=$(now_ns)
        dt=$((t1 - t0))
        if [ -z "$best" ] || [ "$dt" -lt "$best" ]; then best="$dt"; fi
    done
    echo "$best"
}

mbs() { # bytes ns
    awk -v b="$1" -v n="$2" 'BEGIN { printf "%.0f", (n > 0) ? b * 1000.0 / n : 0 }'
}

ratio() { # packed input
    awk -v c="$1" -v i="$2" 'BEGIN { printf "%.2f", (c > 0) ? i / c : 0 }'
}

# A tool that needs a second per run is timed once: the point of REPS is to
# beat the spread on the fast ones, and three runs of zstd -19 is five
# minutes of wall clock for the same answer.
reps_for() {
    case "$1" in
    zstd:19 | lz4:9) echo 1 ;;
    *) echo "$REPS" ;;
    esac
}

rows="$scratch/rows.txt"
: >"$rows"
packed="$scratch/packed.bin"
rt="$scratch/rt.bin"

for t in "${tools[@]}"; do
    if ! enc_cmd "$t" "$in" "$packed" 2>/dev/null; then
        echo "bench_pareto: $t encode failed" >&2
        rc=1
        continue
    fi
    psize=$(wc -c <"$packed")
    if ! dec_cmd "$t" "$packed" "$rt" 2>/dev/null || ! cmp -s "$in" "$rt"; then
        echo "bench_pareto: $t round trip FAILED" >&2
        rc=1
        continue
    fi
    r=$(reps_for "$t")
    ens=$(best_ns "$r" enc_cmd "$t" "$in" /dev/null) ||
        { echo "bench_pareto: $t encode timing failed" >&2; rc=1; continue; }
    des=$(best_ns "$r" dec_cmd "$t" "$packed" /dev/null) ||
        { echo "bench_pareto: $t decode timing failed" >&2; rc=1; continue; }
    printf '%s|%s|%s|%s|%s\n' "$t" "$(ratio "$psize" "$insize")" \
        "$(mbs "$insize" "$ens")" "$(mbs "$insize" "$des")" "$psize" >>"$rows"
done
rm -f "$packed" "$rt"

[ "$(wc -l <"$rows")" -gt 0 ] || fail "no tool produced a row"

cats=$(now_ns)
cat "$in" >/dev/null
cats=$((($(now_ns) - cats)))

# ---- Report -----------------------------------------------------------
{
    echo "| tool | packed | ratio | encode | decode | frontier |"
    echo "| --- | ---: | ---: | ---: | ---: | :---: |"
    awk -F'|' '
        { name[NR] = $1; r[NR] = $2; e[NR] = $3; d[NR] = $4; p[NR] = $5; n = NR }
        END {
            for (i = 1; i <= n; i++) {
                fr = 1
                for (j = 1; j <= n; j++) {
                    if (j == i) continue
                    if (r[j] + 0 >= r[i] + 0 && e[j] + 0 >= e[i] + 0 &&
                        d[j] + 0 >= d[i] + 0 &&
                        (r[j] + 0 > r[i] + 0 || e[j] + 0 > e[i] + 0 ||
                         d[j] + 0 > d[i] + 0)) { fr = 0; break }
                }
                printf "| %s | %s | %s:1 | %s MB/s | %s MB/s | %s |\n",
                       name[i], p[i], r[i], e[i], d[i], (fr ? "**yes**" : "no")
            }
        }' "$rows"
    echo
    echo "MB/s counts uncompressed bytes. Best of $REPS run(s) per tool"
    echo "(zstd -19 and lz4 -9 once). Output to /dev/null, so the rate is the"
    echo "codec's; reading the input still costs $(mbs "$insize" "$cats") MB/s of"
    echo "it, which is the read floor on this host. Every row round-tripped"
    echo "and compared whole with the input before it was timed."
    echo "Frontier: no other row is at least as good on all three columns."
} | tee "$scratch/report.md"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo
        echo "### Pareto: zpse against lz4 and zstd"
        echo
        cat "$scratch/report.md"
    } >>"$GITHUB_STEP_SUMMARY"
fi

exit "$rc"
