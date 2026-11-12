#!/usr/bin/env bash
# Regenerate every README table a harness in this tree can produce.
#
# Usage: bash tests/benchmarks/refresh_tables.sh [--fast]
#
#   --fast   fewer iterations and reps per harness; the numbers are noisier
#
# The corpora and tools the tables name are fetched into a temporary directory
# when the host does not have them -- Silesia, `titanic.csv`,
# `uber-rides-data1.csv`, and an `lz4` CLI built from source -- so the tables
# can be re-measured on a bare machine. Nothing is written into the repository.
#
# Every section prints markdown under a "## table: <name>" line, named for the
# table it refreshes in README.md. A section whose corpus or tool is missing
# says so on stderr and sets the exit status; it never prints a table it did
# not measure. The exit status is 0 only when every section ran.
#
# Env: CC (default cc), REPS (2), ZPSE_LEVELS ("1 5 6 9 F"),
#      SILESIA_DIR, LZ4, ZSTD, KEEP=1 (keep the temp directory and print it).
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
CC="${CC:-cc}"
REPS="${REPS:-2}"
ZPSE_LEVELS="${ZPSE_LEVELS:-1 5 6 9 F}"
fast=0
[ "${1:-}" = "--fast" ] && fast=1
work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-refresh.XXXXXX")" || exit 1
rc=0

cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "refresh_tables: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

say() { echo "## table: $*"; }

missing() {
    echo "refresh_tables: no $* -- that table was not refreshed" >&2
    rc=1
}

have() { command -v "$1" >/dev/null 2>&1; }

# $1 = tool name, $2... = command; prints and returns its status.
run() { "$@"; }

# ---- Tools ------------------------------------------------------------

echo "refresh_tables: building the CLI" >&2
"$CC" -std=c11 -O2 -I "$inc" "$repo/tests/tools/zpse_cli.c" "$unity" \
    -o "$work/zpse" || { echo "refresh_tables: CLI build failed" >&2; exit 1; }
cli="$work/zpse"

"$CC" -std=c11 -O2 -I "$inc" "$repo/tests/corpus/corpus_write.c" \
    -o "$work/corpus_write" 2>/dev/null ||
    "$CC" -std=c11 -O2 -I "$inc" -I "$repo/tests/corpus" \
        "$repo/tests/corpus/corpus_write.c" -o "$work/corpus_write" ||
    { echo "refresh_tables: corpus_write build failed" >&2; exit 1; }

zstd=""
if [ -n "${ZSTD:-}" ] && [ -x "${ZSTD}" ]; then
    zstd="$ZSTD"
elif have zstd; then
    zstd="$(command -v zstd)"
fi

# lz4's CLI, which the Pareto tables drive. Built from source in the temp
# directory when it is not installed: this project's comparisons should not be
# blocked by a distro package.
lz4=""
if [ -n "${LZ4:-}" ] && [ -x "${LZ4}" ]; then
    lz4="$LZ4"
elif have lz4; then
    lz4="$(command -v lz4)"
else
    echo "refresh_tables: fetching lz4" >&2
    if curl -sSL -o "$work/lz4.tar.gz" \
        https://github.com/lz4/lz4/archive/refs/tags/v1.10.0.tar.gz 2>/dev/null &&
        tar xzf "$work/lz4.tar.gz" -C "$work" 2>/dev/null &&
        make -C "$work/lz4-1.10.0" -j4 lz4 >/dev/null 2>&1; then
        lz4="$work/lz4-1.10.0/lz4"
    fi
fi

# ---- Corpora ----------------------------------------------------------

corpus="$work/corpus"
mkdir -p "$corpus" || exit 1
"$work/corpus_write" "$corpus" >/dev/null || exit 1

silesia="${SILESIA_DIR:-}"
if [ -z "$silesia" ]; then
    silesia="$work/silesia"
    echo "refresh_tables: fetching Silesia" >&2
    mkdir -p "$silesia"
    if have curl; then
        curl -sSLo "$work/silesia.zip" \
            http://mattmahoney.net/dc/silesia.zip 2>/dev/null || rc=1
    elif have wget; then
        wget -q -O "$work/silesia.zip" \
            http://mattmahoney.net/dc/silesia.zip || rc=1
    else
        rc=1
    fi
    if have unzip; then
        unzip -q -o "$work/silesia.zip" -d "$silesia" >/dev/null 2>&1 || rc=1
    elif have python3; then
        python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
            "$work/silesia.zip" "$silesia" || rc=1
    else
        rc=1
    fi
fi

# The two real CSV exports the colcheck table names. They live in public
# repositories; a fetch failure means that table is skipped, not guessed.
titanic="$work/titanic.csv"
uber="$work/uber-rides-data1.csv"
curl -sSL --max-time 60 -o "$titanic" \
    https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv \
    2>/dev/null || rm -f "$titanic"
curl -sSL --max-time 180 -o "$uber" \
    https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv \
    2>/dev/null || rm -f "$uber"

# ---- zstd ratio tables ------------------------------------------------

# $1 = section name, $2... = corpus files
ratio_table() {
    local name="$1"
    shift
    local f z p
    say "$name"
    echo "| file | bytes | zpse | zstd -3 | |"
    echo "| --- | ---: | ---: | ---: | --- |"
    for f in "$@"; do
        [ -f "$f" ] || continue
        z=$("$zstd" -q -3 -c "$f" | wc -c)
        # One file at a time through the block bench: its own packed total,
        # 64 KiB blocks, the default level.
        p=$("$cli" bench -q -b 65536 -r 1 "$f" | awk '/TOTAL/ {print $3}')
        awk -v n="$(basename "$f")" -v b="$(wc -c <"$f")" -v p="$p" -v z="$z" \
            'BEGIN {
                pr = p / b; zr = z / b
                if (pr > zr) printf "| %s | %d | %.4f | %.5f | zstd %.1fx smaller |\n", n, b, pr, zr, pr / zr
                else if (zr > pr) printf "| %s | %d | %.4f | %.5f | **zpse %.1fx smaller** |\n", n, b, pr, zr, zr / pr
                else printf "| %s | %d | %.4f | %.5f | tie |\n", n, b, pr, zr
            }'
    done
}

if [ -n "$zstd" ]; then
    ratio_table "zstd ratios, byte patterns" "$corpus"/zeros.bin \
        "$corpus"/const5A.bin "$corpus"/incr.bin "$corpus"/strided.bin \
        "$corpus"/sparse.bin "$corpus"/sparse2.bin "$corpus"/alt00FF.bin \
        "$corpus"/sawtooth.bin "$corpus"/lowentropy.bin "$corpus"/random.bin
    ratio_table "zstd ratios, columnar" "$corpus"/ts_seq.bin \
        "$corpus"/ts_ns.bin "$corpus"/float_sparse.bin \
        "$corpus"/var_string.bin "$corpus"/sparse_mix.bin \
        "$corpus"/geo_point.bin "$corpus"/struct_offsets.bin
else
    missing "zstd CLI"
fi

# ---- Reference column codecs ------------------------------------------

if "$CC" -std=c11 -O2 -I "$inc" "$repo/tests/benchmarks/bench_colref.c" \
    "$unity" -o "$work/colref" 2>/dev/null; then
    if [ "$fast" = 1 ]; then it=3; rep=1; else it=20; rep=3; fi
    say "reference column codecs"
    taskset -c 0 "$work/colref" "$it" "$rep" 2>/dev/null
else
    missing "bench_colref build"
fi

# ---- Transform map and per-case rows ----------------------------------

if "$CC" -std=c11 -O2 -DNDEBUG -I "$inc" \
    "$repo/tests/benchmarks/bench_columnar.c" "$unity" -ldl \
    -o "$work/columnar" 2>/dev/null; then
    if [ "$fast" = 1 ]; then it=1; rep=1; else it=10; rep=3; fi
    taskset -c 0 "$work/columnar" "$it" "$rep" all zpse >"$work/col.txt" \
        2>/dev/null
    say "transform selection"
    grep '^transforms(' "$work/col.txt"
    say "columnar rows (zpse default)"
    grep '| zpse default |' "$work/col.txt"
    # The per-case LZ-form table: the default, MAX, and the tile-only arm
    # (level 4, the deepest rung with no match finder).
    taskset -c 0 "$work/columnar" 1 1 all zpse 64 8 2>/dev/null |
        grep '| zpse default |' >"$work/tile.txt"
    taskset -c 0 "$work/columnar" 1 1 all zpse 64 18 2>/dev/null |
        grep '| zpse default |' >"$work/max.txt"
    say "per-case LZ form (tile-only arm is level 4)"
    echo "| case | \`ZPSE_ENC_SPEED_BALANCED\` | \`ZPSE_ENC_MAX\` | tile stream alone |"
    echo "| --- | ---: | ---: | ---: |"
    cells() { # file -> case|ratio, one per line
        awk -F'|' '{
            c = $2; r = $4
            gsub(/^ +| +$/, "", c); gsub(/^ +| +$/, "", r)
            if (c != "") print c "|" r
        }' "$1" | LC_ALL=C sort
    }
    cells "$work/col.txt" >"$work/d.rat"
    cells "$work/max.txt" >"$work/m.rat"
    cells "$work/tile.txt" >"$work/t.rat"
    join -t'|' "$work/d.rat" "$work/m.rat" | join -t'|' - "$work/t.rat" |
        awk -F'|' '{ printf "| `%s` | %s | %s | %s |\n", $1, $2, $3, $4 }'
else
    missing "bench_columnar build (dlopen?)"
fi

# ---- Pareto tables ----------------------------------------------------

if [ -n "$lz4" ] && [ -n "$zstd" ]; then
    say "pareto, Silesia"
    REPS="$REPS" ZPSE_LEVELS="$ZPSE_LEVELS" SILESIA_DIR="$silesia" \
        LZ4="$lz4" ZSTD="$zstd" CC="$CC" \
        bash "$repo/tests/benchmarks/bench_pareto.sh" 2>/dev/null
    say "pareto, the 17-file columnar corpus"
    REPS="$REPS" ZPSE_LEVELS="$ZPSE_LEVELS" LZ4="$lz4" ZSTD="$zstd" \
        CC="$CC" bash "$repo/tests/benchmarks/bench_pareto.sh" \
        "$corpus"/*.bin 2>/dev/null
else
    missing "lz4 CLI (built from source if the fetch works) or zstd"
fi

# ---- Mixed page tables ------------------------------------------------

# The page tables quote a mixed workload. It is generated here rather than
# scraped from the host, so these rows mean the same thing on every machine:
# 160 MiB interleaving prose, source, executable-like bytes, library-like
# blobs, records and opaque chunks, one 64 KiB chunk at a time.
mixed="$work/mixed"
mfile="$mixed/mixed.bin"
mkdir -p "$mixed" || exit 1
if "$work/corpus_write" "$mixed" 167772160 --mixed && [ -s "$mfile" ]; then
    msize=$(wc -c <"$mfile")
    # Best wall time in ns over REPS runs; fails if any run fails.
    timed() {
        local best="" t0 t1 dt r
        for ((r = 0; r < REPS; r++)); do
            t0=$(date +%s%N)
            "$@" >/dev/null 2>&1 || return 1
            t1=$(date +%s%N)
            dt=$((t1 - t0))
            if [ -z "$best" ] || [ "$dt" -lt "$best" ]; then best=$dt; fi
        done
        printf '%s\n' "$best"
    }
    # One frame per file at 4 KiB blocks -- the method the caption names.
    zcell() { # level -> ratio|encode MB/s|decode MB/s
        local zf="$mixed/m$1.zst" sz te td
        "$zstd" -q -T1 -B4K -"$1" -c "$mfile" >"$zf" || return 1
        sz=$(wc -c <"$zf")
        te=$(timed "$zstd" -q -T1 -B4K -"$1" -c "$mfile") || return 1
        td=$(timed "$zstd" -q -d -c "$zf") || return 1
        awk -v s="$sz" -v n="$msize" -v e="$te" -v d="$td" 'BEGIN {
            printf "%.2f|%.0f|%.1f", 100.0 * s / n,
                n * 1000.0 / e, n * 1000.0 / d }'
    }
    P=()
    for l in 1 6 9; do
        P+=("$("$cli" bench -q -b 4096 -l "$l" -r "$REPS" "$mfile" |
            awk '/TOTAL/ {printf "%.2f|%.0f|%.1f", 100.0*$3/$2, $5, $6}')")
    done
    Z=()
    if [ -n "$zstd" ]; then
        for l in 1 6 9; do
            c=$(zcell "$l") || break
            Z+=("$c")
        done
    fi
    if [ "${#Z[@]}" -eq 3 ]; then
        all=("${P[@]}" "${Z[@]}")
        hdr="| 4 KiB pages, mixed | level 1 | level 6 | level 9 | zstd 1 | zstd 6 | zstd 9 |"
        sep="| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    else
        all=("${P[@]}")
        hdr="| 4 KiB pages, mixed | level 1 | level 6 | level 9 |"
        sep="| --- | ---: | ---: | ---: |"
    fi
    # $1 label, $2 field of "ratio|enc|dec", $3 suffix; rest = cells.
    row() {
        local label="$1" field="$2" suffix="$3" c v out=""
        shift 3
        for c in "$@"; do
            v=$(printf '%s' "$c" | cut -d'|' -f"$field")
            if [ "$suffix" = " GB/s" ]; then
                v=$(awk -v x="$v" 'BEGIN { printf "%.1f", x / 1000.0 }')
            fi
            out="$out $v$suffix |"
        done
        printf '| %s |%s\n' "$label" "$out"
    }
    say "mixed 4 KiB pages"
    echo "$hdr"
    echo "$sep"
    row ratio 1 "%" "${all[@]}"
    row encode 2 " MB/s" "${all[@]}"
    row decode 3 " GB/s" "${all[@]}"

    # The PAGE_FAST row alone: the other rows of that table are the host
    # corpora, which this tree cannot regenerate.
    say "mixed PAGE_FAST row"
    echo "| 4 KiB pages | plain L1 | plain L2 | plain BALANCED | \`PAGE_FAST\` L1 | \`PAGE_FAST\` L2 |"
    echo "| --- | ---: | ---: | ---: | ---: | ---: |"
    pf=""
    for spec in "-l 1" "-l 2" "-l 5" "-l 1 -P" "-l 2 -P"; do
        # The spec is a pair of bench options and is meant to split here.
        pf="$pf $("$cli" bench -q -b 4096 $spec -r "$REPS" "$mfile" |
            awk '/TOTAL/ {printf "%.2f%%|%.0f", 100.0*$3/$2, $5}')"
    done
    pfr=""
    for c in $pf; do
        pfr="$pfr $(printf '%s' "$c" | cut -d'|' -f1) at $(printf '%s' "$c" | cut -d'|' -f2) MB/s |"
    done
    printf '| mixed |%s\n' "$pfr"
else
    missing "mixed corpus (corpus_write --mixed)"
fi

# ---- colcheck ---------------------------------------------------------

if [ -f "$titanic" ] && [ -f "$uber" ]; then
    say "colcheck"
    echo "| file | bytes | text | typed columns | gain |"
    echo "| --- | ---: | ---: | ---: | --- |"
    for f in "$titanic" "$uber"; do
        [ -f "$f" ] || continue
        "$cli" colcheck "$f" 2>/dev/null |
            awk -v n="$(basename "$f")" '{
                printf "| %s | %d | %d (%.2f:1) | %d (%.2f:1) | %s |\n",
                       n, $2, $3, $4, $5, $6, $7
            }'
    done
else
    missing "titanic.csv and uber-rides-data1.csv"
fi

# ---- Scaling ----------------------------------------------------------

if "$CC" -std=c11 -O2 -DZPSE_MT -pthread -I "$inc" \
    "$repo/tests/benchmarks/bench_parallel.c" "$unity" \
    -o "$work/parallel" 2>/dev/null; then
    say "scaling across workers"
    taskset -c 0-5 "$work/parallel" 6 "$REPS" 2>/dev/null
else
    missing "bench_parallel build (ZPSE_MT)"
fi

# ---- Compiler flags ---------------------------------------------------

if "$CC" -std=c11 -O2 -I "$inc" "$repo/tests/benchmarks/bench_levels.c" \
    "$unity" -o "$work/levels_o2" 2>/dev/null &&
    "$CC" -std=c11 -O3 -I "$inc" "$repo/tests/benchmarks/bench_levels.c" \
        "$unity" -o "$work/levels_o3" 2>/dev/null; then
    say "compiler: -O2 vs -O3 on the level corpus"
    if [ "$fast" = 1 ]; then it=2; rep=1; else it=20; rep=3; fi
    for b in "$work/levels_o2" "$work/levels_o3"; do
        printf '%s: ' "$(basename "$b")"
        taskset -c 0 "$b" "$it" "$rep" 2>/dev/null | awk -F'|' '
            $3 ~ / BALANCED/ { printf "%s ", $5 }'
        echo
    done
else
    missing "bench_levels build"
fi

exit "$rc"
