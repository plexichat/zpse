#!/usr/bin/env bash
# The 4 KiB page corpus: ratio and rates for zpse, zstd and lz4hc, one page
# at a time.
#
# Usage: bash tests/benchmarks/page_table.sh [SIZE] [REPS]
#
#   SIZE   bytes per class file, default 16 MiB (4096 pages), must be a
#          multiple of 4 KiB
#   REPS   passed through to the harness (it uses three timing loops)
#
# The headline table is `pages-nonzero.bin`: one page in eight of a real
# footprint is untouched anonymous memory, and those pages compress to almost
# nothing, so counting them would flatter every codec. The all-pages table is
# printed next to it, with the zero share named, so the two answers are both
# visible instead of one hiding the other.
#
# Everything is generated from tests/corpus/corpus_write.c, so the table can be
# re-measured from a checkout on any host. zstd and lz4hc come from the system
# shared libraries (bench_pages dlopens them); no -dev package is required.
# Prints markdown. Nothing is written into the repository.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
CC="${CC:-cc}"
SIZE="${1:-16777216}"
REPS="${2:-3}"
work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pages.XXXXXX")" || exit 1

cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "page_table: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

mkdir -p "$work/pages" || exit 1

echo "page_table: building corpus_write, bench_pages and bench_pagedict" >&2
"$CC" -std=c11 -O2 -I "$inc" -I "$repo/tests/corpus" \
    "$repo/tests/corpus/corpus_write.c" -o "$work/corpus_write" ||
    { echo "page_table: corpus_write build failed" >&2; exit 1; }
"$CC" -std=c11 -O2 -I "$inc" "$repo/tests/benchmarks/bench_pages.c" \
    "$unity" -ldl -o "$work/bench_pages" ||
    { echo "page_table: bench_pages build failed" >&2; exit 1; }
"$CC" -std=c11 -O2 -I "$inc" "$repo/tests/benchmarks/bench_pagedict.c" \
    "$unity" -o "$work/bench_pagedict" ||
    { echo "page_table: bench_pagedict build failed" >&2; exit 1; }

"$work/corpus_write" "$work/pages" "$SIZE" --pages ||
    { echo "page_table: corpus generation failed" >&2; exit 1; }

digest=$(sha256sum "$work/pages/pages-nonzero.bin" | cut -d' ' -f1)
echo "page_table: corpus sha256 $digest" >&2

pin=""
command -v taskset >/dev/null 2>&1 && pin="taskset -c 0"

echo "## table: page corpus, non-zero pages"
echo
$pin "$work/bench_pages" "$work/pages/pages-nonzero.bin"
echo "Corpus: \`pages-nonzero.bin\`, $SIZE bytes, sha256 \`${digest:0:16}…\`."
echo "Every page is one independent block; zstd and lz4 are called once per"
echo "4 KiB page, so no frame spans two pages. Packed bytes include each"
echo "codec's framing (two length bytes are charged to a raw lz4 block)."
echo

echo "## table: page corpus, all pages including zero"
echo
$pin "$work/bench_pages" "$work/pages/pages-all.bin"
echo
$pin "$work/bench_pages" --exclude-zero "$work/pages/pages-all.bin" |
    sed -n '/^| codec/,$p'
echo
echo "The first table counts the zero pages; the second drops them from the"
echo "same file. A real zram device sees both, but only the non-zero number"
echo "describes the content it is actually compressing."
echo

echo "## table: page corpus, per class"
echo
echo "| class | pages | zpse 6 | zstd 6 | lz4hc 12 |"
echo "| --- | ---: | ---: | ---: | ---: |"
for f in "$work"/pages/page-*.bin; do
    name=$(basename "$f" .bin)
    name=${name#page-}
    out=$($pin "$work/bench_pages" "$f")
    zp=$(printf '%s\n' "$out" | awk -F'|' '/\| zpse 6 \(default\)/ {gsub(/ /,"",$3); print $3}')
    zs=$(printf '%s\n' "$out" | awk -F'|' '/\| zstd 6 / {gsub(/ /,"",$3); print $3}')
    lz=$(printf '%s\n' "$out" | awk -F'|' '/\| lz4hc 12 / {gsub(/ /,"",$3); print $3}')
    pages=$(printf '%s\n' "$out" | awk '/^[0-9]+ pages/ {print $1}')
    printf '| `%s` | %s | %s | %s | %s |\n' "$name" "$pages" "$zp" "$zs" "$lz"
done
echo
echo "\`zero\` is listed so the share is visible; it is not part of the"
echo "headline corpus. Classes: code, rodata, heap-pointers, heap-objects,"
echo "slab-strings, btree, hash-table, json-arena, stack, float-series, blob,"
echo "mixed-heap. See \`tests/corpus/corpus_write.c\` for the shape of each."
echo

echo "## table: page dictionary"
echo
echo "\`pages-repeat.bin\` is the same generator with one page in five a byte"
echo "copy of an earlier page, scattered rather than adjacent. Each arm hashes"
echo "the page first, serves a stored block on a hit and retags it, and"
echo "compresses on a miss; packed bytes should be identical across arms."
echo
$pin "$work/bench_pagedict" "$work/pages/pages-repeat.bin" 3
echo
echo "A stream with no repeats pays the hash and serves nothing;"
echo "\`bench_pagedict\` on \`pages-nonzero.bin\` measures that overhead."
