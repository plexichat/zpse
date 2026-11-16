#!/usr/bin/env bash
# The 4 KiB page table without dlopen. bench_pages.c loads zstd and lz4 from
# shared libraries, which this Windows host cannot do; bench_pages_static.c
# links lz4 1.10.0 in instead and times the same per-page loops.
#
# Usage: bash tests/benchmarks/page_table_static.sh [LZ4_DIR] [SIZE] [REPS]
#
#   LZ4_DIR  directory holding lib/lz4.c, lib/lz4hc.c and lib/lz4.h. When
#            omitted, the script fetches the v1.10.0 tarball into a temp dir.
#   SIZE     bytes per class, default 16 MiB, must be a multiple of 4 KiB.
#   REPS     timing reps passed to the harness, default 5.
#
# The harness also supports --no-lz4 and --modes=KEY,KEY,... for a fast pure
# zpse A/B while iterating; this script prints the full tables. The corpus
# sha256 is pinned below, so a generator change fails the run instead of
# silently producing README numbers measured on different data.
#
# Prints markdown. Nothing is written into the repository.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
CC="${CC:-clang}"
LZ4_DIR="${1:-}"
SIZE="${2:-16777216}"
REPS="${3:-5}"
work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pages-static.XXXXXX")" || exit 1

cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "page_table_static: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

if [ -z "$LZ4_DIR" ]; then
    echo "page_table_static: fetching lz4 1.10.0" >&2
    curl -sSL -o "$work/lz4.tar.gz" \
        https://github.com/lz4/lz4/archive/refs/tags/v1.10.0.tar.gz || exit 1
    tar xzf "$work/lz4.tar.gz" -C "$work" || exit 1
    LZ4_DIR="$work/lz4-1.10.0"
fi
lz4lib="$LZ4_DIR/lib"

echo "page_table_static: building corpus_write and bench_pages_static" >&2
"$CC" -std=c11 -O2 -I "$inc" -I "$repo/tests/corpus" \
    "$repo/tests/corpus/corpus_write.c" -o "$work/corpus_write" || exit 1
"$CC" -std=c11 -O2 -I "$inc" -I "$lz4lib" \
    "$repo/tests/benchmarks/bench_pages_static.c" \
    "$lz4lib/lz4.c" "$lz4lib/lz4hc.c" "$unity" -o "$work/bench_pages_static" || exit 1

"$work/corpus_write" "$work/pages" "$SIZE" --pages || exit 1
digest=$(sha256sum "$work/pages/pages-nonzero.bin" | cut -d' ' -f1)
echo "page_table_static: corpus sha256 $digest" >&2

# The README rows are only comparable against the corpus they were measured
# on. Pin it so a corpus generator change cannot silently invalidate them.
expected=C9C0E86052FD430A9E59A589FC6F272B4E93A12001F79EA6D77F82D026BECB1B
digest_upper=$(printf '%s' "$digest" | tr 'a-f' 'A-F')
if [ "$digest_upper" != "$expected" ]; then
    echo "page_table_static: corpus sha256 mismatch: expected $expected, got $digest" >&2
    exit 1
fi

echo "## table: page corpus, non-zero pages"
echo
"$work/bench_pages_static" --exclude-zero --reps="$REPS" "$work/pages/pages-nonzero.bin"
echo "Corpus: \`pages-nonzero.bin\`, $SIZE bytes, sha256 \`${digest:0:16}…\`."
echo "Every page is one independent block; lz4 is called once per 4 KiB page."
echo "Packed bytes include each codec's framing (two length bytes are charged"
echo "to a raw lz4 block). Ratios are exact; rates are best of $REPS."
echo

echo "## table: page corpus, all pages including zero"
echo
"$work/bench_pages_static" --reps="$REPS" "$work/pages/pages-all.bin"
echo

echo "## table: page corpus, per class"
echo
echo "Decode MB/s (best of 3) per corpus class:"
echo
echo "| class | zpse 6 | zpse FLAT | zpse 6 INSTANT | lz4 1 | lz4hc 12 |"
echo "| --- | ---: | ---: | ---: | ---: | ---: |"
for f in "$work"/pages/page-*.bin; do
    name=$(basename "$f" .bin); name=${name#page-}
    out=$("$work/bench_pages_static" --reps=3 "$f")
    zp=$(printf '%s\n' "$out" | awk -F'|' '/zpse 6 \(default\)/ {gsub(/ /,"",$5); print $5}')
    zf=$(printf '%s\n' "$out" | awk -F'|' '/zpse FLAT \(default\)/ {gsub(/ /,"",$5); print $5}')
    zi=$(printf '%s\n' "$out" | awk -F'|' '/zpse 6 PAGE_INSTANT/ {gsub(/ /,"",$5); print $5}')
    l1=$(printf '%s\n' "$out" | awk -F'|' '/lz4 1 / {gsub(/ /,"",$5); print $5}')
    lh=$(printf '%s\n' "$out" | awk -F'|' '/lz4hc 12 / {gsub(/ /,"",$5); print $5}')
    printf '| `%s` | %s | %s | %s | %s | %s |\n' "$name" "$zp" "$zf" "$zi" "$l1" "$lh"
done
