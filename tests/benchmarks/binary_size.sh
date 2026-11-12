#!/usr/bin/env bash
# Compiled size of the library, as the README's "Binary size" table reports it.
#
# Usage: bash tests/benchmarks/binary_size.sh [CC]
#
# Builds src/zpse_unity.c one translation unit at a time and prints the
# object's text/data/bss with `size`. The unity build is what the table is
# about: code that links against it gets the whole object unless the caller
# also builds with -ffunction-sections -fdata-sections -Wl,--gc-sections.
# Prints markdown. Nothing is written into the repository.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
CC="${1:-${CC:-cc}}"
work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-size.XXXXXX")" || exit 1

cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "binary_size: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

command -v "$CC" >/dev/null 2>&1 || { echo "binary_size: no $CC" >&2; exit 1; }
command -v size >/dev/null 2>&1 || { echo "binary_size: no size" >&2; exit 1; }

echo "binary_size: $("$CC" --version | head -n 1)" >&2
echo "| build | text | data | bss | object file |"
echo "| --- | ---: | ---: | ---: | ---: |"

for opt in -O2 -Os; do
    obj="$work/unity$(echo "$opt" | tr -d '-').o"
    "$CC" -std=c11 "$opt" -I "$inc" -c "$unity" -o "$obj" ||
        { echo "binary_size: $opt build failed" >&2; exit 1; }
    set -- $(size "$obj" | awk 'NR == 2 { print $1, $2, $3 }')
    printf '| `%s` | %s B | %s B | %s B | %s B |\n' \
        "$opt" "$1" "$2" "$3" "$(wc -c < "$obj" | tr -d ' ')"
done
