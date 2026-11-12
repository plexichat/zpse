#!/usr/bin/env bash
# Silesia: the codec on real-world data instead of a generator.
#
# Downloads the corpus from mattmahoney.net/dc/silesia.zip into a temporary
# directory, builds the CLI, and benches every file on one pinned core. The
# corpus is 202 MiB of English prose, source, executables, databases and
# medical images -- half of it text, which is the shape this codec is weakest
# on and the synthetic corpus does not contain at all.
#
# Usage: bash tests/benchmarks/silesia.sh [level] [reps]
#   level        1-9, default the CLI's own default (BALANCED)
#   reps         timed runs per file, default 2 (the CLI keeps the best)
#   SILESIA_DIR  bench an already-extracted corpus instead of downloading
#   KEEP=1       keep the download directory and print where it is
#   CC=cc        compiler for the CLI
#
# Prints a markdown table and appends it to $GITHUB_STEP_SUMMARY when set.
# Nothing is written to the repository. Exit 0 on success.
set -uo pipefail

# Two levels up: this script lives in tests/benchmarks/.
repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
cli_src="$repo/tests/tools/zpse_cli.c"
CC="${CC:-cc}"
level="${1:-}"
reps="${2:-2}"
work="${SILESIA_DIR:-}"
tmp=""
bin=""

cleanup() {
    [ -n "$bin" ] && rm -f "$bin"
    if [ -n "$tmp" ] && [ "${KEEP:-0}" != "1" ]; then
        rm -rf "$tmp"
    fi
}
trap cleanup EXIT

if [ -z "$work" ]; then
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/zpse-silesia.XXXXXX")" ||
        { echo "silesia: cannot create a temp directory" >&2; exit 1; }
    work="$tmp"
    url="http://mattmahoney.net/dc/silesia.zip"
    zip="$work/silesia.zip"

    echo "silesia: fetching $url" >&2
    if command -v curl >/dev/null 2>&1; then
        curl -sSLo "$zip" "$url" || { echo "silesia: download failed" >&2; exit 1; }
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$zip" "$url" || { echo "silesia: download failed" >&2; exit 1; }
    else
        echo "silesia: needs curl or wget" >&2
        exit 1
    fi
    # unzip is not in a minimal image (and not in WSL by default); the
    # archive is a plain deflate zip, so Python's zipfile is enough.
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$zip" -d "$work" || { echo "silesia: unzip failed" >&2; exit 1; }
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
            "$zip" "$work" || { echo "silesia: extraction failed" >&2; exit 1; }
    else
        echo "silesia: needs unzip or python3" >&2
        exit 1
    fi
    rm -f "$zip"
elif [ ! -d "$work" ]; then
    echo "silesia: SILESIA_DIR=$work is not a directory" >&2
    exit 1
fi

# The twelve corpus files by name, not "everything in the directory": a
# reused SILESIA_DIR can hold a stray gmon.out, a second .zip or a partially
# extracted file, and a benchmark whose denominator depends on what else is
# lying around is worse than no benchmark. This exact list is also what the
# README's table covers.
names="dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml"
files=""
for n in $names; do
    [ -f "$work/$n" ] && files="$files $work/$n"
done
# shellcheck disable=SC2086
set -- $files
if [ "$#" -eq 0 ]; then
    echo "silesia: none of the twelve corpus files are in $work" >&2
    exit 1
fi
nfiles="$#"
bytes=$(printf '%s\n' "$files" | xargs -d '\n' du -cb 2>/dev/null | tail -1 | cut -f1)

bin="$(mktemp "${TMPDIR:-/tmp}/zpse-cli.XXXXXX")" ||
    { echo "silesia: cannot create a binary path" >&2; exit 1; }
rm -f "$bin"
echo "silesia: building the CLI ($CC)" >&2
"$CC" -std=c11 -O2 -I"$inc" "$cli_src" "$unity" -o "$bin" ||
    { echo "silesia: CLI build failed" >&2; exit 1; }

# Pin to one core when taskset is there: these are single-core numbers, and
# a migrating process is a measurement of the scheduler.
pin=""
command -v taskset >/dev/null 2>&1 && pin="taskset -c 0"

lvl_flag=""
[ -n "$level" ] && lvl_flag="-l $level"

# Read the corpus into RAM before anything is timed. The 202 MiB lives on
# disk, and on a SATA SSD a cold read is several tenths of a second -- the
# size of a whole encode pass -- so a run that started cold would be timing
# the drive. Every file is read twice: the first pass faults it in, the
# second proves it is resident. `zpse_cli bench` reads each file once and
# then re-runs the codec on its own copy, so this is the only disk in the
# run.
for f in $files; do
    cat "$f" > /dev/null
    cat "$f" > /dev/null
done

echo "silesia: $nfiles files, $bytes bytes, level ${level:-default}, ${reps} rep(s)" >&2
out=$($pin "$bin" bench $lvl_flag -r "$reps" $files 2>&1)
rc=$?
printf '%s\n' "$out"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
        echo ""
        echo "### Silesia ($nfiles files, $bytes bytes, level ${level:-default})"
        echo ""
        printf '%s\n' "$out"
    } >>"$GITHUB_STEP_SUMMARY"
fi
exit "$rc"
