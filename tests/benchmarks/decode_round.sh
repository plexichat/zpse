#!/usr/bin/env bash
# The decode round's figures, from one script.
#
# Three measurements over the same inputs:
#   1. batch decode in memory, one worker to N, so the scaling curve belongs to
#      the codec and not to the file read;
#   2. level 6 against level 6|DECODE_FAST, block by block, which is how many
#      blocks the preset moves and what that costs or saves -- the frame tables
#      after it show the same thing at frame granularity;
#   3. the CLI's streaming decode of one concatenated frame, serial against N
#      workers, which is the process figure docs/HISTORY.md quotes (I/O
#      included, which is why 1 exists).
#
# Usage: bash tests/benchmarks/decode_round.sh [FILE ...]
#   No FILE: the twelve corpus files in $SILESIA_DIR when it is set (see
#            silesia.sh); otherwise measurement 1 runs on the built-in corpus
#            and 2-3 are skipped, because a generated 64 MiB frame is not what
#            the numbers in the history were measured on.
#   REPS=3   timed runs per measurement
#   MAXT=6   highest worker count
#   CC=cc    compiler
#
# Prints markdown tables and appends them to $GITHUB_STEP_SUMMARY when set.
# Nothing is written to the repository. Exits non-zero if any step failed.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
inc="$repo/include"
unity="$repo/src/zpse_unity.c"
CC="${CC:-cc}"
REPS="${REPS:-3}"
MAXT="${MAXT:-6}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/zpse-decode.XXXXXX")" || exit 1
log="$tmp/report.md"
bench="$tmp/bench_batch_decode"
first="$tmp/bench_decode_first"
cli="$tmp/zpse_cli"
files=""
rc=0

cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT
: >"$log"

# Every command's output goes to the summary file as well as the terminal,
# with the command's own status kept (PIPESTATUS, not tee's).
run() {
    local st
    "$@" 2>&1 | tee -a "$log"
    st=${PIPESTATUS[0]}
    [ "$st" -ne 0 ] && rc=1
    return "$st"
}

sec() {
    printf '\n### %s\n\n' "$*" >>"$log"
    printf '\n### %s\n\n' "$*"
}

if [ "$#" -gt 0 ]; then
    for f in "$@"; do files="$files $f"; done
elif [ -n "${SILESIA_DIR:-}" ]; then
    for n in dickens mozilla mr nci ooffice osdb reymont samba sao webster \
             x-ray xml; do
        [ -f "$SILESIA_DIR/$n" ] && files="$files $SILESIA_DIR/$n"
    done
fi

if [ -z "$files" ]; then
    echo "decode_round: no inputs: measurement 1 runs on the built-in corpus," \
         "2-3 are skipped" >&2
    echo "decode_round: set SILESIA_DIR or pass FILE... for the rest" >&2
fi

# ZPSE_MT uses pthreads off Windows and its own threads on it, so -pthread is
# only there where the codec wants it.
case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW*|MSYS*|CYGWIN*) mt="" ;;
    *) mt="-pthread" ;;
esac

echo "decode_round: building ($CC${mt:+, $mt})" >&2
# shellcheck disable=SC2086
$CC -std=c11 -O2 -DZPSE_MT $mt -I"$inc" \
    "$repo/tests/benchmarks/bench_batch_decode.c" "$unity" -o "$bench" ||
    { echo "decode_round: batch benchmark build failed" >&2; exit 1; }
$CC -std=c11 -O2 -I"$inc" \
    "$repo/tests/benchmarks/bench_decode_first.c" "$unity" -o "$first" ||
    { echo "decode_round: decode-first benchmark build failed" >&2; exit 1; }
# shellcheck disable=SC2086
$CC -std=c11 -O2 -DZPSE_MT $mt -I"$inc" \
    "$repo/tests/tools/zpse_cli.c" "$unity" -o "$cli" ||
    { echo "decode_round: CLI build failed" >&2; exit 1; }

sec "Batch decode in memory, codec only"
# shellcheck disable=SC2086
run env ZPSE_BD_MAXT="$MAXT" ZPSE_BD_REPS="$REPS" "$bench" $files

if [ -n "$files" ]; then
    sec "Decode-first, block by block"
    # shellcheck disable=SC2086
    run "$first" $files

    sec "Frame size and rate, level 6"
    # shellcheck disable=SC2086
    run "$cli" bench -l 6 -r "$REPS" $files

    sec "Frame size and rate, level 6 -D"
    # shellcheck disable=SC2086
    run "$cli" bench -l 6 -D -r "$REPS" $files

    # One frame over the whole corpus, which is the shape the streaming
    # figure was measured on.
    # shellcheck disable=SC2086
    cat $files >"$tmp/all.bin" 2>/dev/null
    inb=$(wc -c <"$tmp/all.bin")
    if [ "$inb" -gt 0 ] && run "$cli" c -l 6 "$tmp/all.bin" "$tmp/all.z"; then
        fz=$(wc -c <"$tmp/all.z")
        sec "Streaming decode of one frame ($inb bytes in, $fz bytes frame)"
        printf '| workers | ms |\n|---:|---:|\n' | tee -a "$log"
        for t in 1 "$MAXT"; do
            if [ "$t" = 1 ] && [ "$MAXT" = 1 ]; then
                :
            elif [ "$t" = "$MAXT" ] && [ "$MAXT" = 1 ]; then
                continue
            fi
            best=""
            for ((i = 0; i < REPS; i++)); do
                s=$(date +%s%N)
                "$cli" d -t "$t" "$tmp/all.z" /dev/null >/dev/null 2>&1
                st=$?
                e=$(date +%s%N)
                if [ "$st" -ne 0 ]; then
                    echo "decode_round: decode -t $t failed" >&2
                    rc=1
                    break
                fi
                ms=$(( (e - s) / 1000000 ))
                if [ -z "$best" ] || [ "$ms" -lt "$best" ]; then
                    best=$ms
                fi
            done
            printf '| %s | %s |\n' "$t" "${best:-failed}" | tee -a "$log"
        done
    fi
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat "$log" >>"$GITHUB_STEP_SUMMARY"
fi
exit "$rc"
