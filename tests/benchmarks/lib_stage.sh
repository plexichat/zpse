#!/usr/bin/env bash
# Corpus staging shared by the benchmark harnesses.
#
# A harness that times a whole process pays for that process's read of its
# input, and the corpus lives on a disk: on this host a cold 202 MiB read is
# several tenths of a second, which is the size of a whole Silesia encode
# pass. A run that starts cold therefore puts the drive in the rate column.
#
# Two things close that, and this file is both. The bytes are copied into a
# memory-backed directory when the host has one, and every staged copy is
# read back before the clock starts -- once to fault it in and once to prove
# it stayed resident. The second read is reported as the READ FLOOR: it is
# what a whole-process row still pays, so a table can be read against it
# instead of hoping it was free.
#
# The in-process harnesses (bench.c, bench_levels.c, bench_parallel.c, the
# CLI's own `bench`) read their input into memory once and then time the
# codec on that copy, so for them staging is not needed; the scripts that
# run a tool as a process are the ones that stage.
#
# Usage:
#   . "$(dirname "$0")/lib_stage.sh"
#   stage_init
#   p=$(stage_file "$corpus/file")          # copy into the staging root
#   echo "read floor $(stage_floor "$p") MB/s"
#   stage_done                              # on exit, via the caller's trap
#
# ZPSE_BENCH_RAM=/some/dir overrides the staging root. KEEP_STAGE=1 keeps the
# staged copies and prints where they are.

STAGE_ROOT=""
STAGE_DIR=""

# Pick a staging root: the caller's override, then /dev/shm, then TMPDIR. A
# candidate is tried by CREATING the directory in it rather than by testing
# -w: this host reports /dev/shm writable to the shell and then refuses the
# mkdir, and a staging root that cannot hold a file is not a staging root.
# The last candidate is always the temp directory, so a harness still runs
# when the host has no memory-backed one; the read floor it prints is then
# the page cache's, which is what the numbers are read against anyway.
stage_init() {
    local cand d

    if [ -n "${STAGE_DIR:-}" ] && [ -d "${STAGE_DIR:-}" ]; then
        return 0
    fi
    for cand in "${ZPSE_BENCH_RAM:-}" /dev/shm "${TMPDIR:-/tmp}"; do
        [ -n "$cand" ] || continue
        [ -d "$cand" ] || continue
        d=$(mktemp -d "$cand/zpse-stage.XXXXXX" 2>/dev/null) || continue
        STAGE_ROOT="$cand"
        STAGE_DIR="$d"
        return 0
    done
    echo "lib_stage: no writable staging directory" >&2
    return 1
}

# Copy one file into the staging root and print the copy's path. The copy is
# read through once here, so the caller's first timed run does not pay the
# cold pass.
stage_file() {
    local src="$1" name dst

    [ -f "$src" ] || return 1
    [ -n "${STAGE_DIR:-}" ] || stage_init || return 1
    name=$(basename "$src")
    dst="$STAGE_DIR/$name"
    case "$src" in
    "$STAGE_DIR"/*) printf '%s\n' "$src"; return 0 ;;
    esac
    if [ ! -f "$dst" ]; then
        cp -f "$src" "$dst" || return 1
    fi
    dd if="$dst" of=/dev/null bs=1M 2>/dev/null
    printf '%s\n' "$dst"
    return 0
}

# One sequential read of FILE, in decimal MB/s of file bytes. This is the
# read floor: every whole-process row in a cross-tool table still pays it.
stage_floor() {
    local f="$1" t0 t1 bytes ns

    bytes=$(wc -c <"$f") || return 1
    t0=$(date +%s%N)
    dd if="$f" of=/dev/null bs=1M 2>/dev/null || return 1
    t1=$(date +%s%N)
    ns=$((t1 - t0))
    awk -v b="$bytes" -v n="$ns" \
        'BEGIN { printf "%.0f", (n > 0) ? b * 1000.0 / n : 0 }'
}

# Is the staged copy actually resident? Two reads; a second one far below
# the first means the staging root is not memory and the numbers are about
# to include the drive. Reported, not repaired: the harness says so.
stage_resident() {
    local f="$1" a b

    a=$(stage_floor "$f") || return 1
    b=$(stage_floor "$f") || return 1
    if [ -z "$a" ] || [ -z "$b" ]; then
        return 1
    fi
    awk -v a="$a" -v b="$b" 'BEGIN { exit !(b + 0 < a + 0) }' && {
        echo "lib_stage: warning: $f reads at $a then $b MB/s -- not resident" >&2
        return 1
    }
    return 0
}

stage_done() {
    if [ -n "${STAGE_DIR:-}" ] && [ -d "${STAGE_DIR:-}" ]; then
        if [ "${KEEP_STAGE:-0}" = "1" ]; then
            echo "lib_stage: kept $STAGE_DIR" >&2
        else
            rm -rf "$STAGE_DIR"
        fi
    fi
    STAGE_DIR=""
}
