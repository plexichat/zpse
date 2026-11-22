#!/usr/bin/env bash
# Reproducible clang PGO A/B on the twelve Silesia files.
#
# The codec's only measured >5% encode win on this host is codegen: a PGO
# build trained on the Silesia decision traffic. Both arms here are the same
# source, the same clang, the same -O2 and the same unity TU; the only
# difference is -fprofile-generate/-fprofile-use. Output is byte-identical by
# construction -- if the packed-byte column ever differs, the build is wrong.
#
# Usage: bash tests/benchmarks/pgo_silesia.sh [rounds]
#   SILESIA_DIR  staged corpus (all twelve named files)
#   CC           clang with llvm-profdata on PATH (default: clang)
#   ROUNDS       alternating rounds, default 4
# Prints base/pgo medians for encode and decode MB/s and the A/A-style spread.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-pgo.XXXXXX")"
trap 'rm -rf "$work"' EXIT

CC="${CC:-clang}"
PROFDATA="$(dirname "$(command -v "$CC")")/llvm-profdata"
[ -x "$PROFDATA" ] || PROFDATA=llvm-profdata
command -v "$PROFDATA" >/dev/null 2>&1 || {
    echo "pgo_silesia: llvm-profdata not found" >&2; exit 1;
}
rounds="${1:-4}"

names="dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml"
corpus="${SILESIA_DIR:-}"
[ -n "$corpus" ] || { echo "pgo_silesia: set SILESIA_DIR" >&2; exit 1; }
files=""
for n in $names; do
    [ -f "$corpus/$n" ] || { echo "pgo_silesia: missing $n" >&2; exit 1; }
    files="$files $corpus/$n"
done

inc="$repo/include"
cli="$repo/tests/tools/zpse_cli.c"
unity="$repo/src/zpse_unity.c"
mkdir -p "$work/prof"

# A: plain -O2, the reference every README row is currently built from.
"$CC" -std=c11 -O2 -I"$inc" "$cli" "$unity" -o "$work/base" || exit 1

# B stage 1: instrument.
"$CC" -std=c11 -O2 -I"$inc" -fprofile-generate "$cli" "$unity" \
    -o "$work/pg" || exit 1
LLVM_PROFILE_FILE="$work/prof/pgo-%p.profraw" \
    "$work/pg" bench -q -b 65536 -l 6 $files >/dev/null || exit 1
"$PROFDATA" merge -sparse "$work"/prof/*.profraw \
    -o "$work/pgo.profdata" || exit 1

# B stage 2: consume. Keep the suppressions validated by pgo_build.sh.
"$CC" -std=c11 -O2 -I"$inc" -fprofile-use="$work/pgo.profdata" \
    -Wno-ignored-optimization-argument -Wno-profile-instr-out-of-date \
    -Wno-profile-instr-unprofiled -Wno-backend-plugin \
    "$cli" "$unity" -o "$work/pgo" || exit 1

measure() {
    "$1" bench -q -b 65536 -l 6 -r 3 $files 2>/dev/null |
        awk '/^TOTAL/{print $3, $5, $6}'
}

echo "round,arm,packed,enc_mbs,dec_mbs"
for ((r = 1; r <= rounds; r++)); do
    if (( r % 2 )); then order="base pgo pgo base"; else order="pgo base base pgo"; fi
    for arm in $order; do
        read -r packed enc dec <<<"$(measure "$work/$arm")"
        printf '%d,%s,%s,%s,%s\n' "$r" "$arm" "$packed" "$enc" "$dec"
    done
done

echo ""
echo "Packed bytes must be identical in every row above; if not, discard."
echo "Compare the two arms' median enc/dec and require the delta to exceed the"
echo "session's A/A floor (see docs/HISTORY.md, 'Profiles belong to the build')."
