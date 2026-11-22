#!/usr/bin/env bash
# A/B for one 4 KiB page change: two builds, alternating rounds, and a byte
# determinism gate that must pass before any rate is believed.
#
# Why this and not two page_table_static.sh runs. A page optimisation is a
# claim about encode and decode MB/s on the same 16 MiB corpus, and a single
# before/after pair on a shared host carries whatever the machine did between
# the two runs. Here each round runs both builds back to back with the order
# swapped (a,b,b,a,...), the estimator per build is best-of-5 inside a round,
# and the table reports best and median across rounds so one lucky round
# cannot stand in for the run. A sign is only called a win when it holds in
# >=75% of the rounds; everything else is labelled noise.
#
# The gate is the other half of the method. A speed change on an existing page
# mode is only interesting if the bytes it writes are unchanged; if they are
# not, the rate difference is a different codec, not a faster one. The packed
# bytes are not in the harness's stdout (it prints ratios and rates), so a
# small dumper is compiled from each tree and its stdout hashed for every zpse
# row the harness emits. Every row must match byte for byte or the script
# stops before spending time on timing. FLAT and the LZ_FAST row use only live
# flag bits, so they are gated like the default-level rows.
#
# Usage: bash tests/benchmarks/page_ab.sh [options] [A] [B]
#   A          first build, a git ref (default HEAD). Built from git archive.
#   B          second build, a git ref or WORKTREE (default WORKTREE). WORKTREE
#              (also "-", "." or "working-tree") builds the checkout as it
#              stands, including uncommitted changes.
#   -r ROUNDS  alternating rounds, minimum 8 (default 8)
#   -n BESTN   best-of-N timing reps per harness call (default 5)
#   -p REPS    reps for the per-class table (default 1; medians come from the
#              rounds, so a class does not need a five-way min inside each)
#   -c CPU     taskset CPU for timed runs (default empty: no pin)
#
# Environment: CC (default clang), LZ4_DIR (directory with lib/lz4.c,
# lib/lz4hc.c, lib/lz4.h; the v1.10.0 tarball is fetched when unset),
# WITH_LZ4=1 to keep the reference lz4 rows in the tables (they are dropped by
# default, since they are identical in both builds), KEEP=1 to keep the temp
# dir, ROUNDS/BESTN/CLASS_REPS/CPU as the option defaults.
#
# Both builds share one corpus, written once from tests/corpus/corpus_write.c
# and pinned to a sha256 so a generator change cannot change the denominator
# underneath the comparison. Nothing is written into the repository.
set -uo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"

CC="${CC:-clang}"
ROUNDS="${ROUNDS:-8}"
BESTN="${BESTN:-5}"
CLASS_REPS="${CLASS_REPS:-1}"
CPU="${CPU:-}"
SIZE=16777216
EXPECT=C9C0E86052FD430A9E59A589FC6F272B4E93A12001F79EA6D77F82D026BECB1B

usage() { echo "usage: page_ab.sh [-r rounds] [-n best] [-p reps] [-c cpu] A B" >&2; }
die() { echo "page_ab: $*" >&2; exit 1; }

while getopts "r:n:p:c:" opt; do
    case "$opt" in
    r) ROUNDS=$OPTARG ;;
    n) BESTN=$OPTARG ;;
    p) CLASS_REPS=$OPTARG ;;
    c) CPU=$OPTARG ;;
    *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
A="${1:-HEAD}"
B="${2:-WORKTREE}"

for v in ROUNDS BESTN CLASS_REPS; do
    eval val=\$$v
    case "$val" in
    ''|*[!0-9]*) die "$v must be a positive integer" ;;
    esac
done
[ "$ROUNDS" -ge 8 ] || { echo "page_ab: ROUNDS < 8; using 8" >&2; ROUNDS=8; }
[ "$BESTN" -ge 1 ] || die "BESTN must be at least 1"
[ "$CLASS_REPS" -ge 1 ] || die "CLASS_REPS must be at least 1"

work="$(mktemp -d "${TMPDIR:-/tmp}/zpse-page-ab.XXXXXX")" || die "no temp dir"
cleanup() {
    if [ "${KEEP:-0}" = "1" ]; then
        echo "page_ab: kept $work" >&2
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT
mkdir -p "$work/raw" "$work/pages" "$work/a" "$work/b" || exit 1

# ---- tools ----------------------------------------------------------------
if [ -z "${LZ4_DIR:-}" ]; then
    echo "page_ab: fetching lz4 1.10.0" >&2
    curl -sSL -o "$work/lz4.tar.gz" \
        https://github.com/lz4/lz4/archive/refs/tags/v1.10.0.tar.gz ||
        die "lz4 download failed; set LZ4_DIR"
    tar xzf "$work/lz4.tar.gz" -C "$work" || die "lz4 unpack failed"
    LZ4_DIR="$work/lz4-1.10.0"
fi
lz4lib="$LZ4_DIR/lib"
for f in lz4.c lz4hc.c lz4.h; do
    [ -f "$lz4lib/$f" ] || die "LZ4_DIR/lib/$f is missing"
done

# A ref comes from the committed tree; WORKTREE is tarred out of the checkout
# so uncommitted encoder changes are the ones under test.
materialize() {  # ref dest
    local ref="$1" dest="$2"
    mkdir -p "$dest" || return 1
    case "$ref" in
    WORKTREE|worktree|working-tree|@worktree|-|.)
        tar -C "$repo" -cf - include src tests | tar -C "$dest" -xf - ;;
    *)
        git -C "$repo" archive "$ref" | tar -C "$dest" -xf - ;;
    esac
}
materialize "$A" "$work/a" || die "cannot materialise $A"
materialize "$B" "$work/b" || die "cannot materialise $B"

# bench_pages_static.c is untracked, so git archive does not carry it. The
# same harness source goes into both trees; only include/ and src/ differ.
for nm in a b; do
    mkdir -p "$work/$nm/tests/benchmarks" || exit 1
    cp "$repo/tests/benchmarks/bench_pages_static.c" \
        "$work/$nm/tests/benchmarks/" || die "cannot copy harness into $nm"
done

build_harness() {  # tree out
    "$CC" -std=c11 -O2 -I"$1/include" -I"$lz4lib" \
        "$1/tests/benchmarks/bench_pages_static.c" \
        "$lz4lib/lz4.c" "$lz4lib/lz4hc.c" "$1/src/zpse_unity.c" \
        -o "$2"
}
for nm in a b; do
    echo "page_ab: building harness $nm ($( [ "$nm" = a ] && printf '%s' "$A" || printf '%s' "$B" ))" >&2
    build_harness "$work/$nm" "$work/harness_$nm" || die "harness build failed for $nm"
done

# ---- corpus, written once and pinned --------------------------------------
"$CC" -std=c11 -O2 -I"$repo/include" -I"$repo/tests/corpus" \
    "$repo/tests/corpus/corpus_write.c" -o "$work/corpus_write" ||
    die "corpus_write build failed"
"$work/corpus_write" "$work/pages" "$SIZE" --pages || die "corpus generation failed"
headfile="$work/pages/pages-nonzero.bin"
digest=$(sha256sum "$headfile" | awk '{print $1}')
digest_upper=$(printf '%s' "$digest" | tr 'a-f' 'A-F')
if [ "$digest_upper" != "$EXPECT" ]; then
    die "corpus sha256 mismatch: expected $EXPECT, got $digest"
fi
echo "page_ab: corpus sha256 $digest" >&2

# ---- byte dumper ----------------------------------------------------------
# Mirrors the first pass of bench_pages_static.c: one block per 4 KiB page,
# seq 0, is_last 1, and the row's flags from its printed name. Its stdout is
# the exact packed byte stream the harness hashes into its ratio.
cat > "$work/dump_pages.c" <<'CEOF'
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"
#define PAGE 4096u
#define CAP  (PAGE * 2u)
static unsigned char arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));

int main(int argc, char **argv)
{
    const char *label, *path;
    unsigned level = 6;
    uint32_t flags;
    FILE *f;
    unsigned char *in, *comp;
    long sz;
    size_t npages, p;

    if (argc != 3) return 2;
    label = argv[1]; path = argv[2];
    if (strstr(label, "FLAT")) {
        level = 6;  /* FLAT row uses the default level flags */
    } else if (sscanf(label, "zpse %u", &level) != 1) return 3;
    flags = (uint32_t)level << 1;
    if (strstr(label, "LZ_FAST"))   flags |= ZPSE_ENC_LZ_FAST;

    f = fopen(path, "rb");
    if (f == NULL) return 1;
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % (long)PAGE != 0) { fclose(f); return 1; }
    in = (unsigned char *)malloc((size_t)sz);
    comp = (unsigned char *)malloc(CAP);
    if (in == NULL || comp == NULL ||
        fread(in, 1, (size_t)sz, f) != (size_t)sz) {
        free(in); free(comp); fclose(f); return 1;
    }
    fclose(f);
    npages = (size_t)sz / PAGE;
    for (p = 0; p < npages; p++) {
        uint32_t cl = 0;
        if (zpse_compress_block_ex(in + p * PAGE, PAGE, comp, CAP, &cl,
                                   0, 1, flags, arena) != 0 || cl == 0) {
            fprintf(stderr, "dump: encode failed on page %zu (%s)\n", p, label);
            free(in); free(comp); return 1;
        }
        if (fwrite(comp, 1, cl, stdout) != cl) {
            free(in); free(comp); return 1;
        }
    }
    free(in); free(comp);
    return 0;
}
CEOF
for nm in a b; do
    tree="$work/$nm"
    "$CC" -std=c11 -O2 -I"$tree/include" \
        "$work/dump_pages.c" "$tree/src/zpse_unity.c" -o "$work/dump_$nm" ||
        die "byte dumper build failed for $nm"
done

extra=""
[ "${WITH_LZ4:-0}" = "1" ] || extra="--no-lz4"

# Every row uses live flags, so a FAILED row in either tree is a real fault
# and stops the run.
check_failed() {  # file
    local lbl bad=""
    while IFS= read -r lbl; do
        bad="$bad $lbl"
    done < <(awk -F'|' '/FAILED/ { n=$2; gsub(/^ +| +$/,"",n); if (n!="") print n }' "$1")
    [ -z "$bad" ]
}

# Discover the rows from the harnesses themselves, so a newly added zpse mode
# is covered by the gate without editing this script.
: > "$work/modes.txt"
for nm in a b; do
    "$work/harness_$nm" $extra --reps=1 "$headfile" > "$work/baseline.$nm.txt" 2>&1 ||
        die "baseline harness $nm failed: $(head -1 "$work/baseline.$nm.txt")"
    check_failed "$work/baseline.$nm.txt" ||
        die "baseline harness $nm failed on a non-opt-in row: $(grep FAILED "$work/baseline.$nm.txt" | head -1)"
    awk -F'|' '/^\| zpse/ { n=$2; gsub(/^ +| +$/,"",n); if (n!="") print n }' \
        "$work/baseline.$nm.txt" >> "$work/modes.txt"
done
sort -u "$work/modes.txt" -o "$work/modes.txt"
[ -s "$work/modes.txt" ] || die "no zpse rows in harness output"

# ---- gate -----------------------------------------------------------------
gate_rc=0
: > "$work/gate_rows.txt"
while IFS= read -r label; do
    [ -n "$label" ] || continue
    if "$work/dump_a" "$label" "$headfile" > "$work/packed.a.bin" 2> "$work/dump.a.err"; then
        ba=$(wc -c < "$work/packed.a.bin" | tr -d ' ')
        sa=$(sha256sum "$work/packed.a.bin" | awk '{print $1}')
    else
        ba="n/a"; sa="n/a"
    fi
    if "$work/dump_b" "$label" "$headfile" > "$work/packed.b.bin" 2> "$work/dump.b.err"; then
        bb=$(wc -c < "$work/packed.b.bin" | tr -d ' ')
        sb=$(sha256sum "$work/packed.b.bin" | awk '{print $1}')
    else
        bb="n/a"; sb="n/a"
    fi
    if [ "$sa" = "n/a" ] || [ "$sb" = "n/a" ]; then
        verdict="MISSING"; gate_rc=1
    elif [ "$sa" != "$sb" ]; then
        verdict="DIFFER"; gate_rc=1
    else
        verdict="same"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$ba" "$bb" "$sa" "$sb" "$verdict" >> "$work/gate_rows.txt"
done < "$work/modes.txt"

echo
echo "## byte-determinism gate"
echo
echo "Packed bytes of \`pages-nonzero.bin\` under each zpse row the harness"
echo "emits. New opt-in page modes may change their bytes; every other row must"
echo "be identical in both builds."
echo
printf '| zpse row | A bytes | B bytes | A sha256 | B sha256 | verdict |\n'
printf '| --- | ---: | ---: | --- | --- | --- |\n'
while IFS=$'\t' read -r label ba bb sa sb verdict; do
    printf '| %s | %s | %s | %s… | %s… | %s |\n' \
        "$label" "$ba" "$bb" "${sa:0:16}" "${sb:0:16}" "$verdict"
done < "$work/gate_rows.txt"
if [ "$gate_rc" -ne 0 ]; then
    echo "page_ab: byte-determinism gate failed; no rates were measured" >&2
    exit 1
fi

# ---- timing ---------------------------------------------------------------
pin=()
if [ -n "$CPU" ] && command -v taskset >/dev/null 2>&1; then
    pin=(taskset -c "$CPU")
fi

: > "$work/data.tsv"
timed_run() {  # bin name phase key round reps file
    local bin="$1" nm="$2" phase="$3" key="$4" round="$5" reps="$6" file="$7"
    local out="$work/raw/$phase.$key.$nm.$round.txt"
    if ! "${pin[@]}" "$bin" $extra --reps="$reps" "$file" > "$out" 2>&1; then
        echo "page_ab: run failed ($nm $key round $round): $(head -1 "$out")" >&2
        return 1
    fi
    if ! check_failed "$out"; then
        echo "page_ab: non-opt-in FAILED row in $out" >&2
        return 1
    fi
    awk -v ph="$phase" -v key="$key" -v nm="$nm" -v rd="$round" -F'|' '
        /^\| zpse/ {
            c=$2; gsub(/^ +| +$/,"",c)
            e=$4; d=$5; gsub(/ /,"",e); gsub(/ /,"",d)
            if (c!="" && e+0>0 && d+0>0)
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ph,key,nm,rd,c,e,d
        }' "$out" >> "$work/data.tsv" || return 1
}

echo "page_ab: $ROUNDS alternating rounds, best of $BESTN (classes: $CLASS_REPS)" >&2
for ((r = 1; r <= ROUNDS; r++)); do
    if (( r % 2 )); then order=(a b); else order=(b a); fi
    for nm in "${order[@]}"; do
        timed_run "$work/harness_$nm" "$nm" head head "$r" "$BESTN" "$headfile" || exit 1
    done
done
for f in "$work"/pages/page-*.bin; do
    key=$(basename "$f" .bin); key=${key#page-}
    for ((r = 1; r <= ROUNDS; r++)); do
        if (( r % 2 )); then order=(a b); else order=(b a); fi
        for nm in "${order[@]}"; do
            timed_run "$work/harness_$nm" "$nm" class "$key" "$r" "$CLASS_REPS" "$f" || exit 1
        done
    done
done

# ---- statistics -----------------------------------------------------------
cat > "$work/stats.awk" <<'AWKEOF'
BEGIN { FS="\t" }
{
    ph=$1; key=$2; build=$3; rd=$4+0; codec=$5; enc=$6+0; dec=$7+0
    grp[ph SUBSEP key]=1
    kc=ph SUBSEP key SUBSEP codec
    seen[kc]=1
    cc[kc SUBSEP build SUBSEP "enc" SUBSEP rd]=enc
    cc[kc SUBSEP build SUBSEP "dec" SUBSEP rd]=dec
}
function get(ph,key,codec,build,metric,rd,   k) {
    k=ph SUBSEP key SUBSEP codec SUBSEP build SUBSEP metric SUBSEP rd
    return (k in cc) ? cc[k] : 0
}
function bst(ph,key,codec,build,metric,   r,m,v) {
    m=0
    for (r=1; r<=R; r++) { v=get(ph,key,codec,build,metric,r); if (v>m) m=v }
    return m
}
function med(ph,key,codec,build,metric,   a,n,i,j,x,k,r) {
    n=0
    for (r=1; r<=R; r++) {
        k=ph SUBSEP key SUBSEP codec SUBSEP build SUBSEP metric SUBSEP r
        if (k in cc) a[++n]=cc[k]
    }
    if (n==0) return 0
    for (i=2; i<=n; i++) {
        x=a[i]; j=i-1
        while (j>0 && a[j]>x) { a[j+1]=a[j]; j-- }
        a[j+1]=x
    }
    return (n%2) ? a[(n+1)/2] : (a[n/2]+a[n/2+1])/2.0
}
function have(ph,key,codec,build,metric,   r,k) {
    for (r=1; r<=R; r++) {
        k=ph SUBSEP key SUBSEP codec SUBSEP build SUBSEP metric SUBSEP r
        if (k in cc) return 1
    }
    return 0
}
function sgn(ph,key,codec,metric,   r,a,b,p,m) {
    p=0; m=0
    for (r=1; r<=R; r++) {
        a=get(ph,key,codec,na,metric,r); b=get(ph,key,codec,nb,metric,r)
        if (a<=0 || b<=0) continue
        if (b>a) p++; else if (b<a) m++
    }
    if (p*4 >= R*3 && p>m) return "WIN"
    if (m*4 >= R*3 && m>p) return "LOSS"
    return "noise"
}
function sortstr(a,n,   i,j,x) {
    for (i=2; i<=n; i++) {
        x=a[i]; j=i-1
        while (j>0 && a[j]>x) { a[j+1]=a[j]; j-- }
        a[j+1]=x
    }
}
function cell(ph,key,codec,build,metric,
              b,m) { b=bst(ph,key,codec,build,metric); m=med(ph,key,codec,build,metric);
        return sprintf("%7.1f/%-7.1f", b, m) }
function emit(ph,key,label,   codes,codeseen,nc,k,p,i,c,re,rd,se,sd) {
    nc=0
    for (k in seen) {
        split(k,p,SUBSEP)
        if (p[1]==ph && p[2]==key && !(p[3] in codeseen)) {
            codeseen[p[3]]=1; codes[++nc]=p[3]
        }
    }
    sortstr(codes,nc)
    for (i=1; i<=nc; i++) {
        c=codes[i]
        # A mode the older ref cannot run has no rate there: the gate still
        # reports its bytes, but there is no pair to time.
        if (!have(ph,key,c,na,"enc") || !have(ph,key,c,nb,"enc") ||
            !have(ph,key,c,na,"dec") || !have(ph,key,c,nb,"dec")) continue
        re=bst(ph,key,c,nb,"enc")/bst(ph,key,c,na,"enc")
        rd=bst(ph,key,c,nb,"dec")/bst(ph,key,c,na,"dec")
        se=sgn(ph,key,c,"enc"); sd=sgn(ph,key,c,"dec")
        if (ph=="class") printf "| %-12s | %-20s |", label, c
        else printf "| %-20s |", c
        printf " %16s | %16s | %8.4f | %-6s | %16s | %16s | %8.4f | %-6s |\n", \
            cell(ph,key,c,na,"enc"), cell(ph,key,c,nb,"enc"), re, se, \
            cell(ph,key,c,na,"dec"), cell(ph,key,c,nb,"dec"), rd, sd
        if (ph=="head") {
            if (re>0 && rd>0) { ge+=log(re); gd+=log(rd); gn++ }
            if (se=="WIN") we++; else if (se=="LOSS") le++; else ne++
            if (sd=="WIN") wd++; else if (sd=="LOSS") ld++; else nd++
        }
    }
}
END {
    printf "| codec | %s enc best/med | %s enc best/med | enc B/A | enc sign | %s dec best/med | %s dec best/med | dec B/A | dec sign |\n", \
        la, lb, la, lb
    printf "| --- | --- | --- | ---: | --- | --- | --- | ---: | --- |\n"
    emit("head","head","pages-nonzero.bin")
    if (gn>0) {
        printf "\nENCODE geomean %s/%s = %.5f (WIN %d, LOSS %d, noise %d)\n", \
            lb, la, exp(ge/gn), we, le, ne
        printf "DECODE geomean %s/%s = %.5f (WIN %d, LOSS %d, noise %d)\n", \
            lb, la, exp(gd/gn), wd, ld, nd
    }
    nk=0
    for (k in grp) {
        split(k,p,SUBSEP)
        if (p[1]=="class" && !(p[2] in kseen)) { kseen[p[2]]=1; keys[++nk]=p[2] }
    }
    sortstr(keys,nk)
    printf "\n### per class\n\n"
    printf "| class | codec | %s enc best/med | %s enc best/med | enc B/A | enc sign | %s dec best/med | %s dec best/med | dec B/A | dec sign |\n", \
        la, lb, la, lb
    printf "| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | --- |\n"
    for (i=1; i<=nk; i++) emit("class",keys[i],keys[i])
}
AWKEOF

echo
echo "## timing: $A (A) vs $B (B)"
echo
echo "Best/median MB/s across $ROUNDS alternating rounds. Ratios are B/A from"
echo "the best of each round; a sign is a WIN or LOSS only when it holds in at"
echo "least 75% of the rounds, otherwise it is noise."
echo
awk -v R="$ROUNDS" -v na=a -v nb=b -v la="$A" -v lb="$B" \
    -f "$work/stats.awk" "$work/data.tsv" || exit 1

echo
echo "Corpus: \`pages-nonzero.bin\`, $SIZE bytes, sha256 \`${digest:0:16}…\`."
echo "Both builds use the same harness source, lz4 version, compiler and flags;"
echo "only include/ and src/ come from the two refs. Best of $BESTN inside a"
echo "round, best and median across $ROUNDS rounds, order a,b,b,a,... ."
exit 0
