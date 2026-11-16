/* The caller-side page dictionary, measured on a stream with known repeats.
 *
 * Usage: bench_pagedict FILE [REPS]
 *
 * FILE is `pages-repeat.bin` from `corpus_write --pages`, where one page in
 * five is a byte copy of an earlier page. Each arm either compresses every
 * page (none) or hashes it first and serves a stored block on a hit. A hit
 * copies the stored block and calls zpse_block_retag; the arena is not
 * touched, which is the whole point of the API.
 *
 * Every page that misses is inserted, including one whose block came out RAW:
 * a RAW block is a block, and zpse_block_retag rewrites its header just as it
 * does a compressed one. That is what turns a repeated incompressible page
 * from a permanent miss into a hit. Storing one costs a page of slab, which is
 * why the slab is budgeted at a block's worst-case size per page rather than a
 * page's.
 *
 * Packed bytes must be identical across arms -- a retag changes the header,
 * not the payload or its length -- and every emitted block is compared with a
 * fresh encode of the same page at the same position, then decoded and
 * compared with the page, in a verify pass before the rates are taken.
 *
 * The arms are the direct-mapped table sizes and the 4-way sizes worth having
 * (a small table collides, a large one does not), plus one MRU arm that keeps
 * the last page's bytes and slot and memcmp before hashing. That arm also
 * carries a canonical all-zero slot, found with an AVX2 all-zero scan where
 * the host allows one, which is the shape a zram trace has and this corpus
 * has not. Nothing here is a property of the codec; it is a property of the
 * page stream's repeat distance.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "zpse/zpse.h"
#include "zpse/hot.h"

#define PAGE 4096u
#define BLK_CAP (PAGE + 64u)

typedef struct {
    const char *name;
    uint32_t entries;
    uint32_t ways;
    int mru;   /* memcmp the last page's bytes before hashing */
    int zero;  /* canonical all-zero slot via an all-zero scan */
} Arm;

static const Arm arms[] = {
    { "none", 0, 0, 0, 0 },
    { "direct 1024", 1024u, 1u, 0, 0 },
    { "direct 4096", 4096u, 1u, 0, 0 },
    { "direct 65536", 65536u, 1u, 0, 0 },
    { "4-way 4096", 4096u, 4u, 0, 0 },
    { "4-way 65536", 65536u, 4u, 0, 0 },
    { "mru+zero 4096", 4096u, 1u, 1, 1 }
};

#define NARMS ((int)(sizeof arms / sizeof arms[0]))

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static double now_ns(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* All-zero detection for the canonical slot. The AVX2 body is one function:
 * its 256-bit values never cross a call boundary, so it is safe even on hosts
 * where the library keeps its own AVX2 kernels out for ABI reasons
 * (ZPSE_AVX2_ABI_OK). A page is a whole number of 32-byte vectors here. */
#if defined(__GNUC__) && !defined(__clang__) && ZPSE_AVX2_ABI_OK
#define BENCH_HAVE_ZERO_AVX2 1
#include <immintrin.h>

__attribute__((target("avx2")))
static int page_is_zero_avx2(const unsigned char *p)
{
    const __m256i z = _mm256_setzero_si256();
    unsigned int mask = 0xFFFFFFFFu;
    uint32_t i;

    for (i = 0; i < PAGE; i += 32u) {
        __m256i v = _mm256_loadu_si256(
            (const __m256i *)(const void *)(p + i));

        mask &= (unsigned int)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(v, z));
    }
    return mask == 0xFFFFFFFFu;
}
#endif

static int page_is_zero(const unsigned char *p)
{
    uint32_t i;
    uint64_t acc = 0;

#if BENCH_HAVE_ZERO_AVX2
    if ((ZPSE_CPU_CAPS_FAST() & (uint32_t)ZPSE_CPU_AVX2) != 0u)
        return page_is_zero_avx2(p);
#endif
    for (i = 0; i < PAGE; i++) acc |= p[i];
    return acc == 0u;
}

static int run_arm(const Arm *a, const unsigned char *in, size_t npages,
                   unsigned char *slab, uint32_t *off, uint32_t *len,
                   unsigned char *blk, unsigned char *out, int verify,
                   size_t *packed_out, size_t *hits_out)
{
    ZpsePageDict d;
    uint64_t *lo = NULL, *hi = NULL;
    uint32_t *slot = NULL;
    uint8_t *victim = NULL;
    size_t p, used = 0, packed = 0, hits = 0;
    uint32_t next_slot = 0;
    size_t slab_cap = npages * (size_t)BLK_CAP;
    unsigned char ref[BLK_CAP];
    unsigned char last_page[PAGE];
    uint32_t last_slot = 0, zero_slot = 0;
    int have_last = 0, have_zero = 0;
    /* Both cheap checks need a stored slot to serve, so they only make sense
     * on an arm that has a dictionary to put one in. */
    int use_mru = a->mru && a->entries != 0u;
    int use_zero = a->zero && a->entries != 0u;
    int rc = 0;

    if (a->entries) {
        lo = (uint64_t *)malloc((size_t)a->entries * sizeof *lo);
        hi = (uint64_t *)malloc((size_t)a->entries * sizeof *hi);
        slot = (uint32_t *)malloc((size_t)a->entries * sizeof *slot);
        victim = (uint8_t *)malloc((size_t)(a->entries / a->ways));
        if (!lo || !hi || !slot || !victim) {
            free(lo); free(hi); free(slot); free(victim);
            return -1;
        }
        if (a->ways == 1u)
            zpse_pagedict_init(&d, lo, hi, slot, a->entries);
        else
            zpse_pagedict_init_set(&d, lo, hi, slot, victim, a->entries,
                                   a->ways);
    } else {
        memset(&d, 0, sizeof d);
    }

    for (p = 0; p < npages; p++) {
        const unsigned char *src = in + p * PAGE;
        uint32_t seq = (uint32_t)p;
        int is_last = p + 1u == npages;
        uint32_t blen = 0, rlen = 0;
        uint64_t hl = 0, hh = 0;
        uint32_t served_slot = 0;
        int hit = 0;

        /* Cheap checks first: the immediately preceding page, then the
         * canonical zero page. Both serve a slot already in the slab. */
        if (use_mru && have_last && memcmp(src, last_page, PAGE) == 0) {
            hit = 1;
            served_slot = last_slot;
        }
        if (!hit && use_zero && have_zero && page_is_zero(src)) {
            hit = 1;
            served_slot = zero_slot;
        }

        if (!hit && a->entries) {
            zpse_page_hash(src, PAGE, &hl, &hh);
            hit = zpse_pagedict_find(&d, hl, hh, &served_slot);
        }

        if (hit) {
            blen = len[served_slot];
            if (blen > BLK_CAP) { rc = -1; break; }
            memcpy(blk, slab + off[served_slot], blen);
            if (zpse_block_retag(blk, blen, seq, is_last) != ZPSE_OK) {
                rc = -1;
                break;
            }
            hits++;
        } else {
            int n = zpse_compress_block_ex(src, PAGE, blk, BLK_CAP, &blen,
                                           seq, is_last, 0u, enc_arena);
            if (n != 0) { rc = -1; break; }
            /* Store the block whatever form it took. A RAW block is a valid
             * block: retagging it for a repeat writes the same bytes a fresh
             * encode would. */
            if (a->entries) {
                uint32_t s = next_slot++;

                if (s >= npages || used + blen > slab_cap) { rc = -1; break; }
                off[s] = (uint32_t)used;
                len[s] = blen;
                memcpy(slab + used, blk, blen);
                used += blen;
                zpse_pagedict_insert(&d, hl, hh, s);
                served_slot = s;
            }
            if (use_zero && !have_zero && page_is_zero(src)) {
                zero_slot = served_slot;
                have_zero = 1;
            }
        }

        if (use_mru) {
            memcpy(last_page, src, PAGE);
            last_slot = served_slot;
            have_last = 1;
        }

        packed += blen;
        if (verify) {
            uint32_t olen = 0;
            int n = zpse_compress_block_ex(src, PAGE, ref, BLK_CAP, &rlen,
                                           seq, is_last, 0u, enc_arena);

            /* The whole contract: a served block is the byte-for-byte stream
             * a fresh encode of this page at this position would write. */
            if (n != 0 || rlen != blen || memcmp(ref, blk, blen) != 0) {
                rc = -1;
                break;
            }
            if (zpse_decompress_block(blk, blen, out, PAGE, &olen,
                                      dec_arena) != 0 ||
                olen != PAGE || memcmp(out, src, PAGE) != 0) {
                rc = -1;
                break;
            }
        }
    }
    free(lo); free(hi); free(slot); free(victim);
    if (rc == 0) {
        *packed_out = packed;
        *hits_out = hits;
    }
    return rc;
}

int main(int argc, char **argv)
{
    FILE *f;
    long sz;
    unsigned char *in, *slab, *blk, *out;
    uint32_t *off, *len;
    size_t npages, baseline = 0;
    int reps = 3, ai, rep;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: bench_pagedict FILE [REPS]\n");
        return 2;
    }
    if (argc == 3) reps = atoi(argv[2]);
    if (reps < 1) reps = 1;

    f = fopen(argv[1], "rb");
    if (f == NULL) {
        fprintf(stderr, "bench_pagedict: cannot open %s\n", argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % (long)PAGE) != 0) {
        fprintf(stderr, "bench_pagedict: %s is not a page multiple\n", argv[1]);
        fclose(f);
        return 1;
    }
    in = (unsigned char *)malloc((size_t)sz);
    /* One worst-case block per page: a RAW page is stored whole, so a page of
     * slab is not enough headroom. */
    slab = (unsigned char *)malloc(((size_t)sz / PAGE) * (size_t)BLK_CAP);
    off = (uint32_t *)malloc(((size_t)sz / PAGE) * sizeof *off);
    len = (uint32_t *)malloc(((size_t)sz / PAGE) * sizeof *len);
    blk = (unsigned char *)malloc(BLK_CAP);
    out = (unsigned char *)malloc(PAGE);
    if (!in || !slab || !off || !len || !blk || !out ||
        fread(in, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "bench_pagedict: cannot read %s\n", argv[1]);
        free(in); free(slab); free(off); free(len); free(blk); free(out);
        fclose(f);
        return 1;
    }
    fclose(f);
    npages = (size_t)sz / PAGE;

    printf("page dictionary: %zu pages, %d rep(s), level 6, per-page blocks\n\n",
           npages, reps);
    printf("| arm | pages served | encode MB/s | packed bytes | verified |\n");
    printf("| --- | ---: | ---: | ---: | :---: |\n");

    for (ai = 0; ai < NARMS; ai++) {
        size_t packed = 0, hits = 0;
        double best = 0.0;

        if (run_arm(&arms[ai], in, npages, slab, off, len, blk, out, 1,
                    &packed, &hits) != 0) {
            printf("| %s | FAILED | | | |\n", arms[ai].name);
            continue;
        }
        if (ai == 0) baseline = packed;
        for (rep = 0; rep < reps; rep++) {
            double t0 = now_ns(), t1;

            if (run_arm(&arms[ai], in, npages, slab, off, len, blk, out, 0,
                        &packed, &hits) != 0) {
                best = 0.0;
                break;
            }
            t1 = now_ns();
            if (best == 0.0 || t1 - t0 < best) best = t1 - t0;
        }
        if (best == 0.0) {
            printf("| %s | FAILED | | | |\n", arms[ai].name);
            continue;
        }
        printf("| %s | %.2f%% | %.0f | %zu | %s |\n",
               arms[ai].name, 100.0 * (double)hits / (double)npages,
               (double)(npages * PAGE) * 1000.0 / best, packed,
               packed == baseline ? "yes" : "**no**");
    }

    free(in); free(slab); free(off); free(len); free(blk); free(out);
    return 0;
}
