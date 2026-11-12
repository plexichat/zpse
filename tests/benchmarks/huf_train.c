/* Build src/huf_tables.inc: the four fixed Huffman tables the ZPSE_B_HUF
 * form codes against.
 *
 * Usage: huf_train [-o OUT.inc] FILE...
 *
 * The input is any corpus -- the files are read whole and cut into blocks the
 * way the encoder cuts them. Each block is compressed with all three LZ
 * codecs and the smallest payload is kept, because that is the stream a block
 * at the deepest level would actually have won with, and the form only ever
 * wraps a payload that won. Its bytes go into a 256-bucket histogram.
 *
 * The tables are then fitted rather than invented: k-means (k=4, L1 distance
 * on the normalised histograms) partitions the payloads into four families,
 * and each family's summed histogram becomes one table. That is what makes
 * four tables worth 2.1 points over one: an LZ stream over prose, over
 * pointers and over a run of one byte are three different alphabets, and one
 * code for all of them is a compromise none of them likes.
 *
 * Why four and not, say, sixteen: the encoder prices all of them per block
 * (4 x 256 multiply-adds), and the decoder's choice is free (the block names
 * its table), so the cost of another table is binary size -- 8 KiB of lookup
 * plus 512 bytes of codes -- not time. Four is where the marginal table stops
 * paying on the corpora this was trained and checked on.
 *
 * The output is deterministic given the same input files: the k-means
 * initialisation is a fixed greedy farthest-first walk, ties are broken by
 * lowest index, and nothing is random or time-dependent. Regenerating from
 * the same corpus must produce the same file byte for byte, or a block's
 * bytes would depend on when the tables were built.
 *
 * Lengths are built by a plain Huffman tree over the summed counts, clamped
 * to ZPSE_HUF_MAX_LEN, then repaired so the Kraft sum lands exactly on a
 * complete code. A clamp that leaves the code incomplete would let two
 * symbols share an 11-bit prefix, so the repair is not cosmetic; if it cannot
 * find a complete code the table falls back to eight bits for every symbol,
 * which is always complete, and says so on stderr. */

#include "zpse/constants.h"
#include "zpse/lz.h"
#include "zpse/lz2.h"
#include "zpse/lzh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HUF_LOG   11u
#define HUF_SIZE  (1u << HUF_LOG)
#define HUF_SYM   256u
#define HUF_TAB   4u
#define MAX_BLK   40000
#define BLK_SMALL 4096u
#define BLK_BIG   65536u
#define MIN_TRAIN 64u

typedef struct Blk {
    uint16_t c[HUF_SYM];
    uint32_t total;
} Blk;

static Blk *g_blk;
static uint32_t g_nblk;

static uint8_t g_lzbuf[ZPSE_LZ_TABLE_BYTES + 64];
static uint8_t g_out[65568];

/* ---- Corpus -> payload histograms --------------------------------------- */

static void add_block(const uint8_t *p, uint32_t n)
{
    uint32_t a, b, c;
    uint32_t best;
    uint32_t i;
    uint32_t sum;

    if (g_nblk >= MAX_BLK || n < MIN_TRAIN) {
        return;
    }
    /* The three codecs write to the same buffer, so the comparison is on
     * lengths and the winner is re-encoded below. Two passes over one block
     * is nothing here; the trainer runs once per table set. */
    a = zpse_lz_encode(p, n, g_out, (uint32_t)sizeof g_out, g_lzbuf, 13u);
    b = zpse_lz2_encode(p, n, g_out, (uint32_t)sizeof g_out, g_lzbuf, 13u);
    c = zpse_lzh_encode(p, n, g_out, (uint32_t)sizeof g_out, g_lzbuf, 13u);
    best = a;
    if (b != 0u && (best == 0u || b < best)) {
        best = b;
    }
    if (c != 0u && (best == 0u || c < best)) {
        best = c;
    }
    if (best == 0u || best < MIN_TRAIN) {
        return;
    }
    if (a == best) {
        (void)zpse_lz_encode(p, n, g_out, (uint32_t)sizeof g_out, g_lzbuf, 13u);
    } else if (b == best) {
        (void)zpse_lz2_encode(p, n, g_out, (uint32_t)sizeof g_out, g_lzbuf, 13u);
    } else {
        (void)zpse_lzh_encode(p, n, g_out, (uint32_t)sizeof g_out, g_lzbuf, 13u);
    }

    memset(g_blk[g_nblk].c, 0, sizeof g_blk[g_nblk].c);
    for (i = 0; i < best; i++) {
        g_blk[g_nblk].c[g_out[i]]++;
    }
    sum = 0u;
    for (i = 0; i < HUF_SYM; i++) {
        sum += g_blk[g_nblk].c[i];
    }
    g_blk[g_nblk].total = sum;
    g_nblk++;
}

static int load_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    uint8_t *buf;
    long len;
    size_t got;
    uint32_t off;

    if (f == 0) {
        fprintf(stderr, "huf_train: cannot open %s\n", path);
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    len = ftell(f);
    if (len <= 0) {
        fclose(f);
        return 0;
    }
    rewind(f);
    buf = (uint8_t *)malloc((size_t)len);
    if (buf == 0) {
        fclose(f);
        return 0;
    }
    got = fread(buf, 1u, (size_t)len, f);
    fclose(f);
    if (got != (size_t)len) {
        free(buf);
        return 0;
    }

    for (off = 0; off + BLK_SMALL <= (uint32_t)len; off += BLK_SMALL) {
        add_block(buf + off, BLK_SMALL);
    }
    for (off = 0; off + BLK_BIG <= (uint32_t)len; off += BLK_BIG) {
        add_block(buf + off, BLK_BIG);
    }
    if ((uint32_t)len % BLK_SMALL != 0u && (uint32_t)len >= BLK_SMALL) {
        add_block(buf + (uint32_t)len - BLK_SMALL, BLK_SMALL);
    }
    free(buf);
    return 1;
}

/* ---- k-means over the payload histograms -------------------------------- */

static uint32_t g_cent[HUF_TAB][HUF_SYM]; /* fixed point, sum = 1 << 20 */
static uint8_t g_assign[MAX_BLK];

static void normalise(const Blk *b, uint32_t *dst)
{
    uint32_t i;

    for (i = 0; i < HUF_SYM; i++) {
        dst[i] = (uint32_t)(((uint64_t)b->c[i] << 20) / b->total);
    }
}

static uint64_t dist_to(const Blk *b, const uint32_t *c)
{
    uint32_t n[HUF_SYM];
    uint64_t d = 0u;
    uint32_t i;

    normalise(b, n);
    for (i = 0; i < HUF_SYM; i++) {
        d += (n[i] > c[i]) ? (uint64_t)(n[i] - c[i]) : (uint64_t)(c[i] - n[i]);
    }
    return d;
}

static void kmeans(void)
{
    uint32_t pick[HUF_TAB];
    uint64_t bestd[HUF_TAB];
    uint32_t t;
    uint32_t i;
    uint32_t it;

    if (g_nblk == 0u) {
        return;
    }
    /* Farthest-first from block 0: each new centre is the block least well
     * described by the centres already chosen. Deterministic, and it spreads
     * the four over the families actually present instead of seeding them in
     * file order, which would put three of them in the same family whenever
     * the corpus is concatenated per format. */
    pick[0] = 0u;
    for (t = 1; t < HUF_TAB; t++) {
        uint32_t best = 0u;
        uint64_t bd = 0u;

        for (i = 0; i < g_nblk; i++) {
            uint64_t d = ~(uint64_t)0;
            uint32_t j;

            for (j = 0; j < t; j++) {
                uint64_t e = dist_to(&g_blk[i], g_cent[j]);

                if (e < d) {
                    d = e;
                }
            }
            if (d > bd) {
                bd = d;
                best = i;
            }
        }
        pick[t] = best;
        normalise(&g_blk[best], g_cent[t]);
    }
    normalise(&g_blk[0], g_cent[0]);
    for (t = 1; t < HUF_TAB; t++) {
        normalise(&g_blk[pick[t]], g_cent[t]);
    }
    (void)bestd;

    for (it = 0; it < 8u; it++) {
        uint64_t acc[HUF_TAB][HUF_SYM];
        uint64_t tot[HUF_TAB];

        memset(acc, 0, sizeof acc);
        memset(tot, 0, sizeof tot);
        for (i = 0; i < g_nblk; i++) {
            uint32_t best = 0u;
            uint64_t bd = dist_to(&g_blk[i], g_cent[0]);

            for (t = 1; t < HUF_TAB; t++) {
                uint64_t d = dist_to(&g_blk[i], g_cent[t]);

                if (d < bd) {
                    bd = d;
                    best = t;
                }
            }
            g_assign[i] = (uint8_t)best;
            {
                uint32_t s;

                for (s = 0; s < HUF_SYM; s++) {
                    acc[best][s] += g_blk[i].c[s];
                }
                tot[best] += g_blk[i].total;
            }
        }
        for (t = 0; t < HUF_TAB; t++) {
            uint32_t s;

            if (tot[t] == 0u) {
                continue;
            }
            for (s = 0; s < HUF_SYM; s++) {
                g_cent[t][s] = (uint32_t)((acc[t][s] << 20) / tot[t]);
            }
        }
    }
}

/* ---- Huffman: depths, clamp, repair, canonical codes -------------------- */

static const Blk *g_blkref;
static uint8_t g_assign_ref[HUF_TAB];

static void gather(uint32_t t, uint64_t *freq)
{
    uint32_t i;
    uint32_t s;

    for (s = 0; s < HUF_SYM; s++) {
        freq[s] = 1u; /* every symbol keeps a code: an absent one still has to
                       * be codable, and one bit of every count is what makes
                       * the four tables interchangeable. */
    }
    for (i = 0; i < g_nblk; i++) {
        if (g_assign[i] != t) {
            continue;
        }
        for (s = 0; s < HUF_SYM; s++) {
            freq[s] += g_blk[i].c[s];
        }
    }
    (void)g_blkref;
    (void)g_assign_ref;
}

/* Huffman depths via repeated two-smallest merge. 256 leaves, so the O(n^2)
 * scan is 65 536 comparisons and the trainer runs once per table set. */
static void depths(const uint64_t *freq, uint8_t *len)
{
    uint64_t f[2 * HUF_SYM];
    int dad[2 * HUF_SYM];
    int live[2 * HUF_SYM];
    int nlive = 0;
    int next = (int)HUF_SYM;
    int i;
    int j;

    for (i = 0; i < (int)HUF_SYM; i++) {
        f[i] = freq[i];
        dad[i] = -1;
        live[nlive++] = i;
    }
    while (nlive > 1) {
        int ia = 0;
        int ib = -1;

        for (j = 1; j < nlive; j++) {
            if (f[live[j]] < f[live[ia]]) {
                ib = ia;
                ia = j;
            } else if (ib < 0 || f[live[j]] < f[live[ib]]) {
                ib = j;
            }
        }
        if (ia > ib) {
            int tmp = ia;

            ia = ib;
            ib = tmp;
        }
        f[next] = f[live[ia]] + f[live[ib]];
        dad[next] = -1;
        dad[live[ia]] = next;
        dad[live[ib]] = next;
        live[ib] = live[--nlive];
        live[ia] = next;
        next++;
    }
    for (i = 0; i < (int)HUF_SYM; i++) {
        int n = 0;
        int p = i;

        while (dad[p] >= 0) {
            p = dad[p];
            n++;
        }
        len[i] = (uint8_t)(n == 0 ? 1 : n);
    }
}

static uint64_t kraft(const uint8_t *len)
{
    uint64_t k = 0u;
    uint32_t s;

    for (s = 0; s < HUF_SYM; s++) {
        k += (uint64_t)1u << (HUF_LOG - len[s]);
    }
    return k;
}

/* Clamp to HUF_LOG and take the Kraft sum down to exactly HUF_SIZE by
 * lengthening the shortest codes, cheapest reduction first. Returns 0 when it
 * had to fall back to a flat eight-bit code. */
static int make_complete(uint8_t *len, const uint64_t *freq)
{
    uint32_t s;
    uint64_t k;

    for (s = 0; s < HUF_SYM; s++) {
        if (len[s] > HUF_LOG) {
            len[s] = (uint8_t)HUF_LOG;
        }
        if (len[s] == 0u) {
            len[s] = 1u;
        }
    }
    k = kraft(len);
    while (k > HUF_SIZE) {
        uint64_t excess = k - HUF_SIZE;
        uint32_t pick = HUF_SYM;
        uint32_t l;
        int done = 0;

        for (l = 1; l < HUF_LOG && !done; l++) {
            uint64_t r = (uint64_t)1u << (HUF_LOG - l - 1u);

            if (r > excess) {
                continue;
            }
            for (s = 0; s < HUF_SYM; s++) {
                if (len[s] == l &&
                    (pick == HUF_SYM || freq[s] < freq[pick])) {
                    pick = s;
                }
            }
            if (pick != HUF_SYM) {
                len[pick]++;
                k -= r;
                done = 1;
            }
        }
        if (!done) {
            break;
        }
    }
    if (kraft(len) != HUF_SIZE) {
        for (s = 0; s < HUF_SYM; s++) {
            len[s] = (uint8_t)HUF_LOG - 3u; /* eight bits, always complete */
        }
        return 0;
    }
    return 1;
}

static uint32_t rev_bits(uint32_t v, uint32_t n)
{
    uint32_t r = 0u;
    uint32_t i;

    for (i = 0; i < n; i++) {
        r = (r << 1) | (v & 1u);
        v >>= 1;
    }
    return r;
}

/* ---- Emit --------------------------------------------------------------- */

static uint16_t g_ctab[HUF_TAB][HUF_SYM];
static uint16_t g_dtab[HUF_TAB][HUF_SIZE];

static int build_one(uint32_t t, const uint64_t *freq)
{
    uint8_t len[HUF_SYM];
    uint32_t code[HUF_SYM];
    uint32_t rcode[HUF_SYM];
    uint32_t order[HUF_SYM];
    uint32_t i;
    uint32_t j;
    uint32_t s;
    uint32_t c;
    uint32_t prev;
    int ok;

    depths(freq, len);
    ok = make_complete(len, freq);
    if (!ok) {
        fprintf(stderr, "huf_train: table %u fell back to flat 8-bit\n", t);
    }

    /* Canonical assignment in (length, symbol) order. */
    for (s = 0; s < HUF_SYM; s++) {
        order[s] = s;
    }
    for (i = 1; i < HUF_SYM; i++) {
        uint32_t v = order[i];

        j = i;
        while (j > 0u && (len[order[j - 1u]] > len[v] ||
                          (len[order[j - 1u]] == len[v] &&
                           order[j - 1u] > v))) {
            order[j] = order[j - 1u];
            j--;
        }
        order[j] = v;
    }
    c = 0u;
    prev = 0u;
    for (i = 0; i < HUF_SYM; i++) {
        s = order[i];
        c <<= (len[s] - prev);
        prev = len[s];
        code[s] = c;
        c++;
    }
    for (s = 0; s < HUF_SYM; s++) {
        rcode[s] = rev_bits(code[s], len[s]);
        g_ctab[t][s] = (uint16_t)((rcode[s] << 4) | len[s]);
    }

    for (i = 0; i < HUF_SIZE; i++) {
        /* Poison: symbol zero on the longest code. A complete code never
         * reaches one; an incomplete one costs a fixed eleven bits rather
         * than an infinite loop. */
        g_dtab[t][i] = (uint16_t)(HUF_LOG << 8);
    }
    for (s = 0; s < HUF_SYM; s++) {
        uint32_t span = 1u << (HUF_LOG - len[s]);
        uint32_t k;

        for (k = 0; k < span; k++) {
            uint32_t idx = rcode[s] + (k << len[s]);

            if (idx >= HUF_SIZE) {
                fprintf(stderr, "huf_train: table %u overflow at sym %u\n",
                        t, s);
                return 0;
            }
            g_dtab[t][idx] = (uint16_t)(((uint32_t)len[s] << 8) | s);
        }
    }
    printf("table %u: max len %u, kraft exact, %u codes\n", t,
           (unsigned)len[order[HUF_SYM - 1u]], (unsigned)HUF_SYM);
    return 1;
}

static void emit(const char *path)
{
    FILE *f = fopen(path, "wb");
    uint32_t t;
    uint32_t i;

    if (f == 0) {
        fprintf(stderr, "huf_train: cannot write %s\n", path);
        return;
    }
    fprintf(f, "/* Generated by tests/benchmarks/huf_train.c. Do not edit.\n");
    fprintf(f, " *\n");
    fprintf(f, " * Four Huffman code books for the ZPSE_B_HUF form, and the 11-bit\n");
    fprintf(f, " * lookups built from them. See include/zpse/huf.h for the format and\n");
    fprintf(f, " * tests/benchmarks/huf_train.c for how they were fitted.\n");
    fprintf(f, " *\n");
    fprintf(f, " * zpse_huf_ctab[t][s] = (code << 4) | nbits, code bit-reversed.\n");
    fprintf(f, " * zpse_huf_dtab[t][i] = (nbBits << 8) | symbol\n");
    fprintf(f, " */\n\n");
    fprintf(f, "static const uint16_t zpse_huf_ctab[ZPSE_HUF_TABLES]"
               "[ZPSE_HUF_SYMBOLS] = {\n");
    for (t = 0; t < HUF_TAB; t++) {
        fprintf(f, "    {\n");
        for (i = 0; i < HUF_SYM; i += 8u) {
            fprintf(f, "        0x%04x, 0x%04x, 0x%04x, 0x%04x, "
                       "0x%04x, 0x%04x, 0x%04x, 0x%04x,\n",
                    g_ctab[t][i + 0u], g_ctab[t][i + 1u], g_ctab[t][i + 2u],
                    g_ctab[t][i + 3u], g_ctab[t][i + 4u], g_ctab[t][i + 5u],
                    g_ctab[t][i + 6u], g_ctab[t][i + 7u]);
        }
        fprintf(f, "    }%s\n", (t + 1u < HUF_TAB) ? "," : "");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const uint16_t zpse_huf_dtab[ZPSE_HUF_TABLES]"
               "[ZPSE_HUF_TABLE_SIZE] = {\n");
    for (t = 0; t < HUF_TAB; t++) {
        fprintf(f, "    {\n");
        for (i = 0; i < HUF_SIZE; i += 12u) {
            uint32_t k;
            uint32_t lim = (i + 12u <= HUF_SIZE) ? 12u : (HUF_SIZE - i);

            fprintf(f, "        ");
            for (k = 0; k < lim; k++) {
                fprintf(f, "0x%04x,%s", (unsigned)g_dtab[t][i + k],
                        (k + 1u == lim) ? "" : " ");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "    }%s\n", (t + 1u < HUF_TAB) ? "," : "");
    }
    fprintf(f, "};\n");
    fclose(f);
    printf("wrote %s\n", path);
}

int main(int argc, char **argv)
{
    const char *out = "src/huf_tables.inc";
    int i;
    uint32_t t;

    g_blk = (Blk *)calloc(MAX_BLK, sizeof *g_blk);
    if (g_blk == 0) {
        fprintf(stderr, "huf_train: out of memory\n");
        return 1;
    }
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out = argv[++i];
        }
    }
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            i++;
            continue;
        }
        if (argv[i][0] == '-' && argv[i][1] != '\0') {
            continue;
        }
        if (!load_file(argv[i])) {
            return 1;
        }
    }
    if (g_nblk == 0u) {
        fprintf(stderr, "huf_train: no payloads collected\n");
        return 1;
    }
    printf("%u payloads\n", (unsigned)g_nblk);

    kmeans();
    for (t = 0; t < HUF_TAB; t++) {
        uint64_t freq[HUF_SYM];
        uint32_t i2;
        uint32_t members = 0u;

        gather(t, freq);
        for (i2 = 0; i2 < g_nblk; i2++) {
            if (g_assign[i2] == t) {
                members++;
            }
        }
        printf("cluster %u: %u payloads\n", t, (unsigned)members);
        if (!build_one(t, freq)) {
            return 1;
        }
    }
    emit(out);
    free(g_blk);
    return 0;
}
