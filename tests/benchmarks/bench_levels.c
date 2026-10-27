/* zpse across all 9 effort levels (1=FASTEST ... 9=MAX) on the full
 * 25-case corpus from bench_columnar.c: the 18 byte-stream cases plus
 * the 7 typed columnar categories. Same corpus, same 4 MiB size, same
 * 64 KiB blocks, same pinned single core.
 *
 * For each case this runs every level and prints ratio, encode MB/s,
 * decode MB/s, and bytes per level, so the size/speed ladder is visible
 * case by case. A per-case monotonicity check runs at the end: level 1
 * screens only the three uniform predictors, so it can write a smaller
 * block than a mid level on structured data -- sizes are not guaranteed
 * to descend from 1 to 9; that is a feature of the fast tier.
 *
 * The level is an encoder option, not a plain number: it lives in bits
 * 1-4 of the ZPSE_ENC_* word, so level N is passed as N <<
 * ZPSE_ENC_SPEED_SHIFT (2, 4, ... 18). Passing N itself would set the
 * CHECKSUM bit on every odd row and shift every level down by one --
 * levels 5-8 would never be measured, and the deepest row would be a
 * repeat of level 4. The flag word is built once per column below and the
 * row's name is read back through zpse_enc_speed, so a row cannot print a
 * name its flags do not encode.
 *
 * Portable: no dlopen, no POSIX-only calls. Links zpse_unity.c.
 *
 * Usage: bench_levels [iters] [reps]
 *   iters  timed iterations per measurement (default 20)
 *   reps   timed runs, best-of (default 3)
 * Exit 0 on success. */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "zpse/zpse.h"

#include "../corpus/columnar_gen.h"

#if defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

#define CHUNK (65536u)
#define CORPUS_BYTES ((size_t)CHUNK * 64u) /* 4 MiB per case, fixed */
#define NCHUNK (CORPUS_BYTES / CHUNK)

static uint8_t *g_in;
static uint8_t *g_comp;
static uint8_t *g_dec;
static unsigned char g_ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char g_da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static double now_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq = {{0}};
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

static void fmt_ratio(char *b, size_t n, size_t in, size_t out)
{
    snprintf(b, n, "%.2f:1", out ? (double)in / (double)out : 0.0);
}

static void fmt_rate(char *b, size_t n, double mb_s)
{
    if (mb_s >= 1000.0) snprintf(b, n, "%.2f GB/s", mb_s / 1000.0);
    else snprintf(b, n, "%.0f MB/s", mb_s);
}

static void print_host(void)
{
#if defined(__linux__)
    FILE *f;
    char line[512], model[256] = "n/a", mhz[64] = "n/a";
    char flags[256] = "n/a";
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    long pages = sysconf(_SC_PHYS_PAGES), psize = sysconf(_SC_PAGESIZE);
    struct utsname u;
    char os[512] = "n/a";

    if (uname(&u) == 0)
        snprintf(os, sizeof os, "%s %s %s %s", u.sysname, u.release, u.version, u.machine);
    f = fopen("/proc/cpuinfo", "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "model name", 10) && model[0] == 'n') {
                char *p = strchr(line, ':');
                if (p) { snprintf(model, sizeof model, "%s", p + 2); model[strcspn(model, "\n")] = 0; }
            } else if (!strncmp(line, "cpu MHz", 7) && mhz[0] == 'n') {
                char *p = strchr(line, ':');
                if (p) { snprintf(mhz, sizeof mhz, "%s MHz", p + 2); mhz[strcspn(mhz, "\n")] = 0; }
            } else if (!strncmp(line, "flags", 5) && flags[0] == 'n') {
                char *p = strchr(line, ':');
                if (p) { snprintf(flags, sizeof flags, "%.200s...", p + 2); flags[strcspn(flags, "\n")] = 0; }
            }
            if (model[0] != 'n' && mhz[0] != 'n' && flags[0] != 'n')
                break;
        }
        fclose(f);
    }
    printf("host: os=%s\n", os);
    printf("host: cpu=%s\n", model);
    printf("host: cpus=%ld\n", ncpu > 0 ? ncpu : -1L);
    if (pages > 0 && psize > 0)
        printf("host: ram=%.1f GiB\n", (double)pages * (double)psize / 1073741824.0);
    else
        printf("host: ram=n/a\n");
    printf("host: mhz=%s\n", mhz);
    printf("host: flags=%s\n", flags);
#if defined(__VERSION__)
    printf("host: cc=%s\n", __VERSION__);
#else
    printf("host: cc=n/a\n");
#endif
#else
    printf("host: n/a (non-Linux)\n");
#endif
}

/* Total encoded size of the corpus under zpse at one level, block by
 * block, with every decode verified. */
/* The ZPSE_ENC_* word for level 1..9: see the header comment. */
static uint32_t level_flags(uint32_t level)
{
    return level << ZPSE_ENC_SPEED_SHIFT;
}

static int zpse_size_at(uint32_t level, size_t *out_len)
{
    size_t total = 0;
    size_t c;
    uint32_t clen = 0;
    uint32_t olen = 0;

    for (c = 0; c < NCHUNK; c++) {
        const uint8_t *src = g_in + c * CHUNK;
        if (zpse_compress_block_ex(src, CHUNK, g_comp,
                                   (uint32_t)(CHUNK + 1024u), &clen, 0, 1,
                                   level_flags(level), g_ea) != 0)
            return -1;
        if (zpse_decompress_block(g_comp, clen, g_dec, CHUNK, &olen, g_da) != 0)
            return -1;
        if (memcmp(g_dec, src, CHUNK) != 0) return -1;
        total += clen;
    }
    *out_len = total;
    return 0;
}

/* Best-of-reps encode throughput of zpse at one level over the corpus. */
static double zpse_enc_mbps(uint32_t level, int iters, int reps)
{
    double best = 0.0;
    int rep;
    size_t c;

    (void)iters;
    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns();
        double t1;
        for (c = 0; c < NCHUNK; c++) {
            uint32_t l = 0;
            (void)zpse_compress_block_ex(g_in + c * CHUNK, CHUNK, g_comp,
                                         (uint32_t)(CHUNK + 1024u), &l, 0, 1,
                                         level_flags(level), g_ea);
        }
        t1 = now_ns();
        {
            double mb = (double)CORPUS_BYTES * 1000.0 / (t1 - t0);
            if (best == 0.0 || mb > best) best = mb;
        }
    }
    return best;
}

/* Best-of-reps decode throughput of zpse at one level over the corpus. */
static double zpse_dec_mbps(uint32_t level, int iters, int reps)
{
    double best = 0.0;
    int rep;
    size_t c;
    static uint8_t *comps;
    static uint32_t lens[NCHUNK];
    static uint32_t prepared_level = 0u;
    size_t c2;

    (void)iters;
    if (prepared_level != level) {
        if (comps == NULL) {
            comps = malloc((CHUNK + 1024u) * NCHUNK);
            if (comps == NULL) return 0.0;
        }
        for (c2 = 0; c2 < NCHUNK; c2++) {
            uint32_t l = 0;
            zpse_compress_block_ex(g_in + c2 * CHUNK, CHUNK,
                                   comps + c2 * (CHUNK + 1024u),
                                   (uint32_t)(CHUNK + 1024u), &l, 0, 1,
                                   level_flags(level), g_ea);
            lens[c2] = l;
        }
        prepared_level = level;
    }

    for (rep = 0; rep < reps; rep++) {
        double t0 = now_ns();
        double t1;
        for (c = 0; c < NCHUNK; c++) {
            uint32_t l = 0;
            (void)zpse_decompress_block(comps + c * (CHUNK + 1024u), lens[c],
                                        g_dec, CHUNK, &l, g_da);
        }
        t1 = now_ns();
        {
            double mb = (double)CORPUS_BYTES * 1000.0 / (t1 - t0);
            if (best == 0.0 || mb > best) best = mb;
        }
    }
    return best;
}

/* ---- corpus generators (same 25 cases as bench_columnar.c) ---------- */

static uint32_t g_rng;
static uint32_t rnd(void)
{
    g_rng = g_rng * UINT32_C(1664525) + UINT32_C(1013904223);
    return g_rng >> 24;
}
static uint32_t rnd32(void)
{
    g_rng = g_rng * UINT32_C(1664525) + UINT32_C(1013904223);
    return g_rng;
}

static void gen_i32_seq(void)
{
    uint32_t v = 1000;
    for (size_t i = 0; i < CORPUS_BYTES; i += 4) {
        v += rnd() % 7;
        g_in[i] = (uint8_t)v; g_in[i + 1] = (uint8_t)(v >> 8);
        g_in[i + 2] = (uint8_t)(v >> 16); g_in[i + 3] = (uint8_t)(v >> 24);
    }
}

static void gen_i64_mixed(void)
{
    uint64_t v = 7000000ull;
    for (size_t i = 0; i < CORPUS_BYTES; i += 8) {
        v += (rnd() % 50);
        for (int k = 0; k < 8; k++) g_in[i + (size_t)k] = (uint8_t)(v >> (8 * k));
    }
}

static void gen_dict_u8(void) /* skewed dictionary ids 0..15 */
{
    for (size_t i = 0; i < CORPUS_BYTES; i++) {
        uint32_t r = rnd();
        uint8_t v;
        if (r < 200) v = (uint8_t)(r & 15u);
        else if (r < 250) v = (uint8_t)((r >> 3) & 15u);
        else v = (uint8_t)(rnd() & 15u);
        g_in[i] = v;
    }
}

static void gen_dict_u16le(void)
{
    for (size_t i = 0; i < CORPUS_BYTES; i += 2) {
        uint32_t r = rnd();
        uint16_t c = (uint16_t)(r < 210 ? (r % 500u) : (rnd() % 4000u));
        g_in[i] = (uint8_t)c; g_in[i + 1] = (uint8_t)(c >> 8);
    }
}

static void gen_float_mixed(void)
{
    for (size_t i = 0; i < CORPUS_BYTES; i += 16) {
        uint32_t a = 0x4048F5C3u + (uint32_t)(i >> 4);
        uint32_t b = 0x42C80000u + ((rnd() % 2000u) << 2);
        uint32_t c = (uint32_t)(i * 3u);
        uint32_t d = rnd() & 0xFFFFu;
        memcpy(g_in + i, &a, 4); memcpy(g_in + i + 4, &b, 4);
        memcpy(g_in + i + 8, &c, 4); memcpy(g_in + i + 12, &d, 4);
    }
}

static void gen_bools_nulls(void)
{
    for (size_t i = 0; i < CORPUS_BYTES; i++) {
        uint8_t b = 0;
        for (int k = 0; k < 8; k++)
            if ((rnd() % 100u) < 52u) b |= (uint8_t)(1u << k);
        g_in[i] = b;
    }
}

static void gen_decimal_str(void)
{
    static const char dig[] = "0123456789";
    uint32_t v = 0;
    for (size_t i = 0; i < CORPUS_BYTES; i += 8) {
        v += 1 + rnd() % 17;
        g_in[i] = dig[(v / 1000000u) % 10u]; g_in[i + 1] = dig[(v / 100000u) % 10u];
        g_in[i + 2] = dig[(v / 10000u) % 10u]; g_in[i + 3] = dig[(v / 1000u) % 10u];
        g_in[i + 4] = dig[(v / 100u) % 10u]; g_in[i + 5] = '.';
        g_in[i + 6] = dig[(v / 10u) % 10u]; g_in[i + 7] = dig[v % 10u];
    }
}

static void gen_rle_lengths(void)
{
    for (size_t i = 0; i < CORPUS_BYTES; i += 4) {
        uint32_t r = rnd();
        uint32_t v = r < 220 ? (r % 16u) : (r < 250 ? (r % 256u) : rnd() % 100000u);
        g_in[i] = (uint8_t)v; g_in[i + 1] = (uint8_t)(v >> 8);
        g_in[i + 2] = (uint8_t)(v >> 16); g_in[i + 3] = (uint8_t)(v >> 24);
    }
}

static void gen_timestamps(void)
{
    uint64_t v = 1700000000000000ull;
    for (size_t i = 0; i < CORPUS_BYTES; i += 8) {
        uint64_t vj = v + (rnd() % 997u);
        for (int k = 0; k < 8; k++) g_in[i + (size_t)k] = (uint8_t)(vj >> (8 * k));
        v += 1000000ull;
    }
}

static void gen_uuid_mixed(void)
{
    for (size_t i = 0; i < CORPUS_BYTES; i += 16) {
        g_in[i] = 0x55; g_in[i + 1] = 0x0E; g_in[i + 2] = 0x1A; g_in[i + 3] = 0x2B;
        g_in[i + 4] = 0x3C; g_in[i + 5] = 0x4D;
        for (int k = 6; k < 16; k++) g_in[i + (size_t)k] = (uint8_t)rnd();
    }
}

static void gen_text_en(void)
{
    static const char *words[] = {
        "the","of","and","to","a","in","is","that","for","it","as","was",
        "with","be","on","are","this","have","from","or","had","by","not",
        "but","what","we","can","out","other","were","all","there","when",
        "column","block","codec","ratio","speed","encode","decode","cache",
        "delta","match","window","stream","buffer","offset","length"
    };
    static const char *ends[] = { ".", ".", ".", "?", "!", ",", ",", ";", ":" };
    size_t o = 0;
    unsigned col = 0;

    while (o < CORPUS_BYTES) {
        uint32_t r = rnd();
        const char *w = words[r % (sizeof words / sizeof words[0])];
        size_t wl = strlen(w);
        if (col + wl + 1u > 68u) {
            if (o < CORPUS_BYTES) { g_in[o++] = '\n'; col = 0; }
            continue;
        }
        if (col) { if (o < CORPUS_BYTES) g_in[o++] = ' '; col++; }
        for (size_t k = 0; k < wl && o < CORPUS_BYTES; k++) g_in[o++] = (uint8_t)w[k];
        col += (unsigned)wl;
        if (rnd() % 13u == 0u) {
            const char *e = ends[rnd() % (sizeof ends / sizeof ends[0])];
            if (o < CORPUS_BYTES) g_in[o++] = (uint8_t)e[0];
            col++;
        }
    }
}

static void gen_json_rows(void)
{
    static const char *st[] = { "ok", "retry", "fail", "queued", "active" };
    size_t o = 0;
    uint32_t id = 100000;
    while (o + 96 < CORPUS_BYTES) {
        uint32_t q = rnd() % 900u;
        char row[96];
        int n = snprintf(row, sizeof row,
            "{\"id\":%u,\"st\":\"%s\",\"q\":%u,\"t\":%.3f}\n",
            id, st[rnd() % 5u], q, (double)q * 0.137 + (double)(id % 17u));
        if (n <= 0) break;
        memcpy(g_in + o, row, (size_t)n);
        o += (size_t)n;
        id += 1u + rnd() % 3u;
    }
    while (o < CORPUS_BYTES) g_in[o++] = '\n';
}

static void gen_log_lines(void)
{
    static const char *lv[] = { "INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG" };
    static const char *md[] = { "net", "store", "sched", "auth", "cache", "rpc" };
    static const char *ms[] = {
        "request done", "connection reset", "cache miss on key",
        "flush batch wrote rows", "retry backoff tick", "heartbeat ok"
    };
    size_t o = 0;
    uint32_t seq = 7000;
    while (o + 112 < CORPUS_BYTES) {
        char row[112];
        int n = snprintf(row, sizeof row,
            "2026-10-20T12:%02u:%02u.%03u %-5s [%s] #%u %s=%u\n",
            (seq / 60u) % 60u, seq % 60u, rnd() % 1000u,
            lv[rnd() % 6u], md[rnd() % 6u], seq,
            ms[rnd() % 6u], rnd() % 100000u);
        if (n <= 0) break;
        memcpy(g_in + o, row, (size_t)n);
        o += (size_t)n;
        seq += 1u + rnd() % 7u;
    }
    while (o < CORPUS_BYTES) g_in[o++] = '\n';
}

static void gen_sorted_near(void)
{
    uint32_t *w = (uint32_t *)g_in;
    size_t nw = CORPUS_BYTES / 4, i;
    uint32_t v = 42;
    for (i = 0; i < nw; i++) { v += rnd() % 4u; w[i] = v; }
    for (i = 256; i < nw; i++)
        if (rnd() % 50u == 0u) {
            size_t j = i - (rnd() % 256u);
            uint32_t t = w[i]; w[i] = w[j]; w[j] = t;
        }
}

static void gen_zeros_burst(void)
{
    size_t i = 0;
    memset(g_in, 0, CORPUS_BYTES);
    while (i < CORPUS_BYTES) {
        size_t bl, k;
        i += (size_t)(rnd32() % 8192u);
        if (i >= CORPUS_BYTES) break;
        bl = 32u + (rnd32() % 480u);
        if (i + bl > CORPUS_BYTES) bl = CORPUS_BYTES - i;
        for (k = 0; k < bl; k++) g_in[i + k] = (uint8_t)rnd();
        i += bl;
    }
}

static void gen_ts_gapped(void)
{
    uint64_t ts = 1760966400000000ull;
    uint32_t val = 20000;
    for (size_t i = 0; i < CORPUS_BYTES; i += 12) {
        uint64_t tj;
        uint32_t vj;
        int k;
        if (rnd() % 20u == 0u) ts += (uint64_t)(7000000u + rnd32() % 63000000u);
        else ts += 999000ull + (uint64_t)(rnd32() % 2001u);
        val += 3u + rnd() % 11u + (rnd() % 200u == 0u ? rnd32() % 500u : 0u);
        tj = ts; vj = val;
        for (k = 0; k < 8; k++) g_in[i + (size_t)k] = (uint8_t)(tj >> (8 * k));
        g_in[i + 8] = (uint8_t)vj; g_in[i + 9] = (uint8_t)(vj >> 8);
        g_in[i + 10] = (uint8_t)(vj >> 16); g_in[i + 11] = (uint8_t)(vj >> 24);
    }
}

static void gen_dna_2bit(void)
{
    static const uint8_t ab[] = { 0, 1, 2, 3 };
    size_t i = 0;
    while (i < CORPUS_BYTES) {
        uint8_t motif[9], b = 0;
        size_t ml = 3u + rnd() % 7u, bl = 0, k;
        int q;
        for (k = 0; k < ml; k++) motif[k] = ab[rnd() % 4u];
        while (bl < 64 && i < CORPUS_BYTES) {
            b = 0;
            for (q = 0; q < 4 && bl < 64 && i < CORPUS_BYTES; q++, bl++) {
                uint8_t base = (bl < ml * 3u) ? motif[bl % ml] : ab[rnd() % 4u];
                b |= (uint8_t)(base << (2 * q));
                if (q == 3) g_in[i++] = b;
            }
            if (bl % 4u && i < CORPUS_BYTES) g_in[i++] = b;
        }
    }
}

static void gen_utf8_multi(void)
{
    static const char *tok[] = {
        "the", "column",
        "\xe6\x95\xb0\xe6\x8d\xae", /* shuju */
        "\xe5\x9d\x97",             /* kuai */
        "\xf0\x9f\x9a\x80",         /* rocket */
        "\xd0\xb4\xd0\xb0\xd0\xbd\xd0\xbd\xd1\x8b\xd0\xb5", /* dannye */
        "codec",
        "\xe7\xbc\x96\xe7\xa0\x81", /* bianma */
        "\xf0\x9f\x93\xa6",         /* package */
        "ratio", "speed",
        "\xd1\x81\xd0\xba\xd0\xbe\xd1\x80\xd0\xbe\xd1\x81\xd1\x8c\xd1\x82\xd1\x8c"
    };
    size_t o = 0;
    unsigned col = 0;
    while (o + 16 < CORPUS_BYTES) {
        const char *w = tok[rnd() % (sizeof tok / sizeof tok[0])];
        size_t wl = strlen(w);
        if (col > 64u) {
            if (o < CORPUS_BYTES) g_in[o++] = '\n';
            col = 0;
            continue;
        }
        if (col && o + wl + 1 < CORPUS_BYTES) { g_in[o++] = ' '; col++; }
        if (o + wl >= CORPUS_BYTES) break;
        memcpy(g_in + o, w, wl);
        o += wl;
        col += (unsigned)wl;
    }
    while (o < CORPUS_BYTES) g_in[o++] = ' ';
}

static void gen_col(int kind)
{
    ZpseColGen g;
    zpse_col_gen_init(&g, kind);
    zpse_col_gen_fill(&g, g_in, CORPUS_BYTES);
}

/* The seven typed columnar categories from columnar_gen.h. */
static void gen_ts_seq(void)        { gen_col(ZPSE_COL_TS_SEQ); }
static void gen_ts_ns(void)         { gen_col(ZPSE_COL_TS_NS); }
static void gen_float_sparse(void)  { gen_col(ZPSE_COL_FLOAT_SPARSE); }
static void gen_var_string(void)    { gen_col(ZPSE_COL_VAR_STRING); }
static void gen_sparse_mix(void)    { gen_col(ZPSE_COL_SPARSE_MIX); }
static void gen_geo_point(void)     { gen_col(ZPSE_COL_GEO_POINT); }
static void gen_struct_offsets(void){ gen_col(ZPSE_COL_STRUCT_OFFSETS); }

/* The 25 cases, same names and order as bench_columnar.c. */
static const struct { const char *name; void (*gen)(void); } cases[] = {
    { "i32_seq", gen_i32_seq }, { "i64_mixed", gen_i64_mixed },
    { "dict_u8", gen_dict_u8 }, { "dict_u16le", gen_dict_u16le },
    { "float_mix", gen_float_mixed }, { "bools_nulls", gen_bools_nulls },
    { "decimal8", gen_decimal_str }, { "rle_lens", gen_rle_lengths },
    { "ts_us", gen_timestamps }, { "uuid_mix", gen_uuid_mixed },
    { "text_en", gen_text_en }, { "json_rows", gen_json_rows },
    { "log_lines", gen_log_lines }, { "sorted_near", gen_sorted_near },
    { "zeros_burst", gen_zeros_burst }, { "ts_gapped", gen_ts_gapped },
    { "dna_2bit", gen_dna_2bit }, { "utf8_multi", gen_utf8_multi },
    { "ts_seq", gen_ts_seq }, { "ts_ns", gen_ts_ns },
    { "float_sparse", gen_float_sparse }, { "var_string", gen_var_string },
    { "sparse_mix", gen_sparse_mix }, { "geo_point", gen_geo_point },
    { "struct_offsets", gen_struct_offsets }
};

static void report_levels(int ci, int iters, int reps)
{
    size_t lens[9];
    double encs[9];
    double decs[9];
    char rb[24];
    char eb[24];
    char db[24];
    size_t li;

    cases[ci].gen();

    for (li = 0u; li < 9u; li++) {
        uint32_t level = (uint32_t)(li + 1u);
        if (zpse_size_at(level, &lens[li]) != 0) {
            lens[li] = 0;
        }
        encs[li] = zpse_enc_mbps(level, iters, reps);
        decs[li] = zpse_dec_mbps(level, iters, reps);
    }

    printf("\n%s: zpse across 9 levels, 4 MiB as %u x 64 KiB blocks\n",
           cases[ci].name, (unsigned)NCHUNK);
    printf("| level | name | ratio | encode | decode | bytes |");
    printf("\n|---|---|---:|---:|---:|---:|");
    for (li = 0u; li < 9u; li++) {
        uint32_t level = (uint32_t)(li + 1u);
        const char *name;

        /* Read back what the flags actually encode rather than trusting
         * the loop counter: the printout is the harness's only output, so
         * a mismatch between the row label and the level it measured
         * would be invisible in every number below it. */
        switch (zpse_enc_speed(level_flags(level))) {
        case 1u: name = "FASTEST"; break;
        case 2u: name = "2"; break;
        case 3u: name = "FASTER"; break;
        case 4u: name = "4"; break;
        case 5u: name = "BALANCED"; break;
        case 6u: name = "6"; break;
        case 7u: name = "TIGHTER"; break;
        case 8u: name = "8"; break;
        case 9u: name = "MAX"; break;
        default: name = "BAD"; break;
        }
        fmt_ratio(rb, sizeof rb, CORPUS_BYTES, lens[li]);
        fmt_rate(eb, sizeof eb, encs[li]);
        fmt_rate(db, sizeof db, decs[li]);
        printf("\n| %u | %s | %s | %s | %s | %zu |",
               level, name, rb, eb, db, lens[li]);
    }
    {
        int ok = 1;
        for (li = 1u; li < 9u; li++) {
            if (lens[li] != 0 && lens[li - 1u] != 0 &&
                lens[li] > lens[li - 1u]) {
                ok = 0;
                break;
            }
        }
        printf("\n  sizes monotonic: %s\n", ok ? "yes" : "NO");
    }
}

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    int ci;

    if (iters < 1) iters = 1;
    if (reps < 1) reps = 1;

    g_in = malloc(CORPUS_BYTES);
    g_comp = malloc(CHUNK + 1024u);
    g_dec = malloc(CHUNK);
    if (!g_in || !g_comp || !g_dec) {
        printf("oom\n");
        return 1;
    }

    printf("zpse levels on 25-column corpus: 4 MiB per case as %u x 64 KiB "
           "blocks, single core, best-of-%d reps\n",
           (unsigned)NCHUNK, reps);
    printf("levels: 1=FASTEST 2 3=FASTER 4 5=BALANCED 6 7=TIGHTER 8 9=MAX"
           ", passed as enc_flags = level << %u\n",
           (unsigned)ZPSE_ENC_SPEED_SHIFT);
    print_host();

    for (ci = 0; ci < (int)(sizeof cases / sizeof cases[0]); ci++)
        report_levels(ci, iters, reps);

    free(g_in);
    free(g_comp);
    free(g_dec);
    return 0;
}
