/* Columnar in-process shootout: zpse vs zstd / lz4 / zlib / brotli.
 *
 * One thread (pin the runner with taskset -c 0), every codec fed the same
 * bytes, framed the same way (64 KiB blocks), best-of-N, and every decode
 * verified by memcmp before anything is timed. Ratios print as input:output
 * ("x:1"); rates as GB/s above 1000 MB/s. Nothing touches disk in the
 * timed region: corpora are generated in RAM.
 *
 * The competitors are the system runtimes, loaded with dlopen, so no -dev
 * headers are needed anywhere. A runtime that fails to load is announced
 * and skipped, never silently zeroed.
 *
 * Build (Linux; libs on the usual paths):
 *   cc -std=c11 -O2 -DNDEBUG -Iinclude tests/bench_columnar.c \
 *     src/zpse_unity.c -ldl -o /tmp/benchcol
 * Run:
 *   taskset -c 0 /tmp/benchcol [iters] [reps] [case|all] [codec|all]
 * .github/workflows/bench-columnar.yml runs exactly this on every push. */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/utsname.h>
#endif

#include "zpse/zpse.h"

#define BLK (65536u)
#define NB (64u)
#define CSIZE ((size_t)BLK * NB)

static uint8_t *g_in;
static uint8_t *g_comp;
static uint8_t *g_out;
static unsigned char g_ea[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char g_da[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint32_t g_rng;
static uint32_t rnd(void)
{
    g_rng = g_rng * UINT32_C(1664525) + UINT32_C(1013904223);
    return g_rng >> 24;
}

/* ---- columnar generators: 4 MiB each, fixed seed, identical bytes ---- */
static void gen_i32_seq(void)
{
    uint32_t v = 1000;
    for (size_t i = 0; i < CSIZE; i += 4) {
        v += rnd() % 7;
        g_in[i]=(uint8_t)v; g_in[i+1]=(uint8_t)(v>>8);
        g_in[i+2]=(uint8_t)(v>>16); g_in[i+3]=(uint8_t)(v>>24);
    }
}
static void gen_i64_mixed(void)
{
    uint64_t v = 7000000ull;
    for (size_t i = 0; i < CSIZE; i += 8) {
        v += (rnd() % 50);
        for (int k = 0; k < 8; k++) g_in[i+(size_t)k]=(uint8_t)(v>>(8*k));
    }
}
static void gen_dict_u8(void) /* skewed dictionary ids 0..15 */
{
    for (size_t i = 0; i < CSIZE; i++) {
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
    for (size_t i = 0; i < CSIZE; i += 2) {
        uint32_t r = rnd();
        uint16_t c = (uint16_t)(r < 210 ? (r % 500u) : (rnd() % 4000u));
        g_in[i]=(uint8_t)c; g_in[i+1]=(uint8_t)(c>>8);
    }
}
static void gen_float_mixed(void)
{
    for (size_t i = 0; i < CSIZE; i += 16) {
        uint32_t a = 0x4048F5C3u + (uint32_t)(i >> 4);
        uint32_t b = 0x42C80000u + ((rnd() % 2000u) << 2);
        uint32_t c = (uint32_t)(i * 3u);
        uint32_t d = rnd() & 0xFFFFu;
        memcpy(g_in+i,&a,4); memcpy(g_in+i+4,&b,4);
        memcpy(g_in+i+8,&c,4); memcpy(g_in+i+12,&d,4);
    }
}
static void gen_bools_nulls(void)
{
    for (size_t i = 0; i < CSIZE; i++) {
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
    for (size_t i = 0; i < CSIZE; i += 8) {
        v += 1 + rnd() % 17;
        g_in[i]=dig[(v/1000000u)%10u]; g_in[i+1]=dig[(v/100000u)%10u];
        g_in[i+2]=dig[(v/10000u)%10u]; g_in[i+3]=dig[(v/1000u)%10u];
        g_in[i+4]=dig[(v/100u)%10u]; g_in[i+5]='.';
        g_in[i+6]=dig[(v/10u)%10u]; g_in[i+7]=dig[v%10u];
    }
}
static void gen_rle_lengths(void)
{
    for (size_t i = 0; i < CSIZE; i += 4) {
        uint32_t r = rnd();
        uint32_t v = r < 220 ? (r % 16u) : (r < 250 ? (r % 256u) : rnd() % 100000u);
        g_in[i]=(uint8_t)v; g_in[i+1]=(uint8_t)(v>>8);
        g_in[i+2]=(uint8_t)(v>>16); g_in[i+3]=(uint8_t)(v>>24);
    }
}
static void gen_timestamps(void)
{
    uint64_t v = 1700000000000000ull;
    for (size_t i = 0; i < CSIZE; i += 8) {
        uint64_t vj = v + (rnd() % 997u);
        for (int k = 0; k < 8; k++) g_in[i+(size_t)k]=(uint8_t)(vj>>(8*k));
        v += 1000000ull;
    }
}
static void gen_uuid_mixed(void)
{
    for (size_t i = 0; i < CSIZE; i += 16) {
        g_in[i]=0x55; g_in[i+1]=0x0E; g_in[i+2]=0x1A; g_in[i+3]=0x2B;
        g_in[i+4]=0x3C; g_in[i+5]=0x4D;
        for (int k = 6; k < 16; k++) g_in[i+(size_t)k]=(uint8_t)rnd();
    }
}

/* ---- codec abstraction: zpse direct, rivals via dlopen ---- */
typedef struct {
    const char *name, *level;
    int ok;
    size_t (*cbound)(size_t n);
    size_t (*comp)(const uint8_t *in, size_t n, uint8_t *out, size_t cap);
    int (*decomp)(const uint8_t *in, size_t cn, uint8_t *out, size_t n);
} Codec;

static size_t z_cbound(size_t n) { (void)n; return (size_t)BLK + 1024u; }
static size_t z_comp(const uint8_t *in, size_t n, uint8_t *out, size_t cap)
{
    uint32_t l = 0;
    if (zpse_compress_block(in,(uint32_t)n,out,(uint32_t)cap,&l,0,1,g_ea)!=0) return 0;
    return l;
}
static int z_decomp(const uint8_t *in, size_t cn, uint8_t *out, size_t n)
{
    uint32_t l = 0;
    if (zpse_decompress_block(in,(uint32_t)cn,out,(uint32_t)n,&l,g_da)!=0) return -1;
    return l == n ? 0 : -1;
}

static void *h_zstd, *h_lz4, *h_z, *h_bre, *h_brd;
static size_t (*p_ZSTD_compressBound)(size_t);
static size_t (*p_ZSTD_compress)(void*,size_t,const void*,size_t,int);
static size_t (*p_ZSTD_decompress)(void*,size_t,const void*,size_t);
static unsigned (*p_ZSTD_isError)(size_t);
static int (*p_LZ4_compress_default)(const char*,char*,int,int);
static int (*p_LZ4_compressBound)(int);
static int (*p_LZ4_decompress_safe)(const char*,char*,int,int);
static int (*p_compress2)(uint8_t*,unsigned long*,const uint8_t*,unsigned long,int);
static int (*p_uncompress)(uint8_t*,unsigned long*,const uint8_t*,unsigned long);
static size_t (*p_BrotliEncoderMaxCompressedSize)(size_t);
static int (*p_BrotliEncoderCompress)(int,int,int,size_t,const uint8_t*,size_t*,uint8_t*);
static int (*p_BrotliDecoderDecompress)(size_t,const uint8_t*,size_t*,uint8_t*);
/* Version strings, printed once in the header so a per-commit run says what
 * it measured even after the runner image moves on. */
static const char *(*p_ZSTD_versionString)(void);
static const char *(*p_LZ4_versionString)(void);
static const char *(*p_zlibVersion)(void);
static unsigned (*p_BrotliEncoderVersion)(void);

static size_t zs_cbound(size_t n) { return p_ZSTD_compressBound(n); }
static size_t zs_comp(const uint8_t *in, size_t n, uint8_t *out, size_t cap)
{
    size_t r = p_ZSTD_compress(out,cap,in,n,3);
    return p_ZSTD_isError(r) ? 0 : r;
}
static int zs_decomp(const uint8_t *in, size_t cn, uint8_t *out, size_t n)
{
    size_t r = p_ZSTD_decompress(out,n,in,cn);
    return (p_ZSTD_isError(r) || r != n) ? -1 : 0;
}
static size_t lz_cbound(size_t n) { return (size_t)p_LZ4_compressBound((int)n); }
static size_t lz_comp(const uint8_t *in, size_t n, uint8_t *out, size_t cap)
{
    int r = p_LZ4_compress_default((const char*)in,(char*)out,(int)n,(int)cap);
    return r <= 0 ? 0 : (size_t)r;
}
static int lz_decomp(const uint8_t *in, size_t cn, uint8_t *out, size_t n)
{
    int r = p_LZ4_decompress_safe((const char*)in,(char*)out,(int)cn,(int)n);
    return r != (int)n ? -1 : 0;
}
static size_t zl_cbound(size_t n) { return n + n / 100u + 1024u; }
static size_t zl_comp(const uint8_t *in, size_t n, uint8_t *out, size_t cap)
{
    unsigned long d = (unsigned long)cap;
    return p_compress2(out,&d,in,(unsigned long)n,6)==0 ? (size_t)d : 0;
}
static int zl_decomp(const uint8_t *in, size_t cn, uint8_t *out, size_t n)
{
    unsigned long d = (unsigned long)n;
    return p_uncompress(out,&d,in,(unsigned long)cn)==0 && d==n ? 0 : -1;
}
static size_t br_cbound(size_t n) { return p_BrotliEncoderMaxCompressedSize(n); }
static size_t br_comp(const uint8_t *in, size_t n, uint8_t *out, size_t cap)
{
    size_t d = cap;
    return p_BrotliEncoderCompress(5,22,0,n,in,&d,out) ? d : 0;
}
static int br_decomp(const uint8_t *in, size_t cn, uint8_t *out, size_t n)
{
    size_t d = n;
    return p_BrotliDecoderDecompress(cn,in,&d,out)==1 && d==n ? 0 : -1;
}

static void load_all(void)
{
    h_zstd = dlopen("libzstd.so.1", RTLD_NOW);
    if (h_zstd) {
        p_ZSTD_compressBound = dlsym(h_zstd,"ZSTD_compressBound");
        p_ZSTD_compress = dlsym(h_zstd,"ZSTD_compress");
        p_ZSTD_decompress = dlsym(h_zstd,"ZSTD_decompress");
        p_ZSTD_isError = dlsym(h_zstd,"ZSTD_isError");
        p_ZSTD_versionString = dlsym(h_zstd,"ZSTD_versionString");
    }
    h_lz4 = dlopen("liblz4.so.1", RTLD_NOW);
    if (h_lz4) {
        p_LZ4_compress_default = dlsym(h_lz4,"LZ4_compress_default");
        p_LZ4_compressBound = dlsym(h_lz4,"LZ4_compressBound");
        p_LZ4_decompress_safe = dlsym(h_lz4,"LZ4_decompress_safe");
        p_LZ4_versionString = dlsym(h_lz4,"LZ4_versionString");
    }
    h_z = dlopen("libz.so.1", RTLD_NOW);
    if (h_z) {
        p_compress2 = dlsym(h_z,"compress2");
        p_uncompress = dlsym(h_z,"uncompress");
        p_zlibVersion = dlsym(h_z,"zlibVersion");
    }
    h_bre = dlopen("libbrotlienc.so.1", RTLD_NOW);
    h_brd = dlopen("libbrotlidec.so.1", RTLD_NOW);
    if (h_bre) {
        p_BrotliEncoderMaxCompressedSize = dlsym(h_bre,"BrotliEncoderMaxCompressedSize");
        p_BrotliEncoderCompress = dlsym(h_bre,"BrotliEncoderCompress");
        p_BrotliEncoderVersion = dlsym(h_bre,"BrotliEncoderVersion");
    }
    if (h_brd) p_BrotliDecoderDecompress = dlsym(h_brd,"BrotliDecoderDecompress");
}

/* ---- main: same bytes, 64 KiB blocks, best-of-reps, memcmp-checked ----
 * argv: [iters] [reps] [only_case] [only_codec]; "all" = no filter
 * (empty strings cannot be passed through the wsl/powershell boundary).
 *
 * Presentation: ratios as input:output ("x:1"), rates as GB/s above
 * 1000 MB/s -- both computed from the same per-iteration timings. */

static void fmt_ratio(char *b, size_t n, size_t in, size_t out)
{
    snprintf(b, n, "%.2f:1", out ? (double)in / (double)out : 0.0);
}
static void fmt_rate(char *b, size_t n, double mb_s)
{
    if (mb_s >= 1000.0) {
        snprintf(b, n, "%.2f GB/s", mb_s / 1000.0);
    } else {
        snprintf(b, n, "%.0f MB/s", mb_s);
    }
}

/* ---- host: CPU/RAM/OS/compiler, printed above the results so a saved
 * run is self-describing. Linux-only (/proc, sysconf); elsewhere prints
 * what uname gives and "n/a" for the rest. Never fails the run. */

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

int main(int argc, char **argv)
{
    static const struct { const char *name; void (*gen)(void); } cases[] = {
        { "i32_seq", gen_i32_seq }, { "i64_mixed", gen_i64_mixed },
        { "dict_u8", gen_dict_u8 }, { "dict_u16le", gen_dict_u16le },
        { "float_mix", gen_float_mixed }, { "bools_nulls", gen_bools_nulls },
        { "decimal8", gen_decimal_str }, { "rle_lens", gen_rle_lengths },
        { "ts_us", gen_timestamps }, { "uuid_mix", gen_uuid_mixed },
    };
    const int nc = (int)(sizeof cases / sizeof cases[0]);
    int iters = argc > 1 ? atoi(argv[1]) : 30;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    const char *only = (argc > 3 && strcmp(argv[3], "all") != 0) ? argv[3] : "";
    const char *onlyc = (argc > 4 && strcmp(argv[4], "all") != 0) ? argv[4] : "";
    Codec codecs[5];
    int i, c, r, b;

    g_in = malloc(CSIZE);
    g_comp = malloc((size_t)BLK * 2 + 4096u);
    g_out = malloc(BLK);
    if (!g_in || !g_comp || !g_out) { printf("oom\n"); return 1; }
    load_all();

    codecs[0] = (Codec){ "zpse","default",1,z_cbound,z_comp,z_decomp };
    codecs[1] = (Codec){ "zstd","-3",
        h_zstd && p_ZSTD_compress && p_ZSTD_decompress &&
        p_ZSTD_compressBound && p_ZSTD_isError, zs_cbound, zs_comp, zs_decomp };
    codecs[2] = (Codec){ "lz4","fast",
        h_lz4 && p_LZ4_compress_default && p_LZ4_decompress_safe &&
        p_LZ4_compressBound, lz_cbound, lz_comp, lz_decomp };
    codecs[3] = (Codec){ "zlib","-6",
        h_z && p_compress2 && p_uncompress, zl_cbound, zl_comp, zl_decomp };
    codecs[4] = (Codec){ "brotli","q5",
        h_bre && h_brd && p_BrotliEncoderCompress &&
        p_BrotliDecoderDecompress && p_BrotliEncoderMaxCompressedSize,
        br_cbound, br_comp, br_decomp };

    setvbuf(stdout, NULL, _IONBF, 0);
    print_host();
    printf("columnar in-process: 4 MiB/case as 64x64 KiB blocks, single core, "
           "best-of-reps\n");
    printf("levels: zpse default, zstd -3, lz4 fast, zlib -6, brotli q5; "
           "identical bytes per codec, every decode verified\n");
    printf("versions: zpse %s | zstd %s | lz4 %s | zlib %s | brotli %u.%u.%u\n",
           zpse_version_string(),
           p_ZSTD_versionString ? p_ZSTD_versionString() : "n/a",
           p_LZ4_versionString ? p_LZ4_versionString() : "n/a",
           p_zlibVersion ? p_zlibVersion() : "n/a",
           p_BrotliEncoderVersion ? (p_BrotliEncoderVersion() >> 24) & 255u : 0u,
           p_BrotliEncoderVersion ? (p_BrotliEncoderVersion() >> 16) & 255u : 0u,
           p_BrotliEncoderVersion ? (p_BrotliEncoderVersion() >> 8) & 255u : 0u);
    printf("| case | codec | ratio | encode | decode |\n");
    printf("|---|---|---:|---:|---:|\n");
    for (i = 0; i < nc; i++) {
        if (only[0] && !strstr(cases[i].name, only)) continue;
        g_rng = 0x1234u + (uint32_t)i * 7919u;
        cases[i].gen();
        for (c = 0; c < 5; c++) {
            Codec *k = &codecs[c];
            size_t tot_c = 0, worst = 0, *clens;
            double be = 0.0, bd = 0.0;
            int fail = 0;
            int myiters = iters;
            int slow;
            if (!k->ok) continue;
            if (onlyc[0] && !strstr(k->name, onlyc)) continue;
            /* slow codecs get fewer timed iters; ratio is unaffected */
            slow = (k == &codecs[3] || k == &codecs[4]);
            if (slow) myiters = (iters + 3) / 4;
            worst = k->cbound(BLK);
            clens = malloc(sizeof(size_t) * NB);
            if (!clens) { printf("oom\n"); return 1; }
            for (b = 0; b < (int)NB; b++) {
                size_t cl = k->comp(g_in + (size_t)b * BLK, BLK, g_comp, worst);
                if (!cl) { fail = 1; break; }
                if (k->decomp(g_comp, cl, g_out, BLK) != 0 ||
                    memcmp(g_out, g_in + (size_t)b * BLK, BLK) != 0) { fail = 1; break; }
                clens[b] = cl; tot_c += cl;
            }
            if (fail) {
                printf("| %s | %s | FAILED | - | - |\n", cases[i].name, k->name);
                free(clens); continue;
            }
            for (r = 0; r < reps; r++) {
                double t0 = now_ns(), t1, e_ns;
                for (b = 0; b < (int)NB; b++)
                    for (int t = 0; t < myiters; t++)
                        k->comp(g_in + (size_t)b * BLK, BLK, g_comp, worst);
                t1 = now_ns();
                e_ns = (t1 - t0) / ((double)NB * myiters);
                if (be == 0.0 || e_ns < be) be = e_ns;
                for (b = 0; b < (int)NB; b++) {
                    size_t cl = clens[b];
                    double s0, s1, d_ns;
                    k->comp(g_in + (size_t)b * BLK, BLK, g_comp, worst);
                    s0 = now_ns();
                    for (int t = 0; t < myiters * (slow ? 3 : 10); t++)
                        k->decomp(g_comp, cl, g_out, BLK);
                    s1 = now_ns();
                    d_ns = (s1 - s0) / (myiters * (slow ? 3 : 10));
                    if (bd == 0.0 || d_ns < bd) bd = d_ns;
                }
            }
            {
                char rb[16], eb[24], db[24];
                fmt_ratio(rb, sizeof rb, CSIZE, tot_c);
                fmt_rate(eb, sizeof eb, (double)BLK * 1000.0 / be);
                fmt_rate(db, sizeof db, (double)BLK * 1000.0 / bd);
                printf("| %s | %s %s | %s | %s | %s |\n", cases[i].name,
                       k->name, k->level, rb, eb, db);
            }
            free(clens);
        }
    }
    free(g_in); free(g_comp); free(g_out);
    return 0;
}

