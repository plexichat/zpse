/* Randomized differential corpus.
 *
 * tests/test_parity.c pins the encoder's OUTPUT over a fixed list of
 * patterns; this suite attacks the same invariant from the other side, over
 * thousands of shapes and sizes instead of dozens. It is built several ways
 * by tests/run.sh / run.ps1 and the printed summary line must be identical
 * in every build:
 *
 *   default (runtime SIMD dispatch)
 *   -DZPSE_FORCE_SCALAR      (the scalar reference paths)
 *   -DZPSE_FORCE_TRIAL_SWAP  (the block sweep takes the other attempt-order
 *                             decision wherever it has one, so its two width-1
 *                             candidates are tried in the opposite order)
 *
 * The last one is the interesting one for this suite: the block encoder's
 * candidate order is a free performance knob rather than a format decision,
 * so the two orders must produce byte-identical compressed blocks for EVERY
 * input, not just the bench corpus. A wide randomized corpus is what makes
 * that claim testable, so the digest below covers each block's size, its
 * compressed length and its compressed bytes.
 *
 * Nothing printed here may depend on the build: no addresses, no timings, no
 * sizeof-derived counts. Exit 0 = every round trip passed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"

static unsigned char enc_arena[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));
static unsigned char dec_arena[ZPSE_DEC_ARENA_SIZE] __attribute__((aligned(64)));
static uint8_t in[65536];
static uint8_t comp[200000];
static uint8_t out[65536];
static uint8_t fcomp[300000];
static uint8_t fout[65536];

static int fails = 0;
static uint64_t dig = 1469598103934665603ull; /* FNV-1a 64 */

static void mix(const uint8_t *p, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        dig ^= (uint64_t)p[i];
        dig *= 1099511628211ull;
    }
}

static void mix_u32(uint32_t v)
{
    uint8_t b[4];
    b[0] = (uint8_t)v;
    b[1] = (uint8_t)(v >> 8);
    b[2] = (uint8_t)(v >> 16);
    b[3] = (uint8_t)(v >> 24);
    mix(b, 4u);
}

/* Deterministic PRNG: identical corpus on every run and platform. */
static uint32_t rs = 0x2468ACE0u;

static uint8_t nr(void)
{
    rs = rs * 1664525u + 1013904223u;
    return (uint8_t)(rs >> 24);
}

static uint32_t nr32(void)
{
    rs = rs * 1664525u + 1013904223u;
    return rs >> 5;
}

/* ------------------------------------------------------------------ *
 * Shapes. Each fills `dst[0..len)` with one regime; the block shapes
 * below either use one of these or concatenate several of them, so a
 * single block can hold more than one kind of content.
 * ------------------------------------------------------------------ */
static void region_fill(uint8_t *dst, uint32_t len, uint32_t kind,
                        uint32_t seed)
{
    uint32_t i;
    uint32_t acc = seed;

    rs = seed * 2654435761u + 1u;
    switch (kind % 8u) {
    case 0:
        memset(dst, 0, len);
        break;
    case 1:
        memset(dst, 0x5Au, len);
        break;
    case 2:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)((seed + i) & 0xFFu);
        }
        break;
    case 3:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)(nr() & 0x0Fu);
        }
        break;
    case 4:
        for (i = 0; i < len; i++) {
            dst[i] = nr();
        }
        break;
    case 5:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)((i % 251u) * 3u);
        }
        break;
    case 6:
        for (i = 0; i < len; i++) {
            dst[i] = (uint8_t)((i & 1u) ? 0xFFu : 0x00u);
        }
        break;
    default:
        for (i = 0; i < len; i++) {
            acc += 1u + (uint32_t)(nr() % 3u);
            dst[i] = (uint8_t)acc;
        }
        break;
    }
}

#define SHAPE_N 19

static const char *const shape_name[SHAPE_N] = {
    "zeros", "const5A", "constFF", "ramp", "ramp251", "inter2",
    "inter8", "inter16", "alt00FF", "sparse97", "sparse251", "lownib",
    "random", "walk", "walknoise", "regimes", "concat", "shifted",
    "prose"
};

static void fill(uint32_t shape, uint32_t n)
{
    uint32_t i;

    switch (shape) {
    case 0:
        memset(in, 0, n);
        break;
    case 1:
        memset(in, 0x5A, n);
        break;
    case 2:
        memset(in, 0xFF, n);
        break;
    case 3:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)(i & 0xFFu);
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)(i % 251u);
        }
        break;
    case 5:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i & 1u) ? 0xA5u : 0x5Au);
        }
        break;
    case 6:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i % 8u) * 31u);
        }
        break;
    case 7:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i % 16u) * 17u);
        }
        break;
    case 8:
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)((i & 1u) ? 0xFFu : 0x00u);
        }
        break;
    case 9:
        memset(in, 0, n);
        for (i = 0; i < n; i += 97u) {
            in[i] = (uint8_t)i;
        }
        break;
    case 10:
        memset(in, 0, n);
        for (i = 0; i < n; i += 251u) {
            in[i] = (uint8_t)(i * 3u);
        }
        break;
    case 11:
        rs = 0xD1B54A32u + n;
        for (i = 0; i < n; i++) {
            in[i] = (uint8_t)(nr() & 0x0Fu);
        }
        break;
    case 12:
        rs = 0x94D049BBu + n;
        for (i = 0; i < n; i++) {
            in[i] = nr();
        }
        break;
    case 13:
        rs = 0x1B873593u + n;
        {
            uint32_t acc = 0u;
            for (i = 0; i < n; i++) {
                acc += 1u + (uint32_t)(nr() % 3u);
                in[i] = (uint8_t)acc;
            }
        }
        break;
    case 14:
        rs = 0x85EBCA6Bu + n;
        {
            uint32_t acc = 0u;
            for (i = 0; i < n; i++) {
                acc += 1u + (uint32_t)(nr() % 3u);
                if ((nr() % 17u) == 0u) {
                    in[i] = nr();
                } else {
                    in[i] = (uint8_t)acc;
                }
            }
        }
        break;
    case 15: { /* three regimes inside one block */
        uint32_t a = n / 3u;
        uint32_t b = (2u * n) / 3u;

        region_fill(in, a, 2u, 0x1234u);
        region_fill(in + a, b - a, 4u, 0x5678u);
        region_fill(in + b, n - b, 0u, 0x9ABCu);
        break;
    }
    case 16: { /* regions of random kinds at random boundaries */
        uint32_t p = 0u;
        uint32_t r = 0u;
        uint32_t keep;

        rs = 0x51ED270Bu + n;
        while (p < n) {
            uint32_t len = 1u + (nr32() % 3000u);

            if (len > n - p) {
                len = n - p;
            }
            /* region_fill reseeds `rs` for its own stream; keep the
             * boundary stream separate so the split stays its own
             * deterministic sequence. */
            keep = rs;
            region_fill(in + p, len, r, 0x1000u + r * 7u + n);
            rs = keep;
            r++;
            p += len;
        }
        break;
    }
    case 18: { /* deterministic English-like prose: lowercase words + spaces.
                * Every byte is < 0x80, so this shape is eligible for the TEXT
                * (sync 0x50BA) page form on the blocks where it is smaller. */
        static const char *const words[] = {
            "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
            "dog", "pack", "my", "box", "with", "five", "dozen", "liquor",
            "jugs", "hello", "world", "compression", "text", "ascii",
            "byte", "stream", "deterministic", "prose"
        };
        const uint32_t nw = (uint32_t)(sizeof words / sizeof words[0]);
        uint32_t p = 0u;

        rs = 0x4B1D5A2Eu + n; /* fixed per length: reproducible in any order */
        while (p < n) {
            const char *w = words[nr32() % nw];
            uint32_t wl = (uint32_t)strlen(w);
            uint32_t j;

            for (j = 0u; j < wl && p < n; j++, p++) {
                in[p] = (uint8_t)w[j];
            }
            if (p < n) {
                in[p++] = (uint8_t)' ';
            }
        }
        break;
    }
    default: { /* the smooth walk, shifted one byte along */
        uint8_t prev = 0u;

        rs = 0x1B873593u + n;
        {
            uint32_t acc = 0u;
            for (i = 0; i < n; i++) {
                uint8_t cur;

                acc += 1u + (uint32_t)(nr() % 3u);
                cur = (uint8_t)acc;
                in[i] = prev;
                prev = cur;
            }
        }
        break;
    }
    }
}

/* ---- Hand-built TEXT block streams ----------------------------------- *
 * The block decoder dispatches the TEXT sync (0x50BA) before the generic
 * 32-byte-header path: a 16-byte header, then a fixed-rate 7-bit pack whose
 * length is n - (n>>3) for a decoded size of n. We build the header and the
 * packed payload here rather than going through the encoder, so a wrong
 * header, CRC, length or out-of-bounds decode cannot hide behind a matching
 * writer. Two generator modes:
 *
 *   mode 0 -- generate n printable ASCII bytes, pack them with an independent
 *             8->7 model, and check the exact decoded bytes;
 *   mode 1 -- generate arbitrary packed bytes (bits 7 are legal in this form)
 *             for the same fixed rate, then check the exact bytes predicted by
 *             the independent 7->8 model as well as the reject/accept safety
 *             contract.
 *
 * Both feed the public zpse_decompress_block entry point with an output
 * canary, exactly like the hostile-suite reject helpers. */

#define TEXT_TOK_MAX 256u               /* largest decoded n used here */
#define TEXT_PACK_MAX (TEXT_TOK_MAX - (TEXT_TOK_MAX >> 3)) /* 224 */

/* Write a 16-byte TEXT header. n is stored literally (our n < 65536), c is
 * the total block size, and the CRC is the low 16 bits of CRC32-C over the
 * first 14 bytes -- the same check the parser recomputes. */
static void text_block_header(uint8_t h[16], uint32_t n, uint32_t c,
                               uint32_t seq)
{
    uint32_t crc;

    memset(h, 0, 16u);
    zpse_store_u16le(h + 0, (uint16_t)ZPSE_BLOCK_SYNC_TEXT);
    h[2] = (uint8_t)(ZPSE_B_RESET | ZPSE_B_LAST);
    /* h[3..6) reserved, left zero */
    zpse_store_u16le(h + 6, (uint16_t)n);
    zpse_store_u16le(h + 8, (uint16_t)c);
    zpse_store_u32le(h + 10, seq);
    crc = zpse_crc32c(h, 14u) & UINT32_C(0xFFFF);
    zpse_store_u16le(h + 14, (uint16_t)crc);
}

/* Decoded length n whose canonical packed length is exactly plen. A multiple
 * of seven lands on a full 8-byte group (plen=7 -> n=8, plen=14 -> n=16);
 * otherwise the final group holds the remaining 1..6 bytes. */
static uint32_t text_len_for_packed(uint32_t plen)
{
    if (plen == 0u) {
        return 0u;
    }
    if ((plen % 7u) == 0u) {
        return (plen / 7u) * 8u;
    }
    return plen + (plen - 1u) / 7u;
}

/* Independent 8->7 pack: bit 7*i+k of each group is bit k of byte i. */
static uint32_t text_pack_model(const uint8_t *ascii, uint32_t n,
                                uint8_t *packed)
{
    uint32_t ip = 0u, op = 0u;

    while (ip < n) {
        uint32_t r = n - ip, i, k, nb;
        uint64_t v = 0u;

        if (r > 8u) {
            r = 8u;
        }
        for (i = 0u; i < r; i++) {
            for (k = 0u; k < 7u; k++) {
                if (((ascii[ip + i] >> k) & 1u) != 0u) {
                    v |= UINT64_C(1) << (7u * i + k);
                }
            }
        }
        nb = (7u * r + 7u) >> 3;
        for (k = 0u; k < nb; k++) {
            packed[op + k] = (uint8_t)(v >> (8u * k));
        }
        ip += r;
        op += nb;
    }
    return op;
}

/* Independent 7->8 unpack. Used for mode 1, where the packed bytes are
 * arbitrary bit patterns rather than the output of our own packer. */
static void text_unpack_model(const uint8_t *packed, uint32_t n,
                              uint8_t *ascii)
{
    uint32_t ip = 0u, op = 0u;

    while (op < n) {
        uint32_t r = n - op, i, k, nb;
        uint64_t v = 0u;

        if (r > 8u) {
            r = 8u;
        }
        nb = (7u * r + 7u) >> 3;
        for (k = 0u; k < nb; k++) {
            v |= (uint64_t)packed[ip + k] << (8u * k);
        }
        for (i = 0u; i < r; i++) {
            uint8_t b = 0u;

            for (k = 0u; k < 7u; k++) {
                if (((v >> (7u * i + k)) & 1u) != 0u) {
                    b = (uint8_t)(b | (1u << k));
                }
            }
            ascii[op + i] = b;
        }
        ip += nb;
        op += r;
    }
}

/* One hand-built TEXT block through the public decoder. plen is the PACKED
 * length; n is derived from it by text_len_for_packed. mode 0 generates the
 * ASCII source and checks the exact bytes; mode 1 generates arbitrary packed
 * bytes and checks the bytes our independent unpack model predicts. Accepted
 * streams are folded into the shared digest so the determinism gate covers
 * them too. */
static void text_stream_case(uint32_t mode, uint32_t plen, uint32_t seq)
{
    static uint8_t src[TEXT_TOK_MAX];
    static uint8_t tok[TEXT_TOK_MAX];
    static uint8_t exp[TEXT_TOK_MAX];
    uint8_t blk[16u + TEXT_TOK_MAX];
    uint8_t obuf[TEXT_TOK_MAX + 16u];
    uint32_t i;
    uint32_t n;
    uint32_t got;
    uint32_t olen = 0xDEADBEEFu;
    int rc;

    if (plen == 0u || plen > TEXT_PACK_MAX) {
        plen = TEXT_PACK_MAX;
    }
    n = text_len_for_packed(plen);

    if (mode == 0u) {
        /* Structured ASCII: 'a'..'z' plus spaces, all < 0x80. */
        rs = 0xA5A5A5A5u + plen + seq;
        for (i = 0u; i < n; i++) {
            uint32_t r = nr32() % 27u;

            src[i] = (r == 0u) ? (uint8_t)' '
                               : (uint8_t)('a' + (r - 1u));
        }
        got = text_pack_model(src, n, tok);
        memcpy(exp, src, n);
    } else {
        /* Arbitrary packed bits, including bit 7 on every byte: legal in a
         * fixed-rate pack. The independent unpack model names the bytes. */
        rs = 0x5A5A5A5Au + plen + seq;
        for (i = 0u; i < plen; i++) {
            tok[i] = nr();
        }
        got = plen;
        text_unpack_model(tok, n, exp);
    }
    /* Both models must agree with the wire's one structural constraint. */
    if (got != plen) {
        printf("FAIL text_stream mode=%u plen=%u seq=%u model packed %u\n",
               (unsigned)mode, (unsigned)plen, (unsigned)seq, (unsigned)got);
        fails++;
        return;
    }

    text_block_header(blk, n, 16u + plen, seq);
    memcpy(blk + 16u, tok, plen);

    memset(obuf, 0xCC, sizeof obuf);
    rc = zpse_decompress_block(blk, 16u + plen, obuf, n, &olen, dec_arena);

    if (rc < 0) {
        /* Rejected: length untouched and nothing written into or past the
         * declared output length (the decoder writes to scratch first). */
        int clean = (olen == 0xDEADBEEFu);

        for (i = 0u; i < (uint32_t)sizeof obuf; i++) {
            if (obuf[i] != 0xCCu) {
                clean = 0;
            }
        }
        if (!clean) {
            printf("FAIL text_stream mode=%u plen=%u seq=%u rejected but "
                   "wrote (rc=%d olen=%u)\n", (unsigned)mode, (unsigned)plen,
                   (unsigned)seq, rc, (unsigned)olen);
            fails++;
        }
    } else if (rc == ZPSE_OK) {
        /* Accepted: reported length must match, the bytes must equal our
         * independent model, and the canary past n must be intact. */
        int ok = (olen == n) && (memcmp(obuf, exp, n) == 0);

        for (i = n; i < (uint32_t)sizeof obuf; i++) {
            if (obuf[i] != 0xCCu) {
                ok = 0;
            }
        }
        if (!ok) {
            printf("FAIL text_stream mode=%u plen=%u seq=%u accepted wrong "
                   "(rc=%d olen=%u n=%u packed=%u)\n", (unsigned)mode,
                   (unsigned)plen, (unsigned)seq, rc, (unsigned)olen,
                   (unsigned)n, (unsigned)plen);
            fails++;
        }
        mix(blk, 16u + plen);
        mix(obuf, n);
        mix_u32(olen);
    } else {
        printf("FAIL text_stream mode=%u plen=%u seq=%u rc=%d n=%u\n",
               (unsigned)mode, (unsigned)plen, (unsigned)seq, rc,
               (unsigned)n);
        fails++;
    }
}

int main(int argc, char **argv)
{
    /* Optional starting PRNG seed, for the nightly job that must not
     * restart from scratch each time. Pass the seed saved by the previous
     * nightly run and fuzzing continues from where it left off. */
    if (argc > 1) {
        rs = (uint32_t)strtoul(argv[1], NULL, 0);
    }
    /* Size spread: every size up to 64, the tile/segment boundaries, a
     * spread of larger powers of two with their neighbours, and
     * pseudorandom sizes across the whole legal range. */
    static const uint32_t edge[] = {
        127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u, 513u, 1023u, 1024u,
        1025u, 2047u, 2048u, 2049u, 4095u, 4096u, 4097u, 8191u, 8192u,
        8193u, 16383u, 16384u, 16385u, 32768u, 65535u, 65536u
    };
    static uint32_t sizes[192];
    uint32_t nsizes = 0u;
    uint32_t si;
    uint32_t shape;
    uint32_t seq = 0u;
    uint32_t cases = 0u;
    uint32_t frames = 0u;
    int i;

    for (i = 1; i <= 64; i++) {
        sizes[nsizes++] = (uint32_t)i;
    }
    for (i = 0; i < (int)(sizeof edge / sizeof edge[0]); i++) {
        sizes[nsizes++] = edge[i];
    }
    rs = 0xF00Du;
    for (i = 0; i < 60; i++) {
        sizes[nsizes++] = 1u + (nr32() % ZPSE_MAX_BLOCK_SIZE);
    }

    for (si = 0u; si < nsizes; si++) {
        uint32_t n = sizes[si];

        for (shape = 0u; shape < SHAPE_N; shape++) {
            uint32_t clen = 0u;
            uint32_t olen = 0u;
            int rc;

            fill(shape, n);
            rc = zpse_compress_block(in, n, comp, sizeof comp, &clen, seq,
                                     1, enc_arena);
            if (rc != 0) {
                printf("FAIL case=%u shape=%s n=%u compress rc=%d(%s)\n",
                       (unsigned)cases, shape_name[shape], (unsigned)n, rc,
                       zpse_status_name(rc));
                fails++;
                seq++;
                continue;
            }
            /* zpse_compress_block reports 0 on success; the byte count
             * is the out parameter. What the encoder DOES guarantee is
             * that a coded block is only ever chosen while it beats the
             * RAW envelope, so no block can ever exceed it. */
            if (clen > n + (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
                printf("FAIL case=%u shape=%s n=%u clen=%u over envelope\n",
                       (unsigned)cases, shape_name[shape], (unsigned)n,
                       (unsigned)clen);
                fails++;
                seq++;
                continue;
            }
            /* The TEXT form is ASCII-only by definition: any block that holds
             * a byte >= 0x80 must not carry the TEXT sync. Parse the header we
             * just wrote and check the chosen form. */
            {
                uint32_t j;
                int non_ascii = 0;

                for (j = 0u; j < n; j++) {
                    if (in[j] >= 0x80u) {
                        non_ascii = 1;
                        break;
                    }
                }
                if (non_ascii) {
                    ZpseBlockHeader bh;
                    int prc = zpse_block_header_parse(&bh, comp, clen);

                    if (prc == ZPSE_OK &&
                        bh.sync == (uint16_t)ZPSE_BLOCK_SYNC_TEXT) {
                        printf("FAIL case=%u shape=%s n=%u TEXT on non-ASCII "
                               "input\n", (unsigned)cases,
                               shape_name[shape], (unsigned)n);
                        fails++;
                    }
                }
            }
            memset(out, 0xA5, n);
            rc = zpse_decompress_block(comp, clen, out, n, &olen, dec_arena);
            if (rc < 0 || olen != n || memcmp(in, out, n) != 0) {
                printf("FAIL case=%u shape=%s n=%u decode rc=%d olen=%u\n",
                       (unsigned)cases, shape_name[shape], (unsigned)n, rc,
                       (unsigned)olen);
                fails++;
                seq++;
                continue;
            }
            mix_u32(n);
            mix_u32(clen);
            mix(comp, clen);
            cases++;
            seq++;
        }
    }

    /* Frames: same contract, over several nominal block sizes, including
     * one that splits a 64 KiB payload into 256 blocks. */
    {
        static const uint32_t fn[7] = { 20000u, 12345u, 65536u, 3000u,
                                        50000u, 900u, 20000u };
        static const uint32_t fnbs[7] = { 1024u, 4096u, 65536u, 256u,
                                          8192u, 512u, 4096u };
        /* Last case is the all-ASCII prose shape, so the frame path reaches
         * the TEXT form and its bytes enter the frame-level digest. */
        static const uint32_t fshape[7] = { 12u, 13u, 11u, 16u, 14u, 15u,
                                            18u };

        for (i = 0; i < 7; i++) {
            uint32_t n = fn[i];
            size_t flen = 0u;
            size_t olen = 0u;
            int rc;

            fill(fshape[i], n);
            rc = zpse_compress_frame(in, (size_t)n, fcomp, sizeof fcomp,
                                     &flen, fnbs[i], enc_arena);
            if (rc != 0) {
                printf("FAIL frame=%d shape=%s n=%u nbs=%u compress rc=%d\n",
                       i, shape_name[fshape[i]], (unsigned)n,
                       (unsigned)fnbs[i], rc);
                fails++;
                continue;
            }
            memset(fout, 0xA5, n);
            rc = zpse_decompress_frame(fcomp, flen, fout, (size_t)n, &olen,
                                       dec_arena);
            if (rc != 0 || olen != (size_t)n ||
                memcmp(in, fout, (size_t)n) != 0) {
                printf("FAIL frame=%d shape=%s n=%u nbs=%u decode rc=%d "
                       "olen=%u\n", i, shape_name[fshape[i]], (unsigned)n,
                       (unsigned)fnbs[i], rc, (unsigned)olen);
                fails++;
                continue;
            }
            mix_u32(n);
            mix_u32(fnbs[i]);
            mix_u32((uint32_t)flen);
            mix(fcomp, flen);
            mix(fout, (size_t)n);
            frames++;
        }
    }

    /* Hand-built TEXT streams: a finite set of ASCII-source streams (exact
     * decode checked) and arbitrary bit-pattern streams (exact unpack model
     * checked), across PACKED lengths from 1 to 224 -- i.e. decoded lengths
     * from 1 to 256. This is the form's own envelope, built by hand, so it is
     * covered independently of the encoder's form selector. */
    {
        static const uint32_t tplens[8] = {
            1u, 2u, 7u, 8u, 14u, 29u, 88u, TEXT_PACK_MAX
        };
        uint32_t m;
        uint32_t k;

        for (m = 0u; m < 2u; m++) {
            for (k = 0u; k < 8u; k++) {
                text_stream_case(m, tplens[k], seq++);
            }
        }
    }

    /* Continuous fuzzing mode: run until killed by timeout. Gated by
     * ZPSE_FUZZ_CONTINUE so the unit-test runner (run.sh/run.ps1) still
     * executes the finite corpus once and exits with a stable digest; CI
     * sets the env var and bounds the run with timeout 10m / 4h. */
    if (getenv("ZPSE_FUZZ_CONTINUE") != 0) {
        uint64_t cont_cases = 0u;
        for (;;) {
            uint32_t n = 1u + (nr32() % ZPSE_MAX_BLOCK_SIZE);
            uint32_t shape = nr32() % SHAPE_N;
            uint32_t clen = 0u;
            uint32_t olen = 0u;
            int rc;

            /* Also hand a fresh TEXT block to the decoder every iteration. */
            text_stream_case((cont_cases & 1u) ? 1u : 0u,
                             1u + (nr32() % TEXT_PACK_MAX), seq);

            fill(shape, n);
            rc = zpse_compress_block(in, n, comp, sizeof comp, &clen, seq,
                                     1, enc_arena);
            if (rc == 0 && clen <= n + (uint32_t)ZPSE_BLOCK_HEADER_SIZE) {
                memset(out, 0xA5, n);
                rc = zpse_decompress_block(comp, clen, out, n, &olen,
                                           dec_arena);
            }
            if (rc < 0 || olen != n || memcmp(in, out, n) != 0) {
                printf("FAIL cont_case=%llu shape=%u n=%u rc=%d\n",
                       (unsigned long long)cont_cases, shape, n, rc);
                fflush(stdout);
                fails++;
            } else {
                mix_u32(n);
                mix_u32(clen);
                mix(comp, clen);
                cont_cases++;
            }
            seq++;

            if ((cont_cases & 2047u) == 0u) {
                printf("FUZZ cont cases=%llu frames=%llu digest=%016llX seed=%08X\n",
                       (unsigned long long)(cases + cont_cases),
                       (unsigned long long)frames,
                       (unsigned long long)dig, rs);
                fflush(stdout);
            }
        }
    }

    printf("FUZZ cases=%u frames=%u digest=%016llX seed=%08X\n",
           (unsigned)cases, (unsigned)frames, (unsigned long long)dig, rs);
    if (fails == 0) {
        printf("FUZZ: ALL OK\n");
    } else {
        printf("FUZZ: %d FAILURES\n", fails);
    }
    return fails != 0;
}
