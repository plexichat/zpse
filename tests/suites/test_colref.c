/* The columnar reference codecs must round-trip.
 *
 * tests/benchmarks/bench_colref.c reports the size a reference codec
 * reached on each columnar corpus, and the README quotes those sizes
 * as "what this column is worth when it is coded for what it is". A
 * number from a codec that does not decode back to its input is not
 * that: it is a number from a broken codec. So each reference codec
 * is encoded and decoded here, on every column, and the result must
 * equal the original bytes exactly.
 *
 * The columns themselves come from tests/corpus/columnar_gen.h, the
 * same header the benchmarks use, so a reference codec is proven
 * against the bytes it will actually be measured on.
 *
 * Exit 0 on success, 1 on the first failure. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../corpus/columnar_gen.h"
#include "../benchmarks/col_ref.h"

#define CHUNK (65536u)
#define NCHUNK 4u

static uint8_t in[CHUNK * NCHUNK];
static uint8_t enc[2u * CHUNK + 64u];
static uint8_t dec[CHUNK];

static int fails = 0;

static void check(const char *col, const char *what, int ok)
{
    if (!ok) {
        printf("colref FAILED: %s (%s)\n", col, what);
        fails++;
    }
}

/* Every reference codec, over every chunk of the column: the decoded
 * bytes must equal the source bytes. */
static void check_roundtrip(int kind)
{
    const ZpseColRef *r = zpse_colref_lookup(kind);
    ZpseColGen g;
    size_t c;

    zpse_col_gen_init(&g, kind);
    for (c = 0; c < NCHUNK; c++)
        zpse_col_gen_fill(&g, in + c * CHUNK, CHUNK);

    if (r == NULL) {
        /* var_string has no column codec by design: high-cardinality
         * text is a block codec's job, and the benchmark says so
         * instead of inventing a number for it. */
        return;
    }
    for (c = 0; c < NCHUNK; c++) {
        const uint8_t *src = in + c * CHUNK;
        size_t n = 0;

        if (r->encode(src, CHUNK, enc, sizeof enc, &n) != 0) {
            check(zpse_col_names[kind], "encode failed", 0);
            return;
        }
        if (n > sizeof enc) {
            check(zpse_col_names[kind], "encode overflowed its buffer", 0);
            return;
        }
        memset(dec, 0xA5, sizeof dec);
        if (r->decode(enc, n, dec, CHUNK) != 0) {
            check(zpse_col_names[kind], "decode failed", 0);
            return;
        }
        if (memcmp(dec, src, CHUNK) != 0) {
            check(zpse_col_names[kind], "decoded bytes differ from source", 0);
            return;
        }
    }
}

/* The generator must not depend on how the fill is split: the
 * benchmark fills 4 MiB in one buffer, the file writer fills 64 KiB at
 * a time, and both must produce the same bytes. */
static void check_split_invariance(int kind)
{
    static uint8_t whole[CHUNK * NCHUNK];
    static uint8_t split[CHUNK * NCHUNK];
    ZpseColGen a;
    ZpseColGen b;
    size_t c;

    zpse_col_gen_init(&a, kind);
    zpse_col_gen_fill(&a, whole, sizeof whole);

    zpse_col_gen_init(&b, kind);
    for (c = 0; c < NCHUNK; c++)
        zpse_col_gen_fill(&b, split + c * CHUNK, CHUNK);

    check(zpse_col_names[kind], "chunked fill differs from whole fill",
          memcmp(whole, split, sizeof whole) == 0);
}

int main(void)
{
    int k;

    for (k = 0; k < (int)ZPSE_COL_KINDS; k++) {
        check_roundtrip(k);
        check_split_invariance(k);
    }

    if (fails == 0) {
        printf("COLREF: ALL OK (%d columns)\n", (int)ZPSE_COL_KINDS);
        return 0;
    }
    printf("COLREF: %d FAILURES\n", fails);
    return 1;
}
