/* The file-based corpus writer and the in-process harness must produce
 * the same bytes for every columnar category, or the two tables in the
 * README are about different data.
 *
 * tests/corpus/corpus_write.c writes a column as one continuous stream
 * in 64 KiB chunks; tests/benchmarks/bench_columnar.c fills a 4 MiB
 * buffer in one call and cuts it into blocks. This compares the two,
 * byte for byte, for every column.
 *
 * Exit 0 on success. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../corpus/columnar_gen.h"

#define CHUNK (65536u)
#define NCHUNK 4u

static uint8_t whole[CHUNK * NCHUNK];
static uint8_t split[CHUNK * NCHUNK];

int main(void)
{
    int k;
    int fails = 0;

    for (k = 0; k < (int)ZPSE_COL_KINDS; k++) {
        ZpseColGen a;
        ZpseColGen b;
        size_t c;

        zpse_col_gen_init(&a, k);
        zpse_col_gen_fill(&a, whole, sizeof whole);

        zpse_col_gen_init(&b, k);
        for (c = 0; c < NCHUNK; c++)
            zpse_col_gen_fill(&b, split + c * CHUNK, CHUNK);

        if (memcmp(whole, split, sizeof whole) != 0) {
            printf("colcorpus FAILED: %s chunked fill differs\n",
                   zpse_col_names[k]);
            fails++;
        }
    }

    if (fails == 0) {
        printf("COLCORPUS: ALL OK (%d columns)\n", (int)ZPSE_COL_KINDS);
        return 0;
    }
    printf("COLCORPUS: %d FAILURES\n", fails);
    return 1;
}
