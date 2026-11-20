/* What ZPSE_ENC_DECODE_FAST used to move, block by block.
 *
 * The decode-first preset this benchmark was written for has been removed
 * from the encoder; the flag is now accepted but inert, so a current build
 * prints two identical columns. Kept to reproduce the historical table in
 * docs/HISTORY.md, which records what the preset did when it existed.
 *
 * When it existed the preset only decided the shape of blocks whose two forms
 * were close, so a frame total said almost nothing about it: on the Silesia
 * text files it changed no block at all, and on `mozilla` it moved a tenth of
 * them, in both directions. This walks each file at 64 KiB and compresses
 * every block twice, once at the level and once with DECODE_FAST set.
 *
 * Usage: bench_decode_first [--level N] FILE ...
 *   --level N   default 6, the level the entry was measured at
 *
 * Prints a markdown table; the first few moved blocks go to stderr with their
 * sizes and block flags, which is what tells a size increase (a tile form
 * replacing an LZ one) from a decrease (the freed tile stream letting the
 * LZ-over-tile stage win). Links src/zpse_unity.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "zpse/zpse.h"
#include "zpse/block.h"

static unsigned char enc[ZPSE_ENC_ARENA_SIZE] __attribute__((aligned(64)));

int main(int argc, char **argv)
{
    static uint8_t in[ZPSE_MAX_BLOCK_SIZE];
    static uint8_t a[ZPSE_MAX_BLOCK_SIZE + 128u];
    static uint8_t b[ZPSE_MAX_BLOCK_SIZE + 128u];
    unsigned level = 6u;
    int ai;
    int first_file = 1;

    for (ai = 1; ai < argc; ai++) {
        if (strcmp(argv[ai], "--level") == 0 && ai + 1 < argc) {
            level = (unsigned)strtoul(argv[++ai], NULL, 10);
            continue;
        }
        if (first_file) {
            printf("| file | blocks | changed | level %u | -D | delta |\n",
                   level);
            printf("|---|---:|---:|---:|---:|---:|\n");
            first_file = 0;
        }
        {
            FILE *f = fopen(argv[ai], "rb");
            size_t n;
            uint32_t seq = 0u;
            long blocks = 0, changed = 0;
            unsigned long long sa = 0, sb = 0;

            if (f == NULL) {
                perror(argv[ai]);
                return 1;
            }
            while ((n = fread(in, 1u, sizeof in, f)) > 0u) {
                uint32_t la = 0u, lb = 0u;
                int rc;

                rc = zpse_compress_block_ex(in, (uint32_t)n, a, sizeof a, &la,
                                            seq, 0, level << 1, enc);
                if (rc != ZPSE_OK) {
                    fprintf(stderr, "%s block %u: rc %d\n", argv[ai], seq, rc);
                    return 1;
                }
                rc = zpse_compress_block_ex(
                    in, (uint32_t)n, b, sizeof b, &lb, seq, 0,
                    (level << 1) | (uint32_t)ZPSE_ENC_DECODE_FAST, enc);
                if (rc != ZPSE_OK) {
                    fprintf(stderr, "%s block %u: rc %d (-D)\n", argv[ai], seq,
                            rc);
                    return 1;
                }
                blocks++;
                sa += la;
                sb += lb;
                if (la != lb || memcmp(a, b, la) != 0) {
                    changed++;
                    if (changed <= 5) {
                        ZpseBlockHeader ha, hb;

                        zpse_block_header_parse(&ha, a,
                                                ZPSE_BLOCK_HEADER_SIZE);
                        zpse_block_header_parse(&hb, b,
                                                ZPSE_BLOCK_HEADER_SIZE);
                        fprintf(stderr,
                                "%s block %u: %u (flags %02x) -> %u (flags "
                                "%02x)\n",
                                argv[ai], seq, la, ha.block_flags, lb,
                                hb.block_flags);
                    }
                }
                seq++;
            }
            fclose(f);
            printf("| %s | %ld | %ld | %llu | %llu | %.3f%% |\n", argv[ai],
                   blocks, changed, sa, sb,
                   sa == 0u ? 0.0
                            : 100.0 * ((double)sb - (double)sa) / (double)sa);
        }
    }
    if (first_file) {
        fprintf(stderr,
                "usage: bench_decode_first [--level N] FILE ...\n");
        return 2;
    }
    return 0;
}
