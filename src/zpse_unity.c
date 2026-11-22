/* zpse unity build: compile this ONE file to get the whole library in a
 * single translation unit (max inlining, LTO-friendly). Each src file is
 * also independently compilable; all internals are `static` or ZPSE_-
 * prefixed so the amalgamation has no duplicate symbols. */

#include "crc32c.c"
#include "repack.c"
#include "rlefast.c"
#include "arena.c"
#include "types.c"
#include "transform.c"
#include "predictor.c"
#include "mask.c"
#include "residual.c"
#include "dense.c"
/* Vendored LZ4 1.10.0 supplies the single LZ4 codec. lz4block.c keeps
 * the existing block-form API but routes it through LZ4_compress_default /
 * LZ4_decompress_safe. */
#include "vendor/lz4.c"
#include "lz4block.c"
#include "huf.c"
#include "tile.c"
#include "segment.c"
#include "block.c"
#include "frame.c"
#include "flat7.c"
#include "parallel.c"
#include "vector.c"
#include "pagefast.c"
#include "flat.c"
#include "text.c"
#include "api.c"
#include "f64fast.c"
#include "dpcm1.c"
#include "mrd.c"
