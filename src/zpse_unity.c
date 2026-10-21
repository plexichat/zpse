/* zpse unity build: compile this ONE file to get the whole library in a
 * single translation unit (max inlining, LTO-friendly). Each src file is
 * also independently compilable; all internals are `static` or ZPSE_-
 * prefixed so the amalgamation has no duplicate symbols. */

#include "crc32c.c"
#include "arena.c"
#include "types.c"
#include "transform.c"
#include "predictor.c"
#include "combinadic.c"
#include "mask.c"
#include "residual.c"
#include "dense.c"
#include "lz.c"
#include "tile.c"
#include "segment.c"
#include "block.c"
#include "frame.c"
#include "vector.c"
#include "api.c"
