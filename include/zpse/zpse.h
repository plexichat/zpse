#ifndef ZPSE_H
#define ZPSE_H

/* zpse umbrella header — public API, status codes, version.
 *
 * Library version: [stage].[major].[minor]-[build], where stage is one
 * letter: a = alpha, b = beta, c = candidate, r = release.
 *
 * This block is the only place the version is written down.
 * ZPSE_VERSION_STRING is assembled from the parts below by the
 * preprocessor, and zpse_version_string() returns exactly that string, so
 * there is no second literal to fall out of step with them. */
#define ZPSE_VERSION_STAGE      a
#define ZPSE_VERSION_MAJOR      1
#define ZPSE_VERSION_MINOR      0
#define ZPSE_VERSION_BUILD      1

#define ZPSE_VERSION_STR_(x)    #x
#define ZPSE_VERSION_STR(x)     ZPSE_VERSION_STR_(x)
#define ZPSE_VERSION_STRING                                              \
    ZPSE_VERSION_STR(ZPSE_VERSION_STAGE) "."                            \
    ZPSE_VERSION_STR(ZPSE_VERSION_MAJOR) "."                            \
    ZPSE_VERSION_STR(ZPSE_VERSION_MINOR) "-"                            \
    ZPSE_VERSION_STR(ZPSE_VERSION_BUILD)

#include "status.h"
#include "constants.h"
#include "bits.h"
#include "types.h"
#include "crc32c.h"
#include "arena.h"
#include "transform.h"
#include "predictor.h"
#include "combinadic.h"
#include "mask.h"
#include "residual.h"
#include "tile.h"
#include "segment.h"
#include "lz.h"
#include "block.h"
#include "frame.h"
#include "vector.h"
#include "api.h"

#endif /* ZPSE_H */
