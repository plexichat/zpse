#ifndef ZPSE_H
#define ZPSE_H

/* zpse umbrella header — public API, status codes, version.

 * Library version: [stage].[major].[minor]-[build]
 * stage: a = alpha, b = beta, c = candidate, r = release
 * Current: a.1.0-1 (alpha, major 1, minor 0, build 1). */
#define ZPSE_VERSION_STAGE      'a'
#define ZPSE_VERSION_MAJOR      1
#define ZPSE_VERSION_MINOR      0
#define ZPSE_VERSION_PATCH      0
#define ZPSE_VERSION_BUILD      1
#define ZPSE_VERSION_STRING     "a.1.0-1"

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
