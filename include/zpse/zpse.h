#ifndef ZPSE_H
#define ZPSE_H

/* zpse umbrella header — public API, status codes, version.
 *
 * Library version: standard semantic versioning, MAJOR.MINOR.PATCH. The
 * three parts are the whole of it -- no stage letter and no build suffix --
 * so the string is what a package manager and an `--version` caller expect,
 * and it compares by the ordinary semver rules.
 *
 * This block is the only place the version is written down.
 * ZPSE_VERSION_STRING is assembled from the parts below by the
 * preprocessor, and zpse_version_string() returns exactly that string, so
 * there is no second literal to fall out of step with them. */
#define ZPSE_VERSION_MAJOR      1
#define ZPSE_VERSION_MINOR      0
#define ZPSE_VERSION_PATCH      0

#define ZPSE_VERSION_STR_(x)    #x
#define ZPSE_VERSION_STR(x)     ZPSE_VERSION_STR_(x)
#define ZPSE_VERSION_STRING                                              \
    ZPSE_VERSION_STR(ZPSE_VERSION_MAJOR) "."                            \
    ZPSE_VERSION_STR(ZPSE_VERSION_MINOR) "."                            \
    ZPSE_VERSION_STR(ZPSE_VERSION_PATCH)

#include "status.h"
#include "constants.h"
#include "bits.h"
#include "types.h"
#include "crc32c.h"
#include "arena.h"
#include "transform.h"
#include "predictor.h"
#include "mask.h"
#include "residual.h"
#include "tile.h"
#include "segment.h"
#include "lz.h"
#include "huf.h"
#include "block.h"
#include "frame.h"
#include "parallel.h"
#include "vector.h"
#include "api.h"
#include "zpse/flat7.h"
#include "zpse/text.h"

/* Retired page encoder flags (ZPSE_ENC_*). The page path no longer selects a
 * form by flag: every level picks the same page form for the same bytes, so
 * the old page-form flags are gone. ZPSE_ENC_PAGE_FAST, ZPSE_ENC_PAGE_TURBO,
 * ZPSE_ENC_PAGE_HUF and ZPSE_ENC_PAGE_INSTANT are no longer defined, and the
 * API refuses them rather than treating them as no-ops. ZPSE_ENC_HUF remains
 * the one flag that wraps an LZ payload in Huffman streams at any level.
 */

#endif /* ZPSE_H */
