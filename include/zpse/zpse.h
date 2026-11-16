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
#include "lz2.h"
#include "huf.h"
#include "block.h"
#include "frame.h"
#include "parallel.h"
#include "vector.h"
#include "api.h"

/* New page encoder flags (ZPSE_ENC_*). They change no wire form: every stream
 * they select is read by the ordinary decoder. Their values live in
 * constants.h (TURBO, HUF) and api.h (INSTANT); collected here so a caller
 * including the umbrella sees what each one trades.
 *
 * ZPSE_ENC_PAGE_TURBO    Page rate profile: one cheap LZ pass behind the page
 *                        decode-first margin, skipping the multi-candidate
 *                        decision -- lz4-class encode rate at page size, with
 *                        a stream the decoder reads faster than raw LZ.
 * ZPSE_ENC_PAGE_HUF      Page entropy: allow the Huffman wrapper on a 4 KiB
 *                        page at any LZ level, trading an extra entropy pass
 *                        on both sides for a smaller page payload.
 * ZPSE_ENC_PAGE_INSTANT  No-decision classifier: write a page in one shot
 *                        from a cheap classifier, with no probe, trial or
 *                        sweep -- the fastest page encode, paid for in ratio.
 */

#endif /* ZPSE_H */
