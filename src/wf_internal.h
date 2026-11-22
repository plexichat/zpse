#ifndef ZPSE_WF_INTERNAL_H
#define ZPSE_WF_INTERNAL_H

/* Private bridge between the frame API and the parallel frame encode: the
 * whole-frame codec ladder (REPACK, RLEFAST, F64FAST, DPCM1, MRD) is one
 * decision, so both entry points must make it at the same point in a frame.
 * Not installed and not part of the public API. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Try the whole-frame forms in their fixed order; the first commit wins.
 * Returns 1 and sets *out_len_out when `out` holds a complete frame, or 0
 * without touching `out` or *out_len_out when every codec declines. `in`, `out`
 * and `out_len_out` must be non-NULL, n must be nonzero, and enc_flags must
 * have passed the caller's option validation. */
int zpse_whole_frame_try(const uint8_t *in, size_t n, uint8_t *out,
                         size_t out_cap, size_t *out_len_out,
                         uint32_t enc_flags);

#ifdef __cplusplus
}
#endif

#endif /* ZPSE_WF_INTERNAL_H */
