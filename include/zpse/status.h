#ifndef ZPSE_STATUS_H
#define ZPSE_STATUS_H

/* PSE v1.0.0 status codes. 0 = OK; negative = error. */

typedef enum PseStatus {
    PSE_OK                    =  0,
    PSE_BAD_MAGIC               = -1,
    PSE_BAD_VERSION             = -2,
    PSE_BAD_HEADER_CRC          = -3,
    PSE_BAD_HEADER              = -4,
    PSE_BAD_TRANSFORM           = -5,
    PSE_BAD_BLOCK_SIZE          = -6,
    PSE_BAD_SEGMENT_DIRECTORY   = -7,
    PSE_BAD_TILE_SIZE           = -8,
    PSE_BAD_PREDICTOR           = -9,
    PSE_BAD_MASK                = -10,
    PSE_BAD_ENUMERATIVE_RANK    = -11,
    PSE_BAD_RESIDUAL            = -12,
    PSE_UNSUPPORTED_FEATURE     = -13,
    PSE_TRUNCATED               = -14,
    PSE_BAD_ARG                 = -15,
    PSE_OUTPUT_OVERFLOW         = -16,
    PSE_BAD_PAYLOAD_CRC         = -17
} PseStatus;

static inline const char *zpse_status_name(int s) {
    switch (s) {
    case PSE_OK: return "PSE_OK";
    case PSE_BAD_MAGIC: return "PSE_BAD_MAGIC";
    case PSE_BAD_VERSION: return "PSE_BAD_VERSION";
    case PSE_BAD_HEADER_CRC: return "PSE_BAD_HEADER_CRC";
    case PSE_BAD_HEADER: return "PSE_BAD_HEADER";
    case PSE_BAD_TRANSFORM: return "PSE_BAD_TRANSFORM";
    case PSE_BAD_BLOCK_SIZE: return "PSE_BAD_BLOCK_SIZE";
    case PSE_BAD_SEGMENT_DIRECTORY: return "PSE_BAD_SEGMENT_DIRECTORY";
    case PSE_BAD_TILE_SIZE: return "PSE_BAD_TILE_SIZE";
    case PSE_BAD_PREDICTOR: return "PSE_BAD_PREDICTOR";
    case PSE_BAD_MASK: return "PSE_BAD_MASK";
    case PSE_BAD_ENUMERATIVE_RANK: return "PSE_BAD_ENUMERATIVE_RANK";
    case PSE_BAD_RESIDUAL: return "PSE_BAD_RESIDUAL";
    case PSE_UNSUPPORTED_FEATURE: return "PSE_UNSUPPORTED_FEATURE";
    case PSE_TRUNCATED: return "PSE_TRUNCATED";
    case PSE_BAD_ARG: return "PSE_BAD_ARG";
    case PSE_OUTPUT_OVERFLOW: return "PSE_OUTPUT_OVERFLOW";
    case PSE_BAD_PAYLOAD_CRC: return "PSE_BAD_PAYLOAD_CRC";
    default: return "PSE_UNKNOWN";
    }
}

#endif /* ZPSE_STATUS_H */
