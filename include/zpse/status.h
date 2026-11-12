#ifndef ZPSE_STATUS_H
#define ZPSE_STATUS_H

/* ZPSE v1.0.0 status codes. 0 = OK; negative = error. */

typedef enum ZpseStatus {
    ZPSE_OK                     =  0,
    ZPSE_BAD_MAGIC              = -1,
    ZPSE_BAD_VERSION            = -2,
    ZPSE_BAD_HEADER_CRC         = -3,
    ZPSE_BAD_HEADER             = -4,
    ZPSE_BAD_TRANSFORM          = -5,
    ZPSE_BAD_BLOCK_SIZE         = -6,
    ZPSE_BAD_SEGMENT_DIRECTORY  = -7,
    ZPSE_BAD_TILE_SIZE          = -8,
    ZPSE_BAD_PREDICTOR          = -9,
    ZPSE_BAD_MASK               = -10,
    ZPSE_BAD_ENUMERATIVE_RANK   = -11,
    ZPSE_BAD_RESIDUAL           = -12,
    ZPSE_UNSUPPORTED_FEATURE    = -13,
    ZPSE_TRUNCATED              = -14,
    ZPSE_BAD_ARG                = -15,
    ZPSE_OUTPUT_OVERFLOW        = -16,
    ZPSE_BAD_PAYLOAD_CRC        = -17,
    ZPSE_BAD_LZ                 = -18,
    /* The entropy-coded LZ form (include/zpse/huf.h): a payload whose Huffman
     * header, stream lengths or bit accounting do not agree, or an inner
     * stream this decoder will not read. Distinct from ZPSE_BAD_LZ because a
     * Huffman payload is refused before its LZ stream is ever handed to the
     * LZ decoder -- the two failures name different places. */
    ZPSE_BAD_HUF                = -19
} ZpseStatus;

static inline const char *zpse_status_name(int s) {
    switch (s) {
    case ZPSE_OK: return "ZPSE_OK";
    case ZPSE_BAD_MAGIC: return "ZPSE_BAD_MAGIC";
    case ZPSE_BAD_VERSION: return "ZPSE_BAD_VERSION";
    case ZPSE_BAD_HEADER_CRC: return "ZPSE_BAD_HEADER_CRC";
    case ZPSE_BAD_HEADER: return "ZPSE_BAD_HEADER";
    case ZPSE_BAD_TRANSFORM: return "ZPSE_BAD_TRANSFORM";
    case ZPSE_BAD_BLOCK_SIZE: return "ZPSE_BAD_BLOCK_SIZE";
    case ZPSE_BAD_SEGMENT_DIRECTORY: return "ZPSE_BAD_SEGMENT_DIRECTORY";
    case ZPSE_BAD_TILE_SIZE: return "ZPSE_BAD_TILE_SIZE";
    case ZPSE_BAD_PREDICTOR: return "ZPSE_BAD_PREDICTOR";
    case ZPSE_BAD_MASK: return "ZPSE_BAD_MASK";
    case ZPSE_BAD_ENUMERATIVE_RANK: return "ZPSE_BAD_ENUMERATIVE_RANK";
    case ZPSE_BAD_RESIDUAL: return "ZPSE_BAD_RESIDUAL";
    case ZPSE_UNSUPPORTED_FEATURE: return "ZPSE_UNSUPPORTED_FEATURE";
    case ZPSE_TRUNCATED: return "ZPSE_TRUNCATED";
    case ZPSE_BAD_ARG: return "ZPSE_BAD_ARG";
    case ZPSE_OUTPUT_OVERFLOW: return "ZPSE_OUTPUT_OVERFLOW";
    case ZPSE_BAD_PAYLOAD_CRC: return "ZPSE_BAD_PAYLOAD_CRC";
    case ZPSE_BAD_LZ: return "ZPSE_BAD_LZ";
    case ZPSE_BAD_HUF: return "ZPSE_BAD_HUF";
    default: return "ZPSE_UNKNOWN";
    }
}

#endif /* ZPSE_STATUS_H */
