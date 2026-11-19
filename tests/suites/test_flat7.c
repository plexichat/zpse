/* FLAT7: the whole-ASCII fast frame format.
 *
 * A FLAT7 stream is a 40-byte header plus a run of 7-bit packed bytes:
 *   [0..4)    magic "F7P1" (0x31503746 little-endian)
 *   [4..6)    version 1.0
 *   [20..36)  one ASCII7 region record (nregions == 1)
 *   [36..40)  CRC32C over [0..36)
 *   [40..)    payload, packed_len == n - n/8 bytes
 *
 * The suite pins the gate (which inputs are eligible), the byte-exact
 * round-trip across the size boundaries, the frame recogniser and header
 * validator, the decoder's no-write-on-reject guarantee, parity between the
 * two ASCII unpackers, and a real all-ASCII file. Every buffer is static and
 * every case is deterministic, so a failure names the exact n / field.
 *
 * The format lands in parallel with this suite, so every call is guarded on
 * the public header being present. Without it the suite still builds and
 * reports SKIPPED, which keeps the runner green during development. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zpse/zpse.h"
#include "zpse/tile.h"

#if defined(__has_include)
#if __has_include("zpse/flat7.h")
#define ZPSE_TEST_HAVE_FLAT7 1
#include "zpse/flat7.h"
#endif
#endif
/* Include guard fallback for builds where flat7.h is pulled in by the
 * umbrella header but is not findable as a separate include. */
#if !defined(ZPSE_TEST_HAVE_FLAT7) && defined(ZPSE_FLAT7_H)
#define ZPSE_TEST_HAVE_FLAT7 1
#endif

#ifdef ZPSE_TEST_HAVE_FLAT7

#define F7_HDR   40u
#define F7_MAX_N (1u << 20)
#define F7_PAD   256u
#define F7_SMALL 4096u

_Alignas(ZPSE_ARENA_ALIGN) static unsigned char f7_ea[ZPSE_ENC_ARENA_SIZE];
_Alignas(ZPSE_ARENA_ALIGN) static unsigned char f7_da[ZPSE_DEC_ARENA_SIZE];

static uint8_t f7_in[F7_MAX_N];
static uint8_t f7_out[F7_MAX_N + F7_PAD];
static uint8_t f7_frame[F7_MAX_N + F7_PAD];
static uint8_t f7_ref[65536u + F7_PAD];

static int checks;
static int fails;

#define CHECK(cond, ...) do {                       \
    checks++;                                       \
    if (!(cond)) {                                  \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                        \
        printf("\n");                               \
        fails++;                                    \
    }                                               \
} while (0)

/* Deterministic printable ASCII, bytes 32..126, via the same tiny LCG the
 * other suites use. Identical on every host and for every n. */
static void f7_fill_ascii(uint8_t *p, size_t n, uint32_t seed)
{
    uint32_t r = seed;
    size_t i;

    for (i = 0; i < n; i++) {
        r = r * 1664525u + 1013904223u;
        p[i] = (uint8_t)(32u + ((r >> 24) % 95u));
    }
}

/* The bytes past a rejected or short decode must keep the 0xCC canary. */
static int f7_canary(const uint8_t *p, size_t n)
{
    size_t i;

    for (i = 0; i < n; i++) {
        if (p[i] != 0xCCu) return 0;
    }
    return 1;
}

static int f7_make_frame(size_t n, size_t *clen_out)
{
    int rc;

    f7_fill_ascii(f7_in, n, 0xC0FFEEu);
    rc = zpse_flat7_encode(f7_in, n, f7_frame, sizeof f7_frame, clen_out);
    CHECK(rc == ZPSE_OK, "make frame n=%lu: %s", (unsigned long)n,
          zpse_status_name(rc));
    return rc == ZPSE_OK;
}

/* 1. Gate and byte-exact round-trip across the size boundaries. */
static void test_roundtrip(void)
{
    static const size_t ns[] = { 1u, 7u, 8u, 4095u, 4096u, 4097u,
                                 65536u, (1u << 20) };
    unsigned i;

    for (i = 0; i < sizeof ns / sizeof ns[0]; i++) {
        size_t n = ns[i];
        size_t clen = 0;
        size_t got = 0;
        size_t j;
        int rc, want;

        f7_fill_ascii(f7_in, n, 0xC0FFEEu + i);
        want = (n >= F7_SMALL);
        rc = zpse_flat7_should(f7_in, n, 0u, 0, sizeof f7_frame);
        CHECK((rc != 0) == want, "should() at n=%lu is %d, expected %s",
              (unsigned long)n, rc, want ? "true" : "false");
        if (!want) continue;

        rc = zpse_flat7_encode(f7_in, n, f7_frame, sizeof f7_frame, &clen);
        CHECK(rc == ZPSE_OK, "encode n=%lu: %s", (unsigned long)n,
              zpse_status_name(rc));
        if (rc != ZPSE_OK) continue;
        CHECK(clen == F7_HDR + n - n / 8u, "frame length n=%lu is %lu, want %lu",
              (unsigned long)n, (unsigned long)clen,
              (unsigned long)(F7_HDR + n - n / 8u));

        memset(f7_out, 0xCC, sizeof f7_out);
        rc = zpse_flat7_decode(f7_frame, clen, f7_out, n, &got);
        CHECK(rc == ZPSE_OK && got == n && memcmp(f7_out, f7_in, n) == 0,
              "roundtrip n=%lu: %s got=%lu", (unsigned long)n,
              zpse_status_name(rc), (unsigned long)got);
        for (j = n; j < n + 64u; j++) {
            if (f7_out[j] != 0xCCu) break;
        }
        CHECK(j == n + 64u, "decode wrote past n=%lu", (unsigned long)n);
    }
}

/* 2. The gate refuses every input that is not a bare, whole ASCII stream. */
static void test_should_reject(void)
{
    const size_t n = F7_SMALL;

    f7_fill_ascii(f7_in, n, 0x13579u);
    CHECK(zpse_flat7_should(f7_in, n, 0u, 0, sizeof f7_frame) != 0,
          "should(): clean ASCII baseline accepted");

    f7_in[n / 2u] = 0x80u;
    CHECK(zpse_flat7_should(f7_in, n, 0u, 0, sizeof f7_frame) == 0,
          "should(): one non-ASCII byte rejected");
    f7_in[n / 2u] = (uint8_t)'A';

    CHECK(zpse_flat7_should(f7_in, n, 0u, 1, sizeof f7_frame) == 0,
          "should(): indexed request rejected");
    CHECK(zpse_flat7_should(f7_in, n, ZPSE_ENC_CHECKSUM, 0,
                            sizeof f7_frame) == 0,
          "should(): CHECKSUM flag rejected");
    CHECK(zpse_flat7_should(f7_in, n, 0u, 0, n / 2u) == 0,
          "should(): too-small out_cap rejected");
}

/* 3. is_frame recognises only a FLAT7 stream. */
static void test_is_frame(void)
{
    size_t clen = 0;
    uint32_t r = 0xA5A5A5A5u;
    unsigned i;

    /* A real, ordinary zpse frame is not one of ours. zpse_compress_frame
     * never emits FLAT7 — only the CLI routes all-ASCII blocks through the
     * FLAT7 gate — so even an all-ASCII input keeps the "PSE1" magic here.
     * The noisy / non-ASCII fill is simply the ordinary frame baseline. */
    for (i = 0; i < F7_SMALL; i++) {
        r = r * 1664525u + 1013904223u;
        f7_in[i] = (uint8_t)(r >> 24);
    }
    CHECK(zpse_compress_frame(f7_in, F7_SMALL, f7_frame, sizeof f7_frame,
                              &clen, F7_SMALL, f7_ea) == ZPSE_OK,
          "ordinary zpse frame built");
    CHECK(zpse_flat7_is_frame(f7_frame, clen) == 0,
          "is_frame(): ordinary zpse frame refused");

    /* Random bytes, with byte 0 forced away from the magic. */
    for (i = 0; i < 256u; i++) {
        r = r * 1664525u + 1013904223u;
        f7_frame[i] = (uint8_t)(r >> 24);
    }
    f7_frame[0] = 0x00u;
    CHECK(zpse_flat7_is_frame(f7_frame, 256u) == 0,
          "is_frame(): random bytes refused");

    CHECK(f7_make_frame(F7_SMALL, &clen), "flat7 frame for is_frame");
    CHECK(zpse_flat7_is_frame(f7_frame, clen) != 0,
          "is_frame(): FLAT7 frame accepted");
}

/* 4. Header validation: field corruption, CRC and truncation. */
static void test_header_check(void)
{
    const size_t n = F7_SMALL;
    size_t clen = 0;
    size_t got_n = 0;
    size_t got_off = 0;
    uint8_t keep[4];
    int rc;

    CHECK(f7_make_frame(n, &clen), "frame for header_check");

    rc = zpse_flat7_header_check(f7_frame, clen, &got_n, &got_off);
    CHECK(rc == ZPSE_OK && got_n == n && got_off == F7_HDR,
          "header_check valid: %s n=%lu off=%lu", zpse_status_name(rc),
          (unsigned long)got_n, (unsigned long)got_off);

    keep[0] = f7_frame[0];
    f7_frame[0] ^= 0xFFu;
    CHECK(zpse_flat7_header_check(f7_frame, clen, &got_n, &got_off) ==
              ZPSE_BAD_MAGIC,
          "header_check: corrupt magic -> BAD_MAGIC");
    f7_frame[0] = keep[0];

    keep[0] = f7_frame[4];
    f7_frame[4] ^= 0xFFu;
    rc = zpse_flat7_header_check(f7_frame, clen, &got_n, &got_off);
    CHECK(rc < 0, "header_check: bad version rejected (%s)",
          zpse_status_name(rc));
    f7_frame[4] = keep[0];

    keep[0] = f7_frame[36];
    f7_frame[36] ^= 0xFFu;
    rc = zpse_flat7_header_check(f7_frame, clen, &got_n, &got_off);
    CHECK(rc < 0, "header_check: flipped CRC byte rejected (%s)",
          zpse_status_name(rc));
    f7_frame[36] = keep[0];

    /* The region record starts at 20; corrupt its leading fields. */
    memcpy(keep, f7_frame + 20u, sizeof keep);
    memset(f7_frame + 20u, 0xFF, sizeof keep);
    rc = zpse_flat7_header_check(f7_frame, clen, &got_n, &got_off);
    CHECK(rc < 0, "header_check: wrong region type rejected (%s)",
          zpse_status_name(rc));
    memcpy(f7_frame + 20u, keep, sizeof keep);

    rc = zpse_flat7_header_check(f7_frame, clen - 8u, &got_n, &got_off);
    CHECK(rc == ZPSE_TRUNCATED,
          "header_check: truncated payload -> TRUNCATED (%s)",
          zpse_status_name(rc));
}

/* 5. A rejected decode writes nothing into the caller's buffer. */
static void test_decode_no_write(void)
{
    const size_t n = F7_SMALL;
    size_t clen = 0;
    size_t got = 0;
    uint8_t keep;
    int rc;

    CHECK(f7_make_frame(n, &clen), "frame for decode reject");

    memset(f7_out, 0xCC, sizeof f7_out);
    rc = zpse_flat7_decode(f7_frame, clen, f7_out, n - 1u, &got);
    CHECK(rc == ZPSE_OUTPUT_OVERFLOW, "decode: cap n-1 -> OVERFLOW (%s)",
          zpse_status_name(rc));
    CHECK(f7_canary(f7_out, n + 64u), "decode: cap n-1 left the canary alone");

    keep = f7_frame[36];
    f7_frame[36] ^= 0xFFu;
    memset(f7_out, 0xCC, sizeof f7_out);
    rc = zpse_flat7_decode(f7_frame, clen, f7_out, n, &got);
    CHECK(rc < 0, "decode: bad CRC rejected (%s)", zpse_status_name(rc));
    CHECK(f7_canary(f7_out, n + 64u), "decode: bad CRC left the canary alone");
    f7_frame[36] = keep;
}

/* 6. The streaming unpacker and the tile unpacker must agree bit for bit.
 *    Guarded on the stream helper landing alongside the format. */
#ifdef ZPSE_HAVE_ASCII_STREAM
static void test_unpack_parity(void)
{
    static const uint16_t nts[] = { 1u, 7u, 8u, 15u, 16u, 17u, 255u,
                                    256u, 257u, 4095u, 4096u, 65535u };
    uint32_t seed = 0x2468u;
    unsigned i;

    for (i = 0; i < sizeof nts / sizeof nts[0]; i++) {
        uint16_t nt = nts[i];
        uint32_t in_len;
        int rec;

        f7_fill_ascii(f7_in, nt, seed + i);
        rec = zpse_tile_pack_ascii(f7_in, nt, f7_frame,
                                   (uint32_t)sizeof f7_frame);
        CHECK(rec > 8, "ascii pack nt=%u: rec=%d", (unsigned)nt, rec);
        if (rec <= 8) continue;
        in_len = (uint32_t)rec - 8u; /* strip the 8-byte tile record header */

        memset(f7_out, 0xCC, sizeof f7_out);
        memset(f7_ref, 0xCC, sizeof f7_ref);
        zpse_tile_ascii_unpack(f7_frame + 8u, in_len, f7_ref, nt);
        zpse_ascii_unpack_stream(f7_frame + 8u, in_len, f7_out, nt);
        CHECK(memcmp(f7_out, f7_ref, nt) == 0,
              "ascii unpack parity nt=%u", (unsigned)nt);
    }
}
#endif

/* 7. The entropy gate: high-entropy printable ASCII is eligible, but a
 *    repetitive ASCII stream is not (the LZ finder would beat the fixed
 *    7-bit pack, so FLAT7 must decline it). This positive case always runs. */
static void test_entropy_gate(void)
{
    size_t i;

    f7_fill_ascii(f7_in, F7_SMALL, 0xC0FFEEu);
    CHECK(zpse_flat7_should(f7_in, F7_SMALL, 0u, 0, sizeof f7_frame) != 0,
          "should(): high-entropy printable ASCII accepted");

    /* A four-byte cyclic pattern has almost no bigram diversity and many
     * repeated 4-grams: the exact shape the ratio-safe gate refuses. */
    for (i = 0u; i < F7_SMALL; i++) {
        f7_in[i] = (uint8_t)('a' + (i & 3u));
    }
    CHECK(zpse_flat7_should(f7_in, F7_SMALL, 0u, 0, sizeof f7_frame) == 0,
          "should(): low-entropy cyclic ASCII declined");
}

/* 8. A real all-ASCII file: the Silesia webster corpus. Its redundancy makes
 *    the block codec win by ~2.10:1, so the ratio-safe gate must DECLINE it
 *    (webster goes out as an ordinary PSE1 frame, not FLAT7). The path comes
 *    from SILESIA_DIR; the case skips cleanly when the corpus is absent. */
static void test_webster(void)
{
    const char *dir = getenv("SILESIA_DIR");
    char path[1024];
    const size_t want = 1u << 20;
    FILE *f;
    size_t nread;
    int rc;

    if (dir == NULL || dir[0] == '\0') {
        printf("FLAT7: SILESIA_DIR not set, skipping webster case\n");
        return;
    }
    if (snprintf(path, sizeof path, "%s/webster", dir) >= (int)sizeof path) {
        printf("FLAT7: SILESIA_DIR too long, skipping webster case\n");
        return;
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        printf("FLAT7: webster not found, skipping real-file case\n");
        return;
    }
    nread = fread(f7_in, 1u, want, f);
    fclose(f);
    if (nread < F7_SMALL) {
        printf("FLAT7: webster too short (%lu bytes), skipping\n",
               (unsigned long)nread);
        return;
    }

    rc = zpse_flat7_should(f7_in, nread, 0u, 0, sizeof f7_frame);
    CHECK(rc == 0, "webster: should() false, ratio-safe gate declines (%d)", rc);
}

/* 9. FLAT7 frames carry no block index, so the random-access and parallel
 *    entry points must decline them with ZPSE_UNSUPPORTED_FEATURE rather than
 *    misreading the magic as an ordinary frame. */
static void test_refusal_guards(void)
{
    const size_t n = F7_SMALL;
    size_t clen = 0;
    uint32_t olen32 = 0;
    int rc;

    CHECK(f7_make_frame(n, &clen), "frame for refusal guards");

    rc = zpse_seek_frame(f7_frame, clen, 0u, f7_out, (uint32_t)n, &olen32,
                         f7_da);
    CHECK(rc == ZPSE_UNSUPPORTED_FEATURE,
          "seek_frame on FLAT7: %s", zpse_status_name(rc));

    /* The parallel decoder is declared by the umbrella header and its FLAT7
     * guard runs before any worker setup, so no MT build is required. */
#if defined(ZPSE_PARALLEL_H)
    {
        size_t got = 0;
        void *const arenas[1] = { f7_da };

        rc = zpse_decompress_frame_parallel(f7_frame, clen, f7_out, n, &got,
                                            1u, arenas, 1u);
        CHECK(rc == ZPSE_UNSUPPORTED_FEATURE,
              "decompress_frame_parallel on FLAT7: %s", zpse_status_name(rc));
    }
#else
    printf("FLAT7: parallel decode API unavailable, skipping parallel guard\n");
#endif
}

/* 10. The indexed frame API never routes ASCII through FLAT7: the magic stays
 *    PSE1 and the frame still round-trips byte-exact. */
static void test_indexed_not_flat7(void)
{
    const size_t n = F7_SMALL;
    size_t clen = 0;
    size_t got = 0;
    int rc;

    f7_fill_ascii(f7_in, n, 0x13579u);
    rc = zpse_compress_frame_indexed(f7_in, n, f7_frame, sizeof f7_frame,
                                     &clen, F7_SMALL, 0u, f7_ea);
    CHECK(rc == ZPSE_OK, "indexed ASCII frame built: %s",
          zpse_status_name(rc));
    if (rc != ZPSE_OK) return;

    CHECK(clen >= 4u && memcmp(f7_frame, "PSE1", 4u) == 0,
          "indexed ASCII frame carries PSE1 magic (%.4s)", f7_frame);
    CHECK(clen >= 4u && memcmp(f7_frame, "F7P1", 4u) != 0,
          "indexed ASCII frame is not FLAT7 (%.4s)", f7_frame);
    CHECK(zpse_flat7_is_frame(f7_frame, clen) == 0,
          "is_frame(): indexed ASCII frame is not FLAT7");

    memset(f7_out, 0xCC, sizeof f7_out);
    rc = zpse_decompress_frame(f7_frame, clen, f7_out, n, &got, f7_da);
    CHECK(rc == ZPSE_OK && got == n && memcmp(f7_out, f7_in, n) == 0,
          "indexed ASCII frame round-trip: %s got=%lu",
          zpse_status_name(rc), (unsigned long)got);
}

#endif /* ZPSE_TEST_HAVE_FLAT7 */

int main(void)
{
#ifdef ZPSE_TEST_HAVE_FLAT7
    test_roundtrip();
    test_should_reject();
    test_is_frame();
    test_header_check();
    test_decode_no_write();
#ifdef ZPSE_HAVE_ASCII_STREAM
    test_unpack_parity();
#endif
    test_entropy_gate();
    test_webster();
    test_refusal_guards();
    test_indexed_not_flat7();

    printf("FLAT7: checks=%d fails=%d %s\n", checks, fails,
           fails == 0 ? "OK" : "FAIL");
    return fails == 0 ? 0 : 1;
#else
    printf("FLAT7: SKIPPED (no flat7.h)\n");
    return 0;
#endif
}
