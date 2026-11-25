/* Shared non-temporal decode output for opt-in streaming callers.
 *
 * This is the mechanism shipped in f8447d1 for the RAW/no-CRC copy, lifted
 * verbatim so every decode path that writes its output exactly once, in one
 * contiguous forward pass with no read of those bytes before it completes,
 * can use the SAME hint, cutoff and hot-destination guard. Nothing here is a
 * new flag: ZPSE_DEC_STREAMING_OUTPUT and ZPSE_BLOCK_NT_MIN keep the meanings
 * and values they have in constants.h / block.c.
 *
 * The guard is shared across paths on purpose. A caller that decodes
 * successive blocks into the SAME address has a hot destination whichever
 * form those blocks used, so one per-thread "last output" slot covers them
 * all; a streaming caller that rotates buffers sees a new address and gets
 * the NT path. In the unity build (how the library is compiled) this header
 * is included once per translation unit, so there is exactly one slot.
 *
 * FORCE_SCALAR keeps every path on its original write-back store. */
#ifndef ZPSE_NTOUT_H
#define ZPSE_NTOUT_H

#include "zpse/constants.h"
#include "zpse/types.h"

#include <stdint.h>
#include <string.h>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86)) && !defined(ZPSE_FORCE_SCALAR)
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define ZPSE_NTOUT_HAVE 1
#include <emmintrin.h>
#endif
#endif
#ifndef ZPSE_NTOUT_HAVE
#define ZPSE_NTOUT_HAVE 0
#endif
/* Each including TU uses a subset of the helpers (a flat decoder never fills
 * a token page), so silence the unused-static warning without weakening -Wall. */
#if defined(__GNUC__) || defined(__clang__)
#define ZPSE_NT_UNUSED __attribute__((unused))
#else
#define ZPSE_NT_UNUSED
#endif

/* Same measured cutoff text.c uses for its NT unpack, and the one block.c
 * already had: below it the sfence costs more than the RFO it avoids. */
#define ZPSE_NT_MIN 4096u

#if ZPSE_NTOUT_HAVE
#if defined(_MSC_VER)
#define ZPSE_NT_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define ZPSE_NT_TLS _Thread_local
#else
#define ZPSE_NT_TLS
#endif
static ZPSE_NT_TLS const uint8_t *zpse_ntout_last_out;

/* The exact gate: caller hint AND a block big enough to pay for the fence AND
 * a destination not just written by this thread. Always call this, even in
 * scalar builds, so the guard is updated identically on every build. */
static ZPSE_NT_UNUSED int zpse_ntout_ok(uint32_t dec_flags, const uint8_t *dst, uint32_t n)
{
    int use_nt = (dec_flags & (uint32_t)ZPSE_DEC_STREAMING_OUTPUT) != 0u &&
                 n >= (uint32_t)ZPSE_NT_MIN &&
                 dst != zpse_ntout_last_out;

    zpse_ntout_last_out = dst;
    return use_nt;
}

/* Byte-identical memcpy replacement for an arbitrary destination alignment.
 * Advance to the next 64-byte boundary with a scalar head, emit only whole
 * cache lines with four 16-byte streaming stores each, then a scalar tail.
 * The head/tail lines are distinct from the streaming lines, so the sfence
 * both drains the write-combining buffers and keeps a write-back tail from
 * resurrecting stale bytes in a line with pending NT stores. */
static ZPSE_NT_UNUSED void zpse_ntout_copy(uint8_t *dst, const uint8_t *src, uint32_t n)
{
    uintptr_t d = (uintptr_t)(void *)dst;
    uint32_t head = (uint32_t)((64u - (d & (uintptr_t)63u)) & (uintptr_t)63u);
    uint32_t body, lines, i;
    const uint8_t *s;
    uint8_t *p;

    if (head > n) {
        head = n;
    }
    if (head != 0u) {
        memcpy(dst, src, head);
    }
    body = n - head;
    lines = body & ~((uint32_t)63u);
    s = src + head;
    p = dst + head;
    for (i = 0u; i < lines; i += 64u) {
        __m128i v0 = _mm_loadu_si128((const __m128i *)(const void *)(s + i));
        __m128i v1 = _mm_loadu_si128((const __m128i *)(const void *)(s + i + 16u));
        __m128i v2 = _mm_loadu_si128((const __m128i *)(const void *)(s + i + 32u));
        __m128i v3 = _mm_loadu_si128((const __m128i *)(const void *)(s + i + 48u));

        _mm_stream_si128((__m128i *)(void *)(p + i), v0);
        _mm_stream_si128((__m128i *)(void *)(p + i + 16u), v1);
        _mm_stream_si128((__m128i *)(void *)(p + i + 32u), v2);
        _mm_stream_si128((__m128i *)(void *)(p + i + 48u), v3);
    }
    _mm_sfence();
    if (lines != body) {
        memcpy(p + lines, s + lines, body - lines);
    }
}

/* NT memset: the one-value page token writes n bytes of a single value. Same
 * head/line/tail shape, with a broadcast register instead of a source load. */
static ZPSE_NT_UNUSED void zpse_ntout_fill(uint8_t *dst, uint8_t value, uint32_t n)
{
    uintptr_t d = (uintptr_t)(void *)dst;
    uint32_t head = (uint32_t)((64u - (d & (uintptr_t)63u)) & (uintptr_t)63u);
    uint32_t body, lines, i;
    const __m128i v = _mm_set1_epi8((char)value);
    uint8_t *p;

    if (head > n) {
        head = n;
    }
    if (head != 0u) {
        memset(dst, value, head);
    }
    body = n - head;
    lines = body & ~((uint32_t)63u);
    p = dst + head;
    for (i = 0u; i < lines; i += 64u) {
        _mm_stream_si128((__m128i *)(void *)(p + i), v);
        _mm_stream_si128((__m128i *)(void *)(p + i + 16u), v);
        _mm_stream_si128((__m128i *)(void *)(p + i + 32u), v);
        _mm_stream_si128((__m128i *)(void *)(p + i + 48u), v);
    }
    _mm_sfence();
    if (lines != body) {
        memset(p + lines, value, body - lines);
    }
}
#else /* !ZPSE_NTOUT_HAVE */
static ZPSE_NT_UNUSED int zpse_ntout_ok(uint32_t dec_flags, const uint8_t *dst, uint32_t n)
{
    (void)dec_flags;
    (void)dst;
    (void)n;
    return 0;
}
/* No NT path exists here, so the helpers are never reached; keep their
 * write-back bodies so callers compile identically in every build. */
static ZPSE_NT_UNUSED void zpse_ntout_copy(uint8_t *dst, const uint8_t *src, uint32_t n)
{
    memcpy(dst, src, n);
}
static ZPSE_NT_UNUSED void zpse_ntout_fill(uint8_t *dst, uint8_t value, uint32_t n)
{
    memset(dst, value, n);
}
#endif /* ZPSE_NTOUT_HAVE */

#endif /* ZPSE_NTOUT_H */



