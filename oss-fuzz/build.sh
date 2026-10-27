#!/bin/bash -eu
# Build zpse fuzz targets for OSS-Fuzz.
#
# The fuzz target sources live in the project repo (tests/suites/fuzz_target.c
# and tests/suites/fuzz_decode.c) and are compiled here with clang, linked
# against libFuzzer. ASan and UBSan are enabled so OSS-Fuzz's cluster finds
# memory and UB bugs in both the encoder round-trip path and the decoder's
# error-handling / corrupt-stream paths.

cd /src/zpse

# Build the unity translation unit once as an object file (no fuzzer main).
$CC $CFLAGS -std=c11 -O2 -fno-sanitize=fuzzer -Iinclude \
    -c src/zpse_unity.c -o unity.o

# Build the libFuzzer targets with ASan + UBSan + libFuzzer.
# fuzz_target: round-trip fuzzing (compress + decompress across 9 effort levels)
$CC $CFLAGS -std=c11 -O2 -fsanitize=address,undefined -fno-omit-frame-pointer \
    -Iinclude tests/suites/fuzz_target.c unity.o \
    $LIB_FUZZING_ENGINE -o $OUT/fuzz_target

# fuzz_decode: decoder-only fuzzing (arbitrary bytes fed to decompress_block /
# decompress_frame, attacking error-handling and corrupt-stream paths)
$CC $CFLAGS -std=c11 -O2 -fsanitize=address,undefined -fno-omit-frame-pointer \
    -Iinclude tests/suites/fuzz_decode.c unity.o \
    $LIB_FUZZING_ENGINE -o $OUT/fuzz_decode
