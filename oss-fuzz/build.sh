#!/bin/bash -eu
# Build zpse fuzz targets for OSS-Fuzz.
#
# The fuzz target source lives in the project repo (tests/suites/fuzz_target.c)
# and is compiled here with clang, linked against libFuzzer. ASan and UBSan
# are enabled so OSS-Fuzz's cluster finds memory and UB bugs.

cd /src/zpse

# Build the unity translation unit once as an object file (no fuzzer main).
$CC $CFLAGS -std=c11 -O2 -fno-sanitize=fuzzer -Iinclude \
    -c src/zpse_unity.c -o unity.o

# Build the libFuzzer target with ASan + UBSan + libFuzzer.
$CC $CFLAGS -std=c11 -O2 -fsanitize=address,undefined -fno-omit-frame-pointer \
    -Iinclude tests/suites/fuzz_target.c unity.o \
    $LIB_FUZZING_ENGINE -o $OUT/fuzz_target
