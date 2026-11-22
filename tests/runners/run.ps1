# Expects gcc on PATH. Run from anywhere; builds each test against
# src/zpse_unity.c and executes it. Fails on first failing suite.
#
# Also builds tests/test_parity.c and tests/test_fuzz.c
# in three modes -- default (runtime SIMD dispatch), -DZPSE_FORCE_SCALAR and
# -DZPSE_FORCE_TRIAL_SWAP -- and requires every mode's digest and per-case
# output to match. The first two modes prove the SIMD paths are bit-exact
# replacements for the scalar reference; the third permutes the encoder's
# candidate-attempt order and proves order cannot change a compressed byte.
# Two levels up: this script lives in tests/runners/.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$inc = Join-Path $repo "include"
$unity = Join-Path $repo "src/zpse_unity.c"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "opencode/zpse-tests"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$fail = 0
foreach ($t in @("test_roundtrip", "test_transforms", "test_hostile", "test_units", "test_stride3", "test_dense", "test_decide", "test_fillpred", "test_lz", "test_huf", "test_ascii", "test_levels", "test_frame", "test_parallel", "test_sizes", "test_colref", "test_colcorpus", "test_pagedict", "test_pageforms", "test_flat7")) {
    $src = Join-Path $repo "tests/suites/$t.c"
    $exe = Join-Path $tmp "$t.exe"
    Write-Output "=== build $t ==="
    gcc -std=c11 -O2 -Wall -Wextra -I $inc -I (Join-Path $repo "tests/corpus") $src $unity -o $exe 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED: $t"; $fail = 1; continue }
    & $exe 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Output "SUITE FAILED: $t"; $fail = 1 }
}

Write-Output "=== scalar/SIMD/order parity ==="
# `parity` compares whole compressed corpora; `fuzz` does the same over a
# randomized corpus. Each prints one summary line (DIGEST / FUZZ cases=)
# that must match across the three builds, and the rest of their output must
# match too, so a divergence names the case that caused it.
foreach ($suite in @("parity", "fuzz")) {
    $src = Join-Path $repo "tests/suites/test_$suite.c"
    $built = $true
    foreach ($mode in @("def", "scalar", "swap")) {
        $exe = Join-Path $tmp "$suite.$mode.exe"
        Write-Output "=== build $suite ($mode) ==="
        switch ($mode) {
            "def" { gcc -std=c11 -O2 -Wall -Wextra -I $inc -I (Join-Path $repo "tests/corpus") $src $unity -o $exe 2>&1 }
            "scalar" { gcc -std=c11 -O2 -Wall -Wextra -DZPSE_FORCE_SCALAR -I $inc -I (Join-Path $repo "tests/corpus") $src $unity -o $exe 2>&1 }
            "swap" { gcc -std=c11 -O2 -Wall -Wextra -DZPSE_FORCE_TRIAL_SWAP -I $inc -I (Join-Path $repo "tests/corpus") $src $unity -o $exe 2>&1 }
        }
        if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED: $suite ($mode)"; $fail = 1; $built = $false }
    }
    if (-not $built) { continue }

    $logs = @{}
    foreach ($mode in @("def", "scalar", "swap")) {
        $out = & (Join-Path $tmp "$suite.$mode.exe") 2>&1
        $rc = $LASTEXITCODE
        $logs[$mode] = $out
        if ($rc -ne 0) { Write-Output "SUITE FAILED: $suite ($mode)"; $fail = 1 }
    }

    $key = if ($suite -eq "fuzz") { '^FUZZ cases=' } else { '^DIGEST' }
    $digDef = ($logs["def"] | Select-String $key).Line
    $bodyDef = $logs["def"] | Where-Object { $_ -notmatch $key }
    foreach ($mode in @("scalar", "swap")) {
        $dig = ($logs[$mode] | Select-String $key).Line
        $body = $logs[$mode] | Where-Object { $_ -notmatch $key }
        if ($dig -ne $digDef) {
            Write-Output "$suite FAILED: digest $digDef vs $mode $dig"
            $fail = 1
        } elseif (Compare-Object $bodyDef $body) {
            Write-Output "$suite FAILED: per-case output differs in $mode"
            $fail = 1
        } else {
            Write-Output "$suite ok ($mode): $digDef identical streams"
        }
    }
}

Write-Output "=== page kernels without AVX2 (ZPSE_AVX2_ABI_OK=0) ==="
# The page kernels have AVX2 and scalar/SSE2 twins (see hot.h and the page
# kernel sources). Build test_pageforms with the kernels compiled out and
# require every byte of its output to match the default build, so the SIMD
# path cannot drift from the scalar reference on the new page forms. This is
# the same kind of guard the parity/fuzz loop applies to the older
# kernels; it is kept separate so those two digest checks stay unchanged.
$pfSrc = Join-Path $repo "tests/suites/test_pageforms.c"
$pfBuilt = $true
$pfLogs = @{}
foreach ($mode in @("def", "noavx")) {
    $exe = Join-Path $tmp "pageforms.$mode.exe"
    Write-Output "=== build pageforms ($mode) ==="
    switch ($mode) {
        "def" { gcc -std=c11 -O2 -Wall -Wextra -I $inc -I (Join-Path $repo "tests/corpus") $pfSrc $unity -o $exe 2>&1 }
        "noavx" { gcc -std=c11 -O2 -Wall -Wextra -DZPSE_AVX2_ABI_OK=0 -I $inc -I (Join-Path $repo "tests/corpus") $pfSrc $unity -o $exe 2>&1 }
    }
    if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED: pageforms ($mode)"; $fail = 1; $pfBuilt = $false }
}
if ($pfBuilt) {
    foreach ($mode in @("def", "noavx")) {
        $out = & (Join-Path $tmp "pageforms.$mode.exe") 2>&1
        $rc = $LASTEXITCODE
        $pfLogs[$mode] = $out
        if ($rc -ne 0) { Write-Output "SUITE FAILED: pageforms ($mode)"; $fail = 1 }
    }
    if (Compare-Object $pfLogs["def"] $pfLogs["noavx"]) {
        Write-Output "pageforms FAILED: output differs with ZPSE_AVX2_ABI_OK=0"
        $fail = 1
    } else {
        Write-Output "pageforms ok (noavx): $($pfLogs['def']) identical streams"
    }
}

if ($fail -eq 0) { Write-Output "ALL SUITES PASS" } else { Write-Output "FAILURES PRESENT" }
exit $fail
