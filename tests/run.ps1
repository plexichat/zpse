# Expects gcc on PATH. Run from anywhere; builds each test against
# src/zpse_unity.c and executes it. Fails on first failing suite.
#
# Also builds tests/test_parity.c, tests/test_fuzz.c and tests/test_lseed.c
# in three modes -- default (runtime SIMD dispatch), -DZPSE_FORCE_SCALAR and
# -DZPSE_FORCE_TRIAL_SWAP -- and requires every mode's digest and per-case
# output to match. The first two modes prove the SIMD paths are bit-exact
# replacements for the scalar reference; the third permutes the encoder's
# candidate-attempt order and proves order cannot change a compressed byte.
$repo = Split-Path -Parent $PSScriptRoot
$inc = Join-Path $repo "include"
$unity = Join-Path $repo "src/zpse_unity.c"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "opencode/zpse-tests"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$fail = 0
foreach ($t in @("test_roundtrip", "test_combinadic", "test_hostile", "test_units")) {
    $src = Join-Path $PSScriptRoot "$t.c"
    $exe = Join-Path $tmp "$t.exe"
    Write-Output "=== build $t ==="
    gcc -std=c11 -O2 -Wall -Wextra -I $inc $src $unity -o $exe 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED: $t"; $fail = 1; continue }
    & $exe 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Output "SUITE FAILED: $t"; $fail = 1 }
}

Write-Output "=== scalar/SIMD/order parity ==="
# `parity` compares whole compressed corpora; `fuzz` does the same over a
# randomized corpus; `lseed` decodes hand-built foreign segments whose
# ORDER tiles do not reset, against a reference that hardcodes the
# documented segment-start state. Each prints one summary line (DIGEST / FUZZ
# cases= / LSEED:) that must match across the three builds, and the rest of
# their output must match too, so a divergence names the case that caused it.
foreach ($suite in @("parity", "fuzz", "lseed")) {
    $src = Join-Path $PSScriptRoot "test_$suite.c"
    $built = $true
    foreach ($mode in @("def", "scalar", "swap")) {
        $exe = Join-Path $tmp "$suite.$mode.exe"
        Write-Output "=== build $suite ($mode) ==="
        switch ($mode) {
            "def" { gcc -std=c11 -O2 -Wall -Wextra -I $inc $src $unity -o $exe 2>&1 }
            "scalar" { gcc -std=c11 -O2 -Wall -Wextra -DZPSE_FORCE_SCALAR -I $inc $src $unity -o $exe 2>&1 }
            "swap" { gcc -std=c11 -O2 -Wall -Wextra -DZPSE_FORCE_TRIAL_SWAP -I $inc $src $unity -o $exe 2>&1 }
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

    $key = if ($suite -eq "fuzz") { '^FUZZ cases=' } elseif ($suite -eq "lseed") { '^LSEED:' } else { '^DIGEST' }
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

if ($fail -eq 0) { Write-Output "ALL SUITES PASS" } else { Write-Output "FAILURES PRESENT" }
exit $fail
