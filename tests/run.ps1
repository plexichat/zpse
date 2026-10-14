# Expects gcc on PATH. Run from anywhere; builds each test against
# src/zpse_unity.c and executes it. Fails on first failing suite.
#
# Also builds tests/test_parity.c twice -- default (runtime SIMD
# dispatch) and -DZPSE_FORCE_SCALAR -- and requires the two compressed
# corpora digests to match, proving the SIMD paths are bit-exact
# replacements for the scalar reference.
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

Write-Output "=== scalar/SIMD parity ==="
$parity = Join-Path $PSScriptRoot "test_parity.c"
$exeS = Join-Path $tmp "parity_simd.exe"
$exeR = Join-Path $tmp "parity_scalar.exe"
gcc -std=c11 -O2 -Wall -Wextra -I $inc $parity $unity -o $exeS 2>&1
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED: parity (simd)"; $fail = 1 }
gcc -std=c11 -O2 -Wall -Wextra -DZPSE_FORCE_SCALAR -I $inc $parity $unity -o $exeR 2>&1
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED: parity (scalar)"; $fail = 1 }
if ($fail -eq 0) {
    $outS = & $exeS 2>&1
    $rcS = $LASTEXITCODE
    $outR = & $exeR 2>&1
    $rcR = $LASTEXITCODE
    if ($rcS -ne 0) { Write-Output "SUITE FAILED: parity (simd)"; $fail = 1 }
    if ($rcR -ne 0) { Write-Output "SUITE FAILED: parity (scalar)"; $fail = 1 }
    $digS = ($outS | Select-String '^DIGEST').Line
    $digR = ($outR | Select-String '^DIGEST').Line
    $bodyS = $outS | Where-Object { $_ -notmatch '^DIGEST' }
    $bodyR = $outR | Where-Object { $_ -notmatch '^DIGEST' }
    if ($digS -ne $digR) {
        Write-Output "PARITY FAILED: digest $digS vs $digR"
        $fail = 1
    } elseif (Compare-Object $bodyS $bodyR) {
        Write-Output "PARITY FAILED: per-case output differs"
        $fail = 1
    } else {
        Write-Output "parity ok: $digS (identical streams)"
    }
}

if ($fail -eq 0) { Write-Output "ALL SUITES PASS" } else { Write-Output "FAILURES PRESENT" }
exit $fail
