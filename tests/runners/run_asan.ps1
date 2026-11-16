# Opt-in AddressSanitizer + UndefinedBehaviorSanitizer harness. It is NOT
# wired into the normal test runners: invoke it explicitly when you want to
# hunt memory bugs (e.g. in the new FLAT code). It builds src/zpse_unity.c
# together with each suite in tests/suites/ using
#   -fsanitize=address,undefined -fno-omit-frame-pointer -g
# and runs every resulting binary, reporting a clear pass/fail summary.
#
# Mirrors tests/runners/run.ps1: same suite list, same repo-relative paths,
# same tmp layout (a sibling "opencode/zpse-tests-asan" dir so sanitizer
# binaries do not clobber the normal ones). Only the temp dir is written;
# the repo is never modified.
#
# Expects gcc on PATH. Run from anywhere. Exits 0 only when every suite
# passes; 1 when any suite/build fails; 2 when sanitizer support is missing.

# Two levels up: this script lives in tests/runners/.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$inc = Join-Path $repo "include"
$unity = Join-Path $repo "src/zpse_unity.c"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "opencode/zpse-tests-asan"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Output "ASAN UNAVAILABLE: gcc not found on PATH."
    Write-Output "Install a gcc with ASan/UBSan support and retry."
    exit 2
}

# -fno-sanitize-recover makes a UBSan diagnostic abort (nonzero exit);
# otherwise UBSan only prints and the binary still exits 0, which would be
# reported as a false PASS. -O1 keeps stack traces more usable than -O2.
$san = @("-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g", "-fno-sanitize-recover=all")

# Respect a caller-supplied UBSAN_OPTIONS; otherwise make UBSan halt on the
# first diagnostic so it cannot be silently reported as a pass.
if (-not $env:UBSAN_OPTIONS) { $env:UBSAN_OPTIONS = "halt_on_error=1" }

# Probe once: some toolchains accept the flag but cannot link or run a
# sanitized binary (missing libasan/libubsan, unsupported platform, ...).
# Fail early with a clear message instead of a wall of misleading errors.
$probeSrc = Join-Path $tmp "asan_probe.c"
$probeExe = Join-Path $tmp "asan_probe.exe"
Set-Content -Path $probeSrc -Value "int main(void) { return 0; }"
& gcc -std=c11 -O1 @san $probeSrc -o $probeExe 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Output "ASAN UNAVAILABLE: gcc could not build a -fsanitize=address,undefined binary."
    Write-Output "Install gcc with ASan/UBSan support (libasan/libubsan) and retry."
    exit 2
}
& $probeExe 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "ASAN UNAVAILABLE: sanitized binary built but could not run (missing ASan/UBSan runtime?)."
    Write-Output "Check that the compiler's sanitizer runtime libraries are on PATH."
    exit 2
}

# Union of the suite lists parsed from run.ps1 and run.sh (run.sh is the
# superset, adding test_parallel and test_pagedict).
$suites = @("test_roundtrip", "test_transforms", "test_hostile", "test_units", "test_stride3", "test_dense", "test_decide", "test_fillpred", "test_lz", "test_lzh", "test_huf", "test_ascii", "test_levels", "test_frame", "test_parallel", "test_sizes", "test_colref", "test_colcorpus", "test_pagedict", "test_pageforms")

$pass = 0
$fail = 0
foreach ($t in $suites) {
    $src = Join-Path $repo "tests/suites/$t.c"
    $exe = Join-Path $tmp "$t.asan.exe"
    Write-Output "=== build $t (asan+ubsan) ==="
    # tests/corpus is on the include path for suites that share the sized
    # corpus header, matching the main loop of run.ps1.
    & gcc -std=c11 -O1 -Wall -Wextra @san -I $inc -I (Join-Path $repo "tests/corpus") $src $unity -o $exe 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Output "FAIL: $t (build)"
        $fail++
        continue
    }
    $out = & $exe 2>&1
    $rc = $LASTEXITCODE
    Write-Output $out
    if ($rc -eq 0) {
        Write-Output "PASS: $t"
        $pass++
    } else {
        Write-Output "FAIL: $t (exit $rc)"
        $fail++
    }
}

Write-Output "=== sanitizer summary ==="
Write-Output "passed: $pass  failed: $fail  total: $($pass + $fail)"
if ($fail -eq 0) {
    Write-Output "ALL SANITIZER SUITES PASS"
    exit 0
} else {
    Write-Output "SANITIZER FAILURES PRESENT"
    exit 1
}
