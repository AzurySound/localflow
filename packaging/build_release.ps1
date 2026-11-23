<#
.SYNOPSIS
    Builds thin standalone LocalFlow release artifacts (LocalFlow-Setup-0.2.0.exe + portable.zip).
.EXAMPLE
    .\packaging\build_release.ps1
    .\packaging\build_release.ps1 -SkipTests
#>
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PyInstaller = Join-Path $RepoRoot ".venv\Scripts\pyinstaller.exe"
$Version = (& $VenvPy -c "import localflow; print(localflow.__version__)").Trim()
Write-Host "=== LocalFlow Release Build v$Version ===" -ForegroundColor Cyan

if (-not $SkipTests) {
    Write-Host "[1/6] Running test suite..." -ForegroundColor Yellow
    & (Join-Path $RepoRoot ".venv\Scripts\pytest.exe") -q --tb=short
    if ($LASTEXITCODE -ne 0) { throw "Tests failed. Aborting release build." }
}

Write-Host "[2/6] Generating Windows .ico icon..." -ForegroundColor Yellow
& $VenvPy (Join-Path $RepoRoot "packaging\make_icon.py")
if ($LASTEXITCODE -ne 0) { throw "Icon generation failed." }

if (-not (Test-Path $PyInstaller)) {
    Write-Host "PyInstaller not found in .venv. Installing..." -ForegroundColor Yellow
    & uv pip install --python $VenvPy pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller. Run manually: uv pip install --python .venv\Scripts\python.exe pyinstaller"
    }
}

Write-Host "[3/6] Building thin PyInstaller onedir bundle..." -ForegroundColor Yellow
& $PyInstaller --clean --noconfirm (Join-Path $RepoRoot "packaging\localflow.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$DistDir = Join-Path $RepoRoot "dist\LocalFlow"
$DistExe = Join-Path $DistDir "LocalFlow.exe"
if (-not (Test-Path $DistExe)) { throw "Expected $DistExe not found!" }

Write-Host "[4/6] Auditing bundle contents & running smoke test..." -ForegroundColor Yellow
$verOut = (& $DistExe --version 2>&1) -join "`n"
Write-Host "  Smoke --version: $verOut"
$webEngine = Get-ChildItem $DistDir -Recurse -Filter "*WebEngine*" -ErrorAction SilentlyContinue
if ($webEngine) {
    Write-Warning "Found unexpected WebEngine files in bundle: $($webEngine.Name -join ', ')"
}
$vad = Get-ChildItem $DistDir -Recurse -Filter "silero_vad*.onnx" -ErrorAction SilentlyContinue
if (-not $vad) {
    throw "Missing silero_vad onnx asset in dist bundle!"
}
$distSizeMB = [math]::Round(((Get-ChildItem $DistDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "  Uncompressed dist/LocalFlow size: $distSizeMB MB"

$ReleaseDir = Join-Path $RepoRoot "release"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

Write-Host "[5/6] Packaging portable ZIP..." -ForegroundColor Yellow
$PortableZip = Join-Path $ReleaseDir "LocalFlow-$Version-portable.zip"
if (Test-Path $PortableZip) { Remove-Item $PortableZip -Force }
Compress-Archive -Path $DistDir -DestinationPath $PortableZip -CompressionLevel Optimal
$zipSizeMB = [math]::Round(((Get-Item $PortableZip).Length / 1MB), 1)
Write-Host "  Created: $PortableZip ($zipSizeMB MB)" -ForegroundColor Green

Write-Host "[6/6] Building Inno Setup installer..." -ForegroundColor Yellow
$IsccCandidates = @(
    (Get-Command iscc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe",
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
) | Where-Object { $_ -and (Test-Path $_) }

$Iscc = $IsccCandidates | Select-Object -First 1
if ($Iscc) {
    & $Iscc (Join-Path $RepoRoot "packaging\localflow.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }
    $SetupExe = Join-Path $ReleaseDir "LocalFlow-Setup-$Version.exe"
    $setupSizeMB = [math]::Round(((Get-Item $SetupExe).Length / 1MB), 1)
    Write-Host "  Created: $SetupExe ($setupSizeMB MB)" -ForegroundColor Green
} else {
    Write-Warning "Inno Setup 6 (ISCC.exe) not found. Install it with: winget install JRSoftware.InnoSetup"
    Write-Warning "Then re-run .\packaging\build_release.ps1 -SkipTests to create LocalFlow-Setup-$Version.exe."
}

Write-Host "`n=== Done ===" -ForegroundColor Cyan
