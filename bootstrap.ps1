# bootstrap.ps1 - (re)create the repo venv from requirements.lock.txt, then smoke test.
# Usage (from E:\AgentWorkspaces\Tennis\tennis):  powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
#   -Unlocked : install requirements.txt instead and rewrite requirements.lock.txt (only when changing deps)
# Does not request admin / UAC. Everything stays under this folder.

param([switch]$Unlocked)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Lock = Join-Path $PSScriptRoot 'requirements.lock.txt'
$Reqs = Join-Path $PSScriptRoot 'requirements.txt'
if (-not $Unlocked -and -not (Test-Path -LiteralPath $Lock)) {
    throw "Missing lockfile: $Lock (run with -Unlocked once to create it)"
}

$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$SystemPython = 'E:\Installs\Python314\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (-not (Test-Path -LiteralPath $SystemPython)) {
        throw "Missing $SystemPython"
    }
    Write-Host "Creating .venv with $SystemPython"
    & $SystemPython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
    if (-not (Test-Path -LiteralPath $VenvPython)) { throw "venv created but $VenvPython is missing" }
}

# Keep pip cache/temp on E: (never C:)
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.venv\pip-cache'
$env:TEMP = Join-Path $PSScriptRoot '.venv\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

if ($Unlocked) {
    Write-Host "Installing $Reqs (binary wheels only)"
    & $VenvPython -m pip install --only-binary=:all: -r $Reqs
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    $Frozen = & $VenvPython -m pip freeze
    [IO.File]::WriteAllLines($Lock, $Frozen)
    Write-Host "Wrote $Lock"
} else {
    Write-Host "Installing $Lock"
    & $VenvPython -m pip install --only-binary=:all: -r $Lock
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
}

Write-Host 'Smoke test: python -m unittest discover -s tests'
& $VenvPython -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { throw "unit tests failed (exit $LASTEXITCODE)" }

Write-Host 'bootstrap ok'