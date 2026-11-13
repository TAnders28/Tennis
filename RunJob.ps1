# RunJob.ps1 - run a repo Python module in the background with a traceable log.
# Usage: powershell -ExecutionPolicy Bypass -File .\RunJob.ps1 -Name pull_sackmann -Module DataPull.PullSackmann -ModuleArgs "--years 2000 2026"
# Log: logs\<YYYY-MM-DD>_<Name>.log (UTF-8); first line = python --version | git short HEAD | args. Last line = exit code.
param([Parameter(Mandatory=$true)][string]$Name, [Parameter(Mandatory=$true)][string]$Module, [string]$ModuleArgs = '')
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$Log = Join-Path $PSScriptRoot ("logs\{0}_{1}.log" -f (Get-Date -Format 'yyyy-MM-dd'), $Name)
$PyVer = (& $Python --version 2>&1) -join ' '
$Head = (& git rev-parse --short HEAD 2>$null); if (-not $Head) { $Head = 'no-commit' }
$Dirty = if (& git status --porcelain 2>$null) { '+dirty' } else { '' }
[IO.File]::WriteAllText($Log, "$PyVer | git $Head$Dirty | -m $Module $ModuleArgs | started $(Get-Date -Format o)`r`n")
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TEMP = Join-Path $PSScriptRoot '.venv\tmp'; $env:TMP = $env:TEMP
$CmdLine = "/c `"`"$Python`" -m $Module $ModuleArgs >> `"$Log`" 2>&1 && (echo exit 0 >> `"$Log`") || (echo exit FAILED >> `"$Log`")`""
$Proc = Start-Process cmd.exe -ArgumentList $CmdLine -WindowStyle Hidden -PassThru
try { $Proc.PriorityClass = 'BelowNormal' } catch {}
Write-Host "pid $($Proc.Id) log $Log"