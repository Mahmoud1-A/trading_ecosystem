# Phase 12.3A — graceful pause/stop. Keeps chunks, Bronze/Silver, catalog entries.

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Fw = Join-Path $RepoRoot "quant_framework"
$Workspace = Join-Path $Fw "data_import\dukascopy_archive"
$StateDir = Join-Path $Workspace "multiyear"
$StateFile = Join-Path $StateDir "launcher_state.json"

Write-Host "Requesting graceful stop (pause flag + stop flag)..."
Push-Location $Fw
try {
    & $Python -m data.acquire_multiyear --years 2020 --action stop --workspace $Workspace
}
finally {
    Pop-Location
}

if (Test-Path $StateFile) {
    $launcher = Get-Content $StateFile -Raw | ConvertFrom-Json
    if ($launcher.pid) {
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Process -Id $launcher.pid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 1
        }
        $still = Get-Process -Id $launcher.pid -ErrorAction SilentlyContinue
        if ($still) {
            Write-Host "Process still running after 60s; sending terminate (state already persisted on pause checks)."
            Stop-Process -Id $launcher.pid -ErrorAction SilentlyContinue
        } else {
            Write-Host "Process exited cleanly."
        }
    }
}
Write-Host "Valid chunks, Bronze/Silver, and catalog registrations were preserved."
Write-Host "Dashboard / Alpha Miner were not touched."
