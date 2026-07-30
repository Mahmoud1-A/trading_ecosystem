# Phase 12.3A — status for multi-year acquisition.

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Fw = Join-Path $RepoRoot "quant_framework"
$Workspace = Join-Path $Fw "data_import\dukascopy_archive"
$StateDir = Join-Path $Workspace "multiyear"
$StateFile = Join-Path $StateDir "launcher_state.json"
$ParentPath = Join-Path $StateDir "parent_job.json"

Write-Host "=== Multi-year acquisition status ==="
if (Test-Path $StateFile) {
    $launcher = Get-Content $StateFile -Raw | ConvertFrom-Json
    $alive = $false
    if ($launcher.pid) {
        $alive = [bool](Get-Process -Id $launcher.pid -ErrorAction SilentlyContinue)
    }
    Write-Host ("Launcher PID: {0} alive={1}" -f $launcher.pid, $alive)
    Write-Host ("Log: {0}" -f $launcher.log_path)
    Write-Host ("Dashboard started by script: {0}" -f $launcher.dashboard_started)
    Write-Host ("Alpha Miner started by script: {0}" -f $launcher.alpha_miner_started)
}

Push-Location $Fw
try {
    & $Python -m data.acquire_multiyear --years 2020 --action status --workspace $Workspace --json
}
finally {
    Pop-Location
}

if (Test-Path $ParentPath) {
    $parent = Get-Content $ParentPath -Raw | ConvertFrom-Json
    Write-Host ""
    Write-Host ("Parent: {0} state={1}" -f $parent.parent_id, $parent.state)
    foreach ($yk in ($parent.year_jobs.PSObject.Properties.Name | Sort-Object)) {
        $y = $parent.year_jobs.$yk
        $vault = if ($y.vault_locked) { "VAULT_LOCKED" } else { "OPEN" }
        Write-Host ("  {0}: state={1} chunks={2}/{3} failed={4} dataset={5} {6}" -f `
            $yk, $y.state, $y.completed_chunks, $y.total_chunks, $y.failed_chunks, $y.registered_dataset_id, $vault)
    }
    if ($parent.recent_errors -and $parent.recent_errors.Count -gt 0) {
        Write-Host "Recent errors:"
        $parent.recent_errors | Select-Object -Last 5 | ForEach-Object { Write-Host "  - $_" }
    }
    $y2025 = $parent.year_jobs.'2025'
    if ($y2025) {
        Write-Host ("2025 vault_locked flag: {0}" -f $y2025.vault_locked)
    }
}

# Memory of launcher pid if alive
if ((Test-Path $StateFile)) {
    $launcher = Get-Content $StateFile -Raw | ConvertFrom-Json
    $p = Get-Process -Id $launcher.pid -ErrorAction SilentlyContinue
    if ($p) {
        Write-Host ("Memory WS: {0:N1} MB" -f ($p.WorkingSet64 / 1MB))
    }
}
