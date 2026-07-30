# Phase 12.3A — start acquisition-only multi-year Dukascopy download.
# Does NOT start Dashboard or Alpha Miner.

[CmdletBinding()]
param(
    [Parameter()]
    [object]$Years = @(2020, 2021, 2022, 2023, 2025),
    [int]$MaxDownloadYears = 2,
    [int]$MaxSilverBuilds = 1,
    [double]$GlobalRateLimit = 2.0,
    [int]$DecodeWorkers = 2,
    [double]$MemoryCeilingGB = 16
)

$ErrorActionPreference = "Stop"
# Normalize years from int[], object[], or "2020,2021,..." string ( -File CLI quirk ).
if ($Years -is [string]) {
    $Years = @($Years -split '[,;\s]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
} else {
    $Years = @($Years | ForEach-Object { [int]$_ })
}
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Fw = Join-Path $RepoRoot "quant_framework"
$Workspace = Join-Path $Fw "data_import\dukascopy_archive"
$StateDir = Join-Path $Workspace "multiyear"
$LogDir = Join-Path $StateDir "logs"
$StateFile = Join-Path $StateDir "launcher_state.json"

if (-not (Test-Path $Python)) {
    throw "Virtual environment python not found: $Python"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Refuse if launcher already has a live PID
if (Test-Path $StateFile) {
    $prev = Get-Content $StateFile -Raw | ConvertFrom-Json
    if ($prev.pid -and (Get-Process -Id $prev.pid -ErrorAction SilentlyContinue)) {
        Write-Host "Acquisition already running (pid=$($prev.pid)). Use status_multiyear_acquisition.ps1"
        Write-Host "State: $StateFile"
        Write-Host "Log:   $($prev.log_path)"
        exit 0
    }
}

# Verify protected 2024 / March identities before launch
Push-Location $Fw
try {
    & $Python -m data.acquire_multiyear --years @Years --action verify-protected --workspace $Workspace
    if ($LASTEXITCODE -ne 0) { throw "Protected dataset verification failed" }
}
finally {
    Pop-Location
}

$env:QUANT_DUKASCOPY_ENABLE_DOWNLOAD = "1"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "multiyear_$stamp.log"
$YearArgs = ($Years | ForEach-Object { "$_" }) -join " "

$argList = @(
    "-u", "-m", "data.acquire_multiyear",
    "--symbol", "USA500IDXUSD",
    "--years"
) + ($Years | ForEach-Object { "$_" }) + @(
    "--stage", "STAGE_3_HISTORICAL_MULTIYEAR",
    "--allow-network",
    "--resume",
    "--max-concurrent-download-years", "$MaxDownloadYears",
    "--max-concurrent-silver-builds", "$MaxSilverBuilds",
    "--global-rate-limit", "$GlobalRateLimit",
    "--decode-workers", "$DecodeWorkers",
    "--memory-ceiling-gb", "$MemoryCeilingGB",
    "--workspace", $Workspace,
    "--json"
)

Write-Host "Launching multi-year acquisition (no Dashboard / no Alpha Miner)"
Write-Host "Repo:      $RepoRoot"
Write-Host "Python:    $Python"
Write-Host "Workspace: $Workspace"
Write-Host "Years:     $YearArgs"
Write-Host "Log:       $LogPath"

$proc = Start-Process -FilePath $Python `
    -ArgumentList $argList `
    -WorkingDirectory $Fw `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError (Join-Path $LogDir "multiyear_$stamp.err.log") `
    -PassThru `
    -WindowStyle Hidden

# Wait briefly for parent_job.json
$parentPath = Join-Path $StateDir "parent_job.json"
$deadline = (Get-Date).AddSeconds(30)
while (-not (Test-Path $parentPath) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 200
}

$parentId = $null
$yearJobs = @{}
if (Test-Path $parentPath) {
    $parent = Get-Content $parentPath -Raw | ConvertFrom-Json
    $parentId = $parent.parent_id
    $yearJobs = $parent.year_jobs
}

$launcher = [ordered]@{
    pid              = $proc.Id
    parent_id        = $parentId
    years            = $Years
    year_jobs        = $yearJobs
    log_path         = $LogPath
    err_log_path     = (Join-Path $LogDir "multiyear_$stamp.err.log")
    parent_state     = $parentPath
    status_script    = (Join-Path $PSScriptRoot "status_multiyear_acquisition.ps1")
    started_at       = (Get-Date).ToString("o")
    dashboard_started = $false
    alpha_miner_started = $false
}
$launcher | ConvertTo-Json -Depth 8 | Set-Content $StateFile -Encoding utf8

Write-Host "PID:       $($proc.Id)"
Write-Host "Parent ID: $parentId"
Write-Host "State:     $StateFile"
Write-Host "Parent:    $parentPath"
exit 0
