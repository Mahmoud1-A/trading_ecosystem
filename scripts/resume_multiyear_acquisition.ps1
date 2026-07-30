# Phase 12.3A — resume the same parent / yearly jobs without duplication.

[CmdletBinding()]
param(
    [int[]]$Years = @(2020, 2021, 2022, 2023, 2025),
    [int]$MaxDownloadYears = 2,
    [int]$MaxSilverBuilds = 1,
    [double]$GlobalRateLimit = 2.0,
    [int]$DecodeWorkers = 2,
    [double]$MemoryCeilingGB = 16
)

$ErrorActionPreference = "Stop"
$StartScript = Join-Path $PSScriptRoot "start_multiyear_acquisition.ps1"
& $StartScript `
    -Years $Years `
    -MaxDownloadYears $MaxDownloadYears `
    -MaxSilverBuilds $MaxSilverBuilds `
    -GlobalRateLimit $GlobalRateLimit `
    -DecodeWorkers $DecodeWorkers `
    -MemoryCeilingGB $MemoryCeilingGB
