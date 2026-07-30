# Sync live execution sleeve from Windows laptop -> DigitalOcean droplet.
# Discovery stays local.
#
# Usage:
#   $env:DROPLET_HOST = 'ubuntu@1.2.3.4'
#   .\deploy\sync_live_book.ps1
#   .\deploy\sync_live_book.ps1 -Freeze

param(
    [switch]$Freeze,
    [string]$DropletHost = $env:DROPLET_HOST,
    [string]$AppDir = $(if ($env:APP_DIR) { $env:APP_DIR } else { '/opt/trading_ecosystem' })
)

$ErrorActionPreference = 'Stop'
if (-not $DropletHost) {
    throw 'Set DROPLET_HOST e.g. ubuntu@203.0.113.10'
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($Freeze) {
    & (Join-Path $Root '.venv\Scripts\python.exe') -c 'from trading_ecosystem.execution.paper_loop import freeze_book_to_paper_yaml; freeze_book_to_paper_yaml()'
}

$Live = Join-Path $Root 'configs\strategies.paper.live.yaml'
$Prop = Join-Path $Root 'configs\prop_rules.yaml'
if (-not (Test-Path $Live)) { throw ('Missing ' + $Live) }
if (-not (Test-Path $Prop)) { throw ('Missing ' + $Prop) }

Write-Host ('==> Syncing to ' + $DropletHost + ':' + $AppDir)
scp $Live ($DropletHost + ':' + $AppDir + '/configs/strategies.paper.live.yaml')
scp $Prop ($DropletHost + ':' + $AppDir + '/configs/prop_rules.yaml')
scp (Join-Path $Root 'deploy\remote_restart_paper.sh') ($DropletHost + ':/tmp/te_restart_paper.sh')
ssh $DropletHost ('chmod +x /tmp/te_restart_paper.sh; APP_DIR=' + $AppDir + ' bash /tmp/te_restart_paper.sh; rm -f /tmp/te_restart_paper.sh')
Write-Host '==> Sync complete. Discovery remains on this machine.'
