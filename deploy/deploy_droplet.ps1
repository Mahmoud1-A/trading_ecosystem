# Full trade-only deploy to a DigitalOcean Ubuntu droplet (no discovery).
# Usage:
#   $env:DROPLET_HOST = 'root@YOUR_IP'   # DO default; or ubuntu@ if that user exists
#   .\deploy\deploy_droplet.ps1

param(
    [string]$DropletHost = $env:DROPLET_HOST,
    [string]$AppDir = $(if ($env:APP_DIR) { $env:APP_DIR } else { '/opt/trading_ecosystem' }),
    [ValidateSet('systemd', 'docker')]
    [string]$Method = $(if ($env:DEPLOY_METHOD) { $env:DEPLOY_METHOD } else { 'systemd' })
)

$ErrorActionPreference = 'Stop'
if (-not $DropletHost) {
    throw 'DROPLET_HOST is not set. Example: $env:DROPLET_HOST=''root@1.2.3.4''; .\deploy\deploy_droplet.ps1'
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Live = Join-Path $Root 'configs\strategies.paper.live.yaml'
if (-not (Test-Path $Live)) {
    Write-Host 'Freezing live sleeve from local portfolio book...'
    & (Join-Path $Root '.venv\Scripts\python.exe') -c 'from trading_ecosystem.execution.paper_loop import freeze_book_to_paper_yaml; freeze_book_to_paper_yaml()'
}

if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
    throw 'tar not found. Install Git for Windows or use WSL.'
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$archive = Join-Path $env:TEMP ("te_trade_" + $stamp + '.tgz')

Write-Host '==> Creating deploy archive...'
& tar -czf $archive --exclude=.venv --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache --exclude=data/raw --exclude=.env -C $Root .

Write-Host ('==> Uploading to ' + $DropletHost + ' ...')
ssh -o StrictHostKeyChecking=accept-new $DropletHost ('sudo mkdir -p ' + $AppDir)
ssh $DropletHost ('sudo chown -R $(whoami):$(whoami) ' + $AppDir)
scp -o StrictHostKeyChecking=accept-new $archive ($DropletHost + ':/tmp/te_trade.tgz')
ssh $DropletHost ('tar -xzf /tmp/te_trade.tgz -C ' + $AppDir + '; rm -f /tmp/te_trade.tgz')

$envFile = Join-Path $Root '.env'
if (Test-Path $envFile) {
    Write-Host '==> Uploading .env ...'
    scp $envFile ($DropletHost + ':' + $AppDir + '/.env')
} else {
    Write-Host 'WARNING: no local .env - seeding example on server'
    ssh $DropletHost ('cd ' + $AppDir + '; if [ ! -f .env ]; then cp .env.example .env; fi')
}

$remoteName = if ($Method -eq 'docker') { 'remote_docker.sh' } else { 'remote_systemd.sh' }
scp (Join-Path $Root ('deploy\' + $remoteName)) ($DropletHost + ':/tmp/te_remote.sh')

Write-Host ('==> Running remote bootstrap (' + $Method + ')...')
ssh $DropletHost ('chmod +x /tmp/te_remote.sh; APP_DIR=' + $AppDir + ' bash /tmp/te_remote.sh; rm -f /tmp/te_remote.sh')

$ip = ($DropletHost -split '@')[-1]
Write-Host ('==> Health check http://' + $ip + ':8080/health')
Start-Sleep -Seconds 5
try {
    $h = Invoke-RestMethod ('http://' + $ip + ':8080/health') -TimeoutSec 20
    Write-Host ('OK: ' + ($h | ConvertTo-Json -Compress))
} catch {
    Write-Host ('Health not reachable yet (check firewall). ' + $_.Exception.Message)
    Write-Host 'On droplet: sudo systemctl status te-monitor te-paper'
}

Remove-Item $archive -ErrorAction SilentlyContinue
Write-Host ''
Write-Host 'Deploy finished (trade-only).'
Write-Host ('  Monitor: http://' + $ip + ':8080')
Write-Host ('  Sync later: $env:DROPLET_HOST=''' + $DropletHost + '''; .\deploy\sync_live_book.ps1 -Freeze')
Write-Host '  Discovery remains on this laptop.'
