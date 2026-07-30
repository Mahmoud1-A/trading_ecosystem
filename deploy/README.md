# DigitalOcean trade deploy (no discovery)

Runs **monitor + paper/live execution under Master Risk Manager** on a VPS.  
`te-discover` stays on your laptop.

## Droplet size

- Ubuntu 22.04/24.04
- **4GB RAM / 2 vCPU** recommended for monitor + Alpaca paper loop
- Open inbound **TCP 8080** (DigitalOcean Cloud Firewall or `ufw allow 8080`)

## One-shot deploy from Windows

```powershell
$env:DROPLET_HOST = "root@YOUR_DROPLET_IP"   # DO Ubuntu often uses root; ubuntu@ if that user exists
.\deploy\deploy_droplet.ps1
```

Uses systemd by default. For Docker:

```powershell
$env:DEPLOY_METHOD = "docker"
.\deploy\deploy_droplet.ps1
```

## Sync a new live sleeve (after local discovery improves the book)

```powershell
$env:DROPLET_HOST = "root@YOUR_DROPLET_IP"   # DO Ubuntu often uses root; ubuntu@ if that user exists
.\deploy\sync_live_book.ps1 -Freeze
```

## Manual bootstrap on the server

```bash
cd /opt/trading_ecosystem   # after uploading the repo
chmod +x deploy/vps_bootstrap.sh
./deploy/vps_bootstrap.sh
sudo systemctl status te-monitor te-paper
curl -fsS http://127.0.0.1:8080/health
```

## Control panel

Open **http://YOUR_IP:8080/control**

- Start / stop / restart `te-paper` (trade path only)
- Live equity, DD, MRM decisions, Prop policy, journal logs
- Set `CONTROL_TOKEN` in `.env` (same value locally and on the droplet), paste it once in the panel

## Env on the server

Copy keys into `/opt/trading_ecosystem/.env` (never commit):

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_PAPER=true`
- `CONTROL_TOKEN` (for the control panel write actions)

## What is NOT deployed

- Continuous discovery / evolution workers
- Local `.venv`
