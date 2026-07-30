#!/usr/bin/env bash
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/trading_ecosystem}"
# Prefer existing ubuntu; otherwise run services as the SSH login user (often root on DO).
if [[ -z "${APP_USER:-}" ]]; then
  if id -u ubuntu >/dev/null 2>&1; then
    APP_USER=ubuntu
  else
    APP_USER="$(id -un)"
  fi
fi
export APP_USER
cd "$APP_DIR"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip rsync curl ufw
# Trade monitor port
sudo ufw allow OpenSSH || true
sudo ufw allow 8080/tcp || true
sudo ufw --force enable || true
chmod +x deploy/vps_bootstrap.sh deploy/sync_live_book.sh
APP_DIR="$APP_DIR" APP_USER="$APP_USER" ./deploy/vps_bootstrap.sh
. .venv/bin/activate
te-ingest --start 2018-01-01
sudo systemctl restart te-paper
sudo systemctl --no-pager --full status te-monitor te-paper | head -n 40
