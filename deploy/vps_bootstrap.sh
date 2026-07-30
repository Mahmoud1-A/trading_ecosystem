#!/usr/bin/env bash
set -euo pipefail
# Bootstrap Ubuntu VPS for trade-only stack (monitor + paper). No discovery.
# Run from the project root on the droplet (after rsync/scp of the repo).

APP_DIR=${APP_DIR:-/opt/trading_ecosystem}
# DigitalOcean Ubuntu images often only have root; prefer that over a missing ubuntu user.
if [[ -z "${APP_USER:-}" ]]; then
  if id -u ubuntu >/dev/null 2>&1; then
    APP_USER=ubuntu
  else
    APP_USER="$(id -un)"
  fi
fi

echo "==> Installing into ${APP_DIR} as ${APP_USER}"
sudo mkdir -p "$APP_DIR" "$APP_DIR/data/processed"
sudo rsync -a \
  --exclude .venv \
  --exclude .git \
  --exclude 'data/raw' \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  ./ "$APP_DIR/"

cd "$APP_DIR"
sudo chown -R "${APP_USER}:${APP_USER}" "$APP_DIR"

if [[ ! -d .venv ]]; then
  sudo -u "$APP_USER" python3 -m venv .venv
fi
sudo -u "$APP_USER" bash -lc '
  set -euo pipefail
  cd '"$APP_DIR"'
  . .venv/bin/activate
  pip install -U pip
  pip install -e .
'

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "WARNING: Edit ${APP_DIR}/.env with ALPACA_* keys before starting paper."
fi

if [[ ! -f configs/strategies.paper.live.yaml ]]; then
  if [[ -f configs/strategies.paper.yaml ]]; then
    cp configs/strategies.paper.yaml configs/strategies.paper.live.yaml
    echo "Seeded strategies.paper.live.yaml from strategies.paper.yaml"
  else
    echo "WARNING: missing configs/strategies.paper.live.yaml — sync from laptop before paper starts."
  fi
fi

sudo cp deploy/systemd/te-monitor.service /etc/systemd/system/
sudo cp deploy/systemd/te-paper.service /etc/systemd/system/
# Match service user if not ubuntu
if [[ "$APP_USER" != "ubuntu" ]]; then
  sudo sed -i "s/^User=ubuntu/User=${APP_USER}/" /etc/systemd/system/te-monitor.service
  sudo sed -i "s/^User=ubuntu/User=${APP_USER}/" /etc/systemd/system/te-paper.service
fi

sudo systemctl daemon-reload
sudo systemctl enable te-monitor te-paper
sudo systemctl restart te-monitor
sleep 2
sudo systemctl restart te-paper

echo "==> Bootstrap complete (trade-only: monitor + paper)."
echo "    Monitor:  http://$(curl -s ifconfig.me 2>/dev/null || echo DROPLET_IP):8080/health"
echo "    Status:   sudo systemctl status te-monitor te-paper"
echo "    Discovery stays on your laptop — use deploy/sync_live_book.sh to push new sleeves."
