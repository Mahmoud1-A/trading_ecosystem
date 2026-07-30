#!/usr/bin/env bash
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/trading_ecosystem}"
cd "$APP_DIR"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$(whoami)" || true
fi
sudo docker compose -f deploy/docker-compose.yml up -d --build
sudo docker compose -f deploy/docker-compose.yml --profile jobs run --rm ingest
sudo docker compose -f deploy/docker-compose.yml restart paper
sudo docker compose -f deploy/docker-compose.yml ps
