#!/usr/bin/env bash
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/trading_ecosystem}"
if systemctl list-unit-files 2>/dev/null | grep -q te-paper.service; then
  sudo systemctl restart te-paper
  sudo systemctl --no-pager --full status te-paper | head -n 20
elif [ -f "${APP_DIR}/deploy/docker-compose.yml" ]; then
  cd "${APP_DIR}"
  docker compose -f deploy/docker-compose.yml restart paper
  docker compose -f deploy/docker-compose.yml ps
else
  echo "No te-paper service on droplet"
  exit 1
fi
