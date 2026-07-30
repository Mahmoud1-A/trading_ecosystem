#!/usr/bin/env bash
set -euo pipefail
# Sync live execution sleeve from laptop -> DigitalOcean droplet.
# Discovery stays local; only strategies + prop rules are pushed.
#
# Usage:
#   export DROPLET_HOST=root@1.2.3.4   # or ubuntu@IP
#   export APP_DIR=/opt/trading_ecosystem
#   ./deploy/sync_live_book.sh
#
# Optional:
#   FREEZE=1 ./deploy/sync_live_book.sh   # re-freeze sleeve from local discovery book first

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROPLET_HOST="${DROPLET_HOST:?Set DROPLET_HOST e.g. ubuntu@203.0.113.10}"
APP_DIR="${APP_DIR:-/opt/trading_ecosystem}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new}"

cd "$ROOT_DIR"

if [[ "${FREEZE:-0}" == "1" ]]; then
  if [[ -x .venv/Scripts/python.exe ]]; then
    .venv/Scripts/python.exe -c "from trading_ecosystem.execution.paper_loop import freeze_book_to_paper_yaml; freeze_book_to_paper_yaml()"
  elif [[ -x .venv/bin/python ]]; then
    .venv/bin/python -c "from trading_ecosystem.execution.paper_loop import freeze_book_to_paper_yaml; freeze_book_to_paper_yaml()"
  else
    python -c "from trading_ecosystem.execution.paper_loop import freeze_book_to_paper_yaml; freeze_book_to_paper_yaml()"
  fi
fi

LIVE="configs/strategies.paper.live.yaml"
PROP="configs/prop_rules.yaml"
[[ -f "$LIVE" ]] || { echo "Missing $LIVE — run freeze or discovery export first"; exit 1; }
[[ -f "$PROP" ]] || { echo "Missing $PROP"; exit 1; }

echo "==> Syncing live sleeve to ${DROPLET_HOST}:${APP_DIR}"
scp $SSH_OPTS "$LIVE" "${DROPLET_HOST}:${APP_DIR}/configs/strategies.paper.live.yaml"
scp $SSH_OPTS "$PROP" "${DROPLET_HOST}:${APP_DIR}/configs/prop_rules.yaml"

# Restart paper only (monitor stays up). Prefer systemd; fall back to docker compose.
ssh $SSH_OPTS "$DROPLET_HOST" bash -s <<EOF
set -euo pipefail
APP_DIR="${APP_DIR}"
if systemctl list-unit-files | grep -q te-paper.service; then
  sudo systemctl restart te-paper
  sudo systemctl --no-pager --full status te-paper | head -n 20
elif [[ -f "\${APP_DIR}/deploy/docker-compose.yml" ]]; then
  cd "\${APP_DIR}"
  docker compose -f deploy/docker-compose.yml restart paper
  docker compose -f deploy/docker-compose.yml ps
else
  echo "No te-paper.service or docker compose found on droplet"
  exit 1
fi
EOF

echo "==> Sync complete. Discovery remains on this machine."
