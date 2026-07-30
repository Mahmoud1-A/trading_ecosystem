"""Trade-path control helpers (start/stop paper, status, logs). Discovery is out of scope."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, load_yaml


PAPER_UNIT = "te-paper"
MONITOR_UNIT = "te-monitor"
ALLOWED_ACTIONS = frozenset({"start", "stop", "restart"})


def control_token() -> str:
    return (os.getenv("CONTROL_TOKEN") or "").strip()


def token_ok(provided: str | None) -> bool:
    expected = control_token()
    if not expected:
        return False
    return bool(provided) and provided.strip() == expected


def systemd_available() -> bool:
    return platform.system() == "Linux" and shutil.which("systemctl") is not None


def _systemctl(*args: str, timeout: float = 20.0) -> tuple[int, str]:
    cmd = ["systemctl", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def unit_status(unit: str) -> dict[str, Any]:
    if not systemd_available():
        return {
            "unit": unit,
            "available": False,
            "active": "unavailable",
            "detail": "systemctl only on Linux (Droplet). On Windows use CLI te-paper.",
        }
    code, active = _systemctl("is-active", unit)
    _c2, enabled = _systemctl("is-enabled", unit)
    _c3, show = _systemctl(
        "show",
        unit,
        "-p",
        "ActiveEnterTimestamp",
        "-p",
        "MainPID",
        "-p",
        "SubState",
        "-p",
        "Description",
        "--no-pager",
    )
    meta: dict[str, str] = {}
    for line in show.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    return {
        "unit": unit,
        "available": True,
        "active": active or "unknown",
        "running": active == "active",
        "enabled": enabled,
        "main_pid": meta.get("MainPID"),
        "sub_state": meta.get("SubState"),
        "since": meta.get("ActiveEnterTimestamp"),
        "description": meta.get("Description"),
        "ok": code == 0 or active in {"active", "inactive", "failed"},
    }


def paper_action(action: str) -> dict[str, Any]:
    action = action.strip().lower()
    if action not in ALLOWED_ACTIONS:
        return {"ok": False, "error": f"unsupported action: {action}"}
    if not systemd_available():
        return {
            "ok": False,
            "error": "Start/stop via panel requires systemd (Droplet). "
            "Locally run: te-paper --broker alpaca --loop ...",
        }
    code, out = _systemctl(action, PAPER_UNIT)
    status = unit_status(PAPER_UNIT)
    return {
        "ok": code == 0,
        "action": action,
        "output": out,
        "status": status,
    }


def paper_logs(lines: int = 80) -> dict[str, Any]:
    lines = max(10, min(int(lines), 300))
    if not systemd_available():
        return {
            "ok": True,
            "lines": [
                "Logs via journalctl are only available on the Droplet.",
                "Locally watch the te-paper terminal output.",
            ],
            "source": "none",
        }
    try:
        proc = subprocess.run(
            [
                "journalctl",
                "-u",
                PAPER_UNIT,
                "-n",
                str(lines),
                "--no-pager",
                "-o",
                "short-iso",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        text = (proc.stdout or proc.stderr or "").strip()
        return {
            "ok": proc.returncode == 0,
            "lines": text.splitlines() if text else ["(no log lines)"],
            "source": "journalctl",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "lines": [str(exc)], "source": "error"}


def load_prop_policy() -> dict[str, Any]:
    path = CONFIG_DIR / "prop_rules.yaml"
    rules = load_yaml(path)
    from trading_ecosystem.execution.costs import load_execution_costs
    from trading_ecosystem.discovery.universe import get_active_universe, list_universes_public

    costs = {
        u["id"]: load_execution_costs(universe_id=u["id"]).as_dict()
        for u in list_universes_public()
    }
    return {
        "path": str(path),
        "rules": rules,
        "execution_costs": costs,
        "active_universe": get_active_universe(),
        "summary": [
            {"key": "mode", "label": "Mode", "value": rules.get("mode")},
            {
                "key": "starting_equity",
                "label": "Starting equity",
                "value": rules.get("starting_equity"),
            },
            {
                "key": "max_daily_drawdown_pct",
                "label": "Max daily DD %",
                "value": rules.get("max_daily_drawdown_pct"),
            },
            {
                "key": "max_total_drawdown_pct",
                "label": "Max total DD %",
                "value": rules.get("max_total_drawdown_pct"),
            },
            {
                "key": "trailing_total_dd",
                "label": "Trailing total DD",
                "value": rules.get("trailing_total_dd"),
            },
            {
                "key": "risk_per_trade_pct",
                "label": "Risk per trade %",
                "value": rules.get("risk_per_trade_pct"),
            },
            {
                "key": "max_symbol_exposure_pct",
                "label": "Max symbol exposure %",
                "value": rules.get("max_symbol_exposure_pct"),
            },
            {
                "key": "max_gross_exposure_pct",
                "label": "Max gross exposure %",
                "value": rules.get("max_gross_exposure_pct"),
            },
            {
                "key": "max_concurrent_positions",
                "label": "Max concurrent positions",
                "value": rules.get("max_concurrent_positions"),
            },
            {
                "key": "flatten_on_daily_breach",
                "label": "Flatten on daily breach",
                "value": rules.get("flatten_on_daily_breach"),
            },
            {
                "key": "halt_on_total_breach",
                "label": "Halt on total breach",
                "value": rules.get("halt_on_total_breach"),
            },
            {
                "key": "news_block_minutes_before",
                "label": "News block before (min)",
                "value": rules.get("news_block_minutes_before"),
            },
            {
                "key": "news_block_minutes_after",
                "label": "News block after (min)",
                "value": rules.get("news_block_minutes_after"),
            },
        ],
    }


def live_sleeve_info() -> dict[str, Any]:
    path = CONFIG_DIR / "strategies.paper.live.yaml"
    if not path.exists():
        return {"exists": False, "path": str(path), "count": 0}
    try:
        data = load_yaml(path)
        strategies = data.get("strategies") or []
        return {
            "exists": True,
            "path": str(path),
            "count": len(strategies),
            "ids": [
                s.get("strategy_id") or s.get("id") or s.get("class_name")
                for s in strategies[:40]
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "path": str(path), "count": 0, "error": str(exc)}
