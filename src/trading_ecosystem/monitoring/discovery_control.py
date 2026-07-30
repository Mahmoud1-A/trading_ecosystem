"""Start/stop continuous discovery from the control panel (host-local process)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import data_dir, project_root
from trading_ecosystem.discovery.universe import (
    get_active_universe,
    list_universes_public,
    load_universe,
    set_active_universe,
)
from trading_ecosystem.monitoring.state_store import MONITOR


def discovery_allowed() -> bool:
    raw = (os.getenv("ALLOW_DISCOVERY_CONTROL") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _pid_path() -> Path:
    return data_dir() / "processed" / "discovery.pid"


def _stop_flag_path() -> Path:
    return data_dir() / "processed" / "discovery_stop.flag"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def discovery_process_status() -> dict[str, Any]:
    path = _pid_path()
    pid = None
    if path.exists():
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except Exception:  # noqa: BLE001
            pid = None
    running = bool(pid and _pid_alive(pid))
    if not running and path.exists():
        try:
            path.unlink()
        except Exception:  # noqa: BLE001
            pass
        pid = None
    snap = MONITOR.snapshot().get("discovery") or {}
    return {
        "allowed": discovery_allowed(),
        "pid": pid,
        "process_running": running,
        "monitor_status": snap.get("status"),
        "universe": snap.get("universe") or get_active_universe(),
        "generation": snap.get("generation"),
        "message": snap.get("message"),
        "active_universe": get_active_universe(),
        "universes": list_universes_public(),
    }


def request_discovery_stop() -> dict[str, Any]:
    flag = _stop_flag_path()
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("stop\n", encoding="utf-8")
    pid_info = discovery_process_status()
    pid = pid_info.get("pid")
    killed = False
    if pid and _pid_alive(int(pid)):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                os.kill(int(pid), 15)
            killed = True
        except Exception:  # noqa: BLE001
            killed = False
    MONITOR.push_alert("info", "Discovery stop requested from control panel")
    return {"ok": True, "flag": str(flag), "signal_sent": killed, "pid": pid}


def start_discovery(
    *,
    universe: str | None = None,
    start: str = "2018-01-01",
) -> dict[str, Any]:
    if not discovery_allowed():
        return {
            "ok": False,
            "error": "Discovery control disabled on this host (ALLOW_DISCOVERY_CONTROL=false). "
            "Run discovery on your laptop.",
        }
    status = discovery_process_status()
    if status.get("process_running"):
        return {"ok": False, "error": f"Discovery already running (pid={status.get('pid')})"}

    uid = (universe or get_active_universe()).strip().lower()
    uni = set_active_universe(uid)

    # Best-effort ingest for the selected universe before starting.
    try:
        from trading_ecosystem.data_pipeline.provider import YFinanceProvider
        from trading_ecosystem.data_pipeline.store import ParquetBarStore

        provider = YFinanceProvider()
        store = ParquetBarStore()
        bars = provider.fetch_bars(
            list(uni["symbols"]),
            start=start,
            end=None,
            timeframe=str(uni.get("timeframe") or "1d"),
        )
        store.write_bars(bars)
    except Exception as exc:  # noqa: BLE001
        MONITOR.push_alert("warning", f"Universe ingest warning ({uid}): {exc}")

    if _stop_flag_path().exists():
        _stop_flag_path().unlink()

    root = project_root()
    venv_scripts = root / (".venv/Scripts" if sys.platform == "win32" else ".venv/bin")
    te = venv_scripts / ("te-discover.exe" if sys.platform == "win32" else "te-discover")
    if te.exists():
        cmd = [str(te), "--continuous", "--start", start, "--universe", uid]
    else:
        cmd = [
            sys.executable,
            "-c",
            (
                "from trading_ecosystem.cli import discover; "
                f"discover(['--continuous','--start','{start}','--universe','{uid}'])"
            ),
        ]

    log_path = data_dir() / "processed" / f"discovery_{uid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — kept for process lifetime
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        start_new_session=(sys.platform != "win32"),
    )
    _pid_path().write_text(str(proc.pid), encoding="utf-8")
    MONITOR.push_alert("info", f"Discovery started universe={uid} pid={proc.pid}")
    time.sleep(0.5)
    return {
        "ok": True,
        "pid": proc.pid,
        "universe": uni,
        "log": str(log_path),
        "cmd": cmd if isinstance(cmd[0], str) and not cmd[0].endswith("python") else ["te-discover", "--continuous", f"--universe={uid}"],
    }


def select_universe(universe_id: str) -> dict[str, Any]:
    status = discovery_process_status()
    if status.get("process_running"):
        return {
            "ok": False,
            "error": "Stop discovery before switching universe",
            "status": status,
        }
    info = set_active_universe(universe_id)
    MONITOR.discovery_patch(
        universe=info["id"],
        universe_label=info.get("label_en") or info["id"],
        message=f"Active universe set to {info['id']} ({info['count']} symbols)",
    )
    MONITOR.push_alert("info", f"Active discovery universe -> {info['id']}")
    return {"ok": True, "universe": info, "universes": list_universes_public()}
