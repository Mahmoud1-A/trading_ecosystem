"""Start the research control plane and browser dashboard.

Usage:
    python -m control_plane

Opens nothing automatically. Browse to http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quant Framework research control plane (Phase 12)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root",
        default=None,
        help="Persistent state root (default: ./artifacts/control_plane)",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root or os.environ.get("QUANT_CONTROL_PLANE_ROOT", Path.cwd() / "artifacts" / "control_plane"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ["QUANT_CONTROL_PLANE_ROOT"] = str(root.resolve())

    from control_plane.app import create_app
    from control_plane.run_manager import RunManager

    manager = RunManager(root, max_workers=args.workers)
    app = create_app(manager, root=root)

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("QUANT FRAMEWORK — RESEARCH CONTROL PLANE (Phase 12)")
    print("RESEARCH / BACKTEST ONLY — LIVE TRADING IS NOT ENABLED")
    print(f"State root : {root.resolve()}")
    print(f"Dashboard  : {url}")
    print(f"API health : {url}/api/system/health")
    print("=" * 60)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
