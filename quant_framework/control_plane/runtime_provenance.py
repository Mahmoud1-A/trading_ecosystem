"""Runtime provenance for Alpha Miner / control-plane runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_SIGNAL_SOURCE = "candidate_dsl_trees"


def _git_sha(cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def collect_runtime_provenance(*, repo_hint: Path | None = None) -> dict[str, Any]:
    """Capture executable identity so UI/report can prove which code is running.

    Always records ``repository_git_sha`` (exact commit) for audit. Separately
    records ``execution_semantic_hash`` (allowlisted evaluation-critical modules)
    used for search resume compatibility.
    """
    cwd = Path.cwd().resolve()
    repo = (repo_hint or cwd).resolve()
    import control_plane.jobs as jobs_mod
    import control_plane.alpha_results as alpha_mod
    import discovery.event_wfo_backend as backend_mod
    from discovery.execution_semantic_hash import (
        compute_execution_semantic_hash,
        resolve_repo_root,
    )

    repo_root = resolve_repo_root(repo)
    repository_git_sha = _git_sha(repo_root)
    try:
        execution_semantic_hash = compute_execution_semantic_hash(repo_root)
    except Exception:  # noqa: BLE001
        execution_semantic_hash = "UNKNOWN"

    return {
        "git_commit_sha": repository_git_sha,
        "repository_git_sha": repository_git_sha,
        "execution_semantic_hash": execution_semantic_hash,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": str(cwd),
        "module_paths": {
            "jobs.py": str(Path(jobs_mod.__file__).resolve()),
            "event_wfo_backend.py": str(Path(backend_mod.__file__).resolve()),
            "alpha_results.py": str(Path(alpha_mod.__file__).resolve()),
        },
        "sys_path_head": [str(Path(p).resolve()) for p in sys.path[:8] if p],
        "pid": os.getpid(),
        "required_signal_source": REQUIRED_SIGNAL_SOURCE,
    }


def assert_dsl_signal_source(signal_source: str | None, *, where: str) -> None:
    src = str(signal_source or "")
    if src != REQUIRED_SIGNAL_SOURCE:
        raise RuntimeError(
            f"SIGNAL_SOURCE_MISMATCH at {where}: expected {REQUIRED_SIGNAL_SOURCE!r}, "
            f"got {src!r}. Control-plane is not running the validated DSL WFO path."
        )


__all__ = [
    "REQUIRED_SIGNAL_SOURCE",
    "assert_dsl_signal_source",
    "collect_runtime_provenance",
]
