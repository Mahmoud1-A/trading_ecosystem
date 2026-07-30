"""Deterministic hashing for configs, code, and data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: dict[str, Any] | list[Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(blob)


def hash_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def hash_directory_py(root: Path) -> str:
    """Hash all .py files under root for a crude code fingerprint."""
    root = Path(root)
    parts: list[bytes] = []
    for path in sorted(root.rglob("*.py")):
        if ".pytest_cache" in path.parts or "egg-info" in path.parts:
            continue
        parts.append(path.relative_to(root).as_posix().encode("utf-8"))
        parts.append(path.read_bytes())
    return sha256_bytes(b"\0".join(parts))


def git_commit_hash(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
