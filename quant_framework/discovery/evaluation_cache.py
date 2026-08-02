"""Durable evaluation cache keyed by candidate + dataset/config/code fingerprint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from registry.hashing import sha256_json


def build_evaluation_cache_key(
    *,
    candidate_canonical_hash: str,
    dataset_hash: str,
    timeframe: str,
    wfo_config_hash: str,
    cost_model_version: str,
    risk_model_version: str,
    execution_semantic_hash: str = "",
    execution_engine_code_hash: str = "",
) -> str:
    """Build a durable eval-cache key.

    ``execution_semantic_hash`` is preferred; ``execution_engine_code_hash`` is a
    legacy alias accepted for older call sites.
    """
    semantic = str(execution_semantic_hash or execution_engine_code_hash or "")
    payload = {
        "candidate_canonical_hash": str(candidate_canonical_hash),
        "dataset_hash": str(dataset_hash),
        "timeframe": str(timeframe),
        "wfo_config_hash": str(wfo_config_hash),
        "cost_model_version": str(cost_model_version),
        "risk_model_version": str(risk_model_version),
        "execution_semantic_hash": semantic,
    }
    return "evcache_" + sha256_json(payload)[:40]


@dataclass
class EvaluationCacheEntry:
    cache_key: str
    candidate_id: str
    candidate_canonical_hash: str
    evaluation_record: dict[str, Any]
    created_at: str
    fingerprint_components: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "candidate_id": self.candidate_id,
            "candidate_canonical_hash": self.candidate_canonical_hash,
            "evaluation_record": dict(self.evaluation_record),
            "created_at": self.created_at,
            "fingerprint_components": dict(self.fingerprint_components),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> EvaluationCacheEntry:
        return EvaluationCacheEntry(
            cache_key=str(raw["cache_key"]),
            candidate_id=str(raw["candidate_id"]),
            candidate_canonical_hash=str(raw["candidate_canonical_hash"]),
            evaluation_record=dict(raw.get("evaluation_record") or {}),
            created_at=str(raw.get("created_at") or ""),
            fingerprint_components=dict(raw.get("fingerprint_components") or {}),
        )


class EvaluationCache:
    """Atomic per-key JSON files under a program evaluation_cache directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, cache_key: str) -> Path:
        safe = cache_key.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.json"

    def get(self, cache_key: str) -> EvaluationCacheEntry | None:
        path = self._path(cache_key)
        if not path.is_file():
            return None
        return EvaluationCacheEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def put(self, entry: EvaluationCacheEntry) -> None:
        path = self._path(entry.cache_key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry.as_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def invalidate_all(self) -> int:
        """Remove all cached evaluations (REEVALUATE mode)."""
        n = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            n += 1
        return n

    def __contains__(self, cache_key: str) -> bool:
        return self._path(cache_key).is_file()
