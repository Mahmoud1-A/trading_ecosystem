"""Execution-semantic code hashing for Alpha Miner search resume compatibility.

Separates repository provenance (exact git commit) from the allowlisted content
hash of modules that can alter candidate behavior or evaluation results.
Control-plane, dashboard, tests, and search-bookkeeping changes do not change
``execution_semantic_hash`` and therefore do not block true resume.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from registry.hashing import sha256_bytes, sha256_json

# Paths relative to the repository root. Explicit allowlist — never "hash the
# whole tree" and never include control-plane presentation or tests.
EXECUTION_SEMANTIC_ALLOWLIST: tuple[str, ...] = (
    # Candidate DSL and canonicalization
    "quant_framework/discovery/candidate.py",
    "quant_framework/discovery/canonicalization.py",
    "quant_framework/discovery/complexity.py",
    "quant_framework/discovery/expression_tree.py",
    "quant_framework/discovery/grammar.py",
    "quant_framework/discovery/legacy_parameters.py",
    "quant_framework/discovery/operators.py",
    "quant_framework/discovery/repair.py",
    "quant_framework/discovery/typecheck.py",
    "quant_framework/discovery/types.py",
    # Family grammars and candidate generation
    "quant_framework/discovery/family_catalog.py",
    "quant_framework/discovery/family_generator.py",
    "quant_framework/discovery/family_spec.py",
    "quant_framework/discovery/feature_domains.py",
    "quant_framework/discovery/generator.py",
    # Mutation / crossover / evolution
    "quant_framework/discovery/crossover.py",
    "quant_framework/discovery/lineage.py",
    "quant_framework/discovery/mutation.py",
    "quant_framework/discovery/selection.py",
    "quant_framework/discovery/stable_hash.py",
    "quant_framework/discovery/multi_family_campaign.py",
    "quant_framework/discovery/search_controller.py",
    # Feature and signal evaluation
    "quant_framework/discovery/dsl_series.py",
    "quant_framework/discovery/dsl_signal_adapter.py",
    "quant_framework/discovery/evaluator.py",
    "quant_framework/discovery/fitness.py",
    "quant_framework/discovery/freezing.py",
    "quant_framework/discovery/prechecks.py",
    "quant_framework/discovery/regime_gates.py",
    "quant_framework/features/catalog.py",
    "quant_framework/features/contracts.py",
    "quant_framework/features/cross_asset.py",
    "quant_framework/features/definitions.py",
    "quant_framework/features/generator.py",
    "quant_framework/features/liquidity.py",
    "quant_framework/features/microstructure.py",
    "quant_framework/features/normalization.py",
    "quant_framework/features/price.py",
    "quant_framework/features/regime_features.py",
    "quant_framework/features/temporal.py",
    "quant_framework/features/validation.py",
    "quant_framework/features/volatility.py",
    "quant_framework/regime/detector.py",
    # Event-driven BID/ASK execution engine
    "quant_framework/discovery/event_wfo_backend.py",
    "quant_framework/engine/event_execution.py",
    "quant_framework/engine/events.py",
    "quant_framework/engine/fills.py",
    "quant_framework/engine/financing.py",
    "quant_framework/engine/friction.py",
    "quant_framework/engine/ids.py",
    "quant_framework/engine/intrabar_policy.py",
    "quant_framework/engine/orders.py",
    "quant_framework/engine/portfolio.py",
    "quant_framework/engine/positions.py",
    "quant_framework/engine/rollover.py",
    "quant_framework/engine/vectorized_features.py",
    # WFO fold construction and evaluation
    "quant_framework/validation/bootstrap.py",
    "quant_framework/validation/event_driven_wfo.py",
    "quant_framework/validation/purge_embargo.py",
    "quant_framework/validation/walk_forward.py",
    "quant_framework/config/walk_forward_config.py",
    # Spread, costs, slippage, execution delay
    "quant_framework/config/asset_spec.py",
    "quant_framework/config/cost_model.py",
    "quant_framework/config/models.py",
    "quant_framework/config/system_config.py",
    "quant_framework/cfds/cost_policy.py",
    "quant_framework/cfds/execution_profile.py",
    "quant_framework/cfds/financing_profile.py",
    "quant_framework/cfds/instrument_spec.py",
    # Risk and sizing semantics
    "quant_framework/risk/exposure.py",
    "quant_framework/risk/fsm.py",
    "quant_framework/risk/position_sizing.py",
    "quant_framework/config/prop_profile.py",
    "quant_framework/cfds/prop_profile.py",
    # Stress scenarios
    "quant_framework/discovery/stress.py",
    "quant_framework/discovery/stress_backend.py",
    # Robustness neighborhoods
    "quant_framework/discovery/parameter_robustness.py",
    # DSR, PBO, behavioral clustering
    "quant_framework/discovery/behavioral_dedup.py",
    "quant_framework/metrics/dsr.py",
    "quant_framework/metrics/pbo.py",
    "quant_framework/metrics/performance.py",
    "quant_framework/metrics/diagnostics.py",
    "quant_framework/metrics/drawdown.py",
)

# Non-semantic paths commonly touched by control-plane / resume work (for manifests).
EXECUTION_SEMANTIC_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "quant_framework/control_plane/",
    "quant_framework/dashboard/",
    "quant_framework/tests/",
    "quant_framework/api/",
    "quant_framework/scripts/",
    "quant_framework/reports/",
    "quant_framework/live/",
    "quant_framework/brokers/",
    "quant_framework/runtime/",
    "quant_framework/monitoring/",
    "quant_framework/governance/",
    "quant_framework/portfolio/",
    "quant_framework/legacy/",
    "quant_framework/discovery/search_program.py",
    "quant_framework/discovery/search_resume.py",
    "quant_framework/discovery/search_checkpoint.py",
    "quant_framework/discovery/search_budget.py",
    "quant_framework/discovery/evaluation_cache.py",
    "quant_framework/discovery/execution_semantic_hash.py",
    "quant_framework/discovery/vault_gateway.py",
)

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def looks_like_git_sha(value: str | None) -> bool:
    if not value:
        return False
    return bool(_GIT_SHA_RE.fullmatch(str(value).strip()))


def resolve_repo_root(hint: Path | None = None) -> Path:
    """Resolve the git repository root (trading_ecosystem)."""
    if hint is not None:
        cand = Path(hint).resolve()
        if (cand / ".git").exists():
            return cand
        # quant_framework/ package path → parents[1] is repo root
        if cand.name == "quant_framework" and (cand.parent / ".git").exists():
            return cand.parent
    here = Path(__file__).resolve()
    # discovery/ → quant_framework/ → repo
    for parent in [here.parents[2], here.parents[1], Path.cwd().resolve()]:
        if (parent / ".git").exists():
            return parent
    return here.parents[2]


def git_show_bytes(repo_root: Path, rel_path: str, commit: str) -> bytes | None:
    """Return file bytes at ``commit:rel_path``, or None if missing."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{rel_path}"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _working_tree_bytes(repo_root: Path, rel_path: str) -> bytes | None:
    path = Path(repo_root) / rel_path
    if not path.is_file():
        return None
    return path.read_bytes()


def _normalize_text_bytes(raw: bytes) -> bytes:
    """Normalize newlines so Windows working-tree CRLF matches git show LF."""
    if b"\0" in raw[:1024]:
        return raw
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def collect_allowlist_content_hashes(
    repo_root: Path,
    *,
    at_commit: str | None = None,
    allowlist: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Map allowlisted relative paths → sha256 hex of file contents.

    Missing files are omitted (deterministic: only present paths contribute).
    Text newlines are normalized to LF before hashing.
    """
    root = Path(repo_root).resolve()
    paths = allowlist if allowlist is not None else EXECUTION_SEMANTIC_ALLOWLIST
    out: dict[str, str] = {}
    for rel in paths:
        if at_commit:
            raw = git_show_bytes(root, rel, at_commit)
        else:
            raw = _working_tree_bytes(root, rel)
        if raw is None:
            continue
        out[rel] = sha256_bytes(_normalize_text_bytes(raw))
    return out


def compute_execution_semantic_hash(
    repo_root: Path | None = None,
    *,
    at_commit: str | None = None,
    allowlist: tuple[str, ...] | None = None,
) -> str:
    """Deterministic hash of execution-semantic allowlist contents.

    Format: ``esem_`` + first 32 hex chars of sha256(sorted path→content_hash map).
    """
    root = resolve_repo_root(repo_root)
    content = collect_allowlist_content_hashes(
        root, at_commit=at_commit, allowlist=allowlist
    )
    payload = [{"path": p, "sha256": content[p]} for p in sorted(content)]
    return "esem_" + sha256_json(payload)[:32]


def is_excluded_semantic_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    for prefix in EXECUTION_SEMANTIC_EXCLUDED_PREFIXES:
        if norm == prefix.rstrip("/") or norm.startswith(prefix):
            return True
    return False


def build_semantic_hash_manifest(
    *,
    repo_root: Path | None = None,
    stored_repository_git_sha: str,
    current_repository_git_sha: str | None = None,
    stored_execution_semantic_hash: str | None = None,
    current_execution_semantic_hash: str | None = None,
    excluded_change_paths: list[str] | None = None,
    migration_decision: str | None = None,
    migration_reason: str | None = None,
) -> dict[str, Any]:
    """Build a diagnostic manifest for audit / migration verification."""
    root = resolve_repo_root(repo_root)
    current_sha = current_repository_git_sha
    if not current_sha:
        try:
            current_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            current_sha = "UNKNOWN"

    stored_contents = collect_allowlist_content_hashes(
        root, at_commit=stored_repository_git_sha
    )
    current_contents = collect_allowlist_content_hashes(root, at_commit=current_sha)

    stored_sem = stored_execution_semantic_hash or compute_execution_semantic_hash(
        root, at_commit=stored_repository_git_sha
    )
    current_sem = current_execution_semantic_hash or compute_execution_semantic_hash(
        root, at_commit=current_sha
    )

    if migration_decision is None:
        if stored_sem == current_sem:
            migration_decision = "MIGRATE_ALLOW_RESUME"
            migration_reason = (
                "execution_semantic_hash identical across repository commits; "
                "changed paths are outside the semantic allowlist"
            )
        else:
            migration_decision = "REJECT_EXECUTION_SEMANTICS_CHANGED"
            migration_reason = (
                "execution_semantic_hash diverged; refuse true resume"
            )

    changed_allowlist = sorted(
        p
        for p in set(stored_contents) | set(current_contents)
        if stored_contents.get(p) != current_contents.get(p)
    )
    excluded = list(excluded_change_paths or [])
    if not excluded and stored_repository_git_sha and current_sha:
        try:
            diff_out = subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--name-only",
                    stored_repository_git_sha,
                    current_sha,
                ],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            excluded = [
                line.strip().replace("\\", "/")
                for line in diff_out.splitlines()
                if line.strip() and is_excluded_semantic_path(line.strip())
            ]
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            excluded = []

    return {
        "schema_version": 1,
        "included_paths": [
            {"path": p, "content_sha256": stored_contents[p]}
            for p in sorted(stored_contents)
        ],
        "included_paths_current": [
            {"path": p, "content_sha256": current_contents[p]}
            for p in sorted(current_contents)
        ],
        "allowlist_path_count": len(EXECUTION_SEMANTIC_ALLOWLIST),
        "allowlist_present_at_stored": len(stored_contents),
        "allowlist_present_at_current": len(current_contents),
        "changed_allowlist_paths": changed_allowlist,
        "excluded_change_paths": excluded,
        "stored_repository_git_sha": stored_repository_git_sha,
        "current_repository_git_sha": current_sha,
        "stored_execution_semantic_hash": stored_sem,
        "current_execution_semantic_hash": current_sem,
        "semantic_hashes_match": stored_sem == current_sem,
        "migration_decision": migration_decision,
        "migration_reason": migration_reason,
    }


def write_semantic_hash_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def normalize_legacy_code_hash_component(
    components: dict[str, Any],
    *,
    repo_root: Path | None = None,
    current_execution_semantic_hash: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize stored fingerprint components from legacy whole-repo SHA schema.

    Legacy programs stored the repository git SHA in ``execution_engine_code_hash``.
    Returns ``(normalized_components, migration_info)``.
    """
    comps = dict(components)
    info: dict[str, Any] = {
        "legacy_schema": False,
        "legacy_execution_engine_code_hash": None,
        "computed_stored_semantic_hash": None,
        "semantic_match_to_current": None,
    }

    semantic = comps.get("execution_semantic_hash")
    legacy = comps.get("execution_engine_code_hash")
    repo_sha = comps.get("repository_git_sha")

    # Genuine new-schema semantic hash (esem_… or opaque test token, not a git SHA).
    if semantic not in (None, "") and not looks_like_git_sha(semantic):
        if not repo_sha and looks_like_git_sha(legacy):
            comps["repository_git_sha"] = str(legacy)
        return comps, info

    # Semantic slot accidentally holds a git SHA (pre-normalization promotion).
    sha_candidate = None
    if looks_like_git_sha(semantic):
        sha_candidate = str(semantic)
    elif looks_like_git_sha(legacy):
        sha_candidate = str(legacy)
    elif looks_like_git_sha(repo_sha) and (semantic in (None, "") or looks_like_git_sha(semantic)):
        sha_candidate = str(repo_sha)

    if sha_candidate is None:
        # Non-git opaque test hashes — treat the value itself as the semantic hash.
        if legacy not in (None, "") and semantic in (None, ""):
            comps["execution_semantic_hash"] = str(legacy)
        return comps, info

    root = resolve_repo_root(repo_root)
    stored_sem = compute_execution_semantic_hash(root, at_commit=sha_candidate)
    info.update(
        {
            "legacy_schema": True,
            "legacy_execution_engine_code_hash": sha_candidate,
            "computed_stored_semantic_hash": stored_sem,
        }
    )
    if current_execution_semantic_hash is not None:
        info["semantic_match_to_current"] = stored_sem == current_execution_semantic_hash

    comps["repository_git_sha"] = sha_candidate
    comps["execution_semantic_hash"] = stored_sem
    comps["execution_engine_code_hash"] = sha_candidate
    return comps, info


__all__ = [
    "EXECUTION_SEMANTIC_ALLOWLIST",
    "EXECUTION_SEMANTIC_EXCLUDED_PREFIXES",
    "build_semantic_hash_manifest",
    "collect_allowlist_content_hashes",
    "compute_execution_semantic_hash",
    "git_show_bytes",
    "is_excluded_semantic_path",
    "looks_like_git_sha",
    "normalize_legacy_code_hash_component",
    "resolve_repo_root",
    "write_semantic_hash_manifest",
]
