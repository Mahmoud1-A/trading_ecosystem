"""Security controls for the research control plane."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from control_plane.models import FORBIDDEN_LIVE_MODES, RunType
from pydantic import BaseModel, Field, field_validator, model_validator


FORBIDDEN_KEYS = frozenset(
    {
        "code",
        "python",
        "exec",
        "eval",
        "shell",
        "command",
        "cmd",
        "script",
        "pickle",
        "import_path",
        "__import__",
        "subprocess",
        "os_system",
    }
)

PATH_TRAVERSAL = re.compile(r"\.\.|[\\/]\.\.|^\s*/|^\s*[A-Za-z]:")


class ConfigValidationError(ValueError):
    pass


class CreateRunRequest(BaseModel):
    """Validated run creation payload — no arbitrary code or live modes."""

    model_config = {"extra": "forbid"}

    run_type: RunType
    random_seed: int = 42
    symbols: list[str] = Field(default_factory=lambda: ["ES"])
    timeframe: str = "5min"
    dataset: str = "synthetic_demo"
    date_range_start: str | None = None
    date_range_end: str | None = None
    strategy_family: str = "mean_reversion_vwap_bb"
    search_budget: dict[str, Any] = Field(default_factory=dict)
    wfo: dict[str, Any] = Field(default_factory=dict)
    cost_model_version: str = "cost_v1"
    risk_profile: str = "demo"
    feature_set_version: str = "feature_set_v1_phase6b"
    grammar_version: str = "strategy_dsl_v1"
    confirm_alpha_miner: bool = False
    confirm_vault: bool = False
    confirm_paper: bool = False
    # Cooperative pause for cancel / progress tests (research-safe, bounded)
    hold_seconds: float = Field(default=0.0, ge=0.0, le=30.0)
    # Explicit smoke-test flag required to use synthetic_demo with Alpha Miner
    smoke_test: bool = False
    # One-candidate UI canary overrides (research-safe)
    ui_canary: dict[str, Any] = Field(default_factory=dict)
    # Multi-family Strategy Family Generator campaign (above Alpha Miner)
    multi_family: dict[str, Any] = Field(default_factory=dict)
    # Persistent search-program resume controls
    search_mode: str = "NEW_SEARCH"
    search_program_id: str | None = None
    source_run_id: str | None = None
    resumed_from_run_id: str | None = None
    additional_runtime_seconds: float = Field(default=0.0, ge=0.0)
    additional_generated_budget: int = Field(default=0, ge=0)
    additional_full_wfo_budget: int = Field(default=0, ge=0)
    # Explicitly rejected if present via validator on nested dicts
    environment: str | None = None
    trading_mode: str | None = None
    credentials: dict[str, Any] | None = None

    @field_validator("search_mode")
    @classmethod
    def _valid_search_mode(cls, v: str) -> str:
        allowed = {
            "NEW_SEARCH",
            "RESUME_SEARCH",
            "EXTEND_BUDGET",
            "REEVALUATE_FROZEN_CANDIDATES",
        }
        mode = str(v or "NEW_SEARCH").upper()
        if mode not in allowed:
            raise ValueError(f"search_mode must be one of {sorted(allowed)}")
        return mode

    @field_validator("environment", "trading_mode")
    @classmethod
    def _no_live_env(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v.upper() in FORBIDDEN_LIVE_MODES or v.upper() == "LIVE":
            raise ValueError(f"live mode {v!r} cannot be activated from the dashboard")
        if v.lower() not in {"research", "paper", "shadow", "test", ""}:
            # allow only research-safe labels
            if v.upper() in FORBIDDEN_LIVE_MODES:
                raise ValueError(f"forbidden mode {v}")
        return v

    @field_validator("credentials")
    @classmethod
    def _no_live_creds(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return v
        blob = " ".join(f"{k}={x}" for k, x in v.items()).lower()
        if "live" in blob or "production" in blob:
            raise ValueError("live credentials are not accepted")
        return v

    @field_validator("search_budget", "wfo", "ui_canary", "multi_family")
    @classmethod
    def _no_forbidden_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        for k in v:
            if k.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden config key {k!r}")
            if isinstance(v[k], str) and any(
                x in v[k].lower() for x in ("__import__", "os.system", "subprocess")
            ):
                raise ValueError("executable content not allowed in config values")
        return v

    def require_confirmations(self) -> None:
        if self.run_type is RunType.ALPHA_MINER and not self.confirm_alpha_miner:
            raise ConfigValidationError("Alpha Miner requires explicit confirmation")
        if self.run_type is RunType.VAULT_EVALUATION and not self.confirm_vault:
            raise ConfigValidationError("Vault evaluation requires explicit confirmation")
        if self.run_type is RunType.PAPER_RUNTIME and not self.confirm_paper:
            raise ConfigValidationError("Paper runtime requires explicit confirmation")
        if self.environment and self.environment.upper() in FORBIDDEN_LIVE_MODES:
            raise ConfigValidationError("live modes are forbidden")
        if self.trading_mode and self.trading_mode.upper() in FORBIDDEN_LIVE_MODES:
            raise ConfigValidationError("live trading modes are forbidden")

    @model_validator(mode="after")
    def _multi_family_contract(self) -> CreateRunRequest:
        mf = dict(self.multi_family or {})
        if not mf.get("enabled"):
            return self
        if self.run_type is not RunType.ALPHA_MINER:
            raise ValueError("multi_family is only valid for ALPHA_MINER runs")
        from control_plane.multi_family_ui import validate_multi_family_launch

        try:
            preview = validate_multi_family_launch(
                strategy_family=self.strategy_family,
                multi_family=mf,
                random_seed=self.random_seed,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        # Attach preview for run snapshot / frozen config (non-serialized side channel
        # via multi_family dict copy on the model).
        if preview is not None:
            enriched = {
                **mf,
                "enabled": True,
                "preview_family_specs": preview.get("families"),
                "distinct_grammar_fingerprints": preview.get("distinct_grammar_fingerprints"),
                "grammar_fingerprints": preview.get("grammar_fingerprints"),
            }
            object.__setattr__(self, "multi_family", enriched)
        return self

    def validate_against_catalog(self, catalog: Any) -> None:
        """Enforce dataset eligibility and symbol/timeframe constraints."""
        from data.catalog.catalog import SMOKE_DATASET_ID

        entry = catalog.get(self.dataset)
        if entry is None:
            raise ConfigValidationError(f"unknown dataset {self.dataset!r}")
        if self.run_type is RunType.ALPHA_MINER:
            if getattr(entry, "vault_locked", False) or getattr(
                entry, "alpha_miner_access", "ALLOWED"
            ) == "DENIED":
                raise ConfigValidationError(
                    f"dataset {self.dataset!r} is VAULT_LOCKED / "
                    "ALPHA_MINER_ACCESS=DENIED — not selectable for Alpha Miner"
                )
            if entry.smoke_test_only or self.dataset == SMOKE_DATASET_ID:
                if not self.smoke_test:
                    raise ConfigValidationError(
                        "synthetic_demo is SMOKE TEST ONLY — set smoke_test=true "
                        "or select a research_eligible dataset"
                    )
            elif not entry.research_eligible:
                raise ConfigValidationError(
                    f"dataset {self.dataset!r} is not research_eligible: "
                    f"{entry.rejection_reasons}"
                )
            else:
                self._require_real_dataset_gates(entry)
        if not entry.smoke_test_only:
            # Constrain symbols / timeframe to catalog membership
            allowed_symbols = {s.upper() for s in entry.symbols} | {
                c.upper() for c in entry.tradable_contracts
            }
            for sym in self.symbols:
                if sym.upper() not in allowed_symbols:
                    raise ConfigValidationError(
                        f"symbol {sym!r} not in dataset {self.dataset}; "
                        f"allowed={sorted(allowed_symbols)}"
                    )
            from data.catalog.silver_bar_resolution import (
                compatible_timeframes_for_entry,
                display_timeframe,
                normalize_timeframe,
                strategy_allows_timeframe,
            )

            allowed_tfs: set[str] = set()
            try:
                allowed_tfs.add(normalize_timeframe(entry.timeframe))
            except ValueError:
                pass
            for tf in compatible_timeframes_for_entry(entry):
                try:
                    allowed_tfs.add(normalize_timeframe(tf))
                except ValueError:
                    continue
            # Derived bar rules on quote datasets
            for rule in list((entry.schema or {}).get("derived_bar_rules") or []):
                try:
                    allowed_tfs.add(normalize_timeframe(str(rule)))
                except ValueError:
                    continue
            try:
                req_tf = normalize_timeframe(self.timeframe)
            except ValueError as exc:
                raise ConfigValidationError(str(exc)) from exc
            if req_tf not in allowed_tfs:
                raise ConfigValidationError(
                    f"timeframe {self.timeframe!r} incompatible with dataset "
                    f"{self.dataset}; allowed="
                    f"{sorted({display_timeframe(t) for t in allowed_tfs})}"
                )
            if not strategy_allows_timeframe(self.strategy_family, self.timeframe):
                raise ConfigValidationError(
                    f"strategy_family {self.strategy_family!r} does not support "
                    f"timeframe {self.timeframe!r} (raw tick rejected for "
                    "mean_reversion_vwap_bb — use 1m or 5m)"
                )
            multi_enabled = bool((self.multi_family or {}).get("enabled"))
            from control_plane.multi_family_ui import MULTI_FAMILY_STRATEGY_FAMILY

            if multi_enabled and self.strategy_family == MULTI_FAMILY_STRATEGY_FAMILY:
                pass  # dedicated multi-family label — not a catalog MR template
            elif self.strategy_family not in entry.compatible_strategy_families:
                raise ConfigValidationError(
                    f"strategy_family {self.strategy_family!r} not compatible with dataset"
                )
            if self.cost_model_version not in entry.compatible_cost_models:
                raise ConfigValidationError(
                    f"cost_model_version {self.cost_model_version!r} not compatible with dataset"
                )
            if self.risk_profile not in entry.compatible_risk_profiles:
                raise ConfigValidationError(
                    f"risk_profile {self.risk_profile!r} not compatible with dataset"
                )
            if self.feature_set_version not in entry.compatible_feature_sets:
                raise ConfigValidationError(
                    f"feature_set_version {self.feature_set_version!r} not compatible with dataset"
                )

    @staticmethod
    def _require_real_dataset_gates(entry: Any) -> None:
        """
        Phase 12.2 gates for a real (non-smoke) Alpha Miner dataset.

        Passing these grants RESEARCH_ELIGIBLE only. Paper and live eligibility
        are decided separately and require target-broker validation.
        """
        from data.catalog.cfd_import import gate_reasons_for_alpha_miner

        reasons = gate_reasons_for_alpha_miner(entry)
        if reasons:
            raise ConfigValidationError(
                f"dataset {entry.dataset_id!r} cannot start a real Alpha Miner run: "
                + "; ".join(reasons)
            )


def safe_artifact_path(artifact_dir: Path, relative: str, manifest: list[str]) -> Path:
    """Resolve an artifact path; block traversal; require manifest membership."""
    if not relative or PATH_TRAVERSAL.search(relative) or relative.startswith(("/", "\\")):
        raise ConfigValidationError("path traversal blocked")
    name = Path(relative).name
    # Also allow nested relative paths if normalized stays under artifact_dir
    candidate = (artifact_dir / relative).resolve()
    root = artifact_dir.resolve()
    if not str(candidate).startswith(str(root)):
        raise ConfigValidationError("path traversal blocked")
    rel = str(Path(relative).as_posix())
    allowed = set(manifest) | {Path(m).name for m in manifest}
    if rel not in allowed and name not in allowed and relative not in allowed:
        raise ConfigValidationError("artifact not in manifest")
    if not candidate.is_file():
        raise FileNotFoundError(relative)
    return candidate


def reject_arbitrary_command_payload(payload: dict[str, Any]) -> None:
    for k, v in payload.items():
        if k.lower() in FORBIDDEN_KEYS:
            raise ConfigValidationError(f"arbitrary command key rejected: {k}")
        if isinstance(v, dict):
            reject_arbitrary_command_payload(v)
