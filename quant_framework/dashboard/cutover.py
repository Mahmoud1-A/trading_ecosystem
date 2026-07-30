"""Staged dashboard cutover — new backend authoritative after switch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from api.discovery_api import DiscoveryAPI
from api.health_api import HealthAPI
from api.paper_api import PaperAPI, PaperStatus
from api.portfolio_api import PortfolioAPI
from api.types import BackendAuthority
from api.vault_api import VaultAPI
from dashboard.candidate_funnel import candidate_funnel_view
from dashboard.data_quality import data_quality_view
from dashboard.execution_quality import execution_quality_view
from dashboard.paper_status import paper_status_view
from dashboard.portfolio_pipeline import portfolio_pipeline_view
from dashboard.system_health import system_health_view


class CutoverStage(str, Enum):
    PARALLEL = "PARALLEL"  # new backend runs alongside legacy
    LEGACY_READONLY = "LEGACY_READONLY"
    COMPARE = "COMPARE"
    SWITCHED = "SWITCHED"  # dashboard reads new backend only
    LEGACY_DISABLED = "LEGACY_DISABLED"  # old promotion disabled; archive preserved


class CutoverError(RuntimeError):
    pass


@dataclass
class LegacyBackendStub:
    """
    Read-only stand-in for the old dashboard backend during migration.

    Promotion is disabled once cutover advances past PARALLEL.
    """

    read_only: bool = False
    promotion_enabled: bool = True
    archive: dict[str, Any] = field(default_factory=dict)
    funnel_counts: dict[str, int] = field(default_factory=dict)

    def promote(self, candidate_id: str) -> None:
        if self.read_only or not self.promotion_enabled:
            raise CutoverError(
                f"legacy promotion disabled (read_only={self.read_only}, "
                f"promotion_enabled={self.promotion_enabled}); cannot promote {candidate_id}"
            )
        self.archive.setdefault("promoted", []).append(candidate_id)

    def snapshot(self) -> dict[str, Any]:
        return {
            "read_only": self.read_only,
            "promotion_enabled": self.promotion_enabled,
            "funnel_counts": dict(self.funnel_counts),
            "archive_keys": sorted(self.archive.keys()),
        }


@dataclass
class ControlPlane:
    discovery: DiscoveryAPI
    portfolio: PortfolioAPI
    vault: VaultAPI
    paper: PaperAPI
    health: HealthAPI
    legacy: LegacyBackendStub = field(default_factory=LegacyBackendStub)
    stage: CutoverStage = CutoverStage.PARALLEL
    authority: BackendAuthority = BackendAuthority.PARALLEL
    comparison_log: list[dict[str, Any]] = field(default_factory=list)
    rollback_stack: list[CutoverStage] = field(default_factory=list)

    def advance(self, target: CutoverStage) -> None:
        order = list(CutoverStage)
        if order.index(target) < order.index(self.stage) and target is not CutoverStage.PARALLEL:
            # Use rollback() for going backwards except explicit reset handled there
            raise CutoverError(f"cannot advance backwards from {self.stage} to {target}; use rollback()")
        self.rollback_stack.append(self.stage)
        self.stage = target
        self._apply_stage_side_effects()

    def rollback(self) -> CutoverStage:
        """Safely roll cutover back one stage (or to PARALLEL)."""
        if not self.rollback_stack:
            self.stage = CutoverStage.PARALLEL
            self._apply_stage_side_effects()
            return self.stage
        self.stage = self.rollback_stack.pop()
        self._apply_stage_side_effects()
        return self.stage

    def _apply_stage_side_effects(self) -> None:
        if self.stage is CutoverStage.PARALLEL:
            self.legacy.read_only = False
            self.legacy.promotion_enabled = True
            self.authority = BackendAuthority.PARALLEL
        elif self.stage is CutoverStage.LEGACY_READONLY:
            self.legacy.read_only = True
            self.legacy.promotion_enabled = False
            self.authority = BackendAuthority.PARALLEL
        elif self.stage is CutoverStage.COMPARE:
            self.legacy.read_only = True
            self.legacy.promotion_enabled = False
            self.authority = BackendAuthority.PARALLEL
        elif self.stage is CutoverStage.SWITCHED:
            self.legacy.read_only = True
            self.legacy.promotion_enabled = False
            self.authority = BackendAuthority.NEW
        elif self.stage is CutoverStage.LEGACY_DISABLED:
            self.legacy.read_only = True
            self.legacy.promotion_enabled = False
            self.authority = BackendAuthority.NEW

    def compare_outputs(self) -> dict[str, Any]:
        """Compare legacy stub funnel counts vs new discovery API (investigation aid)."""
        if self.stage not in {CutoverStage.COMPARE, CutoverStage.LEGACY_READONLY, CutoverStage.PARALLEL}:
            # Still allow compare after switch for audit
            pass
        new_counts = {c.name: c.value for c in self.discovery.funnel_counters()}
        legacy_counts = dict(self.legacy.funnel_counts)
        diffs = {
            k: {"legacy": legacy_counts.get(k), "new": new_counts.get(k)}
            for k in sorted(set(legacy_counts) | set(new_counts))
            if legacy_counts.get(k) != new_counts.get(k)
        }
        report = {
            "stage": self.stage.value,
            "diffs": diffs,
            "match": len(diffs) == 0,
        }
        self.comparison_log.append(report)
        return report

    def attempt_legacy_promote(self, candidate_id: str) -> None:
        """Legacy promotion path — blocked after cutover leaves PARALLEL."""
        try:
            self.legacy.promote(candidate_id)
        except CutoverError:
            self.paper.promotions_blocked_legacy += 1
            raise

    def dashboard_payload(self) -> dict[str, Any]:
        """Authoritative dashboard read — uses new backend when switched."""
        if self.authority is BackendAuthority.LEGACY:
            return {"authority": self.authority.value, "legacy": self.legacy.snapshot()}
        payload = {
            "authority": self.authority.value,
            "stage": self.stage.value,
            "candidate_funnel": candidate_funnel_view(self.discovery),
            "portfolio_pipeline": portfolio_pipeline_view(self.portfolio),
            "data_quality": data_quality_view(self.health),
            "execution_quality": execution_quality_view(self.health),
            "paper_status": paper_status_view(self.paper),
            "system_health": system_health_view(self.health, self.vault),
        }
        if self.authority is BackendAuthority.PARALLEL:
            payload["legacy_readonly_view"] = self.legacy.snapshot()
        return payload

    def preserve_legacy_archive(self) -> dict[str, Any]:
        """Legacy data is retained until a separate migration audit completes."""
        return {
            "preserved": True,
            "archive": dict(self.legacy.archive),
            "deleted": False,
        }
