"""Portfolio control-plane API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from api.types import CounterKind, TypedCounter
from portfolio.versioning import PortfolioPipelineStage, PortfolioVersion


@dataclass
class PortfolioSessionState:
    versions_built: int = 0
    stability_passed: int = 0
    frozen: int = 0
    vault_eligible: int = 0
    vault_tested: int = 0
    vault_passed: int = 0
    vault_failed: int = 0
    paper_eligible: int = 0


@dataclass
class PortfolioAPI:
    """Portfolio pipeline counters + current membership snapshot."""

    versions: list[PortfolioVersion] = field(default_factory=list)
    session: PortfolioSessionState = field(default_factory=PortfolioSessionState)

    def record_version(self, version: PortfolioVersion) -> None:
        self.versions.append(version)
        self.session.versions_built += 1
        stage = version.stage
        if stage in {
            PortfolioPipelineStage.STABILITY_PASSED,
            PortfolioPipelineStage.FROZEN,
            PortfolioPipelineStage.VAULT_ELIGIBLE,
            PortfolioPipelineStage.VAULT_TESTED,
            PortfolioPipelineStage.VAULT_PASSED,
            PortfolioPipelineStage.PAPER_ELIGIBLE,
        }:
            self.session.stability_passed += 1
        if version.frozen or stage is PortfolioPipelineStage.FROZEN:
            self.session.frozen += 1
        if stage is PortfolioPipelineStage.VAULT_ELIGIBLE:
            self.session.vault_eligible += 1
        if stage is PortfolioPipelineStage.VAULT_TESTED:
            self.session.vault_tested += 1
        if stage is PortfolioPipelineStage.VAULT_PASSED:
            self.session.vault_passed += 1
        if stage is PortfolioPipelineStage.VAULT_FAILED:
            self.session.vault_failed += 1
        if stage is PortfolioPipelineStage.PAPER_ELIGIBLE:
            self.session.paper_eligible += 1

    def current_members(self) -> list[str]:
        """Snapshot of the latest frozen (else latest) portfolio members."""
        frozen = [v for v in self.versions if v.frozen]
        if frozen:
            return list(frozen[-1].member_candidate_ids)
        if self.versions:
            return list(self.versions[-1].member_candidate_ids)
        return []

    def counters(self) -> list[TypedCounter]:
        s = self.session
        return [
            TypedCounter(
                "current_members",
                len(self.current_members()),
                CounterKind.SNAPSHOT,
                "active portfolio membership (snapshot)",
            ),
            TypedCounter("versions_built", s.versions_built, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("stability_passed", s.stability_passed, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("frozen", s.frozen, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("vault_eligible", s.vault_eligible, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("vault_tested", s.vault_tested, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("vault_passed", s.vault_passed, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("vault_failed", s.vault_failed, CounterKind.CUMULATIVE_EVENT),
            TypedCounter("paper_eligible", s.paper_eligible, CounterKind.CUMULATIVE_EVENT),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_members": self.current_members(),
            "counters": [c.as_dict() for c in self.counters()],
        }
