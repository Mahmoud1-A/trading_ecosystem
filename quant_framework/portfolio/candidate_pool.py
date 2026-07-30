"""Candidate pool for portfolio construction — behaviorally unique finalists."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from portfolio.constraints import MemberMeta


class CandidateFunnelStage(str, Enum):
    GENERATED = "GENERATED"
    PRECHECK_REJECTED = "PRECHECK_REJECTED"
    FULLY_EVALUATED = "FULLY_EVALUATED"
    SCORE_QUALIFIED = "SCORE_QUALIFIED"
    BEHAVIORALLY_UNIQUE = "BEHAVIORALLY_UNIQUE"
    FINALIST = "FINALIST"


@dataclass
class PoolMember:
    meta: MemberMeta
    stage: CandidateFunnelStage
    pnl_series: tuple[float, ...]
    vol: float = 0.1

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.meta.candidate_id,
            "lineage_id": self.meta.lineage_id,
            "stage": self.stage.value,
            "expected_oos": self.meta.expected_oos,
            "vol": self.vol,
        }


@dataclass
class CandidatePool:
    members: list[PoolMember] = field(default_factory=list)
    # Event counts — never conflate with snapshot Book size
    event_counts: dict[str, int] = field(default_factory=dict)

    def add(self, member: PoolMember) -> None:
        self.members.append(member)
        key = member.stage.value
        self.event_counts[key] = self.event_counts.get(key, 0) + 1

    def finalists(self) -> list[PoolMember]:
        return [m for m in self.members if m.stage is CandidateFunnelStage.FINALIST]

    def metas(self, stage: CandidateFunnelStage | None = None) -> list[MemberMeta]:
        if stage is None:
            return [m.meta for m in self.members]
        return [m.meta for m in self.members if m.stage is stage]

    def pnl_by_id(self, members: Sequence[PoolMember] | None = None) -> dict[str, tuple[float, ...]]:
        ms = members if members is not None else self.finalists()
        return {m.meta.candidate_id: m.pnl_series for m in ms}

    def as_dict(self) -> dict[str, Any]:
        return {
            "members": [m.as_dict() for m in self.members],
            "event_counts": dict(self.event_counts),
            "finalist_count_snapshot": len(self.finalists()),
        }
