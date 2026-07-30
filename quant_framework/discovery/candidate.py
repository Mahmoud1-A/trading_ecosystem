"""Strategy candidate representation with deterministic identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from discovery.canonicalization import canonical_payload
from discovery.complexity import complexity_score
from discovery.expression_tree import ExprNode
from discovery.types import CreationMethod
from registry.hashing import sha256_json


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    lineage_id: str
    generation: int
    parent_ids: tuple[str, ...]
    creation_method: CreationMethod
    strategy_family: str
    expression_tree: ExprNode | None
    entry_tree: ExprNode
    exit_tree: ExprNode | None
    stop: ExprNode | None
    target: ExprNode | None
    sizing: ExprNode | None
    regime_gates: tuple[ExprNode, ...]
    feature_ids: tuple[str, ...]
    parameters: dict[str, float]
    complexity_score: float
    grammar_version: str
    feature_set_version: str
    cost_model_version: str
    asset_universe: tuple[str, ...]
    random_seed: int
    creation_timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "lineage_id": self.lineage_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "creation_method": self.creation_method.value,
            "strategy_family": self.strategy_family,
            "expression_tree": self.expression_tree.as_dict() if self.expression_tree else None,
            "entry_tree": self.entry_tree.as_dict(),
            "exit_tree": self.exit_tree.as_dict() if self.exit_tree else None,
            "stop": self.stop.as_dict() if self.stop else None,
            "target": self.target.as_dict() if self.target else None,
            "sizing": self.sizing.as_dict() if self.sizing else None,
            "regime_gates": [g.as_dict() for g in self.regime_gates],
            "feature_ids": list(self.feature_ids),
            "parameters": dict(self.parameters),
            "complexity_score": self.complexity_score,
            "grammar_version": self.grammar_version,
            "feature_set_version": self.feature_set_version,
            "cost_model_version": self.cost_model_version,
            "asset_universe": list(self.asset_universe),
            "random_seed": self.random_seed,
            "creation_timestamp": self.creation_timestamp,
        }


def collect_features(*trees: ExprNode | None) -> tuple[str, ...]:
    ids: set[str] = set()
    for t in trees:
        if t is not None:
            ids.update(t.feature_ids())
    return tuple(sorted(ids))


def collect_parameters(*trees: ExprNode | None) -> dict[str, float]:
    params: dict[str, float] = {}
    for t in trees:
        if t is None:
            continue
        for node in t.walk():
            if node.kind.value == "PARAMETER":
                params[node.name] = float(node.meta.get("default", 0.0))
    return params


def build_candidate(
    *,
    entry_tree: ExprNode,
    exit_tree: ExprNode | None = None,
    stop: ExprNode | None = None,
    target: ExprNode | None = None,
    sizing: ExprNode | None = None,
    regime_gates: tuple[ExprNode, ...] = (),
    expression_tree: ExprNode | None = None,
    strategy_family: str = "dsl_mean_reversion",
    creation_method: CreationMethod = CreationMethod.RANDOM,
    generation: int = 0,
    parent_ids: tuple[str, ...] = (),
    grammar_version: str,
    feature_set_version: str,
    cost_model_version: str = "cost_v1",
    asset_universe: tuple[str, ...] = ("ES",),
    random_seed: int,
    lineage_id: str | None = None,
) -> StrategyCandidate:
    features = collect_features(entry_tree, exit_tree, stop, target, sizing, *regime_gates)
    parameters = collect_parameters(entry_tree, exit_tree, stop, target, sizing, *regime_gates)
    payload = canonical_payload(
        entry=entry_tree,
        exit=exit_tree,
        stop=stop,
        target=target,
        regime_gates=regime_gates,
        sizing=sizing,
        strategy_family=strategy_family,
        grammar_version=grammar_version,
        feature_set_version=feature_set_version,
        parameters=parameters,
    )
    candidate_id = "cand_" + sha256_json(payload)[:24]
    # Lineage incorporates creation method + parents so legacy imports get new identities
    lineage_material = {
        **payload,
        "creation_method": creation_method.value,
        "parent_ids": list(parent_ids),
        "generation": generation,
    }
    lin = lineage_id or ("lin_" + sha256_json(lineage_material)[:24])
    score = complexity_score(entry_tree, exit_tree, stop, target, sizing, *regime_gates)
    return StrategyCandidate(
        candidate_id=candidate_id,
        lineage_id=lin,
        generation=generation,
        parent_ids=parent_ids,
        creation_method=creation_method,
        strategy_family=strategy_family,
        expression_tree=expression_tree or entry_tree,
        entry_tree=entry_tree,
        exit_tree=exit_tree,
        stop=stop,
        target=target,
        sizing=sizing,
        regime_gates=regime_gates,
        feature_ids=features,
        parameters=parameters,
        complexity_score=score,
        grammar_version=grammar_version,
        feature_set_version=feature_set_version,
        cost_model_version=cost_model_version,
        asset_universe=asset_universe,
        random_seed=random_seed,
    )
