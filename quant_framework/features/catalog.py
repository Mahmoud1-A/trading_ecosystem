"""Feature catalog — registry of all declared features."""

from __future__ import annotations

from typing import Iterable

from data.events.enums import DataCapability
from features.contracts import FeatureApprovalStatus, FeatureContract


class FeatureCatalog:
    """Mutable registry of feature contracts keyed by feature_id."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureContract] = {}

    def register(self, contract: FeatureContract) -> None:
        if contract.feature_id in self._features:
            existing = self._features[contract.feature_id]
            if existing.feature_version == contract.feature_version:
                raise ValueError(
                    f"Duplicate feature_id={contract.feature_id!r} "
                    f"version={contract.feature_version!r}"
                )
        self._features[contract.feature_id] = contract

    def get(self, feature_id: str) -> FeatureContract:
        if feature_id not in self._features:
            raise KeyError(f"Unknown feature_id={feature_id!r}")
        return self._features[feature_id]

    def all(self) -> list[FeatureContract]:
        return list(self._features.values())

    def approved_causal(
        self,
        *,
        available_capabilities: Iterable[DataCapability] | None = None,
    ) -> list[FeatureContract]:
        caps = set(available_capabilities or ())
        return [f for f in self._features.values() if f.miner_eligible(caps)]

    def by_status(self, status: FeatureApprovalStatus) -> list[FeatureContract]:
        return [f for f in self._features.values() if f.approval_status is status]

    def disable_unsupported(
        self,
        available_capabilities: Iterable[DataCapability],
    ) -> list[str]:
        """
        Return feature_ids that cannot run given available capabilities.

        Does not mutate approval status — callers skip these at generation time.
        """
        caps = set(available_capabilities)
        disabled: list[str] = []
        for f in self._features.values():
            if f.required_capabilities and not all(c in caps for c in f.required_capabilities):
                disabled.append(f.feature_id)
        return disabled
