"""Causal feature generator — deterministic, capability-aware (Phase 6B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.calendars import ExchangeCalendar, get_calendar
from data.events.enums import DataCapability
from features.catalog import FeatureCatalog
from features.contracts import FeatureApprovalStatus, FeatureFrame
from features.cross_asset import compute_cross_asset_features
from features.definitions import build_default_catalog
from features.liquidity import compute_liquidity_features
from features.microstructure import compute_microstructure_features
from features.price import compute_price_features
from features.temporal import compute_temporal_features
from features.validation import assert_no_vault_columns
from features.volatility import compute_volatility_features
from registry.hashing import sha256_json


FEATURE_SET_VERSION = "feature_set_v1_phase6b"


@dataclass
class FeatureGeneratorConfig:
    calendar: ExchangeCalendar | str = "CME"
    bar_end_offset: str = "5min"
    available_capabilities: tuple[DataCapability, ...] = (DataCapability.OHLCV_BARS,)
    include_experimental: bool = False
    volume_type: str | None = None


class FeatureGenerator:
    """
    Build a causal feature frame from OHLCV (+ optional quote/book/secondary).

    Unsupported capabilities disable dependent features rather than synthesizing data.
    """

    def __init__(
        self,
        catalog: FeatureCatalog | None = None,
        *,
        config: FeatureGeneratorConfig | None = None,
        code_hash: str = "unknown",
    ) -> None:
        self.catalog = catalog or build_default_catalog()
        self.config = config or FeatureGeneratorConfig()
        self.code_hash = code_hash

    def _bar_end(self, index: pd.DatetimeIndex) -> pd.Series:
        delta = pd.Timedelta(self.config.bar_end_offset)
        return pd.Series(index + delta, index=index, name="availability_timestamp")

    def _eligible_ids(self) -> set[str]:
        caps = set(self.config.available_capabilities)
        out: set[str] = set()
        for f in self.catalog.all():
            if f.approval_status is FeatureApprovalStatus.DISABLED:
                continue
            if f.approval_status is FeatureApprovalStatus.DEPRECATED:
                continue
            if f.approval_status is FeatureApprovalStatus.EXPERIMENTAL and not self.config.include_experimental:
                continue
            if f.approval_status is FeatureApprovalStatus.APPROVED and f.causal:
                if all(c in caps for c in f.required_capabilities):
                    out.add(f.feature_id)
            elif self.config.include_experimental and f.causal:
                if all(c in caps for c in f.required_capabilities):
                    out.add(f.feature_id)
        return out

    def generate(
        self,
        bars: pd.DataFrame,
        *,
        quotes: pd.DataFrame | None = None,
        book: pd.DataFrame | None = None,
        secondary: pd.DataFrame | None = None,
        feature_ids: list[str] | None = None,
    ) -> FeatureFrame:
        if bars.index.tz is None:
            raise ValueError("FeatureGenerator requires timezone-aware bar index")
        work = bars.sort_index()
        if not work.index.is_monotonic_increasing:
            raise ValueError("bar index must be sorted ascending")

        caps = set(self.config.available_capabilities)
        disabled = self.catalog.disable_unsupported(caps)

        parts = [
            compute_price_features(work),
            compute_liquidity_features(work),
            compute_volatility_features(work),
            compute_temporal_features(work, calendar=self.config.calendar),
        ]
        micro, micro_disabled = compute_microstructure_features(
            quotes=quotes, book=book, available_capabilities=caps
        )
        disabled.extend(micro_disabled)
        if not micro.empty:
            parts.append(micro)

        xasset, x_disabled = compute_cross_asset_features(work, secondary)
        disabled.extend(x_disabled)
        if not xasset.empty:
            parts.append(xasset)

        frame = pd.concat(parts, axis=1)
        # Drop duplicate columns if any
        frame = frame.loc[:, ~frame.columns.duplicated()]

        eligible = self._eligible_ids()
        if feature_ids is not None:
            requested = set(feature_ids)
            missing_caps = sorted(requested - eligible - set(frame.columns))
            keep = [c for c in frame.columns if c in requested and c in eligible]
            for fid in requested:
                if fid not in eligible:
                    disabled.append(fid)
            frame = frame[keep]
        else:
            keep = [c for c in frame.columns if c in eligible]
            frame = frame[keep]

        avail = self._bar_end(pd.DatetimeIndex(work.index))
        source = pd.Series(work.index, index=work.index, name="source_timestamp")
        # Enforce warm-up: null out first warm_up_bars per contract
        for fid in list(frame.columns):
            try:
                contract = self.catalog.get(fid)
            except KeyError:
                continue
            warm = int(contract.warm_up_bars)
            if warm > 0 and fid in frame.columns:
                frame.loc[frame.index[:warm], fid] = pd.NA

        assert_no_vault_columns(frame)

        payload = {
            "feature_set_version": FEATURE_SET_VERSION,
            "columns": list(frame.columns),
            "n_rows": int(len(frame)),
            "capabilities": sorted(c.value for c in caps),
            "calendar": (
                self.config.calendar
                if isinstance(self.config.calendar, str)
                else self.config.calendar.name
            ),
            "volume_type": self.config.volume_type,
        }
        feature_set_hash = sha256_json(payload)

        return FeatureFrame(
            values=frame,
            availability_timestamps=avail,
            source_timestamps=source,
            feature_ids=tuple(frame.columns),
            feature_set_version=FEATURE_SET_VERSION,
            feature_set_hash=feature_set_hash,
            code_hash=self.code_hash,
            disabled_features=tuple(sorted(set(disabled))),
            meta=payload,
        )

    def generate_values(self, bars: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Convenience for Future Invariance tests — values only, same columns."""
        return self.generate(bars, **kwargs).values
