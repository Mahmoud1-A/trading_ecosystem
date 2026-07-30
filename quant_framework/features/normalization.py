"""Stateful transforms — fit only on the active training fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrainOnlyScaler:
    """
    Z-score scaler fitted exclusively on training rows.

    Calling ``transform`` before ``fit`` raises. Never fit on validation/Vault.
    """

    mean_: pd.Series | None = None
    std_: pd.Series | None = None
    fitted: bool = False

    def fit(self, train: pd.DataFrame) -> TrainOnlyScaler:
        self.mean_ = train.mean(numeric_only=True)
        self.std_ = train.std(numeric_only=True, ddof=0).replace(0, np.nan)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.mean_ is None or self.std_ is None:
            raise RuntimeError("TrainOnlyScaler.transform called before fit on training data")
        cols = [c for c in self.mean_.index if c in frame.columns]
        out = frame.copy()
        out[cols] = (frame[cols] - self.mean_[cols]) / self.std_[cols]
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)


@dataclass
class TrainOnlyQuantileThresholds:
    """Per-column quantile thresholds fitted on training only."""

    quantiles: tuple[float, ...] = (0.05, 0.95)
    thresholds_: dict[str, tuple[float, float]] | None = None
    fitted: bool = False

    def fit(self, train: pd.DataFrame) -> TrainOnlyQuantileThresholds:
        thr: dict[str, tuple[float, float]] = {}
        for col in train.select_dtypes(include=[np.number]).columns:
            lo, hi = np.nanquantile(train[col].to_numpy(dtype=float), self.quantiles)
            thr[str(col)] = (float(lo), float(hi))
        self.thresholds_ = thr
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.thresholds_ is None:
            raise RuntimeError("TrainOnlyQuantileThresholds.transform before fit")
        out = frame.copy()
        for col, (lo, hi) in self.thresholds_.items():
            if col in out.columns:
                out[col] = out[col].clip(lo, hi)
        return out
