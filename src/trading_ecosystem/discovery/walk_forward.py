from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from trading_ecosystem.common.contracts import Bar


@dataclass(frozen=True)
class Fold:
    name: str
    start: datetime
    end: datetime
    kind: str  # train | validate | oos | vault | vault_year


def ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _sorted_timestamps(bars_by_symbol: dict[str, list[Bar]]) -> list[datetime]:
    stamps: set[datetime] = set()
    for bars in bars_by_symbol.values():
        for b in bars:
            stamps.add(ensure_aware(b.ts))
    return sorted(stamps)


def filter_panel(
    bars_by_symbol: dict[str, list[Bar]],
    start: datetime,
    end: datetime,
) -> dict[str, list[Bar]]:
    start = ensure_aware(start)
    end = ensure_aware(end)
    out: dict[str, list[Bar]] = {}
    for sym, bars in bars_by_symbol.items():
        out[sym] = [b for b in bars if start <= ensure_aware(b.ts) <= end]
    return out


def ratio_folds(
    bars_by_symbol: dict[str, list[Bar]],
    train_ratio: float = 0.60,
    validate_ratio: float = 0.20,
    holdout_ratio: float = 0.20,
    min_bars_per_fold: int = 60,
    *,
    vault_ratio: float = 0.0,
    seal_vault: bool = True,
) -> list[Fold]:
    """
    Chronological splits.

    When seal_vault=True and vault_ratio>0, the final vault slice is kind='vault'
    and must NOT enter discovery fitness (promotion-only).

    Research window is train + validate (+ optional extra research OOS named kind='oos').
    Do not label research slices as 'holdout' — that name is retired to avoid confusion
    with the sealed Validation Vault.
    """
    stamps = _sorted_timestamps(bars_by_symbol)
    n = len(stamps)
    if n < min_bars_per_fold * 2:
        return []

    vr_vault = max(0.0, float(vault_ratio))
    if seal_vault and vr_vault > 0:
        # Research window uses train+validate(+optional research oos) before vault.
        research_extra = max(0.0, float(holdout_ratio))
        total = train_ratio + validate_ratio + research_extra + vr_vault
        tr = train_ratio / total
        vr = validate_ratio / total
        hr = research_extra / total
        i1 = max(min_bars_per_fold, int(n * tr))
        i2 = max(i1 + min_bars_per_fold // 2, int(n * (tr + vr)))
        i3 = max(i2 + 1, int(n * (tr + vr + hr))) if research_extra > 0 else i2
        i3 = min(i3, n - max(1, min_bars_per_fold // 2))
        if i1 >= i2 or i3 >= n:
            return []
        folds = [
            Fold("train", stamps[0], stamps[i1 - 1], "train"),
            Fold("validate", stamps[i1], stamps[i2 - 1], "validate"),
        ]
        if research_extra > 0 and i3 > i2:
            folds.append(Fold("research_oos", stamps[i2], stamps[i3 - 1], "oos"))
            vault_start_i = i3
        else:
            vault_start_i = i2
        folds.append(Fold("vault", stamps[vault_start_i], stamps[-1], "vault"))
        return folds

    total = train_ratio + validate_ratio + holdout_ratio
    tr = train_ratio / total
    vr = validate_ratio / total
    i1 = max(min_bars_per_fold, int(n * tr))
    i2 = max(i1 + min_bars_per_fold // 2, int(n * (tr + vr)))
    i2 = min(i2, n - max(1, min_bars_per_fold // 2))
    if i1 >= i2 or i2 >= n:
        return []
    return [
        Fold("train", stamps[0], stamps[i1 - 1], "train"),
        Fold("validate", stamps[i1], stamps[i2 - 1], "validate"),
        # Legacy third slice: research OOS (not a vault). Kind must be fitness-eligible.
        Fold("research_oos", stamps[i2], stamps[-1], "oos"),
    ]


def yearly_oos_folds(
    bars_by_symbol: dict[str, list[Bar]],
    min_bars_per_fold: int = 60,
) -> list[Fold]:
    """Expanding train, next calendar year as OOS — no future leakage into train."""
    stamps = _sorted_timestamps(bars_by_symbol)
    if not stamps:
        return []
    years = sorted({s.year for s in stamps})
    if len(years) < 2:
        return []
    folds: list[Fold] = []
    for i in range(1, len(years)):
        oos_year = years[i]
        train_end_year = years[i - 1]
        train_stamps = [s for s in stamps if s.year <= train_end_year]
        oos_stamps = [s for s in stamps if s.year == oos_year]
        if len(train_stamps) < min_bars_per_fold or len(oos_stamps) < max(20, min_bars_per_fold // 3):
            continue
        folds.append(
            Fold(
                name=f"oos_{oos_year}",
                start=oos_stamps[0],
                end=oos_stamps[-1],
                kind="oos",
            )
        )
    return folds


def build_folds(
    bars_by_symbol: dict[str, list[Bar]],
    train_ratio: float = 0.60,
    validate_ratio: float = 0.20,
    holdout_ratio: float = 0.20,
    use_yearly_rolls: bool = True,
    min_bars_per_fold: int = 60,
    *,
    vault_ratio: float = 0.20,
    seal_vault: bool = True,
) -> list[Fold]:
    folds = ratio_folds(
        bars_by_symbol,
        train_ratio=train_ratio,
        validate_ratio=validate_ratio,
        holdout_ratio=holdout_ratio if not seal_vault else 0.0,
        min_bars_per_fold=min_bars_per_fold,
        vault_ratio=vault_ratio if seal_vault else 0.0,
        seal_vault=seal_vault,
    )
    if use_yearly_rolls:
        # Yearly OOS folds that intersect the vault window are tagged vault_overlap and
        # excluded from discovery fitness (handled in evaluate_folds by kind filter).
        vault = next((f for f in folds if f.kind == "vault"), None)
        for yf in yearly_oos_folds(bars_by_symbol, min_bars_per_fold=min_bars_per_fold):
            if vault and ensure_aware(yf.start) >= ensure_aware(vault.start):
                folds.append(Fold(yf.name + "_vault", yf.start, yf.end, "vault_year"))
            else:
                folds.append(yf)
    return folds
