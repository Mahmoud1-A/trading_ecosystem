"""Validation package — WFO, vault, lineage, purge/embargo."""

from validation.bootstrap import block_bootstrap_mean, prop_breach_from_daily
from validation.candidate_lineage import CandidateLineage
from validation.event_driven_wfo import (
    EventDrivenWFOResult,
    EventFoldResult,
    EventWFOContext,
    RankingSourceError,
    WindowRun,
    proxy_metric_evaluate_fn,
    run_event_driven_wfo,
    run_event_window,
)
from validation.purge_embargo import WindowSlice, apply_purge_embargo, assert_no_overlap
from validation.vault import ValidationVault, VaultAccessError
from validation.walk_forward import (
    FoldResult,
    FoldSpec,
    WalkForwardResult,
    expand_param_grid,
    generate_rolling_folds,
    run_walk_forward,
)

__all__ = [
    "CandidateLineage",
    "EventDrivenWFOResult",
    "EventFoldResult",
    "EventWFOContext",
    "FoldResult",
    "FoldSpec",
    "RankingSourceError",
    "ValidationVault",
    "VaultAccessError",
    "WalkForwardResult",
    "WindowRun",
    "WindowSlice",
    "apply_purge_embargo",
    "assert_no_overlap",
    "block_bootstrap_mean",
    "expand_param_grid",
    "generate_rolling_folds",
    "prop_breach_from_daily",
    "proxy_metric_evaluate_fn",
    "run_event_driven_wfo",
    "run_event_window",
]
