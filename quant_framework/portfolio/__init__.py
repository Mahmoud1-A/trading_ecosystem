"""Portfolio construction — Phase 7."""

from portfolio.candidate_pool import CandidateFunnelStage, CandidatePool, PoolMember
from portfolio.constraints import ConstraintViolation, MemberMeta, PortfolioConstraints, validate_weights
from portfolio.correlation import (
    CorrelationReport,
    build_correlation_matrix,
    correlation_penalty,
    pnl_correlation,
    symbol_overlap,
)
from portfolio.marginal_contribution import marginal_contribution, portfolio_quality, risk_parity_weights
from portfolio.optimizer import ConstructionMethod, OptimizerResult, PortfolioOptimizer, UnstableAllocationError
from portfolio.promotion import PortfolioPromoter, PortfolioVaultError
from portfolio.risk_budget import RiskBudget
from portfolio.stress import PORTFOLIO_STRESS_SCENARIOS, PortfolioStressResult, PortfolioStressTester
from portfolio.versioning import (
    PortfolioPipelineStage,
    PortfolioVersion,
    PortfolioVersionError,
    build_portfolio_version,
    freeze_portfolio,
    mutate_frozen_weights,
)

__all__ = [
    "PORTFOLIO_STRESS_SCENARIOS",
    "CandidateFunnelStage",
    "CandidatePool",
    "ConstructionMethod",
    "ConstraintViolation",
    "CorrelationReport",
    "MemberMeta",
    "OptimizerResult",
    "PoolMember",
    "PortfolioConstraints",
    "PortfolioOptimizer",
    "PortfolioPipelineStage",
    "PortfolioPromoter",
    "PortfolioStressResult",
    "PortfolioStressTester",
    "PortfolioVaultError",
    "PortfolioVersion",
    "PortfolioVersionError",
    "RiskBudget",
    "UnstableAllocationError",
    "build_correlation_matrix",
    "build_portfolio_version",
    "correlation_penalty",
    "freeze_portfolio",
    "marginal_contribution",
    "mutate_frozen_weights",
    "pnl_correlation",
    "portfolio_quality",
    "risk_parity_weights",
    "symbol_overlap",
    "validate_weights",
]
