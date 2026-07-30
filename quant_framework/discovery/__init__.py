"""Typed Strategy DSL + Autonomous Alpha Miner — Phases 6C/6D.

Never executes arbitrary candidate Python. Never accesses Vault data during search.
"""

from discovery.behavioral_dedup import (
    BehaviorCluster,
    BehaviorSignature,
    BehavioralDeduper,
    behavioral_similarity,
    signature_from_record,
)
from discovery.candidate import StrategyCandidate, build_candidate, collect_features
from discovery.canonicalization import canonicalize_node, canonical_payload
from discovery.complexity import complexity_score
from discovery.crossover import Crossover, CrossoverError
from discovery.evaluator import (
    BacktestBackend,
    CandidateEvaluator,
    EvalOutcome,
    EvaluationRecord,
    SyntheticOOSBackend,
)
from discovery.expression_tree import (
    DSLValidationError,
    ExprNode,
    constant_node,
    eval_scalar,
    feature_node,
    op_node,
    parameter_node,
    protected_divide,
)
from discovery.fitness import (
    RANKING_SOURCE_OOS,
    FitnessError,
    FitnessResult,
    FoldOOSMetrics,
    RobustFitness,
    SCORE_QUALIFIED_REJECT_REASONS,
    assert_oos_only_promotion,
)
from discovery.family_catalog import DEFAULT_FAMILY_ORDER, FAMILY_BLUEPRINTS
from discovery.family_generator import StrategyFamilyGenerator, materialize_family_spec
from discovery.family_spec import (
    DuplicateFamilyError,
    FamilySpec,
    HomogeneousFamilyGrammarError,
    assert_diverse_family_grammars,
    dedupe_family_specs,
)
from discovery.freezing import FreezeError, FrozenCandidate, freeze_candidate
from discovery.generator import CandidateGenerator
from discovery.grammar import GRAMMAR_VERSION, FeatureLeaf, Grammar, GrammarLimits
from discovery.lineage import legacy_import_candidate, lineage_from_candidate
from discovery.multi_family_campaign import (
    FamilyCampaignConfig,
    FamilyStats,
    MultiFamilyCampaign,
    MultiFamilyCampaignResult,
    adaptive_reallocate_wfo,
    equal_initial_allocation,
    family_campaign_from_config,
)
from discovery.mutation import Mutator
from discovery.operators import FORBIDDEN_BUILTINS, OPERATOR_REGISTRY, OperatorId, OperatorSpec
from discovery.parameter_robustness import ParameterRobustness, RobustnessResult
from discovery.portfolio_candidates import PortfolioCandidate, PortfolioCandidatePool
from discovery.prechecks import PrecheckResult, structural_precheck, validate_limits
from discovery.promotion import PromotionDecision, PromotionGate, PromotionStatus
from discovery.repair import repair_entry, truncate_to_limits
from discovery.search_budget import (
    SEARCH_BUDGET_VERSION,
    BudgetCounters,
    SearchBudget,
    resolve_bucket_caps,
    search_budget_from_config,
)
from discovery.search_controller import DiscoveryRunResult, SearchController
from discovery.selection import DiverseSelector, ScoredCandidate
from discovery.stress import STRESS_SCENARIOS, StressResult, StressTester
from discovery.types import CreationMethod, NodeKind, ValueType
from discovery.vault_gateway import MinerVaultError, MinerVaultGateway, VaultSubmissionRequest

__all__ = [
    "OPERATOR_REGISTRY",
    "FORBIDDEN_BUILTINS",
    "GRAMMAR_VERSION",
    "RANKING_SOURCE_OOS",
    "SEARCH_BUDGET_VERSION",
    "STRESS_SCENARIOS",
    "DEFAULT_FAMILY_ORDER",
    "FAMILY_BLUEPRINTS",
    "BacktestBackend",
    "BehaviorCluster",
    "BehaviorSignature",
    "BehavioralDeduper",
    "BudgetCounters",
    "CandidateEvaluator",
    "CandidateGenerator",
    "CreationMethod",
    "Crossover",
    "CrossoverError",
    "DSLValidationError",
    "DiscoveryRunResult",
    "DiverseSelector",
    "DuplicateFamilyError",
    "EvalOutcome",
    "EvaluationRecord",
    "ExprNode",
    "FamilyCampaignConfig",
    "FamilySpec",
    "FamilyStats",
    "FeatureLeaf",
    "FitnessError",
    "FitnessResult",
    "FoldOOSMetrics",
    "FreezeError",
    "FrozenCandidate",
    "Grammar",
    "GrammarLimits",
    "HomogeneousFamilyGrammarError",
    "MinerVaultError",
    "MinerVaultGateway",
    "MultiFamilyCampaign",
    "MultiFamilyCampaignResult",
    "Mutator",
    "NodeKind",
    "OperatorId",
    "OperatorSpec",
    "ParameterRobustness",
    "PortfolioCandidate",
    "PortfolioCandidatePool",
    "PrecheckResult",
    "PromotionDecision",
    "PromotionGate",
    "PromotionStatus",
    "RobustFitness",
    "RobustnessResult",
    "SCORE_QUALIFIED_REJECT_REASONS",
    "ScoredCandidate",
    "SearchBudget",
    "SearchController",
    "StrategyCandidate",
    "StrategyFamilyGenerator",
    "StressResult",
    "StressTester",
    "SyntheticOOSBackend",
    "ValueType",
    "VaultSubmissionRequest",
    "adaptive_reallocate_wfo",
    "assert_diverse_family_grammars",
    "assert_oos_only_promotion",
    "behavioral_similarity",
    "build_candidate",
    "canonicalize_node",
    "canonical_payload",
    "collect_features",
    "complexity_score",
    "constant_node",
    "dedupe_family_specs",
    "equal_initial_allocation",
    "eval_scalar",
    "family_campaign_from_config",
    "feature_node",
    "freeze_candidate",
    "legacy_import_candidate",
    "lineage_from_candidate",
    "materialize_family_spec",
    "op_node",
    "parameter_node",
    "protected_divide",
    "repair_entry",
    "resolve_bucket_caps",
    "search_budget_from_config",
    "signature_from_record",
    "structural_precheck",
    "truncate_to_limits",
    "validate_limits",
]
