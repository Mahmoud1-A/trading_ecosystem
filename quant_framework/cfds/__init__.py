"""CFD research instrument, source, and execution-target profiles (Phase 12.2)."""

from cfds.cost_policy import (
    CostModelAssessment,
    CostModelEligibility,
    ModeledCostSpec,
    ObservedSpreadStats,
    assess_cost_model,
    derive_observed_spread,
)
from cfds.execution_profile import (
    ComparisonWarning,
    ExecutionTargetProfile,
    PaperEligibilityDecision,
    compare_source_to_target,
    evaluate_paper_eligibility,
    generic_prop_target_profile,
)
from cfds.financing_profile import (
    CFDFinancingProfile,
    dukascopy_us500_financing_profile,
    unknown_financing_profile,
)
from cfds.instrument_spec import (
    CFDCommissionModel,
    CFDResearchInstrument,
    CFDSlippageModel,
    MaintenanceWindow,
    TradingSession,
    default_us500_cfd_instrument,
)
from cfds.prop_profile import GenericCFDPropProfile, default_generic_prop_profile
from cfds.source_profile import (
    HistoricalSourceProfile,
    RetrievalMethod,
    dukascopy_us500_source_profile,
)
from cfds.symbol_mapping import (
    DUKASCOPY_US500,
    US500_CFD,
    SymbolMapping,
    SymbolMappingError,
    assert_not_futures_identifier,
    get_symbol_mapping,
    list_symbol_mappings,
    with_target_broker,
)

__all__ = [
    "CFDCommissionModel",
    "CFDFinancingProfile",
    "CFDResearchInstrument",
    "CFDSlippageModel",
    "ComparisonWarning",
    "CostModelAssessment",
    "CostModelEligibility",
    "DUKASCOPY_US500",
    "ExecutionTargetProfile",
    "GenericCFDPropProfile",
    "HistoricalSourceProfile",
    "MaintenanceWindow",
    "ModeledCostSpec",
    "ObservedSpreadStats",
    "PaperEligibilityDecision",
    "RetrievalMethod",
    "SymbolMapping",
    "SymbolMappingError",
    "TradingSession",
    "US500_CFD",
    "assert_not_futures_identifier",
    "assess_cost_model",
    "compare_source_to_target",
    "default_generic_prop_profile",
    "default_us500_cfd_instrument",
    "derive_observed_spread",
    "dukascopy_us500_financing_profile",
    "dukascopy_us500_source_profile",
    "evaluate_paper_eligibility",
    "generic_prop_target_profile",
    "get_symbol_mapping",
    "list_symbol_mappings",
    "unknown_financing_profile",
    "with_target_broker",
]
