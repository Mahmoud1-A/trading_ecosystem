"""Legacy system bridge — read-only archive, forced reevaluation (Phase 7)."""

from legacy.candidate_adapter import FORBIDDEN_INHERITED_FIELDS, adapt_legacy_record
from legacy.importer import LegacyImporter
from legacy.migration_report import FUNNEL_STAGES, MigrationReport
from legacy.records import LEGACY_BOOK_SIZE, LegacyStrategyRecord, make_legacy_book
from legacy.result_mapper import MappedTrial, map_reevaluation

__all__ = [
    "FORBIDDEN_INHERITED_FIELDS",
    "FUNNEL_STAGES",
    "LEGACY_BOOK_SIZE",
    "LegacyImporter",
    "LegacyStrategyRecord",
    "MappedTrial",
    "MigrationReport",
    "adapt_legacy_record",
    "make_legacy_book",
    "map_reevaluation",
]
