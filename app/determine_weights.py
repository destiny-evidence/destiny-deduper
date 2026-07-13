"""
Backward compatibility layer for refactored algorithm development functions.

This module re-exports functions from algorithm_development and candidate_selection
for backward compatibility with existing notebooks. New code should import directly
from those modules.

Deprecated: Use algorithm_development and candidate_selection modules instead.
"""

# Re-export for backward compatibility
from app.algorithm_development import (
    DEDUPE_FIELDS,
    build_record_cache,
    compare_target_fields,
    read_process_data_from_file,
    score_pairs_with_early_stop,
)
from app.candidate_selection import (
    BLOCK_RULES,
    build_blocked_pairs,
    normalise_block_value,
)
from app.data_models import GoldStandardPaper

__all__ = [
    "BLOCK_RULES",
    "DEDUPE_FIELDS",
    "GoldStandardPaper",
    "build_blocked_pairs",
    "build_record_cache",
    "compare_target_fields",
    "normalise_block_value",
    "read_process_data_from_file",
    "score_pairs_with_early_stop",
]
