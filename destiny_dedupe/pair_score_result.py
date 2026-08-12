"""Structured result types for pair scoring and library configuration."""

import hashlib
import json
from enum import StrEnum
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pydantic import BaseModel, ConfigDict, Field

from destiny_dedupe.config import get_settings


class EarlyStopReason(StrEnum):
    """Stable enum of reasons why pair scoring stopped early."""

    DOI_AND_PAGES_MISMATCH = "doi_and_pages_mismatch"
    DOI_PUB_VERSION_MISMATCH = "doi_pub_version_mismatch"
    PART_NUMBER_MISMATCH = "part_number_mismatch"
    PARTIAL_RATIO_TOO_LOW = "partial_ratio_too_low"
    EXACT_TITLE_WITH_STRUCTURAL_CONFLICT = "exact_title_with_structural_conflict"
    TITLE_WITH_METADATA_MISMATCH = "title_with_metadata_mismatch"
    YEAR_GAP_WITH_ABSTRACT_CONFLICT = "year_gap_with_abstract_conflict"


class FieldStatus(StrEnum):
    """Status of a field comparison for a scored pair."""

    COMPARED = "compared"
    MISSING_A = "missing_a"
    MISSING_B = "missing_b"
    MISSING_BOTH = "missing_both"


class PairLabel(StrEnum):
    """Suggested label for a scored pair."""

    DUPLICATE = "duplicate"
    NOT_DUPLICATE = "not_duplicate"
    UNSCORABLE = "unscorable"


class FieldResult(BaseModel):
    """Result of comparing a single field for a scored pair."""

    model_config = ConfigDict(frozen=True)

    status: FieldStatus
    value_a: str | None = None
    value_b: str | None = None
    score: float | None = None


class PairScoreResult(BaseModel):
    """Structured result of scoring a single candidate pair."""

    model_config = ConfigDict(frozen=True)

    probability: float
    doi_mismatch_adjustment_applied: bool
    field_results: dict[str, FieldResult] = Field(default_factory=dict)
    early_stop_reason: EarlyStopReason | None = None
    label: PairLabel
    unscorable_reason: str | None = None


class LibraryInfo(BaseModel):
    """Library-level configuration exposed once at load time."""

    model_config = ConfigDict(frozen=True)

    package_version: str
    decision_threshold: float
    scoring_config: dict[str, object]
    config_hash: str


@lru_cache(maxsize=1)
def get_library_info() -> LibraryInfo:
    """
    Return library-level configuration.

    Intended to be called once when the library is loaded. The result is
    cached so subsequent calls are free. destiny-repository should store this
    information alongside its decisions.

    Returns:
        LibraryInfo containing the package version, decision threshold,
        effective scoring configuration, and a stable configuration hash.

    """
    settings = get_settings()

    try:
        pkg_version = _pkg_version("deduplication-toolkit")
    except PackageNotFoundError:
        raise PackageNotFoundError("Package 'deduplication-toolkit' is not installed.")

    scoring_config: dict[str, object] = {
        "weights": settings.weights.model_dump(),
        "thresholds": settings.thresholds.model_dump(),
        "decision_threshold": settings.decision_threshold,
    }

    config_json = json.dumps(scoring_config, sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()

    return LibraryInfo(
        package_version=pkg_version,
        decision_threshold=settings.decision_threshold,
        scoring_config=scoring_config,
        config_hash=config_hash,
    )
