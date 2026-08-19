"""Tests for PairScoreResult and get_library_info."""

import pytest
from destiny_sdk.identifiers import DOIIdentifier

from destiny_deduper.data_models import Paper
from destiny_deduper.dedupe import Deduper
from destiny_deduper.pair_score_result import (
    EarlyStopReason,
    FieldStatus,
    PairLabel,
    get_library_info,
)


@pytest.fixture
def duplicate_pair():
    """Two Papers that are very likely duplicates."""
    paper_a = Paper(
        title="Systematic review of hand hygiene compliance in hospitals",
        year=2019,
        journal="Journal of Hospital Infection",
        pages="1-10",
        volume="110",
        authors=None,
    )
    paper_b = Paper(
        title="Systematic review of hand hygiene compliance in hospitals",
        year=2019,
        journal="Journal of Hospital Infection",
        pages="1-10",
        volume="110",
        authors=None,
    )
    return paper_a, paper_b


@pytest.fixture
def non_duplicate_pair():
    """Two Papers with unrelated titles that should not be duplicates."""
    paper_a = Paper(
        title="Machine learning in oncology: a review",
        year=2020,
        journal="Cancer Biology",
        pages="50-60",
    )
    paper_b = Paper(
        title="Economic impacts of climate change on agriculture",
        year=2015,
        journal="Environmental Economics",
        pages="1-20",
    )
    return paper_a, paper_b


@pytest.fixture
def early_stop_pair():
    """Two papers with the same title but confirmed DOI mismatch plus journal/pages mismatch."""
    doi_a = DOIIdentifier(identifier="10.1000/xyz123")
    doi_b = DOIIdentifier(identifier="10.9999/abc999")
    paper_a = Paper(
        title="The effects of exercise on cardiovascular health",
        doi=doi_a,
        journal="Cardiology Today",
        pages="1-5",
        year=2018,
    )
    paper_b = Paper(
        title="The effects of exercise on cardiovascular health",
        doi=doi_b,
        journal="Sports Medicine Review",
        pages="100-110",
        year=2020,
    )
    return paper_a, paper_b


@pytest.fixture
def doi_mismatch_pair():
    """Two Papers with very similar content but different DOIs — no early stop."""
    doi_a = DOIIdentifier(identifier="10.1000/aaa111")
    doi_b = DOIIdentifier(identifier="10.1000/aaa222")
    paper_a = Paper(
        title="Impact of sleep deprivation on cognitive performance",
        year=2021,
        journal="Sleep Medicine",
        pages="10-20",
        doi=doi_a,
    )
    paper_b = Paper(
        title="Impact of sleep deprivation on cognitive performance",
        year=2021,
        journal="Sleep Medicine",
        pages="10-20",
        doi=doi_b,
    )
    return paper_a, paper_b


def test_score_pair_normal_scoring_returns_pair_score_result(duplicate_pair):
    paper_a, paper_b = duplicate_pair
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    assert result.early_stop_reason is None
    assert 0.0 <= result.probability <= 1.0
    assert isinstance(result.doi_mismatch_adjustment_applied, bool)
    assert result.label in (PairLabel.DUPLICATE, PairLabel.NOT_DUPLICATE)


def test_score_pair_normal_scoring_has_field_results(duplicate_pair):
    paper_a, paper_b = duplicate_pair
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    assert len(result.field_results) > 0
    for field_result in result.field_results.values():
        assert field_result.status in FieldStatus.__members__.values()

    pages_result = result.field_results["pages"]
    assert pages_result.normalised_value_a == "1-10"
    assert pages_result.normalised_value_b == "1-10"
    assert pages_result.score == 1.0
    assert "intercept" not in result.field_results


def test_score_pair_duplicate_label(duplicate_pair):
    """Papers with identical titles and metadata should score as duplicates."""
    paper_a, paper_b = duplicate_pair
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    assert result.label == PairLabel.DUPLICATE
    assert result.probability >= 0.85


def test_score_pair_early_stop(early_stop_pair):
    """Early stop should produce probability=0, no field results, stable reason."""
    paper_a, paper_b = early_stop_pair
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    assert result.early_stop_reason is not None
    assert result.early_stop_reason in EarlyStopReason.__members__.values()
    assert result.probability == 0.0
    assert result.field_results == {}
    assert result.label == PairLabel.NOT_DUPLICATE


def test_score_pair_missing_field():
    """A field present on one paper but absent on the other should have MISSING_A/B status."""
    paper_a = Paper(
        title="The role of gut microbiome in mental health",
        year=2022,
        journal="Gut",
        pages="30-40",
    )
    paper_b = Paper(
        title="The role of gut microbiome in mental health",
        year=2022,
        # journal is absent
        pages="30-40",
    )
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    journal_result = result.field_results.get("journal")
    assert journal_result is not None
    assert journal_result.status == FieldStatus.MISSING_B
    assert journal_result.score is None


def test_score_pair_mismatched_field():
    """Fields that are compared and differ should have status=COMPARED with low score."""
    paper_a = Paper(
        title="Overview of neuroplasticity mechanisms",
        year=2018,
        journal="Neuroscience Letters",
        pages="1-8",
    )
    paper_b = Paper(
        title="Overview of neuroplasticity mechanisms",
        year=2018,
        journal="Brain Research Bulletin",  # clearly different journal
        pages="1-8",
    )
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    journal_result = result.field_results.get("journal")
    assert journal_result is not None
    assert journal_result.status == FieldStatus.COMPARED
    assert journal_result.score is not None
    assert journal_result.score < 1.0


def test_score_pair_doi_mismatch_adjustment(doi_mismatch_pair):
    """When DOIs differ and no early stop, the DOI mismatch flag should be True."""
    paper_a, paper_b = doi_mismatch_pair
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    assert result.early_stop_reason is None
    assert result.doi_mismatch_adjustment_applied is True


def test_score_pair_no_doi_mismatch_flag_when_no_doi():
    """When neither paper has a DOI, no penalty is applied."""
    paper_a = Paper(title="Biodiversity in tropical forests", year=2020)
    paper_b = Paper(title="Biodiversity in tropical forests", year=2020)
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    result = deduper.score_pair(paper_a, paper_b)

    assert result.doi_mismatch_adjustment_applied is False


def test_get_library_info_returns_library_info():
    info = get_library_info()

    assert isinstance(info.package_version, str)
    assert isinstance(info.decision_threshold, float)
    assert 0.0 < info.decision_threshold < 1.0
    assert isinstance(info.scoring_config, dict)
    assert isinstance(info.config_hash, str)
    assert len(info.config_hash) == 64  # SHA-256 hex digest


def test_get_library_info_threshold_matches_config():
    from destiny_deduper.config import get_settings

    info = get_library_info()
    assert info.decision_threshold == get_settings().decision_threshold


def test_get_library_info_scoring_config_has_weights_and_thresholds():
    info = get_library_info()

    assert "weights" in info.scoring_config
    assert "thresholds" in info.scoring_config
    assert "decision_threshold" in info.scoring_config


def test_get_library_info_is_stable():
    """Calling get_library_info twice returns the same object (cached)."""
    info1 = get_library_info()
    info2 = get_library_info()

    assert info1 is info2
    assert info1.config_hash == info2.config_hash
