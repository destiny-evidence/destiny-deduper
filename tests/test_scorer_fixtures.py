"""
Fixture-driven tests for score_pair over the Paper input shape.

One case per pair in tests/fixtures/pairs.csv. See that directory's README for
what each pair targets and how the expected values were produced.
"""

from destiny_dedupe.dedupe import Deduper
from destiny_dedupe.early_stop import EARLY_STOP_RULES
from destiny_dedupe.pair_score_result import EarlyStopReason

# Neither is reachable: the first is a subset of doi_and_pages_mismatch, which
# is checked before it, and no rule produces the second at all.
UNREACHABLE_EARLY_STOP_REASONS = {
    EarlyStopReason.EXACT_TITLE_WITH_STRUCTURAL_CONFLICT,
}


def test_scored_pair_matches_fixture(fixture_pair, fixture_papers):
    record_a = fixture_papers[int(fixture_pair["record_id_a"])]
    record_b = fixture_papers[int(fixture_pair["record_id_b"])]

    result = Deduper(record_a, [record_b]).score_pair(record_a, record_b)

    assert result.label == fixture_pair["expected_label"]
    assert (result.early_stop_reason or "") == fixture_pair[
        "expected_early_stop_reason"
    ]
    assert (
        str(result.doi_mismatch_adjustment_applied).lower()
        == fixture_pair["expected_doi_mismatch_adjustment"]
    )
    assert f"{result.probability:.6f}" == fixture_pair["expected_probability"]


def test_fixture_set_covers_every_reachable_early_stop_reason(fixture_pairs):
    covered = {pair["expected_early_stop_reason"] for pair in fixture_pairs} - {""}
    reachable = {reason.value for reason in EarlyStopReason} - {
        reason.value for reason in UNREACHABLE_EARLY_STOP_REASONS
    }

    assert covered == reachable


def test_every_early_stop_rule_reason_is_an_enum_member():
    # An unknown reason is downgraded to None with only a warning, so the pair
    # still early stops but the caller cannot tell why.
    rule_reasons = {rule.reason for rule in EARLY_STOP_RULES}

    assert rule_reasons <= {reason.value for reason in EarlyStopReason}
