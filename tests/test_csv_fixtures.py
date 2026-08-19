"""
Fixture-driven tests for score_pair over records loaded from a CSV.

Covers only the pairs marked ``csv`` in tests/fixtures/pairs.csv. The rest need
fields the default CSV columns cannot carry, so they are Paper-only.
"""

from destiny_dedupe.dedupe import Deduper
from destiny_dedupe.normalisers import normalise_doi

# The fields the scorer weights. A CSV round-trip that loses any of them changes
# the score, which is the failure this comparison is here to catch.
SCORED_FIELDS = ("doi", "title", "authors", "year", "journal", "pages", "issue")


def _comparable(paper, field) -> object:
    value = getattr(paper, field)
    # The CSV import normalises DOIs and direct construction does not, so the
    # two shapes only agree on the normalised form.
    if field == "doi" and value is not None:
        return normalise_doi(value.identifier)
    return value


def test_csv_loaded_pair_matches_fixture(csv_fixture_pair, csv_fixture_papers):
    record_a = csv_fixture_papers[int(csv_fixture_pair["record_id_a"])]
    record_b = csv_fixture_papers[int(csv_fixture_pair["record_id_b"])]

    result = Deduper(record_a, [record_b]).score_pair(record_a, record_b)

    assert result.label == csv_fixture_pair["expected_label"]
    assert (result.early_stop_reason or "") == csv_fixture_pair[
        "expected_early_stop_reason"
    ]
    assert (
        str(result.doi_mismatch_adjustment_applied).lower()
        == csv_fixture_pair["expected_doi_mismatch_adjustment"]
    )
    assert f"{result.probability:.6f}" == csv_fixture_pair["expected_probability"]


def test_csv_import_reproduces_the_paper_records(
    csv_fixture_pair, fixture_papers, csv_fixture_papers
):
    for side in ("a", "b"):
        record_id = int(csv_fixture_pair[f"record_id_{side}"])
        from_csv = csv_fixture_papers[record_id]
        direct = fixture_papers[record_id]

        for field in SCORED_FIELDS:
            assert _comparable(from_csv, field) == _comparable(
                direct, field
            ), f"record {record_id} field {field}"
