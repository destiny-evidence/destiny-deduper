"""Tests for app/dedupe.py."""

import pytest
from destiny_sdk.identifiers import DOIIdentifier, PubMedIdentifier
from loguru import logger

from destiny_dedupe.data_models import Paper
from destiny_dedupe.dedupe import Deduper


@pytest.fixture
def paper_with_doi(valid_complete_destiny_reference):
    return Paper(
        doi=valid_complete_destiny_reference.identifiers[0],
        title="A Study on Deduplication",
        year=2020,
        publisher="Test Publisher",
    )


@pytest.fixture
def paper_with_different_doi():
    # Change the DOI identifier to ensure it's different
    doi = DOIIdentifier(identifier="10.9999/otherdoi")
    return Paper(
        doi=doi, title="Who wins? ", year=2023, publisher="Grandmas Publishing House."
    )


def test_compare_doi_exact_match(paper_with_doi):
    deduper = Deduper(reference=paper_with_doi, candidates=[paper_with_doi])
    score = deduper.compare_doi(
        paper_with_doi, paper_with_doi, string_distance_algorithm="jaro_winkler"
    )
    assert score.score == 1.0


def test_compare_doi_no_match(paper_with_doi, paper_with_different_doi):
    deduper = Deduper(reference=paper_with_doi, candidates=[paper_with_different_doi])
    logger.debug(paper_with_doi)
    logger.debug(paper_with_different_doi)
    score = deduper.compare_doi(
        paper_with_doi,
        paper_with_different_doi,
        string_distance_algorithm="jaro_winkler",
    )
    assert 0.0 <= score.score < 1.0


def test_compare_doi_none_returns_zero(paper_with_doi):
    paper_no_doi = Paper(title="No DOI Paper")
    deduper = Deduper(reference=paper_with_doi, candidates=[paper_no_doi])
    score = deduper.compare_doi(paper_with_doi, paper_no_doi)
    assert score.score == 0.0


def test_compare_isbn_exact_match():
    paper1 = Paper(isbn="8537809667")
    paper2 = Paper(isbn="8537809667")
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_isbn(paper1, paper2).score == 1.0


def test_compare_isbn_no_match():
    paper1 = Paper(isbn="9781453886328")
    paper2 = Paper(isbn="8537809667")
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_isbn(paper1, paper2).score == 0.0


def test_compare_isbn_none_returns_zero():
    paper1 = Paper(title="Missing ISBN means more cites.", isbn=None)
    paper2 = Paper(isbn="8537809667")
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_isbn(paper1, paper2).score == 0.0


def test_compare_year_exact_match():
    paper1 = Paper(year=2022)
    paper2 = Paper(year=2022)
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_year(paper1, paper2).score == 1.0


def test_compare_year_no_match():
    paper1 = Paper(year=2022)
    paper2 = Paper(year=2021)
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_year(paper1, paper2).score == 0.0


def test_compare_year_none_returns_zero():
    paper1 = Paper(title="Missing years mean more cites.", year=None)
    paper2 = Paper(year=2022)
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_year(paper1, paper2).score == 0.0


def test_compare_pubmed_id_exact_match():
    paper1 = Paper(pubmed_id=PubMedIdentifier(identifier=12345))
    paper2 = Paper(pubmed_id=PubMedIdentifier(identifier=12345))
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_pubmed_id(paper1, paper2).score == 1.0


def test_compare_pubmed_id_no_match():
    paper1 = Paper(pubmed_id=PubMedIdentifier(identifier=12345))
    paper2 = Paper(pubmed_id=PubMedIdentifier(identifier=56789))
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_pubmed_id(paper1, paper2).score == 0.0


def test_compare_pubmed_id_none_returns_zero():
    paper1 = Paper(pubmed_id=PubMedIdentifier(identifier=12345))
    paper2 = Paper(title="Missing pubmeds mean more citations", pubmed_id=None)
    deduper = Deduper(reference=paper1, candidates=[paper2])
    assert deduper.compare_pubmed_id(paper1, paper2).score == 0.0


def test_compare_one_to_many_returns_list(paper_with_doi, paper_with_different_doi):
    deduper = Deduper(
        reference=paper_with_doi, candidates=[paper_with_doi, paper_with_different_doi]
    )
    results = deduper.compare_one_to_many()
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(x, float) for x in results)


def test_compare_title_exact_match(paper_with_doi):
    deduper = Deduper(reference=paper_with_doi, candidates=[paper_with_doi])
    score = deduper.compare_title(paper_with_doi, paper_with_doi)
    assert score.score == 1.0


def test_compare_authors_no_authors_returns_zero(paper_with_doi):
    # paper_with_doi has no authors field set
    deduper = Deduper(reference=paper_with_doi, candidates=[paper_with_doi])
    score = deduper.compare_authors(paper_with_doi, paper_with_doi)
    assert score.score == 0.0
