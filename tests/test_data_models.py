"""Tests for app/data_models.py."""

from app.data_models import Paper, convert_ref_to_paper


def test_convert_ref_to_paper_complete_record(valid_complete_destiny_reference):
    paper = convert_ref_to_paper(valid_complete_destiny_reference)
    assert isinstance(paper, Paper)
    assert paper.authors is not None
    assert paper.title is not None
    assert paper.year is not None
    assert paper.publisher is not None


def test_convert_ref_to_paper_incomplete_record(valid_incomplete_destiny_reference):
    paper = convert_ref_to_paper(valid_incomplete_destiny_reference)
    assert isinstance(paper, Paper)
    assert paper.authors is None
