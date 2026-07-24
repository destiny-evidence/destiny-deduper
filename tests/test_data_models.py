from typing import Any

import pytest
from destiny_sdk.identifiers import DOIIdentifier, ExternalIdentifierType
from destiny_sdk.references import ReferenceFileInput
from pydantic import ValidationError

from app.data_models import Paper, convert_ref_to_paper


def test_convert_ref_to_paper_complete_record(valid_complete_destiny_reference):
    paper = convert_ref_to_paper(valid_complete_destiny_reference)
    ref = valid_complete_destiny_reference
    assert isinstance(paper, Paper)
    assert paper.authors is not None
    assert paper.title is not None
    assert paper.year is not None
    assert paper.publisher is not None

    bib = next(
        (
            e.content
            for e in ref.enhancements or []
            if getattr(e.content, "enhancement_type", None) == "bibliographic"
        ),
        None,
    )
    assert paper.doi == ref.identifiers[0]
    assert paper.title == getattr(bib, "title", None)
    assert paper.authors == getattr(bib, "authorship", None)
    assert paper.year == getattr(bib, "publication_year", None)
    assert paper.publisher == getattr(bib, "publisher", None)


def test_convert_ref_to_paper_incomplete_record(valid_incomplete_destiny_reference):
    paper = convert_ref_to_paper(valid_incomplete_destiny_reference)
    ref = valid_incomplete_destiny_reference
    assert isinstance(paper, Paper)
    assert paper.authors is None

    bib = next(
        (
            e.content
            for e in ref.enhancements or []
            if getattr(e.content, "enhancement_type", None) == "bibliographic"
        ),
        None,
    )
    assert paper.doi == ref.identifiers[0]
    assert paper.title == getattr(bib, "title", None)
    assert paper.year == getattr(bib, "publication_year", None)
    assert paper.publisher == getattr(bib, "publisher", None)


def test_convert_ref_string_from_file_to_paper_valid_jsonl(
    valid_reference_jsonl_string,
):
    ref = ReferenceFileInput.from_jsonl(valid_reference_jsonl_string)
    paper = convert_ref_to_paper(ref)
    assert isinstance(paper, Paper)

    bib = next(
        (
            e.content
            for e in ref.enhancements or []
            if getattr(e.content, "enhancement_type", None) == "bibliographic"
        ),
        None,
    )
    doi = next(
        (
            i
            for i in ref.identifiers or []
            if getattr(i, "identifier_type", None) == "doi"
        ),
        None,
    )
    assert paper.doi == doi
    assert paper.title == getattr(bib, "title", None)
    assert paper.authors == getattr(bib, "authorship", None)
    assert paper.year == getattr(bib, "publication_year", None)
    assert paper.publisher == getattr(bib, "publisher", None)


def test_convert_ref_file_input_to_paper_valid_ref_file_obj(
    valid_destiny_reference_file_input,
):
    paper = convert_ref_to_paper(valid_destiny_reference_file_input)
    ref = valid_destiny_reference_file_input
    assert isinstance(paper, Paper)

    bib = next(
        (
            e.content
            for e in ref.enhancements or []
            if getattr(e.content, "enhancement_type", None) == "bibliographic"
        ),
        None,
    )
    assert paper.doi == ref.identifiers[0]
    assert paper.title == getattr(bib, "title", None)
    assert paper.authors == getattr(bib, "authorship", None)
    assert paper.year == getattr(bib, "publication_year", None)
    assert paper.publisher == getattr(bib, "publisher", None)


def test_convert_ref_file_to_paper_invalid_jsonl(
    invalid_reference_jsonl_string,
):
    with pytest.raises(ValidationError):
        ReferenceFileInput.from_jsonl(invalid_reference_jsonl_string)


def test_invalid_paper_instance():
    # invalid if all fields are None
    with pytest.raises(ValidationError):
        Paper(**{})  # noqa: PIE804 - want to be explicit here


def test_invalid_isbn_in_paper_init():
    to_parse: dict[str, Any] = {
        "doi": DOIIdentifier(
            identifier="10.21759465/m2z0z61", identifier_type=ExternalIdentifierType.DOI
        ),
        "isbn": "invalid_isbn1",  # isbn should be 10 or 13 digit int, with additional rules...
    }
    with pytest.raises(ValidationError):
        Paper(**to_parse)
