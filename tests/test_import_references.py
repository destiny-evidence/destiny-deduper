# ruff: noqa: SLF001
from pathlib import Path

import pytest
from destiny_sdk.enhancements import AuthorPosition, Authorship
from destiny_sdk.identifiers import DOIIdentifier, ExternalIdentifierType
from pydantic import BaseModel, ConfigDict

from destiny_dedupe.algorithm import import_references


def test_normalise_column_name_removes_separators_and_lowercases():
    assert (
        import_references._normalise_column_name(" First-Author (ID) ")
        == "firstauthorid"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        ("", None),
        ("   ", None),
        ("  text  ", "text"),
        ("text", "text"),
    ],
)
def test_parse_nan_string(value, expected):
    assert import_references._parse_nan_string(value) == expected


def test_parse_doi_returns_identifier_for_valid_string():
    doi = import_references._parse_doi("https://doi.org/10.1000/xyz123")
    assert isinstance(doi, DOIIdentifier)
    assert doi.identifier_type == ExternalIdentifierType.DOI
    assert doi.identifier == "10.1000/xyz123"


def test_parse_doi_returns_existing_identifier():
    existing = DOIIdentifier(
        identifier="10.1000/xyz123",
        identifier_type=ExternalIdentifierType.DOI,
    )
    assert import_references._parse_doi(existing) is existing


def test_parse_doi_returns_none_when_identifier_construction_fails(monkeypatch):
    class RaisingDOIIdentifier:
        def __init__(self, *args, **kwargs):  # noqa: ANN204, ANN002
            msg = "bad_doi"
            raise ValueError(msg)

    monkeypatch.setattr(import_references, "DOIIdentifier", RaisingDOIIdentifier)
    monkeypatch.setattr(import_references, "normalise_doi", lambda value: value)

    assert import_references._parse_doi("10.1000/xyz123") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (2024.0, 2024),
        (2024, 2024),
        (2024.9, 2024),
    ],
)
def test_parse_year(value, expected):
    assert import_references._parse_year(value) == expected


def test_parse_authors_returns_none_for_empty_and_anonymous():
    assert import_references._parse_authors("") is None
    assert import_references._parse_authors("anonymous") is None
    assert import_references._parse_authors(" Anonymous ") is None


def test_parse_authors_preserves_list_input():
    authors = [Authorship(display_name="A", position=AuthorPosition.FIRST)]
    assert import_references._parse_authors(authors) is authors


def test_parse_authors_parses_positions():
    authors = import_references._parse_authors("Ada Lovelace.Grace Hopper.Alan Turing")
    assert authors is not None
    assert len(authors) == 3
    assert authors[0].display_name == "Ada Lovelace"
    assert authors[0].position == AuthorPosition.FIRST
    assert authors[1].display_name == "Grace Hopper"
    assert authors[1].position == AuthorPosition.MIDDLE
    assert authors[2].display_name == "Alan Turing"
    assert authors[2].position == AuthorPosition.LAST


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        ("", None),
        ("   ", None),
        ("0092-8674", None),  # ISSN-like value in ISBN column
        ("invalid_isbn", None),
        ("978-0-306-40615-x", "978030640615X"),
        ("0-306-40615-2", "0306406152"),
    ],
)
def test_parse_isbn(value, expected):
    assert import_references._parse_isbn(value) == expected


def test_row_to_paper_kwargs_parses_known_fields_and_preserves_extras():
    existing_authors = [
        Authorship(display_name="Existing", position=AuthorPosition.FIRST)
    ]
    row = {
        "doi": "https://doi.org/10.1000/xyz123",
        "title": "  Example Title  ",
        "authors": existing_authors,
        "year": 2024.0,
        "journal": "  Journal Name ",
        "pages": " 12-18 ",
        "abstract": "  Abstract text  ",
        "issue": " 1 ",
        "volume": " 7 ",
        "isbn": "978-0-306-40615-x",
        "issn": " 1234-5678 ",
        "recordid": 11,
        "duplicateid": 12,
    }

    kwargs = import_references.row_to_paper_kwargs(row)

    assert isinstance(kwargs["doi"], DOIIdentifier)
    assert kwargs["title"] == "Example Title"
    assert kwargs["authors"] is existing_authors
    assert kwargs["year"] == 2024
    assert kwargs["journal"] == "Journal Name"
    assert kwargs["pages"] == "12-18"
    assert kwargs["abstract"] == "Abstract text"
    assert kwargs["issue"] == "1"
    assert kwargs["volume"] == "7"
    assert kwargs["isbn"] == "978030640615X"
    assert kwargs["issn"] == "1234-5678"
    assert kwargs["recordid"] == 11
    assert kwargs["duplicateid"] == 12


def test_resolve_requested_columns_uses_aliases_and_reports_missing(monkeypatch):
    monkeypatch.setattr(
        import_references,
        "COLUMN_ALIASES",
        {
            "authors": ("First Author", "Author"),
            "title": ("Title",),
        },
    )

    usecols, rename_map, missing = import_references._resolve_requested_columns(
        available_columns=["DOI", "First Author", "Title", "Extra"],
        requested_columns=["doi", "authors", "title", "publisher"],
    )

    assert usecols == ["DOI", "First Author", "Title"]
    assert rename_map == {
        "DOI": "doi",
        "First Author": "authors",
        "Title": "title",
    }
    assert missing == ["publisher"]


def test_load_reference_csv_uses_fallback_encoding(tmp_path, monkeypatch):
    class FakePaper(BaseModel):
        doi: DOIIdentifier | None = None
        title: str
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    monkeypatch.setattr(import_references, "Paper", FakePaper)

    csv_path: Path = tmp_path / "refs.csv"
    csv_path.write_text(
        "doi,title\n10.1000/xyz123,Café\n",
        encoding="utf-8",
    )

    config = import_references.CsvLoadConfig(
        columns=("doi", "title"),
        encodings=("ascii", "utf-8"),
    )

    papers = import_references.load_reference_csv(csv_path, config)

    assert len(papers) == 1
    assert papers[0].title == "Café"
    assert isinstance(papers[0].doi, DOIIdentifier)


def test_load_reference_csv_skips_invalid_rows(tmp_path, monkeypatch):
    class FakePaper(BaseModel):
        doi: DOIIdentifier | None = None
        title: str
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    monkeypatch.setattr(import_references, "Paper", FakePaper)

    csv_path = tmp_path / "refs.csv"
    csv_path.write_text(
        "doi,title\n10.1000/valid,Valid Title\n10.1000/invalid,\n",
        encoding="utf-8",
    )

    config = import_references.CsvLoadConfig(
        columns=("doi", "title"),
        encodings=("utf-8",),
    )

    papers = import_references.load_reference_csv(csv_path, config)

    assert len(papers) == 1
    assert papers[0].title == "Valid Title"


def test_load_reference_csv_raises_when_required_columns_are_missing(tmp_path):
    csv_path = tmp_path / "refs.csv"
    csv_path.write_text(
        "doi,title\n10.1000/xyz123,Example\n",
        encoding="utf-8",
    )

    config = import_references.CsvLoadConfig(
        columns=("doi", "title", "authors"),
        require_all_columns=True,
        encodings=("utf-8",),
    )

    with pytest.raises(ValueError, match=r"Missing required columns: \['authors'\]"):
        import_references.load_reference_csv(csv_path, config)


def test_load_reference_csv_raises_unicode_decode_error_when_all_encodings_fail(
    tmp_path,
):
    csv_path = tmp_path / "refs.csv"
    csv_path.write_text(
        "doi,title\n10.1000/xyz123,Café\n",
        encoding="utf-8",
    )

    config = import_references.CsvLoadConfig(
        columns=("doi", "title"),
        encodings=("ascii",),
    )

    with pytest.raises(UnicodeDecodeError, match="Unable to decode CSV"):
        import_references.load_reference_csv(csv_path, config)
