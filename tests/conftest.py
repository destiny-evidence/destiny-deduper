import csv
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from destiny_sdk.enhancements import (
    Annotation,
    AnnotationEnhancement,
    AuthorPosition,
    Authorship,
    BibliographicMetadataEnhancement,
    BooleanAnnotation,
    DriverVersion,
    Enhancement,
    EnhancementFileInput,
    EnhancementType,
    Location,
    LocationEnhancement,
)
from destiny_sdk.identifiers import (
    DOIIdentifier,
    ExternalIdentifierType,
    OpenAlexIdentifier,
    PubMedIdentifier,
)
from destiny_sdk.references import Reference, ReferenceFileInput
from destiny_sdk.visibility import Visibility
from faker import Faker
from loguru import logger
from pydantic import HttpUrl

from destiny_deduper.algorithm.import_references import (
    CsvLoadConfig,
    load_reference_csv,
)
from destiny_deduper.data_models import Paper

fa = Faker()


def generate_fake_annotations() -> list[Annotation]:
    """
    Generate synthetic annotations data for reference.

    Returns:
        Annotation: an annotation.

    """
    annotations = [
        BooleanAnnotation(
            scheme="ring",
            value=fa.boolean(),
            label=word,
            score=fa.pyfloat(left_digits=0, right_digits=2, positive=True),
            data={},
        )
        for word in ["earth", "fire", "wind", "water", "heart"]
    ]
    logger.debug("Generated {} fake annotations for fixture data", len(annotations))
    return annotations


def generate_fake_location_enhancement(
    reference_type: Literal["file", "regular"] = "file",
) -> EnhancementFileInput:
    """
    Generate an `extra` annotation.
    This often sits in `enhancements.location` and
    sometimes contains info such as publisher if not
    found elsewhere.

    Returns:
        Annotation: an annotation

    """
    logger.debug(
        "Generating fake location enhancement for reference type: {}", reference_type
    )
    EnhancementDataModel = Enhancement  # noqa: N806
    if reference_type == "file":
        EnhancementDataModel = EnhancementFileInput  # noqa: N806
    org = f"{fa.name} {fa.name_female}"
    openalex_id = f"https://openalex.org/W{fa.random_int(max=99999999)}"
    location = Location(
        is_oa=True,
        version=DriverVersion.PUBLISHED_VERSION,
        landing_page_url=HttpUrl("https://ucl.ac.uk"),
        pdf_url=None,
        license=None,
        extra={
            "id": openalex_id,
            "display_name": f"{fa.last_name} Brothers Publishing",
            "issn_l": ["0953-4563", "1998-409X"],
            "issn": "0953-4563",
            "is_oa": True,
            "is_in_doaj": False,
            "is_indexed_in_scopus": False,
            "is_core": False,
            "host_organization": openalex_id,
            "host_organization_name": org,
            "host_organization_lineage": [
                openalex_id,
                org,
            ],
            "host_organization_lineage_names": [
                f"{fa.name_female} {fa.name_male}",
                org,
            ],
            "type": "journal",
        },
    )
    return EnhancementDataModel(
        reference_id=str(uuid4()),
        source="openalex",
        visibility=Visibility.PUBLIC,
        enhancement_type=EnhancementType.LOCATION,
        robot_version="initial_openalex_import",
        content=LocationEnhancement(
            enhancement_type=EnhancementType.LOCATION,
            locations=[location],
        ),
    )


def generate_fake_enhancements(
    reference_type: Literal["file", "regular"] = "regular",
    bib_content: list | None = None,
) -> list[Enhancement]:
    """
    Generate synthetic enhancements data for reference.

    Returns:
        list[Enhancement]: a list of enhacements.

    """
    logger.debug(
        "Generating fake enhancements for reference_type={} bib_content={}",
        reference_type,
        bib_content,
    )
    EnhancementDataModel = Enhancement  # noqa: N806
    if reference_type == "file":
        EnhancementDataModel = EnhancementFileInput  # noqa: N806

    content = BibliographicMetadataEnhancement()
    if bib_content is None:
        bib_content = ["title", "authors", "year", "publisher"]
    if "title" in bib_content:
        content.title = fa.sentence()
    if "authors" in bib_content:
        content.authorship = [
            Authorship(
                display_name=fa.name_female(), orcid=None, position=AuthorPosition.FIRST
            )
        ]
    if "year" in bib_content:
        content.publication_year = int(fa.year())
    if "publisher" in bib_content:
        content.publisher = f"{fa.last_name_female()} Publishing House"

    return [
        EnhancementDataModel(
            id=str(uuid4()),
            reference_id=str(uuid4()),
            enhancement_type=EnhancementType.BIBLIOGRAPHIC,
            source="Fake",
            visibility=Visibility.PUBLIC,
            robot_version="pytest_robot",
            content=content,
        ),
        EnhancementDataModel(
            id=str(uuid4()),
            reference_id=str(uuid4()),
            enhancement_type=EnhancementType.ANNOTATION,
            source="Fake",
            visibility=Visibility.PUBLIC,
            processor_version="pytest_robot",
            content=AnnotationEnhancement(annotations=generate_fake_annotations()),
        ),
        generate_fake_location_enhancement(reference_type=reference_type),
    ]


def issn_check_digit(issn7: str) -> str:
    """Calculate the ISSN check digit for the first 7 digits."""
    total = sum((8 - i) * int(num) for i, num in enumerate(issn7))
    remainder = total % 11
    check = (11 - remainder) % 11
    return "X" if check == 10 else str(check)


def generate_random_issn() -> str:
    """Generate a random issn."""
    issn7 = fa.numerify("#######")
    return f"{issn7[:4]}-{issn7[4:]}{issn_check_digit(issn7)}"


def generate_reference_jsonl_string(*, valid: bool = True) -> str:
    """
    Generate a jsonl reference string, valid or invalid.

    NOTE: we don't mean that the json string is invalid,
    i.e. not parseable, but that the `ReferenceFileInput`
    data model won't parse it due to `ValidationError`s.

    Args:
        valid (bool, optional): _description_. Defaults to True.

    Returns:
        str: a jsonl string of a `ReferenceFileInput`.

    """
    openalex_id = f"https://openalex.org/W{fa.random_int(max=99999999)}"
    doi = fa.doi()
    author_name = fa.name()
    orcid = f"https://orcid.org/{fa.random_int(1000, 9999)}-{fa.random_int(1000, 9999)}-{fa.random_int(1000, 9999)}-{fa.random_int(1000, 9999)}"
    publication_year = int(fa.year())
    created_date = fa.date_this_decade().isoformat()
    publication_date = fa.date_this_decade().isoformat()
    display_name = fa.company()
    issn_l = None if fa.boolean() else [generate_random_issn(), generate_random_issn()]
    issn = None if fa.boolean() else generate_random_issn()
    is_oa = fa.boolean()
    visibility_key = "visibility" if valid else "visiblity"
    identifiers = (
        [
            {"identifier_type": "open_alex", "identifier": openalex_id},
            {"identifier_type": "doi", "identifier": doi},
        ]
        if valid
        else None
    )

    data = {
        visibility_key: "public",
        "identifiers": identifiers,
        "enhancements": [
            {
                "source": "openalex",
                visibility_key: "public",
                "processor_version": "initial_openalex_import",
                "enhancement_type": "bibliographic",
                "content": {
                    "enhancement_type": "bibliographic",
                    "title": fa.sentence(),
                    "cited_by_count": fa.random_int(0, 100),
                    "created_date": created_date,
                    "publication_date": publication_date,
                    "publication_year": publication_year,
                    "publisher": None,
                    "authorship": [
                        {
                            "display_name": author_name,
                            "orcid": orcid,
                            "position": "first",
                        }
                    ],
                },
            },
            {
                "source": "openalex",
                visibility_key: "public",
                "processor_version": "initial_openalex_import",
                "enhancement_type": "location",
                "content": {
                    "enhancement_type": "location",
                    "locations": [
                        {
                            "is_oa": is_oa,
                            "landing_page_url": f"https://doi.org/{doi}",
                            "extra": {
                                "id": openalex_id,
                                "display_name": display_name,
                                "issn_l": issn_l,
                                "issn": issn,
                                "is_oa": is_oa,
                                "is_in_doaj": False,
                                "is_indexed_in_scopus": False,
                                "is_core": False,
                                "host_organization": None,
                                "host_organization_name": None,
                                "host_organization_lineage": [],
                                "host_organization_lineage_names": [],
                                "type": "ebook platform",
                            },
                        }
                    ],
                },
            },
            {
                "source": "openalex",
                visibility_key: "public",
                "processor_version": "initial_openalex_import",
                "enhancement_type": "annotation",
                "content": {
                    "enhancement_type": "annotation",
                    "annotations": [
                        {
                            "annotation_type": "boolean",
                            "scheme": "openalex:topic",
                            "value": True,
                            "label": fa.catch_phrase(),
                            "data": {
                                "id": f"https://openalex.org/T{fa.random_int(10000, 99999)}",
                                "display_name": fa.catch_phrase(),
                                "score": float(
                                    fa.pyfloat(
                                        left_digits=0, right_digits=4, positive=True
                                    )
                                ),
                                "subfield": {
                                    "id": f"https://openalex.org/subfields/{fa.random_int(1000, 9999)}",
                                    "display_name": fa.job(),
                                },
                                "field": {
                                    "id": f"https://openalex.org/fields/{fa.random_int(10, 99)}",
                                    "display_name": fa.job(),
                                },
                                "domain": {
                                    "id": f"https://openalex.org/domains/{fa.random_int(1, 9)}",
                                    "display_name": fa.job(),
                                },
                            },
                        }
                    ],
                },
            },
        ],
    }

    # remove identifiers if invalid
    # TO DO -- we can add more ways that a jsonl would be invalid here.
    if not valid:
        data.pop("identifiers", None)

    logger.debug(
        "Built reference JSONL fixture with valid={} and identifiers={}",
        valid,
        bool(data.get("identifiers")),
    )
    return json.dumps(data)


@pytest.fixture
def valid_complete_destiny_reference():
    """
    Produce A 'complete' destiny reference.
    We can adjust what constitutes _complete_ in
    due course.
    """
    return Reference(
        id=str(uuid4()),
        identifiers=[
            DOIIdentifier(
                identifier_type=ExternalIdentifierType.DOI,
                identifier=fa.doi(),
            ),
            PubMedIdentifier(identifier=12345),
            OpenAlexIdentifier(identifier="W12345678"),
        ],
        enhancements=generate_fake_enhancements(reference_type="regular"),
    )


@pytest.fixture
def valid_incomplete_destiny_reference():
    return Reference(
        id=str(uuid4()),
        identifiers=[
            DOIIdentifier(
                identifier_type=ExternalIdentifierType.DOI,
                identifier=fa.doi(),
            )
        ],
        enhancements=generate_fake_enhancements(
            reference_type="regular", bib_content=["title", "year"]
        ),
    )


@pytest.fixture
def valid_destiny_reference_file_input():
    return ReferenceFileInput(
        id=str(uuid4()),
        identifiers=[
            DOIIdentifier(
                identifier_type=ExternalIdentifierType.DOI,
                identifier=fa.doi(),
            )
        ],
        enhancements=generate_fake_enhancements(reference_type="file"),
    )


@pytest.fixture
def valid_reference_jsonl_string():
    return generate_reference_jsonl_string(valid=True)


@pytest.fixture
def invalid_reference_jsonl_string():
    return generate_reference_jsonl_string(valid=False)


@pytest.fixture
def valid_paper_instance():
    return Paper(
        doi=DOIIdentifier(
            identifier="10.21759465/m2z0z61", identifier_type=ExternalIdentifierType.DOI
        ),
        isbn=9781453886328,
        title="Das Kapital",
        authors=Authorship(display_name="Karl Marx", position=AuthorPosition.FIRST),
        publisher="Createspace Independent Publishing Platform",
        year=2011,
    )


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture_pairs() -> list[dict]:
    with (FIXTURE_DIR / "pairs.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def pytest_generate_tests(metafunc):
    """Parametrise fixture-driven tests, one case per pair in pairs.csv."""
    if "fixture_pair" in metafunc.fixturenames:
        pairs = _load_fixture_pairs()
        metafunc.parametrize("fixture_pair", pairs, ids=[p["pair_id"] for p in pairs])
    if "csv_fixture_pair" in metafunc.fixturenames:
        pairs = [p for p in _load_fixture_pairs() if "csv" in p["paths"]]
        metafunc.parametrize(
            "csv_fixture_pair", pairs, ids=[p["pair_id"] for p in pairs]
        )


@pytest.fixture(scope="session")
def fixture_pairs() -> list[dict]:
    """Every fixture pair, as rows from pairs.csv."""
    return _load_fixture_pairs()


@pytest.fixture(scope="session")
def fixture_papers() -> dict[int, Paper]:
    """Every fixture record, built directly from the stored Paper fields."""
    stored = json.loads((FIXTURE_DIR / "papers.json").read_text())
    return {int(record_id): Paper(**fields) for record_id, fields in stored.items()}


@pytest.fixture(scope="session")
def csv_fixture_papers() -> dict[int, Paper]:
    """Load the same records through the CSV import path."""
    papers = load_reference_csv(
        FIXTURE_DIR / "records.csv",
        CsvLoadConfig(include_record_id=True),
    )
    return {paper.recordid: paper for paper in papers}
