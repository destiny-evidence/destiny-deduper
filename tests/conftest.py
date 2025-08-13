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
    Enhancement,
    EnhancementFileInput,
    EnhancementType,
)
from destiny_sdk.identifiers import DOIIdentifier, ExternalIdentifierType
from destiny_sdk.references import Reference, ReferenceFileInput
from destiny_sdk.visibility import Visibility
from faker import Faker

from app.data_models import Paper

fa = Faker()


def generate_fake_annotations() -> Annotation:
    """
    Generate synthetic annotations data for reference.

    Returns:
        Annotation: an annotation.

    """
    return [
        BooleanAnnotation(
            scheme="ring",
            value=fa.boolean(),
            label=word,
            score=fa.pyfloat(left_digits=0, right_digits=2, positive=True),
            data={},
        )
        for word in ["earth", "fire", "wind", "water", "heart"]
    ]


def generate_fake_enhancements(
    reference_type: Literal["file", "regular"] = "regular",
    bib_content: list | None = None,
) -> list[Enhancement]:
    """
    Generate synthetic enhancements data for reference.

    Returns:
        list[Enhancement]: a list of enhacements.

    """
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
    ]


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
            )
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
    return """
    {
    "visibility": "public",
    "identifiers": [
        {
            "identifier_type": "open_alex",
            "identifier": "W4411812659"
        },
        {
            "identifier_type": "doi",
            "identifier": "https://doi.org/10.21428/e2759450.9e968b77"
        }
    ],
    "enhancements": [
        {
            "source": "openalex",
            "visibility": "public",
            "processor_version": "initial_openalex_import",
            "enhancement_type": "bibliographic",
            "content": {
                "enhancement_type": "bibliographic",
                "title": "Recurrent Neural Networks",
                "cited_by_count": 15,
                "created_date": "2025-07-01",
                "publication_date": "2025-06-30",
                "publication_year": 2025,
                "publisher": null,
                "authorship": [
                    {
                        "display_name": "James S. Magnuson",
                        "orcid": "https://orcid.org/0000-0003-0158-2367",
                        "position": "first"
                    }
                ]
            }
        },
        {
            "source": "openalex",
            "visibility": "public",
            "processor_version": "initial_openalex_import",
            "enhancement_type": "location",
            "content": {
                "enhancement_type": "location",
                "locations": [
                    {
                        "is_oa": false,
                        "landing_page_url": "https://doi.org/10.21428/e2759450.9e968b77",
                        "extra": {
                            "id": "https://openalex.org/S4306463626",
                            "display_name": "MIT Press eBooks",
                            "issn_l": null,
                            "issn": null,
                            "is_oa": false,
                            "is_in_doaj": false,
                            "is_indexed_in_scopus": false,
                            "is_core": false,
                            "host_organization": null,
                            "host_organization_name": null,
                            "host_organization_lineage": [],
                            "host_organization_lineage_names": [],
                            "type": "ebook platform"
                        }
                    }
                ]
            }
        },
        {
            "source": "openalex",
            "visibility": "public",
            "processor_version": "initial_openalex_import",
            "enhancement_type": "annotation",
            "content": {
                "enhancement_type": "annotation",
                "annotations": [
                    {
                        "annotation_type": "boolean",
                        "scheme": "openalex:topic",
                        "value": true,
                        "label": "Neural Networks and Applications",
                        "data": {
                            "id": "https://openalex.org/T10320",
                            "display_name": "Neural Networks and Applications",
                            "score": 0.0592,
                            "subfield": {
                                "id": "https://openalex.org/subfields/1702",
                                "display_name": "Artificial Intelligence"
                            },
                            "field": {
                                "id": "https://openalex.org/fields/17",
                                "display_name": "Computer Science"
                            },
                            "domain": {
                                "id": "https://openalex.org/domains/3",
                                "display_name": "Physical Sciences"
                            }
                        }
                    }
                ]
            }
        }
    ]
}"""


@pytest.fixture
def invalid_reference_jsonl_string():
    return """
    {
    "visibility": "public",
    "enhancements": [
        {
            "source": "openalex",
            "visiblity": "public",
            "processor_version": "initial_openalex_import",
            "enhancement_type": "bibliographic",
            "content": {
                "enhancement_type": "bibliographic",
                "title": "Recurrent Neural Networks",
                "cited_by_count": 15,
                "created_date": "2025-07-01",
                "publication_date": "2025-06-30",
                "publication_year": 2025,
                "publisher": null,
                "authorship": [
                    {
                        "display_name": "James S. Magnuson",
                        "orcid": "https://orcid.org/0000-0003-0158-2367",
                        "position": "first"
                    }
                ]
            }
        },
        {
            "source": "openalex",
            "visibility": "public",
            "processor_version": "initial_openalex_import",
            "enhancement_type": "location",
            "content": {
                "enhancement_type": "location",
                "locations": [
                    {
                        "is_oa": false,
                        "landing_page_url": "https://doi.org/10.21428/e2759450.9e968b77",
                        "extra": {
                            "id": "https://openalex.org/S4306463626",
                            "display_name": "MIT Press eBooks",
                            "issn_l": null,
                            "issn": null,
                            "is_oa": false,
                            "is_in_doaj": false,
                            "is_indexed_in_scopus": false,
                            "is_core": false,
                            "host_organization": null,
                            "host_organization_name": null,
                            "host_organization_lineage": [],
                            "host_organization_lineage_names": [],
                            "type": "ebook platform"
                        }
                    }
                ]
            }
        },
        {
            "source": "openalex",
            "visibility": "public",
            "processor_version": "initial_openalex_import",
            "enhancement_type": "annotation",
            "content": {
                "enhancement_type": "annotation",
                "annotations": [
                    {
                        "annotation_type": "boolean",
                        "scheme": "openalex:topic",
                        "value": true,
                        "label": "Neural Networks and Applications",
                        "data": {
                            "id": "https://openalex.org/T10320",
                            "display_name": "Neural Networks and Applications",
                            "score": 0.0592,
                            "subfield": {
                                "id": "https://openalex.org/subfields/1702",
                                "display_name": "Artificial Intelligence"
                            },
                            "field": {
                                "id": "https://openalex.org/fields/17",
                                "display_name": "Computer Science"
                            },
                            "domain": {
                                "id": "https://openalex.org/domains/3",
                                "display_name": "Physical Sciences"
                            }
                        }
                    }
                ]
            }
        }
    ]
}"""


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
