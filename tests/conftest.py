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
        content.publication_year = fa.year()
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
    pass


@pytest.fixture
def invalid_reference_jsonl_string_missing_id():
    pass


@pytest.fixture
def invalid_reference_jsonl_string_missing_visibility():
    pass


@pytest.fixture
def invalid_destiny_reference():
    pass


@pytest.fixture
def invalid_destiny_reference_file_input():
    pass
