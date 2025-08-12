"""Data models and associated methods used to specify input and output data."""

from typing import Any, Self

from destiny_sdk.enhancements import Authorship, EnhancementType
from destiny_sdk.identifiers import (
    DOIIdentifier,
    ExternalIdentifier,
    OpenAlexIdentifier,
    OtherIdentifier,
    PubMedIdentifier,
)
from destiny_sdk.references import Reference, ReferenceFileInput
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_extra_types.isbn import ISBN


class IncomingRecord(BaseModel):
    """
    The data structure for incoming records
    for deduplication.

    """

    doi: DOIIdentifier | None = Field(default=None)
    openalex_id: OpenAlexIdentifier | None = Field(default=None)
    pubmed_id: PubMedIdentifier | None = Field(default=None)
    isbn: ISBN | None = Field(default=None)
    title: str | None = Field(default=None)
    authors: list[Authorship] | None = Field(default=None)
    year: int | None = Field(default=None)
    journal: str | None = Field(default=None)
    publisher: str | None = Field(default=None)
    pages: tuple[int, int] | None = Field(default=None)
    abstract: str | None = Field(default=None)

    @model_validator(mode="after")
    def check_for_non_missing(self) -> Self:
        """Ensure there is at least one value in instance."""
        logger.debug(f"model validator, data: {self.model_dump()}")
        for v in self.model_dump().values():
            logger.debug(f"value v: {v}")
            if v is not None:
                return self
        all_none_error = (
            "All values are None.",
            "Please supply at least one non-None value.",
        )
        raise ValueError(all_none_error)


def get_identifier(
    identifiers: list[Any], id_type: type, **kwargs: str
) -> ExternalIdentifier | None:
    """Extract identifier from `Reference` (generic), with optional extra conditions."""
    if identifiers:
        for ident in identifiers:
            if isinstance(ident, id_type):
                # special (rare) ISBN case
                if id_type is OtherIdentifier and kwargs.get("other_identifier_name"):
                    if (
                        getattr(ident, "other_identifier_name", None)
                        == kwargs["other_identifier_name"]
                    ):
                        return ident
                elif id_type is not OtherIdentifier:
                    return ident
    return None


def reference_to_incoming_record(ref: ReferenceFileInput | Reference) -> IncomingRecord:
    """
    Extract relevant fields for `IncomingRecord`
    from a destiny-sdk formatted `Reference` or
    `ReferenceFileInput` object.
    """
    doi = get_identifier(ref.identifiers, DOIIdentifier) if ref.identifiers else None
    openalex_id = (
        get_identifier(ref.identifiers, OpenAlexIdentifier) if ref.identifiers else None
    )
    pubmed_id = (
        get_identifier(ref.identifiers, PubMedIdentifier) if ref.identifiers else None
    )
    isbn = (
        get_identifier(ref.identifiers, OtherIdentifier, other_identifier_name="ISBN")
        if ref.identifiers
        else None
    )
    # get bib enhancement, for title/author/year/publisher/etc.
    bib_enh = None
    if ref.enhancements:
        for enh in ref.enhancements:
            enh_content = getattr(enh, "content", None)
            enh_type = getattr(enh_content, "enhancement_type", None)
            if enh_type == EnhancementType.BIBLIOGRAPHIC:
                bib_enh = enh_content if hasattr(enh, "content") else None
                break

    title = getattr(bib_enh, "title", None) if bib_enh else None
    authors = getattr(bib_enh, "authorship", None) if bib_enh else None
    year = getattr(bib_enh, "publication_year", None) if bib_enh else None
    publisher = getattr(bib_enh, "publisher", None) if bib_enh else None
    journal = None
    abstract = None
    pages = None

    # search for abstract
    if ref.enhancements:
        for enh in ref.enhancements:
            enh_content = getattr(enh, "content", None)
            enh_type = getattr(enh_content, "enhancement_type", None)
            if enh_type == EnhancementType.ABSTRACT:
                if enh_content:
                    abstract = getattr(enh_content, "abstract", None)
                break

    # TODO: get pages -- where?

    return IncomingRecord(
        doi=doi,
        openalex_id=openalex_id,
        pubmed_id=pubmed_id,
        isbn=isbn,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        publisher=publisher,
        pages=pages,
        abstract=abstract,
    )
