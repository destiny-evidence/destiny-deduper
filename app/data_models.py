"""Data models and associated methods used to specify input and output data."""

from typing import Any

from destiny_sdk.enhancements import Authorship, EnhancementType
from destiny_sdk.identifiers import (
    DOIIdentifier,
    ExternalIdentifier,
    OpenAlexIdentifier,
    OtherIdentifier,
    PubMedIdentifier,
)
from destiny_sdk.references import Reference, ReferenceFileInput
from pydantic import BaseModel
from pydantic_extra_types.isbn import ISBN


class IncomingRecord(BaseModel):
    """
    The data structure for incoming records
    for deduplication.

    """

    doi: DOIIdentifier | None
    openalex_id: OpenAlexIdentifier | None
    pubmed_id: PubMedIdentifier | None
    isbn: ISBN | None
    title: str | None
    authors: list[Authorship] | None
    year: int | None
    journal: str | None
    publisher: str | None
    pages: tuple[int, int] | None
    abstract: str | None


def reference_to_incoming_record(ref: ReferenceFileInput | Reference) -> IncomingRecord:
    """
    Extract relevant fields for `IncomingRecord`
    from a destiny-sdk formatted `Reference` or
    `ReferenceFileInput` object.
    """

    def get_identifier(
        identifiers: list[Any], id_type: type, **kwargs: str
    ) -> ExternalIdentifier | None:
        """Extract identifier from `Reference` (generic), with optional extra conditions."""
        if identifiers:
            for ident in identifiers:
                if isinstance(ident, id_type):
                    # special (rare) ISBN case
                    if id_type is OtherIdentifier and kwargs.get(
                        "other_identifier_name"
                    ):
                        if (
                            getattr(ident, "other_identifier_name", None)
                            == kwargs["other_identifier_name"]
                        ):
                            return ident
                    elif id_type is not OtherIdentifier:
                        return ident
        return None

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
            # for ReferenceFileInput, enhancement_type is an attribute; for Reference, it's in content
            enh_type = getattr(enh, "enhancement_type", None)
            if enh_type == EnhancementType.BIBLIOGRAPHIC:
                bib_enh = enh.content if hasattr(enh, "content") else None
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
            enh_type = getattr(enh, "enhancement_type", None)
            if enh_type == EnhancementType.ABSTRACT:
                content = getattr(enh, "content", None)
                if content:
                    abstract = getattr(content, "abstract", None)
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
