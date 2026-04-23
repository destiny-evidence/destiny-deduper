"""Data models and associated methods used to specify input and output data."""

import re
from typing import Self

from destiny_sdk.enhancements import (
    Authorship,
    BibliographicMetadataEnhancement,
    EnhancementType,
    Location,
)
from destiny_sdk.identifiers import (
    DOIIdentifier,
    ExternalIdentifier,
    ExternalIdentifierType,
    OpenAlexIdentifier,
    OtherIdentifier,
    PubMedIdentifier,
)
from destiny_sdk.references import Reference, ReferenceFileInput
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_extra_types.isbn import ISBN


class Paper(BaseModel):
    """
    The data structure for incoming records
    for deduplication.

    """

    doi: DOIIdentifier | None = Field(default=None)
    openalex_id: OpenAlexIdentifier | None = Field(default=None)
    pubmed_id: PubMedIdentifier | None = Field(default=None)
    isbn: ISBN | None = Field(default=None)
    issn: str | None = Field(default=None)
    title: str | None = Field(default=None)
    authors: list[Authorship] | None = Field(default=None)
    year: int | None = Field(default=None)
    journal: str | None = Field(default=None)
    publisher: str | None = Field(default=None)
    pages: tuple[str, str] | None = Field(default=None)  # mirrors destiny
    volume: str | None = Field(default=None)
    issue: str | None = Field(default=None)
    abstract: str | None = Field(default=None)

    @classmethod
    @field_validator("issn", mode="before")
    def check_valid_issn(cls, v: str | None) -> str | None:
        """Ensure issn (if present) is valid."""
        if v:
            issn_regex = re.compile(r"^[0-9]{4}-[0-9]{3}[0-9X]$.")
            if issn_regex.match(v):
                return v
        return None

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


def extract_identifiers(
    identifiers: list[ExternalIdentifier],
) -> dict[ExternalIdentifierType, ExternalIdentifier]:
    """
    Build a mapping from identifier type to identifier object.
    Handles the special case for OTHER/ISBN.
    """
    id_map = {}
    for ident in identifiers:
        if ident.identifier_type == ExternalIdentifierType.OTHER:
            # isbn
            if (
                isinstance(ident, OtherIdentifier)
                and getattr(ident, "other_identifier_name", None) == "ISBN"
            ):
                id_map[(ExternalIdentifierType.OTHER, "ISBN")] = ident
        else:
            id_map[ident.identifier_type] = ident
    return id_map


def convert_ref_to_paper(ref: ReferenceFileInput | Reference) -> Paper:
    """
    Extract relevant fields for `IncomingRecord`
    from a destiny-sdk formatted `Reference` or
    `ReferenceFileInput` object.
    """
    id_map = extract_identifiers(ref.identifiers) if ref.identifiers else {}

    doi = id_map.get(ExternalIdentifierType.DOI)
    openalex_id = id_map.get(ExternalIdentifierType.OPEN_ALEX)
    pubmed_id = id_map.get(ExternalIdentifierType.PM_ID)
    isbn = id_map.get((ExternalIdentifierType.OTHER, "ISBN"))

    # get bib enhancement, for title/author/year/publisher/etc.
    bib_enh: BibliographicMetadataEnhancement | None = None
    loc_enh: list[Location] | None = None
    loc_enh_extra: dict | None = None
    abstract: str | None = None
    if ref.enhancements:
        logger.debug(f"n enhancements: {len(ref.enhancements)}")
        for i, enh in enumerate(ref.enhancements):
            logger.debug(f"on enhancement {i}.")
            logger.debug(f"enhancement: {enh}")
            enh_content = enh.content  # type: ignore[attr-defined]
            enh_type = enh_content.enhancement_type
            if enh_type == EnhancementType.LOCATION and enh_content:
                loc_enh = enh_content.locations
                continue

            if enh_type == EnhancementType.BIBLIOGRAPHIC:
                bib_enh = enh_content
                continue

            if enh_type == EnhancementType.ABSTRACT:
                abstract = enh_content.abstract
                continue

    title = bib_enh.title if bib_enh else None
    authors = bib_enh.authorship if bib_enh else None
    year = bib_enh.publication_year if bib_enh else None
    publisher = bib_enh.publisher if bib_enh else None
    # No DESTINY mapping currently
    journal_bib = None

    logger.debug(loc_enh)
    logger.debug(loc_enh_extra)

    loc_enh_extra = loc_enh[0].extra if loc_enh else None
    journal_loc = loc_enh_extra.get("display_name", None) if loc_enh_extra else None
    issn_list = loc_enh_extra.get("issn", None) if loc_enh_extra else None
    issn = None
    if issn_list:
        issn = issn_list[0] if isinstance(issn_list, list) else issn_list

    # get volume and issue if available
    volume = loc_enh_extra.get("volume", None) if loc_enh_extra else None
    issue = loc_enh_extra.get("issue", None) if loc_enh_extra else None

    journal = journal_bib if journal_bib is not None else journal_loc

    # TODO: get pages -- where?
    pages = None

    return Paper(
        doi=doi,
        openalex_id=openalex_id,
        pubmed_id=pubmed_id,
        isbn=isbn,
        issn=issn,
        title=title,
        authors=authors,
        year=year,
        journal=journal,
        publisher=publisher,
        pages=pages,
        volume=volume,
        issue=issue,
        abstract=abstract,
    )
