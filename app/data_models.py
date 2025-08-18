"""Data models and associated methods used to specify input and output data."""

import re
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
    pages: tuple[int, int] | None = Field(default=None)
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


def convert_ref_to_paper(ref: ReferenceFileInput | Reference) -> Paper:
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
    loc_enh = None
    loc_enh_extra = None
    if ref.enhancements:
        logger.debug(f"n enhancements: {len(ref.enhancements)}")
        for i, enh in enumerate(ref.enhancements):
            logger.debug(f"on enhancement {i}.")
            logger.debug(f"enhancement: {enh}")
            enh_content = getattr(enh, "content", None)
            enh_type = getattr(enh_content, "enhancement_type", None)
            if enh_type == EnhancementType.LOCATION and enh_content:
                loc_enh = getattr(enh_content, "locations", [])
                continue

            if enh_type == EnhancementType.BIBLIOGRAPHIC:
                bib_enh = enh_content if hasattr(enh, "content") else None
                continue

    title = getattr(bib_enh, "title", None) if bib_enh else None
    authors = getattr(bib_enh, "authorship", None) if bib_enh else None
    year = getattr(bib_enh, "publication_year", None) if bib_enh else None
    publisher = getattr(bib_enh, "publisher", None) if bib_enh else None
    journal_bib = getattr(bib_enh, "journal", None) if bib_enh else None

    logger.debug(loc_enh)
    logger.debug(loc_enh_extra)

    loc_enh_extra = getattr(loc_enh[0], "extra", {}) if loc_enh else None
    journal_loc = loc_enh_extra.get("display_name", None) if loc_enh_extra else None
    issn_list = loc_enh_extra.get("issn", None) if loc_enh_extra else None
    issn = None
    if issn_list:
        issn = issn_list[0] if isinstance(issn_list, list) else issn_list

    abstract = None
    pages = None

    journal = journal_bib if journal_bib is not None else journal_loc

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
        abstract=abstract,
    )
