"""Data models and associated methods used to specify input and output data."""

import math
import re
from typing import Self

from destiny_sdk.enhancements import (
    AuthorPosition,
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
    """Data model for paper records used in deduplication.

    Represents a single bibliographic record with metadata fields and identifiers.
    Includes validators to handle pandas NaN values and normalize field inputs,
    making it suitable for both API usage and direct CSV/dataframe parsing.
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
    pages: str | None = Field(default=None)
    volume: str | None = Field(default=None)
    issue: str | None = Field(default=None)
    abstract: str | None = Field(default=None)

    @field_validator("doi", mode="before")
    @classmethod
    def parse_doi(cls, v: DOIIdentifier | str | float | None) -> DOIIdentifier | None:
        """Parse and normalize DOI field to DOIIdentifier object.

        Converts raw DOI strings from CSV/pandas to DOIIdentifier identity objects.
        Handles pandas NaN values and empty strings. Non-string, non-DOIIdentifier
        inputs return None. DOI parsing failures are logged as debug messages.

        Args:
            v: Raw DOI value (may be DOIIdentifier, string, float NaN, or None).

        Returns:
            DOIIdentifier | None: Parsed DOI identity object, or None if missing,
                unparseable, or empty.
        """
        if isinstance(v, DOIIdentifier):
            return v
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            doi = v.strip()
            if not doi:
                return None
            try:
                return DOIIdentifier(
                    identifier=doi,
                    identifier_type=ExternalIdentifierType.DOI,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Unable to parse DOI '{doi}': {e}")
                return None
        return None

    @field_validator("pages", mode="before")
    @classmethod
    def parse_pages(cls, v: str | float | None) -> str | None:
        """Parse and clean pages field, handling pandas NaN and whitespace.

        Strips whitespace and returns None for pandas NaN floats, None values,
        or empty strings. Preserves page range formats (e.g., '123-145').

        Args:
            v: Raw pages value (string, float NaN, or None).

        Returns:
            str | None: Cleaned pages string, or None if missing or empty.
        """
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return None

    @field_validator(
        "volume", "abstract", "issue", "journal", "publisher", "issn", mode="before"
    )
    @classmethod
    def parse_string_fields(cls, v: str | float | None) -> str | None:
        """Parse and clean string fields from pandas data, handling NaN values.

        Applies to volume, abstract, issue, journal, publisher, and issn fields.
        Converts pandas NaN floats to None, strips whitespace, and returns None
        for empty strings. Non-string, non-NaN inputs pass through unchanged.

        Args:
            v: Raw field value (string, float NaN, or None).

        Returns:
            str | None: Cleaned string value, or None if missing or empty.
        """
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return None

    @field_validator("isbn", mode="before")
    @classmethod
    def parse_isbn(cls, v: str | float | None) -> str | None:
        """Parse ISBN field, validating format and detecting ISSN misclassification.

        Accepts ISBN-10 and ISBN-13 formats (with or without hyphens). Rejects
        ISSN values incorrectly placed in ISBN column (pattern: XXXX-XXX[Xx]).
        Returns None for invalid lengths, ISSN misclassifications, or invalid
        ISBN-13 check digits.

        Args:
            v: Raw ISBN value (string, float NaN, or other type).

        Returns:
            str | None: Compact ISBN string (digits only, uppercase), or None
                if invalid, missing, or ISSN.
        """
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if not isinstance(v, str):
            return None

        raw = v.strip()
        if not raw:
            return None

        # Common in source data: ISSN accidentally in ISBN column (e.g. 0092-8674)
        if re.fullmatch(r"\d{4}-\d{3}[\dXx]", raw):
            return None

        compact = re.sub(r"[^0-9Xx]", "", raw)
        if len(compact) not in (10, 13):
            return None

        return raw

    @field_validator("authors", mode="before")
    @classmethod
    def parse_authors(cls, v: str | float | None) -> list[Authorship] | None:
        """Parse author string into list of Authorship objects with position tracking.

        Splits comma and period-delimited author strings into individual authors,
        assigning AuthorPosition (FIRST, MIDDLE, LAST) based on order. Handles
        pandas NaN, "anonymous", and already-parsed Authorship lists.

        Args:
            v: Raw author value (list of Authorship, string, float NaN, or None).
                String format: "Smith, J.Doe, A.Jones, B" splits on periods after
                initials.

        Returns:
            list[Authorship] | None: List of Authorship objects with display_name
                and position, or None if missing or anonymous.
        """
        if isinstance(v, list):
            return v

        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            if v.strip().lower() in ("anonymous", ""):
                return None
            # Split on "." followed by letter (typical author initials)
            author_names = re.split(r"\.(?=\w)", v)
            author_names = [a.strip() for a in author_names if a.strip()]
            if not author_names:
                return None
            authors_list = []
            for i, a in enumerate(author_names):
                position = (
                    AuthorPosition.FIRST
                    if i == 0
                    else AuthorPosition.LAST
                    if i == len(author_names) - 1
                    else AuthorPosition.MIDDLE
                )
                authors_list.append(Authorship(display_name=a, position=position))
            return authors_list
        return None

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
