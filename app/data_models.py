"""Data models and associated methods used to specify input and output data."""

import math
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

PAGES_PARTS = 2
ISBN_LEN = 10


class Paper(BaseModel):
    """
    The data structure for incoming records
    for deduplication.

    """

    doi: DOIIdentifier | str | None = Field(default=None)
    openalex_id: OpenAlexIdentifier | None = Field(default=None)
    pubmed_id: PubMedIdentifier | None = Field(default=None)
    isbn: ISBN | str | None = Field(default=None)
    issn: str | None = Field(default=None)
    title: str | None = Field(default=None)
    authors: list[Authorship] | str | None = Field(default=None)
    year: int | None = Field(default=None)
    journal: str | None = Field(default=None)
    issue: str | None = Field(default=None)
    volume: str | None = Field(default=None)
    publisher: str | None = Field(default=None)
    pages: tuple[int, int] | str | None = Field(default=None)
    abstract: str | None = Field(default=None)

    @field_validator("issn", mode="after")
    @classmethod
    def check_valid_issn(cls, v: str | None) -> str | None:
        """Ensure issn (if present) is valid."""
        if v:
            issn_regex = re.compile(r"^[0-9]{4}-[0-9]{3}[0-9X]$.")
            if issn_regex.match(v):
                return v
        return None

    @field_validator("doi", mode="after")
    @classmethod
    def format_doi(cls, v: str | DOIIdentifier | None) -> DOIIdentifier | None:
        """Make a doi string work with destiny DOIIdentifier."""
        if v is None:
            return None
        if isinstance(v, DOIIdentifier):
            return v
        if isinstance(v, str):
            return DOIIdentifier(identifier=v)
        return None

    @field_validator("isbn", mode="after")
    @classmethod
    def format_isbn(cls, v: ISBN | str | None) -> ISBN | None:
        """Normalize ISBN: keep digits, left-pad with zeros to length ISBN_LEN, return ISBN or None."""
        if v is None:
            return None
        if isinstance(v, ISBN):
            return v
        if isinstance(v, str):
            # extract digits only
            digits = "".join(re.findall(r"\d+", v))
            if not digits:
                return None
            # left-pad until required length
            if len(digits) < ISBN_LEN:
                digits = digits.zfill(ISBN_LEN)
            return ISBN(digits)
        return None

    # @classmethod
    # @field_validator("pages", mode="before")
    # def parse_pages(cls, v: str | tuple[int, int] | None) -> tuple[int, int] | None:
    #     """Parse pages to tuple(int, int)."""
    #     if isinstance(v, str):
    #         # Normalize dash variants
    #         v = re.sub(r"[–—−]", "-", v.strip())  # noqa: RUF001
    #         # Extract two numbers if possible
    #         parts = v.split("-")
    #         if len(parts) == PAGES_PARTS and all(p.strip().isdigit() for p in parts):
    #             return (int(parts[0].strip()), int(parts[1].strip()))
    #         # Handle cases like "685-96" → infer "685-696"
    #         if (
    #             len(parts) == PAGES_PARTS
    #             and parts[0].strip().isdigit()
    #             and not parts[1].strip().isdigit()
    #         ):
    #             start = parts[0].strip()
    #             end = parts[1].strip()
    #             try:
    #                 prefix_len = len(start) - len(end)
    #                 end_full = start[:prefix_len] + end
    #                 return (int(start), int(end_full))
    #             except (ValueError, TypeError):
    #                 return None
    #         else:
    #             return None
    #     return v

    @field_validator("pages", mode="after")
    @classmethod
    def parse_pages(cls, v: str | tuple[int, int] | None) -> tuple[int, int] | None:  # noqa: PLR0911
        """Parse pages to tuple(int, int). Handles '685-96', '70s-85s', and other common forms."""
        if v is None:
            return None
        if isinstance(v, tuple):
            return v
        if isinstance(v, str):
            s = re.sub(r"[–—−]", "-", v.strip())  # normalize dashes
            parts = [p.strip() for p in s.split("-") if p.strip()]
            if len(parts) != 2:
                return None

            def extract_number(token: str) -> str | None:
                # grab the longest contiguous digits substring (e.g. "70s" -> "70", "685" -> "685")
                m = re.search(r"(\d+)", token)
                return m.group(1) if m else None

            start_s = extract_number(parts[0])
            end_s = extract_number(parts[1])

            if not start_s or not end_s:
                return None

            # If end looks shorter than start, try to expand (685-96 -> 685-696)
            try:
                if len(end_s) < len(start_s):
                    prefix_len = len(start_s) - len(end_s)
                    end_full = start_s[:prefix_len] + end_s
                else:
                    end_full = end_s
                start_n = int(start_s)
                end_n = int(end_full)
                return (start_n, end_n)
            except (ValueError, TypeError):
                return None
        return None

    @model_validator(mode="after")
    def check_for_non_missing(self) -> Self:
        """Ensure there is at least one value in instance."""
        for v in self.model_dump().values():
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
        abstract=abstract,
    )
