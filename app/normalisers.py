"""Normalization functions for paper metadata fields."""

import re

from app.data_models import Authorship
from app.utils import roman_to_int

PAGE_PARTS = 2


def normalise_doi(doi: str | None) -> str | None:
    """
    Normalise DOI to consistent lowercase format, remove prefixes, decode symbols.

    Args:
        doi: DOI string to normalise

    Returns:
        Normalised DOI string or None if invalid

    """
    if not doi:
        return None
    doi = doi.replace("%28", "(").replace("%29", ")")
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^DOI[: ]?", "", doi, flags=re.IGNORECASE)
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, flags=re.IGNORECASE)
    if match:
        doi = match.group(0)
    return doi.strip().lower()


def normalise_isbn(isbn: str | None) -> str | None:
    """
    Normalise ISBN string by removing common extra patterns.

    Args:
        isbn: ISBN string to normalise

    Returns:
        Normalised ISBN string or None if invalid

    """
    if not isbn:
        return None
    isbn = re.sub(r"\s*\(PRINT\).*", "", isbn, flags=re.IGNORECASE)
    isbn = re.sub(r"\s*\(ELECTRONIC\).*", "", isbn, flags=re.IGNORECASE)
    isbn = re.sub(r"\\N.*", "", isbn)
    return isbn.strip().lower()


def normalise_pages(page_range: str | None) -> str | None:
    """
    Normalise page strings so style variants compare consistently.

    Args:
        page_range: Page range string to normalise

    Returns:
        Normalised page range string or None if invalid

    """
    if not page_range:
        return None
    page_range = re.sub(r"[---]+", "-", page_range).strip()
    parts = page_range.split("-")
    if len(parts) != PAGE_PARTS:
        return page_range
    start, end = parts[0].strip(), parts[1].strip()
    if len(end) < len(start):
        end = start[: len(start) - len(end)] + end

    step0 = f"{start}-{end}"

    normalised = step0.lower().replace("\u2013", "-").replace("\u2014", "-")
    normalised = re.sub(r"\s+", "", normalised)

    # Canonicalize shorthand prefixed page ranges like s30-50 -> s30-s50.
    match = re.match(r"^([a-z]*)(\d+)-([a-z]*)(\d+)$", normalised)
    if match:
        prefix_start, start, prefix_end, end = match.groups()
        if not prefix_end:
            prefix_end = prefix_start
        return f"{prefix_start}{start}-{prefix_end}{end}"

    return normalised


def normalise_author_name(name: str) -> str:
    """
    Normalise author name (strip, remove dots, lowercase).

    Args:
        name: Author name to normalise

    Returns:
        Normalised author name

    """
    if not name:
        return ""
    return name.replace(".", "").strip().lower()


def normalise_authors(authors: list[str] | None) -> list[Authorship] | None:
    """
    Normalize list of author names.

    Args:
        authors: List of author names to normalize

    Returns:
        List of normalized Authorship objects or None if no authors

    """
    if not authors:
        return None
    return [
        Authorship(author_name=normalise_author_name(a), display_name="", position=0)
        for a in authors
    ]


def normalise_part_number(s: str) -> str:
    """Normalise a part number token to a canonical integer string if possible."""
    try:
        return str(int(s))
    except ValueError:
        roman = roman_to_int(s)
        return str(roman) if roman is not None else s.lower()
