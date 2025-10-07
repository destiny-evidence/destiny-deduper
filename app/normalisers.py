import re
from app.data_models import DOIIdentifier, Authorship

def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.replace("%28", "(").replace("%29", ")")
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^DOI[: ]?", "", doi, flags=re.IGNORECASE)
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, flags=re.IGNORECASE)
    if match:
        doi = match.group(0)
    return doi.strip().lower()

def normalize_isbn(isbn: str | None) -> str | None:
    if not isbn:
        return None
    isbn = re.sub(r"\s*\(PRINT\).*", "", isbn, flags=re.IGNORECASE)
    isbn = re.sub(r"\s*\(ELECTRONIC\).*", "", isbn, flags=re.IGNORECASE)
    isbn = re.sub(r"\\N.*", "", isbn)
    return isbn.strip().lower()

def normalize_pages(page_range: str | None) -> str | None:
    if not page_range:
        return None
    page_range = re.sub(r"[–—−]+", "-", page_range).strip()
    parts = page_range.split("-")
    if len(parts) != 2:
        return page_range
    start, end = parts[0].strip(), parts[1].strip()
    if len(end) < len(start):
        end = start[:len(start)-len(end)] + end
    return f"{start}-{end}"

def normalize_author_name(name: str) -> str:
    """Normalize author name (strip, remove dots, lowercase)."""
    if not name:
        return ""
    return name.replace(".", "").strip().lower()

def normalize_authors(authors: list[str] | None) -> list[Authorship] | None:
    if not authors:
        return None
    return [Authorship(author_name=normalize_author_name(a)) for a in authors]