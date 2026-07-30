"""Module for handling import references and data processing."""

import math
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from destiny_sdk.enhancements import AuthorPosition, Authorship
from destiny_sdk.identifiers import DOIIdentifier, ExternalIdentifierType
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from app.config import CsvImportSettings, get_settings
from app.data_models import GoldStandardPaper, Paper, PaperWithId
from app.normalisers import normalise_doi

settings = get_settings()
csv_import_settings: CsvImportSettings = settings.csv_import

DEFAULT_COLUMNS: tuple[str, ...] = csv_import_settings.default_columns
GOLD_STANDARD_COLUMNS: tuple[str, ...] = csv_import_settings.gold_standard_columns
SUPPORTED_ENCODINGS: tuple[str, ...] = csv_import_settings.supported_encodings
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    field_name: tuple(aliases)
    for field_name, aliases in csv_import_settings.column_aliases.model_dump().items()
}


def _normalise_column_name(column_name: str) -> str:
    return "".join(
        character for character in column_name.lower() if character.isalnum()
    )


def _parse_nan_string(v: str | float | None) -> str | None:
    """Convert pandas NaN and whitespace-only strings to None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return None


def _parse_doi(v: object) -> DOIIdentifier | None:
    """Parse and normalise a DOI value from a raw pandas cell."""
    if isinstance(v, DOIIdentifier):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        doi = normalise_doi(v)
        if not doi:
            return None
        try:
            return DOIIdentifier(
                identifier=doi,
                identifier_type=ExternalIdentifierType.DOI,
            )
        except Exception:  # noqa: BLE001
            logger.debug(f"Unable to parse DOI '{doi}'")
            return None
    return None


def _parse_year(v: float | None) -> int | None:
    """Parse year from a raw pandas cell, handling NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return int(v)


def _parse_authors(v: str | float | list | None) -> list[Authorship] | None:
    """Parse an author string into a list of Authorship objects."""
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        if v.strip().lower() in ("anonymous", ""):
            return None
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


def _parse_isbn(v: str | float | None) -> str | None:
    """Parse and validate an ISBN value from a raw pandas cell."""
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
    return compact.upper()


def row_to_paper_kwargs(row: dict) -> dict:
    """
    Convert a raw pandas row dict to cleaned keyword arguments for Paper.

    Handles pandas NaN values, normalises DOIs, parses author strings, and
    validates ISBNs. Extra keys in ``row`` (e.g. ``recordid``, ``duplicateid``)
    are passed through unchanged so callers that construct a Paper subclass can
    still access them.
    """
    known_fields = {
        "doi": _parse_doi(row.get("doi")),
        "title": _parse_nan_string(row.get("title")),
        "authors": _parse_authors(row.get("authors")),
        "year": _parse_year(row.get("year")),
        "journal": _parse_nan_string(row.get("journal")),
        "pages": _parse_nan_string(row.get("pages")),
        "abstract": _parse_nan_string(row.get("abstract")),
        "issue": _parse_nan_string(row.get("issue")),
        "volume": _parse_nan_string(row.get("volume")),
        "isbn": _parse_isbn(row.get("isbn")),
        "issn": _parse_nan_string(row.get("issn")),
    }
    extra_keys = {k: v for k, v in row.items() if k not in known_fields}
    return {**known_fields, **extra_keys}


def _resolve_requested_columns(
    available_columns: Sequence[str],
    requested_columns: Sequence[str],
) -> tuple[list[str], dict[str, str], list[str]]:
    normalised_available = {
        _normalise_column_name(column_name): column_name
        for column_name in available_columns
    }

    usecols: list[str] = []
    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for requested_column in requested_columns:
        aliases = COLUMN_ALIASES.get(requested_column, (requested_column,))

        matched_column = None
        for alias in aliases:
            matched_column = normalised_available.get(_normalise_column_name(alias))
            if matched_column is not None:
                break

        if matched_column is None:
            missing.append(requested_column)
            continue

        if matched_column not in rename_map:
            usecols.append(matched_column)
        rename_map[matched_column] = requested_column

    return usecols, rename_map, missing


class CsvLoadConfig(BaseModel):
    """Configuration for loading reference CSV files."""

    columns: tuple[str, ...] = Field(default=DEFAULT_COLUMNS)
    include_record_id: bool = False
    include_gold_standard: bool = False
    require_all_columns: bool = False
    encodings: tuple[str, ...] = Field(default=SUPPORTED_ENCODINGS)


def load_reference_csv(
    path: str | Path,
    config: CsvLoadConfig | None = None,
) -> list[Paper]:
    """
    Load a reference CSV and return validated Paper objects.

    Tries each encoding in ``config.encodings`` until the file decodes
    successfully. Columns are resolved via ``COLUMN_ALIASES``. Each row is
    parsed into a ``Paper`` via ``row_to_paper_kwargs``; rows that fail
    validation are logged and skipped.
    """
    config = config or CsvLoadConfig()

    requested_columns = list(config.columns)

    if config.include_record_id:
        requested_columns.append("recordid")

    if config.include_gold_standard:
        requested_columns.extend(GOLD_STANDARD_COLUMNS)

    requested_columns = list(dict.fromkeys(requested_columns))

    paper_model: type[Paper]

    if config.include_gold_standard:
        paper_model = GoldStandardPaper
    elif config.include_record_id:
        paper_model = PaperWithId
    else:
        paper_model = Paper

    csv_path = Path(path)

    for encoding in config.encodings:
        try:
            header = pd.read_csv(
                csv_path,
                nrows=0,
                encoding=encoding,
            )
        except UnicodeDecodeError:
            continue

        usecols, rename_map, missing = _resolve_requested_columns(
            list(header.columns),
            requested_columns,
        )

        if missing and config.require_all_columns:
            err_msg = f"Missing required columns: {missing}"
            raise ValueError(err_msg)

        try:
            dataframe = pd.read_csv(
                csv_path,
                encoding=encoding,
                usecols=usecols,
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue

        dataframe = dataframe.rename(columns=rename_map)

        papers: list[Paper] = []

        for record in dataframe.to_dict(orient="records"):
            try:
                papers.append(paper_model(**row_to_paper_kwargs(record)))
            except ValidationError as exc:
                logger.debug(f"Skipping invalid row: {exc}")

        return papers

    error_encoding = "unknown"
    error_message = "Unable to decode CSV using supported encodings."

    raise UnicodeDecodeError(
        error_encoding,
        b"",
        0,
        1,
        error_message,
    )
