"""Module for handling import references and data processing."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from app.config import CsvImportSettings, get_settings

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


def _resolve_requested_columns(
    available_columns: pd.Index[str] | Sequence[str],
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
    include_gold_standard: bool = False
    require_all_columns: bool = False
    encodings: tuple[str, ...] = Field(default=SUPPORTED_ENCODINGS)


def load_reference_csv(
    path: str | Path,
    config: CsvLoadConfig | None = None,
) -> pd.DataFrame:
    """Load a reference CSV with automatic encoding detection."""
    config = config or CsvLoadConfig()

    requested_columns = list(config.columns)
    if config.include_gold_standard:
        requested_columns.extend(GOLD_STANDARD_COLUMNS)

    csv_path = Path(path)

    for encoding in config.encodings:
        try:
            header = pd.read_csv(csv_path, nrows=0, encoding=encoding)
        except UnicodeDecodeError:
            continue

        usecols, rename_map, missing = _resolve_requested_columns(
            header.columns,
            requested_columns,
        )
        if missing and config.require_all_columns:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)

        try:
            dataframe = pd.read_csv(
                csv_path,
                encoding=encoding,
                usecols=pd.Index(usecols) if usecols else None,
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue

        return dataframe.rename(columns=rename_map)

    error_encoding = "unknown"
    error_message = "Unable to decode CSV using supported encodings."

    raise UnicodeDecodeError(
        error_encoding,
        b"",
        0,
        1,
        error_message,
    )
