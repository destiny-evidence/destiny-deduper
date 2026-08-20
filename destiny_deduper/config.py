"""Configuration management for the deduplication toolkit."""

from functools import lru_cache
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_FILE_PATH = PACKAGE_ROOT / ".config.yaml"  # moved to inside package
USER_CONFIG_FILE_PATH = (
    Path("~/.config/destiny-deduper/").expanduser() / ".config.yaml"
)  # canonical version, but
# won't work on windows just yet.

# NOTE: deleting user config will regenerate on next run.
# this is required if there are any breaking changes in the the config/settings.
if not USER_CONFIG_FILE_PATH.is_file():
    from sys import platform

    if platform == "win32":
        logger.warning(
            "user config not yet implemented, defaulting to packaged config."
        )
    else:
        from shutil import copyfile

        logger.debug(
            f"copying package .config.yaml to user config yaml ({USER_CONFIG_FILE_PATH}) for editability."
        )
        try:
            USER_CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            copyfile(CONFIG_FILE_PATH, USER_CONFIG_FILE_PATH)
        except PermissionError:
            logger.warning(
                f"{USER_CONFIG_FILE_PATH} isn't writeable. defaulting to package config."
            )


class TitleThresholds(BaseModel):
    """Thresholds for paper title similarity comparisons."""

    veto: float
    similarity: float
    similarity_lower: float
    partial_match_ratio: float


class AuthorThresholds(BaseModel):
    """Threshold for author comparisons."""

    similarity: float
    first_chars: int


class JournalThresholds(BaseModel):
    """Threshold for journal-related comparisons."""

    similarity: float
    strong_similarity: float
    abbreviation: float


class PaperThresholds(BaseModel):
    """Threshold for paper comparisons."""

    match: float


class ThresholdSettings(BaseModel):
    """Combined threshold settings."""

    doi_partial_match: float
    partial_title_match: float

    title: TitleThresholds
    author: AuthorThresholds
    journal: JournalThresholds
    paper: PaperThresholds

    strong_metadata_match: float
    min_mismatches_for_veto: int


class WeightSettings(BaseModel):
    """Weights applied during deduplication scoring."""

    doi: float
    title: float
    authors: float
    year: float
    journal: float
    pages: float
    issue: float
    intercept: float


class PatternSettings(BaseModel):
    """Regular-expression pattern settings."""

    html_tag: str
    non_alphanumeric: str
    journal_punctuation: str
    part_number: str


class StopwordSettings(BaseModel):
    """Configured stopword lists."""

    title: list[str]
    journal: list[str]


class CsvColumnAliases(BaseModel):
    """Supported aliases for canonical CSV column names."""

    doi: tuple[str, ...] = ("doi",)
    title: tuple[str, ...] = ("title",)
    authors: tuple[str, ...] = ("authors", "author")
    year: tuple[str, ...] = ("year",)
    journal: tuple[str, ...] = ("journal",)
    pages: tuple[str, ...] = ("pages",)
    issue: tuple[str, ...] = ("issue", "number")
    volume: tuple[str, ...] = ("volume",)
    recordid: tuple[str, ...] = ("record_id", "recordid")
    duplicateid: tuple[str, ...] = ("duplicate_id", "duplicateid")


class CsvImportSettings(BaseModel):
    """Settings used when importing reference CSV files."""

    default_columns: tuple[str, ...] = (
        "doi",
        "title",
        "authors",
        "year",
        "journal",
        "pages",
        "issue",
        "volume",
        "recordid",
    )

    gold_standard_columns: tuple[str, ...] = ("duplicateid",)

    supported_encodings: tuple[str, ...] = (
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    )

    column_aliases: CsvColumnAliases = Field(default_factory=CsvColumnAliases)


class Settings(BaseSettings):
    """Top-level application settings loaded from YAML."""

    decision_threshold: float
    doi_mismatch_penalty: float
    thresholds: ThresholdSettings
    weights: WeightSettings
    patterns: PatternSettings
    roman_numerals: dict[str, int]
    stopwords: StopwordSettings
    languages: list[str]

    csv_import: CsvImportSettings = Field(default_factory=CsvImportSettings)

    model_config = SettingsConfigDict(
        yaml_file=USER_CONFIG_FILE_PATH
        if USER_CONFIG_FILE_PATH.is_file()
        else CONFIG_FILE_PATH,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Configure YAML and other settings sources."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
