"""Configuration management for the deduplication toolkit."""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"


class TitleThresholds(BaseModel):
    """Thresholds for paper titles comparison similarities."""

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


class AbstractThresholds(BaseModel):
    """Threshold for abstract comparisons."""

    similarity: float


class PaperThresholds(BaseModel):
    """Threshold for paper comparisons."""

    match: float


class ThresholdSettings(BaseModel):
    """Combined threshold settings."""

    doi_partial_match: float
    partial_title_match: float

    title: TitleThresholds = Field(default_factory=TitleThresholds)
    author: AuthorThresholds = Field(default_factory=AuthorThresholds)
    journal: JournalThresholds = Field(default_factory=JournalThresholds)
    abstract: AbstractThresholds = Field(default_factory=AbstractThresholds)
    paper: PaperThresholds = Field(default_factory=PaperThresholds)

    strong_metadata_match: float = Field(default=0.97)
    min_mismatches_for_veto: int = Field(default=2)


class WeightSettings(BaseModel):
    """Settings for weight to be applied to deduplication runs."""

    doi: float
    title: float
    authors: float
    year: float
    journal: float
    pages: float
    issue: float
    intercept: float
    abstract: float | None = None
    volume: float | None = None


class PatternSettings(BaseModel):
    """Pattern settings."""

    html_tag: str
    non_alphanumeric: str
    journal_punctuation: str
    part_number: str


class StopwordSettings(BaseModel):
    """Stopwords."""

    title: list[str]
    journal: list[str]


class Settings(BaseSettings):
    """Combined settings class, to be imported elsewhere to access these values."""

    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    weights: WeightSettings = Field(default_factory=WeightSettings)
    patterns: PatternSettings
    roman_numerals: dict[str, int]
    stopwords: StopwordSettings
    languages: list[str]

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_FILE_PATH,
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
        """Get the settings to read from yaml."""
        return (
            YamlConfigSettingsSource(settings_cls),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
