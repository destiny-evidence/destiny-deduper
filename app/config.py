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

    veto: float = Field(default=0.80)
    similarity: float = Field(default=0.97)


class AuthorThresholds(BaseModel):
    """Threshold for author comparisons."""

    similarity: float = Field(default=0.92)
    first_chars: int = Field(default=7)


class JournalThresholds(BaseModel):
    """Threshold for journal-related comparisons."""

    similarity: float = Field(default=0.70)
    strong_similarity: float = Field(default=0.92)
    abbreviation: float = Field(default=0.70)


class AbstractThresholds(BaseModel):
    """Threshold for abstract comparisons."""

    similarity: float = Field(default=0.70)


class PaperThresholds(BaseModel):
    """Threshold for paper comparisons."""

    match: float = Field(default=0.90)


class ThresholdSettings(BaseModel):
    """Combined threshold settings."""

    doi_partial_match: float = Field(default=0.90)
    partial_title_match: float = Field(default=0.90)

    title: TitleThresholds = Field(default_factory=TitleThresholds)
    author: AuthorThresholds = Field(default_factory=AuthorThresholds)
    journal: JournalThresholds = Field(default_factory=JournalThresholds)
    abstract: AbstractThresholds = Field(default_factory=AbstractThresholds)
    paper: PaperThresholds = Field(default_factory=PaperThresholds)

    strong_metadata_match: float = Field(default=0.97)
    min_mismatches_for_veto: int = Field(default=2)


class WeightSettings(BaseModel):
    """Settings for weight to be applied to deduplication runs."""

    doi: float = 2.28
    title: float = 6.38
    authors: float = 2.44
    year: float = 0.12
    journal: float = 1.42
    pages: float = 1.01
    abstract: float = -0.29
    volume: float = 0.38
    issue: float = -0.22


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
