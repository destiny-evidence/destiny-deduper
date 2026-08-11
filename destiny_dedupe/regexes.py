"""Centralised source of truth for compiled regexes."""

import re

from destiny_dedupe.config import get_settings

settings = get_settings()

PART_NUMBER_RE = re.compile(
    settings.patterns.part_number,
    re.IGNORECASE,
)

HTML_TAG_RE = re.compile(settings.patterns.html_tag)
NON_ALNUM_RE = re.compile(settings.patterns.non_alphanumeric)
JOURNAL_PUNCT_RE = re.compile(settings.patterns.journal_punctuation)
