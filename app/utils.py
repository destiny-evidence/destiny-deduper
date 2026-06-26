"""General utility functions for the deduplication toolkit."""

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

from loguru import logger
from rapidfuzz.distance import JaroWinkler as _JaroWinkler
from rapidfuzz.distance import Levenshtein as _Levenshtein

from app.config import get_settings
from app.regexes import HTML_TAG_RE, JOURNAL_PUNCT_RE, NON_ALNUM_RE

settings = get_settings()

# Regex to extract part numbers from titles, e.g. "Part 1", "Part II", "part 3"
_ROMAN_VALUES = settings.roman_numerals
_TITLE_STOP_WORDS = settings.stopwords.title
_JOURNAL_STOP_WORDS = settings.stopwords.journal

# languages
LANGUAGE_NAMES = settings.languages


def split_title_segments(title: str) -> tuple[str, str | None]:
    """
    Split title into first and remaining segments
    if separator present, otherwise strip whitespace.
    """
    for sep in [":", ".", ";", "/", "|", " (", " ["]:
        idx = title.find(sep)
        if idx > 0 and idx < len(title) - 3:  # plausible alternative segment
            first = title[:idx].strip()
            rest = title[idx + 1 :].strip()
            return first, rest
    return title.strip(), None


def contains_language_name(text: str) -> bool:
    """Check if text contains a language name."""
    if not text:
        return False
    return any(lang in text for lang in LANGUAGE_NAMES)


def roman_to_int(s: str) -> int | None:
    """Convert a roman numeral string to an integer, or return None if not a valid roman numeral."""
    s = s.lower()
    if not s or not all(c in _ROMAN_VALUES for c in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total if total > 0 else None


def clean_title_for_partial_ratio(title: str) -> str:
    """
    Normalise a title for partial-ratio comparison.

    Strips HTML/XML markup (e.g. <INF>, <SUP>), punctuation, and common
    English stop words, then collapses whitespace. This makes superficial
    formatting differences (subscripts, brackets, punctuation) invisible
    to the partial-ratio veto.
    """
    t = title.lower()
    t = HTML_TAG_RE.sub(" ", t)  # remove <INF>…</INF>, <b>, etc.
    t = NON_ALNUM_RE.sub(" ", t)  # drop punctuation / brackets
    tokens = [w for w in t.split() if w not in _TITLE_STOP_WORDS]
    return " ".join(tokens)


def is_journal_abbreviation_match(a: str, b: str) -> bool:
    """
    Return True if one journal string is plausibly an abbreviation of the other.

    Two strategies are tried (after lowercasing and stripping punctuation):

    1. **Acronym** — the shorter form is a single token whose letters match the
       initial letter of each significant word in the longer form.
       e.g. "cmaj" == initials of "canadian medical association journal".

    2. **Token-prefix** — each token of the shorter form is a prefix of some
       token in the longer form (in order, skipping stop words).
       e.g. ["proc", "soc", "e"] matches ["proceedings", "society", "experimental",
       "biology", "and", "medicine"] after skipping "of", "the", "for".
    """

    def _tokens(s: str) -> list[str]:
        return JOURNAL_PUNCT_RE.sub(" ", s.lower()).split()

    toks_a = _tokens(a)
    toks_b = _tokens(b)
    if not toks_a or not toks_b:
        return False

    # Ensure `short` is the potentially abbreviated form
    short, long = (toks_a, toks_b) if len(toks_a) <= len(toks_b) else (toks_b, toks_a)

    # Strategy 1: single-token acronym
    if len(short) == 1:
        significant = [w for w in long if w not in _JOURNAL_STOP_WORDS]
        initials = "".join(w[0] for w in significant)
        if short[0] == initials:
            return True

    # Strategy 2: ordered token-prefix match (skipping stop words in the long form)
    long_sig = [w for w in long if w not in _JOURNAL_STOP_WORDS]
    long_idx = 0
    for s_tok in short:
        matched = False
        while long_idx < len(long_sig):
            if long_sig[long_idx].startswith(s_tok):
                matched = True
                long_idx += 1
                break
            long_idx += 1
        if not matched:
            return False
    return True


def calculate_string_distance(
    string_a: str,
    string_b: str,
    string_distance_algorithm: Literal["jaro_winkler", "levenshtein"] = "jaro_winkler",
) -> float:
    """
    Calculate string distance, genericly.

    Args:
        string_a (str): a string
        string_b (str): another string
        string_distance_algorithm (Literal["jaro_winkler", "levenshtein"]): algorithm to use

    Returns:
        float: between 0 and 1

    """
    method_map: dict[str, Callable] = {
        "jaro_winkler": _JaroWinkler.similarity,
        "levenshtein": _Levenshtein.normalized_similarity,
    }

    method: Callable | None = method_map.get(string_distance_algorithm)
    logger.debug(f"selected method: {method}")
    if not method:
        no_method_for_alg_err_msg = (
            f"No method for algorithm: {string_distance_algorithm}"
        )
        raise ValueError(no_method_for_alg_err_msg)
    return method(string_a, string_b)


def extract_abstract_numbers(text: str) -> dict[str, int]:
    """
    Extract numeric tokens from abstract text as a multiset.

    Args:
        text: Abstract text to process

    Returns:
        Dictionary mapping numeric tokens to their frequency

    """
    if not text:
        return {}
    tokens = re.findall(r"\d+(?:\.\d+)?", text)
    # Create a dictionary with counts (multiset)
    result: dict[str, int] = {}
    for token in tokens:
        result[token] = result.get(token, 0) + 1
    return result


def strip_doi_punctuation(doi: str) -> str:
    """
    Strip . - _ from the DOI suffix for fuzzy comparison.

    Handles common database artefacts such as a missing dot (100B3BJJ vs
    100B3.BJJ) or underscore/hyphen substitution (eurrev_2018 vs
    eurrev-2018).  Only the part after the first '/' is altered so the
    registrant prefix (10.XXXX) is never affected.

    Args:
        doi: DOI string to process

    Returns:
        DOI string with punctuation stripped from suffix

    """
    if "/" not in doi:
        return doi
    prefix, suffix = doi.split("/", 1)
    return prefix + "/" + re.sub(r"[-_.]", "", suffix)
