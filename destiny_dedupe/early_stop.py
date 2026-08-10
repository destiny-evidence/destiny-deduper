"""Functions and scaffolding for testing whether a pair fits early stopping rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz

if TYPE_CHECKING:
    from destiny_dedupe.data_models import Paper
    from destiny_dedupe.dedupe import Deduper

from destiny_dedupe.config import get_settings
from destiny_dedupe.normalisers import normalise_doi, normalise_part_number
from destiny_dedupe.regexes import PART_NUMBER_RE
from destiny_dedupe.utils import clean_title_for_partial_ratio

settings = get_settings()


PARTIAL_TITLE_MATCH_RATIO = settings.thresholds.title.partial_match_ratio
JOURNAL_MISMATCH_THRESHOLD = settings.thresholds.journal.similarity
PAPER_MISMATCH_THRESHOLD = settings.thresholds.paper.match
MAX_MISMATCH_COUNT = settings.thresholds.min_mismatches_for_veto
AUTHOR_FIRST_CHARS = settings.thresholds.author.first_chars
STRONG_METADATA_MATCH_THRESHOLD = settings.thresholds.strong_metadata_match
AUTHORS_SIM_THRESHOLD = settings.thresholds.author.similarity
JOURNAL_SIM_THRESHOLD = settings.thresholds.journal.similarity
TITLE_SIM_THRESHOLD = settings.thresholds.title.similarity
TITLE_VETO_THRESHOLD = settings.thresholds.title.veto
TITLE_SIM_THRESHOLD_LOWER = settings.thresholds.title.similarity_lower


class ComparisonContext(BaseModel):
    """
    Comparison context for early stopping rule evaluation.

    Contains the deduper instance and the pair of records being compared,
    allowing early stop rules to call comparison methods without circular
    imports.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deduper: Deduper

    record_a: Paper
    record_b: Paper


class EarlyStopRule(BaseModel):
    """Base model for an early stopping rule."""

    reason: str
    check: Callable[[ComparisonContext], bool]


# early stop check functions
def check_doi_pages_mismatch(ctx: ComparisonContext) -> bool:
    """Check DOI and pages aren't a mismatch for early stopping."""
    if (
        ctx.record_a.doi is not None
        and ctx.record_b.doi is not None
        and ctx.record_a.pages is not None
        and ctx.record_b.pages is not None
    ):
        norm_doi_a = normalise_doi(getattr(ctx.record_a.doi, "identifier", None))
        norm_doi_b = normalise_doi(getattr(ctx.record_b.doi, "identifier", None))
        doi_match = norm_doi_a == norm_doi_b if (norm_doi_a and norm_doi_b) else True
        pages_match = ctx.deduper.compare_pages(ctx.record_a, ctx.record_b) == 1.0

        if not doi_match and not pages_match:
            logger.debug("Early stop: DOI and PAGES both don't match")
            return True
    return False


def check_doi_pub_version_mismatch(ctx: ComparisonContext) -> bool:
    """Check if there's a DOI pub version mismatch."""
    doi_a = normalise_doi(getattr(ctx.record_a.doi, "identifier", None))
    doi_b = normalise_doi(getattr(ctx.record_b.doi, "identifier", None))

    if doi_a and doi_b and doi_a != doi_b:
        base_a = re.sub(r"\.pub\d+$", "", doi_a)
        base_b = re.sub(r"\.pub\d+$", "", doi_b)
        if base_a == base_b:
            logger.debug("Early stop: same DOI base but different .pubN version")
            return True

    return False


def check_part_number_mismatch(ctx: ComparisonContext) -> bool:
    """Check for part number mismatch for early stopping."""
    if ctx.record_a.title and ctx.record_b.title:
        part_match_a = PART_NUMBER_RE.search(ctx.record_a.title)
        part_match_b = PART_NUMBER_RE.search(ctx.record_b.title)
        if part_match_a and part_match_b:
            part_a = normalise_part_number(part_match_a.group(1))
            part_b = normalise_part_number(part_match_b.group(1))
            if part_a != part_b:
                logger.debug(
                    "Early stop: part number mismatch (%s vs %s)", part_a, part_b
                )
                return True
    return False


def check_partial_ratio(ctx: ComparisonContext) -> bool:
    """
    Check for partial ratio.

    if title overlap is too low, treat as non-duplicate.
    Bypass if both DOIs are present and confirmed to match.

    """
    if ctx.record_a.title and ctx.record_b.title:
        _norm_doi_a = (
            normalise_doi(getattr(ctx.record_a.doi, "identifier", None))
            if ctx.record_a.doi
            else None
        )
        _norm_doi_b = (
            normalise_doi(getattr(ctx.record_b.doi, "identifier", None))
            if ctx.record_b.doi
            else None
        )
        doi_confirmed_match = bool(
            _norm_doi_a and _norm_doi_b and _norm_doi_a == _norm_doi_b
        )
        if not doi_confirmed_match:
            clean_a = clean_title_for_partial_ratio(ctx.record_a.title)
            clean_b = clean_title_for_partial_ratio(ctx.record_b.title)
            partial = fuzz.partial_ratio(clean_a, clean_b) / 100.0
            if partial < PARTIAL_TITLE_MATCH_RATIO:
                logger.debug(
                    "Early stop: partial ratio %.3f below 0.90 threshold", partial
                )
                return True

    return False


def check_title_match_with_structural_conflict(ctx: ComparisonContext) -> bool:
    """
    Exact-title structural conflict: identical normalized titles but
    different journal packaging plus a page mismatch.

    Only veto an exact title match when we can *confirm* a DOI mismatch:
    both DOIs present, both normalise successfully, and they differ.
    If either DOI is absent or can't be normalised, we cannot confirm a
    mismatch, so we leave the pair to the scorer (conservative).

    """
    title_a = re.sub(r"\s+", " ", (ctx.record_a.title or "").strip().lower())
    title_b = re.sub(r"\s+", " ", (ctx.record_b.title or "").strip().lower())
    if title_a and title_a == title_b:
        norm_doi_a = (
            normalise_doi(getattr(ctx.record_a.doi, "identifier", None))
            if ctx.record_a.doi
            else None
        )
        norm_doi_b = (
            normalise_doi(getattr(ctx.record_b.doi, "identifier", None))
            if ctx.record_b.doi
            else None
        )
        doi_confirmed_mismatch = (
            norm_doi_a is not None
            and norm_doi_b is not None
            and norm_doi_a != norm_doi_b
        )
        if doi_confirmed_mismatch:
            journal_mismatch = (
                ctx.record_a.journal is not None
                and ctx.record_b.journal is not None
                and ctx.deduper.compare_journal(ctx.record_a, ctx.record_b)
                < JOURNAL_MISMATCH_THRESHOLD
            )
            pages_mismatch = (
                ctx.record_a.pages is not None
                and ctx.record_b.pages is not None
                and ctx.deduper.compare_pages(ctx.record_a, ctx.record_b)
                < PAPER_MISMATCH_THRESHOLD
            )
            if journal_mismatch and pages_mismatch:
                logger.debug(
                    "Early stop: exact title match with confirmed DOI mismatch, journal mismatch and pages mismatch",
                )
                return True

    return False


def check_title_and_metadata_mismatch(ctx: ComparisonContext) -> bool:
    """
    Title + metadata veto.

    Returns True (not a duplicate) if:
      1. Title similarity is below TITLE_VETO_THRESHOLD, or
      2. Title similarity is below TITLE_SIM_THRESHOLD and there are
         enough metadata mismatches. If the title similarity is even lower
         (below TITLE_SIM_THRESHOLD_LOWER), it counts as an additional
         mismatch, making the veto easier to trigger.
    """
    title_sim = ctx.deduper.compare_title(ctx.record_a, ctx.record_b)

    # Extremely different titles -> definitely not duplicates.
    if title_sim < TITLE_VETO_THRESHOLD:
        return True

    if title_sim < TITLE_SIM_THRESHOLD:
        pages_mismatch = (
            ctx.record_a.pages is not None
            and ctx.record_b.pages is not None
            and ctx.record_a.pages != ctx.record_b.pages
        )

        volume_mismatch = (
            ctx.record_a.volume is not None
            and ctx.record_b.volume is not None
            and ctx.record_a.volume != ctx.record_b.volume
        )

        # Check first author's first AUTHOR_FIRST_CHARS characters.
        first_author_mismatch = False
        if ctx.record_a.authors and ctx.record_b.authors:
            auth_a = getattr(ctx.record_a.authors[0], "author_name", None) or getattr(
                ctx.record_a.authors[0], "display_name", ""
            )
            auth_b = getattr(ctx.record_b.authors[0], "author_name", None) or getattr(
                ctx.record_b.authors[0], "display_name", ""
            )

            if auth_a and auth_b:
                first_author_mismatch = (
                    auth_a[:AUTHOR_FIRST_CHARS] != auth_b[:AUTHOR_FIRST_CHARS]
                )

        mismatch_count = sum([pages_mismatch, volume_mismatch, first_author_mismatch])

        # If the titles are even less similar, treat that as one extra mismatch.
        adjusted_mismatch_count = mismatch_count
        if title_sim < TITLE_SIM_THRESHOLD_LOWER:
            adjusted_mismatch_count += 1

        if adjusted_mismatch_count >= MAX_MISMATCH_COUNT:
            mismatch_fields = [
                f
                for f, v in [
                    ("pages", pages_mismatch),
                    ("volume", volume_mismatch),
                    ("first author", first_author_mismatch),
                ]
                if v
            ]

            if title_sim < TITLE_SIM_THRESHOLD_LOWER:
                mismatch_fields.append("very low title similarity")

            logger.debug(
                "Early stop: title similarity %.3f with mismatches in: %s",
                title_sim,
                ", ".join(mismatch_fields),
            )
            return True

    return False


def check_year_gap_abstract_numeric_mismatch(
    ctx: ComparisonContext,
) -> bool:
    """
    Year-gap + abstract-numeric conflict:
    if core metadata still looks very similar but publication years diverge,
    use abstract numbers as a disambiguation veto.

    """
    ctx.record_a = ctx.record_a
    ctx.record_b = ctx.record_b

    if ctx.record_a.year is None or ctx.record_b.year is None:
        return False

    if abs(int(ctx.record_a.year) - int(ctx.record_b.year)) <= 1:
        return False

    title_sim = ctx.deduper.compare_title(ctx.record_a, ctx.record_b)
    authors_sim = ctx.deduper.compare_authors(ctx.record_a, ctx.record_b)
    journal_sim = ctx.deduper.compare_journal(ctx.record_a, ctx.record_b)

    strong_metadata_match = title_sim >= STRONG_METADATA_MATCH_THRESHOLD and (
        authors_sim >= AUTHORS_SIM_THRESHOLD or journal_sim >= JOURNAL_SIM_THRESHOLD
    )

    return strong_metadata_match and ctx.deduper.has_abstract_conflict(
        ctx.record_a, ctx.record_b
    )


EARLY_STOP_RULES = [
    EarlyStopRule(reason="doi_and_pages_mismatch", check=check_doi_pages_mismatch),
    EarlyStopRule(
        reason="doi_pub_version_mismatch", check=check_doi_pub_version_mismatch
    ),
    EarlyStopRule(
        reason="part_number_mismatch",
        check=check_part_number_mismatch,
    ),
    EarlyStopRule(reason="partial_ratio_too_low", check=check_partial_ratio),
    EarlyStopRule(
        reason="exact_title_with_structural_conflict",
        check=check_title_match_with_structural_conflict,
    ),
    EarlyStopRule(
        reason="title_with_metadata_mismatch", check=check_title_and_metadata_mismatch
    ),
    EarlyStopRule(
        reason="year_gap_with_abstract_conflict",
        check=check_year_gap_abstract_numeric_mismatch,
    ),
]
