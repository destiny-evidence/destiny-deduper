"""Candidate pair selection and blocking rules for deduplication."""

import re
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

import pandas as pd

from destiny_dedupe.data_models import Paper
from destiny_dedupe.normalisers import (
    normalise_doi,
    normalise_pages,
    strip_doi_punctuation,
)

BLOCK_RULES = [
    ["title"],
    ["abstract"],
    ["doi"],
    ["year", "journal"],
    ["year", "pages"],
    ["year", "volume"],
    ["pages", "volume"],
]

_MIN_BLOCK_SIZE = 2


def normalise_block_value(field: str, val: object) -> str | None:
    """
    Normalize field value for blocking/grouping operations.

    Applies field-specific normalization (lowercase, whitespace cleanup, DOI
    stripping, etc.) to prepare values for blocking rule comparisons. Returns
    None if value is missing or invalid.

    Args:
        field: Field name (e.g., 'doi', 'title', 'year'). Determines
            normalization rules applied.
        val: Raw field value to normalize. Can be string, int, float, or None.

    Returns:
        str | None: Normalized string suitable for blocking comparisons, or
            None if value is missing/invalid/empty.

    """
    if pd.isna(val) or val is None or val == "":  # ty:ignore[no-matching-overload]
        return None

    text = str(val).strip()
    if text == "":
        return None

    result: str | None
    if field == "doi":
        doi = normalise_doi(text)
        result = strip_doi_punctuation(doi) if doi else None
    elif field == "pages":
        pages = normalise_pages(text)
        result = re.sub(r"\s+", "", pages).lower() if pages else None
    elif field in {"year", "volume", "issue"}:
        cleaned = re.sub(r"\s+", "", text)
        try:
            result = str(int(float(cleaned)))
        except ValueError:
            result = cleaned.lower()
    else:
        result = re.sub(r"\s+", " ", text.lower())

    return result


def build_blocked_pairs(
    papers: Sequence[Paper],
    block_rules: Sequence[Sequence[str]] = BLOCK_RULES,
    *,
    include_block_rules: bool = False,
) -> pd.DataFrame:
    """Generate candidate pairs using indexes in the supplied paper sequence."""
    seen: set[tuple[int, int]] = set()
    rows: list[tuple[int, int]] = []
    pair_rules: dict[tuple[int, int], set[str]] = {}

    for rule in block_rules:
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        rule_label = ",".join(rule)

        for index, paper in enumerate(papers):
            key: list[str] = []

            for field in rule:
                value = normalise_block_value(
                    field,
                    getattr(paper, field, None),
                )

                if value is None:
                    break

                key.append(value)
            else:
                groups[tuple(key)].append(index)

        for group in groups.values():
            if len(group) < _MIN_BLOCK_SIZE:
                continue

            for index_a, index_b in combinations(group, 2):
                pair = (index_a, index_b)

                if include_block_rules:
                    pair_rules.setdefault(pair, set()).add(rule_label)

                if pair in seen:
                    continue

                rows.append(pair)
                seen.add(pair)

    pairs_df = pd.DataFrame(
        rows,
        columns=["index_a", "index_b"],
    )

    if include_block_rules and not pairs_df.empty:
        pairs_df["block_rules"] = pairs_df.apply(
            lambda row: " | ".join(
                sorted(
                    pair_rules[
                        (
                            int(row["index_a"]),
                            int(row["index_b"]),
                        )
                    ]
                )
            ),
            axis=1,
        )

    return pairs_df
