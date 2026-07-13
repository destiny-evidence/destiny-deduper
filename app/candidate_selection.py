"""Candidate pair selection and blocking rules for deduplication."""

import re
from itertools import combinations

import pandas as pd

from app.normalisers import normalise_doi, normalise_pages, strip_doi_punctuation

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
    if pd.isna(val) or val is None or val == "":
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
    df: pd.DataFrame,
    block_rules: list[list[str]] = BLOCK_RULES,
    id_column: str = "recordid",
    dup_column: str = "duplicateid",
    *,
    include_block_rules: bool = False,
) -> pd.DataFrame:
    """
    Generate candidate record pairs using blocking rules with duplicate labels.

    Groups records by normalized field values according to blocking rules.
    Within each group, creates all pairs and labels them as duplicates if both
    records share the same duplicate_id (for gold-standard datasets). Removes
    duplicate pairs and filters groups smaller than 2 records.

    Args:
        df: DataFrame with records and duplicate labels.
        block_rules: List of field combinations to block on (default BLOCK_RULES).
            Each rule is a list of field names; records are grouped by
            normalized values of those fields.
        id_column: Column name for record IDs (default "recordid").
        dup_column: Column name for duplicate group IDs (default "duplicateid").
        include_block_rules: If True, add 'block_rules' column showing which
            rules generated each pair. Defaults to False.

    Returns:
        pd.DataFrame: Columns are [id_a, id_b, is_dupe] (and optionally
            block_rules). Each row is a unique candidate pair with is_dupe=1
            if records share a duplicate_id, else 0. Ordered by (id_a, id_b).

    """
    dup_lookup = df.set_index(id_column)[dup_column].to_dict()
    seen: set[tuple[int, int]] = set()
    rows: list[tuple[int, int, int]] = []
    pair_rules: dict[tuple[int, int], set[str]] = {}

    for rule in block_rules:
        missing = [field for field in rule if field not in df.columns]
        if missing:
            continue

        rule_label = ",".join(rule)
        subset = df[[id_column, *rule]].copy()
        norm_cols = []
        for field in rule:
            norm_col = f"{field}_norm"
            subset[norm_col] = subset[field].apply(
                lambda v, field_name=field: normalise_block_value(field_name, v)
            )
            norm_cols.append(norm_col)

        subset = subset.dropna(subset=norm_cols)
        subset["block_key"] = subset[norm_cols].apply(tuple, axis=1)

        for _, group in subset.groupby("block_key"):
            ids = group[id_column].tolist()
            if len(ids) < _MIN_BLOCK_SIZE:
                continue

            for a, b in combinations(ids, 2):
                id_a, id_b = (a, b) if a < b else (b, a)
                key = (id_a, id_b)
                if include_block_rules:
                    pair_rules.setdefault(key, set()).add(rule_label)
                if key in seen:
                    continue

                dup_a = dup_lookup.get(id_a)
                dup_b = dup_lookup.get(id_b)
                is_dupe = int(pd.notna(dup_a) and pd.notna(dup_b) and dup_a == dup_b)
                rows.append((id_a, id_b, is_dupe))
                seen.add(key)

    pairs_df = pd.DataFrame(rows, columns=["id_a", "id_b", "is_dupe"])
    if include_block_rules and not pairs_df.empty:
        pairs_df["block_rules"] = pairs_df.apply(
            lambda r: " | ".join(
                sorted(pair_rules.get((int(r["id_a"]), int(r["id_b"])), set()))
            ),
            axis=1,
        )
    return pairs_df
