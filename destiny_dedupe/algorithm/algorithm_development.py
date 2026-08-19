"""
Algorithm development functions for model training and evaluation.

This module provides utilities for building and calibrating deduplication models
on labeled datasets. Functions handle data loading, record validation, field
comparison, and pair scoring with early-stop diagnostics.
"""

import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from destiny_dedupe.algorithm.candidate_selection import (
    _MIN_BLOCK_SIZE,
    BLOCK_RULES,
    normalise_block_value,
)
from destiny_dedupe.algorithm.record_cache import build_record_cache
from destiny_dedupe.algorithm.record_resolution import (
    record_level_metrics_for_threshold,
)
from destiny_dedupe.data_models import GoldStandardPaper, Paper
from destiny_dedupe.dedupe import Deduper
from destiny_dedupe.early_stop import EARLY_STOP_RULES, ComparisonContext

SEED = 1234
DEDUPE_FIELDS = [
    "doi",
    "title",
    "authors",
    "year",
    "journal",
    "pages",
    "abstract",
    "volume",
    "issue",
]

random.seed(SEED)

__all__ = [
    "DEDUPE_FIELDS",
    "build_record_cache",
    "compare_target_fields",
    "read_process_data_from_file",
    "record_level_metrics_for_threshold",
    "score_pairs_with_early_stop",
]


def read_process_data_from_file(
    filepath: Path,
    cols_to_drop: list = [  # noqa: B006
        "endnote",
        "bond",
        "asysd",
        "gold",
        "label",
        "nbond",
        "nendnote",
        "nasysd",
        "ngold",
    ],
    **kwargs,
) -> pd.DataFrame:
    """
    Load and preprocess tabular reference data from CSV file.

    Deprecated: Prefer app.import_references.load_reference_csv with CsvLoadConfig
    for new code paths and notebook workflows.

    Reads CSV file, lowercases all column names, drops specified columns,
    and renames common field aliases (author → authors, number → issue).

    Args:
        filepath: Path to CSV file containing reference records.
        cols_to_drop: List of column names to exclude from output. Defaults to
            non-essential columns like endnote, bond, gold, label variants.
        **kwargs: Additional keyword arguments passed to pd.read_csv() (e.g.,
            encoding, sep, dtype).

    Returns:
        pd.DataFrame: Preprocessed dataframe with normalized column names and
            selected columns. All column names are lowercase.

    """
    refdata = pd.read_csv(filepath, **kwargs)
    refdata.columns = [col.lower() for col in refdata.columns]
    final_cols = [col for col in refdata.columns if col not in cols_to_drop]
    refdata = refdata[final_cols]
    return refdata.rename(columns={"number": "issue", "author": "authors"})


def compare_target_fields(
    record_a: Paper,
    record_b: Paper,
    deduper: Deduper,
    fields: list = DEDUPE_FIELDS,
) -> dict[str, float]:
    """
    Compute similarity scores for specified fields between two records.

    Calls the deduper's compare_* methods (compare_doi, compare_title, etc.)
    for each field and returns a dict of field name → similarity score (0-1).
    Errors in individual field comparisons are logged and return None for
    that field.

    Args:
        record_a: First paper record to compare.
        record_b: Second paper record to compare.
        deduper: Deduper instance with compare_* methods.
        fields: List of field names to score. Must correspond to deduper
            methods named compare_{field}. Defaults to DEDUPE_FIELDS.

    Returns:
        dict[str, float]: Field name to similarity score mapping. Score is None
            if the comparison method fails or is unavailable.

    """
    scores = {}
    for field in fields:
        compare_method = getattr(deduper, f"compare_{field}", None)
        if compare_method:
            try:
                scores[field] = compare_method(record_a, record_b).score
            except Exception as e:  # noqa: BLE001
                scores[field] = None
                logger.warning(f"compare_{field} failed: {e}")
        else:
            scores[field] = None

    return scores  # ty:ignore


def score_pairs_with_early_stop(
    pairs_df: pd.DataFrame,
    record_cache: dict[int, GoldStandardPaper],
    score_fields: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score candidate pairs and diagnose early-stopping decisions.

    For each pair, evaluates early-stop rules (veto checks) and computes
    field-level similarity scores. Returns detailed scoring results and a
    summary of early-stop rule application frequencies.

    Args:
        pairs_df: DataFrame with candidate pairs (columns: id_a, id_b, is_dupe).
        record_cache: Dictionary of validated GoldStandardPaper records keyed by ID.
        score_fields: List of fields to score (default DEDUPE_FIELDS). Must
            correspond to Deduper.compare_* methods.

    Returns:
        tuple of:
            - pd.DataFrame: One row per pair with columns [id_a, id_b, is_dupe,
              early_stop_rule, field1_score, field2_score, ...]. early_stop_rule
              is None if no rule triggered; otherwise the reason string.
            - pd.DataFrame: Summary of early-stop rule triggers with columns
              [early_stop_rule, count], sorted by count descending. Empty if
              no rules triggered.

    """
    if not record_cache:
        return pd.DataFrame(), pd.DataFrame(columns=["early_stop_rule", "count"])

    any_paper = next(iter(record_cache.values()))
    deduper = Deduper(reference=any_paper, candidates=[any_paper])
    score_fields = score_fields or DEDUPE_FIELDS

    def _safe_should_early_stop(
        record_a: GoldStandardPaper, record_b: GoldStandardPaper
    ) -> str | None:
        ctx = ComparisonContext.model_construct(
            deduper=deduper,
            record_a=record_a,
            record_b=record_b,
        )
        for rule in EARLY_STOP_RULES:
            if rule.check(ctx):
                return rule.reason
        return None

    results = []
    early_stop_counter: Counter[str] = Counter()

    for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df)):
        rec_a = record_cache.get(int(row.id_a))
        rec_b = record_cache.get(int(row.id_b))
        if rec_a is None or rec_b is None:
            continue

        early_stop_reason = _safe_should_early_stop(rec_a, rec_b)
        key = early_stop_reason if early_stop_reason is not None else "__no_stop__"
        early_stop_counter[key] += 1

        field_scores = compare_target_fields(rec_a, rec_b, deduper, fields=score_fields)
        result = {
            "id_a": row.id_a,
            "id_b": row.id_b,
            "is_dupe": row.is_dupe,
            "early_stop_rule": early_stop_reason,
        }
        for field in score_fields:
            result[field] = field_scores.get(field)
        results.append(result)

    summary_df = pd.DataFrame(
        [
            {"early_stop_rule": k, "count": v}
            for k, v in early_stop_counter.items()
            if k != "__no_stop__"
        ]
    )
    if not summary_df.empty:
        summary_df = summary_df.sort_values("count", ascending=False).reset_index(
            drop=True
        )

    return pd.DataFrame(results), summary_df


def build_blocked_pairs_from_df(
    df: pd.DataFrame,
    block_rules: list[list[str]] = BLOCK_RULES,
    id_column: str = "recordid",
    dup_column: str | None = "duplicateid",
    *,
    include_block_rules: bool = False,
) -> pd.DataFrame:
    """
    Generate candidate record pairs using blocking rules.

    Groups records by normalized field values according to blocking rules.
    Within each group, creates all pairs and, when duplicate labels are
    available, marks them as duplicates if both records share the same
    duplicate_id. Removes duplicate pairs and filters groups smaller than 2
    records.

    Args:
        df: DataFrame with records and duplicate labels.
        block_rules: List of field combinations to block on (default BLOCK_RULES).
            Each rule is a list of field names; records are grouped by
            normalized values of those fields.
        id_column: Column name for record IDs (default "recordid").
        dup_column: Column name for duplicate group IDs. Set to None for
            non-gold-standard datasets where duplicate labels are unavailable.
        include_block_rules: If True, add 'block_rules' column showing which
            rules generated each pair. Defaults to False.

    Returns:
        pd.DataFrame: Columns are [id_a, id_b] for unlabeled datasets, or
            [id_a, id_b, is_dupe] when dup_column is provided. May also include
            block_rules. Rows are unique candidate pairs ordered by discovery.

    """
    include_labels = dup_column is not None
    dup_lookup = df.set_index(id_column)[dup_column].to_dict() if include_labels else {}
    seen: set[tuple[int, int]] = set()
    rows: list[tuple[int, int] | tuple[int, int, int]] = []
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

                if include_labels:
                    dup_a = dup_lookup.get(id_a)
                    dup_b = dup_lookup.get(id_b)
                    is_dupe = int(
                        pd.notna(dup_a) and pd.notna(dup_b) and dup_a == dup_b
                    )
                    rows.append((id_a, id_b, is_dupe))
                else:
                    rows.append((id_a, id_b))
                seen.add(key)

    columns = ["id_a", "id_b", "is_dupe"] if include_labels else ["id_a", "id_b"]
    pairs_df = pd.DataFrame(rows, columns=columns)
    if include_block_rules and not pairs_df.empty:
        pairs_df["block_rules"] = pairs_df.apply(
            lambda r: " | ".join(
                sorted(pair_rules.get((int(r["id_a"]), int(r["id_b"])), set()))
            ),
            axis=1,
        )
    return pairs_df
