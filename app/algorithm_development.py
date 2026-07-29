"""
Algorithm development functions for model training and evaluation.

This module provides utilities for building and calibrating deduplication models
on labeled datasets. Functions handle data loading, record validation, field
comparison, and pair scoring with early-stop diagnostics.
"""

import random
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from app.data_models import GoldStandardPaper, Paper
from app.dedupe import Deduper
from app.early_stop import EARLY_STOP_RULES, ComparisonContext
from app.record_cache import build_record_cache
from app.record_resolution import record_level_metrics_for_threshold

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
    warnings.warn(
        "read_process_data_from_file is deprecated. "
        "Use app.import_references.load_reference_csv instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
) -> dict[str, float | int | None]:
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
                scores[field] = compare_method(record_a, record_b)
            except Exception as e:  # noqa: BLE001
                scores[field] = None
                logger.warning(f"compare_{field} failed: {e}")
        else:
            scores[field] = None

    return scores


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
        return pd.DataFrame(), pd.DataFrame(
            columns=pd.Index(["early_stop_rule", "count"])
        )

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
