"""
Evaluation functions for evaluation of deduplication tool on gold standard datasets.

This module provides utilities for building and calibrating deduplication models
on labeled datasets. Functions handle data loading, record validation, field
comparison, and pair scoring with early-stop diagnostics.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm

from destiny_dedupe.algorithm.candidate_selection import (
    BLOCK_RULES,
    build_blocked_pairs,
)
from destiny_dedupe.algorithm.import_references import (
    DEFAULT_COLUMNS,
    CsvLoadConfig,
    load_reference_csv,
)
from destiny_dedupe.algorithm.record_resolution import (
    record_level_metrics_for_threshold,
)
from destiny_dedupe.data_models import GoldStandardPaper
from destiny_dedupe.dedupe import INTERCEPT, Deduper, ScorePairConfig
from destiny_dedupe.dedupe import WEIGHTS as MODEL_WEIGHTS

repo_root = Path.cwd()
if not (repo_root / "app").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

RESULTS_ROOT = repo_root / "notebooks" / "results"
GLOBAL_SEED = 42
MAX_PAIRS = None
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

SCORE_WEIGHTS = {
    field: weight
    for field, weight in MODEL_WEIGHTS.items()
    if field not in {"volume", "abstract"}
}
SCORE_CONFIG = ScorePairConfig(
    weights=SCORE_WEIGHTS,
    intercept=INTERCEPT,
    fields=list(SCORE_WEIGHTS),
)
SCORE_FIELDS = list(SCORE_WEIGHTS)

INSPECT_FIELDS = [
    "title",
    "authors",
    "year",
    "journal",
    "volume",
    "issue",
    "pages",
    "doi",
    "abstract",
]


def add_gold_standard_labels(
    pairs_df: pd.DataFrame,
    papers: Sequence[GoldStandardPaper],
) -> pd.DataFrame:
    """Add source record IDs and ground-truth duplicate labels."""
    labelled = pairs_df.copy()

    labelled["id_a"] = labelled["index_a"].map(
        lambda index: papers[int(index)].recordid
    )
    labelled["id_b"] = labelled["index_b"].map(
        lambda index: papers[int(index)].recordid
    )

    def is_duplicate(row: pd.Series) -> int:
        paper_a = papers[int(row["index_a"])]
        paper_b = papers[int(row["index_b"])]

        return int(
            paper_a.duplicateid is not None
            and paper_b.duplicateid is not None
            and paper_a.duplicateid == paper_b.duplicateid
        )

    labelled["is_dupe"] = labelled.apply(
        is_duplicate,
        axis=1,
    )

    return labelled


def score_candidate_pairs(
    papers: Sequence[GoldStandardPaper],
    pairs_df: pd.DataFrame,
    config: ScorePairConfig = SCORE_CONFIG,
) -> pd.DataFrame:
    """Score blocked pairs with the app's weighted Deduper implementation."""
    output_columns = [
        *pairs_df.columns,
        "prob",
        "early_stop",
        *[f"score_{field}" for field in config.fields or []],
    ]
    if not papers or pairs_df.empty:
        return pd.DataFrame(columns=output_columns)

    paper_by_id = {
        int(paper.recordid): paper for paper in papers if paper.recordid is not None
    }
    pair_ids = {
        int(recordid) for column in ("id_a", "id_b") for recordid in pairs_df[column]
    }
    missing_ids = sorted(pair_ids.difference(paper_by_id))
    if missing_ids:
        msg = f"Candidate pairs reference missing record IDs: {missing_ids[:10]}"
        raise ValueError(msg)

    deduper = Deduper(reference=papers[0], candidates=[])
    results = []

    for pair in tqdm(
        pairs_df.itertuples(index=False),
        total=len(pairs_df),
        desc="Scoring candidate pairs",
    ):
        probability, field_scores, early_stop = deduper.score_pair(
            paper_by_id[int(pair.id_a)],
            paper_by_id[int(pair.id_b)],
            config=config,
        )

        result = pair._asdict()
        result.update(prob=probability, early_stop=early_stop)
        for field in config.fields or []:
            result[f"score_{field}"] = field_scores.get(field)
        results.append(result)

    return pd.DataFrame(results, columns=output_columns)


BINARY_CLASS_COUNT = 2


def pair_metrics_for_threshold(
    scored_df: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    """Compute pair-level metrics for one probability threshold."""
    y_true = scored_df["is_dupe"].astype(int)
    y_prob = scored_df["prob"].astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if y_true.nunique() == BINARY_CLASS_COUNT
            else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(y_true, y_prob))
            if y_true.sum() > 0
            else float("nan")
        ),
    }


def side_by_side_pair_table(
    records_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    fields: Sequence[str],
) -> pd.DataFrame:
    """Attach selected Paper fields for both sides of each candidate pair."""
    available_fields = [field for field in fields if field in records_df.columns]
    left = records_df[["recordid", *available_fields]].copy()
    left.columns = ["id_a", *[f"{field}_a" for field in available_fields]]
    right = records_df[["recordid", *available_fields]].copy()
    right.columns = ["id_b", *[f"{field}_b" for field in available_fields]]

    review_df = pair_df.merge(left, on="id_a", how="left").merge(
        right, on="id_b", how="left"
    )
    side_by_side_columns = [
        column for field in available_fields for column in (f"{field}_a", f"{field}_b")
    ]
    pair_columns = [column for column in pair_df.columns if column in review_df.columns]
    return review_df[[*pair_columns, *side_by_side_columns]]


def find_best_threshold(
    metrics_df: pd.DataFrame,
    min_sensitivity: float = 0.99,
) -> pd.Series | pd.DataFrame | None:
    """Maximise specificity among thresholds meeting the sensitivity floor."""
    eligible = metrics_df[metrics_df["sensitivity"] >= min_sensitivity]
    if eligible.empty:
        return None
    return eligible.loc[eligible["specificity"].idxmax()]


def build_labelled_pair_table(
    papers: Sequence[GoldStandardPaper],
) -> pd.DataFrame:
    """Build a table of all labelled record pairs from gold-standard papers."""
    rows = []

    for index_a, index_b in build_blocked_pairs(papers):  # ty:ignore TODO: fix
        paper_a = papers[index_a]
        paper_b = papers[index_b]

        rows.append(
            {
                "index_a": index_a,
                "index_b": index_b,
                "id_a": paper_a.recordid,
                "id_b": paper_b.recordid,
                "is_dupe": int(
                    paper_a.duplicateid is not None
                    and paper_a.duplicateid == paper_b.duplicateid
                ),
            }
        )

    return pd.DataFrame(rows)


def _export_error_reviews(
    dataset_dir: Path,
    records_df: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> None:
    """Export false-positive and false-negative pairs for manual review."""
    is_dupe = scored_df["is_dupe"].astype(int)
    probabilities = scored_df["prob"].astype(float)

    for threshold in THRESHOLDS:
        predicted_dupe = probabilities >= threshold

        error_masks = {
            "false_positives": (is_dupe == 0) & predicted_dupe,
            "false_negatives": (is_dupe == 1) & ~predicted_dupe,
        }

        for error_name, mask in error_masks.items():
            review_df = side_by_side_pair_table(
                records_df,
                scored_df.loc[mask].copy(),
                INSPECT_FIELDS,
            )

            review_df.to_csv(
                dataset_dir / f"{error_name}_{threshold}.csv",
                index=False,
            )


def _select_and_report_thresholds(
    pair_metrics_df: pd.DataFrame,
    record_metrics_df: pd.DataFrame,
    min_sensitivity: float,
) -> tuple[object | None, object | None]:
    """Select and report the best pair-level and record-level thresholds."""
    best_pair = find_best_threshold(
        pair_metrics_df,
        min_sensitivity,
    )
    best_record = find_best_threshold(
        record_metrics_df,
        min_sensitivity,
    )

    for _level_name, metrics_df, best in (
        ("pair-level", pair_metrics_df, best_pair),
        ("record-level", record_metrics_df, best_record),
    ):
        if best is not None:
            continue

        metrics_df.loc[metrics_df["sensitivity"].idxmax()]

    return best_pair, best_record


def run_dataset_pipeline(
    dataset_name: str,
    data_path: Path,
    max_pairs: int | None = MAX_PAIRS,
) -> dict[str, object]:
    """Run weighted deduplication evaluation for one gold-standard dataset."""
    loaded_papers = load_reference_csv(
        data_path,
        CsvLoadConfig(
            columns=DEFAULT_COLUMNS,
            include_gold_standard=True,
        ),
    )

    if not loaded_papers:
        msg = f"No valid papers were loaded from {data_path}"
        raise ValueError(msg)

    # load_reference_csv() is typed as returning list[Paper], so validate and
    # narrow the objects for the evaluation-specific workflow.
    papers = [paper for paper in loaded_papers if isinstance(paper, GoldStandardPaper)]

    if len(papers) != len(loaded_papers):
        unexpected_types = sorted(
            {
                type(paper).__name__
                for paper in loaded_papers
                if not isinstance(paper, GoldStandardPaper)
            }
        )
        msg = (
            "Evaluation records must be GoldStandardPaper objects. "
            f"Unexpected types: {unexpected_types}"
        )
        raise TypeError(msg)

    # Generic blocking returns positions in the supplied Paper sequence.
    all_pairs_df = build_blocked_pairs(
        papers,
        block_rules=BLOCK_RULES,
        include_block_rules=True,
    )

    if all_pairs_df.empty:
        msg = f"Blocking produced no candidate pairs for {dataset_name}"
        raise ValueError(msg)

    required_pair_columns = {"index_a", "index_b"}
    missing_pair_columns = required_pair_columns.difference(all_pairs_df.columns)

    if missing_pair_columns:
        msg = (
            "build_blocked_pairs() must return index-based pairs. "
            f"Missing columns: {sorted(missing_pair_columns)}. "
            f"Available columns: {all_pairs_df.columns.tolist()}"
        )
        raise ValueError(msg)

    index_a_values = all_pairs_df["index_a"].astype(int).tolist()
    index_b_values = all_pairs_df["index_b"].astype(int).tolist()

    # Add dataset identifiers for readable evaluation and review outputs.
    all_pairs_df["id_a"] = [papers[index_a].recordid for index_a in index_a_values]
    all_pairs_df["id_b"] = [papers[index_b].recordid for index_b in index_b_values]

    # Add the ground-truth pair label.
    all_pairs_df["is_dupe"] = [
        int(
            papers[index_a].duplicateid is not None
            and papers[index_b].duplicateid is not None
            and papers[index_a].duplicateid == papers[index_b].duplicateid
        )
        for index_a, index_b in zip(
            index_a_values,
            index_b_values,
            strict=True,
        )
    ]

    if max_pairs is not None and len(all_pairs_df) > max_pairs:
        pairs_df = all_pairs_df.sample(
            n=max_pairs,
            random_state=GLOBAL_SEED,
        ).reset_index(drop=True)
    else:
        pairs_df = all_pairs_df.reset_index(drop=True)

    scored_df = score_candidate_pairs(
        papers,
        pairs_df,
    )

    if scored_df.empty:
        msg = f"No candidate pairs were scored for {dataset_name}"
        raise ValueError(msg)

    records_df = pd.DataFrame(
        paper.model_dump(
            mode="json",
            by_alias=True,
        )
        for paper in papers
    )

    required_record_columns = {"recordid", "duplicateid"}
    missing_record_columns = required_record_columns.difference(records_df.columns)

    if missing_record_columns:
        msg = (
            "Gold-standard fields are missing from the evaluation records. "
            f"Missing columns: {sorted(missing_record_columns)}. "
            f"Available columns: {records_df.columns.tolist()}"
        )
        raise ValueError(msg)

    dataset_dir = RESULTS_ROOT / dataset_name
    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs_df.to_csv(
        dataset_dir / f"pairs_seed_{GLOBAL_SEED}.csv",
        index=False,
    )
    scored_df.to_csv(
        dataset_dir / f"scored_pairs_seed_{GLOBAL_SEED}.csv",
        index=False,
    )

    pair_metrics_df = pd.DataFrame(
        pair_metrics_for_threshold(
            scored_df,
            threshold,
        )
        for threshold in THRESHOLDS
    )

    pair_metrics_df.to_csv(
        dataset_dir / "pair_metrics.csv",
        index=False,
    )

    record_metrics_df = pd.DataFrame(
        record_level_metrics_for_threshold(
            records_df,
            scored_df,
            threshold,
        )
        for threshold in THRESHOLDS
    )

    record_metrics_df.to_csv(
        dataset_dir / "record_metrics.csv",
        index=False,
    )

    _export_error_reviews(
        dataset_dir,
        records_df,
        scored_df,
    )

    best_pair, best_record = _select_and_report_thresholds(
        pair_metrics_df,
        record_metrics_df,
        min_sensitivity=0.99,
    )

    return {
        "dataset": dataset_name,
        "papers": papers,
        "records_df": records_df,
        "all_pairs_df": all_pairs_df,
        "pairs_df": pairs_df,
        "scored_df": scored_df,
        "pair_metrics_df": pair_metrics_df,
        "record_metrics_df": record_metrics_df,
        "best_pair": best_pair,
        "best_record": best_record,
    }
