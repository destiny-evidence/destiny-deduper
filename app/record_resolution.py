"""
Cluster-aware record-level deduplication helpers.

This module operates on pandas DataFrames that are usually produced from
Pydantic-validated records (for example via ``build_record_cache`` and
``GoldStandardPaper``). The functions here do not instantiate Pydantic models;
they assume incoming columns are already normalized and trustworthy.

Column naming note:
- Defaults use legacy gold-standard names: ``recordid`` and ``duplicateid``.
- If your frame uses snake_case names (for example ``record_id``), pass the
    relevant ``*_column`` arguments explicitly.
"""

from collections.abc import Iterable
from typing import Literal, cast

import pandas as pd

RetentionStrategy = Literal["min_recordid", "metadata_richness", "prefer_doi_abstract"]

PRESENCE_FIELDS = [
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


def is_present(value: object) -> bool:
    """Return True when a value is present after null and whitespace checks."""
    return not pd.isna(value) and str(value).strip() != ""


def cluster_components(
    record_ids: Iterable[int], edges: Iterable[tuple[int, int]]
) -> dict[int, int]:
    """
    Build connected components from record IDs and predicted duplicate edges.

    Args:
        record_ids: All record IDs in the current dataset slice.
        edges: Candidate duplicate links expressed as ``(id_a, id_b)``.

    Returns:
        Mapping from each record ID to an integer component identifier.

    """
    parent = {int(record_id): int(record_id) for record_id in record_ids}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        if int(left) in parent and int(right) in parent:
            union(int(left), int(right))

    root_to_group: dict[int, int] = {}
    record_to_group: dict[int, int] = {}
    for record_id in parent:
        root = find(record_id)
        group_id = root_to_group.setdefault(root, len(root_to_group))
        record_to_group[record_id] = group_id

    return record_to_group


def choose_canonical_record(
    cluster_df: pd.DataFrame,
    strategy: RetentionStrategy = "prefer_doi_abstract",
) -> int:
    """
    Choose one representative record from a predicted duplicate cluster.

    Args:
        cluster_df: DataFrame slice containing one predicted cluster.
        strategy: Canonical selection strategy.

    Returns:
        The selected ``recordid`` for the cluster.

    """
    scored = cluster_df.copy()
    scored["has_doi"] = (
        scored["doi"].map(is_present).astype(int) if "doi" in scored.columns else 0
    )
    scored["has_abstract"] = (
        scored["abstract"].map(is_present).astype(int)
        if "abstract" in scored.columns
        else 0
    )

    available_fields = [field for field in PRESENCE_FIELDS if field in scored.columns]
    scored["metadata_count"] = scored[available_fields].apply(
        lambda row: sum(is_present(value) for value in row),
        axis=1,
    )
    scored["abstract_len"] = (
        scored["abstract"].fillna("").astype(str).str.len()
        if "abstract" in scored.columns
        else 0
    )

    if strategy == "min_recordid":
        sort_cols = ["recordid"]
        ascending = [True]
    elif strategy == "metadata_richness":
        sort_cols = ["metadata_count", "has_doi", "has_abstract", "recordid"]
        ascending = [False, False, False, True]
    elif strategy == "prefer_doi_abstract":
        sort_cols = [
            "has_doi",
            "has_abstract",
            "metadata_count",
            "abstract_len",
            "recordid",
        ]
        ascending = [False, False, False, False, True]
    else:
        msg = (
            "Unknown strategy. Use one of: "
            "'min_recordid', 'metadata_richness', 'prefer_doi_abstract'."
        )
        raise ValueError(msg)

    return int(scored.sort_values(sort_cols, ascending=ascending).iloc[0]["recordid"])


def enrich_kept_row(kept_row: pd.Series, cluster_df: pd.DataFrame) -> pd.Series:
    """
    Backfill missing fields on the kept row from other rows in the cluster.

    For ``abstract``, the longest non-empty value is preferred (with
    ``recordid`` as tie-break). For other fields, the non-empty value from the
    smallest ``recordid`` is selected.
    """
    enriched = kept_row.copy()
    for field in [field for field in PRESENCE_FIELDS if field in cluster_df.columns]:
        if is_present(enriched.get(field)):
            continue

        candidates = cluster_df[cluster_df[field].map(is_present)]
        if candidates.empty:
            continue

        if field == "abstract":
            best = (
                candidates.assign(_len=candidates[field].astype(str).str.len())
                .sort_values(["_len", "recordid"], ascending=[False, True])
                .iloc[0][field]
            )
        else:
            best = candidates.sort_values("recordid").iloc[0][field]

        enriched[field] = best

    return enriched


def remove_duplicates(  # noqa: PLR0913
    df_records: pd.DataFrame,
    scored_pairs: pd.DataFrame,
    threshold: float = 0.85,
    strategy: RetentionStrategy = "prefer_doi_abstract",
    *,
    enrich_kept_records: bool = True,
    probability_column: str = "probability",
    id_a_column: str = "id_a",
    id_b_column: str = "id_b",
    id_column: str = "recordid",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Resolve pair-level predictions into record-level keep/remove decisions.

    Args:
        df_records: Full record table.
        scored_pairs: Pair scores including record IDs and probability values.
        threshold: Pair score threshold used to create cluster edges.
        strategy: Canonical retention strategy for each cluster.
        enrich_kept_records: Whether to backfill kept records from cluster peers.
        probability_column: Probability column name in ``scored_pairs``.
        id_a_column: Left pair ID column name in ``scored_pairs``.
        id_b_column: Right pair ID column name in ``scored_pairs``.
        id_column: Record ID column name in ``df_records``.

    Returns:
        Tuple of ``(deduplicated_df, removed_df, decisions_df)``.

    """
    dedupe_edges = scored_pairs.loc[
        scored_pairs[probability_column] >= threshold,
        [id_a_column, id_b_column],
    ].to_numpy()

    record_to_cluster = cluster_components(
        df_records[id_column].tolist(),
        dedupe_edges,
    )

    annotated = df_records.copy()
    annotated["predicted_cluster"] = annotated[id_column].map(record_to_cluster)

    decisions = []
    kept_rows = []

    for cluster_id, cluster_df in annotated.groupby("predicted_cluster", sort=True):
        keep_id = choose_canonical_record(cluster_df, strategy=strategy)

        for row in cluster_df.itertuples(index=False):
            row_id = int(getattr(row, id_column))
            decisions.append(
                {
                    id_column: row_id,
                    "predicted_cluster": cast("int", cluster_id),
                    "cluster_size": len(cluster_df),
                    "keep": row_id == keep_id,
                    "removed_as_duplicate": row_id != keep_id,
                    "kept_recordid": keep_id,
                }
            )

        kept_row = cluster_df.loc[cluster_df[id_column] == keep_id].iloc[0]
        if enrich_kept_records and len(cluster_df) > 1:
            kept_row = enrich_kept_row(kept_row, cluster_df)
        kept_rows.append(kept_row)

    decisions_df = pd.DataFrame(decisions).sort_values(
        ["removed_as_duplicate", "cluster_size", id_column],
        ascending=[False, False, True],
    )
    deduplicated_df = (
        pd.DataFrame(kept_rows).sort_values(id_column).reset_index(drop=True)
    )
    removed_df = annotated.loc[
        ~annotated[id_column].isin(deduplicated_df[id_column])
    ].copy()

    return deduplicated_df, removed_df, decisions_df


def record_level_metrics_for_threshold(  # noqa: PLR0913
    df_orig: pd.DataFrame,
    scored_pairs_df: pd.DataFrame,
    threshold: float,
    id_column: str = "recordid",
    true_cluster_column: str = "duplicateid",
    probability_column: str = "prob",
    id_a_column: str = "id_a",
    id_b_column: str = "id_b",
) -> dict:
    """
    Compute ASySD-style record-level metrics at a fixed threshold.

    Predicted duplicate pairs above threshold are clustered with connected
    components, then one record per predicted group is marked as "keep"
    (minimum ID). Those keep/remove decisions are compared against gold-standard
    duplicate groups to produce a record-level confusion matrix.

    Args:
        df_orig: Original records DataFrame.
        scored_pairs_df: Pair scores with IDs and probabilities.
        threshold: Decision threshold applied to ``probability_column``.
        id_column: Record ID column in ``df_orig``.
        true_cluster_column: Gold duplicate-group column in ``df_orig``.
        probability_column: Probability column in ``scored_pairs_df``.
        id_a_column: Left pair ID column in ``scored_pairs_df``.
        id_b_column: Right pair ID column in ``scored_pairs_df``.

    Returns:
        Dictionary containing ``TP``, ``FP``, ``TN``, ``FN`` and derived metrics.

    """
    df_t = df_orig.copy()
    edges = scored_pairs_df.loc[
        scored_pairs_df[probability_column] >= threshold,
        [id_a_column, id_b_column],
    ].to_numpy()

    record_to_group = cluster_components(df_t[id_column].tolist(), edges)

    df_t["predicted_group"] = df_t[id_column].map(record_to_group)
    df_t["pred_keep"] = (
        df_t.groupby("predicted_group")[id_column].transform("min") == df_t[id_column]
    )

    df_t["true_cluster"] = (
        df_t[true_cluster_column].fillna(df_t[id_column].astype(str)).astype(str)
    )
    df_t["true_keep"] = (
        df_t.groupby("true_cluster")[id_column].transform("min") == df_t[id_column]
    )

    true_removed = ~df_t["true_keep"]
    pred_removed = ~df_t["pred_keep"]

    tp = int((true_removed & pred_removed).sum())
    fp = int((~true_removed & pred_removed).sum())
    tn = int((~true_removed & ~pred_removed).sum())
    fn = int((true_removed & ~pred_removed).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }
