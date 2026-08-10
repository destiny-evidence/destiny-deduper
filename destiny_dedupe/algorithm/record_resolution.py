# NOTE: This module is overly complex and will require a refactor
# PURPOSE: Once duplicates are identified, this module is intended to support their removal using a number of different strategies. It is used throughout the notebooks for evaluation purposes but requires more thinking/work before implementation in the main app.

"""Cluster-aware record resolution for validated paper objects."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, cast

import pandas as pd

from destiny_dedupe.data_models import PaperWithId

RetentionStrategy = Literal[
    "min_recordid",
    "metadata_richness",
    "prefer_doi_abstract",
]

PRESENCE_FIELDS = (
    "title",
    "authors",
    "year",
    "journal",
    "volume",
    "issue",
    "pages",
    "doi",
    "abstract",
)

PaperWithIdT = TypeVar("PaperWithIdT", bound=PaperWithId)


class _EmptyClusterError(ValueError):
    def __init__(self) -> None:
        super().__init__("Cannot choose a canonical record from an empty cluster.")


class _UnknownRetentionStrategyError(ValueError):
    def __init__(self, strategy: str) -> None:
        super().__init__(
            "Unknown retention strategy "
            f"{strategy!r}. Use 'min_recordid', 'metadata_richness', "
            "or 'prefer_doi_abstract'."
        )


class _MissingPairColumnsError(ValueError):
    def __init__(self, missing_columns: set[str]) -> None:
        super().__init__(
            "The scored-pair table is missing required columns: "
            f"{sorted(missing_columns)}"
        )


class _InvalidPairPositionsError(IndexError):
    def __init__(
        self,
        invalid_edges: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__(
            "The scored-pair table contains out-of-range record "
            f"positions. Examples: {list(invalid_edges[:5])}"
        )


class _DuplicateRecordIdError(ValueError):
    def __init__(self) -> None:
        super().__init__("PaperWithId.recordid values must be unique.")


@dataclass(frozen=True)
class PairColumnConfig:
    """Column names used to interpret a scored-pair table."""

    probability: str = "probability"
    index_a: str = "index_a"
    index_b: str = "index_b"


@dataclass(frozen=True)
class RecordResolutionConfig:
    """Configuration for resolving pair predictions into record decisions."""

    threshold: float = 0.85
    strategy: RetentionStrategy = "prefer_doi_abstract"
    enrich_kept_records: bool = True
    columns: PairColumnConfig = field(default_factory=PairColumnConfig)


@dataclass(frozen=True)
class RecordResolutionResult(Generic[PaperWithIdT]):
    """Resolved records and the decisions used to produce them."""

    kept_records: list[PaperWithIdT]
    removed_records: list[PaperWithIdT]
    decisions_df: pd.DataFrame


def is_present(value: object) -> bool:
    """Return whether a scalar or collection contains a usable value."""
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list | tuple | set | dict):
        return bool(value)

    try:
        return not bool(pd.isna(value))  # ty:ignore
    except (TypeError, ValueError):
        return True


def cluster_components(
    node_ids: Iterable[int],
    edges: Iterable[tuple[int, int]],
) -> dict[int, int]:
    """Build connected components from integer nodes and duplicate edges."""
    parent = {int(node_id): int(node_id) for node_id in node_ids}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left_id: int, right_id: int) -> None:
        root_left = find(left_id)
        root_right = find(right_id)

        if root_left != root_right:
            parent[root_right] = root_left

    for raw_left, raw_right in edges:
        left_id = int(raw_left)
        right_id = int(raw_right)

        if left_id in parent and right_id in parent:
            union(left_id, right_id)

    root_to_cluster: dict[int, int] = {}
    node_to_cluster: dict[int, int] = {}

    for node_id in parent:
        root = find(node_id)
        cluster_id = root_to_cluster.setdefault(
            root,
            len(root_to_cluster),
        )
        node_to_cluster[node_id] = cluster_id

    return node_to_cluster


def _metadata_count(record: PaperWithId) -> int:
    """Count populated bibliographic fields on one record."""
    return sum(
        is_present(getattr(record, field_name, None)) for field_name in PRESENCE_FIELDS
    )


def choose_canonical_record(
    records: Sequence[PaperWithIdT],
    strategy: RetentionStrategy = "prefer_doi_abstract",
) -> PaperWithIdT:
    """Choose the record retained from one predicted duplicate cluster."""
    if not records:
        raise _EmptyClusterError

    if strategy == "min_recordid":
        return min(
            records,
            key=lambda record: record.recordid,
        )

    if strategy == "metadata_richness":
        return max(
            records,
            key=lambda record: (
                _metadata_count(record),
                int(is_present(record.doi)),
                int(is_present(record.abstract)),
                -record.recordid,
            ),
        )

    if strategy == "prefer_doi_abstract":
        return max(
            records,
            key=lambda record: (
                int(is_present(record.doi)),
                int(is_present(record.abstract)),
                _metadata_count(record),
                (len(record.abstract.strip()) if record.abstract else 0),
                -record.recordid,
            ),
        )

    raise _UnknownRetentionStrategyError(strategy)


def enrich_kept_record(
    kept_record: PaperWithIdT,
    cluster_records: Sequence[PaperWithIdT],
) -> PaperWithIdT:
    """Backfill missing fields from validated records in the same cluster."""
    updates: dict[str, object] = {}

    for field_name in PRESENCE_FIELDS:
        if is_present(getattr(kept_record, field_name, None)):
            continue

        candidates = [
            record
            for record in cluster_records
            if is_present(getattr(record, field_name, None))
        ]
        if not candidates:
            continue

        if field_name == "abstract":
            source_record = max(
                candidates,
                key=lambda record: (
                    len(
                        str(
                            getattr(
                                record,
                                field_name,
                            )
                        )
                    ),
                    -record.recordid,
                ),
            )
        else:
            source_record = min(
                candidates,
                key=lambda record: record.recordid,
            )

        updates[field_name] = getattr(
            source_record,
            field_name,
        )

    if not updates:
        return kept_record

    enriched_data = kept_record.model_dump(mode="python")
    enriched_data.update(updates)

    return cast(  # ty:ignore
        "PaperWithIdT",
        type(kept_record).model_validate(enriched_data),
    )


def _validated_edges(
    scored_pairs: pd.DataFrame,
    record_count: int,
    config: RecordResolutionConfig,
) -> list[tuple[int, int]]:
    """Extract and validate predicted duplicate edges."""
    columns = config.columns
    required_columns = {
        columns.probability,
        columns.index_a,
        columns.index_b,
    }
    missing_columns = required_columns.difference(scored_pairs.columns)

    if missing_columns:
        raise _MissingPairColumnsError(missing_columns)

    predicted_pairs = scored_pairs.loc[
        scored_pairs[columns.probability].astype(float) >= config.threshold,
        [columns.index_a, columns.index_b],
    ]

    edges = [
        (int(left_id), int(right_id))
        for left_id, right_id in (
            predicted_pairs.itertuples(
                index=False,
                name=None,
            )
        )
    ]

    invalid_edges = [
        edge
        for edge in edges
        if (
            edge[0] < 0
            or edge[1] < 0
            or edge[0] >= record_count
            or edge[1] >= record_count
        )
    ]

    if invalid_edges:
        raise _InvalidPairPositionsError(invalid_edges)

    return edges


def _empty_resolution_result() -> RecordResolutionResult[PaperWithIdT]:
    """Return an empty result with stable decision-table columns."""
    decisions_df = pd.DataFrame(
        columns=[
            "index",
            "recordid",
            "predicted_cluster",
            "cluster_size",
            "keep",
            "removed_as_duplicate",
            "kept_recordid",
        ]
    )
    return RecordResolutionResult(
        kept_records=[],
        removed_records=[],
        decisions_df=decisions_df,
    )


def resolve_records(
    records: Sequence[PaperWithIdT],
    scored_pairs: pd.DataFrame,
    config: RecordResolutionConfig | None = None,
) -> RecordResolutionResult[PaperWithIdT]:
    """
    Resolve pair predictions into kept and removed paper objects.

    Candidate-pair edges are expressed as positions in ``records``. This
    keeps clustering independent of external ``recordid`` values while
    retaining those identifiers on every returned object.

    Args:
        records: Validated records with unique ``recordid`` values.
        scored_pairs: Pair table containing positions and probabilities.
        config: Threshold, retention strategy, enrichment, and column names.

    Returns:
        Resolved kept records, removed records, and a decision DataFrame.

    Raises:
        ValueError: If record IDs are duplicated or pair columns are absent.
        IndexError: If a pair references a position outside ``records``.

    """
    resolution_config = config if config is not None else RecordResolutionConfig()
    record_list = list(records)

    if not record_list:
        return _empty_resolution_result()

    record_ids = [record.recordid for record in record_list]
    if len(record_ids) != len(set(record_ids)):
        raise _DuplicateRecordIdError

    edges = _validated_edges(
        scored_pairs,
        len(record_list),
        resolution_config,
    )
    index_to_cluster = cluster_components(
        range(len(record_list)),
        edges,
    )

    cluster_to_indices: dict[int, list[int]] = {}
    for index, cluster_id in index_to_cluster.items():
        cluster_to_indices.setdefault(
            cluster_id,
            [],
        ).append(index)

    kept_records: list[PaperWithIdT] = []
    removed_records: list[PaperWithIdT] = []
    decisions: list[dict[str, int | bool]] = []

    for cluster_id, indices in sorted(cluster_to_indices.items()):
        cluster_records = [record_list[index] for index in indices]
        canonical_record = choose_canonical_record(
            cluster_records,
            strategy=resolution_config.strategy,
        )
        kept_record_id = canonical_record.recordid

        if resolution_config.enrich_kept_records and len(cluster_records) > 1:
            canonical_record = enrich_kept_record(
                canonical_record,
                cluster_records,
            )

        kept_records.append(canonical_record)

        for index in indices:
            record = record_list[index]
            keep = record.recordid == kept_record_id

            decisions.append(
                {
                    "index": index,
                    "recordid": record.recordid,
                    "predicted_cluster": cluster_id,
                    "cluster_size": len(indices),
                    "keep": keep,
                    "removed_as_duplicate": not keep,
                    "kept_recordid": kept_record_id,
                }
            )

            if not keep:
                removed_records.append(record)

    decisions_df = (
        pd.DataFrame(decisions)
        .sort_values(
            [
                "removed_as_duplicate",
                "cluster_size",
                "recordid",
            ],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    kept_records.sort(key=lambda record: record.recordid)
    removed_records.sort(key=lambda record: record.recordid)

    return RecordResolutionResult(
        kept_records=kept_records,
        removed_records=removed_records,
        decisions_df=decisions_df,
    )


def records_to_dataframe(
    records: Sequence[PaperWithId],
) -> pd.DataFrame:
    """Serialize validated paper records into an exportable DataFrame."""
    return pd.DataFrame(
        record.model_dump(
            mode="json",
            by_alias=True,
        )
        for record in records
    )


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
