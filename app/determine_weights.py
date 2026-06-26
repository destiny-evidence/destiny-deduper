"""Determine weights for default deduplation."""

import math
import random
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import Field, ValidationError, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from app.data_models_old import Authorship, Paper
from app.dedupe_old import Deduper

SEED = 1234
BLOCK_RULES_OLD = [
    ["title"],
    # ["first_author"],  # blocking on first author
    ["abstract"],
    ["doi"],
    ["year", "journal"],
    ["year", "pages"],
    ["year", "volume"],
    ["pages", "volume"],
    ["pages", "issue"],
    ["year", "issue"],
]

BLOCK_RULES = [
    ["title"],
    # ["first_author"],  # blocking on first author
    ["abstract"],
    ["doi"],
    ["year", "journal"],
    ["year", "pages"],
    ["year", "volume"],
    ["pages", "volume"],
]
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


class ExtendedPaper(Paper):
    """An extension on the Paper class to include duplicate_id."""

    recordid: int | None = Field(default=None)
    duplicate_id: int | None = Field(default=None, alias="duplicateid")

    @field_validator("pages", mode="before")
    @classmethod
    def parse_pages(cls, v: str | float | None) -> str | None:
        """
        Clean up pages field, handling NaN and empty strings.
        """
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return None

    @field_validator("authors", mode="after")
    @classmethod
    def parse_authors(cls, v: str | float | None) -> list[Authorship] | None:
        """
        Convert raw author string to list of Authorship objects.
        Handles NaN, 'anonymous', and multiple authors separated by '.'.
        """
        if isinstance(v, list):
            return v

        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, str):
            if v.strip().lower() in ("anonymous", ""):
                return None
            # Split on "." followed by letter (typical author initials)
            author_names = re.split(r"\.(?=\w)", v)
            author_names = [a.strip() for a in author_names if a.strip()]
            if not author_names:
                return None
            authors_list = []
            for i, a in enumerate(author_names):
                position = (
                    "first"
                    if i == 0
                    else "last"
                    if i == len(author_names) - 1
                    else "middle"
                )
                authors_list.append(
                    Authorship(author_name=a, display_name=a, position=position)
                )
            return authors_list
        return None  # fallback for unexpected type


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
    """Read and process our gold standard data from csv."""
    df = pd.read_csv(filepath, **kwargs)
    df.columns = [col.lower() for col in df.columns]
    final_cols = [col for col in df.columns if col not in cols_to_drop]
    df = df[final_cols]
    return df.rename(columns={"number": "issue", "author": "authors"})


def get_first_author(authors: list[Authorship] | None) -> str | None:
    """Get first author from a parsed authorship list."""
    if authors is None:
        return None
    for author in authors:
        if author.position.value == "first":
            return author.display_name
    return None


def get_gold_standard_dupes(
    df: pd.DataFrame,
    column: str = "duplicateid",
    id_column: str = "recordid",
    sample: int | None = None,
) -> list[tuple[int, int]]:
    """Get rows from gold standard df which are duplicates."""
    positives = []
    grouped = df[df[column].notna()].groupby(column)

    for _dup_id, grp in grouped:
        ids = grp[
            id_column
        ].tolist()  # Use id_column parameter instead of hardcoded "id"
        for a, b in combinations(ids, 2):
            positives.append((a, b))

    if sample and len(positives) > sample:
        positives = random.sample(positives, k=sample)

    return positives


def get_gold_standard_close_non_dupes(
    df: pd.DataFrame,
    column: str = "duplicateid",
    id_column: str = "recordid",
    sample: int | None = None,
    block_rules: list = BLOCK_RULES,
) -> list[tuple[int, int, int]]:
    """Get rows from gold standard df which are not duplicates, but look like duplicates."""

    def _norm(val) -> str | None:
        """Normalise string."""
        if pd.isna(val) or val is None or val == "":
            return None
        return re.sub(r"\s+", " ", str(val).strip().lower())

    negatives: list[tuple[int, int, int]] = []
    seen_pairs = set()

    # Pre-create lookup dict for duplicate IDs (O(n) once instead of O(n²) lookups)
    dup_lookup = df.set_index(id_column)[column].to_dict()

    logger.info(
        f"Generating hard negatives from {len(df)} records using {len(block_rules)} blocking rules"
    )

    for rule_idx, rule in enumerate(block_rules, 1):
        logger.info(f"Processing rule {rule_idx}/{len(block_rules)}: {rule}")

        # Validate fields exist
        missing_fields = [f for f in rule if f not in df.columns]
        if missing_fields:
            logger.warning(f"Fields {missing_fields} not in dataframe, skipping rule")
            continue

        blocks = defaultdict(list)

        # Vectorized normalization - much faster than iterrows()
        # Create a copy with just the fields we need
        df_subset = df[[id_column] + rule].copy()

        # Normalize all fields at once
        for field in rule:
            df_subset[f"{field}_norm"] = df_subset[field].apply(_norm)

        # Drop rows where any normalized field is None
        norm_cols = [f"{field}_norm" for field in rule]
        df_subset = df_subset.dropna(subset=norm_cols)

        # Create blocking keys using vectorized operations
        df_subset["block_key"] = df_subset[norm_cols].apply(tuple, axis=1)

        # Group by block key - this is much faster than manual grouping
        for block_key, group in df_subset.groupby("block_key"):
            ids = group[id_column].tolist()
            if len(ids) >= 2:
                blocks[block_key] = ids

        total_candidate_pairs = 0
        valid_hard_negatives = 0

        # Generate pairs within each block
        for block_ids in blocks.values():
            for a, b in combinations(block_ids, 2):
                total_candidate_pairs += 1
                pair_id = tuple(sorted((a, b)))

                # Skip if already seen
                if pair_id in seen_pairs:
                    continue

                # Fast lookup instead of .loc (O(1) instead of O(n))
                da = dup_lookup.get(a)
                db = dup_lookup.get(b)

                # Only keep if they have different duplicate IDs
                is_valid_negative = False
                if pd.notna(da) and pd.notna(db):
                    if da != db:
                        is_valid_negative = True
                else:
                    # At least one doesn't have a duplicate ID
                    is_valid_negative = True

                if is_valid_negative:
                    valid_hard_negatives += 1
                    negatives.append((a, b, 0))
                    seen_pairs.add(pair_id)

        logger.info(
            f"Rule {tuple(rule)}: {total_candidate_pairs} candidate pairs, "
            f"{valid_hard_negatives} valid hard negatives, "
            f"total negatives so far: {len(negatives)}"
        )

    logger.info(f"Total hard negatives generated: {len(negatives)}")

    # Sample negatives if requested
    if sample and len(negatives) > sample:
        negatives = random.sample(negatives, k=sample)
        logger.info(f"Sampled down to {sample} negatives")

    return negatives


def get_all_pairs(
    df: pd.DataFrame,
    column: str = "duplicateid",
    id_column: str = "recordid",
    sample: int | None = None,
) -> list[tuple[int, int, int]]:
    """
    Return all unordered pairs of record ids from `df` with labels.

    Each returned tuple is (id_a, id_b, label) where label==1 if both records
    have a non-null duplicate id and that id is equal (i.e. they are true
    duplicates in the gold), otherwise label==0.

    If `sample` is provided and the total number of pairs is larger than
    `sample`, the function will return `sample` randomly sampled unique pairs.

    Notes:
    - This enumerates O(n^2) pairs; for large `df` use `sample` to avoid
      excessive memory/time use.
    - Pair ordering is canonical (a < b) so each unordered pair appears once.

    """
    ids = list(df[id_column].tolist())
    n = len(ids)
    total_pairs = n * (n - 1) // 2
    logger.info(
        f"Preparing all pairs from {n} records (total unordered pairs: {total_pairs})"
    )

    dup_lookup = df.set_index(id_column)[column].to_dict()

    # If sampling requested and total is large, sample unique pairs uniformly
    if sample and total_pairs > sample:
        logger.info(f"Sampling {sample} pairs from {total_pairs} total pairs")
        pairs = []
        seen = set()
        while len(pairs) < sample:
            a, b = random.sample(ids, 2)
            if a == b:
                continue
            pa, pb = (a, b) if a < b else (b, a)
            if (pa, pb) in seen:
                continue
            seen.add((pa, pb))
            da = dup_lookup.get(pa)
            db = dup_lookup.get(pb)
            label = 1 if (pd.notna(da) and pd.notna(db) and da == db) else 0
            pairs.append((pa, pb, label))
        return pairs

    # Otherwise enumerate all pairs deterministically
    out = []
    for i in range(n):
        a = ids[i]
        for j in range(i + 1, n):
            b = ids[j]
            da = dup_lookup.get(a)
            db = dup_lookup.get(b)
            label = 1 if (pd.notna(da) and pd.notna(db) and da == db) else 0
            out.append((a, b, label))

    return out


def build_pre_comparison_training_test_set_df(
    dupes: tuple[int, int], non_dupes: tuple[int, int], non_dupe_ratio: int = 2
) -> pd.DataFrame:
    """Create df of dupe and non-dupe pairs."""
    # get all dupes into our df as id_a and id_b and is_dupe=1
    dupes_df = pd.DataFrame(dupes).rename(columns={0: "id_a", 1: "id_b"})
    dupes_df["is_dupe"] = 1
    # produce sampling target (len(df)*2)
    sample_target = len(dupes_df) * non_dupe_ratio
    sampled_non_dupes = random.sample(non_dupes, sample_target)
    # get sampled non_dupes and add to df
    non_dupes_df = pd.DataFrame(sampled_non_dupes).rename(
        columns={0: "id_a", 1: "id_b", 2: "is_dupe"}
    )
    return pd.concat([dupes_df, non_dupes_df], axis=0)


def build_paired_comparison_df(
    dupes: list[tuple[int, int]], non_dupes: list[tuple[int, int]]
) -> pd.DataFrame:
    """
    Create df of all dupe and non-dupe pairs without ratio sampling.

    Takes all provided dupes and non-dupes and combines them into a single DataFrame
    for comparison. No sampling or ratio logic applied.

    Args:
        dupes: List of (id_a, id_b) tuples that are duplicates
        non_dupes: List of (id_a, id_b) tuples that are non-duplicates

    Returns:
        DataFrame with columns [id_a, id_b, is_dupe]

    """
    dupes_df = pd.DataFrame(dupes).rename(columns={0: "id_a", 1: "id_b"})
    dupes_df["is_dupe"] = 1

    non_dupes_df = pd.DataFrame(non_dupes).rename(columns={0: "id_a", 1: "id_b"})
    non_dupes_df["is_dupe"] = 0

    return pd.concat([dupes_df, non_dupes_df], axis=0).reset_index(drop=True)


def compare_target_fields(
    record_a: ExtendedPaper,
    record_b: ExtendedPaper,
    deduper: Deduper,
    fields: list = DEDUPE_FIELDS,
) -> dict[str, float]:
    """Comapre target fields."""
    scores = {}
    for field in fields:
        compare_method = getattr(deduper, f"compare_{field}", None)
        if compare_method:
            try:
                scores[field] = compare_method(record_a, record_b)
            except Exception as e:
                scores[field] = None
                logger.warning(f"compare_{field} failed: {e}")
        else:
            scores[field] = None

    return scores


def perform_deduplication_on_training_test_set_df(
    training_test_df: pd.DataFrame,
    gold_standard_df: pd.DataFrame,
    comparison_targets: list = DEDUPE_FIELDS,
) -> pd.DataFrame:
    """Perform deduplication using Deduper class on all pairs."""
    out = []
    for _, row in tqdm(training_test_df.iterrows()):
        # id_a (gs) to ExtendedPaper
        gs_id: int = row["id_a"]
        gold_standard_record = (
            gold_standard_df[gold_standard_df["recordid"] == gs_id].iloc[0].to_dict()
        )
        test_id: int = row["id_b"]
        test_record = (
            gold_standard_df[gold_standard_df["recordid"] == test_id].iloc[0].to_dict()
        )

        newline = {"id_a": int(gs_id), "id_b": int(test_id), "is_dupe": row["is_dupe"]}

        try:
            gs_paper = ExtendedPaper(**gold_standard_record)
            test_paper = ExtendedPaper(**test_record)
        except ValidationError as e:
            logger.error(f"validation error on gs {gs_id} test {test_id}.")
            logger.error(f"error msg: {e}")
            continue

        deduper = Deduper(reference=gs_paper, candidates=test_paper)
        comparisons = compare_target_fields(
            record_a=gs_paper,
            record_b=test_paper,
            deduper=deduper,
            fields=comparison_targets,
        )
        newline.update(comparisons)
        out.append(newline)

    return pd.DataFrame(out)


def train_dedup_model(df_training: pd.DataFrame):
    feature_cols = [
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
    X = df_training[feature_cols].fillna(0)
    y = df_training["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    print("[info] Validation metrics:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_prob))

    # Feature importance
    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("[info] Feature importance:\n", feat_imp)

    return clf


# # === 3. Predict duplicates for new candidate pairs ===
# def predict_duplicate_probability(
#     paper_a: Paper, paper_b: Paper, model, feature_cols=None
# ):
#     if feature_cols is None:
#         feature_cols = [
#             "doi",
#             "title",
#             "authors",
#             "year",
#             "journal",
#             "pages",
#             "abstract",
#             "volume",
#             "issue",
#         ]
#     deduper = Deduper(reference=paper_a, candidates=[paper_b])
#     features = []
#     for field in feature_cols:
#         try:
#             features.append(getattr(deduper, f"compare_{field}")(paper_a, paper_b))
#         except:
#             features.append(0.0)
#     prob = model.predict_proba([features])[0, 1]
#     return prob


# # === Example usage ===
# df_training = build_training_pairs_with_scores(papers, negative_ratio=2.0)
# dedup_model = train_dedup_model(df_training)

# # Predict on new pair
# # prob = predict_duplicate_probability(paper1, paper2, model=dedup_model)
