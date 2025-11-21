"""Determine weights for default deduplation."""

import math
import random
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import Field, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.data_models import Authorship, DOIIdentifier, Paper
from app.dedupe import Deduper

BLOCK_RULES = [
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


class ExtendedPaper(Paper):
    """An extension on the Paper class to include duplicate_id."""

    id: int | None = Field(default=None)
    duplicate_id: int | None = Field(default=None, alias="duplicateid")

    @field_validator("authors", mode="after")
    @classmethod
    def parse_authors(cls, v: str | float | None) -> list[Authorship] | None:
        """
        Convert raw author string to list of Authorship objects.
        Handles NaN, 'anonymous', and multiple authors separated by '.'.
        """
        logger.debug("hello!")
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
            logger.debug(author_names)
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
) -> pd.DataFrame:
    """Read and process our gold standard data from csv."""
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    final_cols = [col for col in df.columns if col not in cols_to_drop]
    df = df[final_cols]
    return df.rename(columns={"number": "issue", "author": "authors"})


# # load gold standard data
# gold = pd.read_csv("app/SRSR_duplicates_labelled.csv")
# gold["author_name"] = gold["author"]
# gold["issue"] = gold["number"]
# gold["duplicateid"] = gold["duplicateid"].astype(str)
# gold_sample = (
#     gold.sort_values(by="title").head(50).copy()
# )  # use this for quick dev/testing

# # Inspect columns
# print(gold.head())


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
    id_column: str = "record_id",
    sample: int | None = None
) -> list[tuple[int, int]]:
    """Get rows from gold standard df which are duplicates."""
    positives = []
    grouped = df[df[column].notna()].groupby(column)

    for _dup_id, grp in grouped:
        ids = grp[id_column].tolist()  # Use id_column parameter instead of hardcoded "id"
        for a, b in combinations(ids, 2):
            positives.append((a, b))

    if sample and len(positives) > sample:
        positives = random.sample(positives, k=sample)

    return positives


def get_gold_standard_close_non_dupes(
    df: pd.DataFrame,
    column: str = "duplicateid",
    id_column: str = "record_id",
    sample: int | None = None,
    block_rules: list = BLOCK_RULES,
) -> list[tuple[int, int, int]]:
    """Get rows from gold standard df which are not duplicates, but look like duplicates."""

    def _norm(val: str) -> str | None:
        """Normalise string."""
        if not val:
            return None
        return re.sub(r"\s+", " ", str(val).strip().lower())

    negatives: list[tuple[int, int, int]] = []
    seen_pairs = set()
    logger.debug("Hard negative generation per block rule:")

    # For each blocking rule build blocks of ids
    for rule in block_rules:
        blocks = defaultdict(list)

        # For each row, create a blocking key
        for _, row in df.iterrows():
            # Build the blocking key from the rule fields
            key_parts = []
            for field in rule:
                if field in df.columns:
                    val = _norm(row[field])
                    if val is not None:
                        key_parts.append(val)

            # Skip if any field is missing
            if len(key_parts) != len(rule):
                continue

            # Create blocking key
            key = tuple(key_parts)
            # append the id value using the provided id_column
            blocks[key].append(row[id_column])

        total_candidate_pairs = 0
        valid_hard_negatives = 0

        MIN_BLOCK_SIZE = 2
        # Generate pairs within each block
        for _, ids in blocks.items():
            if len(ids) < MIN_BLOCK_SIZE:
                continue

            for a, b in combinations(ids, 2):
                total_candidate_pairs += 1
                pair_id = tuple(sorted((a, b)))

                # Get duplicate IDs for both papers
                da = df.loc[df[id_column] == a, column].iloc[0]
                db = df.loc[df[id_column] == b, column].iloc[0]

                # Only keep if they have different duplicate IDs (i.e., not true duplicates)
                if pd.notna(da) and pd.notna(db):
                    if da != db and pair_id not in seen_pairs:
                        valid_hard_negatives += 1
                        negatives.append((a, b, 0))  # label=0 for non-duplicates
                        seen_pairs.add(pair_id)
                elif pair_id not in seen_pairs:  # At least one doesn't have a duplicate ID
                    negatives.append((a, b, 0))
                    seen_pairs.add(pair_id)
                    valid_hard_negatives += 1

        logger.debug(
        f"Rule {tuple(rule)}: candidate pairs={total_candidate_pairs}, "
        f"valid hard negatives={valid_hard_negatives}"
        )

    if sample and len(negatives) > sample:
        negatives = random.sample(negatives, k=sample)
        logger.debug("Sampled down to %d negatives", len(negatives))

    return negatives


def run_deduper_on_training_set():
    pass


def build_training_pairs_with_scores(
    papers: list[Paper],
    negative_ratio: float = 2.0,
) -> pd.DataFrame:
    """
    Generate 1:1 candidate pairs for deduplication training using dup_id labels.
    Includes hard negatives via blocking (year+journal, etc.).
    Computes per-field similarity scores using Deduper.
    """

    # --- Step 3: Hard negatives using blocking rules ---

    negatives = []
    seen_pairs = set()
    print("[debug] Hard negative generation per block rule:")

    for rule in block_rules:
        blocks = defaultdict(list)
        for pid, meta in enriched.items():
            vals = [_norm(meta.get(f)) for f in rule]
            if None in vals:
                continue
            key = tuple(vals)
            blocks[key].append(pid)

        total_candidate_pairs = 0
        valid_hard_negatives = 0

        for key, ids in blocks.items():
            if len(ids) < 2:
                continue
            for a, b in combinations(ids, 2):
                total_candidate_pairs += 1
                pair_id = tuple(sorted((a, b)))
                da = enriched[a]["dup_id"]
                db = enriched[b]["dup_id"]
                if da != db:
                    valid_hard_negatives += 1
                    if pair_id not in seen_pairs:
                        negatives.append((a, b, 0))
                        seen_pairs.add(pair_id)

        print(
            f"[debug] Rule {tuple(rule)}: "
            f"candidate pairs={total_candidate_pairs}, "
            f"valid hard negatives={valid_hard_negatives}"
        )

    print(f"[debug] Total hard negatives generated: {len(negatives)}")

    # --- Sample negatives to maintain balance ---
    n_pos = len(positives)
    n_neg_desired = int(n_pos * negative_ratio)
    if len(negatives) > n_neg_desired:
        negatives = random.sample(negatives, n_neg_desired)
    print(f"[debug] negatives sampled: {len(negatives)}")

    all_pairs = positives + negatives

    # --- Step 4: Run Deduper comparisons for all pairs ---
    results = []
    print("[debug] computing Deduper similarity scores...")

    for idx, (a, b, label) in enumerate(all_pairs):
        rec_a = papers_by_id[a]
        rec_b = papers_by_id[b]
        deduper = Deduper(reference=rec_a, candidates=[rec_b])

        scores = {}
        for field in [
            "doi",
            "title",
            "authors",
            "year",
            "journal",
            "pages",
            "abstract",
            "volume",
            "issue",
        ]:
            compare_func = getattr(deduper, f"compare_{field}", None)
            if compare_func:
                try:
                    scores[field] = compare_func(rec_a, rec_b)
                except Exception as e:
                    scores[field] = None
                    print(f"[warn] compare_{field} failed: {e}")
            else:
                scores[field] = None

        results.append(
            {
                "id_1": a,
                "id_2": b,
                "label": label,
                **scores,
            }
        )

        if idx % 200 == 0:
            print(f"[debug] processed {idx}/{len(all_pairs)} pairs")

    df = pd.DataFrame(results)
    print(f"[debug] final dataframe shape: {df.shape}")
    print(
        f"[debug] label distribution: {df['label'].value_counts().to_dict() if not df.empty else 'empty'}"
    )
    return df


# def train_dedup_model(df_training: pd.DataFrame):
#     feature_cols = [
#         "doi",
#         "title",
#         "authors",
#         "year",
#         "journal",
#         "pages",
#         "abstract",
#         "volume",
#         "issue",
#     ]
#     X = df_training[feature_cols].fillna(0)
#     y = df_training["label"]

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     clf = RandomForestClassifier(
#         n_estimators=200, max_depth=None, random_state=42, class_weight="balanced"
#     )
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_val)
#     y_prob = clf.predict_proba(X_val)[:, 1]

#     print("[info] Validation metrics:")
#     print(classification_report(y_val, y_pred))
#     print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
#     print("ROC-AUC:", roc_auc_score(y_val, y_prob))

#     # Feature importance
#     feat_imp = pd.DataFrame(
#         {"feature": feature_cols, "importance": clf.feature_importances_}
#     ).sort_values("importance", ascending=False)
#     print("[info] Feature importance:\n", feat_imp)

#     return clf


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
