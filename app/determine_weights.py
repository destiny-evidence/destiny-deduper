import math
import random
import re
from collections import defaultdict
from itertools import combinations

from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.data_models import Authorship, DOIIdentifier, Paper
from app.dedupe import Deduper


def parse_authors(raw_author: str | float | None) -> list[Authorship] | None:
    """
    Convert raw author string to list of Authorship objects.
    Handles NaN, 'anonymous', and multiple authors separated by '.'.
    """
    if raw_author is None or (isinstance(raw_author, float) and math.isnan(raw_author)):
        return None
    if isinstance(raw_author, str):
        if raw_author.strip().lower() in ("anonymous", ""):
            return None
        # Split on "." followed by letter (typical author initials)
        author_names = re.split(r"\.(?=\w)", raw_author)
        author_names = [a.strip() for a in author_names if a.strip()]
        if not author_names:
            return None
        authors_list = []
        for i, a in enumerate(author_names):
            position = "first" if i == 0 else "last" if i == len(author_names)-1 else "middle"
            authors_list.append(Authorship(author_name=a, display_name=a, position=position))
        return authors_list
    return None  # fallback for unexpected type

# load gold standard data
gold = pd.read_csv("app/SRSR_duplicates_labelled.csv")
gold["author_name"] = gold["author"]
gold["issue"] = gold["number"]
gold["duplicateid"] = gold["duplicateid"].astype(str)
gold_sample = gold.sort_values(by="title").head(50).copy() # use this for quick dev/testing

# Inspect columns
print(gold.head())

# Extend the Paper model to include dup_id
class ExtendedPaper(Paper):
    dup_id: str | None = None  # Add the dup_id field

# Convert to ExtendedPaper format
papers = []
for _, row in gold_sample.iterrows(): #change to gold_sample for quick dev/testing
    try:
        authors_list = parse_authors(row.get("author"))
        pages_tuple = ExtendedPaper.parse_pages(row.get("pages"))

        doi_value = None
        if "doi" in row and pd.notna(row["doi"]) and str(row["doi"]).strip():
            doi_value = DOIIdentifier(identifier=str(row["doi"]).strip())

        paper = ExtendedPaper(
            id=str(row["record_id"]),
            title=row.get("title") if pd.notna(row.get("title")) else None,
            authors=authors_list,
            year=int(row["year"]) if pd.notna(row.get("year")) else None,
            journal=row.get("journal") if pd.notna(row.get("journal")) else None,
            volume=row.get("volume") if pd.notna(row.get("volume")) else None,
            issue=row.get("issue") if pd.notna(row.get("issue")) else None,
            pages=pages_tuple,
            abstract=row.get("abstract") if pd.notna(row.get("abstract")) else None,
            doi=doi_value,
            dup_id=row.get("duplicateid")
        )
        papers.append(paper)
    except Exception as e:
        print(f"Error processing row {row['record_id']}: {e}")

print(f"✅ Successfully created {len(papers)} Paper objects")


def build_training_pairs_with_scores(
    papers: list[Paper],
    negative_ratio: float = 2.0,
) -> pd.DataFrame:
    """
    Generate 1:1 candidate pairs for deduplication training using dup_id labels.
    Includes hard negatives via blocking (year+journal, etc.).
    Computes per-field similarity scores using Deduper.
    """
    logger.debug(f"Received {len(papers)} papers")

    # --- Utility: flatten authors and get first author ---
    def get_authors_info(paper):
        authors_list = getattr(paper, "authors", None)
        if not isinstance(authors_list, list) or len(authors_list) == 0:
            return "", None

        authors_flat = ", ".join(
            getattr(a, "display_name", getattr(a, "author_name", str(a)))
            for a in authors_list if a is not None
        )

        # Get first author by position, fallback to first in list
        first_author_obj = next(
            (a for a in authors_list if getattr(a, "position", "").lower() == "first"),
            authors_list[0]
        )
        first_author = getattr(first_author_obj, "display_name",
                               getattr(first_author_obj, "author_name",
                                       str(first_author_obj)))
        return authors_flat, first_author

    # --- Step 1: convert to simple metadata table ---
    df_meta = pd.DataFrame([
        {
            "id": str(getattr(p, "id", getattr(p, "record_id", id(p)))),
            "dup_id": getattr(p, "dup_id", None),
            "title": getattr(p, "title", None),
            "authors": get_authors_info(p)[0],
            "first_author": get_authors_info(p)[1],
            "year": getattr(p, "year", None),
            "journal": getattr(p, "journal", None),
        }
        for p in papers
    ])

    papers_by_id = {id_: p for id_, p in zip(df_meta["id"], papers)}

    # --- Step 2: Positive pairs (same dup_id) ---
    positives = []
    grouped = df_meta[df_meta["dup_id"].notnull()].groupby("dup_id")
    for dup_id, grp in grouped:
        ids = grp["id"].tolist()
        for a, b in combinations(ids, 2):
            positives.append((a, b, 1))
    print(f"[debug] positives generated: {len(positives)}")

    # --- Step 3: Hard negatives using blocking rules ---
    enriched = {}
    for p in papers:
        pid = str(getattr(p, "id", getattr(p, "record_id", id(p))))
        authors_flat, first_author = get_authors_info(p)

        doi_obj = getattr(p, "doi", None)
        doi_str = getattr(doi_obj, "identifier", None) if doi_obj and not isinstance(doi_obj, str) else doi_obj

        enriched[pid] = {
            "id": pid,
            "dup_id": getattr(p, "dup_id", None),
            "title": getattr(p, "title", None),
            "first_author": first_author,   # for blocking
            "authors": authors_flat,        # full string for Deduper
            "year": getattr(p, "year", None),
            "journal": getattr(p, "journal", None),
            "pages": getattr(p, "pages", None),
            "volume": getattr(p, "volume", None),
            "issue": getattr(p, "issue", None),
            "isbn": getattr(p, "isbn", None),
            "doi": doi_str.lower().strip() if doi_str else None,
            "abstract": getattr(p, "abstract", None),
        }

    def _norm(val):
        if not val:
            return None
        s = str(val).strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    block_rules = [
        ["title"],
        ["first_author"],  # blocking on first author
        ["abstract"],
        ["doi"],
        ["year", "journal"],
        ["year", "pages"],
        ["year", "volume"],
        ["pages", "volume"],
        ["pages", "issue"],
        ["year", "issue"],
    ]

    print(f"[debug] prepping to generate hard negatives using {len(block_rules)} blocking rules...")

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

        print(f"[debug] Rule {tuple(rule)}: "
              f"candidate pairs={total_candidate_pairs}, "
              f"valid hard negatives={valid_hard_negatives}")

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
        for field in ["doi", "title", "authors", "year", "journal", "pages", "abstract", "volume", "issue"]:
            compare_func = getattr(deduper, f"compare_{field}", None)
            if compare_func:
                try:
                    scores[field] = compare_func(rec_a, rec_b)
                except Exception as e:
                    scores[field] = None
                    print(f"[warn] compare_{field} failed: {e}")
            else:
                scores[field] = None

        results.append({
            "id_1": a,
            "id_2": b,
            "label": label,
            **scores,
        })

        if idx % 200 == 0:
            print(f"[debug] processed {idx}/{len(all_pairs)} pairs")

    df = pd.DataFrame(results)
    print(f"[debug] final dataframe shape: {df.shape}")
    print(f"[debug] label distribution: {df['label'].value_counts().to_dict() if not df.empty else 'empty'}")
    return df

def train_dedup_model(df_training: pd.DataFrame):
    feature_cols = ["doi", "title", "authors", "year", "journal", "pages", "abstract", "volume", "issue"]
    X = df_training[feature_cols].fillna(0)
    y = df_training["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    print("[info] Validation metrics:")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_prob))

    # Feature importance
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    print("[info] Feature importance:\n", feat_imp)

    return clf

# === 3. Predict duplicates for new candidate pairs ===
def predict_duplicate_probability(paper_a: Paper, paper_b: Paper, model, feature_cols=None):
    if feature_cols is None:
        feature_cols = ["doi", "title", "authors", "year", "journal", "pages", "abstract", "volume", "issue"]
    deduper = Deduper(reference=paper_a, candidates=[paper_b])
    features = []
    for field in feature_cols:
        try:
            features.append(getattr(deduper, f"compare_{field}")(paper_a, paper_b))
        except:
            features.append(0.0)
    prob = model.predict_proba([features])[0, 1]
    return prob

# === Example usage ===
df_training = build_training_pairs_with_scores(papers, negative_ratio=2.0)
dedup_model = train_dedup_model(df_training)

# Predict on new pair
# prob = predict_duplicate_probability(paper1, paper2, model=dedup_model)  
