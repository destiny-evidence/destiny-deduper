import pandas as pd
import re
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import jellyfish
from loguru import logger
from typing import Optional, List

# ------------------------
# Data Models
# ------------------------
class DOIIdentifier:
    def __init__(self, identifier: str):
        self.identifier = identifier

class Authorship:
    def __init__(self, author_name: str):
        self.author_name = author_name

class Paper:
    def __init__(
        self,
        title: Optional[str] = None,
        authors: Optional[List[Authorship]] = None,
        doi: Optional[DOIIdentifier] = None,
        abstract: Optional[str] = None,
        year: Optional[int] = None,
        journal: Optional[str] = None,
        publisher: Optional[str] = None,
        pages: Optional[str] = None,
        isbn: Optional[str] = None,
    ):
        self.title = title
        self.authors = authors
        self.doi = doi
        self.abstract = abstract
        self.year = year
        self.journal = journal
        self.publisher = publisher
        self.pages = pages
        self.isbn = isbn

# ------------------------
# Normalization Functions
# ------------------------
def normalize_doi(doi: str | None) -> str:
    """Normalize DOI format to lowercase, remove prefixes, decode symbols."""
    if not isinstance(doi, str):
        return ""
    doi = doi.replace("%28", "(").replace("%29", ")")
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^DOI[: ]?", "", doi, flags=re.IGNORECASE)
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, flags=re.IGNORECASE)
    if match:
        doi = match.group(0)
    return doi.strip().lower()


def normalize_isbn(isbn: str | None) -> str:
    """Normalize ISBN string by removing common extra patterns."""
    if not isinstance(isbn, str):
        return ""
    isbn = re.sub(r"\s*\(PRINT\).*", "", isbn, flags=re.IGNORECASE)
    isbn = re.sub(r"\s*\(ELECTRONIC\).*", "", isbn, flags=re.IGNORECASE)
    isbn = re.sub(r"\\N.*", "", isbn)
    return isbn.strip().lower()


def normalize_pages(page_range: str | None) -> str:
    """Normalize page ranges like '123–7' -> '123-127'."""
    if not isinstance(page_range, str):
        return ""
    page_range = re.sub(r"[–—−]", "-", page_range).strip()
    parts = page_range.split("-")
    if len(parts) != 2:
        return page_range
    start, end = parts[0].strip(), parts[1].strip()
    if len(end) < len(start):
        prefix_len = len(start) - len(end)
        end = start[:prefix_len] + end
    return f"{start}-{end}"


def normalize_authors(authors_list: list | None) -> list:
    """Normalize author names; input can be list of dicts/objects or strings."""
    if not isinstance(authors_list, list):
        return []
    normalized = []
    for a in authors_list:
        if isinstance(a, str):
            normalized.append(a.strip())
        elif hasattr(a, "author_name") and isinstance(a.author_name, str):
            normalized.append(a.author_name.strip())
    return normalized

# ------------------------
# Deduper Functions
# ------------------------
class Deduper:
    @staticmethod
    def jaro_winkler_distance(a: str, b: str) -> float:
        return jellyfish.jaro_winkler_similarity(a, b)

    @staticmethod
    def levenshtein_distance(a: str, b: str) -> float:
        dist = jellyfish.levenshtein_distance(a, b)
        max_len = max(len(a), len(b), 1)
        return 1 - dist / max_len

    @staticmethod
    def compare_title(a: Paper, b: Paper) -> float:
        if not a.title or not b.title:
            return 0.0
        return Deduper.levenshtein_distance(a.title, b.title)

    @staticmethod
    def compare_abstract(a: Paper, b: Paper) -> float:
        if not a.abstract or not b.abstract:
            return 0.0
        return Deduper.levenshtein_distance(a.abstract, b.abstract)

    @staticmethod
    def compare_authors(a: Paper, b: Paper) -> float:
        if not a.authors or not b.authors:
            return 0.0
        authors_a = ", ".join([auth.author_name for auth in a.authors])
        authors_b = ", ".join([auth.author_name for auth in b.authors])
        return Deduper.jaro_winkler_distance(authors_a, authors_b)

    @staticmethod
    def compare_doi(a: Paper, b: Paper) -> float:
        doi_a = a.doi.identifier if a.doi else ""
        doi_b = b.doi.identifier if b.doi else ""
        return 1.0 if doi_a == doi_b and doi_a != "" else 0.0

    @staticmethod
    def compare_year(a: Paper, b: Paper) -> float:
        if a.year is None or b.year is None:
            return 0.0
        return 1.0 if a.year == b.year else 0.0

# ------------------------
# Step 1: Load CSV and convert to Paper
# ------------------------
SRSR = pd.read_csv("app/SRSR_duplicates_labelled.csv")
papers = []
for _, row in SRSR.iterrows():
    doi_obj = DOIIdentifier(identifier=normalize_doi(row.get("doi"))) if row.get("doi") else None
    papers.append(Paper(
        title=row.get("title"),
        authors=normalize_authors(row.get("authors")),
        doi=doi_obj,
        abstract=row.get("abstract"),
        year=row.get("year"),
        journal=row.get("journal"),
        publisher=row.get("publisher"),
        pages=normalize_pages(row.get("pages")),
        isbn=normalize_isbn(row.get("isbn"))
    ))

# ------------------------
# Step 2: Generate all pairs + gold labels
# ------------------------
pairs = []
for i, j in combinations(range(len(papers)), 2):
    paper_a, paper_b = papers[i], papers[j]
    is_dup = int(SRSR.loc[i, "duplicateid"] == SRSR.loc[j, "duplicateid"])
    pairs.append({"paper_a": paper_a, "paper_b": paper_b, "is_duplicate": is_dup})

pairs_df = pd.DataFrame(pairs)
logger.info(f"Duplicate distribution:\n{pairs_df['is_duplicate'].value_counts()}")

# ------------------------
# Step 3: Generate field-level features
# ------------------------
def generate_features(row):
    a, b = row["paper_a"], row["paper_b"]
    features = {
        "title_sim": Deduper.compare_title(a, b),
        "abstract_sim": Deduper.compare_abstract(a, b),
        "author_sim": Deduper.compare_authors(a, b),
        "doi_sim": Deduper.compare_doi(a, b),
        "year_sim": Deduper.compare_year(a, b)
    }
    return pd.Series(features)

X = pairs_df.apply(generate_features, axis=1)
y = pairs_df["is_duplicate"]

# ------------------------
# Step 4: Train/test split & logistic regression
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ------------------------
# Step 5: Evaluation
# ------------------------
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, color="blue")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.title(f"ROC Curve (AUC={auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Confusion matrix at 0.6 threshold
pred_class = (y_prob >= 0.6).astype(int)
cm = confusion_matrix(y_test, pred_class)
print("Confusion matrix:\n", cm)

# Feature coefficients
coefs = pd.Series(model.coef_[0], index=X_train.columns)
print("Feature coefficients:\n", coefs)

# False positives / negatives for inspection
false_positives = X_test[(y_test == 0) & (pred_class == 1)]
false_negatives = X_test[(y_test == 1) & (pred_class == 0)]
print("False positives:\n", false_positives.head())
print("False negatives:\n", false_negatives.head())