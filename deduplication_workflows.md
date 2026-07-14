# Deduplication Toolkit Workflow

## Purpose

This guide provides two distinct workflows:

1. New deduplication workflow: use when your dataset is unlabeled, and you want duplicate decisions and deduplicated output.
2. Gold-standard evaluation workflow: use when your dataset has duplicate labels (`duplicateid`), and you want pair-level and record-level metrics to evaluate how this toolkit performs.

## Recommended Default Policy

Use the weighted method in `Deduper`:

- `Deduper.dedupe_weighted(..., weights=WEIGHTS, intercept=INTERCEPT)`

We recommend using this default decision rule unless you are recalibrating on fresh labeled data:

- `probability >= 0.85` -> duplicate
- `probability < 0.85` -> non-duplicate


## Deduplicating unlabelled data

Use this when there is no `duplicateid` and your goal is a deduplicated output file.

Steps:

1. Load unlabeled data 
2. Build a dataset of candidate pairs using inbuilt blocking criteria
3. Build validated record cache 
4. Score with `dedupe_weighted`.
5. Apply duplicate decision rule (`prob > 0.85`).
6. Resolve duplicate clusters and export unique records.

Why build a validated record cache before scoring?

- In this unlabeled workflow, rows are converted into validated `Paper` objects before scoring.
- `dedupe_weighted` compares record objects, so conversion from raw rows is required.
- This class-based object layer also helps align the toolkit interface with DESTINY SDK usage patterns.
- Validation happens once up front, so malformed rows are caught early rather than inside the scoring loop.
- Pair scoring repeatedly looks up `id_a` and `id_b`; cache lookups are fast and avoid repeated dataframe filtering.
- `include_ids` keeps cache scope aligned to blocked pairs, which avoids unnecessary processing.

Minimal example:

```python
import pandas as pd
from app.candidate_selection import BLOCK_RULES, build_blocked_pairs
from app.dedupe import Deduper, INTERCEPT, WEIGHTS
from app.import_references import CsvLoadConfig, DEFAULT_COLUMNS, load_reference_csv
from app.record_cache import build_record_cache
from app.record_resolution import remove_duplicates

THRESHOLD = 0.85

# load references from csv
df = load_reference_csv(
    "notebooks/data/diabetes_data.csv",
    CsvLoadConfig(columns=DEFAULT_COLUMNS, include_gold_standard=False), #no duplicateid column
)

# build a dataset of candidate pairs using inbuilt block rules
pairs_df = build_blocked_pairs(
    df,
    block_rules=BLOCK_RULES,
    id_column="recordid",
    dup_column=None, #no column to indicate whether pairs are duplicates or not 
)

# build a record cache, checking that references pass validation checks and are ready for scoring
sample_ids = set(pairs_df["id_a"]).union(pairs_df["id_b"])
record_cache, validation_errors = build_record_cache(
    df,
    id_column="recordid",
    include_ids=sample_ids,
)
print("Validation errors:", validation_errors)

# score pairs using inbuilt algorithm (using learned weights and intercept)
any_paper = next(iter(record_cache.values()))
deduper = Deduper(reference=any_paper, candidates=[any_paper])

rows = []
for row in pairs_df.itertuples(index=False):
    rec_a = record_cache.get(int(row.id_a))
    rec_b = record_cache.get(int(row.id_b))
    if rec_a is None or rec_b is None:
        continue

    prob = deduper.dedupe_weighted(
        rec_a,
        rec_b,
        weights=WEIGHTS,
        intercept=INTERCEPT,
    )
    rows.append({"id_a": row.id_a, "id_b": row.id_b, "prob": prob})

# apply decision rule to determine dupes from non-dupes
scored_df = pd.DataFrame(rows)
scored_df["is_predicted_duplicate"] = scored_df["prob"] > THRESHOLD

# remove records identified as duplicates using preferred strategy
deduplicated_df, removed_df, decisions_df = remove_duplicates(
    df_records=df,
    scored_pairs=scored_df,
    threshold=THRESHOLD,
    strategy="prefer_doi_abstract",  # recommended: prioritize DOI/abstract, then metadata richness
    probability_column="prob",
    id_a_column="id_a",
    id_b_column="id_b",
    id_column="recordid",
)

print("Input records:", len(df))
print("Kept records:", len(deduplicated_df))
print("Removed records:", len(removed_df))
```

Retention strategy options for `remove_duplicates`:

- `prefer_doi_abstract` (recommended)
- `metadata_richness`
- `min_recordid`

### `dedupe_weighted` Output Modes

`dedupe_weighted` supports two return modes:

1. Probability-only mode (default):

```python
probability = deduper.dedupe_weighted(rec_a, rec_b, weights=WEIGHTS, intercept=INTERCEPT)
```

2. Detailed mode:

```python
probability, field_scores, early_stop_reason = deduper.dedupe_weighted(
    rec_a,
    rec_b,
    weights=WEIGHTS,
    intercept=INTERCEPT,
    return_details=True,
)
```

Use detailed mode for diagnostics and error analysis. Use probability-only mode for lightweight production scoring.

## Gold-Standard Evaluation (Labeled Data)

Use when a `duplicateid` column is available. 

The `duplicateid` column indicates a flag for a group of duplicate records. It provides the ground-truth label for duplicate citations. Records that share the same `duplicateid` refer to the same publication and are considered true duplicates. Records with different `duplicateid` values represent different publications.

| record_id | title | authors | year | journal | duplicateid |
|----------:|-------|---------|-----:|----------|------------:|
| 1001 | Machine Learning for Healthcare | Smith J.; Patel A. | 2021 | Journal of Medical AI | 101 |
| 1002 | Machine Learning for Health Care | Smith, John; Patel, A. | 2021 | J. Medical AI | 101 |
| 1003 | Deep Learning in Radiology | Chen L.; Brown M. | 2020 | Radiology Today | 202 |
| 1004 | Deep Learning in Radiology | L. Chen; M. Brown | 2020 | Radiology Today | 202 |
| 1005 | Natural Language Processing for Clinical Notes | Garcia P.; Lee S. | 2022 | Health Informatics | 333 |
| 1006 | Explainable AI for Medical Diagnosis | Wilson T.; Ahmed N. | 2023 | AI in Medicine | 404 |

In this example:

- Records **1001** and **1002** share `duplicateid = 101`, indicating they are labelled duplicates of the same publication.
- Records **1003** and **1004** share `duplicateid = 202`, representing another labelled duplicate pair.
- Records **1005** and **1006** have unique `duplicateid` values, indicating they have no labelled duplicates in this dataset.

### Steps for evaluation:

1. Load labeled data (`include_gold_standard=True`).
2. Build a dataset of candidate pairs using inbuilt blocking criteria 
3. Build validated cache.
4. Score with `dedupe_weighted`.
5. Compute pair-level metrics and record-level metrics.
6. Pick threshold (default boundary starts at `> 0.85`).

Why build a validated record cache before scoring?

- In this labeled workflow, rows are converted into validated `GoldStandardPaper` objects before scoring.
- `GoldStandardPaper` includes the gold-standard duplicate label context used for evaluation.
- Using explicit record classes here also aligns evaluation flow with DESTINY SDK-style interfaces.
- Conversion and validation happen once up front, then pair scoring uses fast id-based lookups from cache.

Minimal example:

```python
import pandas as pd
from app.algorithm_development import record_level_metrics_for_threshold
from app.candidate_selection import BLOCK_RULES, build_blocked_pairs
from app.dedupe import Deduper, INTERCEPT, WEIGHTS
from app.import_references import CsvLoadConfig, DEFAULT_COLUMNS, load_reference_csv
from app.record_cache import build_record_cache

THRESHOLD = 0.85

# load references
df = load_reference_csv(
    "notebooks/data/srsr_data.csv",
    CsvLoadConfig(columns=("recordid", *DEFAULT_COLUMNS), include_gold_standard=True), #include gold standard columns (e.g. duplicateid)
)

# build dataframe of candidate pairs 
pairs_df = build_blocked_pairs(
    df,
    block_rules=BLOCK_RULES,
    id_column="recordid",
    dup_column="duplicateid", #pass duplicateid column
)

# build a record cache 
sample_ids = set(pairs_df["id_a"]).union(pairs_df["id_b"])
record_cache, validation_errors = build_record_cache(
    df,
    id_column="recordid",
    include_ids=sample_ids,
)
print("Validation errors:", validation_errors)

# score pairs using internal algorithm with learned weights and intercept
any_paper = next(iter(record_cache.values()))
deduper = Deduper(reference=any_paper, candidates=[any_paper])

rows = []
for row in pairs_df.itertuples(index=False):
    rec_a = record_cache.get(int(row.id_a))
    rec_b = record_cache.get(int(row.id_b))
    if rec_a is None or rec_b is None:
        continue

    prob, field_scores, early_stop_reason = deduper.dedupe_weighted(
        rec_a,
        rec_b,
        weights=WEIGHTS,
        intercept=INTERCEPT,
        return_details=True, #return full details rather than just dupe/non-dupe output for more detailed evaluation 
    )

    rows.append(
        {
            "id_a": row.id_a,
            "id_b": row.id_b,
            "is_dupe": row.is_dupe,
            "prob": prob,
            "early_stop": early_stop_reason,
            **{f"score_{k}": v for k, v in field_scores.items()},
        }
    )

scored_df = pd.DataFrame(rows)
scored_df["is_predicted_duplicate"] = scored_df["prob"] > THRESHOLD

# explore record level metrics to determine how deduplication performed on your dataset 
record_metrics = record_level_metrics_for_threshold(
    df_orig=df,
    scored_pairs_df=scored_df,
    threshold=THRESHOLD,
    id_column="recordid",
    true_cluster_column="duplicateid",
    probability_column="prob",
)
print(record_metrics)
```
## Notes

At present, we have import functions for CSV data only. In future, this can be adapted for different reference formats and inputs. 


## Column Naming

Defaults in helper functions are still based on:

- `recordid`
- `duplicateid`

If your data uses snake_case names, pass all column arguments explicitly (`id_column`, `dup_column`, `true_cluster_column`, and related arguments).


## Related Notebooks

Gold-standard evaluation examples:

- `notebooks/Development_dataset_testing.ipynb`
- `notebooks/Test_datasets_eval.ipynb`

Unlabeled dedupe example:

- `notebooks/Example_workflow.ipynb`
