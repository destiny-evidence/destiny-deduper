# Scorer fixtures

A small set of record pairs, each written to exercise one path through
`score_pair`. The data is synthetic. Nothing here comes from a corpus, there are
no abstracts and no author names, and most DOIs use the reserved `10.1000`
example prefix.

## Files

| File | Contents |
| --- | --- |
| `pairs.csv` | One row per pair: the two record ids, which input paths the pair applies to, what it targets, and the expected result. |
| `papers.json` | Every record as stored `Paper` fields, keyed by record id. |
| `records.csv` | The same records in CSV form, for the records used by pairs marked `csv`. |

## Expected values

The `expected_*` columns were produced by scoring each pair with the algorithm
in this repository. They are a regression net, not an independent ground truth:
if a deliberate change moves a score, update the values in the same commit so
the change is visible in review.

`expected_probability` is compared as a string formatted to six decimal places,
which avoids float comparison without needing a tolerance.

## The `paths` column

`paper` means the pair can be built directly from `papers.json`. `paper,csv`
means it can also be reconstructed by loading `records.csv` through
`load_reference_csv`.

`ID-01` and `ID-02` are `paper` only. They carry a PubMed identifier and an
ISBN, and the default CSV columns have no home for either, so a CSV round-trip
cannot reproduce them. Both are still worth keeping: neither identifier changes
the probability, so what these pairs cover is that the records build and score
without raising.

## Coverage

Five of the seven `EarlyStopReason` members are reachable and each has a pair.
`test_fixture_set_covers_every_reachable_early_stop_reason` asserts that, so
adding a new reason to the enum fails until a fixture covers it.

The two that are not reachable are named in `tests/test_scorer_fixtures.py` with
the reason why.
