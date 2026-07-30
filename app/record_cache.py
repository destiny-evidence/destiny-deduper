"""Record validation and cache-building helpers."""

import pandas as pd
from pydantic import ValidationError

from app.data_models import GoldStandardPaper, Paper
from app.import_references import row_to_paper_kwargs


def build_record_cache(
    df: pd.DataFrame,
    id_column: str = "recordid",
    include_ids: set[int] | None = None,
    record_model: type[Paper] = GoldStandardPaper,
    *,
    return_invalid_records: bool = False,
) -> (
    tuple[dict[int, Paper], int] | tuple[dict[int, Paper], int, list[dict[str, object]]]
):
    """
    Parse records into validated Pydantic models with validation tracking.

    Args:
        df: Input records dataframe.
        id_column: Name of the column used as cache key.
        include_ids: Optional subset of IDs to include in the cache.
        record_model: Pydantic model used to parse each row. Defaults to
            GoldStandardPaper and can be set to Paper for unlabeled workflows.
        return_invalid_records: If True, return row-level validation errors.

    """
    cache: dict[int, Paper] = {}
    validation_errors = 0
    invalid_records: list[dict[str, object]] = []

    for record in df.to_dict(orient="records"):
        record_id = record.get(id_column)
        if record_id is None or pd.isna(record_id):
            continue

        parsed_id = int(record_id)
        if include_ids is not None and parsed_id not in include_ids:
            continue

        try:
            cache[parsed_id] = record_model(**row_to_paper_kwargs(record))
        except ValidationError as exc:
            validation_errors += 1
            if return_invalid_records:
                invalid_records.append(
                    {
                        "recordid": parsed_id,
                        "errors": exc.errors(),
                    }
                )

    if return_invalid_records:
        return cache, validation_errors, invalid_records

    return cache, validation_errors
