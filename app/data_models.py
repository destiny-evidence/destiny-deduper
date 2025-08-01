"""Data models and associated methods used to specify input and output data."""

from destiny_sdk.identifiers import DOIIdentifier
from pydantic import BaseModel


class IncomingRecord(BaseModel):
    """
    The data structure for incoming records
    for deduplication.

    """

    title: str | None
    first_author: str | None
    mid_authors: list[str] | None
    last_author: str | None
    doi: DOIIdentifier | None
    year: int | None
