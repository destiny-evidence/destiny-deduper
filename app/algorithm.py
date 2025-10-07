from pydantic import BaseModel
from typing import Optional

# ---- Mock destiny_sdk equivalents ----
class DOIIdentifier(BaseModel):
    identifier: str

class OpenAlexIdentifier(BaseModel):
    identifier: str

class PubMedIdentifier(BaseModel):
    identifier: str

class Authorship(BaseModel):
    author_name: str

# ---- Your Paper model (simplified for test) ----
class Paper(BaseModel):
    doi: Optional[DOIIdentifier] = None
    openalex_id: Optional[OpenAlexIdentifier] = None
    pubmed_id: Optional[PubMedIdentifier] = None
    title: Optional[str] = None
    isbn: Optional[str] = None
    authors: Optional[list[Authorship]] = None
    year: Optional[int] = None
    pages: Optional[str] = None
    journal: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None

papers = [
    # --- Exact duplicate ---
    Paper(
        doi=DOIIdentifier(identifier="HTTPS:/DX.DOI/10.1016/j.neuroimage.2020.1170123"),
        title="Deep learning for brain MRI segmentation",
        authors=[Authorship(author_name="Smith J"), Authorship(author_name="Doe A")],
        year=2020,
        isbn="978-3-16-148410-0",
        pages = "121-35",
        journal="NeuroImage",
        publisher="Elsevier",
        abstract="A review of deep learning approaches for MRI segmentation."
    ),
    Paper(
        doi=DOIIdentifier(identifier="10.1016/j.neuroimage.2020.117012"),
        title="Deep learning for brain MRI segmentations",
        authors=[Authorship(author_name="Smith J."), Authorship(author_name="Doe A.")],
        year=2020,
        isbn="978-3-16-148410-0",
        pages = "121-136",
        journal="NeuroImage",
        publisher="Elsevier",
        abstract="A review of deep learning approaches for MRI segmentation."
    ),

    # --- Near-duplicate (small variation in title/authors) ---
    Paper(
        doi=DOIIdentifier(identifier="10.1016/j.neuroimage.2020.117012"),
        title="Deep learning methods for MRI brain segmentation",
        authors=[Authorship(author_name="J Smith"), Authorship(author_name="A Doe")],
        year=2020,
        journal="NeuroImage",
        publisher="Elsevier",
        abstract="Deep learning models applied to MRI segmentation tasks."
    ),

    # --- Different article (same domain) ---
    Paper(
        doi=DOIIdentifier(identifier="10.1016/j.media.2021.102025"),
        title="Deep learning for CT scan segmentation",
        authors=[Authorship(author_name="Smith J"), Authorship(author_name="Brown T")],
        year=2021,
        journal="Medical Image Analysis",
        publisher="Elsevier",
        abstract="Segmentation of CT scans using convolutional networks."
    ),

    # --- Completely different ---
    Paper(
        doi=DOIIdentifier(identifier="10.1093/bioinformatics/btz123"),
        title="Machine learning in genomics",
        authors=[Authorship(author_name="Lee K"), Authorship(author_name="Zhao M")],
        year=2019,
        journal="Bioinformatics",
        publisher="Oxford University Press",
        abstract="Applications of ML to genomic sequence analysis."
    ),

    Paper(
        doi=DOIIdentifier(identifier="10.1038/s41557-022-01001-7"),
        title="Quantum computing approaches for chemistry",
        authors=[Authorship(author_name="Chen Y"), Authorship(author_name="Patel S")],
        year=2022,
        journal="Nature Chemistry",
        publisher="Nature Publishing Group",
        abstract="Quantum algorithms for chemical modeling."
    ),
]

from itertools import combinations
import pandas as pd
from app.dedupe import Deduper, StringDistanceAlgorithm  # Adjust path as needed

# List of papers
papers_list = papers  # from your example

# Prepare results storage
results = []

# Compare every pair of papers
for a, b in combinations(papers_list, 2):
    deduper = Deduper(reference=a, candidates=b)
    score = deduper.compare_one_to_one(a, b, string_distance_algorithm=StringDistanceAlgorithm.JARO_WINKLER)
    results.append({
        "paper_a_title": a.title,
        "paper_b_title": b.title,
        "similarity_score": score
    })

# Convert to DataFrame for easy viewing
df_results = pd.DataFrame(results)

print(df_results)
