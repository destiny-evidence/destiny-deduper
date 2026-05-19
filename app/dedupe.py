"""Deduplication workflow, in main class `Deduper` with various algorithms."""

import re
from collections import Counter
from enum import StrEnum, auto
from math import exp
from typing import Literal

from rapidfuzz.distance import JaroWinkler as _JaroWinkler
from rapidfuzz.distance import Levenshtein as _Levenshtein
from loguru import logger

from app.data_models_old import Paper

# logit model weights
WEIGHTS = {
    "doi": 2.28,
    "title": 6.38,
    "authors": 2.44,
    "year": 0.12,
    "journal": 1.42,
    "pages": 1.01,
    "abstract": -0.29,
    "volume": 0.38,
    "issue": -0.22,
}
INTERCEPT = -8.80
TITLE_VETO_THRESHOLD = 0.88


class StringDistanceAlgorithm(StrEnum):
    """
    Available string distance algorithms.
    Will need to be implemented in Deduper class.

    Args:
        StrEnum (_type_): name of algorithm

    """

    JARO_WINKLER = auto()
    LEVENSHTEIN = auto()


class Deduper:
    """
    A class that handles deduping.
    Can do one-to-one and one-to-many (sequentially).

    """

    def __init__(
        self,
        reference: Paper,
        candidates: list[Paper] | Paper,
        default_string_distance_algorithm: StringDistanceAlgorithm = StringDistanceAlgorithm.JARO_WINKLER,
    ) -> None:
        """
        Inititialse Deduper instance.

        Args:
            reference (Paper): the record to dedupe.
            candidates (list[Paper] | None): the records to dedupe against.
            default_string_distance_algorithm (StringDistanceAlgorithms)

        """
        self.reference = reference
        self.candidates = candidates
        self.default_string_distance_algorithm = default_string_distance_algorithm
        self.title_veto_threshold = TITLE_VETO_THRESHOLD

    def compare_one_to_one(
        self,
        record_a: Paper,
        record_b: Paper,
        string_distance_algorithm: StringDistanceAlgorithm | None = None,
        **kwargs,
    ) -> float:
        """
        Compare two papers.
        For each comparison, we get a float.
        The return value of this method will be the mean of all those.

        NOTE: This is obviously a naive implementation,
        but it should be simple enough to aggregate
        individual scores into one overal dupe-or-not measure.

        Args:
            reference (Paper): _description_
            candidate (Paper): _description_
            string_distance_algorithm (StringDistanceAlgorithm | None, optional): _description_. Defaults to None.

        Returns:
            float: between 0 and 1 - mean probability that it's a dupe.

        """
        if string_distance_algorithm is None:
            string_distance_algorithm = self.default_string_distance_algorithm

        fields_record_a = [k for k, v in record_a.model_dump().items() if v is not None]
        fields_record_b = [k for k, v in record_b.model_dump().items() if v is not None]

        fields_to_compare = [x for x in fields_record_a if x in fields_record_b]
        logger.debug(f"fields to compare: {', '.join(fields_to_compare)}")

        scores = {}
        for field in fields_to_compare:
            compare_method_name = f"compare_{field}"
            compare_method = getattr(self, compare_method_name, None)
            if compare_method is None:
                logger.warning(f"No compare method for field: {field}")
                continue

            kwargs.update({"string_distance_algorithm": string_distance_algorithm})
            try:
                logger.debug(f"comparison_method: {compare_method}")
                logger.debug(f"overall_kwargs: {kwargs}")
                score = compare_method(record_a, record_b, **kwargs)
                scores[field] = score
            except NotImplementedError as e:
                logger.error(
                    f"method {compare_method} is not yet implemented. skipping."
                )
                logger.error(f"original error message: {e}")
                continue

        logger.debug(f"scores for each field: {scores}")
        return sum(scores.values()) / len(scores)

    def score_pair(
        self,
        record_a: Paper,
        record_b: Paper,
        string_distance_algorithm: StringDistanceAlgorithm | None = None,
        weights: dict[str, float] = WEIGHTS,
        intercept: float = INTERCEPT,
        fields: list[str] | None = None,
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        """
        Score a pair once and return both the probability and per-field matches.
        Optionally restrict to a subset of fields.
        Args:
            record_a (Paper): First record.
            record_b (Paper): Second record.
            string_distance_algorithm: Algorithm to use.
            weights: Field weights.
            intercept: Intercept for logistic regression.
            fields: List of fields to use (if None, use all in weights).
        Returns:
            tuple: (probability, field_scores)
        """
        if string_distance_algorithm is None:
            string_distance_algorithm = self.default_string_distance_algorithm

        kwargs = dict(kwargs)
        kwargs["string_distance_algorithm"] = string_distance_algorithm

        if self._should_early_stop(record_a, record_b):
            return 0.0, {}

        field_scores: dict[str, float] = {}
        weighted_total = 0.0

        # Determine which fields to use
        if fields is not None:
            fields_to_use = [(f, weights[f]) for f in fields if f in weights]
        else:
            fields_to_use = list(weights.items())

        for field, weight in fields_to_use:
            val_a = getattr(record_a, field, None)
            val_b = getattr(record_b, field, None)

            if val_a is None or val_b is None:
                continue

            compare_method = getattr(self, f"compare_{field}", None)
            if compare_method is None:
                logger.warning(f"No compare method for field: {field}")
                continue

            try:
                match_score = compare_method(record_a, record_b, **kwargs)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error comparing {field}: {e}")
                continue

            field_scores[field] = match_score
            weighted_total += match_score * weight
            logger.debug(
                f"{field}: match_score={match_score:.4f}, weight={weight}, "
                f"weighted={match_score * weight:.4f}"
            )

        raw_score = weighted_total + intercept
        logger.debug(f"Raw score (before sigmoid): {raw_score:.4f}")
        probability = 1 / (1 + exp(-raw_score))
        logger.debug(f"Weighted dedup probability: {probability:.4f}")

        return probability, field_scores

    def compare_one_to_many(
        self,
        string_distance_algorithm: StringDistanceAlgorithm | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Compare one record to several candidates.

        Args:
            string_distance_algorithm (StringDistanceAlgorithm | None, optional): _description_. Defaults to None.

        Returns:
            list[float] | float: for each candidate, between 0 and 1 -
            mean probability that it's a dupe.

        """
        if isinstance(self.candidates, Paper):
            self.candidates = [self.candidates]

        dupe_probabilities = []
        for cand in self.candidates:
            prob = self.compare_one_to_one(
                record_a=self.reference,
                record_b=cand,
                string_distance_algorithm=string_distance_algorithm,
                **kwargs,
            )
            logger.debug(
                f"dupe prob for candidate {cand.model_dump().get('title', None)}: {prob}"
            )
            dupe_probabilities.append(prob)

        return dupe_probabilities

    def dedupe_unweighted(
        self,
        record_a: Paper,
        record_b: Paper,
        string_distance_algorithm: StringDistanceAlgorithm | None = None,
        **kwargs,
    ) -> float:
        """
        Hierarchical unweighted deduplication algorithm.

        Implements early stopping logic:
        - If DOI and PAGES both present and both don't match -> STOP (return 0.0)
        - Otherwise compute mean of all available match scores
        - Excludes abstract and issue from the mean calculation

        Args:
            record_a (Paper): first paper to compare
            record_b (Paper): second paper to compare
            string_distance_algorithm (StringDistanceAlgorithm | None): algorithm for string comparisons
            **kwargs: additional arguments passed to comparison methods

        Returns:
            float: 'probability' between 0 and 1 that records are duplicates

        """
        if string_distance_algorithm is None:
            string_distance_algorithm = self.default_string_distance_algorithm

        kwargs.update({"string_distance_algorithm": string_distance_algorithm})

        if self._should_early_stop(record_a, record_b):
            return 0.0

        # fields to compare (excluding abstract and issue)
        fields_to_score = [
            "doi",
            "openalex_id",
            "pubmed_id",
            "isbn",
            "issn",
            "title",
            "authors",
            "year",
            "journal",
            "publisher",
            "pages",
            "volume",
        ]

        scores = {}
        for field in fields_to_score:
            val_a = getattr(record_a, field, None)
            val_b = getattr(record_b, field, None)

            if val_a is None or val_b is None:
                logger.debug(
                    f"missing record. {field} record_a: {val_a},"
                    f"{field} record_b: {val_b}"
                )
                continue

            compare_method_name = f"compare_{field}"
            compare_method = getattr(self, compare_method_name, None)

            if compare_method is None:
                logger.warning(f"No compare method for field: {field}")
                continue

            try:
                score = compare_method(record_a, record_b, **kwargs)
                scores[field] = score
                logger.debug(f"Score for {field}: {score}")
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error comparing {field}: {e}")
                continue

        if not scores:
            logger.warning("No fields available for comparison")
            return 0.0

        mean_score = sum(scores.values()) / len(scores)
        logger.debug(f"Unweighted dedup score: {mean_score:.4f}")

        return mean_score

    def dedupe_weighted(
        self,
        record_a: Paper,
        record_b: Paper,
        string_distance_algorithm: StringDistanceAlgorithm | None = None,
        weights: dict[str, float] = WEIGHTS,
        intercept: float = INTERCEPT,
        **kwargs,
    ) -> float:
        """
        Hierarchical weighted deduplication algorithm using logistic regression.

        Implements early stopping logic:
        - If DOI and PAGES both present and both don't match -> STOP (return 0.0)
        - Otherwise compute weighted scores using logistic regression coefficients
        - Applies sigmoid function to get probability

        Weights from logistic regression:
        - doi: 2.28
        - title: 6.38
        - authors: 2.44
        - year: 0.12
        - journal: 1.42
        - pages: 1.01
        - abstract: -0.29 (negative weight)
        - volume: 0.38
        - issue: -0.22 (negative weight)
        - intercept: -8.80

        Args:
            record_a (Paper): first paper to compare
            record_b (Paper): second paper to compare
            string_distance_algorithm (StringDistanceAlgorithm | None): algorithm for string comparisons
            **kwargs: additional arguments passed to comparison methods

        Returns:
            float: probability between 0 and 1 that records are duplicates

        """
        if string_distance_algorithm is None:
            string_distance_algorithm = self.default_string_distance_algorithm

        kwargs.update({"string_distance_algorithm": string_distance_algorithm})

        if self._should_early_stop(record_a, record_b):
            return 0.0

        weighted_scores = {}
        for field, weight in weights.items():
            # check if both records have field from weights
            val_a = getattr(record_a, field, None)
            val_b = getattr(record_b, field, None)

            if val_a is None or val_b is None:
                logger.debug(
                    f"missing record. {field} record_a: {val_a},"
                    f"{field} record_b: {val_b}"
                )
                continue

            compare_method_name = f"compare_{field}"
            compare_method = getattr(self, compare_method_name, None)

            if compare_method is None:
                logger.warning(f"No compare method for field: {field}")
                continue

            try:
                match_score = compare_method(record_a, record_b, **kwargs)
                weighted_score = match_score * weight
                weighted_scores[field] = weighted_score
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error comparing {field}: {e}")
                continue

        # sum weighted scores + intercept & apply sigmoid
        # to pseudo-standardise the scores b/w 0 and 1.
        # NOTE: if we wanted to use non logit-derived weights,
        # we could use the SD and mean of the actual distribution
        # of scores to standardise. that might in fact be better.
        # but let's stick with this for now.
        raw_score = sum(weighted_scores.values()) + intercept
        logger.debug(f"Raw score (before sigmoid): {raw_score:.4f}")
        probability = 1 / (1 + exp(-raw_score))
        logger.debug(f"Weighted dedup probability: {probability:.4f}")

        # Apply DOI mismatch penalty if both DOIs are present and do not match
        doi_a = getattr(record_a, "doi", None)
        doi_b = getattr(record_b, "doi", None)
        penalty_factor = 1.0
        if doi_a is not None and doi_b is not None:
            norm_a = Deduper._normalize_doi(getattr(doi_a, "identifier", None))
            norm_b = Deduper._normalize_doi(getattr(doi_b, "identifier", None))
            if norm_a and norm_b and norm_a != norm_b:
                penalty_factor = 0.8  # Softer penalty: reduce by 0.2
                logger.debug(f"DOI mismatch penalty applied: {norm_a} != {norm_b}, factor={penalty_factor}")
        return probability * penalty_factor

    def _should_early_stop(self, record_a: Paper, record_b: Paper) -> bool:
        if (
            record_a.doi is not None
            and record_b.doi is not None
            and record_a.pages is not None
            and record_b.pages is not None
        ):
            doi_match = record_a.doi.identifier == record_b.doi.identifier
            pages_match = self.compare_pages(record_a, record_b) == 1.0

            if not doi_match and not pages_match:
                logger.debug("Early stop: DOI and PAGES both don't match")
                return True

        # Page mismatch veto: only trigger if pages differ AND abstracts conflict, with strong metadata match
        if record_a.pages is not None and record_b.pages is not None:
            pages_match = self.compare_pages(record_a, record_b) == 1.0
            if not pages_match:
                # Use same strong metadata match as year-gap rule
                title_sim = self.compare_title(record_a, record_b)
                authors_sim = self.compare_authors(record_a, record_b)
                journal_sim = self.compare_journal(record_a, record_b)
                strong_metadata_match = (
                    title_sim >= 0.97 and (authors_sim >= 0.92 or journal_sim >= 0.92)
                )
                if strong_metadata_match and self._has_abstract_conflict(record_a, record_b):
                    logger.debug("Early stop: normalized pages differ AND abstracts conflict with strong metadata match")
                    return True

        if record_a.doi is not None and record_b.doi is not None:
            doi_a = Deduper._normalize_doi(getattr(record_a.doi, "identifier", None))
            doi_b = Deduper._normalize_doi(getattr(record_b.doi, "identifier", None))

            if doi_a and doi_b and doi_a != doi_b:
                base_a = re.sub(r"\.pub\d+$", "", doi_a)
                base_b = re.sub(r"\.pub\d+$", "", doi_b)
                if base_a == base_b:
                    logger.debug("Early stop: same DOI base but different .pubN version")
                    return True

        # Title mismatch veto: if title similarity is too low,
        # treat as non-dupe to reduce false positives.
        title_sim = self.compare_title(record_a, record_b)
        if title_sim < self.title_veto_threshold:
            logger.debug(
                "Early stop: title similarity %.3f below threshold %.2f",
                title_sim,
                self.title_veto_threshold,
            )
            return True

        # Year-gap + abstract-numeric conflict:
        # if core metadata still looks very similar but publication years diverge,
        # use abstract numbers as a disambiguation veto.
        if record_a.year is not None and record_b.year is not None:
            if abs(int(record_a.year) - int(record_b.year)) > 1:
                title_sim = self.compare_title(record_a, record_b)
                authors_sim = self.compare_authors(record_a, record_b)
                journal_sim = self.compare_journal(record_a, record_b)

                strong_metadata_match = (
                    title_sim >= 0.97 and (authors_sim >= 0.92 or journal_sim >= 0.92)
                )

                if strong_metadata_match and self._has_abstract_conflict(
                    record_a, record_b
                ):
                    logger.debug(
                        "Early stop: year gap > 1 with conflicting abstracts"
                    )
                    return True


        return False

    @staticmethod
    def _extract_abstract_numbers(text: str) -> Counter:
        """Extract numeric tokens from abstract text as a multiset."""
        if not text:
            return Counter()
        tokens = re.findall(r"\d+(?:\.\d+)?", text)
        return Counter(tokens)

    def _has_abstract_conflict(self, record_a: Paper, record_b: Paper) -> bool:
        """Return True when abstracts are present and show conflicting content."""
        abstract_a = getattr(record_a, "abstract", None)
        abstract_b = getattr(record_b, "abstract", None)

        if not abstract_a or not abstract_b:
            return False

        # In the select year-gap subgroup, abstract disagreement is treated as
        # strong evidence against a duplicate.
        abstract_sim = self.compare_abstract(record_a, record_b)
        if abstract_sim < 0.80:
            return True

        nums_a = self._extract_abstract_numbers(abstract_a)
        nums_b = self._extract_abstract_numbers(abstract_b)

        # If neither abstract has numbers, similarity check above decides.
        if not nums_a and not nums_b:
            return False

        return nums_a != nums_b

    @staticmethod
    def _normalize_doi(doi: str) -> str | None:
        """Normalize DOI format to consistent lowercase, remove prefixes, decode symbols."""
        if not doi:
            return None
        doi = doi.replace("%28", "(").replace("%29", ")")
        doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        doi = re.sub(r"^DOI[: ]?", "", doi, flags=re.IGNORECASE)
        match = re.search(
            r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, flags=re.IGNORECASE
        )
        if not match:
            return None
        return match.group(0).strip().lower()

    def compare_doi(
        self,
        record_a: Paper,
        record_b: Paper,
        **kwargs,
    ) -> float:
        """
        Compare DOIs between two records (exact match after normalization).

        Returns:
            float: 1.0 if DOIs match, 0.0 otherwise
        """
        doi_a = Deduper._normalize_doi(getattr(record_a.doi, "identifier", None))
        doi_b = Deduper._normalize_doi(getattr(record_b.doi, "identifier", None))

        if not doi_a or not doi_b:
            return 0.0
        return 1.0 if doi_a == doi_b else 0.0

    def compare_openalex_id(
        self,
        record_a: Paper,
        record_b: Paper,
        method: Literal["string_match", "http", "both"] = "string_match",
        **kwargs,
    ) -> float:
        """
        Compare two openalex ids.

        NOTE: `http` might be a method where we follow the url (requests?) for
        doi_a and doi_b and check if they're the same (maybe string distance on
        the resulting html?).

        Args:
            record_a (Paper): a Paper instance.
            record_b (Paper): a Paper instance.
            method (Literal[&quot;string_match&quot;, &quot;http&quot;, &quot;both&quot;], optional): _description_. Defaults to "string_match".

        Returns:
            float: between 0 and 1.

        """
        if record_a.openalex_id is None or record_b.openalex_id is None:
            return 0.0

        openalex_id_a_str = (
            record_a.openalex_id.identifier
        )  # should have openalex url removed
        openalex_id_b_str = record_b.openalex_id.identifier

        if method == "string_match":
            # NOTE - does this even make sense? i think this is what @kaitlynhair code is doing
            # but does approximate string similarity imply similarity of the
            # underlying record, or should we just do a == b comparison?
            return Deduper.calculate_string_distance(
                openalex_id_a_str, openalex_id_b_str, **kwargs
            )

        not_implemented_err_msg = f"method {method} is not yet implemented."
        raise NotImplementedError(not_implemented_err_msg)

    def compare_pubmed_id(
        self,
        record_a: Paper,
        record_b: Paper,
    ) -> float:
        """
        Compare two pubmed ids.

        Args:
            record_a (Paper): a Paper instance.
            record_b (Paper): a Paper instance.
            string_distance_algorithm (_type_, optional): _description_. Defaults to StringDistanceAlgorithm | str | None=None.

        Returns:
            float: _description_

        """
        if record_a.pubmed_id is None or record_b.pubmed_id is None:
            return 0.0

        pubmed_id_a_str = record_a.pubmed_id.identifier
        pubmed_id_b_str = record_b.pubmed_id.identifier

        return float(pubmed_id_a_str == pubmed_id_b_str)

    def compare_isbn(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 ISBNs after normalization.

        Returns:
            float: 1.0 if equal, 0.0 otherwise.
        """

        def normalize_isbn(isbn: str) -> str:
            """Normalize ISBN string by removing common extra patterns."""
            if not isbn:
                return ""
            isbn = re.sub(r"\s*\(PRINT\).*", "", isbn, flags=re.IGNORECASE)
            isbn = re.sub(r"\s*\(ELECTRONIC\).*", "", isbn, flags=re.IGNORECASE)
            isbn = re.sub(r"\\N.*", "", isbn)
            return isbn.strip().lower()

        isbn_a = normalize_isbn(record_a.isbn)
        isbn_b = normalize_isbn(record_b.isbn)

        if not isbn_a or not isbn_b:
            return 0.0
        return 1.0 if isbn_a == isbn_b else 0.0

    def compare_authors(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        if not record_a.authors or not record_b.authors:
            logger.warning("One or both records have no authors.")
            return 0.0

        try:
            authors_a = ", ".join(
                [
                    getattr(a, "author_name", None) or getattr(a, "display_name", "")
                    for a in record_a.authors
                    if a is not None
                ]
            )
            authors_b = ", ".join(
                [
                    getattr(a, "author_name", None) or getattr(a, "display_name", "")
                    for a in record_b.authors
                    if a is not None
                ]
            )
        except AttributeError as e:
            logger.error(f"Error extracting authors: {e}")
            return 0.0

        # Return 0 if either authors list is empty
        if not authors_a or not authors_b:
            logger.warning("One or both authors lists are empty after extraction.")
            return 0.0

        # Allow override algorithm via kwargs, fallback to Jaro-Winkler
        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )
        return Deduper.calculate_string_distance(
            authors_a, authors_b, string_distance_algorithm=algo
        )

    def compare_title(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare two titles using Levenshtein distance.
        Returns a float between 0 and 1.
        """
        # Return 0 if either title is missing
        if not record_a.title or not record_b.title:
            return 0.0

        # Allow override algorithm via kwargs, fallback to Levenshtein
        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.LEVENSHTEIN
        )

        return Deduper.calculate_string_distance(
            record_a.title, record_b.title, string_distance_algorithm=algo
        )

    def compare_year(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 years.

        NOTE: right now, it'll return 1.0 if the same, 0 otherwise.
        we may want to implement some kind of formula for returning
        a fractional of 1 depending on proximity to 1.

        Args:
            record_a (Paper): a Paper instance.
            record_b (Paper): a Paper instance.

        Returns:
            float: 1 if true, 0 if not.

        """
        if record_a.year is None or record_b.year is None:
            return 0.0

        year_a = record_a.year
        year_b = record_b.year
        return float(year_a == year_b)

    def compare_journal(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare two journal names using Jaro-Winkler distance.
        Returns a float between 0 and 1.
        """
        # Return 0 if either journal is missing
        if not record_a.journal or not record_b.journal:
            return 0.0

        # Allow override algorithm via kwargs, fallback to Jaro-Winkler
        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )

        return Deduper.calculate_string_distance(
            record_a.journal, record_b.journal, string_distance_algorithm=algo
        )

    def compare_publisher(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare two publisher names using string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.publisher is None or record_b.publisher is None:
            return 0.0

        publisher_a = record_a.publisher.lower().strip()
        publisher_b = record_b.publisher.lower().strip()

        return Deduper.calculate_string_distance(publisher_a, publisher_b, **kwargs)

    def compare_pages(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 sets of page delineations.
        Returns 1.0 if canonicalized page values match, 0.0 otherwise.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: 1.0 if match, 0.0 otherwise

        """
        if record_a.pages is None or record_b.pages is None:
            return 0.0

        pages_a = Deduper._normalize_pages(record_a.pages)
        pages_b = Deduper._normalize_pages(record_b.pages)
        return float(pages_a == pages_b)

    @staticmethod
    def _normalize_pages(pages: str) -> str:
        """Normalize page strings so style variants compare consistently."""
        if not pages:
            return ""

        page_str = str(pages).strip().lower()
        page_str = re.sub(r"\s+", "", page_str)
        page_str = page_str.replace("\u2013", "-").replace("\u2014", "-")

        # Canonicalize shorthand page ranges like s30-50 -> s30-s50.
        match = re.match(r"^([a-z]*)(\d+)-([a-z]*)(\d+)$", page_str)
        if match:
            prefix_start, start, prefix_end, end = match.groups()
            if not prefix_end:
                prefix_end = prefix_start
            return f"{prefix_start}{start}-{prefix_end}{end}"

        return page_str

    def compare_abstract(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare abstracts between two papers using Levenshtein distance,
        after removing boilerplate words/phrases (e.g., 'methods', 'results', 'ABSTRACT:').

        Returns:
            float: similarity score between 0 and 1
        """
        if not record_a.abstract or not record_b.abstract:
            return 0.0

        def preprocess_abstract(text: str) -> str:
            # Remove common boilerplate words/section headers
            boilerplate = [
                r"\babstract:?\b",
                r"\bbackground:?\b",
                r"\bmethods?:?\b",
                r"\bresults?:?\b",
                r"\bconclusions?:?\b",
                r"\bobjective[s]?:?\b",
                r"\bintroduction:?\b",
                r"\bpurpose:?\b",
                r"\bdesign:?\b",
                r"\bsetting:?\b",
                r"\bparticipants?:?\b",
                r"\bmain outcome[s]?:?\b",
                r"\bdiscussion:?\b",
                r"\bimplications?:?\b",
                r"\bconclusion:?\b",
            ]
            text = text.lower()
            for pat in boilerplate:
                text = re.sub(pat, " ", text)
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

        abs_a = preprocess_abstract(record_a.abstract)
        abs_b = preprocess_abstract(record_b.abstract)

        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.LEVENSHTEIN
        )
        return Deduper.calculate_string_distance(
            abs_a, abs_b, string_distance_algorithm=algo
        )

    def compare_volume(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 sets of volumes,
        using Jaro-Winkler distance.
        """
        if not record_a.volume or not record_b.volume:
            return 0.0

        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )
        return Deduper.calculate_string_distance(
            record_a.volume, record_b.volume, string_distance_algorithm=algo
        )

    def compare_issue(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 sets of issues,
        using Jaro-Winkler distance.
        """
        if not record_a.issue or not record_b.issue:
            return 0.0

        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )
        return Deduper.calculate_string_distance(
            record_a.issue, record_b.issue, string_distance_algorithm=algo
        )

    def compare_issn(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 ISSN strings.
        Returns 1.0 if exact match, 0.0 otherwise.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: 1.0 if match, 0.0 otherwise

        """
        if record_a.issn is None or record_b.issn is None:
            return 0.0

        return float(record_a.issn == record_b.issn)

    @staticmethod
    def calculate_string_distance(
        string_a: str,
        string_b: str,
        string_distance_algorithm: StringDistanceAlgorithm = StringDistanceAlgorithm.JARO_WINKLER,
    ) -> float:
        """
        Calculate string distance, genericly.

        Args:
            string_a (str): a string
            string_b (str): another string

        Returns:
            float: between 0 and 1

        """
        method_map = {  # @Adam-Hammo @mootpointer -- perhaps this should go elsewhere?
            StringDistanceAlgorithm.JARO_WINKLER: Deduper.jaro_winkler_distance,
            StringDistanceAlgorithm.LEVENSHTEIN: Deduper.levenshtein_distance,
        }

        method = method_map.get(string_distance_algorithm)
        logger.debug(f"selected method: {method}")
        if not method:
            no_method_for_alg_err_msg = (
                f"No method for algorithm: {string_distance_algorithm}"
            )
            raise ValueError(no_method_for_alg_err_msg)
        return method(string_a, string_b)

    @staticmethod
    def jaro_winkler_distance(string_a: str, string_b: str) -> float:
        """
        Calculate Jaro-Winkler distance b/w 2 strings.
        Here: a wrapper around rapidfuzz library, but
        we could easily implement this ourselves.

        Args:
            string_a (str): a string.
            string_b (str): another string.

        Returns:
            float: between 0 and 1.

        """
        return _JaroWinkler.similarity(string_a, string_b)

    @staticmethod
    def levenshtein_distance(string_a: str, string_b: str) -> float:
        """
        Calculate normalized Levenshtein similarity between two strings.
        Returns a float between 0 (completely different) and 1 (identical).

        Args:
            string_a (str): first string
            string_b (str): second string

        Returns:
            float: similarity score between 0 and 1
        """
        if not string_a or not string_b:
            return 0.0

        # lowercase and strip
        a, b = string_a.lower().strip(), string_b.lower().strip()

        return _Levenshtein.normalized_similarity(a, b)
