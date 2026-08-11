"""Deduplication workflow, in main class `Deduper` with various algorithms."""

import re
from enum import StrEnum, auto
from math import exp
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz.distance import JaroWinkler as _JaroWinkler
from rapidfuzz.distance import Levenshtein as _Levenshtein

from destiny_dedupe.config import get_settings
from destiny_dedupe.data_models import Paper
from destiny_dedupe.early_stop import EARLY_STOP_RULES, ComparisonContext, EarlyStopRule
from destiny_dedupe.normalisers import (
    normalise_doi,
    normalise_isbn,
    normalise_pages,
    strip_doi_punctuation,
)
from destiny_dedupe.pair_score_result import (
    EarlyStopReason,
    FieldResult,
    FieldStatus,
    PairLabel,
    PairScoreResult,
)
from destiny_dedupe.utils import (
    contains_language_name,
    extract_abstract_numbers,
    is_journal_abbreviation_match,
    split_title_segments,
)

settings = get_settings()

# constants
WEIGHTS = settings.weights.model_dump()
INTERCEPT: float = settings.weights.intercept
ABSTRACT_SIMILARITY_THRESHOLD = settings.thresholds.abstract.similarity
JOURNAL_ABBREVIATION_THRESHOLD = settings.thresholds.journal.abbreviation


class StringDistanceAlgorithm(StrEnum):
    """
    Available string distance algorithms.
    Will need to be implemented in Deduper class.

    Args:
        StrEnum (_type_): name of algorithm

    """

    JARO_WINKLER = auto()
    LEVENSHTEIN = auto()


class ScorePairConfig(BaseModel):
    """Configuration for scoring a pair of papers."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    string_distance_algorithm: StringDistanceAlgorithm | None = None
    weights: dict[str, float] = Field(default_factory=lambda: WEIGHTS.copy())
    intercept: float = INTERCEPT
    fields: list[str] | None = None


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
        config: ScorePairConfig | None = None,
        **kwargs,
    ) -> PairScoreResult:
        """
        Score a pair and return a structured result.

        Args:
            record_a (Paper): First record.
            record_b (Paper): Second record.
            config (ScorePairConfig | None): Scoring configuration.
            **kwargs: Additional keyword arguments passed to comparison methods.

        Returns:
            PairScoreResult with probability, per-field results, early-stop
            reason, DOI mismatch flag, and suggested label.

        """
        config = config or ScorePairConfig()

        string_distance_algorithm = (
            config.string_distance_algorithm or self.default_string_distance_algorithm
        )
        weights = config.weights
        intercept = config.intercept
        fields = config.fields
        if string_distance_algorithm is None:
            string_distance_algorithm = self.default_string_distance_algorithm

        kwargs = dict(kwargs)
        kwargs["string_distance_algorithm"] = string_distance_algorithm

        threshold = settings.decision_threshold

        early_stop_reason_str = self.should_early_stop(record_a, record_b)
        if early_stop_reason_str is not None:
            try:
                early_stop_reason = EarlyStopReason(early_stop_reason_str)
            except ValueError:
                early_stop_reason = None
            return PairScoreResult(
                probability=0.0,
                doi_mismatch_adjustment_applied=False,
                field_results={},
                early_stop_reason=early_stop_reason,
                label=PairLabel.NOT_DUPLICATE,
            )

        field_results: dict[str, FieldResult] = {}
        weighted_total = 0.0

        # Determine which fields to use
        if fields is not None:
            fields_to_use = [(f, weights[f]) for f in fields if f in weights]
        else:
            fields_to_use = list(weights.items())

        for field_name, weight in fields_to_use:
            val_a = getattr(record_a, field_name, None)
            val_b = getattr(record_b, field_name, None)

            if val_a is None and val_b is None:
                field_results[field_name] = FieldResult(status=FieldStatus.MISSING_BOTH)
                continue

            if val_a is None:
                field_results[field_name] = FieldResult(
                    status=FieldStatus.MISSING_A,
                    value_b=str(val_b),
                )
                continue

            if val_b is None:
                field_results[field_name] = FieldResult(
                    status=FieldStatus.MISSING_B,
                    value_a=str(val_a),
                )
                continue

            compare_method = getattr(self, f"compare_{field_name}", None)
            if compare_method is None:
                logger.warning(f"No compare method for field: {field_name}")
                continue

            try:
                match_score = compare_method(record_a, record_b, **kwargs)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error comparing {field_name}: {e}")
                continue

            field_results[field_name] = FieldResult(
                status=FieldStatus.COMPARED,
                value_a=str(val_a),
                value_b=str(val_b),
                score=match_score,
            )
            weighted_total += match_score * weight
            logger.debug(
                f"{field_name}: match_score={match_score:.4f}, weight={weight}, "
                f"weighted={match_score * weight:.4f}"
            )

        raw_score = weighted_total + intercept
        logger.debug(f"Raw score (before sigmoid): {raw_score:.4f}")
        probability = 1 / (1 + exp(-raw_score))
        logger.debug(f"Weighted dedup probability: {probability:.4f}")

        # Apply DOI mismatch penalty if both DOIs are present and do not match
        doi_a = getattr(record_a, "doi", None)
        doi_b = getattr(record_b, "doi", None)
        doi_mismatch_adjustment_applied = False
        if doi_a is not None and doi_b is not None:
            norm_a = normalise_doi(getattr(doi_a, "identifier", None))
            norm_b = normalise_doi(getattr(doi_b, "identifier", None))
            if norm_a and norm_b and norm_a != norm_b:
                doi_mismatch_adjustment_applied = True
                probability = probability * 0.9
                logger.debug(
                    f"DOI mismatch penalty applied: {norm_a} != {norm_b}, factor=0.9"
                )

        label = PairLabel.DUPLICATE if probability >= threshold else PairLabel.NOT_DUPLICATE

        return PairScoreResult(
            probability=probability,
            doi_mismatch_adjustment_applied=doi_mismatch_adjustment_applied,
            field_results=field_results,
            early_stop_reason=None,
            label=label,
        )

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

        if self.should_early_stop(record_a, record_b) is not None:
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

        if self.should_early_stop(record_a, record_b) is not None:
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
            norm_a = normalise_doi(getattr(doi_a, "identifier", None))
            norm_b = normalise_doi(getattr(doi_b, "identifier", None))
            if norm_a and norm_b and norm_a != norm_b:
                penalty_factor = 0.9
                logger.debug(
                    f"DOI mismatch penalty applied: {norm_a} != {norm_b}, factor={penalty_factor}"
                )
        return probability * penalty_factor

    def should_early_stop(
        self,
        record_a: Paper,
        record_b: Paper,
        rules: list[EarlyStopRule] = EARLY_STOP_RULES,
    ) -> str | None:
        """Return the early-stop reason string if the pair should be vetoed, else None."""
        ctx = ComparisonContext.model_construct(
            deduper=self,
            record_a=record_a,
            record_b=record_b,
        )

        for rule in rules:
            if rule.check(ctx):
                return rule.reason
        logger.info("No early stopping reason detected.")
        return None

    def has_abstract_conflict(self, record_a: Paper, record_b: Paper) -> bool:
        """Return True when abstracts are present and show conflicting content."""
        abstract_a = getattr(record_a, "abstract", None)
        abstract_b = getattr(record_b, "abstract", None)

        if not abstract_a or not abstract_b:
            return False

        # In the select year-gap subgroup, abstract disagreement is treated as
        # strong evidence against a duplicate.
        abstract_sim = self.compare_abstract(record_a, record_b)
        if abstract_sim < ABSTRACT_SIMILARITY_THRESHOLD:
            return True

        nums_a = extract_abstract_numbers(abstract_a)
        nums_b = extract_abstract_numbers(abstract_b)

        # If neither abstract has numbers, similarity check above decides.
        if not nums_a and not nums_b:
            return False

        return nums_a != nums_b

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
        doi_a = normalise_doi(getattr(record_a.doi, "identifier", None))
        doi_b = normalise_doi(getattr(record_b.doi, "identifier", None))

        if not doi_a or not doi_b:
            return 0.0
        if doi_a == doi_b:
            return 1.0
        # Fallback: strip punctuation from suffix to catch variants like
        # missing dots (100B3.BJJ vs 100B3BJJ) or _ vs - substitution.
        if strip_doi_punctuation(doi_a) == strip_doi_punctuation(doi_b):
            return 1.0
        return 0.0

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
        isbn_a = normalise_isbn(record_a.isbn)
        isbn_b = normalise_isbn(record_b.isbn)

        if not isbn_a or not isbn_b:
            return 0.0
        return 1.0 if isbn_a == isbn_b else 0.0

    def compare_authors(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare two lists of authors.

        Args:
            record_a: First paper to compare
            record_b: Second paper to compare
            **kwargs: Additional arguments

        Returns:
            float: Similarity score between 0 and 1

        """
        if not record_a.authors or not record_b.authors:
            logger.warning("One or both records have no authors.")
            return 0.0

        try:
            authors_a = ", ".join(
                [
                    str(
                        getattr(a, "author_name", None)
                        or getattr(a, "display_name", "")
                    )
                    for a in record_a.authors
                    if a is not None
                ]
            )
            authors_b = ", ".join(
                [
                    str(
                        getattr(a, "author_name", None)
                        or getattr(a, "display_name", "")
                    )
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
        Compare two titles using Levenshtein distance (case-insensitive),
        with special handling for alternative-language or subtitle segments.
        Only use segment-aware logic if the alternative
        segment contains a known language name.
        Returns a float between 0 and 1.
        """
        # Return 0 if either title is missing
        if not record_a.title or not record_b.title:
            return 0.0

        a_full = record_a.title.strip().lower()
        b_full = record_b.title.strip().lower()

        # Always compute full-title similarity
        full_sim = Deduper.calculate_string_distance(
            a_full,
            b_full,
            string_distance_algorithm=StringDistanceAlgorithm.LEVENSHTEIN,
        )

        # Segment-aware logic: only if alternative segment contains a language name
        a_first, a_alt = split_title_segments(a_full)
        b_first, b_alt = split_title_segments(b_full)

        use_segment = False
        # Only use segment logic if the alternative segment contains a language name
        if (a_alt and contains_language_name(a_alt)) or (
            b_alt and contains_language_name(b_alt)
        ):
            use_segment = True

        if use_segment:
            seg_sim = Deduper.calculate_string_distance(
                a_first,
                b_first,
                string_distance_algorithm=StringDistanceAlgorithm.LEVENSHTEIN,
            )
            return max(full_sim, seg_sim)

        return full_sim

    def compare_titles(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """Backward-compatible alias for compare_title."""
        return self.compare_title(record_a, record_b, **kwargs)

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

        score = Deduper.calculate_string_distance(
            record_a.journal, record_b.journal, string_distance_algorithm=algo
        )

        # If the raw similarity is low, check whether one name is an abbreviation
        # of the other (e.g. "Cmaj" vs "Canadian Medical Association Journal").
        # When an abbreviation match is detected, return a fixed score of 0.85 so
        # downstream veto rules don't incorrectly flag these as mismatches.
        if score < JOURNAL_ABBREVIATION_THRESHOLD and is_journal_abbreviation_match(
            record_a.journal, record_b.journal
        ):
            return 0.85

        return score

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

        pages_a = normalise_pages(record_a.pages)
        pages_b = normalise_pages(record_b.pages)
        return float(pages_a == pages_b)

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
            return re.sub(r"\s+", " ", text).strip()

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
