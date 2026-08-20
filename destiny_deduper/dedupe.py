"""Deduplication workflow, in main class `Deduper` with various algorithms."""

from enum import StrEnum, auto
from math import exp
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz.distance import JaroWinkler as _JaroWinkler
from rapidfuzz.distance import Levenshtein as _Levenshtein

from destiny_deduper.config import get_settings
from destiny_deduper.data_models import Paper
from destiny_deduper.early_stop import (
    EARLY_STOP_RULES,
    ComparisonContext,
    EarlyStopRule,
)
from destiny_deduper.logger import logger
from destiny_deduper.normalisers import (
    normalise_doi,
    normalise_isbn,
    normalise_pages,
    strip_doi_punctuation,
)
from destiny_deduper.pair_score_result import (
    ComparisonOutput,
    EarlyStopReason,
    FieldResult,
    FieldStatus,
    PairLabel,
    PairScoreResult,
)
from destiny_deduper.utils import (
    contains_language_name,
    is_journal_abbreviation_match,
    split_title_segments,
)

settings = get_settings()

# constants
WEIGHTS = settings.weights.model_dump()
INTERCEPT: float = settings.weights.intercept
DOI_MISMATCH_PENALTY = settings.doi_mismatch_penalty
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
                comparison = compare_method(record_a, record_b, **kwargs)
                scores[field] = comparison.score
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
        """Score a pair and return a structured result."""
        config = config or ScorePairConfig()
        kwargs = self._prepare_score_pair_kwargs(config, kwargs)

        early_stop_result = self._build_early_stop_result(record_a, record_b)
        if early_stop_result is not None:
            return early_stop_result

        fields_to_use = self._resolve_score_fields(config.weights, config.fields)
        field_results, weighted_total = self._score_pair_fields(
            record_a=record_a,
            record_b=record_b,
            fields_to_use=fields_to_use,
            kwargs=kwargs,
        )

        if not self._has_compared_fields(field_results):
            return self._build_unscorable_result(field_results)

        probability = self._calculate_probability(weighted_total, config.intercept)
        probability, doi_mismatch_adjustment_applied = self._apply_doi_mismatch_penalty(
            record_a,
            record_b,
            probability,
        )

        threshold = settings.decision_threshold
        label = (
            PairLabel.DUPLICATE if probability >= threshold else PairLabel.NOT_DUPLICATE
        )

        return PairScoreResult(
            probability=probability,
            doi_mismatch_adjustment_applied=doi_mismatch_adjustment_applied,
            field_results=field_results,
            early_stop_reason=None,
            label=label,
        )

    def _prepare_score_pair_kwargs(
        self,
        config: ScorePairConfig,
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        """Prepare kwargs for score_pair, extracting string distance algo."""
        string_distance_algorithm = (
            config.string_distance_algorithm or self.default_string_distance_algorithm
        )
        if string_distance_algorithm is None:
            string_distance_algorithm = self.default_string_distance_algorithm

        merged_kwargs = dict(kwargs)
        merged_kwargs["string_distance_algorithm"] = string_distance_algorithm
        return merged_kwargs

    def _build_early_stop_result(
        self,
        record_a: Paper,
        record_b: Paper,
    ) -> PairScoreResult | None:
        """Return early-stop result, with reason as PairScoreResult, or None if n/a."""
        early_stop_reason_str = self.should_early_stop(record_a, record_b)
        if early_stop_reason_str is None:
            return None

        try:
            early_stop_reason = EarlyStopReason(early_stop_reason_str)
        except ValueError:
            logger.warning(
                f"Unknown early-stop reason {early_stop_reason_str!r}; "
                "EarlyStopReason enum may be out of sync with early_stop.py"
            )
            early_stop_reason = None

        return PairScoreResult(
            probability=0.0,
            doi_mismatch_adjustment_applied=False,
            field_results={},
            early_stop_reason=early_stop_reason,
            label=PairLabel.NOT_DUPLICATE,
        )

    @staticmethod
    def _resolve_score_fields(
        weights: dict[str, float],
        fields: list[str] | None,
    ) -> list[tuple[str, float]]:
        """Resolve score fields."""
        if fields is None:
            return [
                (field_name, weight)
                for field_name, weight in weights.items()
                if field_name != "intercept"
            ]
        return [
            (field_name, weights[field_name])
            for field_name in fields
            if field_name in weights and field_name != "intercept"
        ]

    def _score_pair_fields(
        self,
        record_a: Paper,
        record_b: Paper,
        fields_to_use: list[tuple[str, float]],
        kwargs: dict[str, object],
    ) -> tuple[dict[str, FieldResult], float]:
        """Produce weighted field-level score."""
        field_results: dict[str, FieldResult] = {}
        weighted_total = 0.0

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
                comparison = compare_method(record_a, record_b, **kwargs)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error comparing {field_name}: {e}")
                continue

            field_results[field_name] = FieldResult(
                status=FieldStatus.COMPARED,
                value_a=str(val_a),
                value_b=str(val_b),
                normalised_value_a=comparison.normalised_value_a,
                normalised_value_b=comparison.normalised_value_b,
                score=comparison.score,
            )
            weighted_total += comparison.score * weight
            logger.debug(
                f"{field_name}: match_score={comparison.score:.4f}, weight={weight}, "
                f"weighted={comparison.score * weight:.4f}"
            )

        return field_results, weighted_total

    @staticmethod
    def _has_compared_fields(field_results: dict[str, FieldResult]) -> bool:
        """Check if a given field has been compared."""
        return any(fr.status == FieldStatus.COMPARED for fr in field_results.values())

    @staticmethod
    def _build_unscorable_result(
        field_results: dict[str, FieldResult],
    ) -> PairScoreResult:
        """Return a PairScoreResult object which was unscorable."""
        return PairScoreResult(
            probability=0.0,
            doi_mismatch_adjustment_applied=False,
            field_results=field_results,
            early_stop_reason=None,
            label=PairLabel.UNSCORABLE,
            unscorable_reason="no_comparable_fields",
        )

    @staticmethod
    def _calculate_probability(weighted_total: float, intercept: float) -> float:
        """Calculate weighted probability."""
        raw_score = weighted_total + intercept
        logger.debug(f"raw score (before sigmoid): {raw_score:.4f}")
        probability = 1 / (1 + exp(-raw_score))
        logger.debug(f"weighted dedupe probability: {probability:.4f}")
        return probability

    @staticmethod
    def _apply_doi_mismatch_penalty(
        record_a: Paper,
        record_b: Paper,
        probability: float,
        doi_mismatch_penalty: float = DOI_MISMATCH_PENALTY,
    ) -> tuple[float, bool]:
        """Apply the doi mismatch penalty, given 2 records and probability."""
        doi_a = getattr(record_a, "doi", None)
        doi_b = getattr(record_b, "doi", None)
        if doi_a is None or doi_b is None:
            return probability, False

        norm_a = normalise_doi(getattr(doi_a, "identifier", None))
        norm_b = normalise_doi(getattr(doi_b, "identifier", None))
        if norm_a and norm_b and norm_a != norm_b:
            logger.debug(
                f"DOI mismatch penalty applied: {norm_a} != {norm_b}, factor={doi_mismatch_penalty}"
            )
            return probability * doi_mismatch_penalty, True

        return probability, False

    @staticmethod
    def _comparison_output(
        normalised_value_a: str,
        normalised_value_b: str,
        score: float,
    ) -> ComparisonOutput:
        """Build a structured result from normalized values and a score."""
        return ComparisonOutput(
            normalised_value_a=normalised_value_a,
            normalised_value_b=normalised_value_b,
            score=score,
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
                comparison = compare_method(record_a, record_b, **kwargs)
                scores[field] = comparison.score
                logger.debug(f"Score for {field}: {comparison.score}")
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
            if field == "intercept":
                continue

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
                comparison = compare_method(record_a, record_b, **kwargs)
                weighted_score = comparison.score * weight
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

    def compare_doi(
        self,
        record_a: Paper,
        record_b: Paper,
        **kwargs,
    ) -> ComparisonOutput:
        """
        Compare DOIs between two records (exact match after normalization).

        Returns:
            float: 1.0 if DOIs match, 0.0 otherwise

        """
        doi_a = normalise_doi(getattr(record_a.doi, "identifier", None)) or ""
        doi_b = normalise_doi(getattr(record_b.doi, "identifier", None)) or ""

        if not doi_a or not doi_b:
            score = 0.0
        elif doi_a == doi_b or strip_doi_punctuation(doi_a) == strip_doi_punctuation(
            doi_b
        ):
            score = 1.0
        else:
            score = 0.0
        return self._comparison_output(doi_a, doi_b, score)

    def compare_openalex_id(
        self,
        record_a: Paper,
        record_b: Paper,
        method: Literal["string_match", "http", "both"] = "string_match",
        **kwargs,
    ) -> ComparisonOutput:
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
        openalex_id_a_str = (
            str(record_a.openalex_id.identifier) if record_a.openalex_id else ""
        )
        openalex_id_b_str = (
            str(record_b.openalex_id.identifier) if record_b.openalex_id else ""
        )

        if method == "string_match":
            # NOTE - does this even make sense? i think this is what @kaitlynhair code is doing
            # but does approximate string similarity imply similarity of the
            # underlying record, or should we just do a == b comparison?
            score = (
                Deduper.calculate_string_distance(
                    openalex_id_a_str, openalex_id_b_str, **kwargs
                )
                if openalex_id_a_str and openalex_id_b_str
                else 0.0
            )
            return self._comparison_output(openalex_id_a_str, openalex_id_b_str, score)

        not_implemented_error = f"method {method} is not yet implemented."
        raise NotImplementedError(not_implemented_error)

    def compare_pubmed_id(
        self,
        record_a: Paper,
        record_b: Paper,
        **kwargs,
    ) -> ComparisonOutput:
        """
        Compare two pubmed ids.

        Args:
            record_a (Paper): a Paper instance.
            record_b (Paper): a Paper instance.
            string_distance_algorithm (_type_, optional): _description_. Defaults to StringDistanceAlgorithm | str | None=None.

        Returns:
            float: _description_

        """
        pubmed_id_a_str = (
            str(record_a.pubmed_id.identifier) if record_a.pubmed_id else ""
        )
        pubmed_id_b_str = (
            str(record_b.pubmed_id.identifier) if record_b.pubmed_id else ""
        )
        return self._comparison_output(
            pubmed_id_a_str,
            pubmed_id_b_str,
            float(
                bool(
                    pubmed_id_a_str
                    and pubmed_id_b_str
                    and pubmed_id_a_str == pubmed_id_b_str
                )
            ),
        )

    def compare_isbn(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """
        Compare 2 ISBNs after normalization.

        Returns:
            float: 1.0 if equal, 0.0 otherwise.

        """
        isbn_a = normalise_isbn(str(record_a.isbn)) if record_a.isbn else ""
        isbn_b = normalise_isbn(str(record_b.isbn)) if record_b.isbn else ""
        return self._comparison_output(
            isbn_a or "",
            isbn_b or "",
            float(bool(isbn_a and isbn_b and isbn_a == isbn_b)),
        )

    def compare_authors(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
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
            return self._comparison_output("", "", 0.0)

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
            return self._comparison_output("", "", 0.0)

        # Return 0 if either authors list is empty
        if not authors_a or not authors_b:
            logger.warning("One or both authors lists are empty after extraction.")
            return self._comparison_output(authors_a, authors_b, 0.0)

        # Allow override algorithm via kwargs, fallback to Jaro-Winkler
        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )
        return self._comparison_output(
            authors_a,
            authors_b,
            Deduper.calculate_string_distance(
                authors_a, authors_b, string_distance_algorithm=algo
            ),
        )

    def compare_title(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """
        Compare two titles using Levenshtein distance (case-insensitive),
        with special handling for alternative-language or subtitle segments.
        Only use segment-aware logic if the alternative
        segment contains a known language name.
        Returns a float between 0 and 1.
        """
        # Return 0 if either title is missing
        if not record_a.title or not record_b.title:
            return self._comparison_output(
                (record_a.title or "").strip().lower(),
                (record_b.title or "").strip().lower(),
                0.0,
            )

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
            full_sim = max(full_sim, seg_sim)

        return self._comparison_output(a_full, b_full, full_sim)

    def compare_titles(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """Backward-compatible alias for compare_title."""
        return self.compare_title(record_a, record_b, **kwargs)

    def compare_year(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
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
            return self._comparison_output(
                "" if record_a.year is None else str(record_a.year),
                "" if record_b.year is None else str(record_b.year),
                0.0,
            )

        year_a = record_a.year
        year_b = record_b.year
        return self._comparison_output(
            str(year_a), str(year_b), float(year_a == year_b)
        )

    def compare_journal(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """
        Compare two journal names using Jaro-Winkler distance.
        Returns a float between 0 and 1.
        """
        # Return 0 if either journal is missing
        if not record_a.journal or not record_b.journal:
            return self._comparison_output(
                (record_a.journal or "").strip(),
                (record_b.journal or "").strip(),
                0.0,
            )

        # Allow override algorithm via kwargs, fallback to Jaro-Winkler
        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )

        journal_a = record_a.journal.strip()
        journal_b = record_b.journal.strip()

        score = Deduper.calculate_string_distance(
            journal_a, journal_b, string_distance_algorithm=algo
        )

        # If the raw similarity is low, check whether one name is an abbreviation
        # of the other (e.g. "Cmaj" vs "Canadian Medical Association Journal").
        # When an abbreviation match is detected, return a fixed score of 0.85 so
        # downstream veto rules don't incorrectly flag these as mismatches.
        if score < JOURNAL_ABBREVIATION_THRESHOLD and is_journal_abbreviation_match(
            journal_a, journal_b
        ):
            score = 0.85

        return self._comparison_output(journal_a, journal_b, score)

    def compare_publisher(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """
        Compare two publisher names using string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.publisher is None or record_b.publisher is None:
            return self._comparison_output(
                (record_a.publisher or "").strip().lower(),
                (record_b.publisher or "").strip().lower(),
                0.0,
            )

        publisher_a = record_a.publisher.lower().strip()
        publisher_b = record_b.publisher.lower().strip()

        return self._comparison_output(
            publisher_a,
            publisher_b,
            Deduper.calculate_string_distance(publisher_a, publisher_b, **kwargs),
        )

    def compare_pages(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
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
            return self._comparison_output(
                normalise_pages(record_a.pages) or "",
                normalise_pages(record_b.pages) or "",
                0.0,
            )

        pages_a = normalise_pages(record_a.pages)
        pages_b = normalise_pages(record_b.pages)
        return self._comparison_output(
            pages_a or "",
            pages_b or "",
            float(bool(pages_a and pages_b and pages_a == pages_b)),
        )

    def compare_volume(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """
        Compare 2 sets of volumes,
        using Jaro-Winkler distance.
        """
        if not record_a.volume or not record_b.volume:
            return self._comparison_output(
                record_a.volume or "", record_b.volume or "", 0.0
            )

        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )
        return self._comparison_output(
            record_a.volume,
            record_b.volume,
            Deduper.calculate_string_distance(
                record_a.volume, record_b.volume, string_distance_algorithm=algo
            ),
        )

    def compare_issue(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
        """
        Compare 2 sets of issues,
        using Jaro-Winkler distance.
        """
        if not record_a.issue or not record_b.issue:
            return self._comparison_output(
                record_a.issue or "", record_b.issue or "", 0.0
            )

        algo = kwargs.get(
            "string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER
        )
        return self._comparison_output(
            record_a.issue,
            record_b.issue,
            Deduper.calculate_string_distance(
                record_a.issue, record_b.issue, string_distance_algorithm=algo
            ),
        )

    def compare_issn(
        self, record_a: Paper, record_b: Paper, **kwargs
    ) -> ComparisonOutput:
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
            return self._comparison_output(
                record_a.issn or "", record_b.issn or "", 0.0
            )

        return self._comparison_output(
            record_a.issn,
            record_b.issn,
            float(record_a.issn == record_b.issn),
        )

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
