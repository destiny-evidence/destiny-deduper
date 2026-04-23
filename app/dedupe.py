"""Deduplication workflow, in main class `Deduper` with various algorithms."""

from enum import StrEnum, auto
from math import exp
from typing import Literal

import jellyfish
from loguru import logger

from app.data_models import Paper

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
        logger.info(f"fields to compare: {', '.join(fields_to_compare)}")

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

        logger.info(f"scores for each field: {scores}")
        return sum(scores.values()) / len(scores)

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
            logger.info(
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

        # early stopping: doi and pages.
        if (
            record_a.doi is not None
            and record_b.doi is not None
            and record_a.pages is not None
            and record_b.pages is not None
        ):
            doi_match = record_a.doi.identifier == record_b.doi.identifier
            pages_match = record_a.pages == record_b.pages

            # if doi and pages between 2 records don't match, stop
            if not doi_match and not pages_match:
                logger.info("Early stop: doi and pages both don't match")
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
        logger.info(f"Unweighted dedup score: {mean_score:.4f}")

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

        # check doi & pages to stop early if required
        if (
            record_a.doi is not None
            and record_b.doi is not None
            and record_a.pages is not None
            and record_b.pages is not None
        ):
            doi_match = record_a.doi.identifier == record_b.doi.identifier
            pages_match = record_a.pages == record_b.pages

            if not doi_match and not pages_match:
                logger.info("Early stop: DOI and PAGES both don't match")
                return 0.0

        weighted_scores = {}
        for field, weight in weights.items():
            # check if both records have field from weights
            logger.debug(f"field: {field}, weight val: {weight}")
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
                logger.debug(
                    f"{field}: match_score={match_score:.4f}, "
                    f"weight={weight}, weighted={weighted_score:.4f}"
                )
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
        logger.info(f"Weighted dedup probability: {probability:.4f}")

        return probability

    def compare_doi(
        self,
        record_a: Paper,
        record_b: Paper,
        method: Literal["string_match", "http", "both"] = "string_match",
        **kwargs,
    ) -> float:
        """
        Compare 2 dois.

        NOTE: `http` might be a method where we follow the url (requests?) for
        doi_a and doi_b and check if they're the same (maybe string distance on
        the resulting html?).

        Args:
            record_a (Paper): a Paper instance.
            record_b (Paper): a Paper instance.
            method (Literal[&quot;string_match&quot;, &quot;http&quot;, &quot;both&quot;], optional): _description_. Defaults to "string_match".

        Raises:
            NotImplementedError: for methods that haven't been implemented yet.

        Returns:
            float: between 0 and 1

        """
        if record_a.doi is None or record_b.doi is None:
            return 0.0
        doi_a = record_a.doi.identifier  # should have doi url removed
        doi_b = record_b.doi.identifier

        if method == "string_match":
            # NOTE - does this even make sense? i think this is what @kaitlynhair code is doing
            # but does approximate string similarity imply similarity of the
            # underlying record, or should we just do a == b comparison?
            return Deduper.calculate_string_distance(doi_a, doi_b, **kwargs)

        not_implemented_err_msg = f"method {method} is not yet implemented."
        raise NotImplementedError(not_implemented_err_msg)

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

    def compare_isbn(self, record_a: Paper, record_b: Paper) -> float:
        """
        Compare 2 isbns.

        Args:
            record_a (Paper): a Paper instance.
            record_b (Paper): a Paper instance.

        Returns:
            float: 1 if equal, 0 if not.

        """
        if record_a.isbn is None or record_b.isbn is None:
            return 0.0

        isbn_a = record_a.isbn
        isbn_b = record_b.isbn
        return float(isbn_a == isbn_b)

    def compare_authors(
        self,
        record_a: Paper,
        record_b: Paper,
        **kwargs,
    ) -> float:
        """
        Compare two authors.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper.

        Returns:
            float: between 0 and 1

        """
        not_impl = "method `compare_authors` is not yet implemented"
        raise NotImplementedError(not_impl)

    def compare_title(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare two titles using string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.title is None or record_b.title is None:
            return 0.0

        title_a = record_a.title.lower().strip()
        title_b = record_b.title.lower().strip()

        return Deduper.calculate_string_distance(title_a, title_b, **kwargs)

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
        Compare two journal names using string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.journal is None or record_b.journal is None:
            return 0.0

        journal_a = record_a.journal.lower().strip()
        journal_b = record_b.journal.lower().strip()

        return Deduper.calculate_string_distance(journal_a, journal_b, **kwargs)

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
        Returns 1.0 if both page ranges match exactly, 0.0 otherwise.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: 1.0 if match, 0.0 otherwise

        """
        if record_a.pages is None or record_b.pages is None:
            return 0.0

        return float(record_a.pages == record_b.pages)

    def compare_abstract(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 abstract strings using string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.abstract is None or record_b.abstract is None:
            return 0.0

        abstract_a = record_a.abstract.lower().strip()
        abstract_b = record_b.abstract.lower().strip()

        return Deduper.calculate_string_distance(abstract_a, abstract_b, **kwargs)

    def compare_volume(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 volume strings.
        Returns 1.0 if exact match, otherwise uses string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.volume is None or record_b.volume is None:
            return 0.0

        volume_a = record_a.volume.strip()
        volume_b = record_b.volume.strip()

        if volume_a == volume_b:
            return 1.0

        return Deduper.calculate_string_distance(volume_a, volume_b, **kwargs)

    def compare_issue(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 issue strings.
        Returns 1.0 if exact match, otherwise uses string distance.

        Args:
            record_a (Paper): a paper
            record_b (Paper): a paper

        Returns:
            float: between 0 and 1

        """
        if record_a.issue is None or record_b.issue is None:
            return 0.0

        issue_a = record_a.issue.strip()
        issue_b = record_b.issue.strip()

        if issue_a == issue_b:
            return 1.0

        return Deduper.calculate_string_distance(issue_a, issue_b, **kwargs)

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
        string_a: str, string_b: str, string_distance_algorithm: StringDistanceAlgorithm
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
        Here: a wrapper around jellyfish library, but
        we could easily implement this ourselves.

        Args:
            string_a (str): a string.
            string_b (str): another string.

        Returns:
            float: between 0 and 1.

        """
        return jellyfish.jaro_winkler_similarity(string_a, string_b)

    @staticmethod
    def levenshtein_distance(string_a: str, string_b: str) -> float:
        """
        Calculate Levenshtein distance b/w 2 strings.
        Here: a wrapper around jellyfish library, but
        we could easily implement this ourselves.

        Args:
            string_a (str): a string
            string_b (str): a string

        Returns:
            float: between 0 and 1.

        """
        return jellyfish.levenshtein_distance(string_a, string_b)
