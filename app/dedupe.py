"""Deduplication workflow, in main class `Deduper` with various algorithms."""

import re
from enum import StrEnum, auto
from typing import Literal

import jellyfish
from loguru import logger

from app.data_models import Paper


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

    def compare_doi(self, 
                    record_a: Paper,
                    record_b: Paper,
                    **kwargs
    ) -> float:
        """
        Compare DOIs between two records (exact match after normalization).

        Returns:
            float: 1.0 if DOIs match, 0.0 otherwise
        """

        def normalize_doi(doi: str) -> str:
            """Normalize DOI format to consistent lowercase, remove prefixes, decode symbols."""
            if not doi:
                return None
            doi = doi.replace("%28", "(").replace("%29", ")")
            doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
            doi = re.sub(r"^DOI[: ]?", "", doi, flags=re.IGNORECASE)
            match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, flags=re.IGNORECASE)
            if not match:
                return None
            return match.group(0).strip().lower()

        doi_a = normalize_doi(getattr(record_a.doi, "identifier", None))
        doi_b = normalize_doi(getattr(record_b.doi, "identifier", None))

        if not doi_a or not doi_b:
            return 0
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


    def compare_isbn(self, record_a: "Paper", record_b: "Paper", **kwargs) -> float:
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
            authors_a = ", ".join([
                getattr(a, "author_name", None) or getattr(a, "display_name", "")
                for a in record_a.authors if a is not None
            ])
            authors_b = ", ".join([
                getattr(a, "author_name", None) or getattr(a, "display_name", "")
                for a in record_b.authors if a is not None
            ])
        except AttributeError as e:
            logger.error(f"Error extracting authors: {e}")
            return 0.0

        # Return 0 if either authors list is empty
        if not authors_a or not authors_b:
            logger.warning("One or both authors lists are empty after extraction.")
            return 0.0

        # Allow override algorithm via kwargs, fallback to Levenshtein/Jaro-Winkler
        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER)
        return Deduper.calculate_string_distance(authors_a, authors_b, string_distance_algorithm=algo)

    def compare_title(
            self, 
            record_a: Paper, 
            record_b: Paper, 
            **kwargs
        ) -> float:
        """
        Compare two titles using Levenshtein distance.
        Returns a float between 0 and 1.
        """
        # Return 0 if either title is missing
        if not record_a.title or not record_b.title:
            return 0.0

        # Allow override algorithm via kwargs, fallback to Levenshtein
        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.LEVENSHTEIN)

        return Deduper.calculate_string_distance(record_a.title, record_b.title, string_distance_algorithm=algo)

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
        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER)

        return Deduper.calculate_string_distance(record_a.journal, record_b.journal, string_distance_algorithm=algo)

    def compare_publisher(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare two publishers names.

        Args:
            record_a (Paper): _description_
            record_b (Paper): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            float: _description_

        """
        not_impl = "method `compare_publisher` is not yet implemented"
        raise NotImplementedError(not_impl)

    def compare_pages(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 sets of page delineations, e.g. (123, 130) vs (123, 136),
        or string ranges like "123-130" vs "123-136".
        Uses Jaro-Winkler distance after normalization.
        """

        def normalize_pages(page_range: str | tuple[int, int] | None) -> str | None:
            if page_range is None:
                return None

            # If it's a tuple, convert to "start-end" string
            if isinstance(page_range, tuple) and len(page_range) == 2:
                return f"{page_range[0]}-{page_range[1]}"

            # Otherwise assume string
            if isinstance(page_range, str):
                page_range = re.sub(r"[–—−]", "-", page_range)
                page_range = page_range.strip()
                parts = page_range.split("-")
                if len(parts) != 2:
                    return page_range
                start, end = parts[0].strip(), parts[1].strip()
                if len(end) < len(start):
                    prefix_len = len(start) - len(end)
                    prefix = start[:prefix_len]
                    end = prefix + end
                return f"{start}-{end}"

            # Fallback
            return None

        if not record_a.pages or not record_b.pages:
            return 0.0

        pages_a = normalize_pages(record_a.pages)
        pages_b = normalize_pages(record_b.pages)

        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER)
        return Deduper.calculate_string_distance(pages_a, pages_b, string_distance_algorithm=algo)


    def compare_volume(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 sets of volumes,
        using Jaro-Winkler distance.
        """
        if not record_a.volume or not record_b.volume:
            return 0.0

        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER)
        return Deduper.calculate_string_distance(record_a.volume, record_b.volume, string_distance_algorithm=algo)

    def compare_issue(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare 2 sets of volumes,
        using Jaro-Winkler distance.
        """
        if not record_a.issue or not record_b.issue:
            return 0.0

        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.JARO_WINKLER)
        return Deduper.calculate_string_distance(record_a.issue, record_b.issue, string_distance_algorithm=algo)

    def compare_abstract(self, record_a: Paper, record_b: Paper, **kwargs) -> float:
        """
        Compare abstracts between two papers using Levenshtein distance.

        Returns:
            float: similarity score between 0 and 1
        """
        if not record_a.abstract or not record_b.abstract:
            return 0.0

        algo = kwargs.get("string_distance_algorithm", StringDistanceAlgorithm.LEVENSHTEIN)
        return Deduper.calculate_string_distance(
            record_a.abstract, record_b.abstract, string_distance_algorithm=algo
        )

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

        # raw edit distance
        dist = jellyfish.levenshtein_distance(a, b)

        # normalize by max string length
        similarity = 1 - (dist / max(len(a), len(b)))
        return max(0.0, min(1.0, similarity))  # clamp between 0 and 1
