"""
SemanticDataAggregator — the top-level entry point for the pipeline.

Usage example
-------------
>>> from extractor import SemanticDataAggregator
>>> agg = SemanticDataAggregator()
>>> results = agg.aggregate(paper_text)
>>> for ref in results["references"]:
...     print(ref["type"], ref["identifier"])
"""

from __future__ import annotations

from extractor.section_extractor import extract_sections
from extractor.reference_extractor import extract_references
from extractor.normalizer import normalize
from extractor.deduplicator import deduplicate


class SemanticDataAggregator:
    """Aggregate dataset references from the text of a scientific paper.

    Parameters
    ----------
    include_sections:
        Which sections to search.  Defaults to both ``"data_availability"``
        and ``"references"``.  Supply a subset to restrict the search.
    """

    DEFAULT_SECTIONS = ("data_availability", "references")

    def __init__(
        self,
        include_sections: tuple[str, ...] | None = None,
    ) -> None:
        self._sections = include_sections or self.DEFAULT_SECTIONS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, text: str) -> dict:
        """Run the full extraction pipeline on *text*.

        Parameters
        ----------
        text:
            Plain-text content of a scientific paper.

        Returns
        -------
        dict with keys:
            ``sections``   – raw text of each extracted section
            ``references`` – list of deduplicated, normalized reference dicts
        """
        sections = extract_sections(text)

        raw_hits: list[dict] = []
        for section_name in self._sections:
            section_text = sections.get(section_name, "")
            if section_text:
                for hit in extract_references(section_text):
                    hit["source_section"] = section_name
                    raw_hits.append(hit)

        normalized = [normalize(h) for h in raw_hits]
        unique = deduplicate(normalized)

        return {
            "sections": {k: sections[k] for k in self._sections if k in sections},
            "references": unique,
        }
