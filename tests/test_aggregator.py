"""Integration tests for SemanticDataAggregator."""

import pytest
from extractor import SemanticDataAggregator


SAMPLE_PAPER = """
Abstract

In this study we analyzed gene expression using RNA-seq.

Methods

RNA was extracted using standard methods.

Data Availability

All raw sequencing data have been deposited in GEO under accession GSE123456.
Processed files are available at https://zenodo.org/record/9999999.
Code is available at https://github.com/labname/rnaseq-pipeline.
Additional data are in SRA: SRR9876543.

References

1. Zhang et al. (2020) https://doi.org/10.1038/s41586-020-00001-x
2. Li et al. (2019) doi: 10.1016/j.cell.2019.01.001
3. Zhang et al. (2020) https://doi.org/10.1038/s41586-020-00001-x
"""


class TestSemanticDataAggregator:
    def setup_method(self):
        self.agg = SemanticDataAggregator()
        self.result = self.agg.aggregate(SAMPLE_PAPER)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def test_result_has_sections_key(self):
        assert "sections" in self.result

    def test_result_has_references_key(self):
        assert "references" in self.result

    def test_sections_contain_da(self):
        assert "data_availability" in self.result["sections"]

    # ------------------------------------------------------------------
    # Reference extraction
    # ------------------------------------------------------------------
    def test_finds_geo_accession(self):
        identifiers = [r["identifier"] for r in self.result["references"]]
        assert "GSE123456" in identifiers

    def test_finds_zenodo(self):
        ids = [r["identifier"] for r in self.result["references"]]
        assert "9999999" in ids

    def test_finds_sra(self):
        ids = [r["identifier"] for r in self.result["references"]]
        assert "SRR9876543" in ids

    def test_finds_github(self):
        types = [r["type"] for r in self.result["references"]]
        assert "GitHub" in types

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def test_no_duplicate_dois(self):
        doi_refs = [r for r in self.result["references"] if r["type"] == "DOI"]
        identifiers = [r["identifier"] for r in doi_refs]
        assert len(identifiers) == len(set(identifiers)), "Duplicate DOIs found"

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def test_doi_is_normalized(self):
        doi_refs = [r for r in self.result["references"] if r["type"] == "DOI"]
        for ref in doi_refs:
            assert ref["identifier"].startswith("10."), (
                f"DOI not normalized: {ref['identifier']}"
            )

    # ------------------------------------------------------------------
    # Custom section filter
    # ------------------------------------------------------------------
    def test_da_only_mode(self):
        agg = SemanticDataAggregator(include_sections=("data_availability",))
        result = agg.aggregate(SAMPLE_PAPER)
        assert "references" not in result["sections"]

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    def test_empty_paper(self):
        result = self.agg.aggregate("")
        assert result["references"] == []

    def test_paper_with_no_datasets(self):
        result = self.agg.aggregate("This paper discusses only theory with no data.")
        assert result["references"] == []
