"""Tests for extractor.section_extractor."""

import pytest
from extractor.section_extractor import extract_sections


PAPER_WITH_DA = """
Abstract

We studied gene expression.

Methods

Cells were grown under standard conditions.

Data Availability

All raw sequencing data are deposited in GEO under accession GSE12345.
The processed data are available at https://doi.org/10.5281/zenodo.9999999.

References

1. Smith et al. Nature 2020.
2. Doe et al. Cell 2021.
"""

PAPER_NO_DA_SECTION = """
Introduction

Genome-wide studies rely on large datasets.

Methods

We used publicly available data from GEO (GSE99999) and SRA (SRR1234567).
The code is at https://github.com/lab/analysis.

Conclusion

Our results are significant.
"""

PAPER_DA_STATEMENT_HEADING = """
Results

New findings here.

Data Availability Statement

Data are available upon reasonable request and have been deposited at
https://doi.org/10.6084/m9.figshare.12345678.

References

Jones et al. 2019.
"""


class TestExtractSections:
    def test_detects_da_section(self):
        result = extract_sections(PAPER_WITH_DA)
        assert "GSE12345" in result["data_availability"]

    def test_detects_references_section(self):
        result = extract_sections(PAPER_WITH_DA)
        assert "Smith et al" in result["references"]

    def test_fallback_keyword_detection(self):
        result = extract_sections(PAPER_NO_DA_SECTION)
        da = result["data_availability"]
        # Should pick up the paragraph mentioning GEO/SRA/github
        assert "GSE99999" in da or "SRR1234567" in da or "github" in da.lower()

    def test_da_statement_heading(self):
        result = extract_sections(PAPER_DA_STATEMENT_HEADING)
        assert "figshare" in result["data_availability"].lower()

    def test_missing_section_is_empty_string(self):
        result = extract_sections("Just some random text with no sections.")
        assert result["references"] == ""
