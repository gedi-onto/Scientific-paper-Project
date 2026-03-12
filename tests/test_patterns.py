"""Tests for extractor.patterns — verify that each regex compiles and fires."""

import pytest
from extractor.patterns import PATTERNS, _DOI, _GEO, _SRA, _BIOSAMPLE, _BIOPROJECT


def _match(pattern, text):
    return pattern.search(text) is not None


# ---------------------------------------------------------------------------
# DOI
# ---------------------------------------------------------------------------
class TestDOIPattern:
    def test_url_form(self):
        assert _match(_DOI, "https://doi.org/10.1038/nature12345")

    def test_dx_doi_form(self):
        assert _match(_DOI, "http://dx.doi.org/10.1016/j.cell.2020.01.001")

    def test_doi_colon_form(self):
        assert _match(_DOI, "doi: 10.1093/nar/gkz123")

    def test_no_match_on_plain_url(self):
        assert not _match(_DOI, "https://www.nature.com/articles/s41586-020-1234-5")


# ---------------------------------------------------------------------------
# GEO
# ---------------------------------------------------------------------------
class TestGEOPattern:
    def test_gse(self):
        assert _match(_GEO, "GSE12345")

    def test_gsm(self):
        assert _match(_GEO, "GSM9876543")

    def test_no_partial(self):
        assert not _match(_GEO, "ABCGSE12")


# ---------------------------------------------------------------------------
# SRA
# ---------------------------------------------------------------------------
class TestSRAPattern:
    def test_srr(self):
        assert _match(_SRA, "SRR1234567")

    def test_srp(self):
        assert _match(_SRA, "SRP000001")

    def test_err(self):
        assert _match(_SRA, "ERR123456")


# ---------------------------------------------------------------------------
# BioSample / BioProject
# ---------------------------------------------------------------------------
class TestBioPatterns:
    def test_biosample_samn(self):
        assert _match(_BIOSAMPLE, "SAMN01234567")

    def test_bioproject_prjna(self):
        assert _match(_BIOPROJECT, "PRJNA123456")


# ---------------------------------------------------------------------------
# All labels are unique in PATTERNS
# ---------------------------------------------------------------------------
def test_pattern_labels_unique():
    labels = [label for label, _ in PATTERNS]
    assert len(labels) == len(set(labels))
