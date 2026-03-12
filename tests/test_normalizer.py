"""Tests for extractor.normalizer."""

import pytest
from extractor.normalizer import normalize


def _ref(rtype, identifier):
    return {"type": rtype, "identifier": identifier, "context": "", "start": 0, "end": 0}


class TestNormalize:
    def test_doi_strips_url_prefix(self):
        result = normalize(_ref("DOI", "https://doi.org/10.1038/NATURE12345"))
        assert result["identifier"] == "10.1038/nature12345"

    def test_doi_strips_doi_colon(self):
        result = normalize(_ref("DOI", "doi: 10.1093/NAR/GKZ123"))
        assert result["identifier"] == "10.1093/nar/gkz123"

    def test_doi_strips_trailing_dot(self):
        result = normalize(_ref("DOI", "10.1038/nature12345."))
        assert result["identifier"] == "10.1038/nature12345"

    def test_geo_uppercase(self):
        result = normalize(_ref("GEO", "gse12345"))
        assert result["identifier"] == "GSE12345"

    def test_sra_uppercase(self):
        result = normalize(_ref("SRA", "srr1234567"))
        assert result["identifier"] == "SRR1234567"

    def test_pdb_uppercase(self):
        result = normalize(_ref("PDB", "1abc"))
        assert result["identifier"] == "1ABC"

    def test_dbgap_lowercase(self):
        result = normalize(_ref("dbGaP", "PHS000001.V1.P1"))
        assert result["identifier"] == "phs000001.v1.p1"

    def test_zenodo_extracts_id(self):
        result = normalize(_ref("Zenodo", "https://zenodo.org/record/3456789"))
        assert result["identifier"] == "3456789"

    def test_github_lowercased(self):
        result = normalize(_ref("GitHub", "https://GitHub.com/Lab/Repo"))
        assert result["identifier"] == "https://github.com/lab/repo"

    def test_unknown_type_passthrough(self):
        result = normalize(_ref("Unknown", "SomeValue"))
        assert result["identifier"] == "SomeValue"

    def test_original_not_mutated(self):
        original = _ref("GEO", "gse99999")
        normalize(original)
        assert original["identifier"] == "gse99999"
