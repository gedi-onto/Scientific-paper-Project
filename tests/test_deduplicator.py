"""Tests for extractor.deduplicator."""

import pytest
from extractor.deduplicator import deduplicate


def _ref(rtype, identifier):
    return {"type": rtype, "identifier": identifier}


class TestDeduplicate:
    def test_removes_exact_duplicate(self):
        refs = [_ref("GEO", "GSE12345"), _ref("GEO", "GSE12345")]
        result = deduplicate(refs)
        assert len(result) == 1

    def test_keeps_different_types(self):
        refs = [_ref("GEO", "GSE12345"), _ref("SRA", "GSE12345")]
        result = deduplicate(refs)
        assert len(result) == 2

    def test_keeps_different_identifiers(self):
        refs = [_ref("GEO", "GSE12345"), _ref("GEO", "GSE99999")]
        result = deduplicate(refs)
        assert len(result) == 2

    def test_preserves_order(self):
        refs = [_ref("GEO", "GSE001"), _ref("SRA", "SRR001"), _ref("GEO", "GSE001")]
        result = deduplicate(refs)
        assert result[0]["identifier"] == "GSE001"
        assert result[1]["identifier"] == "SRR001"

    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_single_item(self):
        refs = [_ref("DOI", "10.1000/xyz123")]
        assert deduplicate(refs) == refs
