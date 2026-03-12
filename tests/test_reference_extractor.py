"""Tests for extractor.reference_extractor."""

import pytest
from extractor.reference_extractor import extract_references, CONTEXT_CHARS


TEXT = (
    "Data are available at https://doi.org/10.1038/nature12345 and "
    "also in GEO under GSE112233. "
    "Raw reads are deposited as SRR7654321. "
    "Protein structure is at PDB: 1ABC. "
    "The zenodo record is https://zenodo.org/record/3456789."
)


class TestExtractReferences:
    def setup_method(self):
        self.hits = extract_references(TEXT)
        self.by_type = {h["type"]: h for h in self.hits}

    def test_finds_doi(self):
        assert "DOI" in self.by_type
        assert "10.1038/nature12345" in self.by_type["DOI"]["identifier"]

    def test_finds_geo(self):
        assert "GEO" in self.by_type
        assert self.by_type["GEO"]["identifier"] == "GSE112233"

    def test_finds_sra(self):
        assert "SRA" in self.by_type
        assert self.by_type["SRA"]["identifier"] == "SRR7654321"

    def test_finds_pdb(self):
        assert "PDB" in self.by_type
        assert self.by_type["PDB"]["identifier"] == "1ABC"

    def test_finds_zenodo(self):
        assert "Zenodo" in self.by_type
        assert self.by_type["Zenodo"]["identifier"] == "3456789"

    def test_context_length(self):
        for hit in self.hits:
            assert len(hit["context"]) <= (2 * CONTEXT_CHARS + 200)

    def test_start_end_offsets(self):
        for hit in self.hits:
            assert hit["start"] >= 0
            assert hit["end"] > hit["start"]

    def test_empty_text_returns_empty_list(self):
        assert extract_references("") == []
