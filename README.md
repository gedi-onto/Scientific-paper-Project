# Scientific-paper-Project

A Python toolkit — the **Semantic Data Aggregator** — that extracts, normalizes,
and deduplicates dataset references from scientific papers.

Given the plain text of a paper the aggregator:

1. **Locates** the *Data Availability* and *References* sections (heading-based
   detection with a keyword-density fallback).
2. **Extracts** every dataset identifier it recognises (DOIs, GEO, SRA,
   BioSample, BioProject, ArrayExpress, PRIDE, PDB, dbGaP, ChEMBL, UniProt,
   Zenodo, Figshare, Dryad, OSF, GitHub) together with ±120 characters of
   surrounding context.
3. **Normalizes** each identifier to its canonical form (e.g. strips URL
   prefixes from DOIs, upper-cases GEO/SRA accessions, extracts the numeric
   Zenodo record ID).
4. **Deduplicates** by (type, identifier) so each dataset is listed exactly once.

---

## Repository layout

```
Scientific-paper-Project/
├── extractor/
│   ├── __init__.py          # exposes SemanticDataAggregator
│   ├── aggregator.py        # top-level pipeline
│   ├── patterns.py          # compiled regex patterns for every dataset type
│   ├── section_extractor.py # locate Data Availability / References sections
│   ├── reference_extractor.py # apply patterns, capture context
│   ├── normalizer.py        # canonical-form normalization
│   └── deduplicator.py      # (type, identifier) deduplication
├── tests/
│   ├── test_aggregator.py
│   ├── test_patterns.py
│   ├── test_section_extractor.py
│   ├── test_reference_extractor.py
│   ├── test_normalizer.py
│   └── test_deduplicator.py
├── requirements.txt
└── README.md
```

---

## Setup

Python 3.10 or later is required. The library has no mandatory runtime
dependencies — only `pytest` is needed to run the tests.

```bash
# Clone the repository
git clone https://github.com/gedi-onto/Scientific-paper-Project.git
cd Scientific-paper-Project

# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install test dependencies
pip install -r requirements.txt
```

---

## Quick start

```python
from extractor import SemanticDataAggregator

paper_text = open("my_paper.txt").read()

agg = SemanticDataAggregator()
result = agg.aggregate(paper_text)

for ref in result["references"]:
    print(ref["type"], ref["identifier"])
    # e.g.  GEO  GSE123456
    #       DOI  10.1038/s41586-020-00001-x
    #       SRA  SRR9876543
```

### Output structure

```python
{
    "sections": {
        "data_availability": "...",   # raw extracted text
        "references": "..."
    },
    "references": [
        {
            "type":           "GEO",
            "identifier":     "GSE123456",   # normalized
            "context":        "... ±120 chars ...",
            "start":          42,            # char offset in section text
            "end":            51,
            "source_section": "data_availability"
        },
        ...
    ]
}
```

### Search only specific sections

```python
# Only scan the Data Availability section
agg = SemanticDataAggregator(include_sections=("data_availability",))
result = agg.aggregate(paper_text)
```

---

## Supported dataset types and regex patterns

| Type | Example identifier | Pattern notes |
|------|--------------------|---------------|
| DOI | `10.1038/nature12345` | Matches `https://doi.org/…`, `http://dx.doi.org/…`, `doi: …` |
| GEO | `GSE12345` | `GSE`, `GSM`, `GPL` prefixes |
| SRA | `SRR1234567` | `SRR/SRX/SRS/SRP`, `ERR/ERX/ERS/ERP`, `DRR/DRX/DRS/DRP` |
| BioSample | `SAMN01234567` | `SAMN`, `SAMD`, `SAME` prefixes |
| BioProject | `PRJNA123456` | `PRJNA/PRJNB/PRJEB/PRJDB` prefixes |
| ArrayExpress | `E-MTAB-1234` | `E-XXXX-####` format |
| PRIDE | `PXD012345` | `PXD` prefix |
| PDB | `1ABC` | Preceded by `PDB` keyword |
| dbGaP | `phs000001.v1.p1` | Full versioned accession |
| ChEMBL | `CHEMBL1234` | `CHEMBL` prefix |
| UniProt | `P12345` | Standard UniProt accession format |
| Zenodo | `3456789` | Full record URL |
| Figshare | full URL | `figshare.com/articles/…` |
| Dryad | full URL | `datadryad.org/…` |
| OSF | `abc12` | `osf.io/XXXXX` |
| GitHub | full URL | `github.com/owner/repo` |

---

## Normalization rules

| Type | Rule |
|------|------|
| DOI | Strip `https://doi.org/` / `doi:` prefix; lower-case; strip trailing `.` |
| GEO, SRA, BioSample, BioProject, ArrayExpress, PRIDE, PDB, ChEMBL, UniProt | Upper-case |
| dbGaP | Lower-case |
| Zenodo | Extract numeric record ID |
| Figshare, Dryad, GitHub | Lower-case, strip trailing `/` |
| OSF | Lower-case |

---

## Section extraction logic

`section_extractor.extract_sections(text)` uses a two-stage approach:

1. **Heading detection** — paragraphs whose first line matches a known section
   heading keyword (e.g. *Data Availability*, *Data Availability Statement*,
   *References*, *Bibliography*) trigger structured collection.
2. **Keyword-density fallback** — when no explicit heading is found, paragraphs
   containing high-density dataset-reference keywords (e.g. *accession number*,
   *deposited in*, *publicly available*, *zenodo*, *figshare*) are collected as
   the data-availability text.

---

## Running the tests

```bash
pytest tests/ -v
```

All 55 tests should pass in under a second.

---

## Issues addressed

| # | Title | Status |
|---|-------|--------|
| 1 | Design and Implement Regex Patterns for Dataset References | ✅ `extractor/patterns.py` |
| 2 | Extract Data Availability Sections | ✅ `extractor/section_extractor.py` |
| 3 | Apply Regex Patterns to Extract Dataset References | ✅ `extractor/reference_extractor.py` |
| 4 | Normalize Dataset References | ✅ `extractor/normalizer.py` |
| 5 | Deduplicate Dataset References | ✅ `extractor/deduplicator.py` |
| 6 | Testing and Validation | ✅ `tests/` |
| 7 | Documentation and GitHub ReadMe | ✅ this file |
