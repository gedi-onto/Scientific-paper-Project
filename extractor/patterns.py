"""
Compiled regex patterns for detecting dataset identifiers in scientific papers.

Each pattern is a tuple of (type_label, compiled_regex).  The label is used as
the ``type`` field in the extracted reference dict.
"""

import re

# ---------------------------------------------------------------------------
# DOI  (Digital Object Identifier)
# ---------------------------------------------------------------------------
_DOI = re.compile(
    r"""
    (?:https?://(?:dx\.)?doi\.org/|doi:\s*)   # URL or doi: prefix
    (10\.\d{4,9}/[^\s,;\"\')\]>]+)            # DOI suffix
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# GEO  (Gene Expression Omnibus)
# ---------------------------------------------------------------------------
_GEO = re.compile(
    r"\b(GS[EM]\d{3,})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# SRA  (Sequence Read Archive)
# ---------------------------------------------------------------------------
_SRA = re.compile(
    r"\b((?:SRR|SRX|SRS|SRP|ERR|ERX|ERS|ERP|DRR|DRX|DRS|DRP)\d{6,})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# BioSample
# ---------------------------------------------------------------------------
_BIOSAMPLE = re.compile(
    r"\b(SAMN?\d{6,}|SAMD\d{6,}|SAME\d{6,})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# BioProject
# ---------------------------------------------------------------------------
_BIOPROJECT = re.compile(
    r"\b(PRJ(?:NA|NB|EB|DB)\d+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ArrayExpress
# ---------------------------------------------------------------------------
_ARRAYEXPRESS = re.compile(
    r"\b(E-\w{4}-\d+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# PRIDE  (PRoteomics IDEntifications)
# ---------------------------------------------------------------------------
_PRIDE = re.compile(
    r"\b(PXD\d{6,})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# PDB  (Protein Data Bank)
# ---------------------------------------------------------------------------
_PDB = re.compile(
    r"\bPDB\s*:?\s*([0-9][A-Z0-9]{3})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# dbGaP
# ---------------------------------------------------------------------------
_DBGAP = re.compile(
    r"\b(phs\d{6}\.v\d+\.p\d+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ChEMBL
# ---------------------------------------------------------------------------
_CHEMBL = re.compile(
    r"\b(CHEMBL\d+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# UniProt
# ---------------------------------------------------------------------------
_UNIPROT = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b",
)

# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------
_ZENODO = re.compile(
    r"https?://(?:www\.)?zenodo\.org/(?:record|deposit)/(\d+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Figshare
# ---------------------------------------------------------------------------
_FIGSHARE = re.compile(
    r"https?://(?:www\.)?figshare\.com/(?:articles|s)/[^\s,;\"\')\]>]+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dryad
# ---------------------------------------------------------------------------
_DRYAD = re.compile(
    r"https?://(?:www\.)?datadryad\.org/(?:stash/)?dataset/[^\s,;\"\')\]>]+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# OSF  (Open Science Framework)
# ---------------------------------------------------------------------------
_OSF = re.compile(
    r"https?://osf\.io/([a-z0-9]{5})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# GitHub repository
# ---------------------------------------------------------------------------
_GITHUB = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/tree/[^\s,;\"\')\]>]+)?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Public catalog — ordered from most-specific to least-specific so that a
# match for a specific type is preferred over a generic URL catch-all.
# ---------------------------------------------------------------------------
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("DOI", _DOI),
    ("GEO", _GEO),
    ("SRA", _SRA),
    ("BioSample", _BIOSAMPLE),
    ("BioProject", _BIOPROJECT),
    ("ArrayExpress", _ARRAYEXPRESS),
    ("PRIDE", _PRIDE),
    ("PDB", _PDB),
    ("dbGaP", _DBGAP),
    ("ChEMBL", _CHEMBL),
    ("UniProt", _UNIPROT),
    ("Zenodo", _ZENODO),
    ("Figshare", _FIGSHARE),
    ("Dryad", _DRYAD),
    ("OSF", _OSF),
    ("GitHub", _GITHUB),
]
