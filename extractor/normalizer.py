"""
Normalizes raw dataset identifiers to their canonical form so that different
representations of the same resource compare equal.

Rules applied
-------------
DOI     – strip leading ``doi:`` / URL prefix → lowercase ``10.xxxx/...``
GEO     – uppercase (``GSE12345``, ``GSM12345``, ``GPL12345``)
SRA     – uppercase (``SRR000001``, ``SRP000001``, …)
BioSample – uppercase (``SAMN12345678``)
BioProject – uppercase (``PRJNA12345``)
ArrayExpress – uppercase (``E-MTAB-1234``)
PRIDE   – uppercase (``PXD012345``)
PDB     – uppercase (``1ABC``)
dbGaP   – lowercase (``phs000001.v1.p1``)
ChEMBL  – uppercase (``CHEMBL1234``)
UniProt – uppercase (``P12345``)
Zenodo  – keep numeric record ID
Figshare – keep full URL (lower-cased)
Dryad   – keep full URL (lower-cased)
OSF     – lowercase 5-char identifier
GitHub  – lowercase full URL
"""

from __future__ import annotations

import re


def normalize(ref: dict) -> dict:
    """Return a copy of *ref* with the ``identifier`` field normalised.

    Parameters
    ----------
    ref:
        A reference dict produced by :func:`extractor.reference_extractor.extract_references`.

    Returns
    -------
    dict
        New dict with the same keys but a canonical ``identifier``.
    """
    ref = dict(ref)  # shallow copy so callers are not surprised
    rtype = ref.get("type", "")
    raw = ref.get("identifier", "")

    ref["identifier"] = _normalise_by_type(rtype, raw)
    return ref


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DOI_PREFIX = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
    re.IGNORECASE,
)


def _normalise_by_type(rtype: str, raw: str) -> str:
    raw = raw.strip()

    if rtype == "DOI":
        # Strip any URL / doi: prefix and lower-case so 10.xxxx/... is stable
        raw = _DOI_PREFIX.sub("", raw)
        return raw.lower().rstrip(".")

    if rtype in ("GEO", "SRA", "BioSample", "BioProject",
                 "ArrayExpress", "PRIDE", "PDB", "ChEMBL", "UniProt"):
        return raw.upper()

    if rtype == "dbGaP":
        return raw.lower()

    if rtype == "Zenodo":
        # Extract the numeric record id if a full URL was matched
        m = re.search(r"(\d+)$", raw)
        return m.group(1) if m else raw

    if rtype in ("Figshare", "Dryad", "GitHub"):
        return raw.lower().rstrip("/")

    if rtype == "OSF":
        return raw.lower()

    return raw
