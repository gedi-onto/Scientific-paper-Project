"""
Extracts Data Availability and References sections from plain-text scientific
papers.

Strategy
--------
1. The text is split into paragraphs (blank-line separated).
2. Each paragraph is tested against a set of section-heading keywords for both
   the *Data Availability* family and the *References* family.
3. Once a matching heading is found, text is collected until the next
   recognised section heading (or end of document).
4. If no explicit heading is found, a sliding-window approach is used to
   collect paragraphs that contain high-density dataset-reference keywords.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Section-heading patterns
# ---------------------------------------------------------------------------
_DA_HEADING = re.compile(
    r"""
    ^(?:
        data\s+availability(?:\s+statement)?  |
        availability\s+of\s+data(?:\s+and\s+(?:material|code))?  |
        data\s+(?:and\s+code\s+)?availability  |
        availability\s+statement  |
        data\s+access  |
        data\s+sharing
    )\s*:?\s*$
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_REF_HEADING = re.compile(
    r"^(?:references?|bibliography|works\s+cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Generic "new section" detector used to stop collection.
_ANY_HEADING = re.compile(
    r"""
    ^(?:
        abstract  |  introduction  |  methods?  |  materials?\s+and\s+methods?  |
        results?  |  discussion  |  conclusion  |  acknowledgements?  |
        supplementary  |  appendix  |  author\s+contributions?  |
        conflict\s+of\s+interest  |  funding  |  ethics
    )\s*:?\s*$
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

# Keywords hinting that a paragraph likely describes data availability
_DA_KEYWORDS = re.compile(
    r"(?:data(?:set)?s?\s+(?:are\s+)?available|accession\s+(?:number|code)|"
    r"deposited\s+(?:in|at|to)|downloaded\s+from|publicly\s+available|"
    r"open\s+access|figshare|zenodo|dryad|osf\.io|github\.com|"
    r"doi\.org|geo\b|sra\b|arrayexpress)",
    re.IGNORECASE,
)


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* on one or more blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _collect_section(
    paragraphs: list[str],
    start_idx: int,
    stop_pattern: re.Pattern[str],
) -> str:
    """Collect paragraphs starting at *start_idx* until *stop_pattern* matches."""
    collected: list[str] = []
    for para in paragraphs[start_idx:]:
        first_line = para.splitlines()[0].strip() if para else ""
        if stop_pattern.match(first_line) or _ANY_HEADING.match(first_line):
            break
        collected.append(para)
    return "\n\n".join(collected)


def extract_sections(text: str) -> dict[str, str]:
    """Return a dict with keys ``"data_availability"`` and ``"references"``.

    Either value may be an empty string when the section is absent.
    """
    paragraphs = _split_paragraphs(text)
    da_text = ""
    ref_text = ""

    for i, para in enumerate(paragraphs):
        first_line = para.splitlines()[0].strip()

        if not da_text and _DA_HEADING.match(first_line):
            # Include the rest of the paragraph after the heading, then collect
            body_lines = para.splitlines()[1:]
            body = "\n".join(body_lines).strip()
            rest = _collect_section(paragraphs, i + 1, _REF_HEADING)
            da_text = (body + "\n\n" + rest).strip()

        if not ref_text and _REF_HEADING.match(first_line):
            body_lines = para.splitlines()[1:]
            body = "\n".join(body_lines).strip()
            rest = _collect_section(paragraphs, i + 1, re.compile(r"^$"))
            ref_text = (body + "\n\n" + rest).strip()

    # Fallback: collect DA-keyword-rich paragraphs if no explicit section found
    if not da_text:
        da_candidates = [p for p in paragraphs if _DA_KEYWORDS.search(p)]
        da_text = "\n\n".join(da_candidates)

    return {"data_availability": da_text, "references": ref_text}
