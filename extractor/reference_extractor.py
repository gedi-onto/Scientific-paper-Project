"""
Applies regex patterns from :mod:`extractor.patterns` to text and returns
structured reference hits with surrounding context.

Each hit is a dict with:
    ``type``        – pattern label (e.g. "DOI", "GEO")
    ``identifier``  – the matched string
    ``context``     – up to ±120 characters around the match
    ``start``       – character offset of the match start in the source text
    ``end``         – character offset of the match end in the source text
"""

from __future__ import annotations

from extractor.patterns import PATTERNS

CONTEXT_CHARS = 120


def extract_references(text: str) -> list[dict]:
    """Return all dataset-reference hits found in *text*.

    Parameters
    ----------
    text:
        Plain-text content of the section(s) to search.

    Returns
    -------
    list[dict]
        One dict per match (see module docstring for fields).
    """
    hits: list[dict] = []

    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()

            ctx_start = max(0, start - CONTEXT_CHARS)
            ctx_end = min(len(text), end + CONTEXT_CHARS)
            context = text[ctx_start:ctx_end].replace("\n", " ").strip()

            # Use the first capturing group as the identifier when available,
            # otherwise fall back to the full match.
            identifier = match.group(1) if match.lastindex else match.group(0)

            hits.append(
                {
                    "type": label,
                    "identifier": identifier,
                    "context": context,
                    "start": start,
                    "end": end,
                }
            )

    return hits
