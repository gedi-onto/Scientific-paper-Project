"""
Deduplicates a list of normalized reference dicts.

Two references are considered duplicates when both their ``type`` and
their normalized ``identifier`` are equal.  The first occurrence is kept
and subsequent duplicates are discarded.
"""

from __future__ import annotations


def deduplicate(refs: list[dict]) -> list[dict]:
    """Return *refs* with duplicate entries removed.

    Parameters
    ----------
    refs:
        List of normalized reference dicts (each must have ``type`` and
        ``identifier`` keys).

    Returns
    -------
    list[dict]
        Deduplicated list preserving original order.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []

    for ref in refs:
        key = (ref.get("type", ""), ref.get("identifier", ""))
        if key not in seen:
            seen.add(key)
            unique.append(ref)

    return unique
