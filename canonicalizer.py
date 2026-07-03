"""Deterministic entity canonicalization for Stage 3."""

from __future__ import annotations

import re

from ontology_manager import normalize_lookup


CANONICAL_ALIASES = {
    "gps": "GPS receiver",
    "gps receiver": "GPS receiver",
    "global positioning system receiver": "GPS receiver",
    "lidar": "lidar sensor",
    "lidar sensor": "lidar sensor",
    "laser scanner": "lidar sensor",
    "imu": "inertial measurement unit",
    "researcher": "scientist",
    "researchers": "scientist",
    "investigator": "scientist",
    "investigators": "scientist",
    "research team": "scientist",
    "results": "experimental results",
    "experimental results": "experimental results",
    "findings": "experimental results",
}


class Canonicalizer:
    def canonicalize(self, surface: str, resolved_text: str | None = None) -> dict:
        source = (resolved_text or surface or "").strip()
        source = re.sub(r"^(?:and|or)\s+", "", source, flags=re.I)
        stripped = re.sub(r"^(?:a|an|the|each|these|those)\s+", "", source, flags=re.I)
        # Epistemic/status modifiers describe a mention, not a new individual.
        # Keeping them in the identity key fragmented the same algorithm across frames.
        stripped = re.sub(r"^(?:proposed|described|aforementioned)\s+", "", stripped, flags=re.I)
        normalized = normalize_lookup(stripped)
        canonical = CANONICAL_ALIASES.get(normalized, stripped)
        aliases = sorted({item for item in (surface.strip(), resolved_text) if item and item != canonical})
        return {
            "surface_form": surface.strip(),
            "canonical_form": canonical.strip(),
            "aliases": aliases,
            "canonicalization_method": (
                "reference_resolution" if resolved_text else
                "controlled_alias" if normalized in CANONICAL_ALIASES else
                "deterministic_normalization"
            ),
        }

    def split_compound(self, value: str) -> list[str]:
        """Split explicit coordinated lists, while leaving ordinary phrases intact."""
        text = (value or "").strip()
        if not text or not ("," in text or re.search(r"\s+and\s+", text, re.I)):
            return [text] if text else []
        parts = re.split(r"\s*,\s*|\s+and\s+", text, flags=re.I)
        # The leading article must be a whole word; otherwise "a"/"an" would strip
        # the first letters of words like "autonomous" ("a|an|the" without a word
        # boundary turned "autonomous vehicles" into "utonomous vehicles").
        cleaned = [
            re.sub(r"^(?:(?:and|or)\s+)?(?:(?:a|an|the)\s+)?", "", part.strip(), flags=re.I)
            for part in parts
        ]
        return [part for part in cleaned if part]
