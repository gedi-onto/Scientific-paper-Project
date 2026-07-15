"""Ground free-text entity mentions to real ontology term IDs, OntoGPT/OAK-style.

The in-graph lookup in :mod:`ontology_manager` can only match what is loaded, and no
sane amount of RAM holds ChEBI + PRO + GO + MONDO + UBERON as rdflib graphs. This module
takes the OntoGPT/SPIRES approach instead: resolve the *mention* to a term ID with a
deterministic lexical annotator rather than asking a model to recall the ID (models
hallucinate ontology IDs; a lexical match does not).

The backend is the Ontology Access Kit (``oaklib``). By default it uses OAK's OLS
adapter -- the EBI Ontology Lookup Service -- which searches every OBO ontology at once
over HTTP with no local database to download. Results are cached to disk, so the network
is hit once per distinct mention across all runs.

``oaklib`` is an optional dependency. If it is not installed, or the service is
unreachable, every method degrades to "not found" and the pipeline falls back to its
existing in-graph typing -- grounding never crashes a run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# Ontology prefixes worth grounding to, roughly best-first. A term found in several
# ontologies is reported from the most specific appropriate source.
#
# NCIT is deliberately absent. It is a broad thesaurus that has an exact label or
# synonym for almost any word, so it grounds generic terms (study, database, culture,
# migration) and lab shorthand to unrelated concepts far more often than it helps -- on
# a real paper it produced 11 groundings, nearly all noise. The OBO ontologies below are
# scoped to what they cover, so a hit is far more likely to be right.
DEFAULT_PREFERRED_PREFIXES = (
    "PR",        # Protein Ontology
    "CHEBI",     # chemicals / drugs
    "GO",        # functions, processes, components
    "MONDO",     # diseases
    "HP",        # phenotypes
    "UBERON",    # anatomy
    "CL",        # cell types
    "NCBITaxon", # organisms
    "PW",        # pathways
)

# Below this length a mention is almost always a lab acronym (PBS, RNA, HT29, PCR) that
# collides with an unrelated exact label, so it is not grounded. Real gene/protein
# symbols the annotator can place (EGFR, COX2) are handled by the loaded-ontology and
# species-aware paths, not by blind acronym matching.
MIN_GROUNDING_LENGTH = 4


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


class GroundingResult(dict):
    """A resolved grounding: ``curie``, ``iri``, ``label``, ``prefix``, ``method``."""


class OakGrounder:
    """Lexical entity grounder backed by oaklib, with a persistent disk cache.

    Parameters
    ----------
    adapter_string:
        Any OAK selector. Defaults to ``"ols:"`` (remote EBI OLS, no download). Use e.g.
        ``"sqlite:obo:chebi"`` to ground against a single local semsql database instead.
    cache_path:
        JSON file the normalised-mention -> result map is persisted to. Deleting it just
        forces re-lookup.
    preferred_prefixes:
        Ontology prefixes to accept, best-first. A hit outside this set is ignored, so a
        stray match in an irrelevant ontology cannot mistype an entity.
    """

    def __init__(
        self,
        adapter_string: str = "ols:",
        cache_path: str | Path | None = None,
        preferred_prefixes: tuple[str, ...] = DEFAULT_PREFERRED_PREFIXES,
    ) -> None:
        self.adapter_string = adapter_string
        self.preferred_prefixes = tuple(p.upper() for p in preferred_prefixes)
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, Optional[dict]] = {}
        self._adapter = None
        self._adapter_failed = False
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self._cache = {}

    # -- adapter is created lazily so importing this module costs nothing ------------
    def _get_adapter(self):
        if self._adapter is not None or self._adapter_failed:
            return self._adapter
        try:
            from oaklib import get_adapter  # optional dependency
            self._adapter = get_adapter(self.adapter_string)
        except Exception:
            # oaklib missing, or the adapter could not be built -> disable, don't crash.
            self._adapter_failed = True
            self._adapter = None
        return self._adapter

    def _search(self, mention: str) -> Optional[dict]:
        adapter = self._get_adapter()
        if adapter is None:
            return None
        try:
            from oaklib.datamodels.search import SearchConfiguration, SearchProperty
            config = SearchConfiguration(
                is_partial=False,  # exact label / synonym only -- precision over recall
                properties=[SearchProperty.LABEL, SearchProperty.ALIAS],
            )
            curies = [c for c in adapter.basic_search(mention, config=config) if c]
        except Exception:
            return None
        # Keep only hits in an accepted ontology, then pick the best-ranked prefix. A
        # match in an ontology we did not ask for (e.g. a niche process ontology that
        # happens to share a label) is dropped, not accepted as a last resort -- a wrong
        # type is worse than none.
        best = None
        best_rank = len(self.preferred_prefixes)
        for curie in curies:
            prefix = curie.split(":", 1)[0].upper()
            if prefix not in self.preferred_prefixes:
                continue
            r = self.preferred_prefixes.index(prefix)
            if r < best_rank:
                best, best_rank = curie, r
        if best is None:
            return None
        try:
            label = adapter.label(best)
        except Exception:
            label = None
        return {
            "curie": best,
            "iri": _curie_to_iri(best),
            "label": label,
            "prefix": best.split(":", 1)[0].upper(),
            "method": "oak_lexical",
        }

    def ground(self, mention: str) -> Optional[GroundingResult]:
        """Resolve one mention to a term, or ``None``. Cached across the process and disk."""
        key = _normalise(mention)
        if not key or len(key) < MIN_GROUNDING_LENGTH:
            return None
        if key in self._cache:
            cached = self._cache[key]
            return GroundingResult(cached) if cached else None
        result = self._search(mention)
        self._cache[key] = result
        return GroundingResult(result) if result else None

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")


# OBO CURIEs share one PURL pattern; a handful of non-OBO ones need explicit bases.
_IRI_BASES = {
    "NCIT": "http://purl.obolibrary.org/obo/NCIT_",
    "NCBITAXON": "http://purl.obolibrary.org/obo/NCBITaxon_",
    "PW": "http://purl.obolibrary.org/obo/PW_",
}


def _curie_to_iri(curie: str) -> str:
    prefix, _, local = curie.partition(":")
    base = _IRI_BASES.get(prefix.upper())
    if base:
        return f"{base}{local}"
    return f"http://purl.obolibrary.org/obo/{prefix}_{local}"
