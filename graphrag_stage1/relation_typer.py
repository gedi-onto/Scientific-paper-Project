"""Rule-based relation typing before ontology object-property lookup.

Mirrors ``semantic_typer.SemanticTyper``: deterministic regex rules assign a
relational type to a Stage 2 relation verb, and each type maps to an ordered
ladder of real ontology property labels ending in a conservative upper-ontology
fallback (``acts on`` / ``affects`` / ``has participant`` / ``is about``).

This only widens recall. ``OntologyManager.resolve_object_property`` still gates
every rung by named OWL/RDFS domain-range axioms, so the typer can never bind an
axiom-incompatible predicate, and every fallback bind is tagged with its
relation type, confidence, and method so it is never mistaken for an exact match.
"""

from __future__ import annotations

import re

from .ontology_manager import normalize_lookup


# Antonymic / contradiction verbs must be caught before any other rule and must
# never fall through to a causal or generic mapping -- asserting "causes" for
# "contradicts" inverts the meaning. There is no faithful contradiction object
# property in the loaded ontologies, so these fail closed (empty ladder).
CONTRADICTION_RE = re.compile(
    r"\b(contradict\w*|conflict\w*|refut\w*|negat\w*|oppos\w*|disagree\w*|"
    r"inconsistent|incompatible|contrary)\b"
)


# Order matters: specific causal/compositional cues are checked before the more
# generic action cues so, e.g., "reduces" is CAUSAL_DECREASE rather than USE.
RELATION_PATTERNS = (
    ("CAUSAL_INCREASE", r"\b(increases?|increased|raises?|raised|boosts?|enhances?|enhanced|improves?|improved)\b"),
    ("CAUSAL_DECREASE", r"\b(reduces?|reduced|decreases?|decreased|lowers?|lowered|minimi[sz]es?|diminishes?|cuts?)\b"),
    ("CAUSATION", r"\b(causes?|caused|drives?|drove|determines?|determined|leads? to|results? in|induces?|triggers?)\b"),
    ("CORRELATION", r"\b(correlat\w*|associat\w*|linked|varies with|covaries)\b"),
    ("COMPARISON", r"\b(compare[ds]? with|compared to|versus|vs|relative to|outperform\w*|exceeds?)\b"),
    ("SUPPORT", r"\b(supports?|supported|corroborat\w*|confirms?|confirmed|demonstrat\w*|validat\w*|provides? evidence)\b"),
    ("CONSISTENCY", r"\b(consistent with|agrees? with|aligns? with|in agreement)\b"),
    ("COMPOSITION_PART", r"\b(part of|belongs? to|member of|component of)\b"),
    ("COMPOSITION", r"\b(combines?|combined|integrat\w*|incorporat\w*|comprises?|consists? of|contains?|includes?|made up of|composed of)\b"),
    ("COORDINATION", r"\b(coordinat\w*|orchestrat\w*|manages?|managed|controls?|controlled|governs?|governed|directs?|regulat\w*|synchroni[sz]\w*)\b"),
    ("ESTIMATION", r"\b(estimat\w*|comput\w*|calculat\w*|infers?|inferred|deriv\w*|predicts?|predicted|forecasts?|assess\w*|evaluat\w*|analy[sz]\w*)\b"),
    ("DETECTION", r"\b(detects?|detected|identif\w*|recogni[sz]\w*|finds?|found|localiz\w*|classif\w*)\b"),
    ("DEPENDENCY", r"\b(depends? on|requires?|required|relies? on|needs?|assumes?|contingent)\b"),
    ("ALLOCATION", r"\b(allocat\w*|assigns?|assigned|distribut\w*|schedules?)\b"),
    ("PRIORITIZATION", r"\b(prioriti[sz]\w*|ranks?|ranked|weights?|weighted)\b"),
    ("CREATION", r"\b(develops?|developed|builds?|built|creates?|created|designs?|designed|proposes?|proposed|introduc\w*|presents?|generat\w*|produc\w*)\b"),
    ("MODIFICATION", r"\b(revis\w*|adjusts?|adjusted|updates?|modif\w*|adapts?|refines?)\b"),
    ("MEASUREMENT", r"\b(measures?|measured|quantif\w*|records?)\b"),
    ("PROVISION", r"\b(provides?|provided|enables?|enabled|facilitates?|allows?|delivers?|offers?)\b"),
    ("USE", r"\b(uses?|used|employs?|leverages?|applies|applied|utili[sz]\w*|involves?)\b"),
    ("PARTICIPATION", r"\b(participat\w*|engages?|interacts? with|collaborat\w*|coordinates? among)\b"),
)


# Every rung below is a property label that exists in the loaded BFO/IAO/CCO/RO
# ontologies. Ladders degrade from a domain-specific verb to a conservative
# upper-ontology relation; the resolver picks the first axiom-compatible rung.
RELATION_ALIASES = {
    "CAUSAL_INCREASE": ["increases", "improves", "causally influences", "affects", "acts on"],
    "CAUSAL_DECREASE": ["reduces", "causally influences", "affects", "acts on"],
    "CAUSATION": ["causes", "causally influences", "causally related to", "affects", "acts on"],
    "CORRELATION": ["correlates with", "correlated with", "causally related to"],
    "COMPARISON": ["compares with", "correlated with", "is about"],
    "SUPPORT": ["supports", "provides evidence for", "is about"],
    "CONSISTENCY": ["is consistent with", "correlated with", "is about"],
    "COMPOSITION": ["has component", "has part", "has participant"],
    "COMPOSITION_PART": ["has part", "has component", "continuant part of"],
    "COORDINATION": ["controls", "regulates", "acts on", "affects", "has participant"],
    "ESTIMATION": ["evaluates", "produces", "has output", "is about"],
    "DETECTION": ["is about", "has output", "produces"],
    "DEPENDENCY": ["depends on", "uses", "has participant"],
    "ALLOCATION": ["allocates", "assigns", "acts on", "affects"],
    "PRIORITIZATION": ["prioritizes", "acts on", "is about"],
    "CREATION": ["develops", "generates", "produces", "has output"],
    "MODIFICATION": ["adjusts", "affects", "acts on"],
    "MEASUREMENT": ["measures", "is about", "has participant"],
    "PROVISION": ["supports", "produces", "has output", "acts on"],
    "USE": ["uses", "has input", "has participant", "acts on"],
    "PARTICIPATION": ["has participant", "acts on", "is about"],
}


# When no lexical rule fires, the Stage 2 frame type supplies a coarse default so
# a bare verb still lands on a defensible relational family.
FRAME_TYPE_DEFAULTS = {
    "MECHANISM": "CAUSATION",
    "CAUSAL_RELATION": "CAUSATION",
    "CORRELATION": "CORRELATION",
    "COMPARISON": "COMPARISON",
    "SUPPORT": "SUPPORT",
    "EVIDENCE": "SUPPORT",
    "CONSISTENCY": "CONSISTENCY",
    "VALIDATION": "SUPPORT",
    "METHOD": "USE",
    "MEASUREMENT": "MEASUREMENT",
    "PART_WHOLE": "COMPOSITION_PART",
    "DEPENDENCY": "DEPENDENCY",
    "PREDICTION": "ESTIMATION",
    "HYPOTHESIS": "ESTIMATION",
}


class RelationTyper:
    """Assign a relational type to a relation verb and expose its property ladder."""

    def type_relation(self, relation: str, frame_type: str | None = None) -> dict:
        normalized = normalize_lookup(relation)
        # Contradiction is checked first and short-circuits the frame-context
        # fallback so a contradictory verb can never be asserted as causal.
        if CONTRADICTION_RE.search(normalized):
            return {"relation_type": "CONTRADICTION", "confidence": 0.75, "method": "deterministic_rules"}
        for relation_type, pattern in RELATION_PATTERNS:
            if re.search(pattern, normalized):
                return {"relation_type": relation_type, "confidence": 0.75, "method": "deterministic_rules"}
        if frame_type in FRAME_TYPE_DEFAULTS:
            return {"relation_type": FRAME_TYPE_DEFAULTS[frame_type], "confidence": 0.5, "method": "frame_context_fallback"}
        return {"relation_type": None, "confidence": 0.0, "method": "no_match"}

    @staticmethod
    def ontology_terms(relation_type: str | None) -> list[str]:
        return list(RELATION_ALIASES.get(relation_type, []))
