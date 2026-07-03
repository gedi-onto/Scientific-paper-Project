"""Rule-based semantic typing before ontology lookup."""

from __future__ import annotations

import re

from .ontology_manager import normalize_lookup


TYPE_PATTERNS = (
    ("GROUP", r"\b(research teams?|teams?|groups?|crews?)\b"),
    ("PERSON", r"\b(researchers?|investigators?|scientists?|persons?|patients?|operators?)\b"),
    # Quantified qualities must precede SYSTEM/TIME.  Otherwise phrases such
    # as "system efficiency" and "mission completion time" are typed by a
    # contextual word rather than by their semantic head.
    ("MEASURABLE_QUALITY", r"\b(energy|power)\s+(consumption|utilization|capacity)\b|\b(stored energy|operational costs?|reliability|accuracy|temperature|pressure|velocity|latency|stability|uncertainty|efficiency|performance|viral load|battery capacity|detection rate|success rate|completion time|availability|coordination)\b"),
    ("QUALITY", r"\b(quality|robustness)\b"),
    ("MEASUREMENT", r"^\s*(?:from\s+)?(?:r\s*=\s*|p\s*[<=>]\s*)?\d+(?:\.\d+)?(?:\s*(?:percent|%|milliseconds?|ms))?(?:\s+to\s+\d+(?:\.\d+)?\s*(?:percent|%))?\s*$|\b(measurement|datum|score|numeric value|coefficient|p[- ]?value)\b"),
    # Vehicles are material artifacts even when their names contain modifiers
    # such as autonomous, aerial, underwater, or unmanned.
    ("VEHICLE", r"\b(vehicles?|aircraft|drones?|uavs?|ugvs?|rovers?|vessels?|ships?|boats?)\b"),
    ("ALGORITHM", r"\b(algorithms?|classifiers?|estimators?|solvers?)\b"),
    # Models, maps, imagery, plans and knowledge representations denote
    # representational artifacts; they are not executable algorithms merely
    # because a phrase contains the word 'model'.
    ("REPRESENTATIONAL_MODEL", r"\b(models?|maps?|imagery|images?|representations?|ontolog(?:y|ies)|knowledge graphs?|terrain data|mission plans?)\b"),
    ("CLAIM", r"\b(claim|hypothesis|proposition)\b|\b(improves?|reduces?|predicts?)\b"),
    ("DEVICE", r"\b(modules?|transceivers?|communication units?|photovoltaic arrays?|solar arrays?|batter(?:y|ies)(?: storage)?|inverters?|controllers?)\b"),
    ("SENSOR", r"\b(sensors?|lidar|radar|cameras?|receivers?|imu|measurement units?)\b"),
    ("SYSTEM", r"\b(system|architecture|framework|platform)\b"),
    ("SYSTEM", r"\bservices?\b"),
    ("DEVICE", r"\b(device|instrument|apparatus|camera|receiver)\b"),
    ("PROCESS", r"\b(process|mechanism|forecasting|production|deployments?|replication|administration|localization|fusion|analysis|weighting)\b"),
    ("HYPOTHESIS", r"\b(hypotheses|hypothesis)\b"),
    ("OBSERVATION", r"\b(observations?|results?|findings?|evidence)\b"),
    ("DOCUMENT", r"\b(document|paper|publication|study|report)\b"),
    ("LOCATION", r"\b(location|environment|region|position)\b"),
    ("TIME", r"\b(time|day|year|duration)\b"),
    ("EVENT", r"\b(event|experiment|trial)\b"),
)


class SemanticTyper:
    def type_entity(self, canonical_form: str, frame_type: str | None = None) -> dict:
        normalized = normalize_lookup(canonical_form)
        for semantic_type, pattern in TYPE_PATTERNS:
            if re.search(pattern, normalized):
                return {"semantic_type": semantic_type, "confidence": 0.9, "method": "deterministic_rules"}
        if frame_type in {"HYPOTHESIS", "PREDICTION", "OBSERVATION", "METHOD"}:
            return {"semantic_type": "INFORMATION_CONTENT_ENTITY", "confidence": 0.65, "method": "frame_context_fallback"}
        return {"semantic_type": "INFORMATION_CONTENT_ENTITY", "confidence": 0.5, "method": "generic_fallback"}

    @staticmethod
    def ontology_terms(semantic_type: str) -> list[str]:
        terms = semantic_type.replace("_", " ").lower()
        aliases = {
            "PERSON": ["person", "agent"],
            "SENSOR": ["sensor", "sensing device", "device"],
            "VEHICLE": ["vehicle", "artifact"],
            "ALGORITHM": ["algorithm"],
            "REPRESENTATIONAL_MODEL": ["representational information content entity", "information content entity"],
            "CLAIM": ["hypothesis textual entity", "information content entity"],
            "HYPOTHESIS": ["hypothesis textual entity", "information content entity"],
            "OBSERVATION": ["information content entity"],
            "SYSTEM": ["information processing artifact", "material artifact"],
            "DEVICE": ["device", "artifact"],
            "MEASUREMENT": ["measurement datum", "measurement information content entity", "information content entity"],
            "PROCESS": ["process"],
            "QUALITY": ["quality"],
            "MEASURABLE_QUALITY": ["quality"],
            "INFORMATION_CONTENT_ENTITY": ["information content entity"],
        }
        return aliases.get(semantic_type, [terms, "information content entity"])
