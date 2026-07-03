"""Map Stage 2 relation concepts to loaded ontology properties."""

from rdflib import URIRef

from ontology_manager import OntologyManager, normalize_lookup
from relation_typer import RelationTyper


RELATION_TERMS = {
    "developed": ["develops"],
    "develops": ["develops"],
    "causation": ["causes"],
    "causes": ["causes"],
    "increased": ["increases"],
    "increase": ["increases"],
    "evaluates": ["evaluates"],
    "prioritizes": ["prioritizes"],
    "generates": ["generates"],
    "correlation": ["correlates with"],
    "correlate": ["is consistent with", "correlates with"],
    "comparison": ["compares with"],
    "improved": ["improves"],
    "allocates": ["allocates"],
    "adjusts": ["adjusts"],
    "assigns": ["assigns"],
    "maintain": ["maintains"],
    "maintains": ["maintains"],
    "suggest": ["suggests"],
    "suggests": ["suggests"],
    "mechanism": ["acts on"],
    "integrates": ["has component", "has part"],
    "contains": ["has component", "has part"],
    "supports": ["supports", "support"],
    "support": ["supports", "support"],
    "support relation": ["supports", "support"],
    "predicts": ["predicts", "is about"],
    "consistent with": ["is consistent with", "consistency agreement relation"],
    "are consistent with": ["is consistent with", "consistency agreement relation"],
    "consistency agreement relation": ["is consistent with"],
    "reduced": ["reduces"],
    "reduce": ["reduces"],
    "compared with": ["compares with"],
    "hypothesize": ["hypothesizes"],
    "hypothesizes": ["hypothesizes"],
    "improves": ["improves"],
}

DISCOURSE_TERMS = {
    "background": ["provides background for"],
    "source context": ["provides source context for"],
    "evidence": ["provides evidence for"],
    "explanation": ["explains"],
    "support": ["provides evidence for"],
    "elaboration": ["elaborates"],
}


class RelationMapper:
    def __init__(self, manager: OntologyManager) -> None:
        self.manager = manager
        self.typer = RelationTyper()

    def resolve(
        self,
        relation: str,
        subject_class: URIRef | None,
        object_class: URIRef | None,
        frame_type: str | None = None,
    ) -> dict:
        normalized = normalize_lookup(relation)
        terms = [relation, *RELATION_TERMS.get(normalized, [])]
        attempts = []
        for term in dict.fromkeys(terms):
            result = self.manager.resolve_object_property(term, subject_class, object_class)
            attempts.append(result)
            if result["status"] == "matched":
                return {**result, "source_relation": relation, "mapping_method": "controlled_relation_map" if term != relation else "ontology_property_exact", "attempts": attempts}
        # Fallback: rule-based relational typing -> upper-ontology property ladder.
        # The controlled/exact paths above are unchanged, so anything that binds
        # today still binds the same way; this only rescues verbs that would
        # otherwise fail closed (e.g. "coordinates", "combines", "estimates").
        typed = self.typer.type_relation(relation, frame_type)
        for term in dict.fromkeys(self.typer.ontology_terms(typed["relation_type"])):
            result = self.manager.resolve_object_property(term, subject_class, object_class)
            attempts.append(result)
            if result["status"] == "matched":
                return {
                    **result,
                    "source_relation": relation,
                    "relation_type": typed["relation_type"],
                    "relation_type_confidence": typed["confidence"],
                    "mapping_method": f"relation_type_fallback:{typed['method']}",
                    "attempts": attempts,
                }
        return {"term": relation, "source_relation": relation, "status": "missing", "iri": None, "mapping_method": "unresolved", "relation_type": typed["relation_type"], "attempts": attempts}

    def resolve_discourse(self, relation: str) -> dict:
        normalized = normalize_lookup(relation)
        terms = [relation, *DISCOURSE_TERMS.get(normalized, [])]
        attempts = []
        for term in dict.fromkeys(terms):
            result = self.manager.resolve_object_property(term, None, None)
            attempts.append(result)
            if result["status"] == "matched":
                return {**result, "source_relation": relation, "mapping_method": "controlled_discourse_map", "attempts": attempts}
        return {"term": relation, "source_relation": relation, "status": "missing", "iri": None, "mapping_method": "unresolved", "attempts": attempts}
