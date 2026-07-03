"""Deterministic ontology individual generation."""

from rdflib import URIRef

from ontology_manager import OntologyManager, normalize_lookup


class InstanceGenerator:
    def __init__(self, manager: OntologyManager) -> None:
        self.manager = manager

    def create(self, ontology_class: URIRef, scope: str, canonical_form: str) -> URIRef:
        return self.manager.create_instance(
            ontology_class,
            f"entity|{scope}|{normalize_lookup(canonical_form)}",
            canonical_form,
        )
