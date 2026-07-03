"""Assertion record construction helpers."""

from rdflib import URIRef

from ontology_manager import OntologyManager


class AssertionBuilder:
    def __init__(self, manager: OntologyManager) -> None:
        self.manager = manager

    def object_assertion(self, subject: URIRef, predicate: URIRef, obj: URIRef, **metadata) -> dict:
        self.manager.add_object_assertion(subject, predicate, obj)
        return {"subject": str(subject), "predicate": str(predicate), "object": str(obj), **metadata}
