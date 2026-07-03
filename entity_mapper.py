"""Semantic-type-driven ontology class and individual mapping."""

from __future__ import annotations

import os

from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, XSD

from canonicalizer import Canonicalizer
from instance_generator import InstanceGenerator
from ontology_manager import OntologyManager
from semantic_typer import SemanticTyper


SEMANTIC_CLASS_IRIS = {
    "PERSON": "https://www.commoncoreontologies.org/ont00001262",
    "SENSOR": "https://www.commoncoreontologies.org/ont00000569",
    "VEHICLE": "https://www.commoncoreontologies.org/ont00000713",
    "ALGORITHM": "http://purl.obolibrary.org/obo/IAO_0000064",
    "REPRESENTATIONAL_MODEL": "https://www.commoncoreontologies.org/ont00001069",
    "SYSTEM": "https://www.commoncoreontologies.org/ont00000117",
    "MEASUREMENT": "http://purl.obolibrary.org/obo/IAO_0000109",
    "LOCATION": "https://www.commoncoreontologies.org/ont00000373",
    "CLAIM": "http://purl.obolibrary.org/obo/IAO_0000415",
    "HYPOTHESIS": "http://purl.obolibrary.org/obo/IAO_0000415",
    "OBSERVATION": "http://purl.obolibrary.org/obo/IAO_0000030",
    "INFORMATION_CONTENT_ENTITY": "http://purl.obolibrary.org/obo/IAO_0000030",
    "PROCESS": "http://purl.obolibrary.org/obo/BFO_0000015",
    "QUALITY": "http://purl.obolibrary.org/obo/BFO_0000019",
    "MEASURABLE_QUALITY": "http://purl.obolibrary.org/obo/BFO_0000019",
    "DEVICE": "https://www.commoncoreontologies.org/ont00000569",
    "GROUP": "https://www.commoncoreontologies.org/ont00001262",
}


class EntityMapper:
    def __init__(self, manager: OntologyManager) -> None:
        self.manager = manager
        self.canonicalizer = Canonicalizer()
        self.typer = SemanticTyper()
        self.instances = InstanceGenerator(manager)

    def _select_class(self, semantic_type: str, canonical: str) -> dict:
        attempts = []
        preferred = SEMANTIC_CLASS_IRIS.get(semantic_type)
        loaded_classes = {item for values in self.manager.class_index.values() for item in values}
        if preferred and URIRef(preferred) in loaded_classes:
            return {"iri": preferred, "method": "semantic_type_registry", "confidence": 0.95, "attempts": attempts}
        # A unique exact domain label is a safe specialization of the semantic
        # type. Fuzzy label ranking is deliberately not used for selection.
        domain_exact = self.manager.explain_class_lookup(canonical)
        attempts.append(domain_exact)
        if domain_exact["status"] == "matched":
            return {"iri": domain_exact["iri"], "method": "semantic_type_domain_exact", "confidence": 0.9, "attempts": attempts}
        for term in self.typer.ontology_terms(semantic_type):
            lookup = self.manager.explain_class_lookup(term)
            attempts.append(lookup)
            if lookup["status"] == "matched":
                return {"iri": lookup["iri"], "method": "semantic_type_exact", "confidence": 0.9, "attempts": attempts}
        # Similarity candidates are diagnostic only: semantic type controls
        # class selection and prevents RO/unrelated classes from winning.
        lexical = self.manager.class_candidates(canonical)
        for term in ("information content entity", "entity", "thing"):
            lookup = self.manager.explain_class_lookup(term)
            attempts.append(lookup)
            if lookup["status"] == "matched":
                return {"iri": lookup["iri"], "method": "ontology_fallback_class", "confidence": 0.35, "attempts": attempts, "candidates": lexical}
        # A loaded class is required by OntologyManager. Select deterministically
        # from loaded classes as the final domain-independent fallback.
        loaded = sorted({str(item) for values in self.manager.class_index.values() for item in values})
        return {"iri": loaded[0] if loaded else None, "method": "loaded_root_fallback", "confidence": 0.2, "attempts": attempts, "candidates": lexical}

    def map(self, surface: str, scope: str, frame_type: str | None = None, resolved_text: str | None = None) -> dict:
        canonical = self.canonicalizer.canonicalize(surface, resolved_text)
        typing = self.typer.type_entity(canonical["canonical_form"], frame_type)
        selection = self._select_class(typing["semantic_type"], canonical["canonical_form"])
        ontology_class = URIRef(selection["iri"]) if selection["iri"] else None
        instance = self.instances.create(ontology_class, scope, canonical["canonical_form"]) if ontology_class else None
        confidence = round(min(typing["confidence"], selection["confidence"]), 4)
        metadata_assertions = []
        if instance:
            metadata = (
                ("canonical form", canonical["canonical_form"], XSD.string),
                ("semantic type", typing["semantic_type"], XSD.string),
                ("mapping method", selection["method"], XSD.string),
                ("mapping confidence", confidence, XSD.decimal),
            )
            for term, value, datatype in metadata:
                prop = self.manager.find_datatype_property(term)
                if prop:
                    self.manager.add_datatype_assertion(instance, prop, value, datatype)
                    metadata_assertions.append({
                        "subject": str(instance), "predicate": str(prop),
                        "value": value, "datatype": str(datatype),
                    })
        inferred_types = []
        materialize_inferred = os.getenv("STAGE3_MATERIALIZE_INFERRED_TYPES", "0") == "1"
        if materialize_inferred and instance and ontology_class:
            for parent in self.manager.graph.transitive_objects(ontology_class, RDFS.subClassOf):
                if isinstance(parent, URIRef) and parent != ontology_class:
                    self.manager.graph.add((instance, RDF.type, parent))
                    inferred_types.append(str(parent))
        return {
            **canonical,
            "semantic_type": typing["semantic_type"],
            "semantic_typing_method": typing["method"],
            "ontology_class": str(ontology_class) if ontology_class else None,
            "rdf_type": str(ontology_class) if ontology_class else None,
            "inferred_types": sorted(set(inferred_types)),
            "inferred_types_materialized": materialize_inferred,
            "instance_id": str(instance) if instance else None,
            "mapping_status": "mapped" if instance else "mapping_error",
            "mapping_method": selection["method"],
            "mapping_confidence": confidence,
            "review_required": confidence < 0.6,
            "mapping_candidates": selection.get("candidates", []),
            "class_selection_audit": selection["attempts"],
            "datatype_assertions": metadata_assertions,
        }
