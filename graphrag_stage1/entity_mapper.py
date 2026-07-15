"""Semantic-type-driven ontology class and individual mapping."""

from __future__ import annotations

import os

from rdflib import URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from .canonicalizer import Canonicalizer
from .instance_generator import InstanceGenerator
from .ontology_manager import OntologyManager
from .semantic_typer import SemanticTyper


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


# BFO and its legacy IFOMIS 1.1 copy are both in the loaded graph, so every upper term
# ("entity", "process", "continuant"...) collides with its own twin and resolves as
# ambiguous. Preferring the canonical OBO namespace breaks the tie. Without this the
# whole ladder below silently misses and entities land on an arbitrary class.
CANONICAL_UPPER_NAMESPACE = "http://purl.obolibrary.org/obo/"

# When the ontology has no class for an entity by name, climb to the nearest class that
# is still TRUE of it -- never sideways into a different kind of thing. A "car" with no
# Car class should become a Vehicle, not an Information Content Entity.
#
# CCO sits between BFO's abstractions and the domain, so its mid-level classes come
# first: landing on "group of agents" or "planned act" is genuinely useful, whereas
# landing on BFO's "entity" is true but says nothing. BFO is the terminal rung, kept
# because it is always true when nothing better applies.
#
# The branches are disjoint on purpose. An occurrent (something that happens) is not a
# continuant (something that persists), and an information content entity is a thing
# that *stands for* something else. Typing a protein as information is not a cautious
# guess, it is a false claim, and it would corrupt any reasoning over the graph.
_MATERIAL_LADDER = ("material entity", "independent continuant", "continuant", "entity")
_ARTIFACT_LADDER = ("material artifact",) + _MATERIAL_LADDER      # CCO first
_AGENT_LADDER = ("agent",) + _MATERIAL_LADDER                     # CCO first
_GROUP_LADDER = ("group of agents", "agent") + _MATERIAL_LADDER   # CCO first
_PROCESS_LADDER = ("act", "process", "occurrent", "entity")       # CCO "act" first
_QUALITY_LADDER = ("quality", "specifically dependent continuant", "continuant", "entity")
_INFORMATION_LADDER = ("information content entity", "generically dependent continuant",
                       "continuant", "entity")
_MEASUREMENT_LADDER = ("measurement information content entity",) + _INFORMATION_LADDER
_DESCRIPTIVE_LADDER = ("descriptive information content entity",) + _INFORMATION_LADDER

FALLBACK_LADDERS = {
    "PROCESS": _PROCESS_LADDER,
    "QUALITY": _QUALITY_LADDER,
    "MEASURABLE_QUALITY": _QUALITY_LADDER,
    # Things that genuinely ARE information.
    "MEASUREMENT": _MEASUREMENT_LADDER,
    "CLAIM": _DESCRIPTIVE_LADDER,
    "HYPOTHESIS": _DESCRIPTIVE_LADDER,
    "OBSERVATION": _DESCRIPTIVE_LADDER,
    "DOCUMENT": _INFORMATION_LADDER,
    "ALGORITHM": _INFORMATION_LADDER,
    "REPRESENTATIONAL_MODEL": _INFORMATION_LADDER,
    "INFORMATION_CONTENT_ENTITY": _INFORMATION_LADDER,
    # Things that are stuff in the world.
    "PERSON": ("person",) + _AGENT_LADDER,
    "GROUP": _GROUP_LADDER,
    "DEVICE": _ARTIFACT_LADDER,
    "SENSOR": _ARTIFACT_LADDER,
    "VEHICLE": _ARTIFACT_LADDER,
    "SYSTEM": _ARTIFACT_LADDER,
    "LOCATION": _MATERIAL_LADDER,
}


def generic_fallback_terms(semantic_type: str) -> tuple[str, ...]:
    """Class ladder for a semantic type: most specific first, always-true last.

    An unrecognised semantic type gets the neutral ladder. BFO's root ``entity`` is true
    of literally anything, which is honest; picking a branch we cannot justify is not.
    """
    return FALLBACK_LADDERS.get(semantic_type, ("entity",))


class EntityMapper:
    def __init__(self, manager: OntologyManager, grounder=None) -> None:
        self.manager = manager
        self.canonicalizer = Canonicalizer()
        self.typer = SemanticTyper()
        self.instances = InstanceGenerator(manager)
        # Optional OntoGPT/OAK-style lexical grounder. When set, an entity the loaded
        # ontologies cannot type is looked up against the wider OBO world (ChEBI, PRO,
        # GO, MONDO, UBERON...) before falling back to a generic upper class. Off by
        # default: the pipeline stays fully offline unless a grounder is supplied.
        self.grounder = grounder

    def _select_class(
        self, semantic_type: str, canonical: str, typing_recognised: bool = True
    ) -> dict:
        """Pick the most SPECIFIC ontology class for an entity.

        Order matters, and it used to be backwards. The semantic-type registry was
        consulted first, but ``SemanticTyper`` has no domain vocabulary -- it types
        almost everything as INFORMATION_CONTENT_ENTITY, which *is* in the registry.
        So the registry short-circuited on a generic catch-all and the domain lookup
        below became dead code: on a real paper, 424 of 430 entities were typed from
        the registry and not one from the loaded ontologies, leaving 88% of the graph
        as ``IAO_0000030``.

        The entity's own name matching an ontology label exactly is strictly more
        informative than a guess from its coarse type, so that is tried first. This is
        still an EXACT match -- fuzzy label ranking remains diagnostic only, so an
        unrelated class cannot win.
        """
        attempts = []
        loaded_classes = {item for values in self.manager.class_index.values() for item in values}

        # 1. The entity's own name, matched exactly against a loaded class label.
        domain_exact = self.manager.explain_class_lookup(canonical)
        attempts.append(domain_exact)
        if domain_exact["status"] == "matched":
            return {"iri": domain_exact["iri"], "method": "domain_exact",
                    "confidence": 0.95, "attempts": attempts}

        # 2. Generalise to the head of the noun phrase. If the ontology has no class for
        #    "Syk inhibitor" but does have "inhibitor", the entity IS an inhibitor --
        #    typing it as one is true, useful, and far better than dumping it in a
        #    generic bucket. The instance keeps its full surface form; only the TYPE is
        #    generalised. Longest head wins, so the most specific true class is chosen:
        #        "Syk signaling pathway" -> signaling pathway   (not "pathway")
        #        "the Syk inhibitor"     -> inhibitor
        head = self.manager.explain_class_head_lookup(canonical, min_tokens=1)
        attempts.append(head)
        if head["status"] == "matched":
            dropped = head.get("dropped_modifiers") or ""
            return {"iri": head["iri"],
                    "method": "domain_head" if dropped else "domain_exact",
                    # A generalisation is a weaker claim than an exact hit; say so.
                    "confidence": 0.85 if dropped else 0.95,
                    "attempts": attempts,
                    "matched_head": head.get("matched_head"),
                    "generalized_from": canonical if dropped else None}

        # 3. The coarse semantic type, via the registry -- but ONLY when the typer
        #    actually recognised something. SemanticTyper returns
        #    INFORMATION_CONTENT_ENTITY at confidence 0.5 with method "generic_fallback"
        #    to mean "I don't know", and the registry was turning that shrug into a
        #    confident 0.9 assertion that a protein is an item of information. A guess
        #    must not be laundered into a fact: when the typer defaulted, drop through
        #    to the upper-ontology ladder, which climbs to something that is at least true.
        # Both of these read the semantic type as a positive identification, so both are
        # gated: SemanticTyper.ontology_terms() also falls back to "information content
        # entity" for anything it does not recognise, which is the same laundering by
        # another route -- and one the production ontology only masked by accident,
        # because that term happens to be ambiguous there.
        if typing_recognised:
            preferred = SEMANTIC_CLASS_IRIS.get(semantic_type)
            if preferred and URIRef(preferred) in loaded_classes:
                return {"iri": preferred, "method": "semantic_type_registry",
                        "confidence": 0.9, "attempts": attempts}

            # 4. Ontology terms associated with that semantic type.
            for term in self.typer.ontology_terms(semantic_type):
                lookup = self.manager.explain_class_lookup(term)
                attempts.append(lookup)
                if lookup["status"] == "matched":
                    return {"iri": lookup["iri"], "method": "semantic_type_exact",
                            "confidence": 0.9, "attempts": attempts}
        # Similarity candidates are diagnostic only: semantic type controls
        # class selection and prevents RO/unrelated classes from winning.
        lexical = self.manager.class_candidates(canonical)

        # 5. Nothing loaded matched. Before giving up to a generic upper class, ask the
        #    external grounder (OAK/OLS) whether the wider OBO world knows this term.
        #    A real grounding to PR/CHEBI/GO/MONDO is vastly more useful than BFO:entity.
        if self.grounder is not None:
            grounded = self.grounder.ground(canonical)
            if grounded and grounded.get("iri"):
                self.manager.register_external_class(grounded["iri"], grounded.get("label"))
                return {"iri": grounded["iri"], "method": "oak_grounded",
                        "confidence": 0.8, "attempts": attempts,
                        "grounded_curie": grounded.get("curie"),
                        "grounded_ontology": grounded.get("prefix")}

        # 6. Last resort: the most general class that is still TRUE of this entity.
        #
        # This used to try "information content entity" FIRST, for everything. That is
        # not a vague answer, it is a false one: p53 is a protein -- a material thing --
        # and calling it an item of information corrupts anything that later reasons over
        # the graph. A fallback must be a genuine ancestor, so pick the upper-ontology
        # branch that matches what the entity actually IS, and only land on ICE for
        # things that really are information.
        # A defaulted semantic type carries no information about WHICH KIND of thing this
        # is, so it must not select a branch. SemanticTyper's default happens to be
        # INFORMATION_CONTENT_ENTITY, and routing that down the information ladder would
        # reintroduce the very bug this is fixing: an unknown protein would be asserted to
        # be an item of information. Unknown means unknown -- climb the neutral ladder.
        ladder = generic_fallback_terms(semantic_type if typing_recognised else None)
        for rung, term in enumerate(ladder):
            lookup = self.manager.explain_class_lookup(
                term, prefer_namespace=CANONICAL_UPPER_NAMESPACE
            )
            attempts.append(lookup)
            if lookup["status"] == "matched":
                # The further up the ladder we had to climb, the less it says.
                confidence = 0.6 if rung == 0 else max(0.25, 0.55 - 0.1 * rung)
                return {"iri": lookup["iri"], "method": "upper_ontology_fallback",
                        "confidence": confidence, "attempts": attempts,
                        "candidates": lexical, "ladder_rung": term}

        # Nothing on the ladder resolved -- the loaded ontologies do not even carry BFO's
        # root. The entity must still enter the graph (dropping it loses a fact), but it
        # must not be given a class we cannot justify: this used to return
        # sorted(classes)[0], an ARBITRARY class, and asserted it as the type of hundreds
        # of entities. owl:Thing is the one type that is true of anything by definition,
        # so it says "we could not classify this" without lying about what it is.
        return {"iri": str(OWL.Thing), "method": "owl_thing_fallback", "confidence": 0.1,
                "attempts": attempts, "candidates": lexical}

    def map(self, surface: str, scope: str, frame_type: str | None = None, resolved_text: str | None = None) -> dict:
        canonical = self.canonicalizer.canonicalize(surface, resolved_text)
        typing = self.typer.type_entity(canonical["canonical_form"], frame_type)
        # "generic_fallback" / "frame_context_fallback" are SemanticTyper shrugging, not
        # recognising -- the registry must not treat them as a positive identification.
        typing_recognised = typing.get("method") == "deterministic_rules"
        selection = self._select_class(
            typing["semantic_type"], canonical["canonical_form"], typing_recognised
        )
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
            # How specific is the class we managed to give it? An exact or head match is a
            # real type; an upper-ontology rung or owl:Thing means "we could not classify
            # this", and downstream consumers need to be able to tell the difference.
            "class_specificity": {
                "domain_exact": "exact",
                "domain_head": "generalized",
                "oak_grounded": "grounded",
                "semantic_type_registry": "typed",
                "semantic_type_exact": "typed",
                "upper_ontology_fallback": "upper_ontology",
                "owl_thing_fallback": "unclassified",
            }.get(selection["method"], "unknown"),
            "generalized_from": selection.get("generalized_from"),
            "mapping_method": selection["method"],
            "mapping_confidence": confidence,
            "review_required": confidence < 0.6,
            "mapping_candidates": selection.get("candidates", []),
            "class_selection_audit": selection["attempts"],
            "datatype_assertions": metadata_assertions,
        }
