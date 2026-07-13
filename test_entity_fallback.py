"""Class fallback: generalize to a TRUE ancestor, never to a convenient one.

A car with no Car class should become a Vehicle. It should never become an
Information Content Entity, and it should never become whichever class happens to
sort first.
"""

import unittest

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS

from graphrag_stage1.entity_mapper import EntityMapper, generic_fallback_terms
from graphrag_stage1.ontology_manager import OntologyManager

OBO = Namespace("http://purl.obolibrary.org/obo/")
CCO = Namespace("https://www.commoncoreontologies.org/")
EX = Namespace("http://example.org/")


def upper_ontology(*extra):
    """A stand-in BFO/CCO/IAO stack, including the duplicate-BFO trap."""
    triples = [
        (OBO.BFO_0000001, RDF.type, OWL.Class),
        (OBO.BFO_0000001, RDFS.label, Literal("entity")),
        (OBO.BFO_0000040, RDF.type, OWL.Class),
        (OBO.BFO_0000040, RDFS.label, Literal("material entity")),
        (OBO.BFO_0000015, RDF.type, OWL.Class),
        (OBO.BFO_0000015, RDFS.label, Literal("process")),
        (OBO.IAO_0000030, RDF.type, OWL.Class),
        (OBO.IAO_0000030, RDFS.label, Literal("information content entity")),
        (CCO.ont00001017, RDF.type, OWL.Class),
        (CCO.ont00001017, RDFS.label, Literal("agent")),
        # The legacy IFOMIS BFO that ships alongside the modern one: same concepts,
        # different IRIs, so every upper term is ambiguous unless disambiguated.
        (Namespace("http://www.ifomis.org/bfo/1.1#").Entity, RDF.type, OWL.Class),
        (Namespace("http://www.ifomis.org/bfo/1.1#").Entity, RDFS.label, Literal("entity")),
        # Sorts before everything -- the class the old fallback used to hand out.
        (OBO.APOLLO_SV_00000008, RDF.type, OWL.Class),
        (OBO.APOLLO_SV_00000008, RDFS.label, Literal("apollo thing")),
    ]
    triples.extend(extra)
    manager = OntologyManager()
    graph = Graph()
    for t in triples:
        graph.add(t)
    manager.graph = graph
    manager._build_indexes()
    return manager


class LadderTests(unittest.TestCase):
    def test_material_things_never_climb_into_the_information_branch(self):
        for stype in ("PERSON", "GROUP", "DEVICE", "VEHICLE"):
            self.assertNotIn("information content entity", generic_fallback_terms(stype), stype)

    def test_processes_climb_the_occurrent_branch(self):
        self.assertIn("process", generic_fallback_terms("PROCESS"))
        self.assertNotIn("material entity", generic_fallback_terms("PROCESS"))

    def test_cco_midlevel_comes_before_abstract_bfo(self):
        ladder = generic_fallback_terms("GROUP")
        self.assertLess(ladder.index("agent"), ladder.index("entity"),
                        "a useful CCO class must be tried before BFO's abstract root")

    def test_unknown_type_gets_the_neutral_ladder_only(self):
        # Unknown means unknown: do not pick a branch we cannot justify.
        self.assertEqual(generic_fallback_terms("SOMETHING_WE_DONT_KNOW"), ("entity",))


class SelectionTests(unittest.TestCase):
    def test_generalizes_to_the_head_when_the_exact_class_is_absent(self):
        # "car" is absent, "vehicle" is present -> the car IS a vehicle.
        manager = upper_ontology(
            (EX.Vehicle, RDF.type, OWL.Class),
            (EX.Vehicle, RDFS.label, Literal("vehicle")),
        )
        selection = EntityMapper(manager)._select_class("VEHICLE", "delivery vehicle")
        self.assertEqual(selection["iri"], str(EX.Vehicle))
        self.assertEqual(selection["method"], "domain_head")

    def test_a_shrug_is_not_laundered_into_an_information_content_entity(self):
        # SemanticTyper defaults to INFORMATION_CONTENT_ENTITY to mean "I don't know".
        # An unrecognised entity must NOT be asserted to be an item of information.
        manager = upper_ontology()
        selection = EntityMapper(manager)._select_class(
            "INFORMATION_CONTENT_ENTITY", "p53", typing_recognised=False
        )
        self.assertNotEqual(selection["iri"], str(OBO.IAO_0000030),
                            "a protein must never be typed as information")
        self.assertEqual(selection["iri"], str(OBO.BFO_0000001))  # 'entity' -- true of anything

    def test_a_recognised_information_entity_still_gets_the_information_class(self):
        manager = upper_ontology()
        selection = EntityMapper(manager)._select_class(
            "INFORMATION_CONTENT_ENTITY", "the dataset", typing_recognised=True
        )
        self.assertEqual(selection["iri"], str(OBO.IAO_0000030))

    def test_duplicate_bfo_is_disambiguated_not_rejected(self):
        # "entity" matches both the modern OBO BFO and the legacy IFOMIS copy. Ambiguity
        # used to make the whole ladder miss, dumping entities on an arbitrary class.
        manager = upper_ontology()
        self.assertEqual(manager.explain_class_lookup("entity")["status"], "ambiguous")
        selection = EntityMapper(manager)._select_class(
            "UNKNOWN", "something", typing_recognised=False
        )
        self.assertEqual(selection["iri"], str(OBO.BFO_0000001))

    def test_never_hands_out_an_arbitrary_class(self):
        manager = upper_ontology()
        selection = EntityMapper(manager)._select_class(
            "UNKNOWN", "something", typing_recognised=False
        )
        self.assertNotEqual(selection["iri"], str(OBO.APOLLO_SV_00000008),
                            "must not assign whichever class happens to sort first")

    def test_unclassifiable_entity_lands_on_owl_thing_not_a_wrong_class(self):
        # An ontology with no upper classes at all: the entity must still be minted,
        # but typed as owl:Thing -- true of anything -- rather than something false.
        manager = OntologyManager()
        graph = Graph()
        graph.add((EX.Widget, RDF.type, OWL.Class))
        graph.add((EX.Widget, RDFS.label, Literal("widget")))
        manager.graph = graph
        manager._build_indexes()
        selection = EntityMapper(manager)._select_class(
            "UNKNOWN", "p53", typing_recognised=False
        )
        self.assertEqual(selection["iri"], str(OWL.Thing))
        self.assertEqual(selection["method"], "owl_thing_fallback")


if __name__ == "__main__":
    unittest.main()
