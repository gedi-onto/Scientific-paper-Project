"""OntoGPT/OAK grounding: prefix ranking, caching, graceful degradation, integration.

No network and no oaklib needed -- the OAK adapter is stubbed, so these run anywhere.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag_stage1.grounding import OakGrounder, _curie_to_iri, _normalise


class FakeAdapter:
    """Stands in for an oaklib adapter: label/synonym map -> CURIEs."""

    def __init__(self, index, labels):
        self._index = {k.casefold(): v for k, v in index.items()}
        self._labels = labels
        self.calls = 0

    def basic_search(self, term, config=None):
        self.calls += 1
        return list(self._index.get(term.casefold(), []))

    def label(self, curie):
        return self._labels.get(curie)


def grounder_with(index, labels, **kw):
    g = OakGrounder(**kw)
    g._adapter = FakeAdapter(index, labels)  # inject, bypassing get_adapter/network
    return g


class RankingTests(unittest.TestCase):
    def test_prefers_more_specific_ontology(self):
        # "catenin beta-1" exists in both PRO (protein) and NCIT (thesaurus). PRO wins.
        g = grounder_with(
            {"beta-catenin": ["NCIT:C17478", "PR:000002198"]},
            {"PR:000002198": "catenin beta-1", "NCIT:C17478": "Catenin Beta-1"},
        )
        r = g.ground("beta-catenin")
        self.assertEqual(r["curie"], "PR:000002198")
        self.assertEqual(r["prefix"], "PR")
        self.assertEqual(r["iri"], "http://purl.obolibrary.org/obo/PR_000002198")

    def test_rejects_hits_outside_accepted_ontologies(self):
        # A match only in a niche ontology we did not ask for is dropped, not accepted.
        g = grounder_with({"cox 2": ["TXPO:0004472"]}, {"TXPO:0004472": "COX-2"})
        self.assertIsNone(g.ground("cox 2"))

    def test_no_hit_returns_none(self):
        g = grounder_with({}, {})
        self.assertIsNone(g.ground("we"))
        self.assertIsNone(g.ground(""))


class CacheTests(unittest.TestCase):
    def test_second_lookup_does_not_hit_the_adapter(self):
        g = grounder_with({"apoptosis": ["GO:0006915"]}, {"GO:0006915": "apoptotic process"})
        g.ground("apoptosis")
        g.ground("apoptosis")
        g.ground("Apoptosis")  # same after normalisation
        self.assertEqual(g._adapter.calls, 1, "repeat mentions must be served from cache")

    def test_cache_persists_across_instances(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            g1 = grounder_with(
                {"colorectal cancer": ["MONDO:0005575"]},
                {"MONDO:0005575": "colorectal cancer"},
                cache_path=path,
            )
            self.assertIsNotNone(g1.ground("colorectal cancer"))
            g1.ground("nonexistent thing")  # a None result must also be remembered
            g1.save_cache()

            # A new grounder with NO adapter must still answer from the saved cache.
            g2 = OakGrounder(cache_path=path)
            g2._adapter_failed = True  # ensure no adapter is built
            hit = g2.ground("colorectal cancer")
            self.assertEqual(hit["curie"], "MONDO:0005575")
            self.assertIsNone(g2.ground("nonexistent thing"))


class DegradationTests(unittest.TestCase):
    def test_missing_oaklib_degrades_to_none_never_raises(self):
        # No adapter injected and oaklib not importable in the test env -> None, no crash.
        g = OakGrounder()
        g._adapter_failed = True
        self.assertIsNone(g.ground("beta-catenin"))


class IntegrationTests(unittest.TestCase):
    def test_grounder_types_an_entity_the_ontology_could_not(self):
        from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS
        from graphrag_stage1.entity_mapper import EntityMapper
        from graphrag_stage1.ontology_manager import OntologyManager

        OBO = Namespace("http://purl.obolibrary.org/obo/")
        manager = OntologyManager()
        graph = Graph()
        # Only an upper ontology is loaded -- no protein classes at all.
        graph.add((OBO.BFO_0000001, RDF.type, OWL.Class))
        graph.add((OBO.BFO_0000001, RDFS.label, Literal("entity")))
        graph.add((OBO.BFO_0000040, RDF.type, OWL.Class))
        graph.add((OBO.BFO_0000040, RDFS.label, Literal("material entity")))
        manager.graph = graph
        manager._build_indexes()

        grounder = grounder_with(
            {"beta-catenin": ["PR:000002198"]}, {"PR:000002198": "catenin beta-1"}
        )
        mapper = EntityMapper(manager, grounder=grounder)
        selection = mapper._select_class("INFORMATION_CONTENT_ENTITY", "beta-catenin",
                                         typing_recognised=False)
        self.assertEqual(selection["method"], "oak_grounded")
        self.assertEqual(selection["iri"], "http://purl.obolibrary.org/obo/PR_000002198")
        # The grounded class must be registered so an instance can be typed with it.
        self.assertIsNotNone(manager.find_class("catenin beta-1"))

    def test_without_grounder_the_same_entity_falls_back_to_upper(self):
        from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS
        from graphrag_stage1.entity_mapper import EntityMapper
        from graphrag_stage1.ontology_manager import OntologyManager

        OBO = Namespace("http://purl.obolibrary.org/obo/")
        manager = OntologyManager()
        graph = Graph()
        graph.add((OBO.BFO_0000001, RDF.type, OWL.Class))
        graph.add((OBO.BFO_0000001, RDFS.label, Literal("entity")))
        manager.graph = graph
        manager._build_indexes()

        mapper = EntityMapper(manager, grounder=None)
        selection = mapper._select_class("INFORMATION_CONTENT_ENTITY", "beta-catenin",
                                         typing_recognised=False)
        self.assertNotEqual(selection["method"], "oak_grounded")


class HelperTests(unittest.TestCase):
    def test_curie_to_iri(self):
        self.assertEqual(_curie_to_iri("GO:0006915"), "http://purl.obolibrary.org/obo/GO_0006915")
        self.assertEqual(_curie_to_iri("NCIT:C17478"), "http://purl.obolibrary.org/obo/NCIT_C17478")
        self.assertEqual(_curie_to_iri("NCBITaxon:10090"),
                         "http://purl.obolibrary.org/obo/NCBITaxon_10090")

    def test_normalise(self):
        self.assertEqual(_normalise("  The Beta-Catenin!! "), "the beta catenin")


if __name__ == "__main__":
    unittest.main()
