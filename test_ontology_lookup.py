"""Class lookup: head-of-phrase matching, OBO synonyms, and namespace disambiguation."""

import unittest

from rdflib import Graph, Namespace, OWL, RDF, RDFS, Literal, URIRef

from graphrag_stage1.ontology_manager import OBO, OBOINOWL, OntologyManager

EX = Namespace("http://example.org/")
ALIGN = Namespace("https://w3id.org/graphrag/alignment/v1/")


def manager_with(triples) -> OntologyManager:
    manager = OntologyManager()
    graph = Graph()
    for t in triples:
        graph.add(t)
    manager.graph = graph
    manager._build_indexes()
    return manager


class HeadPhraseLookupTests(unittest.TestCase):
    """"Syk signaling pathway" IS a "signaling pathway" -- the head carries the type."""

    def setUp(self):
        self.manager = manager_with([
            (EX.SignalingPathway, RDF.type, OWL.Class),
            (EX.SignalingPathway, RDFS.label, Literal("signaling pathway")),
            (EX.Gene, RDF.type, OWL.Class),
            (EX.Gene, RDFS.label, Literal("gene")),
        ])

    def test_exact_phrase_still_matches(self):
        r = self.manager.explain_class_head_lookup("signaling pathway")
        self.assertEqual(r["status"], "matched")
        self.assertEqual(r["iri"], str(EX.SignalingPathway))

    def test_modifiers_are_dropped_to_find_the_head(self):
        r = self.manager.explain_class_head_lookup("Syk signaling pathway")
        self.assertEqual(r["status"], "matched")
        self.assertEqual(r["iri"], str(EX.SignalingPathway))
        self.assertEqual(r["matched_head"], "signaling pathway")
        self.assertEqual(r["dropped_modifiers"], "syk")

    def test_leading_determiners_are_ignored(self):
        r = self.manager.explain_class_head_lookup("the Syk signaling pathway")
        self.assertEqual(r["iri"], str(EX.SignalingPathway))

    def test_unrelated_phrase_does_not_match(self):
        self.assertEqual(
            self.manager.explain_class_head_lookup("fostamatinib")["status"], "missing"
        )

    def test_min_tokens_blocks_over_general_single_word_matches(self):
        # With min_tokens=2 a bare head noun is not accepted, so a long phrase cannot
        # collapse onto an over-general class.
        r = self.manager.explain_class_head_lookup("core driving gene", min_tokens=2)
        self.assertEqual(r["status"], "missing")
        r = self.manager.explain_class_head_lookup("core driving gene", min_tokens=1)
        self.assertEqual(r["iri"], str(EX.Gene))


class OboSynonymTests(unittest.TestCase):
    def test_obo_exact_synonyms_are_indexed(self):
        manager = manager_with([
            (EX.Protein, RDF.type, OWL.Class),
            (EX.Protein, RDFS.label, Literal("protein")),
            (EX.Protein, OBOINOWL.hasExactSynonym, Literal("polypeptide")),
            (EX.Protein, OBO.IAO_0000118, Literal("gene product")),
        ])
        for alias in ("protein", "polypeptide", "gene product"):
            self.assertEqual(
                manager.explain_class_lookup(alias)["iri"], str(EX.Protein), alias
            )

    def test_loose_obo_synonyms_are_NOT_indexed(self):
        # hasRelatedSynonym is deliberately loose in OBO. Indexing it made previously
        # unique terms ambiguous and broke frame-class resolution entirely.
        manager = manager_with([
            (EX.Protein, RDF.type, OWL.Class),
            (EX.Protein, RDFS.label, Literal("protein")),
            (EX.Protein, OBOINOWL.hasRelatedSynonym, Literal("biomolecule")),
        ])
        self.assertEqual(manager.explain_class_lookup("biomolecule")["status"], "missing")


class NamespaceDisambiguationTests(unittest.TestCase):
    """A frame type is DEFINED in the alignment ontology; an alias elsewhere must not win."""

    def setUp(self):
        self.manager = manager_with([
            (ALIGN.MethodFrame, RDF.type, OWL.Class),
            (ALIGN.MethodFrame, RDFS.label, Literal("METHOD")),
            # An unrelated IAO class that merely carries "method" as an alternative term.
            (OBO.IAO_0000317, RDF.type, OWL.Class),
            (OBO.IAO_0000317, OBO.IAO_0000118, Literal("method")),
        ])

    def test_collision_is_ambiguous_without_a_preference(self):
        r = self.manager.explain_class_lookup("METHOD")
        self.assertEqual(r["status"], "ambiguous")
        self.assertIsNone(r["iri"])

    def test_preferred_namespace_resolves_the_collision(self):
        r = self.manager.explain_class_lookup(
            "METHOD", prefer_namespace="https://w3id.org/graphrag/alignment/"
        )
        self.assertEqual(r["status"], "matched")
        self.assertEqual(r["iri"], str(ALIGN.MethodFrame))

    def test_preference_disambiguates_it_does_not_invent(self):
        # If the preferred namespace has no candidate, ambiguity stands -- no guessing.
        r = self.manager.explain_class_lookup("METHOD", prefer_namespace="http://nope.org/")
        self.assertEqual(r["status"], "ambiguous")
        self.assertIsNone(r["iri"])


class RealOntologyStackTests(unittest.TestCase):
    """The shipped ontologies, loaded for real. No BFO file -- CCO and IAO supply it."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        # The foundational stack now ships inside the package, so the default
        # OntologyManager() root is the packaged copy -- exercise exactly what an
        # installed user gets. Only the domain ontology comes from the repo, because
        # that one is deliberately not bundled.
        domain = Path(__file__).resolve().parent / "ontologies" / "Domain" / "ino_merged.owl"
        if not domain.exists():  # pragma: no cover - example domain file is repo-only
            raise unittest.SkipTest("example domain ontology not present")
        cls.manager = OntologyManager().load_all(domain_ontology=str(domain))

    def test_bfo_classes_are_present_without_a_bfo_file(self):
        # BFO ships no file of its own: CCO and IAO are built on it and republish its
        # classes under the canonical obo/BFO_* IRIs.
        self.assertEqual(self.manager.missing_upper_classes(), [])

    def test_no_legacy_ifomis_bfo_remains(self):
        # The legacy IFOMIS BFO 1.1 named the same concepts under DIFFERENT IRIs, which
        # made every upper-ontology term ambiguous and unusable.
        loaded = {str(c) for v in self.manager.class_index.values() for c in v}
        self.assertFalse([c for c in loaded if "ifomis.org" in c])

    def test_upper_terms_resolve_unambiguously(self):
        for term in ("entity", "continuant", "occurrent", "process",
                     "material entity", "independent continuant"):
            self.assertEqual(
                self.manager.explain_class_lookup(term)["status"], "matched", term
            )


if __name__ == "__main__":
    unittest.main()
