import json
import copy
import unittest
from rdflib import Graph
from pathlib import Path
from tempfile import TemporaryDirectory

from ontology_manager import OntologyManager
from canonicalizer import Canonicalizer
from semantic_typer import SemanticTyper
from stage3_ontology_mapper import stage3_pipeline
from stage3_neptune_export import export_neptune_nquads
from stage2_semantic_frames import postprocess_frame


CORE_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <urn:test:core:> .
ex:CoreClass a owl:Class ; rdfs:label "Core Class" .
"""

IAO_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex: <urn:test:iao:> .
ex:MechanismFrame a owl:Class ; skos:altLabel "MECHANISM" .
ex:MeasurementFrame a owl:Class ; skos:altLabel "MEASUREMENT" .
ex:SupportFrame a owl:Class ; skos:altLabel "SUPPORT" .
ex:isAbout a owl:ObjectProperty ; skos:altLabel "is about" .
ex:supports a owl:ObjectProperty ; skos:altLabel "SUPPORT" .
ex:sourceStatementId a owl:DatatypeProperty ; skos:altLabel "source statement id" .
ex:sourceText a owl:DatatypeProperty ; skos:altLabel "source text" .
ex:measurementValue a owl:DatatypeProperty ; skos:altLabel "measurement value" .
ex:measurementUnit a owl:DatatypeProperty ; skos:altLabel "measurement unit" .
ex:paragraphId a owl:DatatypeProperty ; skos:altLabel "source paragraph id" .
ex:page a owl:DatatypeProperty ; skos:altLabel "source page" .
ex:charStart a owl:DatatypeProperty ; skos:altLabel "source character start" .
ex:charEnd a owl:DatatypeProperty ; skos:altLabel "source character end" .
ex:sourceUri a owl:DatatypeProperty ; skos:altLabel "source uri" .
"""

DOMAIN_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex: <urn:test:domain:> .
ex:FusionAlgorithm a owl:Class ; skos:altLabel "fusion algorithm" .
ex:Sensor a owl:Class ; skos:altLabel "sensor" .
ex:ViralLoad a owl:Class ; skos:altLabel "viral load" .
ex:Adjustment a owl:ObjectProperty ; skos:altLabel "adjusts" ;
  rdfs:domain ex:FusionAlgorithm ; rdfs:range ex:Sensor .
"""


def make_ontologies(root: Path) -> None:
    for name, content in (
        ("BFO", CORE_TTL), ("IAO", IAO_TTL), ("CCO", CORE_TTL),
        ("Domain", DOMAIN_TTL),
    ):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / f"{name.lower()}.ttl").write_text(content, encoding="utf-8")


def stage2_mechanism(action="PASS_TO_ONTOLOGY_MAPPING"):
    text = "The fusion algorithm adjusts sensor weighting."
    return {
        "paragraph_id": "p1",
        "frames": [{
            "statement_id": "p1:u1",
            "source_text": text,
            "provenance": {
                "paragraph_id": "p1", "document_id": "doc1", "page": 2,
                "source_uri": "gold://doc1", "char_start": 10, "char_end": 57,
            },
            "stage2_frame": {
                "frame_type": "MECHANISM",
                "semantic_frame": {
                    "primary_entity": "fusion algorithm",
                    "secondary_entity": "sensor",
                    "actor": "fusion algorithm",
                    "affected_entity": "sensor",
                    "measurement": {"value": None, "unit": None},
                    "evidence_statement_ids": [],
                },
                "candidate_entities": ["fusion algorithm", "sensor"],
                "candidate_relations": [{
                    "subject": "fusion algorithm", "relation": "adjusts",
                    "object": "sensor", "qualifier": "weighting",
                    "evidence_text": text,
                }],
                "reference_resolutions": [],
                "validation": {"automation_action": action},
            },
        }],
    }


class OntologyManagerTests(unittest.TestCase):
    def test_reports_missing_and_ambiguous_lookups(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_ontologies(root)
            extra = root / "Domain" / "duplicate.ttl"
            extra.write_text(
                """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
<urn:test:other-sensor> a owl:Class ; skos:altLabel "sensor" .
""",
                encoding="utf-8",
            )
            manager = OntologyManager(root).load_all()
            self.assertEqual(manager.explain_class_lookup("sensor")["status"], "ambiguous")
            self.assertEqual(manager.explain_class_lookup("not present")["status"], "missing")

    def test_loads_flat_core_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bfo.ttl").write_text(CORE_TTL, encoding="utf-8")
            (root / "iao.ttl").write_text(IAO_TTL, encoding="utf-8")
            (root / "CommonCoreOntologiesMerged.ttl").write_text(CORE_TTL, encoding="utf-8")
            (root / "custom-domain.ttl").write_text(DOMAIN_TTL, encoding="utf-8")
            manager = OntologyManager(root).load_all()
            self.assertEqual(len(manager.loaded_files), 4)

    def test_loads_optional_relations_module(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_ontologies(root)
            relations = root / "Relations"
            relations.mkdir()
            (relations / "relations.ttl").write_text(
                """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex: <urn:test:relations:> .
ex:causes a owl:ObjectProperty ; skos:altLabel "causes" .
""",
                encoding="utf-8",
            )
            manager = OntologyManager(root).load_all()
            self.assertTrue(manager.profile()["relations_module_loaded"])
            self.assertIsNotNone(manager.find_object_property("causes"))

    def test_requires_all_core_ontologies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Domain").mkdir()
            (root / "Domain" / "domain.ttl").write_text(DOMAIN_TTL, encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                OntologyManager(root).load_all()

    def test_load_lookup_reasoning_boundary_and_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_ontologies(root)
            manager = OntologyManager(root).load_all()
            self.assertIsNotNone(manager.find_class("MECHANISM"))
            self.assertIsNotNone(manager.find_object_property("adjusts"))
            self.assertEqual(manager.find_property("adjusts"), manager.find_object_property("adjusts"))
            self.assertIsNotNone(manager.find_datatype_property("measurement value"))
            resolved = manager.resolve_object_property(
                "adjusts",
                manager.find_class("fusion algorithm"),
                manager.find_class("sensor"),
            )
            self.assertEqual(resolved["status"], "matched")
            incompatible = manager.resolve_object_property(
                "adjusts",
                manager.find_class("sensor"),
                manager.find_class("fusion algorithm"),
            )
            self.assertEqual(incompatible["status"], "incompatible")
            self.assertIn("MechanismFrame", manager.export(rdf_format="turtle"))
            with self.assertRaises(RuntimeError):
                manager.reason()


class Stage3MapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_ontologies(self.root)
        self.manager = OntologyManager(self.root).load_all()

    def tearDown(self):
        self.temp.cleanup()

    def test_maps_instances_relations_and_provenance_deterministically(self):
        first = stage3_pipeline(stage2_mechanism(), self.manager)
        second = stage3_pipeline(stage2_mechanism(), self.manager)
        mapped = first["mapped_frames"][0]
        self.assertTrue(mapped["ready_for_reasoning"])
        self.assertEqual(mapped["mapping_status"], "mapped")
        self.assertEqual(len(mapped["ontology_individuals"]), 2)
        self.assertTrue(any(item["source_relation"] == "adjusts" for item in mapped["object_property_assertions"]))
        self.assertTrue(any(item["value"] == "p1:u1" for item in mapped["datatype_assertions"]))
        audits = mapped["mapping_audit"]["property_axiom_audits"]
        self.assertTrue(any(item["term"] == "adjusts" and item["status"] == "matched" for item in audits))
        self.assertEqual(
            mapped["frame_instance"]["id"],
            second["mapped_frames"][0]["frame_instance"]["id"],
        )
        self.assertIn("@context", first["jsonld"])
        self.assertTrue(first["jsonld"]["@graph"])
        Graph().parse(data=json.dumps(first["jsonld"]), format="json-ld")

    def test_reference_resolution_controls_canonical_identity(self):
        output = stage2_mechanism()
        frame = output["frames"][0]["stage2_frame"]
        frame["candidate_entities"] = ["the algorithm", "sensor"]
        frame["semantic_frame"]["primary_entity"] = "the algorithm"
        frame["semantic_frame"]["actor"] = "the algorithm"
        frame["candidate_relations"][0]["subject"] = "the algorithm"
        frame["reference_resolutions"] = [{
            "original": "the algorithm",
            "resolved_text": "fusion algorithm",
            "confidence": 0.9,
        }]
        result = stage3_pipeline(output, self.manager)
        entity = result["mapped_frames"][0]["ontology_individuals"][0]
        self.assertEqual(entity["canonical_form"], "fusion algorithm")
        self.assertIn("the algorithm", entity["aliases"])
        self.assertIsNotNone(entity["instance_id"])

    def test_canonicalization_and_semantic_typing_precede_lookup(self):
        canonical = Canonicalizer().canonicalize("Global Positioning System Receiver")
        typed = SemanticTyper().type_entity(canonical["canonical_form"])
        self.assertEqual(canonical["canonical_form"], "GPS receiver")
        self.assertEqual(typed["semantic_type"], "SENSOR")

    def test_navigation_regressions_use_semantic_precedence_and_clean_aliases(self):
        canonicalizer = Canonicalizer()
        typer = SemanticTyper()
        self.assertEqual(canonicalizer.canonicalize("and GPS receiver")["canonical_form"], "GPS receiver")
        self.assertEqual(canonicalizer.canonicalize("These results")["canonical_form"], "experimental results")
        self.assertEqual(canonicalizer.canonicalize("The researchers")["canonical_form"], "scientist")
        self.assertEqual(typer.type_entity("stereo cameras")["semantic_type"], "SENSOR")
        self.assertEqual(typer.type_entity("probabilistic sensor-fusion algorithm")["semantic_type"], "ALGORITHM")
        self.assertEqual(typer.type_entity("claim that sensor fusion improves navigation")["semantic_type"], "CLAIM")

    def test_physical_and_representational_artifacts_are_distinguished(self):
        typer = SemanticTyper()
        self.assertEqual(typer.type_entity("unmanned aerial vehicles", "METHOD")["semantic_type"], "VEHICLE")
        self.assertEqual(typer.type_entity("autonomous underwater vehicle")["semantic_type"], "VEHICLE")
        self.assertEqual(typer.type_entity("terrain models")["semantic_type"], "REPRESENTATIONAL_MODEL")
        self.assertEqual(typer.type_entity("satellite imagery")["semantic_type"], "REPRESENTATIONAL_MODEL")
        self.assertEqual(typer.type_entity("planning algorithm")["semantic_type"], "ALGORITHM")
        self.assertEqual(typer.type_entity("weather prediction service")["semantic_type"], "SYSTEM")
        self.assertEqual(typer.type_entity("system efficiency")["semantic_type"], "MEASURABLE_QUALITY")
        self.assertEqual(typer.type_entity("mission completion time")["semantic_type"], "MEASURABLE_QUALITY")
        self.assertEqual(typer.type_entity("mission success rate")["semantic_type"], "MEASURABLE_QUALITY")

    def test_compound_entity_is_split_into_individuals(self):
        output = stage2_mechanism()
        output["frames"][0]["stage2_frame"]["candidate_entities"].append(
            "GPS, camera and lidar"
        )
        mapped = stage3_pipeline(output, self.manager)["mapped_frames"][0]
        surfaces = {item["surface_form"] for item in mapped["ontology_individuals"]}
        self.assertTrue({"GPS", "camera", "lidar"}.issubset(surfaces))
        self.assertNotIn("GPS, camera and lidar", surfaces)

    def test_reasoning_profile_and_modular_assertion_sections(self):
        mapped = stage3_pipeline(stage2_mechanism(), self.manager)["mapped_frames"][0]
        self.assertIn("reasoning_profile", mapped)
        self.assertIn("ontology_individuals", mapped)
        self.assertNotIn("entity_instances", mapped)
        self.assertIn("class_assertions", mapped)
        self.assertIn("measurement_assertions", mapped)
        self.assertIn("provenance_record", mapped)
        self.assertFalse(any(item["inferred"] for item in mapped["class_assertions"]))

    def test_exports_shacl_valid_neptune_nquads(self):
        output = stage3_pipeline(stage2_mechanism(), self.manager)
        destination = self.root / "publication.nq"
        quarantine = self.root / "quarantine.ttl"
        report = self.root / "validation.txt"
        metadata = export_neptune_nquads(
            output,
            destination,
            quarantine_destination=quarantine,
            report_destination=report,
            shapes_path=Path("ontologies/Alignment/stage3-publication-shapes.ttl"),
        )
        self.assertTrue(destination.exists())
        self.assertIn("urn:graphrag:graph:", destination.read_text(encoding="utf-8"))
        self.assertEqual(metadata["format"], "nquads")
        self.assertGreater(metadata["validated_triple_count"], 0)

    def test_preserves_exact_measurement(self):
        output = stage2_mechanism()
        wrapper = output["frames"][0]
        wrapper["stage2_frame"]["frame_type"] = "MEASUREMENT"
        wrapper["stage2_frame"]["semantic_frame"].update({
            "primary_entity": "viral load",
            "secondary_entity": None,
            "actor": None,
            "affected_entity": None,
            "measurement": {"value": "42", "unit": "percent"},
        })
        wrapper["stage2_frame"]["candidate_entities"] = ["viral load"]
        wrapper["stage2_frame"]["candidate_relations"] = []
        mapped = stage3_pipeline(output, self.manager)["mapped_frames"][0]
        self.assertEqual(mapped["measurement"], {"value": "42", "unit": "percent"})
        values = {str(item["value"]) for item in mapped["datatype_assertions"]}
        self.assertIn("42", values)
        self.assertIn("percent", values)

    def test_stage2_nonpass_is_not_mapped(self):
        mapped = stage3_pipeline(
            stage2_mechanism("REPROCESS_WITH_CONTEXT"), self.manager
        )["mapped_frames"][0]
        self.assertEqual(mapped["mapping_status"], "skipped_by_stage2_routing")
        self.assertFalse(mapped["ready_for_reasoning"])

    def test_unknown_entity_receives_audited_fallback_individual(self):
        output = stage2_mechanism()
        frame = output["frames"][0]["stage2_frame"]
        frame["candidate_entities"].append("unknown scientific entity")
        mapped = stage3_pipeline(output, self.manager)["mapped_frames"][0]
        entity = next(
            item for item in mapped["ontology_individuals"]
            if item["surface_form"] == "unknown scientific entity"
        )
        self.assertEqual(entity["mapping_status"], "mapped")
        self.assertIsNotNone(entity["instance_id"])
        self.assertEqual(entity["semantic_type"], "INFORMATION_CONTENT_ENTITY")
        self.assertTrue(entity["review_required"])

    def test_maps_statement_graph_edges_between_frame_instances(self):
        output = stage2_mechanism()
        second = copy.deepcopy(output["frames"][0])
        second["statement_id"] = "p1:u2"
        output["frames"].append(second)
        output["statement_graph"] = {
            "edges": [{"from": "p1:u1", "to": "p1:u2", "type": "SUPPORT"}]
        }
        result = stage3_pipeline(output, self.manager)
        assertions = result["mapped_frames"][0]["object_property_assertions"]
        self.assertTrue(any(
            item.get("assertion_kind") == "stage1_discourse_edge"
            and item["source_relation"] == "SUPPORT"
            for item in assertions
        ))

    def test_stage1_stage2_stage3_contract(self):
        text = "The fusion algorithm adjusts sensor weighting."
        stage1_statement = {
            "id": "p1:u1", "text": text, "predicate": "adjusts",
            "arguments": [
                {"role": "arg1", "text": "fusion algorithm"},
                {"role": "arg2", "text": "sensor"},
            ],
            "facets": {
                "role": "CONTENT", "proposition_type": "PROCESS",
                "relation": "MECHANISM", "modality": "ASSERTED",
                "polarity": "POSITIVE",
            },
            "provenance": {
                "paragraph_id": "p1", "document_id": "doc1", "page": 1,
                "source_uri": "gold://doc1", "char_start": 0, "char_end": len(text),
            },
        }
        stage1 = {"paragraph_id": "p1", "statements": [stage1_statement], "relations": []}
        proposed = {
            "frame_type": "MECHANISM", "source_statement": text,
            "semantic_frame": {
                "primary_entity": "fusion algorithm", "secondary_entity": "sensor",
                "property": "adjusts", "value": "weighting", "process": "adjusts",
                "condition": None, "basis": None, "context": None,
                "measurement_value": None, "unit": None,
            },
            "candidate_entities": ["fusion algorithm", "sensor"],
            "candidate_relations": [{
                "subject": "fusion algorithm", "relation": "adjusts", "object": "sensor",
                "qualifier": "weighting", "evidence_text": text,
            }],
            "reference_resolutions": [],
            "ontology_readiness": {"ready_for_mapping": True, "mapping_risk": "LOW", "reason": "complete"},
            "validation": {"missing_required_fields": [], "ambiguous_terms": [], "invented_information": False, "automation_action": "PASS_TO_ONTOLOGY_MAPPING"},
            "confidence": 0.95, "reason": "complete",
        }
        stage2_frame = postprocess_frame(proposed, "MECHANISM", stage1_statement, stage1)
        stage2 = {
            "paragraph_id": "p1",
            "frames": [{
                "statement_id": "p1:u1", "source_text": text,
                "provenance": stage1_statement["provenance"], "stage2_frame": stage2_frame,
            }],
            "statement_graph": {"nodes": {}, "edges": []},
        }
        result = stage3_pipeline(stage2, self.manager)
        self.assertTrue(result["mapped_frames"][0]["ready_for_reasoning"])


if __name__ == "__main__":
    unittest.main()
