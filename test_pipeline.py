import unittest
import time
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag_stage1.stage1_classifier import (
    is_duplicate_unit,
    preserve_source_numbers,
    reconcile_decomposition,
)
from graphrag_stage1.stage2_semantic_frames import (
    build_statement_graph,
    derive_artifact_type,
    extract_measurements,
    infer_graph_reference_resolutions,
    postprocess_frame,
    stage2_pipeline,
)
from graphrag_stage1.production_support import RoutingQueue, stable_content_id, validate_paragraph
from graphrag_stage1.validate_adjudication import validate_record
from graphrag_stage1.adjudication_workflow import export_packet, finalize, merge_packet


def base_frame(**semantic_overrides):
    semantic = {
        "primary_entity": "sample",
        "secondary_entity": "response",
        "property": "changes",
        "value": "25 percent",
        "process": None,
        "condition": None,
        "basis": None,
        "context": None,
        "measurement_value": "25",
        "unit": "percent",
    }
    semantic.update(semantic_overrides)
    return {
        "frame_type": "CLAIM",
        "source_statement": "",
        "semantic_frame": semantic,
        "candidate_entities": [],
        "candidate_relations": [],
        "reference_resolutions": [],
        "ontology_readiness": {
            "ready_for_mapping": True,
            "mapping_risk": "LOW",
            "reason": "complete",
        },
        "validation": {
            "missing_required_fields": [],
            "ambiguous_terms": [],
            "invented_information": False,
            "automation_action": "PASS_TO_ONTOLOGY_MAPPING",
        },
        "confidence": 0.9,
        "reason": "test",
    }


class Stage1FidelityTests(unittest.TestCase):
    def test_restores_mutated_measurement(self):
        paragraph = "Treatment reduced the response by 25 percent."
        rewritten = "Treatment reduced the response by 2.5 percent."
        repaired, evidence, fidelity = preserve_source_numbers(paragraph, rewritten)
        self.assertIn("25 percent", repaired)
        self.assertTrue(fidelity["numbers_preserved"])
        self.assertEqual(fidelity["repairs"][0]["kind"], "NUMBER_RESTORED")
        self.assertEqual(evidence["verbatim"], paragraph)

    def test_preserves_scientific_notation(self):
        paragraph = "The concentration was 1.2e-4 mol."
        repaired, _, fidelity = preserve_source_numbers(paragraph, paragraph)
        self.assertEqual(repaired, paragraph)
        self.assertTrue(fidelity["numbers_preserved"])

    def test_number_before_sentence_period_is_preserved(self):
        paragraph = "The publication year was 1969."
        _, _, fidelity = preserve_source_numbers(paragraph, paragraph)
        self.assertEqual(fidelity["source_numbers"], ["1969"])
        self.assertTrue(fidelity["numbers_preserved"])

    def test_bibliographic_numbers_are_not_false_mismatches(self):
        paragraph = "Smith, Example, Vol. 1 (1969). The model assumes symmetry."
        unit = "Smith, Example, Vol. 1 (1969)."
        repaired, _, fidelity = preserve_source_numbers(paragraph, unit)
        self.assertEqual(repaired, unit)
        self.assertTrue(fidelity["numbers_preserved"])

    def test_recall_duplicate_is_rejected(self):
        units = [{"text": "Increasing neutron number raises binding energy per nucleon."}]
        self.assertTrue(is_duplicate_unit(
            "Increasing the neutron number raises the binding energy per nucleon.", units
        ))

    def test_spurious_simple_sentence_split_is_collapsed(self):
        paragraph = "Increasing neutron number raises binding energy up to iron."
        decomposed = {
            "units": [
                {"id": "u1", "text": "Increasing neutron number raises binding energy."},
                {"id": "u2", "text": "Binding energy reaches a maximum at iron."},
            ],
            "relations": [{"from": "u2", "to": "u1", "type": "ELABORATION"}],
        }
        result = reconcile_decomposition(paragraph, decomposed)
        self.assertEqual(result["units"], [{"id": "u1", "text": paragraph}])


class Stage2DeterministicTests(unittest.TestCase):
    def test_preserves_all_measurements_in_compound_statement(self):
        text = (
            "During six months, accuracy increased from 46.3 percent to 71.5 percent "
            "while latency remained below 150 milliseconds (r = 0.92)."
        )
        measurements = extract_measurements(text, {"secondary_entity": "accuracy"})
        self.assertEqual([item["value"] for item in measurements], [0.92, 46.3, 71.5, 150, 6])
        self.assertEqual([item["unit"] for item in measurements], [
            "correlation coefficient", "percent", "percent", "milliseconds", "months"
        ])

    def setUp(self):
        self.statement = {
            "id": "p1:u1",
            "text": "Treatment reduced the response by 25 percent.",
            "facets": {"role": "CONTENT"},
        }
        self.output = {"statements": [self.statement]}

    def test_prediction_precedes_mechanism(self):
        statement = {
            "facets": {
                "proposition_type": "PROCESS",
                "relation": "MECHANISM",
                "modality": "PREDICTED",
            }
        }
        self.assertEqual(derive_artifact_type(statement), "PREDICTION")

    def test_parallel_pipeline_preserves_statement_order(self):
        statements = [
            {"id": f"p1:u{index}", "text": f"Statement {index}", "facets": {}}
            for index in range(1, 5)
        ]
        stage1_output = {"paragraph_id": "p1", "statements": statements, "relations": []}

        def fake_extract(statement, _stage1_output):
            # Reverse completion order without slowing the test materially.
            time.sleep((5 - int(statement["id"].split("u")[-1])) * 0.005)
            return {
                "statement_id": statement["id"],
                "stage2_frame": {"validation": {"automation_action": "PASS_TO_ONTOLOGY_MAPPING"}},
            }

        with patch("graphrag_stage1.stage2_semantic_frames.STAGE2_CONCURRENCY", 4), patch(
            "graphrag_stage1.stage2_semantic_frames.extract_with_routing", side_effect=fake_extract
        ):
            result = stage2_pipeline(stage1_output)

        self.assertEqual(
            [frame["statement_id"] for frame in result["frames"]],
            [statement["id"] for statement in statements],
        )
        self.assertEqual(result["audit"]["concurrency"], 4)

    def test_scientific_relation_frames_are_not_flattened(self):
        expectations = {
            "COMPARISON": "COMPARISON",
            "PART_OF": "PART_WHOLE",
            "IS_A": "CLASSIFICATION",
            "DEPENDENCY": "DEPENDENCY",
            "TEMPORAL_ORDER": "TEMPORAL_RELATION",
        }
        for relation, expected in expectations.items():
            statement = {"facets": {
                "proposition_type": "RELATION",
                "relation": relation,
                "modality": "ASSERTED",
            }}
            self.assertEqual(derive_artifact_type(statement), expected)

    def test_scientific_discourse_has_first_class_frames(self):
        examples = {
            "These findings support the architecture.": "SUPPORT",
            "These results are consistent with prior observations.": "CONSISTENCY",
            "The experiment validates the model.": "VALIDATION",
            "The measurement provides evidence for the hypothesis.": "EVIDENCE",
        }
        for text, expected in examples.items():
            self.assertEqual(derive_artifact_type({"text": text, "facets": {}}), expected)

    def test_statement_graph_normalizes_local_edge_ids(self):
        output = {
            "statements": [
                {"id": "p1:u1", "text": "A", "facets": {}},
                {"id": "p1:u2", "text": "These findings support A", "facets": {}},
            ],
            "relations": [{"from": "u2", "to": "u1", "type": "SUPPORT"}],
        }
        graph = build_statement_graph(output)
        self.assertEqual(graph["edges"][0]["from"], "p1:u2")
        self.assertEqual(graph["edges"][0]["to"], "p1:u1")

    def test_statement_graph_links_clauses_with_identical_provenance(self):
        provenance = {"paragraph_id": "p1", "char_start": 10, "char_end": 80}
        output = {
            "statements": [
                {"id": "p1:u1", "text": "The compound inhibits polymerase.", "facets": {}, "provenance": provenance},
                {"id": "p1:u2", "text": "Inhibition prevents replication.", "facets": {}, "provenance": provenance},
            ],
            "relations": [],
        }
        graph = build_statement_graph(output)
        self.assertTrue(any(edge["type"] == "SOURCE_CONTEXT" for edge in graph["edges"]))

    def test_same_source_clause_context_is_authorized(self):
        provenance = {"paragraph_id": "p1", "char_start": 10, "char_end": 100}
        context = {
            "id": "p1:u1", "text": "The compound inhibits viral polymerase activity.",
            "facets": {"role": "CONTENT"}, "provenance": provenance,
        }
        statement = {
            "id": "p1:u2", "text": "Inhibiting viral polymerase activity prevents genome replication.",
            "facets": {"role": "CONTENT", "polarity": "POSITIVE"}, "provenance": provenance,
        }
        frame = base_frame(
            primary_entity="Inhibiting viral polymerase activity",
            secondary_entity="genome replication",
            property="prevents",
            value="prevents",
            context=context["text"],
            measurement_value=None,
            unit=None,
        )
        result = postprocess_frame(
            frame, "CAUSAL_RELATION", statement, {"statements": [context, statement], "relations": []}
        )
        self.assertFalse(any(
            "context" in error for error in result["validation"]["grounding_errors"]
        ))

    def test_graph_resolves_demonstrative_from_support_edge(self):
        output = {
            "statements": [
                {"id": "p1:u1", "text": "Testing reduced errors.", "facets": {}},
                {"id": "p1:u2", "text": "These results support the architecture.", "facets": {}},
            ],
            "relations": [{"from": "u2", "to": "u1", "type": "SUPPORT"}],
        }
        resolutions = infer_graph_reference_resolutions(output["statements"][1], output)
        self.assertEqual(resolutions[0]["evidence_statement_ids"], ["p1:u1"])
        self.assertEqual(resolutions[0]["method"], "stage1_statement_graph")

    def test_graph_resolves_system_and_compound_mentions(self):
        output = {
            "statements": [
                {"id": "p1:u1", "text": "The autonomous vehicle navigation system integrates sensors.", "facets": {}},
                {"id": "p1:u2", "text": "Researchers tested a new antiviral compound.", "facets": {}},
                {"id": "p1:u3", "text": "Future versions of the system will improve.", "facets": {}},
                {"id": "p1:u4", "text": "The compound inhibits polymerase.", "facets": {}},
            ],
            "relations": [],
        }
        system = infer_graph_reference_resolutions(output["statements"][2], output)
        compound = infer_graph_reference_resolutions(output["statements"][3], output)
        self.assertEqual(system[0]["resolved_text"], "autonomous vehicle navigation system")
        self.assertEqual(compound[0]["resolved_text"], "new antiviral compound")

    def test_discourse_frame_is_not_rejected_as_bibliography(self):
        context = {
            "id": "p1:u1", "text": "Treatment reduced viral load.",
            "facets": {"role": "CONTENT"},
        }
        statement = {
            "id": "p1:u2", "text": "These results support the therapeutic hypothesis.",
            "facets": {"role": "REFERENCE", "polarity": "POSITIVE", "modality": "OBSERVED"},
        }
        output = {
            "statements": [context, statement],
            "relations": [{"from": "u2", "to": "u1", "type": "SUPPORT"}],
        }
        frame = base_frame(
            primary_entity="These results",
            secondary_entity="therapeutic hypothesis",
            property="support",
            value="support",
            measurement_value=None,
            unit=None,
        )
        result = postprocess_frame(frame, "SUPPORT", statement, output)
        self.assertNotEqual(result["validation"]["automation_action"], "REJECT_LOW_VALUE")
        self.assertEqual(result["validation"]["ambiguous_terms"], [])

    def test_deterministic_confidence_replaces_llm_only_score(self):
        frame = base_frame(primary_entity="Treatment", secondary_entity="response")
        frame["confidence"] = 0.1
        result = postprocess_frame(frame, "CLAIM", self.statement, self.output)
        self.assertEqual(result["llm_confidence"], 0.1)
        self.assertIn("grounding", result["confidence_components"])
        self.assertNotEqual(result["confidence"], result["llm_confidence"])

    def test_fills_entities_and_normalizes_evidence(self):
        frame = base_frame(primary_entity="Treatment", secondary_entity="response")
        frame["candidate_relations"] = [{
            "subject": "Treatment",
            "relation": "reduced",
            "object": "response",
            "qualifier": "25 percent",
            "evidence_text": "not exact",
        }]
        result = postprocess_frame(frame, "CLAIM", self.statement, self.output)
        self.assertIn("Treatment", result["candidate_entities"])
        self.assertEqual(
            result["candidate_relations"][0]["evidence_text"], self.statement["text"]
        )
        self.assertTrue(result["validation"]["grounding_errors"])

    def test_unresolved_reference_is_not_passed(self):
        statement = {
            "id": "p1:u2",
            "text": "These findings support the proposed architecture.",
            "facets": {"role": "CONTENT"},
        }
        output = {"statements": [self.statement, statement]}
        frame = base_frame(
            primary_entity="These findings",
            secondary_entity="the proposed architecture",
            property="support",
        )
        result = postprocess_frame(frame, "CLAIM", statement, output)
        self.assertFalse(result["ontology_readiness"]["ready_for_mapping"])
        self.assertEqual(
            result["validation"]["automation_action"], "REPROCESS_WITH_CONTEXT"
        )

    def test_complementizer_that_is_not_a_reference(self):
        statement = {
            "id": "p1:u1",
            "text": "Testing showed that treatment reduced errors by 25 percent.",
            "facets": {"role": "CONTENT"},
        }
        frame = base_frame(primary_entity="treatment", secondary_entity="errors")
        result = postprocess_frame(frame, "CLAIM", statement, {"statements": [statement]})
        self.assertNotIn("that", result["validation"]["ambiguous_terms"])

    def test_supported_reference_can_pass(self):
        context = {
            "id": "p1:u1",
            "text": "Testing confirmed the architecture's accuracy.",
            "facets": {"role": "CONTENT"},
        }
        statement = {
            "id": "p1:u2",
            "text": "These findings support the architecture.",
            "facets": {"role": "CONTENT"},
        }
        output = {"statements": [context, statement]}
        frame = base_frame(primary_entity="testing results", property="support")
        frame["reference_resolutions"] = [{
            "original": "These findings",
            "resolved_text": "testing results",
            "evidence_statement_ids": ["p1:u1"],
            "confidence": 0.9,
        }]
        result = postprocess_frame(frame, "CLAIM", statement, output)
        self.assertEqual(result["validation"]["ambiguous_terms"], [])

    def test_ungrounded_reference_resolution_is_rejected(self):
        context = {
            "id": "p1:u1",
            "text": "Testing measured localization accuracy.",
            "facets": {"role": "CONTENT"},
        }
        statement = {
            "id": "p1:u2",
            "text": "These findings support the architecture.",
            "facets": {"role": "CONTENT"},
        }
        frame = base_frame(primary_entity="unrelated protein pathway", property="support")
        frame["reference_resolutions"] = [{
            "original": "These findings",
            "resolved_text": "unrelated protein pathway",
            "evidence_statement_ids": ["p1:u1"],
            "confidence": 0.99,
        }]
        result = postprocess_frame(frame, "CLAIM", statement, {"statements": [context, statement]})
        self.assertIn("These findings", result["validation"]["ambiguous_terms"])

    def test_number_from_valid_resolved_context_is_not_invented(self):
        context = {
            "id": "p1:u1",
            "text": "Testing reduced localization error by 25 percent.",
            "facets": {"role": "CONTENT"},
        }
        statement = {
            "id": "p1:u2",
            "text": "These findings support the architecture.",
            "facets": {"role": "CONTENT", "polarity": "POSITIVE"},
        }
        output = {
            "statements": [context, statement],
            "relations": [{"from": "u2", "to": "u1", "type": "SUPPORT"}],
        }
        frame = base_frame(
            primary_entity="findings",
            secondary_entity="architecture",
            property="support",
            value="support",
            context=context["text"],
            measurement_value=None,
            unit=None,
        )
        result = postprocess_frame(frame, "SUPPORT", statement, output)
        self.assertFalse(any(
            "invented numbers" in error for error in result["validation"]["grounding_errors"]
        ))

    def test_unsupported_candidate_entity_is_detected(self):
        frame = base_frame(primary_entity="Treatment", secondary_entity="response")
        frame["candidate_entities"] = ["unmentioned quantum reactor"]
        result = postprocess_frame(frame, "CLAIM", self.statement, self.output)
        self.assertTrue(any(
            "unsupported" in error for error in result["validation"]["grounding_errors"]
        ))
        self.assertFalse(result["ontology_readiness"]["ready_for_mapping"])

    def test_stage1_arguments_supply_relation_fallback(self):
        statement = {
            "id": "p1:u1",
            "text": "Dose correlates with recovery.",
            "predicate": "correlates with",
            "arguments": [
                {"role": "arg1", "text": "Dose"},
                {"role": "arg2", "text": "recovery"},
            ],
            "facets": {"role": "CONTENT"},
        }
        frame = base_frame(primary_entity="Dose", secondary_entity="recovery")
        result = postprocess_frame(frame, "CORRELATION", statement, {"statements": [statement]})
        self.assertTrue(any(
            relation["subject"] == "Dose" and relation["object"] == "recovery"
            for relation in result["candidate_relations"]
        ))

    def test_non_content_is_rejected(self):
        statement = {
            "id": "p1:u1",
            "text": "Smith et al. (2024).",
            "facets": {"role": "REFERENCE"},
        }
        frame = base_frame()
        result = postprocess_frame(frame, "CLAIM", statement, {"statements": [statement]})
        self.assertEqual(result["validation"]["automation_action"], "REJECT_LOW_VALUE")

    def test_equation_is_routed_to_specialized_extractor(self):
        statement = {
            "id": "p1:u1",
            "text": "E = mc^2",
            "facets": {"role": "EQUATION"},
        }
        result = postprocess_frame(
            base_frame(), "CLAIM", statement, {"statements": [statement]}
        )
        self.assertEqual(
            result["validation"]["automation_action"], "ROUTE_TO_EQUATION_EXTRACTOR"
        )


class ProductionSupportTests(unittest.TestCase):
    def test_content_ids_are_stable(self):
        self.assertEqual(stable_content_id("same"), stable_content_id("same"))
        self.assertNotEqual(stable_content_id("same"), stable_content_id("different"))

    def test_input_validation(self):
        self.assertEqual(validate_paragraph(" valid ", 20), "valid")
        with self.assertRaises(ValueError):
            validate_paragraph("", 20)
        with self.assertRaises(ValueError):
            validate_paragraph("x" * 21, 20)

    def test_routing_queue_is_idempotent(self):
        with TemporaryDirectory() as directory:
            queue = RoutingQueue(Path(directory) / "queue.sqlite3")
            frame = {
                "statement_id": "p1:u1",
                "stage2_frame": {
                    "validation": {"automation_action": "REPROCESS_WITH_CONTEXT"}
                },
            }
            first = queue.enqueue("p1", frame)
            second = queue.enqueue("p1", frame)
            count = queue.connection.execute(
                "SELECT COUNT(*) FROM routing_queue"
            ).fetchone()[0]
            queue.close()
            self.assertEqual(first, second)
            self.assertEqual(count, 1)

    def test_adjudication_requires_two_reviewer_consensus(self):
        record = {"adjudication": {
            "status": "adjudicated",
            "reviewers": ["reviewer-a", "reviewer-b"],
            "annotations": [
                {"reviewer_id": "reviewer-a", "stage1": {}, "stage2": {}},
                {"reviewer_id": "reviewer-b", "stage1": {}, "stage2": {}},
            ],
            "agreement": "consensus",
            "gold_stage1": {"statements": []},
            "gold_stage2": {"frames": []},
        }}
        self.assertTrue(validate_record(record))
        record["adjudication"]["annotations"][0]["stage1"] = {"statements": []}
        record["adjudication"]["annotations"][0]["stage2"] = {"frames": []}
        record["adjudication"]["annotations"][1]["stage1"] = {"statements": []}
        record["adjudication"]["annotations"][1]["stage2"] = {"frames": []}
        self.assertEqual(validate_record(record), [])

    def test_two_reviewer_workflow_finalizes_identical_annotations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.json"
            corpus = [{
                "id": "r1", "document_id": "d1", "domain_seed": "physics",
                "title": "T", "text": "Energy is 2 MeV.", "source_uri": "gold://r1",
                "adjudication": {"status": "pending_adjudication", "reviewers": [], "annotations": []},
            }]
            corpus_path.write_text(__import__("json").dumps(corpus), encoding="utf-8")
            for reviewer in ("a", "b"):
                packet_path = root / f"{reviewer}.json"
                export_packet(corpus_path, reviewer, packet_path, 0, 1)
                packet = __import__("json").loads(packet_path.read_text(encoding="utf-8"))
                packet["items"][0]["stage1"] = {"statements": [{"text": "Energy is 2 MeV."}]}
                packet["items"][0]["stage2"] = {"frames": [{"frame_type": "MEASUREMENT"}]}
                packet_path.write_text(__import__("json").dumps(packet), encoding="utf-8")
                merge_packet(corpus_path, packet_path)
            result = finalize(corpus_path)
            record = __import__("json").loads(corpus_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(result["adjudicated"], 1)
            self.assertEqual(validate_record(record), [])


if __name__ == "__main__":
    unittest.main()
