"""Tests for hybrid Stage 2: rule-built frames, with the model only for the gaps."""

import unittest

from graphrag_stage1.stage2_semantic_frames import (
    build_deterministic_frame,
    extract_deterministic_frame,
    extract_with_routing,
)


class RecordingClient:
    """LLMClient that fills any schema and records how often it was called."""

    model_id = "recording-model"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, schema, *, stronger=False):
        self.calls += 1
        return self._gen(schema)

    def _gen(self, s):
        if "enum" in s:
            return s["enum"][0]
        t = s.get("type")
        if isinstance(t, list):
            t = t[0]
        if t == "object":
            return {k: self._gen(v) for k, v in s.get("properties", {}).items()}
        if t == "array":
            return [self._gen(s["items"])] if "items" in s else []
        if t == "integer":
            return 1
        if t == "number":
            return 0.9
        if t == "boolean":
            return False
        return "filled"


def statement(text, ptype, *, arg1=None, arg2=None, predicate=None):
    args = []
    if arg1:
        args.append({"role": "arg1", "text": arg1})
    if arg2:
        args.append({"role": "arg2", "text": arg2})
    return {
        "id": "p1:u1",
        "text": text,
        "predicate": predicate,
        "arguments": args,
        "provenance": {"paragraph_id": "p1", "verbatim": text},
        "facets": {
            "role": "CONTENT",
            "proposition_type": ptype,
            "relation": "NONE",
            "modality": "ASSERTED",
            "polarity": "POSITIVE",
            "attribution": "AUTHOR",
            "certainty": "HIGH",
            "has_measurement": False,
        },
    }


def stage1(stmt):
    return {
        "paragraph_id": "p1",
        "paragraph": stmt["text"],
        "statements": [stmt],
        "relations": [],
    }


class DeterministicFrameTests(unittest.TestCase):
    def test_fills_entities_from_stage1_arguments(self):
        stmt = statement(
            "The passivation layer reduced recombination.",
            "CAUSAL_RELATION",
            arg1="passivation layer",
            arg2="recombination",
            predicate="reduce",
        )
        frame = build_deterministic_frame(stmt, stage1(stmt))
        semantic = frame["semantic_frame"]
        self.assertEqual(semantic["primary_entity"], "passivation layer")
        self.assertEqual(semantic["secondary_entity"], "recombination")
        self.assertEqual(
            frame["candidate_relations"],
            [{"subject": "passivation layer", "predicate": "reduce", "object": "recombination"}],
        )
        self.assertEqual(frame["candidate_entities"], ["passivation layer", "recombination"])

    def test_fills_measurement_by_rule(self):
        stmt = statement(
            "Efficiency reached 24.8 percent.",
            "MEASUREMENT",
            arg1="efficiency",
            predicate="reach",
        )
        semantic = build_deterministic_frame(stmt, stage1(stmt))["semantic_frame"]
        self.assertEqual(semantic["primary_entity"], "efficiency")
        self.assertEqual(semantic["measurement_value"], "24.8")
        self.assertIsNotNone(semantic["unit"])

    def test_process_comes_from_the_stage1_predicate(self):
        # MECHANISM defines `process` as the mechanism process; Stage 1's predicate
        # is the statement's main verb lemma. Same thing, and grounded in the source.
        stmt = statement("The system aligns outputs.", "MECHANISM", arg1="system", predicate="align")
        semantic = build_deterministic_frame(stmt, stage1(stmt))["semantic_frame"]
        self.assertEqual(semantic["process"], "align")

    def test_leaves_model_only_fields_unset(self):
        stmt = statement(
            "Radiation drives inflammation.",
            "CAUSAL_RELATION",
            arg1="radiation",
            arg2="inflammation",
            predicate="drive",
        )
        semantic = build_deterministic_frame(stmt, stage1(stmt))["semantic_frame"]
        # A CAUSAL_RELATION's `property` is the AFFECTED property, not the verb --
        # deriving it from the predicate would put a wrong fact in the graph.
        self.assertIsNone(semantic["property"], "predicate must never be used as `property`")
        for field in ("value", "process", "condition", "basis", "context"):
            self.assertIsNone(semantic[field], f"{field} should be left for the model")


class HybridRoutingTests(unittest.TestCase):
    def test_complete_frame_skips_the_model_entirely(self):
        # MEASUREMENT requires primary_entity + measurement_value + unit, all of
        # which Stage 1 structure and the regex pass already supply.
        stmt = statement(
            "Efficiency reached 24.8 percent.",
            "MEASUREMENT",
            arg1="efficiency",
            predicate="reach",
        )
        client = RecordingClient()
        item = extract_with_routing(stmt, stage1(stmt), client, hybrid=True)
        self.assertEqual(client.calls, 0, "no model call should be made for a complete frame")
        self.assertEqual(item["processing"]["model"], "deterministic:stage1-structure")
        self.assertEqual(
            item["stage2_frame"]["validation"]["automation_action"],
            "PASS_TO_ONTOLOGY_MAPPING",
        )

    def test_incomplete_frame_still_escalates_to_the_model(self):
        # CAUSAL_RELATION also requires `property`, which no rule can supply --
        # the quality gate must fall through to the LLM rather than pass a gap.
        stmt = statement(
            "Radiation drives inflammation.",
            "CAUSAL_RELATION",
            arg1="radiation",
            arg2="inflammation",
            predicate="drive",
        )
        client = RecordingClient()
        item = extract_with_routing(stmt, stage1(stmt), client, hybrid=True)
        self.assertGreater(client.calls, 0, "incomplete frame must escalate to the model")
        self.assertNotEqual(item["processing"]["model"], "deterministic:stage1-structure")

    def test_hybrid_off_by_default_preserves_llm_path(self):
        stmt = statement(
            "Efficiency reached 24.8 percent.",
            "MEASUREMENT",
            arg1="efficiency",
            predicate="reach",
        )
        client = RecordingClient()
        extract_with_routing(stmt, stage1(stmt), client, hybrid=False)
        self.assertGreater(client.calls, 0, "default path must still call the model")

    def test_deterministic_frame_has_the_same_shape_as_an_llm_frame(self):
        stmt = statement(
            "Efficiency reached 24.8 percent.",
            "MEASUREMENT",
            arg1="efficiency",
            predicate="reach",
        )
        item = extract_deterministic_frame(stmt, stage1(stmt))
        frame = item["stage2_frame"]
        for key in ("frame_type", "semantic_frame", "candidate_entities",
                    "candidate_relations", "validation", "ontology_readiness", "confidence"):
            self.assertIn(key, frame)
        # Stage 3 consumes these -- they must be populated, not just present.
        self.assertEqual(frame["frame_type"], "MEASUREMENT")
        self.assertIn("confidence_components", frame)


if __name__ == "__main__":
    unittest.main()
