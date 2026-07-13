"""Tests for hybrid Stage 2: rule-built frames, with the model only for the gaps."""

import unittest

from graphrag_stage1.stage2_semantic_frames import (
    build_deterministic_frame,
    extract_deterministic_frame,
    extract_with_routing,
    stage2_pipeline,
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
        self.assertEqual(frame["candidate_entities"], ["passivation layer", "recombination"])

    def test_candidate_relations_are_left_to_the_existing_fallback(self):
        # deterministic_validation synthesises the relation from the Stage 1
        # predicate and owns the key names (subject/relation/object). Emitting our
        # own here previously used the wrong key and failed every 2-argument frame.
        stmt = statement(
            "SYK is a tyrosine kinase.",
            "CLASSIFICATION",
            arg1="SYK",
            arg2="tyrosine kinase",
            predicate="be",
        )
        self.assertEqual(build_deterministic_frame(stmt, stage1(stmt))["candidate_relations"], [])
        # ...and after postprocessing, the fallback has filled it in correctly.
        frame = extract_deterministic_frame(stmt, stage1(stmt))["stage2_frame"]
        relations = frame["candidate_relations"]
        self.assertTrue(relations, "the Stage 1 fallback should have built a relation")
        for key in ("subject", "relation", "object"):
            self.assertTrue(relations[0].get(key), f"relation missing {key}")
        self.assertFalse(frame["validation"].get("grounding_errors"))

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

    def test_two_argument_statement_also_skips_the_model(self):
        # Regression: build_deterministic_frame used to emit candidate_relations with
        # a `predicate` key instead of `relation`, so deterministic_validation flagged
        # every 2-argument frame as incomplete and nothing with arg1+arg2 ever skipped.
        stmt = statement(
            "SYK is a tyrosine kinase.",
            "CLASSIFICATION",
            arg1="SYK",
            arg2="tyrosine kinase",
            predicate="be",
        )
        client = RecordingClient()
        item = extract_with_routing(stmt, stage1(stmt), client, hybrid=True)
        self.assertEqual(client.calls, 0, "CLASSIFICATION has both entities from Stage 1")
        self.assertEqual(item["processing"]["model"], "deterministic:stage1-structure")

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


class BatchingTests(unittest.TestCase):
    """Batched Stage 2: N statements per call, with a fallback to the proven path."""

    def _statements(self, n):
        return [
            statement(
                f"Marker {i} increased by {i} percent.",
                "MEASUREMENT",
                arg1=f"marker {i}",
                predicate="increase",
            ) | {"id": f"p1:u{i}"}
            for i in range(1, n + 1)
        ]

    def test_batch_makes_one_call_for_many_statements(self):
        stmts = self._statements(6)
        s1 = {"paragraph_id": "p1", "paragraph": " ".join(s["text"] for s in stmts),
              "statements": stmts, "relations": []}

        class BatchClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                # Honour the batch schema: return one frame per statement, tagged by id.
                frame = self._gen(schema["properties"]["frames"]["items"])
                return {"frames": [dict(frame, statement_id=s["id"]) for s in stmts]}

        client = BatchClient()
        out = stage2_pipeline(s1, client=client, concurrency=1, batch_size=6)
        self.assertEqual(client.calls, 1, "6 statements should be one batched call")
        self.assertEqual(len(out["frames"]), 6)
        self.assertEqual(out["audit"]["batch_size"], 6)
        for item in out["frames"]:
            self.assertEqual(item["processing"]["batched"], 6)

    def test_frames_stay_aligned_to_their_statements(self):
        stmts = self._statements(4)
        s1 = {"paragraph_id": "p1", "paragraph": "x", "statements": stmts, "relations": []}

        class ShuffledClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                frame = self._gen(schema["properties"]["frames"]["items"])
                # Model returns them out of order -- alignment must follow statement_id.
                return {"frames": [dict(frame, statement_id=s["id"]) for s in reversed(stmts)]}

        out = stage2_pipeline(s1, client=ShuffledClient(), concurrency=1, batch_size=4)
        self.assertEqual(
            [i["statement_id"] for i in out["frames"]],
            [s["id"] for s in stmts],
            "frames must be re-aligned to the input statement order",
        )

    def test_short_batch_falls_back_to_per_statement_calls(self):
        stmts = self._statements(4)
        s1 = {"paragraph_id": "p1", "paragraph": "x", "statements": stmts, "relations": []}

        class ShortClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                if "frames" in schema.get("properties", {}):
                    return {"frames": []}  # truncated / malformed batch
                return self._gen(schema)

        client = ShortClient()
        out = stage2_pipeline(s1, client=client, concurrency=1, batch_size=4)
        # No facts dropped: it fell back to one call per statement.
        self.assertEqual(len(out["frames"]), 4)
        for item in out["frames"]:
            self.assertNotIn("batched", item["processing"])

    def test_flat_cloud_style_response_is_renested(self):
        # Ollama Cloud does not grammar-constrain output: qwen3-coder:480b-cloud returns
        # the semantic slots at the TOP level with an empty "semantic_frame", which used
        # to fail every frame for missing required fields. It must be re-nested in Python.
        stmts = self._statements(2)
        s1 = {"paragraph_id": "p1", "paragraph": "x", "statements": stmts, "relations": []}

        class FlatCloudClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                return {"frames": [
                    {
                        "statement_id": s["id"],
                        "semantic_frame": {},          # empty, as the cloud model sends
                        "primary_entity": f"marker {i}",  # slots at the top level
                        "measurement_value": str(i),
                        "unit": "percent",
                        "property": None, "value": None, "process": None,
                        "condition": None, "basis": None, "context": None,
                        "secondary_entity": None,
                        "confidence": 0.9,
                    }
                    for i, s in enumerate(stmts, start=1)
                ]}

        out = stage2_pipeline(s1, client=FlatCloudClient(), concurrency=1, batch_size=2)
        self.assertEqual(len(out["frames"]), 2)
        for item in out["frames"]:
            semantic = item["stage2_frame"]["semantic_frame"]
            self.assertTrue(
                semantic.get("primary_entity"),
                "top-level slots must be lifted into semantic_frame",
            )
            self.assertEqual(item["processing"]["batched"], 2)

    def test_batch_size_zero_keeps_the_original_path(self):
        stmts = self._statements(3)
        s1 = {"paragraph_id": "p1", "paragraph": "x", "statements": stmts, "relations": []}
        client = RecordingClient()
        out = stage2_pipeline(s1, client=client, concurrency=1, batch_size=0)
        self.assertEqual(len(out["frames"]), 3)
        # At least one call per statement (more if a frame triggered its reprocess
        # retry), and crucially nothing went through the batched path.
        self.assertGreaterEqual(client.calls, 3)
        for item in out["frames"]:
            self.assertNotIn("batched", item["processing"])


if __name__ == "__main__":
    unittest.main()
