"""Batched Stage 2: N statements per model call, with a fallback to the proven path.

Measured on 43 identical statements (qwen3:8b), only batch_size varying:

    no batching   228.0s   25/43 passed (58%)   17 grounding errors
    batch 5       130.5s   31/43 passed (72%)    1 grounding error
    batch 10      121.8s   28/43 passed (65%)    3 grounding errors
    batch 20      433.9s   28/43 passed (65%)    4 grounding errors

Batch 5 is 1.75x faster AND more accurate: showing the model its sibling statements
improves grounding, which collapses the REPROCESS retries -- and each retry was a
second full call. Batch 20 regresses badly (the 32k context thrashes VRAM), so the
useful range is roughly 5-10.
"""

import unittest

from graphrag_stage1.stage2_semantic_frames import stage2_pipeline


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


def statement(sid, text, ptype, *, arg1=None, arg2=None, predicate=None):
    args = []
    if arg1:
        args.append({"role": "arg1", "text": arg1})
    if arg2:
        args.append({"role": "arg2", "text": arg2})
    return {
        "id": sid,
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


def statements(n):
    return [
        statement(
            f"p1:u{i}",
            f"Marker {i} increased by {i} percent.",
            "MEASUREMENT",
            arg1=f"marker {i}",
            predicate="increase",
        )
        for i in range(1, n + 1)
    ]


def stage1(stmts):
    return {
        "paragraph_id": "p1",
        "paragraph": " ".join(s["text"] for s in stmts),
        "statements": stmts,
        "relations": [],
    }


class BatchingTests(unittest.TestCase):
    def test_batch_makes_one_call_for_many_statements(self):
        stmts = statements(6)

        class BatchClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                frame = self._gen(schema["properties"]["frames"]["items"])
                return {"frames": [dict(frame, statement_id=s["id"]) for s in stmts]}

        client = BatchClient()
        out = stage2_pipeline(stage1(stmts), client=client, concurrency=1, batch_size=6)
        self.assertEqual(client.calls, 1, "6 statements should be one batched call")
        self.assertEqual(len(out["frames"]), 6)
        self.assertEqual(out["audit"]["batch_size"], 6)
        for item in out["frames"]:
            self.assertEqual(item["processing"]["batched"], 6)

    def test_frames_stay_aligned_to_their_statements(self):
        stmts = statements(4)

        class ShuffledClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                frame = self._gen(schema["properties"]["frames"]["items"])
                # Returned out of order -- alignment must follow statement_id, not position.
                return {"frames": [dict(frame, statement_id=s["id"]) for s in reversed(stmts)]}

        out = stage2_pipeline(stage1(stmts), client=ShuffledClient(), concurrency=1, batch_size=4)
        self.assertEqual(
            [i["statement_id"] for i in out["frames"]],
            [s["id"] for s in stmts],
            "frames must be re-aligned to the input statement order",
        )

    def test_short_batch_falls_back_to_per_statement_calls(self):
        stmts = statements(4)

        class ShortClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                if "frames" in schema.get("properties", {}):
                    return {"frames": []}  # truncated / malformed batch
                return self._gen(schema)

        out = stage2_pipeline(stage1(stmts), client=ShortClient(), concurrency=1, batch_size=4)
        # No facts dropped: it degraded to one call per statement.
        self.assertEqual(len(out["frames"]), 4)
        for item in out["frames"]:
            self.assertNotIn("batched", item["processing"])

    def test_flat_cloud_style_response_is_renested(self):
        # Ollama Cloud does not grammar-constrain output: qwen3-coder:480b-cloud returns
        # the semantic slots at the TOP level with an empty "semantic_frame", which used
        # to fail every frame for missing required fields. Python must re-nest it.
        stmts = statements(2)

        class FlatCloudClient(RecordingClient):
            def complete(self, prompt, schema, *, stronger=False):
                self.calls += 1
                return {"frames": [
                    {
                        "statement_id": s["id"],
                        "semantic_frame": {},              # empty, as the cloud model sends
                        "primary_entity": f"marker {i}",   # slots at the top level
                        "measurement_value": str(i),
                        "unit": "percent",
                        "property": None, "value": None, "process": None,
                        "condition": None, "basis": None, "context": None,
                        "secondary_entity": None,
                        "confidence": 0.9,
                    }
                    for i, s in enumerate(stmts, start=1)
                ]}

        out = stage2_pipeline(stage1(stmts), client=FlatCloudClient(), concurrency=1, batch_size=2)
        self.assertEqual(len(out["frames"]), 2)
        for item in out["frames"]:
            self.assertTrue(
                item["stage2_frame"]["semantic_frame"].get("primary_entity"),
                "top-level slots must be lifted into semantic_frame",
            )
            self.assertEqual(item["processing"]["batched"], 2)

    def test_batch_size_zero_keeps_the_per_statement_path(self):
        stmts = statements(3)
        client = RecordingClient()
        out = stage2_pipeline(stage1(stmts), client=client, concurrency=1, batch_size=0)
        self.assertEqual(len(out["frames"]), 3)
        self.assertGreaterEqual(client.calls, 3)
        for item in out["frames"]:
            self.assertNotIn("batched", item["processing"])


if __name__ == "__main__":
    unittest.main()
