"""Tests for parallel paper processing and the global concurrency cap."""

import threading
import time
import unittest

from graphrag_stage1 import run_paper, BoundedClient


class InstrumentedClient:
    """Schema-filling fake that records peak concurrent in-flight calls."""

    model_id = "fake-model-v1"

    def __init__(self, delay=0.02):
        self.delay = delay
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.calls = 0

    def complete(self, prompt, schema, *, stronger=False):
        with self._lock:
            self.inflight += 1
            self.calls += 1
            self.peak = max(self.peak, self.inflight)
        try:
            time.sleep(self.delay)
            return self._gen(schema)
        finally:
            with self._lock:
                self.inflight -= 1

    def _gen(self, s):
        if "enum" in s:
            return s["enum"][0]
        t = s.get("type")
        if t == "object":
            return {k: self._gen(v) for k, v in s.get("properties", {}).items()}
        if t == "array":
            return [self._gen(s["items"])]
        if t == "integer":
            return 1
        if t == "number":
            return 0.9
        if t == "boolean":
            return False
        return "sample"


def paragraphs(n):
    return [
        {"text": f"Sample sentence number {i} improved accuracy.", "paragraph_id": f"p{i}"}
        for i in range(n)
    ]


class RunPaperTests(unittest.TestCase):
    def test_preserves_order_and_processes_all(self):
        client = InstrumentedClient(delay=0)
        results = run_paper(paragraphs(6), client=client, max_concurrency=4)
        self.assertEqual(len(results), 6)
        self.assertEqual(
            [r["stage1"]["paragraph_id"] for r in results],
            [f"p{i}" for i in range(6)],
        )

    def test_paragraphs_actually_run_concurrently(self):
        client = InstrumentedClient(delay=0.02)
        run_paper(paragraphs(8), client=client, max_concurrency=8, stage2_concurrency=4)
        # If paragraphs ran serially the peak would be ~1; parallelism must overlap.
        self.assertGreater(client.peak, 1)

    def test_bounded_client_caps_global_concurrency(self):
        client = InstrumentedClient(delay=0.02)
        cap = 3
        run_paper(paragraphs(8), client=client, max_concurrency=cap, stage2_concurrency=4)
        self.assertLessEqual(client.peak, cap)

    def test_explicit_bounded_client_is_respected(self):
        client = InstrumentedClient(delay=0.02)
        bounded = BoundedClient(client, max_concurrency=2)
        # Even if run_paper is asked for more, the pre-wrapped cap wins.
        run_paper(paragraphs(6), client=bounded, max_concurrency=99, stage2_concurrency=4)
        self.assertLessEqual(client.peak, 2)

    def test_one_bad_paragraph_does_not_sink_the_paper(self):
        client = InstrumentedClient(delay=0)
        docs = paragraphs(3)
        docs[1] = {"text": "", "paragraph_id": "p1"}  # empty -> validate_paragraph raises
        results = run_paper(docs, client=client, max_concurrency=3)
        self.assertEqual(len(results), 3)
        self.assertIn("error", results[1])
        self.assertIn("stage1", results[0])
        self.assertIn("stage1", results[2])


if __name__ == "__main__":
    unittest.main()
