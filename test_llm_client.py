"""Transport-level behaviour of OllamaClient: schema enforcement and rate limits."""

import json
import unittest
from unittest.mock import patch

from graphrag_stage1.llm import OllamaClient, _extract_json, _retry_after_seconds

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class SchemaEnforcementTests(unittest.TestCase):
    def test_local_models_are_grammar_constrained(self):
        client = OllamaClient(model="qwen3:8b")
        self.assertTrue(client.schema_is_enforced("qwen3:8b"))

    def test_cloud_models_are_not(self):
        client = OllamaClient(model="qwen3-coder:480b-cloud")
        self.assertFalse(client.schema_is_enforced("qwen3-coder:480b-cloud"))
        self.assertFalse(client.schema_is_enforced("glm-4.6:cloud"))

    def test_cloud_call_restates_the_schema_in_the_prompt(self):
        client = OllamaClient(model="qwen3-coder:480b-cloud")
        seen = {}

        def fake_post(url, json=None, timeout=None):
            seen["prompt"] = json["prompt"]
            return FakeResponse(payload={"response": '{"ok": true}'})

        with patch("graphrag_stage1.llm.requests.post", side_effect=fake_post):
            self.assertEqual(client.complete("do the thing", SCHEMA), {"ok": True})
        # Cloud does not enforce `format`, so the contract must be in the prompt itself.
        self.assertIn("JSON Schema", seen["prompt"])
        self.assertIn('"required"', seen["prompt"])

    def test_local_call_leaves_the_prompt_alone(self):
        client = OllamaClient(model="qwen3:8b")
        seen = {}

        def fake_post(url, json=None, timeout=None):
            seen["prompt"] = json["prompt"]
            return FakeResponse(payload={"response": '{"ok": true}'})

        with patch("graphrag_stage1.llm.requests.post", side_effect=fake_post):
            client.complete("do the thing", SCHEMA)
        self.assertEqual(seen["prompt"], "do the thing")


class JsonExtractionTests(unittest.TestCase):
    def test_bare_json_passes_through(self):
        self.assertEqual(json.loads(_extract_json('{"a": 1}')), {"a": 1})

    def test_markdown_fences_are_stripped(self):
        self.assertEqual(json.loads(_extract_json('```json\n{"a": 1}\n```')), {"a": 1})

    def test_surrounding_prose_is_discarded(self):
        raw = 'Here is the result:\n{"a": 1}\nHope that helps!'
        self.assertEqual(json.loads(_extract_json(raw)), {"a": 1})


class RateLimitTests(unittest.TestCase):
    def test_429_is_waited_out_not_counted_as_a_failure(self):
        # retries=0 means a transport error would fail immediately -- but a 429 is not a
        # failure, it is a request to slow down, so the call must still succeed.
        client = OllamaClient(model="qwen3-coder:480b-cloud", retries=0, rate_limit_backoff=1)
        responses = [
            FakeResponse(429, headers={"Retry-After": "1"}),
            FakeResponse(429, headers={"Retry-After": "1"}),
            FakeResponse(payload={"response": '{"ok": true}'}),
        ]
        with patch("graphrag_stage1.llm.requests.post", side_effect=responses), \
                patch("graphrag_stage1.llm.time.sleep") as sleep:
            self.assertEqual(client.complete("x", SCHEMA), {"ok": True})
        self.assertEqual(sleep.call_count, 2, "should have waited out both 429s")

    def test_persistent_429_gives_up_with_a_clear_message(self):
        client = OllamaClient(model="qwen3-coder:480b-cloud", rate_limit_retries=2)
        with patch("graphrag_stage1.llm.requests.post", return_value=FakeResponse(429)), \
                patch("graphrag_stage1.llm.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                client.complete("x", SCHEMA)
        self.assertIn("rate limited", str(ctx.exception))
        self.assertIn("quota", str(ctx.exception))

    def test_retry_after_header_is_honoured(self):
        self.assertEqual(_retry_after_seconds(FakeResponse(429, headers={"Retry-After": "45"}), 30), 45.0)
        # An HTTP-date Retry-After is unparseable as a float -- fall back, don't crash.
        self.assertEqual(
            _retry_after_seconds(FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), 30),
            30.0,
        )
        self.assertEqual(_retry_after_seconds(FakeResponse(429), 30), 30.0)


if __name__ == "__main__":
    unittest.main()
