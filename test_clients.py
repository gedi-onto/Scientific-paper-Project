"""Tests for the built-in provider clients and analyze_paper, using fake SDKs."""

import json
import types
import unittest

from graphrag_stage1 import AnthropicClient, OpenAIClient, analyze_paper, split_paragraphs
from graphrag_stage1.llm import LLMClient


# --- fake Anthropic SDK -------------------------------------------------------
class FakeAnthropic:
    def __init__(self):
        self.messages = types.SimpleNamespace(create=self._create)
        self.last = {}

    def _create(self, **kwargs):
        self.last = kwargs
        block = types.SimpleNamespace(type="tool_use", input={"ok": True, "echo_model": kwargs["model"]})
        return types.SimpleNamespace(content=[block])


# --- fake OpenAI SDK ----------------------------------------------------------
class FakeOpenAI:
    def __init__(self):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )
        self.last = {}

    def _create(self, **kwargs):
        self.last = kwargs
        call = types.SimpleNamespace(
            function=types.SimpleNamespace(arguments=json.dumps({"ok": True, "echo_model": kwargs["model"]}))
        )
        message = types.SimpleNamespace(tool_calls=[call])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


class AnthropicClientTests(unittest.TestCase):
    def test_is_llmclient_and_parses_tool_output(self):
        c = AnthropicClient(sdk=FakeAnthropic())
        self.assertIsInstance(c, LLMClient)
        out = c.complete("hi", SCHEMA)
        self.assertTrue(out["ok"])

    def test_schema_and_forced_tool_choice_are_passed(self):
        fake = FakeAnthropic()
        AnthropicClient(sdk=fake).complete("hi", SCHEMA)
        self.assertEqual(fake.last["tools"][0]["input_schema"], SCHEMA)
        self.assertEqual(fake.last["tool_choice"], {"type": "tool", "name": "emit"})

    def test_stronger_routes_to_stronger_model(self):
        fake = FakeAnthropic()
        c = AnthropicClient(model="m-fast", stronger_model="m-strong", sdk=fake)
        self.assertEqual(c.complete("hi", SCHEMA)["echo_model"], "m-fast")
        self.assertEqual(c.complete("hi", SCHEMA, stronger=True)["echo_model"], "m-strong")


class OpenAIClientTests(unittest.TestCase):
    def test_is_llmclient_and_parses_tool_output(self):
        c = OpenAIClient(sdk=FakeOpenAI())
        self.assertIsInstance(c, LLMClient)
        out = c.complete("hi", SCHEMA)
        self.assertTrue(out["ok"])

    def test_stronger_routes_to_stronger_model(self):
        c = OpenAIClient(model="fast", stronger_model="strong", sdk=FakeOpenAI())
        self.assertEqual(c.complete("hi", SCHEMA, stronger=True)["echo_model"], "strong")


class AnalyzePaperTests(unittest.TestCase):
    def test_split_paragraphs(self):
        paras = split_paragraphs("One.\n\nTwo.\n\n\n  Three.  ")
        self.assertEqual([p["text"] for p in paras], ["One.", "Two.", "Three."])

    def test_analyze_paper_runs_all_paragraphs(self):
        # Reuse the instrumented schema-filling fake from the throughput tests.
        from test_throughput import InstrumentedClient

        text = "First paragraph improved accuracy.\n\nSecond paragraph reduced error."
        results = analyze_paper(text, client=InstrumentedClient(delay=0), max_concurrency=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("stage1" in r for r in results))


if __name__ == "__main__":
    unittest.main()
