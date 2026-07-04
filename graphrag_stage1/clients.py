"""Ready-made LLMClient implementations for hosted providers.

These save you from writing the structured-output boilerplate by hand. Each one
forces the model to answer in the exact JSON shape the pipeline asks for (via the
provider's tool / function-calling mechanism) and returns it as a dict.

Install the matching extra:

    pip install "graphrag-stage1[anthropic]"     # AnthropicClient
    pip install "graphrag-stage1[openai]"        # OpenAIClient

Both take an optional pre-built SDK object (``sdk=...``) so they are easy to test
or configure; by default they construct one that reads the provider's standard
API-key environment variable.
"""

from __future__ import annotations

import json


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API (Claude).

        client = AnthropicClient()                       # uses ANTHROPIC_API_KEY
        client = AnthropicClient(model="claude-sonnet-5")
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        stronger_model: str = "claude-sonnet-5",
        *,
        max_tokens: int = 2048,
        sdk=None,
    ):
        self.model = model
        self.stronger_model = stronger_model
        self.max_tokens = max_tokens
        if sdk is None:
            import anthropic  # lazy: only needed if you actually use this client

            sdk = anthropic.Anthropic()
        self._sdk = sdk

    @property
    def model_id(self) -> str:
        return self.model

    def resolved_model(self, stronger: bool = False) -> str:
        return self.stronger_model if stronger else self.model

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        message = self._sdk.messages.create(
            model=self.resolved_model(stronger),
            max_tokens=self.max_tokens,
            tools=[{
                "name": "emit",
                "description": "Return the result in the required structure.",
                "input_schema": schema,
            }],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                return block.input
        raise RuntimeError("Anthropic response contained no structured tool output")


class OpenAIClient:
    """LLMClient backed by the OpenAI Chat Completions API.

        client = OpenAIClient()                          # uses OPENAI_API_KEY
        client = OpenAIClient(model="gpt-4o-mini")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        stronger_model: str = "gpt-4o",
        *,
        sdk=None,
    ):
        self.model = model
        self.stronger_model = stronger_model
        if sdk is None:
            import openai  # lazy: only needed if you actually use this client

            sdk = openai.OpenAI()
        self._sdk = sdk

    @property
    def model_id(self) -> str:
        return self.model

    def resolved_model(self, stronger: bool = False) -> str:
        return self.stronger_model if stronger else self.model

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        response = self._sdk.chat.completions.create(
            model=self.resolved_model(stronger),
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "emit",
                    "description": "Return the result in the required structure.",
                    "parameters": schema,
                },
            }],
            tool_choice={"type": "function", "function": {"name": "emit"}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise RuntimeError("OpenAI response contained no structured tool output")
        return json.loads(tool_calls[0].function.arguments)
