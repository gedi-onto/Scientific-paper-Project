"""Cookbook: use graphrag_stage1 with every major LLM provider (and local Ollama).

Each section shows how to obtain an LLMClient for one provider. Built-in clients
(Anthropic, OpenAI, Ollama) need no code. Bedrock and Gemini are shown as small
custom adapters — the same pattern you'd use for any provider not built in.

Run a live demo against one provider:

    python examples/model_clients.py ollama        # local, offline (works out of the box)
    python examples/model_clients.py anthropic     # needs ANTHROPIC_API_KEY + [anthropic] extra
    python examples/model_clients.py openai         # needs OPENAI_API_KEY + [openai] extra
    python examples/model_clients.py bedrock        # needs AWS creds + boto3
    python examples/model_clients.py gemini         # needs GEMINI_API_KEY + google-generativeai
    python examples/model_clients.py vllm           # any OpenAI-compatible endpoint

The one rule for ALL of them: `complete(prompt, schema)` must return a Python dict
that matches `schema`. Every client below does that via the provider's
tool/function-calling (the most reliable way to force schema-valid JSON).

NOTE: model IDs change over time and vary by account/region. Verify the ids below
against what your provider currently offers.
"""

import sys

# Built-in clients ship with the library (no adapter code needed):
from graphrag_stage1 import AnthropicClient, OpenAIClient, OllamaClient, analyze_paper


# ===========================================================================
# 1. Anthropic Claude  (built-in)  —  pip install "graphrag-stage1[anthropic]"
# ===========================================================================
def anthropic_client():
    # Reads ANTHROPIC_API_KEY. Haiku = fast/cheap; Sonnet/Opus = higher quality.
    return AnthropicClient(model="claude-haiku-4-5", stronger_model="claude-sonnet-5")


# ===========================================================================
# 2. OpenAI GPT  (built-in)  —  pip install "graphrag-stage1[openai]"
# ===========================================================================
def openai_client():
    # Reads OPENAI_API_KEY.
    return OpenAIClient(model="gpt-4o-mini", stronger_model="gpt-4o")


# ---- 2b. Azure OpenAI: reuse OpenAIClient with an Azure SDK object ---------
def azure_openai_client():
    from openai import AzureOpenAI
    sdk = AzureOpenAI(
        api_key="...",
        api_version="2024-10-21",
        azure_endpoint="https://YOUR-RESOURCE.openai.azure.com",
    )
    # `model` here is your Azure *deployment name*.
    return OpenAIClient(model="my-gpt4o-mini-deployment", sdk=sdk)


# ===========================================================================
# 3. Local Ollama  (built-in)  —  free, private, offline
# ===========================================================================
def ollama_client():
    # Talks to a local Ollama server (http://localhost:11434). Pull a model first:
    #   ollama pull qwen3:8b
    return OllamaClient(model="qwen3:8b")


# ===========================================================================
# 4. Any OpenAI-compatible endpoint (vLLM / TGI / Groq / Together / LM Studio)
#    Reuse the built-in OpenAIClient — just point its SDK at your base_url.
# ===========================================================================
def vllm_client():
    import openai
    sdk = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    return OpenAIClient(model="Qwen/Qwen2.5-72B-Instruct", sdk=sdk)


# ===========================================================================
# 5. Amazon Bedrock  (custom adapter)  —  Claude etc. inside AWS
#    pip install boto3   ·   uses the Converse API + tool use for JSON output.
# ===========================================================================
class BedrockClient:
    def __init__(
        self,
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
        stronger_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-east-1",
        max_tokens=2048,
        runtime=None,
    ):
        self.model = model
        self.stronger_model = stronger_model
        self.max_tokens = max_tokens
        if runtime is None:
            import boto3
            runtime = boto3.client("bedrock-runtime", region_name=region)
        self._rt = runtime

    @property
    def model_id(self):
        return self.model

    def resolved_model(self, stronger=False):
        return self.stronger_model if stronger else self.model

    def complete(self, prompt, schema, *, stronger=False):
        resp = self._rt.converse(
            modelId=self.resolved_model(stronger),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0},
            toolConfig={
                "tools": [{"toolSpec": {
                    "name": "emit",
                    "description": "Return the result in the required structure.",
                    "inputSchema": {"json": schema},
                }}],
                "toolChoice": {"tool": {"name": "emit"}},
            },
        )
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                return block["toolUse"]["input"]
        raise RuntimeError("Bedrock response contained no tool output")


def bedrock_client():
    return BedrockClient(region="us-east-1")


# ===========================================================================
# 6. Google Gemini  (custom adapter)  —  pip install google-generativeai
#    Uses function calling to force schema-valid JSON.
# ===========================================================================
class GeminiClient:
    def __init__(self, model="gemini-1.5-flash", stronger_model="gemini-1.5-pro", api_key=None):
        import os
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._genai = genai
        self.model = model
        self.stronger_model = stronger_model

    @property
    def model_id(self):
        return self.model

    def resolved_model(self, stronger=False):
        return self.stronger_model if stronger else self.model

    def complete(self, prompt, schema, *, stronger=False):
        tool = {"function_declarations": [{
            "name": "emit",
            "description": "Return the result in the required structure.",
            "parameters": schema,
        }]}
        model = self._genai.GenerativeModel(
            self.resolved_model(stronger),
            tools=[tool],
            tool_config={"function_calling_config": {"mode": "ANY", "allowed_function_names": ["emit"]}},
        )
        resp = model.generate_content(prompt)
        for part in resp.candidates[0].content.parts:
            if getattr(part, "function_call", None):
                # function_call.args is a proto Map; convert to a plain dict.
                return dict(part.function_call.args)
        raise RuntimeError("Gemini response contained no function call")


def gemini_client():
    return GeminiClient()


# ===========================================================================
# Registry + demo runner
# ===========================================================================
CLIENTS = {
    "anthropic": anthropic_client,
    "openai": openai_client,
    "azure": azure_openai_client,
    "ollama": ollama_client,
    "vllm": vllm_client,
    "bedrock": bedrock_client,
    "gemini": gemini_client,
}

SAMPLE = (
    "The fused sensor data reduced positioning error by 35 percent. Because the "
    "fusion provides a more complete state estimate, the guidance system produced "
    "more stable trajectories."
)


def demo(client):
    """Run Stage 1 + 2 on a sample paragraph and print the extracted facts."""
    print(f"model_id: {client.model_id}")
    results = analyze_paper(SAMPLE, client=client, max_concurrency=4)
    for r in results:
        if "error" in r:
            print("  error:", r["error"]); continue
        for s in r["stage1"]["statements"]:
            f = s["facets"]
            print(f"  - [{f['proposition_type']}/{f['relation']}/{f['modality']}] {s['text']}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    if name not in CLIENTS:
        sys.exit(f"unknown provider {name!r}; choose one of: {', '.join(CLIENTS)}")
    demo(CLIENTS[name]())
