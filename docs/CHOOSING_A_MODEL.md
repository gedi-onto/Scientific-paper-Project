# Choosing a model for graphrag_stage1

The library is **model-agnostic** — it works with any model that can return JSON
matching a schema. This guide helps you pick, with **runnable code** and an honest
**pros/cons** for each option. Working code for all of these lives in
[`examples/model_clients.py`](../examples/model_clients.py).

> **The one contract (all providers):** your client's `complete(prompt, schema)`
> must return a Python `dict` that matches `schema`. Every example below does this
> via the provider's **tool / function-calling** — the most reliable way to force
> schema-valid output. Models that can't do reliable structured output (some very
> small local models) will struggle regardless of provider.

---

## Quick comparison

| Provider | Built-in? | Data leaves your env? | Speed at scale | Extraction quality | Cost model | Best for |
|----------|-----------|----------------------|----------------|--------------------|------------|----------|
| **Anthropic Claude** | ✅ | Yes (Anthropic API) | High (parallel) | Excellent | Per token | Fastest path to best quality |
| **OpenAI GPT** | ✅ | Yes (OpenAI API) | High | Excellent | Per token | Mature tooling, strict JSON |
| **Azure OpenAI** | ✅ (via `sdk=`) | Stays in your Azure tenant | High | Excellent | Per token | Microsoft/enterprise compliance |
| **Amazon Bedrock** | Adapter | **No — stays in AWS** | High (quota-bound) | Excellent (Claude) | Per token (AWS bill) | AWS deployments, governance |
| **Google Gemini** | Adapter | Yes (Google API) | High | Very good | Per token | Very long documents |
| **Local Ollama** | ✅ | **No — fully offline** | Low (GPU serializes) | Fair (small models) | Free | Dev, privacy, offline |
| **Self-hosted vLLM/TGI** | ✅ (via `sdk=`) | **No — your infra** | High (if provisioned) | Good→Excellent (large OSS) | Your compute | Private + scalable |

---

## Anthropic Claude *(built-in)*

```python
from graphrag_stage1 import AnthropicClient
client = AnthropicClient(model="claude-haiku-4-5", stronger_model="claude-sonnet-5")
# install: pip install "graphrag-stage1[anthropic]"   ·   env: ANTHROPIC_API_KEY
```

**Pros**
- **Best-in-class for this task** — strong at nuanced scientific prose, faceted
  classification, and reliable tool-use JSON.
- **Zero adapter code** — built in; pick Haiku (fast/cheap), Sonnet (balanced), or
  Opus 4.8 (max quality) with one parameter.
- Scales well: many calls run in parallel, so a paper lands in ~1 minute.
- Clean `stronger` escalation: Haiku by default, Sonnet on hard cases.

**Cons**
- **Paid** per token, and cost scales with paper volume.
- **Data leaves your environment** (Anthropic API) — a factor for sensitive or
  embargoed papers.
- You manage rate limits (via `max_concurrency`).

---

## OpenAI GPT *(built-in)*

```python
from graphrag_stage1 import OpenAIClient
client = OpenAIClient(model="gpt-4o-mini", stronger_model="gpt-4o")
# install: pip install "graphrag-stage1[openai]"   ·   env: OPENAI_API_KEY
```

**Pros**
- **Mature SDK & tooling**, very reliable structured outputs; built in.
- `gpt-4o-mini` is cheap and fast; `gpt-4o` for higher quality.
- Huge ecosystem, easy to get started.

**Cons**
- Paid; data leaves your environment.
- Function-calling schema handling is occasionally stricter than Claude's (the
  built-in client uses function-calling, which is the tolerant path).

**Azure OpenAI variant** — same models, but inside your Azure tenant (compliance,
region control). Reuse `OpenAIClient` with an Azure SDK object; `model` becomes your
**deployment name**. See the `azure` entry in the cookbook.

---

## Amazon Bedrock *(small custom adapter)*

```python
# full BedrockClient in examples/model_clients.py   ·   pip install boto3
client = BedrockClient(model="anthropic.claude-3-5-haiku-20241022-v1:0",
                       region="us-east-1")
```

**Pros**
- **Stays inside AWS** — VPC, IAM, no data egress. The strongest option for **data
  governance** and for the AWS/Neptune deployment this library targets.
- Reuses your existing **AWS account & billing**; no separate vendor.
- Same Claude quality, via the Converse API's tool use.

**Cons**
- **Not built in** — needs the ~25-line adapter (provided).
- Model **availability and IDs vary by region**; you must enable model access.
- Throughput is bound by your **Bedrock service quota** — set `max_concurrency` to it.

---

## Google Gemini *(small custom adapter)*

```python
# full GeminiClient in examples/model_clients.py   ·   pip install google-generativeai
client = GeminiClient(model="gemini-1.5-flash", stronger_model="gemini-1.5-pro")
# env: GEMINI_API_KEY
```

**Pros**
- **Very long context windows** — handy for feeding large sections or whole papers.
- Competitive cost; `flash` tier is fast and cheap.

**Cons**
- **Not built in** — needs the adapter; the SDK and structured-output API change
  more often than others (verify against your installed version).
- Data leaves your environment.
- Structured-output/function-calling can be finickier to get exactly right.

---

## Local Ollama *(built-in)*

```python
from graphrag_stage1 import OllamaClient
client = OllamaClient(model="qwen3:8b")
# install Ollama, then: ollama pull qwen3:8b
```

**Pros**
- **Free** and **fully private/offline** — nothing leaves your machine. Ideal for
  **sensitive, embargoed, or confidential** papers, and for development/testing.
- No API keys, no per-token cost, no rate limits.
- Built in; the library's default when no client is passed.

**Cons**
- **Slow at scale** — a single GPU serializes calls, so concurrency does **not**
  help. A full paper takes tens of minutes, not ~1 minute.
- **Lower accuracy** — small models (8B) produce rougher facets and rougher ontology
  mappings, and sometimes weaker structured output. Fine for trying things out; not
  ideal for production-quality graphs.
- You run and maintain the local server/hardware.

> **Measured (0.2 defaults):** ~25 s per real 70-word paragraph on an RTX 5080 laptop
> (qwen3:8b, batching on, `num_ctx=6144`, `max_concurrency=4`) — an 80-paragraph paper
> in **~33 minutes**. With the pre-0.2 defaults (batching off, Ollama's 4096 window)
> the same paper took **~8 hours**. If local feels impossibly slow, check those two
> settings before blaming the hardware — and see the `num_ctx` × `OLLAMA_NUM_PARALLEL`
> warning in DOCUMENTATION.md §11, which is worth 10× on its own.

---

## Ollama Cloud *(built-in — same `OllamaClient`)*

```python
from graphrag_stage1 import OllamaClient
client = OllamaClient(model="gpt-oss:120b-cloud")   # ollama signin, then any :cloud tag
```

Any model tag ending `-cloud` / `:cloud` is served remotely. `OllamaClient` detects
this (`schema_is_enforced`) and restates the JSON schema in the prompt, because Ollama
Cloud treats `format` as a *hint*, not a decoding grammar — unlike local Ollama, which
compiles it into one. Do **not** pass `num_ctx`: it is a local KV-cache knob and means
nothing remotely.

**Pros**

- Frontier-scale models with no local VRAM (`gpt-oss:120b-cloud`,
  `nemotron-3-super:cloud`, `nemotron-3-ultra:cloud`).
- **Better graphs.** Measured on 4 interaction-dense PMC paragraphs, `gpt-oss:120b-cloud`
  produced **7 fully-`mapped` frames vs local qwen3:8b's 3**, and was the only model to
  reach real domain classes rather than `BFO:entity` fallbacks.

**Cons**

- **Not faster — slower.** 55 s/paragraph vs local 25 s. A bigger model costs more
  decode time per call; reasoning models (`nemotron-*`) cost more still, since they
  emit thinking tokens before answering.
- **Quota-bound.** Free tier throttles hard: `max_concurrency=4` completes cleanly,
  `max_concurrency=12` returned HTTP 429 on **7 of 12 paragraphs** and ran 3× slower per
  survivor. Set `max_concurrency` **to** your allowance, not higher.
- Model tags churn — `qwen3-coder:480b-cloud`, benchmarked in earlier releases, no
  longer exists. Probe availability before pinning one.

> **Choose cloud for quality, not speed.** Parallel capacity is what buys throughput,
> and on a free tier you do not have much of it. If you need both, self-host vLLM.

---

## Self-hosted large models — vLLM / TGI *(built-in via `sdk=`)*

```python
import openai
from graphrag_stage1 import OpenAIClient
client = OpenAIClient(model="Qwen/Qwen2.5-72B-Instruct",
                      sdk=openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x"))
```

**Pros**
- **Private *and* scalable** — no data egress, but real parallel throughput (unlike a
  single-GPU Ollama). The best of both for private production.
- Run strong open models (Llama, Qwen-72B, Mixtral) at a quality approaching the
  hosted APIs.
- Reuses the built-in `OpenAIClient` — no adapter, just a `base_url`.
- Cost is your compute, which can be cheaper at high volume.

**Cons**
- **You run the infrastructure** — GPUs, serving stack, ops. Real setup effort.
- Open models still trail the top hosted models slightly on the hardest extraction.

---

## Which should I pick?

| Your situation | Recommended | Why |
|----------------|-------------|-----|
| Just want the best result, fastest to set up | **Anthropic Claude** (Haiku→Sonnet) | Built-in, top quality, one line |
| Deploying on **AWS** / strict data governance | **Amazon Bedrock** | Data stays in AWS; reuse account |
| **Sensitive / offline / confidential** papers, low volume | **Local Ollama** | Fully private, free |
| Private **and** high-volume production | **Self-hosted vLLM** | Private + real throughput |
| Microsoft / enterprise compliance shop | **Azure OpenAI** | OpenAI models, your tenant |
| **Very long** documents | **Gemini** or **Claude** | Long context windows |
| Prototyping on a laptop | **Local Ollama** | No keys, no cost |
| Bulk-running a corpus on one workstation | **Local Ollama** (batching on) | ~25 s/paragraph, no metering, no 429s — faster than a free-tier cloud model |
| Best graph quality, volume not critical | **Ollama Cloud** `gpt-oss:120b-cloud` | 2× more frames fully mapped; 2× slower per paragraph |

### Practical guidance
- **Quality matters for this task.** Faceted scientific extraction + structured
  output rewards stronger models. Start with a mid/large model; drop to a fast/cheap
  one only after you've confirmed quality on your papers.
- **Use the `stronger` tier.** All clients here escalate hard cases to a bigger model
  automatically (Stage 2's `REPROCESS_WITH_STRONGER_MODEL`). Keep a fast default and a
  strong fallback.
- **Speed is about the endpoint *and* the settings.** Hosted/scalable endpoints hit
  ~1 min/paper; a single local GPU can't (see the user guide, "Make it fast"). But check
  the settings first: with batching off and a 4096 window, a local paper took ~8 hours;
  with the 0.2 defaults it takes ~33 minutes. That is a bigger factor than the endpoint.
- **A bigger model is not a faster one.** It buys graph quality, and costs decode time.
  Speed comes from *parallel capacity* — which is a property of your quota or your
  hardware, not of the model. Set `max_concurrency` to what you're actually entitled to;
  past that you buy 429 backoff and lost paragraphs, not throughput.
- **Mind data governance.** If papers can't leave your environment, use **Bedrock**,
  **Ollama**, or **self-hosted vLLM** — not the public APIs.

---

*Runnable examples for every provider above:* [`examples/model_clients.py`](../examples/model_clients.py)
— `python examples/model_clients.py <provider>`.
