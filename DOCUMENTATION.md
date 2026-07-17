# graphrag_stage1 — Complete Documentation

A Python library that turns unstructured scientific text into a **typed,
provenance-anchored knowledge representation** and a **publishable knowledge
graph**. It is the ingestion/enrichment engine for an ontology-driven GraphRAG
system: text in → verified facts + RDF for a graph store (e.g. Amazon Neptune) out.

This is the single reference: what's in it, what it does, every input, every
output, and exactly which call produces each one. For deep design rationale see
[DESIGN.md](DESIGN.md); for a shorter integration primer see
[INTEGRATION.md](INTEGRATION.md).

---

## Table of contents

1. [What it is (mental model)](#1-what-it-is)
2. [Install](#2-install)
3. [Quick start](#3-quick-start)
4. [The three stages](#4-the-three-stages)
5. [Inputs — everything it accepts](#5-inputs)
6. [Outputs — everything it produces](#6-outputs)
7. [How to get every single output](#7-how-to-get-every-output)
8. [Public API reference](#8-public-api-reference)
9. [Bring your own model (LLMClient)](#9-bring-your-own-model)
10. [Ontologies (core + your domain)](#10-ontologies)
11. [Performance & concurrency](#11-performance--concurrency)
12. [Validation & schemas](#12-validation--schemas)
13. [The faceted vocabulary](#13-the-faceted-vocabulary)
14. [Configuration & environment variables](#14-configuration--environment-variables)
15. [Batch CLI (pipeline_runner)](#15-batch-cli)
16. [AWS deployment](#16-aws-deployment)
17. [Module & file map](#17-module--file-map)
18. [Testing](#18-testing)
19. [Limitations & roadmap](#19-limitations--roadmap)
20. [End-to-end worked example](#20-end-to-end-worked-example)
21. [Jupyter notebook usage guide](#21-jupyter-notebook-usage-guide)

---

## 1. What it is

Standard RAG chops documents into chunks and stores a vector per chunk — throwing
away the reasoning (cause/effect, negation, who-claimed-it, exact numbers).
`graphrag_stage1` keeps all of it. It runs a **multi-pass pipeline** (modeled on a
compiler) that:

- **Stage 1** — breaks a paragraph into atomic, self-contained **facts**, labels
  each along orthogonal facets, and anchors every fact to its exact source span.
- **Stage 2** — turns each fact into a structured, verified **semantic frame** with
  a grounding check and a computed confidence.
- **Stage 3** — maps the verified frames onto a formal **ontology** (BFO/CCO/IAO +
  your domain) and emits **RDF** ready for a graph database.

It is a **library** (not an app, not a service): you install it, hand it your own
language model, and call functions. It brings no model of its own.

**It works on text**, not PDFs. Extract text upstream (Textract, PyMuPDF, GROBID).

### Pipeline at a glance

```
            Scientific Paper
                   │
                   ▼
        +------------------+  ┐
        | Stage 1          |  │
        | Atomic Facts     |  │
        +------------------+  │
                   │         │
                   ▼         │
        +------------------+  │
        | Stage 2          |  │
        | Semantic Frames  |  │
        +------------------+  │
                   │         │   graphrag_stage1
                   ▼         │   (what THIS library does)
        +------------------+  │
        | Stage 3          |  │
        | Ontology Mapping |  │
        +------------------+  │
                   │         │
                   ▼         │
        +------------------+  │
        | RDF / JSON-LD    |  │
        +------------------+  │
                   │         │
                   ▼         │
        +------------------+  │
        | SHACL Validation |  │  ← publication gate
        +------------------+  ┘
                   │
                   ▼
        +------------------+  ┐
        | Neptune          |  │
        +------------------+  │
                   │         │
                   ▼         │
        +------------------+  │   Surrounding system
        | OWL Reasoner     |  │   (you build / provide this;
        +------------------+  │    outside this library)
                   │         │
                   ▼         │
        +------------------+  │
        | GraphRAG         |  │
        +------------------+  ┘
                   │
                   ▼
        Evidence-backed Answers
```

> **Scope:** this library takes you from *Scientific Paper (as text)* down through
> the **SHACL-validated N-Quads** — everything in the top bracket. Loading Neptune,
> OWL reasoning, the GraphRAG retrieval layer, and answer generation are the
> surrounding system you build around it (see [§16 AWS deployment](#16-aws-deployment)).
>
> **Important — Neptune does not perform OWL reasoning.** Amazon Neptune stores RDF
> and answers SPARQL queries; it does **not** automatically infer new facts from the
> ontology's axioms. OWL reasoning / inferred-fact materialization is a **separate
> downstream step** (the design calls it *Stage 3.5*): a reasoner reads the loaded
> graph + ontology, derives implied triples (e.g. subclass and transitive
> `part-of` chains), and writes them back. It is **not** part of this library and is
> **not** triggered by loading Neptune — you add it yourself.

---

## 2. Install

```bash
pip install graphrag-stage1                       # Stage 1 + Stage 2 (dep: requests only)
pip install "graphrag-stage1[ontology]"           # + Stage 3 mapping (rdflib, pyshacl)
pip install "graphrag-stage1[validation]"         # + validate() (jsonschema)
pip install "graphrag-stage1[anthropic]"          # + AnthropicClient (anthropic SDK)
pip install "graphrag-stage1[openai]"             # + OpenAIClient (openai SDK)
pip install "graphrag-stage1[ontology,validation,anthropic]"   # combine extras
```

Requires **Python 3.10+**. Importing the package pulls in nothing heavy — `rdflib`
loads only when you touch Stage 3.

Install straight from source control:

```bash
pip install "graphrag-stage1[ontology] @ git+https://github.com/<you>/<repo>"
```

The **foundational ontologies (IAO, CCO, RO, alignment + SHACL shapes) ship inside the
wheel**, so `[ontology]` gives you a working Stage 3 with nothing else to download. You
supply only your **domain** ontology via `domain_ontology=` — see [§10](#10-ontologies).
(Not published to PyPI yet; on AWS you install it into your container image with
`pip install .`.)

---

## 3. Quick start

```python
from graphrag_stage1 import AnthropicClient, analyze_paper, validate

results = analyze_paper(
    open("paper.txt").read(),
    client=AnthropicClient(),          # or OpenAIClient(), OllamaClient(), or your own
    max_concurrency=16,                # speed dial (match your API rate limit)
    domain_ontology="my_domain.owl",   # omit to skip Stage 3
)

for r in results:                      # one entry per paragraph, in order
    if "error" in r:                   # a failed paragraph never sinks the run
        continue
    validate(r["stage1"]); validate(r["stage2"])
    for fact in r["stage1"]["statements"]:
        print(fact["facets"]["relation"], "|", fact["text"])
```

That is the whole thing. Everything below explains the pieces.

---

## 4. The three stages

| Stage | Function | LLM? | Produces |
|-------|----------|------|----------|
| **Stage 1** — decompose + classify + validate | `process_paragraph` | Yes | Knowledge Artifact Graph (facts + discourse edges + audit) |
| **Stage 2** — semantic frames | `stage2_pipeline` | Yes | Verified frames + statement graph + confidence |
| **Stage 3** — ontology mapping | `stage3_pipeline` | **No** (deterministic) | RDF assertions + JSON-LD |
| **Stage 3 export** — Neptune | `export_neptune_nquads` | No | SHACL-validated N-Quads file |

- **Stage 1 internally** makes ~2–3 LLM calls per paragraph: decompose →
  batch-classify, plus a recall-check *only when the deterministic recall floor says a
  clause went missing*. Deterministic "cue" rules (negation, numbers, "shall", hedges)
  pre-fill or override facets.
- **Stage 2** makes ~1 call per **batch of 5** facts (`STAGE2_BATCH_SIZE`, on by
  default), with an optional bounded retry that can escalate to a stronger model.
- **Stage 3** makes **no** LLM calls — it is pure ontology lookup + RDF minting.
- **Export** runs a SHACL gate: conforming triples are written; non-conforming ones
  are quarantined to a separate file (never uploaded).

---

## 5. Inputs

| You are feeding… | Accepted format | Used by |
|------------------|-----------------|---------|
| **Document content** | **Plain text** (Python `str`) — e.g. a `.txt`/`.md` file or text extracted from a PDF | `analyze_paper`, `process_paragraph` |
| **Paragraph list** | `list[dict]`, each `{"text": str, "paragraph_id"?: str, "source_metadata"?: dict}` | `run_paper` |
| **Batch file (CLI)** | **JSONL**, one paragraph per line: `{"text": ..., "metadata": {...}}` | `pipeline_runner.py` |
| **Ontology** | `.owl` `.rdf` `.ttl` `.nt` `.n3` `.jsonld` | `OntologyManager` |
| **SHACL shapes** | `.ttl` | `export_neptune_nquads` |

**`source_metadata`** (optional, carried into every fact's provenance):
```python
{"document_id": "doc42", "page": 7, "source_uri": "s3://bucket/doc42.pdf"}
```

> **Not accepted:** raw PDF/DOCX/HTML. Convert to text first. There is no built-in
> PDF parser (the design's "L0 ingestion" is roadmap).

---

## 6. Outputs

Everything is plain, JSON-serializable Python dicts (except the final `.nq` file).

### 6.1 Stage 1 output — the Knowledge Artifact Graph

```jsonc
{
  "stage": "stage_1_information_artifact_analysis",
  "pipeline_version": "1.1",
  "model": "<client.model_id>",          // provenance: which model produced this
  "generated_at": "<ISO-8601 UTC>",
  "paragraph_id": "p1",
  "source_metadata": { ... },
  "paragraph": "<the input text>",
  "statements": [ <fact>, ... ],         // the atomic facts
  "relations": [ <discourse_edge>, ... ],// reasoning links between facts
  "audit": { recall floor, coverage, completeness, ... }
}
```

**A `statement` (one fact):**
```jsonc
{
  "id": "p1:u1",
  "text": "The fused sensor data reduced positioning error by 35 percent.",
  "decomposition_text": "...",           // pre-decontextualization form
  "recovered": false,                    // was it recovered by the recall pass?
  "provenance": {
    "paragraph_id": "p1", "document_id": null, "page": null, "source_uri": null,
    "char_start": 0, "char_end": 62,
    "verbatim": "The fused sensor data reduced positioning error by 35 percent.",
    "match": "exact", "similarity": 1.0
  },
  "fidelity": { "source_numbers": [...], "numbers_preserved": true, "repairs": [...] },
  "predicate": "reduce",
  "arguments": [ {"role": "arg1", "text": "..."}, {"role": "arg2", "text": "..."} ],
  "facets": {                            // see §13 for the full vocabulary
    "role": "CONTENT", "proposition_type": "MEASUREMENT", "relation": "CAUSATION",
    "modality": "OBSERVED", "attribution": "PRESENT_WORK", "certainty": "LIKELY",
    "polarity": "POSITIVE", "has_measurement": true
  },
  "cues": { "negation": false, "hedges": [], "connectives": [], "quantities": true, ... },
  "confidence": { "overall": 0.84, "method": "single" },
  "reason": "<model's short justification>"
}
```

**A `discourse_edge` (reasoning link):**
```jsonc
{ "from": "p1:u3", "to": "p1:u2", "type": "EXPLANATION" }   // types: see §13
```

### 6.2 Stage 2 output — semantic frames

```jsonc
{
  "stage": "stage_2_semantic_frame_extraction",
  "pipeline_version": "1.2",
  "model": "<client.model_id>",
  "generated_at": "...",
  "paragraph_id": "p1",
  "frames": [ <frame>, ... ],
  "stage1_relations": [ ... ],
  "statement_graph": { "nodes": {...}, "edges": [...] },   // normalized graph
  "audit": { "frame_count": N, "statements_processed": N,
             "all_statements_processed": true, "concurrency": 4,
             "automation_action_counts": {...} }
}
```

**A `frame`:**
```jsonc
{
  "statement_id": "p1:u1",
  "source_text": "...",
  "stage1_type": "MEASUREMENT",
  "stage1_facets": { ... },
  "provenance": { ... },
  "processing": { "attempt": 1, "model": "<resolved model>" },
  "stage2_frame": {
    "frame_type": "MEASUREMENT",         // see §13
    "semantic_frame": { "primary_entity": ..., "value": "35", "unit": "percent",
                        "process": ..., "condition": ..., "modality": ..., "polarity": ...,
                        "evidence_statement_ids": [...], "measurement": {...} },
    "validation": { "missing_required_fields": [...], "invented_information": false,
                    "automation_action": "PASS_TO_ONTOLOGY_MAPPING" },   // see §13
    "confidence": 0.9,
    "reference_resolutions": [...]
  }
}
```

### 6.3 Stage 3 output — ontology mapping

```jsonc
{
  "stage": "stage_3_ontology_mapping",
  "pipeline_version": "2.0",
  "paragraph_id": "p1",
  "ontology_profile": { loaded ontologies, reasoning boundary, ... },
  "mapped_frames": [ { individuals, object/datatype assertions, evidence, provenance,
                       mappingStatus: "mapped" | "partial_mapping" | "skipped" }, ... ],
  "audit": { lookup audits: missing vs ambiguous terms, candidate IRIs, ... },
  "jsonld": { "@context": {...}, "@graph": [...] }   // the RDF projection
}
```

### 6.4 JSON-LD

`stage3["jsonld"]` is a standard JSON-LD envelope (`@context` + `@graph`). Use it
directly, or feed it to the Neptune exporter.

### 6.5 Neptune N-Quads (the final upload file)

`export_neptune_nquads(...)` writes a **`.nq`** file: SHACL-validated RDF, every
document wrapped in its own **named graph** (`urn:graphrag:graph:<hash>`). It
returns metadata:
```jsonc
{ "format": "nquads", "named_graph": "urn:graphrag:graph:...",
  "destination": "/abs/path/paragraph0.nq",
  "validated_triple_count": 194, "quarantined_triple_count": 0, "quarantined_nodes": [] }
```
Optional side files: a quarantine `.ttl` (rejected triples) and a SHACL report `.txt`.

---

## 7. How to get every output

Each row is a copy-paste recipe. `client` is any `LLMClient` (see §9).

| Output you want | Call | Result |
|-----------------|------|--------|
| **Whole paper, all stages** | `analyze_paper(text, client=c, domain_ontology="d.owl")` | `list` of `{stage1, stage2, stage3}` per paragraph |
| **Whole paper, Stage 1+2 only** | `analyze_paper(text, client=c)` | `list` of `{stage1, stage2}` |
| **Whole paper, parallel, pre-split** | `run_paper(paras, client=c, max_concurrency=16)` | ordered `list` of results |
| **One paragraph, all stages** | `run_pipeline(text, client=c, ontology=mgr)` | `{stage1, stage2, stage3}` |
| **Stage 1 only** (facts) | `process_paragraph(text, client=c)` | Stage 1 dict (§6.1) |
| **Stage 2 from a Stage 1 dict** | `stage2_pipeline(stage1, client=c)` | Stage 2 dict (§6.2) |
| **Stage 3 from a Stage 2 dict** | `stage3_pipeline(stage2, mgr)` | Stage 3 dict (§6.3) |
| **JSON-LD** | `stage3["jsonld"]` | JSON-LD dict (§6.4) |
| **Neptune N-Quads file** | `export_neptune_nquads(stage3, "out.nq")` | writes `.nq`; returns metadata (§6.5) |
| **Just the facts as text** | `[s["text"] for s in stage1["statements"]]` | `list[str]` |
| **Just the fact labels** | `[s["facets"] for s in stage1["statements"]]` | `list[dict]` |
| **Just relationships** | `stage1["relations"]` | `list` of discourse edges |
| **Schema-validate an output** | `validate(stage1)` / `validate(stage2)` | returns it, or raises `SchemaValidationError` |
| **Load ontologies** | `OntologyManager().load_all(domain_ontology="d.owl")` | an `OntologyManager` (core ships in the wheel) |

**Full manual chain (all outputs explicitly):**
```python
from graphrag_stage1 import process_paragraph, stage2_pipeline, stage3_pipeline, OntologyManager
from graphrag_stage1.stage3_neptune_export import export_neptune_nquads

mgr    = OntologyManager().load_all(domain_ontology="my_domain.owl")
s1     = process_paragraph(text, paragraph_id="p1", client=client)   # Stage 1
s2     = stage2_pipeline(s1, client=client)                          # Stage 2
s3     = stage3_pipeline(s2, mgr)                                    # Stage 3
jsonld = s3["jsonld"]                                                # JSON-LD
meta   = export_neptune_nquads(s3, "p1.nq",                          # N-Quads file
                               quarantine_destination="p1_rejected.ttl",
                               report_destination="p1_shacl.txt")
```

---

## 8. Public API reference

Everything importable from `graphrag_stage1`:

### Orchestration
```python
analyze_paper(text, *, client, stage2_client=None, max_concurrency=8,
              stage2_concurrency=4, stage2_batch_size=None, ontology=None,
              domain_ontology=None, ontology_root=None, grounder=None,
              gate_noise=True, on_result=None) -> list[dict]
```
Split raw text into paragraphs and run them in parallel. Pass `domain_ontology`
(a path) or `ontology` (a prebuilt `OntologyManager`) to also run Stage 3.

> ⚠️ **Pass a `grounder` or Stage 3 will type almost nothing.** Without one, an entity
> that is not already a class in the loaded ontology can only be generalised to a *true*
> ancestor — which in practice means `BFO:entity`, the root of the ontology. Measured on
> the 4 most interaction-dense paragraphs of a PMC paper against INO:
>
> | | domain-typed entities |
> |---|---|
> | no grounder | **0 / 10** (every entity → `BFO:entity`) |
> | `OakGrounder()` | **7 / 21 (33%)** — real `CHEBI`, `GO`, `MONDO`, `PR`, `PW` classes |
>
>
> ```python
> from graphrag_stage1.grounding import OakGrounder
> analyze_paper(text, client=c, domain_ontology="ontologies/Domain/ino_merged.owl",
>               grounder=OakGrounder(cache_path=".grounding_cache.json"))
> ```
> The grounder resolves mentions to real term IDs **lexically** (never by asking a model
> to recall a CURIE — models hallucinate ontology IDs, a lexical match does not), and
> caches to disk, so the network is hit once per distinct mention across all runs.

`stage2_client` runs Stage 2 on a **separate model**. Both clients share one global
`max_concurrency` ceiling, so the cap still means what it says.

> ⚠️ **Do not point this at a weaker model to go faster — it backfires.** Measured on
> 43 identical Stage 1 statements, changing *only* the Stage 2 model:
>
> | Stage 2 model | Frames passed | Grounding errors | Wall-clock |
> |---|---|---|---|
> | `qwen3:4b` | 4/43 (9%) | 54 | 268 s |
> | `qwen3:8b` | **25/43 (58%)** | 23 | **243 s** |
>
> The smaller model is **slower and six times worse**. Stage 2's output is
> grounding-checked against the source, and a weak model produces ungrounded frames
> that trigger `REPROCESS_*` — and every retry is a *second full call*. The retries
> cost more than the cheaper model saves. Per-call latency is the wrong thing to
> optimise; total call count is what matters.

Use `stage2_client` to point Stage 2 at a model that is **as strong or stronger**, or
at a different provider (e.g. a hosted endpoint for Stage 2 while Stage 1 stays local).

```python
run_paper(paragraphs, *, client, stage2_client=None, max_concurrency=8,
          stage2_concurrency=4, stage2_batch_size=None, ontology=None,
          grounder=None, gate_noise=True, on_result=None) -> list[dict]
```
Same, but you supply the paragraph list. `on_result(index, result)` fires as each
finishes (for streaming/checkpointing). Order preserved; failures become
`{"paragraph_id", "error", "error_type"}`. Paragraphs dropped by the L1 gate return
`{"paragraph_id", "gated": True, "gate_reason"}` with empty stages — they never reach
a model.

```python
run_pipeline(paragraph, *, client=None, stage2_client=None, paragraph_id="p1",
             source_metadata=None, ontology=None, grounder=None,
             stage2_concurrency=None, stage2_batch_size=None) -> dict
```
One paragraph → `{"stage1", "stage2", ["stage3"]}`. Note `run_pipeline` has **no**
`gate_noise`: L1 gating is a document-level decision and lives in `run_paper`.

```python
split_paragraphs(text) -> list[dict]     # split on blank lines into {"text": ...}
```

### Stages
```python
process_paragraph(paragraph, paragraph_id="p1", source_metadata=None, client=None) -> dict
stage2_pipeline(stage1_output, client=None, concurrency=None) -> dict
stage3_pipeline(stage2_output, manager) -> dict          # manager = OntologyManager
```
If `client=None`, a default local `OllamaClient` is built from env vars.

### Neptune export (import from submodule)
```python
from graphrag_stage1.stage3_neptune_export import export_neptune_nquads
export_neptune_nquads(stage3_output, destination,
                      quarantine_destination=None, report_destination=None,
                      shapes_path=DEFAULT_SHAPES) -> dict   # packaged SHACL shapes
```

### Validation
```python
validate(payload) -> payload            # raises SchemaValidationError on mismatch
SchemaValidationError                   # exception type
```

### Model clients (see §9)
```python
LLMClient                               # the Protocol you implement
OllamaClient(model="qwen3:8b", stronger_model=None, url=..., timeout=180, retries=2,
             temperature=0.0)           # + OllamaClient.from_env(...)
AnthropicClient(model="claude-haiku-4-5", stronger_model="claude-sonnet-5",
                max_tokens=2048, sdk=None)
OpenAIClient(model="gpt-4o-mini", stronger_model="gpt-4o", sdk=None)
BoundedClient(inner_client, max_concurrency)   # global concurrency cap
```

### Ontology & config
```python
OntologyManager(ontology_root=None, instance_namespace="urn:graphrag:instance:")  # None -> packaged core
    .load_all(domain_ontology=None) -> OntologyManager
PipelineConfig                          # max_paragraph_chars, queue_db, audit_log
__version__                             # "0.1.0"
```

---

## 9. Bring your own model

The **only** requirement is one method: given a prompt and a JSON Schema, return a
dict matching it. That is the `LLMClient` protocol.

```python
from graphrag_stage1.llm import LLMClient   # a typing.Protocol (runtime-checkable)

class MyClient:
    model_id = "my-provider/my-model"       # recorded in every output's provenance
    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        ...   # call your model, force JSON matching `schema`, return the dict
```

- **`stronger`** is a hint (Stage 2's escalation route) — route it to a bigger model
  or ignore it.
- **Built-ins** (`AnthropicClient`, `OpenAIClient`) already do the structured-output
  plumbing. Pick a model with `model="..."`.
- **Any OpenAI-compatible endpoint** (vLLM, Together, Groq, LiteLLM, local server):
  `OpenAIClient(sdk=openai.OpenAI(base_url="http://.../v1", api_key="..."))`.
- **Amazon Bedrock** (AWS): write a ~15-line `LLMClient` over `boto3`
  `bedrock-runtime` using Claude tool-use — same shape as `AnthropicClient`.
- **Thread-safety:** `complete` is called concurrently — make it thread-safe.

---

## 10. Ontologies

Stage 3 loads a **fixed core** — which **ships inside the wheel** — plus **one replaceable
domain slot**, which you supply:

```
graphrag_stage1/ontologies/          ← PACKAGED: installed with the library, nothing to fetch
├── iao.owl                          CORE — Information Artifact Ontology
├── CommonCoreOntologiesMerged.ttl   CORE — CCO (also supplies BFO's classes)
├── Relations/ro.owl                 optional — Relation Ontology
└── Alignment/                       the bridge from pipeline vocab → the ontologies
    ├── stage2-alignment-v1.0.ttl
    └── stage3-publication-shapes.ttl   (SHACL rules for the Neptune gate)

<anywhere>/my_domain.owl             ← YOURS: passed via domain_ontology=
```

**BFO ships no file of its own.** CCO and IAO are both built on it and republish its classes
under the canonical `obo/BFO_*` IRIs, so BFO arrives with them. A separate (stale) BFO file is
actively harmful: legacy IFOMIS BFO 1.1 names the same concepts under *different* IRIs, making
every upper-ontology term ambiguous. `missing_upper_classes()` verifies the classes are present
rather than trusting a filename.

**Supply your domain ontology:**
```python
# The core comes from the installed package -- name only your domain file (or a directory):
OntologyManager().load_all(domain_ontology="/path/to/my_domain.owl")

# Override the core location only if you need a pinned / patched copy:
OntologyManager(ontology_root="/path/to/ontologies").load_all(domain_ontology="my_domain.owl")
```

> The domain ontology is deliberately **not** bundled: it is the one module that changes per
> scientific domain, it is your choice, and it can dwarf the core — the whole foundational
> stack is ~3.7 MB, while UBERON alone is 48 MB. The repo keeps examples under the top-level
> `ontologies/Domain/`; those are samples, not part of the library.

- Core ontologies and your domain ontology are **never modified** — the Alignment
  layer does the bridging.
- There are **no hardcoded domain terms in Python**; mappings come from the labels
  inside whatever ontology you load. Swapping domains is plug-and-play.
- Missing/ambiguous ontology terms **fail closed** (logged in the audit), so bad
  mappings don't silently reach the graph. Requirement on your file: proper
  OWL/RDF with human-readable labels; ideally BFO-based like CCO.

---

## 11. Performance & concurrency

Wall-clock ≈ **(total LLM calls × per-call latency) ÷ max_concurrency**.

- **`max_concurrency`** — global cap on in-flight model calls. On a *scalable*
  endpoint this is the main speed dial: set it to your provider's safe concurrent-
  request budget.
- **`BoundedClient`** — enforces that cap no matter how wide the fan-out. `run_paper`
  wraps your client in one automatically.
- **`stage2_concurrency`** — per-paragraph statement fan-out.
- **`stage2_client`** — run Stage 2 on a smaller model than Stage 1 (see §Orchestration).

### Where the time actually goes

Per paragraph: Stage 1 makes **2–3** calls (decompose, batched classify, and the
recall pass *only when the deterministic floor is short* — see below); Stage 2 makes
**one per batch of 5 statements** (~2) plus any reprocess retries. A 6-statement
paragraph therefore costs ~5 calls, not the ~9 it cost before batching became the
default. But call count is not the whole story — **Stage 2 is decode-bound**. Measured
on one paragraph (qwen3, RTX 5080 laptop):

| | prompt | prefill | output | decode | decode share |
|---|---|---|---|---|---|
| Stage 1 decompose | 749 tok | 0.1 s (11k tok/s) | 117 tok | 2.5 s | 86% |
| Stage 2 frame | 2,184 tok | 0.5 s (4.7k tok/s) | **819 tok** | **18.7 s** | **91%** |

Prefill is nearly free; generation is not. A Stage 2 frame emits ~7× the tokens of
a Stage 1 call, and that generation *is* the runtime. Shrinking a prompt saves almost
nothing.

**But per-call latency is the wrong target.** Stage 2 frames are grounding-checked
against the source, and a frame that fails is *reprocessed* — a second full call. So
a weaker, faster model can easily make the run **slower**, by failing more often
(see the ⚠️ under `stage2_client` in §Orchestration: `qwen3:4b` was both slower *and*
6× worse than `qwen3:8b` on identical input). What actually reduces wall-clock is
**fewer total calls**: batch them (below), drop the retry pass
(`STAGE2_MAX_RETRIES=0`), or feed Stage 2 fewer, better statements. Model quality is
a *speed* feature here, not just a quality one.

### Local GPU vs hosted

- **Hosted API** (Claude/OpenAI/Bedrock), `max_concurrency=16`: **~1 minute** for a
  ~50-paragraph paper. Concurrency works because the endpoint scales — *provided your
  quota actually allows 16 concurrent requests*. Concurrency you are not entitled to
  is not speed; see "Concurrency is bounded by quota, not ambition" below.
- **Single local GPU**: raising `max_concurrency` past a few does **not** help — the
  GPU saturates (97–98% utilisation) and extra concurrent requests just time-slice
  the same silicon. Set `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS=2` (so an
  8B Stage 1 and 4B Stage 2 stay resident together rather than swapping), then reduce
  generated tokens.

**Measured, on the defaults this guide now ships** (qwen3:8b, RTX 5080 laptop 16 GB,
4 real 70–75-word paragraphs from a PMC paper, batching on, `num_ctx=6144`):

| Configuration | Wall | Per paragraph | Errors |
|---|---|---|---|
| **Local qwen3:8b, `max_concurrency=4`** | **101 s** | **25 s** | 0 |
| Ollama Cloud `gpt-oss:120b-cloud`, `max_concurrency=4` | 220 s | 55 s | 0 |
| Ollama Cloud `gpt-oss:120b-cloud`, `max_concurrency=12` | 769 s | 154 s | **7 of 12 lost** |

An 80-paragraph paper is therefore **~33 minutes locally** — the "tens of minutes"
this guide has always claimed, but only once batching is on and `num_ctx` is sized.
With the pre-0.2 defaults (batching off, Ollama's 4096 window) the same paper took
**~8 hours**: the shipped defaults, not the pipeline, were the bottleneck.

**A bigger model is not a faster one.** `gpt-oss:120b-cloud` is 2.2× *slower* per
paragraph than a local 8B on a dedicated GPU, and reasoning models
(`nemotron-3-super:cloud`) are slower still because they spend tokens thinking before
answering. What a bigger model buys is a better graph — in the run above it produced
7 fully-`mapped` frames against local's 3, and was the only model to reach a real
domain class rather than falling back to `BFO:entity`. Choose cloud for *quality*;
choose parallel capacity for *speed*. They are different purchases.

### Batched Stage 2 — on by default since 0.2 (`stage2_batch_size`)

Unbatched, Stage 2 makes **one call per statement**. Batching frames several statements
in one call, and it is faster *and* more accurate — so since 0.2 it is the **default**
(`STAGE2_BATCH_SIZE=5`). Set `STAGE2_BATCH_SIZE=0` to restore the per-statement path.
Measured on 43 identical statements (`qwen3:8b`), varying only the batch size:

| Stage 2 | Time | Frames accepted | Grounding errors |
|---|---|---|---|
| no batching | 228.0 s | 25/43 (58%) | 17 |
| **batch 5** | **130.5 s** | **31/43 (72%)** | **1** |
| batch 10 | 121.8 s | 28/43 (65%) | 3 |
| batch 20 | 433.9 s ← worse | 28/43 (65%) | 4 |

**Batch 5 is 1.75× faster and more accurate.** The speedup is not just amortised
prompt overhead (prefill is only ~3% of a call): showing the model its sibling
statements measurably improves *grounding*, which collapses the `REPROCESS` retries —
and every retry was a second full call.

```python
analyze_paper(text, client=c, stage2_batch_size=5)
```

**Do not go past ~10.** A batch of N needs roughly N × 800 output tokens, so the
context window must be sized for it or the batch silently truncates into invalid JSON.
At batch 20 the 32K context thrashes VRAM on a 16 GB card and the run gets *slower*.

### ⚠️ `num_ctx` × `OLLAMA_NUM_PARALLEL` — the sharpest edge in the whole pipeline

`OllamaClient(num_ctx=...)` defaults to **6144** since 0.2, sized for a batch of 5
(~2.2k prompt + 5 × 800 output). It is coupled to batching in *both* directions, and
getting it wrong is expensive in opposite ways:

- **too small** → every batch truncates, falls back to the per-statement path, and the
  run is *slower than not batching at all*;
- **too large** → Ollama allocates `num_ctx` **per parallel slot**. With a common
  `OLLAMA_NUM_PARALLEL=4`, `num_ctx=8192` reserves **32K** of KV cache, which thrashes
  VRAM alongside a 7.4 GB model on a 16 GB card.

Measured on 4 real paragraphs (qwen3:8b, RTX 5080 16 GB, `OLLAMA_NUM_PARALLEL=4`,
batch 5, changing **only** this one value):

| `num_ctx` | Wall | Outcome |
|---|---|---|
| 8192 | **1026 s** | 1 paragraph failed outright (thrashing) |
| **6144** | **101 s** | 0 failures |

**A 10× swing from one number.** If you raise `num_ctx`, lower `OLLAMA_NUM_PARALLEL`
to match your VRAM — otherwise the headroom you were buying for a bigger batch is
exactly what thrashes.

### Concurrency is bounded by quota, not ambition

`max_concurrency` is a *budget*, not a throttle to open as far as it goes. Set it **to**
your provider's safe concurrent-request allowance. Measured on Ollama Cloud
(`gpt-oss:120b-cloud`, free tier): `max_concurrency=4` completes cleanly;
`max_concurrency=12` returns HTTP 429 on **7 of 12 paragraphs**, and each one burns the
full 429 backoff before failing — so the run is both **3× slower per surviving
paragraph and loses more than half the paper**. Past your quota you do not buy speed;
you buy waiting and data loss. `BoundedClient` exists to hold this budget across
arbitrary fan-out — give it the right number.

### The recall pass is now conditional (`STAGE1_ALWAYS_RECALL`)

The L7 recall pass costs one LLM call per paragraph. Since 0.2 it runs **only when the
deterministic recall floor is short** — i.e. when the symbolic layer has positive
evidence that decomposition dropped a clause. A decomposition that already met its
floor has no known gap, and on the gold suite skipping the call changed **nothing**
(13/13 artifact recall, 8/8 unit-count, identical consistency flags). Set
`STAGE1_ALWAYS_RECALL=1` to force it on every paragraph if you want maximum recall over
latency.

### L1 boilerplate gating (`gate_noise`)

`run_paper`/`analyze_paper` drop journal front/back matter — `Keywords:` lines,
supplementary-material DOIs, publisher notices — **before** they cost a model call
(`gate_noise=True`, the default). The gate is deliberately narrow: on a real 63-paragraph
PMC paper it fires on **3** paragraphs (~5%). It does *not* use length or punctuation
heuristics, because the shortest paragraphs in a real paper are figure captions ("A Venn
diagram illustrating the overlap between…") and methods lines ("GAPDH was used as an
internal control") — both content. A dropped paragraph is unrecoverable downstream, so
everything ambiguous is left to Stage 1's `role` facet. Pass `gate_noise=False` to
disable.

Safety: a batch that returns the wrong number of frames, or raises, falls back to the
proven per-statement path — no facts are dropped. Frames are re-aligned to their
statements by `statement_id`, not by position.

---

## 12. Validation & schemas

Versioned JSON Schemas ship in the wheel and describe the IR contract:

```
graphrag_stage1/schemas/stage1-1.1.json
graphrag_stage1/schemas/stage2-1.2.json
```

`validate(payload)` selects the schema from the payload's own `stage` +
`pipeline_version`, and raises `SchemaValidationError` (with a field path) on the
first violation. The schemas lock the **envelope, provenance anchors, and facet
enums** (hard guarantees) while leaving LLM-decided nested content open for
forward compatibility. Pin your integration to a `pipeline_version`.

---

## 13. The faceted vocabulary

**Stage 1 facets** (each an independent axis):

| Facet | Allowed values |
|-------|----------------|
| `role` | CONTENT · NOISE · METADATA · REFERENCE · FIGURE · TABLE · EQUATION · CAPTION |
| `proposition_type` | DEFINITION · CLASSIFICATION · PROPERTY · MEASUREMENT · RELATION · PROCESS · METHOD · EVENT |
| `relation` | NONE · CORRELATION · CAUSATION · MECHANISM · COMPARISON · PART_OF · IS_A · DEPENDENCY · TEMPORAL_ORDER |
| `modality` | ASSERTED · OBSERVED · MEASURED · PREDICTED · HYPOTHESIZED · ASSUMED · DEFINED · REQUIRED |
| `attribution` | PRESENT_WORK · PRIOR_WORK · MODEL_THEORY · DATA · THIRD_PARTY · UNKNOWN |
| `certainty` | CERTAIN · LIKELY · POSSIBLE · UNCERTAIN |
| `polarity` | POSITIVE · NEGATED |
| `has_measurement` | true / false |

**Discourse edge types:** EXPLANATION · CAUSE · EVIDENCE · SUPPORT · CONTRAST ·
CONCESSION · CONDITION · ELABORATION · BACKGROUND · CONTRADICTION

**Stage 2 frame types:** PREDICTION · OBSERVATION · MECHANISM · CORRELATION ·
CAUSAL_RELATION · METHOD · ASSUMPTION · HYPOTHESIS · MEASUREMENT · COMPARISON ·
DEFINITION · CLASSIFICATION · PART_WHOLE · DEPENDENCY · TEMPORAL_RELATION ·
SUPPORT · EVIDENCE · CONSISTENCY · VALIDATION · CLAIM

**Stage 2 automation actions** (`validation.automation_action`):
PASS_TO_ONTOLOGY_MAPPING · REPROCESS_WITH_CONTEXT · REPROCESS_WITH_STRONGER_MODEL ·
SEND_TO_LOW_CONFIDENCE_QUEUE · REJECT_LOW_VALUE · ROUTE_TO_TABLE_EXTRACTOR ·
ROUTE_TO_EQUATION_EXTRACTOR · ROUTE_TO_FIGURE_EXTRACTOR. Only
`PASS_TO_ONTOLOGY_MAPPING` frames are mapped by Stage 3; others are retained as
skipped records.

---

## 14. Configuration & environment variables

Read when a **default** client/config is used (you can always pass explicit objects):

| Variable | Default | Effect |
|----------|---------|--------|
| `STAGE1_MODEL` | `qwen3:8b` | Default Ollama model for Stage 1 |
| `STAGE2_MODEL` | `qwen3:8b` | Default Ollama model for Stage 2 |
| `STAGE2_STRONGER_MODEL` | = `STAGE2_MODEL` | Model used on the `stronger` retry |
| `STAGE2_MAX_RETRIES` | `1` | Bounded automated reprocessing attempts |
| `STAGE2_CONCURRENCY` | `2` | Default statement fan-out |
| `STAGE2_BATCH_SIZE` | `5` | Frame N statements per model call. **On by default since 0.2** — faster *and* more accurate. `0` restores the per-statement path; do not exceed ~10 (see §11) |
| `STAGE1_ALWAYS_RECALL` | unset | Run the L7 recall pass on every paragraph instead of only when the deterministic floor is short. Costs ~1 extra call per paragraph (see §11) |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `OLLAMA_TIMEOUT` | `180` | Per-call timeout (s) |
| `OLLAMA_RETRIES` | `2` | Transport retries |
| `MAX_PARAGRAPH_CHARS` | `50000` | Reject oversized paragraphs |
| `PIPELINE_QUEUE_DB` | `pipeline_queue.sqlite3` | CLI routing queue path |
| `PIPELINE_AUDIT_LOG` | `pipeline_audit.jsonl` | CLI audit log path |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Read by the built-in clients |

---

## 15. Batch CLI

`pipeline_runner.py` (a root script) processes a JSONL file with resumable,
append-only checkpointing and a durable routing queue for frames that can't
auto-advance.

```bash
python pipeline_runner.py input.jsonl output.jsonl
```
`input.jsonl` lines: `{"text": "...", "metadata": {"document_id","page","source_uri"}}`
(source metadata is **required** here). Output is JSONL with `stage1`, `stage2`, and
routing info; already-completed ids are skipped on re-run.

---

## 16. AWS deployment

```
S3 (PDFs) ─► Textract ─► text
                          ├─► [existing chunker+embedder] ─► vector store (keep)
                          └─► graphrag_stage1 on Fargate/Batch
                                 └─ LLMClient → Amazon Bedrock (Claude)
                                 ├─► Stage 1/2 JSON → OpenSearch / S3 / DynamoDB
                                 └─► export_neptune_nquads → .nq → S3 → Neptune loader
Query: vector store (recall) + Neptune (relations, citations) → Claude
```

Notes: install into a **container image** (`pip install .`); prefer **Fargate/Batch**
over Lambda for batches (deps + no 15-min limit); set `max_concurrency` to your
**Bedrock quota**; pair with **Textract** for the PDF→text step.

---

## 17. Module & file map

**Package (`graphrag_stage1/`):**
| Module | Role |
|--------|------|
| `__init__.py` | public API (lazy re-exports) |
| `pipeline.py` | `analyze_paper`, `run_paper`, `run_pipeline`, `split_paragraphs` |
| `llm.py` | `LLMClient`, `OllamaClient`, `BoundedClient` |
| `clients.py` | `AnthropicClient`, `OpenAIClient` |
| `stage1_classifier.py` | Stage 1 (decompose/classify/validate + cue rules) |
| `stage2_semantic_frames.py` | Stage 2 (frames, grounding, confidence, routing) |
| `stage3_ontology_mapper.py` | Stage 3 (ontology mapping) |
| `stage3_jsonld.py` | JSON-LD envelope builder |
| `stage3_neptune_export.py` | SHACL gate + N-Quads export |
| `ontology_manager.py` | ontology loading, indexing, instance minting, RDF |
| `validation.py` | `validate`, `SchemaValidationError` |
| `production_support.py` | `PipelineConfig`, audit logger, routing queue |
| `entity_mapper.py`, `relation_mapper.py`, `measurement_mapper.py`, `assertion_builder.py`, `provenance_builder.py`, `canonicalizer.py`, `semantic_typer.py`, `instance_generator.py`, `relation_typer.py` | Stage 3 mapping helpers |
| `adjudication_workflow.py`, `validate_adjudication.py` | human-review workflow |
| `schemas/*.json` | versioned IR JSON Schemas |

**Root (not part of the installed package):** `pipeline_runner.py` (batch CLI),
`eval*.py` / `readiness_check.py` / `build_osti_corpus.py` (dev/eval), `test_*.py`,
`examples/`, `ontologies/`, `DESIGN.md`, `INTEGRATION.md`, this file.

---

## 18. Testing

```bash
pip install "graphrag-stage1[ontology,validation]" pytest
python -m pytest -q            # 71 deterministic tests, no LLM/network needed
```
Suites: `test_pipeline.py` (Stage 1/2 contracts), `test_stage3.py` (mapping +
Neptune export), `test_schemas.py` (schema/enum sync + validate), `test_throughput.py`
(parallelism + concurrency cap), `test_clients.py` (built-in clients via fake SDKs).

---

## 19. Limitations & roadmap

- **Confidence is uncalibrated.** The evaluation set is small relative to the release
  gate (`release_gates.json`: 500 paragraphs / 50 docs / 8 domains). Trust the
  deterministic guarantees (numeric-token preservation, provenance, schema validity);
  do **not** gate decisions on the numeric confidence until calibrated. **This is the
  top priority before production trust.**
- **No PDF ingestion (L0).** Feed pre-extracted text (Textract/PyMuPDF/GROBID).
- **No cross-sentence coreference (L4) yet.** Roadmap.
- **Not on PyPI.** Install from source / into a container image.
- **No Bedrock client shipped yet** (write one via `LLMClient`, or use `AnthropicClient`).
- **Mapping quality** depends on a domain ontology that matches your papers and on
  model strength; a small/off-domain setup produces rougher type choices.

---

## 20. End-to-end worked example

```python
"""PDF → facts → verified frames → ontology graph → Neptune N-Quads."""
import fitz  # PyMuPDF
from graphrag_stage1 import AnthropicClient, run_paper, validate, OntologyManager
from graphrag_stage1.stage3_neptune_export import export_neptune_nquads

# 1. PDF -> text (your upstream step; Textract on AWS)
text = "\n\n".join(p.get_text() for p in fitz.open("paper.pdf"))
paras = [{"text": p, "source_metadata": {"document_id": "paper", "page": 1}}
         for p in text.split("\n\n") if p.strip()]

# 2. Load core + your domain ontology once
ontology = OntologyManager("ontologies/").load_all(domain_ontology="my_domain.owl")

# 3. Run the whole paper in parallel, all three stages
results = run_paper(paras, client=AnthropicClient(model="claude-haiku-4-5"),
                    max_concurrency=16, ontology=ontology)

# 4. Validate + export each paragraph's graph to a Neptune N-Quads file
for i, r in enumerate(results):
    if "error" in r:
        print(f"[{i}] failed: {r['error']}"); continue
    validate(r["stage1"]); validate(r["stage2"])
    meta = export_neptune_nquads(r["stage3"], f"out/paragraph{i}.nq")
    print(f"[{i}] {len(r['stage1']['statements'])} facts, "
          f"{meta['validated_triple_count']} triples -> {meta['destination']}")

# 5. Upload out/*.nq to S3, then run the Neptune bulk loader.
```

---

## 21. Jupyter notebook usage guide

A complete, runnable Jupyter notebook is available at **`examples/jupyter_usage_demo.ipynb`**. It demonstrates every major workflow with live code cells you can execute.

### 21.1 Open the notebook

```bash
# From the repo root
jupyter lab examples/jupyter_usage_demo.ipynb
# or
jupyter notebook examples/jupyter_usage_demo.ipynb
```

### 21.2 Notebook contents

| Section | What it covers |
|---------|----------------|
| **1. Installation** | `%pip install graphrag-stage1` + optional extras (`[ontology]`, `[anthropic]`, `[openai]`, `[dev]`) |
| **2. Imports & LLM Client Setup** | All imports; choose Anthropic, OpenAI, Ollama, or custom `LLMClient` |
| **3. Stage 1 Only** | `process_paragraph()` → Knowledge Artifact Graph (propositions, discourse relations, coreference clusters) |
| **4. Stage 1 → 2** | `stage2_pipeline()` → Semantic frames with typed slots, grounding checks, confidence |
| **5. Stage 3** | `OntologyManager` + `run_pipeline(ontology=...)` → RDF/JSON-LD + Turtle export |
| **6. Full Paper** | `analyze_paper()` / `run_paper()` for multi-paragraph parallel processing |
| **7. Validation** | `validate()` against versioned JSON Schemas |
| **8. Exploration & Debugging** | Inspect provenance, mapping audits, enable debug logging |
| **9. Patterns & Tips** | Custom clients, checkpointing, error handling, performance tuning, output formats |

### 21.3 Minimal runnable example

```python
# Cell 1: Install
%pip install graphrag-stage1

# Cell 2: Imports + client
from graphrag_stage1 import run_pipeline, OllamaClient
client = OllamaClient()  # or AnthropicClient(), OpenAIClient()

# Cell 3: Run pipeline
text = "The fused sensor data reduced positioning error by 42% because it provides a more complete state estimate."
result = run_pipeline(text, client=client, paragraph_id="demo:p1")

# Cell 4: Inspect outputs
kag = result["stage1"]      # Knowledge Artifact Graph
frames = result["stage2"]   # Validated semantic frames

print(f"Propositions: {len(kag['propositions'])}")
print(f"Frames: {len(frames['frames'])}")
for f in frames["frames"]:
    print(f"  {f['frame_type']}: confidence={f['confidence']:.2f}")
```

### 21.4 With Stage 3 (ontology mapping)

```python
# Requires: %pip install "graphrag-stage1[ontology]"
# And: ontologies/ directory with core + your domain ontology

from graphrag_stage1.ontology_manager import OntologyManager

ontology = OntologyManager("ontologies").load_all(domain_ontology="Domain/your_domain.owl")

result = run_pipeline(
    text, client=client, paragraph_id="demo:p1",
    ontology=ontology  # <-- triggers Stage 3
)

stage3 = result["stage3"]
print(f"Mapped frames: {len(stage3['mapped_frames'])}")
print(f"Ontology individuals: {len(stage3['ontology_individuals'])}")

# Export for Neptune / GraphDB
from graphrag_stage1.stage3_jsonld import build_jsonld
jsonld = build_jsonld(stage3)
import json
with open("output.jsonld", "w") as f:
    json.dump(jsonld, f, indent=2)
```

### 21.5 Key patterns for notebook workflows

| Pattern | Code snippet |
|---------|--------------|
| **Checkpointing** | `run_paper(paras, client=c, on_result=lambda i, r: save_checkpoint(i, r))` |
| **Error handling** | `if "error" in r: log(r["error"]); continue` |
| **Provenance citation** | `prop["provenance"]["source_statement"]` gives verbatim text span |
| **Debug LLM calls** | `logging.getLogger("graphrag_stage1.llm").setLevel(logging.DEBUG)` |
| **Batch + aggregate** | Collect all `stage3["mapped_frames"]` → merge → single KG |

### 21.6 Environment variables for local development

```bash
# .env or shell export
export STAGE1_MODEL="qwen3:8b"
export STAGE2_MODEL="qwen3:8b"
export STAGE2_STRONGER_MODEL="qwen3:14b"
export OLLAMA_URL="http://localhost:11434/api/generate"
```

Or set in notebook:
```python
import os
os.environ["STAGE1_MODEL"] = "qwen3:8b"
os.environ["STAGE2_MODEL"] = "qwen3:8b"
```

---

*graphrag_stage1 v0.1.0 — turn scientific text into a typed, provenance-anchored,
ontology-mapped knowledge graph. See DESIGN.md for the full architecture rationale.*
