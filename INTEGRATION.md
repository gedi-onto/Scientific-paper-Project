# Integrating `graphrag_stage1` into a RAG platform

This library turns a raw paragraph into a typed, provenance-anchored knowledge
representation (a Knowledge Artifact Graph) and validated semantic frames, ready
to feed a graph store or an ontology mapper. It is designed to be embedded in a
host platform: you bring your own language model, we do the extraction.

For the full design rationale see [DESIGN.md](DESIGN.md). This document is the
practical integration contract.

---

## 1. Install

```bash
pip install graphrag-stage1                 # Stage 1 + Stage 2 (dep: requests only)
pip install "graphrag-stage1[ontology]"     # + Stage 3 ontology mapping (rdflib, pyshacl)
```

Requires Python 3.10+. Importing the package pulls in **nothing heavy** — `rdflib`
is imported only when you actually touch Stage 3.

---

## 2. Bring your own model: implement `LLMClient`

The pipeline needs exactly one capability from a model: given a prompt and a JSON
Schema, return schema-valid JSON. There is **no built-in assumption of a local
server** — you inject a client.

```python
from graphrag_stage1.llm import LLMClient   # a typing.Protocol

class MyClient:
    model_id = "my-provider/my-model-2025-01"   # recorded in provenance

    def complete(self, prompt: str, schema: dict, *, stronger: bool = False) -> dict:
        # Call your provider with `schema` as a structured-output / JSON-schema
        # constraint and return the parsed dict. `stronger` is an optional hint
        # that the caller wants a more capable model for this call (Stage 2's
        # escalation route); you may ignore it or route to a bigger model.
        ...
```

`isinstance(MyClient(), LLMClient)` is `True` — the protocol is runtime-checkable.

**Default:** if you pass no client, each stage builds a local
[`OllamaClient`](graphrag_stage1/llm.py) from environment variables
(`STAGE1_MODEL`, `STAGE2_MODEL`, `STAGE2_STRONGER_MODEL`, `OLLAMA_URL`,
`OLLAMA_TIMEOUT`, `OLLAMA_RETRIES`). Good for local dev; supply your own client in
production.

---

## 3. Run it

```python
from graphrag_stage1 import run_pipeline

out = run_pipeline(
    "The fused sensor data reduced positioning error because it gives a more "
    "complete state estimate.",
    client=MyClient(),
    paragraph_id="doc42:p7",
    source_metadata={"document_id": "doc42", "page": 7, "source_uri": "s3://..."},
)

kag    = out["stage1"]   # Knowledge Artifact Graph
frames = out["stage2"]   # validated semantic frames
```

Lower-level entry points if you want to run stages separately:

```python
from graphrag_stage1 import process_paragraph, stage2_pipeline
kag    = process_paragraph(text, paragraph_id="p1", client=MyClient())
frames = stage2_pipeline(kag, client=MyClient())
```

Stage 3 (optional, needs the `[ontology]` extra) maps frames onto an OWL
ontology:

```python
from graphrag_stage1 import OntologyManager
ontology = OntologyManager("ontologies/").load_all()
out = run_pipeline(text, client=MyClient(), ontology=ontology)
graph_records = out["stage3"]
```

---

## 4. The output contract (what you consume)

Both stage outputs are plain JSON-serializable dicts. Top-level shape:

```
stage1 = {
  "stage": "stage_1_information_artifact_analysis",
  "pipeline_version": "1.1",
  "model": "<client.model_id>",          # provenance: which model produced this
  "generated_at": "<ISO-8601 UTC>",
  "paragraph_id", "source_metadata", "paragraph",
  "statements": [ <knowledge artifact>, ... ],
  "relations":  [ <discourse edge>, ... ],
  "audit":      { recall floor, coverage, completeness, ... }
}

statement = {
  "id", "text", "decomposition_text", "recovered",
  "provenance": { paragraph_id, document_id, page, source_uri, char spans, verbatim },
  "fidelity", "predicate", "arguments",
  "facets":     { role, proposition_type, relation, modality, attribution,
                  certainty, polarity, has_measurement },
  "cues", "confidence", "reason"
}

stage2 = {
  "stage": "stage_2_semantic_frame_extraction",
  "pipeline_version": "1.2",
  "model", "generated_at", "paragraph_id",
  "frames": [ { statement_id, source_text, stage1_type, stage1_facets,
                provenance, processing: {attempt, model}, stage2_frame } ],
  "stage1_relations", "statement_graph", "audit"
}
```

Treat `pipeline_version` + `model` as your **compatibility and provenance keys**:
pin behavior to a `pipeline_version`, and every node is auditable back to the
exact model that produced it. See DESIGN.md §4 for the full field-level schema.

### Machine-checkable validation

The IR is described by versioned JSON Schemas shipped in the wheel
(`graphrag_stage1/schemas/stage1-1.1.json`, `stage2-1.2.json`). Assert
conformance in your pipeline (needs the `[validation]` extra for `jsonschema`):

```python
from graphrag_stage1 import validate, SchemaValidationError

try:
    validate(out["stage1"])   # picks the schema from stage + pipeline_version
    validate(out["stage2"])
except SchemaValidationError as e:
    ...  # reject / quarantine before it reaches your graph store
```

The schemas lock the envelope, provenance anchors, and the facet enums (the hard
contract) while leaving LLM-decided nested content open for forward
compatibility. A test keeps the schema enums in sync with the code constants.

---

## 5. Operational notes

- **No hidden I/O.** Importing the package and running the stages performs no disk
  or network access except through the injected client. Durable queuing / JSONL
  checkpointing (`graphrag_stage1.production_support`, root `pipeline_runner.py`)
  is **opt-in** — nothing is written unless you construct those helpers.
- **Concurrency.** Stage 2 fans out statement extraction across threads
  (`STAGE2_CONCURRENCY`, default 2) while preserving input order. Your `complete`
  implementation must be thread-safe.
- **Determinism.** `OllamaClient` calls at temperature 0. If your client is
  non-deterministic, expect run-to-run variation in the LLM-decided facets.

---

## 6. Maturity caveat — read before trusting confidence scores

Confidence values in the output are **not yet calibrated**. The evaluation set is
currently tiny (see [evaluation_status.json](evaluation_status.json)) relative to
the release gate in [release_gates.json](release_gates.json) (500 paragraphs / 50
documents / 8 domains). Until that gate is met, treat every numeric confidence as
*uncalibrated* and do not gate downstream decisions on it. The deterministic
guarantees (numeric-token preservation, provenance completeness, schema validity)
are the parts you can rely on today.
