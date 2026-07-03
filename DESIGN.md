# Stage 1 — Knowledge Artifact Extraction & Validation Layer

> A first-principles redesign. Stage 1 is **not** a classifier and **not** a prompt.
> It is the **compiler front-end** for an ontology-driven knowledge graph: it lowers
> unstructured document bytes into a typed, validated, provenance-anchored
> **Knowledge Artifact Graph (KAG)** — the intermediate representation (IR) that
> Stage 2 compiles into ontology triples.

---

## 0. Reframing the problem

The current design treats Stage 1 as "decompose + label." That conflates several
distinct computations and gives no verifiable guarantee that knowledge was fully and
faithfully recovered. A 10/10 system borrows the discipline of a **multi-pass
compiler**: every pass has a single job, a typed input/output contract, and an
*invariant that can be checked*. Failures are caught between passes, not discovered
downstream in the graph.

| Compiler stage        | Stage-1 layer                              | Invariant it guarantees                         |
|-----------------------|--------------------------------------------|-------------------------------------------------|
| Lexing                | L0 Ingestion & provenance anchoring        | Every character has a stable, addressable span  |
| Parsing               | L1 Structural segmentation & gating        | Every block is typed; content is separated      |
| Syntactic analysis    | L2 Deterministic syntactic pre-analysis    | A **recall floor** on proposition count exists  |
| Semantic analysis     | L3 Decomposition + L4 Coreference          | Every clause maps to ≥1 standalone proposition  |
| Type checking         | L5 Faceted artifact typing                 | Facets obey logical consistency rules           |
| Linking               | L6 Discourse / rhetorical structure        | Every connective yields an edge or proposition  |
| Verification/opt      | L7 Validation, audit, recovery             | Faithfulness + coverage + non-contradiction     |
| Codegen (IR emit)     | L8 Ontology-ready emission                  | Output is reified, normalized, ontology-agnostic|

**Core principle:** neuro-symbolic. Deterministic tools (parser, regex cue
extractors, schema/logic validators) provide *guarantees and recall floors*; LLMs
provide *semantic judgment*; symbolic validators are *gates*. Never let an LLM be
the only thing standing between the document and the graph.

---

## 1. Architecture

```
                          ┌─────────────────────────────────────────────┐
  PDF / DOCX / LaTeX /    │  L0  INGESTION & PROVENANCE ANCHORING        │
  XML / SysML / HTML  ───▶│  layout parse → typed block DOM             │
                          │  content-addressed span IDs (page,bbox,off) │
                          └───────────────────────┬─────────────────────┘
                                                  ▼
                          ┌─────────────────────────────────────────────┐
                          │  L1  STRUCTURAL SEGMENTATION & GATING        │
                          │  block typing (prose/caption/ref/eq/table)  │
                          │  NOISE/METADATA/REFERENCE removed cheaply    │
                          │  science-aware sentence segmentation        │
                          └───────────────────────┬─────────────────────┘
                                                  ▼
                          ┌─────────────────────────────────────────────┐
                          │  L2  SYNTACTIC PRE-ANALYSIS (deterministic) │
                          │  dependency parse · clause inventory        │
                          │  negation/hedge/modal/connective cues       │
                          │  ⇒ RECALL FLOOR (expected #propositions)    │
                          └───────────────────────┬─────────────────────┘
                                                  ▼
            ┌──────────────────────── L3  DECOMPOSITION (LLM) ──────────────────────┐
            │  sentence + clause hints → atomic, minimal, decontextualized props    │
            │  contract: every independent finite clause → ≥1 prop OR "absorbed"    │
            └───────────────────────────────────┬───────────────────────────────────┘
                                                ▼
                          ┌─────────────────────────────────────────────┐
                          │  L4  COREFERENCE & ENTITY ANCHORING          │
                          │  doc-level mention graph → canonical IDs     │
                          └───────────────────────┬─────────────────────┘
                                                  ▼
         ┌───────────────────────────────┐   ┌───────────────────────────────┐
         │ L5 FACETED ARTIFACT TYPING    │   │ L6 DISCOURSE / RHETORICAL      │
         │ orthogonal facets + per-facet │   │ STRUCTURE                      │
         │ confidence + evidence span    │   │ typed directed attributed edges│
         └───────────────┬───────────────┘   └───────────────┬───────────────┘
                         └─────────────────┬──────────────────┘
                                           ▼
                          ┌─────────────────────────────────────────────┐
                          │  L7  VALIDATION · AUDIT · RECOVERY (ensemble)│
                          │  coverage · faithfulness/NLI · consistency  │
                          │  contradiction · calibration → score        │
                          │  closed-loop recovery / human-in-the-loop   │
                          └───────────────────────┬─────────────────────┘
                                                  ▼
                          ┌─────────────────────────────────────────────┐
                          │  L8  ONTOLOGY-READY EMISSION                 │
                          │  reified KA records · pred-arg skeletons    │
                          │  normalized units · ontology hints          │
                          └─────────────────────────────────────────────┘
                                                  ▼
                                       Stage 2: Triple Extraction
```

The output is a **multi-layer graph**, not a list:

- **Proposition layer** — atomic knowledge artifacts (nodes).
- **Discourse layer** — rhetorical/causal edges *between* propositions.
- **Mention layer** — coreference clusters anchoring arguments to entities.
- **Provenance layer** — every node/edge anchored to verbatim source spans.

---

## 2. The 13 questions, answered

**Q1 — Complete architecture?** The 9-layer neuro-symbolic pipeline above. Each
layer is independently testable with a contract. The product is a typed IR (KAG),
not labels.

**Q2 — Separate decomposition and classification?** **Yes — they are different
computations with opposite failure modes.** Decomposition is *recall-bound and
structural* (did we get every proposition?); classification is *precision-bound and
semantic* (is the typing right?). Fusing them couples their errors and makes neither
auditable. Separation lets each pass own a dedicated auditor. (In practice this is
≥3 passes, not 2: decompose → decontextualize → type.)

**Q3 — Syntactic parsing before the LLM?** **Yes — but as a guardrail and feature
source, never as the decomposer.** Rule-based decomposition of scientific prose is
brittle (coordination, nested clauses, math). The parser instead (a) routes trivial
sentences past the LLM, (b) supplies the **recall floor** = independent-finite-clause
count, (c) extracts deterministic cues (negation, hedging, modality, connectives)
that *directly populate facets* without an LLM guess, and (d) audits coverage. This
is what converts "we hope nothing was missed" into "we can prove the floor was met."

**Q4 — Discourse analysis?** **Yes — discourse structure *is* the scientific
reasoning.** "A, **because** B, **therefore** C, which is **consistent with** D" is an
argument. Chunk-embed RAG discards the connectives and keeps only the spans; we model
them as first-class edges. Without this layer you cannot do reasoning-aware retrieval.

**Q5 — RST?** **Use RST's relation *inventory* as inspiration; do not build full RST
trees.** Whole-document RST tree parsing is fragile and rarely pays off. Adopt instead
a **PDTB-style shallow discourse graph**: high-precision intra-sentence relations from
clause structure + adjacent-sentence relations from connectives, expressed as a
*graph, not a spanning tree*, using RST/PDTB relation labels (Cause, Evidence,
Elaboration, Concession, Condition, Background, Contrast).

**Q6 — Represent support / explanation / evidence / contradiction / causality?** As
**typed, directed, attributed discourse edges** between propositions, each carrying:
relation type, lexical trigger, attribution (who asserts the relation), polarity
(affirms vs. contradicts), provenance span, and confidence. Crucially, distinguish
**content causality** (a CAUSATION/MECHANISM *relation_kind* between entities, inside a
proposition) from **discourse causality** (an EVIDENCE/EXPLANATION edge *between*
propositions). They are different layers; conflating them is a common, costly mistake.
Contradiction is produced by a dedicated NLI auditor as `CONTRADICTION` edges over the
proposition set (intra- and cross-document).

**Q7 — Detect a missed statement?** Three independent recall nets:
1. **Recall floor:** produced propositions ≥ parser's independent-clause count, else flag.
2. **Bidirectional entailment:** source ⊨ ⋀propositions (nothing dropped) **and**
   ⋀propositions ⊨ source (nothing invented).
3. **Connective accounting:** every causal/contrastive/conditional connective must
   surface as a proposition or a discourse edge, else flag (this is exactly what
   catches mechanisms buried in `because`-clauses).

**Q8 — Self-audit?** An **ensemble of independent auditors**, each a pass/fail
contract, *independent of the generator* (different prompt / model / sampling, to
avoid correlated blind spots): schema validator (deterministic), coverage auditor,
faithfulness/entailment auditor, decontextualization auditor (no dangling anaphora),
**facet-consistency auditor** (logic rules, e.g. `relation_kind=NONE ⇒ no second
argument`; `MECHANISM ⇒ process phrase present`; `MEASUREMENT ⇒ has_quantity`),
contradiction auditor, and calibration auditor. Think *property-based testing for
extraction*. Failures route to recovery; unresolved → human review.

**Q9 — Confidence?** **Never trust the LLM's self-reported number.** Compute it:
- **Self-consistency:** sample the typing N× at temperature > 0; per-facet agreement
  fraction = empirical confidence.
- **Cue agreement:** deterministic cues agreeing with the LLM facet raises confidence.
- **Auditor agreement:** passing entailment/consistency raises it.
- **Calibrate** against the gold set (reliability curve), report the calibrated value.
- Keep **two distinct quantities**: *extraction confidence* (did we get it right) vs.
  *epistemic certainty* (how sure the **author** is, from hedging). Conflating them is
  a classic bug — a confidently-extracted hypothesis is still a hypothesis.

**Q10 — Provenance?** Three levels: **(a) span anchoring** — every artifact/edge
references content-addressed source spans (block id + char offsets + page + bbox), and
decontextualized text keeps a back-link to its verbatim span(s); **(b) document
lineage** — source hash, version; **(c) processing lineage** — model id, prompt
version, parser version, config hash, timestamp. `hash(source) + hash(pipeline)` =
reproducible artifact version. This makes every node in the graph auditable back to ink
on the page and to the exact code that produced it.

**Q11 — Structure outputs for ontology mapping?** Emit **predicate–argument
skeletons** (neo-Davidsonian: predicate + typed argument roles), each argument carrying
its **mention-cluster ID**; **normalize** quantities/units (SI/QUDT), comparators, and
negation; map facets to ontology constructs (`relation_kind`→object-property type,
`proposition_type`→statement class, `modality`/`attribution`/`confidence`→reification
attributes, `units`→datatype properties). Provide **candidate ontology hints** but do
**not** bind to a specific ontology — keep the IR ontology-agnostic and reusable. Emit
each relational artifact in **reifiable** form (RDF-star ready) so modality, attribution,
provenance, and confidence attach to the statement itself.

**Q12 — What most GraphRAG throws away (and we keep):**
- Discourse / rhetorical structure (the reasoning) ← biggest loss.
- **Negation & polarity** (flips meaning entirely).
- **Epistemic modality & hedging** (fact vs. hypothesis vs. prediction).
- **Attribution** (present work vs. prior work vs. third party).
- **Coreference clusters** (entity identity across the document).
- **Quantities with units and uncertainty** (`2.224 ± 0.001 MeV`).
- **Scope / conditions** ("up to iron", "at high energy", "under vacuum").
- **Contradictions / disagreements** between sources.
- **Equations, tables, figures** as structured knowledge, not flattened noise.
- **Exact provenance** to the source span.
- **Order** (temporal/causal sequence of events and steps).
Most chunk-embed RAG keeps raw text + a vector. We keep the full typed reasoning IR.

**Q13 — What a 10/10 looks like:** compiler-style multi-pass IR with verifiable
contracts; neuro-symbolic (parser recall floor + LLM judgment + symbolic gates);
multi-layer KAG (proposition + discourse + mention + provenance); ensemble self-audit
with a numeric completeness/calibration score and closed-loop recovery; human-in-the-
loop only for flagged items; fully versioned, reproducible, content-addressed,
idempotent; schema-first (JSON-Schema/pydantic, structured outputs everywhere);
model-agnostic and ontology-agnostic; and a CI-gated evaluation harness measuring
per-facet accuracy, calibration, coverage, and faithfulness against a gold set.

---

## 3. The Information Artifact Model (formal, faceted)

Replace the flat 15-label enum (which secretly mixes three independent dimensions and
forces false either/or choices) with **orthogonal facets**. A single artifact is a point
in this product space:

| Facet | Domain | Notes |
|-------|--------|-------|
| `content_role` | CONTENT · NOISE · METADATA · REFERENCE · FIGURE · TABLE · EQUATION · CAPTION | gate; set mostly in L1 |
| `proposition_type` | DEFINITION · CLASSIFICATION · PROPERTY · MEASUREMENT · RELATION · PROCESS · METHOD · EVENT | what *kind* of assertion |
| `relation_kind` | NONE · CORRELATION · CAUSATION · MECHANISM · COMPARISON · PART_OF · IS_A · DEPENDENCY · TEMPORAL_ORDER | MECHANISM = CAUSATION + described pathway |
| `epistemic_modality` | ASSERTED · OBSERVED · MEASURED · PREDICTED · HYPOTHESIZED · ASSUMED · DEFINED · REQUIRED | REQUIRED = MBSE "shall/must" |
| `attribution` | PRESENT_WORK · PRIOR_WORK · MODEL_THEORY · DATA · THIRD_PARTY · UNKNOWN | who asserts it |
| `certainty` | CERTAIN · LIKELY · POSSIBLE · UNCERTAIN | author hedging (≠ extraction confidence) |
| `polarity` | AFFIRMED · NEGATED | from negation cues + LLM |
| `quantitation` | {has_quantity, value, unit, uncertainty, comparator} | normalized to SI/QUDT |
| `temporality` | optional time index / tense | for EVENT/METHOD ordering |

Why this fixes "inconsistent classification": `PREDICTED` (modality) and `CAUSATION`
(relation) are no longer rivals for one slot — a model predicting a cause is simply
`modality=PREDICTED, relation_kind=CAUSATION`. The `relation_kind` axis becomes a single
ordered decision tree (process? → comparative? → compositional? → directional? →
associative? → none) instead of a 15-way guess.

---

## 4. Data schemas

### 4.1 Knowledge Artifact (proposition node)

```json
{
  "id": "doc7b3:s12:p4:a3",
  "text": "Proton and neutron shells align to maximize overlap between their distributions.",
  "provenance": {
    "document_id": "doc7b3",
    "block_id": "doc7b3:b88",
    "char_start": 142, "char_end": 233,
    "page": 4, "bbox": [72, 410, 523, 451],
    "verbatim": "proton and neutron shells tend to align in a manner that maximizes overlap"
  },
  "derived_from_clause": "advcl@because",
  "predicate": {"lemma": "align", "surface": "tend to align"},
  "arguments": [
    {"role": "agent", "text": "proton and neutron shells", "mention_cluster": "m7"},
    {"role": "goal",  "text": "overlap between their distributions", "mention_cluster": "m9"}
  ],
  "facets": {
    "content_role": "CONTENT",
    "proposition_type": "PROCESS",
    "relation_kind": "MECHANISM",
    "epistemic_modality": "ASSERTED",
    "attribution": "PRESENT_WORK",
    "certainty": "LIKELY",
    "polarity": "AFFIRMED",
    "quantitation": {"has_quantity": false}
  },
  "cues": {"negation": false, "hedges": ["tend to"], "connectives": ["because"], "quantities": []},
  "confidence": {"overall": 0.84, "by_facet": {"relation_kind": 1.0, "modality": 0.67},
                 "method": "self_consistency@3", "calibrated": 0.81},
  "evidence_text": "...in a manner that maximizes overlap...",
  "ontology_hints": {"candidate_classes": ["NuclearShell"], "candidate_predicate": "maximizes"},
  "audit": {"coverage_checked": true, "entailment_ok": true, "consistency_flags": []}
}
```

### 4.2 Discourse edge

```json
{
  "id": "doc7b3:p4:e1",
  "from": "doc7b3:s12:p4:a3", "to": "doc7b3:s12:p4:a2",
  "type": "EXPLANATION",                         // CAUSE·EVIDENCE·EXPLANATION·SUPPORT·CONTRAST·CONCESSION·CONDITION·ELABORATION·BACKGROUND·CONTRADICTION·ATTRIBUTION
  "nucleus": "doc7b3:s12:p4:a2",                 // RST nucleus/satellite
  "trigger": "because",
  "attribution": "PRESENT_WORK",
  "polarity": "affirms",
  "provenance": {"block_id": "doc7b3:b88", "char_start": 120, "char_end": 233},
  "confidence": 0.9
}
```

### 4.3 Paragraph audit report

```json
{
  "paragraph_id": "doc7b3:p4",
  "recall_floor": 4, "produced": 4, "floor_ok": true,
  "clause_coverage": [{"clause": "advcl@because", "covered_by": "a3"}],
  "uncovered_connectives": [],
  "entailment": {"source_entails_props": true, "props_entail_source": true},
  "consistency_flags": [],
  "contradiction_pairs": [],
  "completeness_score": 0.97,
  "needs_human_review": false
}
```

---

## 5. Strategies (the parts you asked for explicitly)

### 5.1 Decomposition strategy
- One **finite assertion** per proposition; modifiers/appositives/restrictive relatives
  ride along (no over-splitting).
- **Promote** subordinate clauses that carry their own assertion — especially
  `because / since / due to / in order to / thereby / so that` — to standalone
  propositions, *and* record the discourse edge to the parent.
- **Decontextualize**: resolve every pronoun/ellipsis against a coreference window so each
  proposition stands alone. No proposition may begin with an unresolved anaphor.
- **Contract:** each independent finite clause from L2 maps to ≥1 proposition or is
  explicitly marked `absorbed`/`non-assertional`. This is the checkable coverage guarantee.

### 5.2 Classification strategy
- Faceted, multi-axis (Section 3). `relation_kind` decided by an ordered decision tree.
- Deterministic cues **pre-fill** facets where unambiguous (negation→polarity,
  quantity→`has_quantity`, "shall"→`REQUIRED`, hedge→`certainty`); the LLM resolves the rest.
- Borderline facets resolved by **self-consistency voting**; disagreement lowers confidence
  and may trigger review.

### 5.3 Discourse modeling strategy
- Shallow PDTB-style graph + RST relation inventory (Section 2, Q4/Q5).
- Intra-sentence edges from clause structure (high precision); inter-sentence from
  connectives + LLM. Each edge typed, directed, triggered, attributed, provenance-anchored.

### 5.4 Validation strategy (ensemble auditors)
| Auditor | Type | Contract |
|---------|------|----------|
| Schema | deterministic | output validates against JSON-Schema |
| Coverage | deterministic + LLM | produced ≥ recall floor; every clause covered |
| Faithfulness | NLI/LLM | bidirectional entailment source ↔ propositions |
| Decontextualization | regex + LLM | no dangling anaphora; standalone-resolvable |
| Facet consistency | deterministic rules | `NONE⇒no arg2`; `MECHANISM⇒process phrase`; `MEASUREMENT⇒has_quantity`; `negation cue⇒NEGATED` |
| Contradiction | NLI | no unexplained contradictory proposition pairs |
| Calibration | statistical | reported confidence matches empirical accuracy |

### 5.5 Recovery strategy
Closed loop: a failed auditor triggers a **targeted** re-ask (not a full re-run) — e.g.
coverage failure → "the clause `because …` produced no proposition; extract it";
contradiction → re-examine the pair; low confidence → self-consistency at higher N.
Recover up to K iterations, then flag `needs_human_review`. Every recovery is logged in
processing lineage.

### 5.6 Ontology-readiness strategy
Predicate–argument skeletons + mention clusters + normalized quantities + reifiable
relational frames + ontology hints, all ontology-agnostic (Section 2, Q11). Stage 2
receives structure, not prose, and never has to re-discover what Stage 1 already knows.

---

## 6. Implementation status in this repo

This is an evolving reference implementation, not the full enterprise platform. What the
code currently embodies vs. what is specified here as roadmap:

| Layer | Status in `stage1_classifier.py` |
|-------|----------------------------------|
| L0 Ingestion / provenance | **Partial** — paragraph-level IDs + best-effort char-offset span anchoring (no PDF/layout parser yet) |
| L1 Gating | **Partial** — `content_role` facet; no document DOM yet |
| L2 Syntactic pre-analysis | **Implemented (lightweight)** — regex cue layer + recall floor (no spaCy dependency-parse yet) |
| L3 Decomposition | **Implemented** — LLM, decontextualized, clause-aware |
| L4 Coreference | **Roadmap** — needs cross-sentence mention graph |
| L5 Faceted typing | **Implemented** — full facet set via structured outputs |
| L6 Discourse | **Implemented (basic)** — typed directed edges; RST/PDTB inventory partial |
| L7 Validation/audit | **Implemented** — recall floor, consistency rules, recall pass, completeness score |
| L8 Emission | **Implemented (basic)** — pred-arg + facets + provenance + ontology hints stub |

**Next milestones, in order:** (1) grow the gold set to ~50 paragraphs with per-facet
labels; (2) add self-consistency voting + calibration; (3) introduce spaCy as the real
L2 recall floor; (4) cross-sentence coreference (L4); (5) PDF/LaTeX ingestion (L0).

See `eval.py` for the measurable contracts (artifact recall, decontextualization, unit
count, and — new — facet-consistency and completeness).

### 6.1 Automated Stage 1 -> Stage 2 safety layer

The reference implementation now enforces these domain-agnostic invariants in Python:

- Numeric source tokens are immutable. Stage 1 repairs an unambiguous one-to-one
  mutation and records the repair; uncertain mismatches remain validation failures.
- Stage 2 candidate evidence is normalized to the exact atomic source statement.
- Required semantic-frame fields are checked by artifact type.
- Candidate entities are reconciled with semantic-frame and relation endpoints.
- Context-based reference resolutions must identify valid supporting statement IDs.
- Unresolved references, omitted measurements, incomplete relations, and unsupported
  context use cannot pass automatically to ontology mapping.
- Routing is recomputed deterministically after extraction and supports bounded
  automated contextual/stronger-model retries.
- Model, attempt, version, timestamp, and source provenance are retained.

`pipeline_runner.py` provides append-only JSONL checkpointing for resumable paragraph
batches. `test_pipeline.py` exercises deterministic contracts without requiring an
LLM; model-quality evaluation remains in `eval.py` and should continue expanding across
scientific domains.

### 6.2 Stage 2 semantic-fidelity architecture (v1.2)

Stage 2 now constructs a normalized statement graph from Stage 1 nodes, discourse
edges, provenance, and shared source spans. The graph is used only for reference
resolution, evidence lineage, and context authorization; it does not perform ontology
mapping or introduce new document knowledge.

Scientific discourse is represented explicitly with `SUPPORT`, `EVIDENCE`,
`CONSISTENCY`, and `VALIDATION` frames. Every semantic frame is enriched with Stage 1
modality, polarity, provenance, evidence statement IDs, semantic-role aliases, and an
exact measurement object. Reference resolutions record the original mention, resolved
text, supporting statement IDs, confidence, and resolution method.

Final extraction confidence is computed in Python from required-field completeness,
grounding, context resolution, measurement fidelity, semantic consistency, and the LLM
proposal confidence. Ontology readiness and routing are recomputed from that validated
state. The confidence is explicitly marked uncalibrated until a sufficiently large,
independently adjudicated corpus supports empirical calibration.

### 6.3 Stage 3 ontology mapping architecture (v1.0)

Stage 3 consumes validated Stage 2 frames without performing NLP or reinterpreting
their semantics. `OntologyManager` loads BFO, IAO, CCO, and one replaceable domain
ontology through RDFLib, indexes classes and properties from ontology IRIs and labels,
mints deterministic individuals, records assertions, and serializes RDF. Missing or
ambiguous ontology terms fail closed; there are no Python dictionaries of domain
classes or relation mappings.

The versioned `ontologies/Alignment/` module bridges the frozen Stage 2 frame
vocabulary to OWL without modifying BFO, IAO, CCO, RO, or the replaceable domain
ontology. Alignment changes require a new reviewed version. Lookup audits distinguish
missing terms from ambiguous terms and list every candidate IRI for review.

`Stage3OntologyMapper` creates frame and entity individuals, maps candidate relations
and statement-graph discourse edges to loaded object properties, preserves exact
measurements, evidence, reference resolutions, and source provenance, and emits
RDF-ready object/datatype assertion records. Frames not routed by Stage 2 to ontology
mapping are retained as skipped records. OWL reasoning and inferred-fact
materialization remain explicitly outside Stage 3 and belong to Stage 3.5.

#### Stage 3.1 decisions

Stage 3.1 emits a JSON-LD envelope containing `@context`, `@graph`, and a JSON-valued
mapping audit. Document-scoped individuals preserve canonical forms, original aliases,
and direct links to deterministic source-document and source-statement nodes. Reference
resolutions from Stage 2 control canonical identity while the original wording remains
available for citation.

Relations are represented both as direct triples for traversal and as qualified
assertion nodes carrying evidence and provenance. Measurements are first-class nodes.
Ontology candidate generation is hybrid by design; the implemented deterministic
lexical/synonym retrieval tier supplies auditable candidates while embedding and
constrained-provider tiers remain separately replaceable. Low-confidence mappings do
not create individuals. SHACL publication gating belongs at the Stage 3/3.5 boundary:
conforming assertions may publish while invalid assertions and their reports are
quarantined.

For Amazon Neptune bulk ingestion, Stage 3.1 validates the JSON-LD RDF projection
against `stage3-publication-shapes.ttl`, removes nonconforming focus nodes into a
quarantine graph, and serializes the conforming graph as UTF-8 N-Quads. Each export
uses a deterministic document-scoped named graph. Operational mapping audits remain
outside the publication graph.
