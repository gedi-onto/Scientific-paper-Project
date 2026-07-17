"""End-to-end orchestration convenience for host integrations.

``run_pipeline`` is the one call a RAG platform typically needs: give it a
paragraph and an :class:`~graphrag_stage1.llm.LLMClient`, get back the Stage 1
Knowledge Artifact Graph and the validated Stage 2 semantic frames. Stage 3
ontology mapping runs only if you pass an ``ontology`` manager, so ``rdflib``
stays unimported otherwise.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

from .llm import BoundedClient, LLMClient
from .stage1_classifier import process_paragraph
from .stage2_semantic_frames import stage2_pipeline


def split_paragraphs(text: str) -> list[dict]:
    """Split a document into paragraph records on blank lines.

    A convenience for plain text / already-extracted content. Real PDF or LaTeX
    layout parsing is a separate upstream step; feed this pre-extracted text.
    """
    chunks = re.split(r"\n\s*\n", text)
    return [{"text": chunk.strip()} for chunk in chunks if chunk.strip()]


# L1 gating (DESIGN.md): drop publisher boilerplate before it costs a model call.
#
# Deliberately narrow. These patterns match text that carries no extractable science
# in any paper -- journal front/back matter, not content. Everything else is left to
# the LLM's `role` facet, because the cheap signals that *look* like noise are not:
# on a real paper the shortest paragraphs include figure captions ("A Venn diagram
# illustrating the overlap between...") and methods lines ("GAPDH was used as an
# internal control"), both of which are content. A length or punctuation heuristic
# would silently delete them, and a dropped paragraph is unrecoverable downstream --
# so this gate only fires on phrasing that is unambiguously front/back matter.
BOILERPLATE_PATTERNS = (
    r"^\s*keywords\s*:",
    r"springer nature remains neutral",
    r"^\s*the online version contains supplementary material",
    r"^\s*(competing|conflict of) interests?\b.{0,80}$",
    r"^\s*(open access|this article is licensed under)",
    r"publisher'?s note",
    r"^\s*©\s*\d{4}",  # copyright line
)
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.I | re.S)


def is_boilerplate(text: str) -> bool:
    """True for journal front/back matter that no model call can turn into knowledge.

    Conservative by construction: a false positive silently deletes real science, a
    false negative merely costs one paragraph's worth of model calls. When unsure,
    return False and let Stage 1's `role` facet make the call.
    """
    return bool(_BOILERPLATE_RE.search(str(text or "")))


def run_pipeline(
    paragraph: str,
    *,
    client: LLMClient | None = None,
    stage2_client: LLMClient | None = None,
    paragraph_id: str = "p1",
    source_metadata: dict | None = None,
    ontology: Any | None = None,
    grounder: Any | None = None,
    stage2_concurrency: int | None = None,
    stage2_batch_size: int | None = None,
) -> dict:
    """Run Stage 1 -> Stage 2 (-> Stage 3 if ``ontology`` given) on one paragraph.

    Args:
        paragraph: raw source text.
        client: LLM transport; defaults to a local Ollama client per stage.
        stage2_client: optional separate transport for Stage 2 frame extraction.
            Stage 2 is schema-constrained form-filling, so a smaller / faster
            model than Stage 1's often suffices — the main latency lever for
            local setups. Defaults to ``client``.
        paragraph_id: stable id used in provenance / node ids.
        source_metadata: optional document_id / page / source_uri, carried into
            every statement's provenance.
        ontology: an ``OntologyManager`` (from the ``[ontology]`` extra). When
            provided, Stage 3 mapping runs and its output is included.
        grounder: optional entity grounder (e.g. ``grounding.OakGrounder``) passed
            through to Stage 3. Without it, an entity that is not a class in the
            loaded ontology can only be generalised to a true ancestor -- in
            practice ``BFO:entity`` -- so domain typing collapses to the ontology
            root. Supplying one resolves mentions to real term IDs lexically and
            roughly doubles the share of entities that receive a domain type.

    Returns:
        ``{"stage1": ..., "stage2": ...}`` plus ``"stage3"`` when ``ontology``
        is supplied. Each sub-result carries its own ``pipeline_version`` and
        ``model`` provenance.
    """
    stage1 = process_paragraph(
        paragraph, paragraph_id=paragraph_id, source_metadata=source_metadata, client=client
    )
    stage2 = stage2_pipeline(
        stage1,
        client=stage2_client or client,
        concurrency=stage2_concurrency,
        batch_size=stage2_batch_size,
    )
    result = {"stage1": stage1, "stage2": stage2}
    if ontology is not None:
        from .stage3_ontology_mapper import stage3_pipeline  # rdflib only if asked

        result["stage3"] = stage3_pipeline(stage2, ontology, grounder=grounder)
    return result


def run_paper(
    paragraphs: Iterable[dict],
    *,
    client: LLMClient,
    stage2_client: LLMClient | None = None,
    max_concurrency: int = 8,
    stage2_concurrency: int = 4,
    stage2_batch_size: int | None = None,
    ontology: Any | None = None,
    grounder: Any | None = None,
    gate_noise: bool = True,
    on_result: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Process a whole document's paragraphs in parallel, order-preserving.

    Paragraphs are independent, so they run concurrently. Total in-flight model
    calls are capped by wrapping ``client`` in a
    :class:`~graphrag_stage1.llm.BoundedClient` sized to ``max_concurrency`` --
    set that to your provider's safe concurrent-request budget and it holds no
    matter how wide the fan-out. This is the main lever for hitting a per-paper
    latency target on a scalable (hosted / multi-replica) endpoint.

    Args:
        paragraphs: iterable of dicts, each ``{"text": str, "paragraph_id"?: str,
            "source_metadata"?: dict}``. Order is preserved in the result.
        client: your LLMClient. Wrapped in a BoundedClient unless it already is.
        stage2_client: optional separate (typically smaller / faster) client for
            Stage 2 frame extraction. Shares ``client``'s global concurrency
            ceiling, so total in-flight calls never exceed ``max_concurrency``.
        max_concurrency: global ceiling on concurrent model calls (rate-limit knob).
        stage2_concurrency: per-paragraph statement fan-out (bounded by the same
            global ceiling).
        ontology: optional OntologyManager to also run Stage 3 per paragraph.
        on_result: optional callback ``(index, result)`` invoked as each paragraph
            finishes (e.g. to stream / checkpoint); called from worker threads.

    Returns:
        A list aligned to input order. Each item is the ``run_pipeline`` result,
        or ``{"paragraph_id", "error", "error_type"}`` if that paragraph failed
        (one failure never aborts the paper).
    """
    items = list(paragraphs)
    bounded = client if isinstance(client, BoundedClient) else BoundedClient(client, max_concurrency)
    bounded_stage2: BoundedClient | None = None
    if stage2_client is not None:
        bounded_stage2 = (
            stage2_client
            if isinstance(stage2_client, BoundedClient)
            else BoundedClient(stage2_client, max_concurrency, share_limit_with=bounded)
        )
    results: list[dict] = [None] * len(items)  # type: ignore[list-item]

    def process(index: int, record: dict) -> None:
        pid = record.get("paragraph_id") or f"p{index + 1}"
        if gate_noise and is_boilerplate(record["text"]):
            # L1: never spend a model call on front/back matter.
            results[index] = {
                "paragraph_id": pid,
                "gated": True,
                "gate_reason": "boilerplate",
                "stage1": {"statements": [], "audit": {"consistency_flags": []}},
                "stage2": {"frames": []},
            }
            if on_result is not None:
                on_result(index, results[index])
            return
        try:
            results[index] = run_pipeline(
                record["text"],
                client=bounded,
                stage2_client=bounded_stage2,
                paragraph_id=pid,
                source_metadata=record.get("source_metadata"),
                ontology=ontology,
                grounder=grounder,
                stage2_concurrency=stage2_concurrency,
                stage2_batch_size=stage2_batch_size,
            )
        except Exception as exc:  # one bad paragraph must not sink the paper
            results[index] = {
                "paragraph_id": pid,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        if on_result is not None:
            on_result(index, results[index])

    if items:
        with ThreadPoolExecutor(
            max_workers=min(max_concurrency, len(items)), thread_name_prefix="paper"
        ) as executor:
            for index, record in enumerate(items):
                executor.submit(process, index, record)
    return results


def analyze_paper(
    text: str,
    *,
    client: LLMClient,
    stage2_client: LLMClient | None = None,
    max_concurrency: int = 8,
    stage2_concurrency: int = 4,
    stage2_batch_size: int | None = None,
    ontology: Any | None = None,
    domain_ontology: str | None = None,
    ontology_root: str | None = None,
    grounder: Any | None = None,
    gate_noise: bool = True,
    on_result: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """One call: raw paper text in, structured per-paragraph results out.

    Splits ``text`` into paragraphs and runs them in parallel via
    :func:`run_paper`. For Stage 3 ontology mapping, pass either a prebuilt
    ``ontology`` (an ``OntologyManager``) or a ``domain_ontology`` path -- in the
    latter case the foundational ontologies (IAO, CCO, RO, alignment) are loaded
    from **inside the installed package**, so you only supply your domain file::

        analyze_paper(text, client=c, domain_ontology="my_domain.owl")

    ``ontology_root`` overrides where the foundational ontologies are read from;
    leave it ``None`` to use the packaged copy. The domain ontology is never
    bundled -- it is the module that changes per scientific domain, it is your
    choice, and it can be far larger than the core (UBERON alone is 48 MB).

    ``stage2_client`` lets Stage 2 (schema-constrained frame extraction) run on
    a smaller, faster model than Stage 1 -- the main per-paper latency lever on
    a local single-GPU setup::

        analyze_paper(text,
                      client=OllamaClient(model="qwen3:8b"),
                      stage2_client=OllamaClient(model="qwen3:4b"))
    """
    if ontology is None and domain_ontology is not None:
        from .ontology_manager import OntologyManager  # rdflib only if asked

        ontology = OntologyManager(ontology_root).load_all(domain_ontology=domain_ontology)
    return run_paper(
        split_paragraphs(text),
        client=client,
        stage2_client=stage2_client,
        max_concurrency=max_concurrency,
        stage2_concurrency=stage2_concurrency,
        stage2_batch_size=stage2_batch_size,
        ontology=ontology,
        grounder=grounder,
        gate_noise=gate_noise,
        on_result=on_result,
    )
