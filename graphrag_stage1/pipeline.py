"""End-to-end orchestration convenience for host integrations.

``run_pipeline`` is the one call a RAG platform typically needs: give it a
paragraph and an :class:`~graphrag_stage1.llm.LLMClient`, get back the Stage 1
Knowledge Artifact Graph and the validated Stage 2 semantic frames. Stage 3
ontology mapping runs only if you pass an ``ontology`` manager, so ``rdflib``
stays unimported otherwise.
"""

from __future__ import annotations

from typing import Any

from .llm import LLMClient
from .stage1_classifier import process_paragraph
from .stage2_semantic_frames import stage2_pipeline


def run_pipeline(
    paragraph: str,
    *,
    client: LLMClient | None = None,
    paragraph_id: str = "p1",
    source_metadata: dict | None = None,
    ontology: Any | None = None,
) -> dict:
    """Run Stage 1 -> Stage 2 (-> Stage 3 if ``ontology`` given) on one paragraph.

    Args:
        paragraph: raw source text.
        client: LLM transport; defaults to a local Ollama client per stage.
        paragraph_id: stable id used in provenance / node ids.
        source_metadata: optional document_id / page / source_uri, carried into
            every statement's provenance.
        ontology: an ``OntologyManager`` (from the ``[ontology]`` extra). When
            provided, Stage 3 mapping runs and its output is included.

    Returns:
        ``{"stage1": ..., "stage2": ...}`` plus ``"stage3"`` when ``ontology``
        is supplied. Each sub-result carries its own ``pipeline_version`` and
        ``model`` provenance.
    """
    stage1 = process_paragraph(
        paragraph, paragraph_id=paragraph_id, source_metadata=source_metadata, client=client
    )
    stage2 = stage2_pipeline(stage1, client=client)
    result = {"stage1": stage1, "stage2": stage2}
    if ontology is not None:
        from .stage3_ontology_mapper import stage3_pipeline  # rdflib only if asked

        result["stage3"] = stage3_pipeline(stage2, ontology)
    return result
