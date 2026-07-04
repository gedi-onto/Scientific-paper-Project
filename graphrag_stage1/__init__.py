"""graphrag_stage1 — an ontology-driven knowledge-extraction pipeline.

Stage 1 lowers unstructured paragraphs into a typed, provenance-anchored
Knowledge Artifact Graph (KAG); Stage 2 builds validated semantic frames;
Stage 3 maps those frames onto an OWL ontology. See DESIGN.md.

Public API (stable surface for integrators):

    from graphrag_stage1 import run_pipeline
    from graphrag_stage1.llm import LLMClient   # implement this over your model

    out = run_pipeline(paragraph, client=my_client)
    out["stage1"]   # KAG for the paragraph
    out["stage2"]   # validated semantic frames

Lower-level entry points (process_paragraph, stage2_pipeline) are also
exported. Stage 3 ontology mapping requires the [ontology] extra
(rdflib/pyshacl) and an OntologyManager; see INTEGRATION.md.

Re-exports are lazy (PEP 562): importing this package pulls in nothing heavy;
``rdflib`` is only imported when you first touch a Stage 3 symbol.
"""

__version__ = "0.1.0"

# name -> submodule providing it. Resolved lazily on first attribute access so
# that `import graphrag_stage1` never imports rdflib/pyshacl (Stage 3 only).
_EXPORTS = {
    "analyze_paper": "pipeline",
    "run_pipeline": "pipeline",
    "run_paper": "pipeline",
    "split_paragraphs": "pipeline",
    "AnthropicClient": "clients",
    "OpenAIClient": "clients",
    "process_paragraph": "stage1_classifier",
    "stage2_pipeline": "stage2_semantic_frames",
    "stage3_pipeline": "stage3_ontology_mapper",
    "PipelineConfig": "production_support",
    "OntologyManager": "ontology_manager",
    "LLMClient": "llm",
    "OllamaClient": "llm",
    "BoundedClient": "llm",
    "validate": "validation",
    "SchemaValidationError": "validation",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):  # PEP 562
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(__all__)
