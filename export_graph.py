#!/usr/bin/env python
"""Run the FULL pipeline (Stage 1 -> 2 -> 3) and print the Neptune N-Quads.

    python export_graph.py                 # built-in sample
    python export_graph.py my_paper.txt    # your own text file

It prints the N-Quads (the exact RDF that goes into Neptune) to the screen and
also writes them to  graph_output.nq  (upload that to S3 -> Neptune bulk loader).

Uses your local Ollama by default, and the repo's core ontologies (BFO/IAO/CCO)
plus the sample domain ontology. Swap DOMAIN_ONTOLOGY for your own domain file.
NOTE: this re-runs the language model, so on a local GPU it is slow (minutes).
"""

import sys
import time
from pathlib import Path

# Make graphrag_stage1 importable without installing it:
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphrag_stage1 import analyze_paper, OllamaClient
from graphrag_stage1.stage3_neptune_export import export_neptune_nquads


# --- configuration ---------------------------------------------------------
DOMAIN_ONTOLOGY = "ontologies/Domain/ino_merged.owl"   # <- replace with your domain ontology
OUT_FILE = "graph_output.nq"                           # the combined file to upload
SAMPLE = (
    "The fused sensor data reduced positioning error by 35 percent. Because the "
    "fusion provides a more complete state estimate, the guidance system produced "
    "more stable trajectories."
)


def main():
    text = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else SAMPLE
    client = OllamaClient(model="qwen3:8b")

    print(f"Model: {client.model_id}")
    print("Running Stage 1 -> 2 -> 3 (re-runs the model; local GPU = slow)...\n")
    t0 = time.time()
    results = analyze_paper(
        text, client=client, max_concurrency=4, stage2_concurrency=4,
        domain_ontology=DOMAIN_ONTOLOGY,
    )
    print(f"pipeline finished in {time.time() - t0:.0f}s\n")

    all_nquads = []
    for i, r in enumerate(results):
        if "error" in r:
            print(f"[paragraph {i}] ERROR: {r['error']}")
            continue
        if "stage3" not in r:
            print(f"[paragraph {i}] no Stage 3 output (is the ontology loaded?)")
            continue
        dest = f"paragraph{i}.nq"
        meta = export_neptune_nquads(r["stage3"], destination=dest)
        all_nquads.append(Path(dest).read_text(encoding="utf-8"))
        print(f"[paragraph {i}] {meta['validated_triple_count']} triples validated, "
              f"{meta['quarantined_triple_count']} quarantined  ->  {dest}")

    combined = "".join(all_nquads)
    Path(OUT_FILE).write_text(combined, encoding="utf-8")
    line_count = combined.count("\n")

    print("\n" + "=" * 74)
    print(f"  N-QUADS FOR NEPTUNE  ({line_count} triples)  -  also saved to {OUT_FILE}")
    print("  Upload this file to S3, then run the Neptune bulk loader.")
    print("=" * 74 + "\n")
    print(combined)


if __name__ == "__main__":
    main()
