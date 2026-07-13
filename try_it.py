#!/usr/bin/env python
"""Manual smoke test for graphrag_stage1 — run it directly, no install needed.

    python try_it.py                 # analyze the built-in sample paragraph
    python try_it.py my_paper.txt    # analyze your own plain-text file
    python try_it.py --check         # fast setup check (imports + Ollama reachable)

By default it uses your LOCAL Ollama model (qwen3:8b) — no API key required.
To use a hosted model instead, edit get_client() below.

To also build the ontology graph (Stage 3), set ONTOLOGY to a domain file path.
"""

import sys
import time
from pathlib import Path

# --- Make graphrag_stage1 importable WITHOUT installing it -------------------
# (adds this project folder to Python's search path)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphrag_stage1 import analyze_paper, OllamaClient
# Hosted alternatives (uncomment the import you want in get_client):
# from graphrag_stage1 import AnthropicClient, OpenAIClient


# ---------------------------------------------------------------------------
# Configuration — change these to taste
# ---------------------------------------------------------------------------
def get_client():
    """The model to use. Local Ollama by default (free, offline, no key)."""
    return OllamaClient(model="qwen3:8b")
    # return AnthropicClient()          # needs: pip install "graphrag-stage1[anthropic]" + ANTHROPIC_API_KEY
    # return OpenAIClient()             # needs: pip install "graphrag-stage1[openai]"    + OPENAI_API_KEY


ONTOLOGY = None   # e.g. "ontologies/Domain/ino_merged.owl" to also run Stage 3

SAMPLE = (
    "The fused sensor data reduced positioning error by 35 percent. Because the "
    "fusion provides a more complete state estimate, the guidance system produced "
    "more stable trajectories.\n\n"
    "The model predicts that future versions will achieve higher accuracy in "
    "urban environments."
)


# ---------------------------------------------------------------------------
def check():
    """Fast check: is everything importable and is the model reachable?"""
    import requests
    print("graphrag_stage1 imported OK.")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print("Ollama is reachable. Models available:", ", ".join(models) or "(none)")
        print("\nAll good. Now run:  python try_it.py")
    except Exception as exc:
        print("Ollama NOT reachable:", exc)
        print("Start Ollama and `ollama pull qwen3:8b`, or switch get_client() to a hosted model.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check()

    text = Path(sys.argv[1]).read_text(encoding="utf-8") if len(sys.argv) > 1 else SAMPLE
    client = get_client()
    n_para = text.count("\n\n") + 1

    print(f"Model : {client.model_id}")
    print(f"Input : {n_para} paragraph(s)")
    print("Working... (a local model can take a minute or two per paragraph)\n")

    t0 = time.time()
    results = analyze_paper(
        text, client=client, max_concurrency=4, stage2_concurrency=4,
        domain_ontology=ONTOLOGY,
    )
    elapsed = time.time() - t0

    total = 0
    for i, r in enumerate(results):
        if "error" in r:
            print(f"[paragraph {i}] ERROR: {r['error']}")
            continue
        statements = r["stage1"]["statements"]
        total += len(statements)
        print(f"[paragraph {i}] - {len(statements)} fact(s)")
        for s in statements:
            f = s["facets"]
            print(f"   - ({f['proposition_type']} / {f['relation']} / {f['modality']}) {s['text']}")
            p = s.get("provenance", {})
            print(f"       chars {p.get('char_start')}-{p.get('char_end')}  |  confidence {s['confidence']['overall']}")
        if ONTOLOGY and "stage3" in r:
            mapped = r["stage3"].get("mapped_frames", [])
            print(f"   -> Stage 3: {len(mapped)} frame(s) mapped to the ontology")
        print()

    print(f"Done: {total} fact(s) from {len(results)} paragraph(s) in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
