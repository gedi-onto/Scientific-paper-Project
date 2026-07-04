"""Minimal end-to-end example: analyze a paper with your own ontology.

Run:
    pip install "graphrag-stage1[ontology,validation,anthropic]"
    export ANTHROPIC_API_KEY=sk-...
    python examples/quickstart.py path/to/paper.txt path/to/my_domain.owl
"""

import sys

from graphrag_stage1 import AnthropicClient, analyze_paper, validate


def main(paper_path: str, domain_ontology: str) -> None:
    text = open(paper_path, encoding="utf-8").read()

    # One built-in client = no boilerplate. Swap for OpenAIClient() or your own.
    client = AnthropicClient(model="claude-haiku-4-5")

    # Split + run every paragraph in parallel; load core + your domain ontology.
    results = analyze_paper(
        text,
        client=client,
        max_concurrency=16,            # tune to your API rate limit
        domain_ontology=domain_ontology,
    )

    for i, r in enumerate(results):
        if "error" in r:
            print(f"[{i}] skipped: {r['error']}")
            continue
        validate(r["stage1"])          # assert the output matches the schema
        validate(r["stage2"])
        for fact in r["stage1"]["statements"]:
            print(f"[{i}] {fact['facets']['relation']:<12} {fact['text']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python examples/quickstart.py <paper.txt> <domain_ontology.owl>")
    main(sys.argv[1], sys.argv[2])
