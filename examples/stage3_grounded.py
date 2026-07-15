#!/usr/bin/env python
"""Re-run Stage 3 on a completed run WITH OntoGPT/OAK grounding, and show the lift.

    python examples/stage3_grounded.py examples/dendrobine_output/results.json

Grounding resolves entity mentions the loaded ontologies could not type (proteins,
drugs, diseases) to real OBO term IDs via oaklib's OLS adapter -- no gigabyte
downloads. The first pass hits the network once per distinct mention and caches to
examples/<run>_grounding_cache.json; later passes are instant.

Reports domain-typed entities and ready_for_reasoning, with grounding off vs on, so the
lift is measured on identical Stage 2 output with no model calls.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag_stage1.grounding import OakGrounder
from graphrag_stage1.ontology_manager import OntologyManager
from graphrag_stage1.stage3_ontology_mapper import stage3_pipeline

GENERIC = {
    "http://purl.obolibrary.org/obo/IAO_0000030",
    "http://purl.obolibrary.org/obo/BFO_0000001",
    "http://purl.obolibrary.org/obo/BFO_0000015",
    "http://purl.obolibrary.org/obo/BFO_0000040",
    "http://www.w3.org/2002/07/owl#Thing",
}


def score(results, ontology, grounder):
    classes, methods, ontos = Counter(), Counter(), Counter()
    frames = ready = 0
    for para in results:
        if "stage2" not in para:
            continue
        for m in stage3_pipeline(para["stage2"], ontology, grounder=grounder)["mapped_frames"]:
            frames += 1
            ready += bool(m.get("ready_for_reasoning"))
            for e in m.get("ontology_individuals", []) or []:
                iri = e.get("ontology_class") or ""
                classes[iri] += 1
                methods[e.get("mapping_method", "?")] += 1
                if e.get("mapping_method") == "oak_grounded":
                    curie = e.get("grounded_curie") or ""
                    ontos[curie.split(":", 1)[0]] += 1
    total = sum(classes.values())
    domain = total - sum(n for iri, n in classes.items() if iri in GENERIC)
    return frames, ready, total, domain, methods, ontos


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/dendrobine_output/results.json"
    repo = Path(__file__).resolve().parent.parent
    results = json.loads(Path(path).read_text(encoding="utf-8"))

    ontology = OntologyManager(str(repo / "ontologies")).load_all()

    print("=== grounding OFF (loaded ontologies only) ===")
    f, ready, total, domain, _, _ = score(results, ontology, None)
    print(f"  domain-typed entities  {domain}/{total} ({domain/max(total,1)*100:.0f}%)")
    print(f"  ready_for_reasoning    {ready}/{f} ({ready/max(f,1)*100:.0f}%)\n")

    cache = Path(path).parent / "grounding_cache.json"
    grounder = OakGrounder(cache_path=cache)
    print(f"=== grounding ON (OAK/OLS, cache: {cache.name}) ===")
    print("  first pass queries the network per new mention; please wait...")
    # Fresh manager so grounding-off stubs do not leak into the on run.
    ontology2 = OntologyManager(str(repo / "ontologies")).load_all()
    f, ready, total, domain, methods, ontos = score(results, ontology2, grounder)
    grounder.save_cache()
    print(f"  domain-typed entities  {domain}/{total} ({domain/max(total,1)*100:.0f}%)")
    print(f"  ready_for_reasoning    {ready}/{f} ({ready/max(f,1)*100:.0f}%)")
    print(f"  grounded by OAK        {methods.get('oak_grounded', 0)}")
    if ontos:
        print("  grounded to:")
        for prefix, n in ontos.most_common():
            print(f"      {prefix:<12}{n}")


if __name__ == "__main__":
    main()
