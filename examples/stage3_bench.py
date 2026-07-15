#!/usr/bin/env python
"""Re-run Stage 3 on a completed run's Stage 2 output and score the mapping.

    python examples/stage3_bench.py examples/run_v3_output/results.json

Stage 3 makes NO model calls, so this replays it against stored frames in seconds.
Use it as the measure-fix-measure loop while working on ontology mapping: change the
mapper, re-run this, watch the numbers move.

Scores what actually matters for a knowledge graph:
  * how many entities got a DOMAIN class vs a generic upper-ontology fallback
  * which mapping method won (registry short-circuit? domain lookup? fallback?)
  * how many relation edges the graph carries
  * how many frames are ready_for_reasoning
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag_stage1.ontology_manager import OntologyManager
from graphrag_stage1.stage3_ontology_mapper import stage3_pipeline

# Upper-ontology classes that mean "we gave up and typed it as a generic thing".
GENERIC = {
    "http://purl.obolibrary.org/obo/IAO_0000030",  # information content entity
    "http://purl.obolibrary.org/obo/BFO_0000015",  # process
    "http://purl.obolibrary.org/obo/BFO_0000019",  # quality
    "http://purl.obolibrary.org/obo/BFO_0000001",  # entity  (true of anything)
    "http://purl.obolibrary.org/obo/BFO_0000002",  # continuant
    "http://purl.obolibrary.org/obo/BFO_0000003",  # occurrent
    "http://purl.obolibrary.org/obo/BFO_0000004",  # independent continuant
    "http://purl.obolibrary.org/obo/BFO_0000031",  # generically dependent continuant
    "http://purl.obolibrary.org/obo/BFO_0000040",  # material entity
    "http://www.w3.org/2002/07/owl#Thing",         # could not classify
}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/run_v3_output/results.json"
    repo = Path(__file__).resolve().parent.parent
    results = json.loads(Path(path).read_text(encoding="utf-8"))

    # No domain_ontology argument -> load EVERY file in ontologies/Domain/. Naming one
    # file loads only that file, which silently ignores any other domain ontology you
    # have added (UBERON, ChEBI, ...) and makes it look like they changed nothing.
    ontology = OntologyManager(str(repo / "ontologies")).load_all()
    print(f"loaded {len(ontology.class_index):,} class labels "
          f"from {len(ontology.loaded_files)} files\n")

    classes, methods, types = Counter(), Counter(), Counter()
    edges = Counter()
    frames = ready = 0

    for para in results:
        if "stage2" not in para:
            continue
        out = stage3_pipeline(para["stage2"], ontology)
        for mapped in out["mapped_frames"]:
            frames += 1
            ready += bool(mapped.get("ready_for_reasoning"))
            for edge in mapped.get("object_property_assertions", []) or []:
                edges[str(edge.get("predicate", "")).rsplit("/", 1)[-1]] += 1
            for entity in mapped.get("ontology_individuals", []) or []:
                classes[entity.get("ontology_class") or ""] += 1
                methods[entity.get("mapping_method", "?")] += 1
                types[entity.get("semantic_type", "?")] += 1

    total = sum(classes.values())
    generic = sum(n for iri, n in classes.items() if iri in GENERIC)
    domain = total - generic

    print(f"frames mapped        {frames}")
    print(f"ready_for_reasoning  {ready}  ({ready / max(frames,1) * 100:.0f}%)")
    print()
    print(f"entities typed       {total}")
    print(f"  DOMAIN class       {domain:>5}  ({domain / max(total,1) * 100:.0f}%)   <- the number to raise")
    print(f"  generic fallback   {generic:>5}  ({generic / max(total,1) * 100:.0f}%)   <- the number to lower")
    print()
    print("how the class was chosen:")
    for method, n in methods.most_common():
        print(f"  {method:<32}{n:>5}")
    print()
    print("what SemanticTyper called these entities:")
    for stype, n in types.most_common(6):
        print(f"  {stype:<32}{n:>5}")
    print()
    # Structural edges are pipeline scaffolding (which frame mentions which entity).
    # Domain edges are the actual science (X causes/reduces/increases Y).
    structural = {"hasSemanticParticipant", "providesSourceContextFor",
                  "providesBackgroundFor", "providesEvidenceFor", "elaborates", "supports"}
    domain_edges = {k: v for k, v in edges.items() if k not in structural}
    print(f"relation edges       {sum(edges.values())} total")
    print(f"  domain (the science) {sum(domain_edges.values()):>4}   <- the number to raise")
    for pred, n in sorted(domain_edges.items(), key=lambda x: -x[1])[:6]:
        print(f"      {pred:<28}{n:>5}")
    print(f"  structural (scaffold) {sum(v for k, v in edges.items() if k in structural):>3}")
    print()
    print("top classes:")
    for iri, n in classes.most_common(8):
        tag = "  <- GENERIC" if iri in GENERIC else ""
        print(f"  {iri.rsplit('/', 1)[-1]:<32}{n:>5}{tag}")


if __name__ == "__main__":
    main()
