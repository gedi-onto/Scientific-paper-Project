"""Cross-domain Stage 1/2 release evaluation.

This seed corpus is deliberately small and does not satisfy the production gate in
release_gates.json. It is a regression suite and a schema for growing adjudicated data.
"""

import json

from stage1_classifier import process_paragraph
from stage2_semantic_frames import stage2_pipeline


def facet_matches(statements: list, expected: dict) -> bool:
    for statement in statements:
        facets = statement.get("facets", {})
        if all(facets.get(key) == value for key, value in expected.items()):
            return True
    return False


def evaluate(path: str = "cross_domain_gold.json") -> dict:
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)

    results = []
    for case in cases:
        stage1 = process_paragraph(case["text"], paragraph_id=case["id"])
        stage2 = stage2_pipeline(stage1)
        output_text = " ".join(item["text"] for item in stage1["statements"])
        protected_ok = all(token in output_text for token in case.get("protected", []))
        facets_ok = facet_matches(stage1["statements"], case.get("must_contain", {}))
        discourse_type = case.get("must_discourse")
        discourse_ok = not discourse_type or any(
            edge.get("type") == discourse_type for edge in stage1.get("relations", [])
        )
        resolution_term = case.get("must_resolve")
        resolution_ok = True
        if resolution_term:
            resolution_ok = any(
                any(
                    item.get("original", "").casefold() == resolution_term.casefold()
                    for item in frame["stage2_frame"].get("reference_resolutions", [])
                )
                for frame in stage2["frames"]
            )
        results.append({
            "id": case["id"],
            "domain": case["domain"],
            "protected_ok": protected_ok,
            "facets_ok": facets_ok,
            "discourse_ok": discourse_ok,
            "resolution_ok": resolution_ok,
            "passed": protected_ok and facets_ok and discourse_ok and resolution_ok,
        })

    passed = sum(item["passed"] for item in results)
    return {
        "cases": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0,
        "production_minimum_met": len(results) >= 500,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
