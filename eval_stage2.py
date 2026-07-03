"""Live Stage 2 semantic-frame evaluation against adjudicated fixtures."""

import argparse
import json
from pathlib import Path

from stage2_semantic_frames import stage2_pipeline


def tokens(value: str) -> set:
    import re
    return {item for item in re.findall(r"[a-z0-9]+", str(value).lower()) if len(item) > 1}


def matches(expected: str, actual: str) -> bool:
    wanted = tokens(expected)
    found = tokens(actual)
    return bool(wanted) and len(wanted & found) / len(wanted) >= 0.7


def make_stage1(case: dict) -> dict:
    paragraph_id = case["id"]
    statements = []
    offset = 0
    for context in case.get("context", []):
        text = context["text"]
        statements.append({
            "id": context["id"],
            "text": text,
            "predicate": "",
            "arguments": [],
            "facets": {"role": "CONTENT", "proposition_type": "PROPERTY", "relation": "NONE", "modality": "OBSERVED"},
            "provenance": {"paragraph_id": paragraph_id, "document_id": f"doc-{paragraph_id}", "page": 1, "source_uri": f"gold://{paragraph_id}", "char_start": offset, "char_end": offset + len(text), "verbatim": text, "match": "exact"},
        })
        offset += len(text) + 1
    current_id = f"{paragraph_id}:u{len(statements) + 1}"
    text = case["text"]
    statements.append({
        "id": current_id,
        "text": text,
        "predicate": case.get("predicate", ""),
        "arguments": case.get("arguments", []),
        "facets": case["facets"],
        "provenance": {"paragraph_id": paragraph_id, "document_id": f"doc-{paragraph_id}", "page": 1, "source_uri": f"gold://{paragraph_id}", "char_start": offset, "char_end": offset + len(text), "verbatim": text, "match": "exact"},
    })
    return {"paragraph_id": paragraph_id, "paragraph": " ".join(item["text"] for item in statements), "statements": statements, "relations": []}


def evaluate(path: str = "stage2_gold.json") -> dict:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    totals = {"frames": 0, "frame_correct": 0, "entities": 0, "entities_found": 0,
              "gold_relations": 0, "relations_found": 0, "predicted_scored_relations": 0,
              "predicted_correct_relations": 0, "evidence": 0, "evidence_exact": 0,
              "coreferences": 0, "coreferences_correct": 0, "measurements": 0,
              "measurements_exact": 0, "unsupported_candidates": 0, "candidates": 0,
              "provenance": 0, "provenance_complete": 0}
    details = []

    for case in cases:
        output = stage2_pipeline(make_stage1(case))
        item = output["frames"][-1]
        frame = item["stage2_frame"]
        totals["frames"] += 1
        frame_ok = frame["frame_type"] == case["expected_frame_type"]
        totals["frame_correct"] += frame_ok

        entities = frame.get("candidate_entities", [])
        expected_entities = case.get("expected_entities", [])
        totals["entities"] += len(expected_entities)
        entity_hits = sum(any(matches(expected, actual) for actual in entities) for expected in expected_entities)
        totals["entities_found"] += entity_hits

        predicted = frame.get("candidate_relations", [])
        expected_relations = case.get("expected_relations", [])
        if expected_relations:
            totals["gold_relations"] += len(expected_relations)
            for expected in expected_relations:
                hit = any(matches(expected["subject"], rel["subject"]) and matches(expected["object"], rel["object"]) for rel in predicted)
                totals["relations_found"] += hit
            totals["predicted_scored_relations"] += len(predicted)
            for relation in predicted:
                hit = any(matches(expected["subject"], relation["subject"]) and matches(expected["object"], relation["object"]) for expected in expected_relations)
                totals["predicted_correct_relations"] += hit

        for relation in predicted:
            totals["evidence"] += 1
            totals["evidence_exact"] += relation.get("evidence_text") == case["text"]
        totals["candidates"] += len(entities) + len(predicted) * 2
        totals["unsupported_candidates"] += sum(
            "unsupported" in error for error in frame["validation"].get("grounding_errors", [])
        )

        expected_resolution = case.get("expected_resolution")
        if expected_resolution:
            totals["coreferences"] += 1
            resolution_ok = any(matches(expected_resolution, resolution.get("original")) for resolution in frame.get("reference_resolutions", []))
            totals["coreferences_correct"] += resolution_ok
        else:
            resolution_ok = True

        expected_measurement = case.get("expected_measurement")
        if expected_measurement:
            totals["measurements"] += 1
            semantic = frame["semantic_frame"]
            measurement_ok = str(semantic.get("measurement_value")) == expected_measurement["value"] and semantic.get("unit") == expected_measurement["unit"]
            totals["measurements_exact"] += measurement_ok
        else:
            measurement_ok = True

        provenance = item.get("provenance", {})
        totals["provenance"] += 1
        provenance_ok = all(provenance.get(field) is not None for field in ("paragraph_id", "document_id", "page", "source_uri", "char_start", "char_end"))
        totals["provenance_complete"] += provenance_ok
        details.append({"id": case["id"], "frame_ok": frame_ok, "entity_hits": f"{entity_hits}/{len(expected_entities)}", "resolution_ok": resolution_ok, "measurement_ok": measurement_ok, "provenance_ok": provenance_ok, "action": frame["validation"]["automation_action"]})

    ratio = lambda good, total: good / total if total else 1.0
    metrics = {
        "frame_type_accuracy": ratio(totals["frame_correct"], totals["frames"]),
        "entity_recall": ratio(totals["entities_found"], totals["entities"]),
        "candidate_relation_precision": ratio(totals["predicted_correct_relations"], totals["predicted_scored_relations"]),
        "candidate_relation_recall": ratio(totals["relations_found"], totals["gold_relations"]),
        "evidence_text_exactness": ratio(totals["evidence_exact"], totals["evidence"]),
        "coreference_resolution_accuracy": ratio(totals["coreferences_correct"], totals["coreferences"]),
        "measurement_exactness": ratio(totals["measurements_exact"], totals["measurements"]),
        "provenance_completeness": ratio(totals["provenance_complete"], totals["provenance"]),
        "unsupported_information_rate": ratio(totals["unsupported_candidates"], totals["candidates"]),
    }
    return {"cases": len(cases), "metrics": metrics, "totals": totals, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="stage2_gold.json")
    args = parser.parse_args()
    report = evaluate(args.gold)
    rendered = json.dumps(report, indent=2)
    print(rendered)


if __name__ == "__main__":
    main()
