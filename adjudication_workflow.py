"""Two-reviewer workflow for the real-paper gold corpus.

Commands:
  export   Create a reviewer packet with blank Stage 1/2 annotations.
  merge    Merge a completed packet into the authoritative corpus.
  finalize Mark identical independent annotations as consensus; route differences.
"""

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_packet(corpus_path: Path, reviewer_id: str, output: Path, start: int, count: int) -> None:
    corpus = load(corpus_path)
    selected = corpus[start:start + count]
    packet = {
        "reviewer_id": reviewer_id,
        "instructions": {
            "independence": "Do not inspect another reviewer's annotations.",
            "stage1": "Annotate atomic statements, facets, discourse edges, and exact source spans.",
            "stage2": "Annotate semantic frames, entities, candidate relations, and reference resolutions.",
            "evidence": "Every annotation must be supported by the supplied abstract.",
        },
        "items": [{
            "id": item["id"],
            "document_id": item["document_id"],
            "domain_seed": item["domain_seed"],
            "title": item.get("title"),
            "text": item["text"],
            "source_uri": item["source_uri"],
            "stage1": None,
            "stage2": None,
            "review_notes": None,
        } for item in selected],
    }
    atomic_write(output, packet)


def merge_packet(corpus_path: Path, packet_path: Path) -> dict:
    corpus = load(corpus_path)
    packet = load(packet_path)
    reviewer_id = str(packet.get("reviewer_id") or "").strip()
    if not reviewer_id:
        raise ValueError("packet reviewer_id is required")
    by_id = {item["id"]: item for item in corpus}
    merged = 0
    for annotation in packet.get("items", []):
        if not isinstance(annotation.get("stage1"), dict) or not isinstance(annotation.get("stage2"), dict):
            raise ValueError(f"incomplete annotation for {annotation.get('id')}")
        record = by_id.get(annotation.get("id"))
        if not record:
            raise ValueError(f"unknown corpus id: {annotation.get('id')}")
        adjudication = record.setdefault("adjudication", {})
        annotations = adjudication.setdefault("annotations", [])
        normalized = {
            "reviewer_id": reviewer_id,
            "stage1": annotation["stage1"],
            "stage2": annotation["stage2"],
            "review_notes": annotation.get("review_notes"),
        }
        annotations[:] = [item for item in annotations if item.get("reviewer_id") != reviewer_id]
        annotations.append(normalized)
        adjudication["reviewers"] = sorted({item["reviewer_id"] for item in annotations})
        adjudication["status"] = "under_review"
        merged += 1
    atomic_write(corpus_path, corpus)
    return {"reviewer_id": reviewer_id, "merged": merged}


def finalize(corpus_path: Path) -> dict:
    corpus = load(corpus_path)
    counts = {"adjudicated": 0, "needs_resolution": 0, "insufficient_reviews": 0}
    for record in corpus:
        adjudication = record.setdefault("adjudication", {})
        annotations = adjudication.get("annotations", [])
        unique = {item.get("reviewer_id"): item for item in annotations if item.get("reviewer_id")}
        if len(unique) < 2:
            counts["insufficient_reviews"] += 1
            continue
        reviews = list(unique.values())
        first = {"stage1": reviews[0]["stage1"], "stage2": reviews[0]["stage2"]}
        agreement = all(
            {"stage1": item["stage1"], "stage2": item["stage2"]} == first
            for item in reviews[1:]
        )
        if agreement:
            adjudication.update({
                "status": "adjudicated",
                "agreement": "consensus",
                "reviewers": sorted(unique),
                "gold_stage1": first["stage1"],
                "gold_stage2": first["stage2"],
            })
            counts["adjudicated"] += 1
        else:
            adjudication["status"] = "needs_resolution"
            adjudication["agreement"] = "disagreement"
            counts["needs_resolution"] += 1
    atomic_write(corpus_path, corpus)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("corpus", type=Path)
    export.add_argument("reviewer_id")
    export.add_argument("output", type=Path)
    export.add_argument("--start", type=int, default=0)
    export.add_argument("--count", type=int, default=50)
    merge = sub.add_parser("merge")
    merge.add_argument("corpus", type=Path)
    merge.add_argument("packet", type=Path)
    finish = sub.add_parser("finalize")
    finish.add_argument("corpus", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export_packet(args.corpus, args.reviewer_id, args.output, args.start, args.count)
        result = {"exported": args.count, "output": str(args.output)}
    elif args.command == "merge":
        result = merge_packet(args.corpus, args.packet)
    else:
        result = finalize(args.corpus)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
