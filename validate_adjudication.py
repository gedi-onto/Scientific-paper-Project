"""Validate that real-paper gold records have independent consensus evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_STAGE1 = {"statements"}
REQUIRED_STAGE2 = {"frames"}


def validate_record(record: dict) -> list:
    adjudication = record.get("adjudication", {})
    if adjudication.get("status") != "adjudicated":
        return ["pending"]
    errors = []
    reviewers = {str(item) for item in adjudication.get("reviewers", []) if item}
    annotations = adjudication.get("annotations", [])
    annotation_reviewers = {
        str(item.get("reviewer_id")) for item in annotations if item.get("reviewer_id")
    }
    if len(reviewers) < 2 or len(annotation_reviewers) < 2:
        errors.append("fewer than two independent reviewers")
    if reviewers != annotation_reviewers:
        errors.append("reviewer and annotation identities differ")
    if adjudication.get("agreement") != "consensus":
        errors.append("consensus not recorded")
    gold_stage1 = adjudication.get("gold_stage1")
    gold_stage2 = adjudication.get("gold_stage2")
    if not isinstance(gold_stage1, dict) or not REQUIRED_STAGE1 <= gold_stage1.keys():
        errors.append("gold_stage1 is incomplete")
    if not isinstance(gold_stage2, dict) or not REQUIRED_STAGE2 <= gold_stage2.keys():
        errors.append("gold_stage2 is incomplete")
    if any(not item.get("stage1") or not item.get("stage2") for item in annotations):
        errors.append("reviewer annotation is incomplete")
    return errors


def summarize(path: Path) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    statuses = Counter()
    invalid = []
    valid_adjudicated = 0
    for record in records:
        errors = validate_record(record)
        if errors == ["pending"]:
            statuses["pending"] += 1
        elif errors:
            statuses["invalid_adjudicated"] += 1
            invalid.append({"id": record.get("id"), "errors": errors})
        else:
            statuses["valid_adjudicated"] += 1
            valid_adjudicated += 1
    return {
        "records": len(records),
        "valid_adjudicated": valid_adjudicated,
        "statuses": dict(statuses),
        "invalid": invalid[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("osti_corpus.json"))
    args = parser.parse_args()
    report = summarize(args.corpus)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not report["invalid"] else 1)


if __name__ == "__main__":
    main()
