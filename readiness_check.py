"""Fail-closed production-readiness gate for CI and deployment workflows."""

import json
import subprocess
import sys
from pathlib import Path

from validate_adjudication import validate_record


REQUIRED_FILES = [
    "stage1_classifier.py",
    "stage2_semantic_frames.py",
    "pipeline_runner.py",
    "production_support.py",
    "release_gates.json",
    "evaluation_status.json",
    "cross_domain_gold.json",
    "stage2_gold.json",
    "eval_stage2.py",
    "build_osti_corpus.py",
    "osti_corpus.json",
    "adjudication_workflow.py",
    "doe_authorization.template.json",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    gates = json.loads((root / "release_gates.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "evaluation_status.json").read_text(encoding="utf-8"))
    corpus = json.loads((root / "cross_domain_gold.json").read_text(encoding="utf-8"))
    real_corpus_path = root / "osti_corpus.json"
    real_corpus = json.loads(real_corpus_path.read_text(encoding="utf-8")) if real_corpus_path.exists() else []
    authorization_path = root / "doe_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8")) if authorization_path.exists() else {}

    def approved(section: str, official_field: str, scope_field: str) -> bool:
        value = authorization.get(section, {})
        return value.get("status") == "approved" and all(
            value.get(field) for field in (
                official_field, "approval_date", scope_field, "evidence_reference"
            )
        )
    adjudicated = [
        record for record in real_corpus
        if validate_record(record) == []
    ]
    quality_results = {}
    for name, threshold in gates["required"].items():
        if name.endswith("_max"):
            metric_name = name[:-4]
            value = metrics.get(metric_name)
            quality_results[name] = value is not None and value <= threshold
        else:
            value = metrics.get(name)
            quality_results[name] = value is not None and value >= threshold

    checks = {
        "required_files": all((root / name).exists() for name in REQUIRED_FILES),
        "deterministic_tests": False,
        "minimum_corpus_size": len(adjudicated) >= gates["minimum_evaluation_size"]["paragraphs"],
        "minimum_document_count": len({item.get("document_id") for item in adjudicated})
        >= gates["minimum_evaluation_size"]["documents"],
        "minimum_domain_count": len({item.get("domain_seed") for item in adjudicated})
        >= gates["minimum_evaluation_size"]["scientific_domains"],
        "quality_metrics": all(quality_results.values()),
        "security_authorization": approved(
            "security_authorization", "authorizing_official", "system_scope"
        ),
        "data_governance_approval": approved(
            "data_governance_approval", "approving_official", "data_scope"
        ),
    }
    test = subprocess.run(
        [sys.executable, "-m", "unittest", "-q", "test_pipeline.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    checks["deterministic_tests"] = test.returncode == 0
    ready = all(checks.values())
    print(json.dumps({
        "status": "READY" if ready else "BLOCKED",
        "checks": checks,
        "seed_corpus_paragraphs": len(corpus),
        "real_corpus_downloaded": len(real_corpus),
        "real_corpus_adjudicated": len(adjudicated),
        "required_paragraphs": gates["minimum_evaluation_size"]["paragraphs"],
        "quality_results": quality_results,
        "note": "Authorization gates must be supplied by the target DOE program.",
    }, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
