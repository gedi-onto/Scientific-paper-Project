"""Resumable JSONL runner for the Stage 1 -> Stage 2 pipeline.

Input JSONL records require ``id`` and ``text`` and may include document/page
metadata. One output record is appended and flushed per input paragraph, making
long document jobs restartable without reprocessing completed ids.

Usage:
    python pipeline_runner.py input.jsonl output.jsonl
"""

import argparse
import json
from pathlib import Path

from stage1_classifier import process_paragraph
from stage2_semantic_frames import stage2_pipeline
from production_support import AuditLogger, PipelineConfig, RoutingQueue, stable_content_id


def completed_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("status") == "complete":
                    ids.add(record.get("id"))
            except json.JSONDecodeError:
                continue
    return ids


def run_batch(
    input_path: Path,
    output_path: Path,
    queue_path: Path = None,
    audit_path: Path = None,
) -> dict:
    config = PipelineConfig()
    queue = RoutingQueue(queue_path or Path(config.queue_db))
    audit = AuditLogger(audit_path or Path(config.audit_log))
    done = completed_ids(output_path)
    counts = {"complete": 0, "failed": 0, "skipped": 0}

    with input_path.open(encoding="utf-8") as source, output_path.open(
        "a", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id") or stable_content_id(record.get("text", "")))
            if record_id in done:
                counts["skipped"] += 1
                continue
            try:
                metadata = record.get("metadata") or {}
                missing_provenance = [
                    field for field in ("document_id", "page", "source_uri")
                    if metadata.get(field) in (None, "")
                ]
                if missing_provenance:
                    raise ValueError(
                        f"missing required source metadata: {missing_provenance}"
                    )
                audit.write("paragraph_started", paragraph_id=record_id)
                stage1 = process_paragraph(
                    record["text"],
                    paragraph_id=record_id,
                    source_metadata=metadata,
                )
                stage2 = stage2_pipeline(stage1)
                queue_ids = []
                for frame in stage2["frames"]:
                    action = frame["stage2_frame"]["validation"]["automation_action"]
                    if action != "PASS_TO_ONTOLOGY_MAPPING":
                        queue_ids.append(queue.enqueue(record_id, frame))
                output = {
                    "id": record_id,
                    "status": "complete",
                    "source_metadata": metadata,
                    "stage1": stage1,
                    "stage2": stage2,
                    "routing_queue_ids": queue_ids,
                }
                counts["complete"] += 1
                audit.write(
                    "paragraph_completed",
                    paragraph_id=record_id,
                    frame_count=stage2["audit"]["frame_count"],
                    routing_queue_ids=queue_ids,
                )
            except Exception as exc:  # preserve job progress; error is routed in output
                output = {
                    "id": record_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                counts["failed"] += 1
                audit.write(
                    "paragraph_failed",
                    paragraph_id=record_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            sink.write(json.dumps(output, ensure_ascii=False) + "\n")
            sink.flush()

    queue.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable GraphRAG stages 1 and 2")
    parser.add_argument("input", type=Path, help="Input JSONL with id/text records")
    parser.add_argument("output", type=Path, help="Append-only checkpoint/output JSONL")
    parser.add_argument("--queue-db", type=Path, help="Durable SQLite routing queue")
    parser.add_argument("--audit-log", type=Path, help="Structured JSONL audit log")
    args = parser.parse_args()
    print(json.dumps(run_batch(
        args.input, args.output, queue_path=args.queue_db, audit_path=args.audit_log
    ), indent=2))


if __name__ == "__main__":
    main()
