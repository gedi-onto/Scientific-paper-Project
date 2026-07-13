#!/usr/bin/env python
"""How many Stage 2 model calls would the hybrid actually save on a real run?

    python examples/hybrid_skip_report.py examples/full_run_output/results.json

Replays every Stage 1 statement from a completed run through the hybrid's rule
builder and the same deterministic gate the LLM path must pass. No model calls --
this is pure analysis of an existing run, so it is instant and free.

Reports the artifact-type mix, the exact fraction of statements that would skip
the model, and the projected Stage 2 saving.
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Make graphrag_stage1 importable when run as `python examples/hybrid_skip_report.py`
# (sys.path[0] is examples/, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphrag_stage1.stage2_semantic_frames import (
    _frame_is_complete,
    derive_artifact_type,
    extract_deterministic_frame,
)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/full_run_output/results.json"
    with open(path, encoding="utf-8") as handle:
        results = json.load(handle)

    by_type: Counter = Counter()
    skipped_by_type: Counter = Counter()
    total = skipped = 0

    for para in results:
        if "stage1" not in para:
            continue  # failed paragraph
        stage1 = para["stage1"]
        for statement in stage1.get("statements", []):
            artifact_type = derive_artifact_type(statement, stage1)
            by_type[artifact_type] += 1
            total += 1
            item = extract_deterministic_frame(statement, stage1)
            if _frame_is_complete(item["stage2_frame"], artifact_type):
                skipped += 1
                skipped_by_type[artifact_type] += 1

    if not total:
        raise SystemExit("no Stage 1 statements found in that results file")

    print(f"{'artifact type':<22}{'count':>7}{'skips model':>13}{'rate':>8}")
    print("-" * 50)
    for artifact_type, count in by_type.most_common():
        hit = skipped_by_type[artifact_type]
        print(f"{artifact_type:<22}{count:>7}{hit:>13}{hit / count * 100:>7.0f}%")
    print("-" * 50)
    rate = skipped / total
    print(f"{'TOTAL':<22}{total:>7}{skipped:>13}{rate * 100:>7.0f}%")

    print(
        f"\nStage 2 model calls: {total} -> {total - skipped}"
        f"  ({rate * 100:.0f}% eliminated)"
    )
    if rate > 0:
        print(f"projected Stage 2 speedup: {1 / (1 - rate):.2f}x")
    print("\nNote: this counts calls, not wall-clock. Stage 2 is decode-bound, so")
    print("saved calls translate roughly 1:1 into saved time. Stage 1 is unchanged.")


if __name__ == "__main__":
    main()
