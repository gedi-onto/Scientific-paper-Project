"""
Evaluation harness for Stage 1.

Exact-match alignment against gold units is brittle because decomposition is
non-deterministic. Instead we measure what actually matters for the pipeline:

  1. Artifact recall  -- for each required facet pattern (e.g. "a MECHANISM unit
                         must exist"), did the pipeline produce at least one unit
                         matching it? This directly tracks "did we catch the
                         mechanism / prediction / negation?".
  2. Decontextualization -- fraction of units with NO dangling anaphor. A unit
                         starting with These/This/That/Those/They/It/Such almost
                         always means coreference was not resolved.
  3. Unit-count sanity -- did decomposition land in the expected count range
                         (over/under-splitting check).

Run:  python eval.py
"""

import json
import re

from graphrag_stage1.stage1_classifier import process_paragraph


GOLD_PATH = "gold.json"

# A unit that opens with one of these is almost certainly un-decontextualized.
ANAPHORA_RE = re.compile(
    r"^\s*(These|This|That|Those|They|It|Its|Such|The former|The latter)\b"
)


def matches(statement: dict, pattern: dict) -> bool:
    """A statement matches a gold pattern if every specified facet is equal.

    Facets now live under statement["facets"]; fall back to top level for
    backward compatibility.
    """
    facets = statement.get("facets", statement)
    return all(facets.get(k) == v for k, v in pattern.items())


def evaluate(gold: list) -> None:
    total_constraints = 0
    met_constraints = 0
    total_units = 0
    clean_units = 0
    count_ok = 0
    total_flags = 0
    score_sum = 0.0

    print("=" * 72)
    for case in gold:
        result = process_paragraph(case["text"], case["id"])
        statements = result["statements"]
        audit = result["audit"]
        total_units += len(statements)
        total_flags += len(audit["consistency_flags"])
        score_sum += audit["completeness_score"]

        # 1. Artifact recall
        misses = []
        for pattern in case["must_contain"]:
            total_constraints += 1
            if any(matches(s, pattern) for s in statements):
                met_constraints += 1
            else:
                misses.append(pattern)

        # 2. Decontextualization
        dangling = [s["text"] for s in statements if ANAPHORA_RE.match(s["text"])]
        clean_units += len(statements) - len(dangling)

        # 3. Unit-count sanity
        lo, hi = case["expected_unit_count"]
        in_range = lo <= len(statements) <= hi
        if in_range:
            count_ok += 1

        status = "OK " if not misses and not dangling and in_range else "FAIL"
        print(f"[{status}] {case['id']}")
        print(f"       units={len(statements)} (expected {lo}-{hi})"
              f"{'' if in_range else '  <-- out of range'}")
        if misses:
            print(f"       missing artifacts: {misses}")
        if dangling:
            print(f"       dangling anaphora: {dangling}")
        if audit["consistency_flags"]:
            print(f"       consistency flags: {audit['consistency_flags']}")
        print(f"       completeness={audit['completeness_score']}"
              f"  (floor {audit['produced']}/{audit['recall_floor']})")
        print("-" * 72)

    n = len(gold)
    print("SUMMARY")
    print(f"  Artifact recall      : {met_constraints}/{total_constraints}"
          f" = {met_constraints / total_constraints:.0%}")
    print(f"  Decontextualization  : {clean_units}/{total_units}"
          f" = {clean_units / total_units:.0%}")
    print(f"  Unit-count in range  : {count_ok}/{n} = {count_ok / n:.0%}")
    print(f"  Facet consistency    : {total_units - total_flags}/{total_units}"
          f" clean = {(total_units - total_flags) / total_units:.0%}")
    print(f"  Mean completeness    : {score_sum / n:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    with open(GOLD_PATH, encoding="utf-8") as f:
        gold = json.load(f)
    evaluate(gold)
