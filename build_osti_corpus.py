"""Build a real-paper evaluation corpus from public DOE OSTI abstracts.

Records are intentionally marked ``pending_adjudication``. Merely downloading or
model-labeling text does not make it release evidence; a reviewer must populate the
gold fields and change adjudication.status to ``adjudicated``.
"""

import argparse
import json
import math
import time
from pathlib import Path

import requests


API_URL = "https://www.osti.gov/api/v1/records"
SUBJECT_QUERIES = [
    "physics", "chemistry", "biology", "medicine", "materials",
    "climate", "computer science", "engineering", "mathematics", "astronomy",
]


def get_with_backoff(session: requests.Session, params: dict) -> requests.Response:
    for attempt in range(6):
        response = session.get(API_URL, params=params, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(60, 5 * (2 ** attempt))
        time.sleep(delay)
    response.raise_for_status()


def fetch_records(target: int, rows: int = 100) -> list:
    records, seen = [], set()
    per_subject = math.ceil(target / len(SUBJECT_QUERIES))
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "doe-graphrag-evaluation/1.0"})
    for subject in SUBJECT_QUERIES:
        subject_count = 0
        page = 1
        while len(records) < target and subject_count < per_subject:
            response = get_with_backoff(
                session,
                {"subject": subject, "language": "English", "rows": rows, "page": page},
            )
            batch = response.json()
            if not batch:
                break
            for record in batch:
                osti_id = str(record.get("osti_id", ""))
                abstract = " ".join(str(record.get("description") or "").split())
                if not osti_id or osti_id in seen or len(abstract) < 120:
                    continue
                seen.add(osti_id)
                records.append({
                    "id": f"osti-{osti_id}",
                    "document_id": osti_id,
                    "domain_seed": subject,
                    "title": record.get("title"),
                    "text": abstract,
                    "source_uri": f"https://www.osti.gov/biblio/{osti_id}",
                    "publication_date": record.get("publication_date"),
                    "subjects": record.get("subjects", []),
                    "adjudication": {
                        "status": "pending_adjudication",
                        "reviewers": [],
                        "annotations": [],
                        "agreement": None,
                        "gold_stage1": None,
                        "gold_stage2": None
                    }
                })
                subject_count += 1
                if len(records) >= target or subject_count >= per_subject:
                    break
            page += 1
            time.sleep(1.0)
        if len(records) >= target:
            break

    # Fill sparse-domain shortfalls from a broad multidisciplinary query while
    # preserving the balanced domain sample already collected above.
    page = 1
    while len(records) < target:
        response = get_with_backoff(
            session,
            {"q": "research", "language": "English", "rows": rows, "page": page},
        )
        batch = response.json()
        if not batch:
            break
        for record in batch:
            osti_id = str(record.get("osti_id", ""))
            abstract = " ".join(str(record.get("description") or "").split())
            if not osti_id or osti_id in seen or len(abstract) < 120:
                continue
            seen.add(osti_id)
            records.append({
                "id": f"osti-{osti_id}",
                "document_id": osti_id,
                "domain_seed": "multidisciplinary",
                "title": record.get("title"),
                "text": abstract,
                "source_uri": f"https://www.osti.gov/biblio/{osti_id}",
                "publication_date": record.get("publication_date"),
                "subjects": record.get("subjects", []),
                "adjudication": {
                    "status": "pending_adjudication",
                    "reviewers": [],
                    "annotations": [],
                    "agreement": None,
                    "gold_stage1": None,
                    "gold_stage2": None,
                },
            })
            if len(records) >= target:
                break
        page += 1
        time.sleep(1.0)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()
    records = fetch_records(args.target)
    # This is a generated corpus artifact, written atomically by the corpus tool.
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"downloaded": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
