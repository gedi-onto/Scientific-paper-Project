"""Operational controls shared by production pipeline entry points."""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    max_paragraph_chars: int = int(os.getenv("MAX_PARAGRAPH_CHARS", "50000"))
    queue_db: str = os.getenv("PIPELINE_QUEUE_DB", "pipeline_queue.sqlite3")
    audit_log: str = os.getenv("PIPELINE_AUDIT_LOG", "pipeline_audit.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_content_id(text: str, prefix: str = "p") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def validate_paragraph(text: object, max_chars: int) -> str:
    if not isinstance(text, str):
        raise TypeError("paragraph text must be a string")
    text = text.strip()
    if not text:
        raise ValueError("paragraph text must not be empty")
    if len(text) > max_chars:
        raise ValueError(f"paragraph exceeds MAX_PARAGRAPH_CHARS={max_chars}")
    if "\x00" in text:
        raise ValueError("paragraph contains NUL characters")
    return text


class AuditLogger:
    def __init__(self, path: Path):
        self.path = path

    def write(self, event: str, **fields) -> None:
        record = {"timestamp": utc_now(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class RoutingQueue:
    """Durable idempotent queue for frames that cannot advance automatically."""

    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS routing_queue (
                queue_id TEXT PRIMARY KEY,
                paragraph_id TEXT NOT NULL,
                statement_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def enqueue(self, paragraph_id: str, frame: dict) -> str:
        statement_id = str(frame.get("statement_id"))
        action = frame["stage2_frame"]["validation"]["automation_action"]
        queue_id = stable_content_id(
            f"{paragraph_id}|{statement_id}|{action}", prefix="route"
        )
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO routing_queue (
                queue_id, paragraph_id, statement_id, action, payload_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(queue_id) DO UPDATE SET
                action=excluded.action,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                queue_id, paragraph_id, statement_id, action,
                json.dumps(frame, ensure_ascii=False), now, now,
            ),
        )
        self.connection.commit()
        return queue_id

    def close(self) -> None:
        self.connection.close()

