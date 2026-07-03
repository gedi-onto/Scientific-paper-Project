"""Deterministic assertion provenance records."""

import hashlib


class ProvenanceBuilder:
    @staticmethod
    def statement_provenance(statement_id: str, provenance: dict, source_text: str | None = None) -> dict:
        digest = hashlib.sha256(
            f"{provenance.get('document_id')}|{statement_id}".encode("utf-8")
        ).hexdigest()[:20]
        return {
            "provenance_id": f"urn:graphrag:statement:{digest}",
            "derived_from_statement_id": statement_id,
            "document_id": provenance.get("document_id"),
            "source_uri": provenance.get("source_uri"),
            "page": provenance.get("page"),
            "char_start": provenance.get("char_start"),
            "char_end": provenance.get("char_end"),
            "verbatim": provenance.get("verbatim") or source_text,
        }

    @staticmethod
    def assertion_provenance(assertion: dict, statement_id: str, provenance: dict) -> dict:
        key = "|".join(str(assertion.get(field, "")) for field in ("subject", "predicate", "object", "value"))
        digest = hashlib.sha256(f"{statement_id}|{key}".encode("utf-8")).hexdigest()[:20]
        return {
            "assertion_id": f"urn:graphrag:assertion:{digest}",
            "derived_from_statement_id": statement_id,
            "document_id": provenance.get("document_id"),
            "source_uri": provenance.get("source_uri"),
            "page": provenance.get("page"),
            "char_start": provenance.get("char_start"),
            "char_end": provenance.get("char_end"),
        }
