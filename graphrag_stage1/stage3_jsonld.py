"""Compact research-grade JSON-LD projection for Stage 3 results."""

from __future__ import annotations

import hashlib


CONTEXT = {
    "graphrag": "https://w3id.org/graphrag/vocab/",
    "prov": "http://www.w3.org/ns/prov#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "type": "@type",
    "label": "rdfs:label",
    "derivedFrom": {"@id": "prov:wasDerivedFrom", "@type": "@id"},
    "subject": {"@id": "rdf:subject", "@type": "@id"},
    "predicate": {"@id": "rdf:predicate", "@type": "@id"},
    "object": {"@id": "rdf:object", "@type": "@id"},
    "sourceUri": "graphrag:sourceUri",
    "sourceText": "graphrag:sourceText",
    "page": "graphrag:page",
    "charStart": "graphrag:charStart",
    "charEnd": "graphrag:charEnd",
    "confidence": "graphrag:confidence",
    "mappingStatus": "graphrag:mappingStatus",
    "canonicalForm": "graphrag:canonicalForm",
    "surfaceForm": "graphrag:surfaceForm",
    "alias": "graphrag:alias",
    "value": "graphrag:value",
    "unit": "graphrag:unit",
    "mapping_audit": {"@id": "graphrag:mappingAudit", "@type": "@json"},
}


def _stable(kind: str, *values: object) -> str:
    key = "|".join(str(value or "") for value in values)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"urn:graphrag:{kind}:{digest}"


def build_jsonld(stage3: dict) -> dict:
    nodes: dict[str, dict] = {}

    def add(node: dict) -> None:
        identifier = node["@id"]
        if identifier in nodes:
            nodes[identifier].update({k: v for k, v in node.items() if v not in (None, [], {})})
        else:
            nodes[identifier] = {k: v for k, v in node.items() if v not in (None, [], {})}

    for frame in stage3.get("mapped_frames", []):
        provenance = frame.get("provenance", {}) or {}
        document_id = provenance.get("document_id") or "unknown-document"
        document = _stable("document", document_id, provenance.get("source_uri"))
        statement = _stable("statement", document_id, frame.get("statement_id"))
        add({
            "@id": document,
            "@type": "graphrag:SourceDocument",
            "label": document_id,
            "sourceUri": provenance.get("source_uri"),
        })
        add({
            "@id": statement,
            "@type": "graphrag:SourceStatement",
            "sourceText": frame.get("evidence", {}).get("source_statement"),
            "page": provenance.get("page"),
            "charStart": provenance.get("char_start"),
            "charEnd": provenance.get("char_end"),
            "derivedFrom": document,
        })
        frame_instance = frame.get("frame_instance")
        if frame_instance:
            add({
                "@id": frame_instance["id"],
                "@type": frame_instance["rdf_type"],
                "mappingStatus": frame.get("mapping_status"),
                "derivedFrom": statement,
                "sourceText": frame.get("evidence", {}).get("source_statement"),
                "page": provenance.get("page"),
            })
        for entity in frame.get("ontology_individuals", []):
            if not entity.get("instance_id"):
                continue
            add({
                "@id": entity["instance_id"],
                "@type": entity["rdf_type"],
                "label": entity.get("canonical_form") or entity.get("surface_form"),
                "canonicalForm": entity.get("canonical_form"),
                "surfaceForm": entity.get("surface_form"),
                "alias": entity.get("aliases"),
                "mappingStatus": entity.get("mapping_status"),
                "confidence": entity.get("mapping_confidence"),
                "derivedFrom": statement,
            })
        for assertion in frame.get("object_property_assertions", []):
            subject = assertion["subject"]
            predicate = assertion["predicate"]
            obj = assertion["object"]
            direct = nodes.setdefault(subject, {"@id": subject})
            value = {"@id": obj}
            existing = direct.get(predicate)
            if existing is None:
                direct[predicate] = value
            elif isinstance(existing, list):
                if value not in existing:
                    existing.append(value)
            elif existing != value:
                direct[predicate] = [existing, value]
            assertion_id = _stable("assertion", subject, predicate, obj, frame.get("statement_id"))
            add({
                "@id": assertion_id,
                "@type": "graphrag:QualifiedAssertion",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "sourceText": assertion.get("evidence_text"),
                "confidence": assertion.get("confidence"),
                "derivedFrom": statement,
            })
        mapped_measurements = frame.get("measurement_assertions", [])
        for mapped in mapped_measurements:
            measurement_id = mapped.get("instance_id") or _stable(
                "measurement", frame_instance["id"] if frame_instance else frame.get("statement_id"),
                mapped.get("value"), mapped.get("unit")
            )
            add({
                "@id": measurement_id,
                "@type": "graphrag:Measurement",
                "value": mapped.get("value") if mapped.get("value") is not None else mapped.get("raw_value"),
                "unit": mapped.get("unit"),
                "derivedFrom": statement,
            })
            if frame_instance:
                links = nodes[frame_instance["id"]].setdefault("graphrag:hasMeasurement", [])
                link = {"@id": measurement_id}
                if link not in links:
                    links.append(link)

    return {
        "@context": CONTEXT,
        "@graph": sorted(nodes.values(), key=lambda item: item["@id"]),
        "mapping_audit": stage3.get("audit", {}),
    }
