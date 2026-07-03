"""Stage 3: deterministic ontology mapping and instance generation.

Stage 3 consumes Stage 2 semantics as authoritative input. It performs no NLP,
statement classification, decomposition, or English triple extraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from rdflib import RDF, URIRef
from rdflib.namespace import XSD

from ontology_manager import OntologyManager, normalize_lookup
from assertion_builder import AssertionBuilder
from entity_mapper import EntityMapper
from measurement_mapper import MeasurementMapper
from provenance_builder import ProvenanceBuilder
from relation_mapper import RelationMapper
from stage3_jsonld import build_jsonld


def extract_stage2_frame(wrapper: dict) -> dict:
    return wrapper.get("stage2_frame", wrapper)


def lookup_first(lookup, terms: list[str]) -> Optional[URIRef]:
    for term in terms:
        result = lookup(term)
        if result is not None:
            return result
    return None


def frame_class_terms(frame_type: str) -> list[str]:
    spaced = str(frame_type or "").replace("_", " ")
    return [frame_type, spaced, f"{spaced} statement", f"{spaced} frame"]


def literal_datatype(value: object) -> URIRef:
    if isinstance(value, bool):
        return XSD.boolean
    if isinstance(value, int):
        return XSD.integer
    if isinstance(value, float):
        return XSD.decimal
    text = str(value or "").strip()
    if text and text.lstrip("+-").isdigit():
        return XSD.integer
    try:
        float(text)
        return XSD.decimal
    except (TypeError, ValueError):
        return XSD.string


def literal_assertion(
    subject: URIRef, predicate: URIRef, value: object, datatype: URIRef = XSD.string
) -> dict:
    return {
        "subject": str(subject),
        "predicate": str(predicate),
        "value": value,
        "datatype": str(datatype),
    }


def dedupe_assertions(assertions: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for assertion in assertions:
        key = tuple(str(assertion.get(field, "")) for field in ("subject", "predicate", "object", "value", "datatype"))
        if key not in seen:
            seen.add(key)
            result.append(assertion)
    return result


class Stage3OntologyMapper:
    """Map frozen Stage 2 structures exclusively through OntologyManager."""

    def __init__(self, manager: OntologyManager) -> None:
        self.manager = manager
        self.entity_mapper = EntityMapper(manager)
        self.relation_mapper = RelationMapper(manager)
        self.assertions = AssertionBuilder(manager)
        self.measurements = MeasurementMapper(manager)
        self.provenance_builder = ProvenanceBuilder()
        self.entity_registry: dict[str, dict] = {}
        self.salient_by_type: dict[str, list[dict]] = {}

    def _map_entity(
        self, surface: str, scope: str, canonical: str | None = None,
        aliases: list[str] | None = None, frame_type: str | None = None,
    ) -> dict:
        preview = self.entity_mapper.canonicalizer.canonicalize(
            surface, canonical if canonical and canonical != surface else None
        )
        generic = normalize_lookup(preview["canonical_form"])
        # Only truly generic, near-pronominal heads resolve to the most recent
        # salient individual of their type. "results"/"findings" are deliberately
        # excluded: they routinely denote distinct content and were collapsing
        # unrelated OBSERVATION mentions (e.g. "these findings" -> "new observations").
        generic_heads = {
            "system": "SYSTEM", "algorithm": "ALGORITHM", "model": "REPRESENTATIONAL_MODEL",
            "researcher": "PERSON", "researchers": "PERSON", "investigator": "PERSON",
            "investigators": "PERSON",
            "mechanism": "PROCESS",
        }
        if generic in generic_heads and self.salient_by_type.get(generic_heads[generic]):
            canonical = self.salient_by_type[generic_heads[generic]][-1]["canonical_form"]
        registry_hit = self.entity_registry.get(normalize_lookup(canonical or preview["canonical_form"]))
        if registry_hit:
            result = {**registry_hit, "surface_form": surface.strip()}
            result["aliases"] = sorted(
                set(result.get("aliases", []) + [surface.strip()] + (aliases or []))
                - {result["canonical_form"]}
            )
            result["canonicalization_method"] = "document_coreference"
            return result
        result = self.entity_mapper.map(
            surface, scope, frame_type=frame_type,
            resolved_text=canonical if canonical and canonical != surface else None,
        )
        result["aliases"] = sorted(set(result["aliases"] + (aliases or [])) - {result["canonical_form"]})
        self.entity_registry[normalize_lookup(result["canonical_form"])] = result
        self.salient_by_type.setdefault(result["semantic_type"], []).append(result)
        return result

    def _datatype(
        self,
        subject: URIRef,
        property_terms: list[str],
        value: object,
        assertions: list[dict],
        missing_properties: list[str],
        property_audits: list[dict],
    ) -> None:
        if value in (None, "", [], {}):
            return
        subject_class = next(
            (item for item in self.manager.graph.objects(subject, RDF.type) if isinstance(item, URIRef)),
            None,
        )
        datatype = literal_datatype(value)
        resolutions = [
            self.manager.resolve_datatype_property(term, subject_class, datatype)
            for term in property_terms
        ]
        resolved = next((item for item in resolutions if item["status"] == "matched"), None)
        property_audits.extend(resolutions)
        prop = URIRef(resolved["iri"]) if resolved else None
        if prop is None:
            missing_properties.append(property_terms[0])
            return
        self.manager.add_datatype_assertion(subject, prop, value, datatype)
        assertions.append(literal_assertion(subject, prop, value, datatype))

    def map_frame(self, wrapper: dict, paragraph_id: str | None) -> dict:
        frame = extract_stage2_frame(wrapper)
        statement_id = wrapper.get("statement_id") or frame.get("statement_id")
        source_text = wrapper.get("source_text") or frame.get("source_statement")
        frame_type = frame.get("frame_type") or wrapper.get("stage1_type")
        provenance = dict(wrapper.get("provenance", frame.get("provenance", {})) or {})
        generated_provenance_fields = []
        provenance_defaults = {
            "document_id": str(paragraph_id or "document"),
            "source_uri": f"urn:graphrag:source:{paragraph_id or 'document'}",
            "page": 1,
        }
        for field, default in provenance_defaults.items():
            if provenance.get(field) in (None, ""):
                provenance[field] = default
                generated_provenance_fields.append(field)
        if generated_provenance_fields:
            provenance["generated_fields"] = generated_provenance_fields
        evidence = {
            "source_statement": source_text,
            "statement_id": statement_id,
            "evidence_statement_ids": frame.get("semantic_frame", {}).get(
                "evidence_statement_ids", []
            ),
            "reference_resolutions": frame.get("reference_resolutions", []),
        }
        action = frame.get("validation", {}).get(
            "automation_action", "PASS_TO_ONTOLOGY_MAPPING"
        )
        if action != "PASS_TO_ONTOLOGY_MAPPING":
            return {
                "statement_id": statement_id,
                "mapping_status": "skipped_by_stage2_routing",
                "stage2_action": action,
                "frame_instance": None,
                "ontology_individuals": [],
                "class_assertions": [],
                "object_property_assertions": [],
                "datatype_assertions": [],
                "measurement_assertions": [],
                "provenance_record": self.provenance_builder.statement_provenance(str(statement_id), provenance, source_text),
                "provenance": provenance,
                "evidence": evidence,
                "ready_for_reasoning": False,
                "mapping_audit": {"errors": [f"Stage 2 routing action was {action}"]},
            }

        frame_lookups = [
            self.manager.explain_class_lookup(term)
            for term in frame_class_terms(str(frame_type or ""))
        ]
        frame_match = next((item for item in frame_lookups if item["status"] == "matched"), None)
        frame_class = URIRef(frame_match["iri"]) if frame_match else None
        errors: list[str] = []
        missing_properties: list[str] = []
        property_audits: list[dict] = []
        if frame_class is None:
            errors.append(f"No ontology class matched frame type {frame_type}")
            return {
                "statement_id": statement_id,
                "mapping_status": "unmapped_frame_class",
                "frame_instance": None,
                "ontology_individuals": [],
                "object_property_assertions": [],
                "datatype_assertions": [],
                "provenance": provenance,
                "evidence": evidence,
                "ready_for_reasoning": False,
                "mapping_audit": {
                    "errors": errors,
                    "frame_class_lookups": frame_lookups,
                },
            }

        scope = str(provenance.get("document_id") or paragraph_id or "document")
        frame_instance = self.manager.create_instance(
            frame_class,
            f"frame|{scope}|{statement_id}|{frame_type}",
            f"{frame_type} frame for {statement_id}",
        )

        semantic = frame.get("semantic_frame", {})
        resolutions = {
            normalize_lookup(item.get("original")): item
            for item in frame.get("reference_resolutions", [])
            if item.get("original") and item.get("resolved_text")
        }
        surfaces = list(frame.get("candidate_entities", []))
        for field in ("primary_entity", "secondary_entity", "actor", "affected_entity"):
            value = semantic.get(field)
            if value and value not in surfaces:
                surfaces.append(value)
        split_surfaces = []
        for surface in surfaces:
            if isinstance(surface, str):
                split_surfaces.extend(self.entity_mapper.canonicalizer.split_compound(surface))
        surfaces = split_surfaces
        entity_instances = []
        entity_by_surface = {}
        for surface in surfaces:
            if not isinstance(surface, str) or not surface.strip():
                continue
            key = normalize_lookup(surface)
            if key in entity_by_surface:
                continue
            resolution = resolutions.get(key)
            canonical = resolution.get("resolved_text") if resolution else surface.strip()
            aliases = [surface.strip()] if normalize_lookup(canonical) != key else []
            mapped = self._map_entity(surface.strip(), scope, canonical, aliases, frame_type)
            entity_by_surface[key] = mapped
            entity_by_surface[normalize_lookup(canonical)] = mapped
            entity_instances.append(mapped)

        unique_individuals: dict[str, dict] = {}
        for item in entity_instances:
            identifier = item.get("instance_id") or f"surface:{normalize_lookup(item.get('surface_form'))}"
            if identifier in unique_individuals:
                existing = unique_individuals[identifier]
                existing["aliases"] = sorted(set(
                    existing.get("aliases", []) + item.get("aliases", []) + [item.get("surface_form", "")]
                ) - {"", existing.get("canonical_form")})
            else:
                unique_individuals[identifier] = {**item}
        entity_instances = list(unique_individuals.values())

        object_assertions = []
        for relation in frame.get("candidate_relations", []):
            subject = entity_by_surface.get(normalize_lookup(relation.get("subject")))
            relation_term = relation.get("relation", "")
            subject_class = URIRef(subject["rdf_type"]) if subject and subject.get("rdf_type") else None
            if not subject or not subject.get("instance_id"):
                errors.append(f"Unmapped relation subject: {relation.get('subject')}")
                continue
            object_surfaces = self.entity_mapper.canonicalizer.split_compound(
                relation.get("object", "")
            )
            objects = [
                entity_by_surface.get(normalize_lookup(item))
                for item in object_surfaces
            ]
            objects = [item for item in objects if item and item.get("instance_id")]
            if not objects:
                errors.append(f"Unmapped relation object: {relation.get('object')}")
                continue
            for obj in objects:
                object_class = URIRef(obj["rdf_type"]) if obj.get("rdf_type") else None
                property_lookup = self.relation_mapper.resolve(
                    relation_term, subject_class, object_class, frame_type=frame_type
                )
                property_audits.append(property_lookup)
                prop = URIRef(property_lookup["iri"]) if property_lookup["iri"] else None
                if prop is None:
                    error = f"{property_lookup['status'].title()} object property: {relation_term}"
                    if error not in errors:
                        errors.append(error)
                    continue
                subject_iri = URIRef(subject["instance_id"])
                object_iri = URIRef(obj["instance_id"])
                self.manager.add_object_assertion(subject_iri, prop, object_iri)
                object_assertions.append({
                    "subject": str(subject_iri),
                    "predicate": str(prop),
                    "object": str(object_iri),
                    "source_relation": relation.get("relation"),
                    "qualifier": relation.get("qualifier"),
                    "evidence_text": relation.get("evidence_text"),
                    "assertion_role": "domain",
                })

        about_property = lookup_first(
            self.manager.find_object_property,
            ["has semantic participant", "is about"],
        )
        if about_property:
            participant_ids = set()
            for entity in entity_instances:
                if entity.get("instance_id") and entity["instance_id"] not in participant_ids:
                    participant_ids.add(entity["instance_id"])
                    entity_iri = URIRef(entity["instance_id"])
                    self.manager.add_object_assertion(frame_instance, about_property, entity_iri)
                    object_assertions.append({
                        "subject": str(frame_instance),
                        "predicate": str(about_property),
                        "object": str(entity_iri),
                        "source_relation": "is about",
                        "qualifier": None,
                        "evidence_text": source_text,
                    })
        else:
            missing_properties.append("has semantic participant")

        datatype_assertions: list[dict] = []
        for entity in entity_instances:
            datatype_assertions.extend(entity.get("datatype_assertions", []))
        self._datatype(
            frame_instance, ["source statement id"], statement_id,
            datatype_assertions, missing_properties, property_audits,
        )
        self._datatype(
            frame_instance, ["source text"], source_text,
            datatype_assertions, missing_properties, property_audits,
        )
        measurement = semantic.get("measurement", {}) or {}
        measurement_value = measurement.get("value", semantic.get("measurement_value"))
        unit = measurement.get("unit", semantic.get("unit"))
        measurement_inputs = list(semantic.get("measurements", []) or [])
        if not measurement_inputs and measurement_value not in (None, ""):
            measurement_inputs = [{"value": measurement_value, "raw_value": measurement_value, "unit": unit}]
        measurement_assertions = []
        for index, measurement_input in enumerate(measurement_inputs):
            measured_surface = measurement_input.get("measured_property") or semantic.get("secondary_entity")
            measured = entity_by_surface.get(normalize_lookup(measured_surface))
            mapped_measurement = self.measurements.map(
                measurement_input.get("value"), measurement_input.get("unit"), scope, f"{statement_id}|{index}",
                measured.get("instance_id") if measured else None,
            )
            if not mapped_measurement:
                continue
            mapped_measurement["comparator"] = measurement_input.get("comparator")
            mapped_measurement["raw_value"] = measurement_input.get("raw_value", measurement_input.get("value"))
            measurement_assertions.append(mapped_measurement)
            if mapped_measurement.get("object_assertion"):
                object_assertions.append({
                    **mapped_measurement["object_assertion"],
                    "source_relation": "measures", "qualifier": measurement_input.get("comparator"),
                    "evidence_text": source_text, "assertion_role": "domain",
                })
            self._datatype(
                frame_instance, ["measurement value"], mapped_measurement.get("value"),
                datatype_assertions, missing_properties, property_audits,
            )
            self._datatype(
                frame_instance, ["measurement unit", "unit"], mapped_measurement.get("unit"),
                datatype_assertions, missing_properties, property_audits,
            )
        for field, terms in (
            ("paragraph_id", ["source paragraph id"]),
            ("page", ["source page"]),
            ("char_start", ["source character start"]),
            ("char_end", ["source character end"]),
            ("source_uri", ["source uri"]),
        ):
            self._datatype(
                frame_instance, terms, provenance.get(field),
                datatype_assertions, missing_properties, property_audits,
            )

        unmapped_entities = [
            item["surface_form"] for item in entity_instances
            if item["mapping_status"] != "mapped"
        ]
        class_assertions = []
        seen_classes = set()
        for item in entity_instances:
            if not item.get("instance_id"):
                continue
            key = (item["instance_id"], item["ontology_class"])
            if key not in seen_classes:
                seen_classes.add(key)
                class_assertions.append({"individual": item["instance_id"], "ontology_class": item["ontology_class"], "inferred": False})
        object_assertions = dedupe_assertions(object_assertions)
        datatype_assertions = dedupe_assertions(datatype_assertions)
        all_assertions = object_assertions + datatype_assertions
        provenance_record = self.provenance_builder.statement_provenance(str(statement_id), provenance, source_text)
        for item in all_assertions:
            item["provenance_id"] = provenance_record["provenance_id"]
        reasoning_profile = {
            "class_assertions_ready": bool(frame_instance) and all(item.get("ontology_class") for item in entity_instances),
            "domain_assertions_ready": any(
                item.get("assertion_role") == "domain" for item in object_assertions
            ),
            "object_properties_ready": not any("object property" in error.casefold() for error in errors),
            "datatype_properties_ready": True,
            "provenance_complete": all(
                provenance.get(field) not in (None, "")
                for field in ("document_id", "source_uri", "page", "char_start", "char_end")
            ),
            "shacl_ready": True,
            "owl_ready": bool(frame_instance) and not unmapped_entities,
        }
        ready = all(reasoning_profile.values()) and not errors
        return {
            "statement_id": statement_id,
            "mapping_status": "mapped" if ready else "partial_mapping",
            "frame_instance": {
                "id": str(frame_instance),
                "rdf_type": str(frame_class),
                "frame_type": frame_type,
            },
            "ontology_individuals": entity_instances,
            "class_assertions": class_assertions,
            "object_property_assertions": object_assertions,
            "datatype_assertions": datatype_assertions,
            "measurement_assertions": measurement_assertions,
            "provenance_record": provenance_record,
            "provenance": provenance,
            "evidence": evidence,
            "measurement": {"value": measurement_value, "unit": unit},
            "ready_for_reasoning": ready,
            "reasoning_profile": reasoning_profile,
            "mapping_audit": {
                "errors": errors,
                "unmapped_entities": unmapped_entities,
                "unresolved_entities": [
                    {
                        "surface_form": item["surface_form"],
                        "status": item["mapping_status"],
                        "candidates": item["mapping_candidates"],
                    }
                    for item in entity_instances if item["mapping_status"] != "mapped"
                ],
                "frame_class_lookup": frame_match,
                "missing_optional_properties": sorted(set(missing_properties)),
                "property_axiom_audits": property_audits,
                "entity_count": len(entity_instances),
                "object_assertion_count": len(object_assertions),
                "datatype_assertion_count": len(datatype_assertions),
            },
        }

    def map_output(self, stage2_output: dict) -> dict:
        self.entity_registry.clear()
        self.salient_by_type.clear()
        paragraph_id = stage2_output.get("paragraph_id")
        mapped_frames = [
            self.map_frame(wrapper, paragraph_id)
            for wrapper in stage2_output.get("frames", [])
        ]
        mapped_by_statement = {
            item.get("statement_id"): item for item in mapped_frames
            if item.get("statement_id")
        }
        graph = stage2_output.get("statement_graph", {})
        edges = graph.get("edges", [])
        if not edges:
            local_to_full = {
                str(statement_id).rsplit(":", 1)[-1]: statement_id
                for statement_id in mapped_by_statement
            }
            edges = [{
                "from": local_to_full.get(str(edge.get("from")).rsplit(":", 1)[-1], edge.get("from")),
                "to": local_to_full.get(str(edge.get("to")).rsplit(":", 1)[-1], edge.get("to")),
                "type": edge.get("type"),
            } for edge in stage2_output.get("stage1_relations", [])]

        for edge in edges:
            source = mapped_by_statement.get(edge.get("from"))
            target = mapped_by_statement.get(edge.get("to"))
            if not source or not target or not source.get("frame_instance") or not target.get("frame_instance"):
                continue
            discourse_lookup = self.relation_mapper.resolve_discourse(edge.get("type", ""))
            prop = URIRef(discourse_lookup["iri"]) if discourse_lookup.get("iri") else None
            if prop is None:
                source["mapping_audit"]["errors"].append(
                    f"Unmapped discourse property: {edge.get('type')}"
                )
                source["mapping_status"] = "partial_mapping"
                source["ready_for_reasoning"] = False
                continue
            source_iri = URIRef(source["frame_instance"]["id"])
            target_iri = URIRef(target["frame_instance"]["id"])
            self.manager.add_object_assertion(source_iri, prop, target_iri)
            source["object_property_assertions"].append({
                "subject": str(source_iri),
                "predicate": str(prop),
                "object": str(target_iri),
                "source_relation": edge.get("type"),
                "qualifier": None,
                "evidence_text": None,
                "assertion_kind": "stage1_discourse_edge",
            })
            source["object_property_assertions"][-1]["provenance_id"] = source["provenance_record"]["provenance_id"]
            source["mapping_audit"]["object_assertion_count"] += 1

        ready = sum(item["ready_for_reasoning"] for item in mapped_frames)
        result = {
            "stage": "stage_3_ontology_mapping",
            "pipeline_version": "2.0",
            "paragraph_id": paragraph_id,
            "ontology_profile": self.manager.profile(),
            "mapped_frames": mapped_frames,
            "audit": {
                "frames_received": len(stage2_output.get("frames", [])),
                "frames_mapped": len(mapped_frames),
                "ready_for_reasoning_count": ready,
                "needs_mapping_review_count": len(mapped_frames) - ready,
            },
        }
        result["jsonld"] = build_jsonld(result)
        return result


def stage3_pipeline(stage2_output: dict, manager: OntologyManager) -> dict:
    return Stage3OntologyMapper(manager).map_output(stage2_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map Stage 2 frames to loaded ontologies")
    parser.add_argument("input", nargs="?", default="stage2_output.json")
    parser.add_argument("--ontology-root", default="ontologies")
    parser.add_argument("--domain-ontology")
    parser.add_argument("--neptune-output", help="Write validated UTF-8 N-Quads")
    parser.add_argument("--quarantine-output", help="Write quarantined RDF as Turtle")
    parser.add_argument("--validation-report", help="Write the SHACL validation report")
    args = parser.parse_args()
    stage2_output = json.load(sys.stdin) if args.input == "-" else json.loads(
        Path(args.input).read_text(encoding="utf-8-sig")
    )
    manager = OntologyManager(args.ontology_root).load_all(args.domain_ontology)
    output = stage3_pipeline(stage2_output, manager)
    if args.neptune_output:
        from stage3_neptune_export import export_neptune_nquads

        output["neptune_export"] = export_neptune_nquads(
            output,
            args.neptune_output,
            quarantine_destination=args.quarantine_output,
            report_destination=args.validation_report,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
