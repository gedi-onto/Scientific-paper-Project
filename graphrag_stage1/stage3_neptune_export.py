"""Validate Stage 3 JSON-LD and export Neptune bulk-loader N-Quads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pyshacl import validate
from rdflib import Dataset, Graph, URIRef
from rdflib.namespace import RDF, SH


# Resolved from the package, not the working directory: the shapes ship inside the wheel,
# and a CWD-relative default silently works in the repo while failing for every installed
# user (who has no ./ontologies to find).
DEFAULT_SHAPES = Path(__file__).resolve().parent / "ontologies" / "Alignment" / "stage3-publication-shapes.ttl"


def _data_graph(stage3_output: dict) -> Graph:
    jsonld = stage3_output.get("jsonld")
    if not jsonld:
        raise ValueError("Stage 3 output has no jsonld artifact; rerun with pipeline version 1.1+")
    payload = {"@context": jsonld["@context"], "@graph": jsonld["@graph"]}
    graph = Graph()
    graph.parse(data=json.dumps(payload), format="json-ld")
    return graph


def _invalid_focus_nodes(report_graph: Graph) -> set:
    return {
        focus
        for result in report_graph.subjects(RDF.type, SH.ValidationResult)
        for focus in report_graph.objects(result, SH.focusNode)
    }


def validate_and_partition(
    stage3_output: dict, shapes_path: str | Path = DEFAULT_SHAPES
) -> dict:
    data = _data_graph(stage3_output)
    shapes = Graph().parse(Path(shapes_path), format="turtle")
    conforms, report_graph, report_text = validate(
        data_graph=data,
        shacl_graph=shapes,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    invalid_nodes = _invalid_focus_nodes(report_graph)
    valid_graph = Graph()
    quarantine_graph = Graph()
    for triple in data:
        subject, _, obj = triple
        target = quarantine_graph if subject in invalid_nodes or obj in invalid_nodes else valid_graph
        target.add(triple)
    valid_conforms, valid_report, valid_text = validate(
        data_graph=valid_graph,
        shacl_graph=shapes,
        inference="rdfs",
        abort_on_first=False,
    )
    if not valid_conforms:
        raise ValueError(f"Graph remained invalid after quarantine:\n{valid_text}")
    return {
        "conforms": bool(conforms),
        "valid_graph": valid_graph,
        "quarantine_graph": quarantine_graph,
        "quarantined_nodes": sorted(str(item) for item in invalid_nodes),
        "report_graph": report_graph,
        "report_text": report_text,
        "validated_triple_count": len(valid_graph),
        "quarantined_triple_count": len(quarantine_graph),
    }


def export_neptune_nquads(
    stage3_output: dict,
    destination: str | Path,
    quarantine_destination: str | Path | None = None,
    report_destination: str | Path | None = None,
    shapes_path: str | Path = DEFAULT_SHAPES,
) -> dict:
    partition = validate_and_partition(stage3_output, shapes_path)
    document_key = stage3_output.get("paragraph_id") or "document"
    digest = hashlib.sha256(str(document_key).encode("utf-8")).hexdigest()[:20]
    graph_name = URIRef(f"urn:graphrag:graph:{digest}")
    dataset = Dataset()
    named = dataset.graph(graph_name)
    for triple in partition["valid_graph"]:
        named.add(triple)
    Path(destination).write_text(dataset.serialize(format="nquads"), encoding="utf-8")
    if quarantine_destination:
        Path(quarantine_destination).write_text(
            partition["quarantine_graph"].serialize(format="turtle"), encoding="utf-8"
        )
    if report_destination:
        Path(report_destination).write_text(partition["report_text"], encoding="utf-8")
    return {
        "format": "nquads",
        "encoding": "utf-8",
        "named_graph": str(graph_name),
        "destination": str(Path(destination).resolve()),
        "conforms_before_quarantine": partition["conforms"],
        "validated_triple_count": partition["validated_triple_count"],
        "quarantined_triple_count": partition["quarantined_triple_count"],
        "quarantined_nodes": partition["quarantined_nodes"],
    }
