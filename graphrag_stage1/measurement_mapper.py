"""Measurement individual creation for Stage 3."""

import re

from rdflib import URIRef

from .ontology_manager import OntologyManager


class MeasurementMapper:
    def __init__(self, manager: OntologyManager) -> None:
        self.manager = manager

    def map(self, value, unit, scope: str, statement_id: str, measured_entity: str | None = None) -> dict | None:
        if value in (None, ""):
            return None
        ontology_class = next((self.manager.find_class(term) for term in (
            "measurement datum", "measurement information content entity", "MEASUREMENT"
        ) if self.manager.find_class(term)), None)
        if ontology_class is None:
            ontology_class = sorted({item for values in self.manager.class_index.values() for item in values}, key=str)[0]
        raw_value = str(value).strip()
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", raw_value.replace(",", ""))
        numeric_value = float(numbers[-1]) if numbers else None
        if numeric_value is not None and numeric_value.is_integer():
            numeric_value = int(numeric_value)
        display = raw_value if unit and unit.casefold() in raw_value.casefold() else f"{raw_value} {unit or ''}".strip()
        instance = self.manager.create_instance(
            ontology_class, f"measurement|{scope}|{statement_id}|{raw_value}|{unit}", display
        )
        assertion = None
        measures = self.manager.find_object_property("measures")
        if measured_entity and measures:
            target = URIRef(measured_entity)
            self.manager.add_object_assertion(instance, measures, target)
            assertion = {"subject": str(instance), "predicate": str(measures), "object": measured_entity}
        return {"instance_id": str(instance), "rdf_type": str(ontology_class), "value": numeric_value, "raw_value": value, "unit": unit, "measured_entity": measured_entity, "object_assertion": assertion}
