"""Tests for the versioned IR JSON Schemas and graphrag_stage1.validate."""

import copy
import unittest

from graphrag_stage1 import process_paragraph, stage2_pipeline, validate
from graphrag_stage1.validation import SchemaValidationError, schema_for
from graphrag_stage1 import stage1_classifier as s1


class FakeClient:
    """Fabricates a schema-valid dict from any JSON Schema (no network)."""

    model_id = "fake-model-v1"

    def complete(self, prompt, schema, *, stronger=False):
        return self._gen(schema)

    def _gen(self, s):
        if "enum" in s:
            return s["enum"][0]
        t = s.get("type")
        if t == "object":
            return {k: self._gen(v) for k, v in s.get("properties", {}).items()}
        if t == "array":
            return [self._gen(s["items"])]
        if t == "integer":
            return 1
        if t == "number":
            return 0.9
        if t == "boolean":
            return False
        return "sample"


def run_stage1():
    return process_paragraph("The sensor improved accuracy because fusion helps.", client=FakeClient())


class SchemaEnumSyncTests(unittest.TestCase):
    """The static schema enums must match the code constants (no drift)."""

    def test_facet_enums_match_code(self):
        facets = schema_for("stage_1_information_artifact_analysis", "1.1")["$defs"]["facets"]["properties"]
        self.assertEqual(facets["role"]["enum"], s1.ROLE)
        self.assertEqual(facets["proposition_type"]["enum"], s1.PROP_TYPE)
        self.assertEqual(facets["relation"]["enum"], s1.RELATION)
        self.assertEqual(facets["modality"]["enum"], s1.MODALITY)
        self.assertEqual(facets["attribution"]["enum"], s1.ATTRIBUTION)
        self.assertEqual(facets["certainty"]["enum"], s1.CERTAINTY)
        self.assertEqual(facets["polarity"]["enum"], s1.POLARITY)

    def test_discourse_enum_matches_code(self):
        schema = schema_for("stage_1_information_artifact_analysis", "1.1")
        edge_type = schema["$defs"]["discourse_edge"]["properties"]["type"]["enum"]
        self.assertEqual(edge_type, s1.DISCOURSE)


class ValidateTests(unittest.TestCase):
    def test_real_stage1_output_validates(self):
        out = run_stage1()
        self.assertIs(validate(out), out)  # returns payload for chaining

    def test_real_stage2_output_validates(self):
        out = stage2_pipeline(run_stage1(), client=FakeClient())
        self.assertIs(validate(out), out)

    def test_missing_required_field_fails(self):
        out = run_stage1()
        del out["statements"]
        with self.assertRaises(SchemaValidationError):
            validate(out)

    def test_bad_facet_enum_fails(self):
        out = run_stage1()
        out["statements"][0]["facets"]["relation"] = "NOT_A_REAL_RELATION"
        with self.assertRaises(SchemaValidationError) as ctx:
            validate(out)
        self.assertIn("relation", str(ctx.exception))

    def test_unknown_version_fails(self):
        out = run_stage1()
        out["pipeline_version"] = "9.9"
        with self.assertRaises(SchemaValidationError):
            validate(out)

    def test_non_dict_fails(self):
        with self.assertRaises(SchemaValidationError):
            validate(["not", "a", "dict"])


if __name__ == "__main__":
    unittest.main()
