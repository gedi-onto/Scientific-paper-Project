"""Validate pipeline outputs against the versioned IR JSON Schemas.

The schemas in ``graphrag_stage1/schemas/`` are the machine-checkable contract a
host platform pins to. ``validate`` picks the right schema from the payload's own
``stage`` + ``pipeline_version`` so integrators can assert compatibility without
hardcoding a version.

Requires the ``[validation]`` extra (``pip install graphrag-stage1[validation]``)
for ``jsonschema``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

# (stage, pipeline_version) -> schema filename in graphrag_stage1/schemas/
_SCHEMA_FILES = {
    ("stage_1_information_artifact_analysis", "1.1"): "stage1-1.1.json",
    ("stage_2_semantic_frame_extraction", "1.2"): "stage2-1.2.json",
}


class SchemaValidationError(ValueError):
    """Raised when a payload does not conform to its IR schema."""


@lru_cache(maxsize=None)
def _load_schema(filename: str) -> dict:
    text = resources.files(f"{__package__}.schemas").joinpath(filename).read_text("utf-8")
    return json.loads(text)


def schema_for(stage: str, pipeline_version: str) -> dict:
    """Return the JSON Schema dict for a given stage + version, or raise KeyError."""
    filename = _SCHEMA_FILES.get((stage, pipeline_version))
    if filename is None:
        known = ", ".join(f"{s}@{v}" for s, v in _SCHEMA_FILES)
        raise KeyError(
            f"no IR schema for stage={stage!r} pipeline_version={pipeline_version!r}; "
            f"known: {known}"
        )
    return _load_schema(filename)


def validate(payload: dict) -> dict:
    """Validate a Stage 1 or Stage 2 output against its versioned schema.

    The schema is selected from ``payload['stage']`` and
    ``payload['pipeline_version']``. Returns the payload on success (so calls can
    be chained); raises :class:`SchemaValidationError` on the first violation.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "graphrag_stage1.validation requires jsonschema; "
            "install with: pip install 'graphrag-stage1[validation]'"
        ) from exc

    if not isinstance(payload, dict):
        raise SchemaValidationError(f"payload must be a dict, got {type(payload).__name__}")
    stage = payload.get("stage")
    version = payload.get("pipeline_version")
    try:
        schema = schema_for(stage, version)
    except KeyError as exc:
        raise SchemaValidationError(str(exc)) from exc

    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise SchemaValidationError(f"{stage}@{version}: at {location}: {exc.message}") from exc
    return payload
