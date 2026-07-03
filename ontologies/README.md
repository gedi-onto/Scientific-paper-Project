# Ontology layout

Stage 3 expects the following runtime layout:

```text
ontologies/
  BFO/       # one or more RDF/OWL files
  IAO/       # one or more RDF/OWL files
  CCO/       # one or more RDF/OWL files
  Domain/    # the replaceable domain ontology
  Relations/ # optional relation ontology, such as RO
  Alignment/ # versioned Stage 2-to-OWL bridge
```

Supported serializations are RDF/XML (`.owl`, `.rdf`), Turtle, N-Triples, N3,
and JSON-LD. Classes and properties must expose exact `rdfs:label`,
`skos:prefLabel`, or `skos:altLabel` values matching frozen Stage 2 frame types,
entity surface forms, and candidate relation names. This keeps mappings in the
ontologies rather than in Python.

BFO, IAO, and CCO may also be placed directly under `ontologies/` when their
filenames identify them (for example `bfo.owl`, `iao.owl`, and
`CommonCoreOntologiesMerged.ttl`). A replaceable domain ontology is still
required either under `Domain/`, in a flat filename containing `domain`, or via
the Stage 3 `--domain-ontology` argument.

Stage 3 Neptune publication uses the SHACL shapes in `Alignment/` and emits
UTF-8 N-Quads:

```powershell
python stage3_ontology_mapper.py stage2_output.json `
  --neptune-output stage3-publication.nq `
  --quarantine-output stage3-quarantine.ttl `
  --validation-report stage3-validation.txt
```
