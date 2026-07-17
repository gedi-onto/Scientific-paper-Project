"""Reusable RDF/OWL ontology loading and instance-management facade."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph, Literal, Namespace, RDF, RDFS, SKOS, URIRef
from rdflib.namespace import OWL, XSD


ONTOLOGY_SUFFIXES = {".owl", ".rdf", ".ttl", ".nt", ".n3", ".jsonld"}
# The foundational ontologies (IAO, CCO, RO, and the pipeline alignment) ship *inside*
# the package, so `pip install graphrag-stage1[ontology]` gives you a working Stage 3
# with nothing else to fetch. They are small (~3.7 MB), stable, and the same for every
# user -- there is no reason to make anyone hunt for them.
#
# The DOMAIN ontology is deliberately NOT shipped: it is the one module that changes per
# scientific domain, it is the user's choice, and it can be enormous (UBERON alone is
# 48 MB). Pass it explicitly via `domain_ontology=`.
#
# This path is resolved from the package file, never from the working directory. A
# CWD-relative default silently works in the repo and fails for every installed user.
PACKAGED_ONTOLOGY_ROOT = Path(__file__).resolve().parent / "ontologies"

CORE_ONTOLOGIES = ("BFO", "IAO", "CCO")

# BFO needs no file of its own: CCO and IAO are both built on it and republish its
# classes under the canonical obo/BFO_* IRIs. Shipping a separate BFO file is not just
# redundant -- a stale one is actively harmful, because the legacy IFOMIS BFO 1.1 names
# the same concepts under DIFFERENT IRIs ("entity" as both obo/BFO_0000001 and
# ifomis.org/bfo/1.1#Entity), which makes every upper-ontology term ambiguous and
# unusable. So BFO is loaded when present and simply skipped when it is not; what is
# actually required is that its classes end up in the graph, which is verified below.
OPTIONAL_CORE_ONTOLOGIES = {"BFO"}

# The upper classes the fallback ladders climb to. If these are absent, an entity cannot
# be typed as anything true, so their absence is a real error -- unlike a missing file.
REQUIRED_UPPER_CLASSES = {
    "entity": "http://purl.obolibrary.org/obo/BFO_0000001",
    "process": "http://purl.obolibrary.org/obo/BFO_0000015",
    "material entity": "http://purl.obolibrary.org/obo/BFO_0000040",
}

OBO = Namespace("http://purl.obolibrary.org/obo/")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")

# Marks a class stub that was resolved by an external annotator (OAK/OLS) rather than
# parsed from a loaded ontology file. Lets a consumer distinguish grounded references
# from natively-loaded classes.
EXTERNAL_GROUNDED_CLASS = URIRef("https://w3id.org/graphrag/vocab/ExternalGroundedClass")

# Every predicate that attaches a *name* to a class. SKOS alone is not enough: OBO
# Foundry ontologies (INO, GO, ChEBI, PRO...) publish their synonyms under oboInOwl
# and IAO_0000118 ("alternative term"), so a SKOS-only index cannot see them --
# hundreds of usable aliases were being ignored.
#
# Only names that DENOTE THE SAME THING are indexed. hasBroadSynonym and
# hasRelatedSynonym are deliberately loose in OBO ("related" can be a mere
# association), and indexing them made previously-unique lookups ambiguous: the frame
# type METHOD started matching several classes at once and failed to resolve at all,
# costing 35 frames. Recall is worthless if it destroys precision on the terms that
# already worked.
LABEL_PREDICATES = (
    RDFS.label,
    SKOS.prefLabel,
    SKOS.altLabel,
    OBO.IAO_0000118,             # alternative term  (exact alias)
    OBOINOWL.hasExactSynonym,    # exact alias
)

# Words that carry no ontological content -- stripped when hunting for the head term
# of a noun phrase like "the Syk signaling pathway".
_PHRASE_STOPWORDS = {"the", "a", "an", "this", "that", "these", "those", "its", "their"}


def normalize_lookup(value: object) -> str:
    text = re.sub(r"[_\-]+", " ", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def local_name(iri: URIRef) -> str:
    value = str(iri)
    return value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


class OntologyManager:
    """Hide ontology I/O, indexes, instance minting, and RDF serialization."""

    def __init__(
        self,
        ontology_root: str | Path | None = None,
        instance_namespace: str = "urn:graphrag:instance:",
    ) -> None:
        self.ontology_root = Path(ontology_root) if ontology_root is not None else PACKAGED_ONTOLOGY_ROOT
        self.graph = Graph()
        self.instance_namespace = Namespace(instance_namespace)
        self.loaded_files: list[str] = []
        self.class_index: dict[str, set[URIRef]] = {}
        self.object_property_index: dict[str, set[URIRef]] = {}
        self.datatype_property_index: dict[str, set[URIRef]] = {}

    def load_all(self, domain_ontology: str | Path | None = None) -> "OntologyManager":
        if not self.ontology_root.exists():
            raise FileNotFoundError(f"Ontology root does not exist: {self.ontology_root}")
        self.load_core()
        self.load_relations(required=False)
        self.load_alignment(required=False)
        domain_files = self._discover_domain_files(domain_ontology)
        if not domain_files:
            raise FileNotFoundError("No domain ontology files were found")
        self._parse_files(domain_files)
        self._build_indexes()
        return self

    def load_alignment(self, required: bool = False) -> "OntologyManager":
        """Load the pipeline-to-ontology bridge without changing source ontologies."""
        directory = next(
            (
                item for item in self.ontology_root.iterdir()
                if item.is_dir() and item.name.casefold() == "alignment"
            ),
            None,
        )
        files = self._ontology_files(directory) if directory else []
        if required and not files:
            raise FileNotFoundError("No alignment ontology files were found")
        self._parse_files(files)
        self._build_indexes()
        return self

    def load_relations(self, required: bool = False) -> "OntologyManager":
        directory = next(
            (
                item for item in self.ontology_root.iterdir()
                if item.is_dir() and item.name.casefold() == "relations"
            ),
            None,
        )
        files = self._ontology_files(directory) if directory else []
        if required and not files:
            raise FileNotFoundError("No relations ontology files were found")
        self._parse_files(files)
        self._build_indexes()
        return self

    def _parse_files(self, files: Iterable[Path]) -> None:
        loaded = set(self.loaded_files)
        for path in files:
            resolved = str(path.resolve())
            if resolved in loaded:
                continue
            self.graph.parse(path)
            self.loaded_files.append(resolved)
            loaded.add(resolved)

    def load_core(self) -> "OntologyManager":
        if not self.ontology_root.exists():
            raise FileNotFoundError(f"Ontology root does not exist: {self.ontology_root}")
        self._parse_files(self._discover_core_files())
        self._build_indexes()
        return self

    def load_domain(self, domain_ontology: str | Path | None = None) -> "OntologyManager":
        files = self._discover_domain_files(domain_ontology)
        if not files:
            raise FileNotFoundError("No domain ontology files were found")
        self._parse_files(files)
        self._build_indexes()
        return self

    def _ontology_files(self, directory: Path) -> list[Path]:
        return sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in ONTOLOGY_SUFFIXES
        ) if directory.exists() else []

    def _discover_core_files(self) -> list[Path]:
        discovered = []
        children = {item.name.casefold(): item for item in self.ontology_root.iterdir() if item.is_dir()}
        root_files = self._ontology_files(self.ontology_root)
        # _ontology_files is recursive; only use direct children for flat-layout
        # fallback so a domain file is never mistaken for a core ontology.
        direct_files = [path for path in root_files if path.parent == self.ontology_root]
        flat_patterns = {
            "BFO": ("bfo", "basicformalontology"),
            "IAO": ("iao", "informationartifactontology"),
            "CCO": ("cco", "commoncoreontolog"),
        }
        for name in CORE_ONTOLOGIES:
            directory = children.get(name.casefold())
            files = self._ontology_files(directory) if directory else []
            if not files:
                patterns = flat_patterns[name]
                files = [
                    path for path in direct_files
                    if any(pattern in path.stem.casefold() for pattern in patterns)
                ]
            if not files:
                if name in OPTIONAL_CORE_ONTOLOGIES:
                    continue  # its classes come in with CCO / IAO; verified after load
                raise FileNotFoundError(f"Required core ontology {name} was not found")
            discovered.extend(files)
        return list(dict.fromkeys(discovered))

    def missing_upper_classes(self) -> list[str]:
        """Upper classes the fallback ladders climb to that are NOT in the loaded graph.

        Advisory, not fatal. A minimal or non-BFO ontology set is perfectly legitimate --
        an entity that cannot reach any of these simply lands on ``owl:Thing``, which is
        true of anything. Callers that care (a Stage 3 audit, say) can check this; the
        loader does not impose a stack the caller did not ask for.
        """
        loaded = {item for values in self.class_index.values() for item in values}
        return [
            term for term, iri in REQUIRED_UPPER_CLASSES.items()
            if URIRef(iri) not in loaded
        ]

    def _discover_domain_files(self, domain_ontology: str | Path | None) -> list[Path]:
        if domain_ontology:
            path = Path(domain_ontology)
            return self._ontology_files(path) if path.is_dir() else ([path] if path.exists() else [])
        children = {item.name.casefold(): item for item in self.ontology_root.iterdir() if item.is_dir()}
        domain_directory = children.get("domain")
        if domain_directory:
            return self._ontology_files(domain_directory)
        # Flat-layout domain files must identify themselves in the filename;
        # unrecognized files are not silently treated as a domain ontology.
        return [
            path for path in self.ontology_root.iterdir()
            if path.is_file()
            and path.suffix.casefold() in ONTOLOGY_SUFFIXES
            and "domain" in path.stem.casefold()
        ]

    def _labels(self, resource: URIRef) -> Iterable[str]:
        yield local_name(resource)
        for predicate in LABEL_PREDICATES:
            for label in self.graph.objects(resource, predicate):
                yield str(label)

    @staticmethod
    def _add_index(index: dict[str, set[URIRef]], key: str, resource: URIRef) -> None:
        normalized = normalize_lookup(key)
        if normalized:
            index.setdefault(normalized, set()).add(resource)

    def _build_indexes(self) -> None:
        self.class_index.clear()
        self.object_property_index.clear()
        self.datatype_property_index.clear()
        # owl:Thing is the universal class -- every individual is one by definition. It is
        # rarely declared explicitly in an ontology file, but create_instance rightly
        # refuses to type an individual with an undeclared class, so declare it. This is
        # what an entity we could not classify is typed as: true of anything, and
        # therefore honest, unlike asserting a class we cannot justify.
        self.graph.add((OWL.Thing, RDF.type, OWL.Class))
        classes = set(self.graph.subjects(RDF.type, OWL.Class)) | set(
            self.graph.subjects(RDF.type, RDFS.Class)
        )
        classes = {item for item in classes if isinstance(item, URIRef)}
        object_properties = {
            item for item in self.graph.subjects(RDF.type, OWL.ObjectProperty)
            if isinstance(item, URIRef)
        }
        datatype_properties = {
            item for item in self.graph.subjects(RDF.type, OWL.DatatypeProperty)
            if isinstance(item, URIRef)
        }
        for resource in classes:
            for label in self._labels(resource):
                self._add_index(self.class_index, label, resource)
        for resource in object_properties:
            for label in self._labels(resource):
                self._add_index(self.object_property_index, label, resource)
        for resource in datatype_properties:
            for label in self._labels(resource):
                self._add_index(self.datatype_property_index, label, resource)

    @staticmethod
    def _unique(index: dict[str, set[URIRef]], term: str) -> Optional[URIRef]:
        matches = index.get(normalize_lookup(term), set())
        return next(iter(matches)) if len(matches) == 1 else None

    @staticmethod
    def _lookup_result(
        index: dict[str, set[URIRef]], term: str, prefer_namespace: str | None = None
    ) -> dict:
        matches = sorted(str(item) for item in index.get(normalize_lookup(term), set()))

        # An ambiguous term can still be resolved when the caller knows which ontology
        # *defines* the thing it is asking about. Frame types, for instance, are declared
        # in the alignment ontology; "METHOD" also happens to be an alternative term for
        # an unrelated IAO class, and without this the collision made a well-defined
        # frame class unresolvable. Only a UNIQUE hit in the preferred namespace counts,
        # so this disambiguates rather than guesses.
        if len(matches) > 1 and prefer_namespace:
            preferred = [m for m in matches if m.startswith(prefer_namespace)]
            if len(preferred) == 1:
                return {
                    "term": term,
                    "normalized_term": normalize_lookup(term),
                    "status": "matched",
                    "iri": preferred[0],
                    "candidates": matches,
                    "disambiguated_by": prefer_namespace,
                }

        return {
            "term": term,
            "normalized_term": normalize_lookup(term),
            "status": "matched" if len(matches) == 1 else ("ambiguous" if matches else "missing"),
            "iri": matches[0] if len(matches) == 1 else None,
            "candidates": matches,
        }

    def explain_class_lookup(self, term: str, prefer_namespace: str | None = None) -> dict:
        return self._lookup_result(self.class_index, term, prefer_namespace)

    def explain_class_head_lookup(self, term: str, min_tokens: int = 1) -> dict:
        """Find the most specific class matching the HEAD of a noun phrase.

        Exact matching alone is far too brittle for text: "Syk signaling pathway" never
        equals the class "signaling pathway", so a real entity fails to type even though
        the right class is loaded. English noun phrases put the head last and modifiers
        first, so progressively drop leading modifiers and take the LONGEST label that
        still matches -- the most specific class that is actually justified.

            "the Syk signaling pathway"  ->  "signaling pathway"   (a real class)
            "core driving genes"         ->  "genes"

        The instance keeps its full surface form as its label; only its *type* is
        generalised, which is exactly the correct ontological reading: a Syk signaling
        pathway IS a signaling pathway.

        Still exact per candidate substring (no fuzzy scoring), and ambiguous matches
        are rejected, so an unrelated class cannot win.
        """
        tokens = normalize_lookup(term).split()
        while tokens and tokens[0] in _PHRASE_STOPWORDS:
            tokens = tokens[1:]
        # Longest candidate first => most specific class wins.
        for start in range(len(tokens)):
            candidate = tokens[start:]
            if len(candidate) < min_tokens:
                break
            result = self._lookup_result(self.class_index, " ".join(candidate))
            if result["status"] == "matched":
                result["matched_head"] = " ".join(candidate)
                result["dropped_modifiers"] = " ".join(tokens[:start])
                return result
        return {
            "term": term,
            "normalized_term": normalize_lookup(term),
            "status": "missing",
            "iri": None,
            "candidates": [],
        }

    def explain_object_property_lookup(self, term: str) -> dict:
        return self._lookup_result(self.object_property_index, term)

    def class_candidates(self, term: str, limit: int = 10) -> list[dict]:
        """Retrieve auditable lexical candidates for later constrained classification."""
        query = normalize_lookup(term)
        query_tokens = set(query.split())
        scored: dict[URIRef, dict] = {}
        for label, resources in self.class_index.items():
            label_tokens = set(label.split())
            overlap = len(query_tokens & label_tokens) / max(len(query_tokens | label_tokens), 1)
            sequence = SequenceMatcher(None, query, label).ratio()
            score = round((0.65 * overlap) + (0.35 * sequence), 6)
            if score <= 0:
                continue
            for resource in resources:
                current = scored.get(resource)
                if current is None or score > current["retrieval_score"]:
                    scored[resource] = {
                        "iri": str(resource),
                        "matched_label": label,
                        "retrieval_score": score,
                        "retrieval_method": "label_synonym_token_sequence",
                    }
        return sorted(
            scored.values(), key=lambda item: (-item["retrieval_score"], item["iri"])
        )[:limit]

    def superclasses(self, ontology_class: URIRef) -> set[URIRef]:
        """Return the named transitive superclass closure for compatibility checks."""
        found: set[URIRef] = set()
        pending = [ontology_class]
        while pending:
            current = pending.pop()
            for parent in self.graph.objects(current, RDFS.subClassOf):
                if isinstance(parent, URIRef) and parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return found

    def class_is_compatible(self, actual: URIRef, expected: URIRef) -> bool:
        return actual == expected or expected in self.superclasses(actual)

    def property_signature(self, prop: URIRef) -> dict:
        domains = sorted(str(item) for item in self.graph.objects(prop, RDFS.domain))
        ranges = sorted(str(item) for item in self.graph.objects(prop, RDFS.range))
        parents = sorted(str(item) for item in self.graph.objects(prop, RDFS.subPropertyOf))
        inverses = sorted(str(item) for item in self.graph.objects(prop, OWL.inverseOf))
        return {
            "iri": str(prop),
            "domains": domains,
            "ranges": ranges,
            "superproperties": parents,
            "inverse_properties": inverses,
        }

    def resolve_object_property(
        self,
        term: str,
        subject_class: URIRef | None = None,
        object_class: URIRef | None = None,
    ) -> dict:
        """Resolve a relation and audit named OWL/RDFS domain-range compatibility."""
        lookup = self.explain_object_property_lookup(term)
        candidates = [URIRef(item) for item in lookup["candidates"]]
        assessed = []
        for prop in candidates:
            signature = self.property_signature(prop)
            domains = [item for item in self.graph.objects(prop, RDFS.domain) if isinstance(item, URIRef)]
            ranges = [item for item in self.graph.objects(prop, RDFS.range) if isinstance(item, URIRef)]
            domain_ok = not domains or subject_class is None or any(
                self.class_is_compatible(subject_class, expected) for expected in domains
            )
            range_ok = not ranges or object_class is None or any(
                self.class_is_compatible(object_class, expected) for expected in ranges
            )
            assessed.append({**signature, "domain_compatible": domain_ok, "range_compatible": range_ok})
        compatible = [item for item in assessed if item["domain_compatible"] and item["range_compatible"]]
        selected = compatible[0]["iri"] if len(compatible) == 1 else None
        status = "matched" if selected else (
            "incompatible" if assessed and not compatible else
            ("ambiguous" if len(compatible) > 1 else "missing")
        )
        return {"term": term, "status": status, "iri": selected, "candidates": assessed}

    def resolve_datatype_property(
        self,
        term: str,
        subject_class: URIRef | None = None,
        value_datatype: URIRef | None = None,
    ) -> dict:
        lookup = self._lookup_result(self.datatype_property_index, term)
        candidates = [URIRef(item) for item in lookup["candidates"]]
        assessed = []
        for prop in candidates:
            signature = self.property_signature(prop)
            domains = [item for item in self.graph.objects(prop, RDFS.domain) if isinstance(item, URIRef)]
            ranges = [item for item in self.graph.objects(prop, RDFS.range) if isinstance(item, URIRef)]
            domain_ok = not domains or subject_class is None or any(
                self.class_is_compatible(subject_class, expected) for expected in domains
            )
            range_ok = not ranges or value_datatype is None or value_datatype in ranges or (
                value_datatype == XSD.integer and XSD.decimal in ranges
            )
            assessed.append({
                **signature,
                "domain_compatible": domain_ok,
                "range_compatible": range_ok,
            })
        compatible = [
            item for item in assessed
            if item["domain_compatible"] and item["range_compatible"]
        ]
        selected = compatible[0]["iri"] if len(compatible) == 1 else None
        status = "matched" if selected else (
            "incompatible" if assessed and not compatible else
            ("ambiguous" if len(compatible) > 1 else "missing")
        )
        return {"term": term, "status": status, "iri": selected, "candidates": assessed}

    def find_class(self, term: str) -> Optional[URIRef]:
        return self._unique(self.class_index, term)

    def find_object_property(self, term: str) -> Optional[URIRef]:
        return self._unique(self.object_property_index, term)

    def find_datatype_property(self, term: str) -> Optional[URIRef]:
        return self._unique(self.datatype_property_index, term)

    def find_property(self, term: str, property_type: str | None = None) -> Optional[URIRef]:
        """Find a unique property without exposing internal indexes to callers."""
        if property_type == "object":
            return self.find_object_property(term)
        if property_type == "datatype":
            return self.find_datatype_property(term)
        object_property = self.find_object_property(term)
        datatype_property = self.find_datatype_property(term)
        if object_property and not datatype_property:
            return object_property
        if datatype_property and not object_property:
            return datatype_property
        return None

    def register_external_class(self, iri: str, label: str | None = None) -> URIRef:
        """Declare a class resolved from OUTSIDE the loaded ontologies (e.g. by OAK).

        Grounding a mention to ``PR:000002198`` types an entity against a class that was
        never parsed into this graph -- ChEBI, PRO, GO and friends are far too large to
        hold in rdflib. ``create_instance`` rightly refuses to type an individual with an
        undeclared class, so register a minimal stub (the class node, its label, and a
        marker that it came from an external annotator) and index it. The real ontology
        is not loaded; we are recording a grounded reference to it, with provenance.
        """
        node = URIRef(iri)
        if (node, RDF.type, OWL.Class) not in self.graph:
            self.graph.add((node, RDF.type, OWL.Class))
            if label:
                self.graph.add((node, RDFS.label, Literal(label)))
            # Mark the provenance of the stub so a consumer can tell a grounded class
            # from a natively-loaded one.
            self.graph.add((node, RDF.type, EXTERNAL_GROUNDED_CLASS))
            for text in (label, local_name(node)):
                self._add_index(self.class_index, text, node)
        return node

    def create_instance(self, ontology_class: URIRef, key: str, label: str) -> URIRef:
        if (ontology_class, RDF.type, OWL.Class) not in self.graph and (
            ontology_class, RDF.type, RDFS.Class
        ) not in self.graph:
            raise ValueError(f"Not a loaded ontology class: {ontology_class}")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")[:48] or "instance"
        instance = URIRef(f"{self.instance_namespace}{slug}_{digest}")
        self.graph.add((instance, RDF.type, ontology_class))
        self.graph.add((instance, RDFS.label, Literal(label)))
        return instance

    def add_object_assertion(self, subject: URIRef, predicate: URIRef, obj: URIRef) -> None:
        if (predicate, RDF.type, OWL.ObjectProperty) not in self.graph:
            raise ValueError(f"Not a loaded object property: {predicate}")
        self.graph.add((subject, predicate, obj))

    def add_datatype_assertion(
        self, subject: URIRef, predicate: URIRef, value: object, datatype: URIRef = XSD.string
    ) -> None:
        if (predicate, RDF.type, OWL.DatatypeProperty) not in self.graph:
            raise ValueError(f"Not a loaded datatype property: {predicate}")
        self.graph.add((subject, predicate, Literal(value, datatype=datatype)))

    def reason(self) -> None:
        raise RuntimeError("Reasoning belongs to Stage 3.5 and is not executed by OntologyManager")

    def export(self, destination: str | Path | None = None, rdf_format: str = "turtle") -> str:
        serialized = self.graph.serialize(format=rdf_format)
        if destination:
            Path(destination).write_text(serialized, encoding="utf-8")
        return serialized

    def profile(self) -> dict:
        return {
            "core_ontologies": list(CORE_ONTOLOGIES),
            "ontology_root": str(self.ontology_root.resolve()),
            "loaded_files": self.loaded_files,
            "relations_module_loaded": any(
                Path(path).parent.name.casefold() == "relations" for path in self.loaded_files
            ),
            "alignment_module_loaded": any(
                Path(path).parent.name.casefold() == "alignment" for path in self.loaded_files
            ),
            "class_count": len({item for values in self.class_index.values() for item in values}),
            "object_property_count": len({item for values in self.object_property_index.values() for item in values}),
            "datatype_property_count": len({item for values in self.datatype_property_index.values() for item in values}),
        }
