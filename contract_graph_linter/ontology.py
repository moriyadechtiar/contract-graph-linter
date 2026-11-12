"""The ontology of a contract graph, and how to extend it.

The default ontology is the one introduced in the GRAPH-GRPO-LEX paper: four
node types and six edge types, each edge type connecting a clause to one kind
of node.

An extension adds node types, edge types, and allowed connections on top of the
default. It cannot remove or change anything in the default.

    {
      "node_types": {"OBLIGATION": {"name": "text"}, "SCHEDULE": {"name": "title"}},
      "edge_types": {"IMPOSES": [["CLAUSE", "OBLIGATION"]],
                     "REFERENCES": [["CLAUSE", "SCHEDULE"]]}
    }

For a node type, ``name`` is the attribute that holds the node's name (default
``"name"``; ``null`` when the type has no name to check) and ``id_prefix`` is an
optional prefix that its ids carry, as in ``"party:Acme"``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, Union

_DEFAULT_NODE_TYPES = {
    # type: (attributes that may hold the name, id prefixes)
    "CLAUSE": ((), ("clause",)),
    "PARTY": (("name", "label"), ("party",)),
    "DEFINED_TERM": (("name", "label"), ("term", "defined_term")),
    "VALUE": (("text", "name", "label"), ("value",)),
}
_DEFAULT_EDGE_TYPES = {
    "IS_PART_OF": {("CLAUSE", "CLAUSE")},
    "REFERENCES": {("CLAUSE", "CLAUSE")},
    "DEFINES": {("CLAUSE", "DEFINED_TERM")},
    "USES": {("CLAUSE", "DEFINED_TERM")},
    "MENTIONS_PARTY": {("CLAUSE", "PARTY")},
    "CONTAINS": {("CLAUSE", "VALUE")},
}


class Ontology:
    """Node types, edge types, and which node types each edge type may connect."""

    def __init__(self, node_types: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
                 edge_types: Mapping[str, FrozenSet[Tuple[str, str]]]):
        self._nodes = {k: (tuple(v[0]), tuple(v[1])) for k, v in node_types.items()}
        self._edges = {k: frozenset(v) for k, v in edge_types.items()}

    @property
    def node_types(self) -> Tuple[str, ...]:
        return tuple(self._nodes)

    @property
    def edge_types(self) -> Dict[str, FrozenSet[Tuple[str, str]]]:
        return dict(self._edges)

    def has_node_type(self, node_type: str) -> bool:
        return node_type in self._nodes

    def has_edge_type(self, edge_type: str) -> bool:
        return edge_type in self._edges

    def allows(self, edge_type: str, source_type: str, target_type: str) -> bool:
        return (source_type, target_type) in self._edges.get(edge_type, ())

    def connections(self, edge_type: str) -> str:
        """The allowed connections of an edge type, as text."""
        return " or ".join(f"{s} -> {t}" for s, t in sorted(self._edges.get(edge_type, ())))

    def name_attributes(self, node_type: str) -> Tuple[str, ...]:
        """Attributes that may hold the name of a node of this type. Empty when the type has no name to check."""
        return self._nodes.get(node_type, ((), ()))[0]

    def name_of(self, node_type: str, attrs: Mapping[str, Any]) -> str:
        for key in self.name_attributes(node_type):
            value = attrs.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    def type_for_id_prefix(self, prefix: str) -> Optional[str]:
        for node_type, (_, prefixes) in self._nodes.items():
            if prefix.lower() in prefixes:
                return node_type
        return None

    def extend(self, extension: Mapping[str, Any]) -> "Ontology":
        """A new ontology: this one plus the node types, edge types and connections in ``extension``."""
        if not isinstance(extension, Mapping):
            raise ValueError("An ontology extension is a JSON object with 'node_types' and 'edge_types'.")
        unknown = set(extension) - {"node_types", "edge_types"}
        if unknown:
            raise ValueError(f"Unknown key(s) in the ontology extension: {', '.join(sorted(unknown))}. "
                             "Use 'node_types' and 'edge_types'.")
        nodes = dict(self._nodes)
        for raw_name, spec in (extension.get("node_types") or {}).items():
            name = str(raw_name).strip().upper()
            if name in self._nodes:
                raise ValueError(f"Node type {name} is already in the ontology and cannot be redefined.")
            spec = spec or {}
            if not isinstance(spec, Mapping) or set(spec) - {"name", "id_prefix"}:
                raise ValueError(f"Node type {name}: use an object with optional 'name' and 'id_prefix'.")
            name_attr = spec.get("name", "name")
            prefix = spec.get("id_prefix")
            prefixes = () if prefix is None else tuple(str(p).lower().rstrip(":") for p in ([prefix] if isinstance(prefix, str) else prefix))
            for p in prefixes:
                if self.type_for_id_prefix(p) or any(p in nodes[t][1] for t in nodes):
                    raise ValueError(f"Node type {name}: the id prefix {p!r} is already used by another node type.")
            nodes[name] = ((str(name_attr),) if name_attr else (), prefixes)
        edges = {k: set(v) for k, v in self._edges.items()}
        for raw_name, pairs in (extension.get("edge_types") or {}).items():
            name = str(raw_name).strip().upper()
            if not isinstance(pairs, (list, tuple)) or not pairs or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pairs):
                raise ValueError(f"Edge type {name}: give a list of [source type, target type] pairs, "
                                 'for example [["CLAUSE", "OBLIGATION"]].')
            for source, target in pairs:
                source, target = str(source).strip().upper(), str(target).strip().upper()
                for t in (source, target):
                    if t not in nodes:
                        raise ValueError(f"Edge type {name}: node type {t} is not in the ontology. "
                                         "Add it under 'node_types' first.")
                edges.setdefault(name, set()).add((source, target))
        return Ontology(nodes, edges)


DEFAULT_ONTOLOGY = Ontology(_DEFAULT_NODE_TYPES, _DEFAULT_EDGE_TYPES)

OntologySource = Union[None, Ontology, Mapping[str, Any], str, Path]


def load_ontology(source: OntologySource = None) -> Ontology:
    """The default ontology, or the default extended by a dictionary or a JSON file."""
    if source is None:
        return DEFAULT_ONTOLOGY
    if isinstance(source, Ontology):
        return source
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8") as fh:
            source = json.load(fh)
    return DEFAULT_ONTOLOGY.extend(source)
