"""Reference-free quality scores in [0, 1] (higher is better).

These are the reference-free checks stored in the ContractGraphEval benchmark
under ``reference_free_judges``. They score a predicted graph on its own,
without a gold graph.

  unused_definitions     share of defined terms that the graph defines
  undefined_references   share of REFERENCES targets that are declared nodes of a
                         type the ontology allows a reference to point to
  duplicate_definitions  1 minus the share of defined-term names that repeat
  orphan_nodes           share of nodes that have at least one edge
  isolated_components    1 minus the share of small components apart from the largest
  domain_range_validity  share of edges whose endpoint types fit the edge type
  ontology_compliance    edge typing plus clause hierarchy (parent present,
                         IS_PART_OF edge present)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .checks import normalize_term
from .graph import ContractGraph, load_graph
from .ontology import Ontology, OntologySource, load_ontology

SMALL_COMPONENT_MAX_NODES = 8
SCORE_NAMES = ("unused_definitions", "undefined_references", "duplicate_definitions", "orphan_nodes",
               "isolated_components", "domain_range_validity", "ontology_compliance")

def _unused_definitions(node_type, edges) -> float:
    terms = [nid for nid, t in node_type.items() if t == "DEFINED_TERM"]
    defined = {e.tgt for e in edges if e.type == "DEFINES"}
    return 1.0 - sum(nid not in defined for nid in terms) / max(1, len(terms))


def _undefined_references(node_type, edges, ontology: Ontology) -> float:
    allowed = {target for _, target in ontology.edge_types.get("REFERENCES", ())}
    targets = {e.tgt for e in edges if e.type == "REFERENCES"}
    missing = [t for t in targets if node_type.get(t) not in allowed]
    return 1.0 - len(missing) / max(1, len(targets))


def _duplicate_definitions(nodes, node_type) -> float:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for node in nodes:
        if node.type != "DEFINED_TERM":
            continue
        name = normalize_term(node.attrs.get("name") or node.attrs.get("label") or "")
        if name:
            by_name[name].append(node.id)
    duplicated = sum(1 for ids in by_name.values() if len(ids) > 1)
    total = sum(1 for t in node_type.values() if t == "DEFINED_TERM")
    return 1.0 - duplicated / max(1, total)


def _orphan_nodes(node_type, edges) -> float:
    connected = {e.src for e in edges} | {e.tgt for e in edges}
    return 1.0 - len(set(node_type) - connected) / max(1, len(node_type))


def _isolated_components(node_type, edges) -> float:
    adj = defaultdict(set)
    for e in edges:
        adj[e.src].add(e.tgt)
        adj[e.tgt].add(e.src)
    visited, components = set(), []
    for nid in node_type:
        if nid in visited:
            continue
        comp, stack = [], [nid]
        visited.add(nid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)
    if len(components) <= 1:
        return 1.0
    components.sort(key=len, reverse=True)          # the largest component is the main one, whatever the record order
    isolated = [c for c in components[1:] if len(c) <= SMALL_COMPONENT_MAX_NODES]
    return 1.0 - len(isolated) / max(1, len(components) - 1)


def _domain_range_validity(node_type, edges, ontology: Ontology) -> float:
    invalid = 0
    for e in edges:
        s, t = node_type.get(e.src), node_type.get(e.tgt)
        if not s or not t:
            continue
        if not ontology.allows(e.type, s, t):
            invalid += 1
    return 1.0 - invalid / max(1, len(edges))


def _ontology_compliance(node_type, edges, ontology: Ontology) -> float:
    invalid, part_of = 0, set()
    for e in edges:
        s, t = e.src, e.tgt
        if node_type.get(s) is None or node_type.get(t) is None or not ontology.allows(e.type, node_type[s], node_type[t]):
            invalid += 1
        if e.type == "IS_PART_OF":
            part_of.add((s, t))
    with_parent = missing_parent = missing_edge = 0
    for nid, t in node_type.items():
        if t != "CLAUSE" or "." not in nid:
            continue
        with_parent += 1
        prefix = nid.rsplit(".", 1)[0]
        parents = [p for p in (prefix, f"ARTICLE {prefix}") if node_type.get(p) == "CLAUSE"]
        if not parents:
            missing_parent += 1
        elif not any((nid, p) in part_of for p in parents):
            missing_edge += 1
    expected = len(edges) + 2 * with_parent
    return max(0.0, min(1.0, 1.0 - (invalid + missing_parent + missing_edge) / max(1, expected)))


def reference_free_scores(graph: Any, ontology: OntologySource = None) -> Dict[str, float]:
    """Score a graph: a dictionary ({"nodes": [...], "edges": [...]}) or a loaded ContractGraph.

    The scores describe a graph that could be read in full. A graph that is
    missing, has no nodes, or holds a node or edge record that cannot be read
    (a node with no id, an edge with no source or target) scores 0.0 on every
    check, so a partly unreadable graph never looks sound. ``lint`` says which
    records are affected. The scores do not depend on the order of the records. ``ontology`` is an optional extension of
    the default ontology.
    """
    if not isinstance(graph, (dict, ContractGraph)):
        return {name: 0.0 for name in SCORE_NAMES}
    onto = load_ontology(ontology)
    parsed = load_graph(graph)
    nodes, edges = list(parsed.nodes.values()), parsed.edges
    node_type = {node.id: node.type for node in nodes}
    if not node_type or parsed.invalid_nodes or parsed.invalid_edges:      # nothing to assess, or not assessable as a whole
        return {name: 0.0 for name in SCORE_NAMES}
    return {
        "unused_definitions": _unused_definitions(node_type, edges),
        "undefined_references": _undefined_references(node_type, edges, onto),
        "duplicate_definitions": _duplicate_definitions(nodes, node_type),
        "orphan_nodes": _orphan_nodes(node_type, edges),
        "isolated_components": _isolated_components(node_type, edges),
        "domain_range_validity": _domain_range_validity(node_type, edges, onto),
        "ontology_compliance": _ontology_compliance(node_type, edges, onto),
    }
