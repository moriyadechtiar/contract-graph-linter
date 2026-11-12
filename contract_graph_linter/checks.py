"""Lint checks: observations about a contract graph for a person to review.

Each check returns ``Finding`` records. A finding states what was observed and
which clauses to read. The linter does not decide whether a finding comes from
the contract itself or from how the graph was built; that is left to the
reviewer, who can go to the listed clauses in the contract.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import networkx as nx

from .graph import ContractGraph, GraphSource, load_graph
from .ontology import Ontology, OntologySource, load_ontology

SMALL_COMPONENT_MAX_NODES = 8
# Checks that need the whole contract. On the graph of a single clause they fire
# on ordinary structure (a clause normally uses terms that are defined elsewhere).
CONTRACT_LEVEL_CHECKS = frozenset({
    "undefined_term", "unused_definition", "missing_parent_clause",
    "small_component", "articulation_point",
})

ClauseText = Union[str, Mapping[str, str]]


@dataclass
class Finding:
    check: str
    message: str
    nodes: List[str] = field(default_factory=list)
    edge: Optional[Tuple[str, str, str]] = None
    clauses: List[str] = field(default_factory=list)   # clauses to read in the contract

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_term(label: Any) -> str:
    text = re.sub(r'["\u201c\u201d]', "", str(label if label is not None else ""))
    return text.strip().lower()


def _incoming(graph: ContractGraph, edge_type: str) -> Dict[str, List[str]]:
    table: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.edges_of_type(edge_type):
        table[edge.tgt].append(edge.src)
    return table


def _clauses_for(graph: ContractGraph, nodes: List[str]) -> List[str]:
    """The clause nodes among ``nodes``, plus the clauses linked to any other node in the list."""
    found = set()
    for node in nodes:
        if graph.node_type(node) == "CLAUSE":
            found.add(node)
        else:
            found.update(graph.linked_clauses(node))
    return sorted(found)


def check_records(graph: ContractGraph, ontology: Ontology) -> List[Finding]:
    """Node and edge records: unreadable records, an empty graph, repeated ids and edges, missing names, ids and types that disagree."""
    out = [Finding("invalid_node_record", message) for message in graph.invalid_nodes]
    out += [Finding("invalid_edge_record", message) for message in graph.invalid_edges]
    if not graph.nodes:
        out.append(Finding("empty_graph", "The graph has no nodes"))
    raw_nodes = graph.raw.get("nodes")
    raw_ids = [str(n.get("id")) for n in (raw_nodes if isinstance(raw_nodes, (list, tuple)) else []) if isinstance(n, dict) and n.get("id") is not None]
    for node_id, count in Counter(raw_ids).items():
        if count > 1:
            out.append(Finding("duplicate_node_id", f"Node id {node_id!r} is declared {count} times", [node_id]))
    for (src, etype, tgt), count in Counter((e.src, e.type, e.tgt) for e in graph.edges).items():
        if count > 1:
            out.append(Finding("duplicate_edge", f"{etype} edge {src} -> {tgt} appears {count} times", [src, tgt], (src, etype, tgt)))
    for node in graph.nodes.values():
        if not ontology.has_node_type(node.type):
            out.append(Finding("unknown_node_type", f"Node {node.id} has type {node.type or '(none)'!r}, which is not in the ontology", [node.id]))
            continue
        attributes = ontology.name_attributes(node.type)
        if attributes and not ontology.name_of(node.type, node.attrs):
            out.append(Finding("missing_name", f"{node.type} node {node.id} has no {attributes[0]}", [node.id]))
        prefix = node.id.split(":", 1)[0].lower() if ":" in node.id else None
        expected = ontology.type_for_id_prefix(prefix) if prefix else None
        if expected and expected != node.type:
            out.append(Finding("id_type_mismatch", f"Node {node.id} has the id prefix {prefix!r} but the type {node.type}", [node.id]))
    return out


def check_edges(graph: ContractGraph, ontology: Ontology) -> List[Finding]:
    """Edges: endpoints without a node record, types outside the ontology, a node linked to itself."""
    out = []
    for e in graph.edges:
        triple = (e.src, e.type, e.tgt)
        missing = [x for x in (e.src, e.tgt) if x not in graph.nodes]
        if missing:
            out.append(Finding("dangling_edge", f"{e.type} edge {e.src} -> {e.tgt}: {', '.join(missing)} has no node record", [e.src, e.tgt], triple))
            continue
        if e.src == e.tgt:
            out.append(Finding("self_reference", f"{e.type} edge links {e.src} to itself", [e.src], triple))
        actual = (graph.nodes[e.src].type, graph.nodes[e.tgt].type)
        if not ontology.has_edge_type(e.type):
            out.append(Finding("ontology_violation", f"Edge type {e.type!r} on {e.src} -> {e.tgt} is not in the ontology", [e.src, e.tgt], triple))
        elif not ontology.allows(e.type, *actual) and all(ontology.has_node_type(t) for t in actual):
            out.append(Finding("ontology_violation", f"{e.type} connects {ontology.connections(e.type)} in the ontology; "
                               f"here it connects {actual[0]} -> {actual[1]} ({e.src} -> {e.tgt})", [e.src, e.tgt], triple))
    return out


def check_definition_coverage(graph: ContractGraph) -> List[Finding]:
    """Terms that are used and not defined, and terms that are defined and not used."""
    defined_by = _incoming(graph, "DEFINES")
    used_by = _incoming(graph, "USES")
    out = []
    for term in graph.ids_of_type("DEFINED_TERM"):
        label = graph.nodes[term].label
        if used_by.get(term) and not defined_by.get(term):
            users = sorted(set(used_by[term]))
            out.append(Finding("undefined_term", f"Term {label!r} is used by {', '.join(users)} and no clause defines it", [term] + users))
        if defined_by.get(term) and not used_by.get(term):
            definers = sorted(set(defined_by[term]))
            out.append(Finding("unused_definition", f"Term {label!r} is defined by {', '.join(definers)} and no clause uses it", [term] + definers))
    return out


def check_duplicate_definitions(graph: ContractGraph) -> List[Finding]:
    out = []
    defined_by = _incoming(graph, "DEFINES")
    for term, clauses in defined_by.items():
        distinct = sorted(set(clauses))
        if len(distinct) > 1 and term in graph.nodes:
            out.append(Finding("multiply_defined_term", f"Term {graph.nodes[term].label!r} is defined by more than one clause: "
                               f"{', '.join(distinct)}", [term] + distinct, clauses=distinct))
    by_name: Dict[str, List[str]] = defaultdict(list)
    for term in graph.ids_of_type("DEFINED_TERM"):
        name = normalize_term(graph.nodes[term].label)
        if name:
            by_name[name].append(term)
    for name, ids in by_name.items():
        if len(ids) > 1:
            out.append(Finding("duplicate_term_name", f"{len(ids)} term nodes share the name {name!r}: {', '.join(sorted(ids))}", sorted(ids)))
    return out


def term_dependency_graph(graph: ContractGraph) -> nx.DiGraph:
    """Term A depends on term B when a clause that defines A also uses B."""
    defines: Dict[str, List[str]] = defaultdict(list)
    uses: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges_of_type("DEFINES"):
        defines[e.src].append(e.tgt)
    for e in graph.edges_of_type("USES"):
        uses[e.src].append(e.tgt)
    g = nx.DiGraph()
    g.add_nodes_from(graph.ids_of_type("DEFINED_TERM"))
    for clause, defined_terms in defines.items():
        for a in defined_terms:
            for b in uses.get(clause, []):
                if a != b:
                    g.add_edge(a, b, via=clause)
    return g


def _loops(pairs) -> List[Tuple[List[str], List[str]]]:
    """Groups of nodes that are linked in loops.

    Returns, for each group, one loop written from the group's smallest id, and
    every member of the group. A graph can hold a very large number of distinct
    loops through the same nodes; reporting each group once keeps the work and
    the output bounded.
    """
    edges = sorted({(s, t) for s, t in pairs if s != t})        # a node linked to itself is reported as self_reference
    g = nx.DiGraph()
    g.add_nodes_from(sorted({n for edge in edges for n in edge}))
    g.add_edges_from(edges)                                    # sorted, so the result does not depend on record order
    out = []
    for component in nx.strongly_connected_components(g):
        if len(component) < 2:
            continue
        sub, start = g.subgraph(sorted(component)), min(component)
        returns = [nx.shortest_path(sub, successor, start) for successor in sorted(sub.successors(start))]
        loop = [start] + min(returns, key=lambda path: (len(path), path))[:-1]
        out.append((loop, sorted(component)))
    return sorted(out)


def _loop_text(loop: List[str], members: List[str], names=None, what: str = "clauses") -> str:
    show = [names.get(x, x) for x in loop] if names else loop
    text = " -> ".join(show + show[:1])
    if len(members) > len(loop):
        text += f" ({len(members)} {what} are linked by loops)"
    return text


def check_cycles(graph: ContractGraph) -> List[Finding]:
    out = []
    terms = term_dependency_graph(graph)
    names = {t: graph.nodes[t].label for t in terms if t in graph.nodes}
    for loop, members in _loops(terms.edges()):
        defining = sorted({terms[a][b]["via"] for a, b in terms.edges() if a in members and b in members})
        out.append(Finding("definition_cycle", "Terms are defined in terms of each other: " + _loop_text(loop, members, names, "terms"),
                           members, clauses=defining))
    for loop, members in _loops((e.src, e.tgt) for e in graph.edges_of_type("REFERENCES")):
        out.append(Finding("reference_cycle", "Clauses reference each other in a loop: " + _loop_text(loop, members), members))
    for loop, members in _loops((e.src, e.tgt) for e in graph.edges_of_type("IS_PART_OF")):
        out.append(Finding("hierarchy_cycle", "Clauses are part of each other in a loop: " + _loop_text(loop, members), members))
    return out


def check_hierarchy(graph: ContractGraph) -> List[Finding]:
    """Parents. Numbering is read from dotted ids only: clause 3.4 is expected under 3 (or ARTICLE 3)."""
    part_of = defaultdict(set)
    for e in graph.edges_of_type("IS_PART_OF"):
        if e.src != e.tgt:
            part_of[e.src].add(e.tgt)
    out = []
    for clause, parents in part_of.items():
        if len(parents) > 1:
            out.append(Finding("multiple_parents", f"Clause {clause} is part of more than one clause: {', '.join(sorted(parents))}",
                               [clause] + sorted(parents)))
    for clause in graph.ids_of_type("CLAUSE"):
        if "." not in clause:
            continue
        prefix = clause.rsplit(".", 1)[0]
        parents = [p for p in (prefix, f"ARTICLE {prefix}") if graph.node_type(p) == "CLAUSE"]
        if not parents:
            out.append(Finding("missing_parent_clause", f"Clause {clause} is numbered under {prefix}, and the graph has no clause {prefix}", [clause]))
        elif not any(p in part_of.get(clause, ()) for p in parents):
            out.append(Finding("missing_is_part_of", f"Clause {clause} has no IS_PART_OF edge to {parents[0]}", [clause, parents[0]]))
    return out


def check_connectivity(graph: ContractGraph) -> List[Finding]:
    g = graph.to_networkx()
    out = []
    for node, degree in g.degree():
        if degree == 0:
            out.append(Finding("isolated_node", f"Node {node} has no edges", [node]))
    undirected = g.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    for comp in components[1:]:
        if 1 < len(comp) <= SMALL_COMPONENT_MAX_NODES:
            out.append(Finding("small_component", f"{len(comp)} nodes are not connected to the rest of the graph: "
                               f"{', '.join(sorted(comp))}", sorted(comp)))
    for node in sorted(nx.articulation_points(undirected)):
        out.append(Finding("articulation_point", f"Node {node} is the only connection between parts of the graph", [node]))
    return out


def _flat(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r'["\u201c\u201d\u2018\u2019\']', "", str(text or ""))).strip().lower()


def check_grounding(graph: ContractGraph, text: ClauseText, ontology: Ontology) -> List[Finding]:
    """Named nodes (parties, terms, values, and named types added by an extension) whose name does not appear in the clause text.

    ``text`` is the text of the clause, or a mapping from clause id to text. With
    a mapping, a node is looked for in the clauses linked to it.
    """
    out = []
    whole = _flat(text if isinstance(text, str) else " ".join(map(str, text.values())))
    for node in graph.nodes.values():
        label = ontology.name_of(node.type, node.attrs)
        name = _flat(label)
        if not name:
            continue
        if isinstance(text, str):
            haystack = whole
        else:
            linked = _clauses_for(graph, [node.id])
            haystack = _flat(" ".join(str(text[c]) for c in linked if c in text)) or whole
        if name not in haystack:
            out.append(Finding("ungrounded_node", f"{node.type} {label!r} does not appear in the clause text", [node.id]))
    return out


STRUCTURE_CHECKS = (check_definition_coverage, check_duplicate_definitions, check_cycles, check_hierarchy, check_connectivity)


def run_checks(source: GraphSource, clause_level: bool = False, text: Optional[ClauseText] = None,
               ontology: OntologySource = None) -> List[Finding]:
    """Run every check and return the findings, ordered by check name.

    clause_level  the graph covers a single clause; skip checks that need the whole contract
    text          the clause text, or a mapping from clause id to text; enables ``ungrounded_node``
    ontology      an extension of the default ontology: a dictionary or the path of a JSON file
    """
    graph = load_graph(source)
    onto = load_ontology(ontology)
    findings: List[Finding] = check_records(graph, onto) + check_edges(graph, onto)
    for check in STRUCTURE_CHECKS:
        findings.extend(check(graph))
    if text is not None:
        findings.extend(check_grounding(graph, text, onto))
    if clause_level:
        findings = [f for f in findings if f.check not in CONTRACT_LEVEL_CHECKS]
    for f in findings:
        if not f.clauses:
            f.clauses = _clauses_for(graph, f.nodes)
    findings.sort(key=lambda f: (f.check, f.nodes))
    return findings
