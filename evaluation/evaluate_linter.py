#!/usr/bin/env python3
"""Evaluate contract-graph-linter on real contract graphs with planted defects.

    python evaluation/evaluate_linter.py                      # downloads the public benchmark file
    python evaluation/evaluate_linter.py --graphs FILE.jsonl  # or use a local copy
    python evaluation/evaluate_linter.py --write evaluation/RESULTS.md

Method. Real graphs are taken from the ContractGraphEval benchmark: the
reference graph of each clause, and one graph per contract assembled from its
clause graphs. One change at a time is then made to a graph:

  planted defects   a change with a known expected finding, such as removing the
                    definition of a term that is still used
  controls          a change that does not alter what the graph says, such as
                    shuffling its records; nothing new should be reported
  unusable input    output that is not a usable graph; it must never pass quietly

A planted defect counts as detected when the linter reports the expected check,
on the changed node or edge, and that finding was not present before the change.

Each JSON Lines record needs a contract id, a clause with its text, and a graph
(fields: contract_id, clause {id, text}, gold_graph).
"""
import argparse
import collections
import copy
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contract_graph_linter import GraphReadError, lint, reference_free_scores  # noqa: E402

BENCHMARK_URL = "https://raw.githubusercontent.com/moriyadechtiar/ensemble-graph-eval/main/benchmark/ContractGraphEval_v1.0.jsonl"
STRUCTURE_ONLY = {"articulation_point", "small_component"}     # observations a legitimate addition may change


# ----------------------------------------------------------------- data

def read_records(path):
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = urllib.request.urlopen(BENCHMARK_URL, timeout=60).read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def clause_graphs(records):
    """(graph, clause text, focal clause id) for every record whose focal clause is a node of its graph."""
    out = []
    for r in records:
        graph, clause = r["gold_graph"], r.get("clause") or {}
        focal = str(clause.get("id"))
        if any(str(n.get("id")) == focal for n in graph["nodes"]):
            out.append((graph, clause.get("text", ""), focal))
    return out


def contract_graphs(records):
    """One graph per contract: the union of its clause graphs, with the text of each clause."""
    by_contract = collections.defaultdict(list)
    for r in records:
        by_contract[r["contract_id"]].append(r)
    out = []
    for contract_id, rows in by_contract.items():
        nodes, edges, seen, texts = {}, [], set(), {}
        for r in rows:
            clause = r.get("clause") or {}
            texts[str(clause.get("id"))] = clause.get("text", "")
            for n in r["gold_graph"]["nodes"]:
                nodes.setdefault(str(n["id"]), n)
            for e in r["gold_graph"]["edges"]:
                key = (str(e["src"]), e["type"], str(e["tgt"]))
                if key not in seen:
                    seen.add(key)
                    edges.append(e)
        out.append(({"contract_id": contract_id, "nodes": list(nodes.values()), "edges": edges}, texts))
    return out


# ----------------------------------------------------------------- helpers

def ids(g, node_type=None):
    return [str(n["id"]) for n in g["nodes"] if node_type is None or n.get("node_type") == node_type]


def edges_of(g, edge_type):
    return [e for e in g["edges"] if e["type"] == edge_type]


def finding_keys(report):
    return collections.Counter((f.check, tuple(sorted(f.nodes)), f.message if not f.nodes else "") for f in report.findings)


def new_findings(before, after):
    return list((finding_keys(after) - finding_keys(before)).elements())


# ----------------------------------------------------------------- planted defects
# Each returns (changed graph, ids the expected finding should mention) or None when the graph offers no place for it.

def duplicate_node(g, rng):
    n = rng.choice(g["nodes"]); g["nodes"].append(copy.deepcopy(n)); return g, [str(n["id"])]

def duplicate_edge(g, rng):
    if not g["edges"]: return None
    e = rng.choice(g["edges"]); g["edges"].append(copy.deepcopy(e)); return g, [str(e["src"]), str(e["tgt"])]

def delete_node_record(g, rng):
    used = {str(e["src"]) for e in g["edges"]} | {str(e["tgt"]) for e in g["edges"]}
    pool = [i for i in ids(g) if i in used]
    if not pool: return None
    victim = rng.choice(pool); g["nodes"] = [n for n in g["nodes"] if str(n["id"]) != victim]; return g, [victim]

def unknown_node_type(g, rng):
    n = rng.choice(g["nodes"]); n["node_type"] = "OBLIGATION"; return g, [str(n["id"])]

def unknown_edge_type(g, rng):
    if not g["edges"]: return None
    e = rng.choice(g["edges"]); e["type"] = "QUALIFIES"; return g, [str(e["src"]), str(e["tgt"])]

def wrong_endpoint_type(g, rng):
    swap = {"USES": "MENTIONS_PARTY", "DEFINES": "CONTAINS", "MENTIONS_PARTY": "USES", "CONTAINS": "MENTIONS_PARTY"}
    pool = [e for e in g["edges"] if e["type"] in swap]
    if not pool: return None
    e = rng.choice(pool); e["type"] = swap[e["type"]]; return g, [str(e["src"]), str(e["tgt"])]

def blank_name(g, rng):
    pool = [n for n in g["nodes"] if n.get("node_type") in ("PARTY", "DEFINED_TERM", "VALUE")]
    if not pool: return None
    n = rng.choice(pool)
    for key in ("name", "label", "text"): n.pop(key, None)
    return g, [str(n["id"])]

def wrong_id_prefix(g, rng):
    pool = [n for n in g["nodes"] if n.get("node_type") == "DEFINED_TERM" and str(n["id"]).startswith("term:")]
    if not pool: return None
    n = rng.choice(pool); old = str(n["id"]); new = "party:" + old.split(":", 1)[1]
    if new in ids(g): return None
    n["id"] = new
    for e in g["edges"]:
        if str(e["src"]) == old: e["src"] = new
        if str(e["tgt"]) == old: e["tgt"] = new
    return g, [new]

def drop_node_id(g, rng):
    rng.choice(g["nodes"]).pop("id"); return g, []

def drop_edge_target(g, rng):
    if not g["edges"]: return None
    rng.choice(g["edges"]).pop("tgt"); return g, []

def add_undefined_term(g, rng):
    clauses = ids(g, "CLAUSE")
    g["nodes"].append({"id": "term:Planted Term", "node_type": "DEFINED_TERM", "name": "Planted Term"})
    g["edges"].append({"src": rng.choice(clauses), "tgt": "term:Planted Term", "type": "USES"}); return g, ["term:Planted Term"]

def remove_definition(g, rng):
    used = {str(e["tgt"]) for e in edges_of(g, "USES")}
    pool = sorted({str(e["tgt"]) for e in edges_of(g, "DEFINES")} & used)
    if not pool: return None
    term = rng.choice(pool); g["edges"] = [e for e in g["edges"] if not (e["type"] == "DEFINES" and str(e["tgt"]) == term)]; return g, [term]

def add_unused_definition(g, rng):
    g["nodes"].append({"id": "term:Planted Term", "node_type": "DEFINED_TERM", "name": "Planted Term"})
    g["edges"].append({"src": rng.choice(ids(g, "CLAUSE")), "tgt": "term:Planted Term", "type": "DEFINES"}); return g, ["term:Planted Term"]

def remove_uses(g, rng):
    defined = {str(e["tgt"]) for e in edges_of(g, "DEFINES")}
    pool = sorted({str(e["tgt"]) for e in edges_of(g, "USES")} & defined)
    if not pool: return None
    term = rng.choice(pool); g["edges"] = [e for e in g["edges"] if not (e["type"] == "USES" and str(e["tgt"]) == term)]; return g, [term]

def second_definition(g, rng):
    definers = collections.defaultdict(set)
    for e in edges_of(g, "DEFINES"): definers[str(e["tgt"])].add(str(e["src"]))
    pool = sorted(t for t, c in definers.items() if len(c) == 1)
    if not pool: return None
    term = rng.choice(pool); others = [c for c in ids(g, "CLAUSE") if c not in definers[term]]
    g["edges"].append({"src": rng.choice(others), "tgt": term, "type": "DEFINES"}); return g, [term]

def circular_definition(g, rng):
    clauses = ids(g, "CLAUSE")
    if len(clauses) < 2: return None
    c1, c2 = rng.sample(clauses, 2)
    g["nodes"] += [{"id": "term:Planted A", "node_type": "DEFINED_TERM", "name": "Planted A"}, {"id": "term:Planted B", "node_type": "DEFINED_TERM", "name": "Planted B"}]
    g["edges"] += [{"src": c1, "tgt": "term:Planted A", "type": "DEFINES"}, {"src": c1, "tgt": "term:Planted B", "type": "USES"},
                   {"src": c2, "tgt": "term:Planted B", "type": "DEFINES"}, {"src": c2, "tgt": "term:Planted A", "type": "USES"}]
    return g, ["term:Planted A", "term:Planted B"]

def reference_loop(g, rng):
    clauses = ids(g, "CLAUSE"); existing = {(str(e["src"]), str(e["tgt"])) for e in edges_of(g, "REFERENCES")}
    pairs = [(a, b) for a in clauses for b in clauses if a < b and (a, b) not in existing and (b, a) not in existing]
    if not pairs: return None
    a, b = rng.choice(pairs); g["edges"] += [{"src": a, "tgt": b, "type": "REFERENCES"}, {"src": b, "tgt": a, "type": "REFERENCES"}]; return g, [a, b]

def self_reference(g, rng):
    c = rng.choice(ids(g, "CLAUSE")); g["edges"].append({"src": c, "tgt": c, "type": "REFERENCES"}); return g, [c]

def hierarchy_loop(g, rng):
    pool = [e for e in edges_of(g, "IS_PART_OF") if str(e["src"]) != str(e["tgt"])]
    if not pool: return None
    e = rng.choice(pool); g["edges"].append({"src": e["tgt"], "tgt": e["src"], "type": "IS_PART_OF"}); return g, [str(e["src"]), str(e["tgt"])]

def second_parent(g, rng):
    parents = collections.defaultdict(set)
    for e in edges_of(g, "IS_PART_OF"): parents[str(e["src"])].add(str(e["tgt"]))
    pool = sorted(c for c, p in parents.items() if len(p) == 1)
    if not pool: return None
    child = rng.choice(pool); others = [c for c in ids(g, "CLAUSE") if c != child and c not in parents[child]]
    if not others: return None
    g["edges"].append({"src": child, "tgt": rng.choice(others), "type": "IS_PART_OF"}); return g, [child]

def remove_part_of(g, rng):
    clauses = set(ids(g, "CLAUSE"))
    pool = [e for e in edges_of(g, "IS_PART_OF") if "." in str(e["src"]) and str(e["src"]).rsplit(".", 1)[0] == str(e["tgt"]) and str(e["tgt"]) in clauses]
    if not pool: return None
    e = rng.choice(pool); g["edges"].remove(e); return g, [str(e["src"])]

def parentless_clause(g, rng):
    g["nodes"].append({"id": "97.3", "node_type": "CLAUSE", "title": "97.3", "level": 2})
    g["edges"].append({"src": rng.choice(ids(g, "CLAUSE")[:-1]), "tgt": "97.3", "type": "REFERENCES"}); return g, ["97.3"]

def isolated_clause(g, rng):
    g["nodes"].append({"id": "planted-clause", "node_type": "CLAUSE", "title": "Planted", "level": 1}); return g, ["planted-clause"]

def invented_party(g, rng, focal=None):
    clause = focal or rng.choice(ids(g, "CLAUSE"))
    g["nodes"].append({"id": "party:Zyxgrantor", "node_type": "PARTY", "name": "Zyxgrantor"})
    g["edges"].append({"src": clause, "tgt": "party:Zyxgrantor", "type": "MENTIONS_PARTY"}); return g, ["party:Zyxgrantor"]

def altered_name(g, rng, text=""):
    flat = " ".join(str(text).lower().split())
    pool = [n for n in g["nodes"] if n.get("node_type") in ("PARTY", "DEFINED_TERM") and str(n.get("name", "")).strip()
            and " ".join(str(n["name"]).lower().replace('"', "").split()) in flat]
    if not pool: return None
    n = rng.choice(pool); n["name"] = "Zyx " + str(n["name"])[::-1]; return g, [str(n["id"])]


RECORD_DEFECTS = [   # (label, function, expected check)
    ("Node record repeated", duplicate_node, "duplicate_node_id"),
    ("Edge record repeated", duplicate_edge, "duplicate_edge"),
    ("Node record deleted, its edges kept", delete_node_record, "dangling_edge"),
    ("Node given a type outside the ontology", unknown_node_type, "unknown_node_type"),
    ("Edge given a type outside the ontology", unknown_edge_type, "ontology_violation"),
    ("Edge type swapped for one that points to another kind of node", wrong_endpoint_type, "ontology_violation"),
    ("Name removed from a party, term or value", blank_name, "missing_name"),
    ("Term renamed with a party id prefix", wrong_id_prefix, "id_type_mismatch"),
    ("Id removed from a node record", drop_node_id, "invalid_node_record"),
    ("Target removed from an edge record", drop_edge_target, "invalid_edge_record"),
]
CONTRACT_DEFECTS = [
    ("Term used with no definition (term added)", add_undefined_term, "undefined_term"),
    ("Definition removed from a term still in use", remove_definition, "undefined_term"),
    ("Term defined and never used (term added)", add_unused_definition, "unused_definition"),
    ("Every use removed from a defined term", remove_uses, "unused_definition"),
    ("Second definition added to a term", second_definition, "multiply_defined_term"),
    ("Two terms defined in terms of each other", circular_definition, "definition_cycle"),
    ("Two clauses made to reference each other", reference_loop, "reference_cycle"),
    ("Clause made to reference itself", self_reference, "self_reference"),
    ("Parent made part of its own child", hierarchy_loop, "hierarchy_cycle"),
    ("Second parent added to a clause", second_parent, "multiple_parents"),
    ("Link to the parent removed from a numbered clause", remove_part_of, "missing_is_part_of"),
    ("Numbered clause added with no parent clause", parentless_clause, "missing_parent_clause"),
    ("Clause added with no edges", isolated_clause, "isolated_node"),
]


def run_defect(label, fn, expected, graph, rng, **lint_kw):
    fn_kw = {k: lint_kw.pop(k) for k in ("focal", "fn_text") if k in lint_kw}
    if "fn_text" in fn_kw: fn_kw["text"] = fn_kw.pop("fn_text")
    result = fn(copy.deepcopy(graph), rng, **fn_kw)
    if result is None:
        return None
    changed, targets = result
    fresh = new_findings(lint(graph, **lint_kw), lint(changed, **lint_kw))
    hit = any(check == expected and (not targets or set(targets) & set(nodes)) for check, nodes, _ in fresh)
    return {"detected": hit, "other_new_findings": sum(1 for check, _, _ in fresh if check != expected)}


def run_pair(first, second, graph, rng):
    """Plant two different defects in the same graph. Both expected findings must appear."""
    (_, fn1, exp1), (_, fn2, exp2) = first, second
    r1 = fn1(copy.deepcopy(graph), rng)
    if r1 is None: return None
    middle = copy.deepcopy(r1[0])
    r2 = fn2(r1[0], rng)
    if r2 is None: return None
    changed, t1, t2 = r2[0], r1[1], r2[1]
    edge_set = lambda g: collections.Counter((str(e.get("src")), e.get("type"), str(e.get("tgt"))) for e in g["edges"])
    touched = {x for (a, _, b) in (edge_set(changed) - edge_set(middle)) + (edge_set(middle) - edge_set(changed)) for x in (a, b)}
    if touched & set(t1):
        return None                           # the second change altered the first planted defect itself; not a fair trial
    fresh = new_findings(lint(graph), lint(changed))
    seen = lambda expected, targets: any(c == expected and (not targets or set(targets) & set(n)) for c, n, _ in fresh)
    return seen(exp1, t1) and seen(exp2, t2)


# ----------------------------------------------------------------- controls

def shuffle_records(g, rng, text=None):
    rng.shuffle(g["nodes"]); rng.shuffle(g["edges"]); return g, set()

def alternative_keys(g, rng, text=None):
    for n in g["nodes"]: n["type"] = n.pop("node_type")
    for e in g["edges"]: e["source"], e["target"] = e.pop("src"), e.pop("tgt")
    return g, set()

def lower_case_types(g, rng, text=None):
    for n in g["nodes"]: n["node_type"] = str(n["node_type"]).lower()
    for e in g["edges"]: e["type"] = str(e["type"]).lower()
    return g, set()

def nested_attributes(g, rng, text=None):
    for n in g["nodes"]: n["attrs"] = {k: n.pop(k) for k in list(n) if k not in ("id", "node_type")}
    return g, set()

def _word_from(text, rng, taken=()):
    """A word the clause text really contains and no node already uses as a name. None when the text offers none."""
    words = sorted({w.strip(".,;:()[]\"'") for w in str(text).split()})
    words = [w for w in words if w.isalpha() and len(w) >= 5 and w.lower() not in taken]
    return rng.choice(words) if words else None

def _names(g):
    return {str(n.get("name", "")).lower() for n in g["nodes"]}

def add_named_party(g, rng, text=None):
    name = "Planted Party" if text is None else _word_from(text, rng, _names(g))
    if name is None: return None
    g["nodes"].append({"id": "party:planted", "node_type": "PARTY", "name": name})
    g["edges"].append({"src": rng.choice(ids(g, "CLAUSE")), "tgt": "party:planted", "type": "MENTIONS_PARTY"}); return g, STRUCTURE_ONLY

def add_second_node_for_a_clause(g, rng, text=None):
    c = rng.choice([n for n in g["nodes"] if n.get("node_type") == "CLAUSE"])
    g["nodes"].append({"id": "clause:planted", "node_type": "CLAUSE", "title": c.get("title", ""), "level": c.get("level", 1)})
    g["edges"].append({"src": str(c["id"]), "tgt": "clause:planted", "type": "REFERENCES"}); return g, STRUCTURE_ONLY

def add_term_defined_and_used(g, rng, text=None):
    defining = {str(e["src"]) for e in edges_of(g, "DEFINES")}
    clauses = ids(g, "CLAUSE"); users = [c for c in clauses if c not in defining]     # a user that defines nothing cannot close a circle
    name = "Planted Term" if text is None else _word_from(text, rng, _names(g))
    if name is None or not users or len(clauses) < 2: return None
    c2 = rng.choice(users); c1 = rng.choice([c for c in clauses if c != c2])
    g["nodes"].append({"id": "term:planted", "node_type": "DEFINED_TERM", "name": name})
    g["edges"] += [{"src": c1, "tgt": "term:planted", "type": "DEFINES"}, {"src": c2, "tgt": "term:planted", "type": "USES"}]; return g, STRUCTURE_ONLY

def add_one_way_reference(g, rng, text=None):
    clauses = ids(g, "CLAUSE"); existing = {(str(e["src"]), str(e["tgt"])) for e in edges_of(g, "REFERENCES")}
    reach = collections.defaultdict(set)
    for a_, b_ in existing: reach[a_].add(b_)
    def reaches(x, y, seen=None):
        seen = seen or set()
        return any(n == y or (n not in seen and not seen.add(n) and reaches(n, y, seen)) for n in reach[x])
    pairs = [(x, y) for x in clauses for y in clauses if x != y and (x, y) not in existing and not reaches(y, x)]
    if not pairs: return None
    x, y = rng.choice(pairs); g["edges"].append({"src": x, "tgt": y, "type": "REFERENCES"}); return g, STRUCTURE_ONLY

def add_child_clause_with_parent_link(g, rng, text=None):
    parent = rng.choice(ids(g, "CLAUSE")); child = f"{parent}.99"
    if child in ids(g): return None
    g["nodes"].append({"id": child, "node_type": "CLAUSE", "title": child, "level": 2})
    g["edges"].append({"src": child, "tgt": parent, "type": "IS_PART_OF"}); return g, STRUCTURE_ONLY

CONTROLS = [   # (label, function, the graph says the same thing so findings and scores must be identical)
    ("Records shuffled", shuffle_records, True),
    ("Alternative key names (type, source, target)", alternative_keys, True),
    ("Type names in lower case", lower_case_types, True),
    ("Attributes nested under attrs", nested_attributes, True),
    ("Party added that the clause mentions", add_named_party, False),
    ("Term added with a definition and a use", add_term_defined_and_used, False),
    ("One-way reference added between two clauses", add_one_way_reference, False),
    ("Child clause added with a link to its parent", add_child_clause_with_parent_link, False),
    ("Second node added for the same clause, linked to it", add_second_node_for_a_clause, False),
]


def run_control(fn, same_scores, graph, rng, **lint_kw):
    result = fn(copy.deepcopy(graph), rng, text=lint_kw.get("text") if isinstance(lint_kw.get("text"), str) else None)
    if result is None:
        return None
    changed, allowed = result
    fresh = [k for k in new_findings(lint(graph, **lint_kw), lint(changed, **lint_kw)) if k[0] not in allowed]
    lost = [k for k in new_findings(lint(changed, **lint_kw), lint(graph, **lint_kw)) if k[0] not in allowed] if same_scores else []
    drift = same_scores and reference_free_scores(graph) != reference_free_scores(changed)
    return {"false_alarm": bool(fresh), "finding_lost": bool(lost), "scores_changed": bool(drift), "examples": fresh[:2]}


# ----------------------------------------------------------------- unusable input

UNUSABLE = [
    ("Empty graph", {"nodes": [], "edges": []}),
    ("Empty object", {}),
    ("Nodes given as text", {"nodes": "see above", "edges": []}),
    ("Only edges, no nodes", {"nodes": [], "edges": [{"src": "1", "tgt": "2", "type": "REFERENCES"}]}),
    ("Every node without an id", {"nodes": [{"node_type": "CLAUSE", "title": "1"}], "edges": []}),
    ("JSON cut off mid-way", '{"nodes": [{"id": "1", "node_type": "CLAUSE"'),
    ("Prose instead of JSON", "{Here is the graph you asked for}"),
    ("A list instead of an object", "[1, 2, 3]"),
]


def run_unusable(value):
    try:
        report = lint(value)
    except GraphReadError as e:
        return "rejected with a message", str(e)[:70]
    return ("reported" if report.findings else "PASSED QUIETLY"), ", ".join(sorted(report.counts()))


# ----------------------------------------------------------------- main

def evaluate(records, trials, seed):
    clauses, contracts = clause_graphs(records), contract_graphs(records)
    rows, controls = [], []
    for label, fn, expected in RECORD_DEFECTS:
        outcomes = [run_defect(label, fn, expected, g, random.Random(f"{seed}-{label}-{i}"), clause_level=True) for i, (g, _, _) in enumerate(clauses)]
        rows.append((label, expected, "clause", [o for o in outcomes if o]))
    for label, fn, expected in CONTRACT_DEFECTS:
        outcomes = [run_defect(label, fn, expected, g, random.Random(f"{seed}-{label}-{c}-{t}")) for c, (g, _) in enumerate(contracts) for t in range(trials)]
        rows.append((label, expected, "contract", [o for o in outcomes if o]))
    grounding = [("Party added that the clause text does not mention", invented_party, "focal"), ("Name of a party or term changed to one not in the text", altered_name, "fn_text")]
    for label, fn, how in grounding:
        outcomes = []
        for i, (g, text, focal) in enumerate(clauses):
            extra = {"focal": focal} if how == "focal" else {"fn_text": text}
            outcomes.append(run_defect(label, fn, "ungrounded_node", g, random.Random(f"{seed}-{label}-{i}"), clause_level=True, text=text, **extra))
        rows.append((label, "ungrounded_node", "clause with text", [o for o in outcomes if o]))
    for label, fn, same_scores in CONTROLS:
        outcomes = [run_control(fn, same_scores, g, random.Random(f"{seed}-{label}-{i}"), clause_level=True, text=text) for i, (g, text, _) in enumerate(clauses)]
        outcomes += [run_control(fn, same_scores, g, random.Random(f"{seed}-{label}-c{c}-{t}")) for c, (g, _) in enumerate(contracts) for t in range(trials)]
        controls.append((label, [o for o in outcomes if o]))
    pairs = []
    for c, (g, _) in enumerate(contracts):
        for t in range(trials * 3):
            rng = random.Random(f"{seed}-pair-{c}-{t}")
            first, second = rng.sample(CONTRACT_DEFECTS, 2)
            adds_a_term = (add_undefined_term, add_unused_definition, circular_definition)
            if first[1] in adds_a_term and second[1] in adds_a_term:
                continue                      # these plant nodes with the same ids and cannot be combined
            outcome = run_pair(first, second, g, rng)
            if outcome is not None:
                pairs.append(outcome)
    stored = [(r["gold_graph"], (r.get("prediction") or {}).get("graph"), (r.get("clause") or {}).get("text", "")) for r in records]
    stored = [(g, p, t) for g, p, t in stored if isinstance(p, dict)]
    def tally(index):
        counts = collections.Counter()
        for item in stored:
            counts.update(f.check for f in lint(item[index], clause_level=True, text=item[2]).findings)
        return counts
    real = {"graphs": len(stored), "reference": tally(0), "predicted": tally(1)} if stored else None
    unusable = [(label, *run_unusable(value)) for label, value in UNUSABLE]
    return {"clauses": len(clauses), "contracts": [(g["contract_id"], len(g["nodes"]), len(g["edges"])) for g, _ in contracts],
            "defects": rows, "controls": controls, "unusable": unusable, "pairs": pairs, "real": real}


def render(res, trials, seed):
    out = ["# Evaluation of contract-graph-linter", "",
           "Generated by `evaluation/evaluate_linter.py`. Run it again to regenerate this file.", "",
           "## What was tested", "",
           f"- **Graphs.** {res['clauses']} reference clause graphs from the public ContractGraphEval benchmark, and {len(res['contracts'])} contract graphs assembled from them "
           f"({', '.join(f'{n} nodes and {e} edges' for _, n, e in res['contracts'])}). The contracts are commercial agreements from CUAD.",
           "- **Planted defects.** One change at a time is made to a graph, with a known expected finding. A defect counts as detected when the expected check is reported "
           "on the changed node or edge and was not reported before the change.",
           "- **Controls.** Changes that should not produce a finding: rewriting the graph without changing what it says, or adding something well-formed.",
           "- **Unusable input.** Output that is not a usable graph. It must be reported or rejected, never passed.",
           f"- Clause-level defects are planted once in every clause graph. Contract-level defects are planted {trials} times in each contract graph, at random places (seed {seed}).", ""]
    total = detected = 0
    out += ["## Planted defects", "", "| Planted defect | Expected finding | Level | Trials | Detected | Missed |", "|---|---|---|---|---|---|"]
    for label, expected, level, outcomes in res["defects"]:
        n, d = len(outcomes), sum(o["detected"] for o in outcomes)
        total += n; detected += d
        out.append(f"| {label} | `{expected}` | {level} | {n} | {d} | {n - d} |")
    out += [f"| **All** | | | **{total}** | **{detected}** | **{total - detected}** |", "",
            f"In {sum(1 for _, _, _, os_ in res['defects'] for o in os_ if o['other_new_findings'])} of the {total} trials the change also produced other new findings. "
            "For example, a node record with its id removed is reported, and so are the edges that pointed to it.", ""]
    if res["pairs"]:
        both = sum(res["pairs"])
        out += ["## Two defects at once", "",
                f"Two different contract-level defects were planted in the same graph, to see whether one hides the other. Both expected findings appeared in {both} of {len(res['pairs'])} trials. "
                "Trials in which the second change altered the first planted defect itself were left out.", ""]
    out += ["## Controls", "", "The first four changes rewrite the graph without changing what it says. The rest add something well-formed.", "",
            "| Change | Trials | Unexpected new findings | Findings lost | Scores changed |", "|---|---|---|---|---|"]
    for label, outcomes in res["controls"]:
        same = dict((l, s_) for l, _, s_ in CONTROLS)[label]
        out.append(f"| {label} | {len(outcomes)} | {sum(o['false_alarm'] for o in outcomes)} | {sum(o['finding_lost'] for o in outcomes) if same else 'n/a'} | {sum(o['scores_changed'] for o in outcomes) if same else 'n/a'} |")
    out += ["", "When something is added, a new `articulation_point` or `small_component` finding is expected and not counted: adding a node does change the shape of the graph.", ""]
    if res["real"]:
        real = res["real"]; checks = sorted(set(real["reference"]) | set(real["predicted"]))
        out += ["## Reference graphs and predicted graphs, as stored", "",
                f"No defect is planted here. These are the findings on the {real['graphs']} reference graphs and on the predicted graphs stored with them in the benchmark, each checked against its clause text.", "",
                "| Finding | Reference graphs | Predicted graphs |", "|---|---|---|"]
        out += [f"| `{c}` | {real['reference'].get(c, 0)} | {real['predicted'].get(c, 0)} |" for c in checks]
        out += [f"| **All** | **{sum(real['reference'].values())}** | **{sum(real['predicted'].values())}** |", ""]
    out += ["## Unusable input", "", "| Input | Outcome | Detail |", "|---|---|---|"]
    out += [f"| {label} | {outcome} | {detail} |" for label, outcome, detail in res["unusable"]]
    out += ["", "## Limits of this evaluation", "",
            f"- The graphs come from {len(res['contracts'])} contracts. The planted defects are placed in many positions, but the underlying documents are few.",
            "- Planted defects are constructed. Detecting them shows that each check does what it is specified to do. It does not measure how often such defects occur in practice, "
            "or how useful a finding is to a reviewer.",
            "- The reference graphs were labelled by a language model. Findings on the unmodified graphs are not counted here as right or wrong.",
            "- Checks that compare the graph with the text only test whether a name appears in the clause, not whether the relation drawn from it is correct.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", help="JSON Lines file of records (default: download the public benchmark)")
    ap.add_argument("--trials", type=int, default=40, help="placements per contract for each contract-level defect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", metavar="FILE", help="write the report to this file as well as printing it")
    args = ap.parse_args()
    report = render(evaluate(read_records(args.graphs), args.trials, args.seed), args.trials, args.seed)
    print(report)
    if args.write:
        Path(args.write).write_text(report + "\n", encoding="utf-8")
