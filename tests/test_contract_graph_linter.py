"""Tests for the contract linter. Run with:  python -m pytest -q"""
import copy
import json
from pathlib import Path

import pytest

from contract_graph_linter import (context_slice, dependencies, dependents, graph_metrics, lint,
                             load_graph, reference_free_scores, run_checks)
from contract_graph_linter.__main__ import main

EXAMPLE = Path(__file__).resolve().parent.parent / "contract_graph_linter" / "examples" / "nda.json"
TEXT = "4.1 The Recipient may disclose Confidential Information to its Affiliates."


@pytest.fixture
def base():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def checks(graph, **kw):
    return [f.check for f in run_checks(graph, **kw)]


def clause(i):
    return {"id": i, "node_type": "CLAUSE", "title": i, "level": 1}


# ---- loading -----------------------------------------------------------------

def test_load_from_path_dict_and_json_string(base):
    for source in (EXAMPLE, str(EXAMPLE), base, json.dumps(base)):
        g = load_graph(source)
        assert len(g.nodes) == 8 and len(g.edges) == 10
        assert g.node_type("term:Affiliate") == "DEFINED_TERM"


def test_load_accepts_alternative_keys_and_prediction_wrapper():
    wrapped = {"graph": {"nodes": [{"id": 1, "type": "clause"}, {"id": "t", "type": "defined_term", "label": "X"}],
                         "edges": [{"source": 1, "target": "t", "edge_type": "uses"}]}}
    g = load_graph(wrapped)
    assert g.node_type("1") == "CLAUSE"
    assert (g.edges[0].src, g.edges[0].tgt, g.edges[0].type) == ("1", "t", "USES")


def test_load_accepts_nested_attrs():
    g = load_graph({"nodes": [{"id": "para_1", "type": "CLAUSE", "attrs": {"title": "(a) Export", "level": 1}}], "edges": []})
    assert g.nodes["para_1"].label == "(a) Export" and g.nodes["para_1"].attrs["level"] == 1


# ---- metrics -----------------------------------------------------------------

def test_metrics_on_example(base):
    m = graph_metrics(base)
    assert (m["nodes"], m["edges"]) == (8, 10)
    assert m["dependency_depth"] == 2 and m["orphans"] == 2
    assert m["articulation_point_ids"] == ["1.2", "4.1"]
    assert m["node_types"] == {"CLAUSE": 5, "DEFINED_TERM": 2, "PARTY": 1}
    assert m["edge_types"] == {"DEFINES": 1, "IS_PART_OF": 3, "MENTIONS_PARTY": 2, "REFERENCES": 1, "USES": 3}
    assert m["components"] == 1 and m["largest_component_share"] == 1.0
    assert m["hierarchy_depth"] == 1
    assert m["used_terms_defined"] == 0.5 and m["defined_terms_used"] == 1.0


def test_metrics_on_empty_graph():
    m = graph_metrics({"nodes": [], "edges": []})
    assert m["nodes"] == 0 and m["density"] == 0.0 and m["k_core_k"] is None and m["components"] == 0


# ---- findings ----------------------------------------------------------------

def test_findings_are_observations_with_clauses_to_read(base):
    report = lint(base)
    undefined = [f for f in report.findings if f.check == "undefined_term"]
    assert len(undefined) == 1 and "Confidential Information" in undefined[0].message
    assert undefined[0].clauses == ["4.1"]
    assert set(undefined[0].to_dict()) == {"check", "message", "nodes", "edge", "clauses"}
    assert report.counts() == {"articulation_point": 2, "undefined_term": 1}


def test_unused_definition(base):
    base["nodes"].append({"id": "term:Purpose", "node_type": "DEFINED_TERM", "name": "Purpose"})
    base["edges"].append({"src": "1.2", "tgt": "term:Purpose", "type": "DEFINES"})
    found = [f for f in run_checks(base) if f.check == "unused_definition"]
    assert len(found) == 1 and found[0].clauses == ["1.2"]


def test_duplicate_node_id_and_duplicate_edge():
    g = {"nodes": [clause("1"), clause("1"), clause("2")], "edges": [{"src": "2", "tgt": "1", "type": "REFERENCES"}] * 2}
    found = checks(g)
    assert "duplicate_node_id" in found and "duplicate_edge" in found


def test_dangling_edge(base):
    base["edges"].append({"src": "4.2", "tgt": "9.9", "type": "REFERENCES"})
    found = [f for f in run_checks(base) if f.check == "dangling_edge"]
    assert len(found) == 1 and "9.9 has no node record" in found[0].message and found[0].clauses == ["4.2"]


def test_self_reference_is_not_also_a_cycle():
    g = {"nodes": [clause("1"), clause("2")], "edges": [{"src": "1", "tgt": "1", "type": "REFERENCES"}, {"src": "2", "tgt": "1", "type": "REFERENCES"}]}
    found = checks(g)
    assert "self_reference" in found and "reference_cycle" not in found


def test_unknown_node_type_is_reported_once():
    g = {"nodes": [clause("1"), {"id": "obligation:pay", "node_type": "OBLIGATION", "name": "pay"}],
         "edges": [{"src": "1", "tgt": "obligation:pay", "type": "CONTAINS"}]}
    found = checks(g)
    assert found.count("unknown_node_type") == 1 and "ontology_violation" not in found


def test_ontology_violation(base):
    base["edges"].append({"src": "4.1", "tgt": "party:Recipient", "type": "USES"})     # USES points to a term in the ontology
    base["edges"].append({"src": "4.1", "tgt": "4.2", "type": "QUALIFIES"})            # edge type outside the ontology
    assert checks(base).count("ontology_violation") == 2


def test_clause_to_clause_edges_fit_the_ontology(base):
    assert "ontology_violation" not in checks(base)


def test_missing_name_applies_to_parties_terms_and_values_only():
    g = {"nodes": [{"id": "1", "node_type": "CLAUSE", "title": ""}, {"id": "party:x", "node_type": "PARTY"},
                   {"id": "value:y", "node_type": "VALUE", "unit": "days"}],
         "edges": [{"src": "1", "tgt": "party:x", "type": "MENTIONS_PARTY"}, {"src": "1", "tgt": "value:y", "type": "CONTAINS"}]}
    found = run_checks(g)
    assert sorted(f.nodes[0] for f in found if f.check == "missing_name") == ["party:x", "value:y"]


def test_id_type_mismatch():
    g = {"nodes": [clause("1"), {"id": "party:Acme", "node_type": "DEFINED_TERM", "name": "Acme"}],
         "edges": [{"src": "1", "tgt": "party:Acme", "type": "USES"}]}
    assert "id_type_mismatch" in checks(g)


def test_a_segment_node_and_its_numbered_clause_node_are_two_nodes():
    """A clause can appear as a segment node and as a numbered clause node. That is not a duplicate."""
    g = {"nodes": [{"id": "25", "node_type": "CLAUSE", "title": "16. THE PRINCIPALS.", "level": 1},
                   {"id": "clause:16", "node_type": "CLAUSE", "title": "16. THE PRINCIPALS.", "level": 1},
                   {"id": "term:Principals", "node_type": "DEFINED_TERM", "name": "Principals"}],
         "edges": [{"src": "25", "tgt": "clause:16", "type": "REFERENCES"}, {"src": "clause:16", "tgt": "term:Principals", "type": "DEFINES"}]}
    found = checks(g, clause_level=True)
    assert "duplicate_node_id" not in found and "id_type_mismatch" not in found and "duplicate_term_name" not in found


def test_definition_cycle(base):
    base["nodes"] += [{"id": "1.3", "node_type": "CLAUSE", "title": "1.3", "level": 2},
                      {"id": "term:Control", "node_type": "DEFINED_TERM", "name": "Control"}]
    base["edges"] += [{"src": "1.3", "tgt": "1", "type": "IS_PART_OF"},
                      {"src": "1.3", "tgt": "term:Control", "type": "DEFINES"},
                      {"src": "1.3", "tgt": "term:Affiliate", "type": "USES"},
                      {"src": "1.2", "tgt": "term:Control", "type": "USES"}]
    cycles = [f for f in run_checks(base) if f.check == "definition_cycle"]
    assert len(cycles) == 1 and set(cycles[0].nodes) == {"term:Affiliate", "term:Control"}
    assert cycles[0].clauses == ["1.2", "1.3"]


def test_reference_cycle(base):
    base["edges"].append({"src": "4.1", "tgt": "4.2", "type": "REFERENCES"})
    assert checks(base).count("reference_cycle") == 1


def test_hierarchy_cycle_and_multiple_parents():
    loop = {"nodes": [clause("1"), clause("2")], "edges": [{"src": "1", "tgt": "2", "type": "IS_PART_OF"}, {"src": "2", "tgt": "1", "type": "IS_PART_OF"}]}
    assert "hierarchy_cycle" in checks(loop)
    two = {"nodes": [clause("1"), clause("2"), clause("3")], "edges": [{"src": "3", "tgt": "1", "type": "IS_PART_OF"}, {"src": "3", "tgt": "2", "type": "IS_PART_OF"}]}
    assert "multiple_parents" in checks(two) and "hierarchy_cycle" not in checks(two)


def test_duplicate_definitions(base):
    base["edges"].append({"src": "4.1", "tgt": "term:Affiliate", "type": "DEFINES"})
    base["nodes"].append({"id": "term:affiliate ", "node_type": "DEFINED_TERM", "name": "\u201cAffiliate\u201d"})
    found = checks(base)
    assert "multiply_defined_term" in found and "duplicate_term_name" in found


def test_hierarchy_checks_read_numbering_from_dotted_ids_only(base):
    base["nodes"].append({"id": "7.3", "node_type": "CLAUSE", "title": "7.3", "level": 2})
    assert "missing_parent_clause" in checks(base)
    with_parent = copy.deepcopy(base)
    with_parent["nodes"].append({"id": "7", "node_type": "CLAUSE", "title": "7", "level": 1})
    found = checks(with_parent)
    assert "missing_is_part_of" in found and "missing_parent_clause" not in found
    segment = {"nodes": [{"id": "25", "node_type": "CLAUSE", "title": "16.2 Notices", "level": 2}], "edges": []}
    assert "missing_parent_clause" not in checks(segment)


def test_isolated_node_and_small_component(base):
    base["nodes"] += [clause("9"), clause("10"), clause("11")]
    base["edges"].append({"src": "10", "tgt": "11", "type": "REFERENCES"})
    found = checks(base)
    assert "isolated_node" in found and "small_component" in found


def test_clause_level_mode_skips_whole_contract_checks(base):
    assert "undefined_term" in checks(base)
    assert "undefined_term" not in checks(base, clause_level=True)
    assert "articulation_point" not in checks(base, clause_level=True)


# ---- grounding ---------------------------------------------------------------

def test_names_are_looked_for_in_the_clause_text():
    g = {"nodes": [clause("4.1"), {"id": "party:Recipient", "node_type": "PARTY", "name": "Recipient"},
                   {"id": "term:Affiliates", "node_type": "DEFINED_TERM", "name": "Affiliates"},
                   {"id": "party:Discloser", "node_type": "PARTY", "name": "Discloser"},
                   {"id": "value:30 days", "node_type": "VALUE", "text": "30", "unit": "days"}],
         "edges": [{"src": "4.1", "tgt": "party:Recipient", "type": "MENTIONS_PARTY"}, {"src": "4.1", "tgt": "term:Affiliates", "type": "USES"},
                   {"src": "4.1", "tgt": "party:Discloser", "type": "MENTIONS_PARTY"}, {"src": "4.1", "tgt": "value:30 days", "type": "CONTAINS"}]}
    found = [f for f in run_checks(g, clause_level=True, text=TEXT) if f.check == "ungrounded_node"]
    assert sorted(f.nodes[0] for f in found) == ["party:Discloser", "value:30 days"]
    assert all(f.clauses == ["4.1"] for f in found)
    assert "ungrounded_node" not in checks(g, clause_level=True)                       # no text, no grounding check


def test_grounding_with_text_per_clause(base):
    texts = {"1.2": "1.2 Affiliate means any entity controlled by the Recipient.", "4.1": TEXT, "4.2": "4.2 Section 4.1 applies only to Affiliates."}
    found = [f for f in run_checks(base, text=texts) if f.check == "ungrounded_node"]
    assert found == []
    texts["4.1"] = "4.1 The Recipient may disclose to its Affiliates."
    found = [f for f in run_checks(base, text=texts) if f.check == "ungrounded_node"]
    assert [f.nodes[0] for f in found] == ["term:Confidential Information"]


# ---- slicing -----------------------------------------------------------------

def test_outgoing_slice_misses_the_condition_and_incoming_slice_finds_it(base):
    """Clause 4.2 conditions clause 4.1. Following only what 4.1 points to never reaches 4.2."""
    out = dependencies(base, "4.1")
    assert set(out) == {"1.2", "4"} and "4.2" not in out
    assert "4.2" in dependents(base, "4.1")
    both = context_slice(base, "4.1", direction="both")
    assert set(both) == {"1.2", "4", "4.2"} and both["4.2"]["direction"] == "in"


def test_slice_depth_and_direction(base):
    assert set(dependencies(base, "4.2", depth=1)) == {"4", "4.1", "1.2"}
    with pytest.raises(ValueError):
        context_slice(base, "4.1", direction="sideways")


# ---- reference-free scores ---------------------------------------------------

def test_reference_free_scores(base):
    scores = reference_free_scores(base)
    assert scores["domain_range_validity"] == 1.0 and scores["ontology_compliance"] == 1.0 and scores["orphan_nodes"] == 1.0
    assert scores["unused_definitions"] == 0.5
    base["edges"].append({"src": "4.2", "tgt": "SCHEDULE 9", "type": "REFERENCES"})
    assert reference_free_scores(base)["undefined_references"] == 0.5
    assert set(reference_free_scores(None).values()) == {0.0}


# ---- command line ------------------------------------------------------------

def test_command_line(tmp_path, base, capsys):
    assert main([str(EXAMPLE)]) == 0
    assert "undefined_term" in capsys.readouterr().out
    assert main([str(EXAMPLE), "--strict"]) == 1
    capsys.readouterr()
    text = tmp_path / "clause.txt"; text.write_text("Nothing relevant here.", encoding="utf-8")
    assert main([str(EXAMPLE), "--json", "--scores", "--slice", "4.1", "--text", str(text)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["ungrounded_node"] == 3 and "4.2" in out["slice"]["context"]
    assert all(set(f) == {"check", "message", "nodes", "edge", "clauses"} for f in out["findings"])


# ---- extending the ontology --------------------------------------------------

from contract_graph_linter import DEFAULT_ONTOLOGY, Ontology, load_ontology  # noqa: E402

EXAMPLES = EXAMPLE.parent
LOAN = json.loads((EXAMPLES / "loan.json").read_text(encoding="utf-8"))
EXTENSION = EXAMPLES / "loan_ontology.json"
LOAN_TEXT = (EXAMPLES / "loan.txt").read_text(encoding="utf-8")


def test_new_types_are_outside_the_default_ontology():
    found = checks(LOAN, clause_level=True)
    assert found.count("unknown_node_type") == 2                      # OBLIGATION and SCHEDULE
    assert found.count("ontology_violation") == 1                     # the IMPOSES edge type


def test_an_extension_adds_node_types_edge_types_and_connections():
    for ontology in (EXTENSION, str(EXTENSION), json.loads(EXTENSION.read_text(encoding="utf-8")), load_ontology(EXTENSION)):
        assert checks(LOAN, clause_level=True, text=LOAN_TEXT, ontology=ontology) == []
    report = lint(LOAN, clause_level=True, ontology=EXTENSION)
    assert report.metrics["node_types"]["OBLIGATION"] == 1 and report.metrics["edge_types"]["IMPOSES"] == 1


def test_the_default_ontology_is_not_changed_by_an_extension():
    extended = load_ontology(EXTENSION)
    assert isinstance(extended, Ontology) and extended.has_node_type("OBLIGATION") and extended.allows("REFERENCES", "CLAUSE", "SCHEDULE")
    assert not DEFAULT_ONTOLOGY.has_node_type("OBLIGATION") and not DEFAULT_ONTOLOGY.allows("REFERENCES", "CLAUSE", "SCHEDULE")
    assert extended.allows("REFERENCES", "CLAUSE", "CLAUSE")                         # the built-in connection is kept
    assert load_ontology(None) is DEFAULT_ONTOLOGY


def test_extended_rules_are_checked():
    g = copy.deepcopy(LOAN)
    g["edges"] += [{"src": "5.1", "tgt": "party:Borrower", "type": "IMPOSES"},        # IMPOSES points to an obligation
                   {"src": "5.1", "tgt": "term:Loan", "type": "REFERENCES"}]          # REFERENCES points to a clause or a schedule
    messages = [f.message for f in run_checks(g, clause_level=True, ontology=EXTENSION) if f.check == "ontology_violation"]
    assert len(messages) == 2
    assert any("IMPOSES connects CLAUSE -> OBLIGATION in the ontology" in m for m in messages)
    assert any("REFERENCES connects CLAUSE -> CLAUSE or CLAUSE -> SCHEDULE in the ontology" in m for m in messages)


def test_names_of_new_types_are_checked_and_looked_up_in_the_text():
    g = copy.deepcopy(LOAN)
    g["nodes"] += [{"id": "obligation:insure", "node_type": "OBLIGATION", "text": "insure the Property"},
                   {"id": "obligation:blank", "node_type": "OBLIGATION"}]
    g["edges"] += [{"src": "5.1", "tgt": "obligation:insure", "type": "IMPOSES"}, {"src": "5.1", "tgt": "obligation:blank", "type": "IMPOSES"}]
    found = run_checks(g, clause_level=True, text=LOAN_TEXT, ontology=EXTENSION)
    assert [(f.check, f.nodes[0]) for f in found] == [("missing_name", "obligation:blank"), ("ungrounded_node", "obligation:insure")]
    assert "has no text" in found[0].message


def test_a_new_type_can_have_no_name_and_an_id_prefix_is_checked():
    ontology = {"node_types": {"ANNEX": {"name": None, "id_prefix": "annex"}}, "edge_types": {"REFERENCES": [["CLAUSE", "ANNEX"]]}}
    g = {"nodes": [clause("1"), {"id": "annex:A", "node_type": "ANNEX"}, {"id": "annex:B", "node_type": "PARTY", "name": "B"}],
         "edges": [{"src": "1", "tgt": "annex:A", "type": "REFERENCES"}, {"src": "1", "tgt": "annex:B", "type": "MENTIONS_PARTY"}]}
    assert checks(g, clause_level=True, ontology=ontology) == ["id_type_mismatch"]


@pytest.mark.parametrize("extension, message", [
    ({"node_types": {"PARTY": {"name": "title"}}}, "already in the ontology"),
    ({"edge_types": {"IMPOSES": [["CLAUSE", "OBLIGATION"]]}}, "is not in the ontology"),
    ({"edge_types": {"IMPOSES": ["CLAUSE", "PARTY"]}}, "pairs"),
    ({"nodes": {"OBLIGATION": {}}}, "Unknown key"),
    ({"node_types": {"ALIAS": {"id_prefix": "party"}}}, "already used"),
    ({"node_types": {"OBLIGATION": {"label": "text"}}}, "optional 'name' and 'id_prefix'"),
])
def test_a_malformed_extension_is_rejected_with_a_clear_message(extension, message):
    with pytest.raises(ValueError, match=message):
        load_ontology(extension)


def test_reference_free_scores_follow_the_extension():
    assert reference_free_scores(LOAN)["domain_range_validity"] < 1.0
    assert reference_free_scores(LOAN, ontology=EXTENSION)["domain_range_validity"] == 1.0
    assert reference_free_scores(LOAN, ontology=EXTENSION)["ontology_compliance"] == 1.0


def test_command_line_with_an_ontology(capsys):
    loan, text = str(EXAMPLES / "loan.json"), str(EXAMPLES / "loan.txt")
    assert main([loan, "--clause-level", "--text", text, "--strict"]) == 1
    assert "unknown_node_type" in capsys.readouterr().out
    assert main([loan, "--clause-level", "--text", text, "--ontology", str(EXTENSION), "--strict"]) == 0
    assert "findings: 0" in capsys.readouterr().out


# ---- bundled examples --------------------------------------------------------

from contract_graph_linter.examples import describe_examples, example_names, example_path, load_example  # noqa: E402


def run_example(name):
    e = load_example(name)
    return lint(e["graph"], clause_level=e["clause_level"], text=e["text"], ontology=e["ontology"])


def test_every_example_loads_and_runs():
    assert example_names() == ["nda", "services", "invoice", "loan"] and set(describe_examples()) == set(example_names())
    for name in example_names():
        assert example_path(name).exists()
        run_example(name)
    with pytest.raises(ValueError, match="Unknown example"):
        load_example("missing")


def test_examples_show_what_their_descriptions_say():
    assert run_example("nda").counts()["undefined_term"] == 1
    services = run_example("services").counts()
    for check in ("definition_cycle", "reference_cycle", "unused_definition", "undefined_term", "dangling_edge",
                  "missing_parent_clause", "isolated_node"):
        assert services.get(check) == 1, check
    invoice = run_example("invoice")
    assert [(f.check, f.nodes) for f in invoice.findings] == [("ungrounded_node", ["party:Guarantor"])]
    assert run_example("loan").findings == []


def test_command_line_examples(capsys):
    assert main(["--list-examples"]) == 0
    listed = capsys.readouterr().out
    assert all(name in listed for name in example_names())
    assert main(["--example", "invoice"]) == 0
    assert "ungrounded_node" in capsys.readouterr().out
    assert main(["--example", "loan", "--strict"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main([])


# ---- inputs that cannot be assessed, and independence from record order ------

import random  # noqa: E402


def test_an_empty_graph_is_reported_and_scores_zero(tmp_path):
    for empty in ({"nodes": [], "edges": []}, {}, {"nodes": None}):
        assert checks(empty) == ["empty_graph"]
        assert set(reference_free_scores(empty).values()) == {0.0}
    path = tmp_path / "empty.json"; path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    assert main([str(path), "--strict"]) == 1


def test_records_that_cannot_be_read_are_reported_not_dropped():
    g = {"nodes": [{"node_type": "CLAUSE", "title": "no id"}, clause("1"), "not a record", {"id": "", "node_type": "PARTY", "name": "x"}],
         "edges": [{"type": "REFERENCES"}, {"src": "1", "type": "REFERENCES"}, 7]}
    found = run_checks(g, clause_level=True)
    messages = {f.check: [] for f in found}
    for f in found:
        messages[f.check].append(f.message)
    assert messages["invalid_node_record"] == ["Node record 1 has no id", "Node record 3 is not an object", "Node record 4 has no id"]
    assert messages["invalid_edge_record"] == ["Edge record 1 has no source and target", "Edge record 2 has no target", "Edge record 3 is not an object"]
    assert "empty_graph" not in messages                                           # one node could be read


def test_nodes_given_as_something_other_than_a_list():
    assert set(checks({"nodes": {"id": "1", "node_type": "CLAUSE"}, "edges": "none"})) == {"invalid_edge_record", "isolated_node"}
    assert set(reference_free_scores({"nodes": "x", "edges": []}).values()) == {0.0}


def shuffled(graph, seed):
    rng = random.Random(seed)
    g = copy.deepcopy(graph)
    rng.shuffle(g["nodes"]); rng.shuffle(g["edges"])
    return g


@pytest.mark.parametrize("name", ["nda", "services", "invoice", "loan"])
def test_results_do_not_depend_on_the_order_of_records(name):
    e = load_example(name)
    kw = dict(clause_level=e["clause_level"], text=e["text"], ontology=e["ontology"])
    def result(graph):
        report = lint(graph, **kw)
        return ([(f.check, f.message, f.nodes, f.clauses) for f in report.findings],
                reference_free_scores(graph, ontology=e["ontology"]),
                {k: v for k, v in report.metrics.items() if not k.startswith("top_")})
    expected = result(e["graph"])
    for seed in range(8):
        assert result(shuffled(e["graph"], seed)) == expected


def test_component_score_treats_the_largest_component_as_the_main_one():
    small = [clause("a"), clause("b")]
    large = [clause(f"m{i}") for i in range(10)]
    edges = [{"src": "a", "tgt": "b", "type": "REFERENCES"}] + [{"src": f"m{i}", "tgt": f"m{i + 1}", "type": "REFERENCES"} for i in range(9)]
    first, second = {"nodes": small + large, "edges": edges}, {"nodes": large + small, "edges": edges}
    assert reference_free_scores(first)["isolated_components"] == reference_free_scores(second)["isolated_components"] == 0.0


def test_reference_score_follows_the_ontology_extension():
    assert reference_free_scores(LOAN, ontology=EXTENSION)["undefined_references"] == 1.0     # a schedule is an allowed target
    assert reference_free_scores(LOAN)["undefined_references"] == 0.0                         # not in the default ontology
    assert set(reference_free_scores(LOAN, ontology=EXTENSION).values()) == {0.0, 1.0}


def test_loops_are_written_from_their_smallest_id():
    g = {"nodes": [clause("3"), clause("1"), clause("2")],
         "edges": [{"src": "3", "tgt": "1", "type": "REFERENCES"}, {"src": "1", "tgt": "2", "type": "REFERENCES"}, {"src": "2", "tgt": "3", "type": "REFERENCES"}]}
    messages = [f.message for f in run_checks(g) if f.check == "reference_cycle"]
    assert messages == ["Clauses reference each other in a loop: 1 -> 2 -> 3 -> 1"]


# ---- scores never vouch for a graph that could not be read in full ------------

def test_scores_are_zero_when_a_record_cannot_be_read():
    sound = {"nodes": [clause("1"), clause("2")], "edges": [{"src": "1", "tgt": "2", "type": "REFERENCES"}]}
    assert set(reference_free_scores(sound).values()) == {1.0}
    missing_id = {"nodes": sound["nodes"] + [{"node_type": "CLAUSE", "title": "no id"}], "edges": sound["edges"]}
    missing_target = {"nodes": sound["nodes"], "edges": sound["edges"] + [{"src": "2", "type": "REFERENCES"}]}
    for g in (missing_id, missing_target):
        assert set(reference_free_scores(g).values()) == {0.0}
        report = lint(g, clause_level=True)
        assert report.metrics["unreadable_node_records"] + report.metrics["unreadable_edge_records"] == 1
        assert "records that could not be read: 1" in report.summary()
    assert "could not be read" not in lint(sound, clause_level=True).summary()


# ---- loops are reported once per group, however many there are ----------------

def test_many_loops_through_the_same_clauses_give_one_finding():
    n = 40                                                   # every clause references every other: far too many loops to list
    g = {"nodes": [clause(f"c{i:02d}") for i in range(n)],
         "edges": [{"src": f"c{i:02d}", "tgt": f"c{j:02d}", "type": "REFERENCES"} for i in range(n) for j in range(n) if i != j]}
    found = [f for f in run_checks(g) if f.check == "reference_cycle"]
    assert len(found) == 1 and len(found[0].nodes) == n
    assert found[0].message == "Clauses reference each other in a loop: c00 -> c01 -> c00 (40 clauses are linked by loops)"


def test_separate_loops_are_separate_findings():
    g = {"nodes": [clause(x) for x in "abcd"],
         "edges": [{"src": s, "tgt": t, "type": "REFERENCES"} for s, t in (("a", "b"), ("b", "a"), ("c", "d"), ("d", "c"))]}
    assert [f.nodes for f in run_checks(g) if f.check == "reference_cycle"] == [["a", "b"], ["c", "d"]]


# ---- questions about something that is not there are refused, not answered ----

def test_slicing_a_clause_that_is_not_in_the_graph_is_an_error(base, capsys):
    for fn in (dependencies, dependents, context_slice):
        with pytest.raises(ValueError, match="'9.9' is not in the graph"):
            fn(base, "9.9")
    with pytest.raises(ValueError):
        context_slice(base, "term:Affiliate")                # a term is not a clause
    assert main([str(EXAMPLE), "--slice", "9.9"]) == 2
    assert "not in the graph" in capsys.readouterr().err


def test_an_empty_clause_text_is_still_a_text():
    g = {"nodes": [clause("1"), {"id": "party:X", "node_type": "PARTY", "name": "Xavier"}],
         "edges": [{"src": "1", "tgt": "party:X", "type": "MENTIONS_PARTY"}]}
    assert checks(g, clause_level=True, text="") == ["ungrounded_node"]
    assert checks(g, clause_level=True, text=None) == []


def test_input_that_is_not_a_graph_is_refused_with_a_message(tmp_path, capsys):
    from contract_graph_linter import GraphReadError
    for bad in ('{"nodes": [', "[1, 2, 3]", None, 7):
        with pytest.raises(GraphReadError):
            lint(bad)
    cut = tmp_path / "cut.json"; cut.write_text('{"nodes": [{"id": "1"', encoding="utf-8")
    assert main([str(cut)]) == 2 and main([str(tmp_path / "missing.json")]) == 2
    err = capsys.readouterr().err
    assert "is not valid JSON" in err and "Could not open" in err and "Traceback" not in err
