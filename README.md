# contract-graph-linter

[![Paper](https://img.shields.io/badge/Paper-IEEE%20ICDMW%202025-blue)](https://doi.org/10.1109/ICDMW69685.2025.00092)
[![Research code](https://img.shields.io/badge/Research%20code-graph--grpo--lex-lightgrey)](https://github.com/moriyadechtiar/graph-grpo-lex)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/moriyadechtiar/contract-graph-linter/blob/main/LICENSE)

Checks, graph metrics and context slicing for contract graphs.

The input is a contract that has already been turned into a graph of clauses, parties, defined terms and values. The library reads that graph; it does not extract one from a document.

**Origin.** This library is the contract linter introduced in *GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization* (IEEE ICDMW 2025). The research code, notebooks and training pipeline are in [graph-grpo-lex](https://github.com/moriyadechtiar/graph-grpo-lex). This repository packages the linter so it can be installed and used on its own.

## Install

```bash
pip install contract-graph-linter
```

## Use

Try a bundled example:

```bash
contract-graph-linter --example nda --slice 4.1
```

```
Contract: EXAMPLE-NDA (constructed illustration)
  nodes 8, edges 10, density 0.1786, dependency depth 2, k-core 2
  orphans 2 (25%), leaves 5 (62%), articulation points 2, components 1
  findings: 3
  articulation_point: Node 1.2 is the only connection between parts of the graph  [read: 1.2]
  articulation_point: Node 4.1 is the only connection between parts of the graph  [read: 4.1]
  undefined_term: Term 'Confidential Information' is used by 4.1 and no clause defines it  [read: 4.1]
Context for clause 4.1 (direction=both, depth=1):
  1.2  [out, distance 1]  4.1 uses 'Affiliate', defined in 1.2
  4  [out, distance 1]  4.1 is part of 4
  4.2  [in, distance 1]  4.2 references 4.1
```

Then your own graph:

```bash
contract-graph-linter my_graph.json
```

From Python:

```python
from contract_graph_linter import lint, context_slice, reference_free_scores

report = lint("my_graph.json")
report.metrics            # density, depth, k-core, orphans, leaves, components, definition coverage, ...
report.findings           # list of Finding(check, message, nodes, edge, clauses)

lint(graph, text=clause_text)                           # also checks that names appear in the text
context_slice(report.graph, "4.1", direction="both")    # clauses to read together with 4.1
reference_free_scores(report.graph.raw)                 # seven scores in [0, 1], no gold graph needed
```

## Examples

Four small graphs ship with the library. `contract-graph-linter --list-examples` lists them.

| Example | Shows | Run |
|---|---|---|
| `nda` | A term that is used and never defined; context slicing | `contract-graph-linter --example nda --slice 4.1` |
| `services` | A circular definition, a reference loop, an unused definition, a reference to a clause that is not in the graph | `contract-graph-linter --example services` |
| `invoice` | One clause checked against its text: a party in the graph does not appear in the clause | `contract-graph-linter --example invoice` |
| `loan` | An ontology extension with obligations and schedules | `contract-graph-linter --example loan` |

```python
from contract_graph_linter.examples import load_example

example = load_example("services")          # graph, text, ontology, clause_level, description
report = lint(example["graph"])
```

## The graph format

A contract graph is a JSON object with `nodes` and `edges`.

```json
{
  "nodes": [{"id": "4.1", "node_type": "CLAUSE", "title": "4.1", "level": 2},
            {"id": "term:Affiliate", "node_type": "DEFINED_TERM", "name": "Affiliate"}],
  "edges": [{"src": "4.1", "tgt": "term:Affiliate", "type": "USES"}]
}
```

- **Nodes:** `CLAUSE`, `PARTY`, `DEFINED_TERM`, `VALUE`.
- **Edges**, each from a clause: `IS_PART_OF` and `REFERENCES` (to a clause), `DEFINES` and `USES` (to a defined term), `MENTIONS_PARTY` (to a party), `CONTAINS` (to a value).

Graphs in this format are available in the [contract-clause-graphs](https://huggingface.co/datasets/moriyad/contract-clause-graphs) dataset.

## Extending the ontology

The default ontology is the one above. To lint graphs that carry more node or edge types, such as obligations or schedules in a loan agreement, supply an extension. It adds to the default and cannot change it.

```json
{
  "node_types": {
    "OBLIGATION": {"name": "text", "id_prefix": "obligation"},
    "SCHEDULE": {"name": "title", "id_prefix": "schedule"}
  },
  "edge_types": {
    "IMPOSES": [["CLAUSE", "OBLIGATION"]],
    "REFERENCES": [["CLAUSE", "SCHEDULE"]]
  }
}
```

- **`node_types`**: each new type, with the attribute that holds its name (`name`, default `"name"`; `null` if the type has no name to check) and an optional `id_prefix`.
- **`edge_types`**: for each edge type, the `[source, target]` node types it may connect. Listing a built-in edge type, as `REFERENCES` here, adds a connection to it.

```bash
contract-graph-linter graph.json --ontology my_extension.json
```

```python
lint(graph, ontology="my_extension.json")        # or a dictionary
```

The `loan` example uses this extension. New types are checked against their rules, counted in the metrics, and looked up in the clause text. The checks on definitions, references and hierarchy, and context slicing, follow the built-in edge types.

## Findings

A finding is an observation, with the clauses to read. It may reflect the contract itself or the way the graph was built; the linter does not decide which. The reviewer goes to the listed clauses in the contract.

### Checks

| Check | What is observed |
|---|---|
| `undefined_term` | A term is used and no clause defines it |
| `unused_definition` | A term is defined and no clause uses it |
| `multiply_defined_term`, `duplicate_term_name` | A term is defined by several clauses; two term nodes share a name |
| `definition_cycle` | Terms are defined in terms of each other |
| `reference_cycle`, `self_reference` | Clauses reference each other in a loop (one finding per group of clauses, with one loop shown); a node is linked to itself |
| `hierarchy_cycle`, `multiple_parents` | Clauses are part of each other in a loop; a clause is part of more than one clause |
| `missing_parent_clause`, `missing_is_part_of` | Clause 3.4 with no clause 3 in the graph, or no `IS_PART_OF` edge to it (read from dotted ids) |
| `isolated_node`, `small_component` | A node with no edges; a few nodes not connected to the rest |
| `articulation_point` | A node that is the only connection between parts of the graph |
| `dangling_edge` | An edge points to an id that has no node record |
| `ontology_violation`, `unknown_node_type` | An edge or node type outside the ontology, or an edge between node types the ontology does not pair |
| `duplicate_node_id`, `duplicate_edge` | The same node id or the same edge appears more than once |
| `invalid_node_record`, `invalid_edge_record` | A node record with no id, or an edge record with no source or target |
| `empty_graph` | The graph has no nodes |
| `missing_name`, `id_type_mismatch` | A party, term or value without a name; an id prefix that disagrees with the node type |
| `ungrounded_node` | A party, term or value whose name does not appear in the clause text (needs `text`) |

Options: `--clause-level` skips the checks that need the whole contract, `--text FILE` supplies the clause text, `--ontology FILE` extends the ontology, and `--strict` exits with status 1 when there are findings.

## Graph metrics

| Metric | Definition | Reading |
|---|---|---|
| Density | Edges ÷ maximum possible edges | Overall interconnection |
| Dependency depth | Longest shortest path in the directed graph | How far a reader must follow links |
| Hierarchy depth | Longest `IS_PART_OF` chain | Nesting of the clause structure |
| k-core | Largest k with a non-empty k-core | The most interdependent core |
| Orphans, leaves | Nodes with no incoming, or no outgoing, edge | Completeness and terminal consequences |
| Articulation points, components | Nodes whose removal disconnects the graph; number of connected parts | Single points of failure, fragmentation |
| Definition coverage | Share of used terms that are defined, and of defined terms that are used | Completeness of the definitions |
| Counts by type; top in-degree, out-degree | Nodes and edges by type; most referenced and most referencing nodes | Composition and review priority |

## Context slicing

`dependencies(graph, clause)` follows what a clause relies on: clauses it references, its parent, and the clauses that define the terms it uses. `dependents(graph, clause)` follows the other direction: clauses that reference it, its children, and clauses that use a term it defines.

In the `nda` example, clause 4.2 conditions the permission in clause 4.1. A retrieval step that starts at 4.1 and follows only its outgoing links reaches the definition in 1.2 but never reaches 4.2. `context_slice(..., direction="both")` includes it.

## Reference-free scores

`reference_free_scores` computes seven quality scores for a predicted graph without a gold graph. The scores describe a graph that could be read in full: a graph with no nodes, or with a record that cannot be read, scores 0.0 on each. They do not depend on the order of the records. They are the reference-free checks used in the [ContractGraphEval](https://github.com/moriyadechtiar/ensemble-graph-eval) benchmark. To score a file of predictions:

```bash
python examples/score_predictions.py predictions.jsonl
```

## Evaluation

The linter is evaluated by planting known defects in real contract graphs from the ContractGraphEval benchmark and counting what it reports. The script and the full results are in [`evaluation/`](https://github.com/moriyadechtiar/contract-graph-linter/tree/main/evaluation).

| Test | Result |
|---|---|
| 25 kinds of planted defect, 3,045 placements | 3,045 detected, 0 missed |
| Two defects planted in the same graph | both reported in 274 of 274 trials |
| 9 changes that should produce no finding, 2,283 trials | 0 unexpected findings |
| 8 kinds of unusable input | 8 reported or rejected, 0 passed quietly |

These results show that each check does what it is specified to do, in real graphs. They do not measure how often such defects occur, or how useful a finding is to a reviewer. The graphs come from three contracts.

## Origin and citation

The linter was introduced in the GRAPH-GRPO-LEX paper. If you use this library, please cite it:

```bibtex
@inproceedings{dechtiar2025graphgrpolex,
  title     = {GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization},
  author    = {Dechtiar, Moriya and Katz, Daniel Martin and Sundaresan, Mari and Jaume, Sylvain and Wang, Hongming},
  booktitle = {2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages     = {767--776},
  year      = {2025},
  doi       = {10.1109/ICDMW69685.2025.00092}
}
```

| Related | Link |
|---|---|
| Research code for the paper | [graph-grpo-lex](https://github.com/moriyadechtiar/graph-grpo-lex) |
| Dataset of contract clause graphs | [moriyad/contract-clause-graphs](https://huggingface.co/datasets/moriyad/contract-clause-graphs) |
| Evaluation framework for predicted contract graphs | [ensemble-graph-eval](https://github.com/moriyadechtiar/ensemble-graph-eval) |

## Development

```bash
git clone https://github.com/moriyadechtiar/contract-graph-linter.git
cd contract-graph-linter
pip install -e ".[test]"
python -m pytest -q
python evaluation/evaluate_linter.py          # the evaluation; downloads the public benchmark file
```

## License

Apache License 2.0. See [`LICENSE`](https://github.com/moriyadechtiar/contract-graph-linter/blob/main/LICENSE) and [`NOTICE`](https://github.com/moriyadechtiar/contract-graph-linter/blob/main/NOTICE). Copies and derived works must keep the attribution in `NOTICE`.

## Contact

Moriya Dechtiar, moriya.dechtiar@alumni.harvard.edu. Issues and pull requests are welcome.
