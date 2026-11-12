"""Small example graphs that ship with the library.

    from contract_graph_linter import lint
    from contract_graph_linter.examples import example_names, load_example

    example = load_example("services")
    report = lint(example["graph"], clause_level=example["clause_level"],
                  text=example["text"], ontology=example["ontology"])

From the command line:  contract-graph-linter --example services
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent

# name: (description, graph file, clause text file, ontology extension file, single clause?)
_EXAMPLES = {
    "nda": ("A short confidentiality agreement: one term is used and never defined", "nda.json", None, None, False),
    "services": ("A services agreement with a circular definition, a reference loop, an unused definition "
                 "and a reference to a clause that is not there", "services.json", None, None, False),
    "invoice": ("One clause with its text: a party in the graph does not appear in the clause", "invoice.json", "invoice.txt", None, True),
    "loan": ("One loan clause with an obligation and a schedule, using an ontology extension", "loan.json", "loan.txt", "loan_ontology.json", True),
}


def example_names() -> List[str]:
    return list(_EXAMPLES)


def describe_examples() -> Dict[str, str]:
    return {name: spec[0] for name, spec in _EXAMPLES.items()}


def example_path(name: str) -> Path:
    """Path of the graph file of an example."""
    return _HERE / _spec(name)[1]


def _spec(name: str):
    if name not in _EXAMPLES:
        raise ValueError(f"Unknown example {name!r}. Choose from: {', '.join(_EXAMPLES)}.")
    return _EXAMPLES[name]


def load_example(name: str) -> Dict[str, Any]:
    """An example as a dictionary: graph, text, ontology, clause_level and description."""
    description, graph_file, text_file, ontology_file, clause_level = _spec(name)
    return {
        "name": name,
        "description": description,
        "graph": json.loads((_HERE / graph_file).read_text(encoding="utf-8")),
        "text": (_HERE / text_file).read_text(encoding="utf-8") if text_file else None,
        "ontology": json.loads((_HERE / ontology_file).read_text(encoding="utf-8")) if ontology_file else None,
        "clause_level": clause_level,
    }
