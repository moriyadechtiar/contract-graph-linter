"""Command line:  contract-graph-linter GRAPH.json [options]   or   contract-graph-linter --example NAME

Prints the graph metrics and the findings. With --strict, exits with status 1
when there is at least one finding. Exits with status 2 when the input cannot
be read as a graph.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys

from .examples import describe_examples, load_example
from .graph import GraphReadError
from .report import lint
from .scores import reference_free_scores
from .slicing import context_slice


def main(argv=None) -> int:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # exit quietly when the output is piped to head or less
    ap = argparse.ArgumentParser(prog="contract_graph_linter", description="Lint a contract graph.")
    ap.add_argument("graph", nargs="?", help="path to a graph JSON file ({'nodes': [...], 'edges': [...]})")
    ap.add_argument("--example", metavar="NAME", help="run a bundled example instead of a file (see --list-examples)")
    ap.add_argument("--list-examples", action="store_true", help="list the bundled examples and exit")
    ap.add_argument("--clause-level", action="store_true",
                    help="the graph covers a single clause; skip whole-contract checks")
    ap.add_argument("--text", metavar="FILE", help="file with the clause text; also checks that names appear in it")
    ap.add_argument("--ontology", metavar="FILE", help="JSON file that adds node and edge types to the default ontology")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--scores", action="store_true", help="also print reference-free scores in [0, 1]")
    ap.add_argument("--slice", metavar="CLAUSE", help="list the clauses to read together with CLAUSE")
    ap.add_argument("--direction", choices=("in", "out", "both"), default="both")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--strict", action="store_true", help="exit with status 1 when there are findings")
    args = ap.parse_args(argv)

    if args.list_examples:
        for name, description in describe_examples().items():
            print(f"{name:<10}{description}")
        return 0
    try:
        text = open(args.text, encoding="utf-8").read() if args.text else None
    except OSError as e:
        print(f"contract-graph-linter: could not open {args.text}: {e.strerror or e}", file=sys.stderr)
        return 2
    graph, clause_level, ontology = args.graph, args.clause_level, args.ontology
    if args.example:
        try:
            example = load_example(args.example)
        except ValueError as e:
            ap.error(str(e))
        graph, clause_level = example["graph"], clause_level or example["clause_level"]
        text, ontology = text or example["text"], ontology or example["ontology"]
    elif graph is None:
        ap.error("give a graph file, or --example NAME")
    try:
        report = lint(graph, clause_level=clause_level, text=text, ontology=ontology)
    except (GraphReadError, ValueError, OSError) as e:
        print(f"contract-graph-linter: {e}", file=sys.stderr)
        return 2
    out = report.to_dict()
    if args.scores:
        out["reference_free_scores"] = reference_free_scores(report.graph.raw, ontology=ontology)
    if args.slice:
        try:
            context = context_slice(report.graph, args.slice, args.direction, args.depth)
        except ValueError as e:
            print(f"contract-graph-linter: {e}", file=sys.stderr)
            return 2
        out["slice"] = {"clause": args.slice, "direction": args.direction, "depth": args.depth, "context": context}

    if args.json:
        print(json.dumps(out, indent=2, default=list))
    else:
        print(report.summary())
        if args.scores:
            print("Reference-free scores:")
            for name, value in out["reference_free_scores"].items():
                print(f"  {name:<24}{value:.3f}")
        if args.slice:
            print(f"Context for clause {args.slice} (direction={args.direction}, depth={args.depth}):")
            for clause, info in out["slice"]["context"].items():
                print(f"  {clause}  [{info['direction']}, distance {info['distance']}]  " + "; ".join(info["via"]))
            if not out["slice"]["context"]:
                print("  (none)")
    return 1 if (args.strict and report.findings) else 0


if __name__ == "__main__":
    sys.exit(main())
