"""One-call entry point: metrics plus findings."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .checks import ClauseText, Finding, run_checks
from .graph import ContractGraph, GraphSource, load_graph
from .ontology import OntologySource
from .metrics import graph_metrics


@dataclass
class LintReport:
    graph: ContractGraph
    metrics: Dict[str, Any]
    findings: List[Finding]

    def counts(self) -> Dict[str, int]:
        return dict(sorted(Counter(f.check for f in self.findings).items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.graph.contract_id,
            "metrics": self.metrics,
            "counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary(self, max_findings: int = 50) -> str:
        m = self.metrics
        lines = [
            f"Contract: {self.graph.contract_id or '(no id)'}",
            f"  nodes {m['nodes']}, edges {m['edges']}, density {m['density']}, "
            f"dependency depth {m['dependency_depth']}, k-core {m['k_core_k']}",
            f"  orphans {m['orphans']} ({m['orphan_ratio']:.0%}), leaves {m['leaves']} ({m['leaf_ratio']:.0%}), "
            f"articulation points {m['articulation_points']}, components {m['components']}",
            f"  findings: {len(self.findings)}" + (
                f"   (records that could not be read: {m['unreadable_node_records'] + m['unreadable_edge_records']})"
                if m["unreadable_node_records"] or m["unreadable_edge_records"] else ""),
        ]
        for f in self.findings[:max_findings]:
            where = f"  [read: {', '.join(f.clauses)}]" if f.clauses else ""
            lines.append(f"  {f.check}: {f.message}{where}")
        if len(self.findings) > max_findings:
            lines.append(f"  ... {len(self.findings) - max_findings} more")
        return "\n".join(lines)


def lint(source: GraphSource, clause_level: bool = False, text: Optional[ClauseText] = None,
         ontology: OntologySource = None) -> LintReport:
    """Compute graph metrics and run every check on a contract graph.

    Pass ``text`` (the clause text, or a mapping from clause id to text) to also
    check that parties, terms and values appear in the text. Pass ``ontology``
    (a dictionary or the path of a JSON file) to add node and edge types to the
    default ontology.
    """
    graph = load_graph(source)
    return LintReport(graph, graph_metrics(graph), run_checks(graph, clause_level=clause_level, text=text, ontology=ontology))
