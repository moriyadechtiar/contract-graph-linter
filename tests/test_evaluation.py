"""The evaluation script runs and, on the bundled examples, every planted defect is found."""
import importlib.util
from pathlib import Path

from contract_graph_linter.examples import load_example

SCRIPT = Path(__file__).resolve().parent.parent / "evaluation" / "evaluate_linter.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_linter", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_on_bundled_examples():
    ev = load_script()
    invoice, nda = load_example("invoice"), load_example("nda")
    records = [
        {"contract_id": "supply", "clause": {"id": "8.2", "text": invoice["text"]}, "gold_graph": invoice["graph"],
         "prediction": {"graph": invoice["graph"]}},
        {"contract_id": "nda", "clause": {"id": "4.1", "text": "4.1 The Recipient may disclose Confidential Information to its Affiliate."},
         "gold_graph": nda["graph"], "prediction": {"graph": nda["graph"]}},
    ]
    result = ev.evaluate(records, trials=3, seed=1)
    assert result["clauses"] == 2 and len(result["contracts"]) == 2
    for label, expected, level, outcomes in result["defects"]:
        assert outcomes, label
        assert all(o["detected"] for o in outcomes), label
    for label, outcomes in result["controls"]:
        assert not any(o["false_alarm"] or o["finding_lost"] or o["scores_changed"] for o in outcomes), label
    assert all(result["pairs"]) and all(outcome != "PASSED QUIETLY" for _, outcome, _ in result["unusable"])
    report = ev.render(result, trials=3, seed=1)
    assert "## Planted defects" in report and "## Limits of this evaluation" in report
