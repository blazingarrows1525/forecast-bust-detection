"""The versioned evaluation set, run offline against mocks on every CI run.

A regression in routing, abstention, citation validity, number or directive
checks, or injection isolation fails here with the case id that broke. Add a
case to tests/assistant_eval/cases.jsonl for every failure found in use.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _harness():
    spec = importlib.util.spec_from_file_location("evaluate_assistant", ROOT / "scripts" / "evaluate_assistant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_eval_case_passes_in_mock_mode(tmp_path, capsys):
    assert _harness().main(["--label", "ci-mock"]) == 0, capsys.readouterr().out[-3000:]


def test_eval_set_covers_the_required_kinds():
    cases = [json.loads(ln) for ln in (ROOT / "tests" / "assistant_eval" / "cases.jsonl").read_text().splitlines() if ln.strip()]
    ids = {c["id"] for c in cases}
    for required in ("safety-no-place", "current-no-results", "fabricated-number", "fabricated-citation",
                     "injected-page", "ssrf-result-dropped", "unavailable-feature", "ambiguous",
                     "row-explain", "row-refused-not-low"):
        assert required in ids
    assert len(ids) == len(cases), "case ids are unique"
