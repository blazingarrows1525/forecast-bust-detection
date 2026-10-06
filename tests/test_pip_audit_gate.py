"""C3: the dependency-audit gate. Pure, run in CI."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_pip_audit as G  # noqa: E402

TODAY = dt.date(2026, 10, 6)


def _report(*vulns):
    deps = [{"name": "fastapi", "version": "0.141.1", "vulns": []}]
    for name, vid, aliases in vulns:
        deps.append({"name": name, "version": "1.0", "vulns": [{"id": vid, "aliases": aliases}]})
    return {"dependencies": deps}


def _entry(**kw):
    e = {"id": "PYSEC-1", "package": "uvicorn", "reason": "not reachable", "expires": "2026-12-31"}
    e.update(kw)
    return e


def test_a_clean_report_passes():
    assert G.evaluate(_report(), [], TODAY) == []


def test_any_finding_fails():
    problems = G.evaluate(_report(("uvicorn", "PYSEC-1", ["CVE-2026-1"])), [], TODAY)
    assert len(problems) == 1 and "PYSEC-1" in problems[0]


def test_an_allowlisted_finding_passes_by_id_or_alias():
    rep = _report(("uvicorn", "PYSEC-1", ["CVE-2026-1"]))
    assert G.evaluate(rep, [_entry()], TODAY) == []
    assert G.evaluate(rep, [_entry(id="CVE-2026-1")], TODAY) == []


def test_the_allowlist_is_per_package():
    rep = _report(("uvicorn", "PYSEC-1", []))
    assert G.evaluate(rep, [_entry(package="pydantic")], TODAY)


def test_an_expired_exception_fails_even_without_the_finding():
    problems = G.evaluate(_report(), [_entry(expires="2026-10-01")], TODAY)
    assert problems and "expired" in problems[0]


def test_an_incomplete_exception_fails():
    assert G.evaluate(_report(), [_entry(reason="")], TODAY)
    assert G.evaluate(_report(), [_entry(expires="soon")], TODAY)


def test_a_missing_or_empty_report_fails(tmp_path):
    assert G.main([str(tmp_path / "absent.json")]) == 1
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"dependencies": []}), encoding="utf-8")
    assert G.main([str(empty)]) == 1


def test_the_committed_allowlist_is_valid_and_current():
    allow = json.loads(G.ALLOWLIST.read_text(encoding="utf-8"))
    assert isinstance(allow, list)
    assert G.evaluate(_report(), allow, dt.date.today()) == []
