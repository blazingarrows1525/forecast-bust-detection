"""C2: the UI capture gate. Pure, run in CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_captures as G  # noqa: E402


def _cap(route="landing", vp="1440x900", **kw):
    r = {"route": route, "viewport": vp, "title": "Forecast Bust Intelligence",
         "overflowX": False, "console": [], "exceptions": [], "failed": []}
    r.update(kw)
    return r


def _full():
    return [_cap(r, v) for r in ("landing", "index", "command", "volume")
            for v in ("1440x900", "1280x800", "1024x768", "768x1024", "390x844")]


def test_a_clean_set_passes():
    assert G.problems(_full()) == []


def test_each_defect_fails():
    for kw in ({"overflowX": True}, {"console": ["TypeError: x"]},
               {"exceptions": ["boom"]}, {"failed": ["/api/metrics 500"]}, {"title": ""}):
        caps = _full()
        caps[3] = _cap("landing", "768x1024", **kw)
        assert len(G.problems(caps)) == 1, kw


def test_a_missing_capture_fails():
    assert G.problems(_full()[:-1])


def test_an_unreadable_report_fails(tmp_path):
    assert G.main([str(tmp_path / "absent.json")]) == 1
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(_full()), encoding="utf-8")
    assert G.main([str(ok)]) == 0
