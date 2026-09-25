"""/api/metrics carries every headline number the landing page shows."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from fbd import config  # noqa: E402
from fbd.api import app as app_module  # noqa: E402


def _client(monkeypatch, tmp_path, with_settlement=True, with_store=True,
            with_backtest=True):
    (tmp_path / "results.json").write_text(json.dumps({"overall": {}}))
    (tmp_path / "confidence_intervals.json").write_text(
        json.dumps({"true_ens": {"n_init_dates": 40}}))
    if with_settlement:
        (tmp_path / "ens_settlement.json").write_text(
            json.dumps({"primary": {"verdict": "indistinguishable"}}))
    if with_backtest:
        (tmp_path / "backtest.json").write_text(
            json.dumps({"primary": {"verdict": "indistinguishable", "years": [2019, 2020, 2021]}}))
    db = tmp_path / "bulletins.sqlite"
    if with_store:
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE bulletins (init_date TEXT, status TEXT, actual_bust REAL)")
        con.executemany("INSERT INTO bulletins VALUES (?,?,?)",
                        [("2022-06-01", "OK", 0)] * 9 + [("2022-06-01", "OK", 1)]
                        + [("2022-06-01", "OUT_OF_DISTRIBUTION", 1)]
                        + [("2021-06-01", "OK", 1)] * 50)
        con.commit()
        con.close()
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(app_module, "DB", db)
    return TestClient(app_module.app)


def test_metrics_carries_every_block(monkeypatch, tmp_path):
    body = _client(monkeypatch, tmp_path).get("/api/metrics").json()
    assert body["confidence_intervals"]["true_ens"]["n_init_dates"] == 40
    assert body["ens_settlement"]["primary"]["verdict"] == "indistinguishable"
    assert body["backtest"]["primary"]["years"] == [2019, 2020, 2021]
    assert body["test_year"] == config.TEST_YEARS[0]
    r = body["refusal"]
    assert r["year"] == config.TEST_YEARS[0]
    assert r["scored"] == {"n": 10, "bust_rate": 0.1}, "other years must not leak in"
    assert r["refused"]["bust_rate"] == 1.0 and r["ratio"] == 10.0


def test_missing_pieces_are_null_not_errors(monkeypatch, tmp_path):
    body = _client(monkeypatch, tmp_path, with_settlement=False,
                   with_store=False, with_backtest=False).get("/api/metrics").json()
    assert body["ens_settlement"] is None
    assert body["backtest"] is None
    assert body["refusal"] is None
