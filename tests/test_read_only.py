"""A public, read-only deployment refuses override writes cleanly.

Hugging Face Spaces run the image as UID 1000, which cannot write the image's
data directory, and a public demo should not take anonymous writes to the
forecaster audit log anyway. deploy/huggingface/Dockerfile sets FBD_READ_ONLY=1.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BODY = {"region_id": "ASSAM_MEGHALAYA", "init_date": "2022-06-14", "lead_day": 3,
        "action": "escalate", "reason": "heavy rain signal", "user": "duty forecaster"}


def _client(db, monkeypatch):
    from fbd.api import app as app_module
    importlib.reload(app_module)
    monkeypatch.setattr(app_module, "DB", db)
    return TestClient(app_module.app)


def _with_overrides_table(db):
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE overrides (id INTEGER PRIMARY KEY AUTOINCREMENT, region_id TEXT, "
                "init_date TEXT, lead_day INTEGER, action TEXT, reason TEXT, user TEXT, "
                "created_at TEXT)")
    con.commit()
    con.close()
    return db


def _n_overrides(db):
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM overrides").fetchone()[0]
    con.close()
    return n


def test_read_only_refuses_the_write_and_says_why(bulletin_store, monkeypatch):
    db = _with_overrides_table(bulletin_store)
    monkeypatch.setenv("FBD_READ_ONLY", "1")
    res = _client(db, monkeypatch).post("/api/override", json=BODY)
    assert res.status_code == 403
    assert "read-only" in res.json()["detail"]
    assert _n_overrides(db) == 0


def test_reads_still_work_when_read_only(bulletin_store, monkeypatch):
    monkeypatch.setenv("FBD_READ_ONLY", "1")
    res = _client(bulletin_store, monkeypatch).get(
        "/api/review-queue", params={"init_date": "2022-06-14", "decision_band_only": "false"})
    assert res.status_code == 200 and len(res.json()) >= 1


def test_a_normal_install_still_records_overrides(bulletin_store, monkeypatch):
    db = _with_overrides_table(bulletin_store)
    monkeypatch.delenv("FBD_READ_ONLY", raising=False)
    res = _client(db, monkeypatch).post("/api/override", json=BODY)
    assert res.status_code == 200 and res.json()["n_overrides"] == 1
    assert _n_overrides(db) == 1
