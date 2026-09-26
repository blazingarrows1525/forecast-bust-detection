"""The API serves the combination's ingredients, and tolerates the old store."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PARAMS = {"region_id": "ASSAM_MEGHALAYA", "valid_date": "2022-06-17"}


def _client(db, monkeypatch):
    from fbd.api import app as app_module
    importlib.reload(app_module)
    monkeypatch.setattr(app_module, "DB", db)
    return TestClient(app_module.app)


def test_bulletin_carries_model_probability_and_spread(bulletin_store, monkeypatch):
    rows = _client(bulletin_store, monkeypatch).get("/api/convergence", params=PARAMS).json()
    assert rows[0]["model_probability"] == 0.66 and rows[0]["ens_spread_mm"] == 7.9
    assert rows[0]["model_version"] == "test-fixture"


def test_old_store_reads_as_null(bulletin_store_v1, monkeypatch):
    rows = _client(bulletin_store_v1, monkeypatch).get("/api/convergence", params=PARAMS).json()
    assert rows[0]["model_probability"] is None and rows[0]["ens_spread_mm"] is None
    assert rows[0]["bust_probability"] == 0.706


def test_review_queue_carries_the_ingredients(bulletin_store, monkeypatch):
    q = _client(bulletin_store, monkeypatch).get(
        "/api/review-queue", params={"init_date": "2022-06-14"}).json()
    assert q[0]["model_probability"] == 0.66 and q[0]["ens_spread_mm"] == 7.9


def test_tools_ground_the_new_numbers(bulletin_store):
    from fbd.genai import tools
    _res, grounded = tools.dispatch("get_bulletin", {"region_id": "ASSAM_MEGHALAYA",
                                                     "init_date": "2022-06-14"})
    assert 0.66 in grounded and 7.9 in grounded


def test_tools_still_work_on_the_old_store(bulletin_store_v1, monkeypatch):
    from fbd.genai import tools
    monkeypatch.setattr(tools, "DB", bulletin_store_v1)
    res, _g = tools.dispatch("get_review_queue", {"init_date": "2022-06-14"})
    assert res["items"][0]["bust_probability"] == 0.706
