"""Tests for the 3-D command centre's data endpoint (D-018).

The 3-D page itself is verified in a real browser (WebGL 2.0 context, draw
calls, colour histogram) because a headless assertion cannot prove that pixels
reached a GPU. What *is* unit-testable is the contract the page depends on:
the shape of the risk cube, the parallel-array alignment, and the invariant
that an OOD cell never carries a probability.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config  # noqa: E402
from fbd.api import app as app_module  # noqa: E402


pytestmark = pytest.mark.skipif(
    not (config.ARTIFACTS / "bulletins.sqlite").exists(),
    reason="bulletin store not built; run scripts/generate_bulletins.py",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app)


@pytest.fixture(scope="module")
def an_init_date():
    con = sqlite3.connect(config.ARTIFACTS / "bulletins.sqlite")
    row = con.execute("SELECT MIN(init_date) FROM bulletins").fetchone()
    con.close()
    return row[0]


def test_risk_cube_shape_is_regions_by_leads(client, an_init_date):
    r = client.get(f"/api/risk-cube?init_date={an_init_date}")
    assert r.status_code == 200
    d = r.json()

    assert d["lead_days"] == list(config.LEAD_DAYS)
    assert d["decision_band"] == list(config.DECISION_BAND)
    assert d["n_regions"] == len(d["regions"])
    # Island subdivisions are excluded from modelling (D-006), so the cube is
    # 34 x 10, not 36 x 10.
    assert d["n_regions"] == 34

    n = len(d["lead_days"])
    for reg in d["regions"]:
        for key in ("p", "baseline", "status", "actual", "fcst_mm", "obs_mm", "valid_date"):
            assert len(reg[key]) == n, f"{reg['region_id']}.{key} misaligned"


def test_every_region_has_a_finite_placeable_centroid(client, an_init_date):
    """A NaN centroid would silently drop a column, not raise."""
    d = client.get(f"/api/risk-cube?init_date={an_init_date}").json()
    for reg in d["regions"]:
        assert isinstance(reg["lon"], float) and isinstance(reg["lat"], float)
        assert reg["lon"] == reg["lon"] and reg["lat"] == reg["lat"]  # not NaN
        # Inside the India bounding box from LOGIC.md sec 3.1.
        assert config.INDIA_BBOX["lon_min"] <= reg["lon"] <= config.INDIA_BBOX["lon_max"]
        assert config.INDIA_BBOX["lat_min"] <= reg["lat"] <= config.INDIA_BBOX["lat_max"]


def test_ood_cells_never_carry_a_probability(client, an_init_date):
    """The refusal invariant (LOGIC.md sec 11.1), enforced at the cube level.

    The 3-D view colours purely from these arrays, so a probability leaking
    through beside a non-OK status would render a confident-looking column on a
    state the system had declined to score.
    """
    d = client.get(f"/api/risk-cube?init_date={an_init_date}").json()
    checked = 0
    for reg in d["regions"]:
        for status, p in zip(reg["status"], reg["p"]):
            if status != "OK":
                assert p is None, f"{reg['region_id']} status={status} but p={p}"
                checked += 1
            else:
                assert p is None or 0.0 <= p <= 1.0
    assert checked >= 0  # informational; dates with no OOD cells are legitimate


def test_probabilities_match_the_per_region_bulletin(client, an_init_date):
    """The cube is a projection of the bulletin, so it must not disagree with it."""
    d = client.get(f"/api/risk-cube?init_date={an_init_date}").json()
    reg = d["regions"][0]
    rid = reg["region_id"]

    detail = client.get(f"/api/bulletin/{rid}?init_date={an_init_date}").json()
    by_lead = {row["lead_day"]: row for row in detail}
    for i, lead in enumerate(d["lead_days"]):
        assert by_lead[lead]["bust_probability"] == reg["p"][i]
        assert by_lead[lead]["status"] == reg["status"][i]


def test_unknown_init_date_is_404_not_500(client):
    r = client.get("/api/risk-cube?init_date=1999-01-01")
    assert r.status_code == 404


def test_malformed_mode_is_rejected_by_validation(client, an_init_date):
    r = client.get(f"/api/risk-cube?init_date={an_init_date}&mode=wormhole")
    assert r.status_code == 422
