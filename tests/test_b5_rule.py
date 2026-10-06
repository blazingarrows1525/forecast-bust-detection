"""B5 serving-decision rule: pure, run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import b5  # noqa: E402
from fbd.evaluate import combine as C  # noqa: E402


@pytest.mark.parametrize("lo, hi, v", [(0.001, 0.01, "spatial_better"),
                                       (-0.01, -0.001, "served_better"),
                                       (-0.004, 0.006, "indistinguishable"),
                                       (0.0, 0.01, "indistinguishable")])
def test_verdict_text_and_consequence(lo, hi, v):
    assert b5.verdict(lo, hi) == v
    assert "2019–2022" in b5.TEXT[v]
    assert v in b5.CONSEQUENCE


def test_only_a_win_makes_the_spatial_model_eligible():
    assert "eligible" in b5.CONSEQUENCE["spatial_better"]
    for v in ("served_better", "indistinguishable"):
        assert b5.CONSEQUENCE[v] == "the served product stays XGBoost + ENS"


def test_years_match_the_slate_and_s1c():
    assert b5.YEARS == (2019, 2020, 2021, 2022)
    assert b5.ENS_YEARS == C.PRIMARY_YEARS
