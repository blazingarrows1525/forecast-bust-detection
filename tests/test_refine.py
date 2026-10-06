"""B3/B4 rules and the beta-family combiner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import refine as RF  # noqa: E402


def test_tail_gap_reads_the_top_decile_only():
    p = np.r_[np.full(90, 0.01), np.full(10, 0.5)]
    y = np.r_[np.zeros(90), np.r_[np.ones(3), np.zeros(7)]]
    assert RF.tail_gap(y, p) == pytest.approx(0.2)        # predicts 0.5, observes 0.3
    p2 = p.copy()
    p2[:90] = 0.4                                          # the bulk moves; the top does not
    assert RF.tail_gap(y, p2) == pytest.approx(0.2)


@pytest.mark.parametrize("lo, hi, v", [(0.001, 0.01, "short_lead_fit_better"),
                                       (-0.02, -0.001, "extrapolation_better"),
                                       (-0.01, 0.01, "indistinguishable")])
def test_b4_verdict(lo, hi, v):
    assert RF.b4_verdict(lo, hi) == v
    assert "Days 1–2" in RF.B4_TEXT[v]


@pytest.mark.parametrize("tail, brier, v", [
    ({"lo": -0.03, "hi": -0.01}, {"lo": -0.001, "hi": 0.001}, "adopt"),
    ({"lo": -0.03, "hi": -0.01}, {"lo": 0.0002, "hi": 0.001}, "reject"),   # Brier clearly worse
    ({"lo": 0.002, "hi": 0.01}, {"lo": -0.001, "hi": 0.001}, "reject"),    # tail clearly wider
    ({"lo": -0.03, "hi": 0.01}, {"lo": -0.001, "hi": 0.001}, "not_established"),
])
def test_b3_verdict(tail, brier, v):
    assert RF.b3_verdict(tail, brier) == v
    assert "2019–2022" in RF.B3_TEXT[v]


def test_the_beta_combiner_nests_the_served_form():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.6, 20000)
    s = rng.gamma(2.0, 3.0, 20000)
    z = 0.7 * np.log(p / (1 - p)) + 0.5 * np.log1p(s) - 2.0
    y = (rng.uniform(size=p.size) < 1 / (1 + np.exp(-z))).astype(int)
    c = RF.BetaCombiner().fit(p, s, y).coefficients()
    assert c["ln_p"] == pytest.approx(0.7, abs=0.1)
    assert c["ln_1mp"] == pytest.approx(-0.7, abs=0.3)     # equal and opposite: the logit
    assert c["log1p_ens_spread"] == pytest.approx(0.5, abs=0.1)
    q = RF.BetaCombiner().fit(p, s, y).predict_proba(p, s)
    assert q.shape == p.shape and np.all((q > 0) & (q < 1))
