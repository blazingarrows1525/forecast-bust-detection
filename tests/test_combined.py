"""The served combination: pure arithmetic, readable reasons, honest fallback."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model import combined as K  # noqa: E402

C = {"logit_p_model": 0.5, "log1p_ens_spread": 1.0, "intercept": -4.0, "ref_log1p_spread": 2.0}


def test_apply_is_the_logistic_arithmetic():
    p = K.apply(C, [0.2], [np.expm1(2.0)])[0]
    z = 0.5 * np.log(0.2 / 0.8) + 1.0 * 2.0 - 4.0
    assert p == pytest.approx(1 / (1 + np.exp(-z)))


def test_missing_spread_gives_nan_so_the_caller_falls_back():
    assert np.isnan(K.apply(C, [0.2], [np.nan])[0])


def test_contribution_is_relative_to_the_fit_years_median():
    assert K.contribution(C, [np.expm1(2.0)])[0] == pytest.approx(0.0)
    assert K.contribution(C, [np.expm1(3.0)])[0] == pytest.approx(1.0)


@pytest.mark.parametrize("contrib, words", [(0.9, "high"), (-0.9, "low"), (0.05, "typical")])
def test_ensemble_reason_says_what_the_ensemble_did(contrib, words):
    line = K.ens_reason(9.4, contrib)
    assert "50-member ensemble spread" in line and words in line and "9.4 mm/day" in line
    if words == "high":
        assert "raises the bust odds 2.5x" in line
    if words == "low":
        assert "lowers the bust odds 2.5x" in line


def test_agrees_with_the_fitted_combiner():
    pytest.importorskip("sklearn")
    from fbd.evaluate.combine import Combiner

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    y = (a + b + rng.normal(0, 1, 500) > 1).astype(float)
    p_raw, spread = 1 / (1 + np.exp(-a)), np.expm1(np.clip(b + 3, 0, None))
    fitted = Combiner().fit(p_raw, spread, y)
    c = {**fitted.coefficients(), "ref_log1p_spread": 0.0}
    assert np.allclose(K.apply(c, p_raw, spread), fitted.predict_proba(p_raw, spread))


# ---------------------------------------------------------------- v2 (store v0.3.0)
V2 = {"version": 2, "form": "beta",
      "main": {"ln_p": 0.2, "ln_1mp": -0.8, "log1p_ens_spread": 1.0, "intercept": -5.0},
      "short": {"logit_p_model": 0.6, "log1p_ens_spread": 0.5, "intercept": -3.5},
      "short_leads": [1, 2], "ref_log1p_spread": 2.0}


def test_v2_beta_main_and_logit_short_by_lead():
    p, s = np.array([0.2, 0.2, 0.2]), np.full(3, np.expm1(2.0))
    got = K.apply(V2, p, s, lead=np.array([1, 3, 9]))
    zb = 0.2 * np.log(0.2) - 0.8 * np.log(0.8) + 2.0 - 5.0
    zs = 0.6 * np.log(0.2 / 0.8) + 0.5 * 2.0 - 3.5
    assert got[0] == pytest.approx(1 / (1 + np.exp(-zs)))
    assert got[1] == pytest.approx(1 / (1 + np.exp(-zb))) == got[2]


def test_v2_without_a_short_set_or_leads_uses_main_everywhere():
    no_short = {**V2, "short": None}
    p, s = np.array([0.1, 0.3]), np.array([5.0, 9.0])
    assert np.allclose(K.apply(no_short, p, s, lead=[1, 5]), K.apply(V2, p, s))


def test_v2_model_term_and_contribution_follow_the_set():
    term = K.model_term(V2, [0.2, 0.2], lead=[2, 5])
    assert term[0] == pytest.approx(0.6 * np.log(0.25))
    assert term[1] == pytest.approx(0.2 * np.log(0.2) - 0.8 * np.log(0.8))
    con = K.contribution(V2, [np.expm1(3.0)] * 2, lead=[1, 6])
    assert con[0] == pytest.approx(0.5) and con[1] == pytest.approx(1.0)


def test_v1_is_unchanged_by_the_lead_argument():
    p, s = np.array([0.2, 0.4]), np.array([3.0, 8.0])
    assert np.array_equal(K.apply(C, p, s), K.apply(C, p, s, lead=[1, 5]))


def test_v2_beta_agrees_with_the_fitted_beta_combiner():
    pytest.importorskip("sklearn")
    from fbd.evaluate.refine import BetaCombiner

    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.6, 3000)
    s = rng.gamma(2.0, 3.0, 3000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    b = BetaCombiner().fit(p, s, y)
    c = {"version": 2, "form": "beta", "main": b.coefficients(), "short": None,
         "short_leads": [1, 2], "ref_log1p_spread": 0.0}
    assert np.allclose(K.apply(c, p, s, lead=np.full(p.size, 4)), b.predict_proba(p, s))
