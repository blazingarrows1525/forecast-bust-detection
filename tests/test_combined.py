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
