"""B2 non-inferiority rule and gates: pure, run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import b2  # noqa: E402


@pytest.mark.parametrize("lo, hi, v", [(-0.004, 0.01, "non_inferior"),
                                       (0.001, 0.02, "non_inferior"),
                                       (-0.02, -0.006, "inferior"),
                                       (-0.012, 0.003, "inconclusive"),
                                       (-0.005, 0.01, "inconclusive")])
def test_verdict(lo, hi, v):
    assert b2.verdict(lo, hi) == v
    assert "2019–2022" in b2.VERDICT_TEXT[v]


def test_the_registered_constants():
    assert b2.MARGIN == 0.005 and b2.YEARS == (2019, 2020, 2021, 2022)
    assert set(b2.GATE_YEARS).isdisjoint(b2.YEARS)
    assert b2.G1 == {"rmse_max": 2.0, "abs_bias_max": 1.5}
    assert b2.G2 == {"r_min": 0.95, "rmse_max": 4.0}


def test_field_agreement_and_gates():
    rng = np.random.default_rng(0)
    ref = rng.normal(50, 10, 10_000)
    good = b2.field_agreement(ref + rng.normal(0.5, 1.0, ref.size), ref)
    assert good["rmse"] == pytest.approx(np.sqrt(1.25), rel=0.05)
    assert good["bias"] == pytest.approx(0.5, abs=0.05)
    assert b2.gate_g1(good) and b2.gate_g2(good)
    biased = b2.field_agreement(ref + 2.0, ref)
    assert not b2.gate_g1(biased) and b2.gate_g2(biased)
    noisy = b2.field_agreement(ref + rng.normal(0, 8.0, ref.size), ref)
    assert not b2.gate_g2(noisy)
    nan = ref.copy()
    nan[:10] = np.nan
    assert b2.field_agreement(nan, ref)["n"] == ref.size - 10
