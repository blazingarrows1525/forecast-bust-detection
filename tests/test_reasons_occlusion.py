"""Store v0.3.0: occlusion reasons reuse the TreeSHAP sentences and add the two images."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.explain.reasons import OcclusionExplainer  # noqa: E402

KEYS = ["lagged_spread", "tcwv", "window_pattern", "synoptic_map"]


def _ref():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"subdivision_id": ["A"] * 200,
                         "lagged_spread": rng.gamma(2, 3, 200), "tcwv": rng.normal(50, 5, 200)})


def _rows():
    return pd.DataFrame({"subdivision_id": ["A", "A"], "lagged_spread": [30.0, 1.0],
                         "tcwv": [62.0, 40.0]})


def test_the_window_can_lead_the_reasons():
    ex = OcclusionExplainer(KEYS, _ref())
    out = ex.explain_contributions(_rows(), np.array([[0.2, 0.1, 0.9, -0.3],
                                                      [-0.1, -0.2, -0.3, -0.1]]))
    assert out[0][0].startswith("the network reads the forecast's rainfall pattern")
    assert any("successive model runs disagree" in r for r in out[0])
    assert not any("monsoon map" in r for r in out[0])          # it lowered the risk
    assert out[1] == ["no single dominant factor -- risk comes from the combination of conditions"]


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        OcclusionExplainer(KEYS, _ref()).explain_contributions(_rows(), np.zeros((2, 3)))
