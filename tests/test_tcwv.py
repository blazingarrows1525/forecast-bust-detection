"""B2: the column water vapour integral. Pure numpy, run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.features.tcwv import G, column_water_vapour  # noqa: E402

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def _col(q, ps):
    return column_water_vapour(np.asarray(q, float)[None, :], LEVELS, np.array([ps]))[0]


def test_constant_humidity_integrates_to_q_times_the_column():
    q = np.full(13, 0.01)
    assert _col(q, 100000.0) == pytest.approx(0.01 * (100000.0 - 5000.0) / G)


def test_linear_humidity_is_exact_under_the_trapezoid():
    p = np.array(LEVELS, float) * 100
    q = 1e-7 * p                                 # linear in p
    want = 1e-7 * (100000.0 ** 2 - 5000.0 ** 2) / 2 / G
    assert _col(q, 100000.0) == pytest.approx(want, rel=1e-12)


def test_the_surface_cuts_a_segment_between_levels():
    p = np.array(LEVELS, float) * 100
    q = 1e-7 * p
    ps = 88000.0                                 # between 850 and 925 hPa
    want = 1e-7 * (ps ** 2 - 5000.0 ** 2) / 2 / G
    assert _col(q, ps) == pytest.approx(want, rel=1e-12)


def test_levels_below_the_ground_are_ignored():
    q = np.full(13, 0.004)
    q[-3:] = 99.0                                # 850, 925, 1000 hPa underground on a plateau
    ps = 60000.0                                 # exactly the 600 hPa level
    assert _col(q, ps) == pytest.approx(0.004 * (60000.0 - 5000.0) / G)


def test_a_surface_below_1000_hpa_holds_the_lowest_humidity():
    q = np.full(13, 0.02)
    assert _col(q, 101500.0) == pytest.approx(0.02 * (101500.0 - 5000.0) / G)


def test_level_order_and_axis_do_not_matter():
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 0.02, (4, 13, 3, 2))
    ps = rng.uniform(70000, 101000, (4, 3, 2))
    a = column_water_vapour(q, LEVELS, ps, axis=1)
    rev = column_water_vapour(q[:, ::-1], LEVELS[::-1], ps, axis=1)
    last = column_water_vapour(np.moveaxis(q, 1, -1), LEVELS, ps, axis=-1)
    np.testing.assert_allclose(a, rev, rtol=1e-12)
    np.testing.assert_allclose(a, last, rtol=1e-12)
    assert a.shape == (4, 3, 2) and np.all(a > 0)


def test_a_typical_monsoon_column_is_plausible():
    # q falling roughly exponentially with height, 18 g/kg at the surface
    p = np.array(LEVELS, float)
    q = 0.018 * (p / 1000.0) ** 3.2
    assert 40 < _col(q, 100500.0) < 70


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        column_water_vapour(np.zeros((1, 13)), LEVELS, np.array([4000.0]))
    with pytest.raises(ValueError):
        column_water_vapour(np.zeros((2, 13)), LEVELS, np.array([90000.0]))
