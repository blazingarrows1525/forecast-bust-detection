"""The S1b interval: margins within each year, averaged; dates resampled within year."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import uncertainty as U  # noqa: E402


def mean_p(y, p):
    return float(np.mean(p))


def _data(seed=0):
    rng = np.random.default_rng(seed)
    strata = np.repeat([2019, 2020, 2021], [60, 45, 30])
    clusters = np.array([f"{s}-{i // 3}" for s, i in zip(strata, range(len(strata)))])
    y = rng.integers(0, 2, len(strata)).astype(float)
    a = rng.random(len(strata))
    b = rng.random(len(strata))
    return y, a, b, clusters, strata


def test_same_seed_same_interval():
    args = _data()
    one = U.stratified_mean_difference(mean_p, *args, n_boot=300, seed=7)
    two = U.stratified_mean_difference(mean_p, *args, n_boot=300, seed=7)
    assert (one.point, one.lo, one.hi) == (two.point, two.lo, two.hi)


def test_point_is_the_mean_of_per_stratum_differences():
    y, a, b, clusters, strata = _data()
    iv = U.stratified_mean_difference(mean_p, y, a, b, clusters, strata, n_boot=50)
    per = [a[strata == s].mean() - b[strata == s].mean() for s in (2019, 2020, 2021)]
    assert abs(iv.point - float(np.mean(per))) < 1e-12


def test_strata_are_averaged_not_pooled():
    """Constant per stratum, unequal sizes: averaging gives exactly 2 every time;
    pooling rows would give a row-weighted 1.667 and a non-zero width."""
    strata = np.repeat(["A", "B"], [30, 15])
    clusters = np.array([f"{s}{i // 3}" for s, i in zip(strata, range(45))])
    a = np.where(strata == "A", 1.0, 3.0)
    iv = U.stratified_mean_difference(mean_p, np.zeros(45), a, np.zeros(45),
                                      clusters, strata, n_boot=200)
    assert iv.point == 2.0 and iv.lo == 2.0 and iv.hi == 2.0


def test_one_stratum_is_the_ordinary_paired_bootstrap():
    y, a, b, clusters, _ = _data()
    one = np.zeros(len(y))
    s = U.stratified_mean_difference(mean_p, y, a, b, clusters, one, n_boot=300, seed=3)
    p = U.paired_difference(mean_p, y, a, b, clusters, n_boot=300, seed=3)
    assert (s.point, s.lo, s.hi) == (p.point, p.lo, p.hi)


def test_counts_clusters_across_strata_and_degenerate_resamples():
    y, a, b, clusters, strata = _data()
    iv = U.stratified_mean_difference(mean_p, y, a, b, clusters, strata, n_boot=20)
    assert iv.n_clusters == len(np.unique(clusters))

    def nan_metric(y, p):
        return float("nan")

    bad = U.stratified_mean_difference(nan_metric, y, a, b, clusters, strata, n_boot=20)
    assert bad.n_degenerate == 20 and np.isnan(bad.lo)
