"""Confidence intervals that respect how this data is actually structured.

Every headline number in this project was a point estimate with nothing
attached to it.  The README asserted a margin of **+0.025 AUROC** over a real
operational ensemble without saying whether that margin could be told apart
from zero, which is the first question a reviewer should ask.

Why a plain bootstrap would be wrong
------------------------------------
The 2022 test set is 39,950 rows but only **122 unique init dates**.  Each init
date contributes 36 subdivisions x 10 lead days, and those rows are not
independent observations: they share one synoptic situation.  When a monsoon
depression sits over the Bay of Bengal, the forecast is hard for every
subdivision downstream of it on that day, and the model is right or wrong about
most of them together.

Resampling rows would treat 39,950 correlated observations as 39,950
independent ones and produce intervals several times too narrow -- confidently
declaring significance that the data does not support.  The unit that is
plausibly exchangeable here is the **init date**, so that is what gets
resampled: draw 122 dates with replacement, take every row belonging to each
drawn date, recompute.  This is the standard cluster (block) bootstrap.

What this does not fix
----------------------
Init dates three days apart are themselves correlated -- weather is
autocorrelated on a synoptic timescale of roughly 3-7 days -- so even 122 dates
overstate the independent information in one monsoon season.  These intervals
should be read as a lower bound on the true uncertainty, not an upper one.  The
honest fix is more test years, not a cleverer resampling scheme; see the
rolling-origin backtest in the open items.

Intervals are percentile intervals.  BCa would correct for skew and bias at
some cost in complexity; with 2,000 resamples over 122 clusters the percentile
interval is the standard choice and its limitations are stated rather than
hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260919


@dataclass
class Interval:
    """A point estimate and the interval around it."""

    point: float
    lo: float
    hi: float
    n_boot: int = 0
    n_clusters: int = 0
    n_degenerate: int = 0
    alpha: float = 0.05

    @property
    def excludes_zero(self) -> bool:
        """True if the whole interval sits on one side of zero.

        Only meaningful for a *difference*.  Reported rather than turned into
        a bare "significant: yes/no", because the interval is the finding and
        the threshold is a convention.
        """
        return (self.lo > 0) or (self.hi < 0)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def as_dict(self) -> dict:
        return {
            "point": self.point, "lo": self.lo, "hi": self.hi,
            "n_boot": self.n_boot, "n_clusters": self.n_clusters,
            "n_degenerate": self.n_degenerate, "alpha": self.alpha,
        }

    def fmt(self, places: int = 3) -> str:
        return (f"{self.point:.{places}f} "
                f"[{self.lo:.{places}f}, {self.hi:.{places}f}]")

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.fmt()


def _cluster_index(cluster_ids) -> list:
    """Row indices grouped by cluster, so a resample is a concatenate."""
    ids = np.asarray(cluster_ids)
    _uniq, inverse = np.unique(ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    edges = np.searchsorted(sorted_inv, np.arange(len(_uniq) + 1))
    return [order[edges[i]:edges[i + 1]] for i in range(len(_uniq))]


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    cluster_ids,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Percentile interval for ``statistic``, resampling whole clusters.

    ``statistic`` takes an array of row indices and returns a float.  Keeping
    it index-based is what lets the same function serve AUROC, Brier, ECE and
    the *difference* between two predictors without special-casing any of them.

    A resample that cannot produce a value -- AUROC needs at least one bust and
    one non-bust, and at a 3.4% base rate a draw can miss -- returns NaN and is
    counted in ``n_degenerate`` rather than silently dropped.
    """
    groups = _cluster_index(cluster_ids)
    n_clusters = len(groups)
    all_rows = np.concatenate(groups) if groups else np.array([], dtype=int)

    point = float(statistic(all_rows))
    rng = np.random.default_rng(seed)

    values: list[float] = []
    degenerate = 0
    for _ in range(n_boot):
        picks = rng.integers(0, n_clusters, n_clusters)
        rows = np.concatenate([groups[i] for i in picks])
        try:
            value = float(statistic(rows))
        except (ValueError, ZeroDivisionError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)
        else:
            degenerate += 1

    if not values:
        return Interval(point, float("nan"), float("nan"), n_boot,
                        n_clusters, degenerate, alpha)

    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), n_boot, n_clusters,
                    degenerate, alpha)


def metric_interval(
    metric: Callable,
    y_true,
    y_prob,
    cluster_ids,
    **kwargs,
) -> Interval:
    """Interval for ``metric(y_true, y_prob)`` under the cluster bootstrap."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)

    def statistic(rows: np.ndarray) -> float:
        return metric(y[rows], p[rows])

    return cluster_bootstrap(statistic, cluster_ids, **kwargs)


def paired_difference(
    metric: Callable,
    y_true,
    prob_a,
    prob_b,
    cluster_ids,
    **kwargs,
) -> Interval:
    """Interval for ``metric(a) - metric(b)`` on identical rows.

    Paired, and that matters: both predictors are scored on the *same* resampled
    rows every time, so the shared difficulty of a given set of days cancels.
    Comparing two independently-bootstrapped intervals instead would inflate the
    uncertainty on the difference and could hide a real effect -- overlapping
    marginal intervals do not imply a difference indistinguishable from zero.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)

    def statistic(rows: np.ndarray) -> float:
        return metric(y[rows], a[rows]) - metric(y[rows], b[rows])

    return cluster_bootstrap(statistic, cluster_ids, **kwargs)


def stratified_mean_difference(
    metric: Callable,
    y_true,
    prob_a,
    prob_b,
    cluster_ids,
    strata,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Mean over strata of ``metric(a) - metric(b)``; clusters resampled within strata.

    Built for the S1b backtest. Each test year has its own fold model and its
    own bust rate, so the margin is computed within a year and then averaged:
    one AUROC over rows pooled from several years would partly reward telling
    the years apart. Every resample draws, for each stratum, as many clusters as
    that stratum has, from that stratum only. With a single stratum this is
    exactly ``paired_difference``.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    clusters = np.asarray(cluster_ids)
    strata = np.asarray(strata)

    groups = []
    for s in sorted(np.unique(strata).tolist(), key=str):
        idx = np.flatnonzero(strata == s)
        groups.append([idx[g] for g in _cluster_index(clusters[idx])])

    def diff(rows: np.ndarray) -> float:
        return metric(y[rows], a[rows]) - metric(y[rows], b[rows])

    point = float(np.mean([diff(np.concatenate(g)) for g in groups]))
    rng = np.random.default_rng(seed)

    values: list[float] = []
    degenerate = 0
    for _ in range(n_boot):
        per = []
        for g in groups:
            picks = rng.integers(0, len(g), len(g))
            rows = np.concatenate([g[i] for i in picks])
            try:
                per.append(float(diff(rows)))
            except (ValueError, ZeroDivisionError):
                per.append(float("nan"))
        value = float(np.mean(per))
        if np.isfinite(value):
            values.append(value)
        else:
            degenerate += 1

    n_clusters = sum(len(g) for g in groups)
    if not values:
        return Interval(point, float("nan"), float("nan"), n_boot,
                        n_clusters, degenerate, alpha)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), n_boot, n_clusters,
                    degenerate, alpha)
