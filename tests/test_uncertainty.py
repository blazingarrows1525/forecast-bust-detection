"""Tests for the cluster bootstrap (D-022).

The property that matters is not that the code runs -- it is that it produces
*wider* intervals than a naive per-row bootstrap on clustered data. A confidence
interval that is too narrow is worse than no interval: it converts "we do not
know" into "we have established", which is the failure this whole project is
about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd.evaluate import metrics as M, uncertainty as U


def _clustered(n_days=80, rows_per_day=120, seed=7):
    """Rare events with a strong shared per-day effect.

    Deliberately mirrors the real structure: one synoptic situation per init
    date makes the forecast hard for every subdivision on that day at once.
    """
    rng = np.random.default_rng(seed)
    day_effect = rng.normal(0.0, 1.2, n_days)
    y, p, clusters = [], [], []
    for d in range(n_days):
        logit = day_effect[d] + rng.normal(0.0, 0.5, rows_per_day) - 3.2
        prob = 1.0 / (1.0 + np.exp(-logit))
        y.append(rng.binomial(1, prob))
        p.append(np.clip(prob + rng.normal(0, 0.02, rows_per_day), 0, 1))
        clusters.append(np.full(rows_per_day, d))
    return (np.concatenate(y).astype(float), np.concatenate(p),
            np.concatenate(clusters))


def test_clustered_interval_is_wider_than_a_per_row_interval():
    """The reason this module exists.

    Resampling rows treats correlated observations as independent. On data with
    a real per-day effect that understates the uncertainty substantially.
    """
    y, p, clusters = _clustered()

    clustered = U.metric_interval(M.auroc, y, p, clusters, n_boot=400)
    per_row = U.metric_interval(M.auroc, y, p, np.arange(len(y)), n_boot=400)

    assert clustered.width > per_row.width * 1.4, (
        f"cluster bootstrap width {clustered.width:.4f} should be materially "
        f"wider than the per-row {per_row.width:.4f}; if it is not, the "
        f"clustering is not being respected"
    )
    assert clustered.n_clusters == 80
    assert per_row.n_clusters == len(y)


def test_interval_brackets_its_own_point_estimate():
    y, p, clusters = _clustered()
    iv = U.metric_interval(M.auroc, y, p, clusters, n_boot=300)
    assert iv.lo <= iv.point <= iv.hi
    assert iv.point == pytest.approx(M.auroc(y, p))


def test_identical_predictors_give_a_zero_margin_that_includes_zero():
    """A guard against a paired comparison that quietly unpairs itself."""
    y, p, clusters = _clustered()
    iv = U.paired_difference(M.auroc, y, p, p, clusters, n_boot=200)
    assert iv.point == pytest.approx(0.0, abs=1e-12)
    assert iv.excludes_zero is False


def test_a_real_difference_is_detected():
    y, p, clusters = _clustered()
    noise = np.random.default_rng(1).uniform(0, 1, len(y))  # AUROC ~0.5
    iv = U.paired_difference(M.auroc, y, p, noise, clusters, n_boot=200)
    assert iv.point > 0.2
    assert iv.excludes_zero is True


def test_excludes_zero_is_false_when_the_interval_straddles_it():
    assert U.Interval(0.025, -0.008, 0.058).excludes_zero is False
    assert U.Interval(0.082, 0.062, 0.104).excludes_zero is True
    assert U.Interval(-0.05, -0.09, -0.01).excludes_zero is True


def test_degenerate_resamples_are_counted_not_silently_dropped():
    """AUROC needs one bust and one non-bust; at a low base rate a draw can miss.

    Those resamples must be visible in the result, because a large count means
    the interval rests on fewer effective samples than requested.
    """
    y = np.array([0.0] * 40 + [1.0])
    p = np.linspace(0, 1, 41)
    clusters = np.arange(41)  # each row its own cluster; the single bust is losable
    iv = U.metric_interval(M.auroc, y, p, clusters, n_boot=200)
    assert iv.n_degenerate > 0
    assert iv.n_degenerate + len([1]) <= iv.n_boot


def test_the_same_seed_gives_the_same_interval():
    y, p, clusters = _clustered()
    a = U.metric_interval(M.auroc, y, p, clusters, n_boot=150, seed=42)
    b = U.metric_interval(M.auroc, y, p, clusters, n_boot=150, seed=42)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_reported_intervals_match_the_committed_artifact():
    """The numbers in the README must trace to a file, not to a memory of a run."""
    import json

    from fbd import config

    path = config.ARTIFACTS / "confidence_intervals.json"
    if not path.exists():
        pytest.skip("run scripts/compute_confidence_intervals.py first")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_init_dates"] < 200, "clusters must be init dates, not rows"

    # The ENS margin is published twice: here (2,000 resamples, for the tables)
    # and in the registered settlement (10,000). D-022 pinned the 40-date
    # finding; D-025 superseded it. What must hold now is that the two files
    # never disagree about the verdict or the dates it rests on.
    ens = data.get("true_ens")
    settlement = config.ARTIFACTS / "ens_settlement.json"
    if ens and settlement.exists():
        raw = ens["margins"]["raw ENS spread"]
        primary = json.loads(settlement.read_text(encoding="utf-8"))["primary"]
        assert ens["n_init_dates"] == primary["n_init_dates"]
        assert raw["point"] == pytest.approx(primary["point"], abs=1e-9)
        assert raw["excludes_zero"] == primary["excludes_zero"], (
            "the interval tables and the registered settlement disagree on "
            "whether the ENS margin excludes zero; re-run both, then fix the README"
        )
