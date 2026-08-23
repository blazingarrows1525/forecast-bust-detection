"""Tests for drift monitoring.

The monitor is a safety device, so the tests that matter are the ones proving
it *fires* -- a drift detector that silently returns OK is worse than none,
because it manufactures confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fbd.mlops import drift


@pytest.fixture
def rng():
    return np.random.default_rng(20260920)


def test_psi_is_near_zero_for_identically_distributed_samples(rng):
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert drift.population_stability_index(a, b) < drift.PSI_WATCH


def test_psi_detects_a_shifted_distribution(rng):
    a = rng.normal(size=5000)
    b = rng.normal(loc=1.5, size=5000)
    assert drift.population_stability_index(a, b) > drift.PSI_RETRAIN


def test_psi_detects_a_compressed_distribution(rng):
    """The real 2022 case: same centre, much narrower spread.

    This is the shape that india_shear_z actually took in the held-out year,
    and a detector that only looked at means would have missed it entirely.
    """
    a = rng.normal(size=5000)
    b = rng.normal(scale=0.3, size=5000)
    assert drift.population_stability_index(a, b) > drift.PSI_RETRAIN


def test_psi_is_finite_when_a_bin_empties(rng):
    """An empty bin must not send PSI to infinity."""
    a = rng.normal(size=2000)
    b = rng.normal(size=2000) + 12.0  # no overlap at all
    psi = drift.population_stability_index(a, b)
    assert np.isfinite(psi) and psi > drift.PSI_RETRAIN


def test_psi_is_nan_for_samples_too_small_to_bin():
    assert np.isnan(drift.population_stability_index([1.0, 2.0], [1.0, 2.0]))


def test_constant_feature_does_not_report_drift():
    ref = np.ones(1000)
    cur = np.ones(1000)
    assert drift.population_stability_index(ref, cur) == 0.0


def _frame(rng, n=2000, loc=0.0, proba=0.04, label_rate=0.04):
    return pd.DataFrame(
        {
            "f1": rng.normal(loc=loc, size=n),
            "f2": rng.normal(size=n),
            "proba": np.clip(rng.normal(loc=proba, scale=0.01, size=n), 0.001, 1),
            "bust": (rng.random(n) < label_rate).astype(int),
            "is_ood": np.zeros(n, dtype=int),
        }
    )


def test_stable_data_yields_ok(rng):
    ref, cur = _frame(rng), _frame(rng)
    report = drift.assess(ref, cur, ["f1", "f2"], "proba", "bust", "is_ood")
    assert report.verdict is drift.Verdict.OK


def test_feature_shift_escalates_to_retrain(rng):
    ref, cur = _frame(rng), _frame(rng, loc=2.0)
    report = drift.assess(ref, cur, ["f1", "f2"], "proba")
    assert report.verdict is drift.Verdict.RETRAIN
    assert "f1" in report.feature_psi and report.feature_psi["f1"] > drift.PSI_RETRAIN


def test_calibration_drift_is_detected_when_labels_exist(rng):
    """Predicting 4% while 1% occurs is a decision-relevant failure."""
    ref = _frame(rng)
    cur = _frame(rng, proba=0.04, label_rate=0.01)
    report = drift.assess(ref, cur, ["f1", "f2"], "proba", "bust")
    assert report.verdict is drift.Verdict.RETRAIN
    assert report.calibration_ratio < 0.5
    assert any("calibration" in r for r in report.reasons)


def test_elevated_ood_rate_escalates(rng):
    ref = _frame(rng)
    cur = _frame(rng)
    cur["is_ood"] = (rng.random(len(cur)) < 0.12).astype(int)
    report = drift.assess(ref, cur, ["f1", "f2"], "proba", ood_col="is_ood")
    assert report.verdict is drift.Verdict.RETRAIN
    assert report.ood_rate > drift.OOD_RETRAIN


def test_monitor_works_without_labels(rng):
    """Labels lag verification, so the monitor must say something before them."""
    ref, cur = _frame(rng), _frame(rng, loc=2.0)
    report = drift.assess(ref, cur, ["f1", "f2"])
    assert report.verdict is drift.Verdict.RETRAIN
    assert report.calibration_ratio is None


def test_verdict_never_de_escalates(rng):
    """A later clean check must not erase an earlier severe one."""
    ref = _frame(rng)
    cur = _frame(rng, loc=2.0)  # severe feature drift
    cur["bust"] = (cur["proba"] > 0).astype(int) * 0  # calibration would say fine
    report = drift.assess(ref, cur, ["f1", "f2"], "proba", "bust", "is_ood")
    assert report.verdict is drift.Verdict.RETRAIN


def test_report_serialises_for_an_operator(rng):
    ref, cur = _frame(rng), _frame(rng, loc=1.0)
    payload = drift.assess(ref, cur, ["f1", "f2"], "proba", "bust").as_dict()
    assert payload["verdict"] in {"OK", "WATCH", "RETRAIN"}
    assert isinstance(payload["reasons"], list) and payload["reasons"]
    assert isinstance(payload["top_drifting_features"], list)
