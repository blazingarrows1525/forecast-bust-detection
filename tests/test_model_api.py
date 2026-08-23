"""Tests for metrics, OOD detector, explanation templates, and API schemas.

These tests ensure that decision metrics, calibration, out-of-distribution detection,
TreeSHAP explainability templates, and Pydantic API contracts behave correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config
from fbd.evaluate import metrics
from fbd.ood.detector import MahalanobisOOD
from fbd.explain import reasons
from fbd.api import schema


# ---------------------------------------------------------------------- Metrics
def test_metrics_asymmetric_cost_and_value():
    y_true = np.array([0, 0, 0, 1, 1])
    # Perfect predictions:
    res_perfect = metrics.decision_cost(y_true, [0.0, 0.0, 0.0, 1.0, 1.0], threshold=0.5)
    assert res_perfect["fn"] == 0
    assert res_perfect["fp"] == 0
    assert res_perfect["cost"] == 0.0
    assert res_perfect["pod"] == 1.0

    # All negative (misses both busts):
    res_miss = metrics.decision_cost(y_true, [0.0, 0.0, 0.0, 0.0, 0.0], threshold=0.5)
    assert res_miss["fn"] == 2
    assert res_miss["cost"] == 2 * config.COST_MISSED_BUST


def test_metrics_brier_and_bss():
    y_true = np.array([0, 0, 1, 1])
    # Perfect forecast
    assert metrics.brier(y_true, [0.0, 0.0, 1.0, 1.0]) == 0.0
    assert metrics.brier_skill_score(y_true, [0.0, 0.0, 1.0, 1.0]) == 1.0

    # Climatology forecast
    clim_bss = metrics.brier_skill_score(y_true, [0.5, 0.5, 0.5, 0.5])
    assert np.isclose(clim_bss, 0.0)


def test_metrics_ece_and_reliability():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.array([0.1] * 50 + [0.9] * 50)
    rc = metrics.reliability_curve(y_true, y_prob, n_bins=5)
    assert len(rc) > 0
    assert "mean_predicted" in rc.columns
    assert "observed_frequency" in rc.columns
    ece = metrics.expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece >= 0.0


# ----------------------------------------------------------------- OOD Detector
def test_mahalanobis_ood_detects_extreme_outliers():
    rng = np.random.default_rng(42)
    feats = ["feat_a", "feat_b", "feat_c"]
    # Train data: standard normal
    train_data = pd.DataFrame(rng.normal(0, 1, size=(500, 3)), columns=feats)
    ood = MahalanobisOOD(quantile=0.99)
    ood.fit(train_data, feats)

    # In-distribution test points
    normal_test = pd.DataFrame(rng.normal(0, 1, size=(20, 3)), columns=feats)
    pred_normal = ood.is_ood(normal_test)
    assert (pred_normal == False).sum() >= 15  # Most should not be OOD

    # Extreme outlier test points (+15 standard deviations)
    outlier_test = pd.DataFrame(np.full((5, 3), 15.0), columns=feats)
    pred_outliers = ood.is_ood(outlier_test)
    assert pred_outliers.all()  # All must be flagged as OOD


# -------------------------------------------------------- Explanation Templates
def test_reason_templates_structure_and_coverage():
    for feat, (label, high_t, low_t) in reasons.TEMPLATES.items():
        assert isinstance(label, str) and len(label) > 0
        assert isinstance(high_t, str) and len(high_t) > 0
        if low_t is not None:
            assert isinstance(low_t, str) and len(low_t) > 0


def test_concept_families_cover_templates():
    all_family_features = set(reasons.FAMILY.keys())
    # Check that all features in templates are in family dict
    for key in reasons.TEMPLATES.keys():
        assert key in all_family_features, f"{key} missing in reasons.FAMILY"


def test_percentile_phrases():
    assert reasons._percentile_phrase(0.995) == "top 1% for this subdivision"
    assert reasons._percentile_phrase(0.955) == "top 5% for this subdivision"
    assert reasons._percentile_phrase(0.005) == "bottom 1% for this subdivision"
    assert reasons._percentile_phrase(0.50) == "near normal for this subdivision"


# ------------------------------------------------------------------ API Schemas
def test_api_schema_bust_prediction():
    # Valid normal prediction
    pred = schema.BustPrediction(
        region="Konkan & Goa",
        region_id="KONKAN_GOA",
        lead_day=4,
        init_date="2022-07-10",
        valid_date="2022-07-13",
        status=schema.PredictionStatus.OK,
        bust_probability=0.45,
        confidence_in_estimate=0.85,
        dominant_factors=["High forecast rain", "High moisture transport"],
    )
    assert pred.lead_day == 4
    assert pred.bust_probability == 0.45

    # Out of distribution prediction with None probability
    ood_pred = schema.BustPrediction(
        region="West Rajasthan",
        region_id="WEST_RAJASTHAN",
        lead_day=6,
        init_date="2022-08-01",
        valid_date="2022-08-06",
        status=schema.PredictionStatus.OUT_OF_DISTRIBUTION,
        bust_probability=None,
    )
    assert ood_pred.bust_probability is None
    assert ood_pred.status == schema.PredictionStatus.OUT_OF_DISTRIBUTION


def test_api_schema_bulletin():
    bulletin = schema.Bulletin(
        init_date="2022-07-15",
        issued_at="2022-07-15T08:00:00Z",
        lead_day=3,
        n_regions=34,
        data_quality=schema.DataQuality.OK,
        predictions=[],
    )
    assert bulletin.n_regions == 34
    assert bulletin.data_quality == schema.DataQuality.OK
