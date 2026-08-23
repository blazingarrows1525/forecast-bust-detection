"""Correctness tests for the parts that would fail silently if wrong.

The emphasis is deliberate: a bug in the bust label or a leak in the lagged
ensemble would not raise an exception, it would just produce a better-looking
AUROC.  Those are the failures worth testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config
from fbd.labels import bust
from fbd.features import forecast as ffeat
from fbd.regions import masks


# ---------------------------------------------------------------- categories
def test_rain_categories_match_imd_boundaries():
    # IMD: no rain <2.5, light 2.5-15.6, moderate 15.6-64.5, heavy 64.5-115.5,
    # very heavy 115.5-204.5, extremely heavy >204.5
    got = bust.rain_category([0.0, 2.4, 2.5, 15.5, 15.6, 64.4, 64.5, 115.5, 204.5, 500.0])
    assert list(got) == [0, 0, 1, 1, 2, 2, 3, 4, 5, 5]


def test_rain_category_propagates_nan_as_sentinel():
    assert bust.rain_category([np.nan])[0] == -1


# --------------------------------------------------------------- bust labels
def _pair(fcst, obs, lead=5, sub="KONKAN_GOA", date="2018-07-15"):
    return dict(
        subdivision_id=sub,
        init_date=pd.Timestamp(date) - pd.Timedelta(days=lead - 1),
        lead_day=lead,
        valid_date=pd.Timestamp(date),
        fcst_rain_mm=fcst,
        obs_rain_mm=obs,
    )


def _labelled(rows, percentile=95.0):
    df = pd.DataFrame(rows)
    # Training years must exist for the threshold fit; tag everything 2018.
    return bust.label(df, train_years=(2018,), percentile=percentile)


def test_bust_requires_all_three_conditions():
    # Build a background of small errors so the fitted threshold is small,
    # then probe specific cases.
    rng = np.random.default_rng(0)
    background = [
        _pair(float(v), float(v + rng.normal(0, 1.0)))
        for v in rng.uniform(0, 20, 400)
    ]

    # 1. Large error, category flip, significant rain -> BUST
    big = _pair(80.0, 5.0)
    # 2. Large error but both sides below the "moderate" floor -> NOT a bust
    #    (10 mm vs 0 mm is a big relative miss but not decision-relevant)
    insignificant = _pair(12.0, 0.2)
    # 3. Category flip but tiny error straddling a boundary -> NOT a bust
    straddle = _pair(15.7, 15.5)

    out = _labelled(background + [big, insignificant, straddle])
    got = out.tail(3).bust.tolist()
    assert got[0] == 1, "large + flip + significant must be a bust"
    assert got[2] == 0, "boundary straddle with tiny error must not be a bust"
    # case 2: 12 vs 0.2 flips category (light vs no_rain) but max < 15.6
    assert got[1] == 0, "sub-moderate rainfall must not be decision-relevant"


def test_bust_is_nan_when_either_side_missing():
    rows = [_pair(50.0, np.nan), _pair(np.nan, 50.0)]
    rows += [_pair(float(v), float(v)) for v in np.linspace(0, 30, 300)]
    out = _labelled(rows)
    assert out.head(2).bust.isna().all()
    assert (out.head(2).bust_type == "unlabelled").all()


def test_bust_type_direction():
    rows = [_pair(float(v), float(v)) for v in np.linspace(0, 30, 400)]
    rows += [_pair(90.0, 3.0), _pair(3.0, 90.0)]
    out = _labelled(rows).tail(2)
    assert out.bust_type.tolist() == ["over_forecast", "under_forecast"]


def test_error_threshold_is_fitted_on_training_years_only():
    """The held-out bust rate must be a measurement, not a construction."""
    rng = np.random.default_rng(1)
    rows = []
    for year in (2018, 2022):
        # 2022 deliberately has much larger errors than 2018.
        scale = 1.0 if year == 2018 else 20.0
        for i in range(600):
            v = float(rng.uniform(0, 40))
            rows.append(
                _pair(v, max(0.0, v + rng.normal(0, scale)), date=f"{year}-07-15")
            )
    out = bust.label(pd.DataFrame(rows), train_years=(2018,), percentile=95.0)
    out["year"] = out.valid_date.dt.year
    rate = out.groupby("year").bust.mean()
    # If the threshold were re-fitted per year, both rates would be ~5%.
    # Fitted on 2018 only, the noisier 2022 must bust far more often.
    assert rate[2022] > 3 * rate[2018]


# --------------------------------------------------- lagged-ensemble causality
def test_lagged_ensemble_uses_only_leads_at_or_beyond_L():
    """A forecast at lead L cannot see forecasts issued after it.

    Poison every lead < L with a huge value for one valid day.  If the feature
    leaked, the spread at large L would explode.
    """
    sub, vdate = "ODISHA", pd.Timestamp("2019-08-10")
    rows = []
    for L in range(1, 11):
        val = 1000.0 if L <= 4 else 10.0  # short leads poisoned
        rows.append(
            dict(
                subdivision_id=sub,
                valid_date=vdate,
                lead_day=L,
                fcst_rain_mm=val,
                init_date=vdate - pd.Timedelta(days=L - 1),
                obs_rain_mm=10.0,
            )
        )
    lag = ffeat.lagged_ensemble(pd.DataFrame(rows))
    tail = lag[(lag.lead_day >= 5) & (lag.lead_day <= 8)]
    assert np.allclose(tail.lagged_spread.dropna(), 0.0), (
        "leads >= 5 must not see the poisoned short leads"
    )
    # Lead 4 draws members {4, 5, 6} = {1000, 10, 10}, so it straddles the
    # poison boundary and must show huge spread.  (Leads 1-3 draw only poisoned
    # members, so their spread is legitimately zero.)
    assert lag.loc[lag.lead_day == 4, "lagged_spread"].iloc[0] > 100


def test_lagged_ensemble_marks_thin_membership():
    rows = [
        dict(subdivision_id="BIHAR", valid_date=pd.Timestamp("2019-08-10"),
             lead_day=L, fcst_rain_mm=float(L), init_date=pd.Timestamp("2019-08-01"),
             obs_rain_mm=1.0)
        for L in range(1, 11)
    ]
    lag = ffeat.lagged_ensemble(pd.DataFrame(rows))
    # Lead 10 has only itself (no leads 11, 12) -> spread undefined.
    assert np.isnan(lag.loc[lag.lead_day == 10, "lagged_spread"].iloc[0])
    assert lag.loc[lag.lead_day == 10, "lagged_n_members"].iloc[0] == 1


# ------------------------------------------------------------- area weighting
def test_area_mean_matches_hand_computation():
    W = np.array([[[1.0, 3.0], [0.0, 0.0]]])          # one subdivision
    field = np.array([[[10.0, 20.0], [99.0, 99.0]]])  # one time step
    means, covered = masks.area_mean(field, W)
    assert means.shape == (1, 1)
    assert np.isclose(means[0, 0], (10 * 1 + 20 * 3) / 4)
    assert np.isclose(covered[0, 0], 1.0)


def test_area_mean_renormalises_over_valid_cells_and_reports_coverage():
    W = np.array([[[1.0, 1.0]]])
    field = np.array([[[10.0, np.nan]]])
    means, covered = masks.area_mean(field, W, max_nan_fraction=0.9)
    assert np.isclose(means[0, 0], 10.0)   # mean of the valid half only
    assert np.isclose(covered[0, 0], 0.5)  # but coverage is reported as 50%


def test_area_mean_rejects_thin_coverage():
    W = np.array([[[1.0, 9.0]]])
    field = np.array([[[10.0, np.nan]]])   # only 10% of the area has data
    means, _ = masks.area_mean(field, W, max_nan_fraction=0.4)
    assert np.isnan(means[0, 0]), "coverage below the gate must yield NaN, not a number"


# ------------------------------------------------------------------ regions
def test_subdivision_config_is_an_exact_partition_of_districts():
    """Guards the domain artifact the whole project is aggregated onto."""
    gpd = pytest.importorskip("geopandas")
    shp = config.SHAPES_RAW / "2011_Dist.shp"
    if not shp.exists():
        pytest.skip("district shapefile not downloaded")
    from fbd.regions import build as rbuild

    districts = rbuild.assign_subdivisions(gpd.read_file(shp))
    assert districts.subdivision_id.notna().all()
    assert districts.subdivision_id.nunique() == 36
    assert len(districts) == 641
