"""Poison tests: a fold's test years must never reach a fitted statistic.

Each test fits on early years, then sets the later years to absurd values and
fits again. If anything later leaks into the fit, the two fits differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.features import forecast as ffeat  # noqa: E402
from fbd.features import standardise as S  # noqa: E402
from fbd.labels import bust  # noqa: E402

TRAIN = (2016, 2017)


def _pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2016, 2023):
        for day in pd.date_range(f"{year}-06-01", f"{year}-07-30", freq="D"):
            for sub in ("A", "B"):
                for lead in (1, 2, 3):
                    rows.append(dict(subdivision_id=sub, init_date=day,
                                     lead_day=lead,
                                     valid_date=day + pd.Timedelta(days=lead - 1),
                                     fcst_rain_mm=float(rng.gamma(2.0, 8.0)),
                                     obs_rain_mm=float(rng.gamma(2.0, 8.0))))
    return pd.DataFrame(rows)


def _poison(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    late = out.valid_date.dt.year > max(TRAIN)
    for c in cols:
        out.loc[late, c] = 1e6
    return out


def test_bust_thresholds_ignore_later_years():
    p = _pairs()
    a = bust.label(p, train_years=TRAIN)
    b = bust.label(_poison(p, ["obs_rain_mm"]), train_years=TRAIN)
    pdt.assert_series_equal(a.effective_threshold, b.effective_threshold)


def test_forecast_climatology_ignores_later_years():
    p = _pairs()
    a = ffeat.climatology(p, TRAIN)
    b = ffeat.climatology(_poison(p, ["obs_rain_mm", "fcst_rain_mm"]), TRAIN)
    pdt.assert_frame_equal(a, b)


def test_climatological_bust_rate_ignores_later_years():
    lab = bust.label(_pairs(), train_years=TRAIN)
    poisoned = lab.copy()
    poisoned.loc[poisoned.valid_date.dt.year > max(TRAIN), "bust"] = 1
    pdt.assert_frame_equal(ffeat.climatological_bust_rate(lab, TRAIN),
                           ffeat.climatological_bust_rate(poisoned, TRAIN))


def test_forecast_build_passes_its_years_to_both_climatologies():
    p = _pairs()
    lab = bust.label(p, train_years=TRAIN)
    cols = ["clim_obs_mean", "clim_obs_p90", "clim_fcst_mean", "clim_bust_rate"]
    a = ffeat.build(p, labelled=lab, train_years=TRAIN)
    pp = _poison(p, ["obs_rain_mm"])
    lab2 = lab.copy()
    lab2.loc[lab2.valid_date.dt.year > max(TRAIN), "bust"] = 1
    b = ffeat.build(pp, labelled=lab2, train_years=TRAIN)
    pdt.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def _field_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2016-06-01", "2022-09-30", freq="7D")
    return pd.DataFrame({
        "date": np.repeat(dates, 2),
        "subdivision_id": np.tile(["A", "B"], len(dates)),
        "u850": rng.normal(5, 2, 2 * len(dates)),
        "tcwv": rng.normal(40, 5, 2 * len(dates)),
    })


def test_fit_mask_selects_years_and_none_means_every_row():
    d = pd.Series(pd.to_datetime(["2016-06-01", "2019-06-01", "2022-06-01"]))
    assert S.fit_mask(d, TRAIN).tolist() == [True, False, False]
    assert S.fit_mask(d, None).tolist() == [True, True, True]


def test_standardise_ignores_rows_outside_the_mask():
    f = _field_frame()
    m = S.fit_mask(f.date, TRAIN)
    a = S.standardise(f, ["u850", "tcwv"], m)
    g = f.copy()
    g.loc[~m, ["u850", "tcwv"]] = 1e6
    b = S.standardise(g, ["u850", "tcwv"], m)
    pdt.assert_frame_equal(a[m], b[m])


def test_standardise_is_the_published_national_formula():
    f = _field_frame()
    m = S.fit_mask(f.date, TRAIN)
    mu = f.loc[m, ["u850"]].mean()
    sd = f.loc[m, ["u850"]].std().replace(0, 1.0)
    expected = (f["u850"] - mu["u850"]) / sd["u850"]
    pdt.assert_series_equal(S.standardise(f, ["u850"], m)["u850_z"], expected,
                            check_names=False, check_exact=True)


def test_standardise_within_ignores_rows_outside_the_mask():
    f = _field_frame()
    m = S.fit_mask(f.date, TRAIN)
    a = S.standardise_within(f, ["u850"], "subdivision_id", m)
    g = f.copy()
    g.loc[~m, "u850"] = 1e6
    b = S.standardise_within(g, ["u850"], "subdivision_id", m)
    pdt.assert_frame_equal(a[m], b[m])


def test_standardise_within_every_row_is_the_legacy_regime_formula():
    """Legacy mode must reproduce dataset.parquet bit for bit."""
    f = _field_frame()
    legacy = f.groupby("subdivision_id")["u850"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    got = S.standardise_within(f, ["u850"], "subdivision_id", S.fit_mask(f.date, None))
    pdt.assert_series_equal(got["u850_zl"], legacy, check_names=False, check_exact=True)
