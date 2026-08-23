"""Features derived from the deterministic forecast archive alone.

The headline feature here is the **lagged-ensemble spread**.  LOGIC.md sec 7
wants IFS ENS spread, but the full ensemble archive costs ~105 GB at WB2's
chunking because all 50 members live inside one chunk (DECISIONS.md D-007).
The lagged-average forecast (Hoffman & Kalnay 1983) is the classical substitute:
take the forecasts from *successive initialisations* that all verify on the same
day and measure their disagreement.

Causality is the thing to get right here.  For a forecast issued at ``t0`` with
lead ``L``, verifying on day ``V = t0 + (L-1)``, the other forecasts already in
existence for ``V`` are those with lead ``>= L`` -- they were issued earlier.
Forecasts with lead ``< L`` verify on ``V`` too but are issued *after* ``t0``,
so using them would leak the future into the feature.  Only leads ``>= L`` are
used, and there is a test asserting exactly that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fbd import config

LAGGED_MEMBERS = 3  # leads L, L+1, L+2


def lagged_ensemble(pairs: pd.DataFrame, n_members: int = LAGGED_MEMBERS) -> pd.DataFrame:
    """Spread / mean / range across successive initialisations for the same day.

    Returns one row per (subdivision_id, valid_date, lead_day).
    """
    piv = pairs.pivot_table(
        index=["subdivision_id", "valid_date"],
        columns="lead_day",
        values="fcst_rain_mm",
        aggfunc="first",
    )
    leads = sorted(piv.columns)
    arr = piv.to_numpy(dtype=float)  # (n_rows, n_leads)
    lead_pos = {l: i for i, l in enumerate(leads)}

    recs = []
    for L in leads:
        cols = [lead_pos[l] for l in range(L, L + n_members) if l in lead_pos]
        member = arr[:, cols]                      # (n_rows, <=n_members)
        n_valid = np.isfinite(member).sum(axis=1)
        with np.errstate(invalid="ignore"):
            spread = np.nanstd(member, axis=1, ddof=0)
            mean = np.nanmean(member, axis=1)
            rng = np.nanmax(member, axis=1) - np.nanmin(member, axis=1)
        spread[n_valid < 2] = np.nan
        rng[n_valid < 2] = np.nan
        recs.append(
            pd.DataFrame(
                {
                    "subdivision_id": piv.index.get_level_values(0),
                    "valid_date": piv.index.get_level_values(1),
                    "lead_day": L,
                    "lagged_spread": spread,
                    "lagged_mean": mean,
                    "lagged_range": rng,
                    "lagged_n_members": n_valid,
                }
            )
        )
    out = pd.concat(recs, ignore_index=True)
    # Relative spread: 5 mm of disagreement means something different in Konkan
    # (mean 28 mm/day) than in West Rajasthan (mean 2.7 mm/day).
    out["lagged_spread_rel"] = out.lagged_spread / (out.lagged_mean + 1.0)
    return out


def spread_growth(feat: pd.DataFrame) -> pd.DataFrame:
    """How fast disagreement grows with lead -- fast growth flags instability."""
    feat = feat.sort_values(["subdivision_id", "valid_date", "lead_day"])
    g = feat.groupby(["subdivision_id", "valid_date"], sort=False)
    feat["spread_growth"] = g.lagged_spread.diff()
    feat["spread_growth"] = feat.spread_growth.fillna(0.0)
    return feat


def jumpiness(pairs: pd.DataFrame) -> pd.DataFrame:
    """Run-to-run change: |F(t0, L) - F(t0-1, L+1)| for the same valid day.

    A forecast that changes a lot between consecutive model runs is a forecast
    the model is not confident about, and it is a signal an operational
    forecaster already watches informally.
    """
    p = pairs[["subdivision_id", "valid_date", "lead_day", "fcst_rain_mm"]].copy()
    prev = p.copy()
    prev["lead_day"] = prev.lead_day - 1
    prev = prev.rename(columns={"fcst_rain_mm": "fcst_prev_run"})
    out = p.merge(prev, on=["subdivision_id", "valid_date", "lead_day"], how="left")
    out["jumpiness"] = (out.fcst_rain_mm - out.fcst_prev_run).abs()
    return out[["subdivision_id", "valid_date", "lead_day", "fcst_prev_run", "jumpiness"]]


def climatology(pairs: pd.DataFrame, train_years=None) -> pd.DataFrame:
    """Per (subdivision, month) forecast/observed rainfall climatology.

    Fitted on training years only so that no test-year information reaches the
    features.
    """
    train_years = train_years or config.TRAIN_YEARS
    tr = pairs[pairs.valid_date.dt.year.isin(list(train_years))].copy()
    tr["month"] = tr.valid_date.dt.month
    clim = (
        tr.groupby(["subdivision_id", "month"])
        .agg(
            clim_obs_mean=("obs_rain_mm", "mean"),
            clim_obs_p90=("obs_rain_mm", lambda s: s.quantile(0.90)),
            clim_fcst_mean=("fcst_rain_mm", "mean"),
            clim_abs_error=("fcst_rain_mm", "size"),
        )
        .reset_index()
        .drop(columns="clim_abs_error")
    )
    return clim


def climatological_bust_rate(labelled: pd.DataFrame, train_years=None) -> pd.DataFrame:
    """P(bust | subdivision, month, lead) from the training years.

    This doubles as baseline #1 in LOGIC.md sec 8.1 -- the dumbest predictor --
    and as a feature.  Using it as both is legitimate because it is fitted on
    training data only and the baseline is scored on held-out data.
    """
    train_years = train_years or config.TRAIN_YEARS
    tr = labelled[
        labelled.valid_date.dt.year.isin(list(train_years))
    ].dropna(subset=["bust"])
    tr = tr.assign(month=tr.valid_date.dt.month)

    cell = (
        tr.groupby(["subdivision_id", "month", "lead_day"])
        .bust.agg(["mean", "size"])
        .rename(columns={"mean": "clim_bust_rate", "size": "clim_bust_n"})
        .reset_index()
    )
    # Laplace smoothing: a cell with 40 samples and zero busts should not claim
    # a 0% bust rate.
    prior = float(tr.bust.mean())
    k = 20.0
    cell["clim_bust_rate"] = (
        cell.clim_bust_rate * cell.clim_bust_n + prior * k
    ) / (cell.clim_bust_n + k)
    return cell


def build(pairs: pd.DataFrame, labelled: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble all forecast-derived features onto the pair table."""
    lag = spread_growth(lagged_ensemble(pairs))
    jump = jumpiness(pairs)

    out = pairs.merge(lag, on=["subdivision_id", "valid_date", "lead_day"], how="left")
    out = out.merge(jump, on=["subdivision_id", "valid_date", "lead_day"], how="left")

    out["month"] = out.valid_date.dt.month
    out = out.merge(climatology(pairs), on=["subdivision_id", "month"], how="left")

    # Where does today's forecast sit in this subdivision's own climatology?
    out["fcst_anomaly"] = out.fcst_rain_mm - out.clim_fcst_mean
    out["fcst_rel_to_p90"] = out.fcst_rain_mm / (out.clim_obs_p90 + 1.0)
    out["day_of_season"] = (
        out.valid_date - pd.to_datetime(out.valid_date.dt.year.astype(str) + "-06-01")
    ).dt.days

    if labelled is not None:
        cbr = climatological_bust_rate(labelled)
        out = out.merge(
            cbr[["subdivision_id", "month", "lead_day", "clim_bust_rate"]],
            on=["subdivision_id", "month", "lead_day"],
            how="left",
        )
    return out
