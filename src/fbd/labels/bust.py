"""The bust label.  This is the single most consequential file in the project.

Whoever defines "bust" defines every downstream number, so the definition is
spelled out here in full, with the reasoning, rather than buried in a notebook.

Definition (LOGIC.md sec 4.3, made precise)
-------------------------------------------
For a subdivision ``s``, initialisation ``t0`` and lead day ``L``, with forecast
area-mean rainfall ``F`` and observed (IMD gauge) area-mean rainfall ``O``:

    bust = MAGNITUDE and DECISION_RELEVANCE

    MAGNITUDE          |F - O| >= max(min_abs_error, P_q(s, month))
    DECISION_RELEVANCE F and O fall in different IMD intensity categories
                       and max(F, O) >= moderate threshold (15.6 mm)

Why both conditions, and not either one alone
---------------------------------------------
*Percentile alone is circular.*  Taking the 95th percentile of |error| defines
5% of rows to be busts by construction; the "bust rate" would then be an
artefact of the definition rather than a measurement.

*Category alone is too brittle.*  It fires on a 15.5 mm vs 15.7 mm pair that
merely straddles a boundary, and it almost never fires in dry subdivisions such
as West Rajasthan, whose entire JJAS record maxes out at 29.7 mm/day area-mean.

Requiring both means a bust is *a large error that also flips the rainfall
category into or out of operationally significant rain* -- which is precisely
"would the forecaster's alert decision have changed?" (LOGIC.md sec 4.4).

Leakage control
---------------
``P_q(s, month)`` is estimated on the **training years only** and then applied
unchanged to validation and test years.  Otherwise the held-out bust rate would
be fixed by construction at ``1-q`` and the evaluation would be meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fbd import config

CATEGORY_EDGES = np.array([c[1] for c in config.RAIN_CATEGORIES], dtype=float)
CATEGORY_NAMES = [c[0] for c in config.RAIN_CATEGORIES]
MODERATE_THRESHOLD = config.RAIN_CATEGORIES[2][1]  # 15.6 mm, lower bound of "moderate"
HEAVY_THRESHOLD = config.RAIN_CATEGORIES[config.HEAVY_CATEGORY_INDEX][1]  # 64.5 mm


def rain_category(mm) -> np.ndarray:
    """Map rainfall (mm/day) to an IMD intensity category index."""
    x = np.asarray(mm, dtype=float)
    idx = np.searchsorted(CATEGORY_EDGES, x, side="right") - 1
    return np.where(np.isnan(x), -1, np.clip(idx, 0, len(CATEGORY_NAMES) - 1))


def error_thresholds(
    df: pd.DataFrame, train_years, percentile: float | None = None
) -> pd.DataFrame:
    """Per (subdivision, month) |error| percentile, fitted on training years only.

    Subdivision-months with too few training samples fall back to the
    subdivision-wide percentile, then to the global percentile, so that the
    threshold is always defined and never estimated from a handful of points.
    """
    percentile = config.BUST_ERROR_PERCENTILE if percentile is None else percentile
    tr = df[df.valid_date.dt.year.isin(list(train_years))].dropna(subset=["abs_error"])

    global_thr = float(np.percentile(tr.abs_error, percentile))
    by_sub = (
        tr.groupby("subdivision_id").abs_error.quantile(percentile / 100.0).rename("thr_sub")
    )
    grp = tr.groupby(["subdivision_id", "month"]).abs_error
    by_sub_month = grp.quantile(percentile / 100.0).rename("thr_sub_month")
    counts = grp.size().rename("n")

    thr = pd.concat([by_sub_month, counts], axis=1).reset_index()
    thr = thr.merge(by_sub.reset_index(), on="subdivision_id", how="left")
    # A month with fewer than 200 training rows is too thin for a 95th
    # percentile; back off to the subdivision-wide estimate.
    thr["error_threshold"] = np.where(
        thr.n >= 200, thr.thr_sub_month, thr.thr_sub.fillna(global_thr)
    )
    thr["error_threshold"] = thr.error_threshold.fillna(global_thr)
    return thr[["subdivision_id", "month", "error_threshold", "n"]]


def label(
    pairs: pd.DataFrame,
    train_years=None,
    percentile: float | None = None,
) -> pd.DataFrame:
    """Attach bust labels to a frame of forecast/observation pairs.

    Parameters
    ----------
    pairs : must contain subdivision_id, init_date, lead_day, valid_date,
            fcst_rain_mm, obs_rain_mm.
    """
    train_years = train_years or config.TRAIN_YEARS
    df = pairs.copy()
    df["month"] = df.valid_date.dt.month
    df["error"] = df.fcst_rain_mm - df.obs_rain_mm
    df["abs_error"] = df.error.abs()

    thr = error_thresholds(df, train_years, percentile)
    df = df.merge(thr[["subdivision_id", "month", "error_threshold"]],
                  on=["subdivision_id", "month"], how="left")

    floor = config.BUST_MIN_ABS_ERROR_MM
    df["effective_threshold"] = np.maximum(df.error_threshold, floor)

    df["fcst_category"] = rain_category(df.fcst_rain_mm)
    df["obs_category"] = rain_category(df.obs_rain_mm)

    magnitude = df.abs_error >= df.effective_threshold
    category_flip = df.fcst_category != df.obs_category
    significant = np.maximum(df.fcst_rain_mm, df.obs_rain_mm) >= MODERATE_THRESHOLD

    df["bust"] = (magnitude & category_flip & significant).astype("int8")
    # A bust where heavy rain was involved on either side: the high-impact subset
    # the decision-cost metric weights most.
    df["high_impact"] = (
        np.maximum(df.fcst_rain_mm, df.obs_rain_mm) >= HEAVY_THRESHOLD
    ).astype("int8")
    df["bust_high_impact"] = (df.bust.astype(bool) & df.high_impact.astype(bool)).astype("int8")
    # Direction: did the forecast miss rain that fell, or invent rain that did not?
    df["bust_type"] = np.where(
        df.bust == 0, "none", np.where(df.error < 0, "under_forecast", "over_forecast")
    )
    # Rows where either side is missing cannot be labelled.
    invalid = df.fcst_rain_mm.isna() | df.obs_rain_mm.isna()
    df.loc[invalid, ["bust", "bust_high_impact", "high_impact"]] = np.nan
    df.loc[invalid, "bust_type"] = "unlabelled"
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Bust rate by lead day and split -- the sanity table to eyeball first."""
    d = df.dropna(subset=["bust"]).copy()
    d["year"] = d.valid_date.dt.year
    d["split"] = np.where(
        d.year.isin(config.TEST_YEARS),
        "test",
        np.where(d.year.isin(config.VAL_YEARS), "val", "train"),
    )
    return (
        d.groupby(["split", "lead_day"])
        .agg(
            n=("bust", "size"),
            bust_rate=("bust", "mean"),
            high_impact_bust_rate=("bust_high_impact", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
    )
