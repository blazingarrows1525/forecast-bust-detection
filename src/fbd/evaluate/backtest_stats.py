"""The statistics registered in docs/PREREGISTRATION_S1B.md. Needs scikit-learn.

``rows`` holds every fold's scored test rows: year, init_date, month,
subdivision_id, lead_day, bust, p_model, p_ens, p_proxy.
"""
from __future__ import annotations

import pandas as pd

from fbd.evaluate import metrics as M
from fbd.evaluate import registration as R
from fbd.evaluate import uncertainty as U


def _interval(iv) -> dict:
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero}


def year_margin(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    return _interval(U.paired_difference(
        M.auroc, rows.bust.to_numpy(float), rows[a].to_numpy(float),
        rows[b].to_numpy(float), rows.init_date.to_numpy(), n_boot=n_boot, seed=seed))


def mean_margin(rows: pd.DataFrame, a: str, b: str, years, n_boot: int, seed: int) -> dict:
    r = rows[rows.year.isin(list(years))]
    iv = U.stratified_mean_difference(
        M.auroc, r.bust.to_numpy(float), r[a].to_numpy(float), r[b].to_numpy(float),
        r.init_date.to_numpy(), r.year.to_numpy(), n_boot=n_boot, seed=seed)
    return {**_interval(iv), "years": [int(y) for y in years]}


def primary(rows: pd.DataFrame, years, n_boot: int, seed: int) -> dict:
    out = mean_margin(rows, "p_model", "p_ens", years, n_boot, seed)
    v = R.verdict(out["lo"], out["hi"])
    return {**out, "verdict": v, "text": R.BACKTEST_VERDICT_TEXT[v]}


def per_year(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int,
             who: str = "ENS spread") -> dict:
    out = {}
    for year, r in rows.groupby("year"):
        iv = year_margin(r, a, b, n_boot, seed)
        iv["statement"] = R.year_statement(int(year), iv["lo"], iv["hi"], who=who)
        iv["n_rows"] = int(len(r))
        iv["n_init_dates"] = int(r.init_date.nunique())
        out[str(int(year))] = iv
    return out


def by_month(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    out = {}
    for year, r in rows.groupby("year"):
        out[str(int(year))] = {
            str(int(m)): year_margin(rm, a, b, n_boot, seed)
            for m, rm in r.groupby("month") if rm.bust.nunique() == 2}
    return out
