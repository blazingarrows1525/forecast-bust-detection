"""Cached HRES forecasts -> subdivision area-mean rainfall per (init, lead).

Coastal caveat, stated because it is a real asymmetry
-----------------------------------------------------
IMD truth is NaN over the sea, so an observed subdivision mean is built from
land cells only.  HRES precipitation is defined everywhere, so a 0.7 deg cell
straddling the Konkan coast carries a blended land/ocean value.  The area
overlay already down-weights such a cell by the fraction of it that lies inside
the subdivision polygon, which is the correct treatment; what remains is that
the cell *value* itself is a land/ocean blend.  That is inherent to any coarse
grid, affects forecast and observation comparably for a smooth field, and is
recorded rather than papered over.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from fbd import config
from fbd.regions import masks

HRES_DIR = config.WB2_RAW / "hres"


def open_years(years=None) -> xr.DataArray:
    years = years or config.ALL_YEARS
    paths = [HRES_DIR / f"hres_tp24_india_{y}.nc" for y in years]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing HRES cache: {missing} -- run scripts/fetch_hres.py"
        )
    das = [xr.open_dataset(p)["tp24"] for p in paths]
    da = xr.concat(das, dim="time")
    # Store order is (time, prediction_timedelta, longitude, latitude);
    # the aggregation helpers require (..., lat, lon).
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    da = da.transpose("time", "prediction_timedelta", "lat", "lon")
    return da.sortby("lat").sortby("lon")


def subdivision_forecasts(years=None) -> pd.DataFrame:
    """Tidy frame: subdivision x init_date x lead_day -> forecast rain (mm)."""
    da = open_years(years)
    lats, lons = da["lat"].values, da["lon"].values

    w = masks.overlap_weights(lats, lons)
    sub_ids = sorted(w.subdivision_id.unique())
    W = masks.weights_to_matrix(w, len(lats), len(lons), sub_ids)

    field = da.values  # (time, lead, lat, lon)
    means, covered = masks.area_mean(field, W)  # -> (time, lead, sub)

    inits = pd.DatetimeIndex(da["time"].values)
    leads = (
        pd.to_timedelta(da["prediction_timedelta"].values) // pd.Timedelta(hours=24)
    ).astype(int)

    n_t, n_l, n_s = means.shape
    out = pd.DataFrame(
        {
            "init_date": np.repeat(inits.values, n_l * n_s),
            "lead_day": np.tile(np.repeat(leads.values, n_s), n_t),
            "subdivision_id": np.tile(sub_ids, n_t * n_l),
            "fcst_rain_mm": means.reshape(-1),
            "fcst_coverage": covered.reshape(-1),
        }
    )
    out["init_date"] = pd.to_datetime(out.init_date).dt.normalize()
    # Lead day L describes the 00Z-00Z day init_date + (L-1)  [DECISIONS.md D-004]
    out["valid_date"] = out.init_date + pd.to_timedelta(out.lead_day - 1, unit="D")
    return out


def build_pairs(years=None) -> pd.DataFrame:
    """Join forecasts to IMD truth on (subdivision, valid_date)."""
    from fbd.ingest import imd  # local import keeps module import cheap

    fc = subdivision_forecasts(years)
    truth_path = config.INTERIM / "truth_subdivision_daily.parquet"
    if truth_path.exists():
        tr = pd.read_parquet(truth_path)
    else:
        tr = imd.subdivision_daily(years)
    tr = tr[["subdivision_id", "valid_date", "obs_rain_mm", "obs_coverage"]]

    pairs = fc.merge(tr, on=["subdivision_id", "valid_date"], how="inner")
    # Restrict to JJAS valid days: a Day-10 forecast initialised on 30 September
    # verifies in October, outside the season this project models.
    pairs = pairs[pairs.valid_date.dt.month.isin(list(config.SEASON_MONTHS))]
    return pairs.reset_index(drop=True)
