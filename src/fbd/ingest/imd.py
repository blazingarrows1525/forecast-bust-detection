"""IMD 0.25 degree gauge-based gridded rainfall -> subdivision daily area-means.

This is the *truth* side of the bust label.  IMD gauge data outranks ERA5 for
rainfall over Indian land (LOGIC.md sec 3.1 / strategy sec F.6): it is built from
~6,955 stations, whereas ERA5 precipitation is model output, not observation.

Known alignment caveat, stated rather than hidden
-------------------------------------------------
IMD's daily rainfall day runs 0830 IST -> 0830 IST, i.e. 03Z -> 03Z.  The WB2
24 h accumulation we compare it against runs 00Z -> 00Z.  There is therefore a
fixed 3-hour offset between the observed day and the forecast day.  WB2 does not
publish 3-hourly accumulations at this resolution, so the offset cannot be
removed; it is recorded in DECISIONS.md D-005 and quantified in the evaluation.
Because we work with area-mean rainfall over subdivisions spanning tens of
thousands of km2, a 3 h shift is a second-order effect relative to the bust
signal, but it is a real limitation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from fbd import config
from fbd.regions import masks


def open_year(year: int) -> xr.DataArray:
    """Open one IMD year and normalise coordinate names/order."""
    path = config.IMD_RAW / f"RF25_ind{year}_rfp25.nc"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run scripts/fetch_imd.py")
    ds = xr.open_dataset(path)
    da = ds["RAINFALL"].rename(
        {"TIME": "time", "LATITUDE": "lat", "LONGITUDE": "lon"}
    )
    da = da.sortby("lat").sortby("lon")
    # IMD encodes "outside India / no gauge coverage" as NaN already, but some
    # vintages use -999.  Treat any physically impossible value as missing.
    lo, hi = config.PHYSICAL_RANGES["rainfall_mm"]
    return da.where((da >= lo) & (da <= hi))


def open_years(years=None) -> xr.DataArray:
    years = years or config.ALL_YEARS
    return xr.concat([open_year(y) for y in years], dim="time")


def subdivision_daily(years=None, season_only: bool = True) -> pd.DataFrame:
    """Tidy frame: subdivision_id x date -> observed area-mean rainfall (mm).

    Also returns the fraction of each subdivision's area that had valid gauge
    data on that day, which the data-quality layer uses to reject thin coverage
    instead of silently averaging over two cells.
    """
    da = open_years(years)
    if season_only:
        da = da.sel(time=da["time.month"].isin(list(config.SEASON_MONTHS)))

    lats = da["lat"].values
    lons = da["lon"].values
    w = masks.overlap_weights(lats, lons)
    sub_ids = sorted(w.subdivision_id.unique())
    W = masks.weights_to_matrix(w, len(lats), len(lons), sub_ids)

    field = da.values  # (time, lat, lon)
    means, covered = masks.area_mean(field, W)

    times = pd.DatetimeIndex(da["time"].values).normalize()
    out = pd.DataFrame(
        {
            "valid_date": np.repeat(times, len(sub_ids)),
            "subdivision_id": np.tile(sub_ids, len(times)),
            "obs_rain_mm": means.reshape(-1),
            "obs_coverage": covered.reshape(-1),
        }
    )
    return out


def grid_cell_counts(years=None) -> pd.DataFrame:
    """How many IMD land cells inform each subdivision (data-quality report)."""
    da = open_year((years or config.ALL_YEARS)[0])
    lats, lons = da["lat"].values, da["lon"].values
    w = masks.overlap_weights(lats, lons)
    # A cell "informs" a subdivision if it overlaps it and is ever non-NaN.
    ever_valid = np.isfinite(da.values).any(axis=0)
    w = w.assign(valid=[ever_valid[i, j] for i, j in zip(w.lat_idx, w.lon_idx)])
    return (
        w.groupby("subdivision_id")
        .agg(
            n_cells=("valid", "size"),
            n_land_cells=("valid", "sum"),
            area_km2=("weight_km2", "sum"),
        )
        .reset_index()
    )
