"""Atmospheric-state features from the ERA5 analysis.

CAUSALITY RULE (the easiest way to accidentally cheat on this project)
----------------------------------------------------------------------
Every feature here is evaluated at the **initialisation time t0, or earlier**.
The ERA5 analysis valid on the forecast's *target* day would be a near-perfect
predictor of whether that forecast busted -- and it does not exist yet when the
forecast is issued.  Using it would inflate every score and invalidate the whole
result.  A forecaster at t0 knows the analysis up to t0 and nothing after it.

Monsoon regimes persist for one to two weeks, so the state at t0 genuinely
carries information about the state on day t0+5.  We additionally supply
tendencies over the preceding days, which is how a duty forecaster actually
reads the situation ("the trough has been deepening for three days").

Indices computed
----------------
national, one value per date:
  somali_jet       mean u850 over 5-15N, 50-65E   -- monsoon flow strength
  monsoon_trough   mean mslp anomaly over the trough box
  mcz_precip       ERA5 precipitation over the Monsoon Core Zone
  bob_vorticity    max 850 hPa relative vorticity over the Bay of Bengal
  nw_z500_anom     z500 anomaly over NW India      -- western-disturbance trough
  shear_index      |V200 - V850| averaged over India

per subdivision, one value per date:
  moisture_flux_850, tcwv, shear, z500_grad, onshore_wind
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from fbd import config
from fbd.features import standardise as S
from fbd.regions import masks

ERA5_DIR = config.WB2_RAW / "era5"

# Index boxes (lat_min, lat_max, lon_min, lon_max)
BOX_SOMALI_JET = (5.0, 15.0, 50.0, 65.0)
BOX_MONSOON_TROUGH = (20.0, 28.0, 72.0, 88.0)
BOX_BAY_OF_BENGAL = (10.0, 22.0, 82.0, 95.0)
BOX_NW_INDIA = (28.0, 38.0, 68.0, 80.0)
BOX_INDIA = (6.0, 38.0, 66.0, 100.0)

EARTH_R = 6.371e6


def _open(kind: str, years=None) -> xr.Dataset:
    years = years or config.ALL_YEARS
    paths = [ERA5_DIR / f"era5_{kind}_{y}.nc" for y in years]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing ERA5 cache: {missing} -- run scripts/fetch_era5.py")
    ds = xr.concat([xr.open_dataset(p) for p in paths], dim="time")
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds.sortby("lat").sortby("lon")


def _box_mean(da: xr.DataArray, box) -> xr.DataArray:
    lat0, lat1, lon0, lon1 = box
    sub = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    w = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(w).mean(dim=("lat", "lon"))


def _relative_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Relative vorticity dv/dx - du/dy on a lat/lon grid.

    ``differentiate`` gives derivatives per *degree*, so each is converted to a
    per-metre derivative with the spherical metric: dx = R cos(lat) dlon and
    dy = R dlat, both with degrees->radians folded in.  Centred differences keep
    the result on the original grid, so no realignment is needed.
    """
    deg2rad = np.pi / 180.0
    coslat = np.cos(np.deg2rad(u["lat"]))
    dvdx = v.differentiate("lon") / (EARTH_R * coslat * deg2rad)
    dudy = u.differentiate("lat") / (EARTH_R * deg2rad)
    return dvdx - dudy


def national_indices(years=None) -> pd.DataFrame:
    """One row per analysis timestamp: large-scale circulation indices."""
    d3 = _open("3d", years)
    d2 = _open("2d", years)

    u850 = d3.u_component_of_wind.sel(level=850)
    v850 = d3.v_component_of_wind.sel(level=850)
    u200 = d3.u_component_of_wind.sel(level=200)
    v200 = d3.v_component_of_wind.sel(level=200)
    z500 = d3.geopotential.sel(level=500)
    q850 = d3.specific_humidity.sel(level=850)

    shear = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
    vort850 = _relative_vorticity(u850, v850)

    out = pd.DataFrame({"time": pd.DatetimeIndex(d3.time.values)})
    out["somali_jet"] = _box_mean(u850, BOX_SOMALI_JET).values
    out["monsoon_trough_mslp"] = _box_mean(
        d2.mean_sea_level_pressure, BOX_MONSOON_TROUGH
    ).values
    out["nw_z500"] = _box_mean(z500, BOX_NW_INDIA).values
    out["india_shear"] = _box_mean(shear, BOX_INDIA).values
    out["india_tcwv"] = _box_mean(d2.total_column_water_vapour, BOX_INDIA).values
    out["india_q850"] = _box_mean(q850, BOX_INDIA).values
    out["mcz_q850"] = _box_mean(
        q850,
        (
            config.MONSOON_CORE_ZONE["lat_min"],
            config.MONSOON_CORE_ZONE["lat_max"],
            config.MONSOON_CORE_ZONE["lon_min"],
            config.MONSOON_CORE_ZONE["lon_max"],
        ),
    ).values
    lat0, lat1, lon0, lon1 = BOX_BAY_OF_BENGAL
    bob = vort850.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    out["bob_vorticity_max"] = bob.max(dim=("lat", "lon")).values
    out["bob_mslp_min"] = (
        d2.mean_sea_level_pressure.sel(
            lat=slice(lat0, lat1), lon=slice(lon0, lon1)
        )
        .min(dim=("lat", "lon"))
        .values
    )
    return out


def _daily_at_00z(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the 00 UTC analysis, which is the initialisation time of the forecast."""
    d = df[pd.DatetimeIndex(df.time).hour == 0].copy()
    d["date"] = pd.DatetimeIndex(d.time).normalize()
    return d.drop(columns="time")


def national_daily(years=None, train_years=None) -> pd.DataFrame:
    """Daily 00Z indices plus standardised anomalies and multi-day tendencies.

    ``train_years`` defaults to ``config.TRAIN_YEARS``; an S1b fold passes its own.
    """
    idx = _daily_at_00z(national_indices(years)).sort_values("date").reset_index(drop=True)
    cols = [c for c in idx.columns if c != "date"]

    # Standardise against the training-year climatology only.
    mask = S.fit_mask(idx.date, train_years or config.TRAIN_YEARS)
    z = S.standardise(idx, cols, mask)
    for c in cols:
        idx[f"{c}_z"] = z[f"{c}_z"]

    # Tendencies: how the situation has been evolving up to t0.
    #
    # Computed WITHIN each season, not across the whole concatenated record.
    # The frame holds JJAS only, so a naive .diff() would difference 1 June
    # against the previous 30 September -- an eight-month gap -- and manufacture
    # enormous fake tendencies on the first days of every season.  Those rows
    # then look "unprecedented" to the OOD detector for a purely clerical
    # reason.  The first `lag` days of each season are left NaN instead.
    idx["_year"] = idx.date.dt.year
    for c in cols:
        for lag in (1, 3):
            idx[f"{c}_d{lag}"] = idx.groupby("_year")[c].diff(lag)
    return idx.drop(columns="_year")


def subdivision_fields(years=None) -> pd.DataFrame:
    """Per (subdivision, date) local flow features from the 00Z analysis."""
    d3 = _open("3d", years)
    d2 = _open("2d", years)
    keep = pd.DatetimeIndex(d3.time.values).hour == 0
    d3 = d3.isel(time=keep)
    d2 = d2.isel(time=np.asarray(pd.DatetimeIndex(d2.time.values).hour == 0))

    u850 = d3.u_component_of_wind.sel(level=850)
    v850 = d3.v_component_of_wind.sel(level=850)
    u200 = d3.u_component_of_wind.sel(level=200)
    v200 = d3.v_component_of_wind.sel(level=200)
    q850 = d3.specific_humidity.sel(level=850)
    z500 = d3.geopotential.sel(level=500)

    wind850 = np.sqrt(u850**2 + v850**2)
    fields_3d = {
        "moisture_flux_850": q850 * wind850 * 1000.0,   # g/kg * m/s
        "wind_shear": np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2),
        "u850": u850,
        "v850": v850,
        "z500": z500 / 9.80665,                          # geopotential height (m)
    }
    fields_2d = {
        "tcwv": d2.total_column_water_vapour,
        "mslp": d2.mean_sea_level_pressure,
    }

    frames = []
    for group, fields in (("3d", fields_3d), ("2d", fields_2d)):
        sample = next(iter(fields.values()))
        lats, lons = sample.lat.values, sample.lon.values
        w = masks.overlap_weights(lats, lons)
        sub_ids = sorted(w.subdivision_id.unique())
        W = masks.weights_to_matrix(w, len(lats), len(lons), sub_ids)
        times = pd.DatetimeIndex(sample.time.values).normalize()

        cols = {}
        for name, da in fields.items():
            arr = da.transpose("time", "lat", "lon").values
            means, _ = masks.area_mean(arr, W, max_nan_fraction=0.99)
            cols[name] = means.reshape(-1)
        frame = pd.DataFrame(
            {
                "date": np.repeat(times.values, len(sub_ids)),
                "subdivision_id": np.tile(sub_ids, len(times)),
                **cols,
            }
        )
        frames.append(frame)

    out = frames[0].merge(frames[1], on=["date", "subdivision_id"], how="outer")
    out["onshore_wind"] = out.u850  # westerly component; positive = onshore on the west coast
    return out
