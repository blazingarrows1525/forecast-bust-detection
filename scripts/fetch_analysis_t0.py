"""Cache the operational HRES initial state (WB2 ``hres_t0``) in the ERA5 cache's layout (B2).

    PYTHONPATH=src python scripts/fetch_analysis_t0.py [--years 2016 ...]

For each JJAS season, the 00Z state of every issue day over the monsoon
domain, on the same 1.5 deg grid as the ERA5 cache (37 x 47):

    analysis_t0_3d_<year>.nc  geopotential, specific_humidity, u and v at 200/500/850 hPa
    analysis_t0_2d_<year>.nc  total_column_water_vapour (DERIVED, fbd.features.tcwv),
                              mean_sea_level_pressure, surface_pressure

The archive has no total column water vapour; it is integrated from specific
humidity on all 13 levels and surface pressure, and the files say so in their
attributes. Only 00Z is kept: every analysis-derived input is read at the
issue time (fbd.features.era5). The store is chunked 8 steps x 13 levels x the
globe, so roughly 2.5 GB is read per season for about 40 MB kept.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.features.tcwv import column_water_vapour  # noqa: E402
from fbd.ingest import wb2  # noqa: E402

STORE = "datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
OUT_DIR = config.WB2_RAW / "analysis_t0"
VARS_3D = ["geopotential", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]
VARS_2D = ["mean_sea_level_pressure", "surface_pressure"]
DERIVED = ("integrated from specific_humidity on 13 levels (50-1000 hPa) and "
           "surface_pressure by fbd.features.tcwv.column_water_vapour; the source "
           "archive has no total_column_water_vapour")


def domain(ds: xr.Dataset) -> xr.Dataset:
    d = config.MONSOON_DOMAIN
    lat = ds["latitude"].values
    lat_slice = (slice(d["lat_min"], d["lat_max"]) if lat[0] < lat[-1]
                 else slice(d["lat_max"], d["lat_min"]))
    return ds.sel(latitude=lat_slice, longitude=slice(d["lon_min"], d["lon_max"]))


def season_00z(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-06-01", f"{year}-09-30", freq="D")


def fetch(year: int, out_dir: Path, workers: int) -> tuple[Path, Path]:
    out3, out2 = out_dir / f"analysis_t0_3d_{year}.nc", out_dir / f"analysis_t0_2d_{year}.nc"
    if out3.exists() and out2.exists():
        print(f"  [skip] {year} cached", flush=True)
        return out3, out2
    ds = wb2.open_store(STORE, chunks={"time": 8})
    times = season_00z(year)
    missing = times.difference(pd.DatetimeIndex(ds.time.values))
    if len(missing):
        raise RuntimeError(f"{year}: {len(missing)} issue days missing from the store")
    sub = domain(ds[VARS_3D + VARS_2D].sel(time=times))
    t0 = time.time()
    with dask.config.set(scheduler="threads", num_workers=workers):
        sub = sub.load()
    q = sub.specific_humidity.transpose("time", "level", "longitude", "latitude")
    ps = sub.surface_pressure.transpose("time", "longitude", "latitude")
    tcwv = column_water_vapour(q.values, sub.level.values, ps.values, axis=1)

    d3 = sub[VARS_3D].sel(level=list(config.LEVELS)).transpose(
        "time", "level", "longitude", "latitude")
    d2 = xr.Dataset(
        {"total_column_water_vapour": (("time", "longitude", "latitude"),
                                       tcwv.astype(np.float32),
                                       {"units": "kg m**-2", "derived": DERIVED}),
         "mean_sea_level_pressure": sub.mean_sea_level_pressure.transpose(
             "time", "longitude", "latitude"),
         "surface_pressure": ps},
        coords={"time": sub.time, "longitude": sub.longitude, "latitude": sub.latitude})
    for part in (d3, d2):
        part.attrs.update(source_store=STORE, kept="00Z of each JJAS issue day")
    out_dir.mkdir(parents=True, exist_ok=True)
    d3.to_netcdf(out3, engine="netcdf4")
    d2.to_netcdf(out2, engine="netcdf4")
    print(f"  [ok]   {year}: {len(times)} days in {time.time() - t0:.0f}s -> "
          f"{(out3.stat().st_size + out2.stat().st_size) / 1e6:.1f} MB; "
          f"TCWV {np.nanmin(tcwv):.1f}..{np.nanmax(tcwv):.1f} kg m-2", flush=True)
    return out3, out2


COLUMNS_DIR = config.WB2_RAW / "era5_columns"


def fetch_era5_columns(year: int, out_dir: Path, workers: int) -> Path:
    """ERA5's own 13-level humidity and surface pressure at 00Z, with its TCWV:
    the B2 gate G1 integrates the first two and compares with the third."""
    out = out_dir / f"era5_columns_{year}.nc"
    if out.exists():
        print(f"  [skip] ERA5 columns {year} cached", flush=True)
        return out
    ds = wb2.open_store(config.ERA5_STORE_2D, chunks={"time": 8})
    sub = domain(ds[["specific_humidity", "surface_pressure", "total_column_water_vapour"]]
                 .sel(time=season_00z(year)))
    t0 = time.time()
    with dask.config.set(scheduler="threads", num_workers=workers):
        sub = sub.load()
    sub.attrs.update(source_store=config.ERA5_STORE_2D, kept="00Z of each JJAS issue day")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_netcdf(out, engine="netcdf4")
    print(f"  [ok]   ERA5 columns {year} in {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="*", default=list(config.ALL_YEARS))
    ap.add_argument("--era5-columns", type=int, nargs="*", default=[],
                    help="also cache ERA5's 13-level humidity for these years (gate G1)")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    print(f"hres_t0 -> {OUT_DIR}", flush=True)
    t0 = time.time()
    for y in args.era5_columns:
        fetch_era5_columns(y, COLUMNS_DIR, args.workers)
    for y in args.years:
        fetch(y, OUT_DIR, args.workers)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
