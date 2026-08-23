"""Cache the ERA5 analysis fields used for regime classification and flow features.

Why the analysis and not the forecast?  Because pulling HRES 3-D fields at every
lead costs ~47 GB per variable (DECISIONS.md D-003), while ERA5 is a single time
series shared by every (init, lead) pair.  It is also the more principled
predictor set: these features describe the atmospheric state the forecast
*started from*, which is what determines predictability.

Resolution split, priced from measured chunk sizes:
  * 3-D pressure-level fields -> 128x64 (2.8 deg), ~1.3 GB per variable.
    2.8 deg resolves the Somali jet, the monsoon trough and WD troughs, which
    are all synoptic-to-planetary scale.  The 1.5 deg store costs 3-4x more for
    features that are spatially smooth anyway.
  * 2-D fields -> 240x121 (1.5 deg), only ~0.3 GB each, and finer resolution is
    genuinely useful for subdivision-scale moisture/pressure features.

Domain is the wider monsoon box, not India: cropping to India would cut off the
Somali jet and the Bay of Bengal, where the depressions form.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.ingest import wb2  # noqa: E402

VARS_3D = [
    "geopotential",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]
VARS_2D = ["total_column_water_vapour", "mean_sea_level_pressure"]
# Time-invariant fields, needed to give each subdivision an orography and a
# coastal character for the regime classifier.  One chunk each -- negligible.
VARS_STATIC = ["geopotential_at_surface", "land_sea_mask"]


def fetch_static(out_dir: Path) -> Path | None:
    out = out_dir / "era5_static.nc"
    if out.exists() and out.stat().st_size > 1000:
        print(f"  [skip] static cached ({out.stat().st_size/1e6:.2f} MB)")
        return out
    ds = wb2.open_store(config.ERA5_STORE_2D)
    sub = _domain_slice(ds[VARS_STATIC]).load()
    sub.to_netcdf(out, engine="netcdf4")
    print(f"  [ok]   static -> {out.stat().st_size/1e6:.2f} MB")
    return out


def _domain_slice(ds):
    d = config.MONSOON_DOMAIN
    lat = ds["latitude"].values
    lat_slice = (
        slice(d["lat_min"], d["lat_max"])
        if lat[0] < lat[-1]
        else slice(d["lat_max"], d["lat_min"])
    )
    return ds.sel(
        latitude=lat_slice, longitude=slice(d["lon_min"], d["lon_max"])
    )


def _season_times(ds, year: int) -> pd.DatetimeIndex:
    t = pd.DatetimeIndex(ds.time.values)
    keep = t[(t.year == year) & (t.month.isin(config.SEASON_MONTHS))]
    return keep


def fetch(kind: str, year: int, out_dir: Path, workers: int) -> Path | None:
    out = out_dir / f"era5_{kind}_{year}.nc"
    if out.exists() and out.stat().st_size > 10_000:
        print(f"  [skip] {kind} {year} cached ({out.stat().st_size/1e6:.1f} MB)")
        return out

    store = config.ERA5_STORE_3D if kind == "3d" else config.ERA5_STORE_2D
    variables = VARS_3D if kind == "3d" else VARS_2D
    ds = wb2.open_store(store, chunks={"time": 40})

    times = _season_times(ds, year)
    if len(times) == 0:
        print(f"  [warn] {kind} {year}: no times in store")
        return None

    sub = ds[variables].sel(time=times)
    if kind == "3d":
        sub = sub.sel(level=list(config.LEVELS))
    sub = _domain_slice(sub)

    t0 = time.time()
    with dask.config.set(scheduler="threads", num_workers=workers):
        sub = sub.load()
    sub.attrs["source_store"] = store
    sub.to_netcdf(out, engine="netcdf4")
    print(
        f"  [ok]   {kind} {year}: times={len(times)} "
        f"in {time.time()-t0:.0f}s -> {out.stat().st_size/1e6:.1f} MB"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(config.ALL_YEARS))
    ap.add_argument("--kinds", nargs="*", default=["2d", "3d"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = config.WB2_RAW / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ERA5 -> {out_dir}")
    t0 = time.time()
    fetch_static(out_dir)
    for kind in args.kinds:
        for y in args.years:
            fetch(kind, y, out_dir, args.workers)
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
