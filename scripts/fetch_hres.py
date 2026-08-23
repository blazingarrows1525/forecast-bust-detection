"""Cache the deterministic HRES rainfall forecasts needed for bust labelling.

One NetCDF per year so that a interrupted run resumes instead of restarting.
Only ``total_precipitation_24hr`` is pulled: 3-D flow fields at every lead would
cost ~47 GB per variable at this chunking (see fbd.ingest.wb2 docstring), so
atmospheric state features come from the ERA5 analysis instead.

Units: WB2 precipitation is in metres.  We convert to mm once, here, so that no
downstream code has to remember.
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


def fetch_year(ds, year: int, out_dir: Path, workers: int) -> Path:
    out = out_dir / f"hres_tp24_india_{year}.nc"
    if out.exists() and out.stat().st_size > 10_000:
        print(f"  [skip] {year} cached ({out.stat().st_size/1e6:.1f} MB)")
        return out

    inits = wb2.season_init_times([year], hour=0)
    inits = inits[inits.isin(pd.DatetimeIndex(ds.time.values))]
    leads = wb2.lead_timedeltas()

    t0 = time.time()
    da = ds["total_precipitation_24hr"].sel(
        time=inits, prediction_timedelta=leads
    )
    da = wb2.india_slice(da.to_dataset(name="tp24"))["tp24"]
    with dask.config.set(scheduler="threads", num_workers=workers):
        arr = da.load()

    arr = arr * 1000.0  # metres -> mm
    arr.attrs = {
        "units": "mm",
        "long_name": "HRES forecast 24h accumulated precipitation",
        "accumulation_window": "[valid_time-24h, valid_time], 00Z-00Z",
        "source": config.HRES_STORE,
    }
    out_ds = arr.to_dataset(name="tp24")
    out_ds["lead_day"] = (
        "prediction_timedelta",
        (pd.to_timedelta(arr.prediction_timedelta.values) // pd.Timedelta(hours=24)).astype(int),
    )
    out_ds.to_netcdf(out, engine="netcdf4")
    print(
        f"  [ok]   {year}: {arr.shape} inits={len(inits)} "
        f"in {time.time()-t0:.0f}s -> {out.stat().st_size/1e6:.1f} MB"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(config.ALL_YEARS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--store", default=config.HRES_STORE)
    args = ap.parse_args()

    out_dir = config.WB2_RAW / "hres"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"HRES store: {args.store}\n  -> {out_dir}")

    ds = wb2.open_store(args.store, chunks={"time": 1, "prediction_timedelta": 8})
    t0 = time.time()
    for y in args.years:
        fetch_year(ds, y, out_dir, args.workers)
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
