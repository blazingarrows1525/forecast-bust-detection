"""Fetch true IFS ENS spread for a subsample of init dates.

LOGIC.md sec 8.1 names ensemble spread as *the* baseline to beat, so scoring it
properly matters more than almost any modelling refinement.  The full archive is
unaffordable (~105 GB, DECISIONS.md D-007) because all 50 members sit inside one
chunk, so we buy a stratified subsample instead: every Nth init date of the
held-out year, at full 10-day lead range.

That is enough to answer the question that actually matters -- "does the model
beat the real operational baseline, on the same rows?" -- while being honest
that the ENS comparison is made on a subsample rather than the whole year.
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
from fbd.regions import masks  # noqa: E402

OUT = config.WB2_RAW / "ens"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=config.TEST_YEARS[0])
    ap.add_argument("--every", type=int, default=3, help="take every Nth init date")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"ens_spread_{args.year}_every{args.every}.parquet"
    if out.exists():
        print(f"[skip] {out} exists")
        return 0

    ds = wb2.open_store(config.ENS_STORE, chunks={"time": 1, "prediction_timedelta": 8})
    inits = wb2.season_init_times([args.year], hour=0)
    inits = inits[inits.isin(pd.DatetimeIndex(ds.time.values))][:: args.every]
    leads = wb2.lead_timedeltas()
    print(f"IFS ENS: {len(inits)} init dates in {args.year} (every {args.every} days), "
          f"50 members, leads 1-10")

    da = ds["total_precipitation_24hr"].sel(time=inits, prediction_timedelta=leads)
    da = wb2.india_slice(da.to_dataset(name="tp24"))["tp24"]

    t0 = time.time()
    frames = []
    for i, init in enumerate(inits):
        with dask.config.set(scheduler="threads", num_workers=args.workers):
            arr = da.sel(time=init).load()          # (number, lead, lon, lat)
        arr = arr.rename({"latitude": "lat", "longitude": "lon"})
        arr = arr.transpose("number", "prediction_timedelta", "lat", "lon") * 1000.0

        lats, lons = arr.lat.values, arr.lon.values
        w = masks.overlap_weights(lats, lons)
        sub_ids = sorted(w.subdivision_id.unique())
        W = masks.weights_to_matrix(w, len(lats), len(lons), sub_ids)

        vals = arr.values                            # (member, lead, lat, lon)
        means, _ = masks.area_mean(vals, W)          # (member, lead, sub)
        spread = means.std(axis=0, ddof=1)           # (lead, sub)
        ens_mean = means.mean(axis=0)

        n_l, n_s = spread.shape
        frames.append(pd.DataFrame({
            "init_date": pd.Timestamp(init).normalize(),
            "lead_day": np.repeat(np.arange(1, n_l + 1), n_s),
            "subdivision_id": np.tile(sub_ids, n_l),
            "ens_spread": spread.reshape(-1),
            "ens_mean": ens_mean.reshape(-1),
        }))
        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(inits)} inits  ({el/(i+1):.1f}s each, "
                  f"~{el/(i+1)*(len(inits)-i-1)/60:.1f} min left)")

    df = pd.concat(frames, ignore_index=True)
    df["valid_date"] = df.init_date + pd.to_timedelta(df.lead_day - 1, unit="D")
    df.to_parquet(out, index=False)
    print(f"\nwrote {len(df):,} rows -> {out} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
