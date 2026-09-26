"""Fetch true IFS ENS spread, one resumable shard per init date.

LOGIC.md sec 8.1 names ensemble spread as *the* baseline to beat. The first
fetch took every 3rd date of 2022 (41 dates) and every 6th of 2019 and 2020;
S1 (docs/superpowers/specs/2026-09-23-s1-settle-ens-design.md) completes 2022
and adds all of 2021.

Cost, measured: the store keeps 6-hourly leads, so daily leads 1-10 touch 6
chunks per date, ~123 MB. A full season is ~15 GB, so the fetch must survive
interruption: each date is its own shard, written atomically, and a date that
is already on disk -- as a shard or inside a legacy file -- is skipped.

    python scripts/fetch_ens.py --year 2022
    python scripts/fetch_ens.py --year 2021
    python scripts/fetch_ens.py --year 2022 --verify-legacy 2
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
from fbd.evaluate import ens as E  # noqa: E402
from fbd.ingest import ens_fetch as F  # noqa: E402
from fbd.ingest import wb2  # noqa: E402
from fbd.regions import masks  # noqa: E402

OUT = E.ENS_DIR


def open_field(year: int):
    ds = wb2.open_store(config.ENS_STORE, chunks={"time": 1, "prediction_timedelta": 8})
    inits = wb2.season_init_times([year], hour=0)
    inits = inits[inits.isin(pd.DatetimeIndex(ds.time.values))]
    da = ds["total_precipitation_24hr"].sel(prediction_timedelta=wb2.lead_timedeltas())
    da = wb2.india_slice(da.to_dataset(name="tp24"))["tp24"]
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da, inits


def weights(da):
    lats, lons = da.lat.values, da.lon.values
    w = masks.overlap_weights(lats, lons)
    sub_ids = sorted(w.subdivision_id.unique())
    return masks.weights_to_matrix(w, len(lats), len(lons), sub_ids), sub_ids


def fetch_one(da, init, W, sub_ids, workers: int) -> pd.DataFrame:
    with dask.config.set(scheduler="threads", num_workers=workers):
        arr = da.sel(time=init).load()
    arr = arr.transpose("number", "prediction_timedelta", "lat", "lon") * 1000.0
    F.check_members(int(arr.sizes["number"]))
    means, _ = masks.area_mean(arr.values, W)          # (member, lead, sub)
    return F.reduce_members(means, sub_ids, init)


def verify_legacy(da, W, sub_ids, year: int, n: int, workers: int) -> int:
    """Re-fetch ``n`` legacy dates and compare value by value. Exit 1 on mismatch."""
    legacy = [p for p in E.legacy_paths(OUT) if f"_{year}_" in p.name]
    if not legacy:
        print(f"no legacy file for {year}")
        return 1
    old = pd.read_parquet(legacy[0])
    old["init_date"] = pd.to_datetime(old.init_date).dt.normalize()
    dates = sorted(old.init_date.unique())
    picks = [dates[0], dates[len(dates) // 2]][:n]
    worst = 0.0
    for d in picks:
        new = fetch_one(da, pd.Timestamp(d), W, sub_ids, workers)
        m = old[old.init_date == d].merge(new, on=E.KEY, suffixes=("_old", "_new"))
        for v in E.VALUES:
            a, b = m[f"{v}_old"].to_numpy(float), m[f"{v}_new"].to_numpy(float)
            ok = np.isclose(a, b, rtol=E.RTOL, atol=E.ATOL, equal_nan=True)
            both = np.isfinite(a) & np.isfinite(b)
            if both.any():
                worst = max(worst, float(np.abs(a[both] - b[both]).max()))
            if not ok.all():
                print(f"MISMATCH {pd.Timestamp(d):%Y-%m-%d} {v}: {int((~ok).sum())} rows")
                return 1
        print(f"  {pd.Timestamp(d):%Y-%m-%d}: {len(m)} rows identical within tolerance")
    print(f"determinism check passed; max |difference| {worst:.3g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify-legacy", type=int, default=0, metavar="N",
                    help="re-fetch N legacy dates and compare; fetches nothing else")
    args = ap.parse_args()

    da, inits = open_field(args.year)
    W, sub_ids = weights(da)

    if args.verify_legacy:
        return verify_legacy(da, W, sub_ids, args.year, args.verify_legacy, args.workers)

    todo = F.dates_to_fetch(inits, F.done_dates(OUT, args.year))
    print(f"IFS ENS {args.year}: {len(inits)} JJAS dates in store, "
          f"{len(inits) - len(todo)} already on disk, {len(todo)} to fetch", flush=True)

    failed, flagged, t0 = [], [], time.time()
    for i, init in enumerate(todo):
        try:
            df = F.with_retries(lambda: fetch_one(da, init, W, sub_ids, args.workers))
        except F.ShortEnsemble as exc:
            flagged.append((init, str(exc)))
            print(f"  FLAGGED {init:%Y-%m-%d}: {exc}", flush=True)
            continue
        except Exception as exc:  # after retries
            failed.append((init, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {init:%Y-%m-%d}: {exc}", flush=True)
            continue
        F.write_atomic(df, F.shard_path(OUT, init))
        done = i + 1
        per = (time.time() - t0) / done
        print(f"  {done}/{len(todo)} {init:%Y-%m-%d}  {per:.1f}s each, "
              f"~{per * (len(todo) - done) / 60:.1f} min left", flush=True)

    have = len(F.done_dates(OUT, args.year))
    print(f"\n{args.year}: {have}/{len(inits)} dates on disk")
    for d, why in flagged:
        print(f"  flagged {d:%Y-%m-%d}: {why}")
    for d, why in failed:
        print(f"  missing {d:%Y-%m-%d}: {why}  (re-run to retry)")
    return 0 if have == len(inits) else 1


if __name__ == "__main__":
    sys.exit(main())
