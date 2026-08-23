"""Build the labelled modelling dataset: pairs -> bust labels -> features.

Output: data/processed/dataset.parquet  (one row per subdivision x init x lead)

Run order: fetch_imd.py -> build_truth.py -> fetch_hres.py -> build_dataset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.ingest import hres  # noqa: E402
from fbd.labels import bust  # noqa: E402
from fbd.features import forecast as ffeat  # noqa: E402

OUT = config.PROCESSED / "dataset.parquet"


def attach_state_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Join ERA5 analysis state + regime probabilities onto the feature table.

    CAUSALITY: joined on **init_date**, never valid_date.  The analysis valid on
    the forecast's target day does not exist when the forecast is issued; using
    it would leak the answer and inflate every score.  See fbd.features.era5.
    """
    try:
        from fbd.features import era5 as e5
        from fbd.regime import classify as rg
    except Exception as exc:  # noqa: BLE001
        print(f"   [skip] ERA5 features unavailable: {exc}")
        return feats

    try:
        nat = e5.national_daily()
        loc = e5.subdivision_fields()
        regimes = rg.classify(nat, loc)
        static = rg.static_attributes()
    except FileNotFoundError as exc:
        print(f"   [skip] ERA5 cache incomplete: {exc}")
        return feats

    print(f"   national indices: {len(nat):,} days, "
          f"local fields: {len(loc):,} rows, regimes: {len(regimes):,} rows")

    feats = feats.merge(
        nat.rename(columns={"date": "init_date"}), on="init_date", how="left"
    )
    feats = feats.merge(
        loc.rename(columns={"date": "init_date"}),
        on=["subdivision_id", "init_date"], how="left",
    )
    feats = feats.merge(
        regimes.rename(columns={"date": "init_date"}),
        on=["subdivision_id", "init_date"], how="left",
    )
    feats = feats.merge(static, on="subdivision_id", how="left")

    cov = feats.regime_entropy.notna().mean() if "regime_entropy" in feats else 0.0
    print(f"   state-feature coverage: {cov:.1%} of rows")
    return feats


def main() -> int:
    t0 = time.time()

    print("1. pairing forecasts with IMD truth ...")
    pairs = hres.build_pairs()
    print(f"   {len(pairs):,} rows, {pairs.subdivision_id.nunique()} subdivisions, "
          f"{pairs.init_date.nunique()} init dates")

    print("2. labelling busts ...")
    labelled = bust.label(pairs)
    lab = labelled.dropna(subset=["bust"])
    print(f"   overall bust rate {lab.bust.mean():.3%}  "
          f"(high-impact {lab.bust_high_impact.mean():.3%})")

    print("3. building forecast-derived features ...")
    feats = ffeat.build(pairs, labelled=labelled)

    print("4. building ERA5 state + regime features ...")
    feats = attach_state_features(feats)

    keep = [
        "subdivision_id", "init_date", "lead_day", "valid_date",
        "fcst_rain_mm", "obs_rain_mm", "error", "abs_error",
        "fcst_category", "obs_category", "effective_threshold",
        "bust", "bust_high_impact", "high_impact", "bust_type",
    ]
    ds = labelled[keep].merge(
        feats.drop(columns=[c for c in feats.columns
                            if c in keep and c not in
                            ("subdivision_id", "init_date", "lead_day", "valid_date")]),
        on=["subdivision_id", "init_date", "lead_day", "valid_date"],
        how="left",
    )

    ds["year"] = ds.valid_date.dt.year
    ds["split"] = np.where(
        ds.year.isin(config.TEST_YEARS), "test",
        np.where(ds.year.isin(config.VAL_YEARS), "val", "train"),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  rows={len(ds):,}  cols={ds.shape[1]}  "
          f"in {time.time()-t0:.0f}s")

    print("\nsplit summary:")
    print(
        ds.dropna(subset=["bust"])
        .groupby("split")
        .agg(n=("bust", "size"), bust_rate=("bust", "mean"),
             years=("year", lambda s: sorted(s.unique())))
        .to_string()
    )
    print("\nbust rate by lead day and split:")
    print(bust.summarise(ds).pivot(index="lead_day", columns="split",
                                   values="bust_rate").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
