"""Build the labelled modelling dataset: pairs -> bust labels -> features.

Output: data/processed/dataset.parquet  (one row per subdivision x init x lead)
        --fold YEAR --mode legacy|strict  ->  data/processed/backtest/fold_YEAR_MODE.parquet

Run order: fetch_imd.py -> build_truth.py -> fetch_hres.py -> build_dataset.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.ingest import hres  # noqa: E402
from fbd.labels import bust  # noqa: E402
from fbd.features import forecast as ffeat  # noqa: E402

OUT = config.PROCESSED / "dataset.parquet"


def attach_state_features(feats: pd.DataFrame, train_years=None,
                          regime_fit_years=None, require: bool = False) -> pd.DataFrame:
    """Join ERA5 analysis state + regime probabilities onto the feature table.

    CAUSALITY: joined on **init_date**, never valid_date.  The analysis valid on
    the forecast's target day does not exist when the forecast is issued; using
    it would leak the answer and inflate every score.  See fbd.features.era5.

    ``require=True`` (every fold build) raises instead of skipping: a fold
    silently built without state features would train a different model.
    """
    try:
        from fbd.features import era5 as e5
        from fbd.regime import classify as rg
    except Exception as exc:  # noqa: BLE001
        if require:
            raise
        print(f"   [skip] ERA5 features unavailable: {exc}")
        return feats

    try:
        nat = e5.national_daily(train_years=train_years)
        loc = e5.subdivision_fields()
        regimes = rg.classify(nat, loc, fit_years=regime_fit_years)
        static = rg.static_attributes()
    except FileNotFoundError as exc:
        if require:
            raise
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


def build(train_years=config.TRAIN_YEARS, val_years=config.VAL_YEARS,
          test_years=config.TEST_YEARS, regime_fit_years=None,
          require_state: bool = False) -> pd.DataFrame:
    """The labelled dataset for one choice of years. Defaults = the published one."""
    print("1. pairing forecasts with IMD truth ...")
    pairs = hres.build_pairs()
    print(f"   {len(pairs):,} rows, {pairs.subdivision_id.nunique()} subdivisions, "
          f"{pairs.init_date.nunique()} init dates")

    print(f"2. labelling busts (thresholds fitted on {list(train_years)}) ...")
    labelled = bust.label(pairs, train_years=train_years)
    lab = labelled.dropna(subset=["bust"])
    print(f"   overall bust rate {lab.bust.mean():.3%}  "
          f"(high-impact {lab.bust_high_impact.mean():.3%})")

    print("3. building forecast-derived features ...")
    feats = ffeat.build(pairs, labelled=labelled, train_years=train_years)

    print("4. building ERA5 state + regime features ...")
    feats = attach_state_features(feats, train_years=train_years,
                                  regime_fit_years=regime_fit_years,
                                  require=require_state)

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
    ds["split"] = F.split_labels(ds.year, train_years, val_years, test_years)
    if (ds.split == "excluded").any():
        # Years after a fold's test year: never trained on, never scored.
        ds = ds[ds.split != "excluded"].reset_index(drop=True)
    return ds


def build_fold(test_year: int, mode: str) -> pd.DataFrame:
    fold = F.FOLDS[test_year]
    return build(fold.train, fold.val, (fold.test,),
                 regime_fit_years=F.regime_fit_years(fold, mode),
                 require_state=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, choices=sorted(F.FOLDS),
                    help="build one S1b fold instead of the published dataset")
    ap.add_argument("--mode", choices=F.MODES, default="strict")
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.fold is None:
        ds, out = build(), OUT
    else:
        ds, out = build_fold(args.fold, args.mode), F.fold_path(args.fold, args.mode)
        assert out.resolve() != OUT.resolve(), "a fold must never overwrite dataset.parquet"

    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    print(f"\nwrote {out}  rows={len(ds):,}  cols={ds.shape[1]}  "
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
