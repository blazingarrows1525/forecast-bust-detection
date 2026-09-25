"""Score the model against **true IFS ensemble spread**, on identical rows.

This closes the most important evidentiary gap in the project.  LOGIC.md sec 8.1
names ensemble spread as the baseline to beat; everything up to now used a
lagged-ensemble proxy because the full ENS archive costs ~105 GB
(DECISIONS.md D-007).  Here we use the real thing on a stratified subsample.

Two comparisons are reported, and the distinction matters:

* **AUROC** needs no calibration -- it is rank-based -- so it can be computed
  from raw ENS spread with no fitting at all, on the held-out year alone.  This
  is the honest headline comparison.
* **Brier / value** need a probability, so the spread must be mapped through an
  isotonic fit.  That fit needs training-year ENS data.  If a training-year
  subsample is present the calibrated comparison is reported too; if not, the
  script says so rather than quietly fitting on the test year.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.model import train as T, baselines as B  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402

ENS_DIR = config.WB2_RAW / "ens"


def load_ens() -> pd.DataFrame:
    """Shared loader: legacy files + shards, each forecast once (fbd.evaluate.ens)."""
    from fbd.evaluate import ens as E
    df = E.load_ens(ENS_DIR)
    print(f"loaded {len(df):,} ENS rows; dates per year {E.dates_by_year(df)}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-band-only", action="store_true")
    args = ap.parse_args()

    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    ens = load_ens()
    ens["init_date"] = pd.to_datetime(ens.init_date)
    ds["init_date"] = pd.to_datetime(ds.init_date)

    merged = ds.merge(
        ens[["subdivision_id", "init_date", "lead_day", "ens_spread", "ens_mean"]],
        on=["subdivision_id", "init_date", "lead_day"], how="inner",
    ).dropna(subset=["bust", "ens_spread"])
    print(f"matched {len(merged):,} rows with true ENS spread")
    if merged.empty:
        print("no overlap -- nothing to score")
        return 1

    merged["ens_spread_rel"] = merged.ens_spread / (merged.ens_mean + 1.0)

    tr_all, va_all, _ = T.split_frames(ds)
    tr_all = tr_all.dropna(subset=["bust"])
    va_all = va_all.dropna(subset=["bust"])
    model = T.BustModel.load(config.ARTIFACTS / "bust_model.joblib")

    test = merged[merged.split == "test"].copy()
    train_ens = merged[merged.split.isin(["train", "val"])]
    print(f"  test rows with ENS: {len(test):,}   "
          f"train/val rows with ENS: {len(train_ens):,}")

    if args.decision_band_only:
        test = test[test.lead_day.isin(config.DECISION_BAND)]
        print(f"  restricted to Day {config.DECISION_BAND}: {len(test):,} rows")

    y = test.bust.to_numpy(int)
    p_model = model.predict_proba(test)
    p_lagged = B.SpreadBaseline("lagged_spread").fit(tr_all).predict_proba(test)

    print("\n" + "=" * 78)
    print("AUROC -- rank-based, needs no calibration, so ENS spread is used raw")
    print("=" * 78)
    rows = [
        {"predictor": "true IFS ENS spread (raw)", "auroc": M.auroc(y, test.ens_spread)},
        {"predictor": "true IFS ENS spread / mean", "auroc": M.auroc(y, test.ens_spread_rel)},
        {"predictor": "lagged-ensemble spread (proxy)", "auroc": M.auroc(y, p_lagged)},
        {"predictor": "XGBoost + isotonic (our model)", "auroc": M.auroc(y, p_model)},
    ]
    r = pd.DataFrame(rows).sort_values("auroc")
    print(r.round(4).to_string(index=False))
    best_ens = max(rows[0]["auroc"], rows[1]["auroc"])
    print(f"\n  n = {len(test):,} subdivision-days, bust rate {y.mean():.2%}")
    print(f"  model - true ENS spread = {rows[3]['auroc'] - best_ens:+.4f} AUROC")

    # How well does the cheap proxy track the real thing?
    corr = np.corrcoef(test.ens_spread, test.lagged_spread.fillna(0))[0, 1]
    print(f"\n  correlation(true ENS spread, lagged proxy) = {corr:.3f}")
    print("  (this is the check that justifies using the proxy on the full archive)")

    print("\n" + "=" * 78)
    print("CALIBRATED comparison (Brier / decision cost)")
    print("=" * 78)
    if len(train_ens) < 500:
        print(
            "  SKIPPED: no training-year ENS subsample available, so an isotonic\n"
            "  fit for ENS spread would have to be fitted on the test year itself.\n"
            "  That would be leakage, so it is not done. Run:\n"
            "     python scripts/fetch_ens.py --year 2019 --every 6\n"
            "     python scripts/fetch_ens.py --year 2020 --every 6\n"
            "  to enable it."
        )
    else:
        ens_base = B.SpreadBaseline("ens_spread").fit(train_ens)
        p_ens = ens_base.predict_proba(test)
        out = pd.DataFrame([
            M.evaluate(y, p_ens, label="true IFS ENS spread (calibrated)"),
            M.evaluate(y, p_lagged, label="lagged-ensemble spread (proxy)"),
            M.evaluate(y, p_model, label="XGBoost + isotonic (our model)"),
        ]).sort_values("auroc")
        cols = ["model", "n", "base_rate", "auroc", "brier", "bss", "ece",
                "pod", "far", "cost_per_1000", "value"]
        print(out[cols].round(4).to_string(index=False))
        out.to_csv(config.ARTIFACTS / "ens_baseline_comparison.csv", index=False)

    r.to_csv(config.ARTIFACTS / "ens_auroc_comparison.csv", index=False)
    print(f"\nsaved -> {config.ARTIFACTS/'ens_auroc_comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
