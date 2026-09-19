"""Confidence intervals for every headline number, and for the margins.

Until now this project reported point estimates with nothing attached to them.
The README said the margin over a real operational ensemble was "+0.025 AUROC"
without saying whether +0.025 can be told apart from zero on 41 init dates --
which is the first thing a reviewer should ask and the project had no answer to.

    PYTHONPATH=src python scripts/compute_confidence_intervals.py

Writes ``data/artifacts/confidence_intervals.json`` and prints markdown tables
ready to paste into the README.

Method
------
Cluster bootstrap over **init dates**, not rows.  The test year is 39,950 rows
but only 122 init dates, and rows sharing a date share a synoptic situation, so
resampling rows would treat correlated observations as independent and produce
intervals several times too narrow.  See ``fbd.evaluate.uncertainty``.

Margins are **paired**: both predictors are scored on the same resampled rows,
so the shared difficulty of a given set of days cancels.  Two overlapping
marginal intervals do not mean a difference is indistinguishable from zero, so
the difference is bootstrapped directly rather than eyeballed from the margins.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fbd import config  # noqa: E402
from fbd.evaluate import metrics as M, uncertainty as U  # noqa: E402

OUT = config.ARTIFACTS / "confidence_intervals.json"
PREDICTIONS = config.ARTIFACTS / "test_predictions.parquet"
MODEL_LABEL = "4 XGBoost + isotonic"
SPREAD_LABEL = "2 spread [lagged-ensemble]"

METRICS = [
    ("auroc", M.auroc, 3),
    ("brier", M.brier, 4),
    ("bss", M.brier_skill_score, 3),
    ("ece", M.expected_calibration_error, 4),
]


def load_predictions(rebuild: bool) -> pd.DataFrame:
    """Per-row test predictions, rebuilt from the training pipeline if absent."""
    if PREDICTIONS.exists() and not rebuild:
        df = pd.read_parquet(PREDICTIONS)
        print(f"loaded cached predictions: {len(df):,} rows "
              f"({PREDICTIONS.relative_to(config.ROOT)})")
        return df

    print("building test predictions (fits all five predictors) ...")
    from train_model import build_test_predictions, save_test_predictions

    _tr, _va, te, preds, _feats, _model = build_test_predictions()
    save_test_predictions(te, preds)
    return pd.read_parquet(PREDICTIONS)


def interval_table(df: pd.DataFrame, labels: list, n_boot: int) -> dict:
    """Marginal intervals for every predictor on every metric."""
    y = df.bust.to_numpy(float)
    clusters = df.init_date.to_numpy()
    out: dict = {}
    for label in labels:
        p = df[label].to_numpy(float)
        out[label] = {}
        for name, fn, _places in METRICS:
            iv = U.metric_interval(fn, y, p, clusters, n_boot=n_boot)
            out[label][name] = iv.as_dict()
        print(f"  {label:34s} auroc {out[label]['auroc']['point']:.3f} "
              f"[{out[label]['auroc']['lo']:.3f}, {out[label]['auroc']['hi']:.3f}]")
    return out


def margin(df: pd.DataFrame, a: str, b: str, n_boot: int) -> dict:
    """Paired interval on the AUROC difference between two predictors."""
    iv = U.paired_difference(
        M.auroc, df.bust.to_numpy(float),
        df[a].to_numpy(float), df[b].to_numpy(float),
        df.init_date.to_numpy(), n_boot=n_boot,
    )
    verdict = ("distinguishable from zero" if iv.excludes_zero
               else "NOT distinguishable from zero")
    print(f"  {a}\n    minus {b}\n    delta AUROC {iv.fmt(4)}  -> {verdict}")
    return {"a": a, "b": b, **iv.as_dict(), "excludes_zero": iv.excludes_zero}


# --------------------------------------------------------------------------
# The comparison that matters: a real 50-member operational ensemble
# --------------------------------------------------------------------------
def ens_margin(n_boot: int) -> dict | None:
    """Paired interval on model vs true IFS ENS spread, on rows where both exist.

    This is the number the README leads with, and the one with the least data
    behind it: the full ENS archive costs ~105 GB (D-007), so only a stratified
    subsample was pulled.  If any margin in this project is going to turn out
    indistinguishable from zero, it is this one, and that is worth knowing.
    """
    ens_dir = config.WB2_RAW / "ens"
    paths = sorted(glob.glob(str(ens_dir / "ens_spread_*.parquet")))
    if not paths:
        print("  [skip] no ENS subsample locally; run scripts/fetch_ens.py")
        return None

    from fbd.model import train as T

    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    ens = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    ens["init_date"] = pd.to_datetime(ens.init_date)
    ds["init_date"] = pd.to_datetime(ds.init_date)

    merged = ds.merge(
        ens[["subdivision_id", "init_date", "lead_day", "ens_spread", "ens_mean"]],
        on=["subdivision_id", "init_date", "lead_day"], how="inner",
    ).dropna(subset=["bust", "ens_spread"])
    if merged.empty:
        print("  [skip] no overlap between ENS subsample and dataset")
        return None

    merged["ens_spread_rel"] = merged.ens_spread / (merged.ens_mean + 1.0)
    test = merged[merged.split == "test"]
    test = test[test.lead_day.isin(config.DECISION_BAND)]
    fit = merged[merged.split.isin(["train", "val"])]
    if test.empty or fit.empty:
        print("  [skip] ENS subsample has no usable test/fit split")
        return None

    model = T.BustModel.load(config.ARTIFACTS / "bust_model.joblib")
    p_model = model.predict_proba(test)

    # Which ENS variant to compare against matters a great deal, and getting it
    # wrong flatters us enormously.  Raw ``ens_spread`` scores AUROC 0.807;
    # spread/mean scores 0.554.  Comparing against the weaker one would turn a
    # +0.025 margin into +0.28 and would be indefensible, so the comparison is
    # against the *strongest* ENS variant, which is the raw spread.
    from fbd.model import baselines as B

    raw = test.ens_spread.to_numpy(float)
    rel = test.ens_spread_rel.to_numpy(float)
    if M.auroc(test.bust, rel) > M.auroc(test.bust, raw):  # pragma: no cover
        raw = rel  # only if the relative form ever becomes the stronger one

    # Calibrated variant: fitted on train/val ENS rows only, never the test
    # year.  AUROC is rank-based so this is for Brier/cost; it is reported
    # because isotonic step-fits introduce rank ties and slightly *lower* AUROC.
    p_ens_cal = B.SpreadBaseline("ens_spread").fit(fit).predict_proba(test)

    y = test.bust.to_numpy(float)
    clusters = pd.to_datetime(test.init_date).dt.strftime("%Y-%m-%d").to_numpy()
    n_dates = len(np.unique(clusters))
    print(f"  matched test rows: {len(test):,} over {n_dates} init dates "
          f"(bust rate {y.mean():.3%})")

    result: dict = {"n_rows": int(len(test)), "n_init_dates": int(n_dates),
                    "bust_rate": float(y.mean()), "margins": {}}

    for name, p_ens in (("raw ENS spread", raw), ("calibrated ENS", p_ens_cal)):
        iv = U.paired_difference(M.auroc, y, p_model, p_ens, clusters, n_boot=n_boot)
        verdict = ("distinguishable from zero" if iv.excludes_zero
                   else "NOT distinguishable from zero")
        print(f"  model minus {name:16s} delta AUROC {iv.fmt(4)}  -> {verdict}")
        result["margins"][name] = {**iv.as_dict(), "excludes_zero": iv.excludes_zero}

    for name, p in (("model", p_model), ("raw ENS spread", raw),
                    ("calibrated ENS", p_ens_cal)):
        iv = U.metric_interval(M.auroc, y, p, clusters, n_boot=n_boot)
        result.setdefault("auroc", {})[name] = iv.as_dict()

    return result


def markdown(band: dict, margins: list, ens: dict | None) -> str:
    lines = ["| Predictor | AUROC | Brier | BSS | ECE |", "|---|---|---|---|---|"]
    for label, m in sorted(band.items(), key=lambda kv: kv[1]["auroc"]["point"]):
        cells = []
        for name, _fn, places in METRICS:
            d = m[name]
            cells.append(f"{d['point']:.{places}f} "
                         f"[{d['lo']:.{places}f}, {d['hi']:.{places}f}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "**Paired AUROC margins** (same resampled rows, so shared "
                  "day difficulty cancels):", "",
              "| Comparison | dAUROC [95% CI] | Distinguishable from zero? |",
              "|---|---|---|"]
    for m in margins:
        iv = U.Interval(m["point"], m["lo"], m["hi"])
        lines.append(f"| {m['a']} - {m['b']} | {iv.fmt(4)} | "
                     f"{'yes' if m['excludes_zero'] else '**no**'} |")
    if ens:
        for name, m in ens["margins"].items():
            iv = U.Interval(m["point"], m["lo"], m["hi"])
            lines.append(f"| model - {name} (n={ens['n_rows']:,}, "
                         f"{ens['n_init_dates']} dates) | {iv.fmt(4)} | "
                         f"{'yes' if m['excludes_zero'] else '**no**'} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=U.DEFAULT_N_BOOT)
    ap.add_argument("--rebuild", action="store_true",
                    help="refit the predictors instead of using the cache")
    args = ap.parse_args()
    t0 = time.time()

    df = load_predictions(args.rebuild)
    labels = [c for c in df.columns
              if c not in ("init_date", "subdivision_id", "lead_day", "bust")]

    band = df[df.lead_day.isin(config.DECISION_BAND)]
    n_dates = band.init_date.nunique()
    print(f"\ndecision band Day {min(config.DECISION_BAND)}-"
          f"{max(config.DECISION_BAND)}: {len(band):,} rows over {n_dates} init "
          f"dates\nresampling dates, not rows -- {len(band) // max(n_dates, 1)} "
          f"rows share each date\n")

    print(f"marginal intervals ({args.n_boot} resamples):")
    band_ci = interval_table(band, labels, args.n_boot)

    print("\npaired margins:")
    margins = [margin(band, MODEL_LABEL, SPREAD_LABEL, args.n_boot)]

    print("\nagainst a real 50-member IFS ensemble:")
    ens = ens_margin(args.n_boot)

    payload = {
        "method": "cluster bootstrap over init dates (percentile, 95%)",
        "n_boot": args.n_boot,
        "n_rows_decision_band": int(len(band)),
        "n_init_dates": int(n_dates),
        "decision_band": band_ci,
        "margins": margins,
        "true_ens": ens,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    print(f"\n{markdown(band_ci, margins, ens)}")
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
