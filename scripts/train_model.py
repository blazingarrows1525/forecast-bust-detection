"""Train baselines + bust model, evaluate on the held-out year, write artifacts.

Fairness rule applied throughout: **every** predictor -- baselines included --
gets the same isotonic calibration fitted on the same validation year.  Without
that, the baselines would be compared on Brier score using uncalibrated scores
and the model would 'win' by an artefact of the comparison rather than by being
better.  A class-weighted logistic regression, for instance, emits ~0.5
probabilities and scores a Brier of 0.19 while having a perfectly respectable
AUROC of 0.72.  Beating that on Brier would prove nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.model import baselines as B, train as T  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402

DATASET = config.PROCESSED / "dataset.parquet"
MODEL_PATH = config.ARTIFACTS / "bust_model.joblib"
RESULTS = config.ARTIFACTS / "results.json"


class Calibrated:
    """Wrap any predictor in an isotonic calibration fitted on the val year."""

    def __init__(self, inner, label):
        self.inner, self.label = inner, label
        self.iso = None

    def fit(self, train, val):
        self.inner.fit(train)
        raw = self.inner.predict_proba(val)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(raw, val.bust.to_numpy(float))
        return self

    def predict_proba(self, df):
        return self.iso.predict(self.inner.predict_proba(df))


def main() -> int:
    t0 = time.time()
    ds = pd.read_parquet(DATASET)
    tr, va, te = T.split_frames(ds)
    tr, va, te = (d.dropna(subset=["bust"]) for d in (tr, va, te))
    print(f"train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    print(f"bust rate  train={tr.bust.mean():.3%}  val={va.bust.mean():.3%}  "
          f"test={te.bust.mean():.3%}")

    spread_col = "lagged_spread"
    predictors = [
        ("0 forecast rain only", B.PersistenceBaseline()),
        ("1 climatology (region,month,lead)", B.ClimatologyBaseline()),
        ("2 spread [lagged-ensemble]", B.SpreadBaseline(spread_col)),
        ("3 logistic (spread + lead)", B.LogisticBaseline(spread_col)),
    ]

    rows, preds = [], {}
    for label, p in predictors:
        c = Calibrated(p, label).fit(tr, va)
        pr = c.predict_proba(te)
        preds[label] = pr
        rows.append(M.evaluate(te.bust, pr, label=label))

    feats = T.available_features(ds)
    print(f"\nmodel features ({len(feats)}): {feats}")
    model = T.BustModel().fit(tr, va, features=feats)
    pr = model.predict_proba(te)
    preds["4 XGBoost + isotonic"] = pr
    rows.append(M.evaluate(te.bust, pr, label="4 XGBoost + isotonic"))

    res = pd.DataFrame(rows).sort_values("auroc")
    cols = ["model", "n", "base_rate", "auroc", "brier", "bss", "ece",
            "pod", "far", "csi", "cost_per_1000", "value"]
    print("\n=== HELD-OUT YEAR 2022, all lead days ===")
    print(res[cols].round(4).to_string(index=False))

    # Decision-relevant band, which is what LOGIC.md sec 4.2 says to optimise on.
    band = te[te.lead_day.isin(config.DECISION_BAND)]
    mask = te.lead_day.isin(config.DECISION_BAND).to_numpy()
    band_rows = [
        M.evaluate(band.bust, p[mask], label=lab) for lab, p in preds.items()
    ]
    print(f"\n=== DECISION BAND, Day {min(config.DECISION_BAND)}-{max(config.DECISION_BAND)} ===")
    print(pd.DataFrame(band_rows).sort_values("auroc")[cols].round(4).to_string(index=False))

    # Per-lead AUROC: does the model keep its edge where it matters?
    print("\n=== AUROC by lead day (model vs spread baseline) ===")
    per_lead = []
    for L in sorted(te.lead_day.unique()):
        m = (te.lead_day == L).to_numpy()
        per_lead.append({
            "lead_day": L,
            "n": int(m.sum()),
            "bust_rate": float(te.bust[m].mean()),
            "spread": M.auroc(te.bust[m], preds["2 spread [lagged-ensemble]"][m]),
            "model": M.auroc(te.bust[m], preds["4 XGBoost + isotonic"][m]),
        })
    pl = pd.DataFrame(per_lead)
    pl["gain"] = pl.model - pl.spread
    print(pl.round(4).to_string(index=False))

    # Reliability -- the calibration contract.
    print("\n=== Reliability (model, held-out year) ===")
    print(M.reliability_curve(te.bust, preds["4 XGBoost + isotonic"], 10).round(4).to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    # Snapshot the training-year input distributions next to the model, so the
    # serving-side drift check always compares against the vintage that
    # actually shipped. A reference built at a different time than the model
    # would silently baseline against the wrong thing.
    from fbd.quality import drift as qdrift

    ref_path = qdrift.save_reference(tr)
    print(f"saved drift reference -> {ref_path}")
    RESULTS.write_text(json.dumps({
        "overall": rows,
        "decision_band": band_rows,
        "per_lead": per_lead,
        "features": feats,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }, indent=2, default=float))
    print(f"\nsaved model -> {MODEL_PATH}\nsaved results -> {RESULTS}")
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
