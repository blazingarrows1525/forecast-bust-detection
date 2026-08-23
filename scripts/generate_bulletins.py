"""Batch job: score every (subdivision, init, lead), attach reasons, write SQLite.

This is the daily batch the deployed system would run.  Doing it offline and
serving precomputed bulletins is deliberate: the problem is a once-a-day
decision-support product, not a streaming service (LOGIC.md sec 14 bans
real-time streaming), and a precomputed store is what makes the fully offline
demo possible.

Storage is SQLite rather than PostgreSQL+PostGIS.  LOGIC.md sec 11 names SQLite
as the sanctioned backup precisely for a pure-offline demo, and the hour-33 gate
("must run with the network unplugged") outranks the nicety of PostGIS here.
Schema is written so a Postgres swap is a connection-string change.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.model import train as T, baselines as B  # noqa: E402
from fbd.ood import detector as OOD  # noqa: E402
from fbd.explain import reasons as R  # noqa: E402

DB = config.ARTIFACTS / "bulletins.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS bulletins (
    region_id TEXT NOT NULL,
    region TEXT NOT NULL,
    init_date TEXT NOT NULL,
    lead_day INTEGER NOT NULL,
    valid_date TEXT NOT NULL,
    status TEXT NOT NULL,
    bust_probability REAL,
    confidence_in_estimate REAL,
    pi_low REAL,
    pi_high REAL,
    dominant_factors TEXT,
    regime_json TEXT,
    data_quality TEXT NOT NULL,
    ood_distance REAL,
    forecast_rain_mm REAL,
    observed_rain_mm REAL,
    actual_bust INTEGER,
    baseline_probability REAL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (region_id, init_date, lead_day)
);
CREATE INDEX IF NOT EXISTS idx_init ON bulletins(init_date);
CREATE INDEX IF NOT EXISTS idx_valid ON bulletins(valid_date);
CREATE INDEX IF NOT EXISTS idx_region ON bulletins(region_id);

-- Immutable audit trail: every override kept with user and reason so that a
-- decision can be reconstructed after an event (LOGIC.md sec 15).
CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id TEXT NOT NULL,
    init_date TEXT NOT NULL,
    lead_day INTEGER NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    user TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def bagged_interval(train, val, df, point, features, n_bags: int = 6):
    """Prediction interval from bootstrap-bagged retraining.

    An earlier version derived the interval by truncating the boosting rounds.
    That was wrong: boosting converges upward, so truncated models are
    systematically *lower* than the full model, and the resulting interval could
    exclude its own point estimate (e.g. [0.571, 0.627] around 0.706).

    This version resamples the training years and refits, which measures the
    thing `confidence_in_estimate` is supposed to mean: how much would this
    number move if we had seen a slightly different history?  The interval is
    then recentred on the deployed model's estimate, so it always contains the
    number actually being served.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    draws = []
    for b in range(n_bags):
        idx = rng.choice(len(train), size=len(train), replace=True)
        boot = train.iloc[idx]
        m = T.BustModel().fit(boot, val, features=list(features))
        draws.append(m.predict_proba(df))
        print(f"   bag {b+1}/{n_bags} fitted")
    D = np.vstack(draws)
    half = np.clip((np.percentile(D, 90, axis=0) - np.percentile(D, 10, axis=0)) / 2.0, 0, 1)
    lo = np.clip(point - half, 0.0, 1.0)
    hi = np.clip(point + half, 0.0, 1.0)
    # Narrow interval -> high confidence in the estimate itself.
    conf = np.clip(1.0 - (hi - lo) / 0.30, 0.0, 1.0)
    return lo, hi, conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val", "test"],
                    help="which splits to write bulletins for")
    ap.add_argument("--top-k-reasons", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    tr, va, te = T.split_frames(ds)
    tr = tr.dropna(subset=["bust"])

    model = T.BustModel.load(config.ARTIFACTS / "bust_model.joblib")
    print(f"model loaded, {len(model.features)} features")

    ood = OOD.MahalanobisOOD().fit(tr, model.features)
    clim = B.ClimatologyBaseline().fit(tr)
    spread = B.SpreadBaseline("lagged_spread").fit(tr)
    explainer = R.ReasonExplainer(model, reference=tr)

    target = ds[ds.split.isin(args.splits)].copy()
    target = target.dropna(subset=model.features, how="all")
    print(f"scoring {len(target):,} rows across splits {args.splits}")

    prob = model.predict_proba(target)
    print('computing bagged prediction intervals ...')
    va_ = va.dropna(subset=['bust'])
    lo, hi, conf = bagged_interval(tr, va_, target, prob, model.features)
    dist = ood.score(target)
    is_ood = dist > ood.threshold_
    clim_p = clim.predict_proba(target)
    base_p = spread.predict_proba(target)

    print(f"OOD refusals: {is_ood.sum():,} / {len(target):,} ({is_ood.mean():.2%})")

    print("computing SHAP reasons ...")
    reason_lists = explainer.explain(target, k=args.top_k_reasons)

    names = config.subdivision_names()
    regime_cols = [c for c in target.columns if c.startswith("regime_")]

    rows = []
    for i in range(len(target)):
        r = target.iloc[i]
        ood_hit = bool(is_ood[i])
        status = "OUT_OF_DISTRIBUTION" if ood_hit else "OK"
        regime = {}
        if regime_cols:
            for c in regime_cols:
                if c in ("regime_top", "regime_top_prob"):
                    continue
                regime[c.replace("regime_", "")] = (
                    float(r[c]) if pd.notna(r[c]) else 0.0
                )
        rows.append(
            (
                r.subdivision_id,
                names.get(r.subdivision_id, r.subdivision_id),
                pd.Timestamp(r.init_date).strftime("%Y-%m-%d"),
                int(r.lead_day),
                pd.Timestamp(r.valid_date).strftime("%Y-%m-%d"),
                status,
                None if ood_hit else float(prob[i]),
                None if ood_hit else float(conf[i]),
                None if ood_hit else float(lo[i]),
                None if ood_hit else float(hi[i]),
                json.dumps(
                    ["conditions outside training experience -- confidence unavailable"]
                    if ood_hit
                    else reason_lists[i]
                ),
                json.dumps(regime) if regime else None,
                "OK",
                float(dist[i]),
                float(r.fcst_rain_mm) if pd.notna(r.fcst_rain_mm) else None,
                float(r.obs_rain_mm) if pd.notna(r.obs_rain_mm) else None,
                int(r.bust) if pd.notna(r.bust) else None,
                float(base_p[i]),
                "0.1.0",
            )
        )

    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM bulletins")
    con.executemany(
        "INSERT OR REPLACE INTO bulletins VALUES (" + ",".join("?" * 19) + ")", rows
    )
    con.executemany(
        "INSERT OR REPLACE INTO meta VALUES (?,?)",
        [
            ("model_version", "0.1.0"),
            ("generated_at", pd.Timestamp.utcnow().isoformat()),
            ("ood_threshold", str(ood.threshold_)),
            ("n_features", str(len(model.features))),
            ("splits", ",".join(args.splits)),
        ],
    )
    con.commit()
    con.close()

    print(f"\nwrote {len(rows):,} bulletin rows -> {DB} in {time.time()-t0:.0f}s")
    print(f"db size: {DB.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
