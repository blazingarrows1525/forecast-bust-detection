"""S1b audit: prove the fold pipeline is the published pipeline, before registering.

    PYTHONPATH=src python scripts/audit_s1b.py

1. Rebuild fold 2022 in legacy mode; it must equal dataset.parquet exactly.
2. Retrain on it; decision-band AUROC within 0.001 of the frozen model's.
3. Model minus the lagged proxy within 0.001 of the published +0.0821.
Needs the raw data (data/raw). Writes data/artifacts/s1b_audit.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fbd import config  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.model import baselines as B  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import train as T  # noqa: E402

DATASET = config.PROCESSED / "dataset.parquet"
MODEL = config.ARTIFACTS / "bust_model.joblib"
OUT = config.ARTIFACTS / "s1b_audit.json"
TOL = 0.001


def first_difference(a: pd.DataFrame, b: pd.DataFrame):
    if list(a.columns) != list(b.columns):
        return f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return f"rows differ: {len(a):,} vs {len(b):,}"
    for c in a.columns:
        if not a[c].equals(b[c]):
            try:
                pd.testing.assert_series_equal(a[c], b[c], check_exact=True, check_names=False)
            except AssertionError as exc:
                return f"column {c}: {str(exc).splitlines()[0]}"
    return None


def main() -> int:
    import build_dataset as BD
    from train_model import Calibrated

    t0 = time.time()
    checks = []
    frozen = pd.read_parquet(DATASET)

    print("1. legacy rebuild of fold 2022 ...", flush=True)
    path = F.fold_path(2022, "legacy")
    path.parent.mkdir(parents=True, exist_ok=True)
    BD.build_fold(2022, "legacy").to_parquet(path, index=False)
    legacy = pd.read_parquet(path)  # same parquet round trip as the frozen file
    diff = first_difference(frozen, legacy)
    checks.append({"check": "legacy fold 2022 reproduces dataset.parquet", "ok": diff is None,
                   "detail": diff or f"{len(frozen):,} rows x {frozen.shape[1]} columns identical"})
    print(f"   {'OK' if diff is None else 'MISMATCH'}  {checks[-1]['detail']}", flush=True)

    if diff is None:
        print("2. retrain on the legacy fold ...", flush=True)
        tr, va, te = (d.dropna(subset=["bust"]) for d in T.split_frames(legacy))
        feats = T.available_features(legacy)
        model = T.BustModel().fit(tr, va, features=feats)
        frozen_model = T.BustModel.load(MODEL)
        band = te[te.lead_day.isin(config.DECISION_BAND)]
        auc_frozen = M.auroc(band.bust, frozen_model.predict_proba(band))
        auc_new = M.auroc(band.bust, model.predict_proba(band))
        same_feats = list(feats) == list(frozen_model.features)
        checks.append({"check": "retrained AUROC within 0.001 of the frozen model",
                       "ok": same_feats and abs(auc_new - auc_frozen) <= TOL,
                       "frozen": auc_frozen, "retrained": auc_new,
                       "difference": auc_new - auc_frozen,
                       "same_features": same_feats, "n_features": len(feats)})
        print(f"   frozen {auc_frozen:.4f}  retrained {auc_new:.4f}  "
              f"diff {auc_new - auc_frozen:+.5f}  features identical: {same_feats}", flush=True)

        print("3. lagged-proxy margin ...", flush=True)
        proxy = Calibrated(B.SpreadBaseline("lagged_spread"), "proxy").fit(tr, va).predict_proba(band)
        margin = auc_new - M.auroc(band.bust, proxy)
        ci = json.loads((config.ARTIFACTS / "confidence_intervals.json").read_text())
        published = ci["margins"][0]["point"]
        checks.append({"check": "model minus proxy within 0.001 of the published margin",
                       "ok": abs(margin - published) <= TOL,
                       "published": published, "reproduced": margin,
                       "difference": margin - published})
        print(f"   published {published:+.4f}  reproduced {margin:+.4f}", flush=True)

    payload = {
        "checks": checks,
        "ok": all(c["ok"] for c in checks) and len(checks) == 3,
        "hashes": {"dataset_sha256": P.sha256_file(DATASET),
                   "model_sha256": P.sha256_file(MODEL),
                   "params_sha256": PR.params_sha256()},
        "tolerance": TOL,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{'AUDIT PASSED' if payload['ok'] else 'AUDIT FAILED -- stop, do not register'}"
          f"  ({time.time() - t0:.0f}s)  wrote {OUT}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
