"""S3a audit, before registration.

    PYTHONPATH=src python scripts/audit_s3a.py

1. The harness's view of the incumbent reproduces backtest.json exactly.
2. Two fits of the fold-2019 MLP give bit-identical validation-year output.
3. That MLP's validation-year AUROC is at least 0.60 (a defect check, not tuning).
Touches training and validation years only for 2 and 3.
"""
from __future__ import annotations

# Windows: torch must load before scikit-learn (see fbd.model.mlp).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import settle as S  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.model.mlp import MLPModel  # noqa: E402

BACKTEST = config.ARTIFACTS / "backtest.json"
OUT = config.ARTIFACTS / "s3a_audit.json"
SMOKE_FLOOR = 0.60


def main() -> int:
    t0 = time.time()
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    ens = E.load_ens()
    checks = []

    print("1. incumbent reproduction ...", flush=True)
    folds = []
    for y in (2019, 2020, 2021, 2022):
        ds = pd.read_parquet(F.fold_path(y, "strict"))
        inc = T.BustModel.load(F.model_path(y, "strict"))
        test, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
        yy = test.bust.to_numpy(float)
        _n, comp = S.choose_comparator(yy, test.ens_spread.to_numpy(float),
                                       test.ens_spread_rel.to_numpy(float))
        got = (M.auroc(yy, inc.predict_proba(test)), M.auroc(yy, comp))
        want = (bt["folds"][str(y)]["model_auroc"], bt["folds"][str(y)]["ens_auroc"])
        folds.append({"year": y, "model": [got[0], want[0]], "ens": [got[1], want[1]],
                      "exact": got == want})
        print(f"   {y}: model {got[0]:.6f} vs {want[0]:.6f}, ENS {got[1]:.6f} vs {want[1]:.6f}",
              flush=True)
    checks.append({"check": "harness reproduces the S1b incumbent and ENS AUROC exactly",
                   "ok": all(f["exact"] for f in folds), "folds": folds})

    print("2. MLP determinism on fold 2019 (train 2016-2017, validation 2018) ...", flush=True)
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = list(T.BustModel.load(F.model_path(2019, "strict")).features)
    t1 = time.time()
    one = MLPModel().fit(tr, va, feats)
    fit_s = time.time() - t1
    two = MLPModel().fit(tr, va, feats)
    same = bool(np.array_equal(one.predict_raw(va), two.predict_raw(va)))
    checks.append({"check": "two fits give bit-identical validation-year output", "ok": same,
                   "fit_seconds": round(fit_s, 1), "history": one.history})
    print(f"   identical: {same}  ({fit_s:.0f}s per fit; best epochs "
          f"{[h['best_epoch'] for h in one.history]})", flush=True)

    auc = M.auroc(va.bust.to_numpy(float), one.predict_raw(va))
    checks.append({"check": f"validation-year AUROC at least {SMOKE_FLOOR}",
                   "ok": auc >= SMOKE_FLOOR, "validation_auroc_2018": auc})
    print(f"3. validation-year (2018) AUROC {auc:.4f}", flush=True)

    payload = {"checks": checks, "ok": all(c["ok"] for c in checks),
               "hashes": {"backtest_sha256": P.sha256_file(BACKTEST),
                          "mlp_params_sha256": PR.params_sha256(PR.MLP_PARAMS)}}
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{'AUDIT PASSED' if payload['ok'] else 'AUDIT FAILED -- stop, do not register'}"
          f"  ({time.time() - t0:.0f}s)  wrote {OUT}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
