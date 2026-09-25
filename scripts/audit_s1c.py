"""S1c audit, before registration. Scores no combination on any test year.

    PYTHONPATH=src python scripts/audit_s1c.py

1. Every fold's XGBoost and ENS AUROC reproduces backtest.json, and every MLP
   fold model reproduces mlp.json, exactly.
2. Each fold's combiner rows are its validation year, Day 3-7, with ENS.
3. The combiner is deterministic (two fits on fold 2019's validation rows).
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
from fbd.evaluate import combine as C  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import settle as S  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.model.mlp import MLPModel  # noqa: E402

BACKTEST = config.ARTIFACTS / "backtest.json"
MLP_JSON = config.ARTIFACTS / "candidates" / "mlp.json"
OUT = config.ARTIFACTS / "s1c_audit.json"


def main() -> int:
    t0 = time.time()
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    mj = json.loads(MLP_JSON.read_text(encoding="utf-8"))
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    repro, rows_ok, val_2019 = [], [], None
    for y in C.YEARS:
        ds = pd.read_parquet(F.fold_path(y, "strict"))
        xgb = T.BustModel.load(F.model_path(y, "strict"))
        mlp = MLPModel.load(F.FOLD_DIR / f"candidate_mlp_{y}.joblib")
        test, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
        yy = test.bust.to_numpy(float)
        _n, comp = S.choose_comparator(yy, test.ens_spread.to_numpy(float),
                                       test.ens_spread_rel.to_numpy(float))
        got = [M.auroc(yy, xgb.predict_proba(test)), M.auroc(yy, comp),
               M.auroc(yy, mlp.predict_proba(test))]
        want = [bt["folds"][str(y)]["model_auroc"], bt["folds"][str(y)]["ens_auroc"],
                mj["folds"][str(y)]["candidate_auroc"]]
        repro.append({"year": y, "got": got, "want": want, "exact": got == want})
        val = C.validation_rows(ds, ens, lead_days=config.DECISION_BAND)
        vy = pd.to_datetime(val.init_date).dt.year
        fold_val = F.FOLDS[y].val[0]
        rows_ok.append({"year": y, "fit_year": fold_val, "n_rows": int(len(val)),
                        "n_dates": int(val.init_date.nunique()),
                        "ens_dates_fit_year": int(have.get(fold_val, 0)),
                        "ok": bool(len(val) > 0 and set(val.split) == {"val"}
                                   and set(vy) == {fold_val}
                                   and set(val.lead_day) <= set(config.DECISION_BAND)
                                   and have.get(fold_val, 0) == 122)})
        print(f"   {y}: reproduce {repro[-1]['exact']}; fit on {fold_val}: {len(val):,} rows, "
              f"{val.init_date.nunique()} dates", flush=True)
        if y == 2019:
            val_2019 = (val, xgb)

    val, xgb = val_2019
    p = xgb.predict_raw(val)
    a = C.Combiner().fit(p, val.ens_spread, val.bust).predict_proba(p, val.ens_spread)
    b = C.Combiner().fit(p, val.ens_spread, val.bust).predict_proba(p, val.ens_spread)
    checks = [
        {"check": "fold XGBoost, ENS and MLP AUROC reproduce their records exactly",
         "ok": all(r["exact"] for r in repro), "folds": repro},
        {"check": "combiner rows are the validation year, Day 3-7, with complete ENS",
         "ok": all(r["ok"] for r in rows_ok), "folds": rows_ok},
        {"check": "two combiner fits are identical", "ok": bool(np.array_equal(a, b))},
    ]
    payload = {"checks": checks, "ok": all(c["ok"] for c in checks),
               "hashes": {"backtest_sha256": P.sha256_file(BACKTEST),
                          "mlp_json_sha256": P.sha256_file(MLP_JSON),
                          "combiner_params_sha256": PR.params_sha256(PR.COMBINER_PARAMS)}}
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{'AUDIT PASSED' if payload['ok'] else 'AUDIT FAILED -- stop, do not register'}"
          f"  ({time.time() - t0:.0f}s)  wrote {OUT}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
