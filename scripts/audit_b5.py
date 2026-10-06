"""B5 audit, before registration (design 2026-10-06 §5).

    PYTHONPATH=src python scripts/audit_b5.py

Reproduces, per fold, only numbers already published, and produces no new
test-year information:
1. the served XGBoost + ENS combination, refitted, against combination.json;
2. the pinned spatial fold models, loaded on the GPU, against spatial.json;
3. XGBoost against backtest.json;
and checks every pinned file (fold data and models, spatial models, the 23
gridded inputs). The spatial + ENS combination is not computed here.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402,F401  (Windows: before scikit-learn)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import b5_combination as B  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import b5 as B5  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402

OUT = config.ARTIFACTS / "b5_audit.json"


def main() -> int:
    t0 = time.time()
    bt, cb, sp = B.records()
    B.check_pins(bt, sp)
    print("pins: fold data, XGBoost and spatial fold models, 23 gridded inputs -> ok",
          flush=True)
    ens = E.load_ens()
    folds = {}
    for y in B5.YEARS:
        r = B.reproduce(y, ens, config.DECISION_BAND, bt, cb, sp)
        folds[str(y)] = {"got": r["got"], "want": r["want"], "reproduced": r["reproduced"],
                         "n_fit_rows": int(len(r["val"])), "n_rows": int(len(r["test"]))}
        print(f"  {y}: {r['reproduced']}  {r['got']}", flush=True)
    out = {"folds": folds, "ok": all(f["reproduced"] for f in folds.values()),
           "note": "only published numbers recomputed; spatial + ENS not computed"}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT}; all ok: {out['ok']} ({time.time() - t0:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
