"""B3/B4 audit, before registration: the served combination reproduces, nothing new is computed.

    PYTHONPATH=src python scripts/audit_b3_b4.py

Per fold, the script's served combiner reproduces combination.json's Day 3-7
AUROC exactly, and every pinned fold file and model matches backtest.json.
Neither candidate is fitted, and no Day 1-2 test row is scored.
"""
from __future__ import annotations

try:
    import torch  # noqa: F401  (Windows: before scikit-learn)
except ImportError:
    pass

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import score_b3_b4 as S  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import refine as RF  # noqa: E402

OUT = config.ARTIFACTS / "b3_b4_audit.json"


def main() -> int:
    t0 = time.time()
    bt, cb = S.records()
    S.check_pins(bt, cb)
    ens = E.load_ens()
    folds = {}
    for y in RF.YEARS:
        s = S.served(y, ens, cb)
        folds[str(y)] = {"got": s["got"], "want": s["want"], "reproduced": s["reproduced"],
                         "n_fit_rows": int(len(s["val37"])), "n_rows": int(len(s["test37"]))}
        print(f"  {y}: served combination {s['got']:.6f} vs {s['want']:.6f} -> {s['reproduced']}",
              flush=True)
    out = {"folds": folds, "ok": all(f["reproduced"] for f in folds.values()),
           "note": "only the published served combination recomputed; no candidate fitted"}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT}; all ok: {out['ok']} ({time.time() - t0:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
