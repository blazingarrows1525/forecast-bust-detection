"""The served number's held-out metrics, for the landing page (store v0.3.0, D-037).

    PYTHONPATH=src python scripts/served_metrics.py

The landing's evidence cards used to show the XGBoost model alone. Since
v0.3.0 serves the spatial model + ENS combination, the cards show *that*
number, on exactly the rows the earlier cards used (test_predictions.parquet:
2022, Days 3-7, with a label; 20,060 rows over 120 init dates) and with the same
method (cluster bootstrap over init dates, 2,000 resamples, 95%). Beside it:
the v0.2.0 served combination and the lagged-ensemble baseline on the same rows,
and the registered evidence (B5, V3).

Cross-check: on these rows the v0.3.0 combiner's Day 3-7 set is B5's fold-2022
spatial combiner, so its AUROC must equal b5_serving.json's 2022 value.
Writes data/artifacts/served_metrics.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch  # noqa: F401  (Windows: before scikit-learn)
except ImportError:
    pass

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import fit_combiner as FC  # noqa: E402
from compute_confidence_intervals import METRICS, SPREAD_LABEL  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import uncertainty as U  # noqa: E402
from fbd.model import combined as K  # noqa: E402
from fbd.model import train as T  # noqa: E402

OUT = config.ARTIFACTS / "served_metrics.json"
KEYS = ["init_date", "subdivision_id", "lead_day"]
N_BOOT = 2000


def _rows(ds: pd.DataFrame, keys: pd.DataFrame, ens: pd.DataFrame) -> pd.DataFrame:
    d = ds.assign(init_date=pd.to_datetime(ds.init_date))
    out = keys[KEYS].merge(d, on=KEYS, how="left", validate="one_to_one")
    e = ens.assign(init_date=pd.to_datetime(ens.init_date))[E.KEY + ["ens_spread"]]
    return out.merge(e, on=E.KEY, how="left")


def served(model, comb, rows):
    spread = rows.ens_spread.to_numpy(float)
    has = np.isfinite(spread)
    return np.where(has, K.apply(comb, model.predict_raw(rows), spread,
                                 rows.lead_day.to_numpy(int)),
                    model.predict_proba(rows)), int((~has).sum())


def table(y, p, clusters) -> dict:
    return {name: U.metric_interval(fn, y, p, clusters, n_boot=N_BOOT).as_dict()
            for name, fn, _places in METRICS}


def main() -> int:
    preds = pd.read_parquet(config.ARTIFACTS / "test_predictions.parquet")
    band = preds[preds.lead_day.isin(config.DECISION_BAND)].copy()
    band["init_date"] = pd.to_datetime(band.init_date)
    keys = band[KEYS + ["bust", SPREAD_LABEL]]
    ens = E.load_ens()
    y = keys.bust.to_numpy(float)
    clusters = keys.init_date.to_numpy()

    comb3 = K.load()
    model3, _path = FC.load_spatial()
    rows3 = _rows(pd.read_parquet(F.fold_path(FC.SPATIAL_YEAR, "strict")), keys, ens)
    if not np.array_equal(rows3.bust.to_numpy(float), y):
        raise SystemExit("v0.3.0 rows' labels differ from the evaluation rows")
    p3, n_fallback3 = served(model3, comb3, rows3)

    b5 = json.loads((config.ARTIFACTS / "b5_serving.json").read_text(encoding="utf-8"))
    want = b5["folds"][str(FC.SPATIAL_YEAR)]["auroc"]["spatial_combination"]
    got = M.auroc(y, p3)
    if n_fallback3 == 0 and got != want:
        raise SystemExit(f"cross-check failed: served v0.3.0 AUROC {got!r} != B5 {want!r}")

    comb2 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")) if len(sys.argv) > 1 \
        else None
    model2 = T.BustModel.load(config.ARTIFACTS / "bust_model.joblib")
    rows2 = _rows(pd.read_parquet(config.PROCESSED / "dataset.parquet"), keys, ens)
    out = {
        "store": "0.3.0", "year": config.TEST_YEARS[0],
        "rows": "test_predictions.parquet decision band (Days 3-7 with a label)",
        "n_rows": int(len(keys)), "n_init_dates": int(keys.init_date.nunique()),
        "n_boot": N_BOOT, "method": "cluster bootstrap over init dates (percentile, 95%)",
        "cross_check_b5_auroc": {"got": got, "want": want, "ok": got == want},
        "served": table(y, p3, clusters),
        "baseline": {"label": SPREAD_LABEL,
                     **table(y, keys[SPREAD_LABEL].to_numpy(float), clusters)},
        "registered": {
            "b5_primary": b5["primary"],
            "b5_vs_ens": b5["secondary"]["spatial_combination_minus_ens"]["mean"],
            "b5_per_year": b5["per_year"],
        },
        "combiner": {k: comb3[k] for k in ("form", "main", "short", "short_leads",
                                           "refinements", "base_model")},
        "sources": {"combiner_sha256": P.sha256_file(K.COMBINER),
                    "b5_sha256": P.sha256_file(config.ARTIFACTS / "b5_serving.json")},
    }
    if comb2 is not None:
        p2, _n2 = served(model2, comb2, rows2)
        out["previous_v020"] = table(y, p2, clusters)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    s = out["served"]
    print(f"served v0.3.0, 2022 decision band: AUROC {s['auroc']['point']:.4f} "
          f"[{s['auroc']['lo']:.4f}, {s['auroc']['hi']:.4f}], BSS {s['bss']['point']:.3f}, "
          f"ECE {s['ece']['point']:.4f}; cross-check vs B5 {out['cross_check_b5_auroc']['ok']}")
    if comb2 is not None:
        q = out["previous_v020"]
        print(f"served v0.2.0, same rows: AUROC {q['auroc']['point']:.4f}, BSS "
              f"{q['bss']['point']:.3f}, ECE {q['ece']['point']:.4f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
