"""S3b audit, before registration.

    PYTHONPATH=src python scripts/audit_s3b.py

1. The changed harness re-scores the MLP into a scratch directory and reproduces
   mlp.json exactly (candidate, incumbent and ENS AUROC, every fold), and each
   pinned MLP fold model reproduces its AUROC when loaded.
2. On the real fold-2019 dataset, poisoning everything after each cutoff leaves
   20 sampled 2018 sequences unchanged.
3. Two fits of the fold-2019 temporal model give bit-identical validation-year
   (2018) output, with AUROC at least 0.60 (a defect check, not tuning).
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
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import promote  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import sequence as Q  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.model.temporal import TemporalModel  # noqa: E402

OUT = config.ARTIFACTS / "s3b_audit.json"
SMOKE_FLOOR = 0.60
SEED = 20260919
KEYS = ("candidate_auroc", "incumbent_auroc", "ens_auroc")


def _test(y, ens):
    ds = pd.read_parquet(F.fold_path(y, "strict"))
    test, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
    return test


def neutrality(ens, bt, mlp) -> dict:
    folds = {}
    with tempfile.TemporaryDirectory() as tmp:
        for y in (2019, 2020, 2021, 2022):
            rows, info = promote.score_fold("mlp", y, ens, config.DECISION_BAND, bt, SEED,
                                            out_dir=Path(tmp))
            want = {k: mlp["folds"][str(y)][k] for k in KEYS}
            got = {k: info[k] for k in KEYS}
            test = _test(y, ens)
            p = promote.mlp_probabilities(y, test, test.bust.to_numpy(float), mlp)
            loaded = M.auroc(test.bust.to_numpy(float), p)
            ok = got == want and loaded == want["candidate_auroc"]
            folds[str(y)] = {"got": got, "want": want, "loaded_mlp_auroc": loaded, "ok": ok}
            print(f"  {y}: re-scored {got['candidate_auroc']:.6f} vs {want['candidate_auroc']:.6f}, "
                  f"loaded {loaded:.6f} -> {ok}", flush=True)
    return {"folds": folds, "ok": all(f["ok"] for f in folds.values())}


def causality() -> dict:
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    ds["init_date"] = pd.to_datetime(ds.init_date)
    ds["valid_date"] = pd.to_datetime(ds.valid_date)
    rng = np.random.default_rng(SEED)
    keys = ds[ds.year == 2018][["subdivision_id", "init_date"]].drop_duplicates()
    sample = keys.iloc[rng.choice(len(keys), 20, replace=False)]
    clean = Q.History.from_frame(ds)
    bad = []
    for s, t in sample.itertuples(index=False):
        p = ds.copy()
        p.loc[p.valid_date > t - pd.Timedelta(days=Q.LAG), ["obs_rain_mm", "error", "bust"]] = 1e3
        p.loc[p.init_date > t, Q.WEATHER] = 1e3
        a, b = clean.raw([s], [t]), Q.History.from_frame(p).raw([s], [t])
        if not np.array_equal(a, b, equal_nan=True):
            bad.append(f"{s} {t.date()}")
    return {"n": int(len(sample)), "violations": bad, "ok": not bad}


def determinism() -> dict:
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = list(T.BustModel.load(F.model_path(2019, "strict")).features)
    outs, secs, hist = [], [], None
    for _ in range(2):
        t0 = time.time()
        m = TemporalModel(dataset=ds).fit(tr, va, feats)
        secs.append(round(time.time() - t0, 1))
        outs.append(m.predict_raw(va))
        hist = m.history
    same = bool(np.array_equal(outs[0], outs[1]))
    auc = M.auroc(va.bust.to_numpy(float), outs[0])
    return {"bit_identical": same, "val_year": 2018, "val_auroc": auc, "floor": SMOKE_FLOOR,
            "fit_seconds": secs, "history": hist, "ok": same and auc >= SMOKE_FLOOR}


def main() -> int:
    t0 = time.time()
    bt = json.loads(promote.BACKTEST.read_text(encoding="utf-8"))
    mlp = json.loads(promote.MLP_JSON.read_text(encoding="utf-8"))
    ens = E.load_ens()
    print("1. harness neutrality (MLP re-scored into a scratch directory) ...", flush=True)
    c1 = neutrality(ens, bt, mlp)
    print("2. causality on the real fold-2019 dataset ...", flush=True)
    c2 = causality()
    print(f"  {c2}", flush=True)
    print("3. determinism and smoke, fold 2019, validation year 2018 ...", flush=True)
    c3 = determinism()
    print(f"  bit-identical {c3['bit_identical']}, AUROC {c3['val_auroc']:.4f}, "
          f"{c3['fit_seconds']} s", flush=True)
    out = {"temporal_params_sha256": PR.params_sha256(PR.TEMPORAL_PARAMS),
           "neutrality": c1, "causality": c2, "determinism": c3,
           "ok": c1["ok"] and c2["ok"] and c3["ok"]}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT}; all ok: {out['ok']} ({time.time() - t0:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
