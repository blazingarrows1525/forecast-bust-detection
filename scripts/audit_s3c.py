"""S3c audit, before registration (spec 2026-09-28 §9).

    PYTHONPATH=src python scripts/audit_s3c.py

1. The changed harness re-scores the MLP and the temporal model into a scratch
   directory and reproduces mlp.json and temporal.json exactly (candidate,
   incumbent and ENS AUROC, every fold). The pinned fold models are untouched.
2. On the real 2018 grids, poisoning every run issued after t and every
   analysis after t 00Z leaves the window and the map of 20 sampled rows
   unchanged, and the device's window cut equals the reference cut.
3. Two fits of the fold-2019 primary (train 2016-2017, validation 2018) give
   bit-identical validation-year output, with AUROC at least 0.60 (a defect
   check, not tuning). Fit time and each seed's best epoch are recorded.
4. data/artifacts/s3c_inputs.json is written from the files on disk.
Looks at 2018 only for 2 and 3. No spatial model predicts 2019-2022 here.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Windows: torch must load before scikit-learn (see fbd.model.mlp).
import torch  # noqa: E402

import json  # noqa: E402
import platform  # noqa: E402
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
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.model import grids as G  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.model.spatial import SpatialModel, _Feed, _setup  # noqa: E402

OUT = config.ARTIFACTS / "s3c_audit.json"
TEMPORAL_JSON = promote.OUT_DIR / "temporal.json"
SMOKE_FLOOR = 0.60
SEED = 20260919
KEYS = ("candidate_auroc", "incumbent_auroc", "ens_auroc")


def neutrality(ens, bt) -> dict:
    """Re-score both scored candidates through the changed harness."""
    mlp = json.loads(promote.MLP_JSON.read_text(encoding="utf-8"))
    temporal = json.loads(TEMPORAL_JSON.read_text(encoding="utf-8"))
    pinned = {p.name: P.sha256_file(p) for p in sorted(F.FOLD_DIR.glob("candidate_*.joblib"))}
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, want_json, kw in (("mlp", mlp, {}), ("temporal", temporal, {"mlp": mlp})):
            folds = {}
            for y in (2019, 2020, 2021, 2022):
                _rows, info = promote.score_fold(name, y, ens, config.DECISION_BAND, bt, SEED,
                                                 out_dir=Path(tmp), **kw)
                want = {k: want_json["folds"][str(y)][k] for k in KEYS}
                got = {k: info[k] for k in KEYS}
                folds[str(y)] = {"got": got, "want": want, "ok": got == want}
                print(f"  {name} {y}: {got['candidate_auroc']:.6f} vs "
                      f"{want['candidate_auroc']:.6f} -> {got == want}", flush=True)
            out[name] = {"folds": folds, "ok": all(f["ok"] for f in folds.values())}
    after = {p.name: P.sha256_file(p) for p in sorted(F.FOLD_DIR.glob("candidate_*.joblib"))}
    out["pinned_models_untouched"] = pinned == after
    out["ok"] = out["mlp"]["ok"] and out["temporal"]["ok"] and out["pinned_models_untouched"]
    return out


def _season(year: int):
    import xarray as xr

    def opened(path):
        with xr.open_dataset(path, decode_timedelta=True) as ds:
            return ds.load()

    return (opened(G.HRES_DIR / f"hres_tp24_india_{year}.nc"),
            opened(G.ERA5_DIR / f"era5_2d_{year}.nc"),
            opened(G.ERA5_DIR / f"era5_3d_{year}.nc"))


def causality() -> dict:
    """Poison the real 2018 grids after each sampled row's issue time."""
    import xarray as xr  # noqa: F401

    y = 2018
    h, e2, e3 = _season(y)
    with xr.open_dataset(G.ERA5_DIR / "era5_static.nc") as ds:
        static = ds.load()
    weights = pd.read_parquet(G.WEIGHTS)
    clean = G.Grids.from_datasets({y: h}, {y: e2}, {y: e3}, static, weights)
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    keys = ds[ds.year == y][["subdivision_id", "init_date", "lead_day"]].drop_duplicates()
    rng = np.random.default_rng(SEED)
    sample = keys.iloc[rng.choice(len(keys), 20, replace=False)].reset_index(drop=True)
    bad = []
    for r in sample.itertuples(index=False):
        t = pd.Timestamp(r.init_date).normalize()
        ph, p2, p3 = h.copy(deep=True), e2.copy(deep=True), e3.copy(deep=True)
        ph["tp24"] = ph.tp24.where(ph.time <= t, 1000.0)
        for d in (p2, p3):
            for v in d.data_vars:
                d[v] = d[v].where(d.time <= t, 1000.0)
        poisoned = G.Grids.from_datasets({y: ph}, {y: p2}, {y: p3}, static, weights)
        row = clean.row_index(pd.DataFrame([r._asdict()]))
        a, pa = clean.windows_raw(*row)
        b, pb = poisoned.windows_raw(*row)
        same = np.array_equal(a, b) and np.array_equal(pa, pb) and \
            np.array_equal(clean.maps_raw(row[0]), poisoned.maps_raw(row[0]))
        if not same:
            bad.append(f"{r.subdivision_id} {t.date()} D{r.lead_day}")
    # the device's cut is the reference cut, on the same 20 rows
    rows = clean.row_index(sample)
    sc = G.WindowScaler.fit(clean, *rows)
    want = sc.transform(*clean.windows_raw(*rows))
    feed = _Feed(clean, sc, None, _setup(PR.SPATIAL_PARAMS))
    lt = lambda a: torch.as_tensor(a, dtype=torch.long, device=feed.device)  # noqa: E731
    got = feed.windows(*(lt(a) for a in rows)).cpu().numpy()
    gap = float(np.abs(got - want).max())
    return {"year": y, "n": int(len(sample)), "violations": bad, "device_cut_max_abs_gap": gap,
            "ok": not bad and gap < 1e-4}


def determinism() -> dict:
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = list(T.BustModel.load(F.model_path(2019, "strict")).features)
    outs, secs, hist = [], [], None
    for _ in range(2):
        t0 = time.time()
        m = SpatialModel().fit(tr, va, feats)
        secs.append(round(time.time() - t0, 1))
        outs.append(m.predict_raw(va))
        hist = m.history
    same = bool(np.array_equal(outs[0], outs[1]))
    auc = M.auroc(va.bust.to_numpy(float), outs[0])
    return {"bit_identical": same, "val_year": 2018, "val_auroc": auc, "floor": SMOKE_FLOOR,
            "fit_seconds": secs, "history": hist, "ok": same and auc >= SMOKE_FLOOR}


def environment() -> dict:
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "platform": platform.platform()}


def main() -> int:
    t0 = time.time()
    bt = json.loads(promote.BACKTEST.read_text(encoding="utf-8"))
    ens = E.load_ens()
    print("4. pinning the gridded inputs ...", flush=True)
    manifest = G.write_manifest()
    print(f"  {len(manifest)} files -> {G.MANIFEST.name}", flush=True)
    print("1. harness neutrality (MLP and temporal re-scored into a scratch directory) ...",
          flush=True)
    c1 = neutrality(ens, bt)
    print("2. causality on the real 2018 grids ...", flush=True)
    c2 = causality()
    print(f"  {c2}", flush=True)
    print("3. determinism and smoke, fold 2019, validation year 2018 ...", flush=True)
    c3 = determinism()
    print(f"  bit-identical {c3['bit_identical']}, AUROC {c3['val_auroc']:.4f}, "
          f"{c3['fit_seconds']} s, best epochs {[h['best_epoch'] for h in c3['history']]}",
          flush=True)
    out = {"spatial_params_sha256": PR.params_sha256(PR.SPATIAL_PARAMS),
           "spatial_inputs_sha256": P.sha256_file(G.MANIFEST),
           "n_input_files": len(manifest), "environment": environment(),
           "neutrality": c1, "causality": c2, "determinism": c3,
           "ok": c1["ok"] and c2["ok"] and c3["ok"]}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT}; all ok: {out['ok']} ({time.time() - t0:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
