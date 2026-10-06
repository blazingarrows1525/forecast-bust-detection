"""B2 audit, before registration (design 2026-10-06 §7).

    PYTHONPATH=src python scripts/audit_b2.py

1. Neutrality: with the default source, the changed builder reproduces each
   pinned fold dataset byte for byte (backtest.json hashes).
2. Gates G1 and G2 (design §4) on 2016-2017 at 00Z over the India box:
   G1 integrates ERA5's own 13-level humidity and compares with ERA5's TCWV;
   G2 compares the hres_t0 derived TCWV with ERA5's TCWV.
3. Causality: poisoning every hres_t0 state after the issue time leaves that
   day's analysis-derived inputs unchanged, for sampled 2018 issue days.
4. Determinism: two trainings of the fold-2019 candidate (train 2016-2017,
   validation 2018) give bit-identical validation-year output.
5. Agreement, descriptive: each input that differs between the two sources,
   correlated on the 2016-2017 rows.
6. data/artifacts/b2_inputs.json pins the inputs.
Looks at no test year. No candidate predicts 2019-2022 here.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as BD  # noqa: E402
import score_b2 as SB  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import b2 as B2  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.features import era5 as e5  # noqa: E402
from fbd.features.tcwv import column_water_vapour  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.regime import classify as rg  # noqa: E402

OUT = config.ARTIFACTS / "b2_audit.json"
SEED = 20260919
SMOKE_FLOOR = 0.60


def neutrality(bt: dict) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for y in B2.YEARS:
            path = Path(tmp) / f"fold_{y}.parquet"
            BD.build_fold(y, "strict").to_parquet(path, index=False)
            got, want = P.sha256_file(path), bt["folds"][str(y)]["dataset_sha256"]
            out[str(y)] = {"ok": got == want, "sha256": got}
            print(f"  fold {y}: {'identical' if got == want else 'DIFFERS'}", flush=True)
    return {"folds": out, "ok": all(f["ok"] for f in out.values())}


def _india(da: xr.DataArray) -> xr.DataArray:
    lat0, lat1, lon0, lon1 = B2.INDIA_BOX
    return da.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1)) \
        .transpose("time", "latitude", "longitude")


def gates() -> dict:
    g1_d, g1_r, g2_d, g2_r = [], [], [], []
    for y in B2.GATE_YEARS:
        with xr.open_dataset(config.WB2_RAW / "era5_columns" / f"era5_columns_{y}.nc") as c:
            c = c.load()
        q = c.specific_humidity.transpose("time", "level", "longitude", "latitude")
        ps = c.surface_pressure.transpose("time", "longitude", "latitude")
        derived = c.total_column_water_vapour.copy(
            data=column_water_vapour(q.values, c.level.values, ps.values, axis=1))
        g1_d.append(_india(derived).values)
        g1_r.append(_india(c.total_column_water_vapour).values)

        with xr.open_dataset(config.WB2_RAW / "analysis_t0" / f"analysis_t0_2d_{y}.nc") as a:
            a = a.load()
        with xr.open_dataset(config.WB2_RAW / "era5" / f"era5_2d_{y}.nc") as e:
            e = e.load()
        e00 = e.total_column_water_vapour.sel(time=a.time)          # ERA5 at the same 00Z
        g2_d.append(_india(a.total_column_water_vapour).values)
        g2_r.append(_india(e00).values)
    g1 = B2.field_agreement(np.concatenate(g1_d), np.concatenate(g1_r))
    g2 = B2.field_agreement(np.concatenate(g2_d), np.concatenate(g2_r))
    g1.update(thresholds=B2.G1, passed=B2.gate_g1(g1))
    g2.update(thresholds=B2.G2, passed=B2.gate_g2(g2))
    return {"years": list(B2.GATE_YEARS), "box": list(B2.INDIA_BOX), "G1": g1, "G2": g2,
            "ok": g1["passed"] and g2["passed"]}


def _state(source: str, fold) -> dict:
    nat = e5.national_daily(train_years=fold.train, source=source)
    loc = e5.subdivision_fields(source=source)
    reg = rg.classify(nat, loc, fit_years=fold.train)
    return {"nat": nat, "loc": loc, "reg": reg}


def causality(n_dates: int = 4) -> dict:
    fold = F.fold_for(2019)
    clean = _state("analysis_t0", fold)
    rng = np.random.default_rng(SEED)
    days = pd.date_range("2018-06-05", "2018-09-25", freq="D")
    sample = sorted(rng.choice(days, n_dates, replace=False))
    src_dir, prefix = e5.SOURCES["analysis_t0"]
    bad = []
    for t in sample:
        t = pd.Timestamp(t)
        with tempfile.TemporaryDirectory() as tmp:
            for f in sorted(src_dir.glob(f"{prefix}_*.nc")):
                with xr.open_dataset(f) as ds:
                    ds = ds.load()
                for v in ds.data_vars:
                    ds[v] = ds[v].where(ds.time <= t, 1000.0)
                ds.to_netcdf(Path(tmp) / f.name)
            e5.SOURCES["poisoned"] = (Path(tmp), prefix)
            try:
                dirty = _state("poisoned", fold)
            finally:
                e5.SOURCES.pop("poisoned", None)
        for part in ("nat", "loc", "reg"):
            a = clean[part][clean[part].date == t].reset_index(drop=True)
            b = dirty[part][dirty[part].date == t].reset_index(drop=True)
            if len(a) == 0 or not a.equals(b):
                bad.append(f"{t.date()} {part}")
    return {"dates": [str(pd.Timestamp(t).date()) for t in sample], "violations": bad,
            "ok": not bad}


def determinism_and_agreement() -> tuple[dict, dict]:
    pinned = pd.read_parquet(F.fold_path(2019, "strict"))
    with tempfile.TemporaryDirectory() as tmp:
        cand = pd.read_parquet(SB.candidate_fold(2019, Path(tmp)))
    if not SB.same_rows(cand, pinned):
        raise RuntimeError("fold 2019: rows or labels differ between sources")
    feats = list(T.BustModel.load(F.model_path(2019, "strict")).features)
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(cand))
    outs, secs = [], []
    for _ in range(2):
        t0 = time.time()
        outs.append(T.BustModel().fit(tr, va, features=feats).predict_proba(va))
        secs.append(round(time.time() - t0, 1))
    y = va.bust.to_numpy(float)
    inc = T.BustModel.load(F.model_path(2019, "strict"))
    va_i = pinned[pinned.split == "val"].dropna(subset=["bust"])
    det = {"bit_identical": bool(np.array_equal(outs[0], outs[1])), "val_year": 2018,
           "candidate_val_auroc": M.auroc(y, outs[0]),
           "incumbent_val_auroc": M.auroc(va_i.bust.to_numpy(float), inc.predict_proba(va_i)),
           "floor": SMOKE_FLOOR, "fit_seconds": secs}
    det["ok"] = det["bit_identical"] and det["candidate_val_auroc"] >= SMOKE_FLOOR

    years = pinned.year.isin(B2.GATE_YEARS).to_numpy()
    agree = {}
    for c in feats:
        a, b = cand[c].to_numpy(float)[years], pinned[c].to_numpy(float)[years]
        if np.array_equal(a, b, equal_nan=True):
            continue
        ok = np.isfinite(a) & np.isfinite(b)
        agree[c] = {"r": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None,
                    "n": int(ok.sum())}
    return det, {"years": list(B2.GATE_YEARS), "n_differing": len(agree), "features": agree}


def main() -> int:
    t0 = time.time()
    bt = json.loads(SB.BACKTEST.read_text(encoding="utf-8"))
    print("6. pinning the inputs ...", flush=True)
    pins = SB.write_inputs()
    print(f"  {len(pins)} files -> {SB.INPUTS.name}", flush=True)
    print("1. neutrality (ERA5 folds rebuilt through the changed builder) ...", flush=True)
    c1 = neutrality(bt)
    print("2. gates G1 and G2 on 2016-2017 ...", flush=True)
    c2 = gates()
    print(f"  G1 {c2['G1']}\n  G2 {c2['G2']}", flush=True)
    print("3. causality, poisoned hres_t0 after sampled 2018 issue days ...", flush=True)
    c3 = causality()
    print(f"  {c3}", flush=True)
    print("4-5. determinism on fold 2019 (validation 2018); agreement on 2016-2017 ...",
          flush=True)
    c4, c5 = determinism_and_agreement()
    print(f"  {c4}\n  {c5['n_differing']} inputs differ between the sources", flush=True)
    out = {"inputs_sha256": P.sha256_file(SB.INPUTS), "neutrality": c1, "gates": c2,
           "causality": c3, "determinism": c4, "agreement": c5,
           "ok": c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT}; all ok: {out['ok']} ({time.time() - t0:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
