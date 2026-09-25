"""S3a-C: does the MLP's lead over real ENS spread hold in 2018?

Runs exactly the test registered in docs/PREREGISTRATION_S3A_CONFIRM.md:
fold 2018 (train 2016, calibrate 2017) rebuilt in strict mode, the frozen MLP
and the default XGBoost fitted on it, both scored with ENS spread on 2018's
Day 3-7 comparison rows. Refuses unless the registration is committed as-is,
both models' hyperparameters match their registered hashes, and all 2018 ENS
dates are on disk. Writes data/artifacts/confirm_2018.json.

    PYTHONPATH=src python scripts/confirm_2018.py
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

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import promotion as PM  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S3A_CONFIRM.md"
OUT = config.ARTIFACTS / "confirm_2018.json"
FROZEN = config.ARTIFACTS / "bust_model.joblib"
YEAR = PM.CONFIRM_YEAR


def guards():
    reg = R.guard(PREREG, {}, repo=ROOT)
    fold = F.fold_for(YEAR)
    if (int(reg["year"]), reg["train_years"], reg["val_years"]) != (
            YEAR, ",".join(map(str, fold.train)), ",".join(map(str, fold.val))):
        raise R.RegistrationError("the confirmation fold differs from the registered one")
    for key, got in (("mlp_params_sha256", PR.params_sha256(PR.MLP_PARAMS)),
                     ("xgb_params_sha256", PR.params_sha256())):
        if reg.get(key) != got:
            raise R.RegistrationError(f"{key}: registered {reg.get(key)}, the code has {got}")
    ens = E.load_ens()
    R.check_complete(E.dates_by_year(ens).get(YEAR, 0), int(reg["required_ens_dates"]),
                     year=YEAR)
    return reg, ens


def _interval(iv) -> dict:
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero}


def main() -> int:
    t0 = time.time()
    try:
        reg, ens = guards()
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_sec = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_secondary"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]

    import build_dataset as BD
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.evaluate import uncertainty as U
    from fbd.model import train as T
    from fbd.model.mlp import MLPModel

    print(f"fold {YEAR}: build (strict), fit XGBoost and the MLP ...", flush=True)
    path = F.fold_path(YEAR, "strict")
    path.parent.mkdir(parents=True, exist_ok=True)
    BD.build_fold(YEAR, "strict").to_parquet(path, index=False)
    ds = pd.read_parquet(path)
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = T.available_features(ds)
    if list(feats) != list(T.BustModel.load(FROZEN).features):
        raise RuntimeError("fold 2018 lost features: it must match the frozen model's")
    xgb = T.BustModel().fit(tr, va, features=feats)
    xgb.save(F.model_path(YEAR, "strict"))
    t1 = time.time()
    mlp = MLPModel().fit(tr, va, feats)
    mlp_s = time.time() - t1
    mlp.save(F.FOLD_DIR / f"candidate_mlp_{YEAR}.joblib")

    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    comparator, p_ens = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                            test.ens_spread_rel.to_numpy(float))
    p_mlp, p_xgb = mlp.predict_proba(test), xgb.predict_proba(test)
    clusters = pd.to_datetime(test.init_date).dt.strftime("%Y-%m-%d").to_numpy()
    aucs = {"mlp": M.auroc(y, p_mlp), "xgboost": M.auroc(y, p_xgb), "ens": M.auroc(y, p_ens)}
    print(f"  {len(y):,} rows over {len(set(clusters))} dates; MLP {aucs['mlp']:.4f}, "
          f"XGBoost {aucs['xgboost']:.4f}, ENS ({comparator}) {aucs['ens']:.4f}", flush=True)

    prim = _interval(U.paired_difference(M.auroc, y, p_mlp, p_ens, clusters,
                                         n_boot=n_boot, seed=seed))
    v = R.verdict(prim["lo"], prim["hi"])
    prim.update(verdict=v, text=PM.CONFIRM_TEXT[v])
    print(f"primary: MLP minus ENS spread {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    secondary = {
        "xgboost_minus_ens": _interval(U.paired_difference(M.auroc, y, p_xgb, p_ens, clusters,
                                                           n_boot=n_sec, seed=seed)),
        "mlp_minus_xgboost": _interval(U.paired_difference(M.auroc, y, p_mlp, p_xgb, clusters,
                                                           n_boot=n_sec, seed=seed)),
    }
    for k, iv in secondary.items():
        print(f"  {k}: {iv['point']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]", flush=True)

    payload = {
        "registration_sha256": P.sha256_file(PREREG),
        "params_sha256": {"mlp": PR.params_sha256(PR.MLP_PARAMS), "xgboost": PR.params_sha256()},
        "n_ens_dates": E.dates_by_year(ens),
        "fold": {"test": YEAR, "train": list(F.fold_for(YEAR).train),
                 "val": list(F.fold_for(YEAR).val), "n_features": len(feats),
                 "n_rows": int(len(y)), "n_init_dates": int(len(set(clusters))),
                 "comparator": comparator, "auroc": aucs, "mlp_fit_seconds": round(mlp_s, 1),
                 "mlp_history": mlp.history, "dataset_sha256": P.sha256_file(path)},
        "row_rule": "comparison_rows() on the fold-2018 dataset: Day 3-7 rows with a label "
                    "and ENS spread; every row scored by both models",
        "primary": prim, "secondary": secondary,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
