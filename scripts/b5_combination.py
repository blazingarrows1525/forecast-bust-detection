"""B5: spatial + ENS against the served XGBoost + ENS, exactly as registered in
docs/PREREGISTRATION_B5.md.

    PYTHONPATH=src python scripts/b5_combination.py

Refuses unless the registration is committed as-is; backtest.json,
combination.json, spatial.json and s3c_inputs.json match their pins; every
fold dataset, XGBoost fold model and spatial fold model matches its record;
the combiner's parameters are unchanged; and every ENS year is complete.
Per fold it first reproduces the published numbers (XGBoost AUROC, spatial
AUROC, the served combination's AUROC) and stops if any differs. Then it fits
S1c's combiner on the spatial model's uncalibrated probability over the same
validation rows and scores the test year. Writes data/artifacts/b5_serving.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from fbd import config  # noqa: E402
from fbd.evaluate import b5 as B5  # noqa: E402
from fbd.evaluate import combine as C  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import grids as G  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_B5.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
COMBINATION = config.ARTIFACTS / "combination.json"
SPATIAL_JSON = config.ARTIFACTS / "candidates" / "spatial.json"
OUT = config.ARTIFACTS / "b5_serving.json"


def spatial_path(year: int) -> Path:
    return F.FOLD_DIR / f"candidate_spatial_{year}.joblib"


def records():
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    return load(BACKTEST), load(COMBINATION), load(SPATIAL_JSON)


def check_pins(bt: dict, sp: dict) -> None:
    if PR.params_sha256(PR.COMBINER_PARAMS) != load_combiner_hash():
        raise R.RegistrationError("the combiner's parameters changed since S1c")
    problems = G.verify_manifest()
    if problems:
        raise R.RegistrationError("pinned gridded inputs changed:\n  " + "\n  ".join(problems))
    for y in B5.YEARS:
        for path, want in ((F.fold_path(y, "strict"), bt["folds"][str(y)]["dataset_sha256"]),
                           (F.model_path(y, "strict"), bt["folds"][str(y)]["model_sha256"]),
                           (spatial_path(y), sp["folds"][str(y)]["model_sha256"])):
            if not path.exists() or P.sha256_file(path) != want:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match its record")


def load_combiner_hash() -> str:
    return json.loads(COMBINATION.read_text(encoding="utf-8"))["combiner_params_sha256"]


def guards():
    reg = R.guard(PREREG, {"backtest_sha256": BACKTEST, "combination_sha256": COMBINATION,
                           "spatial_json_sha256": SPATIAL_JSON,
                           "spatial_inputs_sha256": G.MANIFEST}, repo=ROOT)
    bt, cb, sp = records()
    check_pins(bt, sp)
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in sorted({*B5.YEARS, *(F.FOLDS[t].val[0] for t in B5.YEARS)}):
        R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    return reg, ens, bt, cb, sp


def reproduce(year: int, ens, lead_days, bt: dict, cb: dict, sp: dict) -> dict:
    """Everything already published for this fold, recomputed and checked."""
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.model import train as T
    from fbd.model.spatial import SpatialModel

    ds = pd.read_parquet(F.fold_path(year, "strict"))
    xgb = T.BustModel.load(F.model_path(year, "strict"))
    spatial = SpatialModel.load(spatial_path(year))
    val = C.validation_rows(ds, ens, lead_days=lead_days)
    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    comparator, p_ens = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                            test.ens_spread_rel.to_numpy(float))
    comb_xgb = C.Combiner().fit(xgb.predict_raw(val), val.ens_spread.to_numpy(float),
                                val.bust.to_numpy(float))
    p_comb_xgb = comb_xgb.predict_proba(xgb.predict_raw(test), test.ens_spread.to_numpy(float))
    p_xgb = xgb.predict_proba(test)
    p_spatial = spatial.predict_proba(test)
    got = {"xgboost": M.auroc(y, p_xgb), "spatial": M.auroc(y, p_spatial),
           "combination": M.auroc(y, p_comb_xgb)}
    want = {"xgboost": bt["folds"][str(year)]["model_auroc"],
            "spatial": sp["folds"][str(year)]["candidate_auroc"],
            "combination": cb["folds"][str(year)]["auroc"]["combination"]}
    return {"ds": ds, "val": val, "test": test, "y": y, "comparator": comparator,
            "p_ens": p_ens, "p_xgb": p_xgb, "p_spatial": p_spatial, "p_comb_xgb": p_comb_xgb,
            "comb_xgb": comb_xgb, "spatial": spatial, "got": got, "want": want,
            "reproduced": got == want}


def score_fold(year: int, ens, lead_days, bt: dict, cb: dict, sp: dict):
    from fbd.evaluate import metrics as M

    r = reproduce(year, ens, lead_days, bt, cb, sp)
    if not r["reproduced"]:
        raise RuntimeError(f"fold {year}: published numbers not reproduced: "
                           f"{r['got']} vs {r['want']}")
    val, test, y = r["val"], r["test"], r["y"]
    comb_sp = C.Combiner().fit(r["spatial"].predict_raw(val), val.ens_spread.to_numpy(float),
                               val.bust.to_numpy(float))
    p_comb_sp = comb_sp.predict_proba(r["spatial"].predict_raw(test),
                                      test.ens_spread.to_numpy(float))
    init = pd.to_datetime(test.init_date)
    rows = pd.DataFrame({
        "year": year, "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(), "bust": y,
        "p_comb_spatial": p_comb_sp, "p_comb_xgb": r["p_comb_xgb"], "p_ens": r["p_ens"],
        "p_spatial": r["p_spatial"], "p_xgb": r["p_xgb"],
    })
    info = {"comparator": r["comparator"], "n_rows": int(len(rows)),
            "n_fit_rows": int(len(val)),
            "fit_year": int(pd.to_datetime(val.init_date).dt.year.iloc[0]),
            "auroc": {"spatial_combination": M.auroc(y, p_comb_sp), **r["got"],
                      "ens": M.auroc(y, r["p_ens"])},
            "brier_skill": {"spatial_combination": M.brier_skill_score(y, p_comb_sp),
                            "combination": M.brier_skill_score(y, r["p_comb_xgb"])},
            "ece": {"spatial_combination": M.expected_calibration_error(y, p_comb_sp),
                    "combination": M.expected_calibration_error(y, r["p_comb_xgb"])},
            "coefficients": {"spatial": comb_sp.coefficients(),
                             "xgboost": r["comb_xgb"].coefficients()}}
    return rows, info


def main() -> int:
    t0 = time.time()
    try:
        reg, ens, bt, cb, sp = guards()
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in B5.YEARS:
        rows, info = score_fold(y, ens, lead_days, bt, cb, sp)
        frames.append(rows)
        folds[str(y)] = info
        a = info["auroc"]
        print(f"fold {y}: spatial + ENS {a['spatial_combination']:.4f}, served "
              f"{a['combination']:.4f} (spatial {a['spatial']:.4f}, XGBoost "
              f"{a['xgboost']:.4f}, ENS {a['ens']:.4f})", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.mean_margin(rows, "p_comb_spatial", "p_comb_xgb", B5.YEARS, n_boot, seed,
                          alpha=0.05)
    v = B5.verdict(prim["lo"], prim["hi"])
    prim.update(verdict=v, text=B5.TEXT[v], consequence=B5.CONSEQUENCE[v])
    print(f"primary, mean over {list(B5.YEARS)} at 95%: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    per_year = BS.per_year(rows, "p_comb_spatial", "p_comb_xgb", n_year, seed,
                           who="the served combination", whom="spatial + ENS")
    secondary = {
        "spatial_combination_minus_ens": {
            "mean": BS.mean_margin(rows, "p_comb_spatial", "p_ens", B5.ENS_YEARS, n_boot, seed),
            "per_year": BS.per_year(rows, "p_comb_spatial", "p_ens", n_year, seed,
                                    whom="spatial + ENS")},
        "spatial_combination_minus_spatial": {
            "mean": BS.mean_margin(rows, "p_comb_spatial", "p_spatial", B5.YEARS, n_boot, seed),
            "per_year": BS.per_year(rows, "p_comb_spatial", "p_spatial", n_year, seed,
                                    who="the spatial model alone", whom="spatial + ENS")},
    }
    for k, s in secondary.items():
        m = s["mean"]
        print(f"secondary, {k}: {m['point']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]", flush=True)
    payload = {
        "study": "B5", "registration_sha256": P.sha256_file(PREREG),
        "backtest_sha256": P.sha256_file(BACKTEST),
        "combination_sha256": P.sha256_file(COMBINATION),
        "spatial_json_sha256": P.sha256_file(SPATIAL_JSON),
        "combiner_params_sha256": PR.params_sha256(PR.COMBINER_PARAMS),
        "row_rule": ("per fold: both combiners fitted on the validation year's Day 3-7 rows "
                     "with a label and ENS spread; scored on the fold's S1b comparison rows"),
        "folds": folds, "primary": prim, "per_year": per_year, "secondary": secondary,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
