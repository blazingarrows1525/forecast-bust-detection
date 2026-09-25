"""S1c: does the model plus ENS spread outrank ENS spread alone, across folds?

Runs exactly the test registered in docs/PREREGISTRATION_S1C.md. For each S1b
fold: fit the combiner on the validation year's Day 3-7 rows (uncalibrated
fold-model probability + ENS spread), score the test year, compare with ENS
spread alone. Writes data/artifacts/combination.json.

    PYTHONPATH=src python scripts/combine_ens.py
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
from fbd import config  # noqa: E402
from fbd.evaluate import combine as C  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S1C.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
MLP_JSON = config.ARTIFACTS / "candidates" / "mlp.json"
OUT = config.ARTIFACTS / "combination.json"


def mlp_path(year: int) -> Path:
    return F.FOLD_DIR / f"candidate_mlp_{year}.joblib"


def guards():
    reg = R.guard(PREREG, {"backtest_sha256": BACKTEST, "mlp_json_sha256": MLP_JSON}, repo=ROOT)
    if reg.get("combiner_params_sha256") != PR.params_sha256(PR.COMBINER_PARAMS):
        raise R.RegistrationError("the combiner's parameters changed since registration")
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    mj = json.loads(MLP_JSON.read_text(encoding="utf-8"))
    for y in C.YEARS:
        for path, want in ((F.fold_path(y, "strict"), bt["folds"][str(y)]["dataset_sha256"]),
                           (F.model_path(y, "strict"), bt["folds"][str(y)]["model_sha256"]),
                           (mlp_path(y), mj["folds"][str(y)]["model_sha256"])):
            if not path.exists() or P.sha256_file(path) != want:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match its record")
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in sorted({*C.YEARS, *(F.FOLDS[t].val[0] for t in C.YEARS)}):
        R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    return reg, ens, bt, mj


def score_fold(year: int, ens, lead_days, bt: dict, mj: dict):
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.model import train as T
    from fbd.model.mlp import MLPModel

    ds = pd.read_parquet(F.fold_path(year, "strict"))
    xgb = T.BustModel.load(F.model_path(year, "strict"))
    mlp = MLPModel.load(mlp_path(year))
    val = C.validation_rows(ds, ens, lead_days=lead_days)
    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    comparator, p_ens = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                            test.ens_spread_rel.to_numpy(float))

    combos = {}
    for name, model in (("xgb", xgb), ("mlp", mlp)):
        comb = C.Combiner().fit(model.predict_raw(val), val.ens_spread.to_numpy(float),
                                val.bust.to_numpy(float))
        combos[name] = (comb, comb.predict_proba(model.predict_raw(test),
                                                 test.ens_spread.to_numpy(float)))

    init = pd.to_datetime(test.init_date)
    rows = pd.DataFrame({
        "year": year,
        "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(),
        "bust": y,
        "p_comb": combos["xgb"][1],
        "p_ens": p_ens,
        "p_xgb": xgb.predict_proba(test),
        "p_mlp_comb": combos["mlp"][1],
    })
    xgb_auc, mlp_auc = M.auroc(y, rows.p_xgb), M.auroc(y, mlp.predict_proba(test))
    if xgb_auc != bt["folds"][str(year)]["model_auroc"]:
        raise RuntimeError(f"fold {year}: XGBoost AUROC does not reproduce backtest.json")
    if mlp_auc != mj["folds"][str(year)]["candidate_auroc"]:
        raise RuntimeError(f"fold {year}: MLP AUROC does not reproduce mlp.json")
    info = {"comparator": comparator, "n_rows": int(len(rows)),
            "n_init_dates": int(rows.init_date.nunique()),
            "n_fit_rows": int(len(val)), "fit_year": int(pd.to_datetime(val.init_date).dt.year.iloc[0]),
            "auroc": {"combination": M.auroc(y, rows.p_comb), "ens": M.auroc(y, p_ens),
                      "xgboost": xgb_auc, "mlp_combination": M.auroc(y, rows.p_mlp_comb),
                      "mlp": mlp_auc},
            "coefficients": {k: c[0].coefficients() for k, c in combos.items()}}
    return rows, info


def main() -> int:
    t0 = time.time()
    try:
        reg, ens, bt, mj = guards()
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in C.YEARS:
        rows, info = score_fold(y, ens, lead_days, bt, mj)
        frames.append(rows)
        folds[str(y)] = info
        a = info["auroc"]
        print(f"fold {y}: fit on {info['fit_year']} ({info['n_fit_rows']:,} rows); combination "
              f"{a['combination']:.4f}, ENS {a['ens']:.4f}, XGBoost {a['xgboost']:.4f}", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.mean_margin(rows, "p_comb", "p_ens", C.PRIMARY_YEARS, n_boot, seed)
    v = R.verdict(prim["lo"], prim["hi"])
    prim.update(verdict=v, text=C.TEXT[v])
    print(f"primary, mean over {list(C.PRIMARY_YEARS)}: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    per_year = BS.per_year(rows, "p_comb", "p_ens", n_year, seed, whom="the combination")
    for y, iv in per_year.items():
        print(f"  {y}: {iv['point']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]"
              + (f"  -> {iv['statement']}" if iv["statement"] else ""), flush=True)

    print("secondary ...", flush=True)
    secondary = {
        "combination_minus_xgboost": {
            "mean": BS.mean_margin(rows, "p_comb", "p_xgb", C.PRIMARY_YEARS, n_boot, seed),
            "per_year": BS.per_year(rows, "p_comb", "p_xgb", n_year, seed,
                                    who="XGBoost alone", whom="the combination")},
        "mlp_combination_minus_ens": {
            "mean": BS.mean_margin(rows, "p_mlp_comb", "p_ens", C.PRIMARY_YEARS, n_boot, seed),
            "per_year": BS.per_year(rows, "p_mlp_comb", "p_ens", n_year, seed,
                                    whom="the MLP combination")},
    }
    payload = {
        "registration_sha256": P.sha256_file(PREREG),
        "backtest_sha256": P.sha256_file(BACKTEST),
        "mlp_json_sha256": P.sha256_file(MLP_JSON),
        "combiner_params_sha256": PR.params_sha256(PR.COMBINER_PARAMS),
        "n_ens_dates": E.dates_by_year(ens),
        "row_rule": ("per fold: combiner fitted on the validation year's Day 3-7 rows with a "
                     "label and ENS spread; scored on that fold's S1b comparison rows"),
        "folds": folds, "primary": prim, "per_year": per_year, "secondary": secondary,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
