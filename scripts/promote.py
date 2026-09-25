"""Score one declared S3 candidate against the S1b XGBoost incumbent, as registered.

    PYTHONPATH=src python scripts/promote.py --candidate mlp

Refuses unless docs/PREREGISTRATION_S3.md is committed as-is, backtest.json and
every fold dataset and incumbent model match their hashes, the candidate is in
the declared slate with a registered parameter hash that the code still has,
and every ENS year is complete. Writes data/artifacts/candidates/<name>.json.
"""
from __future__ import annotations

# Windows: torch must load before scikit-learn (see fbd.model.mlp).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import promotion as PM  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S3.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
OUT_DIR = config.ARTIFACTS / "candidates"
#: slate name -> (module, class, frozen parameters). S3b and S3c add theirs.
CANDIDATES = {"mlp": ("fbd.model.mlp", "MLPModel", PR.MLP_PARAMS)}


def guards(name: str):
    reg = R.guard(PREREG, {"backtest_sha256": BACKTEST}, repo=ROOT)
    PM.check_candidate(name, reg)
    if name not in CANDIDATES:
        raise R.RegistrationError(f"{name!r} is declared but not implemented yet")
    got = PR.params_sha256(CANDIDATES[name][2])
    if reg[f"{name}_params_sha256"] != got:
        raise R.RegistrationError(f"{name} hyperparameters changed: registered "
                                  f"{reg[f'{name}_params_sha256']}, the code has {got}")
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    for y in PM.YEARS:
        info = bt["folds"][str(y)]
        for path, key in ((F.fold_path(y, "strict"), "dataset_sha256"),
                          (F.model_path(y, "strict"), "model_sha256")):
            if not path.exists() or P.sha256_file(path) != info[key]:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match backtest.json; "
                                          "re-run scripts/backtest.py's fold build")
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in PM.YEARS:
        R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    return reg, ens, bt


def score_fold(name: str, year: int, ens, lead_days, bt: dict):
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.model import train as T

    ds = pd.read_parquet(F.fold_path(year, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    incumbent = T.BustModel.load(F.model_path(year, "strict"))
    module, cls, _p = CANDIDATES[name]
    t0 = time.time()
    cand = getattr(importlib.import_module(module), cls)().fit(tr, va, list(incumbent.features))
    fit_s = time.time() - t0
    mpath = F.FOLD_DIR / f"candidate_{name}_{year}.joblib"
    cand.save(mpath)

    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    comparator, p_ens = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                            test.ens_spread_rel.to_numpy(float))
    init = pd.to_datetime(test.init_date)
    rows = pd.DataFrame({
        "year": year,
        "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(),
        "bust": y,
        "p_cand": cand.predict_proba(test),
        "p_inc": incumbent.predict_proba(test),
        "p_ens": p_ens,
    })
    inc_auc = M.auroc(y, rows.p_inc)
    if inc_auc != bt["folds"][str(year)]["model_auroc"]:
        raise RuntimeError(f"fold {year}: incumbent AUROC {inc_auc!r} does not reproduce "
                           f"backtest.json {bt['folds'][str(year)]['model_auroc']!r}")
    seeds = cand.predict_seeds(test) if hasattr(cand, "predict_seeds") else None
    info = {"candidate_auroc": M.auroc(y, rows.p_cand), "incumbent_auroc": inc_auc,
            "ens_auroc": M.auroc(y, p_ens), "comparator": comparator,
            "n_rows": int(len(rows)), "n_init_dates": int(rows.init_date.nunique()),
            "fit_seconds": round(fit_s, 1), "history": getattr(cand, "history", None),
            "model_sha256": P.sha256_file(mpath),
            "seed_auroc": ([M.auroc(y, s) for s in seeds] if seeds is not None else None)}
    return rows, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    name = ap.parse_args().candidate
    t0 = time.time()
    try:
        reg, ens, bt = guards(name)
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]
    alpha = PM.alpha_per_candidate()
    whom = f"the {PM.DISPLAY[name]}"

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in PM.YEARS:
        print(f"fold {y}: fit {name}, score ...", flush=True)
        rows, info = score_fold(name, y, ens, lead_days, bt)
        frames.append(rows)
        folds[str(y)] = info
        print(f"  {info['n_rows']:,} rows; {name} {info['candidate_auroc']:.4f}, "
              f"incumbent {info['incumbent_auroc']:.4f}, ENS {info['ens_auroc']:.4f} "
              f"({info['fit_seconds']}s)", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.mean_margin(rows, "p_cand", "p_inc", PM.YEARS, n_boot, seed, alpha=alpha)
    v = PM.verdict(prim["lo"], prim["hi"])
    prim.update(verdict=v, text=PM.verdict_text(v, name))
    print(f"primary, mean over {list(PM.YEARS)} at {1 - alpha:.2%}: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    per_year = BS.per_year(rows, "p_cand", "p_inc", n_year, seed,
                           who="the XGBoost incumbent", whom=whom)
    for y, iv in per_year.items():
        print(f"  {y}: {iv['point']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]", flush=True)

    print("secondary: against ENS spread ...", flush=True)
    secondary = {"ens": {
        "mean": BS.mean_margin(rows, "p_cand", "p_ens", PM.ENS_YEARS, n_boot, seed),
        "per_year": BS.per_year(rows, "p_cand", "p_ens", n_year, seed, whom=whom)}}
    exploratory = {
        "by_month": BS.by_month(rows, "p_cand", "p_inc", n_year, seed),
        "seed_auroc": {y: f["seed_auroc"] for y, f in folds.items()},
        "n_boot": n_year, "note": "exploratory: no claims are drawn from these"}

    payload = {
        "candidate": name,
        "registration_sha256": P.sha256_file(PREREG),
        "backtest_sha256": P.sha256_file(BACKTEST),
        "params_sha256": PR.params_sha256(CANDIDATES[name][2]),
        "row_rule": "S1b comparison_rows per fold; both models score every row",
        "folds": folds, "primary": prim, "per_year": per_year,
        "secondary": secondary, "exploratory": exploratory,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
