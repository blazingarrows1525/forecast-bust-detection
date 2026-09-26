"""Run exactly the S1b backtest registered in docs/PREREGISTRATION_S1B.md.

Refuses unless the registration is committed as-is, the frozen dataset, model
and hyperparameters match their registered hashes, and all four test years'
ENS seasons are complete. Then rebuilds four strict folds, trains one model per
fold, scores each test year, and writes data/artifacts/backtest.json.

    PYTHONPATH=src python scripts/backtest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S1B.md"
OUT = config.ARTIFACTS / "backtest.json"
DATASET = config.PROCESSED / "dataset.parquet"
MODEL = config.ARTIFACTS / "bust_model.joblib"
YEARS = tuple(sorted(F.FOLDS))
KEYS = ["init_date", "subdivision_id", "lead_day"]


def guards():
    reg = R.guard(PREREG, {"dataset_sha256": DATASET, "model_sha256": MODEL}, repo=ROOT)
    R.check_params(reg, PR.params_sha256())
    if reg.get("mode") != "strict":
        raise R.RegistrationError(f"registered mode is {reg.get('mode')!r}, not 'strict'")
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in YEARS:
        R.check_complete(have.get(y, 0), int(reg[f"required_ens_dates_{y}"]), year=y)
    return reg, ens


def score_fold(test_year: int, ens: pd.DataFrame, lead_days) -> tuple[pd.DataFrame, dict]:
    import build_dataset as BD
    from train_model import Calibrated
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.model import baselines as B
    from fbd.model import train as T

    fold = F.FOLDS[test_year]
    path = F.fold_path(test_year, "strict")
    path.parent.mkdir(parents=True, exist_ok=True)
    BD.build_fold(test_year, "strict").to_parquet(path, index=False)
    ds = pd.read_parquet(path)

    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = T.available_features(ds)
    frozen_feats = T.BustModel.load(MODEL).features
    if list(feats) != list(frozen_feats):
        raise RuntimeError(f"fold {test_year} has {len(feats)} features, the frozen "
                           f"model {len(frozen_feats)}: a fold must not lose features")
    model = T.BustModel().fit(tr, va, features=feats)
    mpath = F.model_path(test_year, "strict")
    model.save(mpath)

    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    name, comp = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                     test.ens_spread_rel.to_numpy(float))
    proxy = Calibrated(B.SpreadBaseline("lagged_spread"), "proxy").fit(tr, va).predict_proba(test)
    init = pd.to_datetime(test.init_date)
    rows = pd.DataFrame({
        "year": test_year,
        "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(),
        "subdivision_id": test.subdivision_id.to_numpy(),
        "lead_day": test.lead_day.to_numpy(),
        "bust": y,
        "p_model": model.predict_proba(test),
        "p_ens": comp,
        "p_proxy": proxy,
    })
    info = {"train": list(fold.train), "val": list(fold.val), "comparator": name,
            "n_features": len(feats), "dataset_sha256": P.sha256_file(path),
            "model_sha256": P.sha256_file(mpath), "n_rows": int(len(rows)),
            "n_init_dates": int(rows.init_date.nunique()),
            "model_auroc": M.auroc(y, rows.p_model), "ens_auroc": M.auroc(y, comp)}
    return rows, info


def regime_leak(ens, lead_days, strict_2022: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Exploratory: strict fold-2022 model minus the frozen model, on the same 2022 rows."""
    from fbd.evaluate import metrics as M
    from fbd.evaluate import uncertainty as U
    from fbd.model import train as T

    test, _ = E.comparison_rows(pd.read_parquet(DATASET), ens, lead_days=lead_days)
    frozen = pd.DataFrame({
        "init_date": pd.to_datetime(test.init_date).dt.strftime("%Y-%m-%d").to_numpy(),
        "subdivision_id": test.subdivision_id.to_numpy(),
        "lead_day": test.lead_day.to_numpy(),
        "bust_frozen": test.bust.to_numpy(float),
        "p_frozen": T.BustModel.load(MODEL).predict_proba(test),
    })
    m = strict_2022.merge(frozen, on=KEYS, how="inner")
    if len(m) != len(strict_2022) or not (m.bust == m.bust_frozen).all():
        raise RuntimeError("fold-2022 rows or labels differ from dataset.parquet")
    iv = U.paired_difference(M.auroc, m.bust.to_numpy(float), m.p_model.to_numpy(float),
                             m.p_frozen.to_numpy(float), m.init_date.to_numpy(),
                             n_boot=n_boot, seed=seed)
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero,
            "note": ("strict fold-2022 model minus the frozen model on identical 2022 "
                     "rows: the regime-standardisation leak, plus retraining noise "
                     "bounded by the audit's 0.001")}


def main() -> int:
    t0 = time.time()
    try:
        reg, ens = guards()
    except (R.RegistrationError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]
    primary_years = [int(x) for x in reg["primary_years"].split(",")]

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in YEARS:
        print(f"fold {y}: build, train, score ...", flush=True)
        rows, info = score_fold(y, ens, lead_days)
        frames.append(rows)
        folds[str(y)] = info
        print(f"  {info['n_rows']:,} rows over {info['n_init_dates']} dates; "
              f"model {info['model_auroc']:.4f}, ENS ({info['comparator']}) "
              f"{info['ens_auroc']:.4f}", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.primary(rows, primary_years, n_boot, seed)
    print(f"primary, mean over {primary_years}: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    per_year = BS.per_year(rows, "p_model", "p_ens", n_year, seed)
    for y, iv in per_year.items():
        print(f"  {y}: {iv['point']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]"
              + (f"  -> {iv['statement']}" if iv["statement"] else ""), flush=True)

    print("secondary: lagged proxy ...", flush=True)
    secondary = {"proxy": {
        "mean": BS.mean_margin(rows, "p_model", "p_proxy", primary_years, n_boot, seed),
        "per_year": BS.per_year(rows, "p_model", "p_proxy", n_year, seed,
                                who="the lagged proxy")}}
    print("exploratory ...", flush=True)
    exploratory = {
        "by_month": BS.by_month(rows, "p_model", "p_ens", n_year, seed),
        "regime_leak_2022": regime_leak(ens, lead_days, rows[rows.year == 2022],
                                        n_year, seed),
        "n_boot": n_year,
        "note": "exploratory: no claims are drawn from these",
    }

    payload = {
        "registration_sha256": P.sha256_file(PREREG),
        "params_sha256": PR.params_sha256(),
        "hashes": {"dataset_sha256": P.sha256_file(DATASET),
                   "model_sha256": P.sha256_file(MODEL)},
        "n_ens_dates": E.dates_by_year(ens),
        "row_rule": ("per fold: comparison_rows() on that fold's dataset, Day 3-7 test "
                     "rows with a label and ENS spread; the fold model scores every row, "
                     "including ones the served product would refuse"),
        "folds": folds,
        "primary": prim, "per_year": per_year,
        "secondary": secondary, "exploratory": exploratory,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
