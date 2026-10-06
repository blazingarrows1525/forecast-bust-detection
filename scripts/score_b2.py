"""Score B2 exactly as registered in docs/PREREGISTRATION_B2.md: non-inferiority of the
operational-analysis features against the ERA5-built S1b incumbent.

    PYTHONPATH=src python scripts/score_b2.py

Refuses unless the registration is committed as-is, backtest.json and every
pinned fold dataset and model match their hashes, the pinned analysis_t0
inputs are unchanged, XGBoost's hyperparameters are the registered ones and
every ENS year is complete. Then, per fold: rebuild the fold with the
analysis_t0 state (into data/processed/backtest/b2/, never over the pinned
files), train the incumbent's model on it, and score S1b's comparison rows.
Writes data/artifacts/b2_noninferiority.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fbd import config  # noqa: E402
from fbd.evaluate import b2 as B2  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_B2.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
INPUTS = config.ARTIFACTS / "b2_inputs.json"
OUT = config.ARTIFACTS / "b2_noninferiority.json"
CAND_DIR = F.FOLD_DIR / "b2"
KEYS = ["subdivision_id", "init_date", "lead_day"]
SOURCE = "analysis_t0"


def input_files() -> list:
    """The pinned inputs: 14 analysis_t0 season files and the ERA5 static fields."""
    d = config.WB2_RAW / "analysis_t0"
    out = [d / f"analysis_t0_{k}_{y}.nc" for k in ("2d", "3d") for y in config.ALL_YEARS]
    return out + [config.WB2_RAW / "era5" / "era5_static.nc"]


def verify_inputs(path: Path = INPUTS) -> list:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for rel, want in sorted(body.items()):
        f = config.ROOT / rel
        got = P.sha256_file(f) if f.exists() else "missing"
        if got != want:
            problems.append(f"{rel}: pinned {want}, on disk {got}")
    return problems


def write_inputs(path: Path = INPUTS) -> dict:
    body = {f.resolve().relative_to(config.ROOT).as_posix(): P.sha256_file(f)
            for f in input_files()}
    Path(path).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def guards():
    reg = R.guard(PREREG, {"backtest_sha256": BACKTEST, "b2_inputs_sha256": INPUTS}, repo=ROOT)
    R.check_params(reg, PR.params_sha256())
    problems = verify_inputs()
    if problems:
        raise R.RegistrationError("pinned analysis_t0 inputs changed:\n  " + "\n  ".join(problems))
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    for y in B2.YEARS:
        info = bt["folds"][str(y)]
        for path, key in ((F.fold_path(y, "strict"), "dataset_sha256"),
                          (F.model_path(y, "strict"), "model_sha256")):
            if not path.exists() or P.sha256_file(path) != info[key]:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match backtest.json")
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in B2.YEARS:
        R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    return reg, ens, bt


def candidate_fold(year: int, out_dir: Path = CAND_DIR) -> Path:
    """The fold rebuilt with the operational state. Never written over a pinned file."""
    out_dir = Path(out_dir)
    path = out_dir / f"fold_{year}_strict_{SOURCE}.parquet"
    pinned = {F.fold_path(y, m).resolve() for y in F.FOLDS for m in F.MODES}
    if path.resolve() in pinned:
        raise RuntimeError("refusing to overwrite a pinned fold dataset")
    import build_dataset as BD

    out_dir.mkdir(parents=True, exist_ok=True)
    BD.build_fold(year, "strict", state_source=SOURCE).to_parquet(path, index=False)
    return path


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Identical rows, splits and labels: only the state inputs may differ."""
    cols = KEYS + ["split", "bust", "valid_date"]
    x = a[cols].reset_index(drop=True)
    y = b[cols].reset_index(drop=True)
    return len(x) == len(y) and x.equals(y)


def score_fold(year: int, ens, lead_days, bt: dict, out_dir: Path = CAND_DIR):
    from fbd.evaluate import metrics as M
    from fbd.model import train as T
    from fbd.ood import detector as OOD

    pinned = pd.read_parquet(F.fold_path(year, "strict"))
    t0 = time.time()
    cpath = candidate_fold(year, out_dir)
    cand_ds = pd.read_parquet(cpath)
    if not same_rows(cand_ds, pinned):
        raise RuntimeError(f"fold {year}: the rebuilt fold's rows or labels differ")
    incumbent = T.BustModel.load(F.model_path(year, "strict"))
    feats = list(incumbent.features)
    if T.available_features(cand_ds) != feats:
        raise RuntimeError(f"fold {year}: the rebuilt fold does not carry the 52 features")
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(cand_ds))
    model = T.BustModel().fit(tr, va, features=feats)
    mpath = Path(out_dir) / f"model_{year}_strict_{SOURCE}.joblib"
    model.save(mpath)
    fit_s = time.time() - t0

    test_c, _ = E.comparison_rows(cand_ds, ens, lead_days=lead_days)
    test_i, _ = E.comparison_rows(pinned, ens, lead_days=lead_days)
    if not same_rows(test_c, test_i):
        raise RuntimeError(f"fold {year}: comparison rows differ between the two sources")
    y = test_c.bust.to_numpy(float)
    init = pd.to_datetime(test_c.init_date)
    rows = pd.DataFrame({
        "year": year,
        "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(),
        "bust": y,
        "p_cand": model.predict_proba(test_c),
        "p_inc": incumbent.predict_proba(test_i),
    })
    inc_auc = M.auroc(y, rows.p_inc)
    if inc_auc != bt["folds"][str(year)]["model_auroc"]:
        raise RuntimeError(f"fold {year}: incumbent AUROC {inc_auc!r} does not reproduce "
                           "backtest.json")
    tr_i = pinned[pinned.split == "train"].dropna(subset=["bust"])
    refused = {
        "candidate": float(OOD.MahalanobisOOD().fit(tr, feats).is_ood(test_c).mean()),
        "incumbent": float(OOD.MahalanobisOOD().fit(tr_i, feats).is_ood(test_i).mean()),
    }
    info = {"candidate_auroc": M.auroc(y, rows.p_cand), "incumbent_auroc": inc_auc,
            "candidate_bss": M.brier_skill_score(y, rows.p_cand),
            "incumbent_bss": M.brier_skill_score(y, rows.p_inc),
            "refused_share": refused, "n_rows": int(len(rows)),
            "n_init_dates": int(rows.init_date.nunique()), "fit_seconds": round(fit_s, 1),
            "dataset_sha256": P.sha256_file(cpath), "model_sha256": P.sha256_file(mpath)}
    return rows, info


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    t0 = time.time()
    try:
        reg, ens, bt = guards()
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]
    margin = float(reg["margin"])

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in B2.YEARS:
        print(f"fold {y}: rebuild with {SOURCE}, train, score ...", flush=True)
        rows, info = score_fold(y, ens, lead_days, bt)
        frames.append(rows)
        folds[str(y)] = info
        print(f"  {info['n_rows']:,} rows; candidate {info['candidate_auroc']:.4f}, "
              f"incumbent {info['incumbent_auroc']:.4f}; refused "
              f"{info['refused_share']['candidate']:.1%} vs "
              f"{info['refused_share']['incumbent']:.1%} ({info['fit_seconds']}s)", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.mean_margin(rows, "p_cand", "p_inc", B2.YEARS, n_boot, seed, alpha=0.05)
    v = B2.verdict(prim["lo"], prim["hi"], margin)
    prim.update(verdict=v, text=B2.VERDICT_TEXT[v], margin=margin)
    print(f"primary, mean over {list(B2.YEARS)} at 95%: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] against -{margin} -> {prim['text']}",
          flush=True)
    per_year = BS.per_year(rows, "p_cand", "p_inc", n_year, seed,
                           who="the ERA5-built incumbent",
                           whom="the operational-analysis model")
    payload = {
        "study": "B2", "registration_sha256": P.sha256_file(PREREG),
        "backtest_sha256": P.sha256_file(BACKTEST), "inputs_sha256": P.sha256_file(INPUTS),
        "params_sha256": PR.params_sha256(), "source": SOURCE,
        "row_rule": "S1b comparison_rows per fold; both models score every row",
        "folds": folds, "primary": prim, "per_year": per_year,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
