"""V3: do B3 and B4 hold on the spatial base? Exactly as registered in
docs/PREREGISTRATION_V3.md (design: 2026-10-06 v0.3.0, section 1).

    PYTHONPATH=src python scripts/score_v3.py --audit   # reproduce B5 only, nothing new
    PYTHONPATH=src python scripts/score_v3.py           # the registered run

B3's and B4's rules (fbd.evaluate.refine), with the S3c spatial fold models as
the base instead of XGBoost. Per fold it first reproduces b5_serving.json's
spatial + ENS AUROC (Day 3-7) and stops if it differs. Writes
data/artifacts/v3_spatial_refinements.json (or v3_audit.json with --audit).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch  # noqa: F401  (Windows: before scikit-learn)
except ImportError:
    pass

import argparse  # noqa: E402
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
from fbd.evaluate import refine as RF  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import grids as G  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_V3.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
SPATIAL_JSON = config.ARTIFACTS / "candidates" / "spatial.json"
B5_JSON = config.ARTIFACTS / "b5_serving.json"
OUT = config.ARTIFACTS / "v3_spatial_refinements.json"
AUDIT = config.ARTIFACTS / "v3_audit.json"


def spatial_path(year: int) -> Path:
    return F.FOLD_DIR / f"candidate_spatial_{year}.joblib"


def records():
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    return load(BACKTEST), load(SPATIAL_JSON), load(B5_JSON)


def check_pins(bt: dict, sp: dict) -> None:
    combination = json.loads((config.ARTIFACTS / "combination.json").read_text(encoding="utf-8"))
    if PR.params_sha256(PR.COMBINER_PARAMS) != combination["combiner_params_sha256"]:
        raise R.RegistrationError("the combiner's parameters changed since S1c")
    problems = G.verify_manifest()
    if problems:
        raise R.RegistrationError("pinned gridded inputs changed:\n  " + "\n  ".join(problems))
    for y in RF.YEARS:
        for path, want in ((F.fold_path(y, "strict"), bt["folds"][str(y)]["dataset_sha256"]),
                           (spatial_path(y), sp["folds"][str(y)]["model_sha256"])):
            if not path.exists() or P.sha256_file(path) != want:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match its record")


def base(year: int, ens, b5: dict) -> dict:
    """The fold's spatial model and B5's spatial combiner; B5's AUROC reproduced."""
    from fbd.evaluate import metrics as M
    from fbd.model.spatial import SpatialModel

    ds = pd.read_parquet(F.fold_path(year, "strict"))
    sp = SpatialModel.load(spatial_path(year))
    val37 = C.validation_rows(ds, ens, lead_days=config.DECISION_BAND)
    test37, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
    raw_val37, raw_test37 = sp.predict_raw(val37), sp.predict_raw(test37)
    comb = C.Combiner().fit(raw_val37, val37.ens_spread.to_numpy(float), val37.bust.to_numpy(float))
    p37 = comb.predict_proba(raw_test37, test37.ens_spread.to_numpy(float))
    got = M.auroc(test37.bust.to_numpy(float), p37)
    want = b5["folds"][str(year)]["auroc"]["spatial_combination"]
    return {"ds": ds, "sp": sp, "val37": val37, "test37": test37, "raw_val37": raw_val37,
            "raw_test37": raw_test37, "comb": comb, "p37": p37, "got": got, "want": want,
            "reproduced": got == want}


def _frame(year, test, **cols):
    init = pd.to_datetime(test.init_date)
    return pd.DataFrame({"year": year, "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
                         "bust": test.bust.to_numpy(float), **cols})


def score_fold(year, ens, b5):
    from fbd.evaluate import metrics as M

    b = base(year, ens, b5)
    if not b["reproduced"]:
        raise RuntimeError(f"fold {year}: B5's spatial combination not reproduced: "
                           f"{b['got']!r} vs {b['want']!r}")
    ds, sp = b["ds"], b["sp"]
    val12 = C.validation_rows(ds, ens, lead_days=RF.SHORT_LEADS)
    test12, _ = E.comparison_rows(ds, ens, lead_days=RF.SHORT_LEADS)
    raw12 = sp.predict_raw(test12)
    c12 = C.Combiner().fit(sp.predict_raw(val12), val12.ens_spread.to_numpy(float),
                           val12.bust.to_numpy(float))
    b4 = _frame(year, test12, p_short=c12.predict_proba(raw12, test12.ens_spread.to_numpy(float)),
                p_served=b["comb"].predict_proba(raw12, test12.ens_spread.to_numpy(float)))
    beta = RF.BetaCombiner().fit(b["raw_val37"], b["val37"].ens_spread.to_numpy(float),
                                 b["val37"].bust.to_numpy(float))
    b3 = _frame(year, b["test37"],
                p_beta=beta.predict_proba(b["raw_test37"], b["test37"].ens_spread.to_numpy(float)),
                p_served=b["p37"])
    y12, y37 = b4.bust.to_numpy(float), b3.bust.to_numpy(float)
    info = {"b4": {"n_rows": int(len(b4)), "n_fit_rows": int(len(val12)),
                   "auroc": {k: M.auroc(y12, b4[f"p_{k}"]) for k in ("short", "served")},
                   "coefficients": {"short": c12.coefficients(),
                                    "served": b["comb"].coefficients()}},
            "b3": {"n_rows": int(len(b3)),
                   "tail_gap": {k: RF.tail_gap(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
                   "brier": {k: M.brier(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
                   "ece": {k: M.expected_calibration_error(y37, b3[f"p_{k}"])
                           for k in ("beta", "served")},
                   "auroc": {k: M.auroc(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
                   "coefficients": {"beta": beta.coefficients()}}}
    return b4, b3, info


def audit() -> int:
    bt, sp, b5 = records()
    check_pins(bt, sp)
    ens = E.load_ens()
    folds = {}
    for y in RF.YEARS:
        b = base(y, ens, b5)
        folds[str(y)] = {"got": b["got"], "want": b["want"], "reproduced": b["reproduced"]}
        print(f"  {y}: spatial + ENS {b['got']:.6f} vs {b['want']:.6f} -> {b['reproduced']}",
              flush=True)
    out = {"folds": folds, "ok": all(f["reproduced"] for f in folds.values()),
           "note": "only B5's published spatial combination recomputed; no candidate fitted"}
    AUDIT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {AUDIT}; all ok: {out['ok']}")
    return 0 if out["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true")
    if ap.parse_args().audit:
        return audit()
    t0 = time.time()
    try:
        reg = R.guard(PREREG, {"backtest_sha256": BACKTEST, "spatial_json_sha256": SPATIAL_JSON,
                               "b5_json_sha256": B5_JSON,
                               "spatial_inputs_sha256": G.MANIFEST}, repo=ROOT)
        bt, sp, b5 = records()
        check_pins(bt, sp)
        ens = E.load_ens()
        have = E.dates_by_year(ens)
        for y in sorted({*RF.YEARS, *(F.FOLDS[t].val[0] for t in RF.YEARS)}):
            R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2

    from fbd.evaluate import backtest_stats as BS
    from fbd.evaluate import metrics as M
    from fbd.evaluate import uncertainty as U

    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    f4, f3, info = [], [], {}
    for y in RF.YEARS:
        b4, b3, i = score_fold(y, ens, b5)
        f4.append(b4)
        f3.append(b3)
        info[str(y)] = i
        print(f"fold {y}: Day 1-2 short {i['b4']['auroc']['short']:.4f} served "
              f"{i['b4']['auroc']['served']:.4f}; tail gap beta {i['b3']['tail_gap']['beta']:.4f} "
              f"served {i['b3']['tail_gap']['served']:.4f}", flush=True)
    r4, r3 = pd.concat(f4, ignore_index=True), pd.concat(f3, ignore_index=True)

    p4 = BS.mean_margin(r4, "p_short", "p_served", RF.YEARS, n_boot, seed, alpha=0.05)
    v4 = RF.b4_verdict(p4["lo"], p4["hi"])
    p4.update(verdict=v4, text=RF.B4_TEXT[v4], adopt=v4 == "short_lead_fit_better")
    print(f"V3-B4: {p4['point']:+.4f} [{p4['lo']:+.4f}, {p4['hi']:+.4f}] -> {p4['text']}",
          flush=True)

    def margin(metric, years=RF.YEARS, n=n_boot):
        r = r3[r3.year.isin(list(years))]
        iv = U.stratified_mean_difference(metric, r.bust.to_numpy(float), r.p_beta.to_numpy(float),
                                          r.p_served.to_numpy(float), r.init_date.to_numpy(),
                                          r.year.to_numpy(), n_boot=n, seed=seed, alpha=0.05)
        return {**BS._interval(iv), "years": [int(y) for y in years]}

    tail, brier = margin(RF.tail_gap), margin(M.brier)
    v3 = RF.b3_verdict(tail, brier)
    print(f"V3-B3: tail gap {tail['point']:+.4f} [{tail['lo']:+.4f}, {tail['hi']:+.4f}]; Brier "
          f"{brier['point']:+.5f} [{brier['lo']:+.5f}, {brier['hi']:+.5f}] -> {RF.B3_TEXT[v3]}",
          flush=True)
    out = {"study": "V3", "registration_sha256": P.sha256_file(PREREG), "folds": info,
           "b4": {"primary": p4,
                  "per_year": BS.per_year(r4, "p_short", "p_served", n_year, seed,
                                          who="the spatial extrapolation",
                                          whom="the spatial Day 1–2 fit")},
           "b3": {"primary": {"tail_gap": tail, "brier": brier, "verdict": v3,
                              "text": RF.B3_TEXT[v3], "adopt": v3 == "adopt"},
                  "per_year": {str(y): {"tail_gap": margin(RF.tail_gap, (y,), n_year),
                                        "brier": margin(M.brier, (y,), n_year)}
                               for y in RF.YEARS},
                  "secondary": {"auroc": margin(M.auroc), "ece": margin(M.expected_calibration_error)}}}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
