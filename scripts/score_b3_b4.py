"""Score B3 and B4 exactly as registered (docs/PREREGISTRATION_B3.md, _B4.md).

    PYTHONPATH=src python scripts/score_b3_b4.py

Refuses unless both registrations are committed as-is, backtest.json and
combination.json match their pins, every fold dataset and model matches
backtest.json, the combiner's parameters are unchanged, and every ENS year is
complete. Per fold it first reproduces the served combination's published
Day 3-7 AUROC and stops if it differs. Then:
  B4: a combiner fitted on the validation year's Day 1-2 rows against the
      served (Day 3-7-fitted) combiner, on the test year's Day 1-2 rows;
  B3: the beta-family combiner against the served form, on the comparison rows.
Writes data/artifacts/b4_short_leads.json and data/artifacts/b3_tail.json.
"""
from __future__ import annotations

# Windows: torch (if present) must load before scikit-learn (see fbd.model.mlp).
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
from fbd.evaluate import refine as RF  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG_B3 = ROOT / "docs" / "PREREGISTRATION_B3.md"
PREREG_B4 = ROOT / "docs" / "PREREGISTRATION_B4.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
COMBINATION = config.ARTIFACTS / "combination.json"
OUT_B3 = config.ARTIFACTS / "b3_tail.json"
OUT_B4 = config.ARTIFACTS / "b4_short_leads.json"


def records():
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    return load(BACKTEST), load(COMBINATION)


def check_pins(bt: dict, cb: dict) -> None:
    if PR.params_sha256(PR.COMBINER_PARAMS) != cb["combiner_params_sha256"]:
        raise R.RegistrationError("the combiner's parameters changed since S1c")
    for y in RF.YEARS:
        for path, key in ((F.fold_path(y, "strict"), "dataset_sha256"),
                          (F.model_path(y, "strict"), "model_sha256")):
            if not path.exists() or P.sha256_file(path) != bt["folds"][str(y)][key]:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match backtest.json")


def guards():
    pins = {"backtest_sha256": BACKTEST, "combination_sha256": COMBINATION}
    regs = {"B3": R.guard(PREREG_B3, pins, repo=ROOT), "B4": R.guard(PREREG_B4, pins, repo=ROOT)}
    bt, cb = records()
    check_pins(bt, cb)
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in sorted({*RF.YEARS, *(F.FOLDS[t].val[0] for t in RF.YEARS)}):
        for reg in regs.values():
            R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    return regs, ens, bt, cb


def served(year: int, ens, cb: dict):
    """The fold's data, model and served combiner; the published AUROC reproduced."""
    from fbd.evaluate import metrics as M
    from fbd.model import train as T

    ds = pd.read_parquet(F.fold_path(year, "strict"))
    xgb = T.BustModel.load(F.model_path(year, "strict"))
    val37 = C.validation_rows(ds, ens, lead_days=config.DECISION_BAND)
    test37, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
    comb = C.Combiner().fit(xgb.predict_raw(val37), val37.ens_spread.to_numpy(float),
                            val37.bust.to_numpy(float))
    p = comb.predict_proba(xgb.predict_raw(test37), test37.ens_spread.to_numpy(float))
    got = M.auroc(test37.bust.to_numpy(float), p)
    want = cb["folds"][str(year)]["auroc"]["combination"]
    return {"ds": ds, "xgb": xgb, "val37": val37, "test37": test37, "comb": comb,
            "p37": p, "got": got, "want": want, "reproduced": got == want}


def _frame(year, test, **cols):
    init = pd.to_datetime(test.init_date)
    return pd.DataFrame({"year": year, "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
                         "bust": test.bust.to_numpy(float), **cols})


def score_fold(year: int, ens, cb: dict):
    from fbd.evaluate import metrics as M

    s = served(year, ens, cb)
    if not s["reproduced"]:
        raise RuntimeError(f"fold {year}: served combination {s['got']!r} does not reproduce "
                           f"combination.json {s['want']!r}")
    xgb, ds = s["xgb"], s["ds"]
    # B4: Day 1-2
    val12 = C.validation_rows(ds, ens, lead_days=RF.SHORT_LEADS)
    test12, _ = E.comparison_rows(ds, ens, lead_days=RF.SHORT_LEADS)
    raw12 = xgb.predict_raw(test12)
    c12 = C.Combiner().fit(xgb.predict_raw(val12), val12.ens_spread.to_numpy(float),
                           val12.bust.to_numpy(float))
    b4 = _frame(year, test12,
                p_short=c12.predict_proba(raw12, test12.ens_spread.to_numpy(float)),
                p_served=s["comb"].predict_proba(raw12, test12.ens_spread.to_numpy(float)),
                p_model=xgb.predict_proba(test12))
    # B3: Day 3-7, the beta family
    val37, test37 = s["val37"], s["test37"]
    beta = RF.BetaCombiner().fit(xgb.predict_raw(val37), val37.ens_spread.to_numpy(float),
                                 val37.bust.to_numpy(float))
    b3 = _frame(year, test37,
                p_beta=beta.predict_proba(xgb.predict_raw(test37),
                                          test37.ens_spread.to_numpy(float)),
                p_served=s["p37"])
    y12, y37 = b4.bust.to_numpy(float), b3.bust.to_numpy(float)
    info4 = {"n_rows": int(len(b4)), "n_fit_rows": int(len(val12)),
             "auroc": {k: M.auroc(y12, b4[f"p_{k}"]) for k in ("short", "served", "model")},
             "coefficients": {"short": c12.coefficients(), "served": s["comb"].coefficients()}}
    info3 = {"n_rows": int(len(b3)),
             "tail_gap": {k: RF.tail_gap(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
             "brier": {k: M.brier(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
             "ece": {k: M.expected_calibration_error(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
             "auroc": {k: M.auroc(y37, b3[f"p_{k}"]) for k in ("beta", "served")},
             "coefficients": {"beta": beta.coefficients(), "served": s["comb"].coefficients()}}
    return b4, info4, b3, info3


def main() -> int:
    t0 = time.time()
    try:
        regs, ens, bt, cb = guards()
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2

    from fbd.evaluate import backtest_stats as BS
    from fbd.evaluate import metrics as M
    from fbd.evaluate import uncertainty as U

    f4, f3, i4, i3 = [], [], {}, {}
    for y in RF.YEARS:
        b4, info4, b3, info3 = score_fold(y, ens, cb)
        f4.append(b4)
        f3.append(b3)
        i4[str(y)], i3[str(y)] = info4, info3
        print(f"fold {y}: B4 Day 1-2 short {info4['auroc']['short']:.4f} served "
              f"{info4['auroc']['served']:.4f} model {info4['auroc']['model']:.4f}; B3 tail gap "
              f"beta {info3['tail_gap']['beta']:.4f} served {info3['tail_gap']['served']:.4f}",
              flush=True)
    r4, r3 = pd.concat(f4, ignore_index=True), pd.concat(f3, ignore_index=True)

    g4 = regs["B4"]
    n4, s4, ny4 = int(g4["n_boot"]), int(g4["seed"]), int(g4["n_boot_per_year"])
    prim4 = BS.mean_margin(r4, "p_short", "p_served", RF.YEARS, n4, s4, alpha=0.05)
    v4 = RF.b4_verdict(prim4["lo"], prim4["hi"])
    prim4.update(verdict=v4, text=RF.B4_TEXT[v4])
    print(f"B4 primary: {prim4['point']:+.4f} [{prim4['lo']:+.4f}, {prim4['hi']:+.4f}] -> "
          f"{prim4['text']}", flush=True)
    out4 = {"study": "B4", "registration_sha256": P.sha256_file(PREREG_B4),
            "folds": i4, "primary": prim4,
            "per_year": BS.per_year(r4, "p_short", "p_served", ny4, s4,
                                    who="the served extrapolation", whom="the Day 1–2 fit"),
            "secondary": {"short_minus_model": {
                "mean": BS.mean_margin(r4, "p_short", "p_model", RF.YEARS, n4, s4),
                "per_year": BS.per_year(r4, "p_short", "p_model", ny4, s4,
                                        who="the model alone", whom="the Day 1–2 fit")}}}
    OUT_B4.write_text(json.dumps(out4, indent=2, default=float), encoding="utf-8")

    g3 = regs["B3"]
    n3, s3 = int(g3["n_boot"]), int(g3["seed"])

    def margin(metric, years=RF.YEARS, n=n3):
        r = r3[r3.year.isin(list(years))]
        iv = U.stratified_mean_difference(metric, r.bust.to_numpy(float), r.p_beta.to_numpy(float),
                                          r.p_served.to_numpy(float), r.init_date.to_numpy(),
                                          r.year.to_numpy(), n_boot=n, seed=s3, alpha=0.05)
        return {**BS._interval(iv), "years": [int(y) for y in years]}

    tail, brier = margin(RF.tail_gap), margin(M.brier)
    v3 = RF.b3_verdict(tail, brier)
    print(f"B3 tail gap: {tail['point']:+.4f} [{tail['lo']:+.4f}, {tail['hi']:+.4f}]; Brier "
          f"{brier['point']:+.5f} [{brier['lo']:+.5f}, {brier['hi']:+.5f}] -> {RF.B3_TEXT[v3]}",
          flush=True)
    per_year3 = {str(y): {"tail_gap": margin(RF.tail_gap, (y,), int(g3["n_boot_per_year"])),
                          "brier": margin(M.brier, (y,), int(g3["n_boot_per_year"]))}
                 for y in RF.YEARS}
    out3 = {"study": "B3", "registration_sha256": P.sha256_file(PREREG_B3), "folds": i3,
            "primary": {"tail_gap": tail, "brier": brier, "verdict": v3, "text": RF.B3_TEXT[v3]},
            "per_year": per_year3,
            "secondary": {"auroc": margin(M.auroc), "ece": margin(M.expected_calibration_error)}}
    OUT_B3.write_text(json.dumps(out3, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT_B4.name} and {OUT_B3.name} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
