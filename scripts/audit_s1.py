"""S1 audit: reproduce what is published before registering anything new.

Runs on LEGACY ENS files only, so it measures the published baseline even after
new shards exist. Exit 1 if the 40-date margin does not reproduce to four
decimals: S1 must not build on a number the code cannot regenerate.

    PYTHONPATH=src python scripts/audit_s1.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compute_confidence_intervals as CCI  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402

OUT = config.ARTIFACTS / "s1_audit.json"

#: As published in README.md, D-014 and D-022 before S1: (value, places).
PUBLISHED = {
    "raw margin point": (0.0251, 4), "raw margin lo": (-0.0083, 4),
    "raw margin hi": (0.0580, 4), "calibrated margin point": (0.0403, 4),
    "model AUROC": (0.832, 3), "raw ENS AUROC": (0.807, 3),
    "calibrated ENS AUROC": (0.792, 3), "lagged proxy AUROC (calibrated)": (0.731, 3),
    "rows": (6732, 0), "init dates": (40, 0),
    "scored bust rate %": (3.4, 1), "refused bust rate %": (23.4, 1),
}

ROW_RULE = (
    "fbd.evaluate.ens.comparison_rows: inner join of dataset.parquet and ENS on "
    "(subdivision_id, init_date, lead_day); drop rows missing bust or ens_spread; "
    "test = split 'test' and lead_day in 3..7; fit = splits 'train' and 'val', all "
    "leads. The frozen model scores every test row, including rows the served "
    "product would refuse as out-of-distribution."
)


def main() -> int:
    checks, discrepancies = [], []

    def check(name, got):
        want, places = PUBLISHED[name]
        ok = round(float(got), places) == round(want, places)
        checks.append({"name": name, "got": float(got), "published": want, "ok": ok})
        print(f"  {'OK ' if ok else 'XX '} {name:32s} got {float(got):.4f}  published {want}")

    print("reproducing the published ENS margin (legacy ENS files only):")
    ens = E.load_ens(include_shards=False)
    res = CCI.ens_margin(2000, ens=ens)
    raw, cal = res["margins"]["raw ENS spread"], res["margins"]["calibrated ENS"]
    check("raw margin point", raw["point"])
    check("raw margin lo", raw["lo"])
    check("raw margin hi", raw["hi"])
    check("calibrated margin point", cal["point"])
    check("model AUROC", res["auroc"]["model"]["point"])
    check("raw ENS AUROC", res["auroc"]["raw ENS spread"]["point"])
    check("calibrated ENS AUROC", res["auroc"]["calibrated ENS"]["point"])
    check("rows", res["n_rows"])
    check("init dates", res["n_init_dates"])

    # The published 0.731 is the lagged spread after isotonic calibration on
    # the training years (evaluate_ens_baseline.py), not the raw spread.
    from fbd.model import baselines as B
    from fbd.model import train as T
    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    test, _fit = E.comparison_rows(ds, ens)
    tr_all, _va, _te = T.split_frames(ds)
    p_lag = B.SpreadBaseline("lagged_spread").fit(tr_all.dropna(subset=["bust"])).predict_proba(test)
    check("lagged proxy AUROC (calibrated)", M.auroc(test.bust.to_numpy(float), p_lag))

    legacy_2022 = sorted(pd.to_datetime(ens[ens.init_date.dt.year == 2022].init_date).unique())
    used = set(pd.to_datetime(test.init_date).unique())
    dropped = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in legacy_2022 if d not in used]
    if dropped:
        discrepancies.append(
            f"{len(legacy_2022)} legacy 2022 dates fetched but {len(used)} used: "
            f"{', '.join(dropped)} has no Day 3-7 label (its valid dates fall after "
            "30 September). Expected, and stated in the registration. The docstring "
            "of compute_confidence_intervals.py says '41 init dates'; it is 40.")

    db = config.ARTIFACTS / "bulletins.sqlite"
    if db.exists():
        con = sqlite3.connect(db)
        rates = {s: r for s, _n, r in con.execute(
            "SELECT status, COUNT(*), AVG(actual_bust) FROM bulletins "
            "WHERE init_date >= '2022-01-01' AND init_date < '2023-01-01' "
            "AND actual_bust IS NOT NULL GROUP BY status")}
        con.close()
        check("scored bust rate %", 100 * rates["OK"])
        check("refused bust rate %", 100 * rates["OUT_OF_DISTRIBUTION"])

    ci = json.loads((config.ARTIFACTS / "confidence_intervals.json").read_text())
    ece = ci["decision_band"]["4 XGBoost + isotonic"]["ece"]["point"]
    if round(ece, 4) != 0.0107:
        discrepancies.append(
            "landing.html's ECE card said 0.0107, the served-subset figure from "
            "plot_evaluation_figures.py, beside three cards that use the decision "
            f"band, where ECE is {ece:.4f}. The card now reads the decision-band "
            "value and says so.")
    discrepancies.append(
        "docs/FRONTEND_BUILT.md said every landing-page claim reads live from the "
        "API. The ENS sentence, the four stat cards and the refusal chart were "
        "hardcoded; the page fetched /api/metrics and discarded the result.")
    discrepancies.append(
        "README calls the ENS comparison 'on identical rows' without saying the "
        "model scores rows the product would refuse. The row rule is now stated.")

    hashes = {"bust_model.joblib": P.sha256_file(config.ARTIFACTS / "bust_model.joblib"),
              "dataset.parquet": P.sha256_file(config.PROCESSED / "dataset.parquet")}
    reproduced = all(c["ok"] for c in checks if c["name"].startswith("raw margin"))

    OUT.write_text(json.dumps({"reproduced": reproduced, "checks": checks,
                               "hashes": hashes, "row_rule": ROW_RULE,
                               "discrepancies": discrepancies}, indent=2),
                   encoding="utf-8")
    print(f"\nhashes: {json.dumps(hashes, indent=2)}")
    print("discrepancies:")
    for d in discrepancies:
        print(f"  - {d}")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    if not reproduced:
        print("\nSTOP: the published margin does not reproduce. Do not register.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
