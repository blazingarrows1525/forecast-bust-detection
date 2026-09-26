"""Run exactly the analysis registered in docs/PREREGISTRATION_S1.md.

Refuses to run unless the registration is committed as-is, the frozen model and
dataset match their registered hashes, and all 2022 ENS dates are on disk.

    PYTHONPATH=src python scripts/settle_ens.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.evaluate import uncertainty as U  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S1.md"
OUT = config.ARTIFACTS / "ens_settlement.json"
MODEL = config.ARTIFACTS / "bust_model.joblib"
DATASET = config.PROCESSED / "dataset.parquet"
#: Registered: exploratory breakdowns use fewer resamples than the primary.
EXPLORATORY_N_BOOT = 2000


def _clusters(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(frame.init_date).dt.strftime("%Y-%m-%d").to_numpy()


def main() -> int:
    t0 = time.time()
    try:
        reg = R.guard(PREREG, {"model_sha256": MODEL, "dataset_sha256": DATASET}, repo=ROOT)
        ens = E.load_ens()
        have = E.dates_by_year(ens).get(2022, 0)
        R.check_complete(have, int(reg["required_ens_dates_2022"]))
    except (R.RegistrationError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot = int(reg["seed"]), int(reg["n_boot"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]

    # Imported only after the guards pass: they pull in scikit-learn/xgboost.
    from fbd.evaluate import settle as S
    from fbd.model import baselines as B
    from fbd.model import train as T

    ds = pd.read_parquet(DATASET)
    model = T.BustModel.load(MODEL)
    test, fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    p = model.predict_proba(test)
    clusters = _clusters(test)
    name, comp = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                     test.ens_spread_rel.to_numpy(float))

    print(f"primary: {len(test):,} rows over {len(np.unique(clusters))} dates; "
          f"comparator = {name} spread; {n_boot:,} resamples, seed {seed}", flush=True)
    prim = S.primary(y, p, comp, clusters, n_boot, seed)
    prim["comparator"] = name
    print(f"  model minus ENS  {prim['point']:+.4f} [{prim['lo']:+.4f}, {prim['hi']:+.4f}]"
          f"  -> {prim['text']}", flush=True)

    fit21 = fit[(pd.to_datetime(fit.init_date).dt.year == 2021)
                & fit.lead_day.isin(lead_days)]
    have21 = E.dates_by_year(ens).get(2021, 0)
    need21 = int(reg["required_ens_dates_2021"])
    secondary: dict = {}
    if have21 < need21:
        # The primary never waits on 2021; the secondaries that fit on it do.
        skipped = {"skipped": f"2021 incomplete: {have21}/{need21} ENS dates on disk"}
        secondary["a_calibrated_2021"] = skipped
        secondary["b_model_plus_ens"] = skipped
        print(f"secondary (a), (b): {skipped['skipped']}", flush=True)
    else:
        print("secondary (a) calibrated on 2021 ...", flush=True)
        secondary["a_calibrated_2021"] = S.secondary_calibrated(test, fit, p, clusters,
                                                                n_boot, seed)
        print("secondary (b) model + ENS ...", flush=True)
        secondary["b_model_plus_ens"] = S.secondary_combined(
            test, fit21, p, model.predict_proba(fit21), clusters, n_boot, seed)
    print("secondary (c) pooled calibration ...", flush=True)
    p_pool = B.SpreadBaseline("ens_spread").fit(fit).predict_proba(test)
    secondary["c_pooled_calibration"] = {
        "auroc": S.interval(U.paired_difference(M.auroc, y, p, p_pool, clusters,
                                                n_boot=n_boot, seed=seed)),
        "fit_years": sorted(int(x) for x in pd.to_datetime(fit.init_date).dt.year.unique()),
        "note": "continuity with the published pooled fit (train + val ENS rows)"}

    print("exploratory ...", flush=True)
    all_leads, _ = E.comparison_rows(ds, ens, lead_days=range(1, 11))
    ya = all_leads.bust.to_numpy(float)
    pa = model.predict_proba(all_leads)
    _n, compa = S.choose_comparator(ya, all_leads.ens_spread.to_numpy(float),
                                    all_leads.ens_spread_rel.to_numpy(float))
    legacy = set(pd.to_datetime(E.load_ens(include_shards=False).init_date).unique())
    origin = np.where(pd.to_datetime(test.init_date).isin(legacy), "original_40", "new")
    exploratory = {
        "by_lead": S.by_group(ya, pa, compa, all_leads.lead_day.to_numpy(), _clusters(all_leads),
                              EXPLORATORY_N_BOOT, seed),
        "by_month": S.by_group(y, p, comp, pd.to_datetime(test.init_date).dt.month.to_numpy(),
                               clusters, EXPLORATORY_N_BOOT, seed),
        "original_vs_new_dates": S.by_group(y, p, comp, origin, clusters,
                                            EXPLORATORY_N_BOOT, seed),
        "n_boot": EXPLORATORY_N_BOOT,
        "note": "exploratory: no claims are drawn from these",
    }

    payload = {
        "registration_sha256": P.sha256_file(PREREG),
        "hashes": {"model_sha256": P.sha256_file(MODEL),
                   "dataset_sha256": P.sha256_file(DATASET)},
        "n_ens_dates": E.dates_by_year(ens),
        "row_rule": ("comparison_rows(): Day 3-7 test rows with a label and ENS spread; "
                     "the model scores every row, including ones the served product "
                     "would refuse as out-of-distribution"),
        "primary": prim, "secondary": secondary, "exploratory": exploratory,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
