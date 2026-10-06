"""Fit the served combiner once, on 2021, and write data/artifacts/combiner.json.

    PYTHONPATH=src python scripts/fit_combiner.py                  # store v0.3.0 (spatial)
    PYTHONPATH=src python scripts/fit_combiner.py --base xgboost   # store v0.2.0, as before

v0.2.0 (D-029): logistic regression on [logit of the frozen XGBoost model's
uncalibrated probability, log(1 + ENS spread)] over 2021's Day 3-7 rows.

v0.3.0 (D-037): the same recipe on the served spatial model (the S3c fold-2022
model, verified against spatial.json), with the refinements that held on the
spatial base in the registered V3 check (v3_spatial_refinements.json):
  - the beta form on the main set if V3-B3 adopted it, else the logit form;
  - a separate logit set fitted on 2021's Day 1-2 rows if V3-B4 adopted it.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import combine as CB  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import refine as RF  # noqa: E402
from fbd.model import combined as K  # noqa: E402
from fbd.model import train as T  # noqa: E402

MODEL = config.ARTIFACTS / "bust_model.joblib"
SPATIAL_YEAR = config.TEST_YEARS[0]          # the fold whose test year is the served test year
V3 = config.ARTIFACTS / "v3_spatial_refinements.json"
SPATIAL_JSON = config.ARTIFACTS / "candidates" / "spatial.json"


def spatial_model_path() -> Path:
    return F.FOLD_DIR / f"candidate_spatial_{SPATIAL_YEAR}.joblib"


def load_spatial():
    """The served spatial model, refused unless it is the one S3c scored."""
    from fbd.model.spatial import SpatialModel

    path = spatial_model_path()
    want = json.loads(SPATIAL_JSON.read_text(encoding="utf-8"))["folds"][str(SPATIAL_YEAR)]
    if P.sha256_file(path) != want["model_sha256"]:
        raise SystemExit(f"{path.name} does not match spatial.json; refusing to serve it")
    return SpatialModel.load(path), path


def fit_xgboost() -> dict:
    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    val = CB.validation_rows(ds, E.load_ens(), lead_days=config.DECISION_BAND)
    years = sorted(int(y) for y in pd.to_datetime(val.init_date).dt.year.unique())
    assert years == list(config.VAL_YEARS), years
    model = T.BustModel.load(MODEL)
    p_raw = model.predict_raw(val)
    spread = val.ens_spread.to_numpy(float)
    fit = CB.Combiner().fit(p_raw, spread, val.bust.to_numpy(float))
    return {**fit.coefficients(),
            "ref_log1p_spread": float(np.median(np.log1p(spread))),
            "fit_year": years[0], "n_rows": int(len(val)),
            "model_sha256": P.sha256_file(MODEL),
            "recipe": "D-029: logit(uncalibrated model probability), log1p(ENS spread); "
                      "Day 3-7 rows of the validation year"}


def fit_spatial() -> dict:
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    use_beta = bool(v3["b3"]["primary"]["adopt"])
    use_short = bool(v3["b4"]["primary"]["adopt"])
    model, path = load_spatial()
    ds = pd.read_parquet(F.fold_path(SPATIAL_YEAR, "strict"))
    ens = E.load_ens()
    val37 = CB.validation_rows(ds, ens, lead_days=config.DECISION_BAND)
    years = sorted(int(y) for y in pd.to_datetime(val37.init_date).dt.year.unique())
    assert years == list(config.VAL_YEARS), years
    spread37 = val37.ens_spread.to_numpy(float)
    raw37 = model.predict_raw(val37)
    main = (RF.BetaCombiner() if use_beta else CB.Combiner()).fit(
        raw37, spread37, val37.bust.to_numpy(float))
    short = None
    n_short = 0
    if use_short:
        val12 = CB.validation_rows(ds, ens, lead_days=RF.SHORT_LEADS)
        short = CB.Combiner().fit(model.predict_raw(val12), val12.ens_spread.to_numpy(float),
                                  val12.bust.to_numpy(float)).coefficients()
        n_short = int(len(val12))
    return {
        "version": 2, "form": "beta" if use_beta else "logit",
        "main": main.coefficients(), "short": short, "short_leads": list(RF.SHORT_LEADS),
        "ref_log1p_spread": float(np.median(np.log1p(spread37))),
        "fit_year": years[0], "n_rows": int(len(val37)), "n_rows_short": n_short,
        "base_model": f"S3c spatial CNN, fold {SPATIAL_YEAR} (trained 2016-2020, "
                      f"calibrated 2021)",
        "model_sha256": P.sha256_file(path),
        "v3_sha256": P.sha256_file(V3),
        "refinements": {"B3_beta_form": use_beta, "B4_short_leads": use_short,
                        "B3_verdict": v3["b3"]["primary"]["verdict"],
                        "B4_verdict": v3["b4"]["primary"]["verdict"]},
        "recipe": ("D-037: the D-029 recipe on the spatial model's uncalibrated probability "
                   "over 2021's Day 3-7 rows" + ("; beta form (D-035, V3)" if use_beta else
                                                  "; logit form (V3 did not adopt the beta form)")
                   + ("; a separate logit set on 2021's Day 1-2 rows (D-036, V3)" if use_short
                      else "")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", choices=("spatial", "xgboost"), default="spatial")
    out = fit_spatial() if ap.parse_args().base == "spatial" else fit_xgboost()
    K.COMBINER.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
