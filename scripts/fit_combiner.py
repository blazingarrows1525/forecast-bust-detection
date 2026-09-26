"""Fit the served combiner once, on 2021, and write data/artifacts/combiner.json.

D-029's recipe on the served model: logistic regression on [logit of the frozen
model's uncalibrated probability, log(1 + ENS spread)] over 2021's Day 3-7 rows.

    PYTHONPATH=src python scripts/fit_combiner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import combine as CB  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.model import combined as K  # noqa: E402
from fbd.model import train as T  # noqa: E402

MODEL = config.ARTIFACTS / "bust_model.joblib"


def main() -> int:
    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    val = CB.validation_rows(ds, E.load_ens(), lead_days=config.DECISION_BAND)
    years = sorted(int(y) for y in pd.to_datetime(val.init_date).dt.year.unique())
    assert years == list(config.VAL_YEARS), years
    model = T.BustModel.load(MODEL)
    p_raw = model.predict_raw(val)
    spread = val.ens_spread.to_numpy(float)
    fit = CB.Combiner().fit(p_raw, spread, val.bust.to_numpy(float))
    out = {**fit.coefficients(),
           "ref_log1p_spread": float(np.median(np.log1p(spread))),
           "fit_year": years[0], "n_rows": int(len(val)),
           "model_sha256": P.sha256_file(MODEL),
           "recipe": "D-029: logit(uncalibrated model probability), log1p(ENS spread); "
                     "Day 3-7 rows of the validation year"}
    K.COMBINER.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
