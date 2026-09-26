"""S1c: the model + ENS combination sees validation-year rows only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import combine as C  # noqa: E402
from fbd.model import params as PR  # noqa: E402


def _fold():
    rows = []
    for year, split in ((2016, "train"), (2017, "train"), (2018, "val"), (2019, "test")):
        for d in range(1, 4):
            for lead in (1, 3, 5, 8):
                rows.append(dict(subdivision_id="A", init_date=pd.Timestamp(f"{year}-06-0{d}"),
                                 lead_day=lead, bust=float((d + lead) % 3 == 0),
                                 split=split, year=year))
    ds = pd.DataFrame(rows)
    ens = ds[["subdivision_id", "init_date", "lead_day"]].copy()
    ens["ens_spread"] = np.arange(len(ens), dtype=float)
    ens["ens_mean"] = 1.0
    return ds, ens


def test_validation_rows_are_the_validation_year_in_the_decision_band_only():
    ds, ens = _fold()
    val = C.validation_rows(ds, ens, lead_days=(3, 4, 5, 6, 7))
    assert set(val.split) == {"val"} and set(val.year) == {2018}
    assert set(val.lead_day) == {3, 5}
    assert len(val) == 6


def test_years_and_text():
    assert C.PRIMARY_YEARS == (2019, 2020, 2021) and C.YEARS == (2019, 2020, 2021, 2022)
    for v in ("model_better", "ens_better", "indistinguishable"):
        assert "2019–2021" in C.TEXT[v]


def test_combiner_params_are_frozen_and_hashable():
    p = PR.COMBINER_PARAMS
    assert p["inputs"] == ["logit of the uncalibrated model probability", "log(1 + ENS spread)"]
    assert PR.params_sha256(p) != PR.params_sha256(dict(p, C=1.0))


def test_combiner_is_deterministic_and_rewards_both_inputs():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 4000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (a + b + rng.normal(0, 1, n) > 2).astype(float)
    p_raw = 1 / (1 + np.exp(-a))
    spread = np.expm1(np.clip(b + 3, 0, None))
    one = C.Combiner().fit(p_raw, spread, y)
    two = C.Combiner().fit(p_raw, spread, y)
    assert np.array_equal(one.predict_proba(p_raw, spread), two.predict_proba(p_raw, spread))
    assert one.coefficients()["logit_p_model"] > 0 and one.coefficients()["log1p_ens_spread"] > 0
