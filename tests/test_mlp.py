"""The S3a MLP: reproducible, and fitted on training rows only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model.mlp import MLPModel, Preprocessor  # noqa: E402
from fbd.model.params import MLP_PARAMS  # noqa: E402

FAST = dict(MLP_PARAMS, max_epochs=3, seeds=[1, 2])
FEATS = ["a", "b", "c"]


def _frame(n, seed, nan_in=("b",)):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=n), "c": rng.normal(size=n)})
    df["bust"] = (a + rng.normal(0, 0.5, n) > 1.2).astype(float)
    for c in nan_in:
        df.loc[df.index[::7], c] = np.nan
    return df


def test_same_seed_same_fit():
    tr, va = _frame(600, 0), _frame(200, 1)
    one = MLPModel(params=FAST).fit(tr, va, FEATS)
    two = MLPModel(params=FAST).fit(tr, va, FEATS)
    assert np.array_equal(one.predict_raw(va), two.predict_raw(va))


def test_preprocessing_is_fitted_on_training_rows_only():
    tr, va = _frame(600, 0), _frame(200, 1)
    poisoned = va.copy()
    poisoned[FEATS] = 1e6
    a = MLPModel(params=FAST).fit(tr, va, FEATS).prep
    b = MLPModel(params=FAST).fit(tr, poisoned, FEATS).prep
    assert a.medians == b.medians and a.flagged == b.flagged
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.std, b.std)


def test_missing_indicators_are_for_features_missing_in_training():
    tr = _frame(300, 0, nan_in=("b",))
    va = _frame(100, 1, nan_in=("b", "c"))
    prep = Preprocessor.fit(tr, FEATS)
    assert prep.flagged == ["b"]
    assert prep.transform(va).shape == (100, 4)
    assert not np.isnan(prep.transform(va)).any()


def test_probabilities_are_calibrated_and_monotone_in_the_raw_score():
    tr, va = _frame(600, 0), _frame(200, 1)
    m = MLPModel(params=FAST).fit(tr, va, FEATS)
    raw, p = m.predict_raw(va), m.predict_proba(va)
    assert ((p >= 0) & (p <= 1)).all()
    order = np.argsort(raw, kind="stable")
    assert (np.diff(p[order]) >= -1e-12).all()
    assert m.predict_seeds(va).shape == (2, 200)


def test_save_and_load_round_trip(tmp_path):
    tr, va = _frame(600, 0), _frame(200, 1)
    m = MLPModel(params=FAST).fit(tr, va, FEATS)
    path = tmp_path / "mlp.joblib"
    m.save(path)
    back = MLPModel.load(path)
    assert np.array_equal(m.predict_proba(va), back.predict_proba(va))
    assert back.history == m.history
