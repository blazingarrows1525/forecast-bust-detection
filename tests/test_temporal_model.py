"""S3b model on a small synthetic fold. Skips where torch is absent (CI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model import sequence as Q  # noqa: E402
from fbd.model.params import TEMPORAL_PARAMS  # noqa: E402
from fbd.model.temporal import TemporalModel  # noqa: E402

FEATS = ["lead_day", "fcst_rain_mm", "x1"]


def _fold(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year, split in ((2016, "train"), (2017, "val"), (2018, "test")):
        for s in ("A", "B", "C"):
            for t in pd.date_range(f"{year}-06-01", f"{year}-06-30"):
                for L in range(1, 8):
                    fc = float(rng.gamma(2.0, 10.0))
                    ob = float(rng.gamma(2.0, 10.0))
                    row = dict(subdivision_id=s, init_date=t,
                               valid_date=t + pd.Timedelta(days=L - 1),
                               lead_day=L, fcst_rain_mm=fc, obs_rain_mm=ob, error=fc - ob,
                               bust=float(abs(fc - ob) > 25), x1=float(rng.normal()),
                               split=split, year=year)
                    for c in Q.WEATHER:
                        row[c] = float(rng.normal())
                    rows.append(row)
    return pd.DataFrame(rows)


def _small(**kw):
    p = dict(TEMPORAL_PARAMS, seeds=[1, 2], max_epochs=3, patience=2, threads=1)
    p.update(kw)
    return p


def _split(ds):
    return ds[ds.split == "train"], ds[ds.split == "val"], ds[ds.split == "test"]


def test_fit_predict_shapes_and_range():
    ds = _fold()
    tr, va, te = _split(ds)
    m = TemporalModel(dataset=ds, params=_small()).fit(tr, va, FEATS)
    p = m.predict_proba(te)
    assert p.shape == (len(te),) and np.all((p >= 0) & (p <= 1))
    assert m.predict_seeds(te).shape == (2, len(te))
    assert [h["seed"] for h in m.history] == [1, 2]


def test_two_fits_are_bit_identical():
    ds = _fold()
    tr, va, te = _split(ds)
    a = TemporalModel(dataset=ds, params=_small()).fit(tr, va, FEATS).predict_raw(te)
    b = TemporalModel(dataset=ds, params=_small()).fit(tr, va, FEATS).predict_raw(te)
    np.testing.assert_array_equal(a, b)


def test_save_load_round_trip(tmp_path):
    ds = _fold()
    tr, va, te = _split(ds)
    m = TemporalModel(dataset=ds, params=_small()).fit(tr, va, FEATS)
    m.save(tmp_path / "t.joblib")
    back = TemporalModel.load(tmp_path / "t.joblib", dataset=ds)
    np.testing.assert_array_equal(m.predict_proba(te), back.predict_proba(te))


def test_shuffle_changes_only_when_asked_and_is_seeded():
    ds = _fold()
    tr, va, te = _split(ds)
    m = TemporalModel(dataset=ds, params=_small()).fit(tr, va, FEATS)
    base = m.predict_raw(te)
    np.testing.assert_array_equal(base, m.predict_raw(te))
    for g in TemporalModel.SHUFFLE_GROUPS:
        a = m.predict_raw(te, shuffle=g, seed=7)
        np.testing.assert_array_equal(a, m.predict_raw(te, shuffle=g, seed=7))
        assert not np.array_equal(a, base)
    with pytest.raises(KeyError):
        m.predict_raw(te, shuffle="nope", seed=7)


def test_declares_dataset_and_groups():
    assert TemporalModel.needs_dataset is True
    assert set(TemporalModel.SHUFFLE_GROUPS) == set(Q.GROUP_CHANNELS)
