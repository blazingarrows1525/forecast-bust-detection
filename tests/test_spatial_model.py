"""S3c model on small synthetic grids and a synthetic fold. Skips where torch is absent (CI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xarray")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbd.model import grids as G  # noqa: E402
from fbd.model.params import SPATIAL_PARAMS  # noqa: E402
from fbd.model.spatial import SpatialModel, _Feed, _setup, pool_matrix  # noqa: E402
from test_spatial_inputs import _datasets  # noqa: E402

FEATS = ["lead_day", "fcst_rain_mm", "x1"]
BRANCHES = [("window", "map"), ("window",), ("map",)]


@pytest.fixture(scope="module")
def grids():
    return G.Grids.from_datasets(*_datasets(seed=3))


def _fold(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for split, days in (("train", pd.date_range("2016-06-01", "2016-06-30")),
                        ("val", pd.date_range("2016-07-01", "2016-07-20")),
                        ("test", pd.date_range("2017-06-01", "2017-06-15"))):
        for s in ("MID", "EDGE"):
            for t in days:
                for L in range(1, 11):
                    fc = float(rng.gamma(2.0, 10.0))
                    rows.append(dict(subdivision_id=s, init_date=t, lead_day=L,
                                     fcst_rain_mm=fc, x1=float(rng.normal()),
                                     bust=float(rng.uniform() < 0.2), split=split))
    return pd.DataFrame(rows)


def _small(**kw):
    p = dict(SPATIAL_PARAMS, seeds=[1, 2], max_epochs=3, patience=2, threads=1,
             batch_size=64, device="cpu")
    p.update(kw)
    return p


def _split(ds):
    return ds[ds.split == "train"], ds[ds.split == "val"], ds[ds.split == "test"]


@pytest.mark.parametrize("branches", BRANCHES)
def test_fit_predict_for_every_branch_set(grids, branches):
    tr, va, te = _split(_fold())
    m = SpatialModel(branches=branches, params=_small(), grids=grids).fit(tr, va, FEATS)
    p = m.predict_proba(te)
    assert p.shape == (len(te),) and np.all((p >= 0) & (p <= 1))
    assert m.predict_seeds(te).shape == (2, len(te))
    assert [h["seed"] for h in m.history] == [1, 2]
    assert (m.wscaler is not None) == ("window" in branches)
    assert (m.mscaler is not None) == ("map" in branches)


def test_two_fits_are_bit_identical(grids):
    tr, va, te = _split(_fold())
    a = SpatialModel(params=_small(), grids=grids).fit(tr, va, FEATS).predict_raw(te)
    b = SpatialModel(params=_small(), grids=grids).fit(tr, va, FEATS).predict_raw(te)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_two_gpu_fits_are_bit_identical(grids):
    tr, va, te = _split(_fold())
    p = _small(device="cuda")
    a = SpatialModel(params=p, grids=grids).fit(tr, va, FEATS).predict_raw(te)
    b = SpatialModel(params=p, grids=grids).fit(tr, va, FEATS).predict_raw(te)
    assert np.array_equal(a, b)


def test_save_load_round_trip(grids, tmp_path):
    tr, va, te = _split(_fold())
    m = SpatialModel(params=_small(), grids=grids).fit(tr, va, FEATS)
    m.save(tmp_path / "m.joblib")
    back = SpatialModel.load(tmp_path / "m.joblib", grids=grids)
    assert back.branches == m.branches
    assert np.array_equal(back.predict_proba(te), m.predict_proba(te))


def test_shuffles_are_seeded_and_named(grids):
    tr, va, te = _split(_fold())
    m = SpatialModel(params=_small(), grids=grids).fit(tr, va, FEATS)
    base = m.predict_raw(te)
    for grp in SpatialModel.SHUFFLE_GROUPS:
        a = m.predict_raw(te, shuffle=grp, seed=7)
        assert np.array_equal(a, m.predict_raw(te, shuffle=grp, seed=7))
        assert not np.array_equal(a, base)
    with pytest.raises(ValueError):
        m.predict_raw(te, shuffle="history", seed=7)
    only = SpatialModel(branches=("map",), params=_small(), grids=grids).fit(tr, va, FEATS)
    with pytest.raises(ValueError, match="no window branch"):
        only.predict_raw(te, shuffle="window", seed=7)


def test_device_windows_equal_the_reference_cut(grids):
    ds = _fold()
    g, l_, s = grids.row_index(ds)
    sc = G.WindowScaler.fit(grids, g, l_, s)
    raw, present = grids.windows_raw(g, l_, s)
    want = sc.transform(raw, present)
    feed = _Feed(grids, sc, None, _setup(_small()))
    lt = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    got = feed.windows(lt(g), lt(l_), lt(s)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_maps_are_gathered_per_issue_day(grids):
    sc = G.MapScaler.fit(grids, np.arange(30))
    feed = _Feed(grids, None, sc.all_maps(grids), _setup(_small()))
    g = torch.tensor([5, 3, 5, 9, 3])
    maps, inv = feed.maps_for(g)
    assert maps.shape[0] == 3 and inv.tolist() == [1, 0, 1, 2, 0]
    np.testing.assert_array_equal(maps[inv].numpy(), sc.all_maps(grids)[[5, 3, 5, 9, 3]])


@pytest.mark.parametrize("n_in", [9, 11, 4, 2])
def test_pool_matrix_is_adaptive_average_pooling(n_in):
    x = torch.randn(2, 3, n_in, n_in + 2, dtype=torch.float64)
    want = torch.nn.AdaptiveAvgPool2d(4)(x)
    pr = torch.tensor(pool_matrix(n_in, 4))
    pc = torch.tensor(pool_matrix(n_in + 2, 4))
    got = torch.matmul(torch.matmul(pr, x), pc.T)
    torch.testing.assert_close(got, want)


def test_bad_branches_are_refused():
    with pytest.raises(ValueError):
        SpatialModel(branches=())
    with pytest.raises(ValueError):
        SpatialModel(branches=("window", "history"))


def test_a_registered_cuda_device_is_never_silently_replaced(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        _setup(dict(SPATIAL_PARAMS))


def test_occlusion_base_is_predict_raw_and_each_column_moves_one_input(grids):
    tr, va, te = _split(_fold())
    m = SpatialModel(params=_small(), grids=grids).fit(tr, va, FEATS)
    base, occ, keys = m.occlusion(te, chunk=37)
    np.testing.assert_allclose(base, m.predict_raw(te), rtol=1e-5, atol=1e-6)
    assert keys == FEATS + [SpatialModel.WINDOW_KEY, SpatialModel.MAP_KEY]
    assert occ.shape == (len(te), len(keys)) and np.all((occ > 0) & (occ < 1))
    # setting an input that is already at its training mean changes nothing
    te0 = te.copy()
    te0["x1"] = float(m.prep.mean[FEATS.index("x1")])
    b0, o0, _ = m.occlusion(te0)
    np.testing.assert_allclose(o0[:, FEATS.index("x1")], b0, rtol=1e-5, atol=1e-6)


def test_occlusion_drops_a_missing_branch(grids):
    tr, va, te = _split(_fold())
    m = SpatialModel(branches=("window",), params=_small(), grids=grids).fit(tr, va, FEATS)
    _b, occ, keys = m.occlusion(te)
    assert keys == FEATS + [SpatialModel.WINDOW_KEY] and occ.shape[1] == len(FEATS) + 1
