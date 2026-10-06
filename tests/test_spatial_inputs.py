"""S3c inputs on synthetic grids: causality, geometry, scaling, provenance. numpy/xarray only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model import grids as G  # noqa: E402

YEARS = (2016, 2017)
H_LAT = np.round(np.arange(10.0, 10.0 + 0.703125 * 10, 0.703125), 6)   # 10 rows
H_LON = np.round(np.arange(70.0, 70.0 + 0.703125 * 12, 0.703125), 6)   # 12 cols
E_LAT = np.arange(6.0, 6.0 + 1.5 * 18, 1.5)                            # 18 rows
E_LON = np.arange(66.0, 66.0 + 1.5 * 20, 1.5)                          # 20 cols


def _hres(y, rng):
    t = pd.date_range(f"{y}-06-01", periods=G.DAYS, freq="D")
    tp = rng.gamma(1.5, 8.0, (G.DAYS, G.N_LEADS, len(H_LON), len(H_LAT))).astype("float32")
    return xr.Dataset({"tp24": (("time", "prediction_timedelta", "longitude", "latitude"), tp)},
                      coords={"time": t, "latitude": H_LAT, "longitude": H_LON,
                              "prediction_timedelta": pd.to_timedelta(np.arange(1, 11), "D")})


def _era(y, rng):
    t = pd.date_range(f"{y}-06-01", periods=G.DAYS * 4, freq="6h")
    shp2 = (len(t), len(E_LON), len(E_LAT))
    shp3 = (len(t), 3, len(E_LON), len(E_LAT))
    d2 = xr.Dataset({"total_column_water_vapour": (("time", "longitude", "latitude"),
                                                   rng.normal(50, 5, shp2).astype("float32")),
                     "mean_sea_level_pressure": (("time", "longitude", "latitude"),
                                                 rng.normal(1e5, 300, shp2).astype("float32"))},
                    coords={"time": t, "latitude": E_LAT, "longitude": E_LON})
    dims = ("time", "level", "longitude", "latitude")
    d3 = xr.Dataset({k: (dims, rng.normal(0, 1, shp3).astype("float32"))
                     for k in ("geopotential", "specific_humidity", "u_component_of_wind",
                               "v_component_of_wind")},
                    coords={"time": t, "level": [200, 500, 850], "latitude": E_LAT,
                            "longitude": E_LON})
    return d2, d3


def _static(rng):
    shp = (len(E_LON), len(E_LAT))
    return xr.Dataset({"geopotential_at_surface": (("longitude", "latitude"),
                                                   rng.normal(3e3, 1e3, shp).astype("float32")),
                       "land_sea_mask": (("longitude", "latitude"),
                                         rng.uniform(0, 1, shp).astype("float32"))},
                      coords={"latitude": E_LAT, "longitude": E_LON})


def _weights():
    rows = []
    # "MID": a 3 x 3 block in the middle; "EDGE": a 2 x 2 block in the corner
    for r in range(4, 7):
        for c in range(5, 8):
            rows.append(("MID", r, c, 300.0))
    for r in range(0, 2):
        for c in range(0, 2):
            rows.append(("EDGE", r, c, 500.0))
    w = pd.DataFrame(rows, columns=["subdivision_id", "lat_idx", "lon_idx", "weight_km2"])
    w["lat"] = H_LAT[w.lat_idx]
    w["lon"] = H_LON[w.lon_idx]
    return w


def _datasets(seed=0):
    rng = np.random.default_rng(seed)
    hres = {y: _hres(y, rng) for y in YEARS}
    era = {y: _era(y, rng) for y in YEARS}
    return (hres, {y: e[0] for y, e in era.items()}, {y: e[1] for y, e in era.items()},
            _static(rng), _weights())


@pytest.fixture(scope="module")
def built():
    hres, e2, e3, st, w = _datasets()
    return G.Grids.from_datasets(hres, e2, e3, st, w), (hres, e2, e3, st, w)


def _rows(grids, items):
    df = pd.DataFrame(items, columns=["subdivision_id", "init_date", "lead_day"])
    return grids.row_index(df)


def test_shapes_and_day_index(built):
    g, _ = built
    assert g.fc.shape == (2 * G.DAYS, G.N_LEADS, 10 + 2 * G.PAD, 12 + 2 * G.PAD)
    assert g.maps.shape == (2 * G.DAYS, G.N_MAP_DYN, len(E_LAT), len(E_LON))
    gi, li, si = _rows(g, [("MID", "2017-06-03", 4)])
    assert gi[0] == G.DAYS + 2 and li[0] == 3 and si[0] == g.sub_index["MID"]
    with pytest.raises(ValueError):
        _rows(g, [("MID", "2018-06-01", 1)])


def test_window_is_centred_and_contains_the_subdivision(built):
    g, (_h, _a, _b, _s, w) = built
    i = g.sub_index["MID"]
    assert tuple(g.centre[i]) == (5, 6)
    outline = g.static_h[i, 0]
    # padded window rows c..c+12 hold unpadded rows c-6..c+6: MID's cells sit at the centre
    assert outline[G.PAD, G.PAD] > 0 and outline[G.PAD - 1:G.PAD + 2, G.PAD - 1:G.PAD + 2].min() > 0
    assert (outline > 0).sum() == 9


def test_outline_is_area_fraction():
    w = _weights()
    out = G.outlines(w, ["MID"], H_LAT, (10, 12))
    r, c = 5, 6
    assert out[0, r, c] == pytest.approx(300.0 / G.cell_area_km2(H_LAT)[r])
    big = w.assign(weight_km2=1e9)
    assert G.outlines(big, ["MID"], H_LAT, (10, 12)).max() == 1.0


def test_forecast_channels_are_the_two_runs(built):
    g, (hres, *_rest) = built
    gi, li, si = _rows(g, [("MID", "2016-06-10", 3)])
    raw, present = g.windows_raw(gi, li, si)
    tp = hres[2016].tp24.transpose("time", "prediction_timedelta", "latitude", "longitude").values
    c = g.centre[si[0]]
    # window cell (a, b) is unpadded cell (c0 - 6 + a, c1 - 6 + b); MID's centre (5, 6)
    a, b = G.PAD, G.PAD
    assert raw[0, G.FC, a, b] == pytest.approx(np.log1p(tp[9, 2, c[0], c[1]]), rel=1e-6)
    assert raw[0, G.FC_PREV, a, b] == pytest.approx(np.log1p(tp[8, 3, c[0], c[1]]), rel=1e-6)
    assert raw[0, G.PREV_OK, a, b] == 1.0 and present[0, G.FC_PREV, a, b]


@pytest.mark.parametrize("init, lead", [("2016-06-01", 3), ("2017-06-01", 1), ("2016-07-01", 10)])
def test_previous_run_absent_on_1_june_and_at_day_10(built, init, lead):
    g, _ = built
    raw, present = g.windows_raw(*_rows(g, [("MID", init, lead)]))
    assert not raw[0, G.FC_PREV].any() and not raw[0, G.PREV_OK].any()
    assert not present[0, G.FC_PREV].any()


def test_padding_and_the_in_domain_plane_at_the_edge(built):
    g, _ = built
    raw, present = g.windows_raw(*_rows(g, [("EDGE", "2016-07-01", 2)]))
    dom = raw[0, G.IN_DOMAIN]
    c = g.centre[g.sub_index["EDGE"]]
    # rows and cols left of the domain are padding
    assert dom[: G.PAD - c[0]].sum() == 0 and dom[:, : G.PAD - c[1]].sum() == 0
    assert dom.sum() == present[0, G.FC].sum() > 0
    outside = dom == 0
    assert np.all(raw[0][:, outside] == 0)


def _poison(hres, e2, e3, t):
    """Every run issued after t and every analysis after t 00Z set to 1000."""
    h = {y: d.copy(deep=True) for y, d in hres.items()}
    a = {y: d.copy(deep=True) for y, d in e2.items()}
    b = {y: d.copy(deep=True) for y, d in e3.items()}
    for y in h:
        h[y]["tp24"] = h[y].tp24.where(h[y].time <= t, 1000.0)
        for ds in (a[y], b[y]):
            for v in ds.data_vars:
                ds[v] = ds[v].where(ds.time <= t, 1000.0)
    return h, a, b


@pytest.mark.parametrize("init", ["2016-06-01", "2016-08-15", "2017-09-30"])
def test_poisoning_the_future_changes_nothing(built, init):
    g, (hres, e2, e3, st, w) = built
    t = pd.Timestamp(init)
    p = G.Grids.from_datasets(*_poison(hres, e2, e3, t), st, w)
    items = [(s, init, L) for s in ("MID", "EDGE") for L in (1, 4, 9, 10)]
    for grids in (g, p):
        assert grids.row_index(pd.DataFrame(items, columns=["subdivision_id", "init_date",
                                                             "lead_day"]))[0].size == 8
    rows = _rows(g, items)
    a, pa = g.windows_raw(*rows)
    b, pb = p.windows_raw(*rows)
    assert np.array_equal(a, b) and np.array_equal(pa, pb)
    assert np.array_equal(g.maps_raw(rows[0]), p.maps_raw(rows[0]))
    # and the poison did land: the next day's map differs (no next day after 30 September)
    if not init.endswith("09-30"):
        assert not np.array_equal(g.maps_raw(rows[0] + 1), p.maps_raw(rows[0] + 1))


def test_window_scaler_uses_present_entries_only(built):
    g, _ = built
    rows = _rows(g, [(s, d, L) for s in ("MID", "EDGE")
                     for d in ("2016-06-01", "2016-06-02", "2016-07-15") for L in (1, 5, 10)])
    sc = G.WindowScaler.fit(g, *rows, chunk=4)
    raw, present = g.windows_raw(*rows)
    for c in G.STANDARDISED:
        vals = raw[:, c][present[:, c]].astype(float)
        assert sc.mean[c] == pytest.approx(vals.mean(), rel=1e-9)
        assert sc.std[c] == pytest.approx(vals.std(), rel=1e-6)
    for c in set(range(G.N_WIN)) - set(G.STANDARDISED):
        assert sc.mean[c] == 0 and sc.std[c] == 1
    z = sc.transform(raw, present)
    assert np.all(z[:, G.FC][~present[:, G.FC]] == 0)
    assert np.all(z[:, G.FC_PREV][~present[:, G.FC_PREV]] == 0)
    assert np.array_equal(z[:, G.OUTLINE], raw[:, G.OUTLINE])


def test_map_anomalies_use_training_days_only(built):
    g, _ = built
    train = np.arange(0, 50)
    sc = G.MapScaler.fit(g, np.repeat(train, 3))       # duplicates count once
    np.testing.assert_allclose(sc.cell_mean, g.maps[train].astype(float).mean(axis=0))
    assert g.maps.flags["C_CONTIGUOUS"] and g.fc.flags["C_CONTIGUOUS"]
    other = G.Grids(**{**g.__dict__, "maps": g.maps.copy()})
    other.maps[100:] = 1e6
    sc2 = G.MapScaler.fit(other, train)
    assert np.array_equal(sc.cell_mean, sc2.cell_mean) and np.array_equal(sc.sd, sc2.sd)
    z = sc.transform(g.maps_raw(train))
    assert abs(z[:, : G.N_MAP_DYN].mean()) < 1e-6
    assert np.array_equal(z[:, -1], g.maps_raw(train)[:, -1])   # land-sea unscaled


def test_bilinear_matrix_matches_linear_interpolation():
    src = np.array([0.0, 1.5, 3.0, 4.5])
    dst = np.array([0.0, 0.75, 2.0, 4.5])
    A = G.bilinear_matrix(src, dst)
    f = np.array([1.0, 4.0, -2.0, 7.0])
    np.testing.assert_allclose(A @ f, np.interp(dst, src, f))
    with pytest.raises(ValueError):
        G.bilinear_matrix(src, np.array([5.0]))


def test_manifest_pins_every_file(tmp_path):
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    man = tmp_path / "inputs.json"
    body = G.write_manifest(man, files=[a, b], root=tmp_path)
    assert set(body) == {"a.nc", "b.nc"}
    assert G.verify_manifest(man, root=tmp_path) == []
    b.write_bytes(b"changed")
    assert any("b.nc" in x for x in G.verify_manifest(man, root=tmp_path))
    a.unlink()
    assert any("missing" in x for x in G.verify_manifest(man, root=tmp_path))


def test_the_real_input_list_is_23_files():
    files = G.input_files()
    assert len(files) == 23 and len(set(files)) == 23
