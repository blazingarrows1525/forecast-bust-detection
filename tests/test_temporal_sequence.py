"""S3b sequence builder: causality, alignment, masking, scaling. Pure; runs in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model import sequence as Q  # noqa: E402

SUBS = ["A", "B"]
DAYS = pd.date_range("2019-06-01", "2019-07-20", freq="D")


def _frame(drop=None) -> pd.DataFrame:
    """Synthetic fold rows. Values encode (sub, day, lead) so alignment is checkable."""
    rows = []
    for si, s in enumerate(SUBS):
        for t in DAYS:
            for L in range(1, 7):
                v = t + pd.Timedelta(days=L - 1)
                if v > DAYS[-1]:
                    continue
                if drop and drop(s, t, L, v):
                    continue
                k = (t - DAYS[0]).days
                kv = (v - DAYS[0]).days
                row = dict(subdivision_id=s, init_date=t, valid_date=v, lead_day=L,
                           obs_rain_mm=100.0 * si + kv,
                           error=1000.0 * si + 10.0 * kv + L,
                           bust=float((kv + L) % 2))
                for j, c in enumerate(Q.WEATHER):
                    row[c] = 10000.0 * si + 100.0 * j + k
                rows.append(row)
    return pd.DataFrame(rows)


def test_channel_layout():
    assert len(Q.WEATHER) == 17 and Q.LEADS == [1, 3, 5]
    assert Q.WINDOW == 14 and Q.LAG == 2
    assert len(Q.VALUES) == 24 and len(Q.MASKS) == 5 and Q.N_CHANNELS == 29
    assert sorted(Q.GROUP_CHANNELS["weather"] + Q.GROUP_CHANNELS["verification"]) == list(range(29))


def test_step_d_holds_weather_of_d_and_verification_of_d_minus_2():
    H = Q.History.from_frame(_frame())
    t = pd.Timestamp("2019-06-30")
    raw = H.raw(["B"], [t])[0]
    for k in range(Q.WINDOW):
        d = t - pd.Timedelta(days=Q.WINDOW - 1 - k)
        kd = (d - DAYS[0]).days
        kv = kd - Q.LAG
        assert raw[k, 0] == 10000.0 + kd                      # first weather channel, day d
        assert raw[k, 16] == 10000.0 + 100.0 * 16 + kd        # last weather channel, day d
        assert raw[k, 17] == 100.0 + kv                       # obs on d - 2
        for j, L in enumerate(Q.LEADS):
            assert raw[k, 18 + 2 * j] == 1000.0 + 10.0 * kv + L   # error at lead L, valid d - 2
            assert raw[k, 19 + 2 * j] == float((kv + L) % 2)       # bust at lead L


def test_nothing_after_the_cutoffs_reaches_the_sequence():
    t = pd.Timestamp("2019-06-25")
    clean = Q.History.from_frame(_frame()).raw(["A"], [t])
    poisoned = _frame()
    late_obs = poisoned.valid_date > t - pd.Timedelta(days=Q.LAG)
    poisoned.loc[late_obs, ["obs_rain_mm", "error", "bust"]] = 1000.0
    poisoned.loc[poisoned.init_date > t, Q.WEATHER] = 1000.0
    got = Q.History.from_frame(poisoned).raw(["A"], [t])
    np.testing.assert_array_equal(got, clean)


def test_steps_before_the_season_are_absent_and_masked():
    t = pd.Timestamp("2019-06-05")
    H = Q.History.from_frame(_frame())
    raw = H.raw(["A"], [t])
    m = Q.masks_of(raw)[0]
    n_before = (DAYS[0] - (t - pd.Timedelta(days=Q.WINDOW - 1))).days   # 9 weather steps
    assert np.isnan(raw[0, :n_before, :17]).all() and (m[:n_before, 0] == 0).all()
    assert (m[n_before:, 0] == 1).all()
    # verification lags by 2 more days
    assert (m[:n_before + Q.LAG, 1] == 0).all() and (m[n_before + Q.LAG:, 1] == 1).all()
    z = Q.SeqScaler.fit(H.raw(["A", "B"], [DAYS[20], DAYS[30]])).transform(raw)
    assert (z[0, :n_before, :17] == 0).all()


def test_a_missing_lead_blanks_only_that_lead():
    v_gone = pd.Timestamp("2019-06-20")
    frame = _frame(drop=lambda s, t, L, v: s == "A" and L == 3 and v == v_gone)
    t = v_gone + pd.Timedelta(days=Q.LAG)            # step d = t verifies d - 2 = v_gone
    raw = Q.History.from_frame(frame).raw(["A"], [t])
    m = Q.masks_of(raw)[0, -1]
    assert np.isnan(raw[0, -1, 20]) and np.isnan(raw[0, -1, 21])      # lead-3 error, bust
    assert not np.isnan(raw[0, -1, 18]) and not np.isnan(raw[0, -1, 22])
    assert list(m) == [1.0, 1.0, 1.0, 0.0, 1.0]


def test_scaler_uses_present_entries_and_zeroes_absent_ones():
    H = Q.History.from_frame(_frame())
    raw = H.raw(["A", "B"], [pd.Timestamp("2019-06-05"), pd.Timestamp("2019-07-10")])
    sc = Q.SeqScaler.fit(raw)
    assert np.allclose(sc.mean, np.nanmean(raw.reshape(-1, 24), axis=0))
    z = sc.transform(raw)
    assert z.dtype == np.float32 and z.shape == (2, 14, 29)
    assert np.isfinite(z).all()
    np.testing.assert_array_equal(z[..., 24:], Q.masks_of(raw).astype(np.float32))


def test_unique_keys_map_every_row():
    f = _frame()
    keys, idx = Q.unique_keys(f)
    assert len(keys) == f[["subdivision_id", "init_date"]].drop_duplicates().shape[0]
    back = keys.iloc[idx].reset_index(drop=True)
    assert (back.subdivision_id.to_numpy() == f.subdivision_id.to_numpy()).all()
    assert (back.init_date.to_numpy() == pd.to_datetime(f.init_date).to_numpy()).all()
