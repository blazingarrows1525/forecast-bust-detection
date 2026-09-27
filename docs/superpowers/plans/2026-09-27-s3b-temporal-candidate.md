# S3b Temporal Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, audit, register and score S3b: a GRU over 14 days of verification and weather history, in front of S3a's MLP head, under the registered S3 rule.

**Architecture:** A pure sequence builder (`fbd.model.sequence`) turns a fold dataset into dense per-subdivision day arrays and cuts 14-step windows from them, with causality enforced by index arithmetic. `fbd.model.temporal.TemporalModel` wraps S3a's preprocessor and head around a GRU encoder. `scripts/promote.py` gains four additive hooks: dataset, output directory, addenda and the MLP secondary. An audit proves the hooks are neutral before the addendum is registered.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn (isotonic), PyTorch CPU (lazy import; absent in CI), pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-s3b-temporal-candidate-design.md`

## Global Constraints

- Windows: any entry point that trains a torch model imports torch **before** scikit-learn (`try: import torch` at the top).
- CI has no torch: torch tests use `pytest.importorskip("torch")`; `fbd.model.sequence` and `fbd.evaluate.promotion` import no torch.
- Commit messages: plain. No `Co-Authored-By` trailer and no "Generated with" line.
- No SIH or competition wording anywhere.
- The primary is S3's, unchanged: mean 2019–2022 within-year AUROC margin over the XGBoost incumbent, 98.33% interval, 10,000 resamples, seed 20260919.
- No temporal model produces a prediction for 2019–2022 before `PREREGISTRATION_S3B.md` is committed, pushed, and CI is green.
- The audit never overwrites `data/processed/backtest/candidate_mlp_<year>.joblib`.

---

## File Structure

| file | responsibility |
|---|---|
| `src/fbd/model/params.py` (modify) | add `TEMPORAL_PARAMS` |
| `src/fbd/model/sequence.py` (create) | channel lists; `History` (dense day arrays, windows); `masks_of`; `SeqScaler`; `unique_keys`; `GROUP_CHANNELS` |
| `src/fbd/model/temporal.py` (create) | `TemporalModel`: GRU + S3a head, train, predict (with group shuffle), save/load |
| `src/fbd/evaluate/promotion.py` (modify) | `build_candidate` |
| `src/fbd/evaluate/registration.py` (modify) | `merge_addendum` |
| `scripts/promote.py` (modify) | dataset hook, `out_dir`, addenda, MLP secondary, group shuffle |
| `scripts/audit_s3b.py` (create) | neutrality, causality on real data, determinism and smoke |
| `docs/PREREGISTRATION_S3B.md` (create) | the addendum |
| `tests/test_temporal_sequence.py` (create) | causality, alignment, masking, scaling (CI) |
| `tests/test_temporal_model.py` (create) | fit, determinism, save/load, shuffle (skips without torch) |
| `tests/test_promotion.py` (modify) | params hash, addendum merge, `build_candidate` |

---

### Task 0: Branch and PR

- [ ] **Step 1:** `git switch -c s3b-temporal` (from the commit holding the spec and this plan), then `git push -u origin s3b-temporal`.
- [ ] **Step 2:** `gh pr create --base master --head s3b-temporal --title "S3b: temporal candidate (GRU over 14 days of history)" --body-file <scratch>/pr_s3b.md`. The body states the spec path, the plan path, and that no 2019–2022 prediction is made before the addendum is registered. No attribution lines.
- [ ] **Step 3:** Bind it with `mcp__ccd_pr__bind_pr` if the app does not bind it itself.

---

### Task 1: Sequence builder and frozen parameters

**Files:**
- Modify: `src/fbd/model/params.py` (append after `MLP_PARAMS`)
- Create: `src/fbd/model/sequence.py`
- Test: `tests/test_temporal_sequence.py`, `tests/test_promotion.py`

**Interfaces:**
- Produces: `PR.TEMPORAL_PARAMS: dict`; `sequence.WEATHER: list[str]` (17), `LEADS: list[int]` ([1, 3, 5]), `WINDOW = 14`, `LAG = 2`, `VALUES: list[str]` (24), `MASKS: list[str]` (5), `N_CHANNELS = 29`, `GROUP_CHANNELS: dict[str, list[int]]`; `History.from_frame(ds) -> History`; `History.raw(sub_ids, init_dates) -> np.ndarray[n, 14, 24]` (float64, NaN = absent); `masks_of(raw) -> np.ndarray[n, 14, 5]`; `SeqScaler.fit(raw) -> SeqScaler`; `SeqScaler.transform(raw) -> np.ndarray[n, 14, 29]` float32; `unique_keys(df) -> (keys: pd.DataFrame[subdivision_id, init_date], idx: np.ndarray[int64])`.

- [ ] **Step 1: Write the failing tests** — `tests/test_temporal_sequence.py`:

```python
"""S3b sequence builder: causality, alignment, masking, scaling. Pure; runs in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    raw = Q.History.from_frame(_frame()).raw(["A"], [t])
    m = Q.masks_of(raw)[0]
    n_before = (DAYS[0] - (t - pd.Timedelta(days=Q.WINDOW - 1))).days   # 9 weather steps
    assert np.isnan(raw[0, :n_before, :17]).all() and (m[:n_before, 0] == 0).all()
    assert (m[n_before:, 0] == 1).all()
    # verification lags by 2 more days
    assert (m[:n_before + Q.LAG, 1] == 0).all() and (m[n_before + Q.LAG:, 1] == 1).all()
    z = Q.SeqScaler.fit(Q.History.from_frame(_frame()).raw(["A", "B"], [DAYS[20], DAYS[30]])).transform(raw)
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
```

Append to `tests/test_promotion.py`:

```python
def test_temporal_params_are_frozen_and_hashable():
    p = PR.TEMPORAL_PARAMS
    assert p["window"] == 14 and p["verify_lag"] == 2 and p["verify_leads"] == [1, 3, 5]
    assert len(p["weather"]) == 17 and p["gru_hidden"] == 32
    assert p["hidden"] == PR.MLP_PARAMS["hidden"] and p["seeds"] == PR.MLP_PARAMS["seeds"]
    json.dumps(p)
    assert PR.params_sha256(p) == PR.params_sha256(dict(p))
    assert PR.params_sha256(dict(p, window=7)) != PR.params_sha256(p)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_temporal_sequence.py tests/test_promotion.py -q`
Expected: ImportError / AttributeError (`fbd.model.sequence`, `TEMPORAL_PARAMS` missing).

- [ ] **Step 3: Implement.** Append to `src/fbd/model/params.py`:

```python
#: S3b candidate, frozen before any run (spec 2026-09-27, docs/PREREGISTRATION_S3B.md).
#: The channel lists, window, lag and leads are here so that changing any of them
#: changes temporal_params_sha256.
TEMPORAL_PARAMS = dict(
    window=14,
    verify_lag=2,
    verify_leads=[1, 3, 5],
    weather=["moisture_flux_850", "wind_shear", "u850", "v850", "z500", "tcwv", "mslp",
             "onshore_wind", "somali_jet_z", "monsoon_trough_mslp_z", "nw_z500_z",
             "india_shear_z", "india_tcwv_z", "india_q850_z", "mcz_q850_z",
             "bob_vorticity_max_z", "bob_mslp_min_z"],
    verification=["obs_rain_mm on d - 2", "error and bust of leads 1, 3, 5 verifying on d - 2"],
    sequence_scaling="training-rows mean and sd of present entries per value channel; "
                     "absent -> 0 after scaling; 5 presence masks, unscaled",
    gru_hidden=32,
    gru_layers=1,
    static="S3a Preprocessor on the incumbent's 52 features",
    hidden=[128, 64],
    activation="relu",
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=1024,
    max_epochs=60,
    patience=5,
    seeds=[20260920, 20260921, 20260922, 20260923, 20260924],
    threads=8,
    dtype="float32",
    device="cpu",
    loss="bce with pos_weight = negatives / positives on training rows",
    stopping="validation-year weighted bce, restore best epoch",
    calibration="isotonic on the validation year, on the seed-averaged probability",
)
```

Create `src/fbd/model/sequence.py`:

```python
"""S3b: the 14-day history a temporal candidate reads (spec §3). Pure numpy/pandas.

Step d of the window for issue day t holds what had arrived by the morning of
day d: the weather analysed at d, and the verification of forecasts that
verified on d - 2 (IMD's day runs 03Z-03Z, D-005, so d - 2 is the last day
whose observation is complete at a 00Z issue on d). Causality is enforced by
index arithmetic on dense (subdivision, day) arrays, and tested by poisoning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fbd.model.params import TEMPORAL_PARAMS

WEATHER = list(TEMPORAL_PARAMS["weather"])
LEADS = list(TEMPORAL_PARAMS["verify_leads"])
WINDOW = int(TEMPORAL_PARAMS["window"])
LAG = int(TEMPORAL_PARAMS["verify_lag"])
VALUES = WEATHER + ["obs_rain_mm"] + [f"{k}_L{L}" for L in LEADS for k in ("error", "bust")]
MASKS = ["m_weather", "m_obs"] + [f"m_L{L}" for L in LEADS]
N_VALUES = len(VALUES)
N_CHANNELS = N_VALUES + len(MASKS)
_NW = len(WEATHER)
#: channel indices in the scaled tensor that each exploratory shuffle permutes together
GROUP_CHANNELS = {
    "weather": list(range(_NW)) + [N_VALUES],
    "verification": list(range(_NW, N_VALUES)) + list(range(N_VALUES + 1, N_CHANNELS)),
}


def _day(dates, origin) -> np.ndarray:
    return ((pd.to_datetime(pd.Series(dates)).dt.normalize() - origin).dt.days).to_numpy()


@dataclass
class History:
    """Dense per-subdivision day arrays built from one fold dataset (all splits)."""
    index: dict
    origin: pd.Timestamp
    W: np.ndarray    # [S, D, 17] weather analysed on day d (rows with init_date = d)
    O: np.ndarray    # [S, D] observed rain on day d
    E: np.ndarray    # [S, D, len(LEADS)] error of the lead-L forecast verifying on d
    B: np.ndarray    # [S, D, len(LEADS)] bust flag of the same

    @classmethod
    def from_frame(cls, ds: pd.DataFrame) -> "History":
        df = ds[["subdivision_id", "init_date", "valid_date", "lead_day",
                 "obs_rain_mm", "error", "bust", *WEATHER]].copy()
        df["init_date"] = pd.to_datetime(df.init_date).dt.normalize()
        df["valid_date"] = pd.to_datetime(df.valid_date).dt.normalize()
        subs = sorted(df.subdivision_id.unique())
        index = {s: i for i, s in enumerate(subs)}
        origin = min(df.init_date.min(), df.valid_date.min())
        n_days = int((max(df.init_date.max(), df.valid_date.max()) - origin).days) + 1
        S = len(subs)
        W = np.full((S, n_days, _NW), np.nan)
        O = np.full((S, n_days), np.nan)
        E = np.full((S, n_days, len(LEADS)), np.nan)
        B = np.full((S, n_days, len(LEADS)), np.nan)

        w = df.drop_duplicates(["subdivision_id", "init_date"])
        W[w.subdivision_id.map(index).to_numpy(), _day(w.init_date, origin)] = \
            w[WEATHER].to_numpy(float)
        o = df.dropna(subset=["obs_rain_mm"]).drop_duplicates(["subdivision_id", "valid_date"])
        O[o.subdivision_id.map(index).to_numpy(), _day(o.valid_date, origin)] = \
            o.obs_rain_mm.to_numpy(float)
        for j, L in enumerate(LEADS):
            r = df[df.lead_day == L]
            si, di = r.subdivision_id.map(index).to_numpy(), _day(r.valid_date, origin)
            E[si, di, j] = r.error.to_numpy(float)
            B[si, di, j] = r.bust.to_numpy(float)
        return cls(index, origin, W, O, E, B)

    def _take(self, A: np.ndarray, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        ok = (d >= 0) & (d < A.shape[1])
        out = np.full(d.shape + A.shape[2:], np.nan)
        ss = np.broadcast_to(s[:, None], d.shape)
        out[ok] = A[ss[ok], d[ok]]
        return out

    def raw(self, sub_ids, init_dates) -> np.ndarray:
        """[n, WINDOW, N_VALUES] float64; NaN where nothing had arrived."""
        s = np.array([self.index[x] for x in sub_ids], dtype=np.int64)
        t = _day(init_dates, self.origin)
        d = t[:, None] + np.arange(-(WINDOW - 1), 1)[None, :]
        weather = self._take(self.W, s, d)
        v = d - LAG
        obs = self._take(self.O, s, v)[..., None]
        err, bust = self._take(self.E, s, v), self._take(self.B, s, v)
        verif = np.stack([err, bust], axis=-1).reshape(len(s), WINDOW, 2 * len(LEADS))
        return np.concatenate([weather, obs, verif], axis=-1)


def masks_of(raw: np.ndarray) -> np.ndarray:
    """[n, WINDOW, 5]: weather present; observation present; each lead present."""
    fin = np.isfinite(raw)
    m = [fin[..., :_NW].all(-1), fin[..., _NW]]
    for j in range(len(LEADS)):
        m.append(fin[..., _NW + 1 + 2 * j] & fin[..., _NW + 2 + 2 * j])
    return np.stack(m, axis=-1).astype(np.float64)


@dataclass
class SeqScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, raw: np.ndarray) -> "SeqScaler":
        flat = raw.reshape(-1, N_VALUES)
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        return cls(mean, std)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        z = (raw - self.mean) / self.std
        z = np.where(np.isfinite(z), z, 0.0)
        return np.concatenate([z, masks_of(raw)], axis=-1).astype(np.float32)


def unique_keys(df: pd.DataFrame):
    """The (subdivision, issue day) pairs in ``df`` and, per row, which pair it is."""
    k = pd.DataFrame({"subdivision_id": df.subdivision_id.to_numpy(),
                      "init_date": pd.to_datetime(df.init_date).dt.normalize().to_numpy()})
    idx = k.groupby(["subdivision_id", "init_date"], sort=True).ngroup().to_numpy()
    keys = k.drop_duplicates().sort_values(["subdivision_id", "init_date"]).reset_index(drop=True)
    return keys, idx.astype(np.int64)
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src python -m pytest tests/test_temporal_sequence.py tests/test_promotion.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/fbd/model/params.py src/fbd/model/sequence.py tests/test_temporal_sequence.py tests/test_promotion.py
git commit -m "S3b: freeze the temporal parameters; build 14-day history windows causally"
```

---

### Task 2: The temporal model

**Files:**
- Create: `src/fbd/model/temporal.py`
- Test: `tests/test_temporal_model.py`

**Interfaces:**
- Consumes: `sequence.History`, `SeqScaler`, `unique_keys`, `GROUP_CHANNELS`, `N_CHANNELS`; `fbd.model.mlp.Preprocessor`, `fbd.model.mlp._torch`.
- Produces: `TemporalModel(dataset: pd.DataFrame | None = None)`; class attrs `needs_dataset = True`, `SHUFFLE_GROUPS = ("weather", "verification")`; `.fit(train, val, features) -> self`; `.predict_seeds(df, shuffle=None, seed=None) -> np.ndarray[n_seeds, n]`; `.predict_raw(df, shuffle=None, seed=None)`; `.predict_proba(df, shuffle=None, seed=None)`; `.history: list[dict]` (per-seed `seed`, `best_epoch`, `val_loss`); `.save(path)`; `TemporalModel.load(path, dataset) -> TemporalModel`.

- [ ] **Step 1: Write the failing tests** — `tests/test_temporal_model.py`:

```python
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
                    row = dict(subdivision_id=s, init_date=t, valid_date=t + pd.Timedelta(days=L - 1),
                               lead_day=L, fcst_rain_mm=fc, obs_rain_mm=ob, error=fc - ob,
                               bust=float(abs(fc - ob) > 25), x1=float(rng.normal()), split=split,
                               year=year)
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
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_temporal_model.py -q`
Expected: ImportError (`fbd.model.temporal`).

- [ ] **Step 3: Implement** `src/fbd/model/temporal.py`:

```python
"""S3b candidate: a GRU over 14 days of history in front of S3a's MLP head.

Everything except the history encoder is S3a's: the same 52 static inputs and
preprocessing, the same head, loss, optimiser, stopping, seeds and isotonic
calibration. So S3b - MLP measures what the history adds. Frozen in
``fbd.model.params.TEMPORAL_PARAMS``; registered in docs/PREREGISTRATION_S3B.md.
Torch is imported lazily, as in ``fbd.model.mlp``; see its note on Windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fbd.model import sequence as Q
from fbd.model.mlp import Preprocessor, _torch
from fbd.model.params import TEMPORAL_PARAMS


def _net(n_static: int, p: dict):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(Q.N_CHANNELS, p["gru_hidden"], num_layers=p["gru_layers"],
                              batch_first=True)
            layers, width = [], n_static + p["gru_hidden"]
            for h in p["hidden"]:
                layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p["dropout"])]
                width = h
            layers.append(nn.Linear(width, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, x, s):
            _out, h = self.gru(s)
            return self.head(torch.cat([x, h[-1]], dim=1)).squeeze(1)

    return Net()


def _logits(net, x, u, idx, chunk: int = 8192):
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            j = idx[i:i + chunk]
            out.append(net(x[i:i + chunk], u[j]))
    return torch.cat(out)


def _train_one(tr, va, seed: int, p: dict, pos_weight: float):
    torch = _torch()
    xt, ut, it, yt = tr
    xv, uv, iv, yv = va
    torch.manual_seed(seed)
    net = _net(xt.shape[1], p)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, p["max_epochs"] + 1):
        net.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(order), p["batch_size"]):
            b = order[i:i + p["batch_size"]]
            opt.zero_grad()
            loss_fn(net(xt[b], ut[it[b]]), yt[b]).backward()
            opt.step()
        net.eval()
        v = float(loss_fn(_logits(net, xv, uv, iv), yv))
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= p["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"seed": seed, "best_epoch": best_epoch, "val_loss": best}


@dataclass
class TemporalModel:
    dataset: pd.DataFrame | None = None
    params: dict = field(default_factory=lambda: dict(TEMPORAL_PARAMS))
    features: list = field(default_factory=list)
    prep: Preprocessor | None = None
    scaler: Q.SeqScaler | None = None
    nets: list = field(default_factory=list)
    calibrator: IsotonicRegression | None = None
    history: list = field(default_factory=list)

    needs_dataset = True
    SHUFFLE_GROUPS = ("weather", "verification")

    def __post_init__(self):
        self._hist = Q.History.from_frame(self.dataset) if self.dataset is not None else None

    def _tensors(self, df: pd.DataFrame, shuffle=None, seed=None):
        torch = _torch()
        keys, idx = Q.unique_keys(df)
        u = self.scaler.transform(self._hist.raw(keys.subdivision_id, keys.init_date))
        if shuffle is not None:
            cols = Q.GROUP_CHANNELS[shuffle]
            perm = np.random.default_rng(seed).permutation(len(u))
            u[:, :, cols] = u[perm][:, :, cols]
        return (torch.from_numpy(self.prep.transform(df)), torch.from_numpy(u),
                torch.from_numpy(idx))

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features) -> "TemporalModel":
        torch = _torch()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.params["threads"])
        self.features = list(features)
        tr = train.dropna(subset=["bust"])
        va = val.dropna(subset=["bust"])
        self.prep = Preprocessor.fit(tr, self.features)
        keys, _ = Q.unique_keys(tr)
        self.scaler = Q.SeqScaler.fit(self._hist.raw(keys.subdivision_id, keys.init_date))
        xt, ut, it = self._tensors(tr)
        xv, uv, iv = self._tensors(va)
        ytr = torch.from_numpy(tr.bust.to_numpy(np.float32))
        yva = torch.from_numpy(va.bust.to_numpy(np.float32))
        pos_weight = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)

        self.nets, self.history = [], []
        for seed in self.params["seeds"]:
            net, info = _train_one((xt, ut, it, ytr), (xv, uv, iv, yva), seed, self.params,
                                   pos_weight)
            self.nets.append(net)
            self.history.append(info)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(self.predict_raw(va), yva.numpy().astype(float))
        return self

    def predict_seeds(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        torch = _torch()
        x, u, idx = self._tensors(df, shuffle, seed)
        return np.vstack([torch.sigmoid(_logits(net, x, u, idx)).numpy().astype("float64")
                          for net in self.nets])

    def predict_raw(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        return self.predict_seeds(df, shuffle, seed).mean(axis=0)

    def predict_proba(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df, shuffle, seed))

    def save(self, path) -> None:
        import joblib

        joblib.dump({"params": self.params, "features": self.features, "prep": self.prep,
                     "scaler": self.scaler, "states": [n.state_dict() for n in self.nets],
                     "calibrator": self.calibrator, "history": self.history}, path)

    @classmethod
    def load(cls, path, dataset: pd.DataFrame) -> "TemporalModel":
        import joblib

        d = joblib.load(path)
        m = cls(dataset=dataset, params=d["params"], features=d["features"], prep=d["prep"],
                scaler=d["scaler"], calibrator=d["calibrator"], history=d["history"])
        n_static = len(m.features) + len(m.prep.flagged)
        for state in d["states"]:
            net = _net(n_static, m.params)
            net.load_state_dict(state)
            net.eval()
            m.nets.append(net)
        return m
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src python -m pytest tests/test_temporal_model.py -q`
Expected: 5 passed. Then `PYTHONPATH=src python -m pytest tests/ -q`: everything passes.

- [ ] **Step 5: Commit**

```bash
git add src/fbd/model/temporal.py tests/test_temporal_model.py
git commit -m "S3b: the temporal model, a GRU in front of S3a's head"
```

---

### Task 3: Harness hooks

**Files:**
- Modify: `src/fbd/evaluate/promotion.py` (append `build_candidate`)
- Modify: `src/fbd/evaluate/registration.py` (append `merge_addendum`)
- Modify: `scripts/promote.py`
- Test: `tests/test_promotion.py`

**Interfaces:**
- Produces: `PM.build_candidate(cls_, dataset)`; `R.merge_addendum(base: dict, addendum: dict, name: str) -> dict`; `promote.score_fold(name, year, ens, lead_days, bt, seed, out_dir=F.FOLD_DIR, mlp=None) -> (rows, info)`, where `mlp` is the parsed `mlp.json` or None; `rows` gains `p_mlp` when `mlp` is given; `info` gains `shuffle_auroc` for candidates with `SHUFFLE_GROUPS`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_promotion.py`:

```python
def test_addendum_adds_and_never_overrides():
    base = {"slate": "mlp,temporal,spatial", "mlp_params_sha256": "a"}
    got = R.merge_addendum(base, {"temporal_params_sha256": "b"}, "temporal")
    assert got == {**base, "temporal_params_sha256": "b"}
    with pytest.raises(R.RegistrationError, match="never overrides"):
        R.merge_addendum(base, {"mlp_params_sha256": "z"}, "temporal")


def test_build_candidate_gives_the_dataset_only_to_those_that_ask():
    class Plain:
        def __init__(self):
            self.got = None

    class Reader:
        needs_dataset = True

        def __init__(self, dataset=None):
            self.got = dataset

    ds = object()
    assert PM.build_candidate(Plain, ds).got is None
    assert PM.build_candidate(Reader, ds).got is ds
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_promotion.py -q`
Expected: AttributeError (`merge_addendum`, `build_candidate`).

- [ ] **Step 3: Implement.** Append to `src/fbd/evaluate/promotion.py`:

```python
def build_candidate(cls_, dataset):
    """S3b spec §5: a candidate that reads history is given the fold dataset;
    every other candidate is built exactly as before, with no arguments."""
    return cls_(dataset=dataset) if getattr(cls_, "needs_dataset", False) else cls_()
```

Append to `src/fbd/evaluate/registration.py`:

```python
def merge_addendum(base: dict, addendum: dict, name: str) -> dict:
    """An S3 addendum registers one candidate's keys on top of S3's block."""
    clash = sorted(set(base) & set(addendum))
    if clash:
        raise RegistrationError(f"the {name} addendum redefines registered keys {clash}; "
                                "an addendum adds, it never overrides")
    return {**base, **addendum}
```

In `scripts/promote.py`:

1. After `OUT_DIR = ...`, add:

```python
MLP_JSON = OUT_DIR / "mlp.json"
#: addenda that register a candidate's own keys (docs/PREREGISTRATION_S3.md §3)
ADDENDA = {"temporal": ROOT / "docs" / "PREREGISTRATION_S3B.md"}
```

and extend `CANDIDATES`:

```python
CANDIDATES = {"mlp": ("fbd.model.mlp", "MLPModel", PR.MLP_PARAMS),
              "temporal": ("fbd.model.temporal", "TemporalModel", PR.TEMPORAL_PARAMS)}
```

2. In `guards`, replace the first line with:

```python
    reg = R.guard(PREREG, {"backtest_sha256": BACKTEST}, repo=ROOT)
    if name in ADDENDA:
        add = R.guard(ADDENDA[name], {"mlp_json_sha256": MLP_JSON}, repo=ROOT)
        reg = R.merge_addendum(reg, add, name)
```

3. Add, above `score_fold`:

```python
def mlp_probabilities(year: int, test: pd.DataFrame, y, mlp: dict):
    """The pinned S3a fold model's probabilities, verified before use (S3b §5.4)."""
    from fbd.evaluate import metrics as M
    from fbd.model.mlp import MLPModel

    want = mlp["folds"][str(year)]
    path = F.FOLD_DIR / f"candidate_mlp_{year}.joblib"
    if P.sha256_file(path) != want["model_sha256"]:
        raise R.RegistrationError(f"{path.name} does not match mlp.json")
    p = MLPModel.load(path).predict_proba(test)
    if M.auroc(y, p) != want["candidate_auroc"]:
        raise RuntimeError(f"fold {year}: the MLP does not reproduce mlp.json")
    return p
```

4. Change `score_fold`'s signature and body:

```python
def score_fold(name: str, year: int, ens, lead_days, bt: dict, seed: int,
               out_dir: Path = F.FOLD_DIR, mlp: dict | None = None):
```

   - replace `cand = getattr(importlib.import_module(module), cls)().fit(tr, va, list(incumbent.features))` with
     `cand = PM.build_candidate(getattr(importlib.import_module(module), cls), ds).fit(tr, va, list(incumbent.features))`
   - replace `mpath = F.FOLD_DIR / f"candidate_{name}_{year}.joblib"` with `mpath = Path(out_dir) / f"candidate_{name}_{year}.joblib"`
   - after the `rows = pd.DataFrame({...})` block, add:

```python
    if mlp is not None:
        rows["p_mlp"] = mlp_probabilities(year, test, y, mlp)
```

   - after `info = {...}`, add:

```python
    groups = getattr(cand, "SHUFFLE_GROUPS", ())
    if groups:
        info["shuffle_auroc"] = {g: M.auroc(y, cand.predict_proba(test, shuffle=g, seed=seed))
                                 for g in groups}
```

5. In `main`:
   - load the MLP record for the other candidates: `mlp = json.loads(MLP_JSON.read_text(encoding="utf-8")) if name != "mlp" else None`
   - call `score_fold(name, y, ens, lead_days, bt, seed, mlp=mlp)`
   - after `secondary = {...}`, add:

```python
    if mlp is not None:
        secondary["mlp"] = {
            "mean": BS.mean_margin(rows, "p_cand", "p_mlp", PM.YEARS, n_boot, seed),
            "per_year": BS.per_year(rows, "p_cand", "p_mlp", n_year, seed,
                                    who="the MLP", whom=whom)}
        m = secondary["mlp"]["mean"]
        print(f"secondary, {name} - MLP over {list(PM.YEARS)}: {m['point']:+.4f} "
              f"[{m['lo']:+.4f}, {m['hi']:+.4f}]", flush=True)
```

   - add `"shuffle_auroc": {y: f.get("shuffle_auroc") for y, f in folds.items()},` to `exploratory`.

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src python -m pytest tests/ -q` and `python -m bandit -r src/ -ll -q`
Expected: all pass; bandit reports no issues.

- [ ] **Step 5: Commit**

```bash
git add src/fbd/evaluate/promotion.py src/fbd/evaluate/registration.py scripts/promote.py tests/test_promotion.py
git commit -m "Harness: dataset hook, output directory, addenda and the MLP secondary"
```

---

### Task 4: Audit

**Files:**
- Create: `scripts/audit_s3b.py`
- Output: `data/artifacts/s3b_audit.json` (committed)

**Interfaces:**
- Consumes: `promote.score_fold`, `promote.mlp_probabilities`, `promote.CANDIDATES`; `TemporalModel`; `sequence.History`.

- [ ] **Step 1: Write** `scripts/audit_s3b.py`:

```python
"""S3b audit, before registration.

    PYTHONPATH=src python scripts/audit_s3b.py

1. The changed harness re-scores the MLP into a scratch directory and reproduces
   mlp.json exactly (candidate, incumbent and ENS AUROC, every fold), and each
   pinned MLP fold model reproduces its AUROC when loaded.
2. On the real fold-2019 dataset, poisoning everything after each cutoff leaves
   20 sampled 2018 sequences unchanged.
3. Two fits of the fold-2019 temporal model give bit-identical validation-year
   (2018) output, with AUROC at least 0.60 (a defect check, not tuning).
Touches training and validation years only for 2 and 3.
"""
from __future__ import annotations

# Windows: torch must load before scikit-learn (see fbd.model.mlp).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import promote  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import sequence as Q  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.model.temporal import TemporalModel  # noqa: E402

OUT = config.ARTIFACTS / "s3b_audit.json"
SMOKE_FLOOR = 0.60
SEED = 20260919


def neutrality(ens, bt, mlp) -> dict:
    folds = {}
    with tempfile.TemporaryDirectory() as tmp:
        for y in (2019, 2020, 2021, 2022):
            rows, info = promote.score_fold("mlp", y, ens, config.DECISION_BAND, bt, SEED,
                                            out_dir=Path(tmp))
            want = mlp["folds"][str(y)]
            got = {k: info[k] for k in ("candidate_auroc", "incumbent_auroc", "ens_auroc")}
            loaded = M.auroc(rows.bust, promote.mlp_probabilities(y, _test(y, ens), rows.bust, mlp))
            folds[str(y)] = {"got": got, "want": {k: want[k] for k in got},
                             "loaded_mlp_auroc": loaded,
                             "ok": got == {k: want[k] for k in got}
                                   and loaded == want["candidate_auroc"]}
            print(f"  {y}: {folds[str(y)]['ok']}", flush=True)
    return {"folds": folds, "ok": all(f["ok"] for f in folds.values())}


def _test(y, ens):
    ds = pd.read_parquet(F.fold_path(y, "strict"))
    test, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
    return test


def causality() -> dict:
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    ds["init_date"] = pd.to_datetime(ds.init_date)
    ds["valid_date"] = pd.to_datetime(ds.valid_date)
    rng = np.random.default_rng(SEED)
    keys = ds[ds.year == 2018][["subdivision_id", "init_date"]].drop_duplicates()
    sample = keys.iloc[rng.choice(len(keys), 20, replace=False)]
    clean = Q.History.from_frame(ds)
    bad = []
    for s, t in sample.itertuples(index=False):
        p = ds.copy()
        p.loc[p.valid_date > t - pd.Timedelta(days=Q.LAG), ["obs_rain_mm", "error", "bust"]] = 1e3
        p.loc[p.init_date > t, Q.WEATHER] = 1e3
        a, b = clean.raw([s], [t]), Q.History.from_frame(p).raw([s], [t])
        if not np.array_equal(a, b, equal_nan=True):
            bad.append(f"{s} {t.date()}")
    return {"n": len(sample), "violations": bad, "ok": not bad}


def determinism() -> dict:
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = list(T.BustModel.load(F.model_path(2019, "strict")).features)
    outs, secs, hist = [], [], None
    for _ in range(2):
        t0 = time.time()
        m = TemporalModel(dataset=ds).fit(tr, va, feats)
        secs.append(round(time.time() - t0, 1))
        outs.append(m.predict_raw(va))
        hist = m.history
    auc = M.auroc(va.bust.to_numpy(float), outs[0])
    return {"bit_identical": bool(np.array_equal(outs[0], outs[1])), "val_year": 2018,
            "val_auroc": auc, "floor": SMOKE_FLOOR, "fit_seconds": secs, "history": hist,
            "ok": bool(np.array_equal(outs[0], outs[1])) and auc >= SMOKE_FLOOR}


def main() -> int:
    t0 = time.time()
    bt = json.loads(promote.BACKTEST.read_text(encoding="utf-8"))
    mlp = json.loads(promote.MLP_JSON.read_text(encoding="utf-8"))
    ens = E.load_ens()
    print("1. harness neutrality (MLP re-scored into a scratch directory) ...", flush=True)
    c1 = neutrality(ens, bt, mlp)
    print("2. causality on the real fold-2019 dataset ...", flush=True)
    c2 = causality()
    print(f"  {c2}", flush=True)
    print("3. determinism and smoke, fold 2019, validation year 2018 ...", flush=True)
    c3 = determinism()
    print(f"  bit-identical {c3['bit_identical']}, AUROC {c3['val_auroc']:.4f}, "
          f"{c3['fit_seconds']} s", flush=True)
    out = {"temporal_params_sha256": PR.params_sha256(PR.TEMPORAL_PARAMS),
           "neutrality": c1, "causality": c2, "determinism": c3,
           "ok": c1["ok"] and c2["ok"] and c3["ok"]}
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT}; all ok: {out['ok']} ({time.time() - t0:.0f}s)")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run** `PYTHONPATH=src python scripts/audit_s3b.py` (background; it prints per step).
Expected: `all ok: True`. If neutrality fails, stop: the harness change is not neutral, so fix the harness, never the check. If determinism fails, find the non-deterministic op. If AUROC < 0.60, treat it as a defect in the sequence or model code, not a tuning signal.

- [ ] **Step 3: Verify the MLP pins are untouched**

Run: `git status --short data/processed/backtest/` (gitignored) and `sha256sum data/processed/backtest/candidate_mlp_20*.joblib`, compared with `mlp.json`.
Expected: every hash equal.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit_s3b.py data/artifacts/s3b_audit.json
git commit -m "S3b audit: the harness change is neutral, the history is causal, fits are deterministic"
```

---

### Task 5: Register

**Files:**
- Create: `docs/PREREGISTRATION_S3B.md`

- [ ] **Step 1: Write the addendum.** Sections, in order:
  1. **Header:** "Registered 2026-09-27, before any temporal model produced a prediction for 2019–2022." Code: `fbd/model/{sequence,temporal,params}.py`, `scripts/promote.py`, `scripts/audit_s3b.py`, committed before this file. Spec path.
  2. **The question:** spec §1, verbatim.
  3. **Reproduced before registration:** a table from `s3b_audit.json`, with neutrality `ok` per fold, the causality sample size and violations (0), bit-identical, the 2018 validation AUROC with its floor, and the fit seconds.
  4. **The candidate, frozen:** the table in spec §4 plus the channel table in spec §3.
  5. **Primary:** "S3's §4, unchanged", restating the quantity, rows, interval (98.33%, 10,000, seed 20260919) and the verdict.
  6. **Secondary:** spec §6, with S3b − MLP first.
  7. **Exploratory:** spec §6.
  8. **Blinding:** spec §10.
  9. **Consequences:** the table in spec §7.
  10. **What would invalidate this:** changing `TEMPORAL_PARAMS`, `mlp.json`, the harness, this file or S3's after scoring (`promote.py` refuses the parameter and file changes), or scoring before this file is committed.
  11. **Registration block:**

````
```registration
temporal_params_sha256: <PR.params_sha256(PR.TEMPORAL_PARAMS), as printed in s3b_audit.json>
mlp_json_sha256: <sha256 of data/artifacts/candidates/mlp.json>
```
````

   Fill in both hashes by computing them:

```bash
PYTHONPATH=src python -c "from fbd.model import params as PR; print(PR.params_sha256(PR.TEMPORAL_PARAMS))"
sha256sum data/artifacts/candidates/mlp.json
```

- [ ] **Step 2: Guard dry run.** Before committing, `PYTHONPATH=src python scripts/promote.py --candidate temporal` must print `REFUSED: PREREGISTRATION_S3B.md is not committed as-is`. It must not fit anything.

- [ ] **Step 3: Commit and push**

```bash
git add docs/PREREGISTRATION_S3B.md
git commit -m "Register S3b: the temporal candidate, before any 2019-2022 prediction"
git push
```

- [ ] **Step 4:** Wait for the PR's CI to go green, using `mcp__ccd_pr__get_status` rather than polling.

---

### Task 6: Run once, record

**Files:**
- Output: `data/artifacts/candidates/temporal.json`, `docs/figures/candidate_temporal.png`/`.svg`
- Modify: `DECISIONS.md` (D-031), `README.md` ("Other model families"), `FRONTEND_LOGIC.md` §8, `HANDOFF.md`

- [ ] **Step 1:** `PYTHONPATH=src python scripts/promote.py --candidate temporal` (background). Run it once. Do not re-run it to get a different number.
- [ ] **Step 2:** `PYTHONPATH=src python scripts/plot_candidate.py --candidate temporal`.
- [ ] **Step 3:** Write D-031. It records the primary with its verdict, each fold's AUROCs, S3b − MLP (mean and per year), S3b − ENS, the shuffle drops (labelled exploratory), the seed spread, the reading given by the row of spec §7 that applies, and what it does not settle.
- [ ] **Step 4:** Update the README "Other model families" and `FRONTEND_LOGIC.md` §8 as spec §7 dictates, and HANDOFF's queue (S3c next).
- [ ] **Step 5:** Run the full suite, bandit and the secrets grep, then commit and push. Check the PR's CI with `get_status`.

```bash
git add data/artifacts/candidates/temporal.json docs/figures/candidate_temporal.* DECISIONS.md README.md FRONTEND_LOGIC.md HANDOFF.md
git commit -m "Run S3b once: <verdict in words>"
git push
```

---

## Self-review

- Spec §3 → Task 1; §4 → Task 2; §5.1–5.5 → Task 3; §6 → Tasks 3 and 5; §7 → Task 6; §8 → Tasks 1–3; §9 → Task 4; §10 → Task 5; §11 → task order.
- Names are consistent across tasks: `needs_dataset`, `dataset=`, `SHUFFLE_GROUPS`, `GROUP_CHANNELS`, `History.from_frame/raw`, `SeqScaler.fit/transform`, `masks_of`, `unique_keys`, `build_candidate`, `merge_addendum`, `score_fold(..., seed, out_dir, mlp)`, `mlp_probabilities`, `MLP_JSON`, `ADDENDA`.
- The two registration hashes are computed values, not placeholders: Task 5 Step 1 shows the commands.
