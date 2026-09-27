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
        present = np.isfinite(flat)
        n = present.sum(axis=0)
        filled = np.where(present, flat, 0.0)
        mean = np.divide(filled.sum(axis=0), n, out=np.zeros(N_VALUES), where=n > 0)
        dev = np.where(present, flat - mean, 0.0)
        std = np.sqrt(np.divide((dev ** 2).sum(axis=0), n, out=np.ones(N_VALUES), where=n > 0))
        std = np.where(std > 0, std, 1.0)
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
