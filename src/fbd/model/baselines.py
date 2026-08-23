"""The three baselines, built before the model exists (LOGIC.md sec 8.1).

Order matters and is prescribed:
  1. climatological bust rate per (subdivision, month, lead) -- the dumbest thing
     that could possibly work;
  2. spread -- the respected operational baseline, and the one to beat;
  3. logistic regression on spread + lead -- the simplest learned model.

If the tree model cannot beat #2 by a clear margin on held-out data, that has to
be reported as such.  An honest "we matched spread, and here is why" is worth
more than an unverifiable claimed win.

A note on which spread.  The operational baseline is IFS ensemble spread, which
we cannot afford across the whole archive (DECISIONS.md D-007).  These functions
take whichever spread column they are given, so they run on the lagged-ensemble
spread for the full record and on true IFS ENS spread for the subsample where it
exists.  Both results are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fbd import config


class ClimatologyBaseline:
    """P(bust | subdivision, month, lead) estimated on the training years."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.table_: pd.DataFrame | None = None
        self.prior_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "ClimatologyBaseline":
        d = df.dropna(subset=["bust"]).copy()
        d["month"] = d.valid_date.dt.month
        self.prior_ = float(d.bust.mean())
        g = d.groupby(["subdivision_id", "month", "lead_day"]).bust.agg(["mean", "size"])
        g = g.reset_index().rename(columns={"mean": "rate", "size": "n"})
        g["p"] = (g.rate * g.n + self.prior_ * self.smoothing) / (g.n + self.smoothing)
        self.table_ = g[["subdivision_id", "month", "lead_day", "p"]]
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        d = df.copy()
        d["month"] = d.valid_date.dt.month
        m = d.merge(self.table_, on=["subdivision_id", "month", "lead_day"], how="left")
        return m.p.fillna(self.prior_).to_numpy()


class SpreadBaseline:
    """Ensemble spread mapped monotonically to a bust probability.

    Spread is a *score*, not a probability, so comparing it to a calibrated model
    on Brier score would be unfair to spread.  We give it the fairest possible
    treatment: an isotonic fit on the training years, which is the best any
    monotone transform of spread could do.  Beating that is a real win.
    """

    def __init__(self, spread_col: str = "lagged_spread", use_lead: bool = True):
        self.spread_col = spread_col
        self.use_lead = use_lead
        self.models_: dict = {}
        self.prior_: float = 0.0

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.spread_col].to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame) -> "SpreadBaseline":
        d = df.dropna(subset=["bust", self.spread_col])
        self.prior_ = float(d.bust.mean())
        keys = d.lead_day.unique() if self.use_lead else [None]
        for k in keys:
            sub = d if k is None else d[d.lead_day == k]
            if len(sub) < 50 or sub.bust.nunique() < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(self._x(sub), sub.bust.to_numpy(dtype=float))
            self.models_[k] = iso
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._x(df)
        out = np.full(len(df), self.prior_, dtype=float)
        if not self.use_lead:
            iso = self.models_.get(None)
            ok = np.isfinite(x)
            if iso is not None:
                out[ok] = iso.predict(x[ok])
            return out
        for k, iso in self.models_.items():
            m = (df.lead_day.to_numpy() == k) & np.isfinite(x)
            if m.any():
                out[m] = iso.predict(x[m])
        return out


class LogisticBaseline:
    """Logistic regression on spread + lead day -- the simplest learned model."""

    def __init__(self, spread_col: str = "lagged_spread"):
        self.spread_col = spread_col
        self.model_ = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED
        )
        self.prior_ = 0.0

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        s = df[self.spread_col].to_numpy(dtype=float)
        s = np.nan_to_num(s, nan=0.0)
        lead = df.lead_day.to_numpy(dtype=float)
        return np.column_stack([s, np.log1p(s), lead, s * lead])

    def fit(self, df: pd.DataFrame) -> "LogisticBaseline":
        d = df.dropna(subset=["bust"])
        self.prior_ = float(d.bust.mean())
        self.model_.fit(self._design(d), d.bust.to_numpy(dtype=int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._design(df))[:, 1]


class PersistenceBaseline:
    """Bonus baseline: today's forecast magnitude alone.

    Included because a sceptical judge will ask whether the model is really
    learning predictability or just "it rains hard in Konkan".  If raw forecast
    rainfall alone scores nearly as well as the model, we need to say so.
    """

    def __init__(self, col: str = "fcst_rain_mm"):
        self.col = col
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.prior_ = 0.0

    def fit(self, df: pd.DataFrame) -> "PersistenceBaseline":
        d = df.dropna(subset=["bust", self.col])
        self.prior_ = float(d.bust.mean())
        self.iso_.fit(d[self.col].to_numpy(float), d.bust.to_numpy(float))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.col].to_numpy(float)
        out = np.full(len(df), self.prior_)
        ok = np.isfinite(x)
        out[ok] = self.iso_.predict(x[ok])
        return out
