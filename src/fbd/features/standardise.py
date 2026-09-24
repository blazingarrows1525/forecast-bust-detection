"""Standardisation fitted on chosen rows only.

A fold's test years must never reach a fitted mean or standard deviation.
Pure pandas, so the leakage tests run in CI without xarray. With a mask that
selects every row, each function computes exactly what the published pipeline
computed, which is what lets the legacy rebuild reproduce dataset.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_mask(dates, years=None) -> np.ndarray:
    """True for rows whose date falls in ``years``; every row when ``years`` is None."""
    d = pd.to_datetime(pd.Series(dates))
    if years is None:
        return np.ones(len(d), dtype=bool)
    return d.dt.year.isin(list(years)).to_numpy()


def standardise(frame: pd.DataFrame, cols, mask, suffix: str = "_z") -> pd.DataFrame:
    """(x - mean) / sd per column; mean and sd from the masked rows, sd 0 -> 1."""
    m = np.asarray(mask, dtype=bool)
    mu = frame.loc[m, cols].mean()
    sd = frame.loc[m, cols].std().replace(0, 1.0)
    return pd.DataFrame({f"{c}{suffix}": (frame[c] - mu[c]) / sd[c] for c in cols},
                        index=frame.index)


def standardise_within(frame: pd.DataFrame, cols, by, mask,
                       suffix: str = "_zl") -> pd.DataFrame:
    """Per-group (x - mean) / (sd + 1e-9), with mean and sd from masked rows."""
    m = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
    out = {}
    for c in cols:
        def z(s: pd.Series) -> pd.Series:
            f = s[m.loc[s.index].to_numpy()]
            return (s - f.mean()) / (f.std() + 1e-9)
        out[f"{c}{suffix}"] = frame.groupby(by)[c].transform(z)
    return pd.DataFrame(out, index=frame.index)
