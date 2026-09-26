"""One loader for true IFS ENS spread, and one definition of the rows it is compared on.

Two things every ENS script used to do for itself, both of which could go wrong
silently:

* LOADING. ENS arrives in two shapes: legacy files, one per fetch
  (``ens_spread_<year>_every<N>.parquet``), and per-date shards
  (``shards/<year>/<YYYY-MM-DD>.parquet``). Globbing and concatenating them
  counts any date present in both twice -- and the cluster bootstrap resamples
  by date, so that date would silently weigh double in the verdict.
  ``load_ens`` deduplicates, and raises if two sources disagree about one
  forecast, because that is a provenance problem, not a merge problem.

* THE ROW RULE. ``comparison_rows`` is the single definition of which rows the
  model is compared to ENS on. The published margin, the S1 audit and the
  registered settlement all call it, so they cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fbd import config

ENS_DIR = config.WB2_RAW / "ens"
KEY = ["subdivision_id", "init_date", "lead_day"]
VALUES = ["ens_spread", "ens_mean"]
#: Two sources this close are the same forecast read twice. Relative, because
#: spreads run from ~0.01 to 100+ mm; far below physical meaning, above
#: float32 summation-order noise.
RTOL = 1e-5
ATOL = 1e-6


class ConflictingDuplicate(ValueError):
    """Two ENS sources give different values for one (subdivision, init, lead)."""


def legacy_paths(ens_dir: Path = ENS_DIR) -> list[Path]:
    return sorted(Path(ens_dir).glob("ens_spread_*.parquet"))


def shard_paths(ens_dir: Path = ENS_DIR, year: int | None = None) -> list[Path]:
    root = Path(ens_dir) / "shards"
    return sorted(root.glob(f"{year}/*.parquet" if year else "*/*.parquet"))


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per KEY. Equal duplicates keep the first; conflicting ones raise."""
    dup = df.duplicated(KEY, keep=False)
    if not dup.any():
        return df.reset_index(drop=True)
    d = df.loc[dup, KEY + VALUES]
    first = d.groupby(KEY)[VALUES].transform("first")
    # equal_nan: a NaN agrees with a NaN, and never with a number.
    same = np.isclose(d[VALUES].to_numpy(float), first.to_numpy(float),
                      rtol=RTOL, atol=ATOL, equal_nan=True).all(axis=1)
    if not same.all():
        bad = d.loc[~same, KEY].drop_duplicates().head(5)
        raise ConflictingDuplicate(
            f"{int((~same).sum())} duplicate ENS rows disagree, e.g.\n"
            f"{bad.to_string(index=False)}\n"
            "Two values for one forecast means two fetches disagree. Find out "
            "why before trusting either."
        )
    return df.drop_duplicates(KEY, keep="first").reset_index(drop=True)


def load_ens(ens_dir: Path = ENS_DIR, include_shards: bool = True) -> pd.DataFrame:
    """Every ENS row on disk, once. ``include_shards=False`` gives legacy only."""
    paths = legacy_paths(ens_dir) + (shard_paths(ens_dir) if include_shards else [])
    if not paths:
        raise FileNotFoundError(f"no ENS data in {ens_dir}; run scripts/fetch_ens.py")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["init_date"] = pd.to_datetime(df.init_date).dt.normalize()
    return dedupe(df)


def dates_by_year(df: pd.DataFrame) -> dict[int, int]:
    d = pd.Series(pd.to_datetime(df.init_date).unique())
    return {int(y): int(n) for y, n in d.dt.year.value_counts().sort_index().items()}


def comparison_rows(ds: pd.DataFrame, ens: pd.DataFrame,
                    lead_days=config.DECISION_BAND) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(test, fit)``: the rows the model is compared to ENS on.

    The published rule, moved here verbatim from compute_confidence_intervals.py:
      * inner join on (subdivision, init, lead); drop rows missing bust or spread
      * ``test`` = split "test", restricted to ``lead_days``
      * ``fit``  = splits "train" and "val", all leads (for calibrated variants)

    The model scores every test row, including rows the served product would
    refuse as out-of-distribution. This measures ranking skill, not the served
    subset, and every output that uses it says so.
    """
    ds = ds.copy()
    ens = ens.copy()
    ds["init_date"] = pd.to_datetime(ds.init_date)
    ens["init_date"] = pd.to_datetime(ens.init_date)
    merged = ds.merge(ens[KEY + VALUES], on=KEY, how="inner").dropna(
        subset=["bust", "ens_spread"])
    merged["ens_spread_rel"] = merged.ens_spread / (merged.ens_mean + 1.0)
    test = merged[(merged.split == "test") & merged.lead_day.isin(list(lead_days))]
    fit = merged[merged.split.isin(["train", "val"])]
    return test, fit
