"""The pure parts of the ENS fetch, separated so they are testable without a network.

scripts/fetch_ens.py does the I/O (anonymous WeatherBench 2 reads, ~123 MB per
init date). Everything here is deterministic and dependency-light so it runs in
CI, where neither xarray nor gcsfs is installed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

#: IFS ENS perturbed members. Spread from fewer is biased low, so a short date
#: is refused (and reported), never averaged in.
EXPECTED_MEMBERS = 50


class ShortEnsemble(ValueError):
    """An init date arrived with the wrong number of members."""


def shard_path(ens_dir, init_date) -> Path:
    d = pd.Timestamp(init_date).normalize()
    return Path(ens_dir) / "shards" / str(d.year) / f"{d:%Y-%m-%d}.parquet"


def done_dates(ens_dir, year: int) -> set:
    """Dates of ``year`` already on disk: inside a legacy file, or as a shard.

    ``*.parquet.tmp`` never matches, so an interrupted write is never "done".
    """
    from fbd.evaluate.ens import legacy_paths

    done: set = set()
    for p in legacy_paths(ens_dir):
        d = pd.to_datetime(pd.read_parquet(p, columns=["init_date"]).init_date)
        done |= {pd.Timestamp(x).normalize() for x in d.unique()
                 if pd.Timestamp(x).year == year}
    for p in (Path(ens_dir) / "shards" / str(year)).glob("*.parquet"):
        done.add(pd.Timestamp(p.stem))
    return done


def dates_to_fetch(all_dates, done) -> list:
    done = {pd.Timestamp(d).normalize() for d in done}
    return [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d).normalize() not in done]


def write_atomic(df: pd.DataFrame, path) -> None:
    """Write to ``<path>.tmp`` then rename. A crash leaves no file named ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def check_members(n: int) -> None:
    if n != EXPECTED_MEMBERS:
        raise ShortEnsemble(f"{n} members, expected {EXPECTED_MEMBERS}")


def reduce_members(means: np.ndarray, sub_ids: list, init_date) -> pd.DataFrame:
    """(member, lead, sub) area means -> one row per (lead, sub).

    The same arithmetic as the legacy fetch (numpy std, ddof=1), so a re-fetched
    legacy date reproduces its stored values.
    """
    spread = means.std(axis=0, ddof=1)
    mean = means.mean(axis=0)
    n_valid = np.isfinite(means).sum(axis=0)
    n_l, n_s = spread.shape
    d = pd.Timestamp(init_date).normalize()
    df = pd.DataFrame({
        "init_date": d,
        "lead_day": np.repeat(np.arange(1, n_l + 1), n_s),
        "subdivision_id": np.tile(list(sub_ids), n_l),
        "ens_spread": spread.reshape(-1),
        "ens_mean": mean.reshape(-1),
        "n_members": int(means.shape[0]),
        "n_valid_members": n_valid.reshape(-1).astype(int),
    })
    df["valid_date"] = df.init_date + pd.to_timedelta(df.lead_day - 1, unit="D")
    return df


def with_retries(fn, attempts: int = 5, base_delay: float = 2.0,
                 max_delay: float = 32.0, sleep=time.sleep, log=print):
    """Call ``fn``; on failure wait 2, 4, 8, ... (capped) and try again."""
    for k in range(attempts):
        try:
            return fn()
        except Exception as exc:  # network failures arrive as many types
            if k == attempts - 1:
                raise
            delay = min(max_delay, base_delay * 2 ** k)
            log(f"    attempt {k + 1} failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s")
            sleep(delay)
