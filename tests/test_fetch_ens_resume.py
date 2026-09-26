"""The pure parts of the ENS fetch: resume, atomic writes, member checks, retries.

A 25 GB fetch over anonymous cloud storage will be interrupted. These prove an
interruption never costs finished work and never leaves a half-written shard
that later counts as done.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd.ingest import ens_fetch as F  # noqa: E402


def test_shard_path_layout(tmp_path):
    p = F.shard_path(tmp_path, "2022-06-14T00:00")
    assert p == tmp_path / "shards" / "2022" / "2022-06-14.parquet"


def test_done_dates_reads_legacy_and_shards_for_that_year_only(tmp_path):
    pd.DataFrame({"init_date": pd.to_datetime(["2022-06-01", "2022-06-04", "2021-06-01"])}
                 ).to_parquet(tmp_path / "ens_spread_2022_every3.parquet")
    shard = F.shard_path(tmp_path, "2022-06-02")
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"x")
    (shard.parent / "2022-06-03.parquet.tmp").write_bytes(b"half")   # never counts

    done = F.done_dates(tmp_path, 2022)
    assert done == {pd.Timestamp(d) for d in ("2022-06-01", "2022-06-02", "2022-06-04")}


def test_dates_to_fetch_is_all_minus_done():
    all_dates = pd.date_range("2022-06-01", periods=5)
    todo = F.dates_to_fetch(all_dates, {pd.Timestamp("2022-06-02"), pd.Timestamp("2022-06-04")})
    assert todo == [pd.Timestamp(d) for d in ("2022-06-01", "2022-06-03", "2022-06-05")]


def test_write_atomic_leaves_no_tmp(tmp_path):
    p = tmp_path / "shards" / "2022" / "2022-06-01.parquet"
    F.write_atomic(pd.DataFrame({"a": [1]}), p)
    assert p.exists() and not list(p.parent.glob("*.tmp"))


def test_a_failed_write_never_looks_done(tmp_path, monkeypatch):
    p = tmp_path / "shards" / "2022" / "2022-06-01.parquet"

    def boom(self, path, **kw):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        F.write_atomic(pd.DataFrame({"a": [1]}), p)
    assert not p.exists()
    assert not list(p.parent.glob("*.tmp")), "the partial file must be cleaned up"


def test_short_ensemble_is_refused():
    F.check_members(50)
    with pytest.raises(F.ShortEnsemble):
        F.check_members(49)


def test_reduce_members_matches_the_published_arithmetic():
    rng = np.random.default_rng(0)
    means = rng.gamma(2.0, 3.0, size=(50, 10, 3))            # member, lead, sub
    df = F.reduce_members(means, ["A", "B", "C"], "2022-06-14")
    assert len(df) == 30
    row = df[(df.lead_day == 4) & (df.subdivision_id == "B")].iloc[0]
    assert row.ens_spread == pytest.approx(means[:, 3, 1].std(ddof=1))
    assert row.ens_mean == pytest.approx(means[:, 3, 1].mean())
    assert row.valid_date == pd.Timestamp("2022-06-17")
    assert (df.n_members == 50).all()


def test_retries_back_off_then_succeed():
    calls, slept = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("throttled")
        return "ok"

    assert F.with_retries(flaky, sleep=slept.append, log=lambda *_: None) == "ok"
    assert slept == [2.0, 4.0]


def test_retries_cap_the_delay_and_give_up():
    slept = []

    def down():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        F.with_retries(down, attempts=7, sleep=slept.append, log=lambda *_: None)
    assert slept == [2.0, 4.0, 8.0, 16.0, 32.0, 32.0]
