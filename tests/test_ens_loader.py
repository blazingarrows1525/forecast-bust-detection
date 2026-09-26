"""The shared ENS loader and the one row rule every comparison uses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402


def _rows(dates, spread=1.0, subs=("A", "B"), leads=(1, 2)):
    return pd.DataFrame([
        {"init_date": pd.Timestamp(d), "subdivision_id": s, "lead_day": lead,
         "ens_spread": spread, "ens_mean": 2.0}
        for d in dates for s in subs for lead in leads
    ])


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_legacy_and_shards_combine(tmp_path):
    _write(tmp_path / "ens_spread_2022_every3.parquet", _rows(["2022-06-01", "2022-06-04"]))
    _write(tmp_path / "shards/2022/2022-06-02.parquet", _rows(["2022-06-02"]))
    df = E.load_ens(tmp_path)
    assert E.dates_by_year(df) == {2022: 3}
    assert len(df) == 3 * 2 * 2


def test_a_date_in_both_sources_counts_once(tmp_path):
    """The trap this module exists for: the bootstrap resamples by date."""
    _write(tmp_path / "ens_spread_2022_every3.parquet", _rows(["2022-06-01"]))
    _write(tmp_path / "shards/2022/2022-06-01.parquet", _rows(["2022-06-01"]))
    df = E.load_ens(tmp_path)
    assert len(df) == 4
    assert not df.duplicated(E.KEY).any()


def test_conflicting_duplicates_raise(tmp_path):
    _write(tmp_path / "ens_spread_2022_every3.parquet", _rows(["2022-06-01"], spread=1.0))
    _write(tmp_path / "shards/2022/2022-06-01.parquet", _rows(["2022-06-01"], spread=1.5))
    with pytest.raises(E.ConflictingDuplicate):
        E.load_ens(tmp_path)


def test_nan_agrees_with_nan_but_not_with_a_number(tmp_path):
    a = _rows(["2022-06-01"])
    a.loc[0, "ens_spread"] = np.nan
    _write(tmp_path / "ens_spread_2022_every3.parquet", a)
    _write(tmp_path / "shards/2022/2022-06-01.parquet", a.copy())
    assert len(E.load_ens(tmp_path)) == 4

    b = a.copy()
    b.loc[0, "ens_spread"] = 1.0
    _write(tmp_path / "shards/2022/2022-06-01.parquet", b)
    with pytest.raises(E.ConflictingDuplicate):
        E.load_ens(tmp_path)


def test_legacy_only_mode_ignores_shards(tmp_path):
    _write(tmp_path / "ens_spread_2022_every3.parquet", _rows(["2022-06-01"]))
    _write(tmp_path / "shards/2022/2022-06-02.parquet", _rows(["2022-06-02"]))
    assert E.dates_by_year(E.load_ens(tmp_path, include_shards=False)) == {2022: 1}


def test_no_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.load_ens(tmp_path)


def test_comparison_rows_is_the_published_rule():
    rows = []
    for split, year in (("train", 2019), ("val", 2021), ("test", 2022)):
        for lead in range(1, 11):
            rows.append({"init_date": pd.Timestamp(f"{year}-06-01"), "subdivision_id": "A",
                         "lead_day": lead, "split": split, "bust": float(lead % 2)})
    ds = pd.DataFrame(rows)
    ens = ds[E.KEY].assign(ens_spread=1.0, ens_mean=2.0)
    ds.loc[(ds.split == "test") & (ds.lead_day == 4), "bust"] = np.nan  # no label -> dropped
    ens.loc[(ens.init_date.dt.year == 2022) & (ens.lead_day == 5), "ens_spread"] = np.nan

    test, fit = E.comparison_rows(ds, ens)
    assert set(test.lead_day) == set(config.DECISION_BAND) - {4, 5}
    assert (test.split == "test").all()
    assert set(fit.split) == {"train", "val"} and len(fit) == 20  # all leads
    assert "ens_spread_rel" in test.columns
    assert test.bust.notna().all() and test.ens_spread.notna().all()
