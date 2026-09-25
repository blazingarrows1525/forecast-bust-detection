"""S1b statistics on synthetic folds. Needs scikit-learn (AUROC)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import backtest_stats as BS  # noqa: E402


def _rows(ens_wins_in=None, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for year in (2019, 2020, 2021, 2022):
        n_dates, per = 40, 30
        bust = rng.random(n_dates * per) < 0.15
        signal = bust + rng.normal(0, 0.6, len(bust))
        noise = rng.random(len(bust))
        good, weak = (noise, signal) if year == ens_wins_in else (signal, noise)
        frames.append(pd.DataFrame({
            "year": year,
            "init_date": np.repeat([f"{year}-06-{d:02d}" for d in range(1, n_dates + 1)], per),
            "month": 6, "subdivision_id": "A", "lead_day": 3,
            "bust": bust.astype(float), "p_model": good, "p_ens": weak, "p_proxy": weak,
        }))
    return pd.concat(frames, ignore_index=True)


def test_primary_averages_only_the_registered_years():
    rows = _rows()
    p = BS.primary(rows, [2019, 2020, 2021], n_boot=200, seed=1)
    assert p["verdict"] == "model_better" and p["years"] == [2019, 2020, 2021]
    per = BS.per_year(rows, "p_model", "p_ens", n_boot=100, seed=1)
    assert abs(p["point"] - np.mean([per[y]["point"] for y in ("2019", "2020", "2021")])) < 1e-12


def test_per_year_states_a_year_the_ensemble_wins():
    per = BS.per_year(_rows(ens_wins_in=2020), "p_model", "p_ens", n_boot=200, seed=1)
    assert per["2020"]["statement"] == "ENS spread outranks the model in 2020"
    assert per["2019"]["statement"] is None
    assert per["2019"]["n_init_dates"] == 40


def test_by_month_is_keyed_by_year_then_month():
    out = BS.by_month(_rows(), "p_model", "p_ens", n_boot=50, seed=1)
    assert set(out) == {"2019", "2020", "2021", "2022"} and set(out["2019"]) == {"6"}


def test_a_smaller_alpha_widens_the_interval_around_the_same_point():
    rows = _rows()
    wide = BS.mean_margin(rows, "p_model", "p_ens", [2019, 2020, 2021], 300, 1, alpha=0.05 / 3)
    base = BS.mean_margin(rows, "p_model", "p_ens", [2019, 2020, 2021], 300, 1)
    assert wide["point"] == base["point"]
    assert wide["lo"] <= base["lo"] and wide["hi"] >= base["hi"]
    assert wide["alpha"] == pytest.approx(0.05 / 3)
