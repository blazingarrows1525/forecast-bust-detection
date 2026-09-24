"""The S1b folds: an expanding window, one test year each.

Pure on purpose: the fold table and the paths are checked in CI, and a fold
must never be able to write over the frozen dataset S1 was registered on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fbd import config


@dataclass(frozen=True)
class Fold:
    test: int
    train: tuple
    val: tuple


FOLDS = {
    2019: Fold(2019, (2016, 2017), (2018,)),
    2020: Fold(2020, (2016, 2017, 2018), (2019,)),
    2021: Fold(2021, (2016, 2017, 2018, 2019), (2020,)),
    2022: Fold(2022, (2016, 2017, 2018, 2019, 2020), (2021,)),
}
#: Registered: 2022 is reported beside the primary, never in it (S1 saw it).
PRIMARY_YEARS = (2019, 2020, 2021)
#: legacy = today's pipeline exactly; strict = regime statistics on training years.
MODES = ("legacy", "strict")
FOLD_DIR = config.PROCESSED / "backtest"


def regime_fit_years(fold: Fold, mode: str):
    """Years the regime standardisation is fitted on; None means every date."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return None if mode == "legacy" else fold.train


def fold_path(test_year: int, mode: str) -> Path:
    return FOLD_DIR / f"fold_{test_year}_{mode}.parquet"


def model_path(test_year: int, mode: str) -> Path:
    return FOLD_DIR / f"model_{test_year}_{mode}.joblib"


def split_labels(years, train, val, test) -> np.ndarray:
    """train / val / test by year; any other year is "excluded" (e.g. after the test)."""
    y = np.asarray(years)
    return np.where(np.isin(y, list(test)), "test",
                    np.where(np.isin(y, list(val)), "val",
                             np.where(np.isin(y, list(train)), "train", "excluded")))
