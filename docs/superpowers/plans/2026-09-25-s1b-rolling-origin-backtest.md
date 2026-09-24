# S1b — Rolling-Origin Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with a test registered before the 2019–2020 ENS data exists, whether the model's edge over real 50-member IFS ENS spread holds on average over 2019–2021, using one rebuilt dataset and one model per test year.

**Architecture:** Every fit that depends on "training years" takes the years as an explicit parameter (`train_years`, `fit_years`), with defaults that reproduce today's pipeline bit for bit. Pure modules hold the fold table (`fbd/evaluate/folds.py`), the hyperparameters (`fbd/model/params.py`) and the standardisation helpers (`fbd/features/standardise.py`) so CI can test them without xarray or scikit-learn. `scripts/build_dataset.py` gains a `build()` function and `--fold/--mode`; `scripts/backtest.py` runs the registered analysis through `fbd/evaluate/backtest_stats.py` and a new stratified bootstrap in `fbd/evaluate/uncertainty.py`.

**Tech Stack:** Python 3.10/3.11, pandas, numpy, scikit-learn + xgboost (local only), xarray (local only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-s1b-rolling-origin-backtest-design.md`

## Global Constraints

- Commit messages carry **no** `Co-Authored-By` or Claude attribution lines (standing user instruction).
- No SIH / SIH26079 on any product surface.
- `data/processed/dataset.parquet` and `data/artifacts/bust_model.joblib` are **never modified**. Never run `scripts/build_dataset.py` without `--fold`: its default writes `dataset.parquet`.
- Folds: 2019 = train 2016–2017, val 2018; 2020 = train 2016–2018, val 2019; 2021 = train 2016–2019, val 2020; 2022 = train 2016–2020, val 2021.
- Primary years **2019, 2020, 2021**; seed **20260919**; primary and secondary-mean resamples **10,000**; per-year and exploratory resamples **2,000**; alpha **0.05**; decision band **Day 3–7**.
- Audit tolerance **0.001** AUROC for the retrain and proxy reproductions.
- CI installs only `requirements-dev.txt` (numpy, pandas, pyarrow, fastapi, pytest, httpx): modules imported by CI tests must not import scikit-learn, xgboost, xarray or geopandas. Tests needing sklearn use `pytest.importorskip("sklearn")`.
- The 241 existing tests keep passing.

## File Structure

| file | status | responsibility |
|---|---|---|
| `src/fbd/evaluate/folds.py` | create | fold table, modes, fold/model paths, split labels (pure) |
| `src/fbd/model/params.py` | create | `DEFAULT_PARAMS`, `params_sha256()` (pure) |
| `src/fbd/model/train.py` | modify | `BustModel.fit` builds its params from `DEFAULT_PARAMS` |
| `src/fbd/features/standardise.py` | create | `fit_mask`, `standardise`, `standardise_within` (pure) |
| `src/fbd/features/forecast.py` | modify | `build(..., train_years=None)` threads years to both climatologies |
| `src/fbd/features/era5.py` | modify | `national_daily(years=None, train_years=None)` via `standardise` |
| `src/fbd/regime/classify.py` | modify | `regime_scores(..., fit_years=None)`, `classify(..., fit_years=None)` via `standardise_within` |
| `scripts/build_dataset.py` | modify | `build()`, `build_fold()`, `--fold/--mode`; default behaviour unchanged |
| `scripts/audit_s1b.py` | create | legacy, retrain and proxy reproduction → `s1b_audit.json` |
| `src/fbd/evaluate/uncertainty.py` | modify | `stratified_mean_difference()` |
| `src/fbd/evaluate/registration.py` | modify | `check_complete(..., year=2022)`, `check_params`, `year_statement`, `BACKTEST_VERDICT_TEXT` |
| `src/fbd/evaluate/backtest_stats.py` | create | primary, per-year, mean and by-month margins (needs sklearn) |
| `scripts/backtest.py` | create | guards → folds → statistics → `backtest.json` |
| `scripts/plot_backtest.py` | create | per-year and primary intervals on one axis |
| `docs/PREREGISTRATION_S1B.md` | create | the registration |
| `.gitignore` | modify | `data/processed/backtest/` |
| `tests/test_backtest_folds.py` | create | folds, paths, split labels, params hash |
| `tests/test_fold_leakage.py` | create | poison tests for fits 1–5; legacy formula equality |
| `tests/test_stratified_bootstrap.py` | create | the stratified bootstrap |
| `tests/test_backtest_guard.py` | create | registration additions, verdict text, per-year statement |
| `tests/test_backtest_stats.py` | create | statistics on synthetic folds (sklearn, skips in CI) |
| `tests/test_published_numbers.py` | modify | README and D-026 state the backtest interval |
| `.github/workflows/ci.yml` | modify | add the new test files |

(The spec's §10 put the stratified-bootstrap tests in `tests/test_uncertainty.py`; that file is not in the CI list, so they go in a new CI file instead.)

---

### Task 1: Fold table and the hyperparameter hash

**Files:**
- Create: `src/fbd/evaluate/folds.py`, `src/fbd/model/params.py`
- Modify: `src/fbd/model/train.py:112-131` (the `params = dict(...)` block in `BustModel.fit`)
- Test: `tests/test_backtest_folds.py`

**Interfaces:**
- Produces: `Fold(test: int, train: tuple[int, ...], val: tuple[int, ...])`; `FOLDS: dict[int, Fold]`; `PRIMARY_YEARS = (2019, 2020, 2021)`; `MODES = ("legacy", "strict")`; `FOLD_DIR: Path`; `regime_fit_years(fold, mode) -> tuple | None`; `fold_path(test_year, mode) -> Path`; `model_path(test_year, mode) -> Path`; `split_labels(years, train, val, test) -> np.ndarray` (values `"train"|"val"|"test"|"excluded"`). `DEFAULT_PARAMS: dict`; `params_sha256(params=DEFAULT_PARAMS) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/test_backtest_folds.py`

```python
"""S1b folds and hyperparameters: pure, so CI checks them without data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.model import params as PR  # noqa: E402


def test_the_2022_fold_is_the_published_split():
    f = F.FOLDS[2022]
    assert f.train == config.TRAIN_YEARS
    assert f.val == config.VAL_YEARS
    assert (f.test,) == config.TEST_YEARS


def test_folds_are_an_expanding_window():
    years = sorted(F.FOLDS)
    assert years == [2019, 2020, 2021, 2022]
    for a, b in zip(years, years[1:]):
        assert F.FOLDS[b].train == F.FOLDS[a].train + F.FOLDS[a].val


def test_every_fold_trains_then_calibrates_then_tests():
    for f in F.FOLDS.values():
        assert max(f.train) < min(f.val) <= max(f.val) < f.test
        assert not set(f.train) & set(f.val)


def test_primary_years_exclude_the_year_s1_has_seen():
    assert F.PRIMARY_YEARS == (2019, 2020, 2021)
    assert 2022 not in F.PRIMARY_YEARS


def test_no_fold_file_can_overwrite_the_frozen_dataset():
    frozen = (config.PROCESSED / "dataset.parquet").resolve()
    for year in F.FOLDS:
        for mode in F.MODES:
            for p in (F.fold_path(year, mode), F.model_path(year, mode)):
                assert p.resolve() != frozen
                assert p.parent == F.FOLD_DIR


def test_legacy_mode_fits_regimes_on_every_date_and_strict_on_training_years():
    f = F.FOLDS[2019]
    assert F.regime_fit_years(f, "legacy") is None
    assert F.regime_fit_years(f, "strict") == (2016, 2017)


def test_split_labels_reproduce_the_published_split():
    years = np.arange(2016, 2023)
    got = F.split_labels(years, config.TRAIN_YEARS, config.VAL_YEARS, config.TEST_YEARS)
    old = np.where(np.isin(years, config.TEST_YEARS), "test",
                   np.where(np.isin(years, config.VAL_YEARS), "val", "train"))
    assert got.tolist() == old.tolist()


def test_split_labels_exclude_years_after_the_test_year():
    f = F.FOLDS[2019]
    got = F.split_labels(np.arange(2016, 2023), f.train, f.val, (f.test,))
    assert got.tolist() == ["train", "train", "val", "test",
                            "excluded", "excluded", "excluded"]


def test_params_hash_is_stable_and_sensitive():
    assert PR.params_sha256() == PR.params_sha256(dict(PR.DEFAULT_PARAMS))
    changed = dict(PR.DEFAULT_PARAMS, max_depth=PR.DEFAULT_PARAMS["max_depth"] + 1)
    assert PR.params_sha256(changed) != PR.params_sha256()


def test_params_exclude_the_data_dependent_class_weight():
    assert "scale_pos_weight" not in PR.DEFAULT_PARAMS
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_backtest_folds.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'fbd.evaluate.folds'`.

- [ ] **Step 3: Implement** — `src/fbd/evaluate/folds.py`

```python
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
```

- [ ] **Step 4: Implement** — `src/fbd/model/params.py`

```python
"""The model's hyperparameters, in one pure place so S1b can register them.

``scale_pos_weight`` is not here: it is computed from each training set's
class balance (LOGIC.md sec 4.5), so it differs by fold by design.
"""
from __future__ import annotations

import hashlib
import json

from fbd import config

DEFAULT_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.75,
    min_child_weight=20,
    reg_lambda=2.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=config.RANDOM_SEED,
    n_jobs=0,
)


def params_sha256(params: dict = DEFAULT_PARAMS) -> str:
    blob = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
```

- [ ] **Step 5: Point `BustModel.fit` at it** — in `src/fbd/model/train.py` add `from fbd.model.params import DEFAULT_PARAMS` below `from fbd import config`, and replace the whole `params = dict(` … `)` block and the `params.update(overrides)` line with:

```python
        params = dict(DEFAULT_PARAMS)
        # Busts are rare; class weighting is mandated by LOGIC.md sec 4.5.
        params["scale_pos_weight"] = neg / pos
        params.update(overrides)
```

(The values are identical to the literal they replace; the audit in Task 4 proves the retrained model reproduces the published AUROC.)

- [ ] **Step 6: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_backtest_folds.py tests/ -q`
Expected: all pass (241 + 10).

- [ ] **Step 7: Commit**

```bash
git add src/fbd/evaluate/folds.py src/fbd/model/params.py src/fbd/model/train.py tests/test_backtest_folds.py
git commit -m "Add the S1b fold table and a registrable hyperparameter hash"
```

---

### Task 2: Every year-dependent fit takes its years

**Files:**
- Create: `src/fbd/features/standardise.py`
- Modify: `src/fbd/features/forecast.py` (`build`), `src/fbd/features/era5.py:136-146` (`national_daily`), `src/fbd/regime/classify.py:89-98,177-180` (`regime_scores`, `classify`)
- Test: `tests/test_fold_leakage.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `fit_mask(dates, years=None) -> np.ndarray[bool]`; `standardise(frame, cols, mask, suffix="_z") -> DataFrame`; `standardise_within(frame, cols, by, mask, suffix="_zl") -> DataFrame`; `forecast.build(pairs, labelled=None, train_years=None)`; `era5.national_daily(years=None, train_years=None)`; `classify.regime_scores(nat, loc, static, fit_years=None)`; `classify.classify(nat, loc, fit_years=None)`; `classify.LOCAL_FIELDS: list[str]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_fold_leakage.py`

```python
"""Poison tests: a fold's test years must never reach a fitted statistic.

Each test fits on early years, then sets the later years to absurd values and
fits again. If anything later leaks into the fit, the two fits differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.features import forecast as ffeat  # noqa: E402
from fbd.features import standardise as S  # noqa: E402
from fbd.labels import bust  # noqa: E402

TRAIN = (2016, 2017)


def _pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2016, 2023):
        for day in pd.date_range(f"{year}-06-01", f"{year}-07-30", freq="D"):
            for sub in ("A", "B"):
                for lead in (1, 2, 3):
                    rows.append(dict(subdivision_id=sub, init_date=day,
                                     lead_day=lead,
                                     valid_date=day + pd.Timedelta(days=lead - 1),
                                     fcst_rain_mm=float(rng.gamma(2.0, 8.0)),
                                     obs_rain_mm=float(rng.gamma(2.0, 8.0))))
    return pd.DataFrame(rows)


def _poison(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    late = out.valid_date.dt.year > max(TRAIN)
    for c in cols:
        out.loc[late, c] = 1e6
    return out


def test_bust_thresholds_ignore_later_years():
    p = _pairs()
    a = bust.label(p, train_years=TRAIN)
    b = bust.label(_poison(p, ["obs_rain_mm"]), train_years=TRAIN)
    pdt.assert_series_equal(a.effective_threshold, b.effective_threshold)


def test_forecast_climatology_ignores_later_years():
    p = _pairs()
    a = ffeat.climatology(p, TRAIN)
    b = ffeat.climatology(_poison(p, ["obs_rain_mm", "fcst_rain_mm"]), TRAIN)
    pdt.assert_frame_equal(a, b)


def test_climatological_bust_rate_ignores_later_years():
    lab = bust.label(_pairs(), train_years=TRAIN)
    poisoned = lab.copy()
    poisoned.loc[poisoned.valid_date.dt.year > max(TRAIN), "bust"] = 1
    pdt.assert_frame_equal(ffeat.climatological_bust_rate(lab, TRAIN),
                           ffeat.climatological_bust_rate(poisoned, TRAIN))


def test_forecast_build_passes_its_years_to_both_climatologies():
    p = _pairs()
    lab = bust.label(p, train_years=TRAIN)
    cols = ["clim_obs_mean", "clim_obs_p90", "clim_fcst_mean", "clim_bust_rate"]
    a = ffeat.build(p, labelled=lab, train_years=TRAIN)
    pp = _poison(p, ["obs_rain_mm"])
    lab2 = lab.copy()
    lab2.loc[lab2.valid_date.dt.year > max(TRAIN), "bust"] = 1
    b = ffeat.build(pp, labelled=lab2, train_years=TRAIN)
    pdt.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def _field_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2016-06-01", "2022-09-30", freq="7D")
    return pd.DataFrame({
        "date": np.repeat(dates, 2),
        "subdivision_id": np.tile(["A", "B"], len(dates)),
        "u850": rng.normal(5, 2, 2 * len(dates)),
        "tcwv": rng.normal(40, 5, 2 * len(dates)),
    })


def test_fit_mask_selects_years_and_none_means_every_row():
    d = pd.Series(pd.to_datetime(["2016-06-01", "2019-06-01", "2022-06-01"]))
    assert S.fit_mask(d, TRAIN).tolist() == [True, False, False]
    assert S.fit_mask(d, None).tolist() == [True, True, True]


def test_standardise_ignores_rows_outside_the_mask():
    f = _field_frame()
    m = S.fit_mask(f.date, TRAIN)
    a = S.standardise(f, ["u850", "tcwv"], m)
    g = f.copy()
    g.loc[~m, ["u850", "tcwv"]] = 1e6
    b = S.standardise(g, ["u850", "tcwv"], m)
    pdt.assert_frame_equal(a[m], b[m])


def test_standardise_is_the_published_national_formula():
    f = _field_frame()
    m = S.fit_mask(f.date, TRAIN)
    mu = f.loc[m, ["u850"]].mean()
    sd = f.loc[m, ["u850"]].std().replace(0, 1.0)
    expected = (f["u850"] - mu["u850"]) / sd["u850"]
    pdt.assert_series_equal(S.standardise(f, ["u850"], m)["u850_z"], expected,
                            check_names=False, check_exact=True)


def test_standardise_within_ignores_rows_outside_the_mask():
    f = _field_frame()
    m = S.fit_mask(f.date, TRAIN)
    a = S.standardise_within(f, ["u850"], "subdivision_id", m)
    g = f.copy()
    g.loc[~m, "u850"] = 1e6
    b = S.standardise_within(g, ["u850"], "subdivision_id", m)
    pdt.assert_frame_equal(a[m], b[m])


def test_standardise_within_every_row_is_the_legacy_regime_formula():
    """Legacy mode must reproduce dataset.parquet bit for bit."""
    f = _field_frame()
    legacy = f.groupby("subdivision_id")["u850"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    got = S.standardise_within(f, ["u850"], "subdivision_id", S.fit_mask(f.date, None))
    pdt.assert_series_equal(got["u850_zl"], legacy, check_names=False, check_exact=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_fold_leakage.py -q`
Expected: FAIL, `ImportError` on `fbd.features.standardise` (and `build()` has no `train_years`).

- [ ] **Step 3: Implement** — `src/fbd/features/standardise.py`

```python
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
```

- [ ] **Step 4: Thread the years through `forecast.build`** — in `src/fbd/features/forecast.py` change the signature and the two climatology calls:

```python
def build(pairs: pd.DataFrame, labelled: pd.DataFrame | None = None,
          train_years=None) -> pd.DataFrame:
```

```python
    out = out.merge(climatology(pairs, train_years), on=["subdivision_id", "month"], how="left")
```

```python
        cbr = climatological_bust_rate(labelled, train_years)
```

- [ ] **Step 5: `national_daily` through `standardise`** — in `src/fbd/features/era5.py` add `from fbd.features import standardise as S` to the imports, change the signature to `def national_daily(years=None, train_years=None) -> pd.DataFrame:`, and replace the four lines from `train_mask = …` through the `for c in cols: idx[f"{c}_z"] = …` loop with:

```python
    # Standardise against the training-year climatology only.
    mask = S.fit_mask(idx.date, train_years or config.TRAIN_YEARS)
    z = S.standardise(idx, cols, mask)
    for c in cols:
        idx[f"{c}_z"] = z[f"{c}_z"]
```

- [ ] **Step 6: Regimes through `standardise_within`** — in `src/fbd/regime/classify.py` add `from fbd.features import standardise as S`, add below `TEMPERATURE`:

```python
#: Local fields standardised within each subdivision before scoring.
LOCAL_FIELDS = ["moisture_flux_850", "wind_shear", "tcwv", "u850", "v850", "z500", "mslp"]
```

change `regime_scores` to take `fit_years=None` and replace its standardisation loop:

```python
def regime_scores(nat: pd.DataFrame, loc: pd.DataFrame, static: pd.DataFrame,
                  fit_years=None) -> pd.DataFrame:
    """Physically motivated score per regime, per (subdivision, date).

    ``fit_years=None`` standardises over every date, as the published pipeline
    did (test years included -- the leak S1b measures); a fold passes its
    training years.
    """
    df = loc.merge(nat, on="date", how="inner").merge(
        static, on="subdivision_id", how="left"
    )

    # Standardise local fields within subdivision, so "strong flow" means strong
    # *for that place* -- 10 m/s is routine in Konkan and extreme in Vidarbha.
    z = S.standardise_within(df, LOCAL_FIELDS, "subdivision_id",
                             S.fit_mask(df.date, fit_years))
    for c in LOCAL_FIELDS:
        df[f"{c}_zl"] = z[f"{c}_zl"]
```

and `classify`:

```python
def classify(nat: pd.DataFrame, loc: pd.DataFrame, fit_years=None) -> pd.DataFrame:
    """End-to-end: indices -> scores -> soft regime probabilities."""
    static = static_attributes()
    return softmax_probabilities(regime_scores(nat, loc, static, fit_years=fit_years))
```

Delete the now-unused `_z` function once `grep -nw "_z" src/fbd/regime/classify.py` shows only its definition (the old call was `transform(_z)`, which the replacement removes).

- [ ] **Step 7: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/ -q`
Expected: all pass (260: 241 + 10 + 9).

- [ ] **Step 8: Commit**

```bash
git add src/fbd/features/standardise.py src/fbd/features/forecast.py src/fbd/features/era5.py src/fbd/regime/classify.py tests/test_fold_leakage.py
git commit -m "Let every year-dependent fit take its years; poison-test each one"
```

---

### Task 3: `build_dataset.py` builds folds without touching the frozen dataset

**Files:**
- Modify: `scripts/build_dataset.py` (whole `attach_state_features` and `main`), `.gitignore`

**Interfaces:**
- Consumes: `F.FOLDS`, `F.MODES`, `F.fold_path`, `F.regime_fit_years`, `F.split_labels` (Task 1); `ffeat.build(..., train_years)`, `e5.national_daily(train_years=)`, `rg.classify(..., fit_years=)` (Task 2).
- Produces: `build(train_years=config.TRAIN_YEARS, val_years=config.VAL_YEARS, test_years=config.TEST_YEARS, regime_fit_years=None, require_state=False) -> DataFrame`; `build_fold(test_year: int, mode: str) -> DataFrame`; CLI `--fold YEAR --mode legacy|strict`.

- [ ] **Step 1: Replace `attach_state_features`** in `scripts/build_dataset.py`:

```python
def attach_state_features(feats: pd.DataFrame, train_years=None,
                          regime_fit_years=None, require: bool = False) -> pd.DataFrame:
    """Join ERA5 analysis state + regime probabilities onto the feature table.

    CAUSALITY: joined on **init_date**, never valid_date.  The analysis valid on
    the forecast's target day does not exist when the forecast is issued; using
    it would leak the answer and inflate every score.  See fbd.features.era5.

    ``require=True`` (every fold build) raises instead of skipping: a fold
    silently built without state features would train a different model.
    """
    try:
        from fbd.features import era5 as e5
        from fbd.regime import classify as rg
    except Exception as exc:  # noqa: BLE001
        if require:
            raise
        print(f"   [skip] ERA5 features unavailable: {exc}")
        return feats

    try:
        nat = e5.national_daily(train_years=train_years)
        loc = e5.subdivision_fields()
        regimes = rg.classify(nat, loc, fit_years=regime_fit_years)
        static = rg.static_attributes()
    except FileNotFoundError as exc:
        if require:
            raise
        print(f"   [skip] ERA5 cache incomplete: {exc}")
        return feats
```

(the rest of the function, from `print(f"   national indices: …` to `return feats`, is unchanged).

- [ ] **Step 2: Replace `main`** with `build`, `build_fold` and a new `main`. Add `import argparse` and `from fbd.evaluate import folds as F` to the imports.

```python
def build(train_years=config.TRAIN_YEARS, val_years=config.VAL_YEARS,
          test_years=config.TEST_YEARS, regime_fit_years=None,
          require_state: bool = False) -> pd.DataFrame:
    """The labelled dataset for one choice of years. Defaults = the published one."""
    print("1. pairing forecasts with IMD truth ...")
    pairs = hres.build_pairs()
    print(f"   {len(pairs):,} rows, {pairs.subdivision_id.nunique()} subdivisions, "
          f"{pairs.init_date.nunique()} init dates")

    print(f"2. labelling busts (thresholds fitted on {list(train_years)}) ...")
    labelled = bust.label(pairs, train_years=train_years)
    lab = labelled.dropna(subset=["bust"])
    print(f"   overall bust rate {lab.bust.mean():.3%}  "
          f"(high-impact {lab.bust_high_impact.mean():.3%})")

    print("3. building forecast-derived features ...")
    feats = ffeat.build(pairs, labelled=labelled, train_years=train_years)

    print("4. building ERA5 state + regime features ...")
    feats = attach_state_features(feats, train_years=train_years,
                                  regime_fit_years=regime_fit_years,
                                  require=require_state)

    keep = [
        "subdivision_id", "init_date", "lead_day", "valid_date",
        "fcst_rain_mm", "obs_rain_mm", "error", "abs_error",
        "fcst_category", "obs_category", "effective_threshold",
        "bust", "bust_high_impact", "high_impact", "bust_type",
    ]
    ds = labelled[keep].merge(
        feats.drop(columns=[c for c in feats.columns
                            if c in keep and c not in
                            ("subdivision_id", "init_date", "lead_day", "valid_date")]),
        on=["subdivision_id", "init_date", "lead_day", "valid_date"],
        how="left",
    )

    ds["year"] = ds.valid_date.dt.year
    ds["split"] = F.split_labels(ds.year, train_years, val_years, test_years)
    if (ds.split == "excluded").any():
        # Years after a fold's test year: never trained on, never scored.
        ds = ds[ds.split != "excluded"].reset_index(drop=True)
    return ds


def build_fold(test_year: int, mode: str) -> pd.DataFrame:
    fold = F.FOLDS[test_year]
    return build(fold.train, fold.val, (fold.test,),
                 regime_fit_years=F.regime_fit_years(fold, mode),
                 require_state=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, choices=sorted(F.FOLDS),
                    help="build one S1b fold instead of the published dataset")
    ap.add_argument("--mode", choices=F.MODES, default="strict")
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.fold is None:
        ds, out = build(), OUT
    else:
        ds, out = build_fold(args.fold, args.mode), F.fold_path(args.fold, args.mode)
        assert out.resolve() != OUT.resolve(), "a fold must never overwrite dataset.parquet"

    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    print(f"\nwrote {out}  rows={len(ds):,}  cols={ds.shape[1]}  "
          f"in {time.time()-t0:.0f}s")

    print("\nsplit summary:")
    print(
        ds.dropna(subset=["bust"])
        .groupby("split")
        .agg(n=("bust", "size"), bust_rate=("bust", "mean"),
             years=("year", lambda s: sorted(s.unique())))
        .to_string()
    )
    print("\nbust rate by lead day and split:")
    print(bust.summarise(ds).pivot(index="lead_day", columns="split",
                                   values="bust_rate").round(4).to_string())
    return 0
```

Also update the module docstring's `Output:` line to:

```
Output: data/processed/dataset.parquet  (one row per subdivision x init x lead)
        --fold YEAR --mode legacy|strict  ->  data/processed/backtest/fold_YEAR_MODE.parquet
```

- [ ] **Step 3: Ignore the fold outputs** — append to `.gitignore` under the data policy section:

```
# S1b fold datasets and models: rebuilt by scripts/backtest.py, hashes recorded
# in data/artifacts/backtest.json.
data/processed/backtest/
```

- [ ] **Step 4: Check the CLI parses without building anything**

Run: `PYTHONPATH=src python scripts/build_dataset.py --help`
Expected: usage text listing `--fold {2019,2020,2021,2022}` and `--mode {legacy,strict}`. **Do not run it without `--fold`.**

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_dataset.py .gitignore
git commit -m "Build S1b folds from build_dataset.py; the default output is unchanged"
```

---

### Task 4: Audit — the fold pipeline is the published pipeline

**Files:**
- Create: `scripts/audit_s1b.py`
- Create (artifact): `data/artifacts/s1b_audit.json`

**Interfaces:**
- Consumes: `BD.build_fold` (Task 3), `F.fold_path` (Task 1), `PR.params_sha256` (Task 1), `train_model.Calibrated`, `B.SpreadBaseline`, `T.BustModel`, `M.auroc`.
- Produces: `s1b_audit.json` with `checks` (each `check`, `ok`, values), `hashes` (`dataset_sha256`, `model_sha256`, `params_sha256`), and `ok` overall — read by Task 8.

- [ ] **Step 1: Implement** — `scripts/audit_s1b.py`

```python
"""S1b audit: prove the fold pipeline is the published pipeline, before registering.

    PYTHONPATH=src python scripts/audit_s1b.py

1. Rebuild fold 2022 in legacy mode; it must equal dataset.parquet exactly.
2. Retrain on it; decision-band AUROC within 0.001 of the frozen model's.
3. Model minus the lagged proxy within 0.001 of the published +0.0821.
Needs the raw data (data/raw). Writes data/artifacts/s1b_audit.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fbd import config  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.model import baselines as B  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import train as T  # noqa: E402

DATASET = config.PROCESSED / "dataset.parquet"
MODEL = config.ARTIFACTS / "bust_model.joblib"
OUT = config.ARTIFACTS / "s1b_audit.json"
TOL = 0.001


def first_difference(a: pd.DataFrame, b: pd.DataFrame):
    if list(a.columns) != list(b.columns):
        return f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return f"rows differ: {len(a):,} vs {len(b):,}"
    for c in a.columns:
        if not a[c].equals(b[c]):
            try:
                pd.testing.assert_series_equal(a[c], b[c], check_exact=True, check_names=False)
            except AssertionError as exc:
                return f"column {c}: {str(exc).splitlines()[0]}"
    return None


def main() -> int:
    import build_dataset as BD
    from train_model import Calibrated

    t0 = time.time()
    checks = []
    frozen = pd.read_parquet(DATASET)

    print("1. legacy rebuild of fold 2022 ...", flush=True)
    path = F.fold_path(2022, "legacy")
    path.parent.mkdir(parents=True, exist_ok=True)
    BD.build_fold(2022, "legacy").to_parquet(path, index=False)
    legacy = pd.read_parquet(path)  # same parquet round trip as the frozen file
    diff = first_difference(frozen, legacy)
    checks.append({"check": "legacy fold 2022 reproduces dataset.parquet", "ok": diff is None,
                   "detail": diff or f"{len(frozen):,} rows x {frozen.shape[1]} columns identical"})
    print(f"   {'OK' if diff is None else 'MISMATCH'}  {checks[-1]['detail']}", flush=True)

    if diff is None:
        print("2. retrain on the legacy fold ...", flush=True)
        tr, va, te = (d.dropna(subset=["bust"]) for d in T.split_frames(legacy))
        feats = T.available_features(legacy)
        model = T.BustModel().fit(tr, va, features=feats)
        frozen_model = T.BustModel.load(MODEL)
        band = te[te.lead_day.isin(config.DECISION_BAND)]
        auc_frozen = M.auroc(band.bust, frozen_model.predict_proba(band))
        auc_new = M.auroc(band.bust, model.predict_proba(band))
        same_feats = list(feats) == list(frozen_model.features)
        checks.append({"check": "retrained AUROC within 0.001 of the frozen model",
                       "ok": same_feats and abs(auc_new - auc_frozen) <= TOL,
                       "frozen": auc_frozen, "retrained": auc_new,
                       "difference": auc_new - auc_frozen,
                       "same_features": same_feats, "n_features": len(feats)})
        print(f"   frozen {auc_frozen:.4f}  retrained {auc_new:.4f}  "
              f"diff {auc_new - auc_frozen:+.5f}  features identical: {same_feats}", flush=True)

        print("3. lagged-proxy margin ...", flush=True)
        proxy = Calibrated(B.SpreadBaseline("lagged_spread"), "proxy").fit(tr, va).predict_proba(band)
        margin = auc_new - M.auroc(band.bust, proxy)
        ci = json.loads((config.ARTIFACTS / "confidence_intervals.json").read_text())
        published = ci["margins"][0]["point"]
        checks.append({"check": "model minus proxy within 0.001 of the published margin",
                       "ok": abs(margin - published) <= TOL,
                       "published": published, "reproduced": margin,
                       "difference": margin - published})
        print(f"   published {published:+.4f}  reproduced {margin:+.4f}", flush=True)

    payload = {
        "checks": checks,
        "ok": all(c["ok"] for c in checks) and len(checks) == 3,
        "hashes": {"dataset_sha256": P.sha256_file(DATASET),
                   "model_sha256": P.sha256_file(MODEL),
                   "params_sha256": PR.params_sha256()},
        "tolerance": TOL,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{'AUDIT PASSED' if payload['ok'] else 'AUDIT FAILED -- stop, do not register'}"
          f"  ({time.time() - t0:.0f}s)  wrote {OUT}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the audit** (background; builds and trains once)

Run: `PYTHONPATH=src python scripts/audit_s1b.py`
Expected: `AUDIT PASSED`, three OK lines, exit 0. **If check 1 fails, stop**: the refactor changed the pipeline; find the column named in the detail and fix the code, not the check. If check 2 or 3 is outside 0.001, stop and report the numbers.

- [ ] **Step 3: Confirm the frozen inputs are untouched**

Run: `git status --short data/processed/dataset.parquet data/artifacts/bust_model.joblib`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/audit_s1b.py data/artifacts/s1b_audit.json
git commit -m "Audit S1b: the legacy fold rebuild reproduces the published dataset and model"
```

---

### Task 5: Stratified bootstrap

**Files:**
- Modify: `src/fbd/evaluate/uncertainty.py` (append after `paired_difference`)
- Test: `tests/test_stratified_bootstrap.py`

**Interfaces:**
- Consumes: `_cluster_index`, `Interval`, `paired_difference`, `DEFAULT_N_BOOT`, `DEFAULT_SEED` (existing).
- Produces: `stratified_mean_difference(metric, y_true, prob_a, prob_b, cluster_ids, strata, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, alpha=0.05) -> Interval` where `n_clusters` is the total over strata.

- [ ] **Step 1: Write the failing tests** — `tests/test_stratified_bootstrap.py`

```python
"""The S1b interval: margins within each year, averaged; dates resampled within year."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import uncertainty as U  # noqa: E402


def mean_p(y, p):
    return float(np.mean(p))


def _data(seed=0):
    rng = np.random.default_rng(seed)
    strata = np.repeat([2019, 2020, 2021], [60, 45, 30])
    clusters = np.array([f"{s}-{i // 3}" for s, i in zip(strata, range(len(strata)))])
    y = rng.integers(0, 2, len(strata)).astype(float)
    a = rng.random(len(strata))
    b = rng.random(len(strata))
    return y, a, b, clusters, strata


def test_same_seed_same_interval():
    args = _data()
    one = U.stratified_mean_difference(mean_p, *args, n_boot=300, seed=7)
    two = U.stratified_mean_difference(mean_p, *args, n_boot=300, seed=7)
    assert (one.point, one.lo, one.hi) == (two.point, two.lo, two.hi)


def test_point_is_the_mean_of_per_stratum_differences():
    y, a, b, clusters, strata = _data()
    iv = U.stratified_mean_difference(mean_p, y, a, b, clusters, strata, n_boot=50)
    per = [a[strata == s].mean() - b[strata == s].mean() for s in (2019, 2020, 2021)]
    assert abs(iv.point - float(np.mean(per))) < 1e-12


def test_strata_are_averaged_not_pooled():
    """Constant per stratum, unequal sizes: averaging gives exactly 2 every time;
    pooling rows would give a row-weighted 1.667 and a non-zero width."""
    strata = np.repeat(["A", "B"], [30, 15])
    clusters = np.array([f"{s}{i // 3}" for s, i in zip(strata, range(45))])
    a = np.where(strata == "A", 1.0, 3.0)
    iv = U.stratified_mean_difference(mean_p, np.zeros(45), a, np.zeros(45),
                                      clusters, strata, n_boot=200)
    assert iv.point == 2.0 and iv.lo == 2.0 and iv.hi == 2.0


def test_one_stratum_is_the_ordinary_paired_bootstrap():
    y, a, b, clusters, _ = _data()
    one = np.zeros(len(y))
    s = U.stratified_mean_difference(mean_p, y, a, b, clusters, one, n_boot=300, seed=3)
    p = U.paired_difference(mean_p, y, a, b, clusters, n_boot=300, seed=3)
    assert (s.point, s.lo, s.hi) == (p.point, p.lo, p.hi)


def test_counts_clusters_across_strata_and_degenerate_resamples():
    y, a, b, clusters, strata = _data()
    iv = U.stratified_mean_difference(mean_p, y, a, b, clusters, strata, n_boot=20)
    assert iv.n_clusters == len(np.unique(clusters))

    def nan_metric(y, p):
        return float("nan")

    bad = U.stratified_mean_difference(nan_metric, y, a, b, clusters, strata, n_boot=20)
    assert bad.n_degenerate == 20 and np.isnan(bad.lo)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_stratified_bootstrap.py -q`
Expected: FAIL, `AttributeError: module 'fbd.evaluate.uncertainty' has no attribute 'stratified_mean_difference'`.

- [ ] **Step 3: Implement** — append to `src/fbd/evaluate/uncertainty.py`:

```python
def stratified_mean_difference(
    metric: Callable,
    y_true,
    prob_a,
    prob_b,
    cluster_ids,
    strata,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Mean over strata of ``metric(a) - metric(b)``; clusters resampled within strata.

    Built for the S1b backtest. Each test year has its own fold model and its
    own bust rate, so the margin is computed within a year and then averaged:
    one AUROC over rows pooled from several years would partly reward telling
    the years apart. Every resample draws, for each stratum, as many clusters as
    that stratum has, from that stratum only. With a single stratum this is
    exactly ``paired_difference``.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    clusters = np.asarray(cluster_ids)
    strata = np.asarray(strata)

    groups = []
    for s in sorted(np.unique(strata).tolist(), key=str):
        idx = np.flatnonzero(strata == s)
        groups.append([idx[g] for g in _cluster_index(clusters[idx])])

    def diff(rows: np.ndarray) -> float:
        return metric(y[rows], a[rows]) - metric(y[rows], b[rows])

    point = float(np.mean([diff(np.concatenate(g)) for g in groups]))
    rng = np.random.default_rng(seed)

    values: list[float] = []
    degenerate = 0
    for _ in range(n_boot):
        per = []
        for g in groups:
            picks = rng.integers(0, len(g), len(g))
            rows = np.concatenate([g[i] for i in picks])
            try:
                per.append(float(diff(rows)))
            except (ValueError, ZeroDivisionError):
                per.append(float("nan"))
        value = float(np.mean(per))
        if np.isfinite(value):
            values.append(value)
        else:
            degenerate += 1

    n_clusters = sum(len(g) for g in groups)
    if not values:
        return Interval(point, float("nan"), float("nan"), n_boot,
                        n_clusters, degenerate, alpha)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), n_boot, n_clusters,
                    degenerate, alpha)
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_stratified_bootstrap.py tests/test_uncertainty.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/fbd/evaluate/uncertainty.py tests/test_stratified_bootstrap.py
git commit -m "Add a stratified cluster bootstrap for margins averaged over years"
```

---

### Task 6: Registration additions and the S1b statistics

**Files:**
- Modify: `src/fbd/evaluate/registration.py`
- Create: `src/fbd/evaluate/backtest_stats.py`
- Test: `tests/test_backtest_guard.py`, `tests/test_backtest_stats.py`

**Interfaces:**
- Consumes: `U.paired_difference`, `U.stratified_mean_difference` (Task 5), `R.verdict` (existing).
- Produces: `R.check_complete(have, required, year=2022)`; `R.check_params(reg: dict, sha: str) -> None`; `R.year_statement(year: int, lo: float, hi: float, who: str = "ENS spread") -> str | None`; `R.BACKTEST_VERDICT_TEXT: dict[str, str]`. `BS.year_margin(rows, a, b, n_boot, seed) -> dict`; `BS.mean_margin(rows, a, b, years, n_boot, seed) -> dict`; `BS.primary(rows, years, n_boot, seed) -> dict` (adds `verdict`, `text`); `BS.per_year(rows, a, b, n_boot, seed, who="ENS spread") -> dict[str, dict]` (adds `statement`, `n_rows`, `n_init_dates`); `BS.by_month(rows, a, b, n_boot, seed) -> dict[str, dict[str, dict]]`. `rows` columns: `year, init_date, month, subdivision_id, lead_day, bust, p_model, p_ens, p_proxy`.

- [ ] **Step 1: Write the failing guard tests** — `tests/test_backtest_guard.py`

```python
"""S1b guards: dependency-free, run in CI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = Path(__file__).resolve().parents[1] / "docs" / "PREREGISTRATION_S1B.md"


def test_check_complete_names_the_year():
    R.check_complete(122, 122, year=2019)
    with pytest.raises(R.RegistrationError, match="for 2019 on disk"):
        R.check_complete(21, 122, year=2019)


def test_check_complete_keeps_the_s1_wording_by_default():
    with pytest.raises(R.RegistrationError, match="for 2022 on disk"):
        R.check_complete(41, 122)


def test_check_params_refuses_changed_hyperparameters():
    R.check_params({"params_sha256": PR.params_sha256()}, PR.params_sha256())
    with pytest.raises(R.RegistrationError, match="hyperparameters"):
        R.check_params({"params_sha256": "0" * 64}, PR.params_sha256())


@pytest.mark.parametrize("lo, hi, v", [(0.01, 0.05, "model_better"),
                                       (-0.05, -0.01, "ens_better"),
                                       (-0.01, 0.04, "indistinguishable")])
def test_backtest_verdict_text_covers_every_outcome(lo, hi, v):
    assert R.verdict(lo, hi) == v
    assert "2019–2021" in R.BACKTEST_VERDICT_TEXT[v]


def test_a_year_the_ensemble_wins_is_stated_plainly():
    assert R.year_statement(2020, -0.05, -0.01) == "ENS spread outranks the model in 2020"
    assert R.year_statement(2020, -0.05, 0.01) is None
    assert R.year_statement(2020, -0.05, -0.01, who="the lagged proxy") == \
        "the lagged proxy outranks the model in 2020"


@pytest.mark.skipif(not PREREG.exists(), reason="S1b not registered yet")
def test_the_s1b_registration_has_every_key():
    reg = R.parse_registration(PREREG.read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "model_sha256", "params_sha256", "seed", "n_boot",
                "n_boot_per_year", "primary_years", "lead_days", "alpha", "mode",
                *(f"required_ens_dates_{y}" for y in (2019, 2020, 2021, 2022))):
        assert key in reg, key
    assert reg["params_sha256"] == PR.params_sha256()
```

- [ ] **Step 2: Write the failing statistics tests** — `tests/test_backtest_stats.py`

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_backtest_guard.py tests/test_backtest_stats.py -q`
Expected: FAIL (`check_complete() got an unexpected keyword argument 'year'`; `No module named 'fbd.evaluate.backtest_stats'`).

- [ ] **Step 4: Implement the registration additions** — in `src/fbd/evaluate/registration.py` replace `check_complete` and append the rest:

```python
def check_complete(have: int, required: int, year: int = 2022) -> None:
    if have < required:
        raise RegistrationError(
            f"{have} ENS init dates for {year} on disk, {required} registered. "
            f"Re-run scripts/fetch_ens.py --year {year}; a partial season is never "
            "reported as the full one.")
```

```python
# ------------------------------------------------------------------ S1b
BACKTEST_VERDICT_TEXT = {
    "model_better": "the edge replicates in 2019–2021",
    "ens_better": "ENS spread outranks the model in 2019–2021",
    "indistinguishable": "the model is not distinguishable from ENS spread in 2019–2021",
}


def check_params(reg: dict, sha: str) -> None:
    if reg.get("params_sha256") != sha:
        raise RegistrationError(
            f"hyperparameters changed: registered {reg.get('params_sha256')}, "
            f"the code has {sha}")


def year_statement(year: int, lo: float, hi: float, who: str = "ENS spread"):
    """The registered per-year rule: a year the comparator wins is said plainly."""
    return f"{who} outranks the model in {year}" if hi < 0 else None
```

- [ ] **Step 5: Implement** — `src/fbd/evaluate/backtest_stats.py`

```python
"""The statistics registered in docs/PREREGISTRATION_S1B.md. Needs scikit-learn.

``rows`` holds every fold's scored test rows: year, init_date, month,
subdivision_id, lead_day, bust, p_model, p_ens, p_proxy.
"""
from __future__ import annotations

import pandas as pd

from fbd.evaluate import metrics as M
from fbd.evaluate import registration as R
from fbd.evaluate import uncertainty as U


def _interval(iv) -> dict:
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero}


def year_margin(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    return _interval(U.paired_difference(
        M.auroc, rows.bust.to_numpy(float), rows[a].to_numpy(float),
        rows[b].to_numpy(float), rows.init_date.to_numpy(), n_boot=n_boot, seed=seed))


def mean_margin(rows: pd.DataFrame, a: str, b: str, years, n_boot: int, seed: int) -> dict:
    r = rows[rows.year.isin(list(years))]
    iv = U.stratified_mean_difference(
        M.auroc, r.bust.to_numpy(float), r[a].to_numpy(float), r[b].to_numpy(float),
        r.init_date.to_numpy(), r.year.to_numpy(), n_boot=n_boot, seed=seed)
    return {**_interval(iv), "years": [int(y) for y in years]}


def primary(rows: pd.DataFrame, years, n_boot: int, seed: int) -> dict:
    out = mean_margin(rows, "p_model", "p_ens", years, n_boot, seed)
    v = R.verdict(out["lo"], out["hi"])
    return {**out, "verdict": v, "text": R.BACKTEST_VERDICT_TEXT[v]}


def per_year(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int,
             who: str = "ENS spread") -> dict:
    out = {}
    for year, r in rows.groupby("year"):
        iv = year_margin(r, a, b, n_boot, seed)
        iv["statement"] = R.year_statement(int(year), iv["lo"], iv["hi"], who=who)
        iv["n_rows"] = int(len(r))
        iv["n_init_dates"] = int(r.init_date.nunique())
        out[str(int(year))] = iv
    return out


def by_month(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    out = {}
    for year, r in rows.groupby("year"):
        out[str(int(year))] = {
            str(int(m)): year_margin(rm, a, b, n_boot, seed)
            for m, rm in r.groupby("month") if rm.bust.nunique() == 2}
    return out
```

- [ ] **Step 6: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/ -q`
Expected: all pass (the S1B-registration test skips until Task 8).

- [ ] **Step 7: Commit**

```bash
git add src/fbd/evaluate/registration.py src/fbd/evaluate/backtest_stats.py tests/test_backtest_guard.py tests/test_backtest_stats.py
git commit -m "Add the S1b guards, verdict text, per-year rule and statistics"
```

---

### Task 7: `backtest.py` — the registered analysis, and its figure

**Files:**
- Create: `scripts/backtest.py`, `scripts/plot_backtest.py`

**Interfaces:**
- Consumes: everything above; `E.load_ens`, `E.dates_by_year`, `E.comparison_rows`, `S.choose_comparator` (`fbd.evaluate.settle`), `Calibrated` (`scripts/train_model.py`).
- Produces: `data/artifacts/backtest.json` with keys `registration_sha256`, `params_sha256`, `hashes`, `n_ens_dates`, `row_rule`, `folds` (per year: `train`, `val`, `comparator`, `n_features`, `dataset_sha256`, `model_sha256`, `n_rows`, `n_init_dates`, `model_auroc`, `ens_auroc`), `primary`, `per_year`, `secondary.proxy.{mean, per_year}`, `exploratory.{by_month, regime_leak_2022, n_boot, note}`.

- [ ] **Step 1: Implement** — `scripts/backtest.py`

```python
"""Run exactly the S1b backtest registered in docs/PREREGISTRATION_S1B.md.

Refuses unless the registration is committed as-is, the frozen dataset, model
and hyperparameters match their registered hashes, and all four test years'
ENS seasons are complete. Then rebuilds four strict folds, trains one model per
fold, scores each test year, and writes data/artifacts/backtest.json.

    PYTHONPATH=src python scripts/backtest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S1B.md"
OUT = config.ARTIFACTS / "backtest.json"
DATASET = config.PROCESSED / "dataset.parquet"
MODEL = config.ARTIFACTS / "bust_model.joblib"
YEARS = tuple(sorted(F.FOLDS))
KEYS = ["init_date", "subdivision_id", "lead_day"]


def guards():
    reg = R.guard(PREREG, {"dataset_sha256": DATASET, "model_sha256": MODEL}, repo=ROOT)
    R.check_params(reg, PR.params_sha256())
    if reg.get("mode") != "strict":
        raise R.RegistrationError(f"registered mode is {reg.get('mode')!r}, not 'strict'")
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in YEARS:
        R.check_complete(have.get(y, 0), int(reg[f"required_ens_dates_{y}"]), year=y)
    return reg, ens


def score_fold(test_year: int, ens: pd.DataFrame, lead_days) -> tuple[pd.DataFrame, dict]:
    import build_dataset as BD
    from train_model import Calibrated
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.model import baselines as B
    from fbd.model import train as T

    fold = F.FOLDS[test_year]
    path = F.fold_path(test_year, "strict")
    path.parent.mkdir(parents=True, exist_ok=True)
    BD.build_fold(test_year, "strict").to_parquet(path, index=False)
    ds = pd.read_parquet(path)

    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = T.available_features(ds)
    frozen_feats = T.BustModel.load(MODEL).features
    if list(feats) != list(frozen_feats):
        raise RuntimeError(f"fold {test_year} has {len(feats)} features, the frozen "
                           f"model {len(frozen_feats)}: a fold must not lose features")
    model = T.BustModel().fit(tr, va, features=feats)
    mpath = F.model_path(test_year, "strict")
    model.save(mpath)

    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    name, comp = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                     test.ens_spread_rel.to_numpy(float))
    proxy = Calibrated(B.SpreadBaseline("lagged_spread"), "proxy").fit(tr, va).predict_proba(test)
    init = pd.to_datetime(test.init_date)
    rows = pd.DataFrame({
        "year": test_year,
        "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(),
        "subdivision_id": test.subdivision_id.to_numpy(),
        "lead_day": test.lead_day.to_numpy(),
        "bust": y,
        "p_model": model.predict_proba(test),
        "p_ens": comp,
        "p_proxy": proxy,
    })
    info = {"train": list(fold.train), "val": list(fold.val), "comparator": name,
            "n_features": len(feats), "dataset_sha256": P.sha256_file(path),
            "model_sha256": P.sha256_file(mpath), "n_rows": int(len(rows)),
            "n_init_dates": int(rows.init_date.nunique()),
            "model_auroc": M.auroc(y, rows.p_model), "ens_auroc": M.auroc(y, comp)}
    return rows, info


def regime_leak(ens, lead_days, strict_2022: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Exploratory: strict fold-2022 model minus the frozen model, on the same 2022 rows."""
    from fbd.evaluate import metrics as M
    from fbd.evaluate import uncertainty as U
    from fbd.model import train as T

    test, _ = E.comparison_rows(pd.read_parquet(DATASET), ens, lead_days=lead_days)
    frozen = pd.DataFrame({
        "init_date": pd.to_datetime(test.init_date).dt.strftime("%Y-%m-%d").to_numpy(),
        "subdivision_id": test.subdivision_id.to_numpy(),
        "lead_day": test.lead_day.to_numpy(),
        "bust_frozen": test.bust.to_numpy(float),
        "p_frozen": T.BustModel.load(MODEL).predict_proba(test),
    })
    m = strict_2022.merge(frozen, on=KEYS, how="inner")
    if len(m) != len(strict_2022) or not (m.bust == m.bust_frozen).all():
        raise RuntimeError("fold-2022 rows or labels differ from dataset.parquet")
    iv = U.paired_difference(M.auroc, m.bust.to_numpy(float), m.p_model.to_numpy(float),
                             m.p_frozen.to_numpy(float), m.init_date.to_numpy(),
                             n_boot=n_boot, seed=seed)
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero,
            "note": ("strict fold-2022 model minus the frozen model on identical 2022 "
                     "rows: the regime-standardisation leak, plus retraining noise "
                     "bounded by the audit's 0.001")}


def main() -> int:
    t0 = time.time()
    try:
        reg, ens = guards()
    except (R.RegistrationError, FileNotFoundError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]
    primary_years = [int(x) for x in reg["primary_years"].split(",")]

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in YEARS:
        print(f"fold {y}: build, train, score ...", flush=True)
        rows, info = score_fold(y, ens, lead_days)
        frames.append(rows)
        folds[str(y)] = info
        print(f"  {info['n_rows']:,} rows over {info['n_init_dates']} dates; "
              f"model {info['model_auroc']:.4f}, ENS ({info['comparator']}) "
              f"{info['ens_auroc']:.4f}", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.primary(rows, primary_years, n_boot, seed)
    print(f"primary, mean over {primary_years}: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    per_year = BS.per_year(rows, "p_model", "p_ens", n_year, seed)
    for y, iv in per_year.items():
        print(f"  {y}: {iv['point']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]"
              + (f"  -> {iv['statement']}" if iv["statement"] else ""), flush=True)

    print("secondary: lagged proxy ...", flush=True)
    secondary = {"proxy": {
        "mean": BS.mean_margin(rows, "p_model", "p_proxy", primary_years, n_boot, seed),
        "per_year": BS.per_year(rows, "p_model", "p_proxy", n_year, seed,
                                who="the lagged proxy")}}
    print("exploratory ...", flush=True)
    exploratory = {
        "by_month": BS.by_month(rows, "p_model", "p_ens", n_year, seed),
        "regime_leak_2022": regime_leak(ens, lead_days, rows[rows.year == 2022],
                                        n_year, seed),
        "n_boot": n_year,
        "note": "exploratory: no claims are drawn from these",
    }

    payload = {
        "registration_sha256": P.sha256_file(PREREG),
        "params_sha256": PR.params_sha256(),
        "hashes": {"dataset_sha256": P.sha256_file(DATASET),
                   "model_sha256": P.sha256_file(MODEL)},
        "n_ens_dates": E.dates_by_year(ens),
        "row_rule": ("per fold: comparison_rows() on that fold's dataset, Day 3-7 test "
                     "rows with a label and ENS spread; the fold model scores every row, "
                     "including ones the served product would refuse"),
        "folds": folds,
        "primary": prim, "per_year": per_year,
        "secondary": secondary, "exploratory": exploratory,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Implement** — `scripts/plot_backtest.py`

```python
"""S1b: each test year's margin over ENS spread and the registered mean, zero marked.

    PYTHONPATH=src python scripts/plot_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402

INK, MUTED, ACCENT = "#1f2933", "#8b98a5", "#2a6fb0"


def main() -> int:
    s = json.loads((config.ARTIFACTS / "backtest.json").read_text())
    p, per, proxy = s["primary"], s["per_year"], s["secondary"]["proxy"]
    rows = [("PRIMARY  mean 2019–2021, model − ENS", p, ACCENT, True)]
    for y in sorted(per, key=int):
        seen = y == "2022"
        rows.append((f"{y}: model − ENS" + ("  (seen in S1)" if seen else ""),
                     per[y], MUTED if seen else INK, False))
    rows.append(("secondary: mean 2019–2021, model − lagged proxy", proxy["mean"], MUTED, False))

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    left = min(r[1]["lo"] for r in rows)
    for i, (label, iv, colour, bold) in enumerate(rows):
        y = len(rows) - i
        ax.plot([iv["lo"], iv["hi"]], [y, y], color=colour, lw=3 if bold else 1.8)
        ax.plot(iv["point"], y, "o", color=colour, ms=7 if bold else 5)
        ax.text(left - 0.005, y, label, ha="right", va="center", fontsize=9,
                color=colour, fontweight="bold" if bold else "normal")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("ΔAUROC, 95% cluster-bootstrap interval over init dates")
    ax.set_title(f"S1b: {p['text']}", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "backtest"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}.png/.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify it refuses before registration**

Run: `PYTHONPATH=src python scripts/backtest.py`
Expected: `REFUSED: PREREGISTRATION_S1B.md is not committed as-is. …`, exit code 2, nothing built.

- [ ] **Step 4: Run the tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/ -q` → all pass.

```bash
git add scripts/backtest.py scripts/plot_backtest.py
git commit -m "Add the S1b analysis script and its figure; it refuses until registered"
```

---

### Task 8: Register, then push — before any 2019–2020 data

**Files:**
- Create: `docs/PREREGISTRATION_S1B.md`
- Modify: `.github/workflows/ci.yml` (test list)

- [ ] **Step 1: Write** `docs/PREREGISTRATION_S1B.md`. Required contents, in order:

1. The question and why (spec §1), with the S1 result it extends.
2. **Reproduced before registration:** the three checks from `data/artifacts/s1b_audit.json` as a table (check, published/frozen, reproduced, difference, OK), stating the exact retrain difference even when within tolerance.
3. Folds (spec §3 table), modes, frozen hyperparameters, "every test row scored".
4. Primary (spec §4.1 table, verbatim values), the per-year rule, secondary (§4.2), exploratory (§4.3), out of scope (§4.4).
5. Blinding (spec §8, verbatim).
6. Consequences (spec §5 table).
7. What would invalidate this: changing the dataset, model, hyperparameters, this file or the analysis code after the 2019–2020 data is on disk; reporting the primary with any test year short of 122 ENS dates; promoting a per-year, secondary or exploratory result to the headline.
8. The block, hashes copied from `s1b_audit.json`:

````markdown
```registration
dataset_sha256: <hashes.dataset_sha256 from data/artifacts/s1b_audit.json>
model_sha256: <hashes.model_sha256 from data/artifacts/s1b_audit.json>
params_sha256: <hashes.params_sha256 from data/artifacts/s1b_audit.json>
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
primary_years: 2019,2020,2021
required_ens_dates_2019: 122
required_ens_dates_2020: 122
required_ens_dates_2021: 122
required_ens_dates_2022: 122
lead_days: 3,4,5,6,7
alpha: 0.05
mode: strict
```
````

- [ ] **Step 2: Add the new tests to CI** — extend the `pytest -q` list in `.github/workflows/ci.yml`:

```yaml
          tests/test_backtest_folds.py
          tests/test_fold_leakage.py
          tests/test_stratified_bootstrap.py
          tests/test_backtest_guard.py
          tests/test_backtest_stats.py
```

- [ ] **Step 3: Commit and prove the guard passes its first checks**

```bash
git add docs/PREREGISTRATION_S1B.md .github/workflows/ci.yml
git commit -m "Register S1b: the backtest, its code and its frozen inputs, before the data"
PYTHONPATH=src python -m pytest tests/test_backtest_guard.py -q
PYTHONPATH=src python scripts/backtest.py
```

Expected: guard tests all pass (the registration-keys test now runs); the script prints `REFUSED: 21 ENS init dates for 2019 on disk, 122 registered. …` — proving the registration parses, is committed, and every hash matches.

- [ ] **Step 4: Push and watch CI**

```bash
git push origin local-llm-and-image-slimming
gh run list --branch local-llm-and-image-slimming --limit 1
gh run watch <id> --exit-status
```

Expected: 5/5 green. **No fetch before this is green.**

---

### Task 9: Fetch 2019 and 2020

**Files:** data only (`data/raw/wb2/ens/shards/…`, gitignored).

- [ ] **Step 1: Fetch 2019** (background, ≈101 dates, ≈20 s each)

Run: `PYTHONPATH=src python scripts/fetch_ens.py --year 2019`, log to the scratchpad. Report the measured rate after 5 dates. Expected end: `2019: 122/122 dates on disk`, exit 0. On an interruption, re-run the same command (it resumes).

- [ ] **Step 2: Fetch 2020** — `PYTHONPATH=src python scripts/fetch_ens.py --year 2020`. Expected: `2020: 122/122 dates on disk`.

- [ ] **Step 3: Loader sanity**

Run: `PYTHONPATH=src python -c "from fbd.evaluate import ens as E; print(E.dates_by_year(E.load_ens()))"`
Expected: `{2019: 122, 2020: 122, 2021: 122, 2022: 122}`, no `ConflictingDuplicate`. Check no `.tmp` files remain: `find data/raw/wb2/ens/shards -name '*.tmp*' | wc -l` → `0`.

---

### Task 10: Run the registered backtest once

**Files:**
- Create (artifacts): `data/artifacts/backtest.json`, `docs/figures/backtest.png`, `docs/figures/backtest.svg`

- [ ] **Step 1: Run** (background, so a tool timeout cannot cut it)

Run: `PYTHONPATH=src python scripts/backtest.py`, logging to the scratchpad.
Expected: four `fold …` lines with feature counts matching the frozen model, the primary line with its verdict, four per-year lines, exit 0, `wrote …backtest.json`. **Do not re-run with any changed parameter.** If it crashes before writing the JSON, fix only the crash (not the analysis), commit the fix with a message naming it, and re-run; the JSON must come from one complete run.

- [ ] **Step 2: Plot** — `PYTHONPATH=src python scripts/plot_backtest.py`; open the PNG and check labels do not overlap intervals.

- [ ] **Step 3: Frozen inputs untouched** — `git status --short data/processed/dataset.parquet data/artifacts/bust_model.joblib` → no output.

- [ ] **Step 4: Commit**

```bash
git add data/artifacts/backtest.json docs/figures/backtest.png docs/figures/backtest.svg
git commit -m "Run S1b: <primary text from backtest.json>"
```

---

### Task 11: State the result, the way the registration committed to

**Files:**
- Modify: `DECISIONS.md` (append D-026), `README.md`, `FRONTEND_LOGIC.md` §8, `docs/FIGURES.md`, `tests/test_published_numbers.py`, `HANDOFF.md` (ENS bullet)

- [ ] **Step 1: Extend the consistency test** — append to `tests/test_published_numbers.py`:

```python
BACKTEST = config.ARTIFACTS / "backtest.json"


@pytest.mark.skipif(not BACKTEST.exists(), reason="S1b not run yet")
@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_backtest_interval(doc):
    p = json.loads(BACKTEST.read_text())["primary"]
    s = f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert s in text, f"{doc} does not state the S1b interval {s}"
```

Run: `PYTHONPATH=src python -m pytest tests/test_published_numbers.py -q` → Expected: the two new cases FAIL.

- [ ] **Step 2: Apply the consequences row for the recorded verdict** (spec §5), writing the primary interval exactly as the test formats it:

- **DECISIONS.md** — append `## D-026 — S1b: <primary text> — LOCKED|DISCLOSED` (LOCKED only for `model_better`) with: the registration commit (`git log -1 --format=%h -- docs/PREREGISTRATION_S1B.md`); the audit table; the fetch summary; a per-fold table (train, val, rows, dates, comparator, model AUROC, ENS AUROC); the primary; every per-year interval and any per-year statement; the proxy secondary; the by-month exploratory table and whether S1's June–July pattern recurs; the regime-leak size; the consequence for the claim and for S2–S5.
- **README.md** — add a subsection `### Does it hold in other years?` directly after the true-ensemble paragraphs in "Headline result", containing the primary sentence for the verdict, the interval, the per-year intervals as a four-row table (2022 marked "seen in S1"), a link to `docs/PREREGISTRATION_S1B.md` and `docs/figures/backtest.png`. Then apply the verdict row:
  - `model_better`: in "The margin is not spread evenly…" paragraph, replace "It is still one season: whether the edge holds in other years is open." with "It is not one season: on average over 2019–2021 the margin is <interval> (D-026)."
  - `indistinguishable`: replace that sentence with "Outside 2022 it is not established: on average over 2019–2021 the margin is <interval>, an interval that contains zero (D-026)." and add "in 2022" to the bold claim sentence ("…over the full held-out season **of 2022**").
  - `ens_better`: as `indistinguishable`, but "…the ensemble outranks the model on average over 2019–2021: <interval> (D-026)."; and the "The honest reading." paragraph states the claim is 2022-specific.
- **FRONTEND_LOGIC.md §8** — `model_better`: change the may-claim bullet's "always as *one season*" to "in 2022 and on average over 2019–2021 (D-026)" and delete the may-not-claim bullet's "It is one season" clause. Otherwise: add to may-not-claim "That the edge holds outside 2022 (D-026: <interval> over 2019–2021)."
- **docs/FIGURES.md** — add `## 4. The backtest — backtest.png` describing each row, which is primary, that 2022 is drawn muted as "seen in S1", and the honest reading in two sentences.
- **HANDOFF.md** — append one sentence to the ENS bullet stating the S1b result with its interval.

- [ ] **Step 3: Run everything**

Run: `PYTHONPATH=src python -m pytest tests/ -q` → Expected: all pass, including the new cases.

- [ ] **Step 4: Commit**

```bash
git add DECISIONS.md README.md FRONTEND_LOGIC.md docs/FIGURES.md HANDOFF.md tests/test_published_numbers.py
git commit -m "Record D-026 and state the S1b result wherever the ENS claim is made"
```

---

### Task 12: Final audit and push

- [ ] **Step 1: Re-run the S1 audit and the S1b audit** — `PYTHONPATH=src python scripts/audit_s1.py` and `PYTHONPATH=src python scripts/audit_s1b.py` → both still pass; `git diff --stat data/artifacts/s1_audit.json data/artifacts/s1b_audit.json` shows no change.
- [ ] **Step 2: Cross-check** every place the S1b result appears: `backtest.json` vs README vs D-026 vs FRONTEND_LOGIC §8 vs the figure title vs HANDOFF.md — same interval to 4 decimals, same verdict wording, same per-year statements.
- [ ] **Step 3: Frozen inputs** — `git diff --stat b14c2b1 -- data/artifacts/bust_model.joblib data/processed/dataset.parquet` → empty.
- [ ] **Step 4: Every gate locally** — full pytest; the air-gap origin scan and socket-import check from `ci.yml`; `python -m bandit -r src/ -ll` → no issues; secret scan → none.
- [ ] **Step 5: Fix every discrepancy found**, with a test where one can guard it; commit each as `Fix <discrepancy> found in the S1b audit`.
- [ ] **Step 6: Push and watch CI** — `git push`, `gh run watch <id> --exit-status` → 5/5 green.
- [ ] **Step 7: Report** — the verdict, each year, the proxy secondary, the regime-leak size, the audit findings and anything left open.

---

## Self-review

- **Spec coverage:** §2 fits 1–7 → Task 2 (1–5), Task 7 (6 per fold, 7 proxy); §3 folds and modes → Tasks 1, 3; §4.1 primary and per-year rule → Tasks 5, 6, 7; §4.2 proxy → Task 7; §4.3 by-month and regime leak → Task 7; §5 consequences → Task 11; §6 pipeline changes → Tasks 1–3, 5, 7; §7 audit → Task 4; §8 blinding → Task 8; §9 outputs → Tasks 7, 10, 11; §10 tests → Tasks 1, 2, 5, 6 (stratified tests in their own CI file, noted above); §11 failure handling → Tasks 3 (`require_state`), 7 (feature check, refusal), 9; §12 commit order → Tasks 1–3, 4, 5–7, 8, 9, 10, 11–12.
- **Placeholders:** the only fill-ins are values produced by earlier steps (hashes from `s1b_audit.json`, the verdict and intervals from `backtest.json`), each with its exact source.
- **Consistency:** `split_labels(years, train, val, test)`, `regime_fit_years(fold, mode)`, `build_fold(test_year, mode)`, `standardise_within(frame, cols, by, mask)`, `stratified_mean_difference(metric, y, a, b, clusters, strata, …)`, `check_complete(have, required, year)`, `year_statement(year, lo, hi, who)` and the `rows` columns are used with the same names and signatures in every task.
