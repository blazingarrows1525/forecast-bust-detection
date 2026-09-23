# S1 — Settle the ENS Question — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, with a test registered before the data exists, whether the model outranks true 50-member IFS ENS spread over the full 2022 held-out season, and make every surface that states the answer read it from data.

**Architecture:** One shared loader and one shared row rule (`fbd/evaluate/ens.py`) replace per-script globbing. A resumable, sharded fetch (`fbd/ingest/ens_fetch.py` + `scripts/fetch_ens.py`) pulls the rest of 2022 and all of 2021. A dependency-free registration module (`fbd/evaluate/registration.py`) enforces the frozen inputs; `scripts/settle_ens.py` runs exactly the registered analysis via `fbd/evaluate/settle.py`. `/api/metrics` serves the result; the landing page renders it.

**Tech Stack:** Python 3.10/3.11, pandas, numpy, scikit-learn (local only), xarray + gcsfs (fetch, local only), FastAPI, vanilla JS, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-s1-settle-ens-design.md`

## Global Constraints

- Commit messages carry **no** `Co-Authored-By` or Claude attribution lines (standing user instruction).
- No SIH / SIH26079 on any product surface.
- Bootstrap seed **20260919**; primary resamples **10,000**; audit reproduction resamples **2,000**; alpha **0.05**; decision band **Day 3–7** (`config.DECISION_BAND`).
- ENS store unchanged: `config.ENS_STORE` (1.5°, 2018–2022). ≈123 MB per init date, 6 chunks.
- `data/artifacts/bust_model.joblib` and `data/processed/dataset.parquet` are **never modified** during S1.
- CI installs only `requirements-dev.txt` (numpy, pandas, pyarrow, fastapi, pytest, httpx2): **no scikit-learn, xgboost, matplotlib, geopandas, xarray, gcsfs.** Any module imported by a CI test must be pure. Tests needing sklearn use `pytest.importorskip("sklearn")`.
- The 192 existing tests keep passing.

## File Structure

| file | status | responsibility |
|---|---|---|
| `src/fbd/evaluate/ens.py` | create | load legacy + shard ENS, dedupe, per-year counts, the shared row rule |
| `src/fbd/ingest/ens_fetch.py` | create | pure fetch helpers: shard paths, done-dates, to-fetch, atomic write, member reduction, retries |
| `src/fbd/evaluate/provenance.py` | create | SHA-256 of a file; is a path committed in git |
| `src/fbd/evaluate/registration.py` | create | parse the registration block, verify hashes, completeness, verdict mapping, one `guard()` |
| `src/fbd/evaluate/settle.py` | create | the registered statistics (needs sklearn) |
| `scripts/fetch_ens.py` | rewrite | resumable sharded fetch + `--verify-legacy` determinism check |
| `scripts/audit_s1.py` | create | reproduce the 40-date margin and published numbers; hashes; discrepancies |
| `scripts/settle_ens.py` | create | guards, then the registered analysis → `ens_settlement.json` |
| `scripts/plot_ens_settlement.py` | create | forest plot of every interval |
| `scripts/compute_confidence_intervals.py` | modify | `ens_margin` uses the shared loader and row rule |
| `scripts/evaluate_ens_baseline.py` | modify | `load_ens` uses the shared loader |
| `src/fbd/api/app.py` | modify | `/api/metrics` adds `confidence_intervals`, `ens_settlement`, `refusal` |
| `web/landing.html` | modify | ENS sentence, stat cards, refusal chart from `/api/metrics` |
| `docs/PREREGISTRATION_S1.md` | create | the registration (after the audit, before the fetch) |
| `tests/test_ens_loader.py` | create | loader + row rule |
| `tests/test_fetch_ens_resume.py` | create | fetch helpers |
| `tests/test_settle_ens.py` | create | provenance, registration, verdict, settle statistics |
| `tests/test_metrics_endpoint.py` | create | `/api/metrics` blocks present / null |
| `tests/test_published_numbers.py` | create | README and D-025 agree with `ens_settlement.json` |
| `tests/test_web_pages.py` | modify | landing reads numbers from the API; can state every verdict |
| `.github/workflows/ci.yml` | modify | add the new test files |

---

### Task 1: Shared ENS loader and row rule

**Files:**
- Create: `src/fbd/evaluate/ens.py`
- Modify: `scripts/compute_confidence_intervals.py` (`ens_margin`, lines 104–137)
- Modify: `scripts/evaluate_ens_baseline.py` (`load_ens`, lines 36–43)
- Test: `tests/test_ens_loader.py`

**Interfaces:**
- Produces: `ENS_DIR: Path`, `KEY: list[str]`, `VALUES: list[str]`, `class ConflictingDuplicate(ValueError)`, `legacy_paths(ens_dir=ENS_DIR) -> list[Path]`, `shard_paths(ens_dir=ENS_DIR, year: int | None = None) -> list[Path]`, `dedupe(df) -> DataFrame`, `load_ens(ens_dir=ENS_DIR, include_shards: bool = True) -> DataFrame`, `dates_by_year(df) -> dict[int, int]`, `comparison_rows(ds, ens, lead_days=config.DECISION_BAND) -> tuple[DataFrame, DataFrame]` returning `(test, fit)`.
- `ens_margin(n_boot: int, ens: pd.DataFrame | None = None) -> dict | None` (signature extended; default behaviour unchanged).

- [ ] **Step 1: Write the failing tests** — `tests/test_ens_loader.py`

```python
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
        {"init_date": pd.Timestamp(d), "subdivision_id": s, "lead_day": l,
         "ens_spread": spread, "ens_mean": 2.0}
        for d in dates for s in subs for l in leads
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
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_ens_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fbd.evaluate.ens'`

- [ ] **Step 3: Implement** — `src/fbd/evaluate/ens.py`

```python
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
```

- [ ] **Step 4: Point the two scripts at it**

In `scripts/compute_confidence_intervals.py`, add `from fbd.evaluate import ens as E  # noqa: E402` beside the other `fbd` imports and replace the body of `ens_margin` from its first line down to the `if test.empty or fit.empty:` check with:

```python
def ens_margin(n_boot: int, ens: pd.DataFrame | None = None) -> dict | None:
    """Paired interval on model vs true IFS ENS spread, on rows where both exist.

    Loading and the row rule live in fbd.evaluate.ens so this, the S1 audit and
    the registered settlement cannot disagree about which rows count. Pass
    ``ens`` to score a specific subset (the audit passes legacy files only).
    """
    if ens is None:
        try:
            ens = E.load_ens()
        except FileNotFoundError:
            print("  [skip] no ENS data locally; run scripts/fetch_ens.py")
            return None

    from fbd.model import train as T

    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    test, fit = E.comparison_rows(ds, ens)
    if test.empty or fit.empty:
```

(The rest of the function — model load, comparator choice, calibrated fit, intervals — is unchanged.)

In `scripts/evaluate_ens_baseline.py`, replace `load_ens()` with:

```python
def load_ens() -> pd.DataFrame:
    """Shared loader: legacy files + shards, each forecast once (fbd.evaluate.ens)."""
    from fbd.evaluate import ens as E
    df = E.load_ens(ENS_DIR)
    print(f"loaded {len(df):,} ENS rows; dates per year {E.dates_by_year(df)}")
    return df
```

- [ ] **Step 5: Run the tests and the unchanged reproduction**

Run: `PYTHONPATH=src python -m pytest tests/test_ens_loader.py -q` → Expected: 7 passed.
Run: `PYTHONPATH=src python scripts/compute_confidence_intervals.py` → Expected: the ENS block prints `+0.0251 [-0.0083, +0.0580]` on 6,732 rows / 40 dates, exactly as before (only legacy files exist yet).
Run: `git diff --stat data/artifacts/confidence_intervals.json` → Expected: no change in any number (formatting-identical rewrite, or no diff).

- [ ] **Step 6: Commit**

```bash
git add src/fbd/evaluate/ens.py tests/test_ens_loader.py scripts/compute_confidence_intervals.py scripts/evaluate_ens_baseline.py
git commit -m "Share one ENS loader and one row rule across every comparison"
```

---

### Task 2: Resumable, sharded fetch

**Files:**
- Create: `src/fbd/ingest/ens_fetch.py`
- Rewrite: `scripts/fetch_ens.py`
- Test: `tests/test_fetch_ens_resume.py`

**Interfaces:**
- Consumes: `fbd.evaluate.ens.legacy_paths`, `E.ATOL`, `E.RTOL`.
- Produces: `EXPECTED_MEMBERS = 50`, `class ShortEnsemble(ValueError)`, `shard_path(ens_dir, init_date) -> Path`, `done_dates(ens_dir, year: int) -> set[pd.Timestamp]`, `dates_to_fetch(all_dates, done) -> list[pd.Timestamp]`, `write_atomic(df, path) -> None`, `check_members(n: int) -> None`, `reduce_members(means: np.ndarray, sub_ids: list[str], init_date) -> DataFrame`, `with_retries(fn, attempts=5, base_delay=2.0, max_delay=32.0, sleep=time.sleep, log=print)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_fetch_ens_resume.py`

```python
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
    with pytest.raises(ConnectionError):
        F.with_retries(lambda: (_ for _ in ()).throw(ConnectionError("down")),
                       attempts=7, sleep=slept.append, log=lambda *_: None)
    assert slept == [2.0, 4.0, 8.0, 16.0, 32.0, 32.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_fetch_ens_resume.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fbd.ingest.ens_fetch'`

- [ ] **Step 3: Implement** — `src/fbd/ingest/ens_fetch.py`

```python
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
        done |= {pd.Timestamp(x).normalize() for x in d.unique() if pd.Timestamp(x).year == year}
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
```

- [ ] **Step 4: Rewrite** `scripts/fetch_ens.py`

```python
"""Fetch true IFS ENS spread, one resumable shard per init date.

LOGIC.md sec 8.1 names ensemble spread as *the* baseline to beat. The first
fetch took every 3rd date of 2022 (41 dates) and every 6th of 2019 and 2020;
S1 (docs/superpowers/specs/2026-09-23-s1-settle-ens-design.md) completes 2022
and adds all of 2021.

Cost, measured: the store keeps 6-hourly leads, so daily leads 1-10 touch 6
chunks per date, ~123 MB. A full season is ~15 GB, so the fetch must survive
interruption: each date is its own shard, written atomically, and a date that
is already on disk -- as a shard or inside a legacy file -- is skipped.

    python scripts/fetch_ens.py --year 2022
    python scripts/fetch_ens.py --year 2021
    python scripts/fetch_ens.py --year 2022 --verify-legacy 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.ingest import ens_fetch as F  # noqa: E402
from fbd.ingest import wb2  # noqa: E402
from fbd.regions import masks  # noqa: E402

OUT = E.ENS_DIR


def open_field(year: int):
    ds = wb2.open_store(config.ENS_STORE, chunks={"time": 1, "prediction_timedelta": 8})
    inits = wb2.season_init_times([year], hour=0)
    inits = inits[inits.isin(pd.DatetimeIndex(ds.time.values))]
    da = ds["total_precipitation_24hr"].sel(prediction_timedelta=wb2.lead_timedeltas())
    da = wb2.india_slice(da.to_dataset(name="tp24"))["tp24"]
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    return da, inits


def weights(da):
    lats, lons = da.lat.values, da.lon.values
    w = masks.overlap_weights(lats, lons)
    sub_ids = sorted(w.subdivision_id.unique())
    return masks.weights_to_matrix(w, len(lats), len(lons), sub_ids), sub_ids


def fetch_one(da, init, W, sub_ids, workers: int) -> pd.DataFrame:
    with dask.config.set(scheduler="threads", num_workers=workers):
        arr = da.sel(time=init).load()
    arr = arr.transpose("number", "prediction_timedelta", "lat", "lon") * 1000.0
    F.check_members(int(arr.sizes["number"]))
    means, _ = masks.area_mean(arr.values, W)          # (member, lead, sub)
    return F.reduce_members(means, sub_ids, init)


def verify_legacy(da, W, sub_ids, year: int, n: int, workers: int) -> int:
    """Re-fetch ``n`` legacy dates and compare value by value. Exit 1 on mismatch."""
    legacy = [p for p in E.legacy_paths(OUT) if f"_{year}_" in p.name]
    if not legacy:
        print(f"no legacy file for {year}")
        return 1
    old = pd.read_parquet(legacy[0])
    old["init_date"] = pd.to_datetime(old.init_date).dt.normalize()
    dates = sorted(old.init_date.unique())
    picks = [dates[0], dates[len(dates) // 2]][:n]
    worst = 0.0
    for d in picks:
        new = fetch_one(da, pd.Timestamp(d), W, sub_ids, workers)
        m = old[old.init_date == d].merge(new, on=E.KEY, suffixes=("_old", "_new"))
        for v in E.VALUES:
            a, b = m[f"{v}_old"].to_numpy(float), m[f"{v}_new"].to_numpy(float)
            ok = np.isclose(a, b, rtol=E.RTOL, atol=E.ATOL, equal_nan=True)
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
            if not ok.all():
                print(f"MISMATCH {pd.Timestamp(d):%Y-%m-%d} {v}: {int((~ok).sum())} rows")
                return 1
        print(f"  {pd.Timestamp(d):%Y-%m-%d}: {len(m)} rows identical within tolerance")
    print(f"determinism check passed; max |difference| {worst:.3g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify-legacy", type=int, default=0, metavar="N",
                    help="re-fetch N legacy dates and compare; fetches nothing else")
    args = ap.parse_args()

    da, inits = open_field(args.year)
    W, sub_ids = weights(da)

    if args.verify_legacy:
        return verify_legacy(da, W, sub_ids, args.year, args.verify_legacy, args.workers)

    todo = F.dates_to_fetch(inits, F.done_dates(OUT, args.year))
    print(f"IFS ENS {args.year}: {len(inits)} JJAS dates in store, "
          f"{len(inits) - len(todo)} already on disk, {len(todo)} to fetch")

    failed, flagged, t0 = [], [], time.time()
    for i, init in enumerate(todo):
        try:
            df = F.with_retries(lambda: fetch_one(da, init, W, sub_ids, args.workers))
        except F.ShortEnsemble as exc:
            flagged.append((init, str(exc)))
            print(f"  FLAGGED {init:%Y-%m-%d}: {exc}")
            continue
        except Exception as exc:  # after retries
            failed.append((init, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {init:%Y-%m-%d}: {exc}")
            continue
        F.write_atomic(df, F.shard_path(OUT, init))
        done = i + 1
        per = (time.time() - t0) / done
        print(f"  {done}/{len(todo)} {init:%Y-%m-%d}  {per:.1f}s each, "
              f"~{per * (len(todo) - done) / 60:.1f} min left")

    have = len(F.done_dates(OUT, args.year))
    print(f"\n{args.year}: {have}/{len(inits)} dates on disk")
    for d, why in flagged:
        print(f"  flagged {d:%Y-%m-%d}: {why}")
    for d, why in failed:
        print(f"  missing {d:%Y-%m-%d}: {why}  (re-run to retry)")
    return 0 if have == len(inits) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_fetch_ens_resume.py -q` → Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fbd/ingest/ens_fetch.py scripts/fetch_ens.py tests/test_fetch_ens_resume.py
git commit -m "Make the ENS fetch resumable: one atomic shard per init date"
```

---

### Task 3: Provenance helpers and the S1 audit

**Files:**
- Create: `src/fbd/evaluate/provenance.py`
- Create: `scripts/audit_s1.py`
- Test: `tests/test_settle_ens.py` (provenance section)

**Interfaces:**
- Consumes: `E.load_ens`, `E.comparison_rows`, `compute_confidence_intervals.ens_margin(n_boot, ens=...)`.
- Produces: `sha256_file(path) -> str`, `is_committed(path, repo=config.ROOT) -> bool`; artifact `data/artifacts/s1_audit.json` with keys `reproduced` (bool), `checks` (list of `{name, got, published, ok}`), `hashes` (`{"bust_model.joblib": hex, "dataset.parquet": hex}`), `row_rule` (str), `discrepancies` (list of str).

- [ ] **Step 1: Write the failing tests** — `tests/test_settle_ens.py` (first section)

```python
"""S1 machinery: provenance, the registration, the verdict, and the statistics.

The provenance, registration and verdict parts are dependency-free and run in
CI. The statistics need scikit-learn and skip where it is not installed.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd.evaluate import provenance as P  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   cwd=repo, check=True, capture_output=True)


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"forecast" * 1000)
    assert P.sha256_file(f) == hashlib.sha256(b"forecast" * 1000).hexdigest()


def test_is_committed_tracks_state(tmp_path):
    _git(tmp_path, "init", "-q")
    f = tmp_path / "PREREG.md"
    f.write_text("plan")
    assert not P.is_committed(f, repo=tmp_path), "untracked is not committed"
    _git(tmp_path, "add", "PREREG.md")
    _git(tmp_path, "commit", "-q", "-m", "register")
    assert P.is_committed(f, repo=tmp_path)
    f.write_text("plan, edited after seeing data")
    assert not P.is_committed(f, repo=tmp_path), "an edit must be visible"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_settle_ens.py -q` → Expected: FAIL — no module `fbd.evaluate.provenance`.

- [ ] **Step 3: Implement** — `src/fbd/evaluate/provenance.py`

```python
"""Hashes and git state, so a registered analysis can prove what it ran on."""
from __future__ import annotations

import hashlib
import subprocess  # nosec B404 - fixed argv, no shell
from pathlib import Path

from fbd import config


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_committed(path, repo: Path = config.ROOT) -> bool:
    """True if ``path`` is tracked by git and has no uncommitted changes."""
    path = Path(path)
    tracked = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=repo, capture_output=True, text=True)
    if tracked.returncode != 0:
        return False
    dirty = subprocess.run(  # nosec B603 B607
        ["git", "status", "--porcelain", "--", str(path)],
        cwd=repo, capture_output=True, text=True)
    return dirty.returncode == 0 and dirty.stdout.strip() == ""
```

- [ ] **Step 4: Implement** `scripts/audit_s1.py`

```python
"""S1 audit: reproduce what is published before registering anything new.

Runs on LEGACY ENS files only, so it measures the published baseline even after
new shards exist. Exit 1 if the 40-date margin does not reproduce to four
decimals: S1 must not build on a number the code cannot regenerate.

    PYTHONPATH=src python scripts/audit_s1.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import compute_confidence_intervals as CCI  # noqa: E402
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402

OUT = config.ARTIFACTS / "s1_audit.json"

#: As published in README.md, D-014 and D-022 before S1.
PUBLISHED = {
    "raw margin point": (0.0251, 4), "raw margin lo": (-0.0083, 4),
    "raw margin hi": (0.0580, 4), "calibrated margin point": (0.0403, 4),
    "model AUROC": (0.832, 3), "raw ENS AUROC": (0.807, 3),
    "calibrated ENS AUROC": (0.792, 3), "lagged proxy AUROC": (0.731, 3),
    "rows": (6732, 0), "init dates": (40, 0),
    "scored bust rate %": (3.4, 1), "refused bust rate %": (23.4, 1),
}

ROW_RULE = (
    "fbd.evaluate.ens.comparison_rows: inner join of dataset.parquet and ENS on "
    "(subdivision_id, init_date, lead_day); drop rows missing bust or ens_spread; "
    "test = split 'test' and lead_day in 3..7; fit = splits 'train' and 'val', all "
    "leads. The frozen model scores every test row, including rows the served "
    "product would refuse as out-of-distribution."
)


def main() -> int:
    checks, discrepancies = [], []

    def check(name, got):
        want, places = PUBLISHED[name]
        ok = round(float(got), places) == round(want, places)
        checks.append({"name": name, "got": float(got), "published": want, "ok": ok})
        print(f"  {'OK ' if ok else 'XX '} {name:28s} got {got:.4f}  published {want}")

    print("reproducing the published ENS margin (legacy ENS files only):")
    ens = E.load_ens(include_shards=False)
    res = CCI.ens_margin(2000, ens=ens)
    raw, cal = res["margins"]["raw ENS spread"], res["margins"]["calibrated ENS"]
    check("raw margin point", raw["point"])
    check("raw margin lo", raw["lo"])
    check("raw margin hi", raw["hi"])
    check("calibrated margin point", cal["point"])
    check("model AUROC", res["auroc"]["model"]["point"])
    check("raw ENS AUROC", res["auroc"]["raw ENS spread"]["point"])
    check("calibrated ENS AUROC", res["auroc"]["calibrated ENS"]["point"])
    check("rows", res["n_rows"])
    check("init dates", res["n_init_dates"])

    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    test, _fit = E.comparison_rows(ds, ens)
    lag = test.lagged_spread.to_numpy(float)
    ok = np.isfinite(lag)
    check("lagged proxy AUROC", M.auroc(test.bust.to_numpy(float)[ok], lag[ok]))

    legacy_2022 = sorted(pd.to_datetime(ens[ens.init_date.dt.year == 2022].init_date).unique())
    used = sorted(pd.to_datetime(test.init_date).unique())
    dropped = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in legacy_2022 if d not in used]
    if dropped:
        discrepancies.append(
            f"{len(legacy_2022)} legacy 2022 dates fetched but {len(used)} used: "
            f"{', '.join(dropped)} have no Day 3-7 label (valid dates fall after "
            "30 September). Expected, and stated in the registration.")

    db = config.ARTIFACTS / "bulletins.sqlite"
    if db.exists():
        con = sqlite3.connect(db)
        rows = dict((s, r) for s, _n, r in con.execute(
            "SELECT status, COUNT(*), AVG(actual_bust) FROM bulletins "
            "WHERE init_date >= '2022-01-01' AND init_date < '2023-01-01' "
            "AND actual_bust IS NOT NULL GROUP BY status"))
        con.close()
        check("scored bust rate %", 100 * rows["OK"])
        check("refused bust rate %", 100 * rows["OUT_OF_DISTRIBUTION"])

    ci = json.loads((config.ARTIFACTS / "confidence_intervals.json").read_text())
    ece = ci["decision_band"]["4 XGBoost + isotonic"]["ece"]["point"]
    if round(ece, 4) != 0.0107:
        discrepancies.append(
            f"landing.html's ECE card said 0.0107 (the served-subset figure from "
            f"plot_evaluation_figures.py); the decision-band interval beside the "
            f"other cards is {ece:.4f}. The card now reads the decision-band value "
            "and says so.")
    discrepancies.append(
        "docs/FRONTEND_BUILT.md said every landing-page claim reads live from the "
        "API. The ENS sentence, the four stat cards and the refusal chart were "
        "hardcoded (and the page fetched /api/metrics, then discarded it).")
    discrepancies.append(
        "README calls the ENS comparison 'on identical rows' without saying the "
        "model scores rows the product would refuse. Row rule now stated.")

    hashes = {"bust_model.joblib": P.sha256_file(config.ARTIFACTS / "bust_model.joblib"),
              "dataset.parquet": P.sha256_file(config.PROCESSED / "dataset.parquet")}
    reproduced = all(c["ok"] for c in checks if c["name"].startswith("raw margin"))

    OUT.write_text(json.dumps({"reproduced": reproduced, "checks": checks,
                               "hashes": hashes, "row_rule": ROW_RULE,
                               "discrepancies": discrepancies}, indent=2),
                   encoding="utf-8")
    print(f"\nhashes: {hashes}")
    print("discrepancies:")
    for d in discrepancies:
        print(f"  - {d}")
    print(f"\nwrote {OUT.relative_to(config.ROOT)}")
    if not reproduced:
        print("\nSTOP: the published margin does not reproduce. Do not register.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests and the audit**

Run: `PYTHONPATH=src python -m pytest tests/test_settle_ens.py -q` → Expected: 2 passed.
Run: `PYTHONPATH=src python scripts/audit_s1.py` → Expected: exit 0; the three `raw margin` checks OK. Record every `XX` line — each is a discrepancy to carry into the registration and fix in Task 9.

- [ ] **Step 6: Commit**

```bash
git add src/fbd/evaluate/provenance.py scripts/audit_s1.py tests/test_settle_ens.py data/artifacts/s1_audit.json
git commit -m "Audit S1's baseline: reproduce the published ENS margin first"
```

---

### Task 4: Registration guards and the registered analysis

**Files:**
- Create: `src/fbd/evaluate/registration.py`
- Create: `src/fbd/evaluate/settle.py`
- Create: `scripts/settle_ens.py`
- Test: `tests/test_settle_ens.py` (append)

**Interfaces:**
- Consumes: `P.sha256_file`, `P.is_committed`, `E.load_ens`, `E.comparison_rows`, `E.dates_by_year`, `U.paired_difference`, `U.metric_interval`, `M.auroc`, `M.brier`, `M.brier_skill_score`, `M.decision_cost`, `B.SpreadBaseline`, `T.BustModel.load`, `fbd.quality.escalation.REVIEW_THRESHOLD`.
- Produces (registration.py, pure): `class RegistrationError(RuntimeError)`, `parse_registration(text: str) -> dict[str, str]`, `verify_hashes(reg, files: dict[str, Path]) -> list[str]`, `check_complete(have: int, required: int) -> None`, `verdict(lo: float, hi: float) -> str` (one of `"model_better" | "ens_better" | "indistinguishable"`), `VERDICT_TEXT: dict[str, str]`, `guard(prereg: Path, files: dict[str, Path], repo: Path) -> dict[str, str]`.
- Produces (settle.py, sklearn): `choose_comparator(y, raw, rel) -> tuple[str, np.ndarray]`, `primary(y, p_model, comparator, clusters, n_boot, seed) -> dict`, `secondary_calibrated(test, fit, p_model, clusters, n_boot, seed) -> dict`, `secondary_combined(test, fit21, p_model, p_model_21, clusters, n_boot, seed) -> dict`, `by_group(y, p_model, comparator, groups, clusters, n_boot, seed) -> dict[str, dict]`.
- Artifact `data/artifacts/ens_settlement.json` top-level keys: `registration_sha256`, `hashes`, `n_ens_dates_2022`, `primary`, `secondary`, `exploratory`, `row_rule`.

- [ ] **Step 1: Append failing tests** to `tests/test_settle_ens.py`

```python
from fbd.evaluate import registration as R  # noqa: E402

REG = """# Registration

text

```registration
model_sha256: aaa
dataset_sha256: bbb
seed: 20260919
n_boot: 10000
required_ens_dates_2022: 122
```
"""


def test_parse_registration_block():
    reg = R.parse_registration(REG)
    assert reg["seed"] == "20260919" and reg["required_ens_dates_2022"] == "122"


def test_parse_registration_requires_the_block():
    with pytest.raises(R.RegistrationError):
        R.parse_registration("# no block here")


def test_verify_hashes_names_the_mismatch(tmp_path):
    f = tmp_path / "model.joblib"
    f.write_bytes(b"m")
    reg = {"model_sha256": hashlib.sha256(b"m").hexdigest()}
    assert R.verify_hashes(reg, {"model_sha256": f}) == []
    f.write_bytes(b"retrained")
    assert "model_sha256" in R.verify_hashes(reg, {"model_sha256": f})[0]


def test_incomplete_season_is_refused():
    R.check_complete(122, 122)
    with pytest.raises(R.RegistrationError):
        R.check_complete(121, 122)


@pytest.mark.parametrize("lo,hi,want", [
    (0.001, 0.05, "model_better"),
    (-0.05, -0.001, "ens_better"),
    (-0.01, 0.04, "indistinguishable"),
    (0.0, 0.04, "indistinguishable"),      # touching zero is not excluding it
])
def test_verdict_mapping(lo, hi, want):
    assert R.verdict(lo, hi) == want
    assert R.VERDICT_TEXT[want]


def test_guard_refuses_an_uncommitted_registration(tmp_path):
    _git(tmp_path, "init", "-q")
    prereg = tmp_path / "PREREGISTRATION_S1.md"
    prereg.write_text(REG)
    with pytest.raises(R.RegistrationError, match="committed"):
        R.guard(prereg, {}, repo=tmp_path)


def test_guard_refuses_changed_inputs(tmp_path):
    _git(tmp_path, "init", "-q")
    model = tmp_path / "model.joblib"
    model.write_bytes(b"m")
    prereg = tmp_path / "PREREGISTRATION_S1.md"
    prereg.write_text(REG.replace("aaa", hashlib.sha256(b"other").hexdigest()))
    _git(tmp_path, "add", prereg.name)
    _git(tmp_path, "commit", "-q", "-m", "register")
    with pytest.raises(R.RegistrationError, match="model_sha256"):
        R.guard(prereg, {"model_sha256": model}, repo=tmp_path)


# ------------------------------------------------------------- statistics
def _synthetic(n_dates=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for s in range(20):
            for lead in (3, 4, 5, 6, 7):
                y = float(rng.random() < 0.05)
                rows.append({"init_date": pd.Timestamp("2022-06-01") + pd.Timedelta(days=d),
                             "subdivision_id": f"S{s}", "lead_day": lead, "bust": y,
                             "ens_spread": rng.gamma(2, 1) + 1.5 * y,
                             "ens_mean": 5.0, "p_model": 0.02 + 0.5 * y * rng.random()})
    df = pd.DataFrame(rows)
    df["ens_spread_rel"] = df.ens_spread / (df.ens_mean + 1.0)
    return df


def test_primary_is_deterministic_and_finds_a_clear_edge():
    pytest.importorskip("sklearn")
    from fbd.evaluate import settle as S

    df = _synthetic()
    y = df.bust.to_numpy(float)
    clusters = df.init_date.dt.strftime("%Y-%m-%d").to_numpy()
    name, comp = S.choose_comparator(y, df.ens_spread.to_numpy(float),
                                     df.ens_spread_rel.to_numpy(float))
    a = S.primary(y, df.p_model.to_numpy(float), comp, clusters, n_boot=300, seed=7)
    b = S.primary(y, df.p_model.to_numpy(float), comp, clusters, n_boot=300, seed=7)
    assert a == b, "same seed, same result"
    assert a["verdict"] in R.VERDICT_TEXT
    assert a["n_init_dates"] == 60


def test_comparator_is_the_stronger_ens_variant():
    pytest.importorskip("sklearn")
    from fbd.evaluate import settle as S

    y = np.array([0, 0, 1, 1], float)
    strong, weak = np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.9, 0.1, 0.2, 0.8])
    assert S.choose_comparator(y, strong, weak)[0] == "raw"
    assert S.choose_comparator(y, weak, strong)[0] == "relative"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_settle_ens.py -q` → Expected: FAIL — no module `fbd.evaluate.registration`.

- [ ] **Step 3: Implement** — `src/fbd/evaluate/registration.py`

```python
"""The S1 registration, read by machine, so the analysis cannot drift from it.

docs/PREREGISTRATION_S1.md carries a fenced ```registration block of
``key: value`` lines. settle_ens.py refuses to run unless that file is
committed, every hashed input still matches, and the season is complete.
Dependency-free on purpose: these guards are tested in CI.
"""
from __future__ import annotations

from pathlib import Path

from fbd.evaluate import provenance as P

FENCE = "```registration"


class RegistrationError(RuntimeError):
    """The registered analysis cannot run as registered."""


def parse_registration(text: str) -> dict:
    if FENCE not in text:
        raise RegistrationError("no ```registration block found")
    body = text.split(FENCE, 1)[1].split("```", 1)[0]
    out = {}
    for line in body.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def verify_hashes(reg: dict, files: dict) -> list:
    problems = []
    for key, path in files.items():
        want = reg.get(key)
        got = P.sha256_file(path)
        if want != got:
            problems.append(f"{key}: registered {want}, file {Path(path).name} is {got}")
    return problems


def check_complete(have: int, required: int) -> None:
    if have < required:
        raise RegistrationError(
            f"{have} ENS init dates for 2022 on disk, {required} registered. "
            "Re-run scripts/fetch_ens.py --year 2022; a partial season is never "
            "reported as the full one.")


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "model_better"
    if hi < 0:
        return "ens_better"
    return "indistinguishable"


VERDICT_TEXT = {
    "model_better": "the model outranks raw ENS spread over the full held-out season",
    "ens_better": "raw ENS spread outranks the model over the full held-out season",
    "indistinguishable": "the model is not distinguishable from raw ENS spread over the "
                         "full held-out season",
}


def guard(prereg: Path, files: dict, repo: Path) -> dict:
    """Every precondition, in order. Returns the parsed registration."""
    if not P.is_committed(prereg, repo=repo):
        raise RegistrationError(
            f"{Path(prereg).name} is not committed as-is. The analysis runs only "
            "against a registration that git can show existed first.")
    reg = parse_registration(Path(prereg).read_text(encoding="utf-8"))
    problems = verify_hashes(reg, files)
    if problems:
        raise RegistrationError("frozen inputs changed:\n  " + "\n  ".join(problems))
    return reg
```

- [ ] **Step 4: Implement** — `src/fbd/evaluate/settle.py`

```python
"""The statistics registered in docs/PREREGISTRATION_S1.md. Needs scikit-learn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fbd.evaluate import metrics as M
from fbd.evaluate import registration as R
from fbd.evaluate import uncertainty as U
from fbd.model import baselines as B
from fbd.quality.escalation import REVIEW_THRESHOLD


def choose_comparator(y, raw, rel):
    """The published rule: face whichever ENS variant ranks busts better."""
    return ("raw", raw) if M.auroc(y, raw) >= M.auroc(y, rel) else ("relative", rel)


def _iv(iv) -> dict:
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero}


def primary(y, p_model, comparator, clusters, n_boot, seed) -> dict:
    iv = U.paired_difference(M.auroc, y, p_model, comparator, clusters,
                             n_boot=n_boot, seed=seed)
    v = R.verdict(iv.lo, iv.hi)
    return {**_iv(iv), "verdict": v, "text": R.VERDICT_TEXT[v],
            "n_rows": int(len(y)), "n_init_dates": int(len(np.unique(clusters))),
            "model_auroc": M.auroc(y, p_model), "ens_auroc": M.auroc(y, comparator)}


def _cost(y, p):
    return M.decision_cost(y, p, REVIEW_THRESHOLD)["cost_per_1000_rows"]


def secondary_calibrated(test, fit, p_model, clusters, n_boot, seed) -> dict:
    """(a) ENS isotonic fitted on 2021 only, the model's calibration year (D-010)."""
    fit21 = fit[pd.to_datetime(fit.init_date).dt.year == 2021]
    p_ens = B.SpreadBaseline("ens_spread").fit(fit21).predict_proba(test)
    y = test.bust.to_numpy(float)
    out = {"fit_rows_2021": int(len(fit21))}
    for name, fn in (("auroc", M.auroc), ("brier", M.brier),
                     ("bss", M.brier_skill_score), ("cost_per_1000", _cost)):
        out[name] = _iv(U.paired_difference(fn, y, p_model, p_ens, clusters,
                                            n_boot=n_boot, seed=seed))
    out["note"] = "model minus calibrated ENS; lower Brier and cost are better"
    return out


def _design(p, spread):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1 - p)), np.log1p(np.asarray(spread, float))])


def secondary_combined(test, fit21, p_model, p_model_21, clusters, n_boot, seed) -> dict:
    """(b) Does the model add anything on top of ENS? Fitted on 2021, tested on 2022."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_design(p_model_21, fit21.ens_spread), fit21.bust.to_numpy(int))
    p_combo = lr.predict_proba(_design(p_model, test.ens_spread))[:, 1]
    y = test.bust.to_numpy(float)
    raw = test.ens_spread.to_numpy(float)
    return {
        "fit_rows_2021": int(len(fit21)),
        "coefficients": {"logit_p_model": float(lr.coef_[0][0]),
                         "log1p_ens_spread": float(lr.coef_[0][1]),
                         "intercept": float(lr.intercept_[0])},
        "combined_minus_ens": _iv(U.paired_difference(M.auroc, y, p_combo, raw, clusters,
                                                      n_boot=n_boot, seed=seed)),
        "combined_minus_model": _iv(U.paired_difference(M.auroc, y, p_combo, p_model,
                                                        clusters, n_boot=n_boot, seed=seed)),
        "caveat": ("The model's isotonic calibration was fitted on 2021, so this "
                   "combination is fitted on probabilities in-sample for calibration. "
                   "Evaluation on 2022 is out of sample."),
    }


def by_group(y, p_model, comparator, groups, clusters, n_boot, seed) -> dict:
    """Exploratory: the margin within each group (lead, month, old/new dates)."""
    out = {}
    groups = np.asarray(groups)
    for g in sorted(np.unique(groups), key=str):
        m = groups == g
        if len(np.unique(y[m])) < 2:
            continue
        out[str(g)] = _iv(U.paired_difference(M.auroc, y[m], p_model[m], comparator[m],
                                              clusters[m], n_boot=n_boot, seed=seed))
    return out
```

- [ ] **Step 5: Implement** `scripts/settle_ens.py`

```python
"""Run exactly the analysis registered in docs/PREREGISTRATION_S1.md.

Refuses to run unless the registration is committed as-is, the frozen model and
dataset match their registered hashes, and all 2022 ENS dates are on disk.

    PYTHONPATH=src python scripts/settle_ens.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.evaluate import settle as S  # noqa: E402
from fbd.model import train as T  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S1.md"
OUT = config.ARTIFACTS / "ens_settlement.json"
MODEL = config.ARTIFACTS / "bust_model.joblib"
DATASET = config.PROCESSED / "dataset.parquet"
EXPLORATORY_N_BOOT = 2000


def main() -> int:
    try:
        reg = R.guard(PREREG, {"model_sha256": MODEL, "dataset_sha256": DATASET}, repo=ROOT)
        ens = E.load_ens()
        have = E.dates_by_year(ens).get(2022, 0)
        R.check_complete(have, int(reg["required_ens_dates_2022"]))
    except R.RegistrationError as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot = int(reg["seed"]), int(reg["n_boot"])

    ds = pd.read_parquet(DATASET)
    model = T.BustModel.load(MODEL)
    test, fit = E.comparison_rows(ds, ens)
    y = test.bust.to_numpy(float)
    p = model.predict_proba(test)
    clusters = pd.to_datetime(test.init_date).dt.strftime("%Y-%m-%d").to_numpy()
    name, comp = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                     test.ens_spread_rel.to_numpy(float))

    print(f"primary: {len(test):,} rows over {len(np.unique(clusters))} dates; "
          f"comparator = {name} spread; {n_boot:,} resamples, seed {seed}")
    prim = S.primary(y, p, comp, clusters, n_boot, seed)
    prim["comparator"] = name
    print(f"  model minus ENS  {prim['point']:+.4f} [{prim['lo']:+.4f}, {prim['hi']:+.4f}]"
          f"  -> {prim['text']}")

    fit21 = fit[(pd.to_datetime(fit.init_date).dt.year == 2021)
                & fit.lead_day.isin(config.DECISION_BAND)]
    secondary = {
        "a_calibrated_2021": S.secondary_calibrated(test, fit, p, clusters, n_boot, seed),
        "b_model_plus_ens": S.secondary_combined(test, fit21, p, model.predict_proba(fit21),
                                                 clusters, n_boot, seed),
    }
    from fbd.model import baselines as B
    from fbd.evaluate import metrics as M
    from fbd.evaluate import uncertainty as U
    p_pool = B.SpreadBaseline("ens_spread").fit(fit).predict_proba(test)
    secondary["c_pooled_calibration"] = {
        "auroc": S._iv(U.paired_difference(M.auroc, y, p, p_pool, clusters,
                                           n_boot=n_boot, seed=seed)),
        "note": "continuity with the published pooled fit (train + val ENS rows)"}

    all_leads, _ = E.comparison_rows(ds, ens, lead_days=range(1, 11))
    ya = all_leads.bust.to_numpy(float)
    pa = model.predict_proba(all_leads)
    ca = pd.to_datetime(all_leads.init_date).dt.strftime("%Y-%m-%d").to_numpy()
    _n, compa = S.choose_comparator(ya, all_leads.ens_spread.to_numpy(float),
                                    all_leads.ens_spread_rel.to_numpy(float))
    legacy = {pd.Timestamp(d) for d in E.load_ens(include_shards=False).init_date.unique()}
    origin = np.where(pd.to_datetime(test.init_date).isin(legacy), "original_40", "new")
    exploratory = {
        "by_lead": S.by_group(ya, pa, compa, all_leads.lead_day.to_numpy(), ca,
                              EXPLORATORY_N_BOOT, seed),
        "by_month": S.by_group(y, p, comp, pd.to_datetime(test.init_date).dt.month.to_numpy(),
                               clusters, EXPLORATORY_N_BOOT, seed),
        "original_vs_new_dates": S.by_group(y, p, comp, origin, clusters,
                                            EXPLORATORY_N_BOOT, seed),
        "n_boot": EXPLORATORY_N_BOOT,
        "note": "exploratory: no claims are drawn from these",
    }

    payload = {
        "registration_sha256": P.sha256_file(PREREG),
        "hashes": {"model_sha256": P.sha256_file(MODEL), "dataset_sha256": P.sha256_file(DATASET)},
        "n_ens_dates_2022": have,
        "row_rule": ("comparison_rows(): Day 3-7 test rows with a label and ENS spread; "
                     "the model scores every row, including ones the served product "
                     "would refuse as out-of-distribution"),
        "primary": prim, "secondary": secondary, "exploratory": exploratory,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_settle_ens.py -q` → Expected: 14 passed locally (sklearn present).
Run: `PYTHONPATH=src python scripts/settle_ens.py` → Expected: `REFUSED: PREREGISTRATION_S1.md is not committed as-is` (the file does not exist yet), exit 2.

- [ ] **Step 7: Commit**

```bash
git add src/fbd/evaluate/registration.py src/fbd/evaluate/settle.py scripts/settle_ens.py tests/test_settle_ens.py
git commit -m "Add the registered S1 analysis and the guards that make it run as registered"
```

---

### Task 5: `/api/metrics` serves the numbers; the landing page reads them

**Files:**
- Modify: `src/fbd/api/app.py` (`metrics`, lines 421–427; new helpers above it)
- Modify: `web/landing.html` (ENS sentence ~line 213; refusal prose ~181, ~193; boot block ~428–439)
- Modify: `tests/test_web_pages.py` (replace `test_landing_page_states_the_ensemble_margin_is_not_established`)
- Test: `tests/test_metrics_endpoint.py`

**Interfaces:**
- Produces: `GET /api/metrics` returns the `results.json` object plus `confidence_intervals` (object | null), `ens_settlement` (object | null), `refusal` (`{"year": int, "scored": {"n": int, "bust_rate": float}, "refused": {"n": int, "bust_rate": float}, "ratio": float}` | null).

- [ ] **Step 1: Write the failing endpoint test** — `tests/test_metrics_endpoint.py`

```python
"""/api/metrics carries every headline number the landing page shows."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from fbd import config  # noqa: E402
from fbd.api import app as app_module  # noqa: E402


def _client(monkeypatch, tmp_path, with_settlement=True, with_store=True):
    (tmp_path / "results.json").write_text(json.dumps({"overall": {}}))
    (tmp_path / "confidence_intervals.json").write_text(json.dumps({"true_ens": {"n_init_dates": 40}}))
    if with_settlement:
        (tmp_path / "ens_settlement.json").write_text(json.dumps({"primary": {"verdict": "indistinguishable"}}))
    db = tmp_path / "bulletins.sqlite"
    if with_store:
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE bulletins (init_date TEXT, status TEXT, actual_bust REAL)")
        con.executemany("INSERT INTO bulletins VALUES (?,?,?)",
                        [("2022-06-01", "OK", 0)] * 9 + [("2022-06-01", "OK", 1)]
                        + [("2022-06-01", "OUT_OF_DISTRIBUTION", 1)] * 1
                        + [("2021-06-01", "OK", 1)] * 50)
        con.commit()
        con.close()
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(app_module, "DB", db)
    return TestClient(app_module.app)


def test_metrics_carries_every_block(monkeypatch, tmp_path):
    body = _client(monkeypatch, tmp_path).get("/api/metrics").json()
    assert body["confidence_intervals"]["true_ens"]["n_init_dates"] == 40
    assert body["ens_settlement"]["primary"]["verdict"] == "indistinguishable"
    r = body["refusal"]
    assert r["year"] == config.TEST_YEARS[0]
    assert r["scored"] == {"n": 10, "bust_rate": 0.1}, "other years must not leak in"
    assert r["refused"]["bust_rate"] == 1.0 and r["ratio"] == 10.0


def test_missing_pieces_are_null_not_errors(monkeypatch, tmp_path):
    body = _client(monkeypatch, tmp_path, with_settlement=False, with_store=False).get(
        "/api/metrics").json()
    assert body["ens_settlement"] is None
    assert body["refusal"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_metrics_endpoint.py -q` → Expected: FAIL — `KeyError: 'confidence_intervals'`.

- [ ] **Step 3: Implement** in `src/fbd/api/app.py` — replace `metrics()` with:

```python
def _json_or_none(path: Path):
    """A committed artifact, or None. Missing is a state the page renders, not an error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _refusal_rates() -> dict | None:
    """Bust rate of scored vs refused rows over the held-out year, from the store.

    Computed, not quoted, so the landing page's "refused days bust N x more often"
    cannot outlive a regenerated store.
    """
    if not Path(DB).exists():
        return None
    year = config.TEST_YEARS[0]
    con = sqlite3.connect(DB)
    try:
        rows = {s: (int(n), float(r)) for s, n, r in con.execute(
            "SELECT status, COUNT(*), AVG(actual_bust) FROM bulletins "
            "WHERE init_date >= ? AND init_date < ? AND actual_bust IS NOT NULL "
            "GROUP BY status", (f"{year}-01-01", f"{year + 1}-01-01"))}
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if "OK" not in rows or "OUT_OF_DISTRIBUTION" not in rows:
        return None
    (ns, rs), (nr, rr) = rows["OK"], rows["OUT_OF_DISTRIBUTION"]
    return {"year": year, "scored": {"n": ns, "bust_rate": rs},
            "refused": {"n": nr, "bust_rate": rr},
            "ratio": rr / rs if rs else None}


@app.get("/api/metrics")
def metrics() -> dict:
    """Held-out-year evaluation, so the UI can show it is not a cherry-pick.

    Also carries the interval tables, the S1 settlement and the refusal rates,
    so every headline number on the landing page comes from one request. A
    missing piece is null, never an error.
    """
    path = config.ARTIFACTS / "results.json"
    if not path.exists():
        raise HTTPException(503, "results.json missing; run scripts/train_model.py")
    out = json.loads(path.read_text())
    out["confidence_intervals"] = _json_or_none(config.ARTIFACTS / "confidence_intervals.json")
    out["ens_settlement"] = _json_or_none(config.ARTIFACTS / "ens_settlement.json")
    out["refusal"] = _refusal_rates()
    return out
```

(`Path`, `json` and `sqlite3` are already imported in `app.py`; confirm with `grep -n "^import\|^from" src/fbd/api/app.py` and add any that are missing.)

- [ ] **Step 4: Replace the landing test** in `tests/test_web_pages.py` — delete `test_landing_page_states_the_ensemble_margin_is_not_established` and add:

```python
#: Headline numbers the landing page used to write into its own markup.
LANDING_HEADLINE_LITERALS = [
    '"0.840"', "[0.821, 0.859]", '"0.088"', "[0.053, 0.123]", '"0.0107"',
    "+0.025 AUROC", "−0.008", "+0.058", "drawRefusal(0.034, 0.234)",
    "6.9× more often", "23.4 percent", '"6.9×"',
]


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
@pytest.mark.parametrize("literal", LANDING_HEADLINE_LITERALS)
def test_landing_page_reads_headline_numbers_from_the_api(literal: str):
    """S1 changes the ENS margin whichever way it falls. A written-in number
    would go on asserting the old one."""
    html = _read(LANDING)
    assert literal not in html, f"landing.html hardcodes {literal!r}; read it from /api/metrics"
    assert 'getJSON("/api/metrics")' in html


@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
def test_landing_page_can_state_every_ens_verdict():
    """D-022/D-025: whatever the interval says, the page says it plainly."""
    html = _read(LANDING)
    assert "ens_settlement" in html and "true_ens" in html
    for v in ("model_better", "ens_better", "indistinguishable"):
        assert v in html, f"no wording for verdict {v}"
    assert "<strong>is not</strong>" in html, "the undecided case must say so plainly"
```

- [ ] **Step 5: Implement** in `web/landing.html`

Replace the paragraph after `<div class="cards" id="statCards"></div>` with:

```html
    <p style="margin-top:20px" id="ensClaim">
      The margin over a real 50-member operational ensemble — loading…
    </p>
```

At ~line 181 replace `busted <strong>6.9× more often</strong> than the days it accepted.` with
`busted <strong><span id="refuseRatio">—</span> more often</strong> than the days it accepted.`
and in the refusal chart element (~line 193) replace the fixed `aria-label="…"` with `aria-label="Bust rate of refused against scored days"`.

Replace the block from `// Refusal rates and headline metrics come from committed artifacts` through the `statCard("Refused vs scored", …);` line with:

```js
  // Every headline number here comes from /api/metrics (results.json, the
  // interval tables, the S1 settlement and the store's refusal rates). A
  // missing piece renders "—", never a number remembered from an old run.
  let m = null;
  try { m = await getJSON("/api/metrics"); } catch (e) { /* rendered as — */ }
  renderRefusal(m && m.refusal);
  $("#statCards").innerHTML = statCardsHTML(m);
  $("#ensClaim").innerHTML = ensClaimHTML(m);
}

const MODEL_KEY = "4 XGBoost + isotonic", LAGGED_KEY = "2 spread [lagged-ensemble]";
const signed = (x, n) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(n);
const pctOf = (x) => (100 * x).toFixed(1) + "%";

function renderRefusal(r) {
  if (!r) { $("#refuseRatio").textContent = "—"; return; }
  $("#refuseRatio").textContent = r.ratio.toFixed(1) + "×";
  $("#refuseChart").innerHTML = drawRefusal(r.scored.bust_rate, r.refused.bust_rate);
}

function statCardsHTML(m) {
  const band = m && m.confidence_intervals && m.confidence_intervals.decision_band;
  const mod = band && band[MODEL_KEY], lag = band && band[LAGGED_KEY];
  const r = m && m.refusal;
  const iv = (o, n) => `[${o.lo.toFixed(n)}, ${o.hi.toFixed(n)}]`;
  const na = "Unavailable in this build.";
  return (
    statCard("AUROC", mod ? mod.auroc.point.toFixed(3) : "—",
      mod ? `${iv(mod.auroc, 3)} — against ${lag.auroc.point.toFixed(3)} for the ensemble-spread baseline.` : na) +
    statCard("Brier skill", mod ? mod.bss.point.toFixed(3) : "—",
      mod ? `${iv(mod.bss, 3)} over climatology, on the decision band.` : na) +
    statCard("Calibration error", mod ? mod.ece.point.toFixed(4) : "—",
      mod ? `${iv(mod.ece, 4)} expected calibration error, decision band. Overconfident in the extreme tail — see the reliability diagram.` : na) +
    statCard("Refused vs scored", r ? r.ratio.toFixed(1) + "×" : "—",
      r ? `Refused days bust ${pctOf(r.refused.bust_rate)} of the time against ${pctOf(r.scored.bust_rate)} for scored days (${r.year}).` : "Needs the bulletin store.")
  );
}

// The ENS claim, in the wording its interval earns. Prefers the registered S1
// settlement; falls back to the 40-date margin in the interval tables.
function ensClaimHTML(m) {
  const s = m && m.ens_settlement && m.ens_settlement.primary;
  const t = m && m.confidence_intervals && m.confidence_intervals.true_ens;
  const lead = "The margin over a real 50-member operational ensemble is ";
  const fmt = (o, n) => `<strong>${signed(o.point, 3)} AUROC [${signed(o.lo, 3)}, ${signed(o.hi, 3)}]</strong> on ${n} init dates`;
  const words = {
    model_better: "— an interval above zero. Over the full held-out season, the model outranks a real ensemble.",
    ens_better: "— an interval below zero. Over the full held-out season, a real ensemble outranks the model.",
    indistinguishable: "— an interval that contains zero. Beating the cheap proxy is established; beating a real ensemble <strong>is not</strong>.",
  };
  if (s) {
    return lead + fmt(s, s.n_init_dates) + " " + words[s.verdict] +
      " The test was registered before this data was fetched.";
  }
  const raw = t && t.margins && t.margins["raw ENS spread"];
  if (raw) {
    const v = raw.lo > 0 ? "model_better" : raw.hi < 0 ? "ens_better" : "indistinguishable";
    return lead + fmt(raw, t.n_init_dates) + " " + words[v];
  }
  return lead + "unavailable in this build (—).";
}
```

- [ ] **Step 6: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_metrics_endpoint.py tests/test_web_pages.py -q` → Expected: all pass.
Then verify in the browser: `http://localhost:8912/landing.html?v=s1` shows the 40-date sentence "on 40 init dates — an interval that contains zero", four stat cards with numbers, the refusal chart, and a clean console.

- [ ] **Step 7: Commit**

```bash
git add src/fbd/api/app.py web/landing.html tests/test_web_pages.py tests/test_metrics_endpoint.py
git commit -m "Serve the landing page's headline numbers from /api/metrics"
```

---

### Task 6: Register, then push — before any data

**Files:**
- Create: `docs/PREREGISTRATION_S1.md`
- Modify: `.github/workflows/ci.yml` (test list)

- [ ] **Step 1: Write** `docs/PREREGISTRATION_S1.md` from the spec's §5 plus the audit output. Required contents, in order:

1. Question and why it gates S2–S5 (spec §1).
2. **Baseline reproduced before registration**: the audit's check table from `data/artifacts/s1_audit.json` (name, published, reproduced, OK), the row rule, the dropped-date note, and the discrepancies list.
3. Frozen inputs (spec §5.1).
4. Primary, secondary, exploratory (spec §5.2–5.4), including: exploratory resamples **2,000**; secondary b's caveat; the comparator rule.
5. **Completeness:** 122 ENS dates of 2022 on disk (the fetch). The analysis rows span fewer dates, because init dates whose Day 3–7 valid dates fall after 30 September have no label; the audit found this is why 41 legacy dates yield 40.
6. Consequences table (spec §9).
7. The machine-readable block, with the hashes from `s1_audit.json`:

````markdown
```registration
model_sha256: <hashes["bust_model.joblib"] from data/artifacts/s1_audit.json>
dataset_sha256: <hashes["dataset.parquet"] from data/artifacts/s1_audit.json>
seed: 20260919
n_boot: 10000
required_ens_dates_2022: 122
lead_days: 3,4,5,6,7
alpha: 0.05
```
````

(Fill the two hashes by copying the 64-character hex strings; nothing else in the block changes.)

- [ ] **Step 2: Add the new tests to CI** — in `.github/workflows/ci.yml`, extend the `pytest -q` list:

```yaml
          tests/test_ens_loader.py
          tests/test_fetch_ens_resume.py
          tests/test_settle_ens.py
          tests/test_metrics_endpoint.py
```

- [ ] **Step 3: Verify the guard now passes its first two checks**

Run: `git add docs/PREREGISTRATION_S1.md && git commit -m "Register S1: the test, its code and its frozen inputs, before the data" && PYTHONPATH=src python scripts/settle_ens.py`
Expected: `REFUSED: 41 ENS init dates for 2022 on disk, 122 registered.` — proves the registration parses, is committed, and the hashes match.

- [ ] **Step 4: Commit CI and push both**

```bash
git add .github/workflows/ci.yml
git commit -m "Run the S1 machinery tests in CI"
git push origin local-llm-and-image-slimming
```

Check CI: `gh run list --branch local-llm-and-image-slimming --limit 1` then `gh run watch <id> --exit-status`. Expected: 5/5 green (the sklearn statistics tests skip in CI).

---

### Task 7: The fetch

**Files:** data only (`data/raw/wb2/ens/shards/…`, gitignored).

- [ ] **Step 1: Determinism check against legacy data**

Run: `PYTHONPATH=src python scripts/fetch_ens.py --year 2022 --verify-legacy 2`
Expected: `determinism check passed`. **If it fails, stop**: today's code does not reproduce the data the published margin was built on.

- [ ] **Step 2: Fetch 2022** (background; ≈81 dates, ≈10 GB)

Run: `PYTHONPATH=src python scripts/fetch_ens.py --year 2022` in the background, logging to the scratchpad. Report the measured seconds-per-date and ETA after the first 5 dates. Expected end: `2022: 122/122 dates on disk`, exit 0. On a non-zero exit, re-run the same command (it resumes).

- [ ] **Step 3: Fetch 2021** (background; 122 dates, ≈15 GB)

Run: `PYTHONPATH=src python scripts/fetch_ens.py --year 2021`. Expected: `2021: 122/122 dates on disk`.

- [ ] **Step 4: Loader sanity**

Run: `PYTHONPATH=src python -c "from fbd.evaluate import ens as E; print(E.dates_by_year(E.load_ens()))"`
Expected: `{2019: 21, 2020: 21, 2021: 122, 2022: 122}` (2019/2020 from the every-6 legacy files; exact 2019/2020 counts as printed by the legacy files). No `ConflictingDuplicate`.

---

### Task 8: Settle, plot, refresh the interval tables

**Files:**
- Create: `scripts/plot_ens_settlement.py`
- Create (artifacts): `data/artifacts/ens_settlement.json`, `docs/figures/ens_settlement.png`, `docs/figures/ens_settlement.svg`
- Modify (artifact): `data/artifacts/confidence_intervals.json`

- [ ] **Step 1: Run the registered analysis**

Run: `PYTHONPATH=src python scripts/settle_ens.py`
Expected: exit 0; prints the primary interval and verdict text; writes `data/artifacts/ens_settlement.json`. Record the verdict — it selects the row of the consequences table used in Task 9. **Do not re-run with any changed parameter.**

- [ ] **Step 2: Write** `scripts/plot_ens_settlement.py`

```python
"""Every S1 interval on one axis, zero marked. Reads ens_settlement.json.

    PYTHONPATH=src python scripts/plot_ens_settlement.py
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
    s = json.loads((config.ARTIFACTS / "ens_settlement.json").read_text())
    p, sec, ex = s["primary"], s["secondary"], s["exploratory"]
    rows = [
        (f"PRIMARY  model − {p['comparator']} ENS spread", p, ACCENT, True),
        ("(a) model − ENS calibrated on 2021", sec["a_calibrated_2021"]["auroc"], INK, False),
        ("(b) model+ENS − ENS", sec["b_model_plus_ens"]["combined_minus_ens"], INK, False),
        ("(b) model+ENS − model", sec["b_model_plus_ens"]["combined_minus_model"], INK, False),
        ("(c) model − ENS, pooled calibration", sec["c_pooled_calibration"]["auroc"], INK, False),
    ] + [(f"exploratory: Day {k}", v, MUTED, False) for k, v in ex["by_lead"].items()] + [
        (f"exploratory: {k.replace('_', ' ')} dates", v, MUTED, False)
        for k, v in ex["original_vs_new_dates"].items()]

    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rows) + 1.4))
    for i, (label, iv, colour, bold) in enumerate(rows):
        y = len(rows) - i
        ax.plot([iv["lo"], iv["hi"]], [y, y], color=colour, lw=3 if bold else 1.8)
        ax.plot(iv["point"], y, "o", color=colour, ms=7 if bold else 5)
        ax.text(-0.005 + min(r[1]["lo"] for r in rows), y, label, ha="right",
                va="center", fontsize=9, color=colour,
                fontweight="bold" if bold else "normal")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("ΔAUROC, 95% cluster-bootstrap interval over init dates")
    ax.set_title(f"S1: {p['text']} ({p['n_init_dates']} init dates)", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "ens_settlement"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}.png/.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `PYTHONPATH=src python scripts/plot_ens_settlement.py` → Expected: both files written; open the PNG and check labels do not overlap the intervals (widen `figsize` or the label offset if they do).

- [ ] **Step 3: Refresh the interval tables** (secondary c's continuity numbers flow into the README table from here)

Run: `PYTHONPATH=src python scripts/compute_confidence_intervals.py`
Expected: decision-band intervals unchanged (same predictions); `true_ens` now reports the full season. `git diff data/artifacts/confidence_intervals.json` shows changes only under `true_ens`.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_ens_settlement.py data/artifacts/ens_settlement.json data/artifacts/confidence_intervals.json docs/figures/ens_settlement.png docs/figures/ens_settlement.svg
git commit -m "Settle S1: <verdict text from ens_settlement.json>"
```

---

### Task 9: State the verdict everywhere, the way the registration committed to

**Files:**
- Modify: `DECISIONS.md` (append D-025)
- Modify: `README.md` (headline table's true-ENS row and footnote; "Uncertainty" margins table; "The honest reading" paragraph)
- Modify: `FRONTEND_LOGIC.md` §8
- Modify: `docs/FRONTEND_BUILT.md` (§2 landing claim; §1 test count)
- Modify: `docs/FIGURES.md` (add the settlement figure)
- Create: `tests/test_published_numbers.py`

- [ ] **Step 1: Write the failing consistency test** — `tests/test_published_numbers.py`

```python
"""The README and the decision log state the S1 verdict the data gives.

Written because the previous ENS margin lived in five places (README, two
decision entries, the landing page, a logic file) and nothing checked they
agreed with each other or with the code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402

SETTLEMENT = config.ARTIFACTS / "ens_settlement.json"
pytestmark = pytest.mark.skipif(not SETTLEMENT.exists(), reason="S1 not settled yet")


def _interval() -> str:
    p = json.loads(SETTLEMENT.read_text())["primary"]
    return f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"


@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_settled_interval(doc):
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert _interval() in text, f"{doc} does not state the settled interval {_interval()}"


def test_readme_no_longer_leads_with_the_40_date_margin_as_current():
    text = (config.ROOT / "README.md").read_text(encoding="utf-8")
    p = json.loads(SETTLEMENT.read_text())["primary"]
    assert f"{p['n_init_dates']} init dates" in text or f"{p['n_init_dates']} dates" in text
```

Run: `PYTHONPATH=src python -m pytest tests/test_published_numbers.py -q` → Expected: FAIL (README still states the 40-date number).

- [ ] **Step 2: Apply the consequences row for the recorded verdict** (spec §9), writing the settled interval as `+x.xxxx [+lo, +hi]` exactly as `_interval()` formats it:

- **README.md** — in the headline section, replace the paragraph starting "So the margin over a *real* operational ensemble is" and the "**The honest reading.**" paragraph with the verdict's sentence from spec §9, the settled interval, "122 ENS init dates fetched; N used (dates with Day 3–7 labels)", secondary (b)'s result in one sentence, and a link to `docs/PREREGISTRATION_S1.md` and `docs/figures/ens_settlement.png`. In the uncertainty margins table, replace the `model − true IFS ENS, raw spread (6,732 rows, 40 dates)` row with the settled row and keep the 40-date row labelled "(superseded: 40-date subsample)".
- **DECISIONS.md** — append `## D-025 — S1: <verdict text> — LOCKED|DISCLOSED` (LOCKED for `model_better`, DISCLOSED otherwise) containing: the registration commit hash (`git log -1 --format=%h -- docs/PREREGISTRATION_S1.md`), the fetch summary, the primary interval, all three secondaries, the exploratory original-vs-new comparison, the audit discrepancies and their fixes, and the consequence for S2–S5 from spec §9.
- **FRONTEND_LOGIC.md §8** — move or rewrite the ENS bullet per spec §9 (may-claim for `model_better`; the plain statement otherwise), citing the settled interval and 122-date fetch.
- **docs/FRONTEND_BUILT.md** — replace §2's "Every claim on the page reads live from `/api/convergence`" with the accurate statement (convergence chart from `/api/convergence`; headline numbers, ENS sentence and refusal chart from `/api/metrics`), and record that the earlier wording was false until S1. Update the §1 test count to the new total.
- **docs/FIGURES.md** — add a section for `ens_settlement.png` (what each row is; which is primary).

- [ ] **Step 3: Run the consistency test and the full suite**

Run: `PYTHONPATH=src python -m pytest tests/ -q` → Expected: all pass, including `test_published_numbers.py`.

- [ ] **Step 4: Commit**

```bash
git add DECISIONS.md README.md FRONTEND_LOGIC.md docs/FRONTEND_BUILT.md docs/FIGURES.md tests/test_published_numbers.py .github/workflows/ci.yml
git commit -m "Record D-025 and state the S1 verdict wherever the margin is claimed"
```

(Add `tests/test_published_numbers.py` to the CI list in the same commit — `ens_settlement.json` is committed, so it runs in CI.)

---

### Task 10: Final audit and corrections (requested by the user)

**Files:** whatever the audit finds.

- [ ] **Step 1: Re-run the audit on legacy data** — `PYTHONPATH=src python scripts/audit_s1.py` → Expected: still reproduces (the shared loader in legacy mode must be unaffected by the shards).
- [ ] **Step 2: Cross-check every place the verdict appears**: `ens_settlement.json` vs README vs D-025 vs FRONTEND_LOGIC §8 vs the rendered landing page (browser, `?v=` cache-buster) vs the figure title. Every number to 4 decimals where stated to 4, and the same verdict wording.
- [ ] **Step 3: Check the frozen inputs are untouched** — `git diff --stat b14c2b1 -- data/artifacts/bust_model.joblib data/processed/dataset.parquet` → Expected: empty.
- [ ] **Step 4: Run every gate locally** — full pytest; the CI air-gap origin scan; `bandit -r src/ -ll`. Expected: all green.
- [ ] **Step 5: Fix every discrepancy found**, each with a test where one can guard it, then commit (`git commit -m "Fix <discrepancy> found in the S1 audit"`).
- [ ] **Step 6: Push and watch CI** — `git push`, `gh run watch <id> --exit-status` → Expected: 5/5 green.
- [ ] **Step 7: Report** — the verdict, the audit findings and fixes, and anything left open.

---

## Self-review

- **Spec coverage:** §3 facts → Task 3 audit checks; §4 scope (2022 all leads + 2021) → Task 7; §5.1 frozen inputs → Tasks 3, 4, 6; §5.2 primary → Task 4 `primary`, Task 8 Step 1; §5.3 a/b/c → Task 4 `secondary_*` and script; §5.4 exploratory → script's `exploratory`; §6 pipeline → Tasks 1, 2, 7; §7 audit → Task 3; §8 outputs → Tasks 4, 5, 8; §9 consequences → Task 9; §10 tests → Tasks 1–5, 9; §11 failure handling → Tasks 2, 4, 5; §12 commit order → Tasks 1–5 (machinery), 6 (registration pushed), 7 (fetch), 8–9 (verdict).
- **Placeholders:** the only fill-ins are values produced by earlier steps (hashes from `s1_audit.json`, the verdict from `ens_settlement.json`), each with its exact source.
- **Consistency:** `comparison_rows(ds, ens, lead_days)`, `load_ens(ens_dir, include_shards)`, `guard(prereg, files, repo)`, `primary(y, p_model, comparator, clusters, n_boot, seed)` are used with the same signatures everywhere. Task 5's landing wording uses `<strong>is not</strong>`; the test asserts the same string.
