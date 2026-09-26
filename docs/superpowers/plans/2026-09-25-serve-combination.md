# Serve the Model + ENS Combination — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the D-029 combination (frozen model's uncalibrated probability + ENS spread) the number the product serves, with its two ingredients shown beside it, without breaking clones that still hold the v0.1.0 store.

**Architecture:** A pure-numpy module (`fbd.model.combined`) applies a combiner fitted once on 2021 and saved as `data/artifacts/combiner.json`. `generate_bulletins.py` writes store v0.2.0 with the combination in `bust_probability` and two new columns. The API, tools, metrics and pages read the new columns when present and fall back to null.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn (fitting only), FastAPI, SQLite, vanilla JS, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-serve-combination-design.md`

## Global Constraints

- Commit messages carry **no** `Co-Authored-By` or Claude attribution lines.
- No competition branding on product surfaces. Air-gap: pages add no external origin.
- Combiner: logistic regression on [logit of the frozen model's **uncalibrated** probability (clip 1e-6), log(1 + ENS spread)], fitted on **2021** Day 3–7 rows; applied to all leads.
- Store v0.2.0: `bust_probability` = combination (model's calibrated probability where ENS is missing, `data_quality` `ENS_UNAVAILABLE`); new columns appended: `model_probability`, `ens_spread`. OOD refusal unchanged.
- API tolerant of v0.1.0: absent columns → null.
- The landing combination sentence renders only when `served.combined` is true; no number hardcoded.
- Do not publish a Release asset without explicit approval in chat.
- `bust_model.joblib`, `dataset.parquet`, and every registered artifact are never modified.

---

### Task 1: The served combiner

**Files:** Create `src/fbd/model/combined.py`, `scripts/fit_combiner.py`, `tests/test_combined.py`; artifact `data/artifacts/combiner.json`.

**Produces:** `combined.load(path=COMBINER) -> dict`; `combined.apply(c, p_raw, spread) -> np.ndarray` (NaN spread → NaN); `combined.contribution(c, spread) -> np.ndarray`; `combined.ens_reason(spread_mm, contribution) -> str`; `COMBINER: Path`. `combiner.json` keys: `logit_p_model`, `log1p_ens_spread`, `intercept`, `ref_log1p_spread`, `fit_year`, `n_rows`, `model_sha256`.

- [ ] **Step 1: Failing tests** — `tests/test_combined.py`

```python
"""The served combination: pure arithmetic, readable reasons, honest fallback."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model import combined as K  # noqa: E402

C = {"logit_p_model": 0.5, "log1p_ens_spread": 1.0, "intercept": -4.0, "ref_log1p_spread": 2.0}


def test_apply_is_the_logistic_arithmetic():
    p = K.apply(C, [0.2], [np.expm1(2.0)])[0]
    z = 0.5 * np.log(0.2 / 0.8) + 1.0 * 2.0 - 4.0
    assert p == pytest.approx(1 / (1 + np.exp(-z)))


def test_missing_spread_gives_nan_so_the_caller_falls_back():
    assert np.isnan(K.apply(C, [0.2], [np.nan])[0])


def test_contribution_is_relative_to_the_fit_years_median():
    assert K.contribution(C, [np.expm1(2.0)])[0] == pytest.approx(0.0)
    assert K.contribution(C, [np.expm1(3.0)])[0] == pytest.approx(1.0)


@pytest.mark.parametrize("contrib, words", [(0.9, "high"), (-0.9, "low"), (0.05, "typical")])
def test_ensemble_reason_says_what_the_ensemble_did(contrib, words):
    line = K.ens_reason(9.4, contrib)
    assert "50-member ensemble spread" in line and words in line and "9.4 mm/day" in line
    if words == "high":
        assert "raises the bust odds 2.5x" in line
    if words == "low":
        assert "lowers the bust odds 2.5x" in line


def test_agrees_with_the_fitted_combiner():
    pytest.importorskip("sklearn")
    from fbd.evaluate.combine import Combiner

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    y = (a + b + rng.normal(0, 1, 500) > 1).astype(float)
    p_raw, spread = 1 / (1 + np.exp(-a)), np.expm1(np.clip(b + 3, 0, None))
    fitted = Combiner().fit(p_raw, spread, y)
    c = {**fitted.coefficients(), "ref_log1p_spread": 0.0}
    assert np.allclose(K.apply(c, p_raw, spread), fitted.predict_proba(p_raw, spread))
```

- [ ] **Step 2:** run → FAIL (no module).

- [ ] **Step 3: Implement** — `src/fbd/model/combined.py`

```python
"""The served model + ENS combination (D-029, D-030). Pure numpy.

Fitted once by scripts/fit_combiner.py on 2021, the frozen model's calibration
year, from the frozen model's *uncalibrated* probability and the 50-member ENS
spread. Its coefficients live in data/artifacts/combiner.json, so serving needs
no fitting library and the numbers can be read in review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fbd import config

COMBINER = config.ARTIFACTS / "combiner.json"
CLIP = 1e-6
TYPICAL = 0.1  # |log-odds contribution| below this reads as "typical"


def load(path: Path = COMBINER) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def apply(c: dict, p_raw, spread) -> np.ndarray:
    """P(bust) from the model's uncalibrated probability and ENS spread; NaN spread -> NaN."""
    z = (c["logit_p_model"] * _logit(p_raw)
         + c["log1p_ens_spread"] * np.log1p(np.asarray(spread, dtype=float))
         + c["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def contribution(c: dict, spread) -> np.ndarray:
    """Log-odds the ensemble adds against the fit year's median spread."""
    return c["log1p_ens_spread"] * (np.log1p(np.asarray(spread, dtype=float))
                                    - c["ref_log1p_spread"])


def ens_reason(spread_mm: float, contrib: float) -> str:
    """One reason line, in the same plain register as the TreeSHAP lines."""
    if abs(contrib) < TYPICAL:
        return (f"the 50-member ensemble spread is typical ({spread_mm:.1f} mm/day): "
                "little effect on the bust odds")
    level, verb = ("high", "raises") if contrib > 0 else ("low", "lowers")
    return (f"the 50-member ensemble spread is {level} ({spread_mm:.1f} mm/day): "
            f"{verb} the bust odds {np.exp(abs(contrib)):.1f}x")
```

- [ ] **Step 4: Implement** — `scripts/fit_combiner.py`

```python
"""Fit the served combiner once, on 2021, and write data/artifacts/combiner.json.

D-029's recipe on the served model: logistic regression on [logit of the frozen
model's uncalibrated probability, log(1 + ENS spread)] over 2021's Day 3-7 rows.

    PYTHONPATH=src python scripts/fit_combiner.py
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
from fbd.evaluate import combine as CB  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.model import combined as K  # noqa: E402
from fbd.model import train as T  # noqa: E402

MODEL = config.ARTIFACTS / "bust_model.joblib"


def main() -> int:
    ds = pd.read_parquet(config.PROCESSED / "dataset.parquet")
    val = CB.validation_rows(ds, E.load_ens(), lead_days=config.DECISION_BAND)
    years = sorted(pd.to_datetime(val.init_date).dt.year.unique())
    assert years == list(config.VAL_YEARS), years
    model = T.BustModel.load(MODEL)
    p_raw = model.predict_raw(val)
    spread = val.ens_spread.to_numpy(float)
    fit = CB.Combiner().fit(p_raw, spread, val.bust.to_numpy(float))
    out = {**fit.coefficients(),
           "ref_log1p_spread": float(np.median(np.log1p(spread))),
           "fit_year": int(years[0]), "n_rows": int(len(val)),
           "model_sha256": P.sha256_file(MODEL),
           "recipe": "D-029: logit(uncalibrated model probability), log1p(ENS spread); "
                     "Day 3-7 rows of the validation year"}
    K.COMBINER.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5:** run `fit_combiner.py`; run all tests; commit `Fit and serve-side apply the D-029 combiner`.

---

### Task 2: Store v0.2.0

**Files:** Modify `scripts/generate_bulletins.py`, `tests/conftest.py`.

- [ ] **Step 1:** In `SCHEMA`, append after `model_version TEXT NOT NULL,`: `model_probability REAL,` and `ens_spread REAL,` (before `PRIMARY KEY`).
- [ ] **Step 2:** `bagged_interval(..., transform=None)`: each bag's draw is `transform(m) if transform else m.predict_proba(df)`.
- [ ] **Step 3:** In `main()` after loading the model:

```python
    comb = K.load()
    ens = E.load_ens()
    target = target.merge(ens[E.KEY + ["ens_spread"]].assign(
        init_date=lambda d: pd.to_datetime(d.init_date)), on=E.KEY, how="left")
    spread = target.ens_spread.to_numpy(float)
    has_ens = np.isfinite(spread)
    p_model = model.predict_proba(target)
    prob = np.where(has_ens, K.apply(comb, model.predict_raw(target), spread), p_model)
    contrib = K.contribution(comb, spread)
```

(`target.init_date` is converted to datetime before the merge.) The interval uses
`transform=lambda m: np.where(has_ens, K.apply(comb, m.predict_raw(target), spread), m.predict_proba(target))`.
Each row's reasons: `([K.ens_reason(spread[i], contrib[i])] if has_ens[i] else []) + reason_lists[i]`;
`data_quality` = `"OK" if has_ens[i] else "ENS_UNAVAILABLE"`; the row tuple appends
`None if ood_hit else float(p_model[i])` and `float(spread[i]) if has_ens[i] else None`;
the INSERT uses 21 placeholders; `model_version` "0.2.0"; `meta` adds
`("combined", "1")` and `("combiner", json.dumps(comb))`.

- [ ] **Step 4:** `tests/conftest.py`: `_SCHEMA` gains the two columns; each `_ROWS` tuple gains
`model_probability, ens_spread` (0.66, 7.9 / 0.41, 3.1 / None, None). Add a `bulletin_store_v1` fixture that builds the old 19-column schema from `_SCHEMA_V1` and the first 19 values of each row.
- [ ] **Step 5:** full pytest; commit `Write store v0.2.0: the combination is the served probability`.

---

### Task 3: API, tools, metrics

**Files:** Modify `src/fbd/api/schema.py`, `src/fbd/api/app.py`, `src/fbd/genai/tools.py`; tests in `tests/test_metrics_endpoint.py`, `tests/test_combined_api.py` (new).

- [ ] **Step 1: Failing tests** — `tests/test_combined_api.py`

```python
"""The API serves the combination's ingredients, and tolerates the old store."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _client(db, monkeypatch):
    from fbd.api import app as app_module
    importlib.reload(app_module)
    monkeypatch.setattr(app_module, "DB", db)
    return TestClient(app_module.app)


def test_bulletin_carries_model_probability_and_spread(bulletin_store, monkeypatch):
    rows = _client(bulletin_store, monkeypatch).get(
        "/api/convergence", params={"region_id": "ASSAM_MEGHALAYA", "valid_date": "2022-06-17"}).json()
    assert rows[0]["model_probability"] == 0.66 and rows[0]["ens_spread_mm"] == 7.9


def test_old_store_reads_as_null(bulletin_store_v1, monkeypatch):
    rows = _client(bulletin_store_v1, monkeypatch).get(
        "/api/convergence", params={"region_id": "ASSAM_MEGHALAYA", "valid_date": "2022-06-17"}).json()
    assert rows[0]["model_probability"] is None and rows[0]["ens_spread_mm"] is None


def test_tools_ground_the_new_numbers(bulletin_store):
    from fbd.genai import tools
    _res, grounded = tools.dispatch("get_bulletin", {"region_id": "ASSAM_MEGHALAYA",
                                                     "init_date": "2022-06-14"})
    assert 0.66 in grounded and 7.9 in grounded
```

and in `tests/test_metrics_endpoint.py`: the `served` block reads `{"model_version": ..., "combined": ...}` from `meta` (null when no store or no `meta`), and `combination` is served from `combination.json`.

- [ ] **Step 2: Implement.** `schema.py`: `BustPrediction` gets `model_probability: float | None = Field(None, ge=0.0, le=1.0, description="the model alone, before the ensemble is combined in")` and `ens_spread_mm: float | None = Field(None, description="50-member ENS spread for this row, mm/day")`; `ReviewQueueItem` gets the same two optional fields. `app.py`: `_opt(r, key)` returns `r[key] if key in r.keys() else None`; `_to_prediction` sets both fields and `model_version=r["model_version"]`; the review queue sets both; `_served()` opens the store directly and returns `{"model_version": ..., "combined": meta.get("combined") == "1"}` or `None` on any `sqlite3.Error` or missing file; `metrics()` adds `served` and `combination`. `tools.py`: a `_optional_cols(con)` helper appends `, model_probability, ens_spread` to both SELECTs when the columns exist.
- [ ] **Step 3:** full pytest; commit `Serve the combination's ingredients through the API and tools`.

---

### Task 4: Pages

**Files:** Modify `web/landing.html`, `web/index.html`, `tests/test_web_pages.py`.

- [ ] **Step 1: Failing test** — append to `tests/test_web_pages.py`:

```python
@pytest.mark.skipif(not (WEB / LANDING).exists(), reason="no landing page")
def test_landing_states_the_combination_only_when_it_is_served():
    """D-029/D-030: FRONTEND_LOGIC section 8 allows the claim once the product shows it."""
    html = _read(LANDING)
    assert "served.combined" in html and "m.combination" in html
    for literal in ("+0.0244", "+0.024", "0.0178", "0.0308"):
        assert literal not in html, f"landing.html hardcodes {literal!r}"


def test_dashboard_shows_both_ingredients():
    html = _read("index.html")
    assert "model_probability" in html and "ens_spread_mm" in html
```

- [ ] **Step 2: Landing.** `ensClaimHTML` appends `combinationHTML(m)`, which returns "" unless `m.served && m.served.combined && m.combination && m.combination.primary`, else " Together, the model and the ensemble outrank the ensemble alone on average over {first–last year}: <strong>{point} [{lo}, {hi}]</strong> in a registered test, and that combination is the number this product serves." The convergence chart's "this model" label and caption read "model + ensemble" when `served.combined`.
- [ ] **Step 3: Dashboard.** The detail panel's sub-line reads "probability this forecast busts (model + ensemble)" when `r.model_probability != null`, and a line `model alone {pct} · ensemble spread {x.x} mm/day` is added under the interval.
- [ ] **Step 4:** full pytest; commit `Show the combination on the landing page and dashboard, gated on the store`.

---

### Task 5: Regenerate, verify, document

- [ ] **Step 1:** record the current refusal counts (`SELECT status, COUNT(*) … GROUP BY status`), then run `PYTHONPATH=src python scripts/generate_bulletins.py` (background). Expected: same row count (79,900) and identical refusal counts; `meta.combined = 1`.
- [ ] **Step 2:** `PYTHONPATH=src python scripts/plot_evaluation_figures.py`; record the new ECE, Brier, BSS and top-bin numbers it prints.
- [ ] **Step 3:** descriptive numbers for D-030 on 2022 Day 3–7 served rows: AUROC of `bust_probability` vs `model_probability` vs ENS spread; the demo case (Assam & Meghalaya, valid 2022-06-17, leads 7→4) with model alone, combination and ENS spread.
- [ ] **Step 4:** browser check: dashboard detail panel and landing sentence; no console errors.
- [ ] **Step 5:** docs: D-030; README ("What it does", "The two claims, drawn", "The demo case"); `FRONTEND_LOGIC.md` §8 (claim now permitted, as served); `docs/FIGURES.md`; `HANDOFF.md`; commit.
- [ ] **Step 6:** final gates (pytest, air-gap, socket, bandit, secrets, audits unchanged), push, CI.
- [ ] **Step 7:** ask to publish Release `v0.2.0` with the new store; on yes, `gh release create`, pin tag and SHA-256 in `fetch_release_artifacts.py`, commit, push.

---

## Self-review

- Spec §3 → Task 1; §4 → Task 2; §5 → Tasks 3–4; §6 → Task 5; §7 → Tasks 1–4; §8 → task order. The Release step is gated on approval (Task 5 Step 7).
- Names are consistent: `combined.load/apply/contribution/ens_reason`, `model_probability`, `ens_spread` (store) / `ens_spread_mm` (API), `served.combined`, `m.combination`.
