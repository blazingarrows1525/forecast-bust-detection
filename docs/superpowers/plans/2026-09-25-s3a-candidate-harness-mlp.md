# S3a — Candidate Harness and MLP Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a promotion rule for three declared candidate model families, then score the first — an MLP on the incumbent's own 52 features — against the S1b XGBoost fold models across 2019–2022, once.

**Architecture:** Pure modules hold the slate and rule (`fbd/evaluate/promotion.py`) and the frozen MLP hyperparameters (`fbd/model/params.py`). `fbd/model/mlp.py` implements the `BustModel` interface with torch imported lazily. `scripts/promote.py` reuses the S1b fold datasets and incumbent models (hash-checked against `backtest.json`), the S1b row rule and the stratified bootstrap.

**Tech Stack:** Python 3.10, pandas, numpy, scikit-learn, xgboost, PyTorch (CPU build, local only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-s3a-candidate-harness-mlp-design.md`

## Global Constraints

- Commit messages carry **no** `Co-Authored-By` or Claude attribution lines (standing user instruction).
- No SIH / SIH26079 on any product surface.
- Slate **mlp, temporal, spatial**; family alpha **0.05**; per-candidate alpha **0.05/3**; years **2019–2022**; ENS secondary years **2019–2021**; seed **20260919**; primary and ENS-mean resamples **10,000**; per-year and exploratory **2,000**; lead days **3–7**.
- MLP frozen per spec §5: hidden [128, 64], ReLU, dropout 0.2, AdamW lr 1e-3 wd 1e-4, batch 1,024, ≤60 epochs, patience 5 on validation-year weighted BCE, seeds 20260920–20260924, isotonic calibration on the validation year, CPU, deterministic, 8 threads, float32.
- Torch goes in `requirements.txt` only. Nothing CI or the serving image imports may import torch at module level.
- `dataset.parquet`, `bust_model.joblib`, `backtest.json` and the S1b fold files are never modified.
- The 279 existing tests keep passing.

## File Structure

| file | status | responsibility |
|---|---|---|
| `src/fbd/model/params.py` | modify | add `MLP_PARAMS` |
| `src/fbd/evaluate/promotion.py` | create | slate, alphas, verdict, text, candidate check (pure) |
| `src/fbd/evaluate/registration.py` | modify | `year_statement(..., whom="the model")` |
| `src/fbd/evaluate/backtest_stats.py` | modify | `mean_margin(..., alpha=0.05)`, `per_year(..., whom=)` |
| `src/fbd/model/mlp.py` | create | `Preprocessor`, `MLPModel` |
| `scripts/audit_s3a.py` | create | incumbent reproduction, MLP determinism, smoke test |
| `scripts/promote.py` | create | the harness |
| `scripts/plot_candidate.py` | create | the figure |
| `docs/PREREGISTRATION_S3.md` | create | the registration |
| `requirements.txt` | modify | `torch` |
| `tests/test_promotion.py` | create | CI |
| `tests/test_mlp.py` | create | skips without torch |
| `tests/test_backtest_stats.py` | modify | alpha widens the interval |
| `tests/test_published_numbers.py` | modify | README and D-027 state the MLP interval |

---

### Task 1: The rule and the frozen hyperparameters (pure)

**Files:** Modify `src/fbd/model/params.py`, `src/fbd/evaluate/registration.py`; create `src/fbd/evaluate/promotion.py`; test `tests/test_promotion.py`.

**Interfaces — produces:** `MLP_PARAMS: dict`; `SLATE`, `DISPLAY`, `ALPHA_FAMILY`, `YEARS`, `ENS_YEARS`; `alpha_per_candidate() -> float`; `verdict(lo, hi) -> "promoted"|"incumbent_better"|"indistinguishable"`; `verdict_text(v, name) -> str`; `check_candidate(name, reg) -> None` (raises `RegistrationError`); `R.year_statement(year, lo, hi, who="ENS spread", whom="the model")`.

- [ ] **Step 1: Failing tests** — `tests/test_promotion.py`

```python
"""S3 promotion rule: pure, run in CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.evaluate import promotion as PM  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402


def test_the_slate_is_declared_in_full():
    assert PM.SLATE == ("mlp", "temporal", "spatial")
    assert set(PM.DISPLAY) == set(PM.SLATE)


def test_bonferroni_over_the_whole_slate():
    assert PM.ALPHA_FAMILY == 0.05
    assert PM.alpha_per_candidate() == pytest.approx(0.05 / 3)


def test_years():
    assert PM.YEARS == (2019, 2020, 2021, 2022)
    assert PM.ENS_YEARS == (2019, 2020, 2021)


@pytest.mark.parametrize("lo, hi, v", [(0.001, 0.02, "promoted"),
                                       (-0.02, -0.001, "incumbent_better"),
                                       (-0.01, 0.01, "indistinguishable"),
                                       (0.0, 0.01, "indistinguishable")])
def test_verdict(lo, hi, v):
    assert PM.verdict(lo, hi) == v
    text = PM.verdict_text(v, "mlp")
    assert "MLP" in text and "2019–2022" in text


def test_check_candidate_refuses_undeclared_and_unregistered():
    reg = {"mlp_params_sha256": "x"}
    PM.check_candidate("mlp", reg)
    with pytest.raises(R.RegistrationError, match="not in the declared slate"):
        PM.check_candidate("lstm", reg)
    with pytest.raises(R.RegistrationError, match="no registered parameter hash"):
        PM.check_candidate("temporal", reg)


def test_mlp_params_are_frozen_and_hashable():
    p = PR.MLP_PARAMS
    assert p["hidden"] == [128, 64] and p["dropout"] == 0.2 and len(p["seeds"]) == 5
    json.dumps(p)  # serialisable, so the hash is well defined
    assert PR.params_sha256(p) == PR.params_sha256(dict(p))
    assert PR.params_sha256(dict(p, dropout=0.3)) != PR.params_sha256(p)


def test_year_statement_names_the_loser():
    assert R.year_statement(2019, -0.05, -0.01, whom="the MLP") == \
        "ENS spread outranks the MLP in 2019"
    assert R.year_statement(2019, -0.05, -0.01) == "ENS spread outranks the model in 2019"
```

- [ ] **Step 2: Run** `PYTHONPATH=src python -m pytest tests/test_promotion.py -q` → FAIL (no `promotion` module).

- [ ] **Step 3: Implement.** Append to `src/fbd/model/params.py`:

```python
#: S3a candidate, frozen before any run (spec §5, docs/PREREGISTRATION_S3.md).
MLP_PARAMS = dict(
    hidden=[128, 64],
    activation="relu",
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=1024,
    max_epochs=60,
    patience=5,
    seeds=[20260920, 20260921, 20260922, 20260923, 20260924],
    threads=8,
    dtype="float32",
    device="cpu",
    missing="training-rows median + indicator for features missing in training",
    scaling="training-rows mean and sd after imputation; sd 0 -> 1",
    loss="bce with pos_weight = negatives / positives on training rows",
    stopping="validation-year weighted bce, restore best epoch",
    calibration="isotonic on the validation year, on the seed-averaged probability",
)
```

In `src/fbd/evaluate/registration.py` replace `year_statement`:

```python
def year_statement(year: int, lo: float, hi: float, who: str = "ENS spread",
                   whom: str = "the model"):
    """The registered per-year rule: a year the comparator wins is said plainly."""
    return f"{who} outranks {whom} in {year}" if hi < 0 else None
```

Create `src/fbd/evaluate/promotion.py`:

```python
"""The S3 promotion rule, registered in docs/PREREGISTRATION_S3.md. Pure.

Three candidate families are declared before any exists. Each is judged
against the same incumbent -- the S1b XGBoost fold models -- on the mean
within-year AUROC margin over 2019-2022, at a Bonferroni-corrected level so
that scoring them weeks apart cannot inflate the chance of a false promotion.
"""
from __future__ import annotations

from fbd.evaluate.registration import RegistrationError

SLATE = ("mlp", "temporal", "spatial")
DISPLAY = {"mlp": "MLP", "temporal": "temporal model", "spatial": "spatial model"}
ALPHA_FAMILY = 0.05
YEARS = (2019, 2020, 2021, 2022)
ENS_YEARS = (2019, 2020, 2021)


def alpha_per_candidate() -> float:
    return ALPHA_FAMILY / len(SLATE)


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "promoted"
    if hi < 0:
        return "incumbent_better"
    return "indistinguishable"


def verdict_text(v: str, name: str) -> str:
    who = f"the {DISPLAY[name]}"
    return {
        "promoted": f"{who} outranks the XGBoost incumbent across 2019–2022",
        "incumbent_better": f"the XGBoost incumbent outranks {who} across 2019–2022",
        "indistinguishable": f"{who} is not distinguishable from the XGBoost incumbent "
                             "across 2019–2022",
    }[v]


def check_candidate(name: str, reg: dict) -> None:
    if name not in SLATE:
        raise RegistrationError(f"{name!r} is not in the declared slate {SLATE}; "
                                "a new candidate needs a new registration")
    if f"{name}_params_sha256" not in reg:
        raise RegistrationError(f"{name!r} has no registered parameter hash; commit its "
                                "addendum before scoring it")
```

- [ ] **Step 4: Run** `PYTHONPATH=src python -m pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit** `git add src/fbd/model/params.py src/fbd/evaluate/promotion.py src/fbd/evaluate/registration.py tests/test_promotion.py && git commit -m "Add the S3 promotion rule and the frozen MLP hyperparameters"`

---

### Task 2: Install torch and build the MLP

**Files:** Modify `requirements.txt`; create `src/fbd/model/mlp.py`; test `tests/test_mlp.py`.

**Interfaces — produces:** `Preprocessor.fit(train, features) -> Preprocessor` with `features, medians, flagged, mean, std` and `transform(df) -> np.ndarray[float32]`; `MLPModel(params=MLP_PARAMS)` with `fit(train, val, features)`, `predict_seeds(df) -> (n_seeds, n) float64`, `predict_raw(df)`, `predict_proba(df)`, `save(path)`, `MLPModel.load(path)`, attributes `features`, `prep`, `nets`, `calibrator`, `history`.

- [ ] **Step 1: Install** (approved in chat: CPU build, official index):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: a version string and `False`. Add to `requirements.txt` under a comment:

```
# S3 candidate models (local only; not in the serving image or CI). CPU build:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
torch>=2.2
```

- [ ] **Step 2: Failing tests** — `tests/test_mlp.py`

```python
"""The S3a MLP: reproducible, and fitted on training rows only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd.model.mlp import MLPModel, Preprocessor  # noqa: E402
from fbd.model.params import MLP_PARAMS  # noqa: E402

FAST = dict(MLP_PARAMS, max_epochs=3, seeds=[1, 2])
FEATS = ["a", "b", "c"]


def _frame(n, seed, nan_in=("b",)):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=n), "c": rng.normal(size=n)})
    df["bust"] = (a + rng.normal(0, 0.5, n) > 1.2).astype(float)
    for c in nan_in:
        df.loc[df.index[::7], c] = np.nan
    return df


def test_same_seed_same_fit():
    tr, va = _frame(600, 0), _frame(200, 1)
    one = MLPModel(params=FAST).fit(tr, va, FEATS)
    two = MLPModel(params=FAST).fit(tr, va, FEATS)
    assert np.array_equal(one.predict_raw(va), two.predict_raw(va))


def test_preprocessing_is_fitted_on_training_rows_only():
    tr, va = _frame(600, 0), _frame(200, 1)
    poisoned = va.copy()
    poisoned[FEATS] = 1e6
    a = MLPModel(params=FAST).fit(tr, va, FEATS).prep
    b = MLPModel(params=FAST).fit(tr, poisoned, FEATS).prep
    assert a.medians == b.medians and a.flagged == b.flagged
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.std, b.std)


def test_missing_indicators_are_for_features_missing_in_training():
    tr = _frame(300, 0, nan_in=("b",))
    va = _frame(100, 1, nan_in=("b", "c"))
    prep = Preprocessor.fit(tr, FEATS)
    assert prep.flagged == ["b"]
    assert prep.transform(va).shape == (100, 4)
    assert not np.isnan(prep.transform(va)).any()


def test_probabilities_are_calibrated_and_monotone_in_the_raw_score():
    tr, va = _frame(600, 0), _frame(200, 1)
    m = MLPModel(params=FAST).fit(tr, va, FEATS)
    raw, p = m.predict_raw(va), m.predict_proba(va)
    assert ((p >= 0) & (p <= 1)).all()
    order = np.argsort(raw, kind="stable")
    assert (np.diff(p[order]) >= -1e-12).all()
    assert m.predict_seeds(va).shape == (2, 200)


def test_save_and_load_round_trip(tmp_path):
    tr, va = _frame(600, 0), _frame(200, 1)
    m = MLPModel(params=FAST).fit(tr, va, FEATS)
    path = tmp_path / "mlp.joblib"
    m.save(path)
    back = MLPModel.load(path)
    assert np.array_equal(m.predict_proba(va), back.predict_proba(va))
    assert back.history == m.history
```

- [ ] **Step 3: Run** `PYTHONPATH=src python -m pytest tests/test_mlp.py -q` → FAIL (`No module named 'fbd.model.mlp'`).

- [ ] **Step 4: Implement** — `src/fbd/model/mlp.py`

```python
"""S3a candidate: a small MLP on the incumbent's own features.

Everything that could make it differ from XGBoost other than the architecture
is held fixed: the same features, the same class weighting, the same isotonic
calibration on the validation year. Frozen in ``fbd.model.params.MLP_PARAMS``
and registered in docs/PREREGISTRATION_S3.md. Torch is imported inside the
functions that need it, so importing this module needs only numpy, pandas and
scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fbd.model.params import MLP_PARAMS


@dataclass
class Preprocessor:
    """Median imputation + missing indicators + standardisation, fitted on training rows."""
    features: list
    medians: dict
    flagged: list
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def _stack(df: pd.DataFrame, features, medians, flagged) -> np.ndarray:
        X = df[features].astype("float64")
        filled = X.fillna(medians).to_numpy(dtype="float64")
        flags = X[flagged].isna().to_numpy(dtype="float64")
        return np.hstack([filled, flags])

    @classmethod
    def fit(cls, train: pd.DataFrame, features) -> "Preprocessor":
        X = train[list(features)].astype("float64")
        medians = {c: (0.0 if pd.isna(v) else float(v)) for c, v in X.median().items()}
        flagged = [c for c in features if X[c].isna().any()]
        Z = cls._stack(train, list(features), medians, flagged)
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std == 0] = 1.0
        return cls(list(features), medians, flagged, mean, std)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        Z = self._stack(df, self.features, self.medians, self.flagged)
        return ((Z - self.mean) / self.std).astype("float32")


def _net(n_in: int, p: dict):
    import torch.nn as nn

    layers, width = [], n_in
    for h in p["hidden"]:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p["dropout"])]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _train_one(Xtr, ytr, Xva, yva, seed: int, p: dict, pos_weight: float):
    import torch

    torch.manual_seed(seed)
    net = _net(Xtr.shape[1], p)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, p["max_epochs"] + 1):
        net.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(order), p["batch_size"]):
            idx = order[i:i + p["batch_size"]]
            opt.zero_grad()
            loss_fn(net(xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(xv).squeeze(1), yv))
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= p["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"seed": seed, "best_epoch": best_epoch, "val_loss": best}


@dataclass
class MLPModel:
    params: dict = field(default_factory=lambda: dict(MLP_PARAMS))
    features: list = field(default_factory=list)
    prep: Preprocessor | None = None
    nets: list = field(default_factory=list)
    calibrator: IsotonicRegression | None = None
    history: list = field(default_factory=list)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features) -> "MLPModel":
        import torch

        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.params["threads"])
        self.features = list(features)
        tr = train.dropna(subset=["bust"])
        va = val.dropna(subset=["bust"])
        self.prep = Preprocessor.fit(tr, self.features)
        Xtr, Xva = self.prep.transform(tr), self.prep.transform(va)
        ytr = tr.bust.to_numpy(np.float32)
        yva = va.bust.to_numpy(np.float32)
        pos_weight = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)

        self.nets, self.history = [], []
        for seed in self.params["seeds"]:
            net, info = _train_one(Xtr, ytr, Xva, yva, seed, self.params, pos_weight)
            self.nets.append(net)
            self.history.append(info)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(self.predict_raw(va), yva.astype(float))
        return self

    def predict_seeds(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        x = torch.from_numpy(self.prep.transform(df))
        out = []
        with torch.no_grad():
            for net in self.nets:
                out.append(torch.sigmoid(net(x).squeeze(1)).numpy().astype("float64"))
        return np.vstack(out)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_seeds(df).mean(axis=0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df))

    def save(self, path) -> None:
        import joblib

        joblib.dump({"params": self.params, "features": self.features, "prep": self.prep,
                     "states": [n.state_dict() for n in self.nets],
                     "calibrator": self.calibrator, "history": self.history}, path)

    @classmethod
    def load(cls, path) -> "MLPModel":
        import joblib

        d = joblib.load(path)
        m = cls(params=d["params"], features=d["features"], prep=d["prep"],
                calibrator=d["calibrator"], history=d["history"])
        n_in = len(m.features) + len(m.prep.flagged)
        for state in d["states"]:
            net = _net(n_in, m.params)
            net.load_state_dict(state)
            net.eval()
            m.nets.append(net)
        return m
```

- [ ] **Step 5: Run** `PYTHONPATH=src python -m pytest tests/ -q` → all pass (torch present locally).

- [ ] **Step 6: Commit** `git add requirements.txt src/fbd/model/mlp.py tests/test_mlp.py && git commit -m "Add the S3a MLP: frozen, deterministic on CPU, fitted on training rows only"`

---

### Task 3: Statistics hooks, the harness and its figure

**Files:** Modify `src/fbd/evaluate/backtest_stats.py`, `tests/test_backtest_stats.py`; create `scripts/promote.py`, `scripts/plot_candidate.py`.

**Interfaces — consumes:** Tasks 1–2; `F.fold_path`, `F.model_path`, `E.load_ens`, `E.dates_by_year`, `E.comparison_rows`, `S.choose_comparator`, `T.BustModel`, `T.split_frames`, `U.paired_difference`, `M.auroc`. **Produces:** `data/artifacts/candidates/<name>.json` with `registration_sha256`, `backtest_sha256`, `params_sha256`, `folds` (per year: `candidate_auroc`, `incumbent_auroc`, `ens_auroc`, `comparator`, `n_rows`, `n_init_dates`, `history`, `model_sha256`), `primary` (+`verdict`, `text`, `alpha`), `per_year`, `secondary.ens.{mean, per_year}`, `exploratory.{by_month, seed_auroc}`.

- [ ] **Step 1: Failing test** — append to `tests/test_backtest_stats.py`:

```python
def test_a_smaller_alpha_widens_the_interval_around_the_same_point():
    rows = _rows()
    wide = BS.mean_margin(rows, "p_model", "p_ens", [2019, 2020, 2021], 300, 1, alpha=0.05 / 3)
    base = BS.mean_margin(rows, "p_model", "p_ens", [2019, 2020, 2021], 300, 1)
    assert wide["point"] == base["point"]
    assert wide["lo"] <= base["lo"] and wide["hi"] >= base["hi"]
    assert wide["alpha"] == pytest.approx(0.05 / 3)
```

Run → FAIL (`unexpected keyword argument 'alpha'`).

- [ ] **Step 2: Implement** in `src/fbd/evaluate/backtest_stats.py`:

```python
def mean_margin(rows: pd.DataFrame, a: str, b: str, years, n_boot: int, seed: int,
                alpha: float = 0.05) -> dict:
    r = rows[rows.year.isin(list(years))]
    iv = U.stratified_mean_difference(
        M.auroc, r.bust.to_numpy(float), r[a].to_numpy(float), r[b].to_numpy(float),
        r.init_date.to_numpy(), r.year.to_numpy(), n_boot=n_boot, seed=seed, alpha=alpha)
    return {**_interval(iv), "years": [int(y) for y in years]}
```

and give `per_year` a `whom` argument passed through:

```python
def per_year(rows: pd.DataFrame, a: str, b: str, n_boot: int, seed: int,
             who: str = "ENS spread", whom: str = "the model") -> dict:
    out = {}
    for year, r in rows.groupby("year"):
        iv = year_margin(r, a, b, n_boot, seed)
        iv["statement"] = R.year_statement(int(year), iv["lo"], iv["hi"], who=who, whom=whom)
        iv["n_rows"] = int(len(r))
        iv["n_init_dates"] = int(r.init_date.nunique())
        out[str(int(year))] = iv
    return out
```

- [ ] **Step 3: Implement** — `scripts/promote.py`

```python
"""Score one declared S3 candidate against the S1b XGBoost incumbent, as registered.

    PYTHONPATH=src python scripts/promote.py --candidate mlp

Refuses unless docs/PREREGISTRATION_S3.md is committed as-is, backtest.json and
every fold dataset and incumbent model match their hashes, the candidate is in
the declared slate with a registered parameter hash that the code still has,
and every ENS year is complete. Writes data/artifacts/candidates/<name>.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import promotion as PM  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import registration as R  # noqa: E402
from fbd.model import params as PR  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_S3.md"
BACKTEST = config.ARTIFACTS / "backtest.json"
OUT_DIR = config.ARTIFACTS / "candidates"
#: slate name -> (module, class, frozen parameters). S3b and S3c add theirs.
CANDIDATES = {"mlp": ("fbd.model.mlp", "MLPModel", PR.MLP_PARAMS)}


def guards(name: str):
    reg = R.guard(PREREG, {"backtest_sha256": BACKTEST}, repo=ROOT)
    PM.check_candidate(name, reg)
    if name not in CANDIDATES:
        raise R.RegistrationError(f"{name!r} is declared but not implemented yet")
    got = PR.params_sha256(CANDIDATES[name][2])
    if reg[f"{name}_params_sha256"] != got:
        raise R.RegistrationError(f"{name} hyperparameters changed: registered "
                                  f"{reg[f'{name}_params_sha256']}, the code has {got}")
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    for y in PM.YEARS:
        info = bt["folds"][str(y)]
        for path, key in ((F.fold_path(y, "strict"), "dataset_sha256"),
                          (F.model_path(y, "strict"), "model_sha256")):
            if not path.exists() or P.sha256_file(path) != info[key]:
                raise R.RegistrationError(f"fold {y}: {path.name} does not match backtest.json; "
                                          "re-run scripts/backtest.py's fold build")
    ens = E.load_ens()
    have = E.dates_by_year(ens)
    for y in PM.YEARS:
        R.check_complete(have.get(y, 0), int(reg["required_ens_dates"]), year=y)
    return reg, ens, bt


def score_fold(name: str, year: int, ens, lead_days, bt: dict):
    from fbd.evaluate import metrics as M
    from fbd.evaluate import settle as S
    from fbd.model import train as T

    ds = pd.read_parquet(F.fold_path(year, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    incumbent = T.BustModel.load(F.model_path(year, "strict"))
    module, cls, _p = CANDIDATES[name]
    t0 = time.time()
    cand = getattr(importlib.import_module(module), cls)().fit(tr, va, list(incumbent.features))
    fit_s = time.time() - t0
    mpath = F.FOLD_DIR / f"candidate_{name}_{year}.joblib"
    cand.save(mpath)

    test, _fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    y = test.bust.to_numpy(float)
    comparator, p_ens = S.choose_comparator(y, test.ens_spread.to_numpy(float),
                                            test.ens_spread_rel.to_numpy(float))
    init = pd.to_datetime(test.init_date)
    rows = pd.DataFrame({
        "year": year,
        "init_date": init.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": init.dt.month.to_numpy(),
        "bust": y,
        "p_cand": cand.predict_proba(test),
        "p_inc": incumbent.predict_proba(test),
        "p_ens": p_ens,
    })
    inc_auc = M.auroc(y, rows.p_inc)
    if inc_auc != bt["folds"][str(year)]["model_auroc"]:
        raise RuntimeError(f"fold {year}: incumbent AUROC {inc_auc!r} does not reproduce "
                           f"backtest.json {bt['folds'][str(year)]['model_auroc']!r}")
    seeds = cand.predict_seeds(test) if hasattr(cand, "predict_seeds") else None
    info = {"candidate_auroc": M.auroc(y, rows.p_cand), "incumbent_auroc": inc_auc,
            "ens_auroc": M.auroc(y, p_ens), "comparator": comparator,
            "n_rows": int(len(rows)), "n_init_dates": int(rows.init_date.nunique()),
            "fit_seconds": round(fit_s, 1), "history": getattr(cand, "history", None),
            "model_sha256": P.sha256_file(mpath),
            "seed_auroc": ([M.auroc(y, s) for s in seeds] if seeds is not None else None)}
    return rows, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    name = ap.parse_args().candidate
    t0 = time.time()
    try:
        reg, ens, bt = guards(name)
    except (R.RegistrationError, FileNotFoundError, KeyError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    seed, n_boot, n_year = int(reg["seed"]), int(reg["n_boot"]), int(reg["n_boot_per_year"])
    lead_days = [int(x) for x in reg["lead_days"].split(",")]
    alpha = PM.alpha_per_candidate()
    whom = f"the {PM.DISPLAY[name]}"

    from fbd.evaluate import backtest_stats as BS

    frames, folds = [], {}
    for y in PM.YEARS:
        print(f"fold {y}: fit {name}, score ...", flush=True)
        rows, info = score_fold(name, y, ens, lead_days, bt)
        frames.append(rows)
        folds[str(y)] = info
        print(f"  {info['n_rows']:,} rows; {name} {info['candidate_auroc']:.4f}, "
              f"incumbent {info['incumbent_auroc']:.4f}, ENS {info['ens_auroc']:.4f} "
              f"({info['fit_seconds']}s)", flush=True)
    rows = pd.concat(frames, ignore_index=True)

    prim = BS.mean_margin(rows, "p_cand", "p_inc", PM.YEARS, n_boot, seed, alpha=alpha)
    v = PM.verdict(prim["lo"], prim["hi"])
    prim.update(verdict=v, text=PM.verdict_text(v, name))
    print(f"primary, mean over {list(PM.YEARS)} at {1 - alpha:.2%}: {prim['point']:+.4f} "
          f"[{prim['lo']:+.4f}, {prim['hi']:+.4f}] -> {prim['text']}", flush=True)
    per_year = BS.per_year(rows, "p_cand", "p_inc", n_year, seed,
                           who="the XGBoost incumbent", whom=whom)
    for y, iv in per_year.items():
        print(f"  {y}: {iv['point']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]", flush=True)

    print("secondary: against ENS spread ...", flush=True)
    secondary = {"ens": {
        "mean": BS.mean_margin(rows, "p_cand", "p_ens", PM.ENS_YEARS, n_boot, seed),
        "per_year": BS.per_year(rows, "p_cand", "p_ens", n_year, seed, whom=whom)}}
    exploratory = {
        "by_month": BS.by_month(rows, "p_cand", "p_inc", n_year, seed),
        "seed_auroc": {y: f["seed_auroc"] for y, f in folds.items()},
        "n_boot": n_year, "note": "exploratory: no claims are drawn from these"}

    payload = {
        "candidate": name,
        "registration_sha256": P.sha256_file(PREREG),
        "backtest_sha256": P.sha256_file(BACKTEST),
        "params_sha256": PR.params_sha256(CANDIDATES[name][2]),
        "row_rule": "S1b comparison_rows per fold; both models score every row",
        "folds": folds, "primary": prim, "per_year": per_year,
        "secondary": secondary, "exploratory": exploratory,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Implement** — `scripts/plot_candidate.py`

```python
"""S3: one candidate's margin over the XGBoost incumbent, per year and on average.

    PYTHONPATH=src python scripts/plot_candidate.py --candidate mlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import promotion as PM  # noqa: E402

INK, MUTED, ACCENT = "#1f2933", "#8b98a5", "#2a6fb0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    name = ap.parse_args().candidate
    s = json.loads((config.ARTIFACTS / "candidates" / f"{name}.json").read_text(encoding="utf-8"))
    label = PM.DISPLAY[name]
    p, per, ens = s["primary"], s["per_year"], s["secondary"]["ens"]
    rows = [(f"PRIMARY  mean 2019–2022, {label} − XGBoost ({1 - p['alpha']:.2%})", p, ACCENT, True)]
    rows += [(f"{y}: {label} − XGBoost", per[y], INK, False) for y in sorted(per, key=int)]
    rows.append((f"secondary: mean 2019–2021, {label} − ENS spread", ens["mean"], MUTED, False))

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    left = min(r[1]["lo"] for r in rows)
    for i, (text, iv, colour, bold) in enumerate(rows):
        y = len(rows) - i
        ax.plot([iv["lo"], iv["hi"]], [y, y], color=colour, lw=3 if bold else 1.8)
        ax.plot(iv["point"], y, "o", color=colour, ms=7 if bold else 5)
        ax.text(left - 0.005, y, text, ha="right", va="center", fontsize=9, color=colour,
                fontweight="bold" if bold else "normal")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([])
    ax.set_xlabel("ΔAUROC, cluster-bootstrap interval over init dates")
    ax.set_title(f"S3: {p['text']}", fontsize=10, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / f"candidate_{name}"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {out}.png/.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Verify it refuses before registration** — `PYTHONPATH=src python scripts/promote.py --candidate mlp` → `REFUSED: PREREGISTRATION_S3.md is not committed as-is…`, exit 2. Run the full suite → all pass.

- [ ] **Step 6: Commit** `git add src/fbd/evaluate/backtest_stats.py tests/test_backtest_stats.py scripts/promote.py scripts/plot_candidate.py && git commit -m "Add the S3 harness and its figure; it refuses until registered"`

---

### Task 4: Audit

**Files:** Create `scripts/audit_s3a.py`, artifact `data/artifacts/s3a_audit.json`.

- [ ] **Step 1: Implement** — `scripts/audit_s3a.py`

```python
"""S3a audit, before registration.

    PYTHONPATH=src python scripts/audit_s3a.py

1. The harness's view of the incumbent reproduces backtest.json exactly.
2. Two fits of the fold-2019 MLP give bit-identical validation-year output.
3. That MLP's validation-year AUROC is at least 0.60 (a defect check, not tuning).
Touches training and validation years only for 2 and 3.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fbd import config  # noqa: E402
from fbd.evaluate import ens as E  # noqa: E402
from fbd.evaluate import folds as F  # noqa: E402
from fbd.evaluate import metrics as M  # noqa: E402
from fbd.evaluate import provenance as P  # noqa: E402
from fbd.evaluate import settle as S  # noqa: E402
from fbd.model import params as PR  # noqa: E402
from fbd.model import train as T  # noqa: E402
from fbd.model.mlp import MLPModel  # noqa: E402

BACKTEST = config.ARTIFACTS / "backtest.json"
OUT = config.ARTIFACTS / "s3a_audit.json"
SMOKE_FLOOR = 0.60


def main() -> int:
    t0 = time.time()
    bt = json.loads(BACKTEST.read_text(encoding="utf-8"))
    ens = E.load_ens()
    checks = []

    print("1. incumbent reproduction ...", flush=True)
    worst = []
    for y in (2019, 2020, 2021, 2022):
        ds = pd.read_parquet(F.fold_path(y, "strict"))
        inc = T.BustModel.load(F.model_path(y, "strict"))
        test, _ = E.comparison_rows(ds, ens, lead_days=config.DECISION_BAND)
        yy = test.bust.to_numpy(float)
        _n, comp = S.choose_comparator(yy, test.ens_spread.to_numpy(float),
                                       test.ens_spread_rel.to_numpy(float))
        got = (M.auroc(yy, inc.predict_proba(test)), M.auroc(yy, comp))
        want = (bt["folds"][str(y)]["model_auroc"], bt["folds"][str(y)]["ens_auroc"])
        worst.append({"year": y, "model": [got[0], want[0]], "ens": [got[1], want[1]],
                      "exact": got == want})
        print(f"   {y}: model {got[0]:.6f} vs {want[0]:.6f}, ENS {got[1]:.6f} vs {want[1]:.6f}")
    checks.append({"check": "harness reproduces the S1b incumbent and ENS AUROC exactly",
                   "ok": all(w["exact"] for w in worst), "folds": worst})

    print("2. MLP determinism on fold 2019 (train 2016-2017, validation 2018) ...", flush=True)
    ds = pd.read_parquet(F.fold_path(2019, "strict"))
    tr, va, _te = (d.dropna(subset=["bust"]) for d in T.split_frames(ds))
    feats = list(T.BustModel.load(F.model_path(2019, "strict")).features)
    t1 = time.time()
    one = MLPModel().fit(tr, va, feats)
    fit_s = time.time() - t1
    two = MLPModel().fit(tr, va, feats)
    same = bool(np.array_equal(one.predict_raw(va), two.predict_raw(va)))
    checks.append({"check": "two fits give bit-identical validation-year output", "ok": same,
                   "fit_seconds": round(fit_s, 1), "history": one.history})
    print(f"   identical: {same}  ({fit_s:.0f}s per fit; best epochs "
          f"{[h['best_epoch'] for h in one.history]})", flush=True)

    auc = M.auroc(va.bust.to_numpy(float), one.predict_raw(va))
    checks.append({"check": f"validation-year AUROC at least {SMOKE_FLOOR}", "ok": auc >= SMOKE_FLOOR,
                   "validation_auroc_2018": auc})
    print(f"3. validation-year (2018) AUROC {auc:.4f}", flush=True)

    payload = {"checks": checks, "ok": all(c["ok"] for c in checks),
               "hashes": {"backtest_sha256": P.sha256_file(BACKTEST),
                          "mlp_params_sha256": PR.params_sha256(PR.MLP_PARAMS)}}
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{'AUDIT PASSED' if payload['ok'] else 'AUDIT FAILED -- stop, do not register'}"
          f"  ({time.time() - t0:.0f}s)  wrote {OUT}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run** (background) → `AUDIT PASSED`. On failure: stop, fix only the defect, never §5's values.
- [ ] **Step 3: Commit** `git add scripts/audit_s3a.py data/artifacts/s3a_audit.json && git commit -m "Audit S3a: the harness reproduces the incumbent; the MLP is deterministic"`

---

### Task 5: Register, then push

- [ ] **Step 1: Write** `docs/PREREGISTRATION_S3.md`: question (spec §1); audit results table from `s3a_audit.json`; slate and incumbent (§3); primary, secondary, exploratory (§3.1–3.3); the MLP specification (§5 table); blinding (§7); consequences (§8); invalidation; block:

````markdown
```registration
backtest_sha256: <hashes.backtest_sha256 from s3a_audit.json>
slate: mlp,temporal,spatial
alpha_family: 0.05
years: 2019,2020,2021,2022
ens_years: 2019,2020,2021
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
lead_days: 3,4,5,6,7
required_ens_dates: 122
mlp_params_sha256: <hashes.mlp_params_sha256 from s3a_audit.json>
```
````

- [ ] **Step 2: CI list** — add `tests/test_promotion.py` and `tests/test_mlp.py` to `.github/workflows/ci.yml`.
- [ ] **Step 3: Commit, prove the guard passes to the fit, push, watch CI**

```bash
git add docs/PREREGISTRATION_S3.md .github/workflows/ci.yml
git commit -m "Register S3: the promotion rule, the slate and the MLP, before any candidate is scored"
git push origin local-llm-and-image-slimming
gh run watch <id> --exit-status
```

Expected: 5/5 green. `test_mlp.py` skips in CI (no torch).

---

### Task 6: Score the MLP once

- [ ] **Step 1: Run** (background) `PYTHONPATH=src python scripts/promote.py --candidate mlp` → four fold lines, the primary line, exit 0, `wrote …candidates/mlp.json`. **Do not re-run with changed parameters.** A crash before the JSON is written: fix only the crash, commit it naming it, re-run.
- [ ] **Step 2: Plot** `PYTHONPATH=src python scripts/plot_candidate.py --candidate mlp`; check the PNG.
- [ ] **Step 3: Commit** `git add data/artifacts/candidates/mlp.json docs/figures/candidate_mlp.png docs/figures/candidate_mlp.svg && git commit -m "Score S3a: <primary text from mlp.json>"`

---

### Task 7: State it, the way the registration committed to; final audit

- [ ] **Step 1: Consistency test** — append to `tests/test_published_numbers.py`:

```python
MLP = config.ARTIFACTS / "candidates" / "mlp.json"


@pytest.mark.skipif(not MLP.exists(), reason="S3a not scored yet")
@pytest.mark.parametrize("doc", ["README.md", "DECISIONS.md"])
def test_doc_states_the_mlp_interval(doc):
    p = json.loads(MLP.read_text(encoding="utf-8"))["primary"]
    s = f"{p['point']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    text = (config.ROOT / doc).read_text(encoding="utf-8").replace("−", "-")
    assert s in text, f"{doc} does not state the S3a interval {s}"
```

- [ ] **Step 2: Apply the consequences row** (spec §8) for the recorded verdict:
  - **DECISIONS.md** — `## D-027 — S3a: <primary text> — LOCKED|DISCLOSED` (LOCKED only for `promoted`): registration commit, audit table, per-fold table (candidate, incumbent, ENS AUROC, fit seconds, best epochs), primary with its 98.33% interval, per-year margins, ENS secondary with any per-year statement, by-month and seed-spread exploratory, consequence for S3b/S3c.
  - **README.md** — a `### Other model families` subsection after "Does it hold in other years?": the primary sentence, interval, per-year table, links to `docs/PREREGISTRATION_S3.md` and `docs/figures/candidate_mlp.png`.
  - **FRONTEND_LOGIC.md §8** — replace "Any transformer, BERT, LSTM, Random Forest or LightGBM. **None exist in this codebase.**" with the §8 row for the verdict.
  - **docs/FIGURES.md** — `## 5. S3 candidates — candidate_mlp.png`.
- [ ] **Step 3:** full pytest → all pass. Commit `Record D-027 and state the S3a result`.
- [ ] **Step 4: Final audit** — re-run `audit_s1b.py` and `audit_s3a.py` (no JSON change); frozen inputs untouched (`git diff --stat b14c2b1 -- data/artifacts/bust_model.joblib data/processed/dataset.parquet`); cross-check the interval in every doc; gates (pytest, air-gap scan, socket import, bandit, secrets); fix discrepancies; push; CI 5/5; report.

---

## Self-review

- **Spec coverage:** §3 slate/rule → Tasks 1, 3, 5; §3.1 primary → Task 3 (`alpha`), Task 6; §3.2 secondary → Task 3; §3.3 exploratory → Task 3 (`by_month`, `seed_auroc`); §4 harness and guards → Task 3; §5 MLP → Tasks 1 (params), 2 (model); §6 audit → Task 4; §7 blinding → Task 5; §8 consequences → Task 7; §9 files → all; §10 tests → Tasks 1–3, 7; §11 failure handling → Tasks 3 (`REFUSED`, incumbent reproduction check), 4, 6; §12 commit order → Tasks 1–7.
- **Placeholders:** fill-ins are only values produced earlier (audit hashes, the verdict and interval from `mlp.json`), each with its source.
- **Consistency:** `mean_margin(..., alpha)`, `per_year(..., who, whom)`, `year_statement(..., who, whom)`, `verdict_text(v, name)`, `check_candidate(name, reg)`, `MLPModel.fit(train, val, features)`, `predict_seeds/raw/proba`, and the `rows` columns `year, init_date, month, bust, p_cand, p_inc, p_ens` are used identically across tasks.
