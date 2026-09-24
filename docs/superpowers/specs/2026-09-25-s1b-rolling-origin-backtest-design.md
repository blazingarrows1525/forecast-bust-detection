# S1b — Rolling-origin backtest — design

**Date:** 2026-09-25
**Status:** approved in chat (primary rule A; three design sections); awaiting spec review
**Follows:** S1 (`docs/superpowers/specs/2026-09-23-s1-settle-ens-design.md`, D-025)

---

## 1. The question

> Does the model's edge over a real 50-member ensemble hold in seasons other
> than 2022?

S1 settled 2022: **+0.0316 AUROC [+0.0141, +0.0485]** over raw IFS ENS spread,
120 init dates, registered before the data (D-025). Every surface now carries
the same limit: *one season*. D-022 already named the fix, a rolling-origin
backtest, because a single test year cannot say whether 0.840 and the margin
are properties of the model or of 2022.

## 2. Facts verified before designing

| fact | source |
|---|---|
| Split is fixed in config: train 2016–2020, val 2021, test 2022 | `config.py:55-57` |
| ENS store covers JJAS 2018–2022; 2021 and 2022 are complete on disk (122 each) | S1 fetch, `E.dates_by_year` |
| 2019 and 2020 have 21 ENS dates each (legacy every-6) → **≈202 dates to fetch**, ≈20 s each measured in S1 | `data/raw/wb2/ens/` |
| Hyperparameters are fixed defaults in `BustModel.fit` (600 trees, depth 5, lr 0.04, …); no early stopping — `eval_set` is only logged; `scale_pos_weight` is computed from the training rows | `model/train.py:112-138` |
| Calibration: isotonic on the validation year | `model/train.py:143-151` |
| `build_dataset.py` writes to `data/processed/dataset.parquet` unconditionally — **the frozen S1 input** | `scripts/build_dataset.py:22,110` |

**Every fit that depends on which years are "training"** — each must use the
fold's years, or the fold leaks:

| # | fit | where | today |
|---|---|---|---|
| 1 | bust-label error threshold P95(subdivision, month) | `labels/bust.py` `error_thresholds` | `config.TRAIN_YEARS` by default |
| 2 | forecast climatology (mean, obs P90) | `features/forecast.py:98` | default, and `build()` calls it with no years |
| 3 | climatological bust rate feature | `features/forecast.py:121` | same |
| 4 | national ERA5 index standardisation | `features/era5.py:141-144` | reads `config.TRAIN_YEARS` directly |
| 5 | regime local-field standardisation within subdivision | `regime/classify.py:86,97` | **all dates, test years included — an existing leak** |
| 6 | model fit / isotonic calibration | `model/train.py` | train split / val split |
| 7 | lagged-proxy isotonic calibration (secondary only) | `model/baselines.py` `SpreadBaseline` | fit rows passed in |

Fit 5 is a leak in the shipped pipeline, not just a backtest concern: the
2022 model's regime features were standardised with 2022 in the statistics.
D-012 found regime features add no measurable skill, so the effect is likely
small; this design measures it rather than assuming it.

## 3. Folds

Expanding window, one rebuilt dataset and one model per fold:

| fold | train | calibrate | test |
|---|---|---|---|
| 2019 | 2016–2017 | 2018 | 2019 |
| 2020 | 2016–2018 | 2019 | 2020 |
| 2021 | 2016–2019 | 2020 | 2021 |
| 2022 | 2016–2020 | 2021 | 2022 |

- Every fit in §2 uses the fold's training years (1–5, 7) or its train / val
  split (6). Labels therefore differ slightly between folds; within a fold, the
  model and every comparator are scored on the same labels.
- Hyperparameters are the `BustModel` defaults, unchanged; their SHA-256 is
  registered. No per-fold tuning.
- As in S1, the model scores every test row, including rows the served product
  would refuse as out-of-distribution. No OOD detector is fitted.
- Fold 2019 trains on two seasons, so its model is the weakest. That biases
  against replication and is stated with the results.

**Two build modes.** `legacy` reproduces today's pipeline exactly (fit 5 over
all dates). `strict` fits 5 on the fold's training years. All four folds run
in `strict`. `legacy` exists only for the reproduction check in §7.

## 4. The registered analysis

Lives in `docs/PREREGISTRATION_S1B.md`, committed with `scripts/backtest.py`
and pushed before the 2019–2020 fetch runs.

### 4.1 Primary — one test, one verdict

| | |
|---|---|
| quantity | mean over **2019, 2020, 2021** of the within-year margin AUROC(fold model) − AUROC(ENS comparator) |
| rows | each test year's Day 3–7 rows with a label and ENS spread (`comparison_rows`, the S1 rule) |
| comparator | per year, the stronger by AUROC of raw and relative ENS spread — the S1 rule, conservative |
| interval | cluster bootstrap over init dates, **stratified by year** (dates resampled within each year); paired; 10,000 resamples; seed 20260919; 95% percentile |
| requires | all 122 ENS init dates on disk for each of 2019, 2020, 2021, 2022 |
| verdict | lower > 0 → **the edge replicates in 2019–2021**; upper < 0 → **ENS spread outranks the model in 2019–2021**; otherwise → **not distinguishable in 2019–2021** |

2022 is **not** in the primary: S1 has already seen it, and counting it again
would tilt the result toward "replicates". Its fold is reported beside the
primary.

Why a mean of within-year margins rather than one pooled AUROC: each fold has
its own model and calibration and each year its own bust rate, so a pooled
AUROC partly rewards separating years rather than busts.

**Per-year rule, binding whatever the primary says:** every year's margin is
reported with its own interval (2,000 resamples), and any year whose upper
bound is below zero is stated plainly as "ENS spread outranks the model in
<year>".

### 4.2 Secondary — reported, cannot overturn the primary

- **Lagged proxy.** Per year and averaged over 2019–2021: model − the lagged
  proxy, isotonic-calibrated on the fold's validation year exactly as the
  README headline table does it. Same stratified bootstrap.

### 4.3 Exploratory — labelled, no claims drawn

- Margin by month within each year. S1 found the 2022 edge in June–July and
  none detectable in August–September; this asks whether that recurs.
- **Regime-leak size:** on 2022, the strict fold-2022 model against the frozen
  `bust_model.joblib`, paired ΔAUROC.

### 4.4 Out of scope

Calibrated-ENS and model + ENS secondaries (fold 2019 would need 2018 ENS, and
model + ENS belongs to S3), new features, tuning, changing the shipped model.

## 5. Consequences, committed before the data

| primary | README / FRONTEND_LOGIC §8 | D-026 |
|---|---|---|
| lower > 0 | "outranks a real 50-member ensemble in 2022 and on average over 2019–2021", both intervals; the "one season" limit is removed | LOCKED |
| contains 0 | the S1 claim stays, qualified everywhere as "2022; not distinguishable in 2019–2021 (+x [lo, hi])" | DISCLOSED |
| upper < 0 | the S1 claim is restated as 2022-specific everywhere; the product question is reopened before S2–S5 | DISCLOSED |

The per-year rule in §4.1 applies in every row. If the regime leak (§4.3)
turns out material, it is recorded as its own finding; the shipped model is not
changed inside S1b.

## 6. Pipeline changes

- `train_years` is threaded explicitly through fits 1–5 (a parameter, never a
  patched `config`). Defaults stay as today, so every existing caller behaves
  identically.
- Fits 4 and 5 move into small pure helpers (standardise given fit rows) so
  they can be tested in CI without xarray.
- `build_dataset.py` gains `--fold YEAR`, `--mode legacy|strict` and writes fold
  datasets to `data/processed/backtest/fold_<YEAR>_<mode>.parquet`
  (gitignored). Without flags it behaves exactly as today. A test asserts no
  fold path can equal `dataset.parquet`.
- Fold models are saved beside their datasets (gitignored); their hashes and
  the fold datasets' hashes go into the output JSON.
- `fbd.evaluate.uncertainty` gains a stratified paired-mean bootstrap.
- `scripts/backtest.py`: guards → build the four strict folds → train →
  score → primary, secondary, exploratory → `data/artifacts/backtest.json`.
- `scripts/fetch_ens.py` is reused unchanged (`--year 2019`, `--year 2020`).

## 7. Audit (runs first, before registration)

1. **Legacy reproduction:** building with the default years in `legacy` mode
   reproduces `dataset.parquet` row for row, every column. **A mismatch stops
   S1b.**
2. **Retrain reproduction:** training on that legacy dataset gives a 2022
   decision-band AUROC within **0.001** of the frozen model's 0.8400. Within
   tolerance but not identical → recorded in the registration with the exact
   difference. Outside tolerance → S1b stops: the training pipeline would not
   be the one that produced the published numbers.
3. **Proxy reproduction:** in the same retrained fold-2022 legacy run, the
   point estimate of model − lagged proxy is within 0.001 of the published
   +0.0821. Same rule.
4. Hashes of `dataset.parquet`, `bust_model.joblib` (both still untouched) and
   the hyperparameters.

Results go into the registration as "reproduced before registration".

## 8. Blinding, stated honestly

- No fold model exists yet; no out-of-sample margin for 2019, 2020 or 2021 has
  been computed by anyone.
- 2021 ENS is already on disk and was used in S1's secondaries, but only with
  the frozen model, which was calibrated on 2021 (in-sample for calibration).
- The frozen model's in-sample scores on 2019–2021 exist implicitly in
  training; they say nothing about out-of-sample skill.
- 2019 and 2020 ENS beyond the every-6 legacy dates is not yet fetched.

## 9. Outputs

- `data/artifacts/backtest.json` — registration hash, hyperparameter hash, fold
  and model hashes, per-fold rows and dates, primary, per-year, secondary,
  exploratory.
- `docs/figures/backtest.{png,svg}` — per-year margins and the primary mean on
  one axis, zero marked, 2022 drawn distinctly as "seen in S1".
- D-026; README section "Does it hold in other years?"; FRONTEND_LOGIC §8;
  `test_published_numbers.py` extended to the backtest interval.
- The landing page and API are unchanged in S1b.

## 10. Tests

All synthetic, CI-safe (no sklearn / xgboost / xarray in modules CI imports).

| file | proves |
|---|---|
| `tests/test_backtest_folds.py` | fold table: expanding, no overlap, test after train and val; no fold path equals `dataset.parquet` |
| `tests/test_fold_leakage.py` | poison tests for fits 1–5: extreme values in test-year rows leave every fitted statistic unchanged; `legacy` mode differs from `strict` only in fit 5 |
| `tests/test_uncertainty.py` (extended) | stratified bootstrap: same seed → same interval; each resample keeps every year's date count; mean of per-stratum differences |
| `tests/test_backtest_guard.py` | refuses on uncommitted registration, hyperparameter-hash mismatch, any year < 122 ENS dates; verdict mapping for all three outcomes |

The audit's reproduction checks (§7) need raw data and run locally.

## 11. Failure handling

- Fetch: resumable, as in S1; the guard refuses on any incomplete year.
- Legacy or retrain reproduction fails → stop and report, nothing registered.
- A fold fails to build or train → `backtest.py` exits non-zero before any
  statistic is computed; no partial `backtest.json`.
- Changed hyperparameters, registration or code → guard refusal.

## 12. Commit order

```
1  train_years threading, pure helpers, fold build flags, leakage tests   (no fold scored)
2  audit: legacy + retrain + proxy reproduction                           (recorded)
3  stratified bootstrap, backtest.py, guards, tests
4  PREREGISTRATION_S1B.md, CI list — pushed                               (before the data)
5  fetch 2019, 2020                                                        (background)
6  run backtest.py once → json, figure
7  D-026, README, FRONTEND_LOGIC §8, consistency test; final audit; push
```
