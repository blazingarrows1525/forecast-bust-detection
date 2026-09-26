# Pre-registration — S1b: does the edge over a real ensemble hold outside 2022?

**Registered:** 2026-09-25, before any of the 2019–2020 ENS data beyond the
legacy every-6 files was fetched, and before any fold model was scored.
**Code:** `scripts/backtest.py`, `scripts/build_dataset.py`,
`src/fbd/evaluate/{folds,backtest_stats,registration,uncertainty}.py`,
`src/fbd/features/standardise.py`, `src/fbd/model/params.py`, committed before
this file.
**Design:** `docs/superpowers/specs/2026-09-25-s1b-rolling-origin-backtest-design.md`

This file exists so the analysis cannot be chosen after the data is seen.
`backtest.py` refuses to run unless this file is committed unchanged, the
frozen dataset, model and hyperparameters still match the hashes at the end,
and all four test years' ENS seasons are complete.

---

## 1. The question

> Does the model's edge over a real 50-member operational ensemble (IFS ENS)
> hold in seasons other than 2022?

S1 settled 2022 alone: **+0.0316 AUROC [+0.0141, +0.0485]** over raw ENS
spread, 120 init dates, registered before the data (D-025). Every surface now
carries the same limit, *one season*. A single test year cannot say whether the
edge is a property of the model or of 2022 (D-022).

## 2. Reproduced before registration

`scripts/audit_s1b.py`, output `data/artifacts/s1b_audit.json`.

| check | published / frozen | reproduced | difference | ok |
|---|---|---|---|---|
| fold 2022 rebuilt in legacy mode vs `dataset.parquet` | 279,650 rows × 93 columns | identical, compared exactly | none | yes |
| retrained model, 2022 Day 3–7 AUROC (52 features, identical to the frozen model's) | 0.839966 | 0.839966 | **0.0** | yes |
| model − lagged proxy, 2022 Day 3–7 | +0.082107 | +0.082107 | **0.0** | yes |

The tolerance was 0.001; both differences are exactly zero, because training is
deterministic. The fold machinery is therefore the pipeline that produced every
published number, not an approximation of it.

A blind smoke test also built the 2019 strict fold and trained its model
without scoring any test row: 52 features identical to the frozen model's;
2020–2022 excluded; the fold's own label thresholds, national standardisation
and regime scaling in effect on its validation year.

## 3. Folds

Expanding window, one rebuilt dataset and one model per fold:

| fold | train | calibrate | test |
|---|---|---|---|
| 2019 | 2016–2017 | 2018 | 2019 |
| 2020 | 2016–2018 | 2019 | 2020 |
| 2021 | 2016–2019 | 2020 | 2021 |
| 2022 | 2016–2020 | 2021 | 2022 |

- Every year-dependent fit uses the fold's training years: the bust-label error
  threshold, forecast climatology, the climatological bust rate, national ERA5
  standardisation, and (in `strict` mode) the regime standardisation; the model
  is fitted on the training years and isotonic-calibrated on the validation year.
- **Mode `strict` for every fold.** The published pipeline standardised regime
  fields over every date, test years included; `strict` fits them on training
  years only. `legacy` exists only for the reproduction in §2.
- Hyperparameters are the `BustModel` defaults, unchanged, hash below. No
  per-fold tuning.
- The fold model scores every test row, including rows the served product would
  refuse as out-of-distribution. This measures ranking skill, not the served
  subset.
- Fold 2019 trains on two seasons and is the weakest model. That biases
  against replication.

## 4. Primary — one test, one verdict

| | |
|---|---|
| quantity | mean over **2019, 2020, 2021** of the within-year margin AUROC(fold model) − AUROC(ENS comparator) |
| rows | each test year's Day 3–7 rows with a label and ENS spread (`comparison_rows`, the S1 rule) |
| comparator | per year, the stronger by AUROC of raw and relative ENS spread — the S1 rule, conservative |
| interval | cluster bootstrap over init dates, **stratified by year** (dates resampled within each year); paired; **10,000** resamples; seed **20260919**; 95% percentile |
| requires | all 122 ENS init dates on disk for each of 2019, 2020, 2021, 2022 |
| verdict | lower > 0 → **the edge replicates in 2019–2021**; upper < 0 → **ENS spread outranks the model in 2019–2021**; otherwise → **not distinguishable in 2019–2021** |

2022 is **not** in the primary: S1 has already seen it, and counting it again
would tilt the result toward "replicates". Its fold is reported beside the
primary. A mean of within-year margins is used, not one pooled AUROC, because
each fold has its own model and calibration and each year its own bust rate.

**Per-year rule, binding whatever the primary says:** every year's margin is
reported with its own interval (**2,000** resamples, same seed), and any year
whose upper bound is below zero is stated plainly as "ENS spread outranks the
model in <year>".

## 5. Secondary — reported, cannot overturn the primary

- **Lagged proxy.** Per year and averaged over 2019–2021: model − the lagged
  proxy, isotonic-calibrated on the fold's validation year exactly as the README
  headline table does it. Mean with the same stratified bootstrap (10,000);
  per year with 2,000.

## 6. Exploratory — labelled, no claims drawn

2,000 resamples each, same seed.

- Margin by month within each year. S1 found the 2022 edge in June–July and none
  detectable in August–September; this asks whether that recurs.
- **Regime-leak size:** on 2022, the strict fold-2022 model against the frozen
  `bust_model.joblib`, paired ΔAUROC on identical rows.

**Out of scope:** calibrated-ENS and model + ENS secondaries (fold 2019 would
need 2018 ENS; model + ENS belongs to S3), new features, tuning, and any change
to the shipped model.

## 7. Blinding

- No fold model has been scored; no out-of-sample margin for 2019, 2020 or 2021
  has been computed by anyone.
- 2021 ENS is already on disk and was used in S1's secondaries, but only with
  the frozen model, which was calibrated on 2021 (in-sample for calibration).
- The frozen model's in-sample scores on 2019–2021 exist implicitly in its
  training; they say nothing about out-of-sample skill.
- 2019 and 2020 ENS beyond the 21 legacy every-6 dates each is not fetched.

## 8. Consequences, committed now

| primary | README / FRONTEND_LOGIC §8 | D-026 |
|---|---|---|
| lower > 0 | "outranks a real 50-member ensemble in 2022 and on average over 2019–2021", both intervals; the "one season" limit is removed | LOCKED |
| contains 0 | the S1 claim stays, qualified everywhere as "2022; not distinguishable in 2019–2021" with the interval | DISCLOSED |
| upper < 0 | the S1 claim is restated as 2022-specific everywhere; the product question is reopened before S2–S5 | DISCLOSED |

The per-year rule applies in every row. If the regime leak turns out material
it is recorded as its own finding; the shipped model is not changed inside S1b.

## 9. What would invalidate this

- Changing the dataset, the model, the hyperparameters, this file or the
  analysis code after the 2019–2020 data is on disk. The hashes and git history
  make each visible, and `backtest.py` refuses the first four.
- Reporting the primary with any test year short of 122 ENS dates.
- Promoting a per-year, secondary or exploratory result to the headline.

## 10. Registration block

Read by `backtest.py`. Do not edit after registration.

```registration
dataset_sha256: 9f8b4063a14aff844a24071d0d780b27445c9913b8d82e0974e33d906ae0a306
model_sha256: 577578e7391e42eac5bf5390b8f713ce811957e77ac30073ff9d3d15dfb3e020
params_sha256: 205d618ace558e0bc2a72b48feb5ac5f0455604602fb582931d80080d86403cf
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
