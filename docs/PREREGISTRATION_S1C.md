# Pre-registration — S1c: does the model plus ENS spread outrank ENS spread alone?

**Registered:** 2026-09-25, before any model + ENS combination was scored on
2019, 2020 or 2021.
**Code:** `scripts/combine_ens.py`, `src/fbd/evaluate/combine.py`,
`src/fbd/model/params.py` (`COMBINER_PARAMS`), committed before this file
(`4bb3414`).
**Follows:** D-026 (the model alone does not repeatably beat ENS), D-027,
D-028 (neither does the MLP), and S1's secondary (b) (D-025).

`combine_ens.py` refuses to run unless this file is committed unchanged,
`backtest.json` and `candidates/mlp.json` match the hashes below (they in turn
pin every fold dataset and model), the combiner's parameters match, and every
test and validation year's ENS season is complete.

---

## 1. The question

> Forecasters already have the ensemble. Does adding the model to its spread
> rank busts better than the spread alone, across held-out seasons?

Nothing tested so far beats ENS spread repeatably on its own. In 2022, S1's
secondary (b) found that a combination of the model and the spread beat both
parts. Every S1b fold's validation year (2018–2021) now has complete ENS, so a
combiner can be fitted inside every fold.

## 2. Reproduced before registration

`scripts/audit_s1c.py`, output `data/artifacts/s1c_audit.json`. It scores no
combination on any test year.

| check | result |
|---|---|
| each fold's XGBoost and ENS AUROC (vs `backtest.json`) and MLP AUROC (vs `mlp.json`) | **exact**, all twelve |
| combiner rows per fold | the validation year only (2018, 2019, 2020, 2021), Day 3–7, 20,060 rows over 120 dates each, ENS complete |
| two combiner fits on fold 2019's validation rows | identical |

## 3. The combiner, frozen

Logistic regression (lbfgs, C = 1e6, at most 1,000 iterations) on two inputs:
the log-odds of the fold model's **uncalibrated** probability (clipped to
[1e-6, 1 − 1e-6]) and log(1 + ENS spread). Fitted per fold on the validation
year's Day 3–7 rows with a label and ENS spread.

The uncalibrated probability is used because the model's isotonic calibration
was itself fitted on the validation year: calibrated inputs would be in-sample
for the combiner. The raw score of a model trained on the training years is out
of sample there. (S1's secondary (b) used calibrated inputs and said so.)

Models inside: the S1b strict XGBoost fold models for the primary; the S3a MLP
fold models for a secondary. Neither is retrained.

## 4. Primary — one test, one verdict

| | |
|---|---|
| quantity | mean over **2019, 2020, 2021** of the within-year margin AUROC(XGBoost + ENS combination) − AUROC(ENS comparator) |
| rows | each fold's S1b comparison rows (Day 3–7, label and ENS spread), 20,060 per year |
| comparator | per year, the stronger by AUROC of raw and relative ENS spread — the S1 rule |
| interval | stratified cluster bootstrap over init dates; paired; **10,000** resamples; seed **20260919**; 95% percentile |
| verdict | lower > 0 → **model + ENS spread outranks ENS spread alone on average over 2019–2021**; upper < 0 → **ENS spread alone outranks the model + ENS combination**; otherwise → **the model adds nothing detectable to ENS spread** |

2022 is reported beside the primary, not in it: S1 has already seen a
combination on 2022. One primary, so no multiplicity correction.

**Per-year rule:** each year's margin with its own interval (2,000 resamples);
a year whose upper bound is below zero is stated as "ENS spread outranks the
combination in <year>".

## 5. Secondary — reported, cannot change the verdict

- Combination − XGBoost alone: mean over 2019–2021 (10,000) and per year
  (2,000). Does the ensemble add anything to the model?
- MLP + ENS combination − ENS spread: mean over 2019–2021 (10,000) and per year
  (2,000), the same combiner on the MLP's uncalibrated probability.

## 6. What is known, stated honestly

- The components are known: per year, the model against ENS (D-026) and the MLP
  against ENS (D-027, D-028). The combination's performance is not: no
  combination has been scored on 2019–2021.
- S1's secondary (b) found on 2022 that a combination beat both parts; that is
  why 2022 is outside the primary.
- The audit looked only at reproduction, row selection and determinism.

## 7. Consequences, committed now

| primary | README / `FRONTEND_LOGIC.md` §8 | D-029 |
|---|---|---|
| lower > 0 | may say "combining the model with ENS spread outranks the spread alone on average over 2019–2021", with the interval; the combination becomes the candidate to serve (changing the served product is its own step) | LOCKED |
| contains 0 | "the model adds nothing detectable to what ENS spread already says, on average over 2019–2021", stated plainly; the product question ("use the spread") is reopened | DISCLOSED |
| upper < 0 | "ENS spread alone outranks the combination", stated plainly | DISCLOSED |

## 8. What would invalidate this

- Changing this file, `backtest.json`, `mlp.json`, any fold file, or the
  combiner after the first run.
- Reporting the primary with any fitting or test year short of 122 ENS dates.
- Promoting a per-year or secondary result to the headline.

## 9. Registration block

```registration
backtest_sha256: 0e8f23ec784240200d45ce33652f8f23ec5f736db94fec163d9d74196f1e4534
mlp_json_sha256: b8d8b911cee6f2e157c9e27045c9aae1dd87f0bd58ea3e1669ab755ac9b33ae1
combiner_params_sha256: 835dcb7ed5adfe0b23784e47ed16e7b021ce7c0a434dc6d359c8f190db46e780
primary_years: 2019,2020,2021
years: 2019,2020,2021,2022
required_ens_dates: 122
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
lead_days: 3,4,5,6,7
alpha: 0.05
```
