# Pre-registration — B4: a combiner fitted on Days 1–2

**Registered:** 2026-10-06, before any Day 1–2-fitted combiner was computed.
**Code:** `src/fbd/evaluate/refine.py`, `scripts/score_b3_b4.py`,
`scripts/audit_b3_b4.py`, committed before this file.
**Design (committed first):** `docs/superpowers/specs/2026-10-06-b3-b4-combiner-refinements-design.md`.

## 1. The question

> Does a combiner of the served form, fitted on the validation year's Day 1–2
> rows, outrank the served combiner (fitted on Days 3–7 and extrapolated) on
> Day 1–2 test rows across 2019–2022?

D-030 measured that, in 2022 at Days 1–2, the model alone ranks slightly
better than the served combination (−0.002, −0.010): one year, no intervals.

## 2. Reproduced before registration

`scripts/audit_b3_b4.py` (`data/artifacts/b3_b4_audit.json`): the script's
served combiner reproduces `combination.json`'s Day 3–7 AUROC exactly in every
fold:

| fold | AUROC |
|---|---|
| 2019 | 0.817951 |
| 2020 | 0.851478 |
| 2021 | 0.832798 |
| 2022 | 0.856146 |

The audit fitted no candidate and scored no Day 1–2 test row.

## 3. The rule

| | |
|---|---|
| rows | per fold, the test year's Day 1–2 rows with a label and ENS spread |
| incumbent | S1c's combiner (`COMBINER_PARAMS`) on the XGBoost fold model's uncalibrated probability and log1p ENS spread, fitted on the validation year's Day 3–7 rows |
| candidate | the same, fitted on the validation year's Day 1–2 rows |
| primary | mean over 2019–2022 of the within-year AUROC margin, candidate − incumbent; stratified cluster bootstrap over init dates, 10,000 resamples, seed 20260919, two-sided 95% |
| verdict | lower > 0 → **a Day 1–2 fit outranks the extrapolation**; upper < 0 → **the extrapolation outranks it**; otherwise **not distinguishable** |
| secondary | per year (2,000); candidate − the calibrated XGBoost model alone |

## 4. Consequences, committed in the design

- **Outranks:** a lead-split combiner (Days 1–2 and 3–10) is adopted at the
  next store regeneration, and the served claim may extend to Days 1–2.
- **Otherwise:** the extrapolation stays, and the claim stays on Days 3–7.

## 5. Registration block

Read by `scripts/score_b3_b4.py`. Do not edit after registration.

```registration
backtest_sha256: 0e8f23ec784240200d45ce33652f8f23ec5f736db94fec163d9d74196f1e4534
combination_sha256: e4af6751f9fe0752bb9b2f6b76908f42614eb8d7860bdb9546f8a9c8ff679ad1
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
required_ens_dates: 122
```
