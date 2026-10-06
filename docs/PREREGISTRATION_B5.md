# Pre-registration — B5: spatial + ENS against the served XGBoost + ENS

**Registered:** 2026-10-06, before any spatial + ENS combination was computed
for a test year.
**Code:** `src/fbd/evaluate/b5.py`, `scripts/b5_combination.py`,
`scripts/audit_b5.py`, committed before this file.
**Design (committed first):** `docs/superpowers/specs/2026-10-06-b5-serving-decision-design.md`.
**Follows:** D-029 and D-030 (the served combination), D-032 (S3c) and its
addendum.

---

## 1. The question

> Does the S3c spatial model, combined with ENS spread exactly as the served
> XGBoost model is, outrank the served XGBoost + ENS combination across
> 2019–2022?

## 2. Reproduced before registration

`scripts/audit_b5.py`, output `data/artifacts/b5_audit.json`. Only published
numbers were recomputed. The spatial + ENS combination was not computed.

| fold | XGBoost (`backtest.json`) | spatial (`spatial.json`) | served combination (`combination.json`) |
|---|---|---|---|
| 2019 | 0.782907 exact | 0.814762 exact | 0.817951 exact |
| 2020 | 0.841478 exact | 0.853247 exact | 0.851478 exact |
| 2021 | 0.815429 exact | 0.849064 exact | 0.832798 exact |
| 2022 | 0.840829 exact | 0.827345 exact | 0.856146 exact |

The following files match their records:
- every fold dataset
- every XGBoost fold model
- every spatial fold model
- the 23 gridded inputs

The spatial fold models reproduce bit for bit when loaded on the GPU.

## 3. The candidate and the incumbent, frozen

| | incumbent (served) | candidate |
|---|---|---|
| base model | S1b XGBoost fold model | registered S3c primary (window and map) fold model |
| combiner input | the base model's **uncalibrated** probability (XGBoost raw; the spatial model's seed average before isotonic) | the same |
| combiner | S1c's logistic regression on [logit p_raw, log1p ENS spread], `COMBINER_PARAMS` (hash pinned in `combination.json`) | the same |
| fit rows | the fold's validation year, Day 3–7, with a label and ENS spread | the same rows |

The window-only variant is **not** the candidate. Preferring it on 2019–2022
would be choosing on test years (D-032 addendum).

## 4. Primary

The mean over **2019, 2020, 2021, 2022** of the within-year margin
AUROC(spatial + ENS) − AUROC(XGBoost + ENS). The rows are S1b's comparison
rows (20,060 a year). The interval is a stratified, paired cluster bootstrap
over init dates: **10,000** resamples, seed **20260919**, two-sided **95%**.

Verdict:
- lower > 0 → **spatial + ENS outranks the served combination**
- upper < 0 → **the served combination outranks spatial + ENS**
- otherwise → **not distinguishable**

## 5. Secondary — reported, cannot decide

- Per-year margins (2,000 resamples, 95%).
- Spatial + ENS − ENS spread alone, as the mean over 2019–2021 (S1c's years)
  and per year.
- Spatial + ENS − the spatial model alone, as the mean over 2019–2022 and per
  year.
- Brier skill and expected calibration error of both combinations, per fold.
- Both combiners' coefficients per fold.

## 6. Consequences, committed in the design and restated

| verdict | recorded as |
|---|---|
| spatial + ENS outranks | the spatial model is **eligible** to be served. Serving it is a separate, designed change: reasons (TreeSHAP does not apply to a CNN), the OOD detector's inputs, and a store regenerated offline (the serving path stays CPU-only and precomputed). The served number is unchanged until that ships. |
| not distinguishable | the served product stays XGBoost + ENS |
| the served combination outranks | the served product stays XGBoost + ENS, stated plainly |

## 7. Blinding

- No spatial + ENS combination has been computed for any of 2019–2022.
- Known in advance:
  - every base model's per-year AUROC
  - the served combination's per-year AUROC (D-029)
  - that the spatial model alone already reaches or passes the served
    combination in 2020 and 2021, and falls short in 2019 and 2022
  - that ENS spread and the model are partly redundant (D-029's
    coefficients)

  The question is whether combining closes or widens those gaps, and that
  is not known.

## 8. What would invalidate this

- Changing `COMBINER_PARAMS`, `backtest.json`, `combination.json`,
  `spatial.json`, `s3c_inputs.json` or any file it lists, a fold file, a fold
  model, or this file after scoring. `b5_combination.py` refuses each one.
- Scoring before this file is committed and pushed with CI green, or running
  the primary more than once.
- Deciding on a secondary or a per-year result.

## 9. Registration block

Read by `scripts/b5_combination.py`. Do not edit after registration.

```registration
backtest_sha256: 0e8f23ec784240200d45ce33652f8f23ec5f736db94fec163d9d74196f1e4534
combination_sha256: e4af6751f9fe0752bb9b2f6b76908f42614eb8d7860bdb9546f8a9c8ff679ad1
spatial_json_sha256: 56236ac0a0b5eda8dcaff0496f2133e80c0fc8cd8adc9c9f46c5e7ff89d5faca
spatial_inputs_sha256: 1152df9f60a756cded24557d32e0034cd9bad0b76dd542bcea7bc981d1adab6b
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
lead_days: 3,4,5,6,7
required_ens_dates: 122
```
