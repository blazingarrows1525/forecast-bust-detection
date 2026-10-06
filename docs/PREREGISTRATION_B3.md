# Pre-registration — B3: tail calibration with a beta-family combiner

**Registered:** 2026-10-06, before any beta-family combiner was computed.
**Code:** `src/fbd/evaluate/refine.py`, `scripts/score_b3_b4.py`,
`scripts/audit_b3_b4.py`, committed before this file.
**Design (committed first):** `docs/superpowers/specs/2026-10-06-b3-b4-combiner-refinements-design.md`.

## 1. The question

> Does a beta-family combiner narrow the served combination's top-decile
> reliability gap, without a worse Brier score, across 2019–2022?

D-030 measured that in 2022 the served combination's top equal-count bin
predicts 0.207 and observes 0.172: overconfident by 0.036.

## 2. Reproduced before registration

The same audit as B4 (`data/artifacts/b3_b4_audit.json`). The served
combination reproduces `combination.json` exactly in every fold. No candidate
was fitted.

## 3. The rule

| | |
|---|---|
| rows | S1b's comparison rows (Day 3–7 test rows with a label and ENS spread, 20,060 a year) |
| incumbent | the served combiner form: logistic regression on [logit p_raw, log1p ENS spread], fitted per fold on the validation year's Day 3–7 rows |
| candidate | logistic regression on [ln p_raw, ln(1 − p_raw), log1p ENS spread], same rows and settings (`COMBINER_PARAMS`). It nests the incumbent |
| primary | **tail gap** = \|mean predicted − observed rate\| over the top 10% of each model's own predictions in a test year. Mean over 2019–2022 of the within-year difference, candidate − incumbent. Stratified cluster bootstrap over init dates, 10,000 resamples, seed 20260919, two-sided 95% |
| guard | Brier score, candidate − incumbent, same bootstrap |
| verdict | **adopt** if tail-gap upper < 0 and Brier lower ≤ 0. **Reject** if tail-gap lower > 0 or Brier lower > 0. Otherwise **not established** |
| secondary | per year (2,000); ECE; AUROC margin; coefficients |

## 4. Consequences, committed in the design

- **Adopt:** the beta-family combiner replaces the served form at the next
  store regeneration, and the served claim's numbers are re-derived.
- **Otherwise:** the served combiner stays, and its measured overconfidence
  stays stated.

**Not tested:** Venn–Abers. It needs per-row interval outputs and a display
decision.

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
