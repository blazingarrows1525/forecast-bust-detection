# B3 and B4 — two refinements of the served combiner — design

**Date:** 2026-10-06
**Status:** next in the plan the owner said "go ahead" to. This file fixes
both questions, their rules and their consequences **before either candidate
is computed for a test year**.
**Backlog:** `docs/ML_RESEARCH_BACKLOG.md` B3 and B4. **Follows:** D-029,
D-030.

Both studies change only the combiner (the logistic layer that merges the
XGBoost model with ENS spread). They keep the served base model, rows,
folds and seeds. They are registered separately
(`docs/PREREGISTRATION_B3.md`, `docs/PREREGISTRATION_B4.md`) and scored by
one script, so neither result can influence the other's rule.

---

## B4. A combiner fitted on Days 1–2

**Evidence:** D-030 measured that at Days 1–2 in 2022 the model alone ranks
slightly better than the served combination (−0.002, −0.010). The served
combiner was fitted on Days 3–7 and is extrapolated to Days 1–2.

**Question:** does a combiner of the same form, fitted on the validation
year's **Day 1–2** rows, outrank the served (Day 3–7-fitted) combiner on
Day 1–2 test rows across 2019–2022?

| | |
|---|---|
| rows | per fold, the test year's Day 1–2 rows with a label and ENS spread (`comparison_rows` with leads 1, 2) |
| incumbent | S1c's combiner fitted on the validation year's Day 3–7 rows (the served rule) |
| candidate | the same combiner fitted on the validation year's Day 1–2 rows |
| primary | mean over 2019–2022 of the within-year AUROC margin, candidate − incumbent; stratified cluster bootstrap over init dates, 10,000, seed 20260919, 95% |
| verdict | lower > 0: **a Day 1–2 combiner outranks the extrapolation**; upper < 0: **the extrapolation outranks it**; otherwise **not distinguishable** |
| secondary | per year; candidate − the XGBoost model alone (calibrated) on the same rows |
| consequence | outranks → a lead-split combiner (Days 1–2 vs 3–10) is adopted the next time the store is regenerated, and the served claim may extend to Days 1–2. Otherwise → the extrapolation stays, and the claim stays on Days 3–7. |

## B3. Tail calibration: a beta-family combiner

**Evidence:** D-030 measured that in 2022 the served combination's top
equal-count bin (1,988 rows) predicts 0.207 and observes 0.172:
overconfident by 0.036.

**Candidate:** the beta-calibration family applied to the model term. A
logistic regression on [ln p_raw, ln(1 − p_raw), log1p ENS spread] in place
of [logit p_raw, log1p ENS spread]. It nests the served form: equal and
opposite coefficients on the first two give the logit back. So it can bend
the tail without being forced to. It is fitted on the same validation-year
Day 3–7 rows with the same settings (`COMBINER_PARAMS`: C 1e6, lbfgs, 1,000
iterations, clip 1e-6).

**Not tested: Venn–Abers.** Inductive Venn–Abers outputs a probability
interval per row. It needs a per-row isotonic refit (or the specialised
algorithm), and its interval would need a product decision about display.
It is recorded as not done.

| | |
|---|---|
| rows | S1b's comparison rows (Day 3–7 test rows with a label and ENS spread, 20,060 a year) |
| incumbent | the served combiner form (S1c), fitted per fold on the validation year |
| candidate | the beta-family combiner, fitted on the same rows |
| **primary** | the **tail gap**: \|mean predicted − observed rate\| over the top decile of each model's own predictions in a test year. Mean over 2019–2022 of the within-year difference, candidate − incumbent. Stratified cluster bootstrap over init dates, 10,000, seed 20260919, 95% |
| guard | Brier score, candidate − incumbent, same bootstrap |
| verdict | **adopt** if the tail-gap interval's upper bound < 0 **and** the Brier interval's lower bound ≤ 0 (Brier not clearly worse). **Reject** if the tail gap's lower bound > 0 or the Brier interval's lower bound > 0. Otherwise **not established**. |
| secondary | per year; ECE (10 equal-width bins); AUROC margin (the candidate can reorder rows slightly, so it is reported); the coefficients |
| consequence | adopt → the beta-family combiner replaces the served form at the next store regeneration, with the served claim's numbers re-derived. Otherwise → the served combiner stays, and its measured overconfidence stays stated. |

## Shared

- **Folds and base models:** S1b's pinned XGBoost fold models and fold
  datasets (`backtest.json`), verified by SHA-256.
- **Neutrality, before registration:** the script's incumbent reproduces
  `combination.json`'s combination AUROC exactly on Day 3–7 in every fold.
  No candidate is computed in the audit.
- **No GPU:** these are logistic fits on two or three inputs.
