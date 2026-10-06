# B5 — should the spatial model replace XGBoost inside the served combination? — design

**Date:** 2026-10-06
**Status:** the owner said "go ahead" with the plan that named this study
next. This file fixes the question, the rule and the consequences **before
any spatial + ENS combination is computed for a test year**.
**Backlog:** `docs/ML_RESEARCH_BACKLOG.md` B5. **Follows:** D-029, D-030 (the
served combination), D-032 (S3c) and its addendum.

---

## 1. The question

> Does the S3c spatial model, combined with ENS spread exactly as the served
> XGBoost model is, outrank the served XGBoost + ENS combination across
> 2019–2022?

The product serves a combination, not a model alone (D-030). S3c's spatial
model outranks XGBoost on its own (D-032). Whether that survives the
combination is a separate question: ENS spread may already carry what the
forecast window adds.

## 2. The candidate and the incumbent

| | incumbent | candidate |
|---|---|---|
| base model | the S1b XGBoost fold model (`backtest.json`) | the registered S3c primary, both branches (`spatial.json`, models verified by SHA-256) |
| combiner | S1c's: logistic regression on [logit of the *uncalibrated* probability, log1p ENS spread], `COMBINER_PARAMS` | the same, with the spatial model's seed-averaged probability before isotonic |
| fit rows | the fold's validation year, Day 3–7, with a label and ENS spread | the same rows |

**Not the window-only variant.** It looked better on 2019–2022, but those are
test years. Preferring it on them would be choosing on test data (D-032
addendum).

## 3. The rule

- **Rows:** S1b's comparison rows per fold (Day 3–7 test rows with a label
  and ENS spread, 20,060 a year).
- **Primary:** the mean over **2019–2022** of the within-year margin
  AUROC(spatial + ENS) − AUROC(XGBoost + ENS). Stratified, paired cluster
  bootstrap over init dates, **10,000** resamples, seed **20260919**,
  two-sided **95%**.
- **Verdict:**
  - lower > 0: **the spatial combination outranks the served combination**
  - upper < 0: **the served combination outranks it**
  - otherwise: **not distinguishable**
- **Why all four years:** every S3 and B2 comparison used 2019–2022, and
  every year has been seen by some study. S1c kept 2022 out because S1 had
  looked at it. Here both combinations' base models have already been scored
  on all four years, so no year is cleaner than another.
- **Secondary** (reported, cannot decide):
  - per-year margins (2,000 resamples)
  - spatial + ENS − ENS alone (mean over 2019–2021, S1c's years, and per year)
  - spatial + ENS − spatial alone
  - Brier skill and expected calibration error of each combination on the
    comparison rows
  - both combiners' coefficients per fold

## 4. Consequences, committed now

| verdict | recorded as |
|---|---|
| spatial combination outranks | the spatial model is **eligible** to be served. Serving it is a separate, designed change: reasons (TreeSHAP does not apply to a CNN), the OOD detector's inputs, and a store regenerated offline. The serving path stays CPU-only and precomputed. Until that ships, the served number is unchanged. |
| not distinguishable | the served product stays XGBoost + ENS: as good, explainable by TreeSHAP, and with no GPU anywhere near it. S3c stays a research finding. |
| served combination outranks | the same, and stated plainly. |

## 5. Before registration: the audit (`scripts/audit_b5.py`)

No new test-year information is produced. Only numbers already published are
reproduced:
1. **Neutrality of the harness:** the XGBoost + ENS combination, refitted per
   fold, reproduces `combination.json`'s combination AUROC exactly.
2. **The pinned spatial fold models:** loaded on the GPU, they reproduce
   `spatial.json`'s candidate AUROC exactly, and their files are unchanged.
3. **The pinned inputs:** every file in `s3c_inputs.json` is unchanged.

Then `docs/PREREGISTRATION_B5.md`, committed and pushed with CI green. Then
one scoring run.
