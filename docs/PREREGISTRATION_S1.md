# Pre-registration — S1: does the model outrank a real 50-member ensemble?

**Registered:** 2026-09-23, before any of the new ENS data was fetched.
**Code:** `scripts/settle_ens.py`, `src/fbd/evaluate/{ens,settle,registration}.py`,
committed before this file.
**Design:** `docs/superpowers/specs/2026-09-23-s1-settle-ens-design.md`

This file exists so the analysis cannot be chosen after the data is seen.
`settle_ens.py` refuses to run unless this file is committed unchanged and the
frozen inputs still match the hashes in the block at the end.

---

## 1. The question

> Does the model outrank a real 50-member operational ensemble (IFS ENS) at
> ranking which rainfall forecasts are about to bust?

Before S1: **+0.0251 AUROC [−0.0083, +0.0580]** on 40 init dates of the 2022
held-out season. The interval contains zero. Forecasters already have the
ensemble, so if the model does not outrank its spread, the honest operational
answer is "use the spread", and every later sub-project would be building on
the wrong product.

## 2. Baseline reproduced before registration

`scripts/audit_s1.py`, on the legacy ENS files only, using the code as it
stands at registration. Output: `data/artifacts/s1_audit.json`.

| check | published | reproduced | match |
|---|---|---|---|
| raw margin, point | 0.0251 | 0.0251 | yes |
| raw margin, lower | −0.0083 | −0.0083 | yes |
| raw margin, upper | +0.0580 | +0.0580 | yes |
| calibrated margin, point | 0.0403 | 0.0403 | yes |
| model AUROC | 0.832 | 0.8324 | yes |
| raw ENS AUROC | 0.807 | 0.8072 | yes |
| calibrated ENS AUROC | 0.792 | 0.7920 | yes |
| rows | 6,732 | 6,732 | yes |
| init dates | 40 | 40 | yes |
| lagged proxy AUROC (calibrated on training years) | 0.731 | 0.7314 | yes |
| scored bust rate | 3.4% | 3.402% | yes |
| refused bust rate | 23.4% | 23.377% | yes |

**Determinism:** two legacy dates (2022-06-01, 2022-07-31) were re-fetched with
the S1 fetch code and compared with the stored values: 360 rows each, maximum
absolute difference **0**.

**Discrepancies found and to be corrected** (none affects the numbers above):

1. 41 legacy 2022 dates were fetched but 40 are used. 2022-09-29 has no Day 3–7
   label, because its valid dates fall after 30 September. The docstring of
   `compute_confidence_intervals.py` says 41.
2. The landing page's ECE card showed 0.0107, the served-subset figure, beside
   three cards that use the decision band, where ECE is 0.0106.
3. `docs/FRONTEND_BUILT.md` said every landing-page claim reads live from the
   API. The ENS sentence, the stat cards and the refusal chart were hardcoded.
   (Fixed before this registration, in commit `028bab8`.)
4. The README calls the ENS comparison "on identical rows" without saying the
   model scores rows the product would refuse.

## 3. Frozen inputs

- **Model:** `data/artifacts/bust_model.joblib`, scored directly on dataset
  rows, exactly as the published margin was. Not retrained during S1.
- **Data:** `data/processed/dataset.parquet`.
- **ENS spread:** the standard deviation across all 50 members of each
  subdivision's area-mean 24-hour rainfall, as `scripts/fetch_ens.py` computes
  it. A date with any other member count is refused, not averaged in.

Both file hashes are in the block below.

## 4. Rows

`fbd.evaluate.ens.comparison_rows`, the published rule moved into one shared
function:

- inner join of the dataset and ENS on (subdivision, init date, lead day);
- drop rows missing the bust label or ENS spread;
- test rows: split `test` (2022) and lead days 3–7;
- fit rows (for calibrated variants): splits `train` and `val`, all leads.

**The model scores every test row, including rows the served product would
refuse as out-of-distribution.** This measures ranking skill, not the served
subset, and every output says so.

**Completeness.** The fetch must place all 122 JJAS init dates of 2022 on disk,
and `settle_ens.py` refuses otherwise. The analysis rows span fewer dates:
init dates whose Day 3–7 valid dates fall after 30 September have no label
(2022-09-29 and 2022-09-30; the model's own decision-band intervals use 120
dates for the same reason). That is expected and is not an incompleteness.

## 5. Primary — one test, one verdict

| | |
|---|---|
| quantity | model AUROC − ENS-spread AUROC |
| comparator | the stronger, by AUROC on the test rows, of raw spread and relative spread (spread / (mean + 1)). This is the published rule, kept because it is conservative: the model always faces the stronger competitor. |
| interval | paired cluster bootstrap over **init dates**; **10,000** resamples; seed **20260919**; 95% percentile |
| verdict | lower bound > 0 → **the model outranks ENS spread over the full held-out season**. Upper bound < 0 → **ENS spread outranks the model**. Otherwise → **not distinguishable over the full held-out season**. |

The third outcome is final for this season. There will be no re-slicing to find
a significant subset. The published interval used 2,000 resamples; 10,000 at
the same seed only reduces Monte Carlo noise in the endpoints.

## 6. Secondary — reported, cannot overturn the primary

These need all 122 dates of 2021 as well. If 2021 is incomplete they are
reported as "not run", and the primary is unaffected.

- **(a) Calibrated ENS vs model.** ENS isotonic fitted on **2021 only**, the
  model's own calibration year (D-010). Paired intervals on ΔAUROC, ΔBrier,
  ΔBSS, and Δ decision cost per 1,000 rows at the review threshold (0.0909).
- **(b) Does the model add anything on top of ENS?** Logistic regression on
  [log-odds of the model probability, log(1 + ENS spread)], fitted on 2021's
  Day 3–7 rows and tested on 2022, paired against raw ENS alone and against the
  model alone. **Caveat:** the model's isotonic calibration was fitted on 2021,
  so this combination is fitted on probabilities that are in-sample for
  calibration. Evaluation on 2022 is out of sample.
- **(c) Continuity.** Calibrated ENS with the published pooled fit (train + val
  ENS rows), ΔAUROC, so earlier numbers can be compared.

## 7. Exploratory — labelled, no claims drawn

2,000 resamples each, same seed.

- The margin by lead day (1–10) and by month.
- **The original 40 dates vs the new dates.** Tests whether the earlier
  subsample was representative. A large difference is reported as a finding in
  its own right.

## 8. Consequences, committed now

| verdict | README / landing / FRONTEND_LOGIC §8 | D-025 | S2–S5 |
|---|---|---|---|
| lower > 0 | "outranks a real 50-member ensemble over the full held-out season" moves to *may claim*, with the interval and date count | LOCKED | build on an established edge |
| contains 0 | "not distinguishable from a real 50-member ensemble over a full season"; "comparable" stays, now evidenced | DISCLOSED | S3's target becomes model + ENS vs ENS alone (secondary b) |
| upper < 0 | "ENS spread outranks the model over the full season", stated plainly; the value proposition is rewritten around secondary (b), or its absence | DISCLOSED | reopen the product question before building anything else |

## 9. What would invalidate this

- Changing the model, the dataset, this file, or the analysis code after the
  new data is on disk. The hashes and git history make each visible, and
  `settle_ens.py` refuses the first three.
- Reporting the primary on fewer than 122 fetched 2022 dates.
- Promoting a secondary or exploratory result to a headline claim.

## 10. Registration block

Read by `settle_ens.py`. Do not edit after registration.

```registration
model_sha256: 577578e7391e42eac5bf5390b8f713ce811957e77ac30073ff9d3d15dfb3e020
dataset_sha256: 9f8b4063a14aff844a24071d0d780b27445c9913b8d82e0974e33d906ae0a306
seed: 20260919
n_boot: 10000
required_ens_dates_2022: 122
required_ens_dates_2021: 122
lead_days: 3,4,5,6,7
alpha: 0.05
```
