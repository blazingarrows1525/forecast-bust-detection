# Pre-registration — S3: new model families, and the rule that promotes one

**Registered:** 2026-09-25, before any candidate model was scored on any test
year.
**Code:** `scripts/promote.py`, `src/fbd/evaluate/{promotion,backtest_stats,uncertainty}.py`,
`src/fbd/model/{mlp,params}.py`, committed before this file.
**Design:** `docs/superpowers/specs/2026-09-25-s3a-candidate-harness-mlp-design.md`

This file fixes, before any candidate exists on a test year, which candidates
may be tried, what they are compared against, and what counts as a win.
`promote.py` refuses to run unless this file is committed unchanged, the
incumbent it names is byte-identical, and the candidate's hyperparameters match
the hash registered for it.

---

## 1. The question

> Does any of three new model families rank busts better than the XGBoost
> model, across four held-out seasons?

S1b (D-026) found the XGBoost model's edge over a real ensemble holds in 2022
only. S3 asks whether a different family does better than XGBoost itself. The
first candidate, S3a, is the control: an MLP on exactly the incumbent's inputs.

## 2. Reproduced before registration

`scripts/audit_s3a.py`, output `data/artifacts/s3a_audit.json`.

| check | result |
|---|---|
| the harness reproduces every S1b per-year incumbent and ENS AUROC | **exact**, all eight values (2019–2022), with torch loaded first |
| two fits of the fold-2019 MLP (train 2016–2017, validation 2018) | **bit-identical** validation-year output; 12 s per fit |
| that MLP's validation-year (2018) AUROC | **0.8282** (defect floor 0.60) |

The five seeds reached their best validation loss at epochs 1, 3, 1, 3, 3: the
network peaks almost at once under the frozen settings. That is recorded, not
tuned; §5 is fixed.

**Environment note.** On Windows, torch must be imported before scikit-learn
(sklearn bundles an older `msvcp140.dll`; torch's `c10.dll` then fails with
WinError 1114). The entry points and the test session import torch first. The
exact reproduction in check 1 shows the load order changes no incumbent number.

## 3. The slate, the incumbent, the frozen parameters

- **Slate, declared in full:** `mlp` (S3a), `temporal` (S3b), `spatial` (S3c).
  The harness refuses any other name; a new candidate needs a new registration
  and widens the correction.
- **Incumbent, fixed for all three:** the S1b strict XGBoost fold models, pinned
  by the SHA-256 of `data/artifacts/backtest.json` below, which in turn pins
  every fold dataset and model. A promoted candidate does not become the
  comparator for the next one.
- **Frozen parameters:** the MLP's by `mlp_params_sha256` below. S3b and S3c
  are registered in their own addenda (`PREREGISTRATION_S3B.md`,
  `PREREGISTRATION_S3C.md`), each adding its `<name>_params_sha256` under this
  same rule and committed before that candidate is scored.

## 4. Primary, per candidate

| | |
|---|---|
| quantity | mean over **2019, 2020, 2021, 2022** of the within-year margin AUROC(candidate) − AUROC(incumbent) |
| rows | each fold's S1b comparison rows: Day 3–7 test rows with a label and ENS spread, 20,060 per year |
| interval | stratified cluster bootstrap over init dates; paired; **10,000** resamples; seed **20260919**; percentile interval at **1 − 0.05/3 (98.33%)**, Bonferroni over the three declared candidates |
| verdict | lower > 0 → **promoted**; upper < 0 → **the incumbent outranks the candidate**; otherwise → **not distinguishable from the incumbent** |

Bonferroni, not Holm, because the candidates are scored weeks apart. All four
years count because no candidate has been scored on any of them.

## 5. The MLP (S3a), frozen

| | |
|---|---|
| inputs | the incumbent's 52 features, same order |
| missing values | training-rows median; a 0/1 indicator for every feature missing anywhere in the fold's training rows |
| scaling | training-rows mean and SD after imputation; SD 0 → 1 |
| network | inputs → 128 → 64 → 1, ReLU, dropout 0.2 after each hidden layer |
| loss | binary cross-entropy, `pos_weight` = negatives / positives on training rows |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, batch 1,024, seeded shuffle |
| stopping | ≤ 60 epochs; patience 5 on validation-year weighted BCE; best epoch restored |
| seeds | 20260920–20260924, probabilities averaged |
| calibration | isotonic on the validation year, on the averaged probability |
| device | CPU, deterministic algorithms, 8 threads, float32 |

## 6. Secondary — reported, cannot promote

- Per year, candidate − incumbent (2,000 resamples, 95%).
- Candidate − ENS spread: mean over 2019–2021 (10,000 resamples, 95%) and per
  year (2,000), with S1b's rule — a year whose upper bound is below zero is
  stated as "ENS spread outranks the <candidate> in <year>". This cannot reopen
  D-026.

## 7. Exploratory — labelled, no claims drawn

Candidate − incumbent by month within each year; each seed's own AUROC per year.

## 8. Blinding

- No candidate has produced a prediction for any test year.
- The incumbent's per-year results and its late-season weakness (D-026) are
  known; the MLP was specified after them, with identical inputs and an
  architecture fixed without looking at any MLP result.
- The audit looked at fold 2019's validation year (2018) only.

## 9. Consequences, committed now

| MLP primary | recorded as | README / `FRONTEND_LOGIC.md` §8 |
|---|---|---|
| promoted | the neural architecture outranks trees on identical inputs; S3b and S3c are read against that | README "Other model families"; §8: "an MLP outranks the served XGBoost in a registered backtest; it is not served" |
| not distinguishable | architecture alone adds nothing detectable; any S3b/S3c gain is attributable to information | README; §8: "no neural model is served; an MLP was tested and was not distinguishable (D-027)" |
| incumbent outranks | trees beat the network on these inputs | README; §8: "no neural model is served; an MLP was tested and XGBoost outranked it (D-027)" |

The served product is unchanged in every row; promotion into the product is a
separate decision after S3c.

## 10. What would invalidate this

- Changing `backtest.json`, the fold files, this file, the MLP's parameters or
  the harness after a candidate has been scored. `promote.py` refuses the first
  four.
- Scoring a candidate outside the slate, or before its parameter hash is
  registered.
- Promoting a candidate on a per-year, secondary or exploratory result.

## 11. Registration block

Read by `promote.py`. Do not edit after registration; addenda add candidates.

```registration
backtest_sha256: 0e8f23ec784240200d45ce33652f8f23ec5f736db94fec163d9d74196f1e4534
slate: mlp,temporal,spatial
alpha_family: 0.05
years: 2019,2020,2021,2022
ens_years: 2019,2020,2021
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
lead_days: 3,4,5,6,7
required_ens_dates: 122
mlp_params_sha256: f4af16c632f3192f9dd06913cd65880c862df947bbb6b91daeca5898b946f4b4
```
