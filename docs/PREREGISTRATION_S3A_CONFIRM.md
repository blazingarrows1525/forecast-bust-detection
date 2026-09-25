# Pre-registration — S3a-C: is the MLP's lead over real ENS spread confirmed in 2018?

**Registered:** 2026-09-25, before any 2018 ENS data was fetched and before any
model was scored on 2018 as a test year.
**Code:** `scripts/confirm_2018.py`, `scripts/build_dataset.py`,
`src/fbd/evaluate/{folds,promotion}.py`, `src/fbd/model/{mlp,params,train}.py`,
committed before this file (`c9e1a28`).
**Follows:** D-027 (S3a) and `docs/PREREGISTRATION_S3.md`.

`confirm_2018.py` refuses to run unless this file is committed unchanged, both
models' hyperparameters match the hashes below, and all 122 JJAS 2018 ENS init
dates are on disk.

---

## 1. The question

> Does the MLP rank busts better than a real 50-member ensemble's spread in a
> season that no comparison with the ensemble has touched?

D-027 found, as an uncorrected secondary designed after D-026, that the MLP
outranks raw ENS spread on average over 2019–2021 (+0.0238 [+0.0167, +0.0307])
and in each of those years. Every season from 2019 to 2022 has now been used to
compare a model with the ensemble. 2018 is the one season in the ENS archive
that has not: its ENS data has never been fetched.

## 2. The fold

**Fold 2018:** train 2016, calibrate 2017, test 2018, built in `strict` mode
through the S1b fold machinery — every year-dependent fit (bust thresholds,
climatologies, standardisation, regime scaling) on 2016 only. It lives in a
separate table (`CONFIRM_FOLDS`) so the registered S1b and S3 code still see
exactly their four folds.

Models, unchanged from their registrations:

- **MLP:** `MLP_PARAMS` exactly as registered for S3a (hash below).
- **XGBoost:** `BustModel` defaults (hash below), for the secondaries.

Both score every comparison row, including rows the served product would refuse.

## 3. What was seen before registering

- A blind smoke build of fold 2018: splits train 2016 / validation 2017 / test
  2018; the frozen model's 52 features; MLP best epochs 1, 1, 2, 2, 1.
- **Validation-year (2017) AUROC: XGBoost 0.7035, MLP 0.8030.** With one training
  year, the fixed 600-tree XGBoost generalises poorly. This partly anticipates
  the MLP − XGBoost secondary below; it says nothing about ENS, which is the
  primary.
- 2018 was the validation year of fold 2019 and a training year of folds
  2020–2022, and the S3a audit reported a different MLP's AUROC on it (0.8282,
  trained on 2016–2017). No model has been compared with the ensemble on 2018.

## 4. Primary — one test, one verdict

| | |
|---|---|
| quantity | AUROC(MLP) − AUROC(ENS comparator) on 2018 |
| rows | 2018 Day 3–7 rows with a label and ENS spread (`comparison_rows`) |
| comparator | the stronger by AUROC of raw and relative ENS spread — the S1 rule |
| interval | paired cluster bootstrap over init dates; **10,000** resamples; seed **20260919**; 95% percentile |
| requires | all 122 JJAS 2018 ENS init dates on disk |
| verdict | lower > 0 → **the MLP outranks ENS spread in 2018, a season no design decision looked at**; upper < 0 → **ENS spread outranks the MLP in 2018**; otherwise → **the MLP's lead over ENS spread is not confirmed in 2018** |

One test, so no multiplicity correction.

## 5. Secondary — reported, cannot change the verdict

2,000 resamples each, same seed, 95%:

- XGBoost − ENS spread on 2018 (does D-026's picture hold for 2018 too?).
- MLP − XGBoost on 2018 (a fifth year for S3's comparison; not part of S3's
  registered primary, which is fixed at 2019–2022).

## 6. Honest limits, stated now

- One training year weakens both models. A network usually suffers more from
  little data, which biases against confirmation.
- One season is ~120 correlated init dates. "Not confirmed" is a live outcome
  even if the lead is real.

## 7. Consequences, committed now

| primary | README / `FRONTEND_LOGIC.md` §8 | D-028 |
|---|---|---|
| lower > 0 | may say "the MLP outranks real ENS spread in 2018, a season no design decision looked at", with the interval, beside D-027's 2019–2021 secondary | LOCKED |
| contains 0 | the MLP's lead over the ensemble stays unestablished and is said so | DISCLOSED |
| upper < 0 | "ENS spread outranks the MLP in 2018", stated plainly | DISCLOSED |

In every row the served product is unchanged; the S3 rule decides promotion
into the product after S3c.

## 8. What would invalidate this

- Changing this file, the MLP's or XGBoost's parameters, or the fold after the
  2018 ENS data is on disk.
- Running with fewer than 122 2018 ENS dates.

## 9. Registration block

```registration
year: 2018
train_years: 2016
val_years: 2017
required_ens_dates: 122
seed: 20260919
n_boot: 10000
n_boot_secondary: 2000
lead_days: 3,4,5,6,7
alpha: 0.05
mode: strict
mlp_params_sha256: f4af16c632f3192f9dd06913cd65880c862df947bbb6b91daeca5898b946f4b4
xgb_params_sha256: 205d618ace558e0bc2a72b48feb5ac5f0455604602fb582931d80080d86403cf
```
