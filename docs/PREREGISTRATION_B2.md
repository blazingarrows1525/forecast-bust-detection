# Pre-registration — B2: operational-analysis substitution, with derived TCWV

**Registered:** 2026-10-06, before any model built from the operational state
produced a prediction for any of 2019–2022.
**Code:** `src/fbd/features/{tcwv,era5}.py`, `src/fbd/evaluate/b2.py`,
`scripts/{fetch_analysis_t0,build_dataset,score_b2,audit_b2}.py`, committed
before this file.
**Design:** `docs/superpowers/specs/2026-10-06-b2-operational-analysis-design.md`
(committed first, with the gates, the margin and the consequences, before
any derived TCWV was compared with anything).

---

## 1. The question

> Rebuilt from the operational HRES initial state instead of ERA5, do the 30
> analysis-derived inputs cost the bust model no material ranking skill?

30 of the 52 inputs come from a reanalysis that does not exist when a forecast
is issued (`docs/FEATURE_AVAILABILITY_MATRIX.md`). WB2 `hres_t0` is the
operational model's own initial state. It has no total column water vapour,
which 11 of the 30 depend on, so TCWV is integrated from its 13-level humidity
and surface pressure (`fbd.features.tcwv`).

## 2. Audited before registration

`scripts/audit_b2.py`, output `data/artifacts/b2_audit.json`.

| check | result |
|---|---|
| with the default source, the changed builder rebuilds each pinned ERA5 fold | **byte-identical** in all four folds (`backtest.json` hashes) |
| **G1**: ERA5's own 13-level humidity integrated, against ERA5's TCWV (2016–2017, 00Z, India box, 112,728 cells) | RMSE **0.86** kg m⁻², bias **−0.15**, r **0.9992**. Gate: RMSE ≤ 2.0, \|bias\| ≤ 1.5. **Pass** |
| **G2**: `hres_t0` derived TCWV against ERA5's TCWV (same cells) | RMSE **1.57** kg m⁻², bias **−0.09**, r **0.9973**. Gate: r ≥ 0.95, RMSE ≤ 4.0. **Pass** |
| poisoning every `hres_t0` state after the issue time, 4 sampled 2018 issue days | **0 violations** in the national indices, local fields and regimes |
| two trainings of the fold-2019 candidate (train 2016–2017, validation 2018) | **bit-identical**. Validation-year AUROC 0.8084, against 0.8053 for the incumbent's fold model (a smoke check, not a result) |
| inputs that differ between the two sources | exactly **30**, as designed. Agreement on 2016–2017 rows ranges from r 0.890 (`bob_vorticity_max_d1`) to 0.999 (`mslp`, `monsoon_trough_mslp_z`) |
| the 15 inputs (14 `analysis_t0` seasons and ERA5's static fields) | hashed into `b2_inputs.json` |

## 3. The candidate, frozen

Each S1b strict fold is rebuilt with `state_source="analysis_t0"`. Everything
else is unchanged, and `score_b2.py` checks it:
- identical rows, splits and labels
- the same 52 feature names
- the regime classifier refitted on the fold's training years
- the same standardisation years
- `DEFAULT_PARAMS` XGBoost on the CPU, as the incumbent was trained (hash below)
- isotonic calibration on the validation year

The rebuilt folds and models are written to `data/processed/backtest/b2/`,
never over the pinned files.

## 4. Primary

The mean over **2019, 2020, 2021, 2022** of the within-year margin
AUROC(candidate) − AUROC(incumbent), where the incumbent is the pinned S1b
fold models. The rows are S1b's comparison rows (Day 3–7 test rows with a
label and ENS spread, 20,060 a year). The interval is a stratified, paired
cluster bootstrap over init dates: **10,000** resamples, seed **20260919**,
two-sided **95%**.

Margin δ = **0.005** AUROC. Verdict:
- lower > −δ → **non-inferior**
- upper < −δ → **inferior**
- otherwise → **inconclusive**

## 5. Secondary — reported, cannot decide

- The margin per year (2,000 resamples, 95%).
- Brier skill score of each model on the comparison rows.
- The share of comparison rows each model's Mahalanobis OOD detector would
  refuse (fitted on its own training rows, quantile 0.995).

## 6. Consequences, committed in the design and restated

| verdict | recorded as |
|---|---|
| non-inferior | the operational-state features, with derived TCWV, become the live feature set's source; `FEATURE_AVAILABILITY_MATRIX.md` moves the 30 inputs to "operational source, validated" |
| inferior or inconclusive | the live path cannot claim the full model; it ships a reduced model or stays archive-only, and says so |

The served store stays ERA5-built in every case. This study changes no number
the product shows.

## 7. Blinding

- No operational-state model has produced a prediction for 2019–2022.
- Known in advance: the incumbent's per-year results (D-026) and the S3
  results (D-027, D-031, D-032). None of them bears on how well the
  operational state stands in for ERA5.
- The gates and the audit looked at 2016–2018 only, and only 2018's validation
  rows were scored (smoke).

## 8. What would invalidate this

- Changing `DEFAULT_PARAMS`, `backtest.json`, the pinned fold files,
  `b2_inputs.json` or any file it lists, this file, the builder or the TCWV
  integral after scoring. `score_b2.py` refuses each one it can see.
- Scoring before this file is committed, or running the primary more than once.
- Deciding on a secondary or a per-year result.

## 9. Registration block

Read by `score_b2.py`. Do not edit after registration.

```registration
backtest_sha256: 0e8f23ec784240200d45ce33652f8f23ec5f736db94fec163d9d74196f1e4534
b2_inputs_sha256: 9acc6654426023f217a01e0c848b04ca8d59eb78157ade082f968f82aca881e8
params_sha256: 205d618ace558e0bc2a72b48feb5ac5f0455604602fb582931d80080d86403cf
margin: 0.005
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
lead_days: 3,4,5,6,7
required_ens_dates: 122
```
