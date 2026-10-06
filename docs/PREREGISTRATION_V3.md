# Pre-registration — V3: do B3 and B4 hold on the spatial base?

**Registered:** 2026-10-06, before any B3- or B4-style combiner was fitted on
the spatial model.
**Code:** `scripts/score_v3.py` (rules from `src/fbd/evaluate/refine.py`,
unchanged since D-035 and D-036), committed before this file.
**Design (committed first):** `docs/superpowers/specs/2026-10-06-v030-serve-spatial-design.md` §1.

## 1. The question

B3 (D-035) and B4 (D-036) were tested with the XGBoost base model. Store
v0.3.0 serves the spatial model (the owner's decision after D-034). Do both
refinements still pass their own rules when the base is the spatial model?

## 2. Reproduced before registration

`scripts/score_v3.py --audit` (`data/artifacts/v3_audit.json`): B5's spatial +
ENS combination reproduces `b5_serving.json` exactly in every fold:

| fold | AUROC |
|---|---|
| 2019 | 0.829118 |
| 2020 | 0.856138 |
| 2021 | 0.856627 |
| 2022 | 0.841949 |

No candidate was fitted.

## 3. The rules (B3's and B4's, unchanged)

| check | rows | incumbent | candidate | verdict |
|---|---|---|---|---|
| **V3-B4** | per fold, the test year's Day 1–2 rows with a label and ENS spread | the spatial logit combiner fitted on the validation year's Day 3–7 rows (B5's) | the same form fitted on the validation year's Day 1–2 rows | mean 2019–2022 AUROC margin, 95%: **adopt** if lower > 0 |
| **V3-B3** | S1b's comparison rows (Day 3–7) | B5's spatial logit combiner | the beta-family combiner on the spatial model's raw probability | **adopt** if tail-gap upper < 0 and Brier lower ≤ 0; **reject** if tail-gap lower > 0 or Brier lower > 0; otherwise **not established** |

Both use a stratified cluster bootstrap over init dates: 10,000 resamples,
seed 20260919, two-sided 95%. Per-year intervals use 2,000 resamples.

## 4. Consequences, committed in the design

Each refinement that passes goes into store v0.3.0's combiner, fitted on 2021
with the served spatial model. Whatever does not pass is left out, and the
store's metadata and the docs say so.

## 5. Registration block

```registration
backtest_sha256: 0e8f23ec784240200d45ce33652f8f23ec5f736db94fec163d9d74196f1e4534
spatial_json_sha256: 56236ac0a0b5eda8dcaff0496f2133e80c0fc8cd8adc9c9f46c5e7ff89d5faca
b5_json_sha256: b4420c35de71ed50a8bc51442192a13479ce4039fa59c6d487f0d8acae7c5f9f
spatial_inputs_sha256: 1152df9f60a756cded24557d32e0034cd9bad0b76dd542bcea7bc981d1adab6b
seed: 20260919
n_boot: 10000
n_boot_per_year: 2000
required_ens_dates: 122
```
