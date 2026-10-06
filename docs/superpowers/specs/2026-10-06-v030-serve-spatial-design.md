# Store v0.3.0 — serve the spatial model, with B3 and B4 if they hold for it — design

**Date:** 2026-10-06
**Status:** the owner decided "go ahead with cnn and v0.3.0", after D-034
(spatial + ENS eligible), D-035 (B3 adopt) and D-036 (B4 adopt). This file
fixes the serving design and one registered check **before any of it is
computed**.

---

## 1. One registered check first: do B3 and B4 hold on the spatial base?

B3 and B4 were tested with the XGBoost base model. The served base model is
changing, so both are re-tested on the spatial base before either ships. The
rules are exactly B3's and B4's (`fbd.evaluate.refine`); only the base model
changes. The registration is `docs/PREREGISTRATION_V3.md`.

| check | rows | incumbent | candidate | rule |
|---|---|---|---|---|
| **V3-B4** | per fold, the test year's Day 1–2 rows | the spatial combiner fitted on the validation year's Day 3–7 rows (B5's) | the same form fitted on the validation year's Day 1–2 rows | B4's: mean 2019–2022 AUROC margin, 95%; adopt if lower > 0 |
| **V3-B3** | S1b's comparison rows (Day 3–7) | B5's spatial logit combiner | the beta-family combiner on the spatial model's raw probability | B3's: tail-gap upper < 0 and Brier lower ≤ 0 |

The base is the S3c fold models (`spatial.json`, verified by hash), and the
bootstrap is 10,000 resamples with seed 20260919. The audit first reproduces
`b5_serving.json`'s spatial-combination AUROC per fold. **Whatever fails is
left out of v0.3.0, and the store says so.**

## 2. The served configuration

| part | v0.2.0 | v0.3.0 |
|---|---|---|
| base model | XGBoost (`bust_model.joblib`) on `dataset.parquet` | **the S3c spatial model trained on 2016–2020, early-stopped and calibrated on 2021**: the fold-2022 model `candidate_spatial_2022.joblib`, on its own inputs (`fold_2022_strict.parquet` plus the 23 pinned grids) |
| combiner | logit form, fitted on 2021 Day 3–7, extrapolated to Days 1–2 and 8–10 | fitted on 2021 with the spatial model's raw probability: the beta form if V3-B3 passes, else the logit form; a separate Day 1–2 fit if V3-B4 passes, else the extrapolation |
| rows | 2021 (validation) and 2022 (held out), every lead | the same |
| refusals | Mahalanobis on the 52 static inputs, fitted on training rows | the same detector on the same 52 inputs, **as the spatial model reads them** (strict fold features). The images are not checked: stated in the UI and the docs |
| interval | 6 bootstrap refits of XGBoost, through the combiner | 6 bootstrap refits of the spatial model (5 seeds each, on the GPU), through the combiner |
| reasons | TreeSHAP lines plus the ensemble line | **occlusion** lines plus the ensemble line (§3) |
| serving path | CPU, precomputed SQLite | unchanged: everything is computed offline, and the app reads the store |

## 3. Reasons for a network: occlusion, labelled as such

TreeSHAP does not apply to a CNN. For each row, the reason panel shows the
inputs whose replacement by their training average moves the **served
log-odds** most:
- each of the 52 static inputs, set to its training mean (0 after the
  network's standardisation);
- the forecast window, replaced by the window of an average day (all-zero
  standardised channels, present masks kept);
- the synoptic map, replaced by the training-mean map (zero anomalies).

The change is pushed through the served combiner's model term, so the number
is in served log-odds and reads in the same "raises/lowers the bust odds N×"
register as before. It is one-at-a-time, so it is **not additive** like SHAP,
and the panel says "how much the estimate moves if this input were average".
The image encoders are computed once per row, and only the head is re-run for
the static inputs.

## 4. What the product says

- `/api/metrics` gains `served_v030`. It holds the served number's 2022
  decision-band AUROC, Brier skill and ECE with cluster-bootstrap intervals,
  computed from the new store, plus the registered evidence (B5 and V3).
- The landing's evidence cards show the **served** number's held-out
  metrics. The S1, S1b and S1c sentences stay, attributed to the XGBoost
  model they tested, and a new sentence states what is served and why
  (D-034), with the 2022 trade-off in plain words.
- `FRONTEND_LOGIC.md` §8, README and DATA.md say v0.3.0 serves the spatial
  model.

## 5. Release

- `bulletins.sqlite` v0.3.0 is uploaded as a GitHub Release asset (tag
  `v0.3.0`; approved by the owner).
- `scripts/fetch_release_artifacts.py` pins its SHA-256.
- The image is published by the existing workflow (hash-locked, SBOM,
  provenance) and smoke-tested weekly.
